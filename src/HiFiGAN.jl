module HiFiGAN

using AMDGPU
using KernelAbstractions
using NNlib
using Makie
using CairoMakie
using ChainRulesCore: ignore_derivatives
using GPUArrays
using Flux
using FileIO
using FLAC
using Random
using Statistics
using ProgressMeter
using Zygote

import JLD2
import MLUtils

Maybe{T} = Union{Nothing, T}

include("spectral.jl")
include("dataset.jl")
include("generator.jl")
include("discriminator.jl")
include("loss.jl")
include("trainer.jl")

function main()
    kab = ROCBackend()
    GPUArrays.invalidate_cache_allocator!(kab, :trainstep)
    GPUArrays.invalidate_cache_allocator!(kab, :valstep)

    GC.gc(false)
    GC.gc(true)
    AMDGPU.HIP.reclaim()

    CairoMakie.activate!()

    files_list = joinpath(homedir(), "Downloads", "LJSpeech-1.1", "metadata.csv")
    train_files, test_files = load_files(files_list; train_split=0.9)
    train_dataset = LJDataset(train_files)
    test_dataset = LJDataset(test_files)

    train_loader = MLUtils.DataLoader(train_dataset;
        batchsize=16, shuffle=true, partial=false)
    test_loader = MLUtils.DataLoader(test_dataset; shuffle=false, batchsize=1)
    @info "Train loader length: $(length(train_loader))"
    @info "Test loader length: $(length(test_loader))"

    # TODO load from config
    generator = Generator(;
        upsample_kernels=[16, 16, 8],
        upsample_rates=[8, 8, 4],
        upsample_initial_channels=256,

        resblock_kernels=[3, 5, 7],
        resblock_dilations=[[1, 2], [2, 6], [3, 12]],
    )
    period_discriminator = MultiPeriodDiscriminator()
    scale_discriminator = MultiScaleDiscriminator()

    opt_generator = Flux.setup(AdamW(2e-4), generator)
    opt_period_discriminator = Flux.setup(AdamW(2e-4), period_discriminator)
    opt_scale_discriminator = Flux.setup(AdamW(2e-4), scale_discriminator)

    trainer = Trainer(gpu;
        generator, scale_discriminator, period_discriminator,
        opt_generator, opt_scale_discriminator, opt_period_discriminator,
        train_loader, test_loader,
        mel_transform=train_dataset.mel_transform_loss,
    )
    train!(trainer;
        ckpt_path=nothing,
        epochs=3000, save_step=1000, test_step=1000,
        save_dir="/home/pxlth/code/HiFiGAN.jl/runs-2",
    )

    # vlosses = Float32[]

    # # Try loading latest checkpoint.
    # # TODO load current_step as well
    # states = readdir(states_dir)
    # if !isempty(states)
    #     states = sort(states; by=i -> parse(Int, split(i, "-")[2]))
    #     ckpt_path = joinpath(states_dir, states[end])
    #     @info "Loading checkpoint: `$ckpt_path`."
    #     ckpt = JLD2.load(ckpt_path)

    #     opt_generator = ckpt["opt_generator"]
    #     opt_period_discriminator = ckpt["opt_period_discriminator"]
    #     opt_scale_discriminator = ckpt["opt_scale_discriminator"]

    #     Flux.loadmodel!(generator, ckpt["generator"])
    #     Flux.loadmodel!(period_discriminator, ckpt["period_discriminator"])
    #     Flux.loadmodel!(scale_discriminator, ckpt["scale_discriminator"])

    #     vlosses = ckpt["vlosses"]
    # end
    return
end

function eval()
    generator = Generator(;
        upsample_kernels=[16, 16, 8],
        upsample_rates=[8, 8, 4],
        upsample_initial_channels=256,

        resblock_kernels=[3, 5, 7],
        resblock_dilations=[[1, 2], [2, 6], [3, 12]],
    ) |> gpu

    ckpt_path = "/home/pxlth/code/HiFiGAN.jl/runs/states/ckpt-81-79000.jld2"
    ckpt = JLD2.load(ckpt_path)
    Flux.loadmodel!(generator, ckpt["generator"])

    n_fft = 1024
    hop_length = n_fft ÷ 4
    n_freqs = n_fft ÷ 2 + 1
    sp = Spectrogram(;
        n_fft, hop_length, center=false,
        normalized=true, pad=(n_fft - hop_length) ÷ 2)
    ms = MelScale(; n_mels=80, sample_rate=22050, fmin=0f0, fmax=8000f0)
    mel_transform = ms ∘ sp

    wav_file = "/home/pxlth/Downloads/LJSpeech-1.1/wavs/LJ001-0001.wav"
    wav, sample_rate::Int = load(wav_file)
    wav = Float32.(wav)
    segment_size = 8192

    wavs = []
    n_segments = cld(size(wav, 1), segment_size)
    @show n_segments

    for i in 1:n_segments
        s = (i - 1) * segment_size + 1
        e = i * segment_size
        if e ≤ size(wav, 1)
            wav_seg = wav[s:e, :]
        else
            wav_seg = pad_zeros(wav[s:end, :], (0, e - size(wav, 1)); dims=1)
        end

        mel = mel_transform(wav_seg) |> gpu
        wav_gen_seg = generator(mel)
        push!(wavs, reshape(cpu(wav_gen_seg), size(wav_gen_seg)[1:2]))
    end

    wav_gen = cat(wavs...; dims=1)
    save("res.flac", wav_gen, 22050)
    return
end

end
