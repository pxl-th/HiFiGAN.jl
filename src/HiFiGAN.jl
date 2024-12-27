module HiFiGAN

using AMDGPU
using KernelAbstractions
using NNlib
using Makie
using CairoMakie
using ChainRulesCore: ignore_derivatives, @thunk, unthunk
using GPUArrays
using Flux
using ParameterSchedulers
using FileIO
using FLAC
using Random
using Statistics
using ProgressMeter
using Zygote

import ChainRulesCore
import JLD2
import MLUtils
import Optimisers

Maybe{T} = Union{Nothing, T}

include("spectral.jl")
include("dataset.jl")
include("generator.jl")
include("discriminator.jl")
include("loss.jl")
include("trainer.jl")

function main()
    CairoMakie.activate!()

    kab = ROCBackend()
    GPUArrays.invalidate_cache_allocator!(kab, :trainstep)
    GPUArrays.invalidate_cache_allocator!(kab, :valstep)
    GC.gc(false)
    GC.gc(true)

    files_list = joinpath(homedir(), "Downloads", "LJSpeech-1.1", "metadata.csv")
    train_files, test_files = load_files(files_list; train_split=0.9)
    train_dataset = LJDataset(train_files)
    test_dataset = LJDataset(test_files)

    train_loader = MLUtils.DataLoader(train_dataset;
        batchsize=24, shuffle=true, partial=false)
    test_loader = MLUtils.DataLoader(test_dataset; shuffle=false, batchsize=1)
    @info "Train loader length: $(length(train_loader))"
    @info "Test loader length: $(length(test_loader))"

    generator = Generator(;
        upsample_kernels=[16, 16, 8],
        upsample_rates=[8, 8, 4],
        upsample_initial_channels=256,

        resblock_kernels=[3, 5, 7],
        resblock_dilations=[[1, 2], [2, 6], [3, 12]],
    )
    period_discriminator = MultiPeriodDiscriminator()
    scale_discriminator = MultiScaleDiscriminator()

    opt_generator = Flux.setup(Optimisers.AdamW(2e-4), generator)
    opt_period_discriminator = Flux.setup(Optimisers.AdamW(2e-4), period_discriminator)
    opt_scale_discriminator = Flux.setup(Optimisers.AdamW(2e-4), scale_discriminator)

    lr_gen_scheduler = ParameterSchedulers.Stateful(Exp(; start=2e-4, decay=0.999))
    lr_disc_scheduler = ParameterSchedulers.Stateful(Exp(; start=2e-4, decay=0.999))

    trainer = Trainer(gpu;
        generator, scale_discriminator, period_discriminator,
        opt_generator, opt_scale_discriminator, opt_period_discriminator,
        lr_gen_scheduler, lr_disc_scheduler,
        train_loader, test_loader,
        mel_transform=train_dataset.mel_transform_loss,
    )
    train!(trainer;
        ckpt_path=nothing,
        epochs=3000, save_step=5000, test_step=5000,
        save_dir="/home/pxlth/code/HiFiGAN.jl/runs/2",
    )
    return
end

function eval()
    in_dir = "/home/pxlth/Downloads/LJSpeech-1.1/test/"
    out_dir = "/home/pxlth/Downloads/LJSpeech-1.1/eval-test/"
    isdir(out_dir) || mkpath(out_dir)
    ckpt_path = "/home/pxlth/code/HiFiGAN.jl/runs/states/ckpt-81-79000.jld2"

    generator = Generator(;
        upsample_kernels=[16, 16, 8],
        upsample_rates=[8, 8, 4],
        upsample_initial_channels=256,

        resblock_kernels=[3, 5, 7],
        resblock_dilations=[[1, 2], [2, 6], [3, 12]],
    ) |> gpu

    ckpt = JLD2.load(ckpt_path)
    Flux.loadmodel!(generator, ckpt["generator"])

    sample_rate = 22050
    segment_size = 8192

    n_fft = 1024
    hop_length = n_fft ÷ 4
    n_freqs = n_fft ÷ 2 + 1
    sp = Spectrogram(;
        n_fft, hop_length, center=false,
        normalized=true, pad=(n_fft - hop_length) ÷ 2)
    ms = MelScale(; n_mels=80, sample_rate, fmin=0f0, fmax=8000f0)
    mel_transform = ms ∘ sp

    for file in readdir(in_dir)
        endswith(file, ".wav") || endswith(file, ".flac") || continue

        wav, sr = load(joinpath(in_dir, file))
        wav = Float32.(wav)

        wavs = []
        n_segments = cld(size(wav, 1), segment_size)
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
            # TODO trim padding
            push!(wavs, reshape(cpu(wav_gen_seg), size(wav_gen_seg)[1:2]))
        end

        wav_gen = cat(wavs...; dims=1)
        save(
            joinpath(out_dir, replace(file, ".wav" => ".flac")),
            wav_gen, sample_rate)
    end
    return
end

end
