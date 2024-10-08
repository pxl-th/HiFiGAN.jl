module HiFiGAN

using AMDGPU
# using CUDA
using KernelAbstractions
using NNlib
using Makie
using CairoMakie
using Flux
using FileIO
using FLAC
using Random
using Statistics
using ProgressMeter
using Zygote

import JLD2
import MLUtils

include("spectral.jl")
include("dataset.jl")
include("generator.jl")
include("discriminator.jl")
include("loss.jl")

function main()
    CairoMakie.activate!()

    output_dir = joinpath(homedir(), "code", "HiFiGAN.jl", "runs")
    states_dir = joinpath(output_dir, "states")
    val_dir = joinpath(output_dir, "validations")
    vis_dir = joinpath(output_dir, "visualizations")

    isdir(output_dir) || mkpath(output_dir)
    isdir(states_dir) || mkpath(states_dir)
    isdir(val_dir) || mkpath(val_dir)
    isdir(vis_dir) || mkpath(vis_dir)

    train_files, test_files = load_files(joinpath(homedir(), "Downloads", "LJSpeech-1.1", "metadata.csv"))
    train_dataset = LJDataset(train_files)
    test_dataset = LJDataset(test_files)

    train_loader = MLUtils.DataLoader(train_dataset; batchsize=16, shuffle=true)
    test_loader = MLUtils.DataLoader(test_dataset; batchsize=1)
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

    opt_generator = Flux.setup(AdamW(2e-4), generator)
    opt_period_discriminator = Flux.setup(AdamW(2e-4), period_discriminator)
    opt_scale_discriminator = Flux.setup(AdamW(2e-4), scale_discriminator)

    vlosses = Float32[]

    # # Try loading latest checkpoint.
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

    generator = generator |> gpu
    period_discriminator = period_discriminator |> gpu
    scale_discriminator = scale_discriminator |> gpu

    opt_generator = opt_generator |> gpu
    opt_period_discriminator = opt_period_discriminator |> gpu
    opt_scale_discriminator = opt_scale_discriminator |> gpu

    steps = 0
    last_epoch = 0
    epochs = 3000
    save_step = 1000
    test_step = 100

    mel_transform = gpu(train_dataset.mel_transform_loss)
    gloss, dloss, vloss = 0f0, 0f0, 0f0

    # Run train step, but do not update the params.
    # This is done to precompile kernels and reduce memory pressure.
    @info "Precompiling..."
    for batch in test_loader
        train_step(batch,
            generator, period_discriminator, scale_discriminator;
            opt_generator, opt_period_discriminator, opt_scale_discriminator,
            mel_transform, update=false)
        break
    end
    GC.gc(false)
    GC.gc(true)
    AMDGPU.HIP.reclaim()

    @info "Training for $epochs epochs..."
    for epoch in (max(1, last_epoch)):epochs
        bar = Progress(length(train_loader); desc="[$epoch / $epochs] Training")
        for batch in train_loader
            gloss, dloss = train_step(batch,
                generator, period_discriminator, scale_discriminator;
                opt_generator, opt_period_discriminator, opt_scale_discriminator,
                mel_transform)

            # if steps % test_step == 0
            #     vloss = validation_step(generator, test_loader;
            #         mel_transform, val_dir, vis_dir, current_step=steps)
            #     push!(vlosses, vloss)

            #     if length(vlosses) > 1
            #         fig = lines(vlosses)
            #         save(joinpath(vis_dir, "validation-$epoch-$steps.png"), fig)
            #     end
            # end

            # if steps % save_step == 0
            #     JLD2.jldsave(joinpath(states_dir, "ckpt-$epoch-$steps.jld2");
            #         generator=Flux.state(generator |> cpu),
            #         period_discriminator=Flux.state(period_discriminator |> cpu),
            #         scale_discriminator=Flux.state(scale_discriminator |> cpu),

            #         opt_generator=cpu(opt_generator),
            #         opt_period_discriminator=cpu(opt_period_discriminator),
            #         opt_scale_discriminator=cpu(opt_scale_discriminator),
            #         last_epoch, vlosses)
            # end

            next!(bar; showvalues=[(:GLoss, gloss), (:DLoss, dloss), (:VLoss, vloss)])
            steps += 1
            steps == 10 && break
        end
        last_epoch = epoch
        break
    end
    AMDGPU.device_synchronize()
    return
end

function train_step(
    batch, generator, period_discriminator, scale_discriminator;
    opt_generator, opt_period_discriminator, opt_scale_discriminator,
    mel_transform, update::Bool = true,
)
    wavs, mel, mel_loss = gpu.(batch)
    wavs_gen = nothing

    Δ = gpu([1f0])

    # Generator step.
    gloss, gback = Zygote.pullback(generator) do generator
        ŷ = generator(mel)
        wavs_gen = copy(ŷ) # Store for the discriminator step.

        # Reshape from (n_frames, channels, batch) to (n_frames, batch).
        ŷ_mel = mel_transform(reshape(ŷ, (size(ŷ)[[1, 3]])))
        loss_mel = _mae(ŷ_mel, mel_loss)

        period_maps = period_discriminator(wavs)
        period_gen_maps = period_discriminator(ŷ)
        loss_period =
            generator_loss(period_gen_maps) .+
            2f0 .* feature_loss(period_maps, period_gen_maps)

        scale_maps = scale_discriminator(wavs)
        scale_gen_maps = scale_discriminator(ŷ)
        loss_scale =
            generator_loss(scale_gen_maps) .+
            2f0 .* feature_loss(scale_maps, scale_gen_maps)

        45f0 .* loss_mel .+ loss_period .+ loss_scale
    end
    ∇G = gback(Δ)
    update && Flux.update!(opt_generator, generator, ∇G[1])

    # Discriminators step.
    dloss, dback = Zygote.pullback(
        period_discriminator, scale_discriminator,
    ) do period_discriminator, scale_discriminator
        period_maps = period_discriminator(wavs)
        period_gen_maps = period_discriminator(wavs_gen)
        period_loss = discriminator_loss(period_maps, period_gen_maps)

        scale_maps = scale_discriminator(wavs)
        scale_gen_maps = scale_discriminator(wavs_gen)
        scale_loss = discriminator_loss(scale_maps, scale_gen_maps)

        period_loss .+ scale_loss
    end
    ∇D = dback(Δ)

    if update
        Flux.update!(opt_period_discriminator, period_discriminator, ∇D[1])
        Flux.update!(opt_scale_discriminator, scale_discriminator, ∇D[2])
    end

    return gloss, dloss
    # return Array(gloss)[1], Array(dloss)[1]
end

function validation_step(
    generator, test_loader;
    mel_transform, val_dir, vis_dir, current_step::Int,
)
    total_loss = gpu([0f0])
    @showprogress desc="Validating" for (i, batch) in enumerate(test_loader)
        wavs, mel, mel_loss = gpu.(batch)

        ŷ = generator(mel)
        # Reshape from (n_frames, channels, batch) to (n_frames, batch).
        ŷ_mel = mel_transform(reshape(ŷ, (size(ŷ)[[1, 3]])))
        total_loss .+= _mae(ŷ_mel, mel_loss)

        if i ≤ 4
            if current_step == 0
                save(joinpath(val_dir, "real-$current_step-$i.flac"),
                    reshape(cpu(wavs), size(wavs, 1), 1), 16000)

                fig = heatmap(NNlib.power_to_db(cpu(mel_loss))[:, :, 1])
                save(joinpath(vis_dir, "mel-real-$current_step-$i.png"), fig)
            end

            save(joinpath(val_dir, "gen-$current_step-$i.flac"),
                reshape(cpu(ŷ), size(ŷ, 1), 1), 16000)

            fig = heatmap(NNlib.power_to_db(cpu(ŷ_mel))[:, :, 1])
            save(joinpath(vis_dir, "mel-gen-$current_step-$i.png"), fig)
        end
    end
    return Array(total_loss)[1]
end

function tt()
    mpd = MultiPeriodDiscriminator() |> gpu

    wav = gpu(rand(Float32, 8192, 1, 1))
    Δ = gpu(ones(Float32, 1, 1, 1))

    mmaps = mpd(wav)

    l, back = Zygote.pullback(mpd) do mpd
        dmaps = mpd(wav)
        gl = generator_loss(dmaps)
        fl = feature_loss(mmaps, dmaps)
        gl + fl
    end
    back(Δ)

    AMDGPU.device_synchronize()
    return
end

function mm()
    x = rand(Float32, 8192, 1, 16) |> gpu
    # mpd = PeriodDiscriminator(2) |> gpu
    mpd = MultiPeriodDiscriminator() |> gpu
    for i in 1:10
        y = mpd(x)
        @show size(y)
    end
    AMDGPU.device_synchronize()
    return
end

end
