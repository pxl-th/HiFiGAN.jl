module HiFiGAN

using AMDGPU
using KernelAbstractions
using NNlib
using Flux
using FileIO
using FLAC
using Random
using Statistics
using ProgressMeter

import MLUtils

include("spectral.jl")
include("dataset.jl")
include("generator.jl")
include("discriminator.jl")
include("loss.jl")

function main()
    train_files, test_files = load_files("/home/pxlth/Downloads/LJSpeech-1.1/metadata.csv")
    train_dataset = LJDataset(train_files)
    test_dataset = LJDataset(test_files)

    train_loader = MLUtils.DataLoader(train_dataset; batchsize=4)
    test_loader = MLUtils.DataLoader(test_dataset; batchsize=4)
    @info "Train loader length: $(length(train_loader))"
    @info "Test loader length: $(length(test_loader))"

    generator = Generator(;
        upsample_kernels=[16, 16, 8],
        upsample_rates=[8, 8, 4],
        upsample_initial_channels=256,

        resblock_kernels=[3, 5, 7],
        resblock_dilations=[[1, 2], [2, 6], [3, 12]],
    ) |> gpu
    period_discriminator = MultiPeriodDiscriminator() |> gpu
    scale_discriminator = MultiScaleDiscriminator() |> gpu

    opt_generator = Flux.setup(AdamW(2e-4), generator)
    opt_period_discriminator = Flux.setup(AdamW(2e-4), period_discriminator)
    opt_scale_discriminator = Flux.setup(AdamW(2e-4), scale_discriminator)

    # TODO: load from checkpoint

    steps = 0
    epochs = 3000
    save_step = 1000
    test_step = 100

    mel_transform = gpu(train_dataset.mel_transform_loss)

    @info "Training for $epochs epochs..."

    for epoch in 1:epochs # TODO load last epoch
        bar = Progress(length(train_loader); desc="[$epoch / $epochs] Training")
        for batch in train_loader
            gloss, dloss = train_step(batch,
                generator, period_discriminator, scale_discriminator;
                opt_generator, opt_period_discriminator, opt_scale_discriminator,
                mel_transform)

            showvalues = [(:GLoss, gloss), (:DLoss, dloss)]

            if steps % test_step == 0
                val_loss = validation_step(generator, test_loader; mel_transform)
                push!(showvalues, (:VLoss, val_loss))
            end

            next!(bar; showvalues)

            # TODO save

            steps += 1
        end
    end

    return
end

function train_step(
    batch, generator, period_discriminator, scale_discriminator;
    opt_generator, opt_period_discriminator, opt_scale_discriminator,
    mel_transform,
)
    wavs, mel, mel_loss = gpu.(batch)
    wavs_gen = nothing

    # Generator step.
    gloss, ∇ = Flux.withgradient(generator) do generator
        ŷ = generator(mel)
        wavs_gen = copy(ŷ) # Store for the discriminator step.

        # Reshape from (n_frames, channels, batch) to (n_frames, batch).
        ŷ_mel = mel_transform(reshape(ŷ, (size(ŷ)[[1, 3]])))
        loss_mel = Flux.mae(ŷ_mel, mel_loss)

        period_maps = period_discriminator(wavs)
        period_gen_maps = period_discriminator(ŷ)
        loss_period =
            generator_loss(period_gen_maps) +
            feature_loss(period_maps, period_gen_maps)

        scale_maps = scale_discriminator(wavs)
        scale_gen_maps = scale_discriminator(ŷ)
        loss_scale =
            generator_loss(scale_gen_maps) +
            feature_loss(scale_maps, scale_gen_maps)

        45f0 * loss_mel + loss_period + loss_scale
    end
    Flux.update!(opt_generator, generator, ∇[1])

    # Discriminators step.
    dloss, ∇ = Flux.withgradient(
        period_discriminator, scale_discriminator,
    ) do period_discriminator, scale_discriminator
        period_maps = period_discriminator(wavs)
        period_gen_maps = period_discriminator(wavs_gen)
        period_loss = discriminator_loss(period_maps, period_gen_maps)

        scale_maps = scale_discriminator(wavs)
        scale_gen_maps = scale_discriminator(wavs_gen)
        scale_loss = discriminator_loss(scale_maps, scale_gen_maps)

        period_loss + scale_loss
    end
    Flux.update!(opt_period_discriminator, period_discriminator, ∇[1])
    Flux.update!(opt_scale_discriminator, scale_discriminator, ∇[2])

    return gloss, dloss
end

function validation_step(generator, test_loader; mel_transform)
    total_loss = 0f0
    @showprogress desc="Validating" for batch in test_loader
        wavs, mel, mel_loss = gpu.(batch)

        ŷ = generator(mel)
        # Reshape from (n_frames, channels, batch) to (n_frames, batch).
        ŷ_mel = mel_transform(reshape(ŷ, (size(ŷ)[[1, 3]])))
        total_loss += Flux.mae(ŷ_mel, mel_loss)

        # TODO save first few samples to disk
    end
    return total_loss
end

function tt()
    block = ResBlock2(; channels=2, kernel=3, dilation=[1, 2])
    x = rand(Float32, (1024, 2, 1))
    y = block(x)
    @show size(y)

    g = Generator(;
        upsample_kernels=[16, 16, 8],
        upsample_rates=[8, 8, 4],
        upsample_initial_channels=256,

        resblock_kernels=[3, 5, 7],
        resblock_dilations=[[1, 2], [2, 6], [3, 12]],
    )
    x = rand(Float32, 32, 80, 1)
    y = g(x)
    @show size(y)

    loss, grad = Flux.withgradient(g) do g
        sum(g(x))
    end
    @show loss
    return
end

function pp()
    x = rand(Float32, 8192, 1, 1)
    mpd = MultiPeriodDiscriminator()
    msd = MultiScaleDiscriminator()

    maps = mpd(x)
    gen_maps = mpd(x)

    @show discriminator_loss(maps, gen_maps)
    @show generator_loss(gen_maps)
    @show feature_loss(maps, gen_maps)

    maps = msd(x)
    gen_maps = msd(x)

    @show discriminator_loss(maps, gen_maps)
    @show generator_loss(gen_maps)
    @show feature_loss(maps, gen_maps)
    return
end

end
