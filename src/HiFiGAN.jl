module HiFiGAN

using KernelAbstractions
using NNlib
using Flux
using FileIO
using FLAC
using Random
using Statistics
using Zygote

include("spectral.jl")
include("dataset.jl")
include("generator.jl")
include("discriminator.jl")
include("loss.jl")

function main()
    train_files, test_files = load_files("/home/pxlth/Downloads/LJSpeech-1.1/metadata.csv")
    train_dataset = LJDataset(train_files)
    test_dataset = LJDataset(test_files)
    @show length(train_dataset)
    @show length(test_dataset)

    wav, mel_spec, mel_spec_loss = train_dataset[1]
    return
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

    loss, grad = Zygote.withgradient(g) do g
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
