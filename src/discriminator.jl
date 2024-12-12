struct PeriodDiscriminator{C}
    convs::C
    period::Int
end
Flux.@layer PeriodDiscriminator

function PeriodDiscriminator(period::Int; kernel::Int = 5, stride::Int = 3)
    pad = (get_padding(kernel, 1), 0)
    act = leakyrelu
    convs = Chain(
        WeightNorm(Conv((kernel, 1), 1 => 32, act; stride=(stride, 1), pad)),
        WeightNorm(Conv((kernel, 1), 32 => 128, act; stride=(stride, 1), pad)),
        WeightNorm(Conv((kernel, 1), 128 => 512, act; stride=(stride, 1), pad)),
        WeightNorm(Conv((kernel, 1), 512 => 1024, act; stride=(stride, 1), pad)),
        WeightNorm(Conv((kernel, 1), 1024 => 1024, act; stride=1, pad=(2, 0))),
        WeightNorm(Conv((3, 1), 1024 => 1; pad=(1, 0))),
    )
    PeriodDiscriminator(convs, period)
end

function (pd::PeriodDiscriminator)(x)
    time, channels, batch = size(x)

    if time % pd.period != 0
        n_pad = pd.period - time % pd.period
        x = pad_reflect(x, (0, n_pad))
        time += n_pad
    end

    x_tiled = reshape(x, (time ÷ pd.period, pd.period, channels, batch))
    Flux.activations(pd.convs, x_tiled)
end

struct MultiPeriodDiscriminator{D}
    discriminators::D
end
Flux.@layer MultiPeriodDiscriminator

MultiPeriodDiscriminator() = MultiPeriodDiscriminator(Parallel(vcat,
    PeriodDiscriminator(2),
    PeriodDiscriminator(3),
    PeriodDiscriminator(5),
    PeriodDiscriminator(7),
    PeriodDiscriminator(11),
))

(mpd::MultiPeriodDiscriminator)(y) = mpd.discriminators(y)

struct ScaleDiscriminator{C}
    convs::C
end
Flux.@layer ScaleDiscriminator

function ScaleDiscriminator()
    act = leakyrelu
    convs = Chain(
        WeightNorm(Conv((15,), 1 => 128, act; stride=1, pad=7)),
        WeightNorm(Conv((41,), 128 => 128, act; stride=2, pad=20, groups=4)),
        WeightNorm(Conv((41,), 128 => 256, act; stride=2, pad=20, groups=16)),
        WeightNorm(Conv((41,), 256 => 512, act; stride=4, pad=20, groups=16)),
        WeightNorm(Conv((41,), 512 => 1024, act; stride=4, pad=20, groups=16)),
        WeightNorm(Conv((41,), 1024 => 1024, act; stride=1, pad=20, groups=16)),
        WeightNorm(Conv((5,), 1024 => 1024, act; stride=1, pad=1)),
        WeightNorm(Conv((3,), 1024 => 1; stride=1, pad=1)),
    )
    ScaleDiscriminator(convs)
end

(sd::ScaleDiscriminator)(x) = Flux.activations(sd.convs, x)

struct MultiScaleDiscriminator{D}
    discriminators::D
end
Flux.@layer MultiScaleDiscriminator

"""
Parallel
    -> SD
    -> AvgPool -> Parallel
        -> SD
        -> AvgPool -> SD
"""
MultiScaleDiscriminator() = MultiScaleDiscriminator(
    Parallel(vcat,
        ScaleDiscriminator(),
        Chain(MeanPool((4,); pad=2, stride=2), Parallel(vcat,
            ScaleDiscriminator(),
            Chain(MeanPool((4,); pad=2, stride=2), ScaleDiscriminator()),
        ))
))

(msd::MultiScaleDiscriminator)(y) = msd.discriminators(y)
