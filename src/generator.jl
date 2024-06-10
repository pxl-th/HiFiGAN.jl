get_padding(kernel::Int, dilation::Int) = (kernel * dilation - dilation) ÷ 2

struct ResBlock2{C}
    convs::C
end
Flux.@layer ResBlock2

function ResBlock2(; channels::Int, kernel::Int, dilation::Vector{Int},
)
    convs = Chain(
        # x + conv(lrelu(x))
        SkipConnection(
            Chain(
                x -> leakyrelu(x, 0.1),
                Conv((kernel,), channels => channels;
                    dilation=dilation[1], pad=get_padding(kernel, dilation[1])),
            ), +,
        ),
        # x + conv(lrelu(x))
        SkipConnection(
            Chain(
                x -> leakyrelu(x, 0.1),
                Conv((kernel,), channels => channels;
                    dilation=dilation[2], pad=get_padding(kernel, dilation[2])),
            ), +,
        ),
    )
    ResBlock2(convs)
end

(rb::ResBlock2)(x) = rb.convs(x)

struct Generator{U, C1, C2}
    ups::U
    conv_pre::C1
    conv_post::C2
end
Flux.@layer Generator

function Generator(;
    upsample_kernels::Vector{Int},
    upsample_rates::Vector{Int},
    upsample_initial_channels::Int,

    resblock_kernels::Vector{Int},
    resblock_dilations::Vector{Vector{Int}},
)
    n_upsamples = length(upsample_rates)
    n_kernels = length(resblock_kernels)
    scale = 1f0 / Float32(n_kernels)

    ups = []
    channels = 0
    for (i, (kernel, rate)) in enumerate(zip(upsample_kernels, upsample_rates))
        cin = upsample_initial_channels ÷ 2^(i - 1)
        cout = upsample_initial_channels ÷ 2^i
        channels = cout

        rbchain = []
        for (j, (kernel, dilation)) in enumerate(zip(
            resblock_kernels, resblock_dilations,
        ))
            # First chain item: x = r(x)
            # Others:           x = x + r(x)
            if isempty(rbchain)
                push!(rbchain, ResBlock2(; channels, kernel, dilation))
            else
                push!(rbchain, SkipConnection(ResBlock2(; channels, kernel, dilation), +))
            end
        end

        push!(ups, Chain(
            x -> leakyrelu(x, 0.1),
            ConvTranspose((kernel,), cin => cout;
                stride=rate, pad=(kernel - rate) ÷ 2),
            Chain(rbchain..., x -> x .* scale),
        ))
    end

    conv_pre = Conv((7,), 80 => upsample_initial_channels; pad=3)
    conv_post = Chain(Conv((7,), channels => 1; pad=3), x -> leakyrelu(x, 0.1))
    Generator(Chain(ups...), conv_pre, conv_post)
end

(g::Generator)(x) = x |> g.conv_pre |> g.ups |> g.conv_post .|> tanh
