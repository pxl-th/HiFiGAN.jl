_loss1(x) = mean((1f0 .- x).^2; dims=(1:ndims(x)...,))
_loss2(x) = mean(x.^2; dims=(1:ndims(x)...,))
_mae(a, b) = mean(abs.(a .- b); dims=(1:ndims(a)...,))

function discriminator_loss(maps, gen_maps)
    # Discriminator should recognize real waveforms (yr = 1),
    # thus we minimize inverse (Eq. 1).
    real_loss = sum(map(x -> _loss1(x[end]), maps))
    # Same for generated (yg = 0) (Eq. 1).
    gen_loss = sum(map(x -> _loss2(x[end]), gen_maps))
    return real_loss + gen_loss
end

function generator_loss(gen_maps)
    # Generator is trained to fool discriminator making
    # it classify generated samples to 1 (Eq. 2).
    sum(map(x -> _loss1(x[end]), gen_maps))
end

function feature_loss(
    maps::Vector{NTuple{N, T}}, gen_maps::Vector{NTuple{N, T}},
) where {N, T}
    2f0 * sum(map(feature_loss, maps, gen_maps))
end

function feature_loss(maps::NTuple{N, T}, gen_maps::NTuple{N, T}) where {N, T}
    sum(map(_mae, gen_maps, maps))
end
