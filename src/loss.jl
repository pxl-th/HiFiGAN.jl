function discriminator_loss(maps, gen_maps)
    # Discriminator should recognize real waveforms (yr = 1),
    # thus we minimize inverse (Eq. 1).
    real_loss = sum(map(x -> mean((1f0 .- x[end]).^2), maps))
    # Same for generated (yg = 0) (Eq. 1).
    gen_loss = sum(map(x -> mean(x[end].^2), gen_maps))
    return real_loss + gen_loss
end

function generator_loss(gen_maps)
    # Generator is trained to fool discriminator making
    # it classify generated samples to 1 (Eq. 2).
    sum(map(x -> mean((1f0 .- x[end]).^2), gen_maps))
end

function feature_loss(maps, gen_maps)
    2f0 * sum(map(
        (real_feats, gen_feats) ->
            sum(map(Flux.mae, gen_feats, real_feats)),
        maps, gen_maps))
end
