Base.@kwdef mutable struct Trainer{M1, M2, M3, O1, O2, O3, L, T, D}
    generator::M1
    scale_discriminator::M2
    period_discriminator::M3

    opt_generator::O1
    opt_scale_discriminator::O2
    opt_period_discriminator::O3

    train_loader::L
    test_loader::L
    mel_transform::T

    device::D

    current_step::Int
    current_epoch::Int
end

function Trainer(device;
    generator, scale_discriminator, period_discriminator,
    opt_generator, opt_scale_discriminator, opt_period_discriminator,
    train_loader, test_loader, mel_transform,
)
    Trainer(
        generator |> device,
        scale_discriminator |> device,
        period_discriminator |> device,

        opt_generator |> device,
        opt_scale_discriminator |> device,
        opt_period_discriminator |> device,

        train_loader,
        test_loader,
        train_loader.data.mel_transform |> device,

        device,

        0, 0,
    )
end

function train!(trainer::Trainer;
    ckpt_path::Maybe{String} = nothing,
    epochs::Int,
    save_step::Int,
    test_step::Int,
    save_dir::String = ".",
    precompile::Bool = false,
)
    kab = get_backend(trainer.device(Array{Int}(undef, 0)))

    # Cleanup any previous runs.
    GPUArrays.invalidate_cache_allocator!(kab, :trainstep)
    GPUArrays.invalidate_cache_allocator!(kab, :valstep)
    GC.gc(false)
    GC.gc(true)

    ckpt_dir = joinpath(save_dir, "checkpoints")
    vis_dir = joinpath(save_dir, "visualizations")
    demo_dir = joinpath(save_dir, "demo")
    isdir(ckpt_dir) || mkpath(ckpt_dir)
    isdir(vis_dir) || mkpath(vis_dir)
    isdir(demo_dir) || mkpath(demo_dir)

    vlosses = Float32[]
    vloss = 0f0

    if ckpt_path ≢ nothing
        # TODO
    end
    if precompile
        # TODO
    end

    while trainer.current_epoch < epochs
        bar = Progress(
            length(trainer.train_loader);
            desc="[$(trainer.current_epoch) / $epochs] Training")

        for batch in trainer.train_loader
            gloss, dloss = train_step!(trainer, batch)

            if trainer.current_step % test_step == 0
                GPUArrays.invalidate_cache_allocator!(kab, :trainstep)
                vloss = validation_step(trainer; demo_dir, vis_dir)
                push!(vlosses, vloss)

                if length(vlosses) > 1
                    fig = lines(vlosses)
                    save(joinpath(vis_dir, "validation-$(trainer.current_epoch)-$(trainer.current_step).png"), fig)
                end
            end

            if trainer.current_step % save_step == 0
                JLD2.jldsave(joinpath(ckpt_dir, "ckpt-$(trainer.current_epoch)-$(trainer.current_step).jld2");
                    generator=Flux.state(trainer.generator |> cpu),
                    period_discriminator=Flux.state(trainer.period_discriminator |> cpu),
                    scale_discriminator=Flux.state(trainer.scale_discriminator |> cpu),

                    opt_generator=cpu(trainer.opt_generator),
                    opt_period_discriminator=cpu(trainer.opt_period_discriminator),
                    opt_scale_discriminator=cpu(trainer.opt_scale_discriminator),

                    trainer.current_step, trainer.current_epoch,
                    vlosses)
            end

            next!(bar; showvalues=[
                (:steps, trainer.current_step),
                (:gen_loss, gloss),
                (:disc_loss, dloss),
                (:val_loss, vloss),
            ])
            trainer.current_step += 1
        end
        trainer.current_epoch += 1
    end
    return
end

function train_step!(trainer::Trainer, batch; update::Bool = true)
    wavs, mel, mel_loss = trainer.device.(batch)
    kab = get_backend(wavs)
    Δ = trainer.device([1f0])

    wavs_gen = nothing
    GPUArrays.@cache_scope kab :trainstep begin
        # Generator step.
        gloss, gback = Zygote.pullback(trainer.generator) do generator
            ŷ = generator(mel)
            # Store for the discriminator step.
            wavs_gen = ignore_derivatives() do
                GPUArrays.@no_cache_scope copy(ŷ)
            end

            # Reshape from (n_frames, channels, batch) to (n_frames, batch).
            ŷ_mel = trainer.mel_transform(reshape(ŷ, (size(ŷ)[[1, 3]])))
            loss_mel = _mae(ŷ_mel, mel_loss)

            period_maps = trainer.period_discriminator(wavs)
            period_gen_maps = trainer.period_discriminator(ŷ)
            loss_period =
                generator_loss(period_gen_maps) .+
                2f0 .* feature_loss(period_maps, period_gen_maps)

            scale_maps = trainer.scale_discriminator(wavs)
            scale_gen_maps = trainer.scale_discriminator(ŷ)
            loss_scale =
                generator_loss(scale_gen_maps) .+
                2f0 .* feature_loss(scale_maps, scale_gen_maps)

            45f0 .* loss_mel .+ loss_period .+ loss_scale
        end
        ∇G = gback(Δ)
        update && Flux.update!(trainer.opt_generator, trainer.generator, ∇G[1])
    end
    hgloss = Array(gloss)[1]

    GPUArrays.@cache_scope kab :trainstep begin
        # Discriminators step.
        dloss, dback = Zygote.pullback(
            trainer.period_discriminator,
            trainer.scale_discriminator,
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
            Flux.update!(trainer.opt_period_discriminator, trainer.period_discriminator, ∇D[1])
            Flux.update!(trainer.opt_scale_discriminator, trainer.scale_discriminator, ∇D[2])
        end
    end
    hdloss = Array(dloss)[1]

    unsafe_free!(wavs)
    unsafe_free!(mel)
    unsafe_free!(mel_loss)
    unsafe_free!(wavs_gen)
    unsafe_free!(Δ)

    return hgloss, hdloss
end

function validation_step(trainer; demo_dir::String, vis_dir::String)
    total_loss = trainer.device([0f0])
    kab = get_backend(total_loss)
    sample_rate = trainer.test_loader.data.sample_rate

    @showprogress desc="Validating" for (i, batch) in enumerate(trainer.test_loader)
        GPUArrays.@cache_scope kab :valstep begin
            wavs, mel, mel_loss = trainer.device.(batch)

            ŷ = trainer.generator(mel)
            # Reshape from (n_frames, channels, batch) to (n_frames, batch).
            ŷ_mel = trainer.mel_transform(reshape(ŷ, (size(ŷ)[[1, 3]])))
            total_loss .+= _mae(ŷ_mel, mel_loss)

            if i ≤ 4
                if trainer.current_step == 0
                    save(
                        joinpath(demo_dir, "real-$(trainer.current_step)-$i.flac"),
                        reshape(cpu(wavs), size(wavs, 1), 1), sample_rate)

                    fig = heatmap(NNlib.power_to_db(cpu(mel_loss))[:, :, 1])
                    save(joinpath(vis_dir, "mel-real-$(trainer.current_step)-$i.png"), fig)
                end

                save(
                    joinpath(demo_dir, "gen-$(trainer.current_step)-$i.flac"),
                    reshape(cpu(ŷ), size(ŷ, 1), 1), sample_rate)

                fig = heatmap(NNlib.power_to_db(cpu(ŷ_mel))[:, :, 1])
                save(joinpath(vis_dir, "mel-gen-$(trainer.current_step)-$i.png"), fig)
            end
        end
    end
    GPUArrays.invalidate_cache_allocator!(kab, :valstep)
    return Array(total_loss)[1] / length(trainer.test_loader)
end
