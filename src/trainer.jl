Base.@kwdef mutable struct Trainer{M1, M2, M3, O1, O2, O3, S1, S2, L, T, D}
    generator::M1
    scale_discriminator::M2
    period_discriminator::M3

    opt_generator::O1
    opt_scale_discriminator::O2
    opt_period_discriminator::O3

    lr_gen_scheduler::S1
    lr_disc_scheduler::S2

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
    lr_gen_scheduler, lr_disc_scheduler,
    train_loader, test_loader, mel_transform,
)
    Trainer(
        generator |> device,
        scale_discriminator |> device,
        period_discriminator |> device,

        opt_generator |> device,
        opt_scale_discriminator |> device,
        opt_period_discriminator |> device,

        lr_gen_scheduler,
        lr_disc_scheduler,

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

    # Recovert from provided checkpoint.
    if ckpt_path ≢ nothing
        ckpt = JLD2.load(ckpt_path)

        Flux.loadmodel!(trainer.generator, ckpt["generator"])
        Flux.loadmodel!(trainer.period_discriminator, ckpt["period_discriminator"])
        Flux.loadmodel!(trainer.scale_discriminator, ckpt["scale_discriminator"])

        Flux.loadmodel!(trainer.opt_generator, ckpt["opt_generator"])
        Flux.loadmodel!(trainer.opt_scale_discriminator, ckpt["opt_scale_discriminator"])
        Flux.loadmodel!(trainer.opt_period_discriminator, ckpt["opt_period_discriminator"])

        trainer.lr_gen_scheduler.state = ckpt["lr_gen_scheduler"]
        trainer.lr_disc_scheduler.state = ckpt["lr_disc_scheduler"]

        # Update optimizers with schedulers.
        Optimisers.adjust!(trainer.opt_generator,
            trainer.lr_gen_scheduler.schedule(trainer.lr_gen_scheduler.state))
        Optimisers.adjust!(trainer.opt_scale_discriminator,
            trainer.lr_disc_scheduler.schedule(trainer.lr_disc_scheduler.state))
        Optimisers.adjust!(trainer.opt_period_discriminator,
            trainer.lr_disc_scheduler.schedule(trainer.lr_disc_scheduler.state))

        trainer.current_step = ckpt["current_step"]
        trainer.current_epoch = ckpt["current_epoch"]

        vlosses = ckpt["vlosses"]
        vloss = vlosses[end]
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
                ckpt_file = joinpath(ckpt_dir, "ckpt-$(trainer.current_epoch)-$(trainer.current_step).jld2") 
                JLD2.jldsave(ckpt_file;
                    generator=Flux.state(trainer.generator |> cpu),
                    period_discriminator=Flux.state(trainer.period_discriminator |> cpu),
                    scale_discriminator=Flux.state(trainer.scale_discriminator |> cpu),

                    opt_generator=cpu(trainer.opt_generator),
                    opt_period_discriminator=cpu(trainer.opt_period_discriminator),
                    opt_scale_discriminator=cpu(trainer.opt_scale_discriminator),

                    lr_gen_scheduler=trainer.lr_gen_scheduler.state,
                    lr_disc_scheduler=trainer.lr_disc_scheduler.state,

                    trainer.current_step,
                    trainer.current_epoch,

                    vlosses,
                )
            end

            next!(bar; showvalues=[
                (:steps, trainer.current_step),
                (:gen_loss, gloss),
                (:disc_loss, dloss),
                (:val_loss, vloss),
                (:lr_gen, trainer.lr_gen_scheduler.schedule(trainer.lr_gen_scheduler.state)),
                (:lr_disc, trainer.lr_disc_scheduler.schedule(trainer.lr_disc_scheduler.state)),
            ])
            trainer.current_step += 1
        end

        Optimisers.adjust!(trainer.opt_generator,
            ParameterSchedulers.next!(trainer.lr_gen_scheduler))

        lr_disc = ParameterSchedulers.next!(trainer.lr_disc_scheduler)
        Optimisers.adjust!(trainer.opt_scale_discriminator, lr_disc)
        Optimisers.adjust!(trainer.opt_period_discriminator, lr_disc)

        trainer.current_epoch += 1
    end
    return
end

function train_step!(trainer::Trainer, batch)
    wavs, mel, mel_loss = trainer.device.(batch)
    kab = get_backend(wavs)
    Δ = trainer.device([1f0])

    wavs_gen = nothing

    # NOTE
    # Create alias, otherwise Zygote computes grads w.r.t. PD in generator step.
    #
    # TODO make Zygote.lib.accum(::Thunk, ::Thunk) produce another Thunk?
    pd = trainer.period_discriminator
    sd = trainer.scale_discriminator
    mt = trainer.mel_transform
    gen = trainer.generator

    # Generator step.
    GPUArrays.@cache_scope kab :trainstep begin
        gloss, gback = Zygote.pullback(gen) do gen
            ŷ = gen(mel)
            # Store for the discriminator step.
            wavs_gen = ignore_derivatives(() -> GPUArrays.@no_cache_scope copy(ŷ))

            # Reshape from (n_frames, channels, batch) to (n_frames, batch).
            ŷ_mel = mt(reshape(ŷ, (size(ŷ)[[1, 3]])))
            loss_mel = _mae(ŷ_mel, mel_loss)

            pd_maps = pd(wavs)
            pd_gen_maps = pd(ŷ)
            loss_period =
                generator_loss(pd_gen_maps) #.+
                2f0 .* feature_loss(pd_maps, pd_gen_maps)

            sd_maps = sd(wavs)
            sd_gen_maps = sd(ŷ)
            loss_scale =
                generator_loss(sd_gen_maps) .+
                2f0 .* feature_loss(sd_maps, sd_gen_maps)

            45f0 .* loss_mel .+ loss_period .+ loss_scale
        end
        hgloss = Array(gloss)[1]
        ∇G = gback(Δ)
        Flux.update!(trainer.opt_generator, gen, ∇G[1])
    end

    # Discriminators step.
    GPUArrays.@cache_scope kab :trainstep begin
        dloss, dback = Zygote.pullback(pd, sd) do pd, sd
            pd_maps = pd(wavs)
            pd_gen_maps = pd(wavs_gen)
            pd_loss = discriminator_loss(pd_maps, pd_gen_maps)

            sd_maps = sd(wavs)
            sd_gen_maps = sd(wavs_gen)
            sd_loss = discriminator_loss(sd_maps, sd_gen_maps)

            pd_loss .+ sd_loss
        end
        hdloss = Array(dloss)[1]
        ∇D = dback(Δ)
        Flux.update!(trainer.opt_period_discriminator, pd, ∇D[1])
        Flux.update!(trainer.opt_scale_discriminator, sd, ∇D[2])
    end

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
