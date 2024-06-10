struct LJDataset{M}
    items::Vector{Tuple{String, String}}

    mel_transform::M
    mel_transform_loss::M

    segment_size::Int
end

function LJDataset(items;
    segment_size::Int = 8192, sample_rate::Int = 22050,
    n_fft::Int = 1024, n_mels::Int = 80,
    fmin::Float32 = 0f0, fmax::Float32 = 8000f0,
)
    n_freqs = n_fft ÷ 2 + 1
    hop_length = n_fft ÷ 4
    sp = Spectrogram(;
        n_fft, hop_length, center=false,
        normalized=true, pad=(n_fft - hop_length) ÷ 2)
    ms = MelScale(; n_mels, sample_rate, fmin, fmax)
    ms_loss = MelScale(; n_mels, sample_rate, fmin)

    mel_transform = ms ∘ sp
    mel_transform_loss = ms_loss ∘ sp
    LJDataset(items, mel_transform, mel_transform_loss, segment_size)
end

Base.length(d::LJDataset) = length(d.items)

function Base.getindex(d::LJDataset, i::Integer)
    wav_file, transcript = d.items[i]
    wav, sample_rate::Int = load(wav_file)
    wav = Float32.(wav)

    n_frames = size(wav, 1)
    if n_frames > d.segment_size
        s = n_frames - d.segment_size
        s = rand(1:s)
        wav = wav[s:s + d.segment_size - 1, :]
    elseif n_frames < d.segment_size
        wav = pad_zeros(wav, (0, d.segment_size - n_frames); dims=1)
    end

    mel_spec = d.mel_transform(wav)
    mel_spec_loss = d.mel_transform_loss(wav)
    return wav, mel_spec, mel_spec_loss
end

function load_files(data_path::String; train_split::Real = 0.9)
    base_dir = dirname(data_path)

    items = Tuple{String, String}[]
    open(data_path, "r") do f
        while !eof(f)
            line = readline(f)
            # 1st - file name w/o extension;
            # 2nd - transcript including numbers;
            # 3rd - transcript with numbers replaced with words numbers.
            name, _, transcript = split(strip(line), "|")
            push!(items, (joinpath(base_dir, "wavs", "$(name).wav"), transcript))
        end
    end

    # shuffle!(items)
    n_train = ceil(Int, length(items) * train_split)
    train_files, test_files = items[1:n_train], items[n_train + 1:end]
    return train_files, test_files
end
