struct Spectrogram{W}
    window::W

    n_fft::Int
    hop_length::Int
    pad::Int
    power::Real
    center::Bool

    normalized::Bool
    window_normalized::Bool
end
Flux.@layer Spectrogram

function Spectrogram(;
    n_fft::Int = 1024, hop_length::Int = n_fft ÷ 4, pad::Int = 0,
    normalized::Bool = false, window_normalized::Bool = false,
    power::Real = 2.0, center::Bool = true,
    window = NNlib.hann_window(n_fft),
)
    Spectrogram(window,
        n_fft, hop_length, pad, power, center,
        normalized, window_normalized)
end

function (sp::Spectrogram)(waveform)
    get_backend(waveform) != get_backend(sp.window) && throw(ArgumentError(
        "`waveform` must be on the same device as `Spectrogram`, " *
        "instead `$(get_backend(waveform))` vs `$(get_backend(sp.window))`."))
    return spectrogram(waveform;
        sp.n_fft, sp.hop_length, sp.pad, sp.power, sp.center,
        sp.window, sp.normalized, sp.window_normalized)
end

struct MelScale{F}
    filterbanks::F
end
Flux.@layer MelScale

"""
    MelScale(;
        n_mels::Int = 128, sample_rate::Int = 16000,
        fmin::Float32 = 0f0, fmax::Float32 = Float32(sample_rate ÷ 2),
        n_freqs::Int = 201, # n_fft ÷ 2 + 1)

Transform normal spectrogram into mel frequency spectrogram
with triangular filter banks.

# Arguments:

- `n_freqs::Int`: Number of frequencies to highlight.
- `n_mels::Int`: Number of mel filterbanks.
- `sample_rate::Int`: Sample rate of the audio waveform.
- `fmin::Float32`: Minimum frequency in Hz.
- `fmax::Float32`: Maximum frequency in Hz.

# Call Arguments:

- `spec`: Spectrogram of shape `(n_freqs, time, batch)`.

# Returns:

Mel frequency spectrogram of shape `(time, n_mels, batch)`.
"""
function MelScale(;
    n_mels::Int = 128, sample_rate::Int = 16000,
    fmin::Float32 = 0f0, fmax::Float32 = Float32(sample_rate ÷ 2),
    n_freqs::Int = 513, # n_fft ÷ 2 + 1
)
    MelScale(melscale_filterbanks(;
        n_freqs, n_mels, sample_rate, fmin, fmax))
end

function (ms::MelScale)(spec)
    get_backend(spec) != get_backend(ms.filterbanks) && throw(ArgumentError(
        "`spec` must be on the same device as `MelScale`, " *
        "instead `$(get_backend(spec))` vs `$(get_backend(ms.filterbanks))`."))

    n_freqs, time, batch = size(spec)
    n_freqs_fb, n_mels = size(ms.filterbanks)
    n_freqs != n_freqs_fb && throw(ArgumentError(
        "`n_freqs=$n_freqs` for `spec` does not match " *
        "`n_freqs=$n_freqs_fb` for filter banks."))

    return permutedims(spec, (2, 1, 3)) ⊠ ms.filterbanks
end
