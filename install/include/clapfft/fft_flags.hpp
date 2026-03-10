#pragma once

// A small wrapper around FFTW planning flags.  We expose our own symbols so
// that the public API does not expose FFTW constants directly.  These values
// are simply aliases for the underlying FFTW macros, allowing callers to
// express planning preferences (measure, estimate, patient, exhaustive, etc.)
// without depending on <fftw3.h> in their own headers.

namespace clapfft
{
    // Use unsigned to match FFTW's integer flag type.
    using fft_flags = unsigned;

    extern const fft_flags CLAP_FFT_ESTIMATE;
    extern const fft_flags CLAP_FFT_MEASURE;
    extern const fft_flags CLAP_FFT_PATIENT;
    extern const fft_flags CLAP_FFT_EXHAUSTIVE;
    extern const fft_flags CLAP_FFT_UNALIGNED;
} // namespace clapfft
