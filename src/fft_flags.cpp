
// A small wrapper around FFTW planning flags.  We expose our own symbols so
// that the public API does not expose FFTW constants directly.  These values
// are simply aliases for the underlying FFTW macros, allowing callers to
// express planning preferences (measure, estimate, patient, exhaustive, etc.)
// without depending on <fftw3.h> in their own headers.
#include "clapfft/fft_flags.hpp"
#include <fftw3.h>

namespace clapfft
{
    // Use unsigned to match FFTW's integer flag type.
    using fft_flags = unsigned;

    const fft_flags CLAP_FFT_ESTIMATE = FFTW_ESTIMATE;
    const fft_flags CLAP_FFT_MEASURE = FFTW_MEASURE;
    const fft_flags CLAP_FFT_PATIENT = FFTW_PATIENT;
    const fft_flags CLAP_FFT_EXHAUSTIVE = FFTW_EXHAUSTIVE;
    const fft_flags CLAP_FFT_UNALIGNED = FFTW_UNALIGNED;
} // namespace clapfft
