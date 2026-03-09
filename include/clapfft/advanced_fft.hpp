#ifndef CLAPFFT_ADVANCED_FFT_HPP
#define CLAPFFT_ADVANCED_FFT_HPP

#include <complex>

#include "fft_flags.hpp"  // RTTI for planning flags

namespace clapfft
{

    class AdvancedFFT
    {
    public:
        template <typename T>
        static void many_dft(int rank, const int *n, int howmany,
                             std::complex<T> *in, const int *inembed,
                             int istride, int idist,
                             std::complex<T> *out, const int *onembed,
                             int ostride, int odist,
                             int sign,
                             fft_flags flags = CLAP_FFT_ESTIMATE);

        template <typename T>
        static void many_dft_r2c(int rank, const int *n, int howmany,
                                 T *in, const int *inembed,
                                 int istride, int idist,
                                 std::complex<T> *out, const int *onembed,
                                 int ostride, int odist,
                                 fft_flags flags = CLAP_FFT_ESTIMATE);

        template <typename T>
        static void many_dft_c2r(int rank, const int *n, int howmany,
                                 std::complex<T> *in, const int *inembed,
                                 int istride, int idist,
                                 T *out, const int *onembed,
                                 int ostride, int odist,
                                 fft_flags flags = CLAP_FFT_ESTIMATE);

        template <typename T>
        static void many_r2r(int rank, const int *n, int howmany,
                             T *in, const int *inembed,
                             int istride, int idist,
                             T *out, const int *onembed,
                             int ostride, int odist,
                             const int *kind,
                             fft_flags flags = CLAP_FFT_ESTIMATE);
    };

} // namespace clapfft

#endif
