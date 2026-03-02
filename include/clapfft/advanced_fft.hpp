#ifndef CLAPFFT_ADVANCED_FFT_HPP
#define CLAPFFT_ADVANCED_FFT_HPP

#include <complex>

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
                             int sign);

        template <typename T>
        static void many_dft_r2c(int rank, const int *n, int howmany,
                                 T *in, const int *inembed,
                                 int istride, int idist,
                                 std::complex<T> *out, const int *onembed,
                                 int ostride, int odist);

        template <typename T>
        static void many_dft_c2r(int rank, const int *n, int howmany,
                                 std::complex<T> *in, const int *inembed,
                                 int istride, int idist,
                                 T *out, const int *onembed,
                                 int ostride, int odist);

        template <typename T>
        static void many_r2r(int rank, const int *n, int howmany,
                             T *in, const int *inembed,
                             int istride, int idist,
                             T *out, const int *onembed,
                             int ostride, int odist,
                             const int *kind);
    };

} // namespace clapfft

#endif
