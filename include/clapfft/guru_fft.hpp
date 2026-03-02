#ifndef CLAPFFT_GURU_FFT_HPP
#define CLAPFFT_GURU_FFT_HPP

#include <fftw3.h>

namespace clapfft
{
    using fft_plan = fftw_plan;

    class GuruFFT
    {
    public:
        static fft_plan plan_guru_dft(
            int rank, const fftw_iodim *dims,
            int howmany_rank, const fftw_iodim *howmany_dims,
            fftw_complex *in, fftw_complex *out,
            int sign, unsigned flags);

        static fft_plan plan_guru_split_dft(
            int rank, const fftw_iodim *dims,
            int howmany_rank, const fftw_iodim *howmany_dims,
            double *ri, double *ii, double *ro, double *io,
            unsigned flags);

        static fft_plan plan_guru_dft_r2c(
            int rank, const fftw_iodim *dims,
            int howmany_rank, const fftw_iodim *howmany_dims,
            double *in, fftw_complex *out,
            unsigned flags);

        static fft_plan plan_guru_split_dft_r2c(
            int rank, const fftw_iodim *dims,
            int howmany_rank, const fftw_iodim *howmany_dims,
            double *in, double *ro, double *io,
            unsigned flags);

        static fft_plan plan_guru_dft_c2r(
            int rank, const fftw_iodim *dims,
            int howmany_rank, const fftw_iodim *howmany_dims,
            fftw_complex *in, double *out,
            unsigned flags);

        static fft_plan plan_guru_split_dft_c2r(
            int rank, const fftw_iodim *dims,
            int howmany_rank, const fftw_iodim *howmany_dims,
            double *ri, double *ii, double *out,
            unsigned flags);

        static fft_plan plan_guru_r2r(
            int rank, const fftw_iodim *dims,
            int howmany_rank,
            const fftw_iodim *howmany_dims,
            double *in, double *out,
            const fftw_r2r_kind *kind,
            unsigned flags);

        static fft_plan plan_guru64_dft(
            int rank, const fftw_iodim64 *dims,
            int howmany_rank, const fftw_iodim64 *howmany_dims,
            fftw_complex *in, fftw_complex *out,
            int sign, unsigned flags);
    };
}

#endif
