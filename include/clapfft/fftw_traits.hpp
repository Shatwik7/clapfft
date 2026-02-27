#ifndef CLAPFFT_FFTW_TRAITS_HPP
#define CLAPFFT_FFTW_TRAITS_HPP

#include <fftw3.h>

namespace clapfft
{

    // Traits to map T to FFTW types
    template <typename T>
    struct fftw_trait;

    template <>
    struct fftw_trait<float>
    {
        using complex_type = fftwf_complex;
        using plan_type = fftwf_plan;

        static plan_type plan_dft_1d(int n, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftwf_plan_dft_1d(n, in, out, sign, flags);
        }
        static plan_type plan_dft_2d(int n0, int n1, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftwf_plan_dft_2d(n0, n1, in, out, sign, flags);
        }
        static plan_type plan_dft_3d(int n0, int n1, int n2, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftwf_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
        }

        static plan_type plan_dft_c2r_1d(int n, complex_type *in, float *out, unsigned flags)
        {
            return fftwf_plan_dft_c2r_1d(n, in, out, flags);
        }

        static plan_type plan_dft_c2r_2d(int n0, int n1, complex_type *in, float *out, unsigned flags)
        {
            return fftwf_plan_dft_c2r_2d(n0, n1, in, out, flags);
        }

        static plan_type plan_dft_c2r_3d(int n0, int n1, int n2, complex_type *in, float *out, unsigned flags)
        {
            return fftwf_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
        }

        static plan_type plan_dft_r2c_1d(int n, float *in, complex_type *out, unsigned flags)
        {
            return fftwf_plan_dft_r2c_1d(n, in, out, flags);
        }

        static plan_type plan_dft_r2c_2d(int n0, int n1, float *in, complex_type *out, unsigned flags)
        {
            return fftwf_plan_dft_r2c_2d(n0, n1, in, out, flags);
        }

        static plan_type plan_dft_r2c_3d(int n0, int n1, int n2, float *in, complex_type *out, unsigned flags)
        {
            return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
        }

        static plan_type plan_r2r_1d(int n, float *in, float *out, int kind, unsigned flags)
        {
            return fftwf_plan_r2r_1d(n, in, out, static_cast<fftwf_r2r_kind>(kind), flags);
        }

        static plan_type plan_r2r_2d(int n0, int n1, float *in, float *out, int kind0, int kind1, unsigned flags)
        {
            return fftwf_plan_r2r_2d(
                n0,
                n1,
                in,
                out,
                static_cast<fftwf_r2r_kind>(kind0),
                static_cast<fftwf_r2r_kind>(kind1),
                flags);
        }

        static plan_type plan_r2r_3d(int n0, int n1, int n2, float *in, float *out, int kind0, int kind1, int kind2, unsigned flags)
        {
            return fftwf_plan_r2r_3d(
                n0,
                n1,
                n2,
                in,
                out,
                static_cast<fftwf_r2r_kind>(kind0),
                static_cast<fftwf_r2r_kind>(kind1),
                static_cast<fftwf_r2r_kind>(kind2),
                flags);
        }

        static void execute(plan_type plan)
        {
            fftwf_execute(plan);
        }

        static void execute_dft(plan_type plan, complex_type *in, complex_type *out)
        {
            fftwf_execute_dft(plan, in, out);
        }

        static void execute_dft_c2r(plan_type plan, complex_type *in, float *out)
        {
            fftwf_execute_dft_c2r(plan, in, out);
        }

        static void execute_dft_r2c(plan_type plan, float *in, complex_type *out)
        {
            fftwf_execute_dft_r2c(plan, in, out);
        }

        static void execute_r2r(plan_type plan, float *in, float *out)
        {
            fftwf_execute_r2r(plan, in, out);
        }

        static void destroy_plan(plan_type plan)
        {
            fftwf_destroy_plan(plan);
        }
    };

    template <>
    struct fftw_trait<double>
    {
        using complex_type = fftw_complex;
        using plan_type = fftw_plan;

        static plan_type plan_dft_1d(int n, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftw_plan_dft_1d(n, in, out, sign, flags);
        }
        static plan_type plan_dft_2d(int n0, int n1, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftw_plan_dft_2d(n0, n1, in, out, sign, flags);
        }
        static plan_type plan_dft_3d(int n0, int n1, int n2, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftw_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
        }

        static plan_type plan_dft_c2r_1d(int n, complex_type *in, double *out, unsigned flags)
        {
            return fftw_plan_dft_c2r_1d(n, in, out, flags);
        }

        static plan_type plan_dft_c2r_2d(int n0, int n1, complex_type *in, double *out, unsigned flags)
        {
            return fftw_plan_dft_c2r_2d(n0, n1, in, out, flags);
        }

        static plan_type plan_dft_c2r_3d(int n0, int n1, int n2, complex_type *in, double *out, unsigned flags)
        {
            return fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
        }

        static plan_type plan_dft_r2c_1d(int n, double *in, complex_type *out, unsigned flags)
        {
            return fftw_plan_dft_r2c_1d(n, in, out, flags);
        }

        static plan_type plan_dft_r2c_2d(int n0, int n1, double *in, complex_type *out, unsigned flags)
        {
            return fftw_plan_dft_r2c_2d(n0, n1, in, out, flags);
        }

        static plan_type plan_dft_r2c_3d(int n0, int n1, int n2, double *in, complex_type *out, unsigned flags)
        {
            return fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
        }

        static plan_type plan_r2r_1d(int n, double *in, double *out, int kind, unsigned flags)
        {
            return fftw_plan_r2r_1d(n, in, out, static_cast<fftw_r2r_kind>(kind), flags);
        }

        static plan_type plan_r2r_2d(int n0, int n1, double *in, double *out, int kind0, int kind1, unsigned flags)
        {
            return fftw_plan_r2r_2d(
                n0,
                n1,
                in,
                out,
                static_cast<fftw_r2r_kind>(kind0),
                static_cast<fftw_r2r_kind>(kind1),
                flags);
        }

        static plan_type plan_r2r_3d(int n0, int n1, int n2, double *in, double *out, int kind0, int kind1, int kind2, unsigned flags)
        {
            return fftw_plan_r2r_3d(
                n0,
                n1,
                n2,
                in,
                out,
                static_cast<fftw_r2r_kind>(kind0),
                static_cast<fftw_r2r_kind>(kind1),
                static_cast<fftw_r2r_kind>(kind2),
                flags);
        }

        static void execute(plan_type plan)
        {
            fftw_execute(plan);
        }

        static void execute_dft(plan_type plan, complex_type *in, complex_type *out)
        {
            fftw_execute_dft(plan, in, out);
        }

        static void execute_dft_c2r(plan_type plan, complex_type *in, double *out)
        {
            fftw_execute_dft_c2r(plan, in, out);
        }

        static void execute_dft_r2c(plan_type plan, double *in, complex_type *out)
        {
            fftw_execute_dft_r2c(plan, in, out);
        }

        static void execute_r2r(plan_type plan, double *in, double *out)
        {
            fftw_execute_r2r(plan, in, out);
        }

        static void destroy_plan(plan_type plan)
        {
            fftw_destroy_plan(plan);
        }
    };

    template <>
    struct fftw_trait<long double>
    {
        using complex_type = fftwl_complex;
        using plan_type = fftwl_plan;

        static plan_type plan_dft_1d(int n, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftwl_plan_dft_1d(n, in, out, sign, flags);
        }
        static plan_type plan_dft_2d(int n0, int n1, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftwl_plan_dft_2d(n0, n1, in, out, sign, flags);
        }
        static plan_type plan_dft_3d(int n0, int n1, int n2, complex_type *in, complex_type *out, int sign, unsigned flags)
        {
            return fftwl_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
        }

        static plan_type plan_dft_c2r_1d(int n, complex_type *in, long double *out, unsigned flags)
        {
            return fftwl_plan_dft_c2r_1d(n, in, out, flags);
        }

        static plan_type plan_dft_c2r_2d(int n0, int n1, complex_type *in, long double *out, unsigned flags)
        {
            return fftwl_plan_dft_c2r_2d(n0, n1, in, out, flags);
        }

        static plan_type plan_dft_c2r_3d(int n0, int n1, int n2, complex_type *in, long double *out, unsigned flags)
        {
            return fftwl_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
        }

        static plan_type plan_dft_r2c_1d(int n, long double *in, complex_type *out, unsigned flags)
        {
            return fftwl_plan_dft_r2c_1d(n, in, out, flags);
        }

        static plan_type plan_dft_r2c_2d(int n0, int n1, long double *in, complex_type *out, unsigned flags)
        {
            return fftwl_plan_dft_r2c_2d(n0, n1, in, out, flags);
        }

        static plan_type plan_dft_r2c_3d(int n0, int n1, int n2, long double *in, complex_type *out, unsigned flags)
        {
            return fftwl_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
        }

        static plan_type plan_r2r_1d(int n, long double *in, long double *out, int kind, unsigned flags)
        {
            return fftwl_plan_r2r_1d(n, in, out, static_cast<fftwl_r2r_kind>(kind), flags);
        }

        static plan_type plan_r2r_2d(int n0, int n1, long double *in, long double *out, int kind0, int kind1, unsigned flags)
        {
            return fftwl_plan_r2r_2d(
                n0,
                n1,
                in,
                out,
                static_cast<fftwl_r2r_kind>(kind0),
                static_cast<fftwl_r2r_kind>(kind1),
                flags);
        }

        static plan_type plan_r2r_3d(int n0, int n1, int n2, long double *in, long double *out, int kind0, int kind1, int kind2, unsigned flags)
        {
            return fftwl_plan_r2r_3d(
                n0,
                n1,
                n2,
                in,
                out,
                static_cast<fftwl_r2r_kind>(kind0),
                static_cast<fftwl_r2r_kind>(kind1),
                static_cast<fftwl_r2r_kind>(kind2),
                flags);
        }

        static void execute(plan_type plan)
        {
            fftwl_execute(plan);
        }

        static void execute_dft(plan_type plan, complex_type *in, complex_type *out)
        {
            fftwl_execute_dft(plan, in, out);
        }

        static void execute_dft_c2r(plan_type plan, complex_type *in, long double *out)
        {
            fftwl_execute_dft_c2r(plan, in, out);
        }

        static void execute_dft_r2c(plan_type plan, long double *in, complex_type *out)
        {
            fftwl_execute_dft_r2c(plan, in, out);
        }

        static void execute_r2r(plan_type plan, long double *in, long double *out)
        {
            fftwl_execute_r2r(plan, in, out);
        }

        static void destroy_plan(plan_type plan)
        {
            fftwl_destroy_plan(plan);
        }
    };

} // namespace clapfft

#endif // CLAPFFT_FFTW_TRAITS_HPP
