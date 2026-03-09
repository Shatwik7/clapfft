#include <fftw3.h>
#include <clapfft/fft_traits.hpp>

namespace clapfft
{

    // Float trait implementations
    fftwf_plan fft_trait<float>::plan_dft_1d(int n, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags)
    {
        return fftwf_plan_dft_1d(n, in, out, sign, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_2d(int n0, int n1, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags)
    {
        return fftwf_plan_dft_2d(n0, n1, in, out, sign, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_3d(int n0, int n1, int n2, fftwf_complex *in, fftwf_complex *out, int sign, unsigned flags)
    {
        return fftwf_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
    }

    fftwf_plan fft_trait<float>::plan_many_dft(int rank, const int *n, int howmany,
                                               fftwf_complex *in, const int *inembed,
                                               int istride, int idist,
                                               fftwf_complex *out, const int *onembed,
                                               int ostride, int odist,
                                               int sign, unsigned flags)
    {
        return fftwf_plan_many_dft(rank, n, howmany,
                                   in, inembed, istride, idist,
                                   out, onembed, ostride, odist,
                                   sign, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_c2r_1d(int n, fftwf_complex *in, float *out, unsigned flags)
    {
        return fftwf_plan_dft_c2r_1d(n, in, out, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_c2r_2d(int n0, int n1, fftwf_complex *in, float *out, unsigned flags)
    {
        return fftwf_plan_dft_c2r_2d(n0, n1, in, out, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_c2r_3d(int n0, int n1, int n2, fftwf_complex *in, float *out, unsigned flags)
    {
        return fftwf_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_r2c_1d(int n, float *in, fftwf_complex *out, unsigned flags)
    {
        return fftwf_plan_dft_r2c_1d(n, in, out, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_r2c_2d(int n0, int n1, float *in, fftwf_complex *out, unsigned flags)
    {
        return fftwf_plan_dft_r2c_2d(n0, n1, in, out, flags);
    }

    fftwf_plan fft_trait<float>::plan_dft_r2c_3d(int n0, int n1, int n2, float *in, fftwf_complex *out, unsigned flags)
    {
        return fftwf_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
    }

    fftwf_plan fft_trait<float>::plan_many_dft_r2c(int rank, const int *n, int howmany,
                                                   float *in, const int *inembed,
                                                   int istride, int idist,
                                                   fftwf_complex *out, const int *onembed,
                                                   int ostride, int odist,
                                                   unsigned flags)
    {
        return fftwf_plan_many_dft_r2c(rank, n, howmany,
                                       in, inembed, istride, idist,
                                       out, onembed, ostride, odist,
                                       flags);
    }

    fftwf_plan fft_trait<float>::plan_many_dft_c2r(int rank, const int *n, int howmany,
                                                   fftwf_complex *in, const int *inembed,
                                                   int istride, int idist,
                                                   float *out, const int *onembed,
                                                   int ostride, int odist,
                                                   unsigned flags)
    {
        return fftwf_plan_many_dft_c2r(rank, n, howmany,
                                       in, inembed, istride, idist,
                                       out, onembed, ostride, odist,
                                       flags);
    }

    fftwf_plan fft_trait<float>::plan_r2r_1d(int n, float *in, float *out, fftw_r2r_kind kind, unsigned flags)
    {
        return fftwf_plan_r2r_1d(n, in, out, kind, flags);
    }

    fftwf_plan fft_trait<float>::plan_r2r_2d(int n0, int n1, float *in, float *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags)
    {
        return fftwf_plan_r2r_2d(
            n0,
            n1,
            in,
            out,
            kind0,
            kind1,
            flags);
    }

    fftwf_plan fft_trait<float>::plan_r2r_3d(int n0, int n1, int n2, float *in, float *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags)
    {
        return fftwf_plan_r2r_3d(
            n0,
            n1,
            n2,
            in,
            out,
            kind0,
            kind1,
            kind2,
            flags);
    }

    fftwf_plan fft_trait<float>::plan_many_r2r(int rank, const int *n, int howmany,
                                               float *in, const int *inembed,
                                               int istride, int idist,
                                               float *out, const int *onembed,
                                               int ostride, int odist,
                                               const fftw_r2r_kind *kind, unsigned flags)
    {
        return fftwf_plan_many_r2r(rank, n, howmany,
                                   in, inembed, istride, idist,
                                   out, onembed, ostride, odist,
                                   kind, flags);
    }

    void fft_trait<float>::execute(fftwf_plan plan)
    {
        fftwf_execute(plan);
    }

    void fft_trait<float>::execute_dft(fftwf_plan plan, fftwf_complex *in, fftwf_complex *out)
    {
        fftwf_execute_dft(plan, in, out);
    }

    void fft_trait<float>::execute_dft_c2r(fftwf_plan plan, fftwf_complex *in, float *out)
    {
        fftwf_execute_dft_c2r(plan, in, out);
    }

    void fft_trait<float>::execute_dft_r2c(fftwf_plan plan, float *in, fftwf_complex *out)
    {
        fftwf_execute_dft_r2c(plan, in, out);
    }

    void fft_trait<float>::execute_r2r(fftwf_plan plan, float *in, float *out)
    {
        fftwf_execute_r2r(plan, in, out);
    }

    int fft_trait<float>::import_wisdom_from_filename(const char *filename)
    {
        return fftwf_import_wisdom_from_filename(filename);
    }

    void fft_trait<float>::export_wisdom_to_filename(const char *filename)
    {
        fftwf_export_wisdom_to_filename(filename);
    }

    char *fft_trait<float>::export_wisdom_to_string()
    {
        return fftwf_export_wisdom_to_string();
    }

    int fft_trait<float>::import_wisdom_from_string(const char *input_string)
    {
        return fftwf_import_wisdom_from_string(input_string);
    }

    void fft_trait<float>::destroy_plan(fftwf_plan plan)
    {
        fftwf_destroy_plan(plan);
    }

    // Double trait implementations
    fftw_plan fft_trait<double>::plan_dft_1d(int n, fftw_complex *in, fftw_complex *out, int sign, unsigned flags)
    {
        return fftw_plan_dft_1d(n, in, out, sign, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_2d(int n0, int n1, fftw_complex *in, fftw_complex *out, int sign, unsigned flags)
    {
        return fftw_plan_dft_2d(n0, n1, in, out, sign, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_3d(int n0, int n1, int n2, fftw_complex *in, fftw_complex *out, int sign, unsigned flags)
    {
        return fftw_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
    }

    fftw_plan fft_trait<double>::plan_many_dft(int rank, const int *n, int howmany,
                                               fftw_complex *in, const int *inembed,
                                               int istride, int idist,
                                               fftw_complex *out, const int *onembed,
                                               int ostride, int odist,
                                               int sign, unsigned flags)
    {
        return fftw_plan_many_dft(rank, n, howmany,
                                  in, inembed, istride, idist,
                                  out, onembed, ostride, odist,
                                  sign, flags);
    }

    fftw_plan fft_trait<double>::plan_guru_dft(int rank, const fftw_iodim *dims,
                                               int howmany_rank, const fftw_iodim *howmany_dims,
                                               fftw_complex *in, fftw_complex *out,
                                               int sign, unsigned flags)
    {
        guru_many_layout layout;
        if (!convert_guru_dims(rank, dims, howmany_rank, howmany_dims, layout))
        {
            return nullptr;
        }

        return plan_many_dft(rank,
                             layout.n.data(), layout.howmany,
                             in, layout.inembed.data(), layout.istride, layout.idist,
                             out, layout.onembed.data(), layout.ostride, layout.odist,
                             sign, flags);
    }

    fftw_plan fft_trait<double>::plan_guru_split_dft(int rank, const fftw_iodim *dims,
                                                     int howmany_rank, const fftw_iodim *howmany_dims,
                                                     double *ri, double *ii, double *ro, double *io,
                                                     unsigned flags)
    {
        using fn_t = fftw_plan (*)(int, const fftw_iodim *, int, const fftw_iodim *, double *, double *, double *, double *, unsigned);
        static fn_t fn = resolve_fftw_next<fn_t>("fftw_plan_guru_split_dft");
        return fn == nullptr ? nullptr : fn(rank, dims, howmany_rank, howmany_dims, ri, ii, ro, io, flags);
    }

    fftw_plan fft_trait<double>::plan_guru_dft_r2c(int rank, const fftw_iodim *dims,
                                                   int howmany_rank, const fftw_iodim *howmany_dims,
                                                   double *in, fftw_complex *out,
                                                   unsigned flags)
    {
        guru_many_layout layout;
        if (!convert_guru_dims(rank, dims, howmany_rank, howmany_dims, layout))
        {
            return nullptr;
        }

        return plan_many_dft_r2c(rank,
                                 layout.n.data(), layout.howmany,
                                 in, layout.inembed.data(), layout.istride, layout.idist,
                                 out, layout.onembed.data(), layout.ostride, layout.odist,
                                 flags);
    }

    fftw_plan fft_trait<double>::plan_guru_split_dft_r2c(int rank, const fftw_iodim *dims,
                                                         int howmany_rank, const fftw_iodim *howmany_dims,
                                                         double *in, double *ro, double *io,
                                                         unsigned flags)
    {
        using fn_t = fftw_plan (*)(int, const fftw_iodim *, int, const fftw_iodim *, double *, double *, double *, unsigned);
        static fn_t fn = resolve_fftw_next<fn_t>("fftw_plan_guru_split_dft_r2c");
        return fn == nullptr ? nullptr : fn(rank, dims, howmany_rank, howmany_dims, in, ro, io, flags);
    }

    fftw_plan fft_trait<double>::plan_guru_dft_c2r(int rank, const fftw_iodim *dims,
                                                   int howmany_rank, const fftw_iodim *howmany_dims,
                                                   fftw_complex *in, double *out,
                                                   unsigned flags)
    {
        guru_many_layout layout;
        if (!convert_guru_dims(rank, dims, howmany_rank, howmany_dims, layout))
        {
            return nullptr;
        }

        return plan_many_dft_c2r(rank,
                                 layout.n.data(), layout.howmany,
                                 in, layout.inembed.data(), layout.istride, layout.idist,
                                 out, layout.onembed.data(), layout.ostride, layout.odist,
                                 flags);
    }

    fftw_plan fft_trait<double>::plan_guru_split_dft_c2r(int rank, const fftw_iodim *dims,
                                                         int howmany_rank, const fftw_iodim *howmany_dims,
                                                         double *ri, double *ii, double *out,
                                                         unsigned flags)
    {
        using fn_t = fftw_plan (*)(int, const fftw_iodim *, int, const fftw_iodim *, double *, double *, double *, unsigned);
        static fn_t fn = resolve_fftw_next<fn_t>("fftw_plan_guru_split_dft_c2r");
        return fn == nullptr ? nullptr : fn(rank, dims, howmany_rank, howmany_dims, ri, ii, out, flags);
    }

    fftw_plan fft_trait<double>::plan_guru_r2r(int rank, const fftw_iodim *dims,
                                               int howmany_rank, const fftw_iodim *howmany_dims,
                                               double *in, double *out,
                                               const fftw_r2r_kind *kind,
                                               unsigned flags)
    {
        guru_many_layout layout;
        if (!convert_guru_dims(rank, dims, howmany_rank, howmany_dims, layout))
        {
            return nullptr;
        }

        std::vector<fftw_r2r_kind> typed_kind(static_cast<std::size_t>(rank));
        for (int i = 0; i < rank; ++i)
        {
            typed_kind[static_cast<std::size_t>(i)] = kind[i];
        }

        return plan_many_r2r(rank,
                             layout.n.data(), layout.howmany,
                             in, layout.inembed.data(), layout.istride, layout.idist,
                             out, layout.onembed.data(), layout.ostride, layout.odist,
                             typed_kind.data(), flags);
    }

    fftw_plan fft_trait<double>::plan_guru64_dft(int rank, const fftw_iodim64 *dims,
                                                 int howmany_rank, const fftw_iodim64 *howmany_dims,
                                                 fftw_complex *in, fftw_complex *out,
                                                 int sign, unsigned flags)
    {
        guru_many_layout layout;
        if (!convert_guru_dims(rank, dims, howmany_rank, howmany_dims, layout))
        {
            return nullptr;
        }

        return plan_many_dft(rank,
                             layout.n.data(), layout.howmany,
                             in, layout.inembed.data(), layout.istride, layout.idist,
                             out, layout.onembed.data(), layout.ostride, layout.odist,
                             sign, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_c2r_1d(int n, fftw_complex *in, double *out, unsigned flags)
    {
        return fftw_plan_dft_c2r_1d(n, in, out, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_c2r_2d(int n0, int n1, fftw_complex *in, double *out, unsigned flags)
    {
        return fftw_plan_dft_c2r_2d(n0, n1, in, out, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_c2r_3d(int n0, int n1, int n2, fftw_complex *in, double *out, unsigned flags)
    {
        return fftw_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_r2c_1d(int n, double *in, fftw_complex *out, unsigned flags)
    {
        return fftw_plan_dft_r2c_1d(n, in, out, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_r2c_2d(int n0, int n1, double *in, fftw_complex *out, unsigned flags)
    {
        return fftw_plan_dft_r2c_2d(n0, n1, in, out, flags);
    }

    fftw_plan fft_trait<double>::plan_dft_r2c_3d(int n0, int n1, int n2, double *in, fftw_complex *out, unsigned flags)
    {
        return fftw_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
    }

    fftw_plan fft_trait<double>::plan_many_dft_r2c(int rank, const int *n, int howmany,
                                                   double *in, const int *inembed,
                                                   int istride, int idist,
                                                   fftw_complex *out, const int *onembed,
                                                   int ostride, int odist,
                                                   unsigned flags)
    {
        return fftw_plan_many_dft_r2c(rank, n, howmany,
                                      in, inembed, istride, idist,
                                      out, onembed, ostride, odist,
                                      flags);
    }

    fftw_plan fft_trait<double>::plan_many_dft_c2r(int rank, const int *n, int howmany,
                                                   fftw_complex *in, const int *inembed,
                                                   int istride, int idist,
                                                   double *out, const int *onembed,
                                                   int ostride, int odist,
                                                   unsigned flags)
    {
        return fftw_plan_many_dft_c2r(rank, n, howmany,
                                      in, inembed, istride, idist,
                                      out, onembed, ostride, odist,
                                      flags);
    }

    fftw_plan fft_trait<double>::plan_r2r_1d(int n, double *in, double *out, fftw_r2r_kind kind, unsigned flags)
    {
        return fftw_plan_r2r_1d(n, in, out, kind, flags);
    }

    fftw_plan fft_trait<double>::plan_r2r_2d(int n0, int n1, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags)
    {
        return fftw_plan_r2r_2d(
            n0,
            n1,
            in,
            out,
            kind0,
            kind1,
            flags);
    }

    fftw_plan fft_trait<double>::plan_r2r_3d(int n0, int n1, int n2, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags)
    {
        return fftw_plan_r2r_3d(
            n0,
            n1,
            n2,
            in,
            out,
            kind0,
            kind1,
            kind2,
            flags);
    }

    fftw_plan fft_trait<double>::plan_many_r2r(int rank, const int *n, int howmany,
                                               double *in, const int *inembed,
                                               int istride, int idist,
                                               double *out, const int *onembed,
                                               int ostride, int odist,
                                               const fftw_r2r_kind *kind, unsigned flags)
    {
        return fftw_plan_many_r2r(rank, n, howmany,
                                  in, inembed, istride, idist,
                                  out, onembed, ostride, odist,
                                  kind, flags);
    }

    void fft_trait<double>::execute(fftw_plan plan)
    {
        fftw_execute(plan);
    }

    void fft_trait<double>::execute_dft(fftw_plan plan, fftw_complex *in, fftw_complex *out)
    {
        fftw_execute_dft(plan, in, out);
    }

    void fft_trait<double>::execute_dft_c2r(fftw_plan plan, fftw_complex *in, double *out)
    {
        fftw_execute_dft_c2r(plan, in, out);
    }

    void fft_trait<double>::execute_dft_r2c(fftw_plan plan, double *in, fftw_complex *out)
    {
        fftw_execute_dft_r2c(plan, in, out);
    }

    void fft_trait<double>::execute_r2r(fftw_plan plan, double *in, double *out)
    {
        fftw_execute_r2r(plan, in, out);
    }

    int fft_trait<double>::import_wisdom_from_filename(const char *filename)
    {
        return fftw_import_wisdom_from_filename(filename);
    }

    void fft_trait<double>::export_wisdom_to_filename(const char *filename)
    {
        fftw_export_wisdom_to_filename(filename);
    }

    char *fft_trait<double>::export_wisdom_to_string()
    {
        return fftw_export_wisdom_to_string();
    }

    int fft_trait<double>::import_wisdom_from_string(const char *input_string)
    {
        return fftw_import_wisdom_from_string(input_string);
    }

    void fft_trait<double>::destroy_plan(fftw_plan plan)
    {
        fftw_destroy_plan(plan);
    }

    // Long double trait implementations
    fftwl_plan fft_trait<long double>::plan_dft_1d(int n, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags)
    {
        return fftwl_plan_dft_1d(n, in, out, sign, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_2d(int n0, int n1, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags)
    {
        return fftwl_plan_dft_2d(n0, n1, in, out, sign, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_3d(int n0, int n1, int n2, fftwl_complex *in, fftwl_complex *out, int sign, unsigned flags)
    {
        return fftwl_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
    }

    fftwl_plan fft_trait<long double>::plan_many_dft(int rank, const int *n, int howmany,
                                                     fftwl_complex *in, const int *inembed,
                                                     int istride, int idist,
                                                     fftwl_complex *out, const int *onembed,
                                                     int ostride, int odist,
                                                     int sign, unsigned flags)
    {
        return fftwl_plan_many_dft(rank, n, howmany,
                                   in, inembed, istride, idist,
                                   out, onembed, ostride, odist,
                                   sign, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_c2r_1d(int n, fftwl_complex *in, long double *out, unsigned flags)
    {
        return fftwl_plan_dft_c2r_1d(n, in, out, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_c2r_2d(int n0, int n1, fftwl_complex *in, long double *out, unsigned flags)
    {
        return fftwl_plan_dft_c2r_2d(n0, n1, in, out, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_c2r_3d(int n0, int n1, int n2, fftwl_complex *in, long double *out, unsigned flags)
    {
        return fftwl_plan_dft_c2r_3d(n0, n1, n2, in, out, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_r2c_1d(int n, long double *in, fftwl_complex *out, unsigned flags)
    {
        return fftwl_plan_dft_r2c_1d(n, in, out, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_r2c_2d(int n0, int n1, long double *in, fftwl_complex *out, unsigned flags)
    {
        return fftwl_plan_dft_r2c_2d(n0, n1, in, out, flags);
    }

    fftwl_plan fft_trait<long double>::plan_dft_r2c_3d(int n0, int n1, int n2, long double *in, fftwl_complex *out, unsigned flags)
    {
        return fftwl_plan_dft_r2c_3d(n0, n1, n2, in, out, flags);
    }

    fftwl_plan fft_trait<long double>::plan_many_dft_r2c(int rank, const int *n, int howmany,
                                                         long double *in, const int *inembed,
                                                         int istride, int idist,
                                                         fftwl_complex *out, const int *onembed,
                                                         int ostride, int odist,
                                                         unsigned flags)
    {
        return fftwl_plan_many_dft_r2c(rank, n, howmany,
                                       in, inembed, istride, idist,
                                       out, onembed, ostride, odist,
                                       flags);
    }

    fftwl_plan fft_trait<long double>::plan_many_dft_c2r(int rank, const int *n, int howmany,
                                                         fftwl_complex *in, const int *inembed,
                                                         int istride, int idist,
                                                         long double *out, const int *onembed,
                                                         int ostride, int odist,
                                                         unsigned flags)
    {
        return fftwl_plan_many_dft_c2r(rank, n, howmany,
                                       in, inembed, istride, idist,
                                       out, onembed, ostride, odist,
                                       flags);
    }

    fftwl_plan fft_trait<long double>::plan_r2r_1d(int n, long double *in, long double *out, fftw_r2r_kind kind, unsigned flags)
    {
        return fftwl_plan_r2r_1d(n, in, out, kind, flags);
    }

    fftwl_plan fft_trait<long double>::plan_r2r_2d(int n0, int n1, long double *in, long double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags)
    {
        return fftwl_plan_r2r_2d(
            n0,
            n1,
            in,
            out,
            kind0,
            kind1,
            flags);
    }

    fftwl_plan fft_trait<long double>::plan_r2r_3d(int n0, int n1, int n2, long double *in, long double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags)
    {
        return fftwl_plan_r2r_3d(
            n0,
            n1,
            n2,
            in,
            out,
            kind0,
            kind1,
            kind2,
            flags);
    }

    fftwl_plan fft_trait<long double>::plan_many_r2r(int rank, const int *n, int howmany,
                                                     long double *in, const int *inembed,
                                                     int istride, int idist,
                                                     long double *out, const int *onembed,
                                                     int ostride, int odist,
                                                     const fftw_r2r_kind *kind, unsigned flags)
    {
        return fftwl_plan_many_r2r(rank, n, howmany,
                                   in, inembed, istride, idist,
                                   out, onembed, ostride, odist,
                                   kind, flags);
    }

    void fft_trait<long double>::execute(fftwl_plan plan)
    {
        fftwl_execute(plan);
    }

    void fft_trait<long double>::execute_dft(fftwl_plan plan, fftwl_complex *in, fftwl_complex *out)
    {
        fftwl_execute_dft(plan, in, out);
    }

    void fft_trait<long double>::execute_dft_c2r(fftwl_plan plan, fftwl_complex *in, long double *out)
    {
        fftwl_execute_dft_c2r(plan, in, out);
    }

    void fft_trait<long double>::execute_dft_r2c(fftwl_plan plan, long double *in, fftwl_complex *out)
    {
        fftwl_execute_dft_r2c(plan, in, out);
    }

    void fft_trait<long double>::execute_r2r(fftwl_plan plan, long double *in, long double *out)
    {
        fftwl_execute_r2r(plan, in, out);
    }

    int fft_trait<long double>::import_wisdom_from_filename(const char *filename)
    {
        return fftwl_import_wisdom_from_filename(filename);
    }

    void fft_trait<long double>::export_wisdom_to_filename(const char *filename)
    {
        fftwl_export_wisdom_to_filename(filename);
    }

    char *fft_trait<long double>::export_wisdom_to_string()
    {
        return fftwl_export_wisdom_to_string();
    }

    int fft_trait<long double>::import_wisdom_from_string(const char *input_string)
    {
        return fftwl_import_wisdom_from_string(input_string);
    }

    void fft_trait<long double>::destroy_plan(fftwl_plan plan)
    {
        fftwl_destroy_plan(plan);
    }

} // namespace clapfft