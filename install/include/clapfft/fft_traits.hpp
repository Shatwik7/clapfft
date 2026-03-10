#ifndef CLAPFFT_FFT_TRAITS_HPP
#define CLAPFFT_FFT_TRAITS_HPP

#include <dlfcn.h>
#include <limits>
#include <vector>

// FFTW type definitions - compatible with fftw3.h
#ifndef FFTW3_H
typedef float fftwf_complex[2];
typedef double fftw_complex[2];
typedef long double fftwl_complex[2];
typedef void* fftwf_plan;
typedef void* fftw_plan;
typedef void* fftwl_plan;
typedef int fftw_r2r_kind;
typedef int fftwf_r2r_kind;
typedef int fftwl_r2r_kind;
typedef struct {
    int n;
    int is;
    int os;
} fftw_iodim;
typedef struct {
    ptrdiff_t n;
    ptrdiff_t is;
    ptrdiff_t os;
} fftw_iodim64;
#endif

namespace clapfft
{

    // Traits to map T to FFTW types
    template <typename T>
    struct fft_trait;

    template <>
    struct fft_trait<float>
    {
        using complex_type = fftwf_complex;
        using plan_type = fftwf_plan;

        static plan_type plan_dft_1d(int n, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_dft_2d(int n0, int n1, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_dft_3d(int n0, int n1, int n2, complex_type *in, complex_type *out, int sign, unsigned flags);

        static plan_type plan_many_dft(int rank, const int *n, int howmany,
                                       complex_type *in, const int *inembed,
                                       int istride, int idist,
                                       complex_type *out, const int *onembed,
                                       int ostride, int odist,
                                       int sign, unsigned flags);

        static plan_type plan_dft_c2r_1d(int n, complex_type *in, float *out, unsigned flags);
        static plan_type plan_dft_c2r_2d(int n0, int n1, complex_type *in, float *out, unsigned flags);
        static plan_type plan_dft_c2r_3d(int n0, int n1, int n2, complex_type *in, float *out, unsigned flags);
        static plan_type plan_dft_r2c_1d(int n, float *in, complex_type *out, unsigned flags);
        static plan_type plan_dft_r2c_2d(int n0, int n1, float *in, complex_type *out, unsigned flags);
        static plan_type plan_dft_r2c_3d(int n0, int n1, int n2, float *in, complex_type *out, unsigned flags);

        static plan_type plan_many_dft_r2c(int rank, const int *n, int howmany,
                                           float *in, const int *inembed,
                                           int istride, int idist,
                                           complex_type *out, const int *onembed,
                                           int ostride, int odist,
                                           unsigned flags);
        static plan_type plan_many_dft_c2r(int rank, const int *n, int howmany,
                                           complex_type *in, const int *inembed,
                                           int istride, int idist,
                                           float *out, const int *onembed,
                                           int ostride, int odist,
                                           unsigned flags);

        static plan_type plan_r2r_1d(int n, float *in, float *out, fftw_r2r_kind kind, unsigned flags);
        static plan_type plan_r2r_2d(int n0, int n1, float *in, float *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags);
        static plan_type plan_r2r_3d(int n0, int n1, int n2, float *in, float *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags);
        static plan_type plan_many_r2r(int rank, const int *n, int howmany,
                                       float *in, const int *inembed,
                                       int istride, int idist,
                                       float *out, const int *onembed,
                                       int ostride, int odist,
                                       const fftw_r2r_kind *kind, unsigned flags);

        static void execute(plan_type plan);
        static void execute_dft(plan_type plan, complex_type *in, complex_type *out);
        static void execute_dft_c2r(plan_type plan, complex_type *in, float *out);
        static void execute_dft_r2c(plan_type plan, float *in, complex_type *out);
        static void execute_r2r(plan_type plan, float *in, float *out);
        static int import_wisdom_from_filename(const char *filename);
        static void export_wisdom_to_filename(const char *filename);
        static char *export_wisdom_to_string();
        static int import_wisdom_from_string(const char *input_string);
        static void destroy_plan(plan_type plan);
    };

    template <>
    struct fft_trait<double>
    {
        using complex_type = fftw_complex;
        using plan_type = fftw_plan;

        struct guru_many_layout
        {
            std::vector<int> n;
            std::vector<int> inembed;
            std::vector<int> onembed;
            int howmany = 1;
            int istride = 1;
            int ostride = 1;
            int idist = 1;
            int odist = 1;
        };

        template <typename IntLike>
        static bool to_int32_checked(IntLike value, int &out)
        {
            if (value < static_cast<IntLike>(std::numeric_limits<int>::min()) ||
                value > static_cast<IntLike>(std::numeric_limits<int>::max()))
            {
                return false;
            }
            out = static_cast<int>(value);
            return true;
        }

        template <typename IODim>
        static bool fill_embed(const IODim *dims, int rank, bool input, std::vector<int> &embed)
        {
            if (rank <= 0)
            {
                embed.clear();
                return true;
            }

            embed.resize(static_cast<std::size_t>(rank));
            for (int i = rank - 1; i >= 0; --i)
            {
                int ni = 0;
                if (!to_int32_checked(dims[i].n, ni) || ni <= 0)
                {
                    return false;
                }

                if (i == rank - 1)
                {
                    embed[static_cast<std::size_t>(i)] = ni;
                    continue;
                }

                int prev_stride = 0;
                int next_stride = 0;
                if (!to_int32_checked(input ? dims[i].is : dims[i].os, prev_stride) ||
                    !to_int32_checked(input ? dims[i + 1].is : dims[i + 1].os, next_stride))
                {
                    return false;
                }

                if (next_stride == 0 || prev_stride % next_stride != 0)
                {
                    return false;
                }

                const int computed = prev_stride / next_stride;
                if (computed < ni)
                {
                    return false;
                }
                embed[static_cast<std::size_t>(i)] = computed;
            }

            return true;
        }

        template <typename IODim>
        static bool convert_guru_dims(int rank,
                                      const IODim *dims,
                                      int howmany_rank,
                                      const IODim *howmany_dims,
                                      guru_many_layout &layout)
        {
            if (rank <= 0 || dims == nullptr || howmany_rank < 0)
            {
                return false;
            }

            layout.n.resize(static_cast<std::size_t>(rank));
            for (int i = 0; i < rank; ++i)
            {
                int ni = 0;
                if (!to_int32_checked(dims[i].n, ni) || ni <= 0)
                {
                    return false;
                }
                layout.n[static_cast<std::size_t>(i)] = ni;
            }

            if (!fill_embed(dims, rank, true, layout.inembed) || !fill_embed(dims, rank, false, layout.onembed))
            {
                return false;
            }

            if (!to_int32_checked(dims[rank - 1].is, layout.istride) ||
                !to_int32_checked(dims[rank - 1].os, layout.ostride) ||
                layout.istride <= 0 || layout.ostride <= 0)
            {
                return false;
            }

            layout.howmany = 1;
            layout.idist = 1;
            layout.odist = 1;

            if (howmany_rank == 0)
            {
                return true;
            }
            if (howmany_dims == nullptr)
            {
                return false;
            }

            if (!to_int32_checked(howmany_dims[howmany_rank - 1].is, layout.idist) ||
                !to_int32_checked(howmany_dims[howmany_rank - 1].os, layout.odist) ||
                layout.idist <= 0 || layout.odist <= 0)
            {
                return false;
            }

            for (int i = howmany_rank - 1; i >= 0; --i)
            {
                int ni = 0;
                if (!to_int32_checked(howmany_dims[i].n, ni) || ni <= 0)
                {
                    return false;
                }
                if (layout.howmany > std::numeric_limits<int>::max() / ni)
                {
                    return false;
                }
                layout.howmany *= ni;

                if (i < howmany_rank - 1)
                {
                    int this_is = 0;
                    int next_is = 0;
                    int this_os = 0;
                    int next_os = 0;
                    if (!to_int32_checked(howmany_dims[i].is, this_is) ||
                        !to_int32_checked(howmany_dims[i + 1].is, next_is) ||
                        !to_int32_checked(howmany_dims[i].os, this_os) ||
                        !to_int32_checked(howmany_dims[i + 1].os, next_os))
                    {
                        return false;
                    }
                    if (next_is == 0 || next_os == 0)
                    {
                        return false;
                    }

                    const int ratio_is = this_is / next_is;
                    const int ratio_os = this_os / next_os;
                    if (this_is % next_is != 0 || this_os % next_os != 0 || ratio_is != ni || ratio_os != ni)
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        template <typename Fn>
        static Fn resolve_fftw_next(const char *name)
        {
            return reinterpret_cast<Fn>(dlsym(RTLD_NEXT, name));
        }

        static plan_type plan_dft_1d(int n, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_dft_2d(int n0, int n1, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_dft_3d(int n0, int n1, int n2, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_many_dft(int rank, const int *n, int howmany,
                                       complex_type *in, const int *inembed,
                                       int istride, int idist,
                                       complex_type *out, const int *onembed,
                                       int ostride, int odist,
                                       int sign, unsigned flags);

        static plan_type plan_guru_dft(int rank, const fftw_iodim *dims,
                                       int howmany_rank, const fftw_iodim *howmany_dims,
                                       complex_type *in, complex_type *out,
                                       int sign, unsigned flags);
        static plan_type plan_guru_split_dft(int rank, const fftw_iodim *dims,
                                             int howmany_rank, const fftw_iodim *howmany_dims,
                                             double *ri, double *ii, double *ro, double *io,
                                             unsigned flags);
        static plan_type plan_guru_dft_r2c(int rank, const fftw_iodim *dims,
                                           int howmany_rank, const fftw_iodim *howmany_dims,
                                           double *in, complex_type *out,
                                           unsigned flags);
        static plan_type plan_guru_split_dft_r2c(int rank, const fftw_iodim *dims,
                                                 int howmany_rank, const fftw_iodim *howmany_dims,
                                                 double *in, double *ro, double *io,
                                                 unsigned flags);

        static plan_type plan_guru_dft_c2r(int rank, const fftw_iodim *dims,
                                           int howmany_rank, const fftw_iodim *howmany_dims,
                                           complex_type *in, double *out,
                                           unsigned flags);
        static plan_type plan_guru_split_dft_c2r(int rank, const fftw_iodim *dims,
                                                 int howmany_rank, const fftw_iodim *howmany_dims,
                                                 double *ri, double *ii, double *out,
                                                 unsigned flags);
        static plan_type plan_guru_r2r(int rank, const fftw_iodim *dims,
                                       int howmany_rank, const fftw_iodim *howmany_dims,
                                       double *in, double *out,
                                       const fftw_r2r_kind *kind,
                                       unsigned flags);
        static plan_type plan_guru64_dft(int rank, const fftw_iodim64 *dims,
                                         int howmany_rank, const fftw_iodim64 *howmany_dims,
                                         complex_type *in, complex_type *out,
                                         int sign, unsigned flags);

        static plan_type plan_dft_c2r_1d(int n, complex_type *in, double *out, unsigned flags);
        static plan_type plan_dft_c2r_2d(int n0, int n1, complex_type *in, double *out, unsigned flags);
        static plan_type plan_dft_c2r_3d(int n0, int n1, int n2, complex_type *in, double *out, unsigned flags);
        static plan_type plan_dft_r2c_1d(int n, double *in, complex_type *out, unsigned flags);
        static plan_type plan_dft_r2c_2d(int n0, int n1, double *in, complex_type *out, unsigned flags);
        static plan_type plan_dft_r2c_3d(int n0, int n1, int n2, double *in, complex_type *out, unsigned flags);
        static plan_type plan_many_dft_r2c(int rank, const int *n, int howmany,
                                           double *in, const int *inembed,
                                           int istride, int idist,
                                           complex_type *out, const int *onembed,
                                           int ostride, int odist,
                                           unsigned flags);
        static plan_type plan_many_dft_c2r(int rank, const int *n, int howmany,
                                           complex_type *in, const int *inembed,
                                           int istride, int idist,
                                           double *out, const int *onembed,
                                           int ostride, int odist,
                                           unsigned flags);

        static plan_type plan_r2r_1d(int n, double *in, double *out, fftw_r2r_kind kind, unsigned flags);
        static plan_type plan_r2r_2d(int n0, int n1, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags);
        static plan_type plan_r2r_3d(int n0, int n1, int n2, double *in, double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags);
        static plan_type plan_many_r2r(int rank, const int *n, int howmany,
                                       double *in, const int *inembed,
                                       int istride, int idist,
                                       double *out, const int *onembed,
                                       int ostride, int odist,
                                       const fftw_r2r_kind *kind, unsigned flags);
        static void execute(plan_type plan);
        static void execute_dft(plan_type plan, complex_type *in, complex_type *out);
        static void execute_dft_c2r(plan_type plan, complex_type *in, double *out);

        static void execute_dft_r2c(plan_type plan, double *in, complex_type *out);
        static void execute_r2r(plan_type plan, double *in, double *out);
        static int import_wisdom_from_filename(const char *filename);
        static void export_wisdom_to_filename(const char *filename);
        static char *export_wisdom_to_string();
        static int import_wisdom_from_string(const char *input_string);
        static void destroy_plan(plan_type plan);
    };

    template <>
    struct fft_trait<long double>
    {
        using complex_type = fftwl_complex;
        using plan_type = fftwl_plan;

        static plan_type plan_dft_1d(int n, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_dft_2d(int n0, int n1, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_dft_3d(int n0, int n1, int n2, complex_type *in, complex_type *out, int sign, unsigned flags);
        static plan_type plan_many_dft(int rank, const int *n, int howmany,
                                       complex_type *in, const int *inembed,
                                       int istride, int idist,
                                       complex_type *out, const int *onembed,
                                       int ostride, int odist,
                                       int sign, unsigned flags);
        static plan_type plan_dft_c2r_1d(int n, complex_type *in, long double *out, unsigned flags);
        static plan_type plan_dft_c2r_2d(int n0, int n1, complex_type *in, long double *out, unsigned flags);
        static plan_type plan_dft_c2r_3d(int n0, int n1, int n2, complex_type *in, long double *out, unsigned flags);
        static plan_type plan_dft_r2c_1d(int n, long double *in, complex_type *out, unsigned flags);
        static plan_type plan_dft_r2c_2d(int n0, int n1, long double *in, complex_type *out, unsigned flags);
        static plan_type plan_dft_r2c_3d(int n0, int n1, int n2, long double *in, complex_type *out, unsigned flags);
        static plan_type plan_many_dft_r2c(int rank, const int *n, int howmany,
                                           long double *in, const int *inembed,
                                           int istride, int idist,
                                           complex_type *out, const int *onembed,
                                           int ostride, int odist,
                                           unsigned flags);

        static plan_type plan_many_dft_c2r(int rank, const int *n, int howmany,
                                           complex_type *in, const int *inembed,
                                           int istride, int idist,
                                           long double *out, const int *onembed,
                                           int ostride, int odist,
                                           unsigned flags);
        static plan_type plan_r2r_1d(int n, long double *in, long double *out, fftw_r2r_kind kind, unsigned flags);
        static plan_type plan_r2r_2d(int n0, int n1, long double *in, long double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, unsigned flags);
        static plan_type plan_r2r_3d(int n0, int n1, int n2, long double *in, long double *out, fftw_r2r_kind kind0, fftw_r2r_kind kind1, fftw_r2r_kind kind2, unsigned flags);
        static plan_type plan_many_r2r(int rank, const int *n, int howmany,
                                       long double *in, const int *inembed,
                                       int istride, int idist,
                                       long double *out, const int *onembed,
                                       int ostride, int odist,
                                       const fftw_r2r_kind *kind, unsigned flags);

        static void execute(plan_type plan);
        static void execute_dft(plan_type plan, complex_type *in, complex_type *out);
        static void execute_dft_c2r(plan_type plan, complex_type *in, long double *out);
        static void execute_dft_r2c(plan_type plan, long double *in, complex_type *out);
        static void execute_r2r(plan_type plan, long double *in, long double *out);
        static int import_wisdom_from_filename(const char *filename);
        static void export_wisdom_to_filename(const char *filename);
        static char *export_wisdom_to_string();
        static int import_wisdom_from_string(const char *input_string);
        static void destroy_plan(plan_type plan);
    };

} // namespace clapfft

#endif // CLAPFFT_FFT_TRAITS_HPP
