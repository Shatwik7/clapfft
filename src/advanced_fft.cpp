#include <clapfft/advanced_fft.hpp>
#include <clapfft/fft_traits.hpp>
#include <fftw3.h>
#include <mutex>
#include <vector>

namespace clapfft
{

    namespace
    {
        std::mutex planner_mutex;
    }

    template <typename T>
    void AdvancedFFT::many_dft(int rank, const int *n, int howmany,
                               std::complex<T> *in, const int *inembed,
                               int istride, int idist,
                               std::complex<T> *out, const int *onembed,
                               int ostride, int odist,
                               int sign)
    {
        if (rank <= 0 || n == nullptr || in == nullptr || out == nullptr || howmany <= 0)
        {
            return;
        }

        using traits = fft_trait<T>;
        auto in_ptr = reinterpret_cast<typename traits::complex_type *>(in);
        auto out_ptr = reinterpret_cast<typename traits::complex_type *>(out);

        typename traits::plan_type plan;
        {
            std::lock_guard<std::mutex> lock(planner_mutex);
            plan = traits::plan_many_dft(rank, n, howmany,
                                         in_ptr, inembed, istride, idist,
                                         out_ptr, onembed, ostride, odist,
                                         sign, FFTW_ESTIMATE);
        }

        if (plan == nullptr)
        {
            return;
        }

        traits::execute_dft(plan, in_ptr, out_ptr);
        traits::destroy_plan(plan);
    }

    template <typename T>
    void AdvancedFFT::many_dft_r2c(int rank, const int *n, int howmany,
                                   T *in, const int *inembed,
                                   int istride, int idist,
                                   std::complex<T> *out, const int *onembed,
                                   int ostride, int odist)
    {
        if (rank <= 0 || n == nullptr || in == nullptr || out == nullptr || howmany <= 0)
        {
            return;
        }

        using traits = fft_trait<T>;
        auto out_ptr = reinterpret_cast<typename traits::complex_type *>(out);

        typename traits::plan_type plan;
        {
            std::lock_guard<std::mutex> lock(planner_mutex);
            plan = traits::plan_many_dft_r2c(rank, n, howmany,
                                             in, inembed, istride, idist,
                                             out_ptr, onembed, ostride, odist,
                                             FFTW_ESTIMATE);
        }

        if (plan == nullptr)
        {
            return;
        }

        traits::execute_dft_r2c(plan, in, out_ptr);
        traits::destroy_plan(plan);
    }

    template <typename T>
    void AdvancedFFT::many_dft_c2r(int rank, const int *n, int howmany,
                                   std::complex<T> *in, const int *inembed,
                                   int istride, int idist,
                                   T *out, const int *onembed,
                                   int ostride, int odist)
    {
        if (rank <= 0 || n == nullptr || in == nullptr || out == nullptr || howmany <= 0)
        {
            return;
        }

        using traits = fft_trait<T>;
        auto in_ptr = reinterpret_cast<typename traits::complex_type *>(in);

        typename traits::plan_type plan;
        {
            std::lock_guard<std::mutex> lock(planner_mutex);
            plan = traits::plan_many_dft_c2r(rank, n, howmany,
                                             in_ptr, inembed, istride, idist,
                                             out, onembed, ostride, odist,
                                             FFTW_ESTIMATE);
        }

        if (plan == nullptr)
        {
            return;
        }

        traits::execute_dft_c2r(plan, in_ptr, out);
        traits::destroy_plan(plan);
    }

    template <typename T>
    void AdvancedFFT::many_r2r(int rank, const int *n, int howmany,
                               T *in, const int *inembed,
                               int istride, int idist,
                               T *out, const int *onembed,
                               int ostride, int odist,
                               const int *kind)
    {
        if (rank <= 0 || n == nullptr || in == nullptr || out == nullptr || kind == nullptr || howmany <= 0)
        {
            return;
        }

        using traits = fft_trait<T>;

        typename traits::plan_type plan;
        {
            std::lock_guard<std::mutex> lock(planner_mutex);
            plan = traits::plan_many_r2r(rank, n, howmany,
                                         in, inembed, istride, idist,
                                         out, onembed, ostride, odist,
                                         kind, FFTW_ESTIMATE);
        }

        if (plan == nullptr)
        {
            return;
        }

        traits::execute_r2r(plan, in, out);
        traits::destroy_plan(plan);
    }

    template void AdvancedFFT::many_dft<float>(int, const int *, int,
                                               std::complex<float> *, const int *,
                                               int, int,
                                               std::complex<float> *, const int *,
                                               int, int,
                                               int);
    template void AdvancedFFT::many_dft<double>(int, const int *, int,
                                                std::complex<double> *, const int *,
                                                int, int,
                                                std::complex<double> *, const int *,
                                                int, int,
                                                int);
    template void AdvancedFFT::many_dft<long double>(int, const int *, int,
                                                     std::complex<long double> *, const int *,
                                                     int, int,
                                                     std::complex<long double> *, const int *,
                                                     int, int,
                                                     int);

    template void AdvancedFFT::many_dft_r2c<float>(int, const int *, int,
                                                   float *, const int *,
                                                   int, int,
                                                   std::complex<float> *, const int *,
                                                   int, int);
    template void AdvancedFFT::many_dft_r2c<double>(int, const int *, int,
                                                    double *, const int *,
                                                    int, int,
                                                    std::complex<double> *, const int *,
                                                    int, int);
    template void AdvancedFFT::many_dft_r2c<long double>(int, const int *, int,
                                                         long double *, const int *,
                                                         int, int,
                                                         std::complex<long double> *, const int *,
                                                         int, int);

    template void AdvancedFFT::many_dft_c2r<float>(int, const int *, int,
                                                   std::complex<float> *, const int *,
                                                   int, int,
                                                   float *, const int *,
                                                   int, int);
    template void AdvancedFFT::many_dft_c2r<double>(int, const int *, int,
                                                    std::complex<double> *, const int *,
                                                    int, int,
                                                    double *, const int *,
                                                    int, int);
    template void AdvancedFFT::many_dft_c2r<long double>(int, const int *, int,
                                                         std::complex<long double> *, const int *,
                                                         int, int,
                                                         long double *, const int *,
                                                         int, int);

    template void AdvancedFFT::many_r2r<float>(int, const int *, int,
                                               float *, const int *,
                                               int, int,
                                               float *, const int *,
                                               int, int,
                                               const int *);
    template void AdvancedFFT::many_r2r<double>(int, const int *, int,
                                                double *, const int *,
                                                int, int,
                                                double *, const int *,
                                                int, int,
                                                const int *);
    template void AdvancedFFT::many_r2r<long double>(int, const int *, int,
                                                     long double *, const int *,
                                                     int, int,
                                                     long double *, const int *,
                                                     int, int,
                                                     const int *);

} // namespace clapfft
