#include <clapfft/guru_fft.hpp>
#include <clapfft/fft_traits.hpp>
#include <mutex>

namespace
{
    std::mutex planner_mutex;

    bool has_valid_howmany(int howmany_rank, const void *howmany_dims)
    {
        if (howmany_rank < 0)
        {
            return false;
        }
        if (howmany_rank > 0 && howmany_dims == nullptr)
        {
            return false;
        }
        return true;
    }
}

namespace clapfft
{
    plan_type GuruFFT::plan_guru_dft(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        fftw_complex *in, fftw_complex *out,
        int sign, unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || in == nullptr || out == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
    }

    plan_type GuruFFT::plan_guru_split_dft(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *ri, double *ii, double *ro, double *io,
        unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || ri == nullptr || ii == nullptr || ro == nullptr || io == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru_split_dft(rank, dims, howmany_rank, howmany_dims, ri, ii, ro, io, flags);
    }

    plan_type GuruFFT::plan_guru_dft_r2c(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *in, fftw_complex *out,
        unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || in == nullptr || out == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }

    plan_type GuruFFT::plan_guru_split_dft_r2c(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *in, double *ro, double *io,
        unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || in == nullptr || ro == nullptr || io == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru_split_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, ro, io, flags);
    }

    plan_type GuruFFT::plan_guru_dft_c2r(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        fftw_complex *in, double *out,
        unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || in == nullptr || out == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags);
    }

    plan_type GuruFFT::plan_guru_split_dft_c2r(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *ri, double *ii, double *out,
        unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || ri == nullptr || ii == nullptr || out == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru_split_dft_c2r(rank, dims, howmany_rank, howmany_dims, ri, ii, out, flags);
    }

    plan_type GuruFFT::plan_guru_r2r(
        int rank, const fftw_iodim *dims,
        int howmany_rank,
        const fftw_iodim *howmany_dims,
        double *in, double *out,
        const fftw_r2r_kind *kind,
        unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || in == nullptr || out == nullptr || kind == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru_r2r(rank, dims, howmany_rank, howmany_dims, in, out, kind, flags);
    }

    plan_type GuruFFT::plan_guru64_dft(
        int rank, const fftw_iodim64 *dims,
        int howmany_rank, const fftw_iodim64 *howmany_dims,
        fftw_complex *in, fftw_complex *out,
        int sign, unsigned flags)
    {
        if (rank <= 0 || dims == nullptr || in == nullptr || out == nullptr || !has_valid_howmany(howmany_rank, howmany_dims))
        {
            return nullptr;
        }

        using traits = fft_trait<double>;
        std::lock_guard<std::mutex> lock(planner_mutex);
        return traits::plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags);
    }
}

extern "C"
{
    fftw_plan fftw_plan_guru_dft(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        fftw_complex *in, fftw_complex *out,
        int sign, unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags));
    }

    fftw_plan fftw_plan_guru_split_dft(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *ri, double *ii, double *ro, double *io,
        unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru_split_dft(rank, dims, howmany_rank, howmany_dims, ri, ii, ro, io, flags));
    }

    fftw_plan fftw_plan_guru_dft_r2c(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *in, fftw_complex *out,
        unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, out, flags));
    }

    fftw_plan fftw_plan_guru_split_dft_r2c(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *in, double *ro, double *io,
        unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru_split_dft_r2c(rank, dims, howmany_rank, howmany_dims, in, ro, io, flags));
    }

    fftw_plan fftw_plan_guru_dft_c2r(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        fftw_complex *in, double *out,
        unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru_dft_c2r(rank, dims, howmany_rank, howmany_dims, in, out, flags));
    }

    fftw_plan fftw_plan_guru_split_dft_c2r(
        int rank, const fftw_iodim *dims,
        int howmany_rank, const fftw_iodim *howmany_dims,
        double *ri, double *ii, double *out,
        unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru_split_dft_c2r(rank, dims, howmany_rank, howmany_dims, ri, ii, out, flags));
    }

    fftw_plan fftw_plan_guru_r2r(int rank, const fftw_iodim *dims,
                                 int howmany_rank,
                                 const fftw_iodim *howmany_dims,
                                 double *in, double *out,
                                 const fftw_r2r_kind *kind,
                                 unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru_r2r(rank, dims, howmany_rank, howmany_dims, in, out, kind, flags));
    }

    fftw_plan fftw_plan_guru64_dft(
        int rank, const fftw_iodim64 *dims,
        int howmany_rank, const fftw_iodim64 *howmany_dims,
        fftw_complex *in, fftw_complex *out,
        int sign, unsigned flags)
    {
        return reinterpret_cast<fftw_plan>(clapfft::GuruFFT::plan_guru64_dft(rank, dims, howmany_rank, howmany_dims, in, out, sign, flags));
    }
}
