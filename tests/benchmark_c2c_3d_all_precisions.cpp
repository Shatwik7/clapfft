#include <fftw3.h>
#include <clapfft/clapfft_api.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{
    struct BenchmarkConfig
    {
        int n0 = 32;
        int n1 = 32;
        int n2 = 32;
        int warmup = 3;
        int iters = 50;
    };

    BenchmarkConfig parse_args(int argc, char **argv)
    {
        BenchmarkConfig cfg;
        if (argc > 1)
            cfg.n0 = std::max(2, std::atoi(argv[1]));
        if (argc > 2)
            cfg.n1 = std::max(2, std::atoi(argv[2]));
        if (argc > 3)
            cfg.n2 = std::max(2, std::atoi(argv[3]));
        if (argc > 4)
            cfg.iters = std::max(1, std::atoi(argv[4]));
        if (argc > 5)
            cfg.warmup = std::max(0, std::atoi(argv[5]));
        return cfg;
    }

    template <typename T>
    struct FFTWTraits;

    template <>
    struct FFTWTraits<float>
    {
        using ComplexType = fftwf_complex;
        using PlanType = fftwf_plan;

        static PlanType plan_3d(int n0, int n1, int n2, ComplexType *in, ComplexType *out, int sign, unsigned flags)
        {
            return fftwf_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
        }

        static void execute_dft(PlanType p, ComplexType *in, ComplexType *out)
        {
            fftwf_execute_dft(p, in, out);
        }

        static void destroy(PlanType p)
        {
            fftwf_destroy_plan(p);
        }
    };

    template <>
    struct FFTWTraits<double>
    {
        using ComplexType = fftw_complex;
        using PlanType = fftw_plan;

        static PlanType plan_3d(int n0, int n1, int n2, ComplexType *in, ComplexType *out, int sign, unsigned flags)
        {
            return fftw_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
        }

        static void execute_dft(PlanType p, ComplexType *in, ComplexType *out)
        {
            fftw_execute_dft(p, in, out);
        }

        static void destroy(PlanType p)
        {
            fftw_destroy_plan(p);
        }
    };

    template <>
    struct FFTWTraits<long double>
    {
        using ComplexType = fftwl_complex;
        using PlanType = fftwl_plan;

        static PlanType plan_3d(int n0, int n1, int n2, ComplexType *in, ComplexType *out, int sign, unsigned flags)
        {
            return fftwl_plan_dft_3d(n0, n1, n2, in, out, sign, flags);
        }

        static void execute_dft(PlanType p, ComplexType *in, ComplexType *out)
        {
            fftwl_execute_dft(p, in, out);
        }

        static void destroy(PlanType p)
        {
            fftwl_destroy_plan(p);
        }
    };

    template <typename T>
    std::size_t point_count(const BenchmarkConfig &cfg)
    {
        (void)sizeof(T);
        return static_cast<std::size_t>(cfg.n0) * static_cast<std::size_t>(cfg.n1) * static_cast<std::size_t>(cfg.n2);
    }

    template <typename T>
    std::vector<std::complex<T>> make_input(const BenchmarkConfig &cfg)
    {
        std::vector<std::complex<T>> input(point_count<T>(cfg));
        std::mt19937 rng(123456);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            input[i] = std::complex<T>(static_cast<T>(dist(rng)), static_cast<T>(dist(rng)));
        }
        return input;
    }

    template <typename T>
    std::vector<std::vector<std::vector<std::complex<T>>>> to_nested_3d(const std::vector<std::complex<T>> &flat, const BenchmarkConfig &cfg)
    {
        std::vector<std::vector<std::vector<std::complex<T>>>> nested(
            static_cast<std::size_t>(cfg.n0),
            std::vector<std::vector<std::complex<T>>>(
                static_cast<std::size_t>(cfg.n1),
                std::vector<std::complex<T>>(static_cast<std::size_t>(cfg.n2))));

        for (int i = 0; i < cfg.n0; ++i)
        {
            for (int j = 0; j < cfg.n1; ++j)
            {
                for (int k = 0; k < cfg.n2; ++k)
                {
                    const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(cfg.n1) * static_cast<std::size_t>(cfg.n2) +
                                            static_cast<std::size_t>(j) * static_cast<std::size_t>(cfg.n2) +
                                            static_cast<std::size_t>(k);
                    nested[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] = flat[idx];
                }
            }
        }
        return nested;
    }

    template <typename T>
    std::vector<std::complex<T>> flatten_3d(const std::vector<std::vector<std::vector<std::complex<T>>>> &nested, const BenchmarkConfig &cfg)
    {
        std::vector<std::complex<T>> flat(point_count<T>(cfg));
        for (int i = 0; i < cfg.n0; ++i)
        {
            for (int j = 0; j < cfg.n1; ++j)
            {
                for (int k = 0; k < cfg.n2; ++k)
                {
                    const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(cfg.n1) * static_cast<std::size_t>(cfg.n2) +
                                            static_cast<std::size_t>(j) * static_cast<std::size_t>(cfg.n2) +
                                            static_cast<std::size_t>(k);
                    flat[idx] = nested[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)];
                }
            }
        }
        return flat;
    }

    template <typename T>
    void normalize_roundtrip(std::vector<std::complex<T>> &data, const BenchmarkConfig &cfg)
    {
        const T scale = static_cast<T>(cfg.n0) * static_cast<T>(cfg.n1) * static_cast<T>(cfg.n2);
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            data[i] /= scale;
        }
    }

    template <typename T>
    T max_abs_diff(const std::vector<std::complex<T>> &a, const std::vector<std::complex<T>> &b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        T max_err = static_cast<T>(0);
        for (std::size_t i = 0; i < n; ++i)
        {
            const T err = std::abs(a[i] - b[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    template <typename T>
    double benchmark_clapfft(const std::vector<std::complex<T>> &input, const BenchmarkConfig &cfg, std::vector<std::complex<T>> &normalized_out)
    {
        using clock = std::chrono::steady_clock;

        const auto nested_input = to_nested_3d(input, cfg);
        std::vector<std::vector<std::vector<std::complex<T>>>> spectrum;
        std::vector<std::vector<std::vector<std::complex<T>>>> recovered;

        for (int i = 0; i < cfg.warmup; ++i)
        {
            clapfft::FFT::c2c_3d(nested_input, spectrum, FFTW_FORWARD);
            clapfft::FFT::c2c_3d(spectrum, recovered, FFTW_BACKWARD);
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            clapfft::FFT::c2c_3d(nested_input, spectrum, FFTW_FORWARD);
            clapfft::FFT::c2c_3d(spectrum, recovered, FFTW_BACKWARD);
        }
        const auto end = clock::now();

        normalized_out = flatten_3d(recovered, cfg);
        normalize_roundtrip(normalized_out, cfg);

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    template <typename T>
    double benchmark_fftw(const std::vector<std::complex<T>> &input, const BenchmarkConfig &cfg, std::vector<std::complex<T>> &normalized_out)
    {
        using clock = std::chrono::steady_clock;
        using Traits = FFTWTraits<T>;

        std::vector<std::complex<T>> dummy_in(point_count<T>(cfg));
        std::vector<std::complex<T>> dummy_tmp(point_count<T>(cfg));
        std::vector<std::complex<T>> dummy_out(point_count<T>(cfg));

        typename Traits::PlanType forward_plan = Traits::plan_3d(
            cfg.n0,
            cfg.n1,
            cfg.n2,
            reinterpret_cast<typename Traits::ComplexType *>(dummy_in.data()),
            reinterpret_cast<typename Traits::ComplexType *>(dummy_tmp.data()),
            FFTW_FORWARD,
            FFTW_MEASURE | FFTW_UNALIGNED);

        typename Traits::PlanType backward_plan = Traits::plan_3d(
            cfg.n0,
            cfg.n1,
            cfg.n2,
            reinterpret_cast<typename Traits::ComplexType *>(dummy_tmp.data()),
            reinterpret_cast<typename Traits::ComplexType *>(dummy_out.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE | FFTW_UNALIGNED);

        std::vector<std::complex<T>> in = input;
        std::vector<std::complex<T>> spectrum(point_count<T>(cfg));
        std::vector<std::complex<T>> recovered(point_count<T>(cfg));

        for (int i = 0; i < cfg.warmup; ++i)
        {
            Traits::execute_dft(
                forward_plan,
                reinterpret_cast<typename Traits::ComplexType *>(in.data()),
                reinterpret_cast<typename Traits::ComplexType *>(spectrum.data()));
            Traits::execute_dft(
                backward_plan,
                reinterpret_cast<typename Traits::ComplexType *>(spectrum.data()),
                reinterpret_cast<typename Traits::ComplexType *>(recovered.data()));
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            Traits::execute_dft(
                forward_plan,
                reinterpret_cast<typename Traits::ComplexType *>(in.data()),
                reinterpret_cast<typename Traits::ComplexType *>(spectrum.data()));
            Traits::execute_dft(
                backward_plan,
                reinterpret_cast<typename Traits::ComplexType *>(spectrum.data()),
                reinterpret_cast<typename Traits::ComplexType *>(recovered.data()));
        }
        const auto end = clock::now();

        Traits::destroy(forward_plan);
        Traits::destroy(backward_plan);

        normalized_out = recovered;
        normalize_roundtrip(normalized_out, cfg);

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    template <typename T>
    void run_precision(const std::string &label, const BenchmarkConfig &cfg)
    {
        const std::vector<std::complex<T>> input = make_input<T>(cfg);
        std::vector<std::complex<T>> clapfft_out;
        std::vector<std::complex<T>> fftw_out;

        const double clapfft_ms = benchmark_clapfft(input, cfg, clapfft_out);
        const double fftw_ms = benchmark_fftw(input, cfg, fftw_out);

        const T err_cf = max_abs_diff(clapfft_out, fftw_out);
        const T err_ci = max_abs_diff(clapfft_out, input);
        const T err_fi = max_abs_diff(fftw_out, input);

        const double clapfft_per_iter = clapfft_ms / static_cast<double>(cfg.iters);
        const double fftw_per_iter = fftw_ms / static_cast<double>(cfg.iters);
        const double ratio = fftw_per_iter > 0.0 ? clapfft_per_iter / fftw_per_iter : 0.0;

        std::cout << "[" << label << "]\n";
        std::cout << std::fixed << std::setprecision(6);
        std::cout << "clapfft total (ms): " << clapfft_ms << "\n";
        std::cout << "fftw   total (ms): " << fftw_ms << "\n";
        std::cout << "clapfft per iter (ms): " << clapfft_per_iter << "\n";
        std::cout << "fftw   per iter (ms): " << fftw_per_iter << "\n";
        std::cout << "ratio (clapfft/fftw): " << ratio << "x\n";
        std::cout << std::scientific << std::setprecision(4);
        std::cout << "max |clapfft - fftw| after normalization: " << static_cast<double>(err_cf) << "\n";
        std::cout << "max |clapfft - input| after normalization: " << static_cast<double>(err_ci) << "\n";
        std::cout << "max |fftw   - input| after normalization: " << static_cast<double>(err_fi) << "\n\n";
    }
}

int main(int argc, char **argv)
{
    const BenchmarkConfig cfg = parse_args(argc, argv);

    std::cout << "Benchmark: clapfft vs FFTW (c2c 3D, all precisions; forward+backward)\n";
    std::cout << "Dims=" << cfg.n0 << "x" << cfg.n1 << "x" << cfg.n2
              << ", iterations=" << cfg.iters << ", warmup=" << cfg.warmup << "\n\n";

    run_precision<float>("float", cfg);
    run_precision<double>("double", cfg);
    run_precision<long double>("long double", cfg);

    return 0;
}
