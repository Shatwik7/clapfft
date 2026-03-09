#include <fftw3.h>
#include <clapfft/clapfft_api.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

namespace
{
    using Real = float;

    struct BenchmarkConfig
    {
        int n0 = 32;
        int n1 = 32;
        int n2 = 32;
        int warmup = 5;
        int iters = 100;
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

    std::size_t point_count(const BenchmarkConfig &cfg)
    {
        return static_cast<std::size_t>(cfg.n0) * static_cast<std::size_t>(cfg.n1) * static_cast<std::size_t>(cfg.n2);
    }

    std::vector<Real> make_input_flat(const BenchmarkConfig &cfg)
    {
        std::vector<Real> input(point_count(cfg));
        std::mt19937 rng(20260305);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

        for (std::size_t i = 0; i < input.size(); ++i)
        {
            input[i] = dist(rng);
        }
        return input;
    }

    std::vector<std::vector<std::vector<Real>>> to_nested(const std::vector<Real> &flat, const BenchmarkConfig &cfg)
    {
        std::vector<std::vector<std::vector<Real>>> nested(
            static_cast<std::size_t>(cfg.n0),
            std::vector<std::vector<Real>>(
                static_cast<std::size_t>(cfg.n1),
                std::vector<Real>(static_cast<std::size_t>(cfg.n2))));

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

    std::vector<Real> flatten(const std::vector<std::vector<std::vector<Real>>> &nested, const BenchmarkConfig &cfg)
    {
        std::vector<Real> flat(point_count(cfg));
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

    void normalize(std::vector<Real> &data, const BenchmarkConfig &cfg)
    {
        const Real scale = static_cast<Real>((2 * cfg.n0) * (2 * cfg.n1) * (2 * cfg.n2));
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            data[i] /= scale;
        }
    }

    Real max_abs_diff(const std::vector<Real> &a, const std::vector<Real> &b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        Real max_err = 0.0f;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Real err = std::abs(a[i] - b[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    double benchmark_clapfft(const std::vector<Real> &input_flat, const BenchmarkConfig &cfg, std::vector<Real> &normalized_out)
    {
        using clock = std::chrono::steady_clock;

        const auto input_nested = to_nested(input_flat, cfg);
        std::vector<std::vector<std::vector<Real>>> forward;
        std::vector<std::vector<std::vector<Real>>> recovered;

        for (int i = 0; i < cfg.warmup; ++i)
        {
            clapfft::FFT::r2r_3d(input_nested, forward, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10);
            clapfft::FFT::r2r_3d(forward, recovered, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01);
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            clapfft::FFT::r2r_3d(input_nested, forward, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10);
            clapfft::FFT::r2r_3d(forward, recovered, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01);
        }
        const auto end = clock::now();

        normalized_out = flatten(recovered, cfg);
        normalize(normalized_out, cfg);

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    double benchmark_fftw(const std::vector<Real> &input_flat, const BenchmarkConfig &cfg, std::vector<Real> &normalized_out)
    {
        using clock = std::chrono::steady_clock;

        std::vector<Real> dummy_in(point_count(cfg));
        std::vector<Real> dummy_tmp(point_count(cfg));
        std::vector<Real> dummy_out(point_count(cfg));

        fftwf_plan forward_plan = fftwf_plan_r2r_3d(
            cfg.n0,
            cfg.n1,
            cfg.n2,
            dummy_in.data(),
            dummy_tmp.data(),
            FFTW_REDFT10,
            FFTW_REDFT10,
            FFTW_REDFT10,
            FFTW_MEASURE | FFTW_UNALIGNED);

        fftwf_plan backward_plan = fftwf_plan_r2r_3d(
            cfg.n0,
            cfg.n1,
            cfg.n2,
            dummy_tmp.data(),
            dummy_out.data(),
            FFTW_REDFT01,
            FFTW_REDFT01,
            FFTW_REDFT01,
            FFTW_MEASURE | FFTW_UNALIGNED);

        std::vector<Real> in = input_flat;
        std::vector<Real> tmp(point_count(cfg));
        std::vector<Real> recovered(point_count(cfg));

        for (int i = 0; i < cfg.warmup; ++i)
        {
            std::copy(input_flat.begin(), input_flat.end(), in.begin());
            fftwf_execute_r2r(forward_plan, in.data(), tmp.data());
            fftwf_execute_r2r(backward_plan, tmp.data(), recovered.data());
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            std::copy(input_flat.begin(), input_flat.end(), in.begin());
            fftwf_execute_r2r(forward_plan, in.data(), tmp.data());
            fftwf_execute_r2r(backward_plan, tmp.data(), recovered.data());
        }
        const auto end = clock::now();

        fftwf_destroy_plan(forward_plan);
        fftwf_destroy_plan(backward_plan);

        normalized_out = recovered;
        normalize(normalized_out, cfg);

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }
}

int main(int argc, char **argv)
{
    const BenchmarkConfig cfg = parse_args(argc, argv);
    const std::vector<Real> input = make_input_flat(cfg);

    std::vector<Real> clapfft_out;
    std::vector<Real> fftw_out;

    const double clapfft_ms = benchmark_clapfft(input, cfg, clapfft_out);
    const double fftw_ms = benchmark_fftw(input, cfg, fftw_out);

    const float clapfft_vs_fftw = max_abs_diff(clapfft_out, fftw_out);
    const float clapfft_vs_input = max_abs_diff(clapfft_out, input);
    const float fftw_vs_input = max_abs_diff(fftw_out, input);

    const double clapfft_per_iter = clapfft_ms / static_cast<double>(cfg.iters);
    const double fftw_per_iter = fftw_ms / static_cast<double>(cfg.iters);
    const double ratio = fftw_per_iter > 0.0 ? clapfft_per_iter / fftw_per_iter : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Benchmark: clapfft vs FFTW (r2r 3D, float; REDFT10->REDFT01)\n";
    std::cout << "Dims=" << cfg.n0 << "x" << cfg.n1 << "x" << cfg.n2
              << ", iterations=" << cfg.iters << ", warmup=" << cfg.warmup << "\n\n";
    std::cout << "clapfft total (ms): " << clapfft_ms << "\n";
    std::cout << "fftw   total (ms): " << fftw_ms << "\n";
    std::cout << "clapfft per iter (ms): " << clapfft_per_iter << "\n";
    std::cout << "fftw   per iter (ms): " << fftw_per_iter << "\n";
    std::cout << "ratio (clapfft/fftw): " << ratio << "x\n";

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "max |clapfft - fftw| after normalization: " << clapfft_vs_fftw << "\n";
    std::cout << "max |clapfft - input| after normalization: " << clapfft_vs_input << "\n";
    std::cout << "max |fftw   - input| after normalization: " << fftw_vs_input << "\n";

    return 0;
}
