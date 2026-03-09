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
#include <vector>

namespace
{
    using Real = long double;
    using Complex = std::complex<Real>;

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

    std::size_t real_count(const BenchmarkConfig &cfg)
    {
        return static_cast<std::size_t>(cfg.n0) * static_cast<std::size_t>(cfg.n1) * static_cast<std::size_t>(cfg.n2);
    }

    std::size_t complex_count(const BenchmarkConfig &cfg)
    {
        return static_cast<std::size_t>(cfg.n0) * static_cast<std::size_t>(cfg.n1) * static_cast<std::size_t>(cfg.n2 / 2 + 1);
    }

    std::vector<Real> make_real_input(const BenchmarkConfig &cfg)
    {
        std::vector<Real> in(real_count(cfg));
        std::mt19937_64 rng(424242);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (std::size_t i = 0; i < in.size(); ++i)
        {
            in[i] = static_cast<Real>(dist(rng));
        }
        return in;
    }

    std::vector<Complex> make_hermitian_spectrum_from_real(const std::vector<Real> &real_input, const BenchmarkConfig &cfg)
    {
        std::vector<Real> in = real_input;
        std::vector<Complex> spec(complex_count(cfg));

        std::vector<Real> dummy_real(real_count(cfg));
        std::vector<Complex> dummy_spec(complex_count(cfg));
        fftwl_plan r2c_plan = fftwl_plan_dft_r2c_3d(
            cfg.n0,
            cfg.n1,
            cfg.n2,
            dummy_real.data(),
            reinterpret_cast<fftwl_complex *>(dummy_spec.data()),
            FFTW_MEASURE | FFTW_UNALIGNED);

        fftwl_execute_dft_r2c(
            r2c_plan,
            in.data(),
            reinterpret_cast<fftwl_complex *>(spec.data()));
        fftwl_destroy_plan(r2c_plan);
        return spec;
    }

    std::vector<std::vector<std::vector<Complex>>> to_nested_spectrum(const std::vector<Complex> &flat, const BenchmarkConfig &cfg)
    {
        const int n2c = cfg.n2 / 2 + 1;
        std::vector<std::vector<std::vector<Complex>>> nested(
            static_cast<std::size_t>(cfg.n0),
            std::vector<std::vector<Complex>>(
                static_cast<std::size_t>(cfg.n1),
                std::vector<Complex>(static_cast<std::size_t>(n2c))));

        for (int i = 0; i < cfg.n0; ++i)
        {
            for (int j = 0; j < cfg.n1; ++j)
            {
                for (int k = 0; k < n2c; ++k)
                {
                    const std::size_t idx = static_cast<std::size_t>(i) * static_cast<std::size_t>(cfg.n1) * static_cast<std::size_t>(n2c) +
                                            static_cast<std::size_t>(j) * static_cast<std::size_t>(n2c) +
                                            static_cast<std::size_t>(k);
                    nested[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] = flat[idx];
                }
            }
        }
        return nested;
    }

    std::vector<Real> flatten_real_3d(const std::vector<std::vector<std::vector<Real>>> &nested, const BenchmarkConfig &cfg)
    {
        std::vector<Real> flat(real_count(cfg));
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
        const Real scale = static_cast<Real>(cfg.n0) * static_cast<Real>(cfg.n1) * static_cast<Real>(cfg.n2);
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            data[i] /= scale;
        }
    }

    Real max_abs_diff(const std::vector<Real> &a, const std::vector<Real> &b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        Real max_err = 0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Real err = std::abs(a[i] - b[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    double benchmark_clapfft_c2r_3d(const std::vector<Complex> &flat_spectrum, const BenchmarkConfig &cfg, std::vector<Real> &normalized_out)
    {
        using clock = std::chrono::steady_clock;

        std::vector<std::vector<std::vector<Real>>> recovered;
        const std::vector<std::vector<std::vector<Complex>>> nested_spectrum = to_nested_spectrum(flat_spectrum, cfg);

        for (int i = 0; i < cfg.warmup; ++i)
        {
            clapfft::FFT::c2r_3d(nested_spectrum, recovered);
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            clapfft::FFT::c2r_3d(nested_spectrum, recovered);
        }
        const auto end = clock::now();

        normalized_out = flatten_real_3d(recovered, cfg);
        normalize(normalized_out, cfg);

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    double benchmark_fftwl_c2r_3d(const std::vector<Complex> &flat_spectrum, const BenchmarkConfig &cfg, std::vector<Real> &normalized_out)
    {
        using clock = std::chrono::steady_clock;

        std::vector<Complex> dummy_spec(complex_count(cfg));
        std::vector<Real> dummy_real(real_count(cfg));
        fftwl_plan c2r_plan = fftwl_plan_dft_c2r_3d(
            cfg.n0,
            cfg.n1,
            cfg.n2,
            reinterpret_cast<fftwl_complex *>(dummy_spec.data()),
            dummy_real.data(),
            FFTW_MEASURE | FFTW_UNALIGNED);

        std::vector<Complex> in_spec = flat_spectrum;
        std::vector<Real> recovered(real_count(cfg));

        for (int i = 0; i < cfg.warmup; ++i)
        {
            std::copy(flat_spectrum.begin(), flat_spectrum.end(), in_spec.begin());
            fftwl_execute_dft_c2r(
                c2r_plan,
                reinterpret_cast<fftwl_complex *>(in_spec.data()),
                recovered.data());
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            std::copy(flat_spectrum.begin(), flat_spectrum.end(), in_spec.begin());
            fftwl_execute_dft_c2r(
                c2r_plan,
                reinterpret_cast<fftwl_complex *>(in_spec.data()),
                recovered.data());
        }
        const auto end = clock::now();

        fftwl_destroy_plan(c2r_plan);

        normalized_out = recovered;
        normalize(normalized_out, cfg);

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }
}

int main(int argc, char **argv)
{
    const BenchmarkConfig cfg = parse_args(argc, argv);

    const std::vector<Real> real_input = make_real_input(cfg);
    const std::vector<Complex> spectrum = make_hermitian_spectrum_from_real(real_input, cfg);

    std::vector<Real> clapfft_out;
    std::vector<Real> fftwl_out;

    const double clapfft_ms = benchmark_clapfft_c2r_3d(spectrum, cfg, clapfft_out);
    const double fftwl_ms = benchmark_fftwl_c2r_3d(spectrum, cfg, fftwl_out);

    const Real clapfft_vs_fftwl = max_abs_diff(clapfft_out, fftwl_out);
    const Real clapfft_vs_input = max_abs_diff(clapfft_out, real_input);
    const Real fftwl_vs_input = max_abs_diff(fftwl_out, real_input);

    const double clapfft_per_iter = clapfft_ms / static_cast<double>(cfg.iters);
    const double fftwl_per_iter = fftwl_ms / static_cast<double>(cfg.iters);
    const double ratio = fftwl_per_iter > 0.0 ? clapfft_per_iter / fftwl_per_iter : 0.0;

    std::cout << std::fixed << std::setprecision(6);
    std::cout << "Benchmark: clapfft vs FFTW (c2r 3D, long double)\n";
    std::cout << "Dims=" << cfg.n0 << "x" << cfg.n1 << "x" << cfg.n2
              << ", iterations=" << cfg.iters << ", warmup=" << cfg.warmup << "\n\n";
    std::cout << "clapfft total (ms): " << clapfft_ms << "\n";
    std::cout << "fftwl  total (ms): " << fftwl_ms << "\n";
    std::cout << "clapfft per iter (ms): " << clapfft_per_iter << "\n";
    std::cout << "fftwl  per iter (ms): " << fftwl_per_iter << "\n";
    std::cout << "ratio (clapfft/fftwl): " << ratio << "x\n";

    std::cout << std::scientific << std::setprecision(4);
    std::cout << "max |clapfft - fftwl| after normalization: " << static_cast<double>(clapfft_vs_fftwl) << "\n";
    std::cout << "max |clapfft - input| after normalization: " << static_cast<double>(clapfft_vs_input) << "\n";
    std::cout << "max |fftwl  - input| after normalization: " << static_cast<double>(fftwl_vs_input) << "\n";

    return 0;
}
