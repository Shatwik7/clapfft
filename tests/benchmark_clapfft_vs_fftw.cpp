#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>

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
    using Complex = std::complex<double>;

    struct BenchmarkConfig
    {
        int n = 1 << 16;
        int warmup = 5;
        int iters = 100;
    };

    BenchmarkConfig parse_args(int argc, char **argv)
    {
        BenchmarkConfig cfg;

        if (argc > 1)
        {
            cfg.n = std::max(2, std::atoi(argv[1]));
        }
        if (argc > 2)
        {
            cfg.iters = std::max(1, std::atoi(argv[2]));
        }
        if (argc > 3)
        {
            cfg.warmup = std::max(0, std::atoi(argv[3]));
        }
        return cfg;
    }

    std::vector<Complex> make_input(int n)
    {
        std::vector<Complex> data(static_cast<std::size_t>(n));
        std::mt19937 rng(12345);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (int i = 0; i < n; ++i)
        {
            data[static_cast<std::size_t>(i)] = Complex(dist(rng), dist(rng));
        }
        return data;
    }

    double max_abs_diff(const std::vector<Complex> &a, const std::vector<Complex> &b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        double max_err = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const double err = std::abs(a[i] - b[i]);
            if (err > max_err)
            {
                max_err = err;
            }
        }
        return max_err;
    }

    double benchmark_clapfft(const std::vector<Complex> &input, const BenchmarkConfig &cfg, std::vector<Complex> &last_output)
    {
        using clock = std::chrono::steady_clock;

        std::vector<Complex> spectrum;
        std::vector<Complex> recovered;

        for (int i = 0; i < cfg.warmup; ++i)
        {
            clapfft::FFT::c2c_1d(input, spectrum, FFTW_FORWARD);
            clapfft::FFT::c2c_1d(spectrum, recovered, FFTW_BACKWARD);
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            clapfft::FFT::c2c_1d(input, spectrum, FFTW_FORWARD);
            clapfft::FFT::c2c_1d(spectrum, recovered, FFTW_BACKWARD);
        }
        const auto end = clock::now();

        for (std::size_t i = 0; i < recovered.size(); ++i)
        {
            recovered[i] /= static_cast<double>(cfg.n);
        }
        last_output = recovered;

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    double benchmark_fftw(const std::vector<Complex> &input, const BenchmarkConfig &cfg, std::vector<Complex> &last_output)
    {
        using clock = std::chrono::steady_clock;

        std::vector<Complex> in = input;
        std::vector<Complex> spectrum(static_cast<std::size_t>(cfg.n));
        std::vector<Complex> recovered(static_cast<std::size_t>(cfg.n));

        fftw_plan forward_plan = fftw_plan_dft_1d(
            cfg.n,
            reinterpret_cast<fftw_complex *>(in.data()),
            reinterpret_cast<fftw_complex *>(spectrum.data()),
            FFTW_FORWARD,
            FFTW_MEASURE);

        fftw_plan backward_plan = fftw_plan_dft_1d(
            cfg.n,
            reinterpret_cast<fftw_complex *>(spectrum.data()),
            reinterpret_cast<fftw_complex *>(recovered.data()),
            FFTW_BACKWARD,
            FFTW_MEASURE);

        // FFTW_MEASURE may overwrite planning arrays. Restore benchmark input.
        in = input;

        for (int i = 0; i < cfg.warmup; ++i)
        {
            fftw_execute_dft(forward_plan,
                             reinterpret_cast<fftw_complex *>(in.data()),
                             reinterpret_cast<fftw_complex *>(spectrum.data()));
            fftw_execute_dft(backward_plan,
                             reinterpret_cast<fftw_complex *>(spectrum.data()),
                             reinterpret_cast<fftw_complex *>(recovered.data()));
        }

        const auto start = clock::now();
        for (int i = 0; i < cfg.iters; ++i)
        {
            fftw_execute_dft(forward_plan,
                             reinterpret_cast<fftw_complex *>(in.data()),
                             reinterpret_cast<fftw_complex *>(spectrum.data()));
            fftw_execute_dft(backward_plan,
                             reinterpret_cast<fftw_complex *>(spectrum.data()),
                             reinterpret_cast<fftw_complex *>(recovered.data()));
        }
        const auto end = clock::now();

        fftw_destroy_plan(forward_plan);
        fftw_destroy_plan(backward_plan);

        for (std::size_t i = 0; i < recovered.size(); ++i)
        {
            recovered[i] /= static_cast<double>(cfg.n);
        }
        last_output = recovered;

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }
}

int main(int argc, char **argv)
{
    const BenchmarkConfig cfg = parse_args(argc, argv);
    const std::vector<Complex> input = make_input(cfg.n);

    std::vector<Complex> clapfft_out;
    std::vector<Complex> fftw_out;

    const double clapfft_ms = benchmark_clapfft(input, cfg, clapfft_out);
    const double fftw_ms = benchmark_fftw(input, cfg, fftw_out);
    const double error = max_abs_diff(clapfft_out, fftw_out);

    const double clapfft_per_iter = clapfft_ms / static_cast<double>(cfg.iters);
    const double fftw_per_iter = fftw_ms / static_cast<double>(cfg.iters);
    const double speedup = fftw_per_iter > 0.0 ? clapfft_per_iter / fftw_per_iter : 0.0;

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Benchmark: clapfft vs FFTW (c2c 1D, forward+backward)\n";
    std::cout << "N=" << cfg.n << ", iterations=" << cfg.iters << ", warmup=" << cfg.warmup << "\n\n";
    std::cout << "clapfft total (ms): " << clapfft_ms << "\n";
    std::cout << "fftw   total (ms): " << fftw_ms << "\n";
    std::cout << "clapfft per iter (ms): " << clapfft_per_iter << "\n";
    std::cout << "fftw   per iter (ms): " << fftw_per_iter << "\n";
    std::cout << "ratio (clapfft/fftw): " << speedup << "x\n";
    std::cout << "max |clapfft - fftw| after round-trip normalization: " << error << "\n";

    return 0;
}
