#include <fftw3.h>
#include <clapfft/clapfft_api.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace
{
    using Real = double;
    using Complex = std::complex<Real>;

    struct BenchmarkConfig
    {
        int n = 1 << 14;
        int jobs_per_thread = 200;
        int warmup_jobs = 20;
        int max_threads = 0;
    };

    struct WorkerResult
    {
        double checksum = 0.0;
    };

    std::mutex fftw_planner_mutex;

    BenchmarkConfig parse_args(int argc, char **argv)
    {
        BenchmarkConfig cfg;
        if (argc > 1)
            cfg.n = std::max(2, std::atoi(argv[1]));
        if (argc > 2)
            cfg.jobs_per_thread = std::max(1, std::atoi(argv[2]));
        if (argc > 3)
            cfg.warmup_jobs = std::max(0, std::atoi(argv[3]));
        if (argc > 4)
            cfg.max_threads = std::max(1, std::atoi(argv[4]));
        return cfg;
    }

    std::vector<Complex> make_input(int n)
    {
        std::vector<Complex> input(static_cast<std::size_t>(n));
        std::mt19937 rng(987654);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);
        for (int i = 0; i < n; ++i)
        {
            input[static_cast<std::size_t>(i)] = Complex(dist(rng), dist(rng));
        }
        return input;
    }

    void normalize(std::vector<Complex> &data, int n)
    {
        const Real scale = static_cast<Real>(n);
        for (std::size_t i = 0; i < data.size(); ++i)
        {
            data[i] /= scale;
        }
    }

    Real max_abs_diff(const std::vector<Complex> &a, const std::vector<Complex> &b)
    {
        const std::size_t n = std::min(a.size(), b.size());
        Real max_err = 0.0;
        for (std::size_t i = 0; i < n; ++i)
        {
            const Real err = std::abs(a[i] - b[i]);
            if (err > max_err)
                max_err = err;
        }
        return max_err;
    }

    WorkerResult clapfft_worker(const std::vector<Complex> &input, const BenchmarkConfig &cfg)
    {
        std::vector<Complex> spectrum;
        std::vector<Complex> recovered;

        for (int i = 0; i < cfg.warmup_jobs; ++i)
        {
            clapfft::FFT::c2c_1d(input, spectrum, FFTW_FORWARD);
            clapfft::FFT::c2c_1d(spectrum, recovered, FFTW_BACKWARD);
        }

        WorkerResult result;
        for (int i = 0; i < cfg.jobs_per_thread; ++i)
        {
            clapfft::FFT::c2c_1d(input, spectrum, FFTW_FORWARD);
            clapfft::FFT::c2c_1d(spectrum, recovered, FFTW_BACKWARD);
            normalize(recovered, cfg.n);
            result.checksum += recovered[static_cast<std::size_t>(i % cfg.n)].real();
        }
        return result;
    }

    WorkerResult fftw_worker(const std::vector<Complex> &input, const BenchmarkConfig &cfg)
    {
        std::vector<Complex> dummy_in(static_cast<std::size_t>(cfg.n));
        std::vector<Complex> dummy_spec(static_cast<std::size_t>(cfg.n));
        std::vector<Complex> dummy_out(static_cast<std::size_t>(cfg.n));

        fftw_plan fwd;
        fftw_plan bwd;
        {
            std::lock_guard<std::mutex> lock(fftw_planner_mutex);
            fwd = fftw_plan_dft_1d(
                cfg.n,
                reinterpret_cast<fftw_complex *>(dummy_in.data()),
                reinterpret_cast<fftw_complex *>(dummy_spec.data()),
                FFTW_FORWARD,
                FFTW_MEASURE | FFTW_UNALIGNED);

            bwd = fftw_plan_dft_1d(
                cfg.n,
                reinterpret_cast<fftw_complex *>(dummy_spec.data()),
                reinterpret_cast<fftw_complex *>(dummy_out.data()),
                FFTW_BACKWARD,
                FFTW_MEASURE | FFTW_UNALIGNED);
        }

        std::vector<Complex> in = input;
        std::vector<Complex> spectrum(static_cast<std::size_t>(cfg.n));
        std::vector<Complex> recovered(static_cast<std::size_t>(cfg.n));

        for (int i = 0; i < cfg.warmup_jobs; ++i)
        {
            std::copy(input.begin(), input.end(), in.begin());
            fftw_execute_dft(fwd,
                             reinterpret_cast<fftw_complex *>(in.data()),
                             reinterpret_cast<fftw_complex *>(spectrum.data()));
            fftw_execute_dft(bwd,
                             reinterpret_cast<fftw_complex *>(spectrum.data()),
                             reinterpret_cast<fftw_complex *>(recovered.data()));
        }

        WorkerResult result;
        for (int i = 0; i < cfg.jobs_per_thread; ++i)
        {
            std::copy(input.begin(), input.end(), in.begin());
            fftw_execute_dft(fwd,
                             reinterpret_cast<fftw_complex *>(in.data()),
                             reinterpret_cast<fftw_complex *>(spectrum.data()));
            fftw_execute_dft(bwd,
                             reinterpret_cast<fftw_complex *>(spectrum.data()),
                             reinterpret_cast<fftw_complex *>(recovered.data()));
            normalize(recovered, cfg.n);
            result.checksum += recovered[static_cast<std::size_t>(i % cfg.n)].real();
        }

        {
            std::lock_guard<std::mutex> lock(fftw_planner_mutex);
            fftw_destroy_plan(fwd);
            fftw_destroy_plan(bwd);
        }
        return result;
    }

    template <typename WorkerFn>
    double run_parallel(int threads, WorkerFn worker_fn, const std::vector<Complex> &input, const BenchmarkConfig &cfg, double &combined_checksum)
    {
        using clock = std::chrono::steady_clock;

        std::vector<std::thread> pool;
        std::vector<WorkerResult> results(static_cast<std::size_t>(threads));
        pool.reserve(static_cast<std::size_t>(threads));

        const auto start = clock::now();
        for (int t = 0; t < threads; ++t)
        {
            pool.emplace_back([&, t]()
                              { results[static_cast<std::size_t>(t)] = worker_fn(input, cfg); });
        }

        for (std::size_t i = 0; i < pool.size(); ++i)
        {
            pool[i].join();
        }
        const auto end = clock::now();

        combined_checksum = 0.0;
        for (std::size_t i = 0; i < results.size(); ++i)
        {
            combined_checksum += results[i].checksum;
        }

        const std::chrono::duration<double, std::milli> elapsed = end - start;
        return elapsed.count();
    }

    void correctness_check(const std::vector<Complex> &input, int n)
    {
        std::vector<Complex> c_spec;
        std::vector<Complex> c_rec;
        clapfft::FFT::c2c_1d(input, c_spec, FFTW_FORWARD);
        clapfft::FFT::c2c_1d(c_spec, c_rec, FFTW_BACKWARD);
        normalize(c_rec, n);

        std::vector<Complex> in = input;
        std::vector<Complex> f_spec(static_cast<std::size_t>(n));
        std::vector<Complex> f_rec(static_cast<std::size_t>(n));

        fftw_plan fwd = fftw_plan_dft_1d(
            n,
            reinterpret_cast<fftw_complex *>(in.data()),
            reinterpret_cast<fftw_complex *>(f_spec.data()),
            FFTW_FORWARD,
            FFTW_ESTIMATE | FFTW_UNALIGNED);
        fftw_plan bwd = fftw_plan_dft_1d(
            n,
            reinterpret_cast<fftw_complex *>(f_spec.data()),
            reinterpret_cast<fftw_complex *>(f_rec.data()),
            FFTW_BACKWARD,
            FFTW_ESTIMATE | FFTW_UNALIGNED);

        fftw_execute_dft(fwd,
                         reinterpret_cast<fftw_complex *>(in.data()),
                         reinterpret_cast<fftw_complex *>(f_spec.data()));
        fftw_execute_dft(bwd,
                         reinterpret_cast<fftw_complex *>(f_spec.data()),
                         reinterpret_cast<fftw_complex *>(f_rec.data()));
        fftw_destroy_plan(fwd);
        fftw_destroy_plan(bwd);
        normalize(f_rec, n);

        const Real err = max_abs_diff(c_rec, f_rec);
        std::cout << "Correctness max |clapfft-fftw|: " << std::scientific << std::setprecision(4)
                  << static_cast<double>(err) << "\n\n";
    }
}

int main(int argc, char **argv)
{
    const BenchmarkConfig cfg = parse_args(argc, argv);
    const std::vector<Complex> input = make_input(cfg.n);

    const int hw = static_cast<int>(std::thread::hardware_concurrency());
    int limit = cfg.max_threads > 0 ? cfg.max_threads : (hw > 0 ? hw : 4);
    limit = std::max(1, limit);

    std::vector<int> thread_counts;
    for (int t = 1; t <= limit; t *= 2)
    {
        thread_counts.push_back(t);
    }
    if (thread_counts.back() != limit)
    {
        thread_counts.push_back(limit);
    }

    std::cout << "Benchmark: parallel c2c 1D jobs (clapfft vs FFTW)\n";
    std::cout << "N=" << cfg.n
              << ", jobs/thread=" << cfg.jobs_per_thread
              << ", warmup/thread=" << cfg.warmup_jobs
              << ", max_threads=" << limit << "\n\n";

    correctness_check(input, cfg.n);

    std::cout << std::fixed << std::setprecision(4);
    std::cout << "threads,clapfft_ms,fftw_ms,clapfft_jobs_per_s,fftw_jobs_per_s,ratio_clapfft_over_fftw\n";

    for (std::size_t i = 0; i < thread_counts.size(); ++i)
    {
        const int threads = thread_counts[i];
        const double total_jobs = static_cast<double>(threads) * static_cast<double>(cfg.jobs_per_thread);

        double clap_checksum = 0.0;
        const double clap_ms = run_parallel(threads, clapfft_worker, input, cfg, clap_checksum);
        const double clap_jobs_per_s = total_jobs / (clap_ms / 1000.0);

        double fftw_checksum = 0.0;
        const double fftw_ms = run_parallel(threads, fftw_worker, input, cfg, fftw_checksum);
        const double fftw_jobs_per_s = total_jobs / (fftw_ms / 1000.0);

        const double ratio = (fftw_ms > 0.0) ? (clap_ms / fftw_ms) : 0.0;

        std::cout << threads << ","
                  << clap_ms << ","
                  << fftw_ms << ","
                  << clap_jobs_per_s << ","
                  << fftw_jobs_per_s << ","
                  << ratio << "\n";

        if (std::isnan(clap_checksum) || std::isnan(fftw_checksum))
        {
            std::cerr << "Unexpected NaN checksum during benchmark." << std::endl;
            return 1;
        }
    }

    return 0;
}
