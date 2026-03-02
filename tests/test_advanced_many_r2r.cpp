#include <clapfft/advanced_fft.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
void run_many_r2r_1d_test()
{
    const int n = 14;
    const int howmany = 3;
    const T eps = static_cast<T>(1e-4);

    std::vector<T> input(static_cast<std::size_t>(n * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n; ++i)
        {
            input[static_cast<std::size_t>(b * n + i)] = static_cast<T>((b * 5 + i * 3) % 17 - 4);
        }
    }

    std::vector<T> forward(static_cast<std::size_t>(n * howmany));
    std::vector<T> recovered(static_cast<std::size_t>(n * howmany));
    int dims[1] = {n};
    int kind_fwd[1] = {FFTW_REDFT10};
    int kind_inv[1] = {FFTW_REDFT01};

    clapfft::AdvancedFFT::many_r2r<T>(1, dims, howmany,
                                      input.data(), nullptr,
                                      1, n,
                                      forward.data(), nullptr,
                                      1, n,
                                      kind_fwd);

    clapfft::AdvancedFFT::many_r2r<T>(1, dims, howmany,
                                      forward.data(), nullptr,
                                      1, n,
                                      recovered.data(), nullptr,
                                      1, n,
                                      kind_inv);

    const T scale = static_cast<T>(2 * n);
    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= scale;
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

template <typename T>
void run_many_r2r_2d_test()
{
    const int n0 = 4;
    const int n1 = 6;
    const int howmany = 2;
    const int points = n0 * n1;
    const T eps = static_cast<T>(1e-4);

    std::vector<T> input(static_cast<std::size_t>(points * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                input[static_cast<std::size_t>(b * points + i * n1 + j)] = static_cast<T>((b + i * 7 + j * 4) % 19 - 5);
            }
        }
    }

    std::vector<T> forward(static_cast<std::size_t>(points * howmany));
    std::vector<T> recovered(static_cast<std::size_t>(points * howmany));
    int dims[2] = {n0, n1};
    int kind_fwd[2] = {FFTW_REDFT10, FFTW_REDFT10};
    int kind_inv[2] = {FFTW_REDFT01, FFTW_REDFT01};

    clapfft::AdvancedFFT::many_r2r<T>(2, dims, howmany,
                                      input.data(), nullptr,
                                      1, points,
                                      forward.data(), nullptr,
                                      1, points,
                                      kind_fwd);

    clapfft::AdvancedFFT::many_r2r<T>(2, dims, howmany,
                                      forward.data(), nullptr,
                                      1, points,
                                      recovered.data(), nullptr,
                                      1, points,
                                      kind_inv);

    const T scale = static_cast<T>((2 * n0) * (2 * n1));
    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= scale;
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

template <typename T>
void run_many_r2r_3d_test()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 6;
    const int howmany = 2;
    const int points = n0 * n1 * n2;
    const T eps = static_cast<T>(1e-4);

    std::vector<T> input(static_cast<std::size_t>(points * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                for (int k = 0; k < n2; ++k)
                {
                    input[static_cast<std::size_t>(b * points + i * n1 * n2 + j * n2 + k)] = static_cast<T>((b + i * 3 + j * 5 + k * 7) % 23 - 11);
                }
            }
        }
    }

    std::vector<T> forward(static_cast<std::size_t>(points * howmany));
    std::vector<T> recovered(static_cast<std::size_t>(points * howmany));
    int dims[3] = {n0, n1, n2};
    int kind_fwd[3] = {FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10};
    int kind_inv[3] = {FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01};

    clapfft::AdvancedFFT::many_r2r<T>(3, dims, howmany,
                                      input.data(), nullptr,
                                      1, points,
                                      forward.data(), nullptr,
                                      1, points,
                                      kind_fwd);

    clapfft::AdvancedFFT::many_r2r<T>(3, dims, howmany,
                                      forward.data(), nullptr,
                                      1, points,
                                      recovered.data(), nullptr,
                                      1, points,
                                      kind_inv);

    const T scale = static_cast<T>((2 * n0) * (2 * n1) * (2 * n2));
    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= scale;
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

int main()
{
    run_many_r2r_1d_test<float>();
    run_many_r2r_1d_test<double>();
    run_many_r2r_1d_test<long double>();

    run_many_r2r_2d_test<float>();
    run_many_r2r_2d_test<double>();
    run_many_r2r_2d_test<long double>();

    run_many_r2r_3d_test<float>();
    run_many_r2r_3d_test<double>();
    run_many_r2r_3d_test<long double>();
    std::cout << "advanced_many_r2r tests passed." << std::endl;
    return 0;
}
