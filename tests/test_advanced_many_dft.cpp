#include <clapfft/advanced_fft.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_many_dft_1d_test()
{
    const int n = 8;
    const int howmany = 3;
    const T eps = static_cast<T>(1e-4);

    std::vector<std::complex<T>> input(static_cast<std::size_t>(n * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n; ++i)
        {
            const std::size_t idx = static_cast<std::size_t>(b * n + i);
            input[idx] = std::complex<T>(static_cast<T>(b + i * 0.5), static_cast<T>((i % 3) - b));
        }
    }

    std::vector<std::complex<T>> forward(static_cast<std::size_t>(n * howmany));
    std::vector<std::complex<T>> recovered(static_cast<std::size_t>(n * howmany));

    int dims[1] = {n};
    clapfft::AdvancedFFT::many_dft<T>(1, dims, howmany,
                                      input.data(), nullptr,
                                      1, n,
                                      forward.data(), nullptr,
                                      1, n,
                                      FFTW_FORWARD);

    clapfft::AdvancedFFT::many_dft<T>(1, dims, howmany,
                                      forward.data(), nullptr,
                                      1, n,
                                      recovered.data(), nullptr,
                                      1, n,
                                      FFTW_BACKWARD);

    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= static_cast<T>(n);
        assert(std::abs(recovered[i].real() - input[i].real()) <= eps);
        assert(std::abs(recovered[i].imag() - input[i].imag()) <= eps);
    }
}

template <typename T>
void run_many_dft_2d_test()
{
    const int n0 = 4;
    const int n1 = 6;
    const int howmany = 2;
    const int points_per_transform = n0 * n1;
    const T eps = static_cast<T>(1e-4);

    std::vector<std::complex<T>> input(static_cast<std::size_t>(points_per_transform * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                const std::size_t idx = static_cast<std::size_t>(b * points_per_transform + i * n1 + j);
                input[idx] = std::complex<T>(static_cast<T>(b + i * 2 + j * 0.25), static_cast<T>((i - j) % 5));
            }
        }
    }

    std::vector<std::complex<T>> forward(static_cast<std::size_t>(points_per_transform * howmany));
    std::vector<std::complex<T>> recovered(static_cast<std::size_t>(points_per_transform * howmany));

    int dims[2] = {n0, n1};
    clapfft::AdvancedFFT::many_dft<T>(2, dims, howmany,
                                      input.data(), nullptr,
                                      1, points_per_transform,
                                      forward.data(), nullptr,
                                      1, points_per_transform,
                                      FFTW_FORWARD);

    clapfft::AdvancedFFT::many_dft<T>(2, dims, howmany,
                                      forward.data(), nullptr,
                                      1, points_per_transform,
                                      recovered.data(), nullptr,
                                      1, points_per_transform,
                                      FFTW_BACKWARD);

    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= static_cast<T>(points_per_transform);
        assert(std::abs(recovered[i].real() - input[i].real()) <= eps);
        assert(std::abs(recovered[i].imag() - input[i].imag()) <= eps);
    }
}

template <typename T>
void run_many_dft_3d_test()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 5;
    const int howmany = 2;
    const int points_per_transform = n0 * n1 * n2;
    const T eps = static_cast<T>(1e-4);

    std::vector<std::complex<T>> input(static_cast<std::size_t>(points_per_transform * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                for (int k = 0; k < n2; ++k)
                {
                    const std::size_t idx = static_cast<std::size_t>(b * points_per_transform + i * n1 * n2 + j * n2 + k);
                    input[idx] = std::complex<T>(static_cast<T>(b + i * 0.75 + j * 0.5 + k), static_cast<T>((i + j - k) % 7));
                }
            }
        }
    }

    std::vector<std::complex<T>> forward(static_cast<std::size_t>(points_per_transform * howmany));
    std::vector<std::complex<T>> recovered(static_cast<std::size_t>(points_per_transform * howmany));

    int dims[3] = {n0, n1, n2};
    clapfft::AdvancedFFT::many_dft<T>(3, dims, howmany,
                                      input.data(), nullptr,
                                      1, points_per_transform,
                                      forward.data(), nullptr,
                                      1, points_per_transform,
                                      FFTW_FORWARD);

    clapfft::AdvancedFFT::many_dft<T>(3, dims, howmany,
                                      forward.data(), nullptr,
                                      1, points_per_transform,
                                      recovered.data(), nullptr,
                                      1, points_per_transform,
                                      FFTW_BACKWARD);

    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= static_cast<T>(points_per_transform);
        assert(std::abs(recovered[i].real() - input[i].real()) <= eps);
        assert(std::abs(recovered[i].imag() - input[i].imag()) <= eps);
    }
}

int main()
{
    run_many_dft_1d_test<float>();
    run_many_dft_1d_test<double>();
    run_many_dft_1d_test<long double>();

    run_many_dft_2d_test<float>();
    run_many_dft_2d_test<double>();
    run_many_dft_2d_test<long double>();

    run_many_dft_3d_test<float>();
    run_many_dft_3d_test<double>();
    run_many_dft_3d_test<long double>();

    std::cout << "advanced_many_dft tests passed." << std::endl;
    return 0;
}
