#include <clapfft/advanced_fft.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_many_dft_c2r_1d_test()
{
    const int n = 12;
    const int howmany = 3;
    const int n_complex = n / 2 + 1;
    const T eps = static_cast<T>(1e-4);

    std::vector<T> input(static_cast<std::size_t>(n * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n; ++i)
        {
            input[static_cast<std::size_t>(b * n + i)] = static_cast<T>((i * 2 + b * 3) % 9 - 2);
        }
    }

    std::vector<std::complex<T>> spectrum(static_cast<std::size_t>(n_complex * howmany));
    std::vector<T> recovered(static_cast<std::size_t>(n * howmany));
    int dims[1] = {n};

    clapfft::AdvancedFFT::many_dft_r2c<T>(1, dims, howmany,
                                          input.data(), nullptr,
                                          1, n,
                                          spectrum.data(), nullptr,
                                          1, n_complex);

    clapfft::AdvancedFFT::many_dft_c2r<T>(1, dims, howmany,
                                          spectrum.data(), nullptr,
                                          1, n_complex,
                                          recovered.data(), nullptr,
                                          1, n);

    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= static_cast<T>(n);
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

template <typename T>
void run_many_dft_c2r_2d_test()
{
    const int n0 = 5;
    const int n1 = 8;
    const int howmany = 2;
    const int real_points = n0 * n1;
    const int complex_points = n0 * (n1 / 2 + 1);
    const T eps = static_cast<T>(1e-4);

    std::vector<T> input(static_cast<std::size_t>(real_points * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                input[static_cast<std::size_t>(b * real_points + i * n1 + j)] = static_cast<T>((i * 4 + j * 2 + b) % 15 - 6);
            }
        }
    }

    std::vector<std::complex<T>> spectrum(static_cast<std::size_t>(complex_points * howmany));
    std::vector<T> recovered(static_cast<std::size_t>(real_points * howmany));
    int dims[2] = {n0, n1};

    clapfft::AdvancedFFT::many_dft_r2c<T>(2, dims, howmany,
                                          input.data(), nullptr,
                                          1, real_points,
                                          spectrum.data(), nullptr,
                                          1, complex_points);

    clapfft::AdvancedFFT::many_dft_c2r<T>(2, dims, howmany,
                                          spectrum.data(), nullptr,
                                          1, complex_points,
                                          recovered.data(), nullptr,
                                          1, real_points);

    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= static_cast<T>(real_points);
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

template <typename T>
void run_many_dft_c2r_3d_test()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 12;
    const int howmany = 2;
    const int real_points = n0 * n1 * n2;
    const int complex_points = n0 * n1 * (n2 / 2 + 1);
    const T eps = static_cast<T>(1e-4);

    std::vector<T> input(static_cast<std::size_t>(real_points * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                for (int k = 0; k < n2; ++k)
                {
                    input[static_cast<std::size_t>(b * real_points + i * n1 * n2 + j * n2 + k)] = static_cast<T>((b + i * 2 + j * 3 + k) % 21 - 9);
                }
            }
        }
    }

    std::vector<std::complex<T>> spectrum(static_cast<std::size_t>(complex_points * howmany));
    std::vector<T> recovered(static_cast<std::size_t>(real_points * howmany));
    int dims[3] = {n0, n1, n2};

    clapfft::AdvancedFFT::many_dft_r2c<T>(3, dims, howmany,
                                          input.data(), nullptr,
                                          1, real_points,
                                          spectrum.data(), nullptr,
                                          1, complex_points);

    clapfft::AdvancedFFT::many_dft_c2r<T>(3, dims, howmany,
                                          spectrum.data(), nullptr,
                                          1, complex_points,
                                          recovered.data(), nullptr,
                                          1, real_points);

    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= static_cast<T>(real_points);
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

int main()
{
    run_many_dft_c2r_1d_test<float>();
    run_many_dft_c2r_1d_test<double>();
    run_many_dft_c2r_1d_test<long double>();

    run_many_dft_c2r_2d_test<float>();
    run_many_dft_c2r_2d_test<double>();
    run_many_dft_c2r_2d_test<long double>();

    run_many_dft_c2r_3d_test<float>();
    run_many_dft_c2r_3d_test<double>();
    run_many_dft_c2r_3d_test<long double>();
    std::cout << "advanced_many_dft_c2r tests passed." << std::endl;
    return 0;
}
