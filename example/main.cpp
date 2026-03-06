#include <clapfft/advanced_fft.hpp>
#include <clapfft/clapfft_api.hpp>
#include <clapfft/wisdom.hpp>

#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_many_dft_r2c_1d_test()
{
    const int n = 10;
    const int howmany = 4;
    const int n_complex = n / 2 + 1;
    const T eps = static_cast<T>(1e-4);

    std::vector<T> input(static_cast<std::size_t>(n * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n; ++i)
        {
            input[static_cast<std::size_t>(b * n + i)] = static_cast<T>((b + 1) * i - 2 * b + 0.25);
            std::cout << input[static_cast<std::size_t>(b * n + i)] << std::endl;
        }
    }

    std::vector<std::complex<T>> output(static_cast<std::size_t>(n_complex * howmany));
    int dims[1] = {n};

    clapfft::AdvancedFFT::many_dft_r2c<T>(1, dims, howmany,
                                          input.data(), nullptr,
                                          1, n,
                                          output.data(), nullptr,
                                          1, n_complex);

    for (int b = 0; b < howmany; ++b)
    {
        T sum = static_cast<T>(0);
        for (int i = 0; i < n; ++i)
        {
            sum += input[static_cast<std::size_t>(b * n + i)];
        }

        const std::complex<T> dc = output[static_cast<std::size_t>(b * n_complex)];
        assert(std::abs(dc.real() - sum) <= eps);
        assert(std::abs(dc.imag()) <= eps);
    }
}

template <typename T>
void run_many_dft_r2c_2d_test()
{
    const int n0 = 4;
    const int n1 = 8;
    const int howmany = 3;
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
                input[static_cast<std::size_t>(b * real_points + i * n1 + j)] = static_cast<T>((b + 2) * i + j - 0.75);
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
void run_many_dft_r2c_3d_test()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 10;
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
                    input[static_cast<std::size_t>(b * real_points + i * n1 * n2 + j * n2 + k)] = static_cast<T>((i * 3 + j * 5 + k + b) % 13 - 2);
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

template <typename T>
void run_c2r_3d_with_wisdom_test(std::string fileName)
{
    const int n0 = 4;
    const int n1 = 4;
    const int n2_real = 8;
    const int n2_complex = n2_real / 2 + 1;
    const std::string wisdom_file = fileName;

    // 1. Try to import wisdom if it exists
    if (clapfft::Wisdom::import_from_filename<T>(wisdom_file))
    {
        std::cout << "Wisdom imported from " << wisdom_file << std::endl;
    }
    else
    {
        std::cout << "No existing wisdom found, will export after planning." << std::endl;
    }

    // 2. Prepare data
    std::vector<std::vector<std::vector<std::complex<T>>>> input(
        n0, std::vector<std::vector<std::complex<T>>>(
                n1, std::vector<std::complex<T>>(n2_complex, {1.0, 0.0})));
    std::vector<std::vector<std::vector<T>>> output;

    // 3. Run FFT (this will use wisdom if available, or generate it)
    clapfft::FFT::c2r_3d<T>(input, output);

    // 4. Export wisdom for future runs
    clapfft::Wisdom::export_to_filename<T>(wisdom_file);
    std::cout << "Wisdom exported to " << wisdom_file << std::endl;

    assert(output.size() == static_cast<std::size_t>(n0));
    assert(output[0][0].size() == static_cast<std::size_t>(n2_real));
}

int main()
{
    run_many_dft_r2c_1d_test<float>();
    run_many_dft_r2c_1d_test<double>();
    run_many_dft_r2c_1d_test<long double>();

    run_many_dft_r2c_2d_test<float>();
    run_many_dft_r2c_2d_test<double>();
    run_many_dft_r2c_2d_test<long double>();

    run_many_dft_r2c_3d_test<float>();
    run_many_dft_r2c_3d_test<double>();
    run_many_dft_r2c_3d_test<long double>();

    run_c2r_3d_with_wisdom_test<long double>("c2r_3d_wisdom_1.dat");
    run_c2r_3d_with_wisdom_test<float>("c2r_3d_wisdom_2.dat");
    run_c2r_3d_with_wisdom_test<double>("c2r_3d_wisdom_3.dat");

    std::cout << "advanced_many_dft_r2c tests passed." << std::endl;
    return 0;
}
