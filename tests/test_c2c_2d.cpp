#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_c2c_2d_test()
{
    const int n0 = 5;
    const int n1 = 6;
    const T eps = static_cast<T>(1e-4);
    std::vector<std::vector<std::complex<T>>> input(static_cast<std::size_t>(n0), std::vector<std::complex<T>>(static_cast<std::size_t>(n1)));
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = std::complex<T>(static_cast<T>(i * n1 + j), static_cast<T>((i - j) % 3));
        }
    }

    std::vector<std::vector<std::complex<T>>> spectrum;
    std::vector<std::vector<std::complex<T>>> recovered;
    clapfft::FFT::c2c_2d(input, spectrum, FFTW_FORWARD);
    clapfft::FFT::c2c_2d(spectrum, recovered, FFTW_BACKWARD);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] /= static_cast<T>(n0 * n1);
            assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].real() - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].real()) <= eps);
            assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].imag() - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].imag()) <= eps);
        }
    }
}

int main()
{
    run_c2c_2d_test<float>();
    run_c2c_2d_test<double>();
    run_c2c_2d_test<long double>();
    std::cout << "c2c_2d tests passed." << std::endl;
    return 0;
}
