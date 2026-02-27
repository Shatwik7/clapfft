#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_c2c_3d_test()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 5;
    const T eps = static_cast<T>(1e-4);

    std::vector<std::vector<std::vector<std::complex<T>>>> input(
        static_cast<std::size_t>(n0),
        std::vector<std::vector<std::complex<T>>>(static_cast<std::size_t>(n1), std::vector<std::complex<T>>(static_cast<std::size_t>(n2))));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
                    std::complex<T>(static_cast<T>(i * n1 * n2 + j * n2 + k), static_cast<T>((i + j - k) % 5));
            }
        }
    }

    std::vector<std::vector<std::vector<std::complex<T>>>> spectrum;
    std::vector<std::vector<std::vector<std::complex<T>>>> recovered;
    clapfft::FFT::c2c_3d(input, spectrum, FFTW_FORWARD);
    clapfft::FFT::c2c_3d(spectrum, recovered, FFTW_BACKWARD);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] /= static_cast<T>(n0 * n1 * n2);
                assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)].real() - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)].real()) <= eps);
                assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)].imag() - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)].imag()) <= eps);
            }
        }
    }
}

int main()
{
    run_c2c_3d_test<float>();
    run_c2c_3d_test<double>();
    run_c2c_3d_test<long double>();
    std::cout << "c2c_3d tests passed." << std::endl;
    return 0;
}
