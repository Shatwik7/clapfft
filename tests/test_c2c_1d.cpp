#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_c2c_1d_test()
{
    const int n = 16;
    const T eps = static_cast<T>(1e-4);
    std::vector<std::complex<T>> input(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        input[static_cast<std::size_t>(i)] = std::complex<T>(static_cast<T>(i) * static_cast<T>(0.25), static_cast<T>((i % 5) - 2));
    }

    std::vector<std::complex<T>> spectrum;
    std::vector<std::complex<T>> recovered;
    clapfft::FFT::c2c_1d(input, spectrum, FFTW_FORWARD);
    clapfft::FFT::c2c_1d(spectrum, recovered, FFTW_BACKWARD);

    for (std::size_t i = 0; i < recovered.size(); ++i) {
        recovered[i] /= static_cast<T>(n);
        assert(std::abs(recovered[i].real() - input[i].real()) <= eps);
        assert(std::abs(recovered[i].imag() - input[i].imag()) <= eps);
    }
}

int main()
{
    run_c2c_1d_test<float>();
    run_c2c_1d_test<double>();
    run_c2c_1d_test<long double>();
    std::cout << "c2c_1d tests passed." << std::endl;
    return 0;
}
