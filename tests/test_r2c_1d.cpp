#include <clapfft/clapfft_api.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_r2c_1d_test()
{
    const int n = 18;
    const T eps = static_cast<T>(1e-4);
    std::vector<T> input(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        input[static_cast<std::size_t>(i)] = static_cast<T>((i * 3) % 11 - 2);
    }

    std::vector<std::complex<T>> spectrum;
    std::vector<T> recovered;
    clapfft::FFT::r2c_1d(input, spectrum);
    clapfft::FFT::c2r_1d(spectrum, recovered);

    for (std::size_t i = 0; i < recovered.size(); ++i) {
        recovered[i] /= static_cast<T>(n);
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

int main()
{
    run_r2c_1d_test<float>();
    run_r2c_1d_test<double>();
    run_r2c_1d_test<long double>();
    std::cout << "r2c_1d tests passed." << std::endl;
    return 0;
}
