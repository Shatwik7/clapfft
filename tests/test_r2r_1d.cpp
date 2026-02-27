#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
void run_r2r_1d_test()
{
    const int n = 16;
    const T eps = static_cast<T>(1e-4);
    std::vector<T> input(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        input[static_cast<std::size_t>(i)] = static_cast<T>((i * 3) % 11 - 2);
    }

    std::vector<T> forward;
    std::vector<T> recovered;
    clapfft::FFT::r2r_1d(input, forward, FFTW_REDFT10);
    clapfft::FFT::r2r_1d(forward, recovered, FFTW_REDFT01);

    for (std::size_t i = 0; i < recovered.size(); ++i) {
        recovered[i] /= static_cast<T>(2 * n);
        assert(std::abs(recovered[i] - input[i]) <= eps);
    }
}

int main()
{
    run_r2r_1d_test<float>();
    run_r2r_1d_test<double>();
    run_r2r_1d_test<long double>();
    std::cout << "r2r_1d tests passed." << std::endl;
    return 0;
}
