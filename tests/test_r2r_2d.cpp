#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
void run_r2r_2d_test()
{
    const int n0 = 4;
    const int n1 = 6;
    const T eps = static_cast<T>(1e-4);
    std::vector<std::vector<T>> input(static_cast<std::size_t>(n0), std::vector<T>(static_cast<std::size_t>(n1)));
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = static_cast<T>((i * 7 + j * 4) % 19 - 6);
        }
    }

    std::vector<std::vector<T>> forward;
    std::vector<std::vector<T>> recovered;
    clapfft::FFT::r2r_2d(input, forward, FFTW_REDFT10, FFTW_REDFT10);
    clapfft::FFT::r2r_2d(forward, recovered, FFTW_REDFT01, FFTW_REDFT01);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] /= static_cast<T>((2 * n0) * (2 * n1));
            assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) <= eps);
        }
    }
}

int main()
{
    run_r2r_2d_test<float>();
    run_r2r_2d_test<double>();
    run_r2r_2d_test<long double>();
    std::cout << "r2r_2d tests passed." << std::endl;
    return 0;
}
