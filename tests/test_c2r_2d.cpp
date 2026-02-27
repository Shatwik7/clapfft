#include <clapfft/clapfft_api.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_c2r_2d_test()
{
    const int n0 = 4;
    const int n1 = 8;
    const T eps = static_cast<T>(1e-4);
    std::vector<std::vector<T>> input(static_cast<std::size_t>(n0), std::vector<T>(static_cast<std::size_t>(n1)));
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = static_cast<T>((i * 5 + j * 7) % 13 - 4);
        }
    }

    std::vector<std::vector<std::complex<T>>> spectrum;
    std::vector<std::vector<T>> recovered;
    clapfft::FFT::r2c_2d(input, spectrum);
    clapfft::FFT::c2r_2d(spectrum, recovered);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] /= static_cast<T>(n0 * n1);
            assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]) <= eps);
        }
    }
}

int main()
{
    run_c2r_2d_test<float>();
    run_c2r_2d_test<double>();
    run_c2r_2d_test<long double>();
    std::cout << "c2r_2d tests passed." << std::endl;
    return 0;
}
