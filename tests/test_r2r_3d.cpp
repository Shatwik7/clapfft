#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>

template <typename T>
void run_r2r_3d_test()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 6;
    const T eps = static_cast<T>(1e-4);

    std::vector<std::vector<std::vector<T>>> input(
        static_cast<std::size_t>(n0),
        std::vector<std::vector<T>>(static_cast<std::size_t>(n1), std::vector<T>(static_cast<std::size_t>(n2))));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
                    static_cast<T>((i * 3 + j * 5 + k * 7) % 23 - 9);
            }
        }
    }

    std::vector<std::vector<std::vector<T>>> forward;
    std::vector<std::vector<std::vector<T>>> recovered;
    clapfft::FFT::r2r_3d(input, forward, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10);
    clapfft::FFT::r2r_3d(forward, recovered, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01);

    const T scale = static_cast<T>((2 * n0) * (2 * n1) * (2 * n2));
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] /= scale;
                assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]) <= eps);
            }
        }
    }
}

int main()
{
    run_r2r_3d_test<float>();
    run_r2r_3d_test<double>();
    run_r2r_3d_test<long double>();
    std::cout << "r2r_3d tests passed." << std::endl;
    return 0;
}
