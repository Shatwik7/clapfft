#include <clapfft/clapfft_api.hpp>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_c2r_3d_test()
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
                    static_cast<T>((i * 11 + j * 3 + k * 2) % 17 - 8);
            }
        }
    }

    std::vector<std::vector<std::vector<std::complex<T>>>> spectrum;
    std::vector<std::vector<std::vector<T>>> recovered;
    clapfft::FFT::r2c_3d(input, spectrum);
    clapfft::FFT::c2r_3d(spectrum, recovered);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] /= static_cast<T>(n0 * n1 * n2);
                assert(std::abs(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] - input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]) <= eps);
            }
        }
    }
}

int main()
{
    run_c2r_3d_test<float>();
    run_c2r_3d_test<double>();
    run_c2r_3d_test<long double>();
    std::cout << "c2r_3d tests passed." << std::endl;
    return 0;
}
