#include <clapfft/clapfft_api.hpp>
#include <fftw3.h>
#include <cassert>
#include <complex>
#include <cstddef>
#include <iostream>
#include <type_traits>
#include <vector>

namespace {

template <typename T>
T tolerance();

template <>
float tolerance<float>() { return 1e-4f; }

template <>
double tolerance<double>() { return 1e-10; }

template <>
long double tolerance<long double>() { return 1e-10L; }

template <typename T>
std::vector<std::complex<T>> make_complex_signal_1d(int n)
{
    std::vector<std::complex<T>> signal(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        T real = static_cast<T>(i) * static_cast<T>(0.25) + static_cast<T>(1.0);
        T imag = static_cast<T>((i % 5) - 2) * static_cast<T>(0.5);
        signal[static_cast<std::size_t>(i)] = std::complex<T>(real, imag);
    }
    return signal;
}

template <typename T>
std::vector<T> make_real_signal_1d(int n)
{
    std::vector<T> signal(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) {
        signal[static_cast<std::size_t>(i)] = static_cast<T>((i * 3) % 11) - static_cast<T>(2.0);
    }
    return signal;
}

template <typename T>
void assert_complex_close(const std::vector<std::complex<T>>& expected,
                          const std::vector<std::complex<T>>& actual,
                          T eps)
{
    assert(expected.size() == actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(expected[i].real() - actual[i].real()) <= eps);
        assert(std::abs(expected[i].imag() - actual[i].imag()) <= eps);
    }
}

template <typename T>
void assert_real_close(const std::vector<T>& expected,
                       const std::vector<T>& actual,
                       T eps)
{
    assert(expected.size() == actual.size());
    for (std::size_t i = 0; i < expected.size(); ++i) {
        assert(std::abs(expected[i] - actual[i]) <= eps);
    }
}

template <typename T>
void test_c2c_roundtrip_1d()
{
    const int n = 16;
    const T eps = tolerance<T>();
    const std::vector<std::complex<T>> input = make_complex_signal_1d<T>(n);

    for (int run = 0; run < 5; ++run) {
        std::vector<std::complex<T>> spectrum;
        std::vector<std::complex<T>> recovered;

        clapfft::FFT::c2c_1d(input, spectrum, FFTW_FORWARD);
        clapfft::FFT::c2c_1d(spectrum, recovered, FFTW_BACKWARD);

        for (std::size_t i = 0; i < recovered.size(); ++i) {
            recovered[i] /= static_cast<T>(n);
        }

        assert_complex_close(input, recovered, eps);
    }
}

template <typename T>
void test_c2c_roundtrip_2d()
{
    const int n0 = 5;
    const int n1 = 6;
    const T eps = tolerance<T>();

    std::vector<std::vector<std::complex<T>>> input(
        static_cast<std::size_t>(n0),
        std::vector<std::complex<T>>(static_cast<std::size_t>(n1)));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            T real = static_cast<T>(i * n1 + j) * static_cast<T>(0.125);
            T imag = static_cast<T>((i - j) % 4) * static_cast<T>(0.33);
            input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] = std::complex<T>(real, imag);
        }
    }

    for (int run = 0; run < 4; ++run) {
        std::vector<std::vector<std::complex<T>>> spectrum;
        std::vector<std::vector<std::complex<T>>> recovered;

        clapfft::FFT::c2c_2d(input, spectrum, FFTW_FORWARD);
        clapfft::FFT::c2c_2d(spectrum, recovered, FFTW_BACKWARD);

        assert(recovered.size() == static_cast<std::size_t>(n0));
        for (int i = 0; i < n0; ++i) {
            assert(recovered[static_cast<std::size_t>(i)].size() == static_cast<std::size_t>(n1));
            for (int j = 0; j < n1; ++j) {
                recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] /= static_cast<T>(n0 * n1);
            }
            assert_complex_close(input[static_cast<std::size_t>(i)], recovered[static_cast<std::size_t>(i)], eps);
        }
    }
}

template <typename T>
void test_c2c_roundtrip_3d()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 5;
    const T eps = tolerance<T>();

    std::vector<std::vector<std::vector<std::complex<T>>>> input(
        static_cast<std::size_t>(n0),
        std::vector<std::vector<std::complex<T>>>(
            static_cast<std::size_t>(n1),
            std::vector<std::complex<T>>(static_cast<std::size_t>(n2))));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                T real = static_cast<T>(i * n1 * n2 + j * n2 + k) * static_cast<T>(0.05);
                T imag = static_cast<T>((i + j - k) % 7) * static_cast<T>(0.2);
                input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
                    std::complex<T>(real, imag);
            }
        }
    }

    for (int run = 0; run < 3; ++run) {
        std::vector<std::vector<std::vector<std::complex<T>>>> spectrum;
        std::vector<std::vector<std::vector<std::complex<T>>>> recovered;

        clapfft::FFT::c2c_3d(input, spectrum, FFTW_FORWARD);
        clapfft::FFT::c2c_3d(spectrum, recovered, FFTW_BACKWARD);

        assert(recovered.size() == static_cast<std::size_t>(n0));
        for (int i = 0; i < n0; ++i) {
            assert(recovered[static_cast<std::size_t>(i)].size() == static_cast<std::size_t>(n1));
            for (int j = 0; j < n1; ++j) {
                assert(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].size() == static_cast<std::size_t>(n2));
                for (int k = 0; k < n2; ++k) {
                    recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]
                        /= static_cast<T>(n0 * n1 * n2);
                }
                assert_complex_close(input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                                     recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                                     eps);
            }
        }
    }
}

template <typename T>
void test_r2c_c2r_roundtrip_1d()
{
    const int n = 18;
    const T eps = tolerance<T>();
    const std::vector<T> input = make_real_signal_1d<T>(n);

    for (int run = 0; run < 5; ++run) {
        std::vector<std::complex<T>> spectrum;
        std::vector<T> recovered;

        clapfft::FFT::r2c_1d(input, spectrum);
        assert(spectrum.size() == static_cast<std::size_t>(n / 2 + 1));

        clapfft::FFT::c2r_1d(spectrum, recovered);
        assert(recovered.size() == static_cast<std::size_t>(n));

        for (std::size_t i = 0; i < recovered.size(); ++i) {
            recovered[i] /= static_cast<T>(n);
        }

        assert_real_close(input, recovered, eps);
    }
}

template <typename T>
void test_r2c_c2r_roundtrip_2d()
{
    const int n0 = 4;
    const int n1 = 8;
    const T eps = tolerance<T>();

    std::vector<std::vector<T>> input(
        static_cast<std::size_t>(n0),
        std::vector<T>(static_cast<std::size_t>(n1)));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<T>((i * 5 + j * 7) % 13) - static_cast<T>(4);
        }
    }

    for (int run = 0; run < 4; ++run) {
        std::vector<std::vector<std::complex<T>>> spectrum;
        std::vector<std::vector<T>> recovered;

        clapfft::FFT::r2c_2d(input, spectrum);
        assert(spectrum.size() == static_cast<std::size_t>(n0));
        for (int i = 0; i < n0; ++i) {
            assert(spectrum[static_cast<std::size_t>(i)].size() == static_cast<std::size_t>(n1 / 2 + 1));
        }

        clapfft::FFT::c2r_2d(spectrum, recovered);
        assert(recovered.size() == static_cast<std::size_t>(n0));

        for (int i = 0; i < n0; ++i) {
            assert(recovered[static_cast<std::size_t>(i)].size() == static_cast<std::size_t>(n1));
            for (int j = 0; j < n1; ++j) {
                recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] /= static_cast<T>(n0 * n1);
            }
            assert_real_close(input[static_cast<std::size_t>(i)], recovered[static_cast<std::size_t>(i)], eps);
        }
    }
}

template <typename T>
void test_r2c_c2r_roundtrip_3d()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 6;
    const T eps = tolerance<T>();

    std::vector<std::vector<std::vector<T>>> input(
        static_cast<std::size_t>(n0),
        std::vector<std::vector<T>>(
            static_cast<std::size_t>(n1),
            std::vector<T>(static_cast<std::size_t>(n2))));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
                    static_cast<T>((i * 11 + j * 3 + k * 2) % 17) - static_cast<T>(8);
            }
        }
    }

    for (int run = 0; run < 3; ++run) {
        std::vector<std::vector<std::vector<std::complex<T>>>> spectrum;
        std::vector<std::vector<std::vector<T>>> recovered;

        clapfft::FFT::r2c_3d(input, spectrum);
        assert(spectrum.size() == static_cast<std::size_t>(n0));
        for (int i = 0; i < n0; ++i) {
            assert(spectrum[static_cast<std::size_t>(i)].size() == static_cast<std::size_t>(n1));
            for (int j = 0; j < n1; ++j) {
                assert(spectrum[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].size() == static_cast<std::size_t>(n2 / 2 + 1));
            }
        }

        clapfft::FFT::c2r_3d(spectrum, recovered);
        assert(recovered.size() == static_cast<std::size_t>(n0));

        for (int i = 0; i < n0; ++i) {
            assert(recovered[static_cast<std::size_t>(i)].size() == static_cast<std::size_t>(n1));
            for (int j = 0; j < n1; ++j) {
                assert(recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)].size() == static_cast<std::size_t>(n2));
                for (int k = 0; k < n2; ++k) {
                    recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)]
                        /= static_cast<T>(n0 * n1 * n2);
                }
                assert_real_close(input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                                  recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                                  eps);
            }
        }
    }
}

template <typename T>
void test_r2r_roundtrip_1d()
{
    const int n = 16;
    const T eps = tolerance<T>();
    const std::vector<T> input = make_real_signal_1d<T>(n);

    for (int run = 0; run < 5; ++run) {
        std::vector<T> forward;
        std::vector<T> recovered;

        clapfft::FFT::r2r_1d(input, forward, FFTW_REDFT10);
        clapfft::FFT::r2r_1d(forward, recovered, FFTW_REDFT01);

        for (std::size_t i = 0; i < recovered.size(); ++i) {
            recovered[i] /= static_cast<T>(2 * n);
        }

        assert_real_close(input, recovered, eps * static_cast<T>(4));
    }
}

template <typename T>
void test_r2r_roundtrip_2d()
{
    const int n0 = 4;
    const int n1 = 6;
    const T eps = tolerance<T>();

    std::vector<std::vector<T>> input(
        static_cast<std::size_t>(n0),
        std::vector<T>(static_cast<std::size_t>(n1)));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)] =
                static_cast<T>((i * 7 + j * 4) % 19) - static_cast<T>(6);
        }
    }

    for (int run = 0; run < 4; ++run) {
        std::vector<std::vector<T>> forward;
        std::vector<std::vector<T>> recovered;

        clapfft::FFT::r2r_2d(input, forward, FFTW_REDFT10, FFTW_REDFT10);
        clapfft::FFT::r2r_2d(forward, recovered, FFTW_REDFT01, FFTW_REDFT01);

        for (int i = 0; i < n0; ++i) {
            for (int j = 0; j < n1; ++j) {
                recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)]
                    /= static_cast<T>((2 * n0) * (2 * n1));
            }
            assert_real_close(input[static_cast<std::size_t>(i)], recovered[static_cast<std::size_t>(i)], eps * static_cast<T>(8));
        }
    }
}

template <typename T>
void test_r2r_roundtrip_3d()
{
    const int n0 = 3;
    const int n1 = 4;
    const int n2 = 6;
    const T eps = tolerance<T>();

    std::vector<std::vector<std::vector<T>>> input(
        static_cast<std::size_t>(n0),
        std::vector<std::vector<T>>(
            static_cast<std::size_t>(n1),
            std::vector<T>(static_cast<std::size_t>(n2))));

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] =
                    static_cast<T>((i * 3 + j * 5 + k * 7) % 23) - static_cast<T>(9);
            }
        }
    }

    for (int run = 0; run < 3; ++run) {
        std::vector<std::vector<std::vector<T>>> forward;
        std::vector<std::vector<std::vector<T>>> recovered;

        clapfft::FFT::r2r_3d(input, forward, FFTW_REDFT10, FFTW_REDFT10, FFTW_REDFT10);
        clapfft::FFT::r2r_3d(forward, recovered, FFTW_REDFT01, FFTW_REDFT01, FFTW_REDFT01);

        const T scale = static_cast<T>((2 * n0) * (2 * n1) * (2 * n2));
        for (int i = 0; i < n0; ++i) {
            for (int j = 0; j < n1; ++j) {
                for (int k = 0; k < n2; ++k) {
                    recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)][static_cast<std::size_t>(k)] /= scale;
                }
                assert_real_close(input[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                                  recovered[static_cast<std::size_t>(i)][static_cast<std::size_t>(j)],
                                  eps * static_cast<T>(16));
            }
        }
    }
}

template <typename T>
void run_all_precision_tests()
{
    test_c2c_roundtrip_1d<T>();
    test_c2c_roundtrip_2d<T>();
    test_c2c_roundtrip_3d<T>();
    test_r2c_c2r_roundtrip_1d<T>();
    test_r2c_c2r_roundtrip_2d<T>();
    test_r2c_c2r_roundtrip_3d<T>();
    test_r2r_roundtrip_1d<T>();
    test_r2r_roundtrip_2d<T>();
    test_r2r_roundtrip_3d<T>();
}

} // namespace

int main()
{
    run_all_precision_tests<float>();
    run_all_precision_tests<double>();
    run_all_precision_tests<long double>();

    std::cout << "All clapfft tests passed." << std::endl;
    return 0;
}
