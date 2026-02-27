#ifndef CLAPFFT_API_HPP
#define CLAPFFT_API_HPP
#include <vector>
#include <complex>

namespace clapfft
{

    class FFT
    {
    public:
        template <typename T>
        static void c2c_1d(const std::vector<std::complex<T>> &input, std::vector<std::complex<T>> &output, int sign);

        template <typename T>
        static void c2c_2d(const std::vector<std::vector<std::complex<T>>> &input, std::vector<std::vector<std::complex<T>>> &output, int sign);

        template <typename T>
        static void c2c_3d(const std::vector<std::vector<std::vector<std::complex<T>>>> &input, std::vector<std::vector<std::vector<std::complex<T>>>> &output, int sign);

        template <typename T>
        static void c2r_1d(const std::vector<std::complex<T>> &input, std::vector<T> &output);
        template <typename T>
        static void c2r_2d(const std::vector<std::vector<std::complex<T>>> &input, std::vector<std::vector<T>> &output);
        template <typename T>
        static void c2r_3d(const std::vector<std::vector<std::vector<std::complex<T>>>> &input, std::vector<std::vector<std::vector<T>>> &output);

        template <typename T>
        static void r2c_1d(const std::vector<T> &input, std::vector<std::complex<T>> &output);
        template <typename T>
        static void r2c_2d(const std::vector<std::vector<T>> &input, std::vector<std::vector<std::complex<T>>> &output);
        template <typename T>
        static void r2c_3d(const std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<std::vector<std::complex<T>>>> &output);

        template <typename T>
        static void r2r_1d(const std::vector<T> &input, std::vector<T> &output, int kind);
        template <typename T>
        static void r2r_2d(const std::vector<std::vector<T>> &input, std::vector<std::vector<T>> &output, int kind0, int kind1);
        template <typename T>
        static void r2r_3d(const std::vector<std::vector<std::vector<T>>> &input, std::vector<std::vector<std::vector<T>>> &output, int kind0, int kind1, int kind2);
    };

} // namespace clapfft

#endif
