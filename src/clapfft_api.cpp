#include <clapfft/clapfft_api.hpp>
#include <clapfft/fftw_traits.hpp>
#include <vector>
#include <complex>

namespace clapfft {

//c2c
// 1D
template <typename T>
void FFT::fftw_c2c_1d(const std::vector<std::complex<T>>& input, std::vector<std::complex<T>>& output, int sign) {
    using traits = fftw_trait<T>;
    int n = input.size();
    output.resize(n);

    auto in_ptr = reinterpret_cast<typename traits::complex_type*>(const_cast<std::complex<T>*>(input.data()));
    auto out_ptr = reinterpret_cast<typename traits::complex_type*>(output.data());

    typename traits::plan_type plan = traits::plan_dft_1d(n, in_ptr, out_ptr, sign, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);
}

// 2D
template <typename T>
void FFT::fftw_c2c_2d(const std::vector<std::vector<std::complex<T>>>& input, std::vector<std::vector<std::complex<T>>>& output, int sign) {
    using traits = fftw_trait<T>;
    int n0 = input.size();
    if (n0 == 0) return;
    int n1 = input[0].size();

    output.resize(n0, std::vector<std::complex<T>>(n1));

    std::vector<std::complex<T>> flat_input(n0 * n1);
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            flat_input[i * n1 + j] = input[i][j];
        }
    }
    std::vector<std::complex<T>> flat_output(n0 * n1);

    auto in_ptr = reinterpret_cast<typename traits::complex_type*>(flat_input.data());
    auto out_ptr = reinterpret_cast<typename traits::complex_type*>(flat_output.data());

    typename traits::plan_type plan = traits::plan_dft_2d(n0, n1, in_ptr, out_ptr, sign, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);
    
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            output[i][j] = flat_output[i * n1 + j];
        }
    }
}

// 3D
template <typename T>
void FFT::fftw_c2c_3d(const std::vector<std::vector<std::vector<std::complex<T>>>>& input, std::vector<std::vector<std::vector<std::complex<T>>>>& output, int sign) {
    using traits = fftw_trait<T>;
    int n0 = input.size();
    if (n0 == 0) return;
    int n1 = input[0].size();
    if (n1 == 0) return;
    int n2 = input[0][0].size();

    output.resize(n0, std::vector<std::vector<std::complex<T>>>(n1, std::vector<std::complex<T>>(n2)));

    std::vector<std::complex<T>> flat_input(n0 * n1 * n2);
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                flat_input[i * n1 * n2 + j * n2 + k] = input[i][j][k];
            }
        }
    }
    std::vector<std::complex<T>> flat_output(n0 * n1 * n2);

    auto in_ptr = reinterpret_cast<typename traits::complex_type*>(flat_input.data());
    auto out_ptr = reinterpret_cast<typename traits::complex_type*>(flat_output.data());

    typename traits::plan_type plan = traits::plan_dft_3d(n0, n1, n2, in_ptr, out_ptr, sign, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                output[i][j][k] = flat_output[i * n1 * n2 + j * n2 + k];
            }
        }
    }
}


//c2r
//1d
template <typename T>
void FFT::fftw_c2r_1d(const std::vector<std::complex<T>>& input, std::vector<T>& output) {
    using traits = fftw_trait<T>;
    int n_complex = input.size();
    int n_real = 2 * (n_complex - 1);
    output.resize(n_real);

    auto in_ptr = reinterpret_cast<typename traits::complex_type*>(const_cast<std::complex<T>*>(input.data()));
    auto out_ptr = output.data();

    typename traits::plan_type plan = traits::plan_dft_c2r_1d(n_real, in_ptr, out_ptr, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);
}

//c2r 2d
template <typename T>
void FFT::fftw_c2r_2d(const std::vector<std::vector<std::complex<T>>>& input, std::vector<std::vector<T>>& output) {
    using traits = fftw_trait<T>;
    int n0 = input.size();
    if (n0 == 0) return;
    int n1_complex = input[0].size();
    int n1_real = 2 * (n1_complex - 1);

    output.resize(n0, std::vector<T>(n1_real));

    std::vector<std::complex<T>> flat_input(n0 * n1_complex);
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1_complex; ++j) {
            flat_input[i * n1_complex + j] = input[i][j];
        }
    }
    std::vector<T> flat_output(n0 * n1_real);

    auto in_ptr = reinterpret_cast<typename traits::complex_type*>(flat_input.data());
    auto out_ptr = flat_output.data();

    typename traits::plan_type plan = traits::plan_dft_c2r_2d(n0, n1_real, in_ptr, out_ptr, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);
    
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1_real; ++j) {
            output[i][j] = flat_output[i * n1_real + j];
        }
    }
}

//c2r 3d
template <typename T>
void FFT::fftw_c2r_3d(const std::vector<std::vector<std::vector<std::complex<T>>>>& input, std::vector<std::vector<std::vector<T>>>& output) {
    using traits = fftw_trait<T>;
    int n0 = input.size();
    if (n0 == 0) return;
    int n1 = input[0].size();
    if (n1 == 0) return;
    int n2_complex = input[0][0].size();
    int n2_real = 2 * (n2_complex - 1);

    output.resize(n0, std::vector<std::vector<T>>(n1, std::vector<T>(n2_real)));

    std::vector<std::complex<T>> flat_input(n0 * n1 * n2_complex);
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2_complex; ++k) {
                flat_input[i * n1 * n2_complex + j * n2_complex + k] = input[i][j][k];
            }
        }
    }
    std::vector<T> flat_output(n0 * n1 * n2_real);

    auto in_ptr = reinterpret_cast<typename traits::complex_type*>(flat_input.data());
    auto out_ptr = flat_output.data();

    typename traits::plan_type plan = traits::plan_dft_c2r_3d(n0, n1, n2_real, in_ptr, out_ptr, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2_real; ++k) {
                output[i][j][k] = flat_output[i * n1 * n2_real + j * n2_real + k];
            }
        }
    }
}


//r2c 1d
template <typename T>
void FFT::fftw_r2c_1d(const std::vector<T>& input, std::vector<std::complex<T>>& output) {
    using traits = fftw_trait<T>;
    int n = input.size();
    output.resize(n / 2 + 1);

    auto in_ptr = const_cast<T*>(input.data());
    auto out_ptr = reinterpret_cast<typename traits::complex_type*>(output.data());

    typename traits::plan_type plan = traits::plan_dft_r2c_1d(n, in_ptr, out_ptr, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);
}

//r2c 2d
template <typename T>
void FFT::fftw_r2c_2d(const std::vector<std::vector<T>>& input, std::vector<std::vector<std::complex<T>>>& output) {
    using traits = fftw_trait<T>;
    int n0 = input.size();
    if (n0 == 0) return;
    int n1 = input[0].size();

    output.resize(n0, std::vector<std::complex<T>>(n1 / 2 + 1));

    std::vector<T> flat_input(n0 * n1);
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            flat_input[i * n1 + j] = input[i][j];
        }
    }
    std::vector<std::complex<T>> flat_output(n0 * (n1 / 2 + 1));

    auto in_ptr = const_cast<T*>(flat_input.data());
    auto out_ptr = reinterpret_cast<typename traits::complex_type*>(flat_output.data());

    typename traits::plan_type plan = traits::plan_dft_r2c_2d(n0, n1, in_ptr, out_ptr, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);
    
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < (n1 / 2 + 1); ++j) {
            output[i][j] = flat_output[i * (n1 / 2 + 1) + j];
        }
    }
}

//r2c 3d
template <typename T>
void FFT::fftw_r2c_3d(const std::vector<std::vector<std::vector<T>>>& input, std::vector<std::vector<std::vector<std::complex<T>>>>& output) {
    using traits = fftw_trait<T>;
    int n0 = input.size();
    if (n0 == 0) return;
    int n1 = input[0].size();
    if (n1 == 0) return;
    int n2 = input[0][0].size();

    output.resize(n0, std::vector<std::vector<std::complex<T>>>(n1, std::vector<std::complex<T>>(n2 / 2 + 1)));

    std::vector<T> flat_input(n0 * n1 * n2);
    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < n2; ++k) {
                flat_input[i * n1 * n2 + j * n2 + k] = input[i][j][k];
            }
        }
    }
    std::vector<std::complex<T>> flat_output(n0 * n1 * (n2 / 2 + 1));

    auto in_ptr = const_cast<T*>(flat_input.data());
    auto out_ptr = reinterpret_cast<typename traits::complex_type*>(flat_output.data());

    typename traits::plan_type plan = traits::plan_dft_r2c_3d(n0, n1, n2, in_ptr, out_ptr, FFTW_ESTIMATE);

    traits::execute(plan);
    traits::destroy_plan(plan);

    for (int i = 0; i < n0; ++i) {
        for (int j = 0; j < n1; ++j) {
            for (int k = 0; k < (n2 / 2 + 1); ++k) {
                output[i][j][k] = flat_output[i * n1 * (n2 / 2 + 1) + j * (n2 / 2 + 1) + k];
            }
        }
    }
}

// Explicit template instantiations

//c2c
template void FFT::fftw_c2c_1d<float>(const std::vector<std::complex<float>>&, std::vector<std::complex<float>>&, int);
template void FFT::fftw_c2c_1d<double>(const std::vector<std::complex<double>>&, std::vector<std::complex<double>>&, int);
template void FFT::fftw_c2c_1d<long double>(const std::vector<std::complex<long double>>&, std::vector<std::complex<long double>>&, int);

template void FFT::fftw_c2c_2d<float>(const std::vector<std::vector<std::complex<float>>>&, std::vector<std::vector<std::complex<float>>>&, int);
template void FFT::fftw_c2c_2d<double>(const std::vector<std::vector<std::complex<double>>>&, std::vector<std::vector<std::complex<double>>>&, int);
template void FFT::fftw_c2c_2d<long double>(const std::vector<std::vector<std::complex<long double>>>&, std::vector<std::vector<std::complex<long double>>>&, int);

template void FFT::fftw_c2c_3d<float>(const std::vector<std::vector<std::vector<std::complex<float>>>>&, std::vector<std::vector<std::vector<std::complex<float>>>>&, int);
template void FFT::fftw_c2c_3d<double>(const std::vector<std::vector<std::vector<std::complex<double>>>>&, std::vector<std::vector<std::vector<std::complex<double>>>>&, int);
template void FFT::fftw_c2c_3d<long double>(const std::vector<std::vector<std::vector<std::complex<long double>>>>&, std::vector<std::vector<std::vector<std::complex<long double>>>>&, int);

//c2r
template void FFT::fftw_c2r_1d<float>(const std::vector<std::complex<float>>&, std::vector<float>&);
template void FFT::fftw_c2r_1d<long double>(const std::vector<std::complex<long double>>&, std::vector<long double>&);
template void FFT::fftw_c2r_1d<double>(const std::vector<std::complex<double>>&, std::vector<double>&);

template void FFT::fftw_c2r_2d<float>(const std::vector<std::vector<std::complex<float>>>&, std::vector<std::vector<float>>&);
template void FFT::fftw_c2r_2d<long double>(const std::vector<std::vector<std::complex<long double>>>&, std::vector<std::vector<long double>>&);
template void FFT::fftw_c2r_2d<double>(const std::vector<std::vector<std::complex<double>>>&, std::vector<std::vector<double>>&);

template void FFT::fftw_c2r_3d<float>(const std::vector<std::vector<std::vector<std::complex<float>>>>&, std::vector<std::vector<std::vector<float>>>&);
template void FFT::fftw_c2r_3d<long double>(const std::vector<std::vector<std::vector<std::complex<long double>>>>&, std::vector<std::vector<std::vector<long double>>>&);
template void FFT::fftw_c2r_3d<double>(const std::vector<std::vector<std::vector<std::complex<double>>>>&, std::vector<std::vector<std::vector<double>>>&);

//r2c
template void FFT::fftw_r2c_1d<float>(const std::vector<float>&, std::vector<std::complex<float>>&);
template void FFT::fftw_r2c_1d<long double>(const std::vector<long double>&, std::vector<std::complex<long double>>&);
template void FFT::fftw_r2c_1d<double>(const std::vector<double>&, std::vector<std::complex<double>>&);

template void FFT::fftw_r2c_2d<float>(const std::vector<std::vector<float>>&, std::vector<std::vector<std::complex<float>>>&);
template void FFT::fftw_r2c_2d<long double>(const std::vector<std::vector<long double>>&, std::vector<std::vector<std::complex<long double>>>&);
template void FFT::fftw_r2c_2d<double>(const std::vector<std::vector<double>>&, std::vector<std::vector<std::complex<double>>>&);

template void FFT::fftw_r2c_3d<float>(const std::vector<std::vector<std::vector<float>>>&, std::vector<std::vector<std::vector<std::complex<float>>>>&);
template void FFT::fftw_r2c_3d<long double>(const std::vector<std::vector<std::vector<long double>>>&, std::vector<std::vector<std::vector<std::complex<long double>>>>&);
template void FFT::fftw_r2c_3d<double>(const std::vector<std::vector<std::vector<double>>>&, std::vector<std::vector<std::vector<std::complex<double>>>>&);

} // namespace clapfft
