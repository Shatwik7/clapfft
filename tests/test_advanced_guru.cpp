#include <fftw3.h>
#include <cassert>
#include <cmath>
#include <complex>
#include <iostream>
#include <vector>

template <typename T>
void run_guru_dft_roundtrip_test();

template <>
void run_guru_dft_roundtrip_test<double>()
{
    const int n = 8;
    const int howmany = 3;
    const double eps = 1e-9;

    std::vector<std::complex<double>> input(static_cast<std::size_t>(n * howmany));
    for (int b = 0; b < howmany; ++b)
    {
        for (int i = 0; i < n; ++i)
        {
            const std::size_t idx = static_cast<std::size_t>(b * n + i);
            input[idx] = std::complex<double>(b + i * 0.5, (i % 3) - b);
        }
    }

    std::vector<std::complex<double>> forward(static_cast<std::size_t>(n * howmany));
    std::vector<std::complex<double>> recovered(static_cast<std::size_t>(n * howmany));

    fftw_iodim dims[1];
    dims[0].n = n;
    dims[0].is = 1;
    dims[0].os = 1;

    fftw_iodim howmany_dims[1];
    howmany_dims[0].n = howmany;
    howmany_dims[0].is = n;
    howmany_dims[0].os = n;

    fftw_plan p_fwd = fftw_plan_guru_dft(1, dims,
                                         1, howmany_dims,
                                         reinterpret_cast<fftw_complex *>(input.data()),
                                         reinterpret_cast<fftw_complex *>(forward.data()),
                                         FFTW_FORWARD,
                                         FFTW_ESTIMATE);
    assert(p_fwd != nullptr);
    fftw_execute_dft(p_fwd,
                     reinterpret_cast<fftw_complex *>(input.data()),
                     reinterpret_cast<fftw_complex *>(forward.data()));
    fftw_destroy_plan(p_fwd);

    fftw_plan p_inv = fftw_plan_guru_dft(1, dims,
                                         1, howmany_dims,
                                         reinterpret_cast<fftw_complex *>(forward.data()),
                                         reinterpret_cast<fftw_complex *>(recovered.data()),
                                         FFTW_BACKWARD,
                                         FFTW_ESTIMATE);
    assert(p_inv != nullptr);
    fftw_execute_dft(p_inv,
                     reinterpret_cast<fftw_complex *>(forward.data()),
                     reinterpret_cast<fftw_complex *>(recovered.data()));
    fftw_destroy_plan(p_inv);

    for (std::size_t i = 0; i < recovered.size(); ++i)
    {
        recovered[i] /= static_cast<double>(n);
        assert(std::abs(recovered[i].real() - input[i].real()) <= eps);
        assert(std::abs(recovered[i].imag() - input[i].imag()) <= eps);
    }
}

void run_guru64_dft_plan_test()
{
    const int n = 16;
    std::vector<std::complex<double>> in(static_cast<std::size_t>(n));
    std::vector<std::complex<double>> out(static_cast<std::size_t>(n));

    fftw_iodim64 dims[1];
    dims[0].n = static_cast<ptrdiff_t>(n);
    dims[0].is = 1;
    dims[0].os = 1;

    fftw_plan plan64 = fftw_plan_guru64_dft(1, dims,
                                            0, nullptr,
                                            reinterpret_cast<fftw_complex *>(in.data()),
                                            reinterpret_cast<fftw_complex *>(out.data()),
                                            FFTW_FORWARD,
                                            FFTW_ESTIMATE);
    assert(plan64 != nullptr);
    fftw_destroy_plan(plan64);
}

int main()
{
    run_guru_dft_roundtrip_test<double>();
    run_guru64_dft_plan_test();

    std::cout << "advanced_guru tests passed." << std::endl;
    return 0;
}
