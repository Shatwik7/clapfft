#include <fftw3.h>
#include <clapfft/clapfft_api.hpp>
#include <clapfft/fft_plan_cache.hpp>
#include <clapfft/wisdom.hpp>
#include <fftw3.h>
#include <iostream>
#include <vector>
#include <complex>
#include <cstdio> // for remove
#include <string>

void test_wisdom_string()
{
    std::cout << "Testing Wisdom String Export/Import..." << std::endl;

    // 1. Create a plan to ensure some wisdom is generated
    std::vector<std::complex<double>> in(100), out(100);
    clapfft::FFT::c2c_1d(in, out, FFTW_BACKWARD);

    // 2. Export to string
    std::string wisdom_str = clapfft::Wisdom::export_to_string<double>();
    if (wisdom_str.empty())
    {
        std::cerr << "Failed to export wisdom to string (result was empty)." << std::endl;
        exit(1);
    }
    std::cout << "Exported wisdom string length: " << wisdom_str.length() << std::endl;

    // 3. Import from string
    bool success = clapfft::Wisdom::import_from_string<double>(wisdom_str);
    if (!success)
    {
        std::cerr << "Failed to import wisdom from string." << std::endl;
        exit(1);
    }
    std::cout << "Successfully imported wisdom from string." << std::endl;
}

void test_wisdom_file()
{
    std::cout << "Testing Wisdom File Export/Import..." << std::endl;
    std::string filename = "test_wisdom.fftw";

    // 1. Export to filename
    clapfft::Wisdom::export_to_filename<double>(filename);

    // 2. Import from filename
    bool success = clapfft::Wisdom::import_from_filename<double>(filename);
    if (!success)
    {
        std::cerr << "Failed to import wisdom from file." << std::endl;
        std::remove(filename.c_str());
        exit(1);
    }
    std::cout << "Successfully imported wisdom from file." << std::endl;

    // 3. Negative test: Import from non-existent file
    bool fail = clapfft::Wisdom::import_from_filename<double>("non_existent_file_12345.fftw");
    if (fail)
    {
        std::cerr << "Error: Should have failed to import from non-existent file." << std::endl;
        exit(1);
    }

    // Cleanup
    std::remove(filename.c_str());
}

void test_wisdom_c2c_2d_long_double()
{
    std::cout << "Testing Wisdom c2c_2d long double reuse..." << std::endl;

    int n0 = 8;
    int n1 = 8;

    // 1. Run FFT to generate wisdom
    {
        std::vector<std::vector<std::complex<long double>>> in(n0, std::vector<std::complex<long double>>(n1));
        std::vector<std::vector<std::complex<long double>>> out;
        clapfft::FFT::c2c_2d(in, out, FFTW_FORWARD);
    }

    // 2. Export wisdom
    std::string wisdom_str = clapfft::Wisdom::export_to_string<long double>();
    if (wisdom_str.empty())
    {
        std::cerr << "Failed to export long double wisdom." << std::endl;
        exit(1);
    }

    // 3. Cleanup cache to force plan recreation
    clapfft::PlanCache<long double>::cleanup();

    // 4. Import wisdom
    if (!clapfft::Wisdom::import_from_string<long double>(wisdom_str))
    {
        std::cerr << "Failed to import long double wisdom." << std::endl;
        exit(1);
    }

    // 5. Run FFT again (should use wisdom)
    {
        std::vector<std::vector<std::complex<long double>>> in(n0, std::vector<std::complex<long double>>(n1));
        std::vector<std::vector<std::complex<long double>>> out;
        clapfft::FFT::c2c_2d(in, out, FFTW_FORWARD);
    }

    std::cout << "Successfully verified c2c_2d long double wisdom cycle." << std::endl;
}

void test_flag_sensitive_plan_cache()
{
    std::cout << "Testing plan cache flag sensitivity..." << std::endl;
    using wrapper_t = clapfft::PlanCache<double>::Wrapper;
    auto w1 = clapfft::PlanCache<double>::get_c2c_1d(16, FFTW_FORWARD, clapfft::CLAP_FFT_ESTIMATE);
    auto w2 = clapfft::PlanCache<double>::get_c2c_1d(16, FFTW_FORWARD, clapfft::CLAP_FFT_ESTIMATE);
    if (w1 != w2) {
        std::cerr << "Cache failed to return same wrapper for identical flags." << std::endl;
        exit(1);
    }
    auto w3 = clapfft::PlanCache<double>::get_c2c_1d(16, FFTW_FORWARD, clapfft::CLAP_FFT_MEASURE);
    if (w3 == w1) {
        std::cerr << "Cache did not distinguish between different planning flags." << std::endl;
        exit(1);
    }
    std::cout << "Flag sensitivity test passed." << std::endl;
}

int main()
{
    test_wisdom_string();
    test_wisdom_file();
    test_wisdom_c2c_2d_long_double();
    test_flag_sensitive_plan_cache();
    std::cout << "All wisdom tests passed!" << std::endl;
    return 0;
}

