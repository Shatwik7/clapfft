#include <clapfft/clapfft_api.hpp>
#include <iostream>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <cassert>

template <typename T>
void print_vector_1d(const std::vector<std::complex<T>> &vec)
{
    for (const auto &c : vec)
    {
        std::cout << c << " ";
    }
    std::cout << std::endl;
}
void print_vector_1d(const std::vector<float> &vec){
    for (const auto &c : vec)
    {
        std::cout << c << " ";
    }
    std::cout << std::endl;
}

template< typename T>
void print_vector_2d(const std::vector<std::vector<std::complex<T>>>& vec)
{
    for (const auto& row : vec)
    {
        for (const auto& c : row)
        {
            std::cout << c << " ";
        }
        std::cout << std::endl;
    }
}


void test1()
{
    int n = 8;
    std::vector<std::complex<float>> input(n);
    for (int i = 0; i < n; ++i)
    {
        input[i] = std::complex<float>(i, 0);
    }

    std::cout << "Input:" << std::endl;
    print_vector_1d(input);

    std::vector<std::complex<float>> transformed;
    clapfft::FFT::fftw_c2c_1d(input, transformed, FFTW_FORWARD);

    std::cout << "Transformed (Forward):" << std::endl;
    print_vector_1d(transformed);

    std::vector<std::complex<float>> restored;
    clapfft::FFT::fftw_c2c_1d(transformed, restored, FFTW_BACKWARD);

    // FFTW's backward transform is unnormalized, so we need to divide by N.
    for (auto &c : restored)
    {
        c /= n;
    }

    std::cout << "Restored (Backward + normalization):" << std::endl;
    print_vector_1d(restored);

    // Check if the restored vector is close to the original input
    float epsilon = 1e-5;
    for (int i = 0; i < n; ++i)
    {
        assert(std::abs(input[i].real() - restored[i].real()) < epsilon);
        assert(std::abs(input[i].imag() - restored[i].imag()) < epsilon);
    }

    std::cout << "Test passed!" << std::endl;
}

void test_1d_float()
{

    std::cout<<"]\n \n \n TEST : 1D float";
    int n = 8;
    std::vector<std::complex<float>> input(n);
    for (int i = 0; i < n; ++i)
    {
        input[i] = std::complex<float>(i, 0);
    }

    std::cout << "Input:" << std::endl;
    print_vector_1d(input);

    std::vector<std::complex<float>> transformed;
    clapfft::FFT::fftw_c2c_1d(input, transformed, FFTW_FORWARD);

    std::cout<<"Transformation"<<std::endl;
    print_vector_1d(transformed);

    std::vector<std::complex<float>> restored;
    clapfft::FFT::fftw_c2c_1d(transformed, restored, FFTW_BACKWARD);

    std::cout << "Restored (Backward):" << std::endl;
    print_vector_1d(restored);

    std::cout << "Test passed! \n \n \n" << std::endl;
}

void test_1d_double()
{
    int n = 8;
    std::vector<std::complex<double>> input(n);
    for (int i = 0; i < n; ++i)
    {
        input[i] = std::complex<double>(i, 0);
    }

    std::cout << "Input:" << std::endl;
    print_vector_1d(input);

    std::vector<std::complex<double>> transformed;
    clapfft::FFT::fftw_c2c_1d(input, transformed, FFTW_FORWARD);

    std::vector<std::complex<double>> restored;
    clapfft::FFT::fftw_c2c_1d(transformed, restored, FFTW_BACKWARD);
    std::cout << "Test passed!" << std::endl;
}

void test_1d_long_double()
{
    int n = 8;
    std::vector<std::complex<long double>> input(n);
    for (int i = 0; i < n; ++i)
    {
        input[i] = std::complex<long double>(i, 0);
    }

    std::cout << "Input:" << std::endl;
    print_vector_1d(input);

    std::vector<std::complex<long double>> transformed;
    clapfft::FFT::fftw_c2c_1d(input, transformed, FFTW_FORWARD);

    std::vector<std::complex<long double>> restored;
    clapfft::FFT::fftw_c2c_1d(transformed, restored, FFTW_BACKWARD);

    std::cout << "Test passed!" << std::endl;
}

void test_2d_float()
{
    std::cout<<"TEST : 2D float";
    int n0 = 3;
    int n1 = 4;
    std::vector<std::vector<std::complex<float>>> input(n0, std::vector<std::complex<float>>(n1));
    for (int i = 0; i < n0; ++i)
    {
        for (int j = 0; j < n1; ++j)
        {
            input[i][j] = std::complex<float>(i * n1 + j, 0);
        }
    }
    std::cout << "Input:" << std::endl;
    print_vector_2d(input);

    std::vector<std::vector<std::complex<float>>> transformed;
    clapfft::FFT::fftw_c2c_2d(input, transformed, FFTW_FORWARD);

    std::cout<< "Transformed (Forward):" << std::endl;
    print_vector_2d(transformed);


    std::vector<std::vector<std::complex<float>>> restored;
    clapfft::FFT::fftw_c2c_2d(transformed, restored, FFTW_BACKWARD);

    std::cout<< "Restored (Backward):" << std::endl;
    print_vector_2d(restored);

    std::cout << "Test passed!" << std::endl;

}

void test_2d_double()
{
    int n0 = 3;
    int n1 = 4;
    std::vector<std::vector<std::complex<double>>> input(n0, std::vector<std::complex<double>>(n1));
    for (int i = 0; i < n0; ++i)
    {
        for (int j = 0; j < n1; ++j)
        {
            input[i][j] = std::complex<double>(i * n1 + j, 0);
        }
    }
    std::cout << "Input:" << std::endl;
    print_vector_2d(input);

    std::vector<std::vector<std::complex<double>>> transformed;
    clapfft::FFT::fftw_c2c_2d(input, transformed, FFTW_FORWARD);

    print_vector_2d(transformed);

    clapfft::FFT::fftw_c2c_2d(transformed, input, FFTW_BACKWARD);

    print_vector_2d(input);
}

void c2r_test()
{
    std::cout<<" \n \n TEST : c2r";
    int n = 8;
    std::vector<std::complex<float>> input(n);
    for (int i = 0; i < n; ++i)
    {
        input[i] = std::complex<float>(i, 0);
    }

    std::cout << "Input:" << std::endl;
    print_vector_1d(input);

    std::vector<float> transformed;
    clapfft::FFT::fftw_c2r_1d(input, transformed);

    std::cout << "Transformed (Forward):" << std::endl;
    print_vector_1d(transformed);
    
    std::cout<< "Test passed!" << std::endl;
}
void c2r_test();
void r2c_c2r_test();

void r2c_c2r_test()
{
    // 1D test
    {
        int n = 8;
        std::vector<float> input(n);
        for (int i = 0; i < n; ++i)
        {
            input[i] = i;
        }

        std::vector<std::complex<float>> complex_output;
        clapfft::FFT::fftw_r2c_1d(input, complex_output);

        std::vector<float> real_output;
        clapfft::FFT::fftw_c2r_1d(complex_output, real_output);

        // Normalization
        for (auto &val : real_output)
        {
            val /= n;
        }

        float epsilon = 1e-5;
        for (int i = 0; i < n; ++i)
        {
            assert(std::abs(input[i] - real_output[i]) < epsilon);
        }
        std::cout << "1D r2c/c2r test passed!" << std::endl;
    }

    // 2D test
    {
        int n0 = 3;
        int n1 = 4;
        std::vector<std::vector<float>> input(n0, std::vector<float>(n1));
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                input[i][j] = i * n1 + j;
            }
        }

        std::vector<std::vector<std::complex<float>>> complex_output;
        clapfft::FFT::fftw_r2c_2d(input, complex_output);

        std::vector<std::vector<float>> real_output;
        clapfft::FFT::fftw_c2r_2d(complex_output, real_output);

        // Normalization
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                real_output[i][j] /= (n0 * n1);
            }
        }

        float epsilon = 1e-5;
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                assert(std::abs(input[i][j] - real_output[i][j]) < epsilon);
            }
        }
        std::cout << "2D r2c/c2r test passed!" << std::endl;
    }

    // 3D test
    {
        int n0 = 2;
        int n1 = 3;
        int n2 = 4;
        std::vector<std::vector<std::vector<float>>> input(n0, std::vector<std::vector<float>>(n1, std::vector<float>(n2)));
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                for(int k = 0; k < n2; ++k)
                {
                    input[i][j][k] = i * n1 * n2 + j * n2 + k;
                }
            }
        }

        std::vector<std::vector<std::vector<std::complex<float>>>> complex_output;
        clapfft::FFT::fftw_r2c_3d(input, complex_output);

        std::vector<std::vector<std::vector<float>>> real_output;
        clapfft::FFT::fftw_c2r_3d(complex_output, real_output);

        // Normalization
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                for(int k = 0; k < n2; ++k)
                {
                    real_output[i][j][k] /= (n0 * n1 * n2);
                }
            }
        }

        float epsilon = 1e-5;
        for (int i = 0; i < n0; ++i)
        {
            for (int j = 0; j < n1; ++j)
            {
                for(int k = 0; k < n2; ++k)
                {
                    assert(std::abs(input[i][j][k] - real_output[i][j][k]) < epsilon);
                }
            }
        }
        std::cout << "3D r2c/c2r test passed!" << std::endl;
    }
}

int main()
{
    test1();
    test_1d_float();
    test_1d_double();
    test_1d_long_double();
    test_2d_float();
    test_2d_double();

    c2r_test();
    r2c_c2r_test();
    return 0;
}
