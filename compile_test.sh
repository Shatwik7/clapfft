# explicity compiles the test.cpp using g++

g++ tests/test_c2c.cpp src/clapfft_api.cpp \
    -I./include -std=c++17 \
    -DCLAPFFT_HAS_FFTW3F -DCLAPFFT_HAS_FFTW3L \
    -lfftw3 -lfftw3f -lfftw3l -lm \
    -o fft_test

