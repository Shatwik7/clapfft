cmake_minimum_required(VERSION 3.10)
project(clfft CXX)

# Standard C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find FFTW3 library. It's a common dependency.
# Users might need to install it, e.g., 'sudo apt-get install libfftw3-dev'
find_package(FFTW3 REQUIRED)

# Add our include directory so that #include <clapfft/clapfft_api.hpp> works
include_directories(include)

# --- Library ---
# Create a static library for our project
add_library(clapfft STATIC
    src/clapfft_api.cpp
)

# Link FFTW3 to our library. We need to link all three precision versions
# because our library supports float, double, and long double.
target_link_libraries(clapfft PRIVATE FFTW3::fftw3f FFTW3::fftw3 FFTW3::fftw3l)


# --- Test Executable ---
# Enable testing with CTest
enable_testing()

# Add the test executable
add_executable(test_c2c
    tests/test_c2c.cpp
)

# Link the test executable with our library.
# The test uses the float version, so it needs fftw3f.
target_link_libraries(test_c2c PRIVATE clapfft)

# Add a CTest test
add_test(NAME c2c_test COMMAND test_c2c)


# --- Installation ---
# This part is for making the library easily reusable in other projects.

# Install the public headers
install(DIRECTORY include/
    DESTINATION include
)

# Install the library binary
install(TARGETS clapfft
    EXPORT clapfft-targets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Export the clapfft target, making it importable by other CMake projects
install(EXPORT clapfft-targets
    FILE clapfft-targets.cmake
    NAMESPACE clapfft::
    DESTINATION lib/cmake/clapfft
)

# Generate a config file that other projects can use with find_package(clapfft)
include(CMakePackageConfigHelpers)

# This creates a clapfft-config.cmake file
configure_package_config_file(
    "cmake/clapfft-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/clapfft-config.cmake"
    INSTALL_DESTINATION "lib/cmake/clapfft"
    PATH_VARS "lib"
)

# This installs the generated clapfft-config.cmake
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/clapfft-config.cmake"
    DESTINATION "lib/cmake/clapfft"
)

# Create a simple version file
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/clapfft-config-version.cmake"
    VERSION "1.0.0"
    COMPATIBILITY AnyNewerVersion
)

# Install the version file
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/clapfft-config-version.cmake"
    DESTINATION "lib/cmake/clapfft"
)
