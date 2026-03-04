cmake_minimum_required(VERSION 3.10)
project(clfft CXX)

# Include GNUInstallDirs to use standard installation directories
include(GNUInstallDirs)

# Standard C++
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(CLAPFFT_LINK_STATIC_FFTW "Link FFTW statically into shared clapfft library" ON)

# Find FFTW3 using pkg-config
find_package(PkgConfig REQUIRED)
pkg_check_modules(FFTW3F REQUIRED fftw3f)
pkg_check_modules(FFTW3D REQUIRED fftw3)
pkg_check_modules(FFTW3L REQUIRED fftw3l)

# Add our include directory so that #include <clapfft/clapfft_api.hpp> works
include_directories(include)
include_directories(${FFTW3F_INCLUDE_DIRS})
include_directories(${FFTW3D_INCLUDE_DIRS})
include_directories(${FFTW3L_INCLUDE_DIRS})

# --- Library ---
# Create a shared library for our project
add_library(clapfft SHARED
    src/clapfft_api.cpp
    src/advanced_fft.cpp
    src/guru_fft.cpp
)

set_target_properties(clapfft PROPERTIES
    POSITION_INDEPENDENT_CODE ON
)

# Link FFTW3 to our library. We need all three precision versions because the
# wrapper supports float, double, and long double.
if(CLAPFFT_LINK_STATIC_FFTW)
    set(_saved_suffixes "${CMAKE_FIND_LIBRARY_SUFFIXES}")
    set(CMAKE_FIND_LIBRARY_SUFFIXES .a)

    find_library(FFTW3F_STATIC_LIB NAMES fftw3f HINTS ${FFTW3F_LIBRARY_DIRS} REQUIRED)
    find_library(FFTW3D_STATIC_LIB NAMES fftw3 HINTS ${FFTW3D_LIBRARY_DIRS} REQUIRED)
    find_library(FFTW3L_STATIC_LIB NAMES fftw3l HINTS ${FFTW3L_LIBRARY_DIRS} REQUIRED)

    set(CMAKE_FIND_LIBRARY_SUFFIXES "${_saved_suffixes}")

    target_link_libraries(clapfft PRIVATE
        ${FFTW3F_STATIC_LIB}
        ${FFTW3D_STATIC_LIB}
        ${FFTW3L_STATIC_LIB}
        m
    )
else()
    target_link_libraries(clapfft PRIVATE ${FFTW3F_LIBRARIES} ${FFTW3D_LIBRARIES} ${FFTW3L_LIBRARIES})
endif()


# --- Test Executable ---
# Enable testing with CTest
enable_testing()

# Add separate test executables per transform and dimension
set(CLAPFFT_TEST_CASES
    c2c_1d
    c2c_2d
    c2c_3d
    r2c_1d
    r2c_2d
    r2c_3d
    c2r_1d
    c2r_2d
    c2r_3d
    r2r_1d
    r2r_2d
    r2r_3d
    advanced_many_dft
    advanced_many_dft_r2c
    advanced_many_dft_c2r
    advanced_many_r2r
    advanced_guru
)

foreach(case IN LISTS CLAPFFT_TEST_CASES)
    set(target_name "test_${case}")
    add_executable(${target_name}
        tests/test_${case}.cpp
    )
    target_link_libraries(${target_name} PRIVATE clapfft)
    add_test(NAME ${case}_test COMMAND ${target_name})
endforeach()


# --- Installation ---
# This part is for making the library easily reusable in other projects.

set(CMAKECONFIG_INSTALL_DIR "${CMAKE_INSTALL_LIBDIR}/cmake/clapfft")

# Install the public headers
install(DIRECTORY include/
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# Install the library binary
install(TARGETS clapfft
    EXPORT clapfft-targets
    ARCHIVE DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    LIBRARY DESTINATION "${CMAKE_INSTALL_LIBDIR}"
    RUNTIME DESTINATION "${CMAKE_INSTALL_BINDIR}"
)

# Export the clapfft target, making it importable by other CMake projects
install(EXPORT clapfft-targets
    FILE clapfft-targets.cmake
    NAMESPACE clapfft::
    DESTINATION ${CMAKECONFIG_INSTALL_DIR}
)

# Generate a config file that other projects can use with find_package(clapfft)
include(CMakePackageConfigHelpers)

# This creates a clapfft-config.cmake file
configure_package_config_file(
    "cmake/clapfft-config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/clapfft-config.cmake"
    INSTALL_DESTINATION ${CMAKECONFIG_INSTALL_DIR}
    PATH_VARS CMAKE_INSTALL_LIBDIR
)

# This installs the generated clapfft-config.cmake
install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/clapfft-config.cmake"
    DESTINATION ${CMAKECONFIG_INSTALL_DIR}
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
