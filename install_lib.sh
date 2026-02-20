#!/bin/bash
# This script builds and "installs" the clapfft library to a local directory.

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
BUILD_DIR="build"
# This is where the final library and headers will be placed.
INSTALL_DIR="install"

# --- Build ---
echo "--- Configuring and Building ---"
# Create a build directory (or clear it)
rm -rf $BUILD_DIR
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# Run CMake. We specify the CMAKE_INSTALL_PREFIX to control where the library is installed.
cmake -DCMAKE_INSTALL_PREFIX=../$INSTALL_DIR ..

# Run make to compile the library.
make

# --- Installation ---
echo "--- Installing library to ../$INSTALL_DIR ---"
# This command runs the 'install' steps defined in our CMakeLists.txt
make install
cd ..

echo ""
echo "--- Library installation complete! ---"
echo "You can find the compiled library and headers in the './$INSTALL_DIR' directory."
echo "To use it in another CMake project, add the following to your CMakeLists.txt:"
echo ""
echo '    find_package(clapfft REQUIRED HINTS "${CMAKE_CURRENT_SOURCE_DIR}/../clfft/'$INSTALL_DIR'/lib/cmake/clapfft")'
echo '    target_link_libraries(your_target_name PRIVATE clapfft::clapfft)'
echo ""
