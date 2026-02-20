#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Create a build directory if it doesn't exist
BUILD_DIR="build"
mkdir -p $BUILD_DIR

# --- Build ---
echo "--- Configuring and Building ---"
# Move into the build directory
cd $BUILD_DIR
# Run CMake to configure the project. This generates Makefiles.
cmake ..
# Run make to compile the code (library and test executable).
make

# --- Test ---
echo "--- Running Test ---"
# Run the test executable. ctest is a tool that comes with CMake.
ctest --verbose

echo "--- Build and test script finished successfully! ---"
