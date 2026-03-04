#!/bin/bash

# Get the absolute path of the directory where this script is located
PROJECT_ROOT="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Define paths relative to the project root
INCLUDE_DIR="${PROJECT_ROOT}/include"
LIB_DIR="${PROJECT_ROOT}/build"

echo "--- Compiling example/main.cpp ---"

# Compile the code
# -I: Header search path
# -L: Library search path
# -Wl,-rpath: Set runtime library search path for the executable
# -lclapfft: Link against libclapfft.so/a
parascc -std=c++17 example/main.cpp \
    -I"${INCLUDE_DIR}" \
    -L"${LIB_DIR}" \
    -Wl,-rpath,"${LIB_DIR}" \
    -lclapfft \
    -o run_for_parass

# Check if compilation was successful
if [ $? -eq 0 ]; then
    echo "--- Compilation successful! ---"
else
    echo "--- Compilation failed ---"
    exit 1
fi
