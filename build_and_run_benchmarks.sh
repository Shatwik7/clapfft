#!/bin/sh

set -eu

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_TYPE="${BUILD_TYPE:-Release}"
BUILD_DIR="$ROOT_DIR/build-bench-${BUILD_TYPE}"

# Optional override knobs via environment variables
B1_ARGS="${B1_ARGS:-65536 100 5}"
B2_ARGS="${B2_ARGS:-32 32 32 100 5}"
B3_ARGS="${B3_ARGS:-32 32 32 100 5}"
B4_ARGS="${B4_ARGS:-32 32 32 20 3}"
B5_ARGS="${B5_ARGS:-16384 200 20 8}"

echo "--- Configuring project ---"
cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"

echo "--- Building benchmark targets ---"
for target in \
  benchmark_clapfft_vs_fftw \
  benchmark_c2r_3d_long_double \
  benchmark_r2r_3d_float \
  benchmark_c2c_3d_all_precisions \
  benchmark_parallel_c2c_1d_threads
do
  echo "Building: $target"
  cmake --build "$BUILD_DIR" --target "$target"
done

echo "--- Running benchmarks ---"
echo

echo ">>> benchmark_clapfft_vs_fftw $B1_ARGS"
"$BUILD_DIR/benchmark_clapfft_vs_fftw" $B1_ARGS

echo

echo ">>> benchmark_c2r_3d_long_double $B2_ARGS"
"$BUILD_DIR/benchmark_c2r_3d_long_double" $B2_ARGS

echo

echo ">>> benchmark_r2r_3d_float $B3_ARGS"
"$BUILD_DIR/benchmark_r2r_3d_float" $B3_ARGS

echo

echo ">>> benchmark_c2c_3d_all_precisions $B4_ARGS"
"$BUILD_DIR/benchmark_c2c_3d_all_precisions" $B4_ARGS

echo

echo ">>> benchmark_parallel_c2c_1d_threads $B5_ARGS"
"$BUILD_DIR/benchmark_parallel_c2c_1d_threads" $B5_ARGS

echo
echo "--- All benchmarks completed successfully ---"
