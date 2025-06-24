# CUDA Matrix Transposition – Project Summary

## Project Overview

This project focused on implementing and optimizing matrix transposition using CUDA. The main goal was to explore different GPU programming techniques and see how much faster we could make the operation compared to a standard CPU approach.

## Key Results

- **Maximum Speedup:** Up to 183x faster than the CPU version for large matrices (2048x1024).
- **Peak Throughput:** Achieved 195 GB/s using shared memory.
- **Tested Matrix Sizes:** Ranged from 512x256 up to 4096x2048.
- **All Implementations Verified:** Every version produced correct results.

## What Was Implemented

1. **Basic CUDA Transpose:** Started with a straightforward GPU kernel, which already gave a noticeable speedup.
2. **Coalesced Memory Access:** Improved memory access patterns, leading to a huge jump in performance.
3. **Shared Memory Optimization:** Used shared memory to further boost speed, especially for larger matrices.
4. **Unified Memory:** Made memory management easier and more flexible.

## Performance Breakdown

| Matrix Size      | CPU Time | Best GPU Time | Speedup      |
|------------------|----------|--------------|--------------|
| 512 x 256        | 0.478 ms | 0.139 ms     | 3.4x         |
| 1024 x 512       | 3.379 ms | 0.054 ms     | 62.6x        |
| 2048 x 1024      | 15.802 ms| 0.086 ms     | 183.7x       |
| 4096 x 2048      | 62.860 ms| 0.724 ms     | 86.8x        |

The biggest gains came from optimizing memory access and using shared memory. For small matrices, the speedup was modest, but for larger ones, the GPU really shined.

## Technical Highlights

- **Tiling and Thread Blocks:** Used 32x32 tiles and 32x8 thread blocks for best performance.
- **Boundary Checks:** Made sure kernels handled matrices that aren't multiples of the tile size.
- **Synchronization:** Used `__syncthreads()` to coordinate threads within a block.
- **High-Resolution Timing:** Measured performance with microsecond accuracy.


## How to Build and Run

**On macOS/Linux:**
```bash
make clean && make run
# or
nvcc -O3 -arch=sm_50 -std=c++11 -o matrix_transpose main_final.cu
./matrix_transpose
```

**On Windows:**
```bash
.\build_and_run_final.bat
# or
nvcc -O3 -arch=sm_50 -std=c++11 -o matrix_transpose_final.exe main_final.cu
matrix_transpose_final.exe
```
