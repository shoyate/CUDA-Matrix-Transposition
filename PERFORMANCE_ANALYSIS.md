# CUDA Matrix Transposition – Performance Analysis

## Overview

I started with a basic CPU version, then moved to a simple GPU kernel, and finally layered on optimizations like coalesced memory access and shared memory. Each step brought big improvements, especially for larger matrices.

## What Was Tried

- **CPU Baseline:** Standard nested loops.
- **Basic GPU Kernel:** First CUDA version, not optimized.
- **Coalesced Memory Access:** Made sure threads accessed memory in a way that's fast for the GPU.
- **Shared Memory:** Used the GPU's fast on-chip memory to cut down on slow global memory accesses.

## Performance Results

| Matrix Size      | CPU Time | Basic GPU | Coalesced | Shared Memory |
|------------------|---------|-----------|-----------|--------------|
| 512 x 256        | 0.478ms | 24.652ms  | 0.169ms   | 0.139ms      |
| 1024 x 512       | 3.379ms | 1.657ms   | 0.405ms   | 0.054ms      |
| 2048 x 1024      | 15.802ms| 5.964ms   | 0.277ms   | 0.086ms      |
| 4096 x 2048      | 62.860ms| 21.731ms  | 0.939ms   | 0.724ms      |

- For small matrices, the GPU overhead means the CPU can actually be faster.
- As the matrices get bigger, the GPU's advantage becomes huge—up to 183x faster with shared memory.

## Why Shared Memory Helped So Much

- **Fewer Global Memory Accesses:** Data is loaded once into shared memory, then reused.
- **Better Memory Patterns:** Coalesced access and padding to avoid bank conflicts.
- **Synchronization:** Threads work together efficiently within a block.

## Lessons Learned

- **GPU Overhead is Real:** For small problems, setup and memory transfer time can outweigh the benefits.
- **Memory Patterns are Everything:** The way you access memory on the GPU makes or breaks performance.
- **Optimizations Stack Up:** Each layer of optimization multiplies the speedup.
- **Big Problems, Big Gains:** The GPU really shines with large workloads.

## What Could Be Improved Next

- Try using streams to overlap computation and memory transfer.
- Experiment with even larger matrices or multiple GPUs.
- Look into warp-level primitives for even more fine-tuned optimization.

## Conclusion

This project was a great hands-on way to see how much of a difference good memory access and shared memory can make on the GPU. The final version is dramatically faster than the CPU, especially for big matrices, and the process really drove home the importance of understanding the GPU's memory hierarchy.
