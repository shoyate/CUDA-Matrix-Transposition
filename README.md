# CUDA Matrix Transposition

## Build and Run

### Prerequisites

- NVIDIA CUDA Toolkit
- CUDA-capable GPU
- C++ compiler

### Compilation

```bash
# Using Makefile
make

# Or directly with nvcc
nvcc -O3 -arch=sm_50 -o matrix_transpose main.cu
```

### Execution

```bash
# Using Makefile
make run

# Or directly
./matrix_transpose
```
