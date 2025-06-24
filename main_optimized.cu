#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <assert.h>

// Function prototypes
void transposeMatrixCPU(float *input, float *output, int rows, int cols);
__global__ void transposeMatrixGPU_Basic(float *input, float *output, int rows, int cols);
__global__ void transposeMatrixGPU_Coalesced(float *input, float *output, int rows, int cols);
__global__ void transposeMatrixGPU_SharedMem(float *input, float *output, int rows, int cols);
void initializeMatrix(float *matrix, int rows, int cols);
bool verifyTranspose(float *original, float *transposed, int rows, int cols);
void printMatrix(float *matrix, int rows, int cols, const char* name);
double measureTime(clock_t start, clock_t end);
void runPerformanceTest(const char* testName, int rows, int cols);

// Matrix dimensions - starting with 2048x1024, will test multiple sizes
#define DEFAULT_ROWS 2048
#define DEFAULT_COLS 1024
#define TILE_DIM 32
#define BLOCK_ROWS 8

int main()
{
    printf("CUDA Matrix Transposition - Optimized Implementation\n");
    printf("==================================================\n\n");

    // Test different matrix sizes
    printf("Testing multiple matrix sizes:\n");
    printf("==============================\n");
    
    runPerformanceTest("Small Matrix", 512, 256);
    runPerformanceTest("Medium Matrix", 1024, 512);
    runPerformanceTest("Large Matrix", 2048, 1024);
    runPerformanceTest("XLarge Matrix", 4096, 2048);

    return 0;
}

void runPerformanceTest(const char* testName, int rows, int cols)
{
    printf("\n%s (%d x %d):\n", testName, rows, cols);
    printf("----------------------------------------\n");
    
    int matrix_size = rows * cols;
    size_t size = matrix_size * sizeof(float);

    // Use unified memory for easier management and potentially better performance
    float *unified_input, *unified_output_basic, *unified_output_coalesced, *unified_output_shared;
    float *h_output_cpu;

    // Allocate unified memory
    cudaMallocManaged(&unified_input, size);
    cudaMallocManaged(&unified_output_basic, size);
    cudaMallocManaged(&unified_output_coalesced, size);
    cudaMallocManaged(&unified_output_shared, size);
    
    // Allocate host memory for CPU implementation
    h_output_cpu = (float*)malloc(size);

    if (!h_output_cpu) {
        fprintf(stderr, "Memory allocation failed!\n");
        return;
    }

    // Initialize input matrix
    initializeMatrix(unified_input, rows, cols);

    // CPU Implementation
    clock_t cpu_start = clock();
    transposeMatrixCPU(unified_input, h_output_cpu, rows, cols);
    clock_t cpu_end = clock();
    double cpu_time = measureTime(cpu_start, cpu_end);

    // GPU Implementation 1: Basic (from Part 1)
    dim3 blockSize_basic(16, 16);
    dim3 gridSize_basic((cols + blockSize_basic.x - 1) / blockSize_basic.x, 
                       (rows + blockSize_basic.y - 1) / blockSize_basic.y);

    clock_t gpu_basic_start = clock();
    transposeMatrixGPU_Basic<<<gridSize_basic, blockSize_basic>>>(unified_input, unified_output_basic, rows, cols);
    cudaDeviceSynchronize();
    clock_t gpu_basic_end = clock();
    double gpu_basic_time = measureTime(gpu_basic_start, gpu_basic_end);

    // GPU Implementation 2: Coalesced Memory Access
    dim3 blockSize_coalesced(TILE_DIM, BLOCK_ROWS);
    dim3 gridSize_coalesced((cols + TILE_DIM - 1) / TILE_DIM, 
                           (rows + TILE_DIM - 1) / TILE_DIM);

    clock_t gpu_coalesced_start = clock();
    transposeMatrixGPU_Coalesced<<<gridSize_coalesced, blockSize_coalesced>>>(unified_input, unified_output_coalesced, rows, cols);
    cudaDeviceSynchronize();
    clock_t gpu_coalesced_end = clock();
    double gpu_coalesced_time = measureTime(gpu_coalesced_start, gpu_coalesced_end);

    // GPU Implementation 3: Shared Memory Optimization
    clock_t gpu_shared_start = clock();
    transposeMatrixGPU_SharedMem<<<gridSize_coalesced, blockSize_coalesced>>>(unified_input, unified_output_shared, rows, cols);
    cudaDeviceSynchronize();
    clock_t gpu_shared_end = clock();
    double gpu_shared_time = measureTime(gpu_shared_start, gpu_shared_end);

    // Verify correctness
    bool cpu_correct = verifyTranspose(unified_input, h_output_cpu, rows, cols);
    bool gpu_basic_correct = verifyTranspose(unified_input, unified_output_basic, rows, cols);
    bool gpu_coalesced_correct = verifyTranspose(unified_input, unified_output_coalesced, rows, cols);
    bool gpu_shared_correct = verifyTranspose(unified_input, unified_output_shared, rows, cols);

    printf("Correctness: CPU=%s, GPU_Basic=%s, GPU_Coalesced=%s, GPU_Shared=%s\n",
           cpu_correct ? "✓" : "✗",
           gpu_basic_correct ? "✓" : "✗",
           gpu_coalesced_correct ? "✓" : "✗",
           gpu_shared_correct ? "✓" : "✗");

    // Performance Results
    printf("\nPerformance Results:\n");
    printf("CPU Time:           %.2f ms (%.2f GB/s)\n", cpu_time, (2.0 * size) / (cpu_time * 1e-3) / 1e9);
    printf("GPU Basic:          %.2f ms (%.2f GB/s) - Speedup: %.2fx\n", gpu_basic_time, (2.0 * size) / (gpu_basic_time * 1e-3) / 1e9, cpu_time / gpu_basic_time);
    printf("GPU Coalesced:      %.2f ms (%.2f GB/s) - Speedup: %.2fx\n", gpu_coalesced_time, (2.0 * size) / (gpu_coalesced_time * 1e-3) / 1e9, cpu_time / gpu_coalesced_time);
    printf("GPU Shared Mem:     %.2f ms (%.2f GB/s) - Speedup: %.2fx\n", gpu_shared_time, (2.0 * size) / (gpu_shared_time * 1e-3) / 1e9, cpu_time / gpu_shared_time);

    // Cleanup
    free(h_output_cpu);
    cudaFree(unified_input);
    cudaFree(unified_output_basic);
    cudaFree(unified_output_coalesced);
    cudaFree(unified_output_shared);
}

// CPU matrix transpose implementation
void transposeMatrixCPU(float *input, float *output, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

// Basic GPU matrix transpose kernel (from Part 1)
__global__ void transposeMatrixGPU_Basic(float *input, float *output, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int input_idx = row * cols + col;
        int output_idx = col * rows + row;
        output[output_idx] = input[input_idx];
    }
}

// Coalesced memory access version
__global__ void transposeMatrixGPU_Coalesced(float *input, float *output, int rows, int cols)
{
    int blockIdx_x = blockIdx.x;
    int blockIdx_y = blockIdx.y;

    // Handle non-square tiles by swapping block indices for half the blocks
    if (blockIdx_x > blockIdx_y) {
        int temp = blockIdx_x;
        blockIdx_x = blockIdx_y;
        blockIdx_y = temp;
    }

    int xIndex = blockIdx_x * TILE_DIM + threadIdx.x;
    int yIndex = blockIdx_y * TILE_DIM + threadIdx.y;
    
    int width = cols;
    int height = rows;

    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (xIndex < width && (yIndex + j) < height) {
            int input_idx = (yIndex + j) * width + xIndex;
            int output_idx = xIndex * height + (yIndex + j);
            output[output_idx] = input[input_idx];
        }
    }
}

// Shared memory optimized version
__global__ void transposeMatrixGPU_SharedMem(float *input, float *output, int rows, int cols)
{
    __shared__ float tile[TILE_DIM][TILE_DIM + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_DIM + threadIdx.x;
    int y = blockIdx.y * TILE_DIM + threadIdx.y;
    int width = cols;
    int height = rows;

    // Load data into shared memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < width && (y + j) < height) {
            tile[threadIdx.y + j][threadIdx.x] = input[(y + j) * width + x];
        }
    }

    __syncthreads();

    // Calculate transposed coordinates
    x = blockIdx.y * TILE_DIM + threadIdx.x;
    y = blockIdx.x * TILE_DIM + threadIdx.y;

    // Write transposed data from shared memory to global memory
    for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
        if (x < height && (y + j) < width) {
            output[(y + j) * height + x] = tile[threadIdx.x][threadIdx.y + j];
        }
    }
}

// Initialize matrix with random values
void initializeMatrix(float *matrix, int rows, int cols)
{
    srand(time(NULL));
    for (int i = 0; i < rows * cols; i++) {
        matrix[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

// Verify transpose correctness
bool verifyTranspose(float *original, float *transposed, int rows, int cols)
{
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            float orig = original[i * cols + j];
            float trans = transposed[j * rows + i];
            if (abs(orig - trans) > 1e-5) {
                return false;
            }
        }
    }
    return true;
}

// Print matrix (for debugging small matrices)
void printMatrix(float *matrix, int rows, int cols, const char* name)
{
    if (rows > 10 || cols > 10) {
        printf("%s: Matrix too large to print (size: %dx%d)\n", name, rows, cols);
        return;
    }
    
    printf("%s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%.2f ", matrix[i * cols + j]);
        }
        printf("\n");
    }
    printf("\n");
}

// Measure time in milliseconds
double measureTime(clock_t start, clock_t end)
{
    return ((double)(end - start) / CLOCKS_PER_SEC) * 1000.0;
}
