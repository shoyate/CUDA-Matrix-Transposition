#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <time.h>
#include <assert.h>

// Function prototypes
void transposeMatrixCPU(float *input, float *output, int rows, int cols);
__global__ void transposeMatrixGPU(float *input, float *output, int rows, int cols);
void initializeMatrix(float *matrix, int rows, int cols);
bool verifyTranspose(float *original, float *transposed, int rows, int cols);
void printMatrix(float *matrix, int rows, int cols, const char* name);
double measureTime(clock_t start, clock_t end);

// Matrix dimensions - starting with 2048x1024 as specified
#define ROWS 2048
#define COLS 1024
#define MATRIX_SIZE (ROWS * COLS)

int main()
{
    printf("CUDA Matrix Transposition Performance Analysis\n");
    printf("============================================\n");
    printf("Matrix size: %d x %d\n", ROWS, COLS);
    printf("Total elements: %d\n\n", MATRIX_SIZE);

    // Allocate host memory
    size_t size = MATRIX_SIZE * sizeof(float);
    float *h_input = (float*)malloc(size);
    float *h_output_cpu = (float*)malloc(size);
    float *h_output_gpu = (float*)malloc(size);

    if (!h_input || !h_output_cpu || !h_output_gpu) {
        fprintf(stderr, "Host memory allocation failed!\n");
        return -1;
    }

    // Initialize input matrix with random values
    printf("Initializing matrix with random values...\n");
    initializeMatrix(h_input, ROWS, COLS);

    // CPU Implementation
    printf("Running CPU transpose...\n");
    clock_t cpu_start = clock();
    transposeMatrixCPU(h_input, h_output_cpu, ROWS, COLS);
    clock_t cpu_end = clock();
    double cpu_time = measureTime(cpu_start, cpu_end);

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc((void**)&d_input, size);
    cudaMalloc((void**)&d_output, size);

    // Copy input data to device
    cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice);

    // GPU Implementation
    printf("Running GPU transpose...\n");
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((COLS + blockSize.x - 1) / blockSize.x, 
                  (ROWS + blockSize.y - 1) / blockSize.y);

    clock_t gpu_start = clock();
    transposeMatrixGPU<<<gridSize, blockSize>>>(d_input, d_output, ROWS, COLS);
    cudaDeviceSynchronize();
    clock_t gpu_end = clock();
    double gpu_time = measureTime(gpu_start, gpu_end);

    // Copy result back to host
    cudaMemcpy(h_output_gpu, d_output, size, cudaMemcpyDeviceToHost);

    // Verify correctness
    printf("Verifying results...\n");
    bool correct = verifyTranspose(h_input, h_output_cpu, ROWS, COLS) && 
                   verifyTranspose(h_input, h_output_gpu, ROWS, COLS);
    
    if (correct) {
        printf("✓ Transpose results are CORRECT!\n\n");
    } else {
        printf("✗ Transpose results are INCORRECT!\n\n");
    }

    // Performance Results
    printf("Performance Results:\n");
    printf("==================\n");
    printf("CPU Time: %.2f ms\n", cpu_time);
    printf("GPU Time: %.2f ms\n", gpu_time);
    printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    printf("CPU Throughput: %.2f GB/s\n", (2.0 * size) / (cpu_time * 1e-3) / 1e9);
    printf("GPU Throughput: %.2f GB/s\n", (2.0 * size) / (gpu_time * 1e-3) / 1e9);

    // Cleanup
    free(h_input);
    free(h_output_cpu);
    free(h_output_gpu);
    cudaFree(d_input);
    cudaFree(d_output);

    return 0;
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

// Basic GPU matrix transpose kernel
__global__ void transposeMatrixGPU(float *input, float *output, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int input_idx = row * cols + col;
        int output_idx = col * rows + row;
        output[output_idx] = input[input_idx];
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
