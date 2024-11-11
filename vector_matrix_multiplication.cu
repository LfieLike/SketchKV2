#include <cuda_runtime.h>
#include <iostream>

__global__ void bitMatrixVectorBatchMul(const unsigned int* matrix, const float* vectors, float* results, int rows, int cols, int batchSize) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < rows) {
        int num_ints = (cols + 31) / 32;  // Number of unsigned int per row
        for (int batch = 0; batch < batchSize; ++batch) {
            float sum = 0.0f;
            for (int i = 0; i < num_ints; ++i) {
                unsigned int bits = matrix[row * num_ints + i];
                for (int bit = 0; bit < 32; ++bit) {
                    int colIndex = i * 32 + bit;
                    if (colIndex < cols) {
                        if (bits & (1 << bit)) {
                            sum += vectors[batch * cols + colIndex];
                        } else {
                            sum -= vectors[batch * cols + colIndex];
                        }
                    }
                }
            }
            results[batch * rows + row] = sum;
        }
    }
}

void initBitMatrix(unsigned int* matrix, int rows, int cols) {
    int num_ints = (cols + 31) / 32;  // Calculate number of unsigned int per row
    for (int i = 0; i < rows * num_ints; ++i) {
        matrix[i] = rand();
    }
}

void initVectorBatch(float* vectors, int cols, int batchSize) {
    for (int i = 0; i < cols * batchSize; ++i) {
        vectors[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int rows = 12000;  // Number of rows in the matrix
    int cols = 128;   // Number of columns in the matrix

    // Initialize the bit matrix only once
    size_t bitMatrixSize = rows * ((cols + 31) / 32) * sizeof(unsigned int);
    unsigned int* h_matrix = (unsigned int*)malloc(bitMatrixSize);
    initBitMatrix(h_matrix, rows, cols);

    // Allocate memory on the device for the matrix
    unsigned int* d_matrix;
    cudaMalloc(&d_matrix, bitMatrixSize);
    cudaMemcpy(d_matrix, h_matrix, bitMatrixSize, cudaMemcpyHostToDevice);

    // Define block and grid sizes
    int blockSize = 256;
    int gridSize = (rows + blockSize - 1) / blockSize;

    // Create cuBLAS handle
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test different batch sizes
    for (int batchSize = 1; batchSize <= 680; batchSize += 2) {
        size_t vectorBatchSize = cols * batchSize * sizeof(float);
        size_t resultBatchSize = rows * batchSize * sizeof(float);

        // Allocate memory on the host for vectors and results
        float* h_vectors = (float*)malloc(vectorBatchSize);
        float* h_results = (float*)malloc(resultBatchSize);

        // Initialize vectors for the current batch size
        initVectorBatch(h_vectors, cols, batchSize);

        // Allocate memory on the device for vectors and results
        float* d_vectors;
        float* d_results;
        cudaMalloc(&d_vectors, vectorBatchSize);
        cudaMalloc(&d_results, resultBatchSize);

        // Copy vectors to the device
        cudaMemcpy(d_vectors, h_vectors, vectorBatchSize, cudaMemcpyHostToDevice);

        // Record start time
        cudaEventRecord(start);
        
        // Launch the kernel
        bitMatrixVectorBatchMul<<<gridSize, blockSize>>>(d_matrix, d_vectors, d_results, rows, cols, batchSize);
        
        // Record stop time
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "Batch size: " << batchSize << " - Time: " << milliseconds << " ms" << std::endl;

        // Copy results back to the host (optional, depending on whether you need the results)
        cudaMemcpy(h_results, d_results, resultBatchSize, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(d_vectors);
        cudaFree(d_results);

        // Free host memory
        free(h_vectors);
        free(h_results);
    }

    // Cleanup
    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}
