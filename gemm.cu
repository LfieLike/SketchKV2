#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <iostream>

void initMatrix(float* matrix, int M, int N) {
    for (int i = 0; i < M * N; ++i) {
        matrix[i] = static_cast<float>(rand() % 2); // 01矩阵
    }
}

void initVectorBatch(float* vectors, int N, int batchSize) {
    for (int i = 0; i < N * batchSize; ++i) {
        vectors[i] = static_cast<float>(rand()) / RAND_MAX;
    }
}

int main() {
    int M = 12000;  // 矩阵的行数
    int N = 128;   // 矩阵的列数以及向量的长度

    // Initialize matrix only once as it's the same across different batch sizes
    size_t matrixSize = M * N * sizeof(float);
    float* h_matrix = (float*)malloc(matrixSize);
    initMatrix(h_matrix, M, N);

    // Allocate memory on the device for the matrix
    float* d_matrix;
    cudaMalloc(&d_matrix, matrixSize);
    cudaMemcpy(d_matrix, h_matrix, matrixSize, cudaMemcpyHostToDevice);

    // 创建cuBLAS句柄
    cublasHandle_t handle;
    cublasCreate(&handle);

    // Define the range of batch sizes to test
    int minBatchSize = 1;
    int maxBatchSize = 680;
    int step = 1;

    for (int batchSize = minBatchSize; batchSize <= maxBatchSize; batchSize += 2) {
        std::cout << "Testing batch size: " << batchSize << std::endl;

        // Adjust memory allocation based on batch size
        size_t vectorBatchSize = N * batchSize * sizeof(float);
        size_t resultBatchSize = M * batchSize * sizeof(float);

        // 在主机上分配内存
        float* h_vectors = (float*)malloc(vectorBatchSize);
        float* h_results = (float*)malloc(resultBatchSize);

        // Initialize the batch of vectors
        initVectorBatch(h_vectors, N, batchSize);

        // 在设备上分配内存
        float* d_vectors;
        float* d_results;
        cudaMalloc(&d_vectors, vectorBatchSize);
        cudaMalloc(&d_results, resultBatchSize);

        // 将数据复制到设备
        cudaMemcpy(d_vectors, h_vectors, vectorBatchSize, cudaMemcpyHostToDevice);

        // 记录开始时间
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaEventRecord(start);

        // 使用cuBLAS执行矩阵乘法
        float alpha = 1.0f;
        float beta = 0.0f;

        cublasSgemmStridedBatched(
            handle,
            CUBLAS_OP_N, CUBLAS_OP_N,
            M, 1, N,
            &alpha,
            d_matrix, M, 0,       // A matrix (no stride since matrix is reused)
            d_vectors, N, N,      // B vectors with stride N between each vector in the batch
            &beta,
            d_results, M, M,      // C results with stride M between each result in the batch
            batchSize             // Number of matrices in the batch
        );

        // 记录结束时间
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        std::cout << "cuBLAS SgemmStridedBatched time for batch size " << batchSize << ": " << milliseconds << " ms" << std::endl;

        // 将结果复制回主机
        cudaMemcpy(h_results, d_results, resultBatchSize, cudaMemcpyDeviceToHost);

        // 清理
        cudaFree(d_vectors);
        cudaFree(d_results);
        free(h_vectors);
        free(h_results);

        // Increase step size for larger batch sizes if desired
        if (batchSize < 16) {
            step = 2;
        } else if (batchSize < 128) {
            step = 2;
        } else if (batchSize < 1024) {
            step = 2;
        }
    }

    // Cleanup
    cublasDestroy(handle);
    cudaFree(d_matrix);
    free(h_matrix);

    return 0;
}
