#include <cuda_runtime.h>
#include <iostream>
#include <cublas_v2.h>

// CUDA kernel for dot product without multiplication
__global__ void dotProductWithoutMultiplication(const float* A, const float* B, float* C, int N) {
    __shared__ float partialSum[1024];
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int lane = threadIdx.x;
    float sum = 0.0f;
    if (tid < N) {
        sum = (B[tid] == 1.0f) ? A[tid] : -A[tid];
    }
    partialSum[lane] = sum;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (lane < stride) {
            partialSum[lane] += partialSum[lane + stride];
        }
        __syncthreads();
    }
    if (lane == 0) {
        atomicAdd(C, partialSum[0]);
    }
}

// Host function to call the custom kernel
float dotProductWithoutMultiplicationHost(const float* A, const float* B, int N) {
    float *d_A, *d_B, *d_C;
    float h_C = 0.0f;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMalloc(&d_C, sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_C, &h_C, sizeof(float), cudaMemcpyHostToDevice);

    int blockSize = 1024;
    int gridSize = (N + blockSize - 1) / blockSize;

    // Start and stop CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);
    dotProductWithoutMultiplication<<<gridSize, blockSize>>>(d_A, d_B, d_C, N);
    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_C, d_C, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);

    // Output the timing information
    std::cout << "Custom Kernel Time: " << milliseconds << " ms" << std::endl;

    return h_C;
}

// Function to perform matrix multiplication using cublasSgemm
void cublasDotProduct(const float* A, const float* B, int N, float* result) {
    cublasHandle_t handle;
    cublasCreate(&handle);
    float *d_A, *d_B;

    cudaMalloc(&d_A, N * sizeof(float));
    cudaMalloc(&d_B, N * sizeof(float));
    cudaMemcpy(d_A, A, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, N * sizeof(float), cudaMemcpyHostToDevice);

    float alpha = 1.0f, beta = 0.0f;

    // Start and stop CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Record start event
    cudaEventRecord(start);
    cublasSdot(handle, N, d_A, 1, d_B, 1, result);
    cudaEventRecord(stop);

    // Wait for the event to complete
    cudaEventSynchronize(stop);

    // Calculate the elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Output the timing information
    std::cout << "cuBLAS Time: " << milliseconds << " ms" << std::endl;

    cudaFree(d_A);
    cudaFree(d_B);
    cublasDestroy(handle);
}

int main() {
    int N = 10240;
    float *h_A = new float[N];
    float *h_B = new float[N];

    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = (rand() % 2) * 2.0f - 1.0f;
    }

    float custom_result = dotProductWithoutMultiplicationHost(h_A, h_B, N);

    float cublas_result;
    cublasDotProduct(h_A, h_B, N, &cublas_result);

    std::cout << "Custom Kernel Result: " << custom_result << std::endl;
    std::cout << "cuBLAS Result: " << cublas_result << std::endl;

    delete[] h_A;
    delete[] h_B;

    return 0;
}
