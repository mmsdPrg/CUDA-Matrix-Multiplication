// Step 2: Unified Memory (Managed Memory) Implementation
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_kernel(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

extern "C" {

void matmul_forward(
    float* h_A,
    float* h_B,
    float* h_C,
    int M, int K, int N
) {
    float *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(float);
    size_t size_B = K * N * sizeof(float);
    size_t size_C = M * N * sizeof(float);

    // Using Unified Memory
    cudaMallocManaged(&d_A, size_A);
    cudaMallocManaged(&d_B, size_B);
    cudaMallocManaged(&d_C, size_C);

    // Copy data using memcpy instead of cudaMemcpy
    memcpy(d_A, h_A, size_A);
    memcpy(d_B, h_B, size_B);

    // Prefetch data to GPU for better performance
    int device = -1;
    cudaGetDevice(&device);
    cudaMemPrefetchAsync(d_A, size_A, device, NULL);
    cudaMemPrefetchAsync(d_B, size_B, device, NULL);
    cudaMemPrefetchAsync(d_C, size_C, device, NULL);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_kernel<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();

    // Copy result back using memcpy
    memcpy(h_C, d_C, size_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}
