// Step 5: Loop Unrolling Implementation (Factor of 4)
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void matmul_kernel_unroll(
    const float* A,
    const float* B,
    float* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        int k;
        
        // Loop unrolling: process 4 elements at a time
        for (k = 0; k < K - 3; k += 4) {
            sum += A[row * K + k] * B[k * N + col];
            sum += A[row * K + k + 1] * B[(k + 1) * N + col];
            sum += A[row * K + k + 2] * B[(k + 2) * N + col];
            sum += A[row * K + k + 3] * B[(k + 3) * N + col];
        }
        
        // Handle remainder
        for (; k < K; k++) {
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

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_kernel_unroll<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C, d_C, size_C, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}
