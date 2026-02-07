// Step 4: Half Precision (FP16) Implementation
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdio.h>

__global__ void matmul_kernel_fp16(
    const half* A,
    const half* B,
    half* C,
    int M, int K, int N
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;  // Accumulate in FP32 for numerical stability
        for (int k = 0; k < K; k++) {
            sum += __half2float(A[row * K + k]) * __half2float(B[k * N + col]);
        }
        C[row * N + col] = __float2half(sum);
    }
}

extern "C" {

void matmul_forward(
    float* h_A,
    float* h_B,
    float* h_C,
    int M, int K, int N
) {
    half *d_A, *d_B, *d_C;
    float *h_A_temp, *h_B_temp, *h_C_temp;

    size_t size_A = M * K * sizeof(half);
    size_t size_B = K * N * sizeof(half);
    size_t size_C = M * N * sizeof(half);

    // Allocate host memory for conversion
    h_A_temp = (float*)malloc(M * K * sizeof(float));
    h_B_temp = (float*)malloc(K * N * sizeof(float));
    h_C_temp = (float*)malloc(M * N * sizeof(float));

    // Convert FP32 to FP16 on host
    half *h_A_half = (half*)malloc(size_A);
    half *h_B_half = (half*)malloc(size_B);
    half *h_C_half = (half*)malloc(size_C);

    for (int i = 0; i < M * K; i++) {
        h_A_half[i] = __float2half(h_A[i]);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_half[i] = __float2half(h_B[i]);
    }

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A_half, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_half, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_kernel_fp16<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C_half, d_C, size_C, cudaMemcpyDeviceToHost);

    // Convert FP16 back to FP32
    for (int i = 0; i < M * N; i++) {
        h_C[i] = __half2float(h_C_half[i]);
    }

    free(h_A_half);
    free(h_B_half);
    free(h_C_half);
    free(h_A_temp);
    free(h_B_temp);
    free(h_C_temp);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}
