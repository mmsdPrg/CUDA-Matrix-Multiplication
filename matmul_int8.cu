// Step 4: INT8 Quantized Implementation (Bonus)
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

__global__ void matmul_kernel_int8(
    const int8_t* A,
    const int8_t* B,
    int8_t* C,
    int M, int K, int N,
    float scale_A, float scale_B, float scale_C
) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        int32_t sum = 0;  // Accumulate in INT32 to prevent overflow
        for (int k = 0; k < K; k++) {
            sum += (int32_t)A[row * K + k] * (int32_t)B[k * N + col];
        }
        // Dequantize and requantize
        float result = (float)sum * scale_A * scale_B / scale_C;
        C[row * N + col] = (int8_t)fmaxf(-127.0f, fminf(127.0f, result));
    }
}

extern "C" {

void matmul_forward(
    float* h_A,
    float* h_B,
    float* h_C,
    int M, int K, int N
) {
    int8_t *d_A, *d_B, *d_C;

    size_t size_A = M * K * sizeof(int8_t);
    size_t size_B = K * N * sizeof(int8_t);
    size_t size_C = M * N * sizeof(int8_t);

    // Simple quantization: scale = max_value / 127
    float max_A = 0.0f, max_B = 0.0f;
    for (int i = 0; i < M * K; i++) {
        max_A = fmaxf(max_A, fabsf(h_A[i]));
    }
    for (int i = 0; i < K * N; i++) {
        max_B = fmaxf(max_B, fabsf(h_B[i]));
    }

    float scale_A = max_A > 0 ? max_A / 127.0f : 1.0f;
    float scale_B = max_B > 0 ? max_B / 127.0f : 1.0f;
    float scale_C = scale_A * scale_B * K;

    // Quantize to INT8
    int8_t *h_A_int8 = (int8_t*)malloc(size_A);
    int8_t *h_B_int8 = (int8_t*)malloc(size_B);
    int8_t *h_C_int8 = (int8_t*)malloc(size_C);

    for (int i = 0; i < M * K; i++) {
        h_A_int8[i] = (int8_t)(h_A[i] / scale_A);
    }
    for (int i = 0; i < K * N; i++) {
        h_B_int8[i] = (int8_t)(h_B[i] / scale_B);
    }

    cudaMalloc(&d_A, size_A);
    cudaMalloc(&d_B, size_B);
    cudaMalloc(&d_C, size_C);

    cudaMemcpy(d_A, h_A_int8, size_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B_int8, size_B, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((N + 15) / 16, (M + 15) / 16);

    matmul_kernel_int8<<<gridDim, blockDim>>>(d_A, d_B, d_C, M, K, N, scale_A, scale_B, scale_C);

    cudaDeviceSynchronize();

    cudaMemcpy(h_C_int8, d_C, size_C, cudaMemcpyDeviceToHost);

    // Dequantize back to FP32
    for (int i = 0; i < M * N; i++) {
        h_C[i] = (float)h_C_int8[i] * scale_C;
    }

    free(h_A_int8);
    free(h_B_int8);
    free(h_C_int8);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

}
