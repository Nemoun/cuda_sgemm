#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// C(M×N) = A(M×K) × B(K×N)
__global__ void sgemm_naive(int M, int N, int K, const float *A, const float *B, float *C) {
       const int col = blockIdx.x * blockDim.x + threadIdx.x;
       const int row = blockIdx.y * blockDim.y + threadIdx.y;

       if(col < M && row < N){
              float tmp = 0;
              for(int i = 0; i < K; i++){
                     tmp += A[col * K + i] * B[i * N + row];
              }
              C[N * col + row] = tmp;
       }
}

void launch_sgemm_naive(const float *A, const float *B, float *C,
                        int M, int N, int K) {
  dim3 blockDim(16, 16, 1);
  dim3 gridDim((M + 15) / 16, (N + 15) / 16, 1);
  sgemm_naive<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
