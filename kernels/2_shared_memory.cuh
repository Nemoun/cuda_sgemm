#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// tile 尺寸（常量放进 namespace，避免与其他 kernel 头文件的常量重名）
namespace sgemm_k2 {
constexpr int BM = 16;  // C tile 行数
constexpr int BN = 16;  // C tile 列数
constexpr int BK = 16;  // K 方向分块宽度

// C(M×N) = A(M×K) × B(K×N)
__global__ void sgemm_shared_memory(int M, int N, int K, const float *A, const float *B, float *C) {
  const int col = blockIdx.x * BN + threadIdx.x;  // 0..N-1
  const int row = blockIdx.y * BM + threadIdx.y;  // 0..M-1

  // 声明 shared memory
  __shared__ float As[BM][BK];
  __shared__ float Bs[BK][BN];

  float tmp = 0.0f;

  // K 方向按 BK 分块
  for (int k0 = 0; k0 < K; k0 += BK) {
    // 可能越界，越界处填 0.0f
    if (k0 + threadIdx.x >= K || row >= M) {
      As[threadIdx.y][threadIdx.x] = 0.0f;
    } else {
      As[threadIdx.y][threadIdx.x] = A[row * K + k0 + threadIdx.x];
    }

    if (col >= N || k0 + threadIdx.y >= K) {
      Bs[threadIdx.y][threadIdx.x] = 0.0f;
    } else {
      Bs[threadIdx.y][threadIdx.x] = B[(k0 + threadIdx.y) * N + col];
    }

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      tmp += As[threadIdx.y][k] * Bs[k][threadIdx.x];
    }

    __syncthreads();
  }

  // row/col 可能越界
  if (row < M && col < N) {
    C[row * N + col] = tmp;
  }
}

}  // namespace sgemm_k2

void launch_sgemm_shared_memory(const float *A, const float *B, float *C,
                                int M, int N, int K) {
  dim3 blockDim(sgemm_k2::BN, sgemm_k2::BM, 1);
  dim3 gridDim((N + sgemm_k2::BN - 1) / sgemm_k2::BN,
               (M + sgemm_k2::BM - 1) / sgemm_k2::BM, 1);
  sgemm_k2::sgemm_shared_memory<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
