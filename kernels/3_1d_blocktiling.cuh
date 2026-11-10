#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace sgemm_k3 {
constexpr int BM = 64;  // C tile 行数
constexpr int BN = 64;  // C tile 列数
constexpr int BK = 8;   // K 方向分块宽度
constexpr int TM = 8;   // 每线程沿行方向算的结果数

// C(M×N) = A(M×K) × B(K×N)
__global__ void sgemm_1d_blocktiling(int M, int N, int K, const float *A, const float *B, float *C) {
  const int col = blockIdx.x * BN + threadIdx.x;           // 0..N-1
  const int rowBase = blockIdx.y * BM + threadIdx.y * TM;  // 本线程的第 0 个结果的行

  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float results[TM] = {0.0};

  for (int k0 = 0; k0 < K; k0 += BK) {
    const int tid = threadIdx.y * BN + threadIdx.x;  // 0..511
    const int mA = tid / BK, kA = tid % BK;
    const int kB = tid / BN, nB = tid % BN;

    const int rowA = blockIdx.y * BM + mA;
    const int aK = k0 + kA;

    const int bK = k0 + kB;
    const int colB = blockIdx.x * BN + nB; 

    As[mA * BK + kA] = (rowA < M && aK < K) ? A[rowA * K + aK] : 0.0f;
    Bs[kB * BN + nB] = (bK < K && colB < N) ? B[bK * N + colB] : 0.0f;

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      float tmpB = Bs[k * BN + threadIdx.x];
      for (int r = 0; r < TM; r++) {
        results[r] += As[(threadIdx.y * TM + r) * BK + k] * tmpB;
      }
    }

    __syncthreads();
  }

  for (int r = 0; r < TM; r++) {
    if (rowBase + r < M && col < N) C[(rowBase + r) * N + col] = results[r];
  }
}

}  // namespace sgemm_k3

void launch_sgemm_1d_blocktiling(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
  dim3 blockDim(sgemm_k3::BN, sgemm_k3::BM / sgemm_k3::TM, 1);
  dim3 gridDim((N + sgemm_k3::BN - 1) / sgemm_k3::BN,
               (M + sgemm_k3::BM - 1) / sgemm_k3::BM, 1);
  sgemm_k3::sgemm_1d_blocktiling<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
