#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// tile 尺寸
namespace sgemm_k4 {
constexpr int BM = 128;  // C tile 行数
constexpr int BN = 128;  // C tile 列数
constexpr int BK = 8;    // K 方向分块宽度
constexpr int TM = 8;    // 每线程沿行方向算的结果数
constexpr int TN = 8;    // 每线程沿列方向算的结果数

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_2d_blocktiling(int M, int N, int K, const float *A, const float *B, float *C) {
  const int colBase = blockIdx.x * BN + threadIdx.x * TN;  // 0..N-1
  const int rowBase = blockIdx.y * BM + threadIdx.y * TM;  // 0..M-1

  // shared memory
  __shared__ float As[BM * BK];
  __shared__ float Bs[BK * BN];

  float results[TM * TN] = {0.0};

  const int numThreads = (BN / TN) * (BM / TM);            // 256
  const int tid = threadIdx.y * (BN / TN) + threadIdx.x;   // 0..255
  const int innerRow = tid / BK, innerCol = tid % BK;
  const int strideA = numThreads / BK;                     // 沿 m 的步长
  const int innerK = tid / BN, innerN = tid % BN;
  const int strideB = numThreads / BN;                     // 沿 k 的步长

  for (int k0 = 0; k0 < K; k0 += BK) {
    for (int m = innerRow; m < BM; m += strideA) {
      int aRow = blockIdx.y * BM + m;
      int aK = k0 + innerCol;
      As[m * BK + innerCol] = (aRow < M && aK < K) ? A[aRow * K + aK] : 0.0f;
    }
    for (int k = innerK; k < BK; k += strideB) {
      int bK = k0 + k;
      int bCol = blockIdx.x * BN + innerN;
      Bs[k * BN + innerN] = (bK < K && bCol < N) ? B[bK * N + bCol] : 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      float regM[TM], regN[TN];
      for (int r = 0; r < TM; r++) regM[r] = As[(threadIdx.y * TM + r) * BK + k];
      for (int c = 0; c < TN; c++) regN[c] = Bs[k * BN + threadIdx.x * TN + c];
      for (int r = 0; r < TM; r++)
        for (int c = 0; c < TN; c++)
          results[r * TN + c] += regM[r] * regN[c];
    }

    __syncthreads();
  }

  for (int r = 0; r < TM; r++)
    for (int c = 0; c < TN; c++)
      if (rowBase + r < M && colBase + c < N)
        C[(rowBase + r) * N + (colBase + c)] = results[r * TN + c];
}

}  // namespace sgemm_k4

void launch_sgemm_2d_blocktiling(const float *A, const float *B, float *C,
                                 int M, int N, int K) {
  dim3 blockDim(sgemm_k4::BN / sgemm_k4::TN, sgemm_k4::BM / sgemm_k4::TM, 1);
  dim3 gridDim((N + sgemm_k4::BN - 1) / sgemm_k4::BN,
               (M + sgemm_k4::BM - 1) / sgemm_k4::BM, 1);
  sgemm_k4::sgemm_2d_blocktiling<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
