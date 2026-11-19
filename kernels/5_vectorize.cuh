#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// tile 尺寸
namespace sgemm_k5 {
constexpr int BM = 128;  // C tile 行数
constexpr int BN = 128;  // C tile 列数
constexpr int BK = 8;    // K 方向分块宽度
constexpr int TM = 8;    // 每线程沿行方向算的结果数
constexpr int TN = 8;    // 每线程沿列方向算的结果数

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_vectorize(int M, int N, int K, const float *A, const float *B, float *C) {
  const int colBase = blockIdx.x * BN + threadIdx.x * TN;  // 0..N-1
  const int rowBase = blockIdx.y * BM + threadIdx.y * TM;  // 0..M-1

  // shared memory
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  const int tid = threadIdx.y * (BN / TN) + threadIdx.x;   // 0..255
  const int innerRowA = tid / (BK / 4);                    // 0..127，A tile 的行（转置后的列）
  const int innerColA = tid % (BK / 4);                    // 0..1，k 方向的 float4 段
  const int innerRowB = tid / (BN / 4);                    // 0..7，B tile 的 k 行
  const int innerColB = tid % (BN / 4);                    // 0..31，B tile 的 float4 段

  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  float results[TM * TN] = {0.0};

  for (int k0 = 0; k0 < K; k0 += BK) {
    const int aRow = blockIdx.y * BM + innerRowA;      // 全局行
    const int aK = k0 + innerColA * 4;                 // 本 float4 的 K 起点
    if (vecA && aRow < M && aK + 4 <= K) {
      float4 tmp = reinterpret_cast<const float4 *>(&A[aRow * K + aK])[0];
      As[(innerColA * 4 + 0) * BM + innerRowA] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA] = tmp.w;
    } else {  // 尺寸不满足 4 对齐或 tile 越界：标量 + 填 0
      for (int j = 0; j < 4; j++)
        As[(innerColA * 4 + j) * BM + innerRowA] =
            (aRow < M && aK + j < K) ? A[aRow * K + aK + j] : 0.0f;
    }

    const int bK = k0 + innerRowB;                     // 全局 K 行
    const int bCol = blockIdx.x * BN + innerColB * 4;  // 本 float4 的列起点
    if (vecB && bK < K && bCol + 4 <= N) {
      reinterpret_cast<float4 *>(&Bs[innerRowB * BN + innerColB * 4])[0] =
          reinterpret_cast<const float4 *>(&B[bK * N + bCol])[0];
    } else {  // 尺寸不满足 4 对齐或 tile 越界：标量 + 填 0
      for (int j = 0; j < 4; j++)
        Bs[innerRowB * BN + innerColB * 4 + j] =
            (bK < K && bCol + j < N) ? B[bK * N + bCol + j] : 0.0f;
    }

    __syncthreads();

    for (int k = 0; k < BK; k++) {
      float regM[TM], regN[TN];
      for (int r = 0; r < TM; r++) regM[r] = As[k * BM + threadIdx.y * TM + r];
      for (int c = 0; c < TN; c++) regN[c] = Bs[k * BN + threadIdx.x * TN + c];
      for (int r = 0; r < TM; r++)
        for (int c = 0; c < TN; c++)
          results[r * TN + c] += regM[r] * regN[c];
    }

    __syncthreads();
  }

  for (int r = 0; r < TM; r++) {
    if (rowBase + r >= M) continue;
    if (vecB && colBase + TN <= N) {
      for (int c4 = 0; c4 < TN / 4; c4++) {
        float4 tmp;
        tmp.x = results[r * TN + c4 * 4 + 0];
        tmp.y = results[r * TN + c4 * 4 + 1];
        tmp.z = results[r * TN + c4 * 4 + 2];
        tmp.w = results[r * TN + c4 * 4 + 3];
        reinterpret_cast<float4 *>(&C[(rowBase + r) * N + colBase + c4 * 4])[0] = tmp;
      }
    } else {
      for (int c = 0; c < TN; c++)
        if (colBase + c < N) C[(rowBase + r) * N + colBase + c] = results[r * TN + c];
    }
  }
}

}  // namespace sgemm_k5

void launch_sgemm_vectorize(const float *A, const float *B, float *C,
                            int M, int N, int K) {
  dim3 blockDim(sgemm_k5::BN / sgemm_k5::TN, sgemm_k5::BM / sgemm_k5::TM, 1);
  dim3 gridDim((N + sgemm_k5::BN - 1) / sgemm_k5::BN,
               (M + sgemm_k5::BM - 1) / sgemm_k5::BM, 1);
  sgemm_k5::sgemm_vectorize<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
