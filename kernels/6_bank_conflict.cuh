#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// tile 尺寸
namespace sgemm_k6 {
constexpr int BM = 128;  // C tile 行数
constexpr int BN = 128;  // C tile 列数
constexpr int BK = 8;    // K 方向分块宽度
constexpr int TM = 8;    // 每线程沿行方向算的结果数
constexpr int TN = 8;    // 每线程沿列方向算的结果数

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_no_bank_conflict(int M, int N, int K, const float *A, const float *B, float *C) {
  const int colBase = blockIdx.x * BN + threadIdx.x * TN;  // 0..N-1
  const int rowBase = blockIdx.y * BM + threadIdx.y * TM;  // 0..M-1

  // shared memory
  __shared__ float As[BK * BM];
  __shared__ float4 Bs4[BK * (TN / 4) * (BN / TN)];  // 8*2*16 = 256 个 float4

  // 搬运排班（每线程各搬 1 个 float4，256 × 4 = 1024 恰好铺满每个 tile）
  const int tid = threadIdx.y * (BN / TN) + threadIdx.x;   // 0..255
  const int innerRowA = tid / (BK / 4);                    // 0..127，A tile 的行（转置后的列）
  const int innerColA = tid % (BK / 4);                    // 0..1，k 方向的 float4 段
  const int innerRowB = tid / (BN / 4);                    // 0..7，B tile 的 k 行
  const int innerColB = tid % (BN / 4);                    // 0..31，B tile 的 float4 段
  // Bs4 的落位
  const int f4 = innerColB % 2;
  const int txB = innerColB / 2;

  // 行首地址 16B 对齐才允许 float4 访问全局内存
  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  float results[TM * TN] = {0.0};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile 搬运：GMEM 读一个 float4，scatter 进转置的 As 的 4 行
    const int aRow = blockIdx.y * BM + innerRowA;      // 全局行
    const int aK = k0 + innerColA * 4;                 // 本 float4 的 K 起点
    float4 tmpA;
    if (vecA && aRow < M && aK + 4 <= K) {
      tmpA = reinterpret_cast<const float4 *>(&A[aRow * K + aK])[0];
    } else {
      tmpA.x = (aRow < M && aK + 0 < K) ? A[aRow * K + aK + 0] : 0.0f;
      tmpA.y = (aRow < M && aK + 1 < K) ? A[aRow * K + aK + 1] : 0.0f;
      tmpA.z = (aRow < M && aK + 2 < K) ? A[aRow * K + aK + 2] : 0.0f;
      tmpA.w = (aRow < M && aK + 3 < K) ? A[aRow * K + aK + 3] : 0.0f;
    }
    As[(innerColA * 4 + 0) * BM + innerRowA] = tmpA.x;
    As[(innerColA * 4 + 1) * BM + innerRowA] = tmpA.y;
    As[(innerColA * 4 + 2) * BM + innerRowA] = tmpA.z;
    As[(innerColA * 4 + 3) * BM + innerRowA] = tmpA.w;

    // B tile 搬运：GMEM 读一个 float4，落进 swizzle 布局的 Bs4
    const int bK = k0 + innerRowB;                     // 全局 K 行
    const int bCol = blockIdx.x * BN + innerColB * 4;  // 本 float4 的列起点
    float4 tmpB;
    if (vecB && bK < K && bCol + 4 <= N) {
      tmpB = reinterpret_cast<const float4 *>(&B[bK * N + bCol])[0];
    } else {
      tmpB.x = (bK < K && bCol + 0 < N) ? B[bK * N + bCol + 0] : 0.0f;
      tmpB.y = (bK < K && bCol + 1 < N) ? B[bK * N + bCol + 1] : 0.0f;
      tmpB.z = (bK < K && bCol + 2 < N) ? B[bK * N + bCol + 2] : 0.0f;
      tmpB.w = (bK < K && bCol + 3 < N) ? B[bK * N + bCol + 3] : 0.0f;
    }
    Bs4[(innerRowB * (TN / 4) + f4) * (BN / TN) + txB] = tmpB;

    __syncthreads();

    // 内积：smem 读全部 float4
    for (int k = 0; k < BK; k++) {
      float4 regM[TM / 4], regN[TN / 4];
      // As 读
      for (int r4 = 0; r4 < TM / 4; r4++)
        regM[r4] = reinterpret_cast<const float4 *>(
            &As[k * BM + threadIdx.y * TM + r4 * 4])[0];
      // Bs4 读
      for (int c4 = 0; c4 < TN / 4; c4++)
        regN[c4] = Bs4[(k * (TN / 4) + c4) * (BN / TN) + threadIdx.x];
      // 64 个 FMA
      for (int r = 0; r < TM; r++)
        for (int c = 0; c < TN; c++)
          results[r * TN + c] += ((float *)regM)[r] * ((float *)regN)[c];
    }

    __syncthreads();
  }

  // 写回：每行 2 个 float4 store
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

}  // namespace sgemm_k6

void launch_sgemm_no_bank_conflict(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
  dim3 blockDim(sgemm_k6::BN / sgemm_k6::TN, sgemm_k6::BM / sgemm_k6::TM, 1);
  dim3 gridDim((N + sgemm_k6::BN - 1) / sgemm_k6::BN,
               (M + sgemm_k6::BM - 1) / sgemm_k6::BM, 1);
  sgemm_k6::sgemm_no_bank_conflict<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
