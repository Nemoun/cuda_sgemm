#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// tile 尺寸
namespace sgemm_k7 {
constexpr int BM = 128;     // C tile 行数
constexpr int BN = 128;     // C tile 列数
constexpr int BK = 16;      // K 方向分块宽度
constexpr int WM = 64;      // 每 warp 的 C 子 tile 行数
constexpr int WN = 64;      // 每 warp 的 C 子 tile 列数
constexpr int WNITER = 4;   // 子 tile 在 N 方向的迭代数
constexpr int TM = 8;       // 每线程沿行方向算的结果数
constexpr int TN = 4;       // 每线程沿列方向算的结果数
constexpr int THREADS = 128;
constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);  // = 1
constexpr int WSUBM = WM / WMITER;   // 64
constexpr int WSUBN = WN / WNITER;   // 16

__global__ void __launch_bounds__(THREADS, 1)
    sgemm_warptiling(int M, int N, int K, const float *A, const float *B, float *C) {
  // ---- 线性线程号、warp 归属、线程在 warp 子 tile 中的位置 ----
  const int linear = threadIdx.y * blockDim.x + threadIdx.x;   // 0..127
  const int warpIdx = linear / 32;                             // 0..3
  const int warpRow = warpIdx / (BN / WN);                     // 0..1
  const int warpCol = warpIdx % (BN / WN);                     // 0..1
  const int threadInWarp = linear % 32;                        // 0..31
  const int threadRowInWarp = threadInWarp / (WSUBN / TN);     // 0..7
  const int threadColInWarp = threadInWarp % (WSUBN / TN);     // 0..3

  // shared memory
  __shared__ float As[BK * BM];
  __shared__ float Bs[BK * BN];

  // 搬运排班
  const int innerRowA = linear / (BK / 4);   // 0..31
  const int innerColA = linear % (BK / 4);   // 0..3
  const int strideA = (THREADS * 4) / BK;    // 32，A tile 的偏移步长
  const int innerRowB = linear / (BN / 4);   // 0..3
  const int innerColB = linear % (BN / 4);   // 0..31
  const int strideB = THREADS / (BN / 4);    // 4，B tile 的偏移步长

  // 行首地址 16B 对齐才允许 float4 访问全局内存
  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  float results[WMITER * TM * WNITER * TN] = {0.0};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile 搬运
    for (int off = 0; off < BM; off += strideA) {
      const int aRow = blockIdx.y * BM + innerRowA + off;
      const int aK = k0 + innerColA * 4;
      float4 tmp;
      if (vecA && aRow < M && aK + 4 <= K) {
        tmp = reinterpret_cast<const float4 *>(&A[aRow * K + aK])[0];
      } else {
        tmp.x = (aRow < M && aK + 0 < K) ? A[aRow * K + aK + 0] : 0.0f;
        tmp.y = (aRow < M && aK + 1 < K) ? A[aRow * K + aK + 1] : 0.0f;
        tmp.z = (aRow < M && aK + 2 < K) ? A[aRow * K + aK + 2] : 0.0f;
        tmp.w = (aRow < M && aK + 3 < K) ? A[aRow * K + aK + 3] : 0.0f;
      }
      As[(innerColA * 4 + 0) * BM + innerRowA + off] = tmp.x;
      As[(innerColA * 4 + 1) * BM + innerRowA + off] = tmp.y;
      As[(innerColA * 4 + 2) * BM + innerRowA + off] = tmp.z;
      As[(innerColA * 4 + 3) * BM + innerRowA + off] = tmp.w;
    }

    // B tile 搬运
    for (int off = 0; off < BK; off += strideB) {
      const int bK = k0 + innerRowB + off;
      const int bCol = blockIdx.x * BN + innerColB * 4;
      float4 tmp;
      if (vecB && bK < K && bCol + 4 <= N) {
        tmp = reinterpret_cast<const float4 *>(&B[bK * N + bCol])[0];
      } else {
        tmp.x = (bK < K && bCol + 0 < N) ? B[bK * N + bCol + 0] : 0.0f;
        tmp.y = (bK < K && bCol + 1 < N) ? B[bK * N + bCol + 1] : 0.0f;
        tmp.z = (bK < K && bCol + 2 < N) ? B[bK * N + bCol + 2] : 0.0f;
        tmp.w = (bK < K && bCol + 3 < N) ? B[bK * N + bCol + 3] : 0.0f;
      }
      reinterpret_cast<float4 *>(&Bs[(innerRowB + off) * BN + innerColB * 4])[0] = tmp;
    }

    __syncthreads();

    // 内积：每 dotIdx 24 次标量 LDS 换 128 个 FMA
    for (int k = 0; k < BK; k++) {
      float regM[WMITER * TM], regN[WNITER * TN];
      for (int i = 0; i < TM; i++)
        regM[i] = As[k * BM + warpRow * WM + threadRowInWarp * TM + i];
      for (int wsc = 0; wsc < WNITER; wsc++)
        for (int i = 0; i < TN; i++)
          regN[wsc * TN + i] =
              Bs[k * BN + warpCol * WN + wsc * WSUBN + threadColInWarp * TN + i];
      for (int wsr = 0; wsr < WMITER; wsr++)
        for (int wsc = 0; wsc < WNITER; wsc++)
          for (int r = 0; r < TM; r++)
            for (int c = 0; c < TN; c++)
              results[(wsr * TM + r) * (WNITER * TN) + wsc * TN + c] +=
                  regM[wsr * TM + r] * regN[wsc * TN + c];
    }

    __syncthreads();
  }

  // 写回
  const int rowTop = blockIdx.y * BM + warpRow * WM + threadRowInWarp * TM;
  const int colLeft = blockIdx.x * BN + warpCol * WN + threadColInWarp * TN;
  for (int wsr = 0; wsr < WMITER; wsr++)
    for (int wsc = 0; wsc < WNITER; wsc++) {
      const int gCol = colLeft + wsc * WSUBN;
      for (int r = 0; r < TM; r++) {
        const int gRow = rowTop + wsr * WSUBM + r;
        if (gRow >= M) continue;
        const int idx = (wsr * TM + r) * (WNITER * TN) + wsc * TN;
        if (vecB && gCol + TN <= N) {
          float4 tmp;
          tmp.x = results[idx + 0];
          tmp.y = results[idx + 1];
          tmp.z = results[idx + 2];
          tmp.w = results[idx + 3];
          reinterpret_cast<float4 *>(&C[gRow * N + gCol])[0] = tmp;
        } else {
          for (int c = 0; c < TN; c++)
            if (gCol + c < N) C[gRow * N + gCol + c] = results[idx + c];
        }
      }
    }
}

}  // namespace sgemm_k7

void launch_sgemm_warptiling(const float *A, const float *B, float *C,
                             int M, int N, int K) {
  dim3 blockDim(16, sgemm_k7::THREADS / 16, 1);   // (16, 8) = 128 线程 = 4 warps
  dim3 gridDim((N + sgemm_k7::BN - 1) / sgemm_k7::BN,
               (M + sgemm_k7::BM - 1) / sgemm_k7::BM, 1);
  sgemm_k7::sgemm_warptiling<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
