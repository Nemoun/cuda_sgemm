#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 本头文件自己的 CUDA 错误检查
#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e));          \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

namespace sgemm_k9 {

// 模板化 kernel
template <int BM, int BN, int BK, int TM, int TN>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_autotuned(int M, int N, int K, const float *A, const float *B, float *C) {
  constexpr int THREADS = (BM / TM) * (BN / TN);

  // 配置合法性检查
  static_assert(BK % 4 == 0 && BN % 4 == 0, "BK/BN 必须是 4 的倍数");
  static_assert((THREADS * 4) % BK == 0, "搬运排班要求 THREADS*4 % BK == 0");
  static_assert((THREADS * 4) / BK <= BM && BM % ((THREADS * 4) / BK) == 0,
                "A tile 偏移步长必须整除 BM 且不超过 BM");
  static_assert(THREADS % (BN / 4) == 0, "搬运排班要求 THREADS % (BN/4) == 0");
  static_assert(THREADS / (BN / 4) <= BK && BK % (THREADS / (BN / 4)) == 0,
                "B tile 偏移步长必须整除 BK 且不超过 BK");

  const int colBase = blockIdx.x * BN + threadIdx.x * TN;  // 0..N-1
  const int rowBase = blockIdx.y * BM + threadIdx.y * TM;  // 0..M-1

  __shared__ float As[BK * BM];  // 转置布局
  __shared__ float Bs[BK * BN];  // 行主序

  // 搬运排班
  const int linear = threadIdx.y * (BN / TN) + threadIdx.x;  // 0..THREADS-1
  const int innerRowA = linear / (BK / 4);
  const int innerColA = linear % (BK / 4);
  const int strideA = (THREADS * 4) / BK;
  const int innerRowB = linear / (BN / 4);
  const int innerColB = linear % (BN / 4);
  const int strideB = THREADS / (BN / 4);

  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  float results[TM * TN] = {0.0};

  for (int k0 = 0; k0 < K; k0 += BK) {
    // A tile
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

    // B tile
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

    // 内积
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

  // 写回
  for (int r = 0; r < TM; r++) {
    if (rowBase + r >= M) continue;
    if (vecB && colBase + TN <= N && TN % 4 == 0) {
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

// 计时辅助：某配置跑 1 次预热 + 3 次计时取最快
template <int BM, int BN, int BK, int TM, int TN>
float time_config(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 block(BN / TN, BM / TM, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  auto launch = [&] { sgemm_autotuned<BM, BN, BK, TM, TN><<<grid, block>>>(M, N, K, A, B, C); };
  launch();
  CHECK_CUDA(cudaGetLastError());
  CHECK_CUDA(cudaDeviceSynchronize());
  cudaEvent_t s, e;
  CHECK_CUDA(cudaEventCreate(&s));
  CHECK_CUDA(cudaEventCreate(&e));
  float best = 1e30f;
  for (int i = 0; i < 3; i++) {
    CHECK_CUDA(cudaEventRecord(s));
    launch();
    CHECK_CUDA(cudaEventRecord(e));
    CHECK_CUDA(cudaEventSynchronize(e));
    float ms = 0.0f;
    CHECK_CUDA(cudaEventElapsedTime(&ms, s, e));
    if (ms < best) best = ms;
  }
  CHECK_CUDA(cudaEventDestroy(s));
  CHECK_CUDA(cudaEventDestroy(e));
  return best;
}

// 运行时配置分发
float dispatch(int BM, int BN, int BK, int TM, int TN,
               const float *A, const float *B, float *C, int M, int N, int K) {
#define TRY(bm, bn, bk, tm, tn)                                      \
  if (BM == bm && BN == bn && BK == bk && TM == tm && TN == tn)      \
    return time_config<bm, bn, bk, tm, tn>(A, B, C, M, N, K);
  TRY(128, 128, 8, 8, 8)
  TRY(128, 128, 16, 8, 8)
  TRY(128, 128, 32, 8, 8)
  TRY(128, 128, 16, 8, 4)
  TRY(128, 128, 32, 8, 4)
  TRY(128, 128, 16, 4, 8)
  TRY(64, 128, 16, 8, 8)
  TRY(128, 64, 16, 8, 8)
  TRY(64, 64, 16, 8, 8)
  TRY(64, 64, 8, 8, 8)
  TRY(64, 64, 32, 8, 8)
#undef TRY
  fprintf(stderr, "autotuned: 未注册的配置 (%d,%d,%d,%d,%d)\n", BM, BN, BK, TM, TN);
  return 1e30f;
}

}  // namespace sgemm_k9

// 对外入口
float launch_sgemm_autotuned(const float *A, const float *B, float *C,
                             int M, int N, int K) {
  struct Cfg { int BM, BN, BK, TM, TN; };
  const Cfg cfg_list[] = {
      {128, 128, 8, 8, 8},    // vectorized 基准
      {128, 128, 16, 8, 8},   // BK 翻倍
      {128, 128, 32, 8, 8},   // BK 再翻倍
      {128, 128, 16, 8, 4},   // 512 线程、32 累加器
      {128, 128, 32, 8, 4},
      {128, 128, 16, 4, 8},   // 512 线程（TM 方向）
      {64, 128, 16, 8, 8},    // 128 线程窄 tile
      {128, 64, 16, 8, 8},    // 128 线程扁 tile
      {64, 64, 16, 8, 8},     // 64 线程小 tile
      {64, 64, 8, 8, 8},
      {64, 64, 32, 8, 8},
  };
  const int ncfg = (int)(sizeof(cfg_list) / sizeof(cfg_list[0]));

  printf("autotuned: 扫描 %d 个配置\n", ncfg);
  float best_ms = 1e30f;
  int best_i = -1;
  for (int i = 0; i < ncfg; i++) {
    const Cfg &c = cfg_list[i];
    float ms = sgemm_k9::dispatch(c.BM, c.BN, c.BK, c.TM, c.TN, A, B, C, M, N, K);
    double gf = 2.0 * (double)M * N * K / (ms * 1e6);
    printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%d TN=%d : %8.2f GFLOPS\n",
           i, c.BM, c.BN, c.BK, c.TM, c.TN, gf);
    if (ms < best_ms) { best_ms = ms; best_i = i; }
  }

  // 用最优配置再跑一次，最终结果写进 C
  const Cfg &best = cfg_list[best_i];
  float final_ms = sgemm_k9::dispatch(best.BM, best.BN, best.BK, best.TM, best.TN,
                                      A, B, C, M, N, K);
  printf("autotuned: 最优配置 [%2d] BM=%d BN=%d BK=%d TM=%d TN=%d\n",
         best_i, best.BM, best.BN, best.BK, best.TM, best.TN);
  return final_ms;
}
