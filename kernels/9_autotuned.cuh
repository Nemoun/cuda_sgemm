#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda_runtime.h>

#define CHECK_CUDA(call)                                                     \
  do {                                                                       \
    cudaError_t _e = (call);                                                 \
    if (_e != cudaSuccess) {                                                 \
      fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(_e));          \
      exit(EXIT_FAILURE);                                                    \
    }                                                                        \
  } while (0)

namespace sgemm_k9 {

// 模板化 kernel：几何 + SWIZZLE + DBUF 三个维度
template <int BM, int BN, int BK, int TM, int TN, bool SWIZZLE, bool DBUF>
__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_autotuned(int M, int N, int K, const float *A, const float *B, float *C) {
  constexpr int THREADS = (BM / TM) * (BN / TN);
  constexpr int NBUFS = DBUF ? 2 : 1;

  // 配置合法性检查（编译期）
  static_assert(BK % 4 == 0 && BN % 4 == 0, "BK/BN 必须是 4 的倍数");
  static_assert(TM % 4 == 0 && TN % 4 == 0, "TM/TN 必须是 4 的倍数（swizzle 内积读）");
  static_assert((THREADS * 4) % BK == 0, "搬运排班要求 THREADS*4 % BK == 0");
  static_assert((THREADS * 4) / BK <= BM && BM % ((THREADS * 4) / BK) == 0,
                "A tile 偏移步长必须整除 BM 且不超过 BM");
  static_assert(THREADS % (BN / 4) == 0, "搬运排班要求 THREADS % (BN/4) == 0");
  static_assert(THREADS / (BN / 4) <= BK && BK % (THREADS / (BN / 4)) == 0,
                "B tile 偏移步长必须整除 BK 且不超过 BK");

  const int colBase = blockIdx.x * BN + threadIdx.x * TN;  // 0..N-1
  const int rowBase = blockIdx.y * BM + threadIdx.y * TM;  // 0..M-1

  // shared memory
  __shared__ float As[NBUFS * BK * BM];
  __shared__ float4 Bs4[NBUFS * (BK * BN / 4)];

  const int linear = threadIdx.y * (BN / TN) + threadIdx.x;  // 0..THREADS-1

  // DBUF 的双 barrier
  cuda::barrier<cuda::thread_scope_block> *front = nullptr;
  cuda::barrier<cuda::thread_scope_block> *back = nullptr;
  if constexpr (DBUF) {
    __shared__ cuda::barrier<cuda::thread_scope_block> frontBarrier;
    __shared__ cuda::barrier<cuda::thread_scope_block> backBarrier;
    front = &frontBarrier;
    back = &backBarrier;
    if (linear == 0) {
      init(front, THREADS);  // 期望到达数 = 线程数（每个 memcpy_async 再 +1）
      init(back, THREADS);
    }
    __syncthreads();
  }

  // 搬运排班（带偏移循环，支持任意合法 tile 尺寸）
  const int innerRowA = linear / (BK / 4);
  const int innerColA = linear % (BK / 4);
  const int strideA = (THREADS * 4) / BK;
  const int innerRowB = linear / (BN / 4);
  const int innerColB = linear % (BN / 4);
  const int strideB = THREADS / (BN / 4);
  // swizzle 落位：本线程搬的 float4（4 列）属于线程 txB 的第 f4 段
  const int f4 = innerColB % (TN / 4);
  const int txB = innerColB / (TN / 4);

  // 行首地址 16B 对齐才允许 float4 访问全局内存
  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  float results[TM * TN] = {0.0};

  // ---- 搬运第 k 个 tile 进 buffer buf（DBUF 时异步拷贝报到 bar） ----
  auto load_tile = [&](int k0, int buf,
                       cuda::barrier<cuda::thread_scope_block> *bar) {
    float *As_dst = As + buf * (BK * BM);

    // A tile：GMEM float4 → scatter 进转置 As（4 个分量，DBUF 时 4 次 4B 异步拷贝）
    for (int off = 0; off < BM; off += strideA) {
      const int aRow = blockIdx.y * BM + innerRowA + off;
      const int aK = k0 + innerColA * 4;
      if (DBUF && vecA && aRow < M && aK + 4 <= K) {
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 0) * BM + innerRowA + off],
                           &A[aRow * K + aK + 0], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), *bar);
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 1) * BM + innerRowA + off],
                           &A[aRow * K + aK + 1], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), *bar);
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 2) * BM + innerRowA + off],
                           &A[aRow * K + aK + 2], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), *bar);
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 3) * BM + innerRowA + off],
                           &A[aRow * K + aK + 3], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), *bar);
      } else {  // 尺寸不满足 4 对齐或 tile 越界：同步拷贝 + 填 0
        As_dst[(innerColA * 4 + 0) * BM + innerRowA + off] = (aRow < M && aK + 0 < K) ? A[aRow * K + aK + 0] : 0.0f;
        As_dst[(innerColA * 4 + 1) * BM + innerRowA + off] = (aRow < M && aK + 1 < K) ? A[aRow * K + aK + 1] : 0.0f;
        As_dst[(innerColA * 4 + 2) * BM + innerRowA + off] = (aRow < M && aK + 2 < K) ? A[aRow * K + aK + 2] : 0.0f;
        As_dst[(innerColA * 4 + 3) * BM + innerRowA + off] = (aRow < M && aK + 3 < K) ? A[aRow * K + aK + 3] : 0.0f;
      }
    }

    // B tile：GMEM float4 → plain 行主序或 swizzle 重排落位
    for (int off = 0; off < BK; off += strideB) {
      const int bK = k0 + innerRowB + off;
      const int bCol = blockIdx.x * BN + innerColB * 4;
      // 目的 float4 下标：plain = (k 行, innerColB 段)；swizzle = (k 行, f4 段, txB 线程)
      const int dst4 = buf * (BK * BN / 4) + (SWIZZLE
          ? ((innerRowB + off) * (TN / 4) + f4) * (BN / TN) + txB
          : (innerRowB + off) * (BN / 4) + innerColB);
      if (DBUF && vecB && bK < K && bCol + 4 <= N) {
        cuda::memcpy_async(&Bs4[dst4], &B[bK * N + bCol],
                           cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), *bar);
      } else {  // 尺寸不满足 4 对齐或 tile 越界：同步拷贝 + 填 0
        float4 tmp;
        if (vecB && bK < K && bCol + 4 <= N) {
          tmp = reinterpret_cast<const float4 *>(&B[bK * N + bCol])[0];
        } else {
          tmp.x = (bK < K && bCol + 0 < N) ? B[bK * N + bCol + 0] : 0.0f;
          tmp.y = (bK < K && bCol + 1 < N) ? B[bK * N + bCol + 1] : 0.0f;
          tmp.z = (bK < K && bCol + 2 < N) ? B[bK * N + bCol + 2] : 0.0f;
          tmp.w = (bK < K && bCol + 3 < N) ? B[bK * N + bCol + 3] : 0.0f;
        }
        Bs4[dst4] = tmp;
      }
    }
  };

  // ---- 计算 buffer buf 里的 tile ----
  auto compute_tile = [&](int buf) {
    const float *As_cur = As + buf * (BK * BM);
    const float *Bs_cur = reinterpret_cast<const float *>(Bs4 + buf * (BK * BN / 4));
    for (int k = 0; k < BK; k++) {
      if (SWIZZLE) {
        // 内积读全部 float4：每 dotIdx 2×(TM/4)+2×(TN/4) 次 LDS.128，零 bank conflict
        float4 regM[TM / 4], regN[TN / 4];
        for (int r4 = 0; r4 < TM / 4; r4++)
          regM[r4] = reinterpret_cast<const float4 *>(
              &As_cur[k * BM + threadIdx.y * TM + r4 * 4])[0];
        for (int c4 = 0; c4 < TN / 4; c4++)
          regN[c4] = Bs4[buf * (BK * BN / 4) +
                         (k * (TN / 4) + c4) * (BN / TN) + threadIdx.x];
        for (int r = 0; r < TM; r++)
          for (int c = 0; c < TN; c++)
            results[r * TN + c] += ((float *)regM)[r] * ((float *)regN)[c];
      } else {
        // 标量内积读（plain 布局）
        float regM[TM], regN[TN];
        for (int r = 0; r < TM; r++) regM[r] = As_cur[k * BM + threadIdx.y * TM + r];
        for (int c = 0; c < TN; c++) regN[c] = Bs_cur[k * BN + threadIdx.x * TN + c];
        for (int r = 0; r < TM; r++)
          for (int c = 0; c < TN; c++)
            results[r * TN + c] += regM[r] * regN[c];
      }
    }
  };

  // ---- 主循环：DBUF = 预取下一 tile → 等当前 → 算当前 → 交换 ----
  int buf = 0;
  if (DBUF) {
    load_tile(0, 0, front);  // 冷启动预取
    for (int k0 = 0; k0 + BK < K; k0 += BK) {
      load_tile(k0 + BK, 1 - buf, back);
      front->arrive_and_wait();  // 所有线程到达 + 当前 tile 拷贝完成
      compute_tile(buf);
      buf = 1 - buf;
      cuda::barrier<cuda::thread_scope_block> *tmp = front;
      front = back;
      back = tmp;
      __syncthreads();  // 所有线程算完，下一轮才能覆盖刚用过的 buffer
    }
    front->arrive_and_wait();
    compute_tile(buf);
  } else {
    for (int k0 = 0; k0 < K; k0 += BK) {
      load_tile(k0, 0, nullptr);
      __syncthreads();
      compute_tile(0);
      __syncthreads();
    }
  }

  // ---- 写回（与第一期相同） ----
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

// ---- 计时辅助：某配置跑 1 次预热 + 3 次计时取最快 ----
template <int BM, int BN, int BK, int TM, int TN, bool SWIZZLE, bool DBUF>
float time_config(const float *A, const float *B, float *C, int M, int N, int K) {
  dim3 block(BN / TN, BM / TM, 1);
  dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM, 1);
  auto launch = [&] { sgemm_autotuned<BM, BN, BK, TM, TN, SWIZZLE, DBUF><<<grid, block>>>(M, N, K, A, B, C); };
  launch();  // 预热
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

// ---- 运行时配置分发（编译期实例化） ----
float dispatch(int BM, int BN, int BK, int TM, int TN, bool swizzle, bool dbuf,
               const float *A, const float *B, float *C, int M, int N, int K) {
#define TRY(bm, bn, bk, tm, tn, sw, db)                                      \
  if (BM == bm && BN == bn && BK == bk && TM == tm && TN == tn &&            \
      swizzle == sw && dbuf == db)                                           \
    return time_config<bm, bn, bk, tm, tn, sw, db>(A, B, C, M, N, K);
  // plain + sync（第一期全集）
  TRY(128, 128, 8, 8, 8, false, false)
  TRY(128, 128, 16, 8, 8, false, false)
  TRY(128, 128, 32, 8, 8, false, false)
  TRY(128, 128, 16, 8, 4, false, false)
  TRY(128, 128, 32, 8, 4, false, false)
  TRY(128, 128, 16, 4, 8, false, false)
  TRY(64, 128, 16, 8, 8, false, false)
  TRY(128, 64, 16, 8, 8, false, false)
  TRY(64, 64, 16, 8, 8, false, false)
  TRY(64, 64, 8, 8, 8, false, false)
  TRY(64, 64, 32, 8, 8, false, false)
  // swizzle + sync
  TRY(128, 128, 8, 8, 8, true, false)
  TRY(128, 128, 16, 8, 8, true, false)
  TRY(128, 128, 32, 8, 8, true, false)
  TRY(128, 128, 16, 8, 4, true, false)
  TRY(128, 128, 32, 8, 4, true, false)
  TRY(128, 128, 16, 4, 8, true, false)
  TRY(64, 128, 16, 8, 8, true, false)
  TRY(128, 64, 16, 8, 8, true, false)
  TRY(64, 64, 16, 8, 8, true, false)
  TRY(64, 64, 8, 8, 8, true, false)
  TRY(64, 64, 32, 8, 8, true, false)
  // plain + dbuf（只扫 BK≥16）
  TRY(128, 128, 16, 8, 8, false, true)
  TRY(128, 128, 32, 8, 8, false, true)
  TRY(128, 128, 16, 8, 4, false, true)
  TRY(128, 128, 32, 8, 4, false, true)
  TRY(128, 128, 16, 4, 8, false, true)
  TRY(64, 128, 16, 8, 8, false, true)
  TRY(128, 64, 16, 8, 8, false, true)
  TRY(64, 64, 16, 8, 8, false, true)
  TRY(64, 64, 32, 8, 8, false, true)
  // swizzle + dbuf
  TRY(128, 128, 16, 8, 8, true, true)
  TRY(128, 128, 32, 8, 8, true, true)
  TRY(128, 128, 16, 8, 4, true, true)
  TRY(128, 128, 32, 8, 4, true, true)
  TRY(128, 128, 16, 4, 8, true, true)
  TRY(64, 128, 16, 8, 8, true, true)
  TRY(128, 64, 16, 8, 8, true, true)
  TRY(64, 64, 16, 8, 8, true, true)
  TRY(64, 64, 32, 8, 8, true, true)
#undef TRY
  fprintf(stderr, "autotuned: 未注册的配置 (%d,%d,%d,%d,%d,%d,%d)\n",
          BM, BN, BK, TM, TN, (int)swizzle, (int)dbuf);
  return 1e30f;
}

}  // namespace sgemm_k9

// ---- 对外入口：扫配置表 → 选最优 → 用最优再跑一次，返回耗时 ----
// 结果留在 C 里供对拍；扫描明细打印到 stdout
float launch_sgemm_autotuned(const float *A, const float *B, float *C,
                             int M, int N, int K) {
  struct Cfg {
    int BM, BN, BK, TM, TN;
    bool swizzle, dbuf;
    const char *variant;  // 打印用变体名
  };
  const Cfg cfg_list[] = {
      // plain + sync（第一期基线）
      {128, 128, 8, 8, 8, false, false, "plain sync"},
      {128, 128, 16, 8, 8, false, false, "plain sync"},
      {128, 128, 32, 8, 8, false, false, "plain sync"},
      {128, 128, 16, 8, 4, false, false, "plain sync"},
      {128, 128, 32, 8, 4, false, false, "plain sync"},
      {128, 128, 16, 4, 8, false, false, "plain sync"},
      {64, 128, 16, 8, 8, false, false, "plain sync"},
      {128, 64, 16, 8, 8, false, false, "plain sync"},
      {64, 64, 16, 8, 8, false, false, "plain sync"},
      {64, 64, 8, 8, 8, false, false, "plain sync"},
      {64, 64, 32, 8, 8, false, false, "plain sync"},
      // swizzle + sync
      {128, 128, 8, 8, 8, true, false, "swizzle sync"},
      {128, 128, 16, 8, 8, true, false, "swizzle sync"},
      {128, 128, 32, 8, 8, true, false, "swizzle sync"},
      {128, 128, 16, 8, 4, true, false, "swizzle sync"},
      {128, 128, 32, 8, 4, true, false, "swizzle sync"},
      {128, 128, 16, 4, 8, true, false, "swizzle sync"},
      {64, 128, 16, 8, 8, true, false, "swizzle sync"},
      {128, 64, 16, 8, 8, true, false, "swizzle sync"},
      {64, 64, 16, 8, 8, true, false, "swizzle sync"},
      {64, 64, 8, 8, 8, true, false, "swizzle sync"},
      {64, 64, 32, 8, 8, true, false, "swizzle sync"},
      // plain + dbuf（BK≥16）
      {128, 128, 16, 8, 8, false, true, "plain dbuf"},
      {128, 128, 32, 8, 8, false, true, "plain dbuf"},
      {128, 128, 16, 8, 4, false, true, "plain dbuf"},
      {128, 128, 32, 8, 4, false, true, "plain dbuf"},
      {128, 128, 16, 4, 8, false, true, "plain dbuf"},
      {64, 128, 16, 8, 8, false, true, "plain dbuf"},
      {128, 64, 16, 8, 8, false, true, "plain dbuf"},
      {64, 64, 16, 8, 8, false, true, "plain dbuf"},
      {64, 64, 32, 8, 8, false, true, "plain dbuf"},
      // swizzle + dbuf
      {128, 128, 16, 8, 8, true, true, "swizzle dbuf"},
      {128, 128, 32, 8, 8, true, true, "swizzle dbuf"},
      {128, 128, 16, 8, 4, true, true, "swizzle dbuf"},
      {128, 128, 32, 8, 4, true, true, "swizzle dbuf"},
      {128, 128, 16, 4, 8, true, true, "swizzle dbuf"},
      {64, 128, 16, 8, 8, true, true, "swizzle dbuf"},
      {128, 64, 16, 8, 8, true, true, "swizzle dbuf"},
      {64, 64, 16, 8, 8, true, true, "swizzle dbuf"},
      {64, 64, 32, 8, 8, true, true, "swizzle dbuf"},
  };
  const int ncfg = (int)(sizeof(cfg_list) / sizeof(cfg_list[0]));

  printf("autotuned: 扫描 %d 个配置\n", ncfg);
  float best_ms = 1e30f;
  int best_i = -1;
  for (int i = 0; i < ncfg; i++) {
    const Cfg &c = cfg_list[i];
    float ms = sgemm_k9::dispatch(c.BM, c.BN, c.BK, c.TM, c.TN, c.swizzle, c.dbuf,
                                  A, B, C, M, N, K);
    double gf = 2.0 * (double)M * N * K / (ms * 1e6);
    printf("  [%2d] BM=%3d BN=%3d BK=%2d TM=%d TN=%d %-12s : %8.2f GFLOPS\n",
           i, c.BM, c.BN, c.BK, c.TM, c.TN, c.variant, gf);
    if (ms < best_ms) { best_ms = ms; best_i = i; }
  }

  // 用最优配置再跑一次，最终结果写进 C
  const Cfg &best = cfg_list[best_i];
  float final_ms = sgemm_k9::dispatch(best.BM, best.BN, best.BK, best.TM, best.TN,
                                      best.swizzle, best.dbuf, A, B, C, M, N, K);
  printf("autotuned: 最优配置 [%2d] BM=%d BN=%d BK=%d TM=%d TN=%d %s\n",
         best_i, best.BM, best.BN, best.BK, best.TM, best.TN, best.variant);
  return final_ms;
}
