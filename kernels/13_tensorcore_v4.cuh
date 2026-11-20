#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <mma.h>

// kernel 13：Tensor Core v4 —— 大 tile + 8 warps + BK=32 + 多级流水
namespace sgemm_k13 {
namespace wmma = nvcuda::wmma;

constexpr int BLOCK_M = 128;  // block tile 行数
constexpr int BLOCK_N = 128;  // block tile 列数
constexpr int BK = 32;        // K 方向分块宽度
constexpr int WARP_M = 32;    // 每 warp 子 tile 行数
constexpr int WARP_N = 64;    // 每 warp 子 tile 列数
constexpr int THREADS = 256;  // 8 warps
constexpr int STAGES = 2;     // 流水级数
constexpr int PAD_A = 4;      // As 行 padding
constexpr int PAD_B = 4;      // Bs 行 padding

__global__ void __launch_bounds__(THREADS, 1)
    sgemm_tensorcore_v4(int M, int N, int K, const float *A, const float *B, float *C) {
  const int linear = threadIdx.y * blockDim.x + threadIdx.x;  // 0..255
  const int warpIdx = linear / 32;                    // 0..7
  const int warpRow = warpIdx / (BLOCK_N / WARP_N);   // 0..3（128/32）
  const int warpCol = warpIdx % (BLOCK_N / WARP_N);   // 0..1（128/64）

  // 多级流水 smem + per-warp 写回 scratch
  __shared__ float As[STAGES * BLOCK_M * (BK + PAD_A)];
  __shared__ float Bs[STAGES * BK * (BLOCK_N + PAD_B)];
  __shared__ float c_scratch[THREADS / 32][16][16];

  // 每级一个 barrier
  __shared__ cuda::barrier<cuda::thread_scope_block> bars[STAGES];
  if (linear == 0)
    for (int s = 0; s < STAGES; s++) init(&bars[s], THREADS);
  __syncthreads();

  // 搬运排班
  const int innerRowA = linear / (BK / 4);          // 0..31
  const int innerColA = linear % (BK / 4);          // 0..7
  const int strideA = (THREADS * 4) / BK;           // 32，A tile 的偏移步长
  const int innerRowB = linear / (BLOCK_N / 4);     // 0..7
  const int innerColB = linear % (BLOCK_N / 4);     // 0..31
  const int strideB = THREADS / (BLOCK_N / 4);      // 8，B tile 的偏移步长

  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  // 8 个累加器 fragment
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][4];
  for (int fi = 0; fi < 2; fi++)
    for (int fj = 0; fj < 4; fj++)
      wmma::fill_fragment(acc[fi][fj], 0.0f);

  // 把第 k 个 tile 异步预取进 stage buf
  auto load_tile_async = [&](int k0, int buf, cuda::barrier<cuda::thread_scope_block> &bar) {
    float *As_dst = As + buf * (BLOCK_M * (BK + PAD_A));
    float *Bs_dst = Bs + buf * (BK * (BLOCK_N + PAD_B));

    for (int off = 0; off < BLOCK_M; off += strideA) {
      const int aRow = blockIdx.y * BLOCK_M + innerRowA + off;
      const int aK = k0 + innerColA * 4;
      if (vecA && aRow < M && aK + 4 <= K) {
        cuda::memcpy_async(&As_dst[(innerRowA + off) * (BK + PAD_A) + innerColA * 4],
                           &A[aRow * K + aK],
                           cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), bar);
      } else {
        float4 tmp;
        tmp.x = (aRow < M && aK + 0 < K) ? A[aRow * K + aK + 0] : 0.0f;
        tmp.y = (aRow < M && aK + 1 < K) ? A[aRow * K + aK + 1] : 0.0f;
        tmp.z = (aRow < M && aK + 2 < K) ? A[aRow * K + aK + 2] : 0.0f;
        tmp.w = (aRow < M && aK + 3 < K) ? A[aRow * K + aK + 3] : 0.0f;
        reinterpret_cast<float4 *>(&As_dst[(innerRowA + off) * (BK + PAD_A) + innerColA * 4])[0] = tmp;
      }
    }

    for (int off = 0; off < BK; off += strideB) {
      const int bK = k0 + innerRowB + off;
      const int bCol = blockIdx.x * BLOCK_N + innerColB * 4;
      if (vecB && bK < K && bCol + 4 <= N) {
        cuda::memcpy_async(&Bs_dst[(innerRowB + off) * (BLOCK_N + PAD_B) + innerColB * 4],
                           &B[bK * N + bCol],
                           cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), bar);
      } else {
        float4 tmp;
        tmp.x = (bK < K && bCol + 0 < N) ? B[bK * N + bCol + 0] : 0.0f;
        tmp.y = (bK < K && bCol + 1 < N) ? B[bK * N + bCol + 1] : 0.0f;
        tmp.z = (bK < K && bCol + 2 < N) ? B[bK * N + bCol + 2] : 0.0f;
        tmp.w = (bK < K && bCol + 3 < N) ? B[bK * N + bCol + 3] : 0.0f;
        reinterpret_cast<float4 *>(&Bs_dst[(innerRowB + off) * (BLOCK_N + PAD_B) + innerColB * 4])[0] = tmp;
      }
    }
  };

  // 计算 stage buf 里的 tile
  auto mma_tile = [&](int buf) {
    const float *As_cur = As + buf * (BLOCK_M * (BK + PAD_A));
    const float *Bs_cur = Bs + buf * (BK * (BLOCK_N + PAD_B));
    for (int fi = 0; fi < 2; fi++) {
      for (int fj = 0; fj < 4; fj++) {
        for (int k = 0; k < BK; k += 8) {
          wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
          wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
          wmma::load_matrix_sync(a_frag,
              As_cur + (warpRow * WARP_M + fi * 16) * (BK + PAD_A) + k, BK + PAD_A);
          wmma::load_matrix_sync(b_frag,
              Bs_cur + k * (BLOCK_N + PAD_B) + (warpCol * WARP_N + fj * 16), BLOCK_N + PAD_B);
          wmma::mma_sync(acc[fi][fj], a_frag, b_frag, acc[fi][fj]);
        }
      }
    }
  };

  // STAGES 级流水主循环
  for (int s = 0; s < STAGES - 1; s++)
    if (s * BK < K) load_tile_async(s * BK, s, bars[s]);

  int step = 0;
  for (int k0 = 0; k0 < K; k0 += BK, step++) {
    // 预取 STAGES-1 步之后的 tile 到刚空闲的 stage
    const int next_k = k0 + (STAGES - 1) * BK;
    const int next_buf = (step + STAGES - 1) % STAGES;
    if (next_k < K) load_tile_async(next_k, next_buf, bars[next_buf]);
    // 等当前 tile 的拷贝完成，然后 mma
    bars[step % STAGES].arrive_and_wait();
    mma_tile(step % STAGES);
    __syncthreads();
  }

  // 写回
  for (int fi = 0; fi < 2; fi++) {
    for (int fj = 0; fj < 4; fj++) {
      const int fr = blockIdx.y * BLOCK_M + warpRow * WARP_M + fi * 16;
      const int fc = blockIdx.x * BLOCK_N + warpCol * WARP_N + fj * 16;
      if (fr + 16 <= M && fc + 16 <= N) {
        wmma::store_matrix_sync(C + fr * N + fc, acc[fi][fj], N, wmma::mem_row_major);
      } else {
        wmma::store_matrix_sync(&c_scratch[warpIdx][0][0], acc[fi][fj], 16, wmma::mem_row_major);
        __syncwarp();
        for (int e = linear % 32; e < 256; e += 32) {
          const int r = e / 16, c = e % 16;
          if (fr + r < M && fc + c < N)
            C[(fr + r) * N + fc + c] = c_scratch[warpIdx][r][c];
        }
      }
    }
  }
}

}  // namespace sgemm_k13

void launch_sgemm_tensorcore_v4(const float *A, const float *B, float *C,
                                int M, int N, int K) {
  dim3 blockDim(32, sgemm_k13::THREADS / 32, 1);
  dim3 gridDim((N + sgemm_k13::BLOCK_N - 1) / sgemm_k13::BLOCK_N,
               (M + sgemm_k13::BLOCK_M - 1) / sgemm_k13::BLOCK_M, 1);
  sgemm_k13::sgemm_tensorcore_v4<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
