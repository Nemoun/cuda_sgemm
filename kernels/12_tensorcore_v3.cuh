#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda_runtime.h>
#include <mma.h>

// kernel 12：Tensor Core v3 —— cp.async 双缓冲流水
// 在 v2（smem tiling）基础上，把同步搬运换成 cp.async 预取
namespace sgemm_k12 {
namespace wmma = nvcuda::wmma;

constexpr int BLOCK_M = 64;   // block tile 行数
constexpr int BLOCK_N = 64;   // block tile 列数
constexpr int BK = 16;        // K 方向分块宽度
constexpr int WARP_M = 32;    // 每 warp 子 tile 行数
constexpr int WARP_N = 32;    // 每 warp 子 tile 列数
constexpr int THREADS = 128;  // 4 warps
constexpr int PAD_A = 4;      // As 行 padding
constexpr int PAD_B = 4;      // Bs 行 padding

__global__ void __launch_bounds__(THREADS, 1)
    sgemm_tensorcore_v3(int M, int N, int K, const float *A, const float *B, float *C) {
  const int linear = threadIdx.y * blockDim.x + threadIdx.x;  // 0..127
  const int warpIdx = linear / 32;                    // 0..3
  const int warpRow = warpIdx / (BLOCK_N / WARP_N);   // 0..1
  const int warpCol = warpIdx % (BLOCK_N / WARP_N);   // 0..1

  // 双 buffer smem + per-warp 写回 scratch
  __shared__ float As[2 * BLOCK_M * (BK + PAD_A)];
  __shared__ float Bs[2 * BK * (BLOCK_N + PAD_B)];
  __shared__ float c_scratch[THREADS / 32][16][16];

  // 双 barrier
  __shared__ cuda::barrier<cuda::thread_scope_block> frontBarrier;
  __shared__ cuda::barrier<cuda::thread_scope_block> backBarrier;
  cuda::barrier<cuda::thread_scope_block> *front = &frontBarrier;
  cuda::barrier<cuda::thread_scope_block> *back = &backBarrier;
  if (linear == 0) {
    init(front, THREADS);
    init(back, THREADS);
  }
  __syncthreads();

  // 搬运排班
  const int innerRowA = linear / (BK / 4);          // 0..31
  const int innerColA = linear % (BK / 4);          // 0..3
  const int strideA = (THREADS * 4) / BK;           // 32
  const int innerRowB = linear / (BLOCK_N / 4);     // 0..7
  const int innerColB = linear % (BLOCK_N / 4);     // 0..15
  const int strideB = THREADS / (BLOCK_N / 4);      // 8

  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  // 4 个累加器 fragment
  wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc[2][2];
  for (int fi = 0; fi < 2; fi++)
    for (int fj = 0; fj < 2; fj++)
      wmma::fill_fragment(acc[fi][fj], 0.0f);

  // 把第 k 个 tile 异步预取进 buffer buf
  auto load_tile_async = [&](int k0, int buf, cuda::barrier<cuda::thread_scope_block> &bar) {
    float *As_dst = As + buf * (BLOCK_M * (BK + PAD_A));
    float *Bs_dst = Bs + buf * (BK * (BLOCK_N + PAD_B));

    // A tile
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

    // B tile
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

  // 计算 buffer buf 里的 tile
  auto mma_tile = [&](int buf) {
    const float *As_cur = As + buf * (BLOCK_M * (BK + PAD_A));
    const float *Bs_cur = Bs + buf * (BK * (BLOCK_N + PAD_B));
    for (int fi = 0; fi < 2; fi++) {
      for (int fj = 0; fj < 2; fj++) {
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

  // 主循环
  int buf = 0;
  load_tile_async(0, 0, *front);
  for (int k0 = 0; k0 + BK < K; k0 += BK) {
    load_tile_async(k0 + BK, 1 - buf, *back);
    front->arrive_and_wait();
    mma_tile(buf);
    buf = 1 - buf;
    cuda::barrier<cuda::thread_scope_block> *tmp = front;
    front = back;
    back = tmp;
    __syncthreads();  // 所有线程算完，下一轮才能覆盖刚用过的 buffer
  }
  front->arrive_and_wait();
  mma_tile(buf);

  // 写回
  for (int fi = 0; fi < 2; fi++) {
    for (int fj = 0; fj < 2; fj++) {
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

}  // namespace sgemm_k12

void launch_sgemm_tensorcore_v3(const float *A, const float *B, float *C,
                                int M, int N, int K) {
  dim3 blockDim(32, sgemm_k12::THREADS / 32, 1);
  dim3 gridDim((N + sgemm_k12::BLOCK_N - 1) / sgemm_k12::BLOCK_N,
               (M + sgemm_k12::BLOCK_M - 1) / sgemm_k12::BLOCK_M, 1);
  sgemm_k12::sgemm_tensorcore_v3<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
