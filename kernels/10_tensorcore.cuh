#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <mma.h>

// kernel 10：Tensor Core v1 —— wmma TF32，fragment 直接从 global 载入
namespace sgemm_k10 {
namespace wmma = nvcuda::wmma;

constexpr int BLOCK_M = 64;   // block tile 行数
constexpr int BLOCK_N = 64;   // block tile 列数
constexpr int WARP_M = 32;    // 每 warp 子 tile 行数
constexpr int WARP_N = 32;    // 每 warp 子 tile 列数
constexpr int THREADS = 128;  // 4 warps

__global__ void __launch_bounds__(THREADS, 1)
    sgemm_tensorcore(int M, int N, int K, const float *A, const float *B, float *C) {
  const int linear = threadIdx.y * blockDim.x + threadIdx.x;  // 0..127
  const int warpIdx = linear / 32;                    // 0..3
  const int warpRow = warpIdx / (BLOCK_N / WARP_N);   // 0..1
  const int warpCol = warpIdx % (BLOCK_N / WARP_N);   // 0..1
  const int lane = linear % 32;                      // warp 内线程号

  // warp 子 tile 左上角
  const int m0 = blockIdx.y * BLOCK_M + warpRow * WARP_M;
  const int n0 = blockIdx.x * BLOCK_N + warpCol * WARP_N;

  // 2×2 个 16×16 fragment
  for (int fi = 0; fi < WARP_M / 16; fi++) {
    for (int fj = 0; fj < WARP_N / 16; fj++) {
      const int fr = m0 + fi * 16;   // fragment 的行起点
      const int fc = n0 + fj * 16;   // fragment 的列起点

      if (fr + 16 <= M && fc + 16 <= N && K % 8 == 0) {
        // wmma 路径：fragment 直接从 global 载入
        wmma::fragment<wmma::matrix_a, 16, 16, 8, wmma::precision::tf32, wmma::row_major> a_frag;
        wmma::fragment<wmma::matrix_b, 16, 16, 8, wmma::precision::tf32, wmma::row_major> b_frag;
        wmma::fragment<wmma::accumulator, 16, 16, 8, float> acc;
        wmma::fill_fragment(acc, 0.0f);
        for (int k0 = 0; k0 < K; k0 += 8) {
          wmma::load_matrix_sync(a_frag, A + fr * K + k0, K);
          wmma::load_matrix_sync(b_frag, B + k0 * N + fc, N);
          wmma::mma_sync(acc, a_frag, b_frag, acc);
        }
        wmma::store_matrix_sync(C + fr * N + fc, acc, N, wmma::mem_row_major);
      } else {
        // 标量回退
        const int c_row = (lane / 4) * 2;   // 0..14
        const int c_col = (lane % 4) * 4;   // 0..12
        float c[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        for (int k = 0; k < K; k++) {
          const float a0 = (fr + c_row + 0 < M) ? A[(fr + c_row + 0) * K + k] : 0.0f;
          const float a1 = (fr + c_row + 1 < M) ? A[(fr + c_row + 1) * K + k] : 0.0f;
          const float b0 = (fc + c_col + 0 < N) ? B[k * N + fc + c_col + 0] : 0.0f;
          const float b1 = (fc + c_col + 1 < N) ? B[k * N + fc + c_col + 1] : 0.0f;
          const float b2 = (fc + c_col + 2 < N) ? B[k * N + fc + c_col + 2] : 0.0f;
          const float b3 = (fc + c_col + 3 < N) ? B[k * N + fc + c_col + 3] : 0.0f;
          c[0] += a0 * b0; c[1] += a0 * b1; c[2] += a0 * b2; c[3] += a0 * b3;
          c[4] += a1 * b0; c[5] += a1 * b1; c[6] += a1 * b2; c[7] += a1 * b3;
        }
        if (fr + c_row + 0 < M)
          for (int cc = 0; cc < 4; cc++)
            if (fc + c_col + cc < N)
              C[(fr + c_row + 0) * N + fc + c_col + cc] = c[cc];
        if (fr + c_row + 1 < M)
          for (int cc = 0; cc < 4; cc++)
            if (fc + c_col + cc < N)
              C[(fr + c_row + 1) * N + fc + c_col + cc] = c[4 + cc];
      }
    }
  }
}

}  // namespace sgemm_k10

void launch_sgemm_tensorcore(const float *A, const float *B, float *C,
                             int M, int N, int K) {
  dim3 blockDim(32, sgemm_k10::THREADS / 32, 1);
  dim3 gridDim((N + sgemm_k10::BLOCK_N - 1) / sgemm_k10::BLOCK_N,
               (M + sgemm_k10::BLOCK_M - 1) / sgemm_k10::BLOCK_M, 1);
  sgemm_k10::sgemm_tensorcore<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
