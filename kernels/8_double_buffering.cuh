#pragma once
#include <cstdio>
#include <cstdlib>
#include <cuda/barrier>
#include <cuda_runtime.h>


// tile 尺寸
namespace sgemm_k8 {
constexpr int BM = 128;  // C tile 行数
constexpr int BN = 128;  // C tile 列数
constexpr int BK = 16;   // K 方向分块宽度（教程 kernel 12 用 16：计算量足以遮住预取延迟）
constexpr int TM = 8;    // 每线程沿行方向算的结果数
constexpr int TN = 8;    // 每线程沿列方向算的结果数

__global__ void __launch_bounds__((BM * BN) / (TM * TN), 1)
    sgemm_double_buffering(int M, int N, int K, const float *A, const float *B, float *C) {
  const int colBase = blockIdx.x * BN + threadIdx.x * TN;  // 0..N-1
  const int rowBase = blockIdx.y * BM + threadIdx.y * TM;  // 0..M-1

  // 双 buffer：buffer 0 与 buffer 1 交替
  __shared__ float As[2 * BK * BM];
  __shared__ float Bs[2 * BK * BN];

  // 双 barrier：front = 当前 tile 的拷贝完成信号，back = 预取 tile 的信号
  __shared__ cuda::barrier<cuda::thread_scope_block> frontBarrier;
  __shared__ cuda::barrier<cuda::thread_scope_block> backBarrier;
  cuda::barrier<cuda::thread_scope_block> *front = &frontBarrier;
  cuda::barrier<cuda::thread_scope_block> *back = &backBarrier;

  const int linear = threadIdx.y * blockDim.x + threadIdx.x;  // 0..255
  if (linear == 0) {
    init(front, blockDim.x * blockDim.y);
    init(back, blockDim.x * blockDim.y);
  }
  __syncthreads();

  // 搬运排班
  const int innerRowA = linear / (BK / 4);   // 0..63
  const int innerColA = linear % (BK / 4);   // 0..3
  const int strideA = (256 * 4) / BK;        // 64，A tile 的偏移步长
  const int innerRowB = linear / (BN / 4);   // 0..7
  const int innerColB = linear % (BN / 4);   // 0..31
  const int strideB = 256 / (BN / 4);        // 8，B tile 的偏移步长

  const bool vecA = (K % 4 == 0);
  const bool vecB = (N % 4 == 0);

  float results[TM * TN] = {0.0};

  auto load_tile_async = [&](int k0, int buf, cuda::barrier<cuda::thread_scope_block> &bar) {
    float *As_dst = As + buf * (BK * BM);
    float *Bs_dst = Bs + buf * (BK * BN);

    // A tile
    for (int off = 0; off < BM; off += strideA) {
      const int aRow = blockIdx.y * BM + innerRowA + off;
      const int aK = k0 + innerColA * 4;
      if (vecA && aRow < M && aK + 4 <= K) {
        // 发 4 次 4B 异步拷贝
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 0) * BM + innerRowA + off],
                           &A[aRow * K + aK + 0], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), bar);
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 1) * BM + innerRowA + off],
                           &A[aRow * K + aK + 1], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), bar);
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 2) * BM + innerRowA + off],
                           &A[aRow * K + aK + 2], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), bar);
        cuda::memcpy_async(&As_dst[(innerColA * 4 + 3) * BM + innerRowA + off],
                           &A[aRow * K + aK + 3], cuda::aligned_size_t<sizeof(float)>(sizeof(float)), bar);
      } else {
        As_dst[(innerColA * 4 + 0) * BM + innerRowA + off] = (aRow < M && aK + 0 < K) ? A[aRow * K + aK + 0] : 0.0f;
        As_dst[(innerColA * 4 + 1) * BM + innerRowA + off] = (aRow < M && aK + 1 < K) ? A[aRow * K + aK + 1] : 0.0f;
        As_dst[(innerColA * 4 + 2) * BM + innerRowA + off] = (aRow < M && aK + 2 < K) ? A[aRow * K + aK + 2] : 0.0f;
        As_dst[(innerColA * 4 + 3) * BM + innerRowA + off] = (aRow < M && aK + 3 < K) ? A[aRow * K + aK + 3] : 0.0f;
      }
    }

    // B tile
    for (int off = 0; off < BK; off += strideB) {
      const int bK = k0 + innerRowB + off;
      const int bCol = blockIdx.x * BN + innerColB * 4;
      if (vecB && bK < K && bCol + 4 <= N) {
        cuda::memcpy_async(&Bs_dst[(innerRowB + off) * BN + innerColB * 4],
                           &B[bK * N + bCol],
                           cuda::aligned_size_t<sizeof(float4)>(sizeof(float4)), bar);
      } else {
        Bs_dst[(innerRowB + off) * BN + innerColB * 4 + 0] = (bK < K && bCol + 0 < N) ? B[bK * N + bCol + 0] : 0.0f;
        Bs_dst[(innerRowB + off) * BN + innerColB * 4 + 1] = (bK < K && bCol + 1 < N) ? B[bK * N + bCol + 1] : 0.0f;
        Bs_dst[(innerRowB + off) * BN + innerColB * 4 + 2] = (bK < K && bCol + 2 < N) ? B[bK * N + bCol + 2] : 0.0f;
        Bs_dst[(innerRowB + off) * BN + innerColB * 4 + 3] = (bK < K && bCol + 3 < N) ? B[bK * N + bCol + 3] : 0.0f;
      }
    }
  };

  // 计算 buffer 里的 tile
  auto compute_tile = [&](int buf) {
    const float *As_cur = As + buf * (BK * BM);
    const float *Bs_cur = Bs + buf * (BK * BN);
    for (int k = 0; k < BK; k++) {
      float regM[TM], regN[TN];
      for (int r = 0; r < TM; r++) regM[r] = As_cur[k * BM + threadIdx.y * TM + r];
      for (int c = 0; c < TN; c++) regN[c] = Bs_cur[k * BN + threadIdx.x * TN + c];
      for (int r = 0; r < TM; r++)
        for (int c = 0; c < TN; c++)
          results[r * TN + c] += regM[r] * regN[c];
    }
  };

  // 预取第 0 个 tile
  int buf = 0;
  load_tile_async(0, buf, *front);

  for (int k0 = 0; k0 + BK < K; k0 += BK) {
    load_tile_async(k0 + BK, 1 - buf, *back);
    front->arrive_and_wait();
    compute_tile(buf);
    buf = 1 - buf;
    // 交换 front/back
    cuda::barrier<cuda::thread_scope_block> *tmp = front;
    front = back;
    back = tmp;
    __syncthreads();
  }

  // 计算最后一个 tile
  front->arrive_and_wait();
  compute_tile(buf);

  // 写回
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

}  // namespace sgemm_k8

void launch_sgemm_double_buffering(const float *A, const float *B, float *C,
                                   int M, int N, int K) {
  dim3 blockDim(sgemm_k8::BN / sgemm_k8::TN, sgemm_k8::BM / sgemm_k8::TM, 1);
  dim3 gridDim((N + sgemm_k8::BN - 1) / sgemm_k8::BN,
               (M + sgemm_k8::BM - 1) / sgemm_k8::BM, 1);
  sgemm_k8::sgemm_double_buffering<<<gridDim, blockDim>>>(M, N, K, A, B, C);
}
