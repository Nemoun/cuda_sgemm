#pragma once
#include <cstdio>

#include "11_tensorcore_v2.cuh"
#include "12_tensorcore_v3.cuh"
#include "13_tensorcore_v4.cuh"

// kernel 14：TC 尺寸自适应 —— 按 (M, N) 选择 tile 尺寸
void launch_sgemm_tc_adaptive(const float *A, const float *B, float *C,
                              int M, int N, int K) {
  // tc4（128×128）在这个尺寸下的 block 数
  const long blocks4 = ((long)M + 127) / 128 * (((long)N + 127) / 128);
  const bool use_v4 = blocks4 >= 128;

  // 打印一次选择结果
  static bool printed = false;
  if (!printed) {
    printf("tcauto: %s\n", use_v4 ? "大尺寸 → tc4（128×128 tile）"
                                 : "小尺寸 → tc3（64×64 tile）");
    printed = true;
  }

  if (use_v4) {
    launch_sgemm_tensorcore_v4(A, B, C, M, N, K);
  } else {
    launch_sgemm_tensorcore_v3(A, B, C, M, N, K);
  }
}
