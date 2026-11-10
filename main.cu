// ============================================================
// CUDA SGEMM benchmark 入口
// 职责：分配矩阵 → 启动 kernel（计时）→ CPU/cuBLAS 对拍 → 输出 GFLOPS
// 用法：./sgemm [M [N [K]]] [--dump out.bin]
// ============================================================
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include "kernels/naive.cuh"
#include "kernels/1_global_memory.cuh"
#include "kernels/2_shared_memory.cuh"
#include "kernels/3_1d_blocktiling.cuh"

// CUDA API 错误检查
#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// cuBLAS 错误检查
#define CHECK_CUBLAS(call)                                                     \
  do {                                                                         \
    cublasStatus_t st = (call);                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
      fprintf(stderr, "cuBLAS error at %s:%d: status %d\n", __FILE__,          \
              __LINE__, (int)st);                                              \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

// ---------------- CPU 参考实现（行主序三重循环） ----------------
void sgemm_cpu(const float *A, const float *B, float *C, int M, int N, int K) {
  for (int i = 0; i < M; i++)
    for (int j = 0; j < N; j++) {
      float sum = 0.0f;
      for (int k = 0; k < K; k++)
        sum += A[i * K + k] * B[k * N + j];
      C[i * N + j] = sum;
    }
}

// ------- 确定性初始化（与 bench.py 保持一致，方便跨语言对拍） -------
// 不用 rand()：同一输入每次运行可复现，且 Python 能生成完全相同的矩阵
void init_matrices(float *A, float *B, int M, int N, int K) {
  for (int i = 0; i < M * K; i++) A[i] = (float)((i % 101) - 50) * 0.01f;
  for (int j = 0; j < K * N; j++) B[j] = (float)((j % 97) - 48) * 0.01f;
}

// ---------- 计时：launch() 里启动 kernel，用 CUDA event 量耗时 ----------
// 跑 iters 次，返回最快一次（对拍数值用，性能曲线也建议用最快值）
template <typename F>
float time_ms(F &&launch, int iters) {
  cudaEvent_t start, stop;
  CHECK(cudaEventCreate(&start));
  CHECK(cudaEventCreate(&stop));
  float best = 1e30f;
  for (int i = 0; i < iters; i++) {
    CHECK(cudaEventRecord(start));
    launch();
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    float ms = 0.0f;
    CHECK(cudaEventElapsedTime(&ms, start, stop));
    if (ms < best) best = ms;
  }
  CHECK(cudaEventDestroy(start));
  CHECK(cudaEventDestroy(stop));
  return best;
}

// ---------- 对拍：|x - ref| <= 1e-3 * (1 + |ref|) 即认为一致 ----------
// float32 累加顺序不同会带来 ~1e-4 量级的相对误差，阈值 1e-3 足够安全。
// 相对误差以 ref 的最大绝对值为基准（逐元素除会因 ref≈0 产生虚高的假象）
int check_against(const float *x, const float *ref, int n, float *max_rel_err) {
  float scale = 1e-6f;
  for (int i = 0; i < n; i++) scale = fmaxf(scale, fabsf(ref[i]));
  int bad = 0;
  float max_abs_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float abs_err = fabsf(x[i] - ref[i]);
    max_abs_err = fmaxf(max_abs_err, abs_err);
    if (abs_err > 1e-3f * (1.0f + fabsf(ref[i]))) bad++;
  }
  *max_rel_err = max_abs_err / scale;
  return bad;
}

// ---------- 把结果写入二进制文件（供 bench.py 对拍） ----------
// 格式：3 个 int32（M, N, K）+ M*N 个 float32（C，行主序）
void dump(const char *path, const float *c, int M, int N, int K) {
  FILE *f = fopen(path, "wb");
  if (!f) {
    fprintf(stderr, "cannot open %s\n", path);
    exit(EXIT_FAILURE);
  }
  fwrite(&M, sizeof(int), 1, f);
  fwrite(&N, sizeof(int), 1, f);
  fwrite(&K, sizeof(int), 1, f);
  fwrite(c, sizeof(float), (size_t)M * N, f);
  fclose(f);
  printf("已写入 kernel 结果: %s\n", path);
}

int main(int argc, char **argv) {
  // ---- 参数解析：./sgemm [M [N [K]]] [--dump out.bin] ----
  int M = 1024, N = 1024, K = 1024;
  const char *dump_path = nullptr;
  int pos = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (argv[i][0] >= '0' && argv[i][0] <= '9') {
      if (pos == 0) M = atoi(argv[i]);
      else if (pos == 1) N = atoi(argv[i]);
      else if (pos == 2) K = atoi(argv[i]);
      pos++;
    }
  }

  // 只给了一个数字参数时，M=N=K 都用它（见 README）
  if (pos == 1) {
    N = M;
    K = M;
  }

  cudaDeviceProp prop;
  CHECK(cudaGetDeviceProperties(&prop, 0));
  printf("Device: %s (sm_%d%d)\n", prop.name, prop.major, prop.minor);
  printf("SGEMM: M=%d N=%d K=%d\n", M, N, K);

  // ---- 主机端分配与初始化 ----
  size_t bytes_a = (size_t)M * K * sizeof(float);
  size_t bytes_b = (size_t)K * N * sizeof(float);
  size_t bytes_c = (size_t)M * N * sizeof(float);
  float *h_a = (float *)malloc(bytes_a);
  float *h_b = (float *)malloc(bytes_b);
  float *h_c = (float *)malloc(bytes_c);
  float *h_c2 = (float *)malloc(bytes_c);
  float *h_c3 = (float *)malloc(bytes_c);
  float *h_c4 = (float *)malloc(bytes_c);
  float *h_ref = (float *)malloc(bytes_c);
  float *h_cblas = (float *)malloc(bytes_c);
  if (!h_a || !h_b || !h_c || !h_c2 || !h_c3 || !h_c4 || !h_ref || !h_cblas) {
    fprintf(stderr, "host malloc failed\n");
    return EXIT_FAILURE;
  }
  init_matrices(h_a, h_b, M, N, K);

  // ---- 设备端分配 + H2D ----
  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc(&d_a, bytes_a));
  CHECK(cudaMalloc(&d_b, bytes_b));
  CHECK(cudaMalloc(&d_c, bytes_c));
  CHECK(cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice));

  // ---- 1. 你的 kernel：预热 + 计时 ----
  launch_sgemm_naive(d_a, d_b, d_c, M, N, K);  // 预热，同时暴露启动配置错误
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  float ms = time_ms([&] { launch_sgemm_naive(d_a, d_b, d_c, M, N, K); }, 3);
  double gflops = 2.0 * M * N * K / (ms * 1e6);
  printf("naive:  %10.3f ms  %10.2f GFLOPS\n", ms, gflops);

  // 注意：先把 kernel 结果拷回 host，后面 cuBLAS 会覆盖 d_c
  CHECK(cudaMemcpy(h_c, d_c, bytes_c, cudaMemcpyDeviceToHost));

  // ---- 1b. kernel 1：global memory 合并访问 ----
  launch_sgemm_global_memory(d_a, d_b, d_c, M, N, K);  // 预热，同时暴露启动配置错误
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  float ms_coalesced =
      time_ms([&] { launch_sgemm_global_memory(d_a, d_b, d_c, M, N, K); }, 3);
  double gflops_coalesced = 2.0 * M * N * K / (ms_coalesced * 1e6);
  printf("coalesced: %9.3f ms  %10.2f GFLOPS  (%.2fx naive)\n", ms_coalesced,
         gflops_coalesced, gflops_coalesced / gflops);
  CHECK(cudaMemcpy(h_c2, d_c, bytes_c, cudaMemcpyDeviceToHost));

  // ---- 1c. kernel 2：shared memory tiling ----
  launch_sgemm_shared_memory(d_a, d_b, d_c, M, N, K);  // 预热，同时暴露启动配置错误
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  float ms_smem =
      time_ms([&] { launch_sgemm_shared_memory(d_a, d_b, d_c, M, N, K); }, 3);
  double gflops_smem = 2.0 * M * N * K / (ms_smem * 1e6);
  printf("smem:     %9.3f ms  %10.2f GFLOPS  (%.2fx naive)\n", ms_smem,
         gflops_smem, gflops_smem / gflops);
  CHECK(cudaMemcpy(h_c3, d_c, bytes_c, cudaMemcpyDeviceToHost));

  // ---- 1d. kernel 3：1D blocktiling（寄存器分块） ----
  launch_sgemm_1d_blocktiling(d_a, d_b, d_c, M, N, K);  // 预热，同时暴露启动配置错误
  CHECK(cudaGetLastError());
  CHECK(cudaDeviceSynchronize());
  float ms_1d =
      time_ms([&] { launch_sgemm_1d_blocktiling(d_a, d_b, d_c, M, N, K); }, 3);
  double gflops_1d = 2.0 * M * N * K / (ms_1d * 1e6);
  printf("1dblock:  %9.3f ms  %10.2f GFLOPS  (%.2fx naive)\n", ms_1d,
         gflops_1d, gflops_1d / gflops);
  CHECK(cudaMemcpy(h_c4, d_c, bytes_c, cudaMemcpyDeviceToHost));

  // ---- 2. CPU 参考对拍（尺寸太大则跳过，避免 CPU 跑太久） ----
  if ((size_t)M * N * K <= (size_t)1 << 30) {
    sgemm_cpu(h_a, h_b, h_ref, M, N, K);
    float rel = 0.0f;
    int bad = check_against(h_c, h_ref, M * N, &rel);
    printf("CPU 对拍 (naive): %s (max rel err = %.2e)\n",
           bad == 0 ? "PASS" : "FAIL", rel);
    if (bad) {
      printf("  不一致的元素: %d\n", bad);
      return EXIT_FAILURE;
    }
    float rel_c = 0.0f;
    int bad_c = check_against(h_c2, h_ref, M * N, &rel_c);
    printf("CPU 对拍 (coalesced): %s (max rel err = %.2e)\n",
           bad_c == 0 ? "PASS" : "FAIL", rel_c);
    if (bad_c) {
      printf("  不一致的元素: %d\n", bad_c);
      return EXIT_FAILURE;
    }
    float rel_s = 0.0f;
    int bad_s = check_against(h_c3, h_ref, M * N, &rel_s);
    printf("CPU 对拍 (smem): %s (max rel err = %.2e)\n",
           bad_s == 0 ? "PASS" : "FAIL", rel_s);
    if (bad_s) {
      printf("  不一致的元素: %d\n", bad_s);
      return EXIT_FAILURE;
    }
    float rel_t = 0.0f;
    int bad_t = check_against(h_c4, h_ref, M * N, &rel_t);
    printf("CPU 对拍 (1dblock): %s (max rel err = %.2e)\n",
           bad_t == 0 ? "PASS" : "FAIL", rel_t);
    if (bad_t) {
      printf("  不一致的元素: %d\n", bad_t);
      return EXIT_FAILURE;
    }
  } else {
    printf("CPU 对拍: 跳过（尺寸太大）\n");
  }

  // ---- 3. cuBLAS 基线（对比目标） ----
  cublasHandle_t handle;
  CHECK_CUBLAS(cublasCreate(&handle));
  // 显式用默认 FP32 数学模式（TF32 是 opt-in，这里先不开）
  CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_DEFAULT_MATH));
  const float alpha = 1.0f, beta = 0.0f;
  // 行主序 C(M,N) = A(M,K) @ B(K,N) 转成列主序视角：C^T = B^T @ A^T
  // cublasSgemm 的矩阵按列主序存储，利用转置把行主序内存直接喂进去
  auto launch_cublas = [&] {
    CHECK_CUBLAS(cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K, &alpha,
                             d_b, N, d_a, K, &beta, d_c, N));
  };
  launch_cublas();  // 预热
  CHECK(cudaDeviceSynchronize());
  float ms_cublas = time_ms(launch_cublas, 3);
  double gflops_cublas = 2.0 * M * N * K / (ms_cublas * 1e6);
  printf("cublas: %10.3f ms  %10.2f GFLOPS  (naive = %.2f%% of cublas)\n",
         ms_cublas, gflops_cublas, 100.0 * gflops / gflops_cublas);

  // ---- 4. kernel vs cuBLAS 对拍（两种 GPU 实现互相对比） ----
  CHECK(cudaMemcpy(h_cblas, d_c, bytes_c, cudaMemcpyDeviceToHost));
  float rel2 = 0.0f;
  int bad2 = check_against(h_c, h_cblas, M * N, &rel2);
  printf("cuBLAS 对拍 (naive): %s (max rel err = %.2e)\n",
         bad2 == 0 ? "PASS" : "FAIL", rel2);
  if (bad2) {
    printf("  不一致的元素: %d\n", bad2);
    return EXIT_FAILURE;
  }
  float rel3 = 0.0f;
  int bad3 = check_against(h_c2, h_cblas, M * N, &rel3);
  printf("cuBLAS 对拍 (coalesced): %s (max rel err = %.2e)\n",
         bad3 == 0 ? "PASS" : "FAIL", rel3);
  if (bad3) {
    printf("  不一致的元素: %d\n", bad3);
    return EXIT_FAILURE;
  }
  float rel4 = 0.0f;
  int bad4 = check_against(h_c3, h_cblas, M * N, &rel4);
  printf("cuBLAS 对拍 (smem): %s (max rel err = %.2e)\n",
         bad4 == 0 ? "PASS" : "FAIL", rel4);
  if (bad4) {
    printf("  不一致的元素: %d\n", bad4);
    return EXIT_FAILURE;
  }
  float rel5 = 0.0f;
  int bad5 = check_against(h_c4, h_cblas, M * N, &rel5);
  printf("cuBLAS 对拍 (1dblock): %s (max rel err = %.2e)\n",
         bad5 == 0 ? "PASS" : "FAIL", rel5);
  if (bad5) {
    printf("  不一致的元素: %d\n", bad5);
    return EXIT_FAILURE;
  }

  // ---- 5. 可选 dump（h_c4 是当前最新 kernel 的结果，供 bench.py 对拍） ----
  if (dump_path) dump(dump_path, h_c4, M, N, K);

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
  free(h_a);
  free(h_b);
  free(h_c);
  free(h_c2);
  free(h_c3);
  free(h_c4);
  free(h_ref);
  free(h_cblas);
  return EXIT_SUCCESS;
}
