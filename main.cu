// ============================================================
// CUDA SGEMM benchmark 入口
// 职责：分配矩阵 → 启动 kernel（计时）→ CPU/cuBLAS 对拍 → 输出 GFLOPS
// 用法：./sgemm [M [N [K]]] [--kernel <名字|all>] [--dump out.bin]
//   --kernel naive | coalesced | smem | 1dblock | 2dblock | vectorized
//            | nobank | warp | dbuf | auto    （默认 all，全部按顺序跑）
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
#include "kernels/4_2d_blocktiling.cuh"
#include "kernels/5_vectorize.cuh"
#include "kernels/6_bank_conflict.cuh"
#include "kernels/7_warptiling.cuh"
#include "kernels/8_double_buffering.cuh"
#include "kernels/9_autotuned.cuh"
#include "kernels/10_tensorcore.cuh"

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

// ---------- 对拍：|x - ref| <= tol * (1 + |ref|) 即认为一致 ----------
// FFMA 类 kernel 用 1e-3（fp32 累加顺序差异 ~1e-4，余量足够）；
// TF32 Tensor Core kernel 用 5e-3（TF32 尾数只有 10 位，输入截断误差 ~1e-3）。
// 相对误差以 ref 的最大绝对值为基准（逐元素除会因 ref≈0 产生虚高的假象）
int check_against(const float *x, const float *ref, int n, float tol, float *max_rel_err) {
  float scale = 1e-6f;
  for (int i = 0; i < n; i++) scale = fmaxf(scale, fabsf(ref[i]));
  int bad = 0;
  float max_abs_err = 0.0f;
  for (int i = 0; i < n; i++) {
    float abs_err = fabsf(x[i] - ref[i]);
    max_abs_err = fmaxf(max_abs_err, abs_err);
    if (abs_err > tol * (1.0f + fabsf(ref[i]))) bad++;
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

// ---------------- kernel 注册表 ----------------
struct KernelEntry {
  const char *name;   // --kernel 参数名
  const char *label;  // 输出标签
  float *h_out;       // 结果 buffer
  float tol;          // 对拍阈值（FFMA 类 1e-3，TF32 Tensor Core 5e-3）
  bool ran = false;   // 本轮是否运行
  float ms = 0.0f;    // 最快耗时
};

int main(int argc, char **argv) {
  // ---- 参数解析：./sgemm [M [N [K]]] [--kernel name] [--dump out.bin] ----
  int M = 1024, N = 1024, K = 1024;
  const char *kernel_sel = "all";
  const char *dump_path = nullptr;
  int pos = 0;
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "--dump") == 0 && i + 1 < argc) {
      dump_path = argv[++i];
    } else if (strcmp(argv[i], "--kernel") == 0 && i + 1 < argc) {
      kernel_sel = argv[++i];
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
  float *h_bufs[11];  // 每个 kernel 一个结果 buffer
  for (int i = 0; i < 11; i++) h_bufs[i] = (float *)malloc(bytes_c);
  float *h_ref = (float *)malloc(bytes_c);
  float *h_cblas = (float *)malloc(bytes_c);
  bool alloc_ok = h_a && h_b && h_ref && h_cblas;
  for (int i = 0; i < 11; i++) alloc_ok = alloc_ok && h_bufs[i];
  if (!alloc_ok) {
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

  // ---- kernel 注册表（顺序即默认执行顺序） ----
  KernelEntry entries[] = {
      {"naive", "naive:", h_bufs[0], 1e-3f},
      {"coalesced", "coalesced:", h_bufs[1], 1e-3f},
      {"smem", "smem:", h_bufs[2], 1e-3f},
      {"1dblock", "1dblock:", h_bufs[3], 1e-3f},
      {"2dblock", "2dblock:", h_bufs[4], 1e-3f},
      {"vectorized", "vectorized:", h_bufs[5], 1e-3f},
      {"nobank", "nobank:", h_bufs[6], 1e-3f},
      {"warp", "warp:", h_bufs[7], 1e-3f},
      {"dbuf", "dbuf:", h_bufs[8], 1e-3f},
      {"auto", "auto:", h_bufs[9], 1e-3f},
      {"tc", "tc:", h_bufs[10], 5e-3f},  // TF32：输入截断误差 ~1e-3，放宽阈值
  };
  const int n_entries = (int)(sizeof(entries) / sizeof(entries[0]));

  // ---- 运行某个 kernel：预热 + 计时，结果拷回 host ----
  auto run_entry = [&](KernelEntry &e) {
    if (strcmp(e.name, "naive") == 0) {
      launch_sgemm_naive(d_a, d_b, d_c, M, N, K);  // 预热，同时暴露启动配置错误
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_naive(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "coalesced") == 0) {
      launch_sgemm_global_memory(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_global_memory(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "smem") == 0) {
      launch_sgemm_shared_memory(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_shared_memory(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "1dblock") == 0) {
      launch_sgemm_1d_blocktiling(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_1d_blocktiling(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "2dblock") == 0) {
      launch_sgemm_2d_blocktiling(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_2d_blocktiling(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "vectorized") == 0) {
      launch_sgemm_vectorize(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_vectorize(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "nobank") == 0) {
      launch_sgemm_no_bank_conflict(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_no_bank_conflict(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "warp") == 0) {
      launch_sgemm_warptiling(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_warptiling(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "dbuf") == 0) {
      launch_sgemm_double_buffering(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_double_buffering(d_a, d_b, d_c, M, N, K); }, 3);
    } else if (strcmp(e.name, "auto") == 0) {
      // 内部完成配置扫描 + 最优重跑（结果已留在 d_c）
      e.ms = launch_sgemm_autotuned(d_a, d_b, d_c, M, N, K);
    } else if (strcmp(e.name, "tc") == 0) {
      launch_sgemm_tensorcore(d_a, d_b, d_c, M, N, K);
      CHECK(cudaGetLastError());
      CHECK(cudaDeviceSynchronize());
      e.ms = time_ms([&] { launch_sgemm_tensorcore(d_a, d_b, d_c, M, N, K); }, 3);
    } else {
      fprintf(stderr, "未知 kernel: %s\n", e.name);
      exit(EXIT_FAILURE);
    }
    CHECK(cudaMemcpy(e.h_out, d_c, bytes_c, cudaMemcpyDeviceToHost));
    e.ran = true;
  };

  // ---- 执行所选 kernel ----
  if (strcmp(kernel_sel, "all") == 0) {
    for (int i = 0; i < n_entries; i++) run_entry(entries[i]);
  } else {
    bool found = false;
    for (int i = 0; i < n_entries; i++)
      if (strcmp(entries[i].name, kernel_sel) == 0) {
        run_entry(entries[i]);
        found = true;
        break;
      }
    if (!found) {
      fprintf(stderr, "未知 kernel: %s\n", kernel_sel);
      return EXIT_FAILURE;
    }
  }

  // ---- 打印结果（比值以 naive 为基准，若本轮没跑 naive 则不打印比值） ----
  const double flops = 2.0 * (double)M * N * K;
  double naive_gf = 0.0;
  bool naive_ran = false;
  float *last_out = nullptr;
  for (int i = 0; i < n_entries; i++) {
    KernelEntry &e = entries[i];
    if (!e.ran) continue;
    double gf = flops / (e.ms * 1e6);
    last_out = e.h_out;
    if (strcmp(e.name, "naive") == 0) {
      naive_gf = gf;
      naive_ran = true;
      printf("%-10s %9.3f ms  %10.2f GFLOPS\n", e.label, e.ms, gf);
    } else if (naive_ran) {
      printf("%-10s %9.3f ms  %10.2f GFLOPS  (%.2fx naive)\n", e.label, e.ms,
             gf, gf / naive_gf);
    } else {
      printf("%-10s %9.3f ms  %10.2f GFLOPS\n", e.label, e.ms, gf);
    }
  }

  // ---- CPU 参考对拍（只查本轮跑过的 kernel；尺寸太大则跳过） ----
  if ((size_t)M * N * K <= (size_t)1 << 30) {
    sgemm_cpu(h_a, h_b, h_ref, M, N, K);
    for (int i = 0; i < n_entries; i++) {
      KernelEntry &e = entries[i];
      if (!e.ran) continue;
      float rel = 0.0f;
      int bad = check_against(e.h_out, h_ref, M * N, e.tol, &rel);
      printf("CPU 对拍 (%s): %s (max rel err = %.2e)\n", e.name,
             bad == 0 ? "PASS" : "FAIL", rel);
      if (bad) {
        printf("  不一致的元素: %d\n", bad);
        return EXIT_FAILURE;
      }
    }
  } else {
    printf("CPU 对拍: 跳过（尺寸太大）\n");
  }

  // ---- cuBLAS 基线（对比目标） ----
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
  double gflops_cublas = flops / (ms_cublas * 1e6);
  if (naive_ran) {
    printf("cublas: %10.3f ms  %10.2f GFLOPS  (naive = %.2f%% of cublas)\n",
           ms_cublas, gflops_cublas, 100.0 * naive_gf / gflops_cublas);
  } else {
    printf("cublas: %10.3f ms  %10.2f GFLOPS\n", ms_cublas, gflops_cublas);
  }

  // ---- cuBLAS 对拍（只查本轮跑过的 kernel） ----
  CHECK(cudaMemcpy(h_cblas, d_c, bytes_c, cudaMemcpyDeviceToHost));
  for (int i = 0; i < n_entries; i++) {
    KernelEntry &e = entries[i];
    if (!e.ran) continue;
    float rel = 0.0f;
    int bad = check_against(e.h_out, h_cblas, M * N, e.tol, &rel);
    printf("cuBLAS 对拍 (%s): %s (max rel err = %.2e)\n", e.name,
           bad == 0 ? "PASS" : "FAIL", rel);
    if (bad) {
      printf("  不一致的元素: %d\n", bad);
      return EXIT_FAILURE;
    }
  }

  // ---- 可选 dump（导出最后跑的那个 kernel 的结果，供 bench.py 对拍） ----
  if (dump_path) dump(dump_path, last_out, M, N, K);

  CHECK_CUBLAS(cublasDestroy(handle));
  CHECK(cudaFree(d_a));
  CHECK(cudaFree(d_b));
  CHECK(cudaFree(d_c));
  free(h_a);
  free(h_b);
  for (int i = 0; i < 11; i++) free(h_bufs[i]);
  free(h_ref);
  free(h_cblas);
  return EXIT_SUCCESS;
}
