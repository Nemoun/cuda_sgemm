#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 错误检查宏：cudaMalloc / cudaMemcpy 等 API 调用都包上，出错直接报行号退出
#define CHECK(call)                                                            \
  do {                                                                         \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
      fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,         \
              cudaGetErrorString(err));                                        \
      exit(EXIT_FAILURE);                                                      \
    }                                                                          \
  } while (0)

__global__ void add(float *a, float *b, float *c, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}

int main() {
  const int n = 1 << 20;  // 1M 个元素，超出单 block 线程数，验证 grid 索引正确

  // 1. 主机端分配并初始化
  float *h_a = (float *)malloc(n * sizeof(float));
  float *h_b = (float *)malloc(n * sizeof(float));
  float *h_c = (float *)malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) {
    h_a[i] = 1.0f;
    h_b[i] = 2.0f;
  }

  // 2. 设备端分配
  float *d_a, *d_b, *d_c;
  CHECK(cudaMalloc(&d_a, n * sizeof(float)));
  CHECK(cudaMalloc(&d_b, n * sizeof(float)));
  CHECK(cudaMalloc(&d_c, n * sizeof(float)));

  // 3. 主机 → 设备
  CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

  // 4. 启动 kernel
  const int blockSize = 256;
  const int gridSize = (n + blockSize - 1) / blockSize;
  add<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
  CHECK(cudaGetLastError());        // kernel 启动错误（如非法 grid/block 配置）
  CHECK(cudaDeviceSynchronize());   // 等 kernel 跑完

  // 5. 设备 → 主机
  CHECK(cudaMemcpy(h_c, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

  // 6. 核对：1.0 + 2.0 = 3.0（IEEE float 下精确相等）
  int errors = 0;
  for (int i = 0; i < n; i++) {
    if (h_c[i] != 3.0f) errors++;
  }
  printf("%s: %d/%d elements correct\n", errors == 0 ? "PASS" : "FAIL",
         n - errors, n);

  cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
  free(h_a); free(h_b); free(h_c);
  return errors == 0 ? 0 : 1;
}
