# CUDA SGEMM 优化

从 naive GEMM 出发，逐步优化到共享内存分块、向量化、Tensor Core 的学习项目。

## 环境

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU（Ada，compute capability **89**）
- CUDA Toolkit 12.6（WSL2 Ubuntu 24.04，仓库与开发均在 WSL 内）
- 编译命令见 `build.sh`

## 快速开始

```bash
bash build.sh                              # 编译
./sgemm                                    # 默认 1024×1024×1024
./sgemm 512                                # 自定义尺寸 M=N=K=512
./sgemm 512 512 1024                       # M N K 分别指定
./sgemm 1024 --dump c.bin                  # 把 kernel 结果写到二进制文件
python3 bench.py 1024 --check c.bin        # numpy 对拍（需在 WSL 里装 numpy）
```

每次运行会输出：kernel 与 cuBLAS 的耗时/GFLOPS/比值，以及 CPU 参考对拍和 cuBLAS 对拍结果。

## 目录结构

```
main.cu            # benchmark 入口：计时 + CPU 参考 + cuBLAS 基线 + 三方对拍
kernels/           # 每个优化版本一个 .cuh，main.cu 只改一行 include/调用
bench.py           # numpy 参考实现与对拍（后续性能曲线绘图也在这里加）
build.sh           # 编译脚本（nvcc -arch=sm_89）
vector_add.cu      # 环境验证 hello world
```

## 对拍原理

A、B 用确定性公式初始化（`main.cu` 与 `bench.py` 完全一致，不用随机数，可复现），
同一结果做三方对拍：CPU 三重循环参考、numpy matmul、cuBLAS sgemm。
float32 累加顺序不同会带来 ~1e-4 量级相对误差，阈值取 1e-3。
实测 max rel err：CPU 对拍 4.77e-07、cuBLAS 对拍 1.62e-06。

## 任务

- [x] 环境跑通（vector_add）
- [x] 0. naive GEMM —— 三重循环，起点 219.64 GFLOPS（cuBLAS 的 2.16%）（`kernels/naive.cuh`）
- [ ] 1. global memory 合并访问
- [ ] 2. shared memory tiling
- [ ] 3. blocktiling + float4 向量化
- [ ] 4. 消除 bank conflict（padding / swizzle）
- [ ] 5. warp tiling / double buffering
- [ ] 6. Tensor Core（wmma）
- [ ] 7. autotuning + Nsight Compute 分析 + roofline 图

## 性能记录

| kernel | 1024³ GFLOPS | 1024³ 耗时 | 4096³ GFLOPS | 4096³ 耗时 |
|---|---|---|---|---|
| naive | 219.64 | 9.777 ms | 299.05 | 459.578 ms |
| cuBLAS | 10180.35 | 0.211 ms | 13832.60 | 9.936 ms |

两种尺寸下 naive 均为 cuBLAS 的约 2.16%（cuBLAS 快约 46 倍）。
4096³ 时 CPU 对拍会自动跳过（`main.cu` 中 M×N×K > 2^30 即跳过，CPU 三重循环太慢），只保留 cuBLAS 对拍。

## 参考

- How to Optimize a GEMM：https://siboehm.com/articles/21/CUDA.html
- SGEMM_CUDA：https://github.com/siboehm/SGEMM_CUDA
