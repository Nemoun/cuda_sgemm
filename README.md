# CUDA SGEMM 优化

从 naive GEMM 出发，逐步优化到共享内存分块、向量化、Tensor Core 的学习项目。

## 环境

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU（Ada，compute capability **89**）
- CUDA Toolkit 12.6
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
- [x] 1. global memory 合并访问 —— 5.2× naive（`kernels/1_global_memory.cuh`）
- [x] 2. shared memory tiling —— 7.3× naive（`kernels/2_shared_memory.cuh`）
- [x] 3. 1D blocktiling（寄存器分块）—— 19× naive（`kernels/3_1d_blocktiling.cuh`）
- [x] 4. 2D blocktiling（TM×TN 寄存器分块）—— 26.5× naive（`kernels/4_2d_blocktiling.cuh`）
- [ ] 5. float4 向量化搬运（顺带处理 bank conflict）
- [ ] 6. warp tiling / double buffering
- [ ] 7. Tensor Core（wmma）
- [ ] 8. autotuning + roofline 图

## 性能记录

| kernel | 1024³ GFLOPS | 1024³ 耗时 | 4096³ GFLOPS | 4096³ 耗时 |
|---|---|---|---|---|
| naive | 216.81 | 9.905 ms | 302.96 | 453.652 ms |
| coalesced | 1224.35 | 1.754 ms | 1044.89 | 131.534 ms |
| smem | 1592.37 | 1.349 ms | 1432.16 | 95.966 ms |
| 1dblock | 4246.86 | 0.506 ms | 4657.59 | 29.509 ms |
| 2dblock | 5745.62 | 0.374 ms | 7340.72 | 18.723 ms |
| cuBLAS | 10746.02 | 0.200 ms | 13094.41 | 10.496 ms |

同一次运行的加速比（相对 naive，1024³）：coalesced 5.7×、smem 7.3×、1dblock 19.6×、
2dblock **26.5×**。4096³ 依次为 3.5× / 4.7× / 15.4× / 24.2×。数值为最近一次运行的实测，
笔记本环境受功耗墙影响有波动，看同一次运行的比值更有意义。
4096³ 时 CPU 对拍会自动跳过（`main.cu` 中 M×N×K > 2^30 即跳过，CPU 三重循环太慢），只保留 cuBLAS 对拍。

ncu 数据（1024³）：naive 的瓶颈是 L1/TEX 被 32 个碎片请求/warp 打满（DRAM 仅 5.4%）；
合并访问后瓶颈转移到 Compute(SM) 98.3%——每 FMA 配 2 条 load；
smem 版把 load 搬进 shared memory，但 2:1 的 load:FMA 没变，瓶颈仍在访存发射；
1dblock 用 TM=8 个寄存器累加器把 load:FMA 压到约 1:1，GFLOPS 翻 2.7 倍（Compute 72.5%）；
2dblock 用 TM×TN=64 个累加器把 load:FMA 压到 1:4，Compute 降到 43.6%，
此时最忙的部件变成 L1/TEX 71%（内积每 dotIdx 16 次 LDS + Bs 行读取的 4-way bank conflict）；
occupancy 只有 28%（寄存器限制）但 ILP 补了回来，GFLOPS 仍大涨——
下一步 float4 向量化搬运 + 消除 bank conflict（教程 kernel 6/7）。

## 参考

- How to Optimize a GEMM：https://siboehm.com/articles/21/CUDA.html
- SGEMM_CUDA：https://github.com/siboehm/SGEMM_CUDA
