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
python3 bench.py 1024 --check c.bin        # numpy 对拍
```

每次运行会输出：kernel 与 cuBLAS 的耗时/GFLOPS/比值，以及 CPU 参考对拍和 cuBLAS 对拍结果。

## 目录结构

```
main.cu            # benchmark 入口：计时 + CPU 参考 + cuBLAS 基线 + 三方对拍
kernels/           # 每个优化版本一个 .cuh，main.cu 只改一行 include/调用
bench.py           # numpy 参考实现与对拍
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
- [x] 5. float4 向量化搬运 —— 29.6× naive（`kernels/5_vectorize.cuh`）
- [ ] 6. 消除 bank conflict + smem 读向量化
- [ ] 6. warp tiling / double buffering
- [ ] 7. Tensor Core（wmma）
- [ ] 8. autotuning + roofline 图

## 性能记录

| kernel | 1024³ GFLOPS | 1024³ 耗时 | 4096³ GFLOPS | 4096³ 耗时 |
|---|---|---|---|---|
| naive | 269.43 | 7.971 ms | 303.30 | 453.146 ms |
| coalesced | 1222.85 | 1.756 ms | 1067.80 | 128.712 ms |
| smem | 1593.62 | 1.348 ms | 1436.19 | 95.697 ms |
| 1dblock | 4144.57 | 0.518 ms | 4361.97 | 31.508 ms |
| 2dblock | 5777.28 | 0.372 ms | 7063.72 | 19.457 ms |
| vectorized | 7973.96 | 0.269 ms | 8447.74 | 16.269 ms |
| cuBLAS | 9939.11 | 0.216 ms | 13780.05 | 9.974 ms |

同一次运行的加速比（相对 naive，1024³）：coalesced 4.5×、smem 5.9×、1dblock 15.4×、
2dblock 21.4×、vectorized **29.6×**。4096³ 依次为 3.5× / 4.7× / 14.4× / 23.3× / 27.9×。
4096³ 下 vectorized 已达到 cuBLAS 的 61%。数值为最近一次运行的实测，
笔记本环境受功耗墙影响有波动，看同一次运行的比值更有意义。
4096³ 时 CPU 对拍会自动跳过（`main.cu` 中 M×N×K > 2^30 即跳过，CPU 三重循环太慢），只保留 cuBLAS 对拍。

ncu 数据（1024³）：naive 的瓶颈是 L1/TEX 被 32 个碎片请求/warp 打满（DRAM 仅 5.4%）；
合并访问后瓶颈转移到 Compute(SM) 98.3%——每 FMA 配 2 条 load；
smem 版把 load 搬进 shared memory，但 2:1 的 load:FMA 没变，瓶颈仍在访存发射；
1dblock 用 TM=8 个寄存器累加器把 load:FMA 压到约 1:1（Compute 72.5%）；
2dblock 用 TM×TN=64 个累加器压到 1:4（Compute 43.6%，L1/TEX 71% 成最忙部件）；
vectorized 用 float4 把搬运次数 ÷4（As 转置 scatter、Bs/C 直存），
L1/TEX 降到 64.7%、GFLOPS 再涨 38%、Compute 升到 50.1%、DRAM 20.7%；
occupancy 仍 33%（寄存器限制）。剩余大头：内积的标量 smem 读（含 Bs 行的
4-way bank conflict）——下一个里程碑：消除 bank conflict + smem 读向量化。

## 参考

- How to Optimize a GEMM：https://siboehm.com/articles/21/CUDA.html
- SGEMM_CUDA：https://github.com/siboehm/SGEMM_CUDA
