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
- [x] 1. global memory 合并访问 —— 5.2× naive（`kernels/1_global_memory.cuh`）
- [x] 2. shared memory tiling —— 7.3× naive（`kernels/2_shared_memory.cuh`）
- [x] 3. 1D blocktiling（寄存器分块）—— 19× naive（`kernels/3_1d_blocktiling.cuh`）
- [ ] 4. 2D blocktiling（TM×TN 寄存器分块）
- [ ] 5. float4 向量化搬运（顺带处理 bank conflict）
- [ ] 6. warp tiling / double buffering
- [ ] 7. Tensor Core（wmma）
- [ ] 8. autotuning + roofline 图

## 性能记录

| kernel | 1024³ GFLOPS | 1024³ 耗时 | 4096³ GFLOPS | 4096³ 耗时 |
|---|---|---|---|---|
| naive | 223.77 | 9.597 ms | 300.31 | 457.665 ms |
| coalesced | 1225.69 | 1.752 ms | 1061.18 | 129.515 ms |
| smem | 1597.22 | 1.345 ms | 1445.63 | 95.072 ms |
| 1dblock | 4245.25 | 0.506 ms | 4558.27 | 30.152 ms |
| cuBLAS | 10038.72 | 0.214 ms | 13797.05 | 9.961 ms |

同一次运行的加速比（相对 naive，1024³）：coalesced 5.5×、smem 7.1×、1dblock **19.0×**。
4096³ 依次为 3.5× / 4.8× / 15.2×。数值为最近一次运行的实测，
笔记本环境受功耗墙影响有波动，看同一次运行的比值更有意义。
4096³ 时 CPU 对拍会自动跳过（`main.cu` 中 M×N×K > 2^30 即跳过，CPU 三重循环太慢），只保留 cuBLAS 对拍。

ncu 数据（1024³）：naive 的瓶颈是 L1/TEX 被 32 个碎片请求/warp 打满（DRAM 仅 5.4%）；
合并访问后瓶颈转移到 Compute(SM) 98.3%——每 FMA 配 2 条 load；
smem 版把 load 搬进 shared memory，但 2:1 的 load:FMA 没变，瓶颈仍在访存发射；
1dblock 用 TM=8 个寄存器累加器把 load:FMA 压到约 1:1，GFLOPS 翻 2.7 倍，
此时 Compute 72.5% / DRAM 29% / L1/TEX 78.7%，没有单一瓶颈打满；
occupancy 62%（受寄存器数量限制，2 blocks/SM），ncu 提示仍有 ~33% 提升空间——
下一步 2D blocktiling（TM×TN）继续摊薄 load，float4 向量化加快搬运。

## 参考

- How to Optimize a GEMM：https://siboehm.com/articles/21/CUDA.html
- SGEMM_CUDA：https://github.com/siboehm/SGEMM_CUDA
