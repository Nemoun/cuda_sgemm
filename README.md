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
- [x] 6. 消除 bank conflict + smem 读向量化 —— 31× naive，收益有限（见性能记录）（`kernels/6_bank_conflict.cuh`）
- [x] 7. warp tiling —— 26× naive，A6000 配置在 4070 Laptop 上慢于 vectorized（见性能记录）（`kernels/7_warptiling.cuh`）
- [x] 8. double buffering（cp.async）—— 23.8× / 26.4× naive，重叠收益随 DRAM 延迟增大而显现（见性能记录）（`kernels/8_double_buffering.cuh`）
- [ ] 9. autotuning（扫 warp tiling 参数）+ roofline 图
- [ ] 10. Tensor Core（wmma）

## 性能记录

| kernel | 1024³ GFLOPS | 1024³ 耗时 | 4096³ GFLOPS | 4096³ 耗时 |
|---|---|---|---|---|
| naive | 268.04 | 8.012 ms | 306.19 | 448.868 ms |
| coalesced | 1217.86 | 1.763 ms | 1041.43 | 131.971 ms |
| smem | 1594.07 | 1.347 ms | 1433.75 | 95.860 ms |
| 1dblock | 4271.19 | 0.503 ms | 4603.43 | 29.856 ms |
| 2dblock | 5698.78 | 0.377 ms | 7137.34 | 19.256 ms |
| vectorized | 8104.94 | 0.265 ms | 8853.41 | 15.524 ms |
| nobank | 8435.00 | 0.255 ms | 8888.00 | 15.463 ms |
| warp | 6969.45 | 0.308 ms | 6484.66 | 21.194 ms |
| dbuf | 6373.72 | 0.337 ms | 8079.08 | 17.012 ms |
| cuBLAS | 10230.01 | 0.210 ms | 13875.37 | 9.905 ms |

同一次运行的加速比（相对 naive，1024³）：coalesced 4.5×、smem 5.9×、1dblock 15.9×、
2dblock 21.3×、vectorized 30.2×、nobank **31.5×**、warp 26.0×、dbuf 23.8×。
4096³ 依次为 3.4× / 4.7× / 15.0× / 23.3× / 28.9× / 29.0× / 21.2× / 26.4×。
4096³ 下 nobank 已达到 cuBLAS 的 64%。
4096³ 时 CPU 对拍会自动跳过（`main.cu` 中 M×N×K > 2^30 即跳过，CPU 三重循环太慢），只保留 cuBLAS 对拍。

ncu 数据（1024³）：naive 的瓶颈是 L1/TEX 被 32 个碎片请求/warp 打满（DRAM 仅 5.4%）；
合并访问后瓶颈转移到 Compute(SM) 98.3%——每 FMA 配 2 条 load；
smem 版把 load 搬进 shared memory，但 2:1 的 load:FMA 没变，瓶颈仍在访存发射；
1dblock 用 TM=8 个寄存器累加器把 load:FMA 压到约 1:1（Compute 72.5%）；
2dblock 用 TM×TN=64 个累加器压到 1:4（Compute 43.6%，L1/TEX 71% 成最忙部件）；
vectorized 用 float4 把搬运次数 ÷4（As 转置 scatter、Bs/C 直存），L1/TEX 降到 64.7%；
nobank 用 swizzle 布局（Bs4 按 float4 单位重排）+ 内积 LDS.128，实测**读冲突 0**，
但 Bs4 的散列存储带来 52 万次**存储冲突**，整体收益有限且不稳定
（1024³ +13%、4096³ −1.7%，ncu 下对比还会翻转）——与教程作者在 A6000 上的
经验一致，这也是文章跳过 bank conflict 章节直接讲 autotuning 的原因。
warp tiling（教程 kernel 10 的 A6000 配置：128 线程、128 个寄存器累加器）在
4070 Laptop 上**慢于 vectorized**：occupancy 只有 16.7%（寄存器限制 2 blocks/SM），
8 warps/SM 遮不住访存延迟，所有吞吐指标都 <50%（Compute 40.5%）——A6000 的
「重 ILP 轻 occupancy」配置在我们这块 46 SM 的笔记本芯片上水土不服。
double buffering（cp.async 双 barrier，教程 kernel 12 模式，BK=16）在
vectorized 底座上的重叠收益**随 DRAM 延迟增大而显现**：1024³ 同轮比
vectorized 慢 21%（L2 命中高、延迟短，重叠的净收益小），4096³ 只慢 9%
（8079 vs 8853——DRAM 延迟长，预取遮住了更多等待）。4B cp.async 的
指令开销、barrier 相位管理、双倍 smem 都是实打实的成本，教程的大幅
收益来自 warptiling 组合底座 + A6000 的硬件特性。
至此所有「教程主线」的优化手法都已实现并实测：真正的问题是——
**每种手法与硬件的匹配度不同**，这正是下一步 autotuning 要解决的：
扫参数（THREADS/BK/TM/TN/底座组合/流水级数），在 4070 Laptop 上找出
真正的最优配置。

## 参考

- How to Optimize a GEMM：https://siboehm.com/articles/21/CUDA.html
- SGEMM_CUDA：https://github.com/siboehm/SGEMM_CUDA
