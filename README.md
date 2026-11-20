# CUDA SGEMM 优化

从 naive GEMM 出发，逐步优化到共享内存分块、向量化、Tensor Core 的学习项目。

## 环境

- GPU：NVIDIA GeForce RTX 4070 Laptop GPU（Ada，compute capability **89**）
- CUDA Toolkit 12.6
- 编译命令见 `build.sh`

## 快速开始

```bash
bash build.sh                              # 编译
./sgemm                                    # 默认 1024×1024×1024，全部 kernel 按顺序跑
./sgemm 512                                # 自定义尺寸 M=N=K=512
./sgemm 512 512 1024                       # M N K 分别指定
./sgemm --kernel nobank 1024               # 只跑某个 kernel（同样计时 + 三方对拍）
./sgemm --kernel auto 4096                 # 只跑 autotuning 扫描
./sgemm 1024 --dump c.bin                  # 把（最后跑的）kernel 结果写到二进制文件
python3 bench.py 1024 --check c.bin        # numpy 对拍
```

`--kernel` 可选名字：`naive`、`coalesced`、`smem`、`1dblock`、`2dblock`、
`vectorized`、`nobank`、`warp`、`dbuf`、`auto`、`tc`（默认 `all`，全部按顺序跑）。

每次运行会输出：kernel 与 cuBLAS 的耗时/GFLOPS/比值，以及 CPU 参考对拍和 cuBLAS 对拍结果（对拍只覆盖本轮跑过的 kernel）。

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
- [x] 9. autotuning（一期几何扫描 → 二期「几何 × swizzle × dbuf」三维扫描）—— 32.4× / 33.5× naive（`kernels/9_autotuned.cuh`）
- [ ] 10. roofline 图 + 性能曲线绘图（bench.py）
- [x] 11. Tensor Core v1（wmma TF32，fragment 直载 global）—— 正确性 ✓（max rel err 仅 1.85e-4），速度待 v2 释放（`kernels/10_tensorcore.cuh`）
- [ ] 12. Tensor Core v2（smem tiling）/ v3（cp.async 流水 + 3xTF32 精度恢复）

## 性能记录

| kernel | 1024³ GFLOPS | 1024³ 耗时 | 4096³ GFLOPS | 4096³ 耗时 |
|---|---|---|---|---|
| naive | 269.21 | 7.977 ms | 305.79 | 449.452 ms |
| coalesced | 1220.87 | 1.759 ms | 1048.86 | 131.037 ms |
| smem | 1595.21 | 1.346 ms | 1443.80 | 95.193 ms |
| 1dblock | 4271.19 | 0.503 ms | 4684.41 | 29.340 ms |
| 2dblock | 5729.43 | 0.375 ms | 7406.00 | 18.558 ms |
| vectorized | 8004.40 | 0.268 ms | 8824.92 | 15.574 ms |
| nobank | 8388.61 | 0.256 ms | 9243.01 | 14.870 ms |
| warp | 6990.51 | 0.307 ms | 6100.00 | 22.531 ms |
| dbuf | 6413.31 | 0.335 ms | 8057.74 | 17.057 ms |
| autotuned | 8739.27 | 0.246 ms | 10074.89 | 13.642 ms |
| tc | 5045.40 | 0.426 ms | 2684.52 | 51.197 ms |
| cuBLAS | 10180.35 | 0.211 ms | 13856.88 | 9.918 ms |

### 加速比速览（同一次运行，相对 naive）

- 1024³：coalesced 4.5× → smem 5.9× → 1dblock 15.9× → 2dblock 21.3× → vectorized 29.7× → nobank 31.2× → **autotuned 32.8×**
- 4096³：coalesced 3.4× → smem 4.7× → 1dblock 15.3× → 2dblock 24.2× → vectorized 28.9× → nobank 30.2× → **autotuned 36.4×**
- 4096³ 下 autotuned 达到 cuBLAS 的 **80%**
- 说明：数值为最近一次运行的实测，有 ±5% 波动，看同一次运行的比值更有意义；4096³ 时 CPU 对拍自动跳过（`main.cu` 中 M×N×K > 2^30，CPU 三重循环太慢），只保留 cuBLAS 对拍

### 优化历程与 ncu 诊断（1024³）

1. **naive**：L1/TEX 被 32 个碎片请求/warp 打满（DRAM 仅 5.4%）
2. **coalesced**：请求合并后，瓶颈转移到 Compute(SM) 98.3%——每 FMA 配 2 条 load
3. **smem**：load 搬进 shared memory，但 2:1 的 load:FMA 没变，瓶颈仍在访存发射
4. **1dblock**：TM=8 个寄存器累加器把 load:FMA 压到约 1:1（Compute 72.5%）
5. **2dblock**：TM×TN=64 个累加器压到 1:4（Compute 43.6%，L1/TEX 71% 成最忙部件）
6. **vectorized**：float4 把搬运次数 ÷4，L1/TEX 降到 64.7%
7. **nobank**：swizzle 布局 + 内积 LDS.128，实测读冲突 0，但 Bs4 散列存储带来 52 万次存储冲突——整体收益有限且不稳定（1024³ +13%、4096³ −1.7%）。与教程作者在 A6000 上的经验一致，这也是文章跳过 bank conflict 章节直接讲 autotuning 的原因
8. **warp tiling**：教程 A6000 配置（128 线程、128 累加器）在 4070 Laptop 上慢于 vectorized——occupancy 仅 16.7%，8 warps/SM 遮不住访存延迟，「重 ILP 轻 occupancy」水土不服
9. **double buffering**：cp.async 双 barrier（BK=16）的重叠收益随 DRAM 延迟增大而显现——1024³ 慢 21%（L2 命中高、重叠净收益小），4096³ 只慢 9%
10. **tensorcore v1**（wmma TF32，fragment 直载 global）：正确跑通，实测 max rel err 仅 1.85e-4（远好于预估的 1e-3 量级——确定性小数值数据分布温和）；速度 5045（fragment 从 global 载入、算术强度仅 8 FLOP/B，受载入带宽限制）

### autotuning 扫描发现（一期 11 个几何配置 + 二期 40 个三维配置）

- **最优配置随尺寸变化**：一期 1024³ 选 (128,128,16,8,8)、4096³ 选 (64,128,16,8,8)；
  二期 1024³ 选 plain sync (128,128,32,8,8)，4096³ 选 swizzle sync (128,128,16,8,8)
- **swizzle 的收益随尺寸显现**：1024³ plain 占优，4096³ swizzle 反超 ~6%
  （读冲突的代价在 DRAM 延迟长的尺寸下更贵）
- **dbuf 组合在两个尺寸全部落败**：cp.async 双缓冲在这台机器上无净收益，
  结论收敛——多级流水大概率同样落空，不再投入
- BK=16/32 是甜点、TM=4 是坑；同一个 kernel 不同 tile 尺寸性能差近 2 倍
- 经验：冠军配置在 plain/swizzle 的 BK=16/32 之间随运行漂移（±5% 噪声）——
  扫描结果只能当参考，别迷信单次测量的「最优」

### 总结与后续

最优配置是 (几何, 布局, 搬运方式) 与尺寸、硬件的联合函数，运行时扫描比手工挑参数可靠。FFMA 路径的搜索空间已基本收敛（dbuf/多级流水路线在 4070 Laptop 上无净收益）。Tensor Core v1 已跑通，后续方向：

- Tensor Core v2（smem tiling，fragment 从 smem 载入）+ v3（cp.async 流水——多级流水在 TC 语境下复活；3xTF32 精度恢复）
- roofline 图 + 性能曲线（bench.py 绘图）——收尾产出

## 参考

- How to Optimize a GEMM：https://siboehm.com/articles/21/CUDA.html
- SGEMM_CUDA：https://github.com/siboehm/SGEMM_CUDA
