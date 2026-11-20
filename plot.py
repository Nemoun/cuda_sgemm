#!/usr/bin/env python3
# ============================================================
# 性能曲线 + roofline 图
# 用法：python3 plot.py [--no-run] [--sizes 1024,4096]
#   - 默认运行 ./sgemm <size> 收集数据（缓存到 results.json）
#   - --no-run 用缓存的 results.json 直接画图
# 输出：performance.png（优化历程柱状图）、roofline.png（4096³ roofline）
#
# 硬件参数（RTX 4070 Laptop GPU，实测）：
#   - SM 频率 ~2.07 GHz（ncu 实测），36 SMs × 128 FMA × 2 FLOP
#     → fp32 计算顶 ≈ 19.1 TFLOPS
#   - 显存 8001 MHz × 256-bit（nvidia-smi 实测）→ 带宽 ≈ 512 GB/s
#   - TF32 Tensor Core 顶 ≈ 36 SM × 4 TC × 512 FLOP × 2.07 GHz ≈ 152.6 TFLOPS
#   - ridge ≈ 19.1e12 / 512e9 ≈ 37.3 FLOP/byte
#
# 算术强度（AI，DRAM 流量口径的近似）：
#   AI ≈ 2MNK / (4MK·N/BN + 4KN·M/BM + 4MN)，由 tile 尺寸 (BM,BN) 决定；
#   cuBLAS 内部 tile 未知，按 256×128 假设。
# ============================================================
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---- 设计系统参数（dataviz skill 参考调色板，已通过 validator 校验） ----
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"
AXIS = "#c3c2b7"
C_BLUE = "#2a78d6"    # categorical slot 1
C_ORANGE = "#eb6834"  # categorical slot 2
C_AQUA = "#1baf7a"    # categorical slot 3（浅色面下 2.74:1，需直接标签）

plt.rcParams.update({
    "font.family": "sans-serif",
    "text.color": INK_PRIMARY,
    "axes.edgecolor": AXIS,
    "axes.labelcolor": INK_SECONDARY,
    "xtick.color": INK_MUTED,
    "ytick.color": INK_MUTED,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "grid.color": GRIDLINE,
    "grid.linewidth": 0.8,
    "legend.frameon": False,
})

# ---- kernel 元数据：journey 顺序 + AI（FLOP/byte，4096³ 口径） ----
# (label, display, series, AI@4096³)
KERNELS = [
    ("naive", "naive", "ffma", 0.25),
    ("coalesced", "coalesced", "ffma", 0.25),
    ("smem", "smem", "ffma", 4.0),
    ("1dblock", "1dblock", "ffma", 16.0),
    ("2dblock", "2dblock", "ffma", 32.0),
    ("vectorized", "vectorized", "ffma", 32.0),
    ("nobank", "nobank", "ffma", 32.0),
    ("warp", "warp", "ffma", 32.0),
    ("dbuf", "dbuf", "ffma", 32.0),
    ("auto", "auto", "ffma", 21.3),   # 4096³ 冠军 (64×128) 口径
    ("tc", "tc", "tc", 16.0),
    ("tc2", "tc2", "tc", 16.0),
    ("tc3", "tc3", "tc", 16.0),
    ("tc4", "tc4", "tc", 32.0),
    ("tcauto", "tcauto", "tc", 32.0),
    ("cublas", "cuBLAS", "cublas", 42.7),  # 内部 tile 未知，按 256×128 假设
]

# ---- 硬件参数 ----
BW_GBS = 512.0              # GB/s
FP32_PEAK_GFLOPS = 19080.0  # ≈ 19.1 TFLOPS @ 2.07 GHz
TF32_PEAK_GFLOPS = 152600.0 # ≈ 152.6 TFLOPS @ 2.07 GHz
RIDGE = FP32_PEAK_GFLOPS / BW_GBS


def run_harness(sizes):
    """运行 ./sgemm 并解析 label: ms GFLOPS 行，返回 {label: {size: gflops}}"""
    results = {}
    for size in sizes:
        out = subprocess.run(["./sgemm", str(size)], capture_output=True, text=True)
        pat = re.compile(r"^(\w+):\s+([\d.]+) ms\s+([\d.]+) GFLOPS")
        for line in out.stdout.splitlines():
            m = pat.match(line)
            if m:
                label, ms, gf = m.group(1), float(m.group(2)), float(m.group(3))
                if label not in results:
                    results[label] = {}
                results[label][str(size)] = gf
        if len(results) == 0:
            print(out.stdout[-2000:])
            print(out.stderr[-2000:], file=sys.stderr)
            sys.exit("harness 输出解析失败")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-run", action="store_true", help="用 results.json 缓存画图")
    ap.add_argument("--sizes", default="1024,4096", help="逗号分隔的矩阵尺寸")
    args = ap.parse_args()
    sizes = [int(s) for s in args.sizes.split(",")]

    if args.no_run and Path("results.json").exists():
        results = json.load(open("results.json"))
    else:
        results = run_harness(sizes)
        json.dump(results, open("results.json", "w"), indent=2)

    # 缺失数据的 kernel 跳过并提示
    present = [k for k, _, _, _ in KERNELS if all(str(s) in results.get(k, {}) for s in sizes)]
    if len(present) < len(KERNELS):
        print("跳过缺失数据的 kernel:", [k for k, *_ in KERNELS if k not in present])

    # ================= 图 1：优化历程（分组柱状，log-y） =================
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=150)
    x = range(len(present))
    width = 0.36
    for si, size in enumerate(sizes):
        vals = [results[k][str(size)] for k in present]
        color = C_BLUE if si == 0 else C_ORANGE
        bars = ax.bar([i + (si - 0.5) * width + width / 2 for i in x], vals,
                      width=width, color=color, label=f"{size}³")
        # 端点标注：只标最后一个 kernel（tcauto）
        if si == len(sizes) - 1:
            last = bars[-1]
            ax.annotate(f"{last.get_height():,.0f}",
                        (last.get_x() + last.get_width() / 2, last.get_height()),
                        textcoords="offset points", xytext=(0, 5), ha="center",
                        fontsize=9, color=INK_SECONDARY)
    # cuBLAS 参考线（每尺寸一条虚线，颜色跟随系列）
    for si, size in enumerate(sizes):
        if "cublas" in results and str(size) in results["cublas"]:
            v = results["cublas"][str(size)]
            color = C_BLUE if si == 0 else C_ORANGE
            ax.axhline(v, color=color, linestyle="--", linewidth=1.2, alpha=0.7)
            ax.annotate(f"cuBLAS {size}³  {v:,.0f}", (len(present) - 0.55, v),
                        textcoords="offset points", xytext=(0, 4), ha="right",
                        fontsize=9, color=INK_SECONDARY)
    ax.set_yscale("log")
    labels = [next(d for k, d, _, _ in KERNELS if k == pk) for pk in present]
    ax.set_xticks(list(x), labels, rotation=30, ha="right")
    ax.set_ylabel("GFLOPS (log)")
    ax.set_title("CUDA SGEMM optimization journey — RTX 4070 Laptop (WSL2, CUDA 12.6)", loc="left",
                 fontsize=13, color=INK_PRIMARY)
    ax.grid(axis="y", which="major", linewidth=0.8)
    ax.grid(axis="y", which="minor", linewidth=0.4)
    ax.legend(loc="lower right", ncol=2, fontsize=10)
    fig.tight_layout()
    fig.savefig("performance.png", facecolor=SURFACE)
    plt.close(fig)

    # ================= 图 2：roofline（4096³） =================
    size = sizes[-1]
    fig, ax = plt.subplots(figsize=(9, 7.5), dpi=150)

    # 屋顶线（退居次要：灰色系）
    ai = [0.1, 1000.0]
    ax.plot(ai, [BW_GBS * a for a in ai], color=INK_MUTED, linestyle="--", linewidth=1.2,
            label=f"DRAM bandwidth {BW_GBS:.0f} GB/s")
    ax.axhline(FP32_PEAK_GFLOPS, color=INK_MUTED, linestyle="-", linewidth=1.2,
               label=f"fp32 compute peak {FP32_PEAK_GFLOPS/1000:.1f} TFLOPS")
    ax.axhline(TF32_PEAK_GFLOPS, color=INK_MUTED, linestyle=":", linewidth=1.2,
               label=f"TF32 Tensor Core peak {TF32_PEAK_GFLOPS/1000:.0f} TFLOPS")
    ax.axvline(RIDGE, color=AXIS, linestyle=":", linewidth=1.0)
    ax.annotate(f"ridge ≈ {RIDGE:.0f} FLOP/B", (RIDGE * 1.05, 300),
                fontsize=9, color=INK_MUTED)

    # 数据点（按系列着色 + 形状区分）
    styles = {
        "ffma": (C_BLUE, "o", "FFMA path"),
        "tc": (C_ORANGE, "^", "Tensor Core path"),
        "cublas": (C_AQUA, "D", "cuBLAS fp32"),
    }
    label_done = set()
    for k, disp, series, ai_v in KERNELS:
        if k not in present or str(size) not in results[k]:
            continue
        gf = results[k][str(size)]
        color, marker, name = styles[series]
        lbl = name if name not in label_done else None
        label_done.add(name)
        ax.scatter(ai_v, gf, s=70, color=color, marker=marker, label=lbl,
                   edgecolors=SURFACE, linewidths=1.8, zorder=3)
        # 直接标签：只标关键点（aqua 低对比度必须标）
        if k in ("naive", "tc", "auto", "tcauto", "cublas"):
            dy = 1.45 if k == "tcauto" else 1.3
            ax.annotate(disp, (ai_v, gf), textcoords="offset points",
                        xytext=(6, 6), fontsize=9, color=INK_SECONDARY)

    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(0.1, 300)
    ax.set_ylim(100, 3e5)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte, DRAM-traffic basis)")
    ax.set_ylabel("GFLOPS (log)")
    ax.set_title(f"Roofline — {size}³, RTX 4070 Laptop", loc="left",
                 fontsize=13, color=INK_PRIMARY)
    ax.grid(axis="both", which="major", linewidth=0.8)
    ax.grid(axis="both", which="minor", linewidth=0.4)
    ax.legend(loc="upper left", fontsize=10)
    fig.tight_layout()
    fig.savefig("roofline.png", facecolor=SURFACE)
    plt.close(fig)

    # 统一转成 RGB（matplotlib 输出 RGBA，部分看图工具不兼容；用 PIL 强制转）
    from PIL import Image
    for name in ("performance.png", "roofline.png"):
        Image.open(name).convert("RGB").save(name)
    print("saved: performance.png, roofline.png, results.json")


if __name__ == "__main__":
    main()
