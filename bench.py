"""numpy 参考实现与对拍脚本（与 main.cu 的确定性初始化保持一致）

用法（在 WSL 里）：
  python3 bench.py                  # 只跑 numpy 参考，打印 GFLOPS
  python3 bench.py 1024 --check c.bin
                                    # 与 ./sgemm 1024 --dump c.bin 的输出对拍
"""

import struct
import sys
import time

import numpy as np


def init_a(m, k):
    """与 main.cu 的 init_matrices 完全一致：A[i] = ((i % 101) - 50) * 0.01"""
    return ((np.arange(m * k, dtype=np.float64) % 101) - 50) * 0.01


def init_b(k, n):
    """与 main.cu 的 init_matrices 完全一致：B[j] = ((j % 97) - 48) * 0.01"""
    return ((np.arange(k * n, dtype=np.float64) % 97) - 48) * 0.01


def load_dump(path):
    """读取 main.cu --dump 输出的二进制：3 个 int32 (M,N,K) + M*N 个 float32"""
    with open(path, "rb") as f:
        m, n, k = struct.unpack("iii", f.read(12))
        c = np.frombuffer(f.read(m * n * 4), dtype=np.float32)
    return m, n, k, c.reshape(m, n)


def main():
    args = sys.argv[1:]
    check_path = None
    if "--check" in args:
        check_path = args[args.index("--check") + 1]
        args.remove("--check")
        args.remove(check_path)
    m = n = k = int(args[0]) if args else 1024

    a = init_a(m, k).reshape(m, k)
    b = init_b(k, n).reshape(k, n)

    t0 = time.perf_counter()
    c = a @ b  # numpy 内部走 BLAS
    dt = time.perf_counter() - t0
    print(f"numpy 参考: {2 * m * n * k / dt / 1e9:.2f} GFLOPS ({dt * 1e3:.1f} ms)")

    if check_path:
        m2, n2, k2, c_cuda = load_dump(check_path)
        assert (m2, n2, k2) == (m, n, k), f"维度不匹配: {(m2, n2, k2)} vs {(m, n, k)}"
        diff = np.abs(c.astype(np.float32) - c_cuda)
        rel = diff.max() / max(np.abs(c).max(), 1e-6)
        print(f"与 CUDA 输出对拍: max abs diff = {diff.max():.3e}, "
              f"max rel diff = {rel:.3e}")
        ok = np.all(diff <= 1e-3 * (1 + np.abs(c)))
        print("PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
