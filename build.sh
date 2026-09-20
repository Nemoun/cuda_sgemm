#!/usr/bin/env bash
# 编译脚本：在 WSL 里执行 bash build.sh
# -arch=sm_89 对应 RTX 4070 Laptop（Ada, compute capability 89）
set -e
cd "$(dirname "$0")"
# 从 Windows 调 wsl 走的是非交互 shell，不加载 .bashrc/.profile，这里兜底 PATH
if [ -x /usr/local/cuda/bin/nvcc ]; then
  export PATH="/usr/local/cuda/bin:$PATH"
fi
nvcc -arch=sm_89 -O3 -Xcompiler -O3 main.cu -o sgemm -lcublas
echo "build OK: ./sgemm [M [N [K]]] [--dump out.bin]"
