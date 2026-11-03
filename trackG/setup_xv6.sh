#!/bin/bash
# Clone upstream xv6-riscv at the commit this patch was made against, apply the A8 patch, build.
# Needs: git, qemu-system-misc, gcc-riscv64-linux-gnu (Ubuntu/Debian: apt install qemu-system-misc gcc-riscv64-linux-gnu)
set -e
DEST=${1:-external/xv6-riscv}; HERE=$(cd "$(dirname "$0")" && pwd)
if [ ! -d "$DEST" ]; then git clone https://github.com/mit-pdos/xv6-riscv.git "$DEST"; fi
cd "$DEST" && git checkout -q e93dabeb98431235c6e1a101cb96c73c923fdbc4
git apply "$HERE/xv6-a8.patch"
make TOOLPREFIX=riscv64-linux-gnu- kernel/kernel fs.img
echo "xv6 built in $DEST — run: python3 trackG/qemu_run.py $DEST 'schedbench 1 3'"
