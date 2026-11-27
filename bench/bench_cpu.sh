#!/bin/bash
# bench_cpu.sh [PACRUNNER [TAG]] [-- bench.py options]: the CPU-only benchmark: bench_full.sh without the GPU variants
# (stock, cpu, cpu-jit, simd-avx2/simd-neon, simd-split-avx2/simd-split-neon, simd-avx512, simd-split-avx512). Paths: run.sh.
exec "$(dirname "$0")/run.sh" cpu "$@"
