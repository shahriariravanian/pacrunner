#!/bin/bash
# bench_full.sh [PACRUNNER [TAG]] [-- bench.py options]: the full benchmark: stock, cpu, cpu-jit, the SIMD devices (AVX2/NEON,
# AVX-512, and their split kernels), the GPU with MadSpace on the CPU (cuda-f64, cuda-mixed or metal-mixed) and
# everything on the GPU (cuda-full-f64, cuda-full-mixed), each where this machine has it. Paths: run.sh.
exec "$(dirname "$0")/run.sh" full "$@"
