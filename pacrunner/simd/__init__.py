"""The SIMD devices (simd-avx2, simd-avx512, simd-neon): a program as one symjit kernel (the library of
pacops operations, lib/, spliced by codegen/compile_simd.py), its machine code as an object (elf.py);
the split kernels' model kernels (split_kernel.py) and fragments written as MIR (mir.py)."""
