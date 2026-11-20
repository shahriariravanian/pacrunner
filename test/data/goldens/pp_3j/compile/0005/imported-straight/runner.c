#include <stdlib.h>
#include "pacops.h"
#include "runner.h"

const double scales[] = {
	0.0, 0.0,
	0.7071067811865476, 0.0,
	0.0, 0.7071067811865476,
	};

const double factors[] = {
	1.0, 0.0,
	-1.0, 0.0,
	};

#include <math.h>
#include <stdint.h>
#include <string.h>

/* a function of the host and, under nvcc, of the device (as pacops.h's) */
#ifndef PAC_FN
#ifdef __CUDACC__
#define PAC_FN __host__ __device__
#else
#define PAC_FN
#endif
#endif

#ifdef __CUDACC__
#pragma nv_diag_suppress 177  /* a helper a program's kernels do not use */
#endif

/* the symjit kernels' bit operations on doubles (kernel_c.jl); a program's kernels use some of
   them, so each is marked unused (clang's -Wunused-function warns of an unused static inline) */
#define SJ_FN PAC_FN static inline __attribute__((unused))
SJ_FN uint64_t sj_bits(double x) { uint64_t u; memcpy(&u, &x, sizeof u); return u; }
SJ_FN double sj_double(uint64_t u) { double x; memcpy(&x, &u, sizeof x); return x; }
SJ_FN double sj_mask(int b) { return sj_double(b ? ~(uint64_t)0 : 0); }
SJ_FN double sj_and(double a, double b) { return sj_double(sj_bits(a) & sj_bits(b)); }
SJ_FN double sj_andnot(double a, double b) { return sj_double(~sj_bits(a) & sj_bits(b)); }
SJ_FN double sj_or(double a, double b) { return sj_double(sj_bits(a) | sj_bits(b)); }
SJ_FN double sj_xor(double a, double b) { return sj_double(sj_bits(a) ^ sj_bits(b)); }
SJ_FN double sj_not(double a) { return sj_double(~sj_bits(a)); }
SJ_FN double sj_sign(double a) { return sj_double(sj_bits(a) & ((uint64_t)1 << 63)); }
/* the symjit kernel k_50a8d7ac89185a73 (72 MIR instructions), from its .plane.symjit by kernel_c.jl */
PAC_FN static int k_50a8d7ac89185a73(double *mem) {
	double stack2 = 0.0, stack144 = 0.0, stack145 = 0.0, stack146 = 0.0, stack147 = 0.0;
	double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0, g6 = 0.0, g7 = 0.0, g8 = 0.0, g9 = 0.0, r0 = 0.0, r1 = 0.0, g10 = 0.0, g11 = 0.0, g12 = 0.0, g13 = 0.0;
	int ret = -1;
	g4 = 0.5;
	g5 = sj_xor(g5, g5);
	g0 = g4;
	g1 = g5;
	ret = 0; goto L__40_complex_root; R_0:;
	g4 = r0;
	g5 = r1;
	goto L__40_jump_over_complex_root;
	L__40_complex_root:;
	g2 = sj_xor(g2, g2);
	g2 = sj_mask(g2 < g0);
	stack2 = g2;
	g2 = g0 * g0;
	g2 = fma(g1, g1, g2);
	g2 = sqrt(g2);
	g3 = fabs(g0);
	g2 = g2 + g3;
	g2 = g2 * 0.5;
	g2 = sqrt(g2);
	g3 = g1 / g2;
	g3 = g3 * 0.5;
	g0 = sj_mask(g3 == g3);
	g3 = sj_and(g3, g0);
	g0 = stack2 != 0.0 ? g2 : g3;
	g1 = stack2 != 0.0 ? g3 : g2;
	g2 = sj_sign(g0);
	r0 = sj_xor(g0, g2);
	r1 = sj_xor(g1, g2);
	goto L_return;
	L__40_jump_over_complex_root:;
	stack144 = g4; stack145 = g5;
	g6 = mem[6]; g7 = mem[7];
	g3 = -g7;
	g9 = g6;
	g8 = g3;
	g10 = mem[4]; g11 = mem[5];
	g0 = mem[10]; g1 = mem[11];
	g12 = g10 + g0; g13 = g11 + g1;
	g3 = -g13;
	g13 = g12;
	g12 = g3;
	g0 = mem[0]; g1 = mem[1];
	g2 = g13 * g1; g2 = fma(g12, g0, -g2); g3 = g12 * g1; g13 = fma(g0, g13, g3); g12 = g2;
	g0 = mem[8]; g1 = mem[9];
	g4 = g8 + g0; g5 = g9 + g1;
	r0 = mem[2]; r1 = mem[3];
	g2 = g5 * r1; g2 = fma(g4, r0, -g2); g3 = g4 * r1; r1 = fma(r0, g5, g3); r0 = g2;
	g4 = r0 + g12; g5 = r1 + g13;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	stack146 = g4; stack147 = g5;
	g0 = mem[8]; g1 = mem[9];
	g12 = g8 - g0; g13 = g9 - g1;
	g0 = mem[0]; g1 = mem[1];
	g2 = g13 * g1; g2 = fma(g12, g0, -g2); g3 = g12 * g1; g13 = fma(g0, g13, g3); g12 = g2;
	g3 = -g11;
	g5 = g10;
	g4 = g3;
	g6 = mem[10]; g7 = mem[11];
	g3 = g7;
	g7 = -g6;
	g6 = g3;
	g4 = g4 + g6; g5 = g5 + g7;
	r0 = mem[2]; r1 = mem[3];
	g2 = g5 * r1; g2 = fma(g4, r0, -g2); g3 = g4 * r1; r1 = fma(r0, g5, g3); r0 = g2;
	g4 = r0 + g12; g5 = r1 + g13;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	g6 = stack146; g7 = stack147;
	mem[12] = g6; mem[13] = g7;
	mem[12] = g6; mem[13] = g7;
	mem[14] = g4; mem[15] = g5;
	return 0;
	L_return:
	switch (ret) {
		case 0: goto R_0;
		}
	return -1;  /* a return outside a funclet */
	}

/* the symjit kernel k_e2d4ab976ad89705 (88 MIR instructions), from its .plane.symjit by kernel_c.jl */
PAC_FN static int k_e2d4ab976ad89705(double *mem) {
	double stack2 = 0.0, stack144 = 0.0, stack145 = 0.0, stack146 = 0.0, stack147 = 0.0, stack148 = 0.0, stack149 = 0.0, stack150 = 0.0, stack151 = 0.0, stack152 = 0.0, stack153 = 0.0;
	double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0, g6 = 0.0, g7 = 0.0, g8 = 0.0, g9 = 0.0, r0 = 0.0, r1 = 0.0, g10 = 0.0, g11 = 0.0, g12 = 0.0, g13 = 0.0;
	int ret = -1;
	g4 = 0.5;
	g5 = sj_xor(g5, g5);
	g0 = g4;
	g1 = g5;
	ret = 0; goto L__40_complex_root; R_0:;
	g4 = r0;
	g5 = r1;
	goto L__40_jump_over_complex_root;
	L__40_complex_root:;
	g2 = sj_xor(g2, g2);
	g2 = sj_mask(g2 < g0);
	stack2 = g2;
	g2 = g0 * g0;
	g2 = fma(g1, g1, g2);
	g2 = sqrt(g2);
	g3 = fabs(g0);
	g2 = g2 + g3;
	g2 = g2 * 0.5;
	g2 = sqrt(g2);
	g3 = g1 / g2;
	g3 = g3 * 0.5;
	g0 = sj_mask(g3 == g3);
	g3 = sj_and(g3, g0);
	g0 = stack2 != 0.0 ? g2 : g3;
	g1 = stack2 != 0.0 ? g3 : g2;
	g2 = sj_sign(g0);
	r0 = sj_xor(g0, g2);
	r1 = sj_xor(g1, g2);
	goto L_return;
	L__40_jump_over_complex_root:;
	stack144 = g4; stack145 = g5;
	r0 = mem[0]; r1 = mem[1];
	g6 = mem[2]; g7 = mem[3];
	g0 = mem[4]; g1 = mem[5];
	g2 = r1 * g1; g2 = fma(r0, g0, -g2); g3 = r0 * g1; r1 = fma(g0, r1, g3); r0 = g2;
	g0 = mem[6]; g1 = mem[7];
	g2 = g7 * g1; g2 = fma(g6, g0, -g2); g3 = g6 * g1; g9 = fma(g0, g7, g3); g8 = g2;
	stack146 = r0; stack147 = r1;
	r0 = stack146; r1 = stack147;
	g10 = r0 + g8; g11 = r1 + g9;
	g0 = stack144; g1 = stack145;
	g2 = g11 * g1; g2 = fma(g10, g0, -g2); g3 = g10 * g1; g11 = fma(g0, g11, g3); g10 = g2;
	g3 = -g11;
	g11 = g10;
	g10 = g3;
	r0 = mem[0]; r1 = mem[1];
	g0 = mem[6]; g1 = mem[7];
	g2 = r1 * g1; g2 = fma(r0, g0, -g2); g3 = r0 * g1; r1 = fma(g0, r1, g3); r0 = g2;
	g0 = mem[4]; g1 = mem[5];
	g2 = g7 * g1; g2 = fma(g6, g0, -g2); g3 = g6 * g1; g13 = fma(g0, g7, g3); g12 = g2;
	stack148 = r0; stack149 = r1;
	stack150 = g12; stack151 = g13;
	r0 = stack148; r1 = stack149;
	g4 = r0 + g12; g5 = r1 + g13;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	g3 = g5;
	g5 = -g4;
	g4 = g3;
	stack152 = g4; stack153 = g5;
	g4 = stack148; g5 = stack149;
	g0 = stack150; g1 = stack151;
	g4 = g4 - g0; g5 = g5 - g1;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	stack150 = g4; stack151 = g5;
	g4 = stack146; g5 = stack147;
	g3 = g5;
	g5 = -g4;
	g4 = g3;
	g3 = -g9;
	g7 = g8;
	g6 = g3;
	g4 = g4 + g6; g5 = g5 + g7;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	stack144 = g4; stack145 = g5;
	stack144 = g4; stack145 = g5;
	mem[8] = g10; mem[9] = g11;
	mem[8] = g10; mem[9] = g11;
	g6 = stack152; g7 = stack153;
	mem[10] = g6; mem[11] = g7;
	mem[10] = g6; mem[11] = g7;
	g4 = stack150; g5 = stack151;
	mem[12] = g4; mem[13] = g5;
	mem[12] = g4; mem[13] = g5;
	g4 = stack144; g5 = stack145;
	mem[14] = g4; mem[15] = g5;
	return 0;
	L_return:
	switch (ret) {
		case 0: goto R_0;
		}
	return -1;  /* a return outside a funclet */
	}

/* the symjit kernel k_d76e7388753f86a9 (57 MIR instructions), from its .plane.symjit by kernel_c.jl */
PAC_FN static int k_d76e7388753f86a9(double *mem) {
	double stack144 = 0.0, stack145 = 0.0, stack146 = 0.0, stack147 = 0.0, stack148 = 0.0, stack149 = 0.0;
	double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0, g6 = 0.0, g7 = 0.0, g8 = 0.0, g9 = 0.0, r0 = 0.0, r1 = 0.0, g10 = 0.0, g11 = 0.0, g12 = 0.0, g13 = 0.0;
	g4 = mem[6]; g5 = mem[7];
	g3 = g5;
	g7 = -g4;
	g6 = g3;
	stack144 = g6; stack145 = g7;
	g2 = g5 * g5; g2 = fma(g4, g4, -g2); g3 = g4 * g5; g9 = fma(g4, g5, g3); g8 = g2;
	g10 = mem[4]; g11 = mem[5];
	g2 = g11 * g11; g2 = fma(g10, g10, -g2); g3 = g10 * g11; r1 = fma(g10, g11, g3); r0 = g2;
	g8 = g8 - r0; g9 = g9 - r1;
	g12 = mem[8]; g13 = mem[9];
	g2 = g13 * g13; g2 = fma(g12, g12, -g2); g3 = g12 * g13; g5 = fma(g12, g13, g3); g4 = g2;
	g6 = mem[10]; g7 = mem[11];
	g2 = g7 * g7; g2 = fma(g6, g6, -g2); g3 = g6 * g7; r1 = fma(g6, g7, g3); r0 = g2;
	g4 = r0 + g4; g5 = r1 + g5;
	g4 = g8 + g4; g5 = g9 + g5;
	g2 = g4 * g4;
	g2 = fma(g5, g5, g2);
	g4 = g4 / g2;
	g5 = g5 / g2;
	g5 = -g5;
	stack146 = g4; stack147 = g5;
	g3 = -g11;
	g9 = g10;
	g8 = g3;
	g3 = g7;
	g5 = -g6;
	g4 = g3;
	g4 = g8 + g4; g5 = g9 + g5;
	g0 = mem[0]; g1 = mem[1];
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	g8 = stack144; g9 = stack145;
	g0 = mem[8]; g1 = mem[9];
	g6 = g8 - g0; g7 = g9 - g1;
	r0 = mem[2]; r1 = mem[3];
	g2 = g7 * r1; g2 = fma(g6, r0, -g2); g3 = g6 * r1; r1 = fma(r0, g7, g3); r0 = g2;
	g4 = r0 + g4; g5 = r1 + g5;
	g0 = stack146; g1 = stack147;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	stack148 = g4; stack149 = g5;
	g0 = stack144; g1 = stack145;
	g6 = g12 + g0; g7 = g13 + g1;
	g0 = mem[0]; g1 = mem[1];
	g2 = g7 * g1; g2 = fma(g6, g0, -g2); g3 = g6 * g1; g7 = fma(g0, g7, g3); g6 = g2;
	g0 = mem[10]; g1 = mem[11];
	g4 = g10 + g0; g5 = g11 + g1;
	g8 = mem[2]; g9 = mem[3];
	g3 = -g9;
	g9 = g8;
	g8 = g3;
	g2 = g5 * g9; g2 = fma(g4, g8, -g2); g3 = g4 * g9; r1 = fma(g8, g5, g3); r0 = g2;
	g4 = r0 + g6; g5 = r1 + g7;
	g0 = stack146; g1 = stack147;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	g6 = stack148; g7 = stack149;
	mem[12] = g6; mem[13] = g7;
	mem[12] = g6; mem[13] = g7;
	mem[14] = g4; mem[15] = g5;
	return 0;
	}

/* the symjit kernel k_5acfa4bfbabd18e1 (105 MIR instructions), from its .plane.symjit by kernel_c.jl */
PAC_FN static int k_5acfa4bfbabd18e1(double *mem) {
	double stack2 = 0.0, stack144 = 0.0, stack145 = 0.0, stack146 = 0.0, stack147 = 0.0, stack148 = 0.0, stack149 = 0.0, stack150 = 0.0, stack151 = 0.0, stack152 = 0.0, stack153 = 0.0, stack154 = 0.0, stack155 = 0.0, stack156 = 0.0, stack157 = 0.0;
	double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0, g6 = 0.0, g7 = 0.0, g8 = 0.0, g9 = 0.0, r0 = 0.0, r1 = 0.0, g10 = 0.0, g11 = 0.0, g12 = 0.0, g13 = 0.0;
	int ret = -1;
	g4 = 0.5;
	g5 = sj_xor(g5, g5);
	g0 = g4;
	g1 = g5;
	ret = 0; goto L__40_complex_root; R_0:;
	g4 = r0;
	g5 = r1;
	goto L__40_jump_over_complex_root;
	L__40_complex_root:;
	g2 = sj_xor(g2, g2);
	g2 = sj_mask(g2 < g0);
	stack2 = g2;
	g2 = g0 * g0;
	g2 = fma(g1, g1, g2);
	g2 = sqrt(g2);
	g3 = fabs(g0);
	g2 = g2 + g3;
	g2 = g2 * 0.5;
	g2 = sqrt(g2);
	g3 = g1 / g2;
	g3 = g3 * 0.5;
	g0 = sj_mask(g3 == g3);
	g3 = sj_and(g3, g0);
	g0 = stack2 != 0.0 ? g2 : g3;
	g1 = stack2 != 0.0 ? g3 : g2;
	g2 = sj_sign(g0);
	r0 = sj_xor(g0, g2);
	r1 = sj_xor(g1, g2);
	goto L_return;
	L__40_jump_over_complex_root:;
	stack144 = g4; stack145 = g5;
	r0 = mem[0]; r1 = mem[1];
	g6 = mem[2]; g7 = mem[3];
	g0 = mem[4]; g1 = mem[5];
	g2 = r1 * g1; g2 = fma(r0, g0, -g2); g3 = r0 * g1; r1 = fma(g0, r1, g3); r0 = g2;
	g0 = mem[6]; g1 = mem[7];
	g2 = g7 * g1; g2 = fma(g6, g0, -g2); g3 = g6 * g1; g9 = fma(g0, g7, g3); g8 = g2;
	stack146 = r0; stack147 = r1;
	stack148 = g8; stack149 = g9;
	g10 = 0.0;
	g12 = 0.3333333333333333;
	g11 = g12;
	g12 = stack146; g13 = stack147;
	g0 = stack148; g1 = stack149;
	g4 = g12 + g0; g5 = g13 + g1;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	g2 = g11 * g5; g2 = fma(g10, g4, -g2); g3 = g10 * g5; g5 = fma(g4, g11, g3); g4 = g2;
	stack150 = g4; stack151 = g5;
	r0 = mem[0]; r1 = mem[1];
	g0 = mem[6]; g1 = mem[7];
	g2 = r1 * g1; g2 = fma(r0, g0, -g2); g3 = r0 * g1; r1 = fma(g0, r1, g3); r0 = g2;
	g0 = mem[4]; g1 = mem[5];
	g2 = g7 * g1; g2 = fma(g6, g0, -g2); g3 = g6 * g1; g11 = fma(g0, g7, g3); g10 = g2;
	stack152 = r0; stack153 = r1;
	stack154 = g10; stack155 = g11;
	g4 = 0.0;
	g6 = -0.3333333333333333;
	g5 = g6;
	g6 = stack152; g7 = stack153;
	g0 = stack154; g1 = stack155;
	g6 = g6 + g0; g7 = g7 + g1;
	g0 = stack144; g1 = stack145;
	g2 = g7 * g1; g2 = fma(g6, g0, -g2); g3 = g6 * g1; g7 = fma(g0, g7, g3); g6 = g2;
	g2 = g5 * g7; g2 = fma(g4, g6, -g2); g3 = g4 * g7; g5 = fma(g6, g5, g3); g4 = g2;
	stack156 = g4; stack157 = g5;
	g6 = stack152; g7 = stack153;
	g0 = stack154; g1 = stack155;
	g4 = g6 - g0; g5 = g7 - g1;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	g0 = -0.3333333333333333;
	g5 = g5 * g0;
	g4 = g4 * g0;
	stack154 = g4; stack155 = g5;
	g4 = 0.0;
	g6 = -0.3333333333333333;
	g5 = g6;
	g0 = stack146; g1 = stack147;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	g6 = 0.0;
	g8 = 0.3333333333333333;
	g7 = g8;
	r0 = stack148; r1 = stack149;
	g2 = g7 * r1; g2 = fma(g6, r0, -g2); g3 = g6 * r1; r1 = fma(r0, g7, g3); r0 = g2;
	g4 = r0 + g4; g5 = r1 + g5;
	g0 = stack144; g1 = stack145;
	g2 = g5 * g1; g2 = fma(g4, g0, -g2); g3 = g4 * g1; g5 = fma(g0, g5, g3); g4 = g2;
	stack144 = g4; stack145 = g5;
	stack144 = g4; stack145 = g5;
	g6 = stack150; g7 = stack151;
	mem[8] = g6; mem[9] = g7;
	mem[8] = g6; mem[9] = g7;
	g8 = stack156; g9 = stack157;
	mem[10] = g8; mem[11] = g9;
	mem[10] = g8; mem[11] = g9;
	g4 = stack154; g5 = stack155;
	mem[12] = g4; mem[13] = g5;
	mem[12] = g4; mem[13] = g5;
	g4 = stack144; g5 = stack145;
	mem[14] = g4; mem[15] = g5;
	return 0;
	L_return:
	switch (ret) {
		case 0: goto R_0;
		}
	return -1;  /* a return outside a funclet */
	}

void adapter_k_50a8d7ac89185a73(double *out, double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8, double x9, double x10, double x11,  const int *outputs, int direct, int mode, const double *factor) {
	double mem[17];
	mem[0] = x0;
	mem[1] = x1;
	mem[2] = x2;
	mem[3] = x3;
	mem[4] = x4;
	mem[5] = x5;
	mem[6] = x6;
	mem[7] = x7;
	mem[8] = x8;
	mem[9] = x9;
	mem[10] = x10;
	mem[11] = x11;
	if (k_50a8d7ac89185a73(mem) != 0) abort();  /* the kernel failed */
	cplx f = get(factor, 0);
	for (int n = 0; n < 2; n++) {
		cplx pair = get(mem + 12, n);
		put(out, outputs[n], combine(get(out, outputs[n]), direct ? pair : mul(pair, f), mode));
		}
	}

void adapter_k_e2d4ab976ad89705(double *out, double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7,  const int *outputs, int direct, int mode, const double *factor) {
	double mem[17];
	mem[0] = x0;
	mem[1] = x1;
	mem[2] = x2;
	mem[3] = x3;
	mem[4] = x4;
	mem[5] = x5;
	mem[6] = x6;
	mem[7] = x7;
	if (k_e2d4ab976ad89705(mem) != 0) abort();  /* the kernel failed */
	cplx f = get(factor, 0);
	for (int n = 0; n < 4; n++) {
		cplx pair = get(mem + 8, n);
		put(out, outputs[n], combine(get(out, outputs[n]), direct ? pair : mul(pair, f), mode));
		}
	}

void adapter_k_d76e7388753f86a9(double *out, double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7, double x8, double x9, double x10, double x11,  const int *outputs, int direct, int mode, const double *factor) {
	double mem[17];
	mem[0] = x0;
	mem[1] = x1;
	mem[2] = x2;
	mem[3] = x3;
	mem[4] = x4;
	mem[5] = x5;
	mem[6] = x6;
	mem[7] = x7;
	mem[8] = x8;
	mem[9] = x9;
	mem[10] = x10;
	mem[11] = x11;
	if (k_d76e7388753f86a9(mem) != 0) abort();  /* the kernel failed */
	cplx f = get(factor, 0);
	for (int n = 0; n < 2; n++) {
		cplx pair = get(mem + 12, n);
		put(out, outputs[n], combine(get(out, outputs[n]), direct ? pair : mul(pair, f), mode));
		}
	}

void adapter_k_5acfa4bfbabd18e1(double *out, double x0, double x1, double x2, double x3, double x4, double x5, double x6, double x7,  const int *outputs, int direct, int mode, const double *factor) {
	double mem[17];
	mem[0] = x0;
	mem[1] = x1;
	mem[2] = x2;
	mem[3] = x3;
	mem[4] = x4;
	mem[5] = x5;
	mem[6] = x6;
	mem[7] = x7;
	if (k_5acfa4bfbabd18e1(mem) != 0) abort();  /* the kernel failed */
	cplx f = get(factor, 0);
	for (int n = 0; n < 4; n++) {
		cplx pair = get(mem + 8, n);
		put(out, outputs[n], combine(get(out, outputs[n]), direct ? pair : mul(pair, f), mode));
		}
	}

void forms(double *p, const double *momenta) {
	p[0] = 0.0 + -1.0 * (1.0 * momenta[0]);
	p[1] = 0.0 + -1.0 * (1.0 * momenta[1]);
	p[2] = 0.0 + -1.0 * (1.0 * momenta[2]);
	p[3] = 0.0 + -1.0 * (1.0 * momenta[3]);
	p[4] = 0.0 + -1.0 * (1.0 * momenta[0]) + -1.0 * (1.0 * momenta[4]);
	p[5] = 0.0 + -1.0 * (1.0 * momenta[1]) + -1.0 * (1.0 * momenta[5]);
	p[6] = 0.0 + -1.0 * (1.0 * momenta[2]) + -1.0 * (1.0 * momenta[6]);
	p[7] = 0.0 + -1.0 * (1.0 * momenta[3]) + -1.0 * (1.0 * momenta[7]);
	p[8] = 0.0 + -1.0 * (1.0 * momenta[0]) + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[12]);
	p[9] = 0.0 + -1.0 * (1.0 * momenta[1]) + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[13]);
	p[10] = 0.0 + -1.0 * (1.0 * momenta[2]) + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[14]);
	p[11] = 0.0 + -1.0 * (1.0 * momenta[3]) + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[15]);
	p[12] = 0.0 + -1.0 * (1.0 * momenta[0]) + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[12]) + 1.0 * (1.0 * momenta[16]);
	p[13] = 0.0 + -1.0 * (1.0 * momenta[1]) + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[13]) + 1.0 * (1.0 * momenta[17]);
	p[14] = 0.0 + -1.0 * (1.0 * momenta[2]) + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[14]) + 1.0 * (1.0 * momenta[18]);
	p[15] = 0.0 + -1.0 * (1.0 * momenta[3]) + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[15]) + 1.0 * (1.0 * momenta[19]);
	p[16] = 0.0 + -1.0 * (1.0 * momenta[0]) + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[16]);
	p[17] = 0.0 + -1.0 * (1.0 * momenta[1]) + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[17]);
	p[18] = 0.0 + -1.0 * (1.0 * momenta[2]) + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[18]);
	p[19] = 0.0 + -1.0 * (1.0 * momenta[3]) + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[19]);
	p[20] = 0.0 + -1.0 * (1.0 * momenta[0]) + 1.0 * (1.0 * momenta[12]);
	p[21] = 0.0 + -1.0 * (1.0 * momenta[1]) + 1.0 * (1.0 * momenta[13]);
	p[22] = 0.0 + -1.0 * (1.0 * momenta[2]) + 1.0 * (1.0 * momenta[14]);
	p[23] = 0.0 + -1.0 * (1.0 * momenta[3]) + 1.0 * (1.0 * momenta[15]);
	p[24] = 0.0 + -1.0 * (1.0 * momenta[0]) + 1.0 * (1.0 * momenta[12]) + 1.0 * (1.0 * momenta[16]);
	p[25] = 0.0 + -1.0 * (1.0 * momenta[1]) + 1.0 * (1.0 * momenta[13]) + 1.0 * (1.0 * momenta[17]);
	p[26] = 0.0 + -1.0 * (1.0 * momenta[2]) + 1.0 * (1.0 * momenta[14]) + 1.0 * (1.0 * momenta[18]);
	p[27] = 0.0 + -1.0 * (1.0 * momenta[3]) + 1.0 * (1.0 * momenta[15]) + 1.0 * (1.0 * momenta[19]);
	p[28] = 0.0 + -1.0 * (1.0 * momenta[0]) + 1.0 * (1.0 * momenta[16]);
	p[29] = 0.0 + -1.0 * (1.0 * momenta[1]) + 1.0 * (1.0 * momenta[17]);
	p[30] = 0.0 + -1.0 * (1.0 * momenta[2]) + 1.0 * (1.0 * momenta[18]);
	p[31] = 0.0 + -1.0 * (1.0 * momenta[3]) + 1.0 * (1.0 * momenta[19]);
	p[32] = 0.0 + -1.0 * (1.0 * momenta[4]);
	p[33] = 0.0 + -1.0 * (1.0 * momenta[5]);
	p[34] = 0.0 + -1.0 * (1.0 * momenta[6]);
	p[35] = 0.0 + -1.0 * (1.0 * momenta[7]);
	p[36] = 0.0 + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[12]);
	p[37] = 0.0 + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[13]);
	p[38] = 0.0 + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[14]);
	p[39] = 0.0 + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[15]);
	p[40] = 0.0 + -1.0 * (1.0 * momenta[4]) + 1.0 * (1.0 * momenta[16]);
	p[41] = 0.0 + -1.0 * (1.0 * momenta[5]) + 1.0 * (1.0 * momenta[17]);
	p[42] = 0.0 + -1.0 * (1.0 * momenta[6]) + 1.0 * (1.0 * momenta[18]);
	p[43] = 0.0 + -1.0 * (1.0 * momenta[7]) + 1.0 * (1.0 * momenta[19]);
	p[44] = 0.0 + 1.0 * (1.0 * momenta[12]);
	p[45] = 0.0 + 1.0 * (1.0 * momenta[13]);
	p[46] = 0.0 + 1.0 * (1.0 * momenta[14]);
	p[47] = 0.0 + 1.0 * (1.0 * momenta[15]);
	p[48] = 0.0 + 1.0 * (1.0 * momenta[16]);
	p[49] = 0.0 + 1.0 * (1.0 * momenta[17]);
	p[50] = 0.0 + 1.0 * (1.0 * momenta[18]);
	p[51] = 0.0 + 1.0 * (1.0 * momenta[19]);
	p[52] = 0.0 + 1.0 * (1.0 * momenta[8]);
	p[53] = 0.0 + 1.0 * (1.0 * momenta[9]);
	p[54] = 0.0 + 1.0 * (1.0 * momenta[10]);
	p[55] = 0.0 + 1.0 * (1.0 * momenta[11]);
	}

static const double cf1[] = {1.0, 0.0, 1.0, 0.0};
static const int outputs1[] = {0, 1};
static const int outputs2[] = {0, 1, 2, 3};
static const int outputs3[] = {0, 1};
static const int outputs4[] = {0, 1, 2, 3};

void body(double *out, double *p) {
	// 1: CLEAR_AMPLITUDE a0
	p[56] = 0.0; p[57] = 0.0;
	// 2: CLEAR_AMPLITUDE a1
	p[58] = 0.0; p[59] = 0.0;
	// 3: CLEAR_AMPLITUDE a2
	p[60] = 0.0; p[61] = 0.0;
	// 4: CLEAR_AMPLITUDE a3
	p[62] = 0.0; p[63] = 0.0;
	// 5: CLEAR_AMPLITUDE a4
	p[64] = 0.0; p[65] = 0.0;
	// 6: CLEAR_AMPLITUDE a5
	p[66] = 0.0; p[67] = 0.0;
	// 7: CLEAR_AMPLITUDE a6
	p[68] = 0.0; p[69] = 0.0;
	// 8: CLEAR_AMPLITUDE a7
	p[70] = 0.0; p[71] = 0.0;
	// 9: CLEAR_AMPLITUDE a8
	p[72] = 0.0; p[73] = 0.0;
	// 10: CLEAR_AMPLITUDE a9
	p[74] = 0.0; p[75] = 0.0;
	// 11: CLEAR_AMPLITUDE a10
	p[76] = 0.0; p[77] = 0.0;
	// 12: CLEAR_AMPLITUDE a11
	p[78] = 0.0; p[79] = 0.0;
	// 13: CLEAR_AMPLITUDE a12
	p[80] = 0.0; p[81] = 0.0;
	// 14: CLEAR_AMPLITUDE a13
	p[82] = 0.0; p[83] = 0.0;
	// 15: CLEAR_AMPLITUDE a14
	p[84] = 0.0; p[85] = 0.0;
	// 16: CLEAR_AMPLITUDE a15
	p[86] = 0.0; p[87] = 0.0;
	// 17: CLEAR_AMPLITUDE a16
	p[88] = 0.0; p[89] = 0.0;
	// 18: CLEAR_AMPLITUDE a17
	p[90] = 0.0; p[91] = 0.0;
	// 19: CLEAR_AMPLITUDE a18
	p[92] = 0.0; p[93] = 0.0;
	// 20: CLEAR_AMPLITUDE a19
	p[94] = 0.0; p[95] = 0.0;
	// 21: CLEAR_AMPLITUDE a20
	p[96] = 0.0; p[97] = 0.0;
	// 22: CLEAR_AMPLITUDE a21
	p[98] = 0.0; p[99] = 0.0;
	// 23: CLEAR_AMPLITUDE a22
	p[100] = 0.0; p[101] = 0.0;
	// 24: CLEAR_AMPLITUDE a23
	p[102] = 0.0; p[103] = 0.0;
	// 25: SOURCE c0@q13 = WEYL antiparticle, helicity 1, chirality 1, mass s0; x0
	source(&p[144], FAMILY_WEYL, ANTIPARTICLE, &p[52], 1, 1, 0.0, &factors[0]);
	// 26: SOURCE c1@q11 = WEYL, helicity -1, chirality -1, mass s0; x0
	source(&p[124], FAMILY_WEYL, PARTICLE, &p[44], -1, -1, 0.0, &factors[0]);
	// 27: SOURCE c2@q12 = WEYL, helicity -1, chirality -1, mass s0; x0
	source(&p[132], FAMILY_WEYL, PARTICLE, &p[48], -1, -1, 0.0, &factors[0]);
	// 28: SOURCE c3@q12 = WEYL, helicity 1, chirality 1, mass s0; x0
	source(&p[136], FAMILY_WEYL, PARTICLE, &p[48], 1, 1, 0.0, &factors[0]);
	// 29: SOURCE c4@q11 = WEYL, helicity 1, chirality 1, mass s0; x0
	source(&p[128], FAMILY_WEYL, PARTICLE, &p[44], 1, 1, 0.0, &factors[0]);
	// 30: SOURCE c5@q8 = VECTOR, helicity 1, chirality 0, mass s0; x0
	source(&p[116], FAMILY_VECTOR, PARTICLE, &p[32], 1, 0, 0.0, &factors[0]);
	// 31: SOURCE c6@q8 = VECTOR, helicity -1, chirality 0, mass s0; x0
	source(&p[108], FAMILY_VECTOR, PARTICLE, &p[32], -1, 0, 0.0, &factors[0]);
	// 32: SOURCE c7@q13 = WEYL antiparticle, helicity -1, chirality -1, mass s0; x0
	source(&p[140], FAMILY_WEYL, ANTIPARTICLE, &p[52], -1, -1, 0.0, &factors[0]);
	// 33: SOURCE c8@q0 = WEYL antiparticle, helicity -1, chirality 1, mass s0; x0
	source(&p[104], FAMILY_WEYL, ANTIPARTICLE, &p[0], -1, 1, 0.0, &factors[0]);
	// 34: CLEAR c9@q1, 2 components
	clear(&p[180], 2);
	// 35: CLEAR c10@q1, 2 components
	clear(&p[184], 2);
	// 36: CLEAR c11, 4 components
	clear(&p[188], 4);
	// 37: CLEAR c12, 2 components
	clear(&p[196], 2);
	// 38: CLEAR c13, 2 components
	clear(&p[200], 2);
	// 39: CLEAR c14, 2 components
	clear(&p[204], 2);
	// 40: CLEAR c15, 2 components
	clear(&p[208], 2);
	// 41: KERNEL c9@q1 = c16, c17; binding instance 1 x0
	adapter_k_50a8d7ac89185a73(&p[180], p[104], p[105], p[106], p[107], p[108], p[109], p[110], p[111], p[112], p[113], p[114], p[115], outputs1, 0, MODE_SET, &factors[0]);
	// 42: KERNEL c10@q1 = c16, c18; binding instance 1 x0
	adapter_k_50a8d7ac89185a73(&p[184], p[104], p[105], p[106], p[107], p[116], p[117], p[118], p[119], p[120], p[121], p[122], p[123], outputs1, 0, MODE_SET, &factors[0]);
	// 43: V_WEYL_VECTOR_B c19@q9 = c1@q11, c6@q8; s1 x0
	vertex(&p[196], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[124], &p[108], NULL, NULL, &scales[2], &factors[0]);
	// 44: V_WEYL_VECTOR_B c20@q9 = c1@q11, c5@q8; s1 x0
	vertex(&p[204], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[124], &p[116], NULL, NULL, &scales[2], &factors[0]);
	// 45: KERNEL c11 = c16, c21; binding instance 2 x0
	adapter_k_e2d4ab976ad89705(&p[188], p[104], p[105], p[106], p[107], p[132], p[133], p[134], p[135], outputs2, 0, MODE_SET, &factors[0]);
	// 46: V_WEYL_VECTOR_A c22@q9 = c4@q11, c6@q8; s1 x0
	vertex(&p[200], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[128], &p[108], NULL, NULL, &scales[2], &factors[0]);
	// 47: V_WEYL_VECTOR_A c23@q9 = c4@q11, c5@q8; s1 x0
	vertex(&p[208], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[128], &p[116], NULL, NULL, &scales[2], &factors[0]);
	// 48: KERNEL c9@q1 = c9@q1, -; binding instance 3 x0
	adapter_k_d76e7388753f86a9(&p[180], p[180], p[181], p[182], p[183], p[4], scales[0], p[5], scales[0], p[6], scales[0], p[7], scales[0], outputs3, 0, MODE_SET, &factors[0]);
	// 49: KERNEL c10@q1 = c10@q1, -; binding instance 3 x0
	adapter_k_d76e7388753f86a9(&p[184], p[184], p[185], p[186], p[187], p[4], scales[0], p[5], scales[0], p[6], scales[0], p[7], scales[0], outputs3, 0, MODE_SET, &factors[0]);
	// 50: P_VECTOR c24@q7; x0
	propagator(&p[188], PROPAGATOR_VECTOR, &p[188], &p[28], &factors[0]);
	// 51: P_WEYL_A c19@q9; x0
	propagator(&p[196], PROPAGATOR_WEYL_A, &p[196], &p[36], &factors[0]);
	// 52: P_WEYL_A c20@q9; x0
	propagator(&p[204], PROPAGATOR_WEYL_A, &p[204], &p[36], &factors[0]);
	// 53: P_WEYL_B c22@q9; x0
	propagator(&p[200], PROPAGATOR_WEYL_B, &p[200], &p[36], &factors[0]);
	// 54: P_WEYL_B c23@q9; x0
	propagator(&p[208], PROPAGATOR_WEYL_B, &p[208], &p[36], &factors[0]);
	// 55: CLEAR c25, 4 components
	clear(&p[304], 4);
	// 56: CLEAR c26, 4 components
	clear(&p[312], 4);
	// 57: CLEAR c27, 4 components
	clear(&p[320], 4);
	// 58: CLEAR c28, 4 components
	clear(&p[328], 4);
	// 59: KERNEL c25 = c29, c9@q1; binding instance 4 x1
	adapter_k_5acfa4bfbabd18e1(&p[304], p[124], p[125], p[126], p[127], p[180], p[181], p[182], p[183], outputs4, 0, MODE_SET, &factors[2]);
	// 60: KERNEL c25 += c12, c16; binding instance 4 x1
	adapter_k_5acfa4bfbabd18e1(&p[304], p[196], p[197], p[198], p[199], p[104], p[105], p[106], p[107], outputs4, 0, MODE_ADD, &factors[2]);
	// 61: KERNEL c26 = c29, c10@q1; binding instance 4 x1
	adapter_k_5acfa4bfbabd18e1(&p[312], p[124], p[125], p[126], p[127], p[184], p[185], p[186], p[187], outputs4, 0, MODE_SET, &factors[2]);
	// 62: KERNEL c26 += c14, c16; binding instance 4 x1
	adapter_k_5acfa4bfbabd18e1(&p[312], p[204], p[205], p[206], p[207], p[104], p[105], p[106], p[107], outputs4, 0, MODE_ADD, &factors[2]);
	// 63: KERNEL c27 = c9@q1, c21; binding instance 2 x0
	adapter_k_e2d4ab976ad89705(&p[320], p[180], p[181], p[182], p[183], p[132], p[133], p[134], p[135], outputs2, 0, MODE_SET, &factors[0]);
	// 64: KERNEL c28 = c10@q1, c21; binding instance 2 x0
	adapter_k_e2d4ab976ad89705(&p[328], p[184], p[185], p[186], p[187], p[132], p[133], p[134], p[135], outputs2, 0, MODE_SET, &factors[0]);
	// 65: V_THREE_VECTOR c30@q4 += c6@q8, c24@q7; s2 x0
	vertex(&p[320], VERTEX_THREE_VECTOR, MODE_ADD, &p[108], &p[188], &p[32], &p[28], &scales[4], &factors[0]);
	// 66: V_THREE_VECTOR c31@q4 += c5@q8, c24@q7; s2 x0
	vertex(&p[328], VERTEX_THREE_VECTOR, MODE_ADD, &p[116], &p[188], &p[32], &p[28], &scales[4], &factors[0]);
	// 67: P_VECTOR c32@q2; x0
	propagator(&p[304], PROPAGATOR_VECTOR, &p[304], &p[8], &factors[0]);
	// 68: P_VECTOR c33@q2; x0
	propagator(&p[312], PROPAGATOR_VECTOR, &p[312], &p[8], &factors[0]);
	// 69: P_VECTOR c30@q4; x0
	propagator(&p[320], PROPAGATOR_VECTOR, &p[320], &p[16], &factors[0]);
	// 70: P_VECTOR c31@q4; x0
	propagator(&p[328], PROPAGATOR_VECTOR, &p[328], &p[16], &factors[0]);
	// 71: CLEAR c34, 2 components
	clear(&p[216], 2);
	// 72: CLEAR c35, 2 components
	clear(&p[244], 2);
	// 73: CLEAR c36, 2 components
	clear(&p[248], 2);
	// 74: CLEAR c37, 2 components
	clear(&p[396], 2);
	// 75: CLEAR c38, 2 components
	clear(&p[400], 2);
	// 76: CLEAR c39, 2 components
	clear(&p[404], 2);
	// 77: V_WEYL_VECTOR_B c40 = c2@q12, c32@q2; s1 x0
	vertex(&p[216], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[132], &p[304], NULL, NULL, &scales[2], &factors[0]);
	// 78: V_WEYL_VECTOR_B c41 = c2@q12, c33@q2; s1 x0
	vertex(&p[244], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[132], &p[312], NULL, NULL, &scales[2], &factors[0]);
	// 79: V_WEYL_VECTOR_B c40 += c1@q11, c30@q4; s1 x1
	vertex(&p[216], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[124], &p[320], NULL, NULL, &scales[2], &factors[2]);
	// 80: V_WEYL_VECTOR_B c40 += c19@q9, c24@q7; s1 x1
	vertex(&p[216], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[196], &p[188], NULL, NULL, &scales[2], &factors[2]);
	// 81: V_WEYL_VECTOR_B c41 += c1@q11, c31@q4; s1 x1
	vertex(&p[244], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[124], &p[328], NULL, NULL, &scales[2], &factors[2]);
	// 82: V_WEYL_VECTOR_B c41 += c20@q9, c24@q7; s1 x1
	vertex(&p[244], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[204], &p[188], NULL, NULL, &scales[2], &factors[2]);
	// 83: V_WEYL_VECTOR_A c42 = c4@q11, c30@q4; s1 x1
	vertex(&p[248], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[128], &p[320], NULL, NULL, &scales[2], &factors[2]);
	// 84: V_WEYL_VECTOR_A c42 += c22@q9, c24@q7; s1 x1
	vertex(&p[248], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[200], &p[188], NULL, NULL, &scales[2], &factors[2]);
	// 85: V_WEYL_VECTOR_A c43 = c4@q11, c31@q4; s1 x1
	vertex(&p[396], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[128], &p[328], NULL, NULL, &scales[2], &factors[2]);
	// 86: V_WEYL_VECTOR_A c43 += c23@q9, c24@q7; s1 x1
	vertex(&p[396], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[208], &p[188], NULL, NULL, &scales[2], &factors[2]);
	// 87: V_WEYL_VECTOR_A c44 = c3@q12, c32@q2; s1 x0
	vertex(&p[400], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[136], &p[304], NULL, NULL, &scales[2], &factors[0]);
	// 88: V_WEYL_VECTOR_A c45 = c3@q12, c33@q2; s1 x0
	vertex(&p[404], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[136], &p[312], NULL, NULL, &scales[2], &factors[0]);
	// 89: CLOSURE a6 += c46 . c34; list 1 x0
	closure(&p[68], &p[144], &p[216], cf1, 2, &factors[0]);
	// 90: CLOSURE a7 += c38 . c47; list 1 x0
	closure(&p[70], &p[400], &p[140], cf1, 2, &factors[0]);
	// 91: CLOSURE a8 += c36 . c47; list 1 x0
	closure(&p[72], &p[248], &p[140], cf1, 2, &factors[0]);
	// 92: CLOSURE a9 += c46 . c35; list 1 x0
	closure(&p[74], &p[144], &p[244], cf1, 2, &factors[0]);
	// 93: CLOSURE a10 += c39 . c47; list 1 x0
	closure(&p[76], &p[404], &p[140], cf1, 2, &factors[0]);
	// 94: CLOSURE a11 += c37 . c47; list 1 x0
	closure(&p[78], &p[396], &p[140], cf1, 2, &factors[0]);
	// 95: REDUCE out[1] += w1 × |a6 (× x0)|²
	out[1] = reduce_term(out[1], &p[68], &factors[0], 0.45849874307927724);
	// 96: REDUCE out[30] += w1 × |a6 (× x0)|²
	out[30] = reduce_term(out[30], &p[68], &factors[0], 0.45849874307927724);
	// 97: REDUCE out[2] += w1 × |a7 (× x0)|²
	out[2] = reduce_term(out[2], &p[70], &factors[0], 0.45849874307927724);
	// 98: REDUCE out[29] += w1 × |a7 (× x0)|²
	out[29] = reduce_term(out[29], &p[70], &factors[0], 0.45849874307927724);
	// 99: REDUCE out[4] += w1 × |a8 (× x0)|²
	out[4] = reduce_term(out[4], &p[72], &factors[0], 0.45849874307927724);
	// 100: REDUCE out[27] += w1 × |a8 (× x0)|²
	out[27] = reduce_term(out[27], &p[72], &factors[0], 0.45849874307927724);
	// 101: REDUCE out[9] += w1 × |a9 (× x0)|²
	out[9] = reduce_term(out[9], &p[74], &factors[0], 0.45849874307927724);
	// 102: REDUCE out[22] += w1 × |a9 (× x0)|²
	out[22] = reduce_term(out[22], &p[74], &factors[0], 0.45849874307927724);
	// 103: REDUCE out[10] += w1 × |a10 (× x0)|²
	out[10] = reduce_term(out[10], &p[76], &factors[0], 0.45849874307927724);
	// 104: REDUCE out[21] += w1 × |a10 (× x0)|²
	out[21] = reduce_term(out[21], &p[76], &factors[0], 0.45849874307927724);
	// 105: REDUCE out[12] += w1 × |a11 (× x0)|²
	out[12] = reduce_term(out[12], &p[78], &factors[0], 0.45849874307927724);
	// 106: REDUCE out[19] += w1 × |a11 (× x0)|²
	out[19] = reduce_term(out[19], &p[78], &factors[0], 0.45849874307927724);
	}

int run(double *out, double *p, const double *momenta) {
	for(int i=0; i<456; i++) p[i] = 0.0;
	for(int h=0; h<32; h++) out[h] = 0.0;
	forms(p, momenta);
	body(out, p);
	return 0;
	}
