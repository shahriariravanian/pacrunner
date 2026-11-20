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

void forms(double *p, const double *momenta) {
	p[0] = 0.0 + -1.0 * momenta[0];
	p[1] = 0.0 + -1.0 * momenta[1];
	p[2] = 0.0 + -1.0 * momenta[2];
	p[3] = 0.0 + -1.0 * momenta[3];
	p[4] = 0.0 + -1.0 * momenta[0] + 1.0 * momenta[8];
	p[5] = 0.0 + -1.0 * momenta[1] + 1.0 * momenta[9];
	p[6] = 0.0 + -1.0 * momenta[2] + 1.0 * momenta[10];
	p[7] = 0.0 + -1.0 * momenta[3] + 1.0 * momenta[11];
	p[8] = 0.0 + -1.0 * momenta[0] + 1.0 * momenta[8] + 1.0 * momenta[12];
	p[9] = 0.0 + -1.0 * momenta[1] + 1.0 * momenta[9] + 1.0 * momenta[13];
	p[10] = 0.0 + -1.0 * momenta[2] + 1.0 * momenta[10] + 1.0 * momenta[14];
	p[11] = 0.0 + -1.0 * momenta[3] + 1.0 * momenta[11] + 1.0 * momenta[15];
	p[12] = 0.0 + -1.0 * momenta[0] + 1.0 * momenta[8] + 1.0 * momenta[12] + 1.0 * momenta[16];
	p[13] = 0.0 + -1.0 * momenta[1] + 1.0 * momenta[9] + 1.0 * momenta[13] + 1.0 * momenta[17];
	p[14] = 0.0 + -1.0 * momenta[2] + 1.0 * momenta[10] + 1.0 * momenta[14] + 1.0 * momenta[18];
	p[15] = 0.0 + -1.0 * momenta[3] + 1.0 * momenta[11] + 1.0 * momenta[15] + 1.0 * momenta[19];
	p[16] = 0.0 + -1.0 * momenta[0] + 1.0 * momenta[8] + 1.0 * momenta[16];
	p[17] = 0.0 + -1.0 * momenta[1] + 1.0 * momenta[9] + 1.0 * momenta[17];
	p[18] = 0.0 + -1.0 * momenta[2] + 1.0 * momenta[10] + 1.0 * momenta[18];
	p[19] = 0.0 + -1.0 * momenta[3] + 1.0 * momenta[11] + 1.0 * momenta[19];
	p[20] = 0.0 + -1.0 * momenta[0] + 1.0 * momenta[12];
	p[21] = 0.0 + -1.0 * momenta[1] + 1.0 * momenta[13];
	p[22] = 0.0 + -1.0 * momenta[2] + 1.0 * momenta[14];
	p[23] = 0.0 + -1.0 * momenta[3] + 1.0 * momenta[15];
	p[24] = 0.0 + -1.0 * momenta[0] + 1.0 * momenta[12] + 1.0 * momenta[16];
	p[25] = 0.0 + -1.0 * momenta[1] + 1.0 * momenta[13] + 1.0 * momenta[17];
	p[26] = 0.0 + -1.0 * momenta[2] + 1.0 * momenta[14] + 1.0 * momenta[18];
	p[27] = 0.0 + -1.0 * momenta[3] + 1.0 * momenta[15] + 1.0 * momenta[19];
	p[28] = 0.0 + -1.0 * momenta[0] + 1.0 * momenta[16];
	p[29] = 0.0 + -1.0 * momenta[1] + 1.0 * momenta[17];
	p[30] = 0.0 + -1.0 * momenta[2] + 1.0 * momenta[18];
	p[31] = 0.0 + -1.0 * momenta[3] + 1.0 * momenta[19];
	p[32] = 0.0 + -1.0 * momenta[4];
	p[33] = 0.0 + -1.0 * momenta[5];
	p[34] = 0.0 + -1.0 * momenta[6];
	p[35] = 0.0 + -1.0 * momenta[7];
	p[36] = 0.0 + 1.0 * momenta[8];
	p[37] = 0.0 + 1.0 * momenta[9];
	p[38] = 0.0 + 1.0 * momenta[10];
	p[39] = 0.0 + 1.0 * momenta[11];
	p[40] = 0.0 + 1.0 * momenta[8] + 1.0 * momenta[16];
	p[41] = 0.0 + 1.0 * momenta[9] + 1.0 * momenta[17];
	p[42] = 0.0 + 1.0 * momenta[10] + 1.0 * momenta[18];
	p[43] = 0.0 + 1.0 * momenta[11] + 1.0 * momenta[19];
	p[44] = 0.0 + 1.0 * momenta[12];
	p[45] = 0.0 + 1.0 * momenta[13];
	p[46] = 0.0 + 1.0 * momenta[14];
	p[47] = 0.0 + 1.0 * momenta[15];
	p[48] = 0.0 + 1.0 * momenta[12] + 1.0 * momenta[16];
	p[49] = 0.0 + 1.0 * momenta[13] + 1.0 * momenta[17];
	p[50] = 0.0 + 1.0 * momenta[14] + 1.0 * momenta[18];
	p[51] = 0.0 + 1.0 * momenta[15] + 1.0 * momenta[19];
	p[52] = 0.0 + 1.0 * momenta[16];
	p[53] = 0.0 + 1.0 * momenta[17];
	p[54] = 0.0 + 1.0 * momenta[18];
	p[55] = 0.0 + 1.0 * momenta[19];
	}

static const double cf1[] = {1.0, 0.0, 1.0, 0.0};
static const int outputs1[] = {0, 1, 2, 3};
static const int outputs2[] = {0, 1, 2, 3};

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
	// 25: SOURCE c0@q8 = WEYL antiparticle, helicity 1, chirality -1, mass s0; x0
	source(&p[112], FAMILY_WEYL, ANTIPARTICLE, &p[32], 1, -1, 0.0, &factors[0]);
	// 26: SOURCE c1@q9 = WEYL, helicity -1, chirality -1, mass s0; x0
	source(&p[116], FAMILY_WEYL, PARTICLE, &p[36], -1, -1, 0.0, &factors[0]);
	// 27: SOURCE c2@q11 = WEYL, helicity -1, chirality -1, mass s0; x0
	source(&p[124], FAMILY_WEYL, PARTICLE, &p[44], -1, -1, 0.0, &factors[0]);
	// 28: SOURCE c3@q11 = WEYL, helicity 1, chirality 1, mass s0; x0
	source(&p[128], FAMILY_WEYL, PARTICLE, &p[44], 1, 1, 0.0, &factors[0]);
	// 29: SOURCE c4@q9 = WEYL, helicity 1, chirality 1, mass s0; x0
	source(&p[120], FAMILY_WEYL, PARTICLE, &p[36], 1, 1, 0.0, &factors[0]);
	// 30: SOURCE c5@q13 = VECTOR, helicity -1, chirality 0, mass s0; x0
	source(&p[132], FAMILY_VECTOR, PARTICLE, &p[52], -1, 0, 0.0, &factors[0]);
	// 31: SOURCE c6@q13 = VECTOR, helicity 1, chirality 0, mass s0; x0
	source(&p[140], FAMILY_VECTOR, PARTICLE, &p[52], 1, 0, 0.0, &factors[0]);
	// 32: SOURCE c7@q8 = WEYL antiparticle, helicity -1, chirality 1, mass s0; x0
	source(&p[108], FAMILY_WEYL, ANTIPARTICLE, &p[32], -1, 1, 0.0, &factors[0]);
	// 33: SOURCE c8@q0 = WEYL antiparticle, helicity -1, chirality 1, mass s0; x0
	source(&p[104], FAMILY_WEYL, ANTIPARTICLE, &p[0], -1, 1, 0.0, &factors[0]);
	// 34: CLEAR c9, 4 components
	clear(&p[148], 4);
	// 35: CLEAR c10, 4 components
	clear(&p[156], 4);
	// 36: CLEAR c11, 2 components
	clear(&p[164], 2);
	// 37: CLEAR c12, 2 components
	clear(&p[168], 2);
	// 38: CLEAR c13, 2 components
	clear(&p[172], 2);
	// 39: CLEAR c14, 2 components
	clear(&p[176], 2);
	// 40: KERNEL c9 = c15, c16; binding instance 1 x1
	adapter_k_5acfa4bfbabd18e1(&p[148], p[116], p[117], p[118], p[119], p[104], p[105], p[106], p[107], outputs1, 0, MODE_SET, &factors[2]);
	// 41: V_WEYL_VECTOR_B c17@q12 = c2@q11, c5@q13; s1 x0
	vertex(&p[164], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[124], &p[132], NULL, NULL, &scales[2], &factors[0]);
	// 42: V_WEYL_VECTOR_B c18@q12 = c2@q11, c6@q13; s1 x0
	vertex(&p[168], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[124], &p[140], NULL, NULL, &scales[2], &factors[0]);
	// 43: KERNEL c10 = c16, c19; binding instance 2 x0
	adapter_k_e2d4ab976ad89705(&p[156], p[104], p[105], p[106], p[107], p[124], p[125], p[126], p[127], outputs2, 0, MODE_SET, &factors[0]);
	// 44: V_WEYL_VECTOR_A c20@q12 = c3@q11, c5@q13; s1 x0
	vertex(&p[172], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[128], &p[132], NULL, NULL, &scales[2], &factors[0]);
	// 45: V_WEYL_VECTOR_A c21@q12 = c3@q11, c6@q13; s1 x0
	vertex(&p[176], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[128], &p[140], NULL, NULL, &scales[2], &factors[0]);
	// 46: P_VECTOR c22@q1; x0
	propagator(&p[148], PROPAGATOR_VECTOR, &p[148], &p[4], &factors[0]);
	// 47: P_VECTOR c23@q5; x0
	propagator(&p[156], PROPAGATOR_VECTOR, &p[156], &p[20], &factors[0]);
	// 48: P_WEYL_A c17@q12; x0
	propagator(&p[164], PROPAGATOR_WEYL_A, &p[164], &p[48], &factors[0]);
	// 49: P_WEYL_A c18@q12; x0
	propagator(&p[168], PROPAGATOR_WEYL_A, &p[168], &p[48], &factors[0]);
	// 50: P_WEYL_B c20@q12; x0
	propagator(&p[172], PROPAGATOR_WEYL_B, &p[172], &p[48], &factors[0]);
	// 51: P_WEYL_B c21@q12; x0
	propagator(&p[176], PROPAGATOR_WEYL_B, &p[176], &p[48], &factors[0]);
	// 52: CLEAR c24, 4 components
	clear(&p[276], 4);
	// 53: CLEAR c25, 4 components
	clear(&p[284], 4);
	// 54: CLEAR c26, 2 components
	clear(&p[292], 2);
	// 55: CLEAR c27, 2 components
	clear(&p[296], 2);
	// 56: CLEAR c28, 2 components
	clear(&p[300], 2);
	// 57: V_WEYL_VECTOR_B c29@q2 = c2@q11, c22@q1; s1 x0
	vertex(&p[292], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[124], &p[148], NULL, NULL, &scales[2], &factors[0]);
	// 58: V_WEYL_VECTOR_B c29@q2 += c1@q9, c23@q5; s1 x1
	vertex(&p[292], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[116], &p[156], NULL, NULL, &scales[2], &factors[2]);
	// 59: KERNEL c24 = c16, c11; binding instance 2 x0
	adapter_k_e2d4ab976ad89705(&p[276], p[104], p[105], p[106], p[107], p[164], p[165], p[166], p[167], outputs2, 0, MODE_SET, &factors[0]);
	// 60: KERNEL c25 = c16, c12; binding instance 2 x0
	adapter_k_e2d4ab976ad89705(&p[284], p[104], p[105], p[106], p[107], p[168], p[169], p[170], p[171], outputs2, 0, MODE_SET, &factors[0]);
	// 61: V_WEYL_VECTOR_A c30@q2 = c4@q9, c23@q5; s1 x1
	vertex(&p[296], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[120], &p[156], NULL, NULL, &scales[2], &factors[2]);
	// 62: V_THREE_VECTOR c31@q6 += c5@q13, c23@q5; s2 x1
	vertex(&p[276], VERTEX_THREE_VECTOR, MODE_ADD, &p[132], &p[156], &p[52], &p[20], &scales[4], &factors[2]);
	// 63: V_THREE_VECTOR c32@q6 += c6@q13, c23@q5; s2 x1
	vertex(&p[284], VERTEX_THREE_VECTOR, MODE_ADD, &p[140], &p[156], &p[52], &p[20], &scales[4], &factors[2]);
	// 64: V_WEYL_VECTOR_A c33@q2 = c3@q11, c22@q1; s1 x0
	vertex(&p[300], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[128], &p[148], NULL, NULL, &scales[2], &factors[0]);
	// 65: P_VECTOR c31@q6; x0
	propagator(&p[276], PROPAGATOR_VECTOR, &p[276], &p[24], &factors[0]);
	// 66: P_VECTOR c32@q6; x0
	propagator(&p[284], PROPAGATOR_VECTOR, &p[284], &p[24], &factors[0]);
	// 67: P_WEYL_A c29@q2; x0
	propagator(&p[292], PROPAGATOR_WEYL_A, &p[292], &p[8], &factors[0]);
	// 68: P_WEYL_B c33@q2; x0
	propagator(&p[300], PROPAGATOR_WEYL_B, &p[300], &p[8], &factors[0]);
	// 69: P_WEYL_B c30@q2; x0
	propagator(&p[296], PROPAGATOR_WEYL_B, &p[296], &p[8], &factors[0]);
	// 70: CLEAR c16, 2 components
	clear(&p[104], 2);
	// 71: CLEAR c34, 2 components
	clear(&p[156], 2);
	// 72: CLEAR c35, 2 components
	clear(&p[160], 2);
	// 73: CLEAR c36, 2 components
	clear(&p[188], 2);
	// 74: CLEAR c37, 2 components
	clear(&p[192], 2);
	// 75: CLEAR c38, 2 components
	clear(&p[220], 2);
	// 76: V_WEYL_VECTOR_B c8@q0 = c17@q12, c22@q1; s1 x0
	vertex(&p[104], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[164], &p[148], NULL, NULL, &scales[2], &factors[0]);
	// 77: V_WEYL_VECTOR_B c39 = c18@q12, c22@q1; s1 x0
	vertex(&p[156], VERTEX_WEYL_VECTOR_B, MODE_SET, &p[168], &p[148], NULL, NULL, &scales[2], &factors[0]);
	// 78: V_WEYL_VECTOR_B c8@q0 += c1@q9, c31@q6; s1 x1
	vertex(&p[104], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[116], &p[276], NULL, NULL, &scales[2], &factors[2]);
	// 79: V_WEYL_VECTOR_B c8@q0 += c29@q2, c5@q13; s1 x0
	vertex(&p[104], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[292], &p[132], NULL, NULL, &scales[2], &factors[0]);
	// 80: V_WEYL_VECTOR_B c39 += c1@q9, c32@q6; s1 x1
	vertex(&p[156], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[116], &p[284], NULL, NULL, &scales[2], &factors[2]);
	// 81: V_WEYL_VECTOR_B c39 += c29@q2, c6@q13; s1 x0
	vertex(&p[156], VERTEX_WEYL_VECTOR_B, MODE_ADD, &p[292], &p[140], NULL, NULL, &scales[2], &factors[0]);
	// 82: V_WEYL_VECTOR_A c40 = c33@q2, c5@q13; s1 x0
	vertex(&p[192], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[300], &p[132], NULL, NULL, &scales[2], &factors[0]);
	// 83: V_WEYL_VECTOR_A c41 = c33@q2, c6@q13; s1 x0
	vertex(&p[220], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[300], &p[140], NULL, NULL, &scales[2], &factors[0]);
	// 84: V_WEYL_VECTOR_A c42 = c4@q9, c31@q6; s1 x1
	vertex(&p[160], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[120], &p[276], NULL, NULL, &scales[2], &factors[2]);
	// 85: V_WEYL_VECTOR_A c42 += c30@q2, c5@q13; s1 x0
	vertex(&p[160], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[296], &p[132], NULL, NULL, &scales[2], &factors[0]);
	// 86: V_WEYL_VECTOR_A c43 = c4@q9, c32@q6; s1 x1
	vertex(&p[188], VERTEX_WEYL_VECTOR_A, MODE_SET, &p[120], &p[284], NULL, NULL, &scales[2], &factors[2]);
	// 87: V_WEYL_VECTOR_A c43 += c30@q2, c6@q13; s1 x0
	vertex(&p[188], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[296], &p[140], NULL, NULL, &scales[2], &factors[0]);
	// 88: V_WEYL_VECTOR_A c40 += c20@q12, c22@q1; s1 x0
	vertex(&p[192], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[172], &p[148], NULL, NULL, &scales[2], &factors[0]);
	// 89: V_WEYL_VECTOR_A c41 += c21@q12, c22@q1; s1 x0
	vertex(&p[220], VERTEX_WEYL_VECTOR_A, MODE_ADD, &p[176], &p[148], NULL, NULL, &scales[2], &factors[0]);
	// 90: CLOSURE a0 += c44 . c16; list 1 x0
	closure(&p[56], &p[108], &p[104], cf1, 2, &factors[0]);
	// 91: CLOSURE a1 += c44 . c34; list 1 x0
	closure(&p[58], &p[108], &p[156], cf1, 2, &factors[0]);
	// 92: CLOSURE a2 += c37 . c45; list 1 x0
	closure(&p[60], &p[192], &p[112], cf1, 2, &factors[0]);
	// 93: CLOSURE a3 += c38 . c45; list 1 x0
	closure(&p[62], &p[220], &p[112], cf1, 2, &factors[0]);
	// 94: CLOSURE a4 += c35 . c45; list 1 x0
	closure(&p[64], &p[160], &p[112], cf1, 2, &factors[0]);
	// 95: CLOSURE a5 += c36 . c45; list 1 x0
	closure(&p[66], &p[188], &p[112], cf1, 2, &factors[0]);
	// 96: REDUCE out[0] += w1 × |a0 (× x0)|²
	out[0] = reduce_term(out[0], &p[56], &factors[0], 1.2226633148780726);
	// 97: REDUCE out[31] += w1 × |a0 (× x0)|²
	out[31] = reduce_term(out[31], &p[56], &factors[0], 1.2226633148780726);
	// 98: REDUCE out[1] += w1 × |a1 (× x0)|²
	out[1] = reduce_term(out[1], &p[58], &factors[0], 1.2226633148780726);
	// 99: REDUCE out[30] += w1 × |a1 (× x0)|²
	out[30] = reduce_term(out[30], &p[58], &factors[0], 1.2226633148780726);
	// 100: REDUCE out[10] += w1 × |a2 (× x0)|²
	out[10] = reduce_term(out[10], &p[60], &factors[0], 1.2226633148780726);
	// 101: REDUCE out[21] += w1 × |a2 (× x0)|²
	out[21] = reduce_term(out[21], &p[60], &factors[0], 1.2226633148780726);
	// 102: REDUCE out[11] += w1 × |a3 (× x0)|²
	out[11] = reduce_term(out[11], &p[62], &factors[0], 1.2226633148780726);
	// 103: REDUCE out[20] += w1 × |a3 (× x0)|²
	out[20] = reduce_term(out[20], &p[62], &factors[0], 1.2226633148780726);
	// 104: REDUCE out[12] += w1 × |a4 (× x0)|²
	out[12] = reduce_term(out[12], &p[64], &factors[0], 1.2226633148780726);
	// 105: REDUCE out[19] += w1 × |a4 (× x0)|²
	out[19] = reduce_term(out[19], &p[64], &factors[0], 1.2226633148780726);
	// 106: REDUCE out[13] += w1 × |a5 (× x0)|²
	out[13] = reduce_term(out[13], &p[66], &factors[0], 1.2226633148780726);
	// 107: REDUCE out[18] += w1 × |a5 (× x0)|²
	out[18] = reduce_term(out[18], &p[66], &factors[0], 1.2226633148780726);
	}

int run(double *out, double *p, const double *momenta) {
	for(int i=0; i<456; i++) p[i] = 0.0;
	for(int h=0; h<32; h++) out[h] = 0.0;
	forms(p, momenta);
	body(out, p);
	return 0;
	}
