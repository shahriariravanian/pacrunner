/* p_p_to_j_j_j_101 flow:3,2,5,1,4 on the UIR interpreter (uirvm.c), compiled by PacRunner's compile */
#include <stdlib.h>
#include "uirvm.c"
#ifdef __CUDACC__
#include "pacops.c"  /* one translation unit for the device */
#endif
#include "runner.h"

UIR_DATA uint32_t rows[] = {
	0x0000000eu, 0x00000000u, 0x0000020eu, 0x00000000u, 0x0000040eu, 0x00000000u, 0x0000060eu, 0x00000000u,
	0x0000080eu, 0x00000000u, 0x00000a0eu, 0x00000000u, 0x00000c0eu, 0x00000000u, 0x00000e0eu, 0x00000000u,
	0x0000100eu, 0x00000000u, 0x0000120eu, 0x00000000u, 0x0000140eu, 0x00000000u, 0x0000160eu, 0x00000000u,
	0x0000180eu, 0x00000000u, 0x00001a0eu, 0x00000000u, 0x00001c0eu, 0x00000000u, 0x00001e0eu, 0x00000000u,
	0x0000200eu, 0x00000000u, 0x0000220eu, 0x00000000u, 0x0000240eu, 0x00000000u, 0x0000260eu, 0x00000000u,
	0x0000280eu, 0x00000000u, 0x00002a0eu, 0x00000000u, 0x00002c0eu, 0x00000000u, 0x00002e0eu, 0x00000000u,
	0x0000000du, 0x03ff013au, 0x0000020du, 0x03ff0012u, 0x0000040du, 0x03ff0012u, 0x0000060du, 0x03ff0132u,
	0x0000080du, 0x03ff0132u, 0x00000a0du, 0x03ff00b4u, 0x00000c0du, 0x03ff0094u, 0x00000e0du, 0x03ff001au,
	0x0000100du, 0x03ff011au, 0x0000120au, 0x00000004u, 0x0000140au, 0x00000004u, 0x0000160au, 0x00000002u,
	0x0000180au, 0x00000002u, 0x00001a0au, 0x00000002u, 0x00001c0au, 0x00000002u, 0x0000124bu, 0x0402000fu,
	0x004023c1u, 0x0000c002u, 0x004025c1u, 0x0000a002u, 0x0040144bu, 0x00026010u, 0x004029c0u, 0x0000c003u,
	0x00402bc0u, 0x0000a003u, 0x00002c07u, 0x00000000u, 0x00002e07u, 0x00000000u, 0x00002205u, 0x00000000u,
	0x00002405u, 0x00000000u, 0x00002806u, 0x00000000u, 0x00002a06u, 0x00000000u, 0x0000300au, 0x00000004u,
	0x0000320au, 0x00000004u, 0x0000340au, 0x00000002u, 0x0000360au, 0x00000002u, 0x0000380au, 0x00000002u,
	0x00403bc1u, 0x0002c002u, 0x00403b81u, 0x0402e001u, 0x0040304bu, 0x00016010u, 0x0040324bu, 0x0001a010u,
	0x00403dc0u, 0x0402e004u, 0x00803f84u, 0x0402e006u, 0x00804184u, 0x0402e005u, 0x004043c0u, 0x0002c003u,
	0x00003e07u, 0x00000000u, 0x00004007u, 0x00000000u, 0x00003a05u, 0x00000000u, 0x00004206u, 0x00000000u,
	0x00003c06u, 0x00000000u, 0x0000200au, 0x00000002u, 0x0000440au, 0x00000002u, 0x0000460au, 0x00000002u,
	0x0000480au, 0x00000002u, 0x00004a0au, 0x00000002u, 0x00004c0au, 0x00000002u, 0x004011c1u, 0x0002c011u,
	0x00404fc1u, 0x0002c012u, 0x00401181u, 0x0403e001u, 0x00401181u, 0x0000c01du, 0x00404f81u, 0x04040001u,
	0x00404f81u, 0x0000a01du, 0x004051c0u, 0x0000c021u, 0x004053c0u, 0x0403e004u, 0x00405380u, 0x0000c01eu,
	0x004055c0u, 0x0000a021u, 0x004057c0u, 0x04040004u, 0x00405780u, 0x0000a01eu, 0x00405180u, 0x0002c014u,
	0x00405580u, 0x0002c015u, 0x00000008u, 0x0002002cu, 0x00000208u, 0x0005a023u, 0x00000408u, 0x0005a022u,
	0x00000608u, 0x0004802cu, 0x00000808u, 0x0005a026u, 0x00000a08u, 0x0005a025u, 0x00000211u, 0x00000000u,
	0x00003c11u, 0x00000000u, 0x00400411u, 0x00000000u, 0x00403a11u, 0x00000000u, 0x00800811u, 0x00000000u,
	0x00803611u, 0x00000000u, 0x00c01211u, 0x00000000u, 0x00c02c11u, 0x00000000u, 0x01001411u, 0x00000000u,
	0x01002a11u, 0x00000000u, 0x01401811u, 0x00000000u, 0x01402611u, 0x00000000u, 0x0000001fu, 0x00000000u,
	};
UIR_DATA uint32_t edges[] = {
	285638666, 285573125, 285605895, 285605896, 285573126, 570687491, 570687489, 285638665,
	285212672, 41910283, 41910285, 25133071, 25133072, 25133073, 25133074, 25133061,
	25133056, 285540367, 285540369, 25133063, 285540368, 285540370, 570589195, 570654733,
	41910315, 41910317, 25133103, 25133104, 25133105, 285409327, 285409328, 570556459,
	570556461, 285409329, 25133069, 25133070, 25133075, 25133076, 25133083, 293568531,
	293568526, 293568525, 293568539, 293568532, 25133066, 25133065,
	};
UIR_DATA double scales[] = {
	0.0, 0.0, 0.7071067811865476, 0.0,
	0.0, 0.7071067811865476,
	};
UIR_DATA double factors[] = {
	1.0, 0.0, -1.0, 0.0,
	};
UIR_DATA uint32_t form_first[] = {
	0, 1, 3, 6, 10, 13, 15, 18,
	20, 21, 23, 25, 26, 27, 28,
	};
UIR_DATA uint32_t term_leg[] = {
	0, 0, 1, 0, 1, 2, 0, 1,
	2, 3, 0, 1, 3, 0, 2, 0,
	2, 3, 0, 3, 1, 1, 2, 1,
	3, 2, 3, 4,
	};
UIR_DATA double term_coefficient[] = {
	-1.0, -1.0, -1.0, -1.0,
	-1.0, 1.0, -1.0, -1.0,
	1.0, 1.0, -1.0, -1.0,
	1.0, -1.0, 1.0, -1.0,
	1.0, 1.0, -1.0, 1.0,
	-1.0, -1.0, 1.0, -1.0,
	1.0, 1.0, 1.0, 1.0,
	};
UIR_DATA uint32_t crossing_leg[] = {
	0, 1, 3, 4, 2,
	};
UIR_DATA double crossing_sign[] = {
	1.0, 1.0, 1.0, 1.0,
	1.0,
	};
UIR_DATA uint32_t list_first[] = {
	0, 2,
	};
UIR_DATA uint32_t list_factor[] = {
	0, 0,
	};
UIR_DATA uint32_t embedding_first[] = {
	0,
	};
UIR_DATA int32_t embedding_component[1] = {0};
UIR_DATA int32_t embedding_wave[1] = {0};
UIR_DATA uint32_t embedding_factor[1] = {0};
UIR_DATA uint32_t scaleset_first[] = {
	0, 0,
	};
UIR_DATA uint32_t scaleset_scale[1] = {0};
UIR_DATA uint32_t instance[] = {
	0, 0, 1, 0,
	};
UIR_DATA double weights[] = {
	0.45849874307927724,
	};
#ifdef PAC_MIXED
UIR_DATA pac_real scales_r[] = {
	0.0, 0.0, 0.7071067811865476, 0.0,
	0.0, 0.7071067811865476,
	};
UIR_DATA pac_real factors_r[] = {
	1.0, 0.0, -1.0, 0.0,
	};
#else
#define scales_r scales
#define factors_r factors
#endif

UIR_DATA uir_program program = {
	UIR_VERSION, 5, 14, 176, 24, 32,
	rows, 108, edges, scales, factors,
	form_first, term_leg, term_coefficient, crossing_leg, crossing_sign,
	list_first, list_factor,
	embedding_first, embedding_component, embedding_wave, embedding_factor,
	scaleset_first, scaleset_scale, instance,
	weights, 1L, 1L,
	scales_r, factors_r
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
#ifdef PAC_MIXED
/* the symjit kernel k_5acfa4bfbabd18e1 in mixed precision (kernel_c.jl's kernel_c_mixed): 82 f32 and 17 f64 operations, 48 conversions */
PAC_FN static int k_5acfa4bfbabd18e1(double *mem) {
	double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0, r0 = 0.0, r1 = 0.0, stack2 = 0.0, stack144 = 0.0, stack145 = 0.0;
	float g0_f = 0.0f, g1_f = 0.0f, g2_f = 0.0f, g3_f = 0.0f, g4_f = 0.0f, g5_f = 0.0f, g6_f = 0.0f, g7_f = 0.0f, g8_f = 0.0f, g9_f = 0.0f, r0_f = 0.0f, r1_f = 0.0f, g10_f = 0.0f, g11_f = 0.0f, g12_f = 0.0f, g13_f = 0.0f, stack144_f = 0.0f, stack145_f = 0.0f, stack146_f = 0.0f, stack147_f = 0.0f, stack148_f = 0.0f, stack149_f = 0.0f, stack150_f = 0.0f, stack151_f = 0.0f, stack152_f = 0.0f, stack153_f = 0.0f, stack154_f = 0.0f, stack155_f = 0.0f, stack156_f = 0.0f, stack157_f = 0.0f;
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
	g3_f = (float)g3 * ((float)0.5);
	g0 = sj_mask((double)g3_f == (double)g3_f);
	g3 = sj_and((double)g3_f, g0);
	g0 = stack2 != 0.0 ? g2 : g3;
	g1 = stack2 != 0.0 ? g3 : g2;
	g2 = sj_sign(g0);
	r0 = sj_xor(g0, g2);
	r1 = sj_xor(g1, g2);
	goto L_return;
	L__40_jump_over_complex_root:;
	stack144 = g4;
	stack145 = g5;
	r0_f = (float)mem[0];
	r1_f = (float)mem[1];
	g6_f = (float)mem[2];
	g7_f = (float)mem[3];
	g0_f = (float)mem[4];
	g1_f = (float)mem[5];
	g2_f = r1_f * g1_f;
	g2_f = fmaf(r0_f, g0_f, -g2_f);
	g3_f = r0_f * g1_f;
	r1_f = fmaf(g0_f, r1_f, g3_f);
	r0_f = g2_f;
	g0_f = (float)mem[6];
	g1_f = (float)mem[7];
	g2_f = g7_f * g1_f;
	g2_f = fmaf(g6_f, g0_f, -g2_f);
	g3_f = g6_f * g1_f;
	g9_f = fmaf(g0_f, g7_f, g3_f);
	g8_f = g2_f;
	stack146_f = r0_f;
	stack147_f = r1_f;
	stack148_f = g8_f;
	stack149_f = g9_f;
	g10_f = ((float)0.0);
	g12_f = ((float)0.3333333333333333);
	g11_f = g12_f;
	g10_f = g10_f;
	g12_f = stack146_f;
	g13_f = stack147_f;
	g0_f = stack148_f;
	g1_f = stack149_f;
	g4_f = g12_f + g0_f;
	g5_f = g13_f + g1_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g5_f * (float)g1;
	g2_f = fmaf(g4_f, (float)g0, -g2_f);
	g3_f = g4_f * (float)g1;
	g5_f = fmaf((float)g0, g5_f, g3_f);
	g4_f = g2_f;
	g2_f = g11_f * g5_f;
	g2_f = fmaf(g10_f, g4_f, -g2_f);
	g3_f = g10_f * g5_f;
	g5_f = fmaf(g4_f, g11_f, g3_f);
	g4_f = g2_f;
	stack150_f = g4_f;
	stack151_f = g5_f;
	r0_f = (float)mem[0];
	r1_f = (float)mem[1];
	g0_f = (float)mem[6];
	g1_f = (float)mem[7];
	g2_f = r1_f * g1_f;
	g2_f = fmaf(r0_f, g0_f, -g2_f);
	g3_f = r0_f * g1_f;
	r1_f = fmaf(g0_f, r1_f, g3_f);
	r0_f = g2_f;
	g0_f = (float)mem[4];
	g1_f = (float)mem[5];
	g2_f = g7_f * g1_f;
	g2_f = fmaf(g6_f, g0_f, -g2_f);
	g3_f = g6_f * g1_f;
	g11_f = fmaf(g0_f, g7_f, g3_f);
	g10_f = g2_f;
	stack152_f = r0_f;
	stack153_f = r1_f;
	stack154_f = g10_f;
	stack155_f = g11_f;
	g4_f = ((float)0.0);
	g6_f = ((float)-0.3333333333333333);
	g5_f = g6_f;
	g4_f = g4_f;
	g6_f = stack152_f;
	g7_f = stack153_f;
	g0_f = stack154_f;
	g1_f = stack155_f;
	g6_f = g6_f + g0_f;
	g7_f = g7_f + g1_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g7_f * (float)g1;
	g2_f = fmaf(g6_f, (float)g0, -g2_f);
	g3_f = g6_f * (float)g1;
	g7_f = fmaf((float)g0, g7_f, g3_f);
	g6_f = g2_f;
	g2_f = g5_f * g7_f;
	g2_f = fmaf(g4_f, g6_f, -g2_f);
	g3_f = g4_f * g7_f;
	g5_f = fmaf(g6_f, g5_f, g3_f);
	g4_f = g2_f;
	stack156_f = g4_f;
	stack157_f = g5_f;
	g6_f = stack152_f;
	g7_f = stack153_f;
	g0_f = stack154_f;
	g1_f = stack155_f;
	g4_f = g6_f - g0_f;
	g5_f = g7_f - g1_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g5_f * (float)g1;
	g2_f = fmaf(g4_f, (float)g0, -g2_f);
	g3_f = g4_f * (float)g1;
	g5_f = fmaf((float)g0, g5_f, g3_f);
	g4_f = g2_f;
	g0_f = ((float)-0.3333333333333333);
	g5_f = g5_f * g0_f;
	g4_f = g4_f * g0_f;
	stack154_f = g4_f;
	stack155_f = g5_f;
	g4_f = ((float)0.0);
	g6_f = ((float)-0.3333333333333333);
	g5_f = g6_f;
	g4_f = g4_f;
	g0_f = stack146_f;
	g1_f = stack147_f;
	g2_f = g5_f * g1_f;
	g2_f = fmaf(g4_f, g0_f, -g2_f);
	g3_f = g4_f * g1_f;
	g5_f = fmaf(g0_f, g5_f, g3_f);
	g4_f = g2_f;
	g6_f = ((float)0.0);
	g8_f = ((float)0.3333333333333333);
	g7_f = g8_f;
	g6_f = g6_f;
	r0_f = stack148_f;
	r1_f = stack149_f;
	g2_f = g7_f * r1_f;
	g2_f = fmaf(g6_f, r0_f, -g2_f);
	g3_f = g6_f * r1_f;
	r1_f = fmaf(r0_f, g7_f, g3_f);
	r0_f = g2_f;
	g4_f = r0_f + g4_f;
	g5_f = r1_f + g5_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g5_f * (float)g1;
	g2_f = fmaf(g4_f, (float)g0, -g2_f);
	g3_f = g4_f * (float)g1;
	g5_f = fmaf((float)g0, g5_f, g3_f);
	g4_f = g2_f;
	stack144_f = g4_f;
	stack145_f = g5_f;
	stack144_f = g4_f;
	stack145_f = g5_f;
	g6_f = stack150_f;
	g7_f = stack151_f;
	mem[8] = (double)g6_f;
	mem[9] = (double)g7_f;
	mem[8] = (double)g6_f;
	mem[9] = (double)g7_f;
	g8_f = stack156_f;
	g9_f = stack157_f;
	mem[10] = (double)g8_f;
	mem[11] = (double)g9_f;
	mem[10] = (double)g8_f;
	mem[11] = (double)g9_f;
	g4_f = stack154_f;
	g5_f = stack155_f;
	mem[12] = (double)g4_f;
	mem[13] = (double)g5_f;
	mem[12] = (double)g4_f;
	mem[13] = (double)g5_f;
	g4_f = stack144_f;
	g5_f = stack145_f;
	mem[14] = (double)g4_f;
	mem[15] = (double)g5_f;
	return 0;
	L_return:
	switch (ret) {
		case 0: goto R_0;
		}
	return -1;  /* a return outside a funclet */
	}
#else
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
#endif

#ifdef PAC_MIXED
/* the symjit kernel k_e2d4ab976ad89705 in mixed precision (kernel_c.jl's kernel_c_mixed): 59 f32 and 17 f64 operations, 48 conversions */
PAC_FN static int k_e2d4ab976ad89705(double *mem) {
	double g0 = 0.0, g1 = 0.0, g2 = 0.0, g3 = 0.0, g4 = 0.0, g5 = 0.0, r0 = 0.0, r1 = 0.0, stack2 = 0.0, stack144 = 0.0, stack145 = 0.0;
	float g0_f = 0.0f, g1_f = 0.0f, g2_f = 0.0f, g3_f = 0.0f, g4_f = 0.0f, g5_f = 0.0f, g6_f = 0.0f, g7_f = 0.0f, g8_f = 0.0f, g9_f = 0.0f, r0_f = 0.0f, r1_f = 0.0f, g10_f = 0.0f, g11_f = 0.0f, g12_f = 0.0f, g13_f = 0.0f, stack144_f = 0.0f, stack145_f = 0.0f, stack146_f = 0.0f, stack147_f = 0.0f, stack148_f = 0.0f, stack149_f = 0.0f, stack150_f = 0.0f, stack151_f = 0.0f, stack152_f = 0.0f, stack153_f = 0.0f;
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
	g3_f = (float)g3 * ((float)0.5);
	g0 = sj_mask((double)g3_f == (double)g3_f);
	g3 = sj_and((double)g3_f, g0);
	g0 = stack2 != 0.0 ? g2 : g3;
	g1 = stack2 != 0.0 ? g3 : g2;
	g2 = sj_sign(g0);
	r0 = sj_xor(g0, g2);
	r1 = sj_xor(g1, g2);
	goto L_return;
	L__40_jump_over_complex_root:;
	stack144 = g4;
	stack145 = g5;
	r0_f = (float)mem[0];
	r1_f = (float)mem[1];
	g6_f = (float)mem[2];
	g7_f = (float)mem[3];
	g0_f = (float)mem[4];
	g1_f = (float)mem[5];
	g2_f = r1_f * g1_f;
	g2_f = fmaf(r0_f, g0_f, -g2_f);
	g3_f = r0_f * g1_f;
	r1_f = fmaf(g0_f, r1_f, g3_f);
	r0_f = g2_f;
	g0_f = (float)mem[6];
	g1_f = (float)mem[7];
	g2_f = g7_f * g1_f;
	g2_f = fmaf(g6_f, g0_f, -g2_f);
	g3_f = g6_f * g1_f;
	g9_f = fmaf(g0_f, g7_f, g3_f);
	g8_f = g2_f;
	stack146_f = r0_f;
	stack147_f = r1_f;
	r0_f = stack146_f;
	r1_f = stack147_f;
	g10_f = r0_f + g8_f;
	g11_f = r1_f + g9_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g11_f * (float)g1;
	g2_f = fmaf(g10_f, (float)g0, -g2_f);
	g3_f = g10_f * (float)g1;
	g11_f = fmaf((float)g0, g11_f, g3_f);
	g10_f = g2_f;
	g3_f = -g11_f;
	g11_f = g10_f;
	g10_f = g3_f;
	r0_f = (float)mem[0];
	r1_f = (float)mem[1];
	g0_f = (float)mem[6];
	g1_f = (float)mem[7];
	g2_f = r1_f * g1_f;
	g2_f = fmaf(r0_f, g0_f, -g2_f);
	g3_f = r0_f * g1_f;
	r1_f = fmaf(g0_f, r1_f, g3_f);
	r0_f = g2_f;
	g0_f = (float)mem[4];
	g1_f = (float)mem[5];
	g2_f = g7_f * g1_f;
	g2_f = fmaf(g6_f, g0_f, -g2_f);
	g3_f = g6_f * g1_f;
	g13_f = fmaf(g0_f, g7_f, g3_f);
	g12_f = g2_f;
	stack148_f = r0_f;
	stack149_f = r1_f;
	stack150_f = g12_f;
	stack151_f = g13_f;
	r0_f = stack148_f;
	r1_f = stack149_f;
	g4_f = r0_f + g12_f;
	g5_f = r1_f + g13_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g5_f * (float)g1;
	g2_f = fmaf(g4_f, (float)g0, -g2_f);
	g3_f = g4_f * (float)g1;
	g5_f = fmaf((float)g0, g5_f, g3_f);
	g4_f = g2_f;
	g3_f = g5_f;
	g5_f = -g4_f;
	g4_f = g3_f;
	stack152_f = g4_f;
	stack153_f = g5_f;
	g4_f = stack148_f;
	g5_f = stack149_f;
	g0_f = stack150_f;
	g1_f = stack151_f;
	g4_f = g4_f - g0_f;
	g5_f = g5_f - g1_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g5_f * (float)g1;
	g2_f = fmaf(g4_f, (float)g0, -g2_f);
	g3_f = g4_f * (float)g1;
	g5_f = fmaf((float)g0, g5_f, g3_f);
	g4_f = g2_f;
	stack150_f = g4_f;
	stack151_f = g5_f;
	g4_f = stack146_f;
	g5_f = stack147_f;
	g3_f = g5_f;
	g5_f = -g4_f;
	g4_f = g3_f;
	g3_f = -g9_f;
	g7_f = g8_f;
	g6_f = g3_f;
	g4_f = g4_f + g6_f;
	g5_f = g5_f + g7_f;
	g0 = stack144;
	g1 = stack145;
	g2_f = g5_f * (float)g1;
	g2_f = fmaf(g4_f, (float)g0, -g2_f);
	g3_f = g4_f * (float)g1;
	g5_f = fmaf((float)g0, g5_f, g3_f);
	g4_f = g2_f;
	stack144_f = g4_f;
	stack145_f = g5_f;
	stack144_f = g4_f;
	stack145_f = g5_f;
	mem[8] = (double)g10_f;
	mem[9] = (double)g11_f;
	mem[8] = (double)g10_f;
	mem[9] = (double)g11_f;
	g6_f = stack152_f;
	g7_f = stack153_f;
	mem[10] = (double)g6_f;
	mem[11] = (double)g7_f;
	mem[10] = (double)g6_f;
	mem[11] = (double)g7_f;
	g4_f = stack150_f;
	g5_f = stack151_f;
	mem[12] = (double)g4_f;
	mem[13] = (double)g5_f;
	mem[12] = (double)g4_f;
	mem[13] = (double)g5_f;
	g4_f = stack144_f;
	g5_f = stack145_f;
	mem[14] = (double)g4_f;
	mem[15] = (double)g5_f;
	return 0;
	L_return:
	switch (ret) {
		case 0: goto R_0;
		}
	return -1;  /* a return outside a funclet */
	}
#else
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
#endif

PAC_FN int uir_stub(uint32_t stub, const uir_program *g, const uir_arena *a, const uint32_t *w, const uint32_t *scaleset) {
	switch (stub) {
		case 0: {  /* k_5acfa4bfbabd18e1 */
			double mem[17];
			mem[0] = uir_input(g, a, w, 0u, scaleset);
			mem[1] = uir_input(g, a, w, 8u, scaleset);
			mem[2] = uir_input(g, a, w, 16u, scaleset);
			mem[3] = uir_input(g, a, w, 24u, scaleset);
			mem[4] = uir_input(g, a, w, 1u, scaleset);
			mem[5] = uir_input(g, a, w, 9u, scaleset);
			mem[6] = uir_input(g, a, w, 17u, scaleset);
			mem[7] = uir_input(g, a, w, 25u, scaleset);
			int status = k_5acfa4bfbabd18e1(mem);
			if (status != 0) return status;
			uir_output(g, a, w, 0, cx(mem[8], mem[9]));
			uir_output(g, a, w, 1, cx(mem[10], mem[11]));
			uir_output(g, a, w, 2, cx(mem[12], mem[13]));
			uir_output(g, a, w, 3, cx(mem[14], mem[15]));
			return 0;
			}
		case 1: {  /* k_e2d4ab976ad89705 */
			double mem[17];
			mem[0] = uir_input(g, a, w, 0u, scaleset);
			mem[1] = uir_input(g, a, w, 8u, scaleset);
			mem[2] = uir_input(g, a, w, 16u, scaleset);
			mem[3] = uir_input(g, a, w, 24u, scaleset);
			mem[4] = uir_input(g, a, w, 1u, scaleset);
			mem[5] = uir_input(g, a, w, 9u, scaleset);
			mem[6] = uir_input(g, a, w, 17u, scaleset);
			mem[7] = uir_input(g, a, w, 25u, scaleset);
			int status = k_e2d4ab976ad89705(mem);
			if (status != 0) return status;
			uir_output(g, a, w, 0, cx(mem[8], mem[9]));
			uir_output(g, a, w, 1, cx(mem[10], mem[11]));
			uir_output(g, a, w, 2, cx(mem[12], mem[13]));
			uir_output(g, a, w, 3, cx(mem[14], mem[15]));
			return 0;
			}
		default: return -1;
		}
	}

#ifndef __CUDACC__
int run(double *out, double *p, const double *momenta) {
	for (int i = 0; i < 456; i++) p[i] = 0.0;
	uir_arena a = {p + 0, p + 56, p + 104, 1, 0};
	return uir_run(&program, &a, momenta, out);  /* 0, or a failing kernel's status */
	}
#else
#define UIRGPU_LEGS 5
#define UIRGPU_OUTPUTS 32
#define UIRGPU_FORMS 14
#define UIRGPU_AMPLITUDES 24
#define UIRGPU_PLANES 176
#ifndef UIRGPU_SLOT_BYTES
#define UIRGPU_SLOT_BYTES ((size_t)268435456)  /* 4064 bytes per point (2656 mixed) */
#endif
#include "uirgpu.c"
#endif
