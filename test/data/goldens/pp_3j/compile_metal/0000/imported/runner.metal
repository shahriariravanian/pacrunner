// p_p_to_j_j_j_1 flow:3,1,4,5,2: the program's Metal source, by PacRunner's compile_metal
#ifndef MSL_SHIM_H
#include <metal_stdlib>
using namespace metal;
#pragma METAL fp contract(off)
#endif

// ---- c/metal/df64.h ----
/* df64.h: a double emulated by a pair of floats (hi, lo), value hi + lo with |lo| <= ulp(hi)/2, for
 * Apple GPUs (Metal has no double). The algorithms are pacuda's (runner/recurrence_runner.cu):
 * Joldes, Muller, Popescu, "Tight and rigorous error bounds for basic building blocks of
 * double-word arithmetic", ACM TOMS 44(2), 2017: + AccurateDWPlusDW (Alg. 6), x DWTimesDW3
 * (Alg. 12), / DWDivDW2 (Alg. 17) with DWTimesFP3 (Alg. 9). Relative error about 2^-46 or better.
 *
 * They need correctly rounded f32 operations without contraction or reassociation (Metal:
 * MTLMathModeSafe and `#pragma METAL fp contract(off)`, which Safe does not imply; measured on the
 * M4: f32 +, -, x, /, fma and sqrt are the CPU's bits) and a real fma. Apple GPUs flush f32
 * denormals to zero: a low part below 2^-126 is lost (a relative error of 2^-126 / |x|, only for
 * values near the f32 range's bottom), and the range is f32's (~1e-38..3e38).
 *
 * Plain C++ over float: the same source is MSL (with <metal_stdlib>) and C++ on the CPU (with
 * msl_shim.h), so the Linux tests run these very functions. The `df64` struct is not a float2 on
 * purpose: a float2's + would be componentwise. Buffers hold df64 values as float2 (hi, lo). */

#ifndef DF64_H
#define DF64_H

struct df64 {
    float hi, lo;
};

static inline df64 dd(float hi, float lo) { df64 z; z.hi = hi; z.lo = lo; return z; }
static inline df64 dd_f(float x) { return dd(x, 0.0f); }                 /* exact */
static inline float dd_to_f(df64 x) { return x.hi + x.lo; }              /* rounded once */
static inline df64 dd_load(float2 v) { return dd(v.x, v.y); }
static inline float2 dd_store(df64 x) { return float2(x.hi, x.lo); }
/* an integer (|n| < 2^48) exactly */
static inline df64 dd_int(long n) { float h = (float)n; return dd(h, (float)(n - (long)h)); }

/* ---- error-free transformations ---- */
static inline df64 two_sum(float a, float b)
{
    float s = a + b, bb = s - a;
    return dd(s, (a - (s - bb)) + (b - bb));
}
static inline df64 fast_two_sum(float a, float b)    /* |a| >= |b| (or a = 0) */
{
    float s = a + b;
    return dd(s, b - (s - a));
}
static inline df64 two_prod(float a, float b)
{
    float p = a * b;
    return dd(p, fma(a, b, -p));
}

/* ---- arithmetic ---- */
static inline df64 operator-(df64 x) { return dd(-x.hi, -x.lo); }

static inline df64 operator+(df64 x, df64 y)          /* AccurateDWPlusDW */
{
    df64 s = two_sum(x.hi, y.hi), t = two_sum(x.lo, y.lo);
    df64 v = fast_two_sum(s.hi, s.lo + t.hi);
    return fast_two_sum(v.hi, t.lo + v.lo);
}
static inline df64 operator-(df64 x, df64 y) { return x + (-y); }

static inline df64 operator*(df64 x, df64 y)          /* DWTimesDW3 */
{
    float ch = x.hi * y.hi;
    float cl1 = fma(x.hi, y.hi, -ch);
    float tl0 = x.lo * y.lo;
    float tl1 = fma(x.hi, y.lo, tl0);
    float cl2 = fma(x.lo, y.hi, tl1);
    return fast_two_sum(ch, cl1 + cl2);
}

static inline df64 operator/(df64 x, df64 y)          /* DWDivDW2 */
{
    float th = x.hi / y.hi;
    float rh = y.hi * th;                              /* r = y x th (DWTimesFP3) */
    float rl1 = fma(y.hi, th, -rh);
    float rl3 = fma(y.lo, th, rl1);
    df64 r = fast_two_sum(rh, rl3);
    float pi_h = x.hi - r.hi;
    float delta = pi_h + (x.lo - r.lo);
    return fast_two_sum(th, delta / y.hi);
}

static inline thread df64 &operator+=(thread df64 &x, df64 y) { x = x + y; return x; }

/* a × b + c (the kernels' fused multiply-add: two df64 roundings, not one) */
static inline df64 dd_fma(df64 a, df64 b, df64 c) { return a * b + c; }

/* sqrt: Karp-Markstein, one correction of the f32 root; sqrt(0) = 0, sqrt(<0) = NaN */
static inline df64 dd_sqrt(df64 x)
{
    if (!(x.hi > 0.0f)) return dd(sqrt(x.hi), 0.0f);
    float s = sqrt(x.hi);
    df64 r = x - two_prod(s, s);
    return fast_two_sum(s, r.hi / (2.0f * s));
}

/* ---- comparisons (normalized pairs: hi decides, lo breaks ties) ---- */
static inline bool operator==(df64 a, df64 b) { return a.hi == b.hi && a.lo == b.lo; }
static inline bool operator!=(df64 a, df64 b) { return !(a == b); }
static inline bool operator<(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo < b.lo); }
static inline bool operator<=(df64 a, df64 b) { return a.hi < b.hi || (a.hi == b.hi && a.lo <= b.lo); }
static inline bool operator>(df64 a, df64 b) { return b < a; }
static inline bool operator>=(df64 a, df64 b) { return b <= a; }
static inline bool dd_isnan(df64 x) { return isnan(x.hi) || isnan(x.lo); }
static inline bool dd_signbit(df64 x) { return (as_type<uint>(x.hi) >> 31) != 0; }
static inline bool dd_is_zero(df64 x) { return x.hi == 0.0f && x.lo == 0.0f; }

static inline df64 dd_abs(df64 x) { return dd_signbit(x) ? -x : x; }
/* sign of s on the magnitude of x (as copysign: on hi's sign bit) */
static inline df64 dd_copysign(df64 x, df64 s) { return dd_signbit(x) != dd_signbit(s) ? -x : x; }

/* Julia's max and min (pacops' jmax, jmin): NaN propagates, max(-0, 0) = 0, min(-0, 0) = -0 */
static inline df64 dd_max(df64 x, df64 y)
{
    if (dd_isnan(x) || dd_isnan(y)) return x + y;
    if (x != y) return x > y ? x : y;
    return dd_signbit(x) ? y : x;
}
static inline df64 dd_min(df64 x, df64 y)
{
    if (dd_isnan(x) || dd_isnan(y)) return x + y;
    if (x != y) return x < y ? x : y;
    return dd_signbit(x) ? x : y;
}

/* ---- roundings to an integer (the kernels' Floor, Ceiling, Trunc, Round) ---- */
static inline df64 dd_floor(df64 x)
{
    float h = floor(x.hi);
    return h != x.hi ? dd(h, 0.0f) : fast_two_sum(h, floor(x.lo));
}
static inline df64 dd_ceil(df64 x)
{
    float h = ceil(x.hi);
    return h != x.hi ? dd(h, 0.0f) : fast_two_sum(h, ceil(x.lo));
}
static inline df64 dd_trunc(df64 x) { return dd_signbit(x) ? dd_ceil(x) : dd_floor(x); }
/* to nearest, ties to even (vroundsd mode 0) */
static inline df64 dd_rint(df64 x)
{
    df64 f = dd_floor(x), d = x - f;
    if (d < dd_f(0.5f)) return f;
    if (d > dd_f(0.5f)) return f + dd_f(1.0f);
    df64 halved = dd_floor(f * dd_f(0.5f));               /* a tie: the even neighbour */
    return halved * dd_f(2.0f) == f ? f : f + dd_f(1.0f);
}

/* ---- the kernels' bit operations (kernel_msl.jl): componentwise on (hi, lo) ----
 * A mask (comparisons, IsZero) has all bits set in both halves, and a sign (Sign) the sign bit in
 * both, so that x And mask selects x, x Xor x is 0 and x Xor Sign(y) is ±x, as in f64. */
static inline df64 dd_bits(uint h, uint l) { return dd(as_type<float>(h), as_type<float>(l)); }
static inline df64 dd_mask(bool b) { uint m = b ? 0xffffffffu : 0u; return dd_bits(m, m); }
static inline df64 dd_sign(df64 x) { uint s = as_type<uint>(x.hi) & 0x80000000u; return dd_bits(s, s); }
static inline df64 dd_and(df64 a, df64 b)
{
    return dd_bits(as_type<uint>(a.hi) & as_type<uint>(b.hi), as_type<uint>(a.lo) & as_type<uint>(b.lo));
}
static inline df64 dd_andnot(df64 a, df64 b)
{
    return dd_bits(~as_type<uint>(a.hi) & as_type<uint>(b.hi), ~as_type<uint>(a.lo) & as_type<uint>(b.lo));
}
static inline df64 dd_or(df64 a, df64 b)
{
    return dd_bits(as_type<uint>(a.hi) | as_type<uint>(b.hi), as_type<uint>(a.lo) | as_type<uint>(b.lo));
}
static inline df64 dd_xor(df64 a, df64 b)
{
    return dd_bits(as_type<uint>(a.hi) ^ as_type<uint>(b.hi), as_type<uint>(a.lo) ^ as_type<uint>(b.lo));
}
static inline df64 dd_not(df64 a) { return dd_bits(~as_type<uint>(a.hi), ~as_type<uint>(a.lo)); }
/* a condition (IfElse, BranchIf): x != 0, as the C kernels' `x != 0.0` (NaN masks are true) */
static inline bool dd_true(df64 x) { return !(x.hi == 0.0f && x.lo == 0.0f); }

/* ---- complex df64 ---- */
struct cdf64 {
    df64 re, im;
};
static inline cdf64 cdd(df64 re, df64 im) { cdf64 z; z.re = re; z.im = im; return z; }
static inline cdf64 cdd_add(cdf64 a, cdf64 b) { return cdd(a.re + b.re, a.im + b.im); }
static inline cdf64 cdd_sub(cdf64 a, cdf64 b) { return cdd(a.re - b.re, a.im - b.im); }
static inline cdf64 cdd_neg(cdf64 a) { return cdd(-a.re, -a.im); }
static inline cdf64 cdd_mul(cdf64 a, cdf64 b) { return cdd(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
static inline cdf64 cdd_mulr(cdf64 a, df64 x) { return cdd(a.re * x, a.im * x); }
static inline cdf64 cdd_divr(cdf64 a, df64 x) { return cdd(a.re / x, a.im / x); }

#endif

// ---- c/metal/pacops.metal ----
/* pacops.metal: c/pacops.{h,c} for Metal, in the GPU's mixed precision with df64 (df64.h) for f64:
 * the same functions, the same operations in the same order, as PAC_MIXED does on CUDA:
 *
 *   f32 (pac_real): a current's components and the current rows' arithmetic (vertices, propagator
 *                   numerators, scale, closure products, copy, scatter), the rows' factors
 *   df64:           momentum forms, propagator denominators (rounded once to f32), wavefunctions
 *                   (then rounded into the f32 planes), amplitudes, the reduction, the group factor
 *
 * Differences from c/pacops.c, all forced by MSL: no double (df64); pointer arguments name their
 * address space (`thread`: the interpreter's local operand arrays), so factors come by value
 * (rcplx, cdf64); no recursion (massive_vector flips its momentum in place). The straight-line
 * runner's `vertex`, `kernel_call` and `kernel_store` are not here: Metal runs the interpreter,
 * and uirvm.metal stores kernel outputs itself.
 *
 * Keep in step with c/pacops.c: a change there belongs here too (test/test_metal_shim.jl compares
 * the two on the CPU). Needs df64.h, and <metal_stdlib> or msl_shim.h, before it. */

#ifndef PACOPS_METAL
#define PACOPS_METAL

/* main.json's strings as numbers (pacops.h's) */
enum { MODE_SET = 0, MODE_ADD = 1 };
enum { FAMILY_SCALAR, FAMILY_WEYL, FAMILY_DIRAC, FAMILY_VECTOR };
enum { PARTICLE = 0, ANTIPARTICLE = 1 };
enum { VERTEX_WEYL_VECTOR_A, VERTEX_WEYL_VECTOR_B, VERTEX_TENSOR_VECTOR, VERTEX_VECTOR_WEDGE,
       VERTEX_THREE_VECTOR };
enum { PROPAGATOR_WEYL_A, PROPAGATOR_WEYL_B, PROPAGATOR_VECTOR };

/* ---- f32 complex numbers: the currents (pacops.h's rcplx under PAC_MIXED) ---- */

typedef float pac_real;
struct rcplx {
    float re, im;
};

static inline rcplx rx(float re, float im) { rcplx z; z.re = re; z.im = im; return z; }
static inline rcplx rget(thread const float *p, int j) { return rx(p[2 * j], p[2 * j + 1]); }
static inline void rput(thread float *p, int j, rcplx z) { p[2 * j] = z.re; p[2 * j + 1] = z.im; }
static inline rcplx radd(rcplx a, rcplx b) { return rx(a.re + b.re, a.im + b.im); }
static inline rcplx rsub(rcplx a, rcplx b) { return rx(a.re - b.re, a.im - b.im); }
static inline rcplx rneg(rcplx a) { return rx(-a.re, -a.im); }
static inline rcplx rmul(rcplx a, rcplx b) { return rx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re); }
static inline rcplx rmulr(rcplx a, float x) { return rx(a.re * x, a.im * x); }
static inline rcplx rmul_i(rcplx a) { return rx(-a.im, a.re); }
static inline rcplx rcombine(rcplx old, rcplx value, int mode) { return mode == MODE_ADD ? radd(old, value) : value; }
static inline void rput2(thread float *out, rcplx a, rcplx b) { rput(out, 0, a); rput(out, 1, b); }
static inline void rput4(thread float *out, rcplx a, rcplx b, rcplx c, rcplx d)
{
    rput(out, 0, a); rput(out, 1, b); rput(out, 2, c); rput(out, 3, d);
}
/* to and from df64: rounded once, widened exactly */
static inline rcplx rof(cdf64 z) { return rx(dd_to_f(z.re), dd_to_f(z.im)); }
static inline cdf64 dof(rcplx z) { return cdd(dd_f(z.re), dd_f(z.im)); }

#define RZERO (rx(0.0f, 0.0f))

/* ---- df64 complex helpers of the wavefunctions ---- */

static inline df64 D(float x) { return dd_f(x); }                 /* an exact small constant */
static inline cdf64 creal(df64 x) { return cdd(x, D(0.0f)); }
#define CZERO (cdd(D(0.0f), D(0.0f)))
static inline void put2(thread cdf64 *out, cdf64 a, cdf64 b) { out[0] = a; out[1] = b; }
static inline void put4(thread cdf64 *out, cdf64 a, cdf64 b, cdf64 c, cdf64 d)
{
    out[0] = a; out[1] = b; out[2] = c; out[3] = d;
}
static inline cdf64 combine(cdf64 old, cdf64 value, int mode) { return mode == MODE_ADD ? cdd_add(old, value) : value; }

/* ==== wavefunctions (k: a momentum, h: helicity, ch: chirality) ================================ */

static inline void scalar(thread cdf64 *out) { out[0] = creal(D(1.0f)); }

/* ---- Weyl spinors (ext_quark_weyl_array, ext_antiquark_weyl_array) ---- */

static inline void weyl_particle(thread cdf64 *out, thread const df64 *k, int h, int ch)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(hd * px / s, -py / s);
        if (h == +1 && ch == +1) put2(out, a, b);
        else if (h == -1 && ch == -1) put2(out, b, a);
        else put2(out, CZERO, CZERO);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : -dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(hd * px / s, py / s);
        if (h == -1 && ch == +1) put2(out, a, b);
        else if (h == +1 && ch == -1) put2(out, b, a);
        else put2(out, CZERO, CZERO);
    }
}

static inline void weyl_antiparticle(thread cdf64 *out, thread const df64 *k, int h, int ch)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : -dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(-hd * px / s, py / s);
        if (h == +1 && ch == +1) put2(out, b, a);
        else if (h == -1 && ch == -1) put2(out, a, b);
        else put2(out, CZERO, CZERO);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(-hd * px / s, -py / s);
        if (h == -1 && ch == +1) put2(out, b, a);
        else if (h == +1 && ch == -1) put2(out, a, b);
        else put2(out, CZERO, CZERO);
    }
}

/* ---- massless Dirac spinors (ext_quark_dirac, ext_antiquark_dirac) ---- */

static inline void quark_dirac(thread cdf64 *out, thread const df64 *k, int h)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(hd * px / s, -py / s);
        if (h == +1) put4(out, a, b, CZERO, CZERO);
        else put4(out, CZERO, CZERO, b, a);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : -dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(hd * px / s, py / s);
        if (h == -1) put4(out, a, b, CZERO, CZERO);
        else put4(out, CZERO, CZERO, b, a);
    }
}

static inline void antiquark_dirac(thread cdf64 *out, thread const df64 *k, int h)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f), hd = D((float)h);
    if (E > zero) {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz < zero ? zero : -dd_sqrt(dd_max(E + pz, zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * E)) : cdd(-hd * px / s, py / s);
        if (h == -1) put4(out, CZERO, CZERO, a, b);
        else put4(out, b, a, CZERO, CZERO);
    } else {
        df64 s = dd_is_zero(px) && dd_is_zero(py) && pz > zero ? zero : dd_sqrt(dd_max(-(E + pz), zero));
        cdf64 a = creal(s);
        cdf64 b = dd_is_zero(s) ? creal(D((float)-h) * dd_sqrt(D(2.0f) * dd_abs(E))) : cdd(-hd * px / s, -py / s);
        if (h == +1) put4(out, CZERO, CZERO, a, b);
        else put4(out, b, a, CZERO, CZERO);
    }
}

/* ---- Dirac spinors (ext_quark_dirac_massive, ext_antiquark_dirac_massive) ----
 * |mass| < 1e-8 falls back to the massless spinors (the threshold compared in f32: a mass is never
 * near it). The two differ in nsf, the sign of py in chi2 and the order of the products. */

static inline void dirac_massive(thread cdf64 *out, thread const df64 *k, int h, df64 mass, int anti)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f);
    int nsf = anti ? (E > zero ? -1 : 1) : (E > zero ? 1 : -1);
    int nh = nsf * h;

    df64 pp = dd_sqrt(px * px + py * py + pz * pz);
    df64 omega1 = dd_sqrt(dd_abs(E) + pp);
    df64 omega2 = mass / omega1;

    df64 sf1 = D((float)(1 + nsf + (1 - nsf) * nh)) * D(0.5f);
    df64 sf2 = D((float)(1 + nsf - (1 - nsf) * nh)) * D(0.5f);

    df64 spx = E > zero ? px : -px, spy = E > zero ? py : -py, spz = E > zero ? pz : -pz;
    df64 pp3 = dd_max(pp + spz, zero);
    cdf64 chi1 = dd_is_zero(pp) ? creal(D(1.0f)) : creal(dd_sqrt(pp3 * D(0.5f) / pp));
    cdf64 chi2 = dd_is_zero(pp3) || dd_is_zero(pp) ? creal(D((float)-nh))
                                                  : cdd_divr(cdd(D((float)nh) * spx, anti ? spy : -spy), dd_sqrt(D(2.0f) * pp * pp3));

    /* the quark's (upper, lower) weights; the antiquark's are swapped */
    df64 up1 = anti ? sf1 * omega2 : sf2 * omega1, lo1 = anti ? sf2 * omega1 : sf1 * omega2;
    df64 up2 = anti ? sf1 * omega1 : sf2 * omega2, lo2 = anti ? sf2 * omega2 : sf1 * omega1;
    if (nh > 0) put4(out, cdd_mulr(chi1, up1), cdd_mulr(chi2, up1), cdd_mulr(chi1, lo1), cdd_mulr(chi2, lo1));
    else put4(out, cdd_mulr(chi2, up2), cdd_mulr(chi1, up2), cdd_mulr(chi2, lo2), cdd_mulr(chi1, lo2));
}

static inline void quark_dirac_massive(thread cdf64 *out, thread const df64 *k, int h, df64 mass)
{
    if (dd_abs(mass).hi < 1.0e-8f) quark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 0);
}

static inline void antiquark_dirac_massive(thread cdf64 *out, thread const df64 *k, int h, df64 mass)
{
    if (dd_abs(mass).hi < 1.0e-8f) antiquark_dirac(out, k, h);
    else dirac_massive(out, k, h, mass, 1);
}

/* ---- massless vectors: gluon, photon (ext_gluon) ---- */

static inline void gluon(thread cdf64 *out, thread const df64 *k, int h)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f);
    df64 sqh = dd_sqrt(D(0.5f));
    cdf64 wf1, wf2, wf3;
    if (E > zero) {
        df64 hel = D((float)h);
        df64 pp = E;
        df64 pt = dd_sqrt(px * px + py * py);
        wf3 = creal(hel * pt / pp * sqh);
        if (!dd_is_zero(pt)) {
            df64 pzpt = pz / (pp * pt) * sqh * hel;
            wf1 = cdd(-px * pzpt, -py / pt * sqh);
            wf2 = cdd(-py * pzpt, px / pt * sqh);
        } else {
            wf1 = creal(-hel * sqh);
            wf2 = cdd(zero, dd_copysign(sqh, pz));
        }
    } else {
        df64 hel = D((float)-h);
        df64 pp = -E;
        df64 pt = dd_sqrt(px * px + py * py);
        wf3 = creal(hel * pt / pp * sqh);
        if (!dd_is_zero(pt)) {
            df64 pzpt = -pz / (pp * pt) * sqh * hel;
            wf1 = cdd(px * pzpt, py / pt * sqh);
            wf2 = cdd(py * pzpt, -px / pt * sqh);
        } else {
            wf1 = creal(-hel * sqh);
            wf2 = cdd(zero, -dd_copysign(sqh, pz));
        }
    }
    put4(out, CZERO, wf1, wf2, wf3);
}

/* ---- massive vectors: Z, W (ext_massive_vector) ---- */

static inline void massive_vector(thread cdf64 *out, thread const df64 *k, int h, df64 mass)
{
    df64 E = k[0], px = k[1], py = k[2], pz = k[3], zero = D(0.0f);
    if (E < zero) {   /* c/pacops.c recurses with (-k, -h) */
        E = -E; px = -px; py = -py; pz = -pz; h = -h;
    }

    df64 sqh = dd_sqrt(D(0.5f));
    df64 hel = D((float)h);
    df64 nsvahl = D((float)abs(h));
    df64 pt2 = px * px + py * py;
    df64 pp = dd_min(E, dd_sqrt(pt2 + pz * pz));
    df64 pt = dd_min(pp, dd_sqrt(pt2));
    df64 hel0 = D(1.0f) - dd_abs(hel);

    if (dd_is_zero(pp)) {
        put4(out, CZERO, creal(-hel * sqh), cdd(zero, nsvahl * sqh), creal(hel0));
        return;
    }

    df64 emp = E / (mass * pp);
    cdf64 wf0 = creal(hel0 * pp / mass);
    cdf64 wf3 = creal(hel0 * pz * emp + hel * pt / pp * sqh);
    cdf64 wf1, wf2;
    if (!dd_is_zero(pt)) {
        df64 pzpt = pz / (pp * pt) * sqh * hel;
        wf1 = cdd(hel0 * px * emp - px * pzpt, -nsvahl * py / pt * sqh);
        wf2 = cdd(hel0 * py * emp - py * pzpt, nsvahl * px / pt * sqh);
    } else {
        wf1 = creal(-hel * sqh);
        wf2 = cdd(zero, nsvahl * dd_copysign(sqh, pz));
    }
    put4(out, wf0, wf1, wf2, wf3);
}

/* ---- a source current ---- */

/* the runner's pick: a Dirac source always through the massive spinors, a vector by whether its
 * mass is exactly 0; returns the components */
static inline int wavefunction(thread cdf64 *out, int family, int orientation, thread const df64 *k, int h, int ch,
                               df64 mass)
{
    int anti = orientation == ANTIPARTICLE;
    switch (family) {
    case FAMILY_SCALAR: scalar(out); return 1;
    case FAMILY_WEYL:
        if (anti) weyl_antiparticle(out, k, h, ch); else weyl_particle(out, k, h, ch);
        return 2;
    case FAMILY_DIRAC:
        if (anti) antiquark_dirac_massive(out, k, h, mass); else quark_dirac_massive(out, k, h, mass);
        return 4;
    case FAMILY_VECTOR:
        if (dd_is_zero(mass)) gluon(out, k, h); else massive_vector(out, k, h, mass);
        return 4;
    default: return 0;
    }
}

/* a source current: factor × its wavefunction; returns its components */
static inline int source(thread cdf64 *out, int family, int orientation, thread const df64 *k, int h, int ch, df64 mass,
                         cdf64 factor)
{
    int n = wavefunction(out, family, orientation, k, h, ch, mass);
    for (int j = 0; j < n; j++) out[j] = cdd_mul(factor, out[j]);
    return n;
}

/* ==== vertex: o, before the coupling and the factor (f32) ======================================== */

static inline void vertex_weyl_vector_a(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o0 = radd(radd(rsub(rneg(rmul_i(rmul(l0, r3))), rmul_i(rmul(l1, r1))), rmul(l1, r2)), rmul_i(rmul(l0, r0)));
    rcplx o1 = radd(radd(rsub(rneg(rmul(l0, r2)), rmul_i(rmul(l0, r1))), rmul_i(rmul(l1, r0))), rmul_i(rmul(l1, r3)));
    rput2(out, o0, o1);
}

static inline void vertex_weyl_vector_b(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o0 = radd(radd(radd(rneg(rmul(l1, r2)), rmul_i(rmul(l0, r0))), rmul_i(rmul(l0, r3))), rmul_i(rmul(l1, r1)));
    rcplx o1 = radd(radd(radd(rneg(rmul_i(rmul(l1, r3))), rmul(l0, r2)), rmul_i(rmul(l0, r1))), rmul_i(rmul(l1, r0)));
    rput2(out, o0, o1);
}

static inline void vertex_tensor_vector(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1), l2 = rget(l, 2), l3 = rget(l, 3), l4 = rget(l, 4), l5 = rget(l, 5);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rput4(out,
          radd(radd(rmul(l0, r1), rmul(l1, r2)), rmul(l2, r3)),
          radd(radd(rmul(l0, r0), rmul(l3, r2)), rmul(l4, r3)),
          radd(rsub(rmul(l1, r0), rmul(l3, r1)), rmul(l5, r3)),
          rsub(rsub(rmul(l2, r0), rmul(l4, r1)), rmul(l5, r2)));
}

static inline void vertex_vector_wedge(thread float *out, thread const float *l, thread const float *r)
{
    rcplx l0 = rget(l, 0), l1 = rget(l, 1), l2 = rget(l, 2), l3 = rget(l, 3);
    rcplx r0 = rget(r, 0), r1 = rget(r, 1), r2 = rget(r, 2), r3 = rget(r, 3);
    rcplx o[6] = {rsub(rmul(l0, r1), rmul(l1, r0)), rsub(rmul(l0, r2), rmul(l2, r0)), rsub(rmul(l0, r3), rmul(l3, r0)),
                  rsub(rmul(l1, r2), rmul(l2, r1)), rsub(rmul(l1, r3), rmul(l3, r1)), rsub(rmul(l2, r3), rmul(l3, r2))};
    for (int j = 0; j < 6; j++) rput(out, j, o[j]);
}

static inline rcplx dot(thread const float *a, thread const float *b)
{
    return rsub(rsub(rsub(rmul(rget(a, 0), rget(b, 0)), rmul(rget(a, 1), rget(b, 1))), rmul(rget(a, 2), rget(b, 2))),
                rmul(rget(a, 3), rget(b, 3)));
}

static inline rcplx dot_real(thread const float *a, thread const float *b)
{
    return rsub(rsub(rsub(rmulr(rget(a, 0), b[0]), rmulr(rget(a, 1), b[1])), rmulr(rget(a, 2), b[2])), rmulr(rget(a, 3), b[3]));
}

/* the colour-ordered three-gluon vertex; u = momenta[0], v = momenta[1] (rounded to f32) */
static inline void vertex_three_vector(thread float *out, thread const float *l, thread const float *r, thread const float *u,
                                       thread const float *v)
{
    rcplx lr = dot(l, r), lv = dot_real(l, v), ru = dot_real(r, u);
    rcplx o[4];
    for (int j = 0; j < 4; j++)
        o[j] = radd(rmulr(lr, u[j] - v[j]), rmulr(rsub(rmul(lv, rget(r, j)), rmul(ru, rget(l, j))), 2.0f));
    for (int j = 0; j < 4; j++) rput(out, j, o[j]);
}

/* ==== propagator: out[j] = o[j] × factor / (k·k) ================================================= */

/* 1 / (k·k) in df64; the numerators take k rounded to f32 (kr) */
static inline df64 reciprocal_norm(thread const df64 *k)
{
    return D(1.0f) / (((k[0] * k[0] - k[1] * k[1]) - k[2] * k[2]) - k[3] * k[3]);
}

static inline void propagator_weyl_a(thread float *out, thread const float *c, thread const df64 *k, rcplx factor)
{
    float kr[4] = {dd_to_f(k[0]), dd_to_f(k[1]), dd_to_f(k[2]), dd_to_f(k[3])};
    rcplx c0 = rget(c, 0), c1 = rget(c, 1);
    rcplx s = rmulr(factor, dd_to_f(reciprocal_norm(k)));
    rcplx o0 = radd(radd(rsub(rneg(rmul_i(rmulr(c0, kr[3]))), rmul_i(rmulr(c1, kr[1]))), rmulr(c1, kr[2])), rmul_i(rmulr(c0, kr[0])));
    rcplx o1 = radd(radd(rsub(rneg(rmulr(c0, kr[2])), rmul_i(rmulr(c0, kr[1]))), rmul_i(rmulr(c1, kr[0]))), rmul_i(rmulr(c1, kr[3])));
    rput2(out, rmul(o0, s), rmul(o1, s));
}

static inline void propagator_weyl_b(thread float *out, thread const float *c, thread const df64 *k, rcplx factor)
{
    float kr[4] = {dd_to_f(k[0]), dd_to_f(k[1]), dd_to_f(k[2]), dd_to_f(k[3])};
    rcplx c0 = rget(c, 0), c1 = rget(c, 1);
    rcplx s = rmulr(factor, dd_to_f(reciprocal_norm(k)));
    rcplx o0 = radd(radd(radd(rneg(rmulr(c1, kr[2])), rmul_i(rmulr(c0, kr[0]))), rmul_i(rmulr(c0, kr[3]))), rmul_i(rmulr(c1, kr[1])));
    rcplx o1 = radd(radd(radd(rneg(rmul_i(rmulr(c1, kr[3]))), rmulr(c0, kr[2])), rmul_i(rmulr(c0, kr[1]))), rmul_i(rmulr(c1, kr[0])));
    rput2(out, rmul(o0, s), rmul(o1, s));
}

/* c vector, Feynman gauge: o[j] = -i c[j] */
static inline void propagator_vector(thread float *out, thread const float *c, thread const df64 *k, rcplx factor)
{
    rcplx s = rmulr(rmul(factor, rx(0.0f, -1.0f)), dd_to_f(reciprocal_norm(k)));
    for (int j = 0; j < 4; j++) rput(out, j, rmul(rget(c, j), s));
}

static inline void propagator(thread float *out, int type, thread const float *c, thread const df64 *k, rcplx factor)
{
    switch (type) {
    case PROPAGATOR_WEYL_A: propagator_weyl_a(out, c, k, factor); break;
    case PROPAGATOR_WEYL_B: propagator_weyl_b(out, c, k, factor); break;
    case PROPAGATOR_VECTOR: propagator_vector(out, c, k, factor); break;
    }
}

/* ==== scale, clear, closure ====================================================================== */

static inline void scale(thread float *c, int components, rcplx f)
{
    for (int j = 0; j < components; j++) rput(c, j, rmul(f, rget(c, j)));
}

static inline void clear(thread float *c, int components)
{
    for (int j = 0; j < components; j++) rput(c, j, RZERO);
}

/* amplitude += Σ_j (l[j] × r[j]) × (component_factors[j] × factor): products in f32, the sum in df64 */
static inline void closure(thread cdf64 &amplitude, thread const float *l, thread const float *r,
                           thread const float *component_factors, int components, rcplx f)
{
    df64 re = amplitude.re, im = amplitude.im;
    for (int j = 0; j < components; j++) {
        rcplx t = rmul(rmul(rget(l, j), rget(r, j)), rmul(rget(component_factors, j), f));
        re = re + dd_f(t.re);
        im = im + dd_f(t.im);
    }
    amplitude = cdd(re, im);
}

/* ==== copy, scatter (f32) ======================================================================== */

static inline void copy_row(thread float *out, thread const float *src, int n, rcplx f)
{
    bool zero = f.re == 0.0f && f.im == 0.0f;
    for (int j = 0; j < n; j++) rput(out, j, zero ? RZERO : rmul(rget(src, j), f));
}

static inline rcplx scatter_value(rcplx v, rcplx f)
{
    if (f.im == 0.0f) {
        if (f.re == 1.0f) return v;
        if (f.re == -1.0f) return rneg(v);
        return rx(v.re * f.re, v.im * f.re);
    }
    if (f.re == 0.0f) return rx(-v.im * f.im, v.re * f.im);
    return rmul(v, f);
}

static inline void scatter(thread float *out, thread const float *src, int n, int mode, rcplx f)
{
    for (int j = 0; j < n; j++) rput(out, j, rcombine(rget(out, j), scatter_value(rget(src, j), f), mode));
}

/* ==== reduction (df64) =========================================================================== */

/* acc + weight × |a|², a = the amplitude × scale when there is one (pacops' fma(re, re, im im): here
 * re re + im im, each product and the sum in df64) */
static inline df64 reduce_term(df64 acc, cdf64 a, bool scaled, cdf64 scale_, df64 weight)
{
    if (scaled) a = cdd_mul(a, scale_);
    return acc + weight * (a.re * a.re + a.im * a.im);
}

/* x × the group factor: x when numerator = denominator, else x × numerator ÷ denominator */
static inline df64 group(df64 x, long numerator, long denominator)
{
    return numerator == denominator ? x : x * dd_int(numerator) / dd_int(denominator);
}

#endif

// ---- c/metal/uirvm.metal ----
/* uirvm.metal: c/uirvm.{h,c}, the interpreter of UIR v0.8, for Metal, in the GPU's mixed precision
 * with df64 for f64 (pacops.metal): f32 planes and current rows, df64 forms, amplitudes, outputs,
 * sources and kernels' wide values. One thread per point (uir_points, at the end).
 *
 * Differences from c/uirvm.c:
 *   - The program is one device buffer: a uir_header (written by compile_metal.jl's metal_program, its
 *     fields in METAL_TABLES order) of counts and byte offsets, then its tables. uir_view makes a thread's
 *     uir_program, its tables as device pointers. df64 tables (scales, factors, coefficients,
 *     crossing signs, weights) are float2 (hi, lo); rscales and rfactors f32.
 *   - The arena (uir_arena): planes f32, forms and amplitudes df64 (float2), plane-major with
 *     ld = the launch's points, as on CUDA.
 *   - Momenta and outputs have their own strides (uir_launch), so the host stages MadSpace's
 *     layout as it is and a call runs in chunks of points.
 *   - The arena is cleared by each thread for its point (no host memset).
 *   - Factors are passed by value (pacops.metal). No PAC_ACC64: the vertex runs' sums are f32.
 *
 * Keep in step with c/uirvm.c. Needs df64.h and pacops.metal before it; the program's generated
 * source (compile_metal.jl) defines uir_stub after it. */

#ifndef UIRVM_METAL
#define UIRVM_METAL

#define UIR_MAX_COMPONENTS 6
#define UIR_VERSION ((0u << 16) | 8u)

enum {
    UIR_V_WEYL_VECTOR_A, UIR_V_WEYL_VECTOR_B, UIR_V_TENSOR_VECTOR, UIR_V_VECTOR_WEDGE, UIR_V_THREE_VECTOR,
    UIR_P_WEYL_A, UIR_P_WEYL_B, UIR_P_VECTOR,
    UIR_CLOSURE, UIR_SCALE, UIR_CLEAR, UIR_KERNEL, UIR_KERNEL_DIRECT, UIR_SOURCE,
    UIR_CLEAR_AMPLITUDE, UIR_COPY, UIR_SCATTER, UIR_REDUCE,
    UIR_END = 31
};
enum { UIR_IN_L, UIR_IN_R, UIR_OUT, UIR_MOMENTUM_U, UIR_MOMENTUM_V, UIR_IN_SCALE, UIR_IN_FACTOR };

/* a row's fields (uirvm.h's; w: the row's two 32-bit halves) */
#define UIR_OP(w) ((w)[0] & 31u)
#define UIR_SET(w) (((w)[0] >> 6) & 1u)
#define UIR_FIRST(w) (((w)[0] >> 7) & 1u)
#define UIR_LAST(w) (((w)[0] >> 8) & 1u)
#define UIR_D(w) (((w)[0] >> 9) & 0x1fffu)
#define UIR_T(w) ((w)[0] >> 22)
#define UIR_L(w) ((w)[1] & 0x1fffu)
#define UIR_R(w) (((w)[1] >> 13) & 0x1fffu)
#define UIR_N(w) ((w)[1] & 0xffffu)
#define UIR_X(w) ((w)[1] >> 26)
#define UIR_E(w) (((w)[1] >> 16) & 0x3ffu)

/* the program buffer's header: counts, then the byte offset of each table in the buffer (every
 * offset a multiple of 8), as compile_metal.jl writes it (METAL_TABLES; 128 bytes with padding) */
struct uir_header {
    uint version, legs, forms, planes, amplitudes, outputs, words, has_crossing;
    int group_numerator, group_denominator;
    uint rows, edges, scales, factors, rscales, rfactors, form_first, term_leg, term_coefficient, crossing_leg,
        crossing_sign, list_first, list_factor, embedding_first, embedding_component, embedding_wave, embedding_factor,
        scaleset_first, scaleset_scale, instance, weights;
};

/* a thread's view of the program: its tables as device pointers */
struct uir_program {
    uint legs, forms, planes, amplitudes, outputs;
    bool has_crossing;
    long group_numerator, group_denominator;
    device const uint *rows, *edges;
    device const float2 *scales, *factors;              /* df64 (re, im) each */
    device const float *rscales, *rfactors;             /* f32 (re, im) each */
    device const uint *form_first, *term_leg;
    device const float2 *term_coefficient;
    device const uint *crossing_leg;
    device const float2 *crossing_sign;
    device const uint *list_first, *list_factor, *embedding_first;
    device const int *embedding_component, *embedding_wave;
    device const uint *embedding_factor, *scaleset_first, *scaleset_scale, *instance;
    device const float2 *weights;
};

#define UIR_TABLE(T, off) ((device const T *)(blob + (off)))

static inline uir_program uir_view(device const uchar *blob)
{
    device const uir_header *h = (device const uir_header *)blob;
    uir_program g;
    g.legs = h->legs; g.forms = h->forms; g.planes = h->planes; g.amplitudes = h->amplitudes; g.outputs = h->outputs;
    g.has_crossing = h->has_crossing != 0;
    g.group_numerator = h->group_numerator; g.group_denominator = h->group_denominator;
    g.rows = UIR_TABLE(uint, h->rows); g.edges = UIR_TABLE(uint, h->edges);
    g.scales = UIR_TABLE(float2, h->scales); g.factors = UIR_TABLE(float2, h->factors);
    g.rscales = UIR_TABLE(float, h->rscales); g.rfactors = UIR_TABLE(float, h->rfactors);
    g.form_first = UIR_TABLE(uint, h->form_first); g.term_leg = UIR_TABLE(uint, h->term_leg);
    g.term_coefficient = UIR_TABLE(float2, h->term_coefficient);
    g.crossing_leg = UIR_TABLE(uint, h->crossing_leg); g.crossing_sign = UIR_TABLE(float2, h->crossing_sign);
    g.list_first = UIR_TABLE(uint, h->list_first); g.list_factor = UIR_TABLE(uint, h->list_factor);
    g.embedding_first = UIR_TABLE(uint, h->embedding_first);
    g.embedding_component = UIR_TABLE(int, h->embedding_component); g.embedding_wave = UIR_TABLE(int, h->embedding_wave);
    g.embedding_factor = UIR_TABLE(uint, h->embedding_factor);
    g.scaleset_first = UIR_TABLE(uint, h->scaleset_first); g.scaleset_scale = UIR_TABLE(uint, h->scaleset_scale);
    g.instance = UIR_TABLE(uint, h->instance); g.weights = UIR_TABLE(float2, h->weights);
    return g;
}

/* the arena of one point: plane k (re, im) at planes[(2k) ld + i], [(2k + 1) ld + i]; form f's
 * component j at forms[(4f + j) ld + i]; amplitude k at amplitudes[(2k) ld + i], [(2k + 1) ld + i] */
struct uir_arena {
    device float2 *forms;
    device float2 *amplitudes;
    device float *planes;
    uint ld, i;
};

/* a table's complex value: f32 (rscales, rfactors) or df64 (scales, factors) */
static inline rcplx rgetd(device const float *p, uint j) { return rx(p[2 * j], p[2 * j + 1]); }
static inline cdf64 cgetd(device const float2 *p, uint j) { return cdd(dd_load(p[2 * j]), dd_load(p[2 * j + 1])); }

/* an edge-table entry's fields */
static inline uint edge_start(thread const uir_program &g, uint e) { return (g.edges[e] & 0x7fffu) * 2; }
static inline uint edge_form(thread const uir_program &g, uint e) { return (g.edges[e] >> 15) & 0xffu; }
static inline int edge_components(thread const uir_program &g, uint e) { return (int)((g.edges[e] >> 23) & 0xfu); }

/* ---- the arena ------------------------------------------------------------------------------- */

static inline device float *plane_re(thread const uir_arena &a, uint k) { return a.planes + (2 * k) * a.ld + a.i; }
static inline device float *plane_im(thread const uir_arena &a, uint k) { return a.planes + (2 * k + 1) * a.ld + a.i; }

static inline void load(thread const uir_arena &a, uint start, int n, thread float *v)
{
    for (int j = 0; j < n; j++) {
        v[2 * j] = *plane_re(a, start + j);
        v[2 * j + 1] = *plane_im(a, start + j);
    }
}

static inline void store(thread const uir_arena &a, uint start, int n, thread const float *v)
{
    for (int j = 0; j < n; j++) {
        *plane_re(a, start + j) = v[2 * j];
        *plane_im(a, start + j) = v[2 * j + 1];
    }
}

static inline rcplx load1(thread const uir_arena &a, uint k) { return rx(*plane_re(a, k), *plane_im(a, k)); }
static inline void store1(thread const uir_arena &a, uint k, rcplx z) { *plane_re(a, k) = z.re; *plane_im(a, k) = z.im; }
/* a component as df64 (exact), and a df64 value into a component (rounded once) */
static inline cdf64 load1d(thread const uir_arena &a, uint k) { return dof(load1(a, k)); }
static inline void store1d(thread const uir_arena &a, uint k, cdf64 z) { store1(a, k, rof(z)); }

/* momentum form f, in df64 and rounded to f32 (the three-vector vertex's) */
static inline void load_form(thread const uir_arena &a, uint f, thread df64 *q)
{
    for (int j = 0; j < 4; j++) q[j] = dd_load(a.forms[(4 * f + j) * a.ld + a.i]);
}
static inline void load_form_r(thread const uir_arena &a, uint f, thread float *q)
{
    for (int j = 0; j < 4; j++) q[j] = dd_to_f(dd_load(a.forms[(4 * f + j) * a.ld + a.i]));
}

static inline device float2 *amplitude_re(thread const uir_arena &a, uint k) { return a.amplitudes + (2 * k) * a.ld + a.i; }
static inline device float2 *amplitude_im(thread const uir_arena &a, uint k) { return a.amplitudes + (2 * k + 1) * a.ld + a.i; }

/* ---- the rows -------------------------------------------------------------------------------- */

static inline void vertex_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w,
                              thread rcplx *acc, uint op, int nl, int nr, int n)
{
    float l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], v[2 * UIR_MAX_COMPONENTS];
    load(a, edge_start(g, UIR_L(w)), nl, l);
    load(a, edge_start(g, UIR_R(w)), nr, r);
    switch (op) {
    case UIR_V_WEYL_VECTOR_A: vertex_weyl_vector_a(v, l, r); break;
    case UIR_V_WEYL_VECTOR_B: vertex_weyl_vector_b(v, l, r); break;
    case UIR_V_TENSOR_VECTOR: vertex_tensor_vector(v, l, r); break;
    case UIR_V_VECTOR_WEDGE: vertex_vector_wedge(v, l, r); break;
    default: {
        float u[4], q[4];
        load_form_r(a, edge_form(g, UIR_L(w)), u);
        load_form_r(a, edge_form(g, UIR_R(w)), q);
        vertex_three_vector(v, l, r, u, q);
    }
    }
    uint d = edge_start(g, UIR_D(w));
    rcplx s = rmul(rgetd(g.rscales, UIR_T(w)), rgetd(g.rfactors, UIR_X(w)));
    uint first = UIR_FIRST(w), set = UIR_SET(w);
    for (int j = 0; j < n; j++) {
        rcplx c = rmul(rget(v, j), s);
        if (!first) acc[j] = radd(acc[j], c);
        else if (set) acc[j] = c;
        else acc[j] = radd(load1(a, d + j), c);
    }
    if (UIR_LAST(w))
        for (int j = 0; j < n; j++) store1(a, d + j, acc[j]);
}

static inline void run_vertex(thread const uir_program &g, thread const uir_arena &a, device const uint *w, thread rcplx *acc)
{
    switch (UIR_OP(w)) {
    case UIR_V_WEYL_VECTOR_A: vertex_row(g, a, w, acc, UIR_V_WEYL_VECTOR_A, 2, 4, 2); break;
    case UIR_V_WEYL_VECTOR_B: vertex_row(g, a, w, acc, UIR_V_WEYL_VECTOR_B, 2, 4, 2); break;
    case UIR_V_TENSOR_VECTOR: vertex_row(g, a, w, acc, UIR_V_TENSOR_VECTOR, 6, 4, 4); break;
    case UIR_V_VECTOR_WEDGE: vertex_row(g, a, w, acc, UIR_V_VECTOR_WEDGE, 4, 4, 6); break;
    default: vertex_row(g, a, w, acc, UIR_V_THREE_VECTOR, 4, 4, 4); break;
    }
}

static inline void propagator_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int type, int n)
{
    float c[8], v[8];
    df64 q[4];
    uint d = edge_start(g, UIR_D(w));
    load(a, d, n, c);
    load_form(a, edge_form(g, UIR_D(w)), q);
    propagator(v, type, c, q, rgetd(g.rfactors, UIR_X(w)));
    store(a, d, n, v);
}

static inline void run_propagator(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    switch (UIR_OP(w)) {
    case UIR_P_WEYL_A: propagator_row(g, a, w, PROPAGATOR_WEYL_A, 2); break;
    case UIR_P_WEYL_B: propagator_row(g, a, w, PROPAGATOR_WEYL_B, 2); break;
    default: propagator_row(g, a, w, PROPAGATOR_VECTOR, 4); break;
    }
}

static inline void closure_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, uint first, int n)
{
    float l[2 * UIR_MAX_COMPONENTS], r[2 * UIR_MAX_COMPONENTS], cf[2 * UIR_MAX_COMPONENTS];
    for (int j = 0; j < n; j++) {
        uint f = g.list_factor[first + j];
        cf[2 * j] = g.rfactors[2 * f];
        cf[2 * j + 1] = g.rfactors[2 * f + 1];
    }
    load(a, edge_start(g, UIR_L(w)), n, l);
    load(a, edge_start(g, UIR_R(w)), n, r);
    uint d = UIR_D(w);
    cdf64 amp = cdd(dd_load(*amplitude_re(a, d)), dd_load(*amplitude_im(a, d)));
    closure(amp, l, r, cf, n, rgetd(g.rfactors, UIR_X(w)));
    *amplitude_re(a, d) = dd_store(amp.re);
    *amplitude_im(a, d) = dd_store(amp.im);
}

static inline void run_closure(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    uint t = UIR_T(w), first = g.list_first[t];
    switch (g.list_first[t + 1] - first) {   /* 0 to 6 (compile refuses more), each a constant count */
    case 0: break;
    case 1: closure_row(g, a, w, first, 1); break;
    case 2: closure_row(g, a, w, first, 2); break;
    case 3: closure_row(g, a, w, first, 3); break;
    case 4: closure_row(g, a, w, first, 4); break;
    case 5: closure_row(g, a, w, first, 5); break;
    case 6: closure_row(g, a, w, first, 6); break;
    default: break;
    }
}

static inline void scale_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int n)
{
    float c[2 * UIR_MAX_COMPONENTS];
    uint d = edge_start(g, UIR_D(w));
    load(a, d, n, c);
    scale(c, n, rgetd(g.rfactors, UIR_X(w)));
    store(a, d, n, c);
}

static inline void run_scale(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    switch (UIR_N(w)) {
    case 0: break;
    case 1: scale_row(g, a, w, 1); break;
    case 2: scale_row(g, a, w, 2); break;
    case 3: scale_row(g, a, w, 3); break;
    case 4: scale_row(g, a, w, 4); break;
    case 5: scale_row(g, a, w, 5); break;
    case 6: scale_row(g, a, w, 6); break;
    default: break;
    }
}

static inline void run_clear(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    uint d = edge_start(g, UIR_D(w)), n = UIR_N(w);
    for (uint j = 0; j < n; j++) store1(a, d + j, RZERO);
}

/* kernel row w's input of code `code` (kind bits 0-2, imaginary part 3, n from 4), as df64 */
static inline df64 uir_input(thread const uir_program &g, thread const uir_arena &a, device const uint *w, uint code,
                             device const uint *scaleset)
{
    uint kind = code & 7, im = (code >> 3) & 1, n = code >> 4;
    df64 q[4];
    switch (kind) {
    case UIR_IN_L: return dd_f(im ? *plane_im(a, edge_start(g, UIR_L(w)) + n) : *plane_re(a, edge_start(g, UIR_L(w)) + n));
    case UIR_IN_R: return dd_f(im ? *plane_im(a, edge_start(g, UIR_R(w)) + n) : *plane_re(a, edge_start(g, UIR_R(w)) + n));
    case UIR_OUT: return dd_f(im ? *plane_im(a, edge_start(g, UIR_D(w)) + n) : *plane_re(a, edge_start(g, UIR_D(w)) + n));
    case UIR_MOMENTUM_U: load_form(a, edge_form(g, UIR_L(w)), q); return q[n];
    case UIR_MOMENTUM_V: load_form(a, edge_form(g, UIR_R(w)), q); return q[n];
    case UIR_IN_SCALE: return dd_load(g.scales[2 * scaleset[n]]);
    default: return dd_load(g.factors[2 * UIR_X(w) + im]);
    }
}

/* kernel row w's output pair into component `component` of its current, with the row's mode
 * (KERNEL_DIRECT: the pair as it is; else × factor), in df64, rounded once into the plane */
static inline void uir_output(thread const uir_program &g, thread const uir_arena &a, device const uint *w, uint component,
                              cdf64 pair)
{
    uint j = edge_start(g, UIR_D(w)) + component;
    cdf64 value = UIR_OP(w) == UIR_KERNEL_DIRECT ? pair : cdd_mul(pair, cgetd(g.factors, UIR_X(w)));
    store1d(a, j, combine(load1d(a, j), value, UIR_SET(w) ? MODE_SET : MODE_ADD));
}

/* the program's kernel stubs (its generated source defines it, after this file) */
static int uir_stub(uint stub, thread const uir_program &g, thread const uir_arena &a, device const uint *w,
                    device const uint *scaleset);

static inline int run_kernel(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    uint t = UIR_T(w), s = g.instance[2 * t], set = g.instance[2 * t + 1];
    return uir_stub(s, g, a, w, g.scaleset_scale + g.scaleset_first[set]);
}

static inline void source_row(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int family, int n)
{
    df64 q[4];
    cdf64 v[4];
    uint t = UIR_N(w), e = UIR_E(w), d = edge_start(g, UIR_D(w));
    int orientation = (t >> 3) & 1 ? ANTIPARTICLE : PARTICLE, h = (int)((t >> 4) & 7) - 2, ch = (int)((t >> 7) & 3) - 1;
    df64 mass = dd_load(g.scales[2 * UIR_T(w)]);
    load_form(a, edge_form(g, UIR_D(w)), q);
    if (e == 0x3ffu) {
        source(v, family, orientation, q, h, ch, mass, cgetd(g.factors, UIR_X(w)));
        for (int j = 0; j < n; j++) store1d(a, d + j, v[j]);
        return;
    }
    wavefunction(v, family, orientation, q, h, ch, mass);
    for (uint k = g.embedding_first[e]; k < g.embedding_first[e + 1]; k++) {
        int wave = g.embedding_wave[k];
        store1d(a, d + (uint)g.embedding_component[k], wave < 0 ? CZERO : cdd_mul(cgetd(g.factors, g.embedding_factor[k]), v[wave]));
    }
}

static inline void run_source(thread const uir_program &g, thread const uir_arena &a, device const uint *w)
{
    switch (UIR_N(w) & 7) {
    case 1: source_row(g, a, w, FAMILY_SCALAR, 1); break;
    case 2: source_row(g, a, w, FAMILY_WEYL, 2); break;
    case 3: source_row(g, a, w, FAMILY_DIRAC, 4); break;
    case 4: source_row(g, a, w, FAMILY_VECTOR, 4); break;
    default: break;
    }
}

static inline void copy_row_n(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int scatter_, int n)
{
    float s[2 * UIR_MAX_COMPONENTS], d[2 * UIR_MAX_COMPONENTS];
    uint dst = edge_start(g, UIR_D(w));
    load(a, edge_start(g, UIR_L(w)), n, s);
    if (scatter_) {
        load(a, dst, n, d);
        scatter(d, s, n, UIR_SET(w) ? MODE_SET : MODE_ADD, rgetd(g.rfactors, UIR_X(w)));
    } else {
        copy_row(d, s, n, rgetd(g.rfactors, UIR_X(w)));
    }
    store(a, dst, n, d);
}

static inline void run_copy(thread const uir_program &g, thread const uir_arena &a, device const uint *w, int scatter_)
{
    switch (edge_components(g, UIR_D(w))) {
    case 1: copy_row_n(g, a, w, scatter_, 1); break;
    case 2: copy_row_n(g, a, w, scatter_, 2); break;
    case 3: copy_row_n(g, a, w, scatter_, 3); break;
    case 4: copy_row_n(g, a, w, scatter_, 4); break;
    case 5: copy_row_n(g, a, w, scatter_, 5); break;
    case 6: copy_row_n(g, a, w, scatter_, 6); break;
    default: break;
    }
}

/* REDUCE: output D += weight N × |amplitude T (× its scale X, unless none)|², in df64 */
static inline void run_reduce(thread const uir_program &g, thread const uir_arena &a, device const uint *w,
                              device float2 *out, uint out_step)
{
    uint t = UIR_T(w), x = UIR_X(w);
    uint d = UIR_D(w) * out_step;
    cdf64 amp = cdd(dd_load(*amplitude_re(a, t)), dd_load(*amplitude_im(a, t)));
    bool scaled = x != 0x3fu;
    cdf64 s = scaled ? cgetd(g.factors, x) : CZERO;
    out[d] = dd_store(reduce_term(dd_load(out[d]), amp, scaled, s, dd_load(g.weights[UIR_N(w)])));
}

static inline int uir_rows(thread const uir_program &g, thread const uir_arena &a, device float2 *out, uint out_step)
{
    rcplx acc[UIR_MAX_COMPONENTS];   /* the fused run's sum */
    for (uint k = 0;;) {
        device const uint *w = g.rows + 2 * k++;
        uint op = UIR_OP(w);
        if (op == UIR_END) break;
        if (op <= UIR_V_THREE_VECTOR) {
            run_vertex(g, a, w, acc);
        } else if (op <= UIR_P_VECTOR) {
            run_propagator(g, a, w);
        } else if (op == UIR_CLOSURE) {
            run_closure(g, a, w);
        } else if (op == UIR_SCALE) {
            run_scale(g, a, w);
        } else if (op == UIR_CLEAR) {
            run_clear(g, a, w);
        } else if (op == UIR_SOURCE) {
            run_source(g, a, w);
        } else if (op == UIR_CLEAR_AMPLITUDE) {
            *amplitude_re(a, UIR_D(w)) = float2(0.0f, 0.0f);
            *amplitude_im(a, UIR_D(w)) = float2(0.0f, 0.0f);
        } else if (op == UIR_COPY || op == UIR_SCATTER) {
            run_copy(g, a, w, op == UIR_SCATTER);
        } else if (op == UIR_REDUCE) {
            run_reduce(g, a, w, out, out_step);
        } else {
            int status = run_kernel(g, a, w);
            if (status != 0) return status;
        }
    }
    return 0;
}

/* ---- forms, outputs, group factor ------------------------------------------------------------ */

/* the forms of one point: component j of leg l at momenta[leg_step l + component_step j] (df64);
 * q_f = 0 + Σ coefficient × leg, in term order; with a crossing, leg l is sign_l × the caller's
 * leg crossing_leg_l */
static inline void uir_forms_layout(thread const uir_program &g, thread const uir_arena &a, device const float2 *momenta,
                                    uint leg_step, uint component_step)
{
    for (uint f = 0; f < g.forms; f++)
        for (uint j = 0; j < 4; j++) {
            df64 v = dd_f(0.0f);
            for (uint t = g.form_first[f]; t < g.form_first[f + 1]; t++) {
                uint leg = g.term_leg[t];
                df64 x = g.has_crossing
                    ? dd_load(g.crossing_sign[leg]) * dd_load(momenta[leg_step * g.crossing_leg[leg] + component_step * j])
                    : dd_load(momenta[leg_step * leg + component_step * j]);
                v = v + dd_load(g.term_coefficient[t]) * x;
            }
            a.forms[(4 * f + j) * a.ld + a.i] = dd_store(v);
        }
}

static inline void uir_group(thread const uir_program &g, device float2 *out, uint out_step)
{
    if (g.group_numerator == g.group_denominator) return;
    for (uint h = 0; h < g.outputs; h++)
        out[h * out_step] = dd_store(group(dd_load(out[h * out_step]), g.group_numerator, g.group_denominator));
}

/* one point: its arena cleared, the forms, its outputs at zero, the rows, the group factor.
 * Returns 0, or the first failing kernel's status. */
static inline int uir_run_layout(thread const uir_program &g, thread const uir_arena &a, device const float2 *momenta,
                                 uint leg_step, uint component_step, device float2 *out, uint out_step)
{
    for (uint k = 0; k < 2 * g.planes; k++) a.planes[k * a.ld + a.i] = 0.0f;
    for (uint k = 0; k < 2 * g.amplitudes; k++) a.amplitudes[k * a.ld + a.i] = float2(0.0f, 0.0f);
    uir_forms_layout(g, a, momenta, leg_step, component_step);
    for (uint h = 0; h < g.outputs; h++) out[h * out_step] = float2(0.0f, 0.0f);
    int status = uir_rows(g, a, out, out_step);
    if (status != 0) return status;
    uir_group(g, out, out_step);
    return 0;
}

/* ---- the compute function: one thread per point ----------------------------------------------
 * A launch of `points` points, the call's points first .. first + points: the momenta of point p
 * (of the call) at momenta[p + leg_step l + component_step j], its outputs at out[p + out_step h];
 * the arena holds the launch's points (ld = points). A failing point stores its status. */
struct uir_launch {
    uint points, first, leg_step, component_step, out_step;
};

kernel void uir_points(device const uchar *program [[buffer(0)]], constant uir_launch &launch [[buffer(1)]],
                       device const float2 *momenta [[buffer(2)]], device float2 *out [[buffer(3)]],
                       device float2 *forms [[buffer(4)]], device float2 *amplitudes [[buffer(5)]],
                       device float *planes [[buffer(6)]], device atomic_int *status [[buffer(7)]],
                       uint i [[thread_position_in_grid]])
{
    if (i >= launch.points) return;
    uir_program g = uir_view(program);
    uir_arena a;
    a.forms = forms; a.amplitudes = amplitudes; a.planes = planes; a.ld = launch.points; a.i = i;
    uint p = launch.first + i;
    int s = uir_run_layout(g, a, momenta + p, launch.leg_step, launch.component_step, out + p, launch.out_step);
    if (s != 0) atomic_store_explicit(status, s, memory_order_relaxed);
}

#endif

// ---- the program's kernels (kernel_msl.jl) and stubs ----
/* the symjit kernel k_5acfa4bfbabd18e1 in mixed precision (kernel_msl.jl): 82 f32 and 17 df64 operations, 48 conversions */
static inline int k_5acfa4bfbabd18e1(thread df64 *mem) {
df64 g0 = dd_f(0.0f), g1 = dd_f(0.0f), g2 = dd_f(0.0f), g3 = dd_f(0.0f), g4 = dd_f(0.0f), g5 = dd_f(0.0f), r0 = dd_f(0.0f), r1 = dd_f(0.0f), stack2 = dd_f(0.0f), stack144 = dd_f(0.0f), stack145 = dd_f(0.0f);
float g0_f = 0.0f, g1_f = 0.0f, g2_f = 0.0f, g3_f = 0.0f, g4_f = 0.0f, g5_f = 0.0f, g6_f = 0.0f, g7_f = 0.0f, g8_f = 0.0f, g9_f = 0.0f, r0_f = 0.0f, r1_f = 0.0f, g10_f = 0.0f, g11_f = 0.0f, g12_f = 0.0f, g13_f = 0.0f, stack144_f = 0.0f, stack145_f = 0.0f, stack146_f = 0.0f, stack147_f = 0.0f, stack148_f = 0.0f, stack149_f = 0.0f, stack150_f = 0.0f, stack151_f = 0.0f, stack152_f = 0.0f, stack153_f = 0.0f, stack154_f = 0.0f, stack155_f = 0.0f, stack156_f = 0.0f, stack157_f = 0.0f;
g4 = dd(0.5e0f, 0.0e0f);
g5 = dd_xor(g5, g5);
g0 = g4;
g1 = g5;
g2 = dd_xor(g2, g2);
g2 = dd_mask(g2 < g0);
stack2 = g2;
g2 = g0 * g0;
g2 = dd_fma(g1, g1, g2);
g2 = dd_sqrt(g2);
g3 = dd_abs(g0);
g2 = g2 + g3;
g2 = g2 * dd(0.5e0f, 0.0e0f);
g2 = dd_sqrt(g2);
g3 = g1 / g2;
g3_f = dd_to_f(g3) * 0.5e0f;
g0 = dd_mask(dd_f(g3_f) == dd_f(g3_f));
g3 = dd_and(dd_f(g3_f), g0);
g0 = dd_true(stack2) ? g2 : g3;
g1 = dd_true(stack2) ? g3 : g2;
g2 = dd_sign(g0);
r0 = dd_xor(g0, g2);
r1 = dd_xor(g1, g2);
g4 = r0;
g5 = r1;
stack144 = g4;
stack145 = g5;
r0_f = dd_to_f(mem[0]);
r1_f = dd_to_f(mem[1]);
g6_f = dd_to_f(mem[2]);
g7_f = dd_to_f(mem[3]);
g0_f = dd_to_f(mem[4]);
g1_f = dd_to_f(mem[5]);
g2_f = r1_f * g1_f;
g2_f = fma(r0_f, g0_f, -g2_f);
g3_f = r0_f * g1_f;
r1_f = fma(g0_f, r1_f, g3_f);
r0_f = g2_f;
g0_f = dd_to_f(mem[6]);
g1_f = dd_to_f(mem[7]);
g2_f = g7_f * g1_f;
g2_f = fma(g6_f, g0_f, -g2_f);
g3_f = g6_f * g1_f;
g9_f = fma(g0_f, g7_f, g3_f);
g8_f = g2_f;
stack146_f = r0_f;
stack147_f = r1_f;
stack148_f = g8_f;
stack149_f = g9_f;
g10_f = 0.0e0f;
g12_f = 0.33333334e0f;
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
g2_f = g5_f * dd_to_f(g1);
g2_f = fma(g4_f, dd_to_f(g0), -g2_f);
g3_f = g4_f * dd_to_f(g1);
g5_f = fma(dd_to_f(g0), g5_f, g3_f);
g4_f = g2_f;
g2_f = g11_f * g5_f;
g2_f = fma(g10_f, g4_f, -g2_f);
g3_f = g10_f * g5_f;
g5_f = fma(g4_f, g11_f, g3_f);
g4_f = g2_f;
stack150_f = g4_f;
stack151_f = g5_f;
r0_f = dd_to_f(mem[0]);
r1_f = dd_to_f(mem[1]);
g0_f = dd_to_f(mem[6]);
g1_f = dd_to_f(mem[7]);
g2_f = r1_f * g1_f;
g2_f = fma(r0_f, g0_f, -g2_f);
g3_f = r0_f * g1_f;
r1_f = fma(g0_f, r1_f, g3_f);
r0_f = g2_f;
g0_f = dd_to_f(mem[4]);
g1_f = dd_to_f(mem[5]);
g2_f = g7_f * g1_f;
g2_f = fma(g6_f, g0_f, -g2_f);
g3_f = g6_f * g1_f;
g11_f = fma(g0_f, g7_f, g3_f);
g10_f = g2_f;
stack152_f = r0_f;
stack153_f = r1_f;
stack154_f = g10_f;
stack155_f = g11_f;
g4_f = 0.0e0f;
g6_f = -0.33333334e0f;
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
g2_f = g7_f * dd_to_f(g1);
g2_f = fma(g6_f, dd_to_f(g0), -g2_f);
g3_f = g6_f * dd_to_f(g1);
g7_f = fma(dd_to_f(g0), g7_f, g3_f);
g6_f = g2_f;
g2_f = g5_f * g7_f;
g2_f = fma(g4_f, g6_f, -g2_f);
g3_f = g4_f * g7_f;
g5_f = fma(g6_f, g5_f, g3_f);
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
g2_f = g5_f * dd_to_f(g1);
g2_f = fma(g4_f, dd_to_f(g0), -g2_f);
g3_f = g4_f * dd_to_f(g1);
g5_f = fma(dd_to_f(g0), g5_f, g3_f);
g4_f = g2_f;
g0_f = -0.33333334e0f;
g5_f = g5_f * g0_f;
g4_f = g4_f * g0_f;
stack154_f = g4_f;
stack155_f = g5_f;
g4_f = 0.0e0f;
g6_f = -0.33333334e0f;
g5_f = g6_f;
g4_f = g4_f;
g0_f = stack146_f;
g1_f = stack147_f;
g2_f = g5_f * g1_f;
g2_f = fma(g4_f, g0_f, -g2_f);
g3_f = g4_f * g1_f;
g5_f = fma(g0_f, g5_f, g3_f);
g4_f = g2_f;
g6_f = 0.0e0f;
g8_f = 0.33333334e0f;
g7_f = g8_f;
g6_f = g6_f;
r0_f = stack148_f;
r1_f = stack149_f;
g2_f = g7_f * r1_f;
g2_f = fma(g6_f, r0_f, -g2_f);
g3_f = g6_f * r1_f;
r1_f = fma(r0_f, g7_f, g3_f);
r0_f = g2_f;
g4_f = r0_f + g4_f;
g5_f = r1_f + g5_f;
g0 = stack144;
g1 = stack145;
g2_f = g5_f * dd_to_f(g1);
g2_f = fma(g4_f, dd_to_f(g0), -g2_f);
g3_f = g4_f * dd_to_f(g1);
g5_f = fma(dd_to_f(g0), g5_f, g3_f);
g4_f = g2_f;
stack144_f = g4_f;
stack145_f = g5_f;
stack144_f = g4_f;
stack145_f = g5_f;
g6_f = stack150_f;
g7_f = stack151_f;
mem[8] = dd_f(g6_f);
mem[9] = dd_f(g7_f);
mem[8] = dd_f(g6_f);
mem[9] = dd_f(g7_f);
g8_f = stack156_f;
g9_f = stack157_f;
mem[10] = dd_f(g8_f);
mem[11] = dd_f(g9_f);
mem[10] = dd_f(g8_f);
mem[11] = dd_f(g9_f);
g4_f = stack154_f;
g5_f = stack155_f;
mem[12] = dd_f(g4_f);
mem[13] = dd_f(g5_f);
mem[12] = dd_f(g4_f);
mem[13] = dd_f(g5_f);
g4_f = stack144_f;
g5_f = stack145_f;
mem[14] = dd_f(g4_f);
mem[15] = dd_f(g5_f);
return 0;
}

/* the symjit kernel k_e2d4ab976ad89705 in mixed precision (kernel_msl.jl): 59 f32 and 17 df64 operations, 48 conversions */
static inline int k_e2d4ab976ad89705(thread df64 *mem) {
df64 g0 = dd_f(0.0f), g1 = dd_f(0.0f), g2 = dd_f(0.0f), g3 = dd_f(0.0f), g4 = dd_f(0.0f), g5 = dd_f(0.0f), r0 = dd_f(0.0f), r1 = dd_f(0.0f), stack2 = dd_f(0.0f), stack144 = dd_f(0.0f), stack145 = dd_f(0.0f);
float g0_f = 0.0f, g1_f = 0.0f, g2_f = 0.0f, g3_f = 0.0f, g4_f = 0.0f, g5_f = 0.0f, g6_f = 0.0f, g7_f = 0.0f, g8_f = 0.0f, g9_f = 0.0f, r0_f = 0.0f, r1_f = 0.0f, g10_f = 0.0f, g11_f = 0.0f, g12_f = 0.0f, g13_f = 0.0f, stack144_f = 0.0f, stack145_f = 0.0f, stack146_f = 0.0f, stack147_f = 0.0f, stack148_f = 0.0f, stack149_f = 0.0f, stack150_f = 0.0f, stack151_f = 0.0f, stack152_f = 0.0f, stack153_f = 0.0f;
g4 = dd(0.5e0f, 0.0e0f);
g5 = dd_xor(g5, g5);
g0 = g4;
g1 = g5;
g2 = dd_xor(g2, g2);
g2 = dd_mask(g2 < g0);
stack2 = g2;
g2 = g0 * g0;
g2 = dd_fma(g1, g1, g2);
g2 = dd_sqrt(g2);
g3 = dd_abs(g0);
g2 = g2 + g3;
g2 = g2 * dd(0.5e0f, 0.0e0f);
g2 = dd_sqrt(g2);
g3 = g1 / g2;
g3_f = dd_to_f(g3) * 0.5e0f;
g0 = dd_mask(dd_f(g3_f) == dd_f(g3_f));
g3 = dd_and(dd_f(g3_f), g0);
g0 = dd_true(stack2) ? g2 : g3;
g1 = dd_true(stack2) ? g3 : g2;
g2 = dd_sign(g0);
r0 = dd_xor(g0, g2);
r1 = dd_xor(g1, g2);
g4 = r0;
g5 = r1;
stack144 = g4;
stack145 = g5;
r0_f = dd_to_f(mem[0]);
r1_f = dd_to_f(mem[1]);
g6_f = dd_to_f(mem[2]);
g7_f = dd_to_f(mem[3]);
g0_f = dd_to_f(mem[4]);
g1_f = dd_to_f(mem[5]);
g2_f = r1_f * g1_f;
g2_f = fma(r0_f, g0_f, -g2_f);
g3_f = r0_f * g1_f;
r1_f = fma(g0_f, r1_f, g3_f);
r0_f = g2_f;
g0_f = dd_to_f(mem[6]);
g1_f = dd_to_f(mem[7]);
g2_f = g7_f * g1_f;
g2_f = fma(g6_f, g0_f, -g2_f);
g3_f = g6_f * g1_f;
g9_f = fma(g0_f, g7_f, g3_f);
g8_f = g2_f;
stack146_f = r0_f;
stack147_f = r1_f;
r0_f = stack146_f;
r1_f = stack147_f;
g10_f = r0_f + g8_f;
g11_f = r1_f + g9_f;
g0 = stack144;
g1 = stack145;
g2_f = g11_f * dd_to_f(g1);
g2_f = fma(g10_f, dd_to_f(g0), -g2_f);
g3_f = g10_f * dd_to_f(g1);
g11_f = fma(dd_to_f(g0), g11_f, g3_f);
g10_f = g2_f;
g3_f = -g11_f;
g11_f = g10_f;
g10_f = g3_f;
r0_f = dd_to_f(mem[0]);
r1_f = dd_to_f(mem[1]);
g0_f = dd_to_f(mem[6]);
g1_f = dd_to_f(mem[7]);
g2_f = r1_f * g1_f;
g2_f = fma(r0_f, g0_f, -g2_f);
g3_f = r0_f * g1_f;
r1_f = fma(g0_f, r1_f, g3_f);
r0_f = g2_f;
g0_f = dd_to_f(mem[4]);
g1_f = dd_to_f(mem[5]);
g2_f = g7_f * g1_f;
g2_f = fma(g6_f, g0_f, -g2_f);
g3_f = g6_f * g1_f;
g13_f = fma(g0_f, g7_f, g3_f);
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
g2_f = g5_f * dd_to_f(g1);
g2_f = fma(g4_f, dd_to_f(g0), -g2_f);
g3_f = g4_f * dd_to_f(g1);
g5_f = fma(dd_to_f(g0), g5_f, g3_f);
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
g2_f = g5_f * dd_to_f(g1);
g2_f = fma(g4_f, dd_to_f(g0), -g2_f);
g3_f = g4_f * dd_to_f(g1);
g5_f = fma(dd_to_f(g0), g5_f, g3_f);
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
g2_f = g5_f * dd_to_f(g1);
g2_f = fma(g4_f, dd_to_f(g0), -g2_f);
g3_f = g4_f * dd_to_f(g1);
g5_f = fma(dd_to_f(g0), g5_f, g3_f);
g4_f = g2_f;
stack144_f = g4_f;
stack145_f = g5_f;
stack144_f = g4_f;
stack145_f = g5_f;
mem[8] = dd_f(g10_f);
mem[9] = dd_f(g11_f);
mem[8] = dd_f(g10_f);
mem[9] = dd_f(g11_f);
g6_f = stack152_f;
g7_f = stack153_f;
mem[10] = dd_f(g6_f);
mem[11] = dd_f(g7_f);
mem[10] = dd_f(g6_f);
mem[11] = dd_f(g7_f);
g4_f = stack150_f;
g5_f = stack151_f;
mem[12] = dd_f(g4_f);
mem[13] = dd_f(g5_f);
mem[12] = dd_f(g4_f);
mem[13] = dd_f(g5_f);
g4_f = stack144_f;
g5_f = stack145_f;
mem[14] = dd_f(g4_f);
mem[15] = dd_f(g5_f);
return 0;
}

static int uir_stub(uint stub, thread const uir_program &g, thread const uir_arena &a, device const uint *w,
                    device const uint *scaleset) {
switch (stub) {
case 0: {  /* k_5acfa4bfbabd18e1 */
df64 mem[17];
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
uir_output(g, a, w, 0u, cdd(mem[8], mem[9]));
uir_output(g, a, w, 1u, cdd(mem[10], mem[11]));
uir_output(g, a, w, 2u, cdd(mem[12], mem[13]));
uir_output(g, a, w, 3u, cdd(mem[14], mem[15]));
return 0;
}
case 1: {  /* k_e2d4ab976ad89705 */
df64 mem[17];
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
uir_output(g, a, w, 0u, cdd(mem[8], mem[9]));
uir_output(g, a, w, 1u, cdd(mem[10], mem[11]));
uir_output(g, a, w, 2u, cdd(mem[12], mem[13]));
uir_output(g, a, w, 3u, cdd(mem[14], mem[15]));
return 0;
}
default: return -1;
}
}
