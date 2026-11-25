/* A UMAMI matrix element (madspace/include/madspace/umami.h) for one program: one process, one LC
 * colour flow, one leg permutation, compiled by PacRunner (runner.c: run(out, p, momenta), the
 * outputs of every runtime helicity at one point). PacRunner's umami_libraries writes one library per
 * (process, colour flow, permutation) of a pyAmpliCol artifact's UMAMI provider, with this file, the
 * program's runner.c and pacops.c, and umami_program.h (its constants, below). The flavour
 * splitting is MadSpace's: a flavour of the provider is a cross section of its own
 * (flavor_diff_xs_indices), and each library sees flavour 0.
 *
 * Per event, as the stock pyAmpliCol wrapper (API/umami/umami.c) evaluates one contribution:
 *   - the momenta, input[event + stride*(particle + particle_count*component)], read in the
 *     caller's leg order: the library's leg permutation (runtime leg j = input leg permutation[j])
 *     is folded into the program's crossing when the library is built (umami_libraries);
 *   - run(): the runtime helicities' values;
 *   - with a helicity index input (exported numbering, the reweight pass): that helicity's value ×
 *     factor (0 when the program does not have it);
 *   - else the values summed in order, × factor; with a helicity index output (the LC pass) one
 *     helicity drawn by inverse CDF from the random helicity (zeros skipped).
 * Bit for bit the stock library in both passes MadGraph's [amplicol] makes. Without either index
 * the stock library asks RustiCol for the helicity sum, which RustiCol takes over helicity orbits
 * (one representative × the orbit's weight): the export does not describe that shortcut, and the
 * sum here agrees with it to rounding (about 2e-16).
 *   - with an alpha_s input different from the export's: each value × (alpha_s / alpha_s0)^power,
 *     the power the export proved (the programs' couplings do not depend on alpha_s; RustiCol
 *     recomputes (4 pi alpha_s)^power instead, so values at another alpha_s agree to rounding).
 *     The input is always accepted (MadGraph passes it whenever the model has alpha_s); a program
 *     without a proven power accepts only the export's alpha_s (relative 1e-12) and fails otherwise;
 *   - the colour index is the program's.
 * The end of each event (scale, fixed or summed helicity, draw, factor) is pac_finish
 * (umami_finish.h), shared with the GPU build.
 * The model parameters are those of the export: umami_set_parameter accepts a known parameter at
 * its exported value (relative 1e-12) and refuses any other value.
 * The flavour input must be 0; the channel input is not read (the library is one program).
 *
 * With PAC_CUDA (umami_libraries(...; device = :cuda)) the program runs on the GPU (uirgpu.h: the
 * UIR interpreter, one thread per point) and the library still identifies as a CPU provider:
 * MadSpace passes host memory and calls from its CPU threads. Per call, the host checks the inputs
 * and computes the alpha_s scales (glibc's pow, as the CPU build); one GPU run (pac_gpu_umami)
 * uploads the momentum rows of the call as they are, runs the program and pac_finish in each
 * event's thread, and returns only the matrix elements and helicity indices: the CPU build's bits.
 * The GPU memory and streams are the process-wide slot pool's (libpacpool.so, pool.h): a call
 * borrows a free slot, so MadSpace's threads run on the GPU at the same time (up to the pool's
 * slots; by default one per CPU), whatever the number of libraries.
 *
 * With PAC_CUDA_PROVIDER (umami_libraries(...; device = :cuda_provider)) the library identifies as
 * CUDA: MadSpace's GPU runtime passes device pointers (count = stride = the batch, offset 0) and
 * its stream as the output UMAMI_OUT_GPU_STREAM. One call queues the events on that stream
 * (pac_gpu_provider: the checks, the program and pac_finish in each event's thread, the outputs
 * written in place) and returns without waiting. An event that fails a check (pac_check, the
 * momenta, the result) gets a NaN matrix element and leaves its code in an error flag; the next
 * call that sees the flag prints it, clears it and returns UMAMI_ERROR without queuing its own
 * events (umami_free prints one still pending). The alpha_s scale is the device's pow, which may
 * differ from glibc's by an ulp.
 *
 * With PAC_SIMD (umami_libraries(...; device = "simd")) the program is one symjit kernel (compile_simd:
 * kernel4.o, pac_simd4, AVX2; kernel8.o, pac_simd8, AVX-512, when the building CPU has AVX-512 and
 * runner.h defines PAC_SIMD_HAVE8). An instance takes AVX-512 if the library has it and the CPU too, else
 * AVX2; PAC_SIMD_LANES=4 or 8 asks for one. A call's
 * events go to the kernel PAC_SIMD_CHUNK at a time: their momenta as columns (the last chunk padded with
 * its last event to a multiple of the lanes), the kernel on 4 or 8 events per call, then each event's
 * end (pac_finish) on its column of outputs. The checks, the scales and the end are the CPU build's, and
 * the kernel computes runner.c's values: the CPU build's bits. A kernel call uses PAC_SIMD8_STACK_BYTES
 * (PAC_SIMD4_STACK_BYTES) of the calling thread's stack. The kernels are symjit's JIT code in object files
 * (compile_simd, simd/elf.py): no relocations, no library calls.
 *
 * With PAC_SIMD and PAC_SIMD_SPLIT (device "simd", precision "mixed") the kernels are the split ones,
 * two a chunk (compile_simd(precision = "mixed"), codegen/compile_split.py): kernel A in f64 (kernelA4.o /
 * kernelA8.o, pac_split_a4 / a8) from the momenta to the transfers (forms, 1 / k·k, sources, the model
 * kernels' wide values), rounded to floats, then kernel B in f32 (kernelB8.o / kernelB16.o, AVX-512 with
 * PAC_SIMD_HAVE16; float columns: it reads element idx at byte 8 idx, called with idx / 2) to the
 * outputs, widened for the event's end; PAC_SIMD_LANES=8 or 16 asks for B's lanes (A runs half as many). */
#include "umami.h"
#include "runner.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if defined(PAC_CUDA) || defined(PAC_CUDA_PROVIDER)
#include "uirgpu.h"
#endif
#ifdef PAC_SIMD
#ifndef PAC_SIMD_CHUNK
#define PAC_SIMD_CHUNK 64                      /* events per kernel batch: a multiple of 16 */
#endif
#if defined(PAC_SIMD_SPLIT)
#include "kernelA4.h"
#include "kernelB8.h"
#ifdef PAC_SIMD_HAVE16
#include "kernelA8.h"
#include "kernelB16.h"
#endif
#define PAC_SIMD_STATES PAC_SPLIT_A4_COUNT_STATES      /* kernel A's: 4 x the legs, the momenta */
#define PAC_SPLIT_TRANSFERS PAC_SPLIT_A4_COUNT_OBS     /* A's outputs, B's states */
#define PAC_SIMD_COLUMNS (PAC_SPLIT_A4_COUNT_STATES + PAC_SPLIT_A4_COUNT_OBS)
typedef double pac_simd_real;
#else
#include "kernel4.h"
#ifdef PAC_SIMD_HAVE8
#include "kernel8.h"
#endif
#define PAC_SIMD_STATES PAC_SIMD4_COUNT_STATES   /* 4 x the legs: the momenta */
#define PAC_SIMD_COLUMNS (PAC_SIMD4_COUNT_STATES + PAC_SIMD4_COUNT_OBS)
typedef double pac_simd_real;
#endif
typedef int (*pac_simd_kernel)(double *, const symjit_slice *, size_t, const double *);
#endif

typedef struct {
    const char *name;
    double real, imaginary;
} PacParameter;

#include "umami_finish.h"
#include "umami_program.h"

#define PAC_EXPORT __attribute__((visibility("default")))

typedef struct {
    double *arena;    /* run()'s scratch, RUNNER_ARENA_SIZE (CPU) */
    double momenta[4 * PAC_PARTICLES];
    double values[RUNNER_OUTPUTS];
#ifdef PAC_SIMD
    pac_simd_kernel kernel;
    size_t lanes;
    pac_simd_real *columns;                    /* PAC_SIMD_COLUMNS columns of PAC_SIMD_CHUNK events */
#ifdef PAC_SIMD_SPLIT
    double widened[RUNNER_OUTPUTS * PAC_SIMD_CHUNK];   /* kernel B's outputs, widened */
    pac_simd_kernel kernel_b;
    float *bcolumns;                           /* B's columns: the transfers (rounded), then its outputs */
    symjit_slice bslices[PAC_SPLIT_TRANSFERS + RUNNER_OUTPUTS];
#endif
    symjit_slice slices[PAC_SIMD_COLUMNS];
    double scale[PAC_SIMD_CHUNK], random[PAC_SIMD_CHUNK];
    int scaled[PAC_SIMD_CHUNK], fixed[PAC_SIMD_CHUNK];
#endif
#ifdef PAC_CUDA
    size_t capacity;  /* events the scale buffer holds */
    double *scales;   /* the alpha_s scale of each event */
#endif
#ifdef PAC_CUDA_PROVIDER
    int *error;       /* the error flag (mapped pinned): a failing event's code, until reported */
#endif
} Instance;

#ifdef PAC_SIMD
/* the lanes of this CPU's kernel (8: AVX-512, 4: AVX2; split: kernel B's, 16, 8), PAC_SIMD_LANES if it asks for
 * one; 0: none */
static size_t simd_lanes(void) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && defined(PAC_SIMD_SPLIT)
    __builtin_cpu_init();
#ifdef PAC_SIMD_HAVE16
    bool has16 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#else
    bool has16 = false;   /* built without the AVX-512 kernel */
#endif
    bool has8 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    char const *want = getenv("PAC_SIMD_LANES");
    if (want != NULL && strcmp(want, "8") == 0) return has8 ? 8 : 0;
    if (want != NULL && strcmp(want, "16") == 0) return has16 ? 16 : 0;
    return has16 ? 16 : has8 ? 8 : 0;
#elif defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
    __builtin_cpu_init();
#ifdef PAC_SIMD_HAVE8
    bool has8 = __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq");
#else
    bool has8 = false;   /* built without the AVX-512 kernel */
#endif
    bool has4 = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
    char const *want = getenv("PAC_SIMD_LANES");
    if (want != NULL && strcmp(want, "4") == 0) return has4 ? 4 : 0;
    if (want != NULL && strcmp(want, "8") == 0) return has8 ? 8 : 0;
    return has8 ? 8 : has4 ? 4 : 0;
#else
    return 0;
#endif
}

static bool simd_setup(Instance *instance) {
    instance->lanes = simd_lanes();
    if (instance->lanes == 0) {
        fprintf(stderr, "umami (%s %s): device simd needs an x86-64 CPU with AVX2 and FMA%s\n", RUNNER_PROCESS,
                RUNNER_COLOUR, getenv("PAC_SIMD_LANES") ? " (or the kernel PAC_SIMD_LANES asks for: 8 needs AVX-512 "
                "F and DQ and a library built on such a CPU)" : "");
        return false;
    }
#if defined(PAC_SIMD_SPLIT) && defined(PAC_SIMD_HAVE16)
    instance->kernel = instance->lanes == 16 ? pac_split_a8 : pac_split_a4;
    instance->kernel_b = instance->lanes == 16 ? pac_split_b16 : pac_split_b8;
#elif defined(PAC_SIMD_SPLIT)
    instance->kernel = pac_split_a4;
    instance->kernel_b = pac_split_b8;
#elif defined(PAC_SIMD_HAVE8)
    instance->kernel = instance->lanes == 8 ? pac_simd8 : pac_simd4;
#else
    instance->kernel = pac_simd4;
#endif
    instance->columns = calloc((size_t)PAC_SIMD_COLUMNS * PAC_SIMD_CHUNK, sizeof(pac_simd_real));
    if (instance->columns == NULL) return false;
    for (size_t j = 0; j < PAC_SIMD_COLUMNS; ++j) {
        instance->slices[j].ptr = (double *)(instance->columns + j * PAC_SIMD_CHUNK);
        instance->slices[j].len = PAC_SIMD_CHUNK;
    }
#ifdef PAC_SIMD_SPLIT
    instance->bcolumns = calloc((size_t)(PAC_SPLIT_TRANSFERS + RUNNER_OUTPUTS) * PAC_SIMD_CHUNK, sizeof(float));
    if (instance->bcolumns == NULL) return false;
    for (size_t j = 0; j < PAC_SPLIT_TRANSFERS + RUNNER_OUTPUTS; ++j) {
        instance->bslices[j].ptr = (double *)(instance->bcolumns + j * PAC_SIMD_CHUNK);
        instance->bslices[j].len = PAC_SIMD_CHUNK;
    }
#endif
    return true;
}

/* the chunk's n events (the first is `first`): padded to the lanes, the kernel, each event's end */
static bool simd_flush(Instance *instance, size_t n, size_t first, bool draw, void *const *out) {
    size_t const lanes = instance->lanes, padded = (n + lanes - 1) / lanes * lanes;
    for (size_t j = 0; j < PAC_SIMD_STATES; ++j)
        for (size_t c = n; c < padded; ++c)
            instance->columns[j * PAC_SIMD_CHUNK + c] = instance->columns[j * PAC_SIMD_CHUNK + n - 1];
#if defined(PAC_SIMD_SPLIT)
    /* kernel A (f64, lanes / 2 a call), its transfers rounded to floats, kernel B (f32, lanes a call) */
    for (size_t i = 0; i < padded; i += lanes / 2)
        if (instance->kernel(NULL, instance->slices, i, NULL) != 0) return false;
    double const *transfers = instance->columns + (size_t)PAC_SIMD_STATES * PAC_SIMD_CHUNK;
    for (size_t t = 0; t < (size_t)PAC_SPLIT_TRANSFERS * PAC_SIMD_CHUNK; ++t)
        instance->bcolumns[t] = (float)transfers[t];
    for (size_t i = 0; i < padded; i += lanes)   /* floats: element i at byte 4 i = 8 (i / 2) */
        if (instance->kernel_b(NULL, instance->bslices, i / 2, NULL) != 0) return false;
    float const *bout = instance->bcolumns + (size_t)PAC_SPLIT_TRANSFERS * PAC_SIMD_CHUNK;
    for (size_t h = 0; h < RUNNER_OUTPUTS; ++h)
        for (size_t c = 0; c < n; ++c)
            instance->widened[h * PAC_SIMD_CHUNK + c] = (double)bout[h * PAC_SIMD_CHUNK + c];
    double const *values = instance->widened;
#else
    for (size_t i = 0; i < padded; i += lanes)
        if (instance->kernel(NULL, instance->slices, i, NULL) != 0) return false;
    double const *values = instance->columns + (size_t)PAC_SIMD_STATES * PAC_SIMD_CHUNK;
#endif
    for (size_t c = 0; c < n; ++c) {
        double me;
        int label;
        if (pac_finish(values + c, PAC_SIMD_CHUNK, instance->scaled[c], instance->scale[c], instance->fixed[c],
                       instance->random[c], draw, &me, &label) != 0)
            return false;
        if (out[UMAMI_OUT_MATRIX_ELEMENT]) ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[first + c] = me;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[first + c] = label;
    }
    return true;
}
#endif

#ifdef PAC_CUDA
/* the instance's scale buffer for `count` events */
static bool reserve_batch(Instance *instance, size_t count) {
    if (count <= instance->capacity) return true;
    size_t n = count > 2 * instance->capacity ? count : 2 * instance->capacity;
    double *x = realloc(instance->scales, (n + 1) * sizeof(double));
    if (x == NULL) return false;
    instance->scales = x;
    instance->capacity = n;
    return true;
}
#endif

static bool const supported_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
    [UMAMI_IN_ALPHA_S] = true,   /* as the stock library; see umami_matrix_element */
    [UMAMI_IN_FLAVOR_INDEX] = true,
    [UMAMI_IN_RANDOM_HELICITY] = true,
    [UMAMI_IN_HELICITY_INDEX] = true,
    [UMAMI_IN_CHANNEL_INDEX] = true,
};
static bool const required_inputs[UMAMI_INPUT_KEY_COUNT] = {
    [UMAMI_IN_MOMENTA] = true,
};
static bool const supported_outputs[UMAMI_OUTPUT_KEY_COUNT] = {
    [UMAMI_OUT_MATRIX_ELEMENT] = true,
    [UMAMI_OUT_COLOR_INDEX] = true,
    [UMAMI_OUT_HELICITY_INDEX] = true,
#ifdef PAC_CUDA_PROVIDER
    [UMAMI_OUT_GPU_STREAM] = true,
#endif
};

PAC_EXPORT UmamiStatus umami_get_meta(UmamiMetaKey key, void *result) {
    if (result == NULL) return UMAMI_ERROR;
    switch (key) {
#ifdef PAC_CUDA_PROVIDER
    case UMAMI_META_DEVICE: *(UmamiDevice *)result = UMAMI_DEVICE_CUDA; break;
#else
    case UMAMI_META_DEVICE: *(UmamiDevice *)result = UMAMI_DEVICE_CPU; break;
#endif
    case UMAMI_META_PARTICLE_COUNT: *(int *)result = PAC_PARTICLES; break;
    case UMAMI_META_HELICITY_COUNT: *(int *)result = PAC_HELICITY_COUNT; break;
    case UMAMI_META_COLOR_COUNT: *(int *)result = PAC_COLOR_COUNT; break;
    case UMAMI_META_MASSES: memcpy(result, pac_masses, sizeof(pac_masses)); break;
    default: return UMAMI_ERROR_UNSUPPORTED_META;
    }
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_supported_inputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_required_inputs(bool const **required, int *count) {
    if (required == NULL || count == NULL) return UMAMI_ERROR;
    *required = required_inputs; *count = UMAMI_INPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_supported_outputs(bool const **supported, int *count) {
    if (supported == NULL || count == NULL) return UMAMI_ERROR;
    *supported = supported_outputs; *count = UMAMI_OUTPUT_KEY_COUNT;
    return UMAMI_SUCCESS;
}

/* param_card_path is not read: the program carries its parameters */
PAC_EXPORT UmamiStatus umami_initialize(UmamiHandle *handle, char const *param_card_path) {
    (void)param_card_path;
    if (handle == NULL) return UMAMI_ERROR;
    Instance *instance = calloc(1, sizeof(*instance));
    if (instance == NULL) return UMAMI_ERROR;
#if defined(PAC_CUDA_PROVIDER)
    instance->error = pac_gpu_error_flag();
    if (instance->error == NULL) { free(instance); return UMAMI_ERROR; }
#elif defined(PAC_SIMD)
    if (!simd_setup(instance)) { free(instance->columns); free(instance); return UMAMI_ERROR; }
#elif !defined(PAC_CUDA)
    instance->arena = calloc(RUNNER_ARENA_SIZE, sizeof(double));
    if (instance->arena == NULL) { free(instance); return UMAMI_ERROR; }
#endif
    *handle = instance;
    return UMAMI_SUCCESS;
}

static PacParameter const *find_parameter(char const *name) {
    for (size_t i = 0; i < sizeof(pac_parameters) / sizeof(pac_parameters[0]); ++i)
        if (pac_parameters[i].name != NULL && strcmp(pac_parameters[i].name, name) == 0)
            return &pac_parameters[i];
    return NULL;
}

static bool close_to(double x, double y) { return pac_close_to(x, y); }

static bool alpha_s_warned = false;   /* the unknown-power refusal is reported once */

#ifdef PAC_CUDA_PROVIDER
/* an earlier call's failing event (the flag), printed and cleared; true: there was one */
static bool report_error(Instance *instance) {
    int code = *(volatile int *)instance->error;
    if (code == 0) return false;
    *(volatile int *)instance->error = 0;   /* reported once: later calls go on */
    static char const *const what[] = {"", "a flavour index other than 0", "a random number outside [0, 1)",
        "a helicity index outside the exported ones", "an alpha_s it cannot take (non-finite, negative, or "
        "another than the exported one without a proven power)", "a non-finite momentum",
        "a failing kernel", "a non-finite matrix element"};
    fprintf(stderr, "umami (%s %s): an event of an earlier call (its matrix element is NaN) had %s\n",
            RUNNER_PROCESS, RUNNER_COLOUR,
            code > 0 && code < (int)(sizeof(what) / sizeof(what[0])) ? what[code] : "an unknown failure");
    return true;
}
#endif

PAC_EXPORT UmamiStatus umami_set_parameter(UmamiHandle handle, char const *name,
                                           double real, double imaginary) {
    if (handle == NULL || name == NULL) return UMAMI_ERROR;
    PacParameter const *p = find_parameter(name);
    if (p == NULL) return UMAMI_ERROR_UNKNOWN_PARAMETER;
    if (close_to(real, p->real) && close_to(imaginary, p->imaginary)) return UMAMI_SUCCESS;
    fprintf(stderr, "umami (%s %s): parameter %s = %.17g + %.17gi, but the program was exported "
            "with %.17g + %.17gi: export it again with these parameters\n", RUNNER_PROCESS,
            RUNNER_COLOUR, name, real, imaginary, p->real, p->imaginary);
    return UMAMI_ERROR;
}

PAC_EXPORT UmamiStatus umami_get_parameter(UmamiHandle handle, char const *name,
                                           double *real, double *imaginary) {
    if (handle == NULL || name == NULL || real == NULL) return UMAMI_ERROR;
    PacParameter const *p = find_parameter(name);
    if (p == NULL) return UMAMI_ERROR_UNKNOWN_PARAMETER;
    *real = p->real;
    if (imaginary != NULL) *imaginary = p->imaginary;
    return UMAMI_SUCCESS;
}

PAC_EXPORT UmamiStatus umami_matrix_element(UmamiHandle handle, size_t count, size_t stride, size_t offset,
    size_t input_count, UmamiInputKey const *input_keys, void const *const *inputs,
    size_t output_count, UmamiOutputKey const *output_keys, void *const *outputs) {
    if (handle == NULL || offset > stride || count > stride - offset ||
        (input_count && (input_keys == NULL || inputs == NULL)) ||
        (output_count && (output_keys == NULL || outputs == NULL))) return UMAMI_ERROR;
    void const *in[UMAMI_INPUT_KEY_COUNT] = {0};
    void *out[UMAMI_OUTPUT_KEY_COUNT] = {0};
#ifdef PAC_CUDA_PROVIDER
    bool has_stream = false;
    void *stream = NULL;
#endif
    for (size_t i = 0; i < input_count; ++i) {
        int key = input_keys[i];
        if (key < 0 || key >= UMAMI_INPUT_KEY_COUNT || !supported_inputs[key])
            return UMAMI_ERROR_UNSUPPORTED_INPUT;
        if (inputs[i] == NULL || in[key] != NULL) return UMAMI_ERROR;
        in[key] = inputs[i];
    }
    for (size_t i = 0; i < output_count; ++i) {
        int key = output_keys[i];
        if (key < 0 || key >= UMAMI_OUTPUT_KEY_COUNT || !supported_outputs[key])
            return UMAMI_ERROR_UNSUPPORTED_OUTPUT;
#ifdef PAC_CUDA_PROVIDER
        if (key == UMAMI_OUT_GPU_STREAM) {   /* a cudaStream_t: NULL is the default stream */
            if (has_stream) return UMAMI_ERROR;
            has_stream = true;
            stream = outputs[i];
            continue;
        }
#endif
        if (outputs[i] == NULL || out[key] != NULL) return UMAMI_ERROR;
        out[key] = outputs[i];
    }
    if (count == 0) return UMAMI_SUCCESS;
    if (in[UMAMI_IN_MOMENTA] == NULL) return UMAMI_ERROR_MISSING_INPUT;
    Instance *instance = handle;
#ifdef PAC_CUDA_PROVIDER
    /* device pointers: everything happens on the stream; an earlier call's failing event first */
    if (report_error(instance)) return UMAMI_ERROR;
    uirgpu_call call = {
        (double const *)in[UMAMI_IN_MOMENTA] + offset, stride,
        in[UMAMI_IN_ALPHA_S] ? (double const *)in[UMAMI_IN_ALPHA_S] + offset : NULL,
        in[UMAMI_IN_FLAVOR_INDEX] ? (int const *)in[UMAMI_IN_FLAVOR_INDEX] + offset : NULL,
        in[UMAMI_IN_RANDOM_HELICITY] ? (double const *)in[UMAMI_IN_RANDOM_HELICITY] + offset : NULL,
        in[UMAMI_IN_HELICITY_INDEX] ? (int const *)in[UMAMI_IN_HELICITY_INDEX] + offset : NULL,
        out[UMAMI_OUT_MATRIX_ELEMENT] ? (double *)out[UMAMI_OUT_MATRIX_ELEMENT] + offset : NULL,
        out[UMAMI_OUT_HELICITY_INDEX] ? (int *)out[UMAMI_OUT_HELICITY_INDEX] + offset : NULL,
        out[UMAMI_OUT_COLOR_INDEX] ? (int *)out[UMAMI_OUT_COLOR_INDEX] + offset : NULL,
        out[UMAMI_OUT_HELICITY_INDEX] != NULL && in[UMAMI_IN_HELICITY_INDEX] == NULL,
    };
    return pac_gpu_provider(&call, count, stream, instance->error) == 0 ? UMAMI_SUCCESS : UMAMI_ERROR;
#else
    double const *momenta = in[UMAMI_IN_MOMENTA];
    size_t const n = PAC_PARTICLES;
    bool const draw = out[UMAMI_OUT_HELICITY_INDEX] != NULL && in[UMAMI_IN_HELICITY_INDEX] == NULL;
#ifdef PAC_CUDA
    if (!reserve_batch(instance, count)) return UMAMI_ERROR;
    bool scaled = false;
#endif
    for (size_t event = offset; event < offset + count; ++event) {
        double random = in[UMAMI_IN_RANDOM_HELICITY] ?
            ((double const *)in[UMAMI_IN_RANDOM_HELICITY])[event] : 0.5;
        /* a fixed exported helicity; one the program does not have contributes nothing */
        int fixed = in[UMAMI_IN_HELICITY_INDEX] ? ((int const *)in[UMAMI_IN_HELICITY_INDEX])[event] : -1;
        double alpha_s = in[UMAMI_IN_ALPHA_S] ? ((double const *)in[UMAMI_IN_ALPHA_S])[event] : 0.0;
        double scale;
        int event_scaled;
        int code = pac_check(in[UMAMI_IN_FLAVOR_INDEX] != NULL,
                             in[UMAMI_IN_FLAVOR_INDEX] ? ((int const *)in[UMAMI_IN_FLAVOR_INDEX])[event] : 0, random,
                             in[UMAMI_IN_HELICITY_INDEX] != NULL, fixed, in[UMAMI_IN_ALPHA_S] != NULL, alpha_s,
                             &scale, &event_scaled);
        if (code != 0) {
            if (code == PAC_E_ALPHA_S && isfinite(alpha_s) && alpha_s >= 0.0 && !alpha_s_warned) {
                alpha_s_warned = true;
                fprintf(stderr, "umami (%s %s): alpha_s = %.17g, but the program was exported with "
                        "%.17g and its alpha_s power is unknown: export it again at this alpha_s\n",
                        RUNNER_PROCESS, RUNNER_COLOUR, alpha_s, PAC_ALPHA_S0);
            }
            return UMAMI_ERROR;
        }
#ifdef PAC_CUDA
        /* × 1.0 for an event at alpha_s0 changes no bit: one scale array for the call */
        instance->scales[event - offset] = scale;
        scaled = scaled || event_scaled;
#elif defined(PAC_SIMD)
        size_t c = (event - offset) % PAC_SIMD_CHUNK;   /* the event's place in its chunk */
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < 4; ++k) {
                double value = momenta[event + stride * (n * k + j)];
                if (!isfinite(value)) return UMAMI_ERROR;
                instance->columns[(4 * j + k) * PAC_SIMD_CHUNK + c] = value;
            }
        instance->scale[c] = scale;
        instance->scaled[c] = event_scaled;
        instance->fixed[c] = fixed;
        instance->random[c] = random;
        if ((c == PAC_SIMD_CHUNK - 1 || event == offset + count - 1) && !simd_flush(instance, c + 1, event - c, draw, out))
            return UMAMI_ERROR;
#else
        for (size_t j = 0; j < n; ++j)
            for (size_t k = 0; k < 4; ++k) {
                double value = momenta[event + stride * (n * k + j)];
                if (!isfinite(value)) return UMAMI_ERROR;
                instance->momenta[4 * j + k] = value;
            }
        if (run(instance->values, instance->arena, instance->momenta) != 0) return UMAMI_ERROR;
        double me;
        int label;
        if (pac_finish(instance->values, 1, event_scaled, scale, fixed, random, draw, &me, &label) != 0)
            return UMAMI_ERROR;
        if (out[UMAMI_OUT_MATRIX_ELEMENT]) ((double *)out[UMAMI_OUT_MATRIX_ELEMENT])[event] = me;
        if (out[UMAMI_OUT_HELICITY_INDEX]) ((int *)out[UMAMI_OUT_HELICITY_INDEX])[event] = label;
#endif
        if (out[UMAMI_OUT_COLOR_INDEX]) ((int *)out[UMAMI_OUT_COLOR_INDEX])[event] = PAC_COLOR_INDEX;
    }
#ifdef PAC_CUDA
    double *me = out[UMAMI_OUT_MATRIX_ELEMENT] ? (double *)out[UMAMI_OUT_MATRIX_ELEMENT] + offset : NULL;
    int *label = out[UMAMI_OUT_HELICITY_INDEX] ? (int *)out[UMAMI_OUT_HELICITY_INDEX] + offset : NULL;
    if (pac_gpu_umami(momenta + offset, stride, count, scaled ? instance->scales : NULL,
                      in[UMAMI_IN_HELICITY_INDEX] ? (int const *)in[UMAMI_IN_HELICITY_INDEX] + offset : NULL,
                      in[UMAMI_IN_RANDOM_HELICITY] ? (double const *)in[UMAMI_IN_RANDOM_HELICITY] + offset : NULL,
                      draw, me, label) != 0)
        return UMAMI_ERROR;
    (void)n;
#endif
    return UMAMI_SUCCESS;
#endif   /* PAC_CUDA_PROVIDER */
}

PAC_EXPORT UmamiStatus umami_free(UmamiHandle handle) {
    if (handle == NULL) return UMAMI_SUCCESS;
    Instance *instance = handle;
#ifdef PAC_CUDA_PROVIDER
    report_error(instance);
    pac_gpu_free_error_flag(instance->error);
#endif
#ifdef PAC_CUDA
    free(instance->scales);
#endif
#ifdef PAC_SIMD
    free(instance->columns);
#endif
#ifdef PAC_SIMD_SPLIT
    free(instance->bcolumns);
#endif
    free(instance->arena);
    free(instance);
    return UMAMI_SUCCESS;
}
