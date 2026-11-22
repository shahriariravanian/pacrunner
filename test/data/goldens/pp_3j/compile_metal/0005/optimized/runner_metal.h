/* p_p_to_j_j_j_101 flow:3,5,1,2,4: the program for Metal, by PacRunner's compile_metal
 * (runner_metal.c: its source and program; uirmetal.h: the entries). */
#ifndef RUNNER_METAL_H
#define RUNNER_METAL_H

#define RUNNER_PROCESS "p_p_to_j_j_j_101"
#define RUNNER_COLOUR "flow:3,5,1,2,4"
#define RUNNER_LEGS 5
#define RUNNER_OUTPUTS 32
#define RUNNER_FORMS 14
#define RUNNER_AMPLITUDES 24
#define RUNNER_PLANES 176

#include <stddef.h>
extern const char uir_metal_source[];
extern const size_t uir_metal_source_size;
extern const unsigned char uir_metal_program[];
extern const size_t uir_metal_program_size;

#include "uirmetal.h"

#endif
