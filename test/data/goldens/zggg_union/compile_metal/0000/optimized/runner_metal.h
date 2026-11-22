/* zggg_union flow:2,4,5,6,1: the program for Metal, by PacRunner's compile_metal
 * (runner_metal.c: its source and program; uirmetal.h: the entries). */
#ifndef RUNNER_METAL_H
#define RUNNER_METAL_H

#define RUNNER_PROCESS "zggg_union"
#define RUNNER_COLOUR "flow:2,4,5,6,1"
#define RUNNER_LEGS 6
#define RUNNER_OUTPUTS 96
#define RUNNER_FORMS 25
#define RUNNER_AMPLITUDES 6
#define RUNNER_PLANES 166

#include <stddef.h>
extern const char uir_metal_source[];
extern const size_t uir_metal_source_size;
extern const unsigned char uir_metal_program[];
extern const size_t uir_metal_program_size;

#include "uirmetal.h"

#endif
