/* p_p_to_j_j_j_1 flow:3,5,1,4,2: the program compiled by PacRunner's compile (runner.c).
 * run(out, p, momenta) evaluates one phase-space point: momenta[4 * RUNNER_LEGS] are the caller's
 * legs (E, px, py, pz each, before the crossing), out[RUNNER_OUTPUTS] receives the helicity
 * outputs (group factor applied; the matrix element is their sum), p[RUNNER_ARENA_SIZE] is
 * scratch. Returns 0. */
#ifndef RUNNER_H
#define RUNNER_H

#define RUNNER_PROCESS "p_p_to_j_j_j_1"
#define RUNNER_COLOUR "flow:3,5,1,4,2"
#define RUNNER_LEGS 5
#define RUNNER_OUTPUTS 32
#define RUNNER_ARENA_SIZE 456

#ifdef __cplusplus
extern "C" {
#endif

int run(double *out, double *p, const double *momenta);

#ifdef __cplusplus
}
#endif

#endif
