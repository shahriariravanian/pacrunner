# p_p_to_j_j_j_1 flow:3,1,4,5,2, compiled by PacRunner's compile.
#   make         runner.o and pacops.o, and testing
#   make test    runs testing: the points of examples.json against RustiCol's outputs
#   make gpu     runner_gpu.o (nvcc): run_gpu, every point on the GPU (uirvm.h)
#   make clean
# CFLAGS may be changed; PACFLAGS may not: the values are bit for bit the Julia runner's only
# without contracted multiply-adds (-ffp-contract=off).
CC ?= cc
CFLAGS ?= -O2
PACFLAGS = -std=c99 -ffp-contract=off
all: runner.o pacops.o testing

runner.o: runner.c runner.h pacops.h uirvm.c uirvm.h uirgpu.c uirgpu.h uirgpu_pool.c uirgpu_umami.c uirgpu_provider.c pool.cu pool.h
	$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ runner.c

pacops.o: pacops.c pacops.h
	$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ pacops.c

# the same runner.c on the GPU; NVFLAGS may be changed, NVPACFLAGS may not (bit for bit the CPU's
# only without contracted multiply-adds)
NVCC ?= nvcc
NVFLAGS ?= -O3 -arch=native
NVPACFLAGS = --fmad=false
gpu: runner_gpu.o

runner_gpu.o: runner.c runner.h pacops.c pacops.h uirvm.c uirvm.h uirgpu.c uirgpu.h uirgpu_pool.c uirgpu_umami.c uirgpu_provider.c pool.cu pool.h
	$(NVCC) $(NVFLAGS) $(NVPACFLAGS) -x cu -c -o $@ runner.c

testing: testing.c runner.h runner.o pacops.o
	$(CC) $(CFLAGS) $(PACFLAGS) -o $@ testing.c runner.o pacops.o -lm

test: testing
	./testing

clean:
	rm -f runner.o pacops.o testing runner_gpu.o

.PHONY: all test gpu clean
