# p_p_to_j_j_j_1 flow:3,1,4,5,2 for Metal, compiled by PacRunner's compile_metal (macOS).
#   make         librunner_metal.dylib: run_metal, pac_metal_run, pac_metal_values (uirmetal.h)
#   make clean
CC ?= cc
CFLAGS ?= -O2
PACFLAGS = -ffp-contract=off -fPIC
FRAMEWORKS = -framework Metal -framework Foundation
all: librunner_metal.dylib

librunner_metal.dylib: uirmetal.m uirmetal.h metalpool.m metalpool.h runner_metal.c runner_metal.h
	$(CC) $(CFLAGS) $(PACFLAGS) -fobjc-arc -dynamiclib -o $@ uirmetal.m metalpool.m runner_metal.c $(FRAMEWORKS)

clean:
	rm -f librunner_metal.dylib

.PHONY: all clean
