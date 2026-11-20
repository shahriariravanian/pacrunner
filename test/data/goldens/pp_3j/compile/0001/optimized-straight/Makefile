# p_p_to_j_j_j_1 flow:3,5,1,4,2, compiled by PacRunner's compile.
#   make         runner.o and pacops.o, and testing
#   make test    runs testing: the points of examples.json against RustiCol's outputs
#   make clean
# CFLAGS may be changed; PACFLAGS may not: the values are bit for bit the Julia runner's only
# without contracted multiply-adds (-ffp-contract=off).
CC ?= cc
CFLAGS ?= -O2
PACFLAGS = -std=c99 -ffp-contract=off
all: runner.o pacops.o testing

runner.o: runner.c runner.h pacops.h
	$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ runner.c

pacops.o: pacops.c pacops.h
	$(CC) $(CFLAGS) $(PACFLAGS) -c -o $@ pacops.c

testing: testing.c runner.h runner.o pacops.o
	$(CC) $(CFLAGS) $(PACFLAGS) -o $@ testing.c runner.o pacops.o -lm

test: testing
	./testing

clean:
	rm -f runner.o pacops.o testing

.PHONY: all test clean
