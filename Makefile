CC = gcc
CFLAGS = -O3 -march=native -fopenmp -Iinclude -Wall -Wextra
LDFLAGS = -lm -ldl

all: zero-llm

zero-llm: src/zero_llm.c include/auto_hw.h include/zero_accel.h include/zero_gpu_ptx.h
	$(CC) $(CFLAGS) src/zero_llm.c $(LDFLAGS) -o zero-llm

clean:
	rm -f zero-llm
