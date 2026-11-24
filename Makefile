CC = gcc
CFLAGS_COLD = -Os -march=native -fopenmp -fno-plt -fno-asynchronous-unwind-tables -fomit-frame-pointer -Iinclude -Wall -Wextra
CFLAGS_HOT  = -O3 -march=native -fopenmp -fno-plt -fno-asynchronous-unwind-tables -fomit-frame-pointer -Iinclude -Wall -Wextra
LDFLAGS = -lm -ldl -Wl,--gc-sections -s

all: zero-llm

zero-llm: src/zero_llm.c src/hot_core.c include/zero_engine.h include/zero_tokamak_hot.h
	$(CC) $(CFLAGS_COLD) -c src/zero_llm.c -o /tmp/zk_cold.o
	$(CC) $(CFLAGS_HOT)  -c src/hot_core.c -o /tmp/zk_hot.o
	$(CC) -fopenmp /tmp/zk_cold.o /tmp/zk_hot.o $(LDFLAGS) -o zero-llm
	rm -f /tmp/zk_cold.o /tmp/zk_hot.o

clean:
	rm -f zero-llm
