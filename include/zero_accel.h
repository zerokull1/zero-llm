// ZEROMANI (zerokull1) | zero-llm Pure-Speed Tensor Accelerator
#ifndef ZERO_LLM_ACCEL_H
#define ZERO_LLM_ACCEL_H

#include <stdint.h>

static inline float zero_fast_silu(float x) {
    if (x < -8.0f) return 0.0f;
    if (x > 8.0f) return x;
    float ax = (x >= 0.0f) ? x : -x;
    float sig = 0.5f + 0.5f * (x / (1.0f + ax + 0.25f * x * x));
    return x * sig;
}

static inline uint64_t zero_gf64_route(uint64_t state, uint64_t token_id) {
    state ^= token_id * 0x9E3779B97F4A7C15ULL;
    uint64_t mask = (uint64_t)((int64_t)state >> 63);
    return (state << 1) ^ (mask & 0x87ULL);
}

#endif
