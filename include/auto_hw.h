// ZEROMANI (zerokull1) | zero-llm Hardware-Adaptive Token & Memory Planner
#ifndef ZERO_LLM_AUTO_HW_H
#define ZERO_LLM_AUTO_HW_H

#include <stdint.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    uint64_t avail_ram_bytes;
    uint32_t max_seq;
    uint32_t max_gen;
    uint32_t bytes_per_tok;
    size_t   kv_bytes;
} AutoHwPlan;

static inline uint64_t hw_get_avail_ram(void) {
    FILE *f = fopen("/proc/meminfo", "r");
    if (!f) return 2ULL * 1024ULL * 1024ULL * 1024ULL;
    char key[64], unit[16];
    uint64_t val = 0, avail_kb = 0;
    while (fscanf(f, "%63s %llu %15s", key, (unsigned long long *)&val, unit) == 3) {
        if (strcmp(key, "MemAvailable:") == 0) { avail_kb = val; break; }
    }
    fclose(f);
    return avail_kb * 1024ULL;
}

static inline AutoHwPlan hw_calc_token_plan(
    uint32_t n_layers, uint32_t n_kv_heads, uint32_t head_dim, uint32_t model_ctx_cap
) {
    AutoHwPlan p = {0};
    p.avail_ram_bytes = hw_get_avail_ram();
    p.bytes_per_tok = 2U * n_layers * n_kv_heads * head_dim * 4U;

    uint64_t os_reserve = 2ULL * 1024ULL * 1024ULL * 1024ULL;
    uint64_t usable = (p.avail_ram_bytes > os_reserve) ? (p.avail_ram_bytes - os_reserve) : (256ULL * 1024ULL * 1024ULL);
    uint64_t kv_budget = usable / 4ULL;

    uint64_t hw_toks = kv_budget / (uint64_t)p.bytes_per_tok;
    if (model_ctx_cap == 0) model_ctx_cap = 32768U;
    uint32_t seq = (hw_toks > model_ctx_cap) ? model_ctx_cap : (uint32_t)hw_toks;
    seq = (seq / 512U) * 512U;
    if (seq < 512U) seq = 512U;

    p.max_seq = seq;
    p.max_gen = (seq / 8U > 1024U) ? 1024U : (seq / 8U);
    if (p.max_gen < 128U) p.max_gen = 128U;
    p.kv_bytes = (size_t)seq * (size_t)p.bytes_per_tok;
    return p;
}

#endif
