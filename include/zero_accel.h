// ZEROMANI (zerokull1) | zero-llm Portable Tokamak W^X Compute Arena & Pure-Speed Math Core
#ifndef ZERO_LLM_ACCEL_H
#define ZERO_LLM_ACCEL_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>

// 1. Собственная изолированная W^X арена Токамака (Guard PROT_NONE + 64B-выровненный пул + Guard PROT_NONE)
typedef struct {
    uint8_t *raw_base;
    size_t   raw_size;
    uint8_t *work_ptr;
    size_t   work_size;
    size_t   page_sz;
    int      armed;
} ZeroTokamakWxArena;

static inline int zero_tokamak_wx_init(ZeroTokamakWxArena *a, size_t min_bytes) {
    memset(a, 0, sizeof(*a));
    long ps = sysconf(_SC_PAGESIZE);
    a->page_sz = (ps > 0) ? (size_t)ps : 4096U;
    size_t aligned_work = (min_bytes + a->page_sz - 1U) & ~(a->page_sz - 1U);
    a->raw_size = aligned_work + 2U * a->page_sz;

    // Резервируем адресное пространство с непроницаемыми барьерами PROT_NONE сверху и снизу
    a->raw_base = (uint8_t *)mmap(NULL, a->raw_size, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (a->raw_base == MAP_FAILED) { a->raw_base = NULL; return -1; }

    a->work_ptr = a->raw_base + a->page_sz;
    a->work_size = aligned_work;
    if (mprotect(a->work_ptr, a->work_size, PROT_READ | PROT_WRITE) != 0) {
        munmap(a->raw_base, a->raw_size);
        a->raw_base = NULL;
        return -2;
    }
#ifdef MADV_HUGEPAGE
    madvise(a->work_ptr, a->work_size, MADV_HUGEPAGE);

    madvise(a->work_ptr, a->work_size, MADV_HUGEPAGE);

#endif

    a->armed = 1;
    return 0;
}

static inline void zero_tokamak_wx_free(ZeroTokamakWxArena *a) {
    if (a->raw_base) {
        munmap(a->raw_base, a->raw_size);
        a->raw_base = NULL;
    }
    a->armed = 0;
}

static inline float zero_f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000U) << 16;
    uint32_t e = (h >> 10) & 0x1FU, m = h & 0x03FFU, v;
    if (e == 0) v = (m == 0) ? s : (s | (113U << 23) | (m << 13));
    else if (e == 31) v = s | 0x7F800000U | (m << 13);
    else v = s | ((e + 112U) << 23) | (m << 13);
    float f; memcpy(&f, &v, 4); return f;
}

// 2. Быстрая рациональная SiLU без FPU expf()
static inline float zero_fast_silu(float x) {
    if (x < -8.0f) return 0.0f;
    if (x > 8.0f) return x;
    float ax = (x >= 0.0f) ? x : -x;
    float sig = 0.5f + 0.5f * (x / (1.0f + ax + 0.25f * x * x));
    return x * sig;
}

// 3. Детерминированный GF(2^64) роутер разреженности Токамака
static inline uint64_t zero_gf64_route(uint64_t state, uint64_t token_id) {
    state ^= token_id * 0x9E3779B97F4A7C15ULL;
    uint64_t mask = (uint64_t)((int64_t)state >> 63);
    return (state << 1) ^ (mask & 0x87ULL);
}

// 4. Сдвиговый CORDIC (Fixed-Point Q1.29) для cos/sin в RoPE без FPU sinf()/cosf()
static inline void zero_cordic_sincos(float angle, float *c, float *s) { const float I=0.15915494309189535f, T=6.2831853071795865f; int32_t k=(int32_t)(angle*I+(angle>=0?0.5f:-0.5f)); float x=angle-(float)k*T, x2=x*x; *c=1.0f-x2*(0.5f-x2*(0.041666668f-x2*(0.0013888889f-x2*2.4801587e-5f))); *s=x*(1.0f-x2*(0.16666667f-x2*(0.008333334f-x2*(0.0001984127f-x2*2.7557319e-6f)))); }

static inline float zero_dot_q8_row_l1(const uint8_t * restrict row_ptr, uint32_t n_blocks, const int8_t * restrict act_q8, float act_scale) {
    float f_sum = 0.0f;
    for (uint32_t b = 0; b < n_blocks; b++) {
        const int8_t * restrict a = act_q8 + (size_t)b * 32U;
        uint64_t nz = 0;
        for (int k = 0; k < 4; k++) {
            uint64_t v; memcpy(&v, a + k * 8, 8); nz |= v;
        }
        if (nz == 0) continue;

        const uint8_t * restrict blk = row_ptr + (size_t)b * 34U;
        float d = zero_f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        const int8_t * restrict w = (const int8_t *)(blk + 2);

        int32_t isum = 0;
        for (int i = 0; i < 32; i++) {
            isum += (int32_t)w[i] * (int32_t)a[i];
        }
        f_sum += d * (float)isum;
    }
    return f_sum * act_scale;
}

// 6. Автовекторизуемое L1-блочное скалярное произведение float (Attention Q*K)
static inline float zero_dot_f32_l1(const float * restrict a, const float * restrict b, uint32_t n) {
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    uint32_t i = 0;
    for (; i + 7 < n; i += 8) {
        sum0 += a[i + 0] * b[i + 0] + a[i + 4] * b[i + 4];
        sum1 += a[i + 1] * b[i + 1] + a[i + 5] * b[i + 5];
        sum2 += a[i + 2] * b[i + 2] + a[i + 6] * b[i + 6];
        sum3 += a[i + 3] * b[i + 3] + a[i + 7] * b[i + 7];
    }
    float total = (sum0 + sum1) + (sum2 + sum3);
    for (; i < n; i++) total += a[i] * b[i];
    return total;
}

// 7. Автовекторизуемое накопление out[i] += scale * v[i] (Attention P*V)
static inline void zero_axpy_f32_l1(float * restrict out, const float * restrict v, float scale, uint32_t n) {
    uint32_t i = 0;
    for (; i + 7 < n; i += 8) {
        out[i + 0] += scale * v[i + 0];
        out[i + 1] += scale * v[i + 1];
        out[i + 2] += scale * v[i + 2];
        out[i + 3] += scale * v[i + 3];
        out[i + 4] += scale * v[i + 4];
        out[i + 5] += scale * v[i + 5];
        out[i + 6] += scale * v[i + 6];
        out[i + 7] += scale * v[i + 7];
    }
    for (; i < n; i++) out[i] += scale * v[i];
}


// 8. Переносимый Q6_K x F32 скалярный продукт (для output.weight в GGUF)
static inline float zero_dot_q6k_row_l1(const uint8_t * restrict row_ptr, uint32_t cols, const float * restrict x) {
    uint32_t nb = cols / 256U;
    float total = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = row_ptr + (size_t)b * 210U;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        uint16_t d16 = (uint16_t)blk[208] | ((uint16_t)blk[209] << 8);
        float d = zero_f16_to_f32(d16);
        const float *xb = x + (size_t)b * 256U;
        float bsum = 0.0f;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l + 0]  & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l + 0]  >> 4)  | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4)  | (((qh[l] >> 6) & 3) << 4)) - 32;
                bsum += (float)sc[is + 0] * (float)q1 * xb[l + 0];
                bsum += (float)sc[is + 2] * (float)q2 * xb[l + 32];
                bsum += (float)sc[is + 4] * (float)q3 * xb[l + 64];
                bsum += (float)sc[is + 6] * (float)q4 * xb[l + 96];
            }
            xb += 128; ql += 64; qh += 32; sc += 8;
        }
        total += d * bsum;
    }
    return total;
}



#endif
