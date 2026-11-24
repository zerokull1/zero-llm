// ZEROMANI (zerokull1) | zero-llm L1i-Compact Tokamak 4-Way L1d-Orbit Kernels
#ifndef ZERO_TOKAMAK_HOT_H
#define ZERO_TOKAMAK_HOT_H
#include <math.h>
#include <string.h>
#include "zero_engine.h"

__attribute__((noinline))
static float dot_q8_row_hot(const uint8_t * restrict rp, uint32_t nb, const int8_t * restrict aq, float asc) {
    return zero_dot_q8_row_l1(rp, nb, aq, asc);
}

__attribute__((noinline))
static void matvec_tokamak_hot(float *out, const uint8_t *wb, TensorRef tr, const int8_t *aq, float asc, const float *bias) {
    uint32_t nb = tr.cols / 32U; size_t rb = (size_t)nb * 34U; const uint8_t *mat = wb + tr.off;
    uint32_t r4 = tr.rows & ~3U;
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < r4; r += 4) {
        const uint8_t *p0 = mat + (size_t)(r + 0) * rb, *p1 = p0 + rb, *p2 = p1 + rb, *p3 = p2 + rb;
        float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
        for (uint32_t b = 0; b < nb; b++) {
            const int8_t *a = aq + (size_t)b * 32U;
            uint64_t nz = 0;
            for (int k = 0; k < 4; k++) { uint64_t v; memcpy(&v, a + k * 8, 8); nz |= v; }
            if (!nz) continue;
            const uint8_t *b0 = p0 + (size_t)b * 34U, *b1 = p1 + (size_t)b * 34U, *b2 = p2 + (size_t)b * 34U, *b3 = p3 + (size_t)b * 34U;
            float d0 = zero_f16_to_f32((uint16_t)b0[0] | ((uint16_t)b0[1] << 8));
            float d1 = zero_f16_to_f32((uint16_t)b1[0] | ((uint16_t)b1[1] << 8));
            float d2 = zero_f16_to_f32((uint16_t)b2[0] | ((uint16_t)b2[1] << 8));
            float d3 = zero_f16_to_f32((uint16_t)b3[0] | ((uint16_t)b3[1] << 8));
            const int8_t *w0 = (const int8_t*)(b0 + 2), *w1 = (const int8_t*)(b1 + 2), *w2 = (const int8_t*)(b2 + 2), *w3 = (const int8_t*)(b3 + 2);
            int32_t i0 = 0, i1 = 0, i2 = 0, i3 = 0;
            for (int i = 0; i < 32; i++) {
                int32_t av = (int32_t)a[i];
                i0 += (int32_t)w0[i] * av; i1 += (int32_t)w1[i] * av; i2 += (int32_t)w2[i] * av; i3 += (int32_t)w3[i] * av;
            }
            s0 += d0 * (float)i0; s1 += d1 * (float)i1; s2 += d2 * (float)i2; s3 += d3 * (float)i3;
        }
        out[r + 0] = s0 * asc + (bias ? bias[r + 0] : 0.0f); out[r + 1] = s1 * asc + (bias ? bias[r + 1] : 0.0f);
        out[r + 2] = s2 * asc + (bias ? bias[r + 2] : 0.0f); out[r + 3] = s3 * asc + (bias ? bias[r + 3] : 0.0f);
    }
    for (uint32_t r = r4; r < tr.rows; r++) { float v = dot_q8_row_hot(mat + (size_t)r * rb, nb, aq, asc); out[r] = bias ? (v + bias[r]) : v; }
}
#endif
