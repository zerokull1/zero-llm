// ZEROMANI (zerokull1) | zero-llm L1i-Compact Tokamak Hot Core (Universal C11 + AVX2/NEON)
#ifndef ZERO_TOKAMAK_HOT_H
#define ZERO_TOKAMAK_HOT_H
#include <math.h>
#include <string.h>
#include "zero_engine.h"
#if defined(__AVX2__)
#include <immintrin.h>
#endif

__attribute__((noinline))
static float dot_q8_row_hot(const uint8_t * restrict rp, uint32_t nb, const int8_t * restrict aq, float asc) {
    float f_sum = 0.0f; uint32_t b = 0;
#if defined(__AVX2__)
    const __m256i ones16 = _mm256_set1_epi16(1);
    for (; b + 3 < nb; b += 4) {
        __m256i a0 = _mm256_loadu_si256((const __m256i*)(aq + (b+0)*32U)), a1 = _mm256_loadu_si256((const __m256i*)(aq + (b+1)*32U));
        __m256i a2 = _mm256_loadu_si256((const __m256i*)(aq + (b+2)*32U)), a3 = _mm256_loadu_si256((const __m256i*)(aq + (b+3)*32U));
        __m256i any = _mm256_or_si256(_mm256_or_si256(a0,a1), _mm256_or_si256(a2,a3));
        if (_mm256_testz_si256(any, any)) continue;
        const uint8_t *b0 = rp + (size_t)(b+0)*34U, *b1 = rp + (size_t)(b+1)*34U, *b2 = rp + (size_t)(b+2)*34U, *b3 = rp + (size_t)(b+3)*34U;
        __m256i w0 = _mm256_loadu_si256((const __m256i*)(b0+2)), w1 = _mm256_loadu_si256((const __m256i*)(b1+2));
        __m256i w2 = _mm256_loadu_si256((const __m256i*)(b2+2)), w3 = _mm256_loadu_si256((const __m256i*)(b3+2));
        __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(a0,w0)), ones16);
        __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(a1,w1)), ones16);
        __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w2), _mm256_sign_epi8(a2,w2)), ones16);
        __m256i p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w3), _mm256_sign_epi8(a3,w3)), ones16);
        __m128i s0 = _mm_add_epi32(_mm256_castsi256_si128(p0), _mm256_extracti128_si256(p0,1)), s1 = _mm_add_epi32(_mm256_castsi256_si128(p1), _mm256_extracti128_si256(p1,1));
        __m128i s2 = _mm_add_epi32(_mm256_castsi256_si128(p2), _mm256_extracti128_si256(p2,1)), s3 = _mm_add_epi32(_mm256_castsi256_si128(p3), _mm256_extracti128_si256(p3,1));
        __m128 pr = _mm_mul_ps(_mm_cvtepi32_ps(_mm_hadd_epi32(_mm_hadd_epi32(s0,s1), _mm_hadd_epi32(s2,s3))),
            _mm_set_ps(zero_f16_to_f32(*(const uint16_t*)b3), zero_f16_to_f32(*(const uint16_t*)b2), zero_f16_to_f32(*(const uint16_t*)b1), zero_f16_to_f32(*(const uint16_t*)b0)));
        pr = _mm_hadd_ps(pr, pr); pr = _mm_hadd_ps(pr, pr); f_sum += _mm_cvtss_f32(pr);
    }
#endif
    for (; b < nb; b++) {
        const int8_t * restrict a = aq + (size_t)b * 32U; const uint8_t * restrict blk = rp + (size_t)b * 34U;
        float d = zero_f16_to_f32(*(const uint16_t*)blk); const int8_t * restrict w = (const int8_t*)(blk + 2);
        int32_t isum = 0; for (int i = 0; i < 32; i++) isum += (int32_t)w[i] * (int32_t)a[i];
        f_sum += d * (float)isum;
    }
    return f_sum * asc;
}

__attribute__((noinline))
static void matvec_tokamak_hot(float *out, const uint8_t *wb, TensorRef tr, const int8_t *aq, float asc, const float *bias) {
    uint32_t nb = tr.cols / 32U; size_t rb = (size_t)nb * 34U; const uint8_t *mat = wb + tr.off;
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < tr.rows; r++) { float v = dot_q8_row_hot(mat + (size_t)r * rb, nb, aq, asc); out[r] = bias ? (v + bias[r]) : v; }
}
#endif
