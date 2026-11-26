// ZEROKULL (zerokull1) | Author Signature: 0x5A45524F4B554C4CULL
#ifndef ZK_GPU_RING_H
#define ZK_GPU_RING_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>

#if defined(__x86_64__)
#include <immintrin.h>
#define ZK_SFENCE() __asm__ __volatile__("sfence" ::: "memory")
#define ZK_PAUSE()  __builtin_ia32_pause()
#elif defined(__aarch64__)
#define ZK_SFENCE() __asm__ __volatile__("dmb ishst" ::: "memory")
#define ZK_PAUSE()  __asm__ __volatile__("yield" ::: "memory")
#else
#define ZK_SFENCE() __sync_synchronize()
#define ZK_PAUSE()  do {} while (0)
#endif

typedef enum {
    ZK_ACCEL_CPU_SWAR     = 0,
    ZK_ACCEL_GENERIC_GPU  = 1,
    ZK_ACCEL_DIRECT_RING  = 2
} ZkAccelMode;

typedef struct {
    ZkAccelMode mode;
    uint32_t vendor_id;
    uint32_t ndesc;
    uint32_t cur_put;
    uint32_t ring_mask;
    uint32_t soliton_delta_en;
    uint32_t _pad0;
    uint64_t *ring_ptr;
    const volatile uint32_t *get_ptr;
    volatile uint32_t *put_ptr;
    uint64_t soliton_lines_sent;
    uint64_t soliton_lines_skipped;
    __attribute__((aligned(64))) uint64_t desc[64];
    __attribute__((aligned(64))) uint64_t shadow_desc[64];
    char device_name[128];
} ZkGpuRingCtx;

static inline uint32_t zk_swar_batch_entropy_score(const uint8_t *buf, size_t len) {
    if (!buf || len == 0) return 0;
    uint64_t acc = 0x9E3779B97F4A7C15ULL;
    size_t i = 0;
    for (; i + 8 <= len; i += 8) {
        uint64_t w = 0;
        memcpy(&w, buf + i, 8);
        acc ^= w;
        acc *= 0xBF58476D1CE4E5B9ULL;
        acc ^= acc >> 31;
    }
    for (; i < len; i++) {
        acc ^= (uint64_t)buf[i] << ((i & 7u) * 8u);
        acc *= 0x94D049BB133111EBULL;
    }
    return (uint32_t)(acc ^ (acc >> 32));
}

static inline size_t zk_soliton_delta_stream_64b(
    void * __restrict__ dst64,
    const void * __restrict__ src64,
    void * __restrict__ shadow64,
    size_t len_bytes,
    int delta_filter
) {
    size_t lines = len_bytes >> 6;
    size_t sent = 0;
    uint8_t *d = (uint8_t *)dst64;
    const uint8_t *s = (const uint8_t *)src64;
    uint8_t *sh = (uint8_t *)shadow64;
#if defined(__x86_64__) && defined(__AVX2__)
    if (((uintptr_t)d & 63u) == 0 && ((uintptr_t)s & 63u) == 0 &&
        (!delta_filter || (sh && ((uintptr_t)sh & 63u) == 0))) {
        for (size_t l = 0; l < lines; l++) {
            const __m256i *sp = (const __m256i *)(s + (l << 6));
            __m256i *dp = (__m256i *)(d + (l << 6));
            __m256i s0 = _mm256_load_si256(sp);
            __m256i s1 = _mm256_load_si256(sp + 1);
            if (delta_filter && sh) {
                __m256i *pp = (__m256i *)(sh + (l << 6));
                __m256i d0 = _mm256_xor_si256(s0, _mm256_load_si256(pp));
                __m256i d1 = _mm256_xor_si256(s1, _mm256_load_si256(pp + 1));
                __m256i diff = _mm256_or_si256(d0, d1);
                if (_mm256_testz_si256(diff, diff)) continue;
                _mm256_store_si256(pp, s0);
                _mm256_store_si256(pp + 1, s1);
            }
            _mm256_stream_si256(dp, s0);
            _mm256_stream_si256(dp + 1, s1);
            sent++;
        }
        if (sent > 0) ZK_SFENCE();
        return sent;
    }
#endif
    for (size_t l = 0; l < lines; l++) {
        const uint64_t *sp = (const uint64_t *)(s + (l << 6));
        uint64_t *dp = (uint64_t *)(d + (l << 6));
        if (delta_filter && sh) {
            uint64_t *pp = (uint64_t *)(sh + (l << 6));
            uint64_t diff = 0;
            for (size_t k = 0; k < 8; k++) diff |= (sp[k] ^ pp[k]);
            if (diff == 0) continue;
            memcpy(pp, sp, 64);
        }
        memcpy(dp, sp, 64);
        sent++;
    }
    if (sent > 0) ZK_SFENCE();
    return sent;
}

static inline void zk_gpu_ring_init(ZkGpuRingCtx *ctx) {
    if (!ctx) return;
    memset(ctx, 0, sizeof(*ctx));
    ctx->mode = ZK_ACCEL_CPU_SWAR;
    ctx->soliton_delta_en = 1u;
    memcpy(ctx->device_name, "Universal-SWAR-CPU", 19);
}

static inline void zk_gpu_ring_enable_soliton_delta(ZkGpuRingCtx *ctx, uint32_t enable) {
    if (!ctx) return;
    ctx->soliton_delta_en = enable ? 1u : 0u;
    if (enable) memset(ctx->shadow_desc, 0, sizeof(ctx->shadow_desc));
}

static inline int zk_gpu_ring_submit_pipelined(ZkGpuRingCtx *ctx, uint32_t win_slots) {
    if (!ctx) return -1;
    if (ctx->mode == ZK_ACCEL_DIRECT_RING && ctx->ring_ptr && ctx->put_ptr && ctx->get_ptr && ctx->ndesc > 0) {
        uint32_t put = ctx->cur_put;
        uint32_t mask = ctx->ring_mask;
        uint32_t limit = (win_slots > 0 ? win_slots : 4u) * ctx->ndesc;
        uint64_t spin = 0;
        while (((put - *ctx->get_ptr) & mask) >= limit && spin++ < 50000000ULL) {
            ZK_PAUSE();
        }
        uint32_t idx = put & mask;
        uint32_t lines64 = ctx->ndesc >> 3;
        if (lines64 > 0 && (ctx->ndesc & 7u) == 0 && (idx + ctx->ndesc) <= (mask + 1u) &&
            (((uintptr_t)&ctx->ring_ptr[idx]) & 63u) == 0) {
            size_t sent = zk_soliton_delta_stream_64b(
                &ctx->ring_ptr[idx], ctx->desc, ctx->shadow_desc,
                (size_t)ctx->ndesc * sizeof(uint64_t), (int)ctx->soliton_delta_en);
            ctx->soliton_lines_sent += sent;
            ctx->soliton_lines_skipped += (lines64 - sent);
        } else {
            for (uint32_t d = 0; d < ctx->ndesc; d++) {
                ctx->ring_ptr[(put + d) & mask] = ctx->desc[d];
            }
            ZK_SFENCE();
        }
        put = (put + ctx->ndesc) & mask;
        ctx->cur_put = put;
        *ctx->put_ptr = put;
        return 0;
    }
    return 0;
}

static inline int zk_gpu_ring_sync_drain(ZkGpuRingCtx *ctx) {
    if (!ctx) return -1;
    if (ctx->mode == ZK_ACCEL_DIRECT_RING && ctx->get_ptr) {
        ZK_SFENCE();
        uint64_t spin = 0;
        while (*ctx->get_ptr != ctx->cur_put && spin++ < 50000000ULL) {
            ZK_PAUSE();
        }
        return (*ctx->get_ptr == ctx->cur_put) ? 0 : -2;
    }
    return 0;
}

static inline int zk_gpu_ring_submit_batch(ZkGpuRingCtx *ctx) {
    int rc = zk_gpu_ring_submit_pipelined(ctx, 1u);
    return (rc == 0) ? zk_gpu_ring_sync_drain(ctx) : rc;
}

#endif
