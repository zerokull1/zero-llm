// ZEROMANI (zerokull1) | Author Signature: 0x5A45524F4D414E49ULL
#ifndef ZERO_GPU_RING_FASTPATH_H
#define ZERO_GPU_RING_FASTPATH_H

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <dlfcn.h>

#if defined(__x86_64__)
#define ZLLM_SFENCE() __asm__ __volatile__("sfence" ::: "memory")
#define ZLLM_PAUSE()  __builtin_ia32_pause()
#elif defined(__aarch64__)
#define ZLLM_SFENCE() __asm__ __volatile__("dmb ishst" ::: "memory")
#define ZLLM_PAUSE()  __asm__ __volatile__("yield" ::: "memory")
#else
#define ZLLM_SFENCE() __sync_synchronize()
#define ZLLM_PAUSE()  do {} while (0)
#endif

typedef enum {
    ZLLM_PATH_CPU_ADAPTIVE = 0,
    ZLLM_PATH_GENERIC_GPU  = 1,
    ZLLM_PATH_DIRECT_RING  = 2
} ZeroLlmAccelPath;

typedef struct {
    ZeroLlmAccelPath path;
    uint32_t ndesc;
    uint32_t cur_put;
    uint32_t ring_mask;
    uint64_t *ring_ptr;
    const volatile uint32_t *get_ptr;
    volatile uint32_t *put_ptr;
    uint64_t desc[64];
} ZeroLlmRingFastpath;

static inline void zero_llm_ring_probe(ZeroLlmRingFastpath *fp) {
    if (!fp) return;
    memset(fp, 0, sizeof(*fp));
    fp->path = ZLLM_PATH_CPU_ADAPTIVE;
    void *vk = dlopen("libvulkan.so.1", RTLD_LAZY | RTLD_LOCAL);
    if (vk) dlclose(vk);
}

static inline int zero_llm_ring_dispatch(ZeroLlmRingFastpath *fp) {
    if (!fp) return -1;
    if (fp->path == ZLLM_PATH_DIRECT_RING && fp->ring_ptr && fp->put_ptr && fp->get_ptr && fp->ndesc > 0) {
        uint32_t put = fp->cur_put;
        uint32_t mask = fp->ring_mask;
        for (uint32_t d = 0; d < fp->ndesc; d++) {
            fp->ring_ptr[(put + d) & mask] = fp->desc[d];
        }
        put = (put + fp->ndesc) & mask;
        fp->cur_put = put;
        ZLLM_SFENCE();
        *fp->put_ptr = put;
        ZLLM_SFENCE();
        uint64_t spin = 0;
        while (*fp->get_ptr != put && spin++ < 50000000ULL) {
            ZLLM_PAUSE();
        }
        return (*fp->get_ptr == put) ? 0 : -2;
    }
    return 0;
}

#endif
