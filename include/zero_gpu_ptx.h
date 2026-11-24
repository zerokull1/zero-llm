// ZEROMANI (zerokull1) | zero-llm Universal Auto-VRAM PTX Backend
#ifndef ZERO_LLM_GPU_PTX_H
#define ZERO_LLM_GPU_PTX_H

#include <stdint.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dlfcn.h>

typedef int CUresult;
typedef int CUdevice;
typedef void* CUcontext;
typedef void* CUmodule;
typedef void* CUfunction;
typedef void* CUstream;
typedef uint64_t CUdeviceptr;

typedef struct {
    void *lib;
    int active;
    char gpu_name[128];
    int gpu_layers;
    size_t free_vram_start;
    size_t vram_used;
    CUcontext ctx;
    CUmodule mod;
    CUfunction fn_q8;
    CUstream stream;
    CUdeviceptr d_weights;
    CUdeviceptr d_act;
    CUdeviceptr d_out;
    int8_t *h_act_pinned;
    float *h_out_pinned;

    CUresult (*cuInit)(unsigned int);
    CUresult (*cuDeviceGet)(CUdevice*, int);
    CUresult (*cuDeviceGetName)(char*, int, CUdevice);
    CUresult (*cuCtxCreate)(CUcontext*, unsigned int, CUdevice);
    CUresult (*cuCtxDestroy)(CUcontext);
    CUresult (*cuMemGetInfo)(size_t*, size_t*);
    CUresult (*cuModuleLoadData)(CUmodule*, const void*);
    CUresult (*cuModuleGetFunction)(CUfunction*, CUmodule, const char*);
    CUresult (*cuMemAlloc)(CUdeviceptr*, size_t);
    CUresult (*cuMemFree)(CUdeviceptr);
    CUresult (*cuMemHostAlloc)(void**, size_t, unsigned int);
    CUresult (*cuMemFreeHost)(void*);
    CUresult (*cuMemcpyHtoD)(CUdeviceptr, const void*, size_t);
    CUresult (*cuMemcpyHtoDAsync)(CUdeviceptr, const void*, size_t, CUstream);
    CUresult (*cuMemcpyDtoHAsync)(void*, CUdeviceptr, size_t, CUstream);
    CUresult (*cuLaunchKernel)(CUfunction, unsigned int, unsigned int, unsigned int,
                               unsigned int, unsigned int, unsigned int,
                               unsigned int, CUstream, void**, void**);
    CUresult (*cuStreamCreate)(CUstream*, unsigned int);
    CUresult (*cuStreamSynchronize)(CUstream);
    CUresult (*cuStreamDestroy)(CUstream);
} ZeroGpuBackend;

static const char *ZERO_Q8_PTX =
".version 6.0\n"
".target sm_61\n"
".address_size 64\n"
".visible .entry q8_matvec_warp(\n"
"    .param .u64 p_mat,\n"
"    .param .u64 p_act,\n"
"    .param .u64 p_out,\n"
"    .param .u32 p_rows,\n"
"    .param .u32 p_nb,\n"
"    .param .f32 p_scale\n"
") {\n"
"    .reg .pred %p<4>;\n"
"    .reg .b16 %h<4>;\n"
"    .reg .b32 %r<24>;\n"
"    .reg .b64 %rd<16>;\n"
"    .reg .f32 %f<12>;\n"
"    ld.param.u64 %rd1, [p_mat];\n"
"    ld.param.u64 %rd2, [p_act];\n"
"    ld.param.u64 %rd3, [p_out];\n"
"    ld.param.u32 %r1, [p_rows];\n"
"    ld.param.u32 %r2, [p_nb];\n"
"    ld.param.f32 %f1, [p_scale];\n"
"    mov.u32 %r3, %ctaid.x;\n"
"    mov.u32 %r4, %tid.x;\n"
"    shr.u32 %r5, %r4, 5;\n"
"    and.b32 %r6, %r4, 31;\n"
"    mad.lo.u32 %r7, %r3, 4, %r5;\n"
"    setp.ge.u32 %p1, %r7, %r1;\n"
"    @%p1 bra DONE;\n"
"    mul.lo.u32 %r8, %r2, 34;\n"
"    mul.wide.u32 %rd4, %r7, %r8;\n"
"    add.u64 %rd5, %rd1, %rd4;\n"
"    mov.f32 %f2, 0f00000000;\n"
"    mov.u32 %r9, %r6;\n"
"LOOP:\n"
"    setp.ge.u32 %p2, %r9, %r2;\n"
"    @%p2 bra REDUCE;\n"
"    mul.wide.u32 %rd6, %r9, 34;\n"
"    add.u64 %rd7, %rd5, %rd6;\n"
"    ld.global.u16 %h1, [%rd7];\n"
"    cvt.f32.f16 %f3, %h1;\n"
"    mul.wide.u32 %rd8, %r9, 32;\n"
"    add.u64 %rd9, %rd2, %rd8;\n"
"    mov.u32 %r10, 0;\n"
"    ld.global.u16 %h2, [%rd7+2];  ld.global.u16 %h3, [%rd7+4];  mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9];    dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    ld.global.u16 %h2, [%rd7+6];  ld.global.u16 %h3, [%rd7+8];  mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9+4];  dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    ld.global.u16 %h2, [%rd7+10]; ld.global.u16 %h3, [%rd7+12]; mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9+8];  dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    ld.global.u16 %h2, [%rd7+14]; ld.global.u16 %h3, [%rd7+16]; mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9+12]; dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    ld.global.u16 %h2, [%rd7+18]; ld.global.u16 %h3, [%rd7+20]; mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9+16]; dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    ld.global.u16 %h2, [%rd7+22]; ld.global.u16 %h3, [%rd7+24]; mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9+20]; dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    ld.global.u16 %h2, [%rd7+26]; ld.global.u16 %h3, [%rd7+28]; mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9+24]; dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    ld.global.u16 %h2, [%rd7+30]; ld.global.u16 %h3, [%rd7+32]; mov.b32 %r11, {%h2, %h3}; ld.global.u32 %r12, [%rd9+28]; dp4a.s32.s32 %r10, %r11, %r12, %r10;\n"
"    cvt.rn.f32.s32 %f4, %r10;\n"
"    fma.rn.f32 %f2, %f3, %f4, %f2;\n"
"    add.u32 %r9, %r9, 32;\n"
"    bra LOOP;\n"
"REDUCE:\n"
"    shfl.sync.bfly.b32 %f5, %f2, 16, 31, 0xffffffff; add.f32 %f2, %f2, %f5;\n"
"    shfl.sync.bfly.b32 %f5, %f2, 8,  31, 0xffffffff; add.f32 %f2, %f2, %f5;\n"
"    shfl.sync.bfly.b32 %f5, %f2, 4,  31, 0xffffffff; add.f32 %f2, %f2, %f5;\n"
"    shfl.sync.bfly.b32 %f5, %f2, 2,  31, 0xffffffff; add.f32 %f2, %f2, %f5;\n"
"    shfl.sync.bfly.b32 %f5, %f2, 1,  31, 0xffffffff; add.f32 %f2, %f2, %f5;\n"
"    setp.eq.u32 %p3, %r6, 0;\n"
"    @!%p3 bra DONE;\n"
"    mul.f32 %f2, %f2, %f1;\n"
"    mul.wide.u32 %rd10, %r7, 4;\n"
"    add.u64 %rd11, %rd3, %rd10;\n"
"    st.global.f32 [%rd11], %f2;\n"
"DONE:\n"
"    ret;\n"
"}\n";

static inline int zero_gpu_auto_init(ZeroGpuBackend *g, int total_layers, size_t bytes_per_layer) {
    memset(g, 0, sizeof(*g));
    g->lib = dlopen("libcuda.so.1", RTLD_LAZY);
    if (!g->lib) return -1;

    g->cuInit = dlsym(g->lib, "cuInit");
    g->cuDeviceGet = dlsym(g->lib, "cuDeviceGet");
    g->cuDeviceGetName = dlsym(g->lib, "cuDeviceGetName");
    g->cuCtxCreate = dlsym(g->lib, "cuCtxCreate_v2");
    g->cuCtxDestroy = dlsym(g->lib, "cuCtxDestroy_v2");
    g->cuMemGetInfo = dlsym(g->lib, "cuMemGetInfo_v2");
    g->cuModuleLoadData = dlsym(g->lib, "cuModuleLoadData");
    g->cuModuleGetFunction = dlsym(g->lib, "cuModuleGetFunction");
    g->cuMemAlloc = dlsym(g->lib, "cuMemAlloc_v2");
    g->cuMemFree = dlsym(g->lib, "cuMemFree_v2");
    g->cuMemHostAlloc = dlsym(g->lib, "cuMemHostAlloc");
    g->cuMemFreeHost = dlsym(g->lib, "cuMemFreeHost");
    g->cuMemcpyHtoD = dlsym(g->lib, "cuMemcpyHtoD_v2");
    g->cuMemcpyHtoDAsync = dlsym(g->lib, "cuMemcpyHtoDAsync_v2");
    g->cuMemcpyDtoHAsync = dlsym(g->lib, "cuMemcpyDtoHAsync_v2");
    g->cuLaunchKernel = dlsym(g->lib, "cuLaunchKernel");
    g->cuStreamCreate = dlsym(g->lib, "cuStreamCreate");
    g->cuStreamSynchronize = dlsym(g->lib, "cuStreamSynchronize");
    g->cuStreamDestroy = dlsym(g->lib, "cuStreamDestroy_v2");

    if (!g->cuInit || g->cuInit(0) != 0) return -2;
    CUdevice dev = 0;
    if (g->cuDeviceGet(&dev, 0) != 0) return -3;
    if (g->cuDeviceGetName) g->cuDeviceGetName(g->gpu_name, sizeof(g->gpu_name), dev);
    if (g->cuCtxCreate(&g->ctx, 0, dev) != 0) return -4;

    size_t free_vram = 0, total_vram = 0;
    if (g->cuMemGetInfo) g->cuMemGetInfo(&free_vram, &total_vram);
    g->free_vram_start = free_vram;

    size_t reserve = 320ULL * 1024ULL * 1024ULL;
    size_t usable = (free_vram > reserve) ? (free_vram - reserve) : 0;
    int fit_layers = (bytes_per_layer > 0) ? (int)(usable / bytes_per_layer) : 0;
    if (fit_layers > total_layers) fit_layers = total_layers;
    if (fit_layers <= 0) { g->cuCtxDestroy(g->ctx); dlclose(g->lib); return -5; }

    if (g->cuModuleLoadData(&g->mod, ZERO_Q8_PTX) != 0) return -6;
    if (g->cuModuleGetFunction(&g->fn_q8, g->mod, "q8_matvec_warp") != 0) return -7;

    size_t vram_bytes = (size_t)fit_layers * bytes_per_layer;
    if (g->cuMemAlloc(&g->d_weights, vram_bytes) != 0) return -8;
    if (g->cuMemAlloc(&g->d_act, 65536) != 0) return -9;
    if (g->cuMemAlloc(&g->d_out, 524288) != 0) return -10;
    g->cuMemHostAlloc((void**)&g->h_act_pinned, 65536, 0);
    g->cuMemHostAlloc((void**)&g->h_out_pinned, 524288, 0);
    if (g->cuStreamCreate) g->cuStreamCreate(&g->stream, 0);

    g->gpu_layers = fit_layers;
    g->vram_used = vram_bytes;
    g->active = 1;
    return 0;
}

static inline void zero_gpu_matvec(ZeroGpuBackend *g, float *out, size_t vram_off, uint32_t rows, uint32_t cols,
                                   const int8_t *act_q8, float scale, const float *bias) {
    uint32_t nb = cols / 32U;
    memcpy(g->h_act_pinned, act_q8, cols);
    g->cuMemcpyHtoDAsync(g->d_act, g->h_act_pinned, cols, g->stream);
    CUdeviceptr p_mat = g->d_weights + vram_off;
    void *args[6] = { &p_mat, &g->d_act, &g->d_out, &rows, &nb, &scale };
    g->cuLaunchKernel(g->fn_q8, (rows + 3U) / 4U, 1, 1, 128, 1, 1, 0, g->stream, args, NULL);
    g->cuMemcpyDtoHAsync(g->h_out_pinned, g->d_out, (size_t)rows * sizeof(float), g->stream);
    g->cuStreamSynchronize(g->stream);
    if (bias) {
        for (uint32_t r = 0; r < rows; r++) out[r] = g->h_out_pinned[r] + bias[r];
    } else {
        memcpy(out, g->h_out_pinned, (size_t)rows * sizeof(float));
    }
}

static inline void zero_gpu_qkv_stream(
    ZeroGpuBackend *g,
    float *q_out, size_t off_q, uint32_t rows_q, const float *bq,
    float *k_out, size_t off_k, uint32_t rows_k, const float *bk,
    float *v_out, size_t off_v, uint32_t rows_v, const float *bv,
    uint32_t cols, const int8_t *act_q8, float scale
) {
    uint32_t nb = cols / 32U;
    memcpy(g->h_act_pinned, act_q8, cols);
    g->cuMemcpyHtoDAsync(g->d_act, g->h_act_pinned, cols, g->stream);
    CUdeviceptr pq = g->d_weights + off_q, pk = g->d_weights + off_k, pv = g->d_weights + off_v;
    CUdeviceptr dq = g->d_out, dk = dq + (size_t)rows_q * sizeof(float), dv = dk + (size_t)rows_k * sizeof(float);
    void *aq[6] = { &pq, &g->d_act, &dq, &rows_q, &nb, &scale };
    void *ak[6] = { &pk, &g->d_act, &dk, &rows_k, &nb, &scale };
    void *av[6] = { &pv, &g->d_act, &dv, &rows_v, &nb, &scale };
    g->cuLaunchKernel(g->fn_q8, (rows_q + 3U) / 4U, 1, 1, 128, 1, 1, 0, g->stream, aq, NULL);
    g->cuLaunchKernel(g->fn_q8, (rows_k + 3U) / 4U, 1, 1, 128, 1, 1, 0, g->stream, ak, NULL);
    g->cuLaunchKernel(g->fn_q8, (rows_v + 3U) / 4U, 1, 1, 128, 1, 1, 0, g->stream, av, NULL);
    g->cuMemcpyDtoHAsync(g->h_out_pinned, g->d_out, ((size_t)rows_q + rows_k + rows_v) * sizeof(float), g->stream);
    g->cuStreamSynchronize(g->stream);
    const float *pq_h = g->h_out_pinned, *pk_h = pq_h + rows_q, *pv_h = pk_h + rows_k;
    for (uint32_t r = 0; r < rows_q; r++) q_out[r] = pq_h[r] + (bq ? bq[r] : 0.0f);
    for (uint32_t r = 0; r < rows_k; r++) k_out[r] = pk_h[r] + (bk ? bk[r] : 0.0f);
    for (uint32_t r = 0; r < rows_v; r++) v_out[r] = pv_h[r] + (bv ? bv[r] : 0.0f);
}

static inline void zero_gpu_gate_up_stream(
    ZeroGpuBackend *g, float *gate_out, size_t off_g, float *up_out, size_t off_u,
    uint32_t rows, uint32_t cols, const int8_t *act_q8, float scale
) {
    uint32_t nb = cols / 32U;
    memcpy(g->h_act_pinned, act_q8, cols);
    g->cuMemcpyHtoDAsync(g->d_act, g->h_act_pinned, cols, g->stream);
    CUdeviceptr pg = g->d_weights + off_g, pu = g->d_weights + off_u;
    CUdeviceptr dg = g->d_out, du = dg + (size_t)rows * sizeof(float);
    void *ag[6] = { &pg, &g->d_act, &dg, &rows, &nb, &scale };
    void *au[6] = { &pu, &g->d_act, &du, &rows, &nb, &scale };
    g->cuLaunchKernel(g->fn_q8, (rows + 3U) / 4U, 1, 1, 128, 1, 1, 0, g->stream, ag, NULL);
    g->cuLaunchKernel(g->fn_q8, (rows + 3U) / 4U, 1, 1, 128, 1, 1, 0, g->stream, au, NULL);
    g->cuMemcpyDtoHAsync(g->h_out_pinned, g->d_out, (size_t)rows * 2U * sizeof(float), g->stream);
    g->cuStreamSynchronize(g->stream);
    memcpy(gate_out, g->h_out_pinned, (size_t)rows * sizeof(float));
    memcpy(up_out, g->h_out_pinned + rows, (size_t)rows * sizeof(float));
}

static inline void zero_gpu_free(ZeroGpuBackend *g) {
    if (!g->active) return;
    g->cuMemFree(g->d_weights);
    g->cuMemFree(g->d_act);
    g->cuMemFree(g->d_out);
    g->cuMemFreeHost(g->h_act_pinned);
    g->cuMemFreeHost(g->h_out_pinned);
    if (g->stream && g->cuStreamDestroy) g->cuStreamDestroy(g->stream);
    g->cuCtxDestroy(g->ctx);
    dlclose(g->lib);
    g->active = 0;
}

#endif
