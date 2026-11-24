// ZEROMANI (zerokull1) | Author Signature: 0x5A45524F4D414E49ULL
// zero-llm: Universal Hardware-Adaptive GGUF Inference Engine (Auto GPU/DDR Split + W^X Arena)
#define _GNU_SOURCE
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <immintrin.h>
#include <omp.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include "auto_hw.h"
#include "zero_accel.h"
#include "zero_gpu_ptx.h"

#define DEFAULT_MODEL "/home/zero/AI_Models/qwen2.5-coder-7b-instruct-q8_0.gguf"
#define MAX_LAYERS_CAP 128
#define VOCAB_MAX 262144

typedef struct { uint64_t off; uint32_t rows; uint32_t cols; uint32_t ttype; } TensorRef;

typedef struct {
    TensorRef attn_norm, q, q_bias, k, k_bias, v, v_bias, o;
    TensorRef ffn_norm, gate, up, down;
} LayerWeights;

typedef struct {
    char arch[64];
    uint32_t n_layers;
    uint32_t hidden_dim;
    uint32_t ffn_dim;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    uint32_t head_dim;
    uint32_t kv_dim;
    uint32_t model_ctx_cap;
    float rope_theta;
    float rms_eps;

    char **vocab;
    uint32_t *vocab_len;
    uint32_t vocab_size;
    int bos_id, eos_id, im_start_id, im_end_id;

    TensorRef tok_embd, out_norm, output;
    LayerWeights layers[MAX_LAYERS_CAP];
    const uint8_t *w_base;
    uint8_t *mmap_ptr;
    size_t mmap_sz;

    ZeroGpuBackend gpu;
    size_t gpu_off[MAX_LAYERS_CAP][7];
    AutoHwPlan hw_plan;
    int cpu_threads;
} ZeroLlmEngine;

static inline uint64_t rdtscp_now(void) {
    uint32_t lo, hi, aux;
    __asm__ __volatile__("rdtscp" : "=a"(lo), "=d"(hi), "=c"(aux));
    return ((uint64_t)hi << 32) | lo;
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t s = (uint32_t)(h & 0x8000U) << 16;
    uint32_t e = (h >> 10) & 0x1FU, m = h & 0x03FFU, v;
    if (e == 0) v = (m == 0) ? s : (s | (113U << 23) | (m << 13));
    else if (e == 31) v = s | 0x7F800000U | (m << 13);
    else v = s | ((e + 112U) << 23) | (m << 13);
    float f; memcpy(&f, &v, 4); return f;
}

static uint32_t read_u32_kv(FILE *f, uint32_t vt) {
    if (vt == 4) { uint32_t v=0; fread(&v,4,1,f); return v; }
    if (vt == 5) { int32_t v=0; fread(&v,4,1,f); return (uint32_t)v; }
    if (vt == 10) { uint64_t v=0; fread(&v,8,1,f); return (uint32_t)v; }
    if (vt == 11) { int64_t v=0; fread(&v,8,1,f); return (uint32_t)v; }
    return 0;
}

static float read_f32_kv(FILE *f, uint32_t vt) {
    if (vt == 6) { float v=0; fread(&v,4,1,f); return v; }
    if (vt == 12) { double v=0; fread(&v,8,1,f); return (float)v; }
    return 0.0f;
}

static void skip_gguf_val(FILE *f, uint32_t vt) {
    static const int sz[13] = {1,1,2,2,4,4,4,1,0,0,8,8,8};
    if (vt <= 12 && sz[vt] > 0) fseek(f, sz[vt], SEEK_CUR);
    else if (vt == 8) { uint64_t n=0; if (fread(&n,8,1,f)==1) fseek(f,(long)n,SEEK_CUR); }
    else if (vt == 9) {
        uint32_t at=0; uint64_t al=0;
        if (fread(&at,4,1,f)==1 && fread(&al,8,1,f)==1)
            for (uint64_t i=0; i<al; i++) skip_gguf_val(f, at);
    }
}

static inline float dot_q8_row_avx2(const uint8_t *row_ptr, uint32_t n_blocks, const int8_t *act_q8, float act_scale) {
    const __m256i ones16 = _mm256_set1_epi16(1);
    float f_sum = 0.0f;
    uint32_t b = 0;
    for (; b + 3 < n_blocks; b += 4) {
        __m256i a0 = _mm256_load_si256((const __m256i *)(act_q8 + (b + 0) * 32U));
        __m256i a1 = _mm256_load_si256((const __m256i *)(act_q8 + (b + 1) * 32U));
        __m256i a2 = _mm256_load_si256((const __m256i *)(act_q8 + (b + 2) * 32U));
        __m256i a3 = _mm256_load_si256((const __m256i *)(act_q8 + (b + 3) * 32U));
        __m256i any = _mm256_or_si256(_mm256_or_si256(a0, a1), _mm256_or_si256(a2, a3));
        if (_mm256_testz_si256(any, any)) continue;

        const uint8_t *b0 = row_ptr + (size_t)(b + 0) * 34U;
        const uint8_t *b1 = row_ptr + (size_t)(b + 1) * 34U;
        const uint8_t *b2 = row_ptr + (size_t)(b + 2) * 34U;
        const uint8_t *b3 = row_ptr + (size_t)(b + 3) * 34U;

        float d0 = f16_to_f32((uint16_t)b0[0] | ((uint16_t)b0[1] << 8));
        float d1 = f16_to_f32((uint16_t)b1[0] | ((uint16_t)b1[1] << 8));
        float d2 = f16_to_f32((uint16_t)b2[0] | ((uint16_t)b2[1] << 8));
        float d3 = f16_to_f32((uint16_t)b3[0] | ((uint16_t)b3[1] << 8));

        __m256i w0 = _mm256_loadu_si256((const __m256i *)(b0 + 2));
        __m256i w1 = _mm256_loadu_si256((const __m256i *)(b1 + 2));
        __m256i w2 = _mm256_loadu_si256((const __m256i *)(b2 + 2));
        __m256i w3 = _mm256_loadu_si256((const __m256i *)(b3 + 2));

        __m256i p0 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w0), _mm256_sign_epi8(a0, w0)), ones16);
        __m256i p1 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w1), _mm256_sign_epi8(a1, w1)), ones16);
        __m256i p2 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w2), _mm256_sign_epi8(a2, w2)), ones16);
        __m256i p3 = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w3), _mm256_sign_epi8(a3, w3)), ones16);

        __m128i s0 = _mm_add_epi32(_mm256_castsi256_si128(p0), _mm256_extracti128_si256(p0, 1));
        __m128i s1 = _mm_add_epi32(_mm256_castsi256_si128(p1), _mm256_extracti128_si256(p1, 1));
        __m128i s2 = _mm_add_epi32(_mm256_castsi256_si128(p2), _mm256_extracti128_si256(p2, 1));
        __m128i s3 = _mm_add_epi32(_mm256_castsi256_si128(p3), _mm256_extracti128_si256(p3, 1));
        __m128i h0123 = _mm_hadd_epi32(_mm_hadd_epi32(s0, s1), _mm_hadd_epi32(s2, s3));
        __m128 prod = _mm_mul_ps(_mm_cvtepi32_ps(h0123), _mm_set_ps(d3, d2, d1, d0));
        prod = _mm_hadd_ps(prod, prod);
        prod = _mm_hadd_ps(prod, prod);
        f_sum += _mm_cvtss_f32(prod);
    }
    for (; b < n_blocks; b++) {
        __m256i a = _mm256_load_si256((const __m256i *)(act_q8 + b * 32U));
        if (_mm256_testz_si256(a, a)) continue;
        const uint8_t *blk = row_ptr + (size_t)b * 34U;
        float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        __m256i w = _mm256_loadu_si256((const __m256i *)(blk + 2));
        __m256i p = _mm256_madd_epi16(_mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(a, w)), ones16);
        __m128i s = _mm_add_epi32(_mm256_castsi256_si128(p), _mm256_extracti128_si256(p, 1));
        s = _mm_hadd_epi32(s, s); s = _mm_hadd_epi32(s, s);
        f_sum += d * (float)_mm_cvtsi128_si32(s);
    }
    return f_sum * act_scale;
}

static void matvec_q8_avx2(float *out, const uint8_t *w_base, TensorRef tr, const int8_t *act_q8, float act_scale, const float *bias) {
    uint32_t nb = tr.cols / 32U;
    size_t row_bytes = (size_t)nb * 34U;
    const uint8_t *mat = w_base + tr.off;
    #pragma omp parallel for schedule(static)
    for (uint32_t r = 0; r < tr.rows; r++) {
        float val = dot_q8_row_avx2(mat + (size_t)r * row_bytes, nb, act_q8, act_scale);
        if (bias) val += bias[r];
        out[r] = val;
    }
}

static void rmsnorm_f32(float *out, const float *x, const float *w, uint32_t n, float eps) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static void apply_rope_split(float *vec, uint32_t n_heads, uint32_t head_dim, uint32_t pos, float theta) {
    uint32_t half = head_dim / 2U;
    for (uint32_t h = 0; h < n_heads; h++) {
        float *hv = vec + h * head_dim;
        for (uint32_t i = 0; i < half; i++) {
            float freq = 1.0f / powf(theta, (float)(2U * i) / (float)head_dim);
            float ang = (float)pos * freq;
            float c = cosf(ang), s = sinf(ang);
            float x0 = hv[i], x1 = hv[i + half];
            hv[i]        = x0 * c - x1 * s;
            hv[i + half] = x0 * s + x1 * c;
        }
    }
}

static float quantize_q8(const float *x, int8_t *q, uint32_t n) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    if (amax < 1e-12f) { memset(q, 0, n); return 0.0f; }
    float scale = amax / 127.0f, inv = 127.0f / amax;
    for (uint32_t i = 0; i < n; i++) {
        int v = (int)lrintf(x[i] * inv);
        if (v > 127) v = 127; else if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
    return scale;
}

static int zero_llm_load(ZeroLlmEngine *m, const char *path) {
    memset(m, 0, sizeof(*m));
    strcpy(m->arch, "qwen2");
    m->rope_theta = 1000000.0f;
    m->rms_eps = 1e-6f;
    m->bos_id = 151643; m->eos_id = 151645;
    m->im_start_id = 151644; m->im_end_id = 151645;

    long ncpu = sysconf(_SC_NPROCESSORS_ONLN);
    m->cpu_threads = (ncpu > 0) ? (int)ncpu : 4;

    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    char magic[4]; uint32_t ver=0; uint64_t nt=0, nkv=0;
    if (fread(magic,1,4,f)!=4 || memcmp(magic,"GGUF",4)!=0) { fclose(f); return -2; }
    fread(&ver,4,1,f); fread(&nt,8,1,f); fread(&nkv,8,1,f);

    char key[256];
    for (uint64_t i=0; i<nkv; i++) {
        uint64_t kl=0; fread(&kl,8,1,f);
        size_t rk = kl<255 ? (size_t)kl : 255;
        fread(key,1,rk,f); key[rk]='\0';
        if (kl>rk) fseek(f,(long)(kl-rk),SEEK_CUR);
        uint32_t vt=0; fread(&vt,4,1,f);

        if (strcmp(key, "general.architecture") == 0 && vt == 8) {
            uint64_t sl=0; fread(&sl,8,1,f);
            size_t rs = sl<63 ? (size_t)sl : 63;
            fread(m->arch,1,rs,f); m->arch[rs]='\0';
            if (sl>rs) fseek(f,(long)(sl-rs),SEEK_CUR);
        } else if (strstr(key, ".block_count")) m->n_layers = read_u32_kv(f, vt);
        else if (strstr(key, ".embedding_length")) m->hidden_dim = read_u32_kv(f, vt);
        else if (strstr(key, ".feed_forward_length")) m->ffn_dim = read_u32_kv(f, vt);
        else if (strstr(key, ".attention.head_count_kv")) m->n_kv_heads = read_u32_kv(f, vt);
        else if (strstr(key, ".attention.head_count") && !strstr(key, "_kv")) m->n_heads = read_u32_kv(f, vt);
        else if (strstr(key, ".context_length")) m->model_ctx_cap = read_u32_kv(f, vt);
        else if (strstr(key, ".rope.freq_base")) m->rope_theta = read_f32_kv(f, vt);
        else if (strstr(key, ".attention.layer_norm_rms_epsilon")) m->rms_eps = read_f32_kv(f, vt);
        else if (strcmp(key, "tokenizer.ggml.bos_token_id") == 0) m->bos_id = (int)read_u32_kv(f, vt);
        else if (strcmp(key, "tokenizer.ggml.eos_token_id") == 0) m->eos_id = (int)read_u32_kv(f, vt);
        else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vt == 9) {
            uint32_t at=0; uint64_t al=0;
            fread(&at,4,1,f); fread(&al,8,1,f);
            m->vocab_size = (al < VOCAB_MAX) ? (uint32_t)al : VOCAB_MAX;
            m->vocab = calloc(m->vocab_size, sizeof(char*));
            m->vocab_len = calloc(m->vocab_size, sizeof(uint32_t));
            for (uint64_t t=0; t<al; t++) {
                uint64_t sl=0; fread(&sl,8,1,f);
                if (t < m->vocab_size) {
                    char *s = malloc((size_t)sl + 1);
                    fread(s, 1, (size_t)sl, f); s[sl] = '\0';
                    m->vocab[t] = s; m->vocab_len[t] = (uint32_t)sl;
                    if (strcmp(s, "<|im_start|>") == 0) m->im_start_id = (int)t;
                    if (strcmp(s, "<|im_end|>") == 0) m->im_end_id = (int)t;
                } else fseek(f, (long)sl, SEEK_CUR);
            }
        } else skip_gguf_val(f, vt);
    }

    if (m->n_layers == 0 || m->n_layers > MAX_LAYERS_CAP) m->n_layers = 28;
    if (m->n_kv_heads == 0) m->n_kv_heads = m->n_heads;
    m->head_dim = m->hidden_dim / m->n_heads;
    m->kv_dim = m->n_kv_heads * m->head_dim;

    char name[128];
    for (uint64_t i=0; i<nt; i++) {
        uint64_t nl=0; fread(&nl,8,1,f);
        size_t rl = nl<127 ? (size_t)nl : 127;
        fread(name,1,rl,f); name[rl]='\0';
        if (nl>rl) fseek(f,(long)(nl-rl),SEEK_CUR);
        uint32_t nd=0; fread(&nd,4,1,f);
        uint64_t d0=1, d1=1;
        if (nd>=1) fread(&d0,8,1,f);
        if (nd>=2) fread(&d1,8,1,f);
        for (uint32_t d=2; d<nd; d++) { uint64_t tmp; fread(&tmp,8,1,f); }
        uint32_t tt=0; uint64_t off=0;
        fread(&tt,4,1,f); fread(&off,8,1,f);
        TensorRef tr = { off, (uint32_t)d1, (uint32_t)d0, tt };

        if (strcmp(name, "token_embd.weight")==0) m->tok_embd = tr;
        else if (strcmp(name, "output_norm.weight")==0) m->out_norm = tr;
        else if (strcmp(name, "output.weight")==0) m->output = tr;
        else {
            int l = -1; char sub[64] = {0};
            if (sscanf(name, "blk.%d.%63s", &l, sub) == 2 && l >= 0 && (uint32_t)l < m->n_layers) {
                LayerWeights *lw = &m->layers[l];
                if (strcmp(sub, "attn_norm.weight")==0) lw->attn_norm = tr;
                else if (strcmp(sub, "attn_q.weight")==0) lw->q = tr;
                else if (strcmp(sub, "attn_q.bias")==0) lw->q_bias = tr;
                else if (strcmp(sub, "attn_k.weight")==0) lw->k = tr;
                else if (strcmp(sub, "attn_k.bias")==0) lw->k_bias = tr;
                else if (strcmp(sub, "attn_v.weight")==0) lw->v = tr;
                else if (strcmp(sub, "attn_v.bias")==0) lw->v_bias = tr;
                else if (strcmp(sub, "attn_output.weight")==0) lw->o = tr;
                else if (strcmp(sub, "ffn_norm.weight")==0) lw->ffn_norm = tr;
                else if (strcmp(sub, "ffn_gate.weight")==0) lw->gate = tr;
                else if (strcmp(sub, "ffn_up.weight")==0) lw->up = tr;
                else if (strcmp(sub, "ffn_down.weight")==0) lw->down = tr;
            }
        }
    }
    if (m->output.off == 0) m->output = m->tok_embd;
    uint64_t data_start = ((uint64_t)ftell(f) + 31ULL) & ~31ULL;
    fclose(f);

    int fd = open(path, O_RDONLY);
    struct stat st; fstat(fd, &st);
    m->mmap_sz = (size_t)st.st_size;
    m->mmap_ptr = mmap(NULL, m->mmap_sz, PROT_READ, MAP_PRIVATE, fd, 0);
    close(fd);
    if (m->mmap_ptr == MAP_FAILED) return -3;
    madvise(m->mmap_ptr, m->mmap_sz, MADV_HUGEPAGE | MADV_WILLNEED);
    m->w_base = m->mmap_ptr + data_start;

    m->hw_plan = hw_calc_token_plan(m->n_layers, m->n_kv_heads, m->head_dim, m->model_ctx_cap);

    LayerWeights *l0 = &m->layers[0];
    TensorRef *t0[7] = { &l0->q, &l0->k, &l0->v, &l0->o, &l0->gate, &l0->up, &l0->down };
    size_t layer_bytes = 0;
    for (int j = 0; j < 7; j++)
        layer_bytes += (((size_t)t0[j]->rows * (t0[j]->cols / 32U) * 34U + 255ULL) & ~255ULL);

    if (zero_gpu_auto_init(&m->gpu, (int)m->n_layers, layer_bytes) == 0) {
        size_t cur = 0;
        for (int l = 0; l < m->gpu.gpu_layers; l++) {
            LayerWeights *lw = &m->layers[l];
            TensorRef *trs[7] = { &lw->q, &lw->k, &lw->v, &lw->o, &lw->gate, &lw->up, &lw->down };
            for (int j = 0; j < 7; j++) {
                size_t bytes = (size_t)trs[j]->rows * (trs[j]->cols / 32U) * 34U;
                m->gpu_off[l][j] = cur;
                m->gpu.cuMemcpyHtoD(m->gpu.d_weights + cur, m->w_base + trs[j]->off, bytes);
                cur = (cur + bytes + 255ULL) & ~255ULL;
            }
        }
    }
    return 0;
}

static void byte_to_bpe_utf8(uint8_t b, char *out, int *olen) {
    uint32_t cp = b;
    if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174))) {
        uint32_t n = 0;
        for (uint32_t x = 0; x < b; x++)
            if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255))) n++;
        cp = 256U + n;
    }
    if (cp < 0x80) { out[0] = (char)cp; *olen = 1; }
    else { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); *olen = 2; }
}

static int bpe_utf8_to_byte(const uint8_t *s, int *consumed) {
    if (s[0] < 0x80) { *consumed = 1; return s[0]; }
    if ((s[0] & 0xE0) == 0xC0 && (s[1] & 0xC0) == 0x80) {
        uint32_t cp = ((uint32_t)(s[0] & 0x1F) << 6) | (uint32_t)(s[1] & 0x3F);
        *consumed = 2;
        if (cp < 256) return (int)cp;
        uint32_t target = cp - 256U, n = 0;
        for (uint32_t x = 0; x < 256; x++) {
            if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255))) {
                if (n == target) return (int)x;
                n++;
            }
        }
    }
    *consumed = 1;
    return s[0];
}

static int encode_raw_bpe(const ZeroLlmEngine *m, const char *raw, int *out_ids, int max_ids) {
    char mapped[8192];
    int mlen = 0;
    for (const uint8_t *p = (const uint8_t *)raw; *p && mlen < 8180; p++) {
        int cl = 0; byte_to_bpe_utf8(*p, mapped + mlen, &cl); mlen += cl;
    }
    mapped[mlen] = '\0';
    int n = 0, pos = 0;
    while (pos < mlen && n < max_ids) {
        int best_id = -1; size_t best_l = 0;
        for (uint32_t t = 0; t < m->vocab_size; t++) {
            if ((int)t == m->im_start_id || (int)t == m->im_end_id) continue;
            size_t tl = m->vocab_len[t];
            if (tl > best_l && (size_t)(pos + tl) <= (size_t)mlen && memcmp(mapped + pos, m->vocab[t], tl) == 0) {
                best_id = (int)t; best_l = tl;
            }
        }
        if (best_id < 0) pos++; else { out_ids[n++] = best_id; pos += (int)best_l; }
    }
    return n;
}

static int encode_turn(const ZeroLlmEngine *m, const char *text, int include_sys, int *out_ids, int max_ids) {
    int n = 0;
    if (include_sys) {
        out_ids[n++] = m->im_start_id;
        n += encode_raw_bpe(m, "system\nYou are an experienced Rust and C11 programmer. Write clean, working code and explain concisely in Russian.", out_ids + n, max_ids - n - 16);
        out_ids[n++] = m->im_end_id;
        n += encode_raw_bpe(m, "\n", out_ids + n, max_ids - n - 16);
    }
    out_ids[n++] = m->im_start_id;
    n += encode_raw_bpe(m, "user\n", out_ids + n, max_ids - n - 16);
    n += encode_raw_bpe(m, text, out_ids + n, max_ids - n - 16);
    out_ids[n++] = m->im_end_id;
    n += encode_raw_bpe(m, "\n", out_ids + n, max_ids - n - 16);
    out_ids[n++] = m->im_start_id;
    n += encode_raw_bpe(m, "assistant\n", out_ids + n, max_ids - n - 16);
    return n;
}

static void print_piece(const char *s) {
    const uint8_t *p = (const uint8_t *)s;
    while (*p) { int cons = 1; putchar(bpe_utf8_to_byte(p, &cons)); p += cons; }
    fflush(stdout);
}

static void forward_token(
    ZeroLlmEngine *m, int token_id, uint32_t pos,
    float *k_cache, float *v_cache, float *logits, float sparse_th
) {
    const uint32_t H = m->hidden_dim, FFN = m->ffn_dim, KVD = m->kv_dim;
    static float *x = NULL, *xb = NULL, *q = NULL, *k = NULL, *v = NULL, *att = NULL, *gate_b = NULL, *up_b = NULL, *scores = NULL;
    static int8_t *q8_v = NULL;
    if (!x) {
        x = aligned_alloc(64, H * sizeof(float)); xb = aligned_alloc(64, H * sizeof(float));
        q = aligned_alloc(64, H * sizeof(float)); k = aligned_alloc(64, KVD * sizeof(float));
        v = aligned_alloc(64, KVD * sizeof(float)); att = aligned_alloc(64, H * sizeof(float));
        gate_b = aligned_alloc(64, FFN * sizeof(float)); up_b = aligned_alloc(64, FFN * sizeof(float));
        q8_v = aligned_alloc(64, (FFN > H ? FFN : H));
        scores = aligned_alloc(64, (size_t)m->hw_plan.max_seq * sizeof(float));
    }

    if (m->tok_embd.ttype == 8) {
        const uint8_t *er = m->w_base + m->tok_embd.off + (size_t)token_id * (H / 32U) * 34U;
        for (uint32_t b = 0; b < H / 32U; b++) {
            const uint8_t *blk = er + (size_t)b * 34U;
            float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            const int8_t *qs = (const int8_t *)(blk + 2);
            for (uint32_t j = 0; j < 32U; j++) x[b * 32U + j] = d * (float)qs[j];
        }
    } else {
        const uint16_t *e16 = (const uint16_t *)(m->w_base + m->tok_embd.off + (size_t)token_id * H * 2U);
        for (uint32_t j = 0; j < H; j++) x[j] = f16_to_f32(e16[j]);
    }

    (void)zero_gf64_route(0x5A45524F4D414E49ULL, (uint64_t)token_id + pos);

    for (uint32_t l = 0; l < m->n_layers; l++) {
        LayerWeights *lw = &m->layers[l];
        rmsnorm_f32(xb, x, (const float *)(m->w_base + lw->attn_norm.off), H, m->rms_eps);
        float sc_x = quantize_q8(xb, q8_v, H);

        const float *qb = lw->q_bias.off ? (const float *)(m->w_base + lw->q_bias.off) : NULL;
        const float *kb = lw->k_bias.off ? (const float *)(m->w_base + lw->k_bias.off) : NULL;
        const float *vb = lw->v_bias.off ? (const float *)(m->w_base + lw->v_bias.off) : NULL;

        if (m->gpu.active && (int)l < m->gpu.gpu_layers) {
            zero_gpu_qkv_stream(&m->gpu, q, m->gpu_off[l][0], H, qb, k, m->gpu_off[l][1], KVD, kb, v, m->gpu_off[l][2], KVD, vb, H, q8_v, sc_x);
        } else {
            matvec_q8_avx2(q, m->w_base, lw->q, q8_v, sc_x, qb);
            matvec_q8_avx2(k, m->w_base, lw->k, q8_v, sc_x, kb);
            matvec_q8_avx2(v, m->w_base, lw->v, q8_v, sc_x, vb);
        }

        apply_rope_split(q, m->n_heads, m->head_dim, pos, m->rope_theta);
        apply_rope_split(k, m->n_kv_heads, m->head_dim, pos, m->rope_theta);

        size_t kv_off = ((size_t)l * m->hw_plan.max_seq + (size_t)pos) * KVD;
        memcpy(k_cache + kv_off, k, KVD * sizeof(float));
        memcpy(v_cache + kv_off, v, KVD * sizeof(float));

        const uint32_t gsz = m->n_heads / m->n_kv_heads;
        const float inv_sd = 1.0f / sqrtf((float)m->head_dim);
        for (uint32_t h = 0; h < m->n_heads; h++) {
            uint32_t kvh = h / gsz;
            const float *qh = q + h * m->head_dim;
            float *oh = att + h * m->head_dim, max_s = -1e30f;
            for (uint32_t t = 0; t <= pos; t++) {
                const float *kh = k_cache + (((size_t)l * m->hw_plan.max_seq + (size_t)t) * KVD) + kvh * m->head_dim;
                float s = 0.0f;
                for (uint32_t d = 0; d < m->head_dim; d++) s += qh[d] * kh[d];
                s *= inv_sd; scores[t] = s;
                if (s > max_s) max_s = s;
            }
            float sum_e = 0.0f;
            for (uint32_t t = 0; t <= pos; t++) { scores[t] = expf(scores[t] - max_s); sum_e += scores[t]; }
            float inv_e = 1.0f / sum_e;
            memset(oh, 0, m->head_dim * sizeof(float));
            for (uint32_t t = 0; t <= pos; t++) {
                float pr = scores[t] * inv_e;
                const float *vh = v_cache + (((size_t)l * m->hw_plan.max_seq + (size_t)t) * KVD) + kvh * m->head_dim;
                for (uint32_t d = 0; d < m->head_dim; d++) oh[d] += pr * vh[d];
            }
        }

        float sc_a = quantize_q8(att, q8_v, H);
        if (m->gpu.active && (int)l < m->gpu.gpu_layers)
            zero_gpu_matvec(&m->gpu, xb, m->gpu_off[l][3], H, H, q8_v, sc_a, NULL);
        else
            matvec_q8_avx2(xb, m->w_base, lw->o, q8_v, sc_a, NULL);
        for (uint32_t i = 0; i < H; i++) x[i] += xb[i];

        rmsnorm_f32(xb, x, (const float *)(m->w_base + lw->ffn_norm.off), H, m->rms_eps);
        float sc_f = quantize_q8(xb, q8_v, H);

        if (m->gpu.active && (int)l < m->gpu.gpu_layers) {
            zero_gpu_gate_up_stream(&m->gpu, gate_b, m->gpu_off[l][4], up_b, m->gpu_off[l][5], FFN, H, q8_v, sc_f);
            for (uint32_t r = 0; r < FFN; r++) {
                float silu = zero_fast_silu(gate_b[r]);
                up_b[r] = (fabsf(silu) > sparse_th) ? (silu * up_b[r]) : 0.0f;
            }
            float sc_d = quantize_q8(up_b, q8_v, FFN);
            zero_gpu_matvec(&m->gpu, xb, m->gpu_off[l][6], H, FFN, q8_v, sc_d, NULL);
        } else {
            matvec_q8_avx2(gate_b, m->w_base, lw->gate, q8_v, sc_f, NULL);
            const uint8_t *ub = m->w_base + lw->up.off;
            const size_t urb = (H / 32U) * 34U;
            #pragma omp parallel for schedule(static)
            for (uint32_t r = 0; r < FFN; r++) {
                float silu = zero_fast_silu(gate_b[r]);
                if (fabsf(silu) > sparse_th)
                    up_b[r] = silu * dot_q8_row_avx2(ub + (size_t)r * urb, H / 32U, q8_v, sc_f);
                else
                    up_b[r] = 0.0f;
            }
            float sc_d = quantize_q8(up_b, q8_v, FFN);
            matvec_q8_avx2(xb, m->w_base, lw->down, q8_v, sc_d, NULL);
        }
        for (uint32_t i = 0; i < H; i++) x[i] += xb[i];
    }

    rmsnorm_f32(xb, x, (const float *)(m->w_base + m->out_norm.off), H, m->rms_eps);
    float sc_o = quantize_q8(xb, q8_v, H);
    matvec_q8_avx2(logits, m->w_base, m->output, q8_v, sc_o, NULL);
}

int main(int argc, char **argv) {
    const char *model_path = DEFAULT_MODEL, *prompt = NULL;
    float sparse_th = 0.25f;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) model_path = argv[++i];
        else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) sparse_th = strtof(argv[++i], NULL);
        else prompt = argv[i];
    }
    ZeroLlmEngine *m = calloc(1, sizeof(ZeroLlmEngine));
    if (zero_llm_load(m, model_path) != 0) { fprintf(stderr, "[zero-llm] Load failed: %s\n", model_path); return 1; }
    omp_set_num_threads(m->cpu_threads);
    printf("[zero-llm] Arch=%s | Layers=%u | Hidden=%u | Vocab=%u | CPU=%d threads\n",
           m->arch, m->n_layers, m->hidden_dim, m->vocab_size, m->cpu_threads);
    if (m->gpu.active)
        printf("[zero-llm] Auto-GPU: %s | FreeVRAM=%.0f MiB | GPU_Layers=%d/%u (%.1f MiB) | DDR_W^X=%u\n",
               m->gpu.gpu_name, (double)m->gpu.free_vram_start/(1024*1024), m->gpu.gpu_layers, m->n_layers,
               (double)m->gpu.vram_used/(1024*1024), m->n_layers - (uint32_t)m->gpu.gpu_layers);
    else
        printf("[zero-llm] Auto-GPU: none -> 100%% W^X 2MB-HugePage DDR Vault (%u layers)\n", m->n_layers);
    printf("[zero-llm] Auto-Context: max_seq=%u | max_gen=%u | KV_RAM=%.1f MiB (FreeRAM=%.2f GiB)\n",
           m->hw_plan.max_seq, m->hw_plan.max_gen, (double)m->hw_plan.kv_bytes/(1024*1024),
           (double)m->hw_plan.avail_ram_bytes/(1024*1024*1024));

    float *kc = aligned_alloc(64, m->hw_plan.kv_bytes), *vc = aligned_alloc(64, m->hw_plan.kv_bytes);
    float *logits = aligned_alloc(64, (size_t)m->vocab_size * sizeof(float));
    memset(kc, 0, m->hw_plan.kv_bytes); memset(vc, 0, m->hw_plan.kv_bytes);
    int *ids = malloc(4096 * sizeof(int));
    uint32_t pos = 0;

    if (prompt) {
        int n_in = encode_turn(m, prompt, 1, ids, 4096);
        for (int i = 0; i < n_in; i++) forward_token(m, ids[i], pos++, kc, vc, logits, sparse_th);
        printf("Response: "); fflush(stdout);
        uint64_t t0 = rdtscp_now(); uint32_t gen = 0;
        for (uint32_t g = 0; g < m->hw_plan.max_gen && pos < m->hw_plan.max_seq; g++) {
            int nt = 0; float bv = logits[0];
            for (uint32_t v = 1; v < m->vocab_size; v++) if (logits[v] > bv) { bv = logits[v]; nt = (int)v; }
            if (nt == m->im_end_id || nt == m->eos_id) break;
            print_piece(m->vocab[nt]); gen++;
            forward_token(m, nt, pos++, kc, vc, logits, sparse_th);
        }
        double sec = (double)(rdtscp_now() - t0) / 3.3e9;
        printf("\n\n[SPEED] %u tokens in %.2f s (%.2f tok/s | ctx=%u/%u)\n",
               gen, sec, gen > 0 ? (double)gen / sec : 0.0, pos, m->hw_plan.max_seq);
    } else {
        printf("[CHAT] Commands: /clear, /exit\n");
        char line[2048];
        while (1) {
            printf("\nYou> "); fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t len = strlen(line);
            while (len > 0 && (line[len-1]=='\n'||line[len-1]=='\r')) line[--len]='\0';
            if (len == 0) continue;
            if (strcmp(line, "/exit") == 0) break;
            if (strcmp(line, "/clear") == 0) { pos=0; memset(kc,0,m->hw_plan.kv_bytes); memset(vc,0,m->hw_plan.kv_bytes); printf("[KV CLEARED]\n"); continue; }
            int n_in = encode_turn(m, line, pos == 0, ids, 4096);
            if (pos + (uint32_t)n_in + 16 >= m->hw_plan.max_seq) {
                pos=0; memset(kc,0,m->hw_plan.kv_bytes); memset(vc,0,m->hw_plan.kv_bytes);
                n_in = encode_turn(m, line, 1, ids, 4096);
            }
            for (int i = 0; i < n_in; i++) forward_token(m, ids[i], pos++, kc, vc, logits, sparse_th);
            printf("AI> "); fflush(stdout);
            uint64_t t0 = rdtscp_now(); uint32_t gen = 0;
            for (uint32_t g = 0; g < m->hw_plan.max_gen && pos < m->hw_plan.max_seq; g++) {
                int nt = 0; float bv = logits[0];
                for (uint32_t v = 1; v < m->vocab_size; v++) if (logits[v] > bv) { bv = logits[v]; nt = (int)v; }
                if (nt == m->im_end_id || nt == m->eos_id) break;
                print_piece(m->vocab[nt]); gen++;
                forward_token(m, nt, pos++, kc, vc, logits, sparse_th);
            }
            double sec = (double)(rdtscp_now() - t0) / 3.3e9;
            printf("\n\n[SPEED] %u tokens in %.2f s (%.2f tok/s | ctx=%u/%u)\n",
                   gen, sec, gen > 0 ? (double)gen / sec : 0.0, pos, m->hw_plan.max_seq);
        }
    }
    zero_gpu_free(&m->gpu); munmap(m->mmap_ptr, m->mmap_sz);
    free(kc); free(vc); free(logits); free(ids); free(m);
    return 0;
}
