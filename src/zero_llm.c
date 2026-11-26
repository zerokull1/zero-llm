#include "zero_gpu_ring_fastpath.h"
// ZEROMANI (zerokull1) | Author Signature: 0x5A45524F4D414E49ULL
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
#include "zero_engine.h"

#define DEFAULT_MODEL "/home/zero/AI_Models/qwen2.5-coder-7b-instruct-q8_0.gguf"
#define NUM_LAYERS ((int)m->n_layers)
#define HIDDEN (m->hidden_dim)
#define FFN_DIM (m->ffn_dim)
#define N_HEADS ((int)m->n_heads)
#define N_KV_HEADS ((int)m->n_kv_heads)
#define HEAD_DIM (m->head_dim)
#define KV_DIM (m->kv_dim)
#define MAX_SEQ ((int)m->hw_plan.max_seq)
#define ROPE_THETA (m->rope_theta)
#define RMS_EPS 1e-6f

typedef ZeroLlmModel QwenWxModel;

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

static int load_qwen_gguf(QwenWxModel *m, const char *path) {
    memset(m, 0, sizeof(*m)); strcpy(m->arch,"qwen2"); strcpy(m->tok_model,"gpt2"); m->rope_theta=1000000.0f; m->rms_eps=1e-6f; m->bos_id=151643; m->eos_id=151645; m->im_start_id=151644; m->im_end_id=151645; m->nl_id=198;
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

        if (strstr(key, ".block_count")) { uint32_t v=0; fread(&v,4,1,f); m->n_layers=v; }
        else if (strstr(key, ".embedding_length")) { uint32_t v=0; fread(&v,4,1,f); m->hidden_dim=v; }
        else if (strstr(key, ".feed_forward_length")) { uint32_t v=0; fread(&v,4,1,f); m->ffn_dim=v; }
        else if (strstr(key, ".attention.head_count_kv")) { uint32_t v=0; fread(&v,4,1,f); m->n_kv_heads=v; }
        else if (strstr(key, ".attention.head_count") && !strstr(key, "_kv")) { uint32_t v=0; fread(&v,4,1,f); m->n_heads=v; }
        else if (strstr(key, ".context_length")) { uint32_t v=0; fread(&v,4,1,f); m->model_ctx_cap=v; }
        else if (strstr(key, ".rope.freq_base")) { float v=0; fread(&v,4,1,f); if(v>0) m->rope_theta=v; }
        else if (strstr(key, ".attention.layer_norm_rms_epsilon")) { float v=0; fread(&v,4,1,f); if(v>0) m->rms_eps=v; }
        else if (!strcmp(key, "tokenizer.ggml.model") && vt==8) { uint64_t sl=0; fread(&sl,8,1,f); size_t rs=sl<31?sl:31; fread(m->tok_model,1,rs,f); m->tok_model[rs]=0; if(sl>rs) fseek(f,(long)(sl-rs),SEEK_CUR); }
        else if (!strcmp(key, "tokenizer.chat_template") && vt==8) { uint64_t sl=0; fread(&sl,8,1,f); char tpl[1024]={0}; size_t rs=sl<1023?sl:1023; fread(tpl,1,rs,f); if(sl>rs) fseek(f,(long)(sl-rs),SEEK_CUR); zk_detect_chat_tpl(tpl, &m->chat_fmt); }
        else if (strcmp(key, "tokenizer.ggml.tokens") == 0 && vt == 9) {
            uint32_t at=0; uint64_t al=0;
            fread(&at,4,1,f); fread(&al,8,1,f);
            m->vocab_size = (al < VOCAB_MAX) ? (uint32_t)al : VOCAB_MAX;
            for (uint64_t t=0; t<al; t++) {
                uint64_t sl=0; fread(&sl,8,1,f);
                if (t < VOCAB_MAX) {
                    char *s = malloc((size_t)sl + 1);
                    fread(s, 1, (size_t)sl, f); s[sl] = '\0';
                    m->vocab[t] = s;
                    m->vocab_len[t] = (uint32_t)sl;
                } else fseek(f, (long)sl, SEEK_CUR);
            }
        } else {
            skip_gguf_val(f, vt);
        }
    }

    if (!m->n_layers) m->n_layers = 28;
    if (!m->n_kv_heads) m->n_kv_heads = m->n_heads;
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
            if (sscanf(name, "blk.%d.%63s", &l, sub) == 2 && l >= 0 && l < NUM_LAYERS) {
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
    LayerWeights *l0 = &m->layers[0]; TensorRef *t0[7] = {&l0->q,&l0->k,&l0->v,&l0->o,&l0->gate,&l0->up,&l0->down}; size_t bpl=0; for(int j=0;j<7;j++) bpl+=(((size_t)t0[j]->rows*(t0[j]->cols/32U)*34U+255ULL)&~255ULL);
    if (zero_gpu_auto_init(&m->gpu, (int)m->n_layers, bpl) == 0) {
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
        printf("[ZERO-LLM] GTX 1050 Pascal dp4a synced: %d/%d layers (%.1f MiB GDDR5)\n",
               m->gpu.gpu_layers, NUM_LAYERS, (double)cur / (1024.0 * 1024.0));
    }
    size_t wx_need = (4U*m->hidden_dim + 2U*m->kv_dim + 2U*m->ffn_dim)*4U + m->ffn_dim + 4096U;
    if (zero_tokamak_wx_init(&m->wx_arena, wx_need) != 0) return -4;
    return 0;
}

static inline float quantize_f32_to_q8(const float *x, int8_t *q, uint32_t n) {
    float amax = 0.0f;
    for (uint32_t i = 0; i < n; i++) {
        float a = fabsf(x[i]);
        if (a > amax) amax = a;
    }
    float scale = (amax > 1e-12f) ? (amax / 127.0f) : 1.0f;
    float inv = 1.0f / scale;
    for (uint32_t i = 0; i < n; i++) {
        int v = (int)roundf(x[i] * inv);
        if (v > 127) v = 127;
        if (v < -127) v = -127;
        q[i] = (int8_t)v;
    }
    return scale;
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
        __m128i h01 = _mm_hadd_epi32(s0, s1);
        __m128i h23 = _mm_hadd_epi32(s2, s3);
        __m128i h0123 = _mm_hadd_epi32(h01, h23);
        __m128 vd = _mm_set_ps(d3, d2, d1, d0);
        __m128 prod = _mm_mul_ps(_mm_cvtepi32_ps(h0123), vd);
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
        s = _mm_hadd_epi32(s, s);
        s = _mm_hadd_epi32(s, s);
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

static void rmsnorm_f32(float *out, const float *x, const float *w, uint32_t n) {
    float ss = 0.0f;
    for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + RMS_EPS);
    for (uint32_t i = 0; i < n; i++) out[i] = x[i] * inv * w[i];
}

static void apply_rope_qwen(const QwenWxModel *m, float *vec, uint32_t n_heads, uint32_t pos) {
    // Qwen 2.5 использует split-half RoPE (первая половина 64 dims + вторая половина 64 dims)
    uint32_t half = HEAD_DIM / 2;
    for (uint32_t h = 0; h < n_heads; h++) {
        float *hv = vec + h * HEAD_DIM;
        for (uint32_t i = 0; i < half; i++) {
            float freq = 1.0f / powf(ROPE_THETA, (float)(2 * i) / (float)HEAD_DIM);
            float ang = (float)pos * freq;
            float c = cosf(ang), s = sinf(ang);
            float x0 = hv[i], x1 = hv[i + half];
            hv[i]        = x0 * c - x1 * s;
            hv[i + half] = x0 * s + x1 * c;
        }
    }
}

static void dequant_q8_row(float *out, const uint8_t *row_ptr, uint32_t cols) {
    uint32_t nb = cols / 32U;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = row_ptr + (size_t)b * 34U;
        float d = f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
        const int8_t *qs = (const int8_t *)(blk + 2);
        for (uint32_t k = 0; k < 32; k++) out[b * 32U + k] = d * (float)qs[k];
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

static float dot_q6k_row(const uint8_t *row_ptr, uint32_t cols, const float *x) {
    uint32_t nb = cols / 256U;
    float sum = 0.0f;
    for (uint32_t b = 0; b < nb; b++) {
        const uint8_t *blk = row_ptr + (size_t)b * 210U;
        const uint8_t *ql = blk, *qh = blk + 128;
        const int8_t *sc = (const int8_t *)(blk + 192);
        float d = f16_to_f32((uint16_t)blk[208] | ((uint16_t)blk[209] << 8));
        const float *xb = x + b * 256U;
        for (int n = 0; n < 256; n += 128) {
            for (int l = 0; l < 32; l++) {
                int is = l / 16;
                int8_t q1 = (int8_t)((ql[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                int8_t q3 = (int8_t)((ql[l] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                int8_t q4 = (int8_t)((ql[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                sum += d * (float)sc[is] * (float)q1 * xb[l];
                sum += d * (float)sc[is + 2] * (float)q2 * xb[l + 32];
                sum += d * (float)sc[is + 4] * (float)q3 * xb[l + 64];
                sum += d * (float)sc[is + 6] * (float)q4 * xb[l + 96];
            }
            xb += 128; ql += 64; qh += 32; sc += 8;
        }
    }
    return sum;
}

static void qwen_prefill_batch(
    const QwenWxModel *m, const int *toks, int n_tok,
    float *k_cache, float *v_cache, float *last_logits, float sparse_thresh
) {
    float *X = aligned_alloc(64, (size_t)n_tok * HIDDEN * sizeof(float));
    float *XB = aligned_alloc(64, (size_t)n_tok * HIDDEN * sizeof(float));
    float *Q = aligned_alloc(64, (size_t)n_tok * HIDDEN * sizeof(float));
    float *K = aligned_alloc(64, (size_t)n_tok * KV_DIM * sizeof(float));
    float *V = aligned_alloc(64, (size_t)n_tok * KV_DIM * sizeof(float));
    float *ATT = aligned_alloc(64, (size_t)n_tok * HIDDEN * sizeof(float));
    float *UP = aligned_alloc(64, (size_t)n_tok * FFN_DIM * sizeof(float));
    int8_t *Q8 = aligned_alloc(64, (size_t)n_tok * FFN_DIM);
    float *SC = aligned_alloc(64, (size_t)n_tok * sizeof(float));

    for (int p = 0; p < n_tok; p++) {
        float *xp = X + (size_t)p * HIDDEN;
        if (m->tok_embd.ttype == 8)
            dequant_q8_row(xp, m->w_base + m->tok_embd.off + (size_t)toks[p] * (HIDDEN / 32) * 34U, HIDDEN);
        else {
            const uint16_t *e16 = (const uint16_t *)(m->w_base + m->tok_embd.off + (size_t)toks[p] * HIDDEN * 2U);
            for (uint32_t j = 0; j < HIDDEN; j++) xp[j] = f16_to_f32(e16[j]);
        }
    }

    for (int l = 0; l < NUM_LAYERS; l++) {
        const LayerWeights *lw = &m->layers[l];
        for (int p = 0; p < n_tok; p++) {
            rmsnorm_f32(XB + (size_t)p * HIDDEN, X + (size_t)p * HIDDEN, (const float *)(m->w_base + lw->attn_norm.off), HIDDEN);
            SC[p] = quantize_q8(XB + (size_t)p * HIDDEN, Q8 + (size_t)p * HIDDEN, HIDDEN);
        }
        const float *qb = lw->q_bias.off ? (const float *)(m->w_base + lw->q_bias.off) : NULL;
        const float *kb = lw->k_bias.off ? (const float *)(m->w_base + lw->k_bias.off) : NULL;
        const float *vb = lw->v_bias.off ? (const float *)(m->w_base + lw->v_bias.off) : NULL;

        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < HIDDEN; r++) {
            const uint8_t *row = m->w_base + lw->q.off + (size_t)r * (HIDDEN / 32) * 34U;
            float b = qb ? qb[r] : 0.0f;
            for (int p = 0; p < n_tok; p++)
                Q[(size_t)p * HIDDEN + r] = dot_q8_row_avx2(row, HIDDEN / 32, Q8 + (size_t)p * HIDDEN, SC[p]) + b;
        }
        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < KV_DIM; r++) {
            const uint8_t *rk = m->w_base + lw->k.off + (size_t)r * (HIDDEN / 32) * 34U;
            const uint8_t *rv = m->w_base + lw->v.off + (size_t)r * (HIDDEN / 32) * 34U;
            float bk = kb ? kb[r] : 0.0f, bv = vb ? vb[r] : 0.0f;
            for (int p = 0; p < n_tok; p++) {
                K[(size_t)p * KV_DIM + r] = dot_q8_row_avx2(rk, HIDDEN / 32, Q8 + (size_t)p * HIDDEN, SC[p]) + bk;
                V[(size_t)p * KV_DIM + r] = dot_q8_row_avx2(rv, HIDDEN / 32, Q8 + (size_t)p * HIDDEN, SC[p]) + bv;
            }
        }
        for (int p = 0; p < n_tok; p++) {
            apply_rope_qwen(m, Q + (size_t)p * HIDDEN, N_HEADS, (uint32_t)p);
            apply_rope_qwen(m, K + (size_t)p * KV_DIM, N_KV_HEADS, (uint32_t)p);
            size_t kv_off = ((size_t)l * MAX_SEQ + (size_t)p) * KV_DIM;
            memcpy(k_cache + kv_off, K + (size_t)p * KV_DIM, KV_DIM * sizeof(float));
            memcpy(v_cache + kv_off, V + (size_t)p * KV_DIM, KV_DIM * sizeof(float));
        }

        const int group_sz = N_HEADS / N_KV_HEADS;
        const float inv_sqrt_d = 1.0f / sqrtf((float)HEAD_DIM);
        for (int p = 0; p < n_tok; p++) {
            #pragma omp parallel for schedule(static)
            for (int h = 0; h < N_HEADS; h++) {
                int kv_h = h / group_sz;
                const float *qh = Q + (size_t)p * HIDDEN + h * HEAD_DIM;
                float *oh = ATT + (size_t)p * HIDDEN + h * HEAD_DIM, scores[4096], max_s = -1e30f;
                for (int t = 0; t <= p; t++) {
                    const float *kh = k_cache + (((size_t)l * MAX_SEQ + (size_t)t) * KV_DIM) + kv_h * HEAD_DIM;
                    float s = 0.0f;
                    for (uint32_t d = 0; d < HEAD_DIM; d++) s += qh[d] * kh[d];
                    s *= inv_sqrt_d; scores[t] = s;
                    if (s > max_s) max_s = s;
                }
                float sum_exp = 0.0f;
                for (int t = 0; t <= p; t++) { scores[t] = expf(scores[t] - max_s); sum_exp += scores[t]; }
                float inv_sum = 1.0f / sum_exp;
                memset(oh, 0, HEAD_DIM * sizeof(float));
                for (int t = 0; t <= p; t++) {
                    float pr = scores[t] * inv_sum;
                    const float *vh = v_cache + (((size_t)l * MAX_SEQ + (size_t)t) * KV_DIM) + kv_h * HEAD_DIM;
                    for (uint32_t d = 0; d < HEAD_DIM; d++) oh[d] += pr * vh[d];
                }
            }
            SC[p] = quantize_q8(ATT + (size_t)p * HIDDEN, Q8 + (size_t)p * HIDDEN, HIDDEN);
        }

        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < HIDDEN; r++) {
            const uint8_t *ro = m->w_base + lw->o.off + (size_t)r * (HIDDEN / 32) * 34U;
            for (int p = 0; p < n_tok; p++)
                X[(size_t)p * HIDDEN + r] += dot_q8_row_avx2(ro, HIDDEN / 32, Q8 + (size_t)p * HIDDEN, SC[p]);
        }
        for (int p = 0; p < n_tok; p++) {
            rmsnorm_f32(XB + (size_t)p * HIDDEN, X + (size_t)p * HIDDEN, (const float *)(m->w_base + lw->ffn_norm.off), HIDDEN);
            SC[p] = quantize_q8(XB + (size_t)p * HIDDEN, Q8 + (size_t)p * HIDDEN, HIDDEN);
        }
        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < FFN_DIM; r++) {
            const uint8_t *rg = m->w_base + lw->gate.off + (size_t)r * (HIDDEN / 32) * 34U;
            const uint8_t *ru = m->w_base + lw->up.off + (size_t)r * (HIDDEN / 32) * 34U;
            for (int p = 0; p < n_tok; p++) {
                float g = dot_q8_row_avx2(rg, HIDDEN / 32, Q8 + (size_t)p * HIDDEN, SC[p]);
                float silu = g / (1.0f + expf(-g));
                if (fabsf(silu) > sparse_thresh) {
                    float u = dot_q8_row_avx2(ru, HIDDEN / 32, Q8 + (size_t)p * HIDDEN, SC[p]);
                    UP[(size_t)p * FFN_DIM + r] = silu * u;
                } else UP[(size_t)p * FFN_DIM + r] = 0.0f;
            }
        }
        for (int p = 0; p < n_tok; p++)
            SC[p] = quantize_q8(UP + (size_t)p * FFN_DIM, Q8 + (size_t)p * FFN_DIM, FFN_DIM);
        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < HIDDEN; r++) {
            const uint8_t *rd = m->w_base + lw->down.off + (size_t)r * (FFN_DIM / 32) * 34U;
            for (int p = 0; p < n_tok; p++)
                X[(size_t)p * HIDDEN + r] += dot_q8_row_avx2(rd, FFN_DIM / 32, Q8 + (size_t)p * FFN_DIM, SC[p]);
        }
        fprintf(stderr, "\r[PREFILL] Layer %02d/28...", l + 1);
        fflush(stderr);
    }
    fprintf(stderr, "\n");

    float *x_last = X + (size_t)(n_tok - 1) * HIDDEN;
    float xb_last[HIDDEN];
    int8_t q8_last[HIDDEN] __attribute__((aligned(32)));
    rmsnorm_f32(xb_last, x_last, (const float *)(m->w_base + m->out_norm.off), HIDDEN);
    if (m->output.ttype == 8) {
        float sc_o = quantize_q8(xb_last, q8_last, HIDDEN);
        matvec_q8_avx2(last_logits, m->w_base, m->output, q8_last, sc_o, NULL);
    } else {
        const uint8_t *omat = m->w_base + m->output.off;
        size_t rb = (HIDDEN / 256U) * 210U;
        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < m->vocab_size; r++)
            last_logits[r] = dot_q6k_row(omat + (size_t)r * rb, HIDDEN, xb_last);
    }
    free(X); free(XB); free(Q); free(K); free(V); free(ATT); free(UP); free(Q8); free(SC);
}

static void byte_to_qwen_utf8(uint8_t b, char *out, int *olen) {
    uint32_t cp = b;
    if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174))) {
        // Сдвиг управляющих/пробельных байт (0..32, 127..160, 173) в U+0100..U+0143 как в GPT-2/Qwen
        uint32_t n = 0;
        for (uint32_t x = 0; x < b; x++) {
            if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255))) n++;
        }
        cp = 256U + n;
    }
    if (cp < 0x80) { out[0] = (char)cp; *olen = 1; }
    else { out[0] = (char)(0xC0 | (cp >> 6)); out[1] = (char)(0x80 | (cp & 0x3F)); *olen = 2; }
}

static int qwen_utf8_to_byte(const uint8_t *s, int *consumed) {
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

static int encode_raw_bpe(const QwenWxModel *m, const char *raw, int *out_ids, int max_ids) {
    char mapped[4096];
    int mlen = 0;
    for (const uint8_t *p = (const uint8_t *)raw; *p && mlen < 4090; p++) {
        int cl = 0;
        byte_to_qwen_utf8(*p, mapped + mlen, &cl);
        mlen += cl;
    }
    mapped[mlen] = '\0';

    int n = 0, pos = 0;
    while (pos < mlen && n < max_ids) {
        int best_id = -1; size_t best_l = 0;
        for (uint32_t t = 0; t < m->vocab_size && t < 151643; t++) {
            size_t tl = m->vocab_len[t];
            if (tl > best_l && (size_t)(pos + tl) <= (size_t)mlen && memcmp(mapped + pos, m->vocab[t], tl) == 0) {
                best_id = (int)t; best_l = tl;
            }
        }
        if (best_id < 0) pos++; else { out_ids[n++] = best_id; pos += (int)best_l; }
    }
    return n;
}

static int encode_prompt(const QwenWxModel *m, const char *text, int *out_ids, int max_ids) {
    int n = 0;
    out_ids[n++] = 151644; // <|im_start|>
    n += encode_raw_bpe(m, "system\nYou are an experienced Rust and C11 programmer. Write clean, working code examples and explain concisely in Russian.", out_ids + n, max_ids - n - 16);
    out_ids[n++] = 151645; // <|im_end|>
    out_ids[n++] = 198;

    out_ids[n++] = 151644; // <|im_start|>
    n += encode_raw_bpe(m, "user\n", out_ids + n, max_ids - n - 16);
    n += encode_raw_bpe(m, text, out_ids + n, max_ids - n - 16);
    out_ids[n++] = 151645; // <|im_end|>
    out_ids[n++] = 198;
    out_ids[n++] = 151644; // <|im_start|>
    out_ids[n++] = 77091;  // assistant
    out_ids[n++] = 198;
    return n;
}

static void print_token_piece(const char *s) {
    const uint8_t *p = (const uint8_t *)s;
    while (*p) {
        int cons = 1;
        int b = qwen_utf8_to_byte(p, &cons);
        putchar(b);
        p += cons;
    }
    fflush(stdout);
}

static void generate_reply(
    const QwenWxModel *m, int *pos, int max_gen,
    float *k_cache, float *v_cache, float *logits, float sparse_th
) {
    uint64_t t_gen0 = rdtscp_now();
    int gen_count = 0;
    for (int g = 0; g < max_gen && *pos < MAX_SEQ; g++) {
        int next_tok = 0;
        float best_v = logits[0];
        for (uint32_t v = 1; v < m->vocab_size; v++) {
            if (logits[v] > best_v) { best_v = logits[v]; next_tok = (int)v; }
        }
        if (next_tok == 151645 || next_tok == 151643) break;
        print_token_piece(m->vocab[next_tok]);
        gen_count++;
        zero_forward_token_hot(m, next_tok, (*pos)++, k_cache, v_cache, logits, sparse_th);
    }
    uint64_t dt = rdtscp_now() - t_gen0;
    double sec = (double)dt / 3.3e9;
    printf("\n\n[SPEED] %d tokens in %.2f s (%.2f tokens/sec | ctx=%d/%d)\n",
           gen_count, sec, gen_count > 0 ? (double)gen_count / sec : 0.0, *pos, MAX_SEQ);
}

int main(int argc, char **argv) {
    int chat_mode = (argc == 1) || (argc > 1 && strcmp(argv[1], "--chat") == 0);
    const char *prompt = (!chat_mode && argc > 1) ? argv[1] : "Hello! Reply in 5 words:";
    int max_gen = (argc > 2) ? atoi(argv[2]) : (chat_mode ? 512 : 128);
    float sparse_th = (argc > 3) ? strtof(argv[3], NULL) : 0.25f;

    QwenWxModel *m = calloc(1, sizeof(QwenWxModel));
    if (load_qwen_gguf(m, DEFAULT_MODEL) != 0) { fprintf(stderr, "Failed to load GGUF\n"); return 1; }
    printf("[ZERO-LLM] Arch=%s | Layers=%u | Hidden=%u | Vocab=%u | Tok=%s (chat=%d)\n[ZERO-LLM] Auto-HW: %s | GPU=%d/%u (%.1f MiB) | DDR_W^X=%u | Ctx=%u (FreeRAM=%.2f GiB)\n", m->arch, m->n_layers, m->hidden_dim, m->vocab_size, m->tok_model, m->chat_fmt, m->gpu.active ? m->gpu.gpu_name : "CPU-Only", m->gpu.gpu_layers, m->n_layers, (double)m->gpu.vram_used/(1024.0*1024.0), m->n_layers-(uint32_t)m->gpu.gpu_layers, m->hw_plan.max_seq, (double)m->hw_plan.avail_ram_bytes/(1024.0*1024.0*1024.0));

    size_t kv_bytes = m->hw_plan.kv_bytes;
    float *k_cache = aligned_alloc(64, kv_bytes);
    float *v_cache = aligned_alloc(64, kv_bytes);
    float *logits  = aligned_alloc(64, (size_t)m->vocab_size * sizeof(float));
    memset(k_cache, 0, kv_bytes);
    memset(v_cache, 0, kv_bytes);

    omp_set_num_threads(4);
    int prompt_ids[256];

    if (!chat_mode) {
        int n_prompt = encode_prompt(m, prompt, prompt_ids, 256);
        printf("Prompt (%d tokens): %s\n", n_prompt, prompt);
        qwen_prefill_batch(m, prompt_ids, n_prompt, k_cache, v_cache, logits, sparse_th);
        printf("Response: "); fflush(stdout);
        int pos = n_prompt;
        generate_reply(m, &pos, max_gen, k_cache, v_cache, logits, sparse_th);
    } else {
        printf("[CHAT MODE] Model sealed in W^X Vault. Commands: /clear (reset KV), /exit\n");
        char line[1024];
        int pos = 0;
        while (1) {
            printf("\nYou> ");
            fflush(stdout);
            if (!fgets(line, sizeof(line), stdin)) break;
            size_t len = strlen(line);
            while (len > 0 && (line[len - 1] == '\n' || line[len - 1] == '\r')) line[--len] = '\0';
            if (len == 0) continue;
            if (strcmp(line, "/exit") == 0 || strcmp(line, "/quit") == 0) break;
            if (strcmp(line, "/clear") == 0) {
                pos = 0;
                memset(k_cache, 0, kv_bytes);
                memset(v_cache, 0, kv_bytes);
                printf("[KV-CACHE CLEARED]\n");
                continue;
            }
            int n_turn = encode_prompt(m, line, prompt_ids, 256);
            if (pos + n_turn + 8 >= MAX_SEQ) {
                printf("[CTX FULL -> AUTO-CLEAR]\n");
                pos = 0;
                memset(k_cache, 0, kv_bytes);
                memset(v_cache, 0, kv_bytes);
            }
            for (int i = 0; i < n_turn; i++) {
                zero_forward_token_hot(m, prompt_ids[i], pos++, k_cache, v_cache, logits, sparse_th);
            }
            printf("Qwen> "); fflush(stdout);
            generate_reply(m, &pos, max_gen, k_cache, v_cache, logits, sparse_th);
        }
    }

    zero_gpu_free(&m->gpu);
    munmap(m->mmap_ptr, m->mmap_sz);
    free(k_cache); free(v_cache); free(logits); free(m);
    return 0;
}
