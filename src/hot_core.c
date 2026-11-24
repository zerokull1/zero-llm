// ZEROMANI (zerokull1) | zero-llm L1i-Resident Hot Core (<= 7 KB L1i + W^X Arena)
#include "zero_tokamak_hot.h"

__attribute__((noinline)) static void rmsnorm_hot(float *o, const float *x, const float *w, uint32_t n, float eps) {
    float ss = 0.0f; for (uint32_t i = 0; i < n; i++) ss += x[i] * x[i];
    float inv = 1.0f / sqrtf(ss / (float)n + eps);
    for (uint32_t i = 0; i < n; i++) o[i] = x[i] * inv * w[i];
}
__attribute__((noinline)) static void rope_hot(float *vec, uint32_t nh, uint32_t hd, uint32_t pos, float theta) {
    uint32_t half = hd / 2U;
    for (uint32_t i = 0; i < half; i++) {
        float ang = (float)pos / powf(theta, (float)(2U * i) / (float)hd), c = cosf(ang), s = sinf(ang);
        for (uint32_t h = 0; h < nh; h++) { float *hv = vec + h * hd, x0 = hv[i], x1 = hv[i + half]; hv[i] = x0*c - x1*s; hv[i + half] = x0*s + x1*c; }
    }
}
__attribute__((noinline)) static float quant_q8_hot(const float *x, int8_t *q, uint32_t n) {
    float am = 0.0f; for (uint32_t i = 0; i < n; i++) { float a = fabsf(x[i]); if (a > am) am = a; }
    if (am < 1e-12f) { memset(q, 0, n); return 0.0f; }
    float sc = am / 127.0f, inv = 127.0f / am;
    for (uint32_t i = 0; i < n; i++) { int v = (int)lrintf(x[i] * inv); q[i] = (int8_t)(v > 127 ? 127 : (v < -127 ? -127 : v)); }
    return sc;
}
__attribute__((noinline)) static float attn_dot_hot(const float *a, const float *b, uint32_t n) { return zero_dot_f32_l1(a, b, n); }
__attribute__((noinline)) static void attn_axpy_hot(float *o, const float *v, float s, uint32_t n) { zero_axpy_f32_l1(o, v, s, n); }

__attribute__((noinline)) static void gpu_qkv_hot(const ZeroLlmModel *m, uint32_t l, float *q, const float *qb, float *k, const float *kb, float *v, const float *vb, const int8_t *q8, float sc) {
    zero_gpu_qkv_stream((ZeroGpuBackend*)&m->gpu, q, m->gpu_off[l][0], m->hidden_dim, qb, k, m->gpu_off[l][1], m->kv_dim, kb, v, m->gpu_off[l][2], m->kv_dim, vb, m->hidden_dim, q8, sc);
}
__attribute__((noinline)) static void gpu_mv_hot(const ZeroLlmModel *m, float *o, size_t off, uint32_t r, uint32_t c, const int8_t *q8, float sc) {
    zero_gpu_matvec((ZeroGpuBackend*)&m->gpu, o, off, r, c, q8, sc, NULL);
}
__attribute__((noinline)) static void gpu_gu_hot(const ZeroLlmModel *m, uint32_t l, float *gb, float *ub, const int8_t *q8, float sc) {
    zero_gpu_gate_up_stream((ZeroGpuBackend*)&m->gpu, gb, m->gpu_off[l][4], ub, m->gpu_off[l][5], m->ffn_dim, m->hidden_dim, q8, sc);
}

void zero_forward_token_hot(const ZeroLlmModel *m, int token_id, int pos, float *kc, float *vc, float *logits, float sparse_th) {
    const uint32_t H = m->hidden_dim, FFN = m->ffn_dim, KVD = m->kv_dim, HD = m->head_dim, MS = m->hw_plan.max_seq;
    uint8_t *p = m->wx_arena.work_ptr;
    float *x = (float*)p; p += H*4U; float *xb = (float*)p; p += H*4U; float *q = (float*)p; p += H*4U;
    float *att = (float*)p; p += H*4U; float *k = (float*)p; p += KVD*4U; float *v = (float*)p; p += KVD*4U;
    float *gb = (float*)p; p += FFN*4U; float *ub = (float*)p; p += FFN*4U; int8_t *q8 = (int8_t*)p;

    if (m->tok_embd.ttype == 8) {
        const uint8_t *er = m->w_base + m->tok_embd.off + (size_t)token_id * (H / 32U) * 34U;
        for (uint32_t b = 0; b < H / 32U; b++) {
            const uint8_t *blk = er + (size_t)b * 34U; float d = zero_f16_to_f32((uint16_t)blk[0] | ((uint16_t)blk[1] << 8));
            for (uint32_t j = 0; j < 32U; j++) x[b * 32U + j] = d * (float)((const int8_t*)(blk + 2))[j];
        }
    } else {
        const uint16_t *e16 = (const uint16_t*)(m->w_base + m->tok_embd.off + (size_t)token_id * H * 2U);
        for (uint32_t j = 0; j < H; j++) x[j] = zero_f16_to_f32(e16[j]);
    }
    for (uint32_t l = 0; l < m->n_layers; l++) {
        const LayerWeights *lw = &m->layers[l];
        rmsnorm_hot(xb, x, (const float*)(m->w_base + lw->attn_norm.off), H, m->rms_eps); float scx = quant_q8_hot(xb, q8, H);
        const float *qb = lw->q_bias.off ? (const float*)(m->w_base + lw->q_bias.off) : NULL;
        const float *kb = lw->k_bias.off ? (const float*)(m->w_base + lw->k_bias.off) : NULL;
        const float *vb = lw->v_bias.off ? (const float*)(m->w_base + lw->v_bias.off) : NULL;
        if (m->gpu.active && (int)l < m->gpu.gpu_layers)
            gpu_qkv_hot(m, l, q, qb, k, kb, v, vb, q8, scx);
        else { matvec_tokamak_hot(q, m->w_base, lw->q, q8, scx, qb); matvec_tokamak_hot(k, m->w_base, lw->k, q8, scx, kb); matvec_tokamak_hot(v, m->w_base, lw->v, q8, scx, vb); }
        rope_hot(q, m->n_heads, HD, (uint32_t)pos, m->rope_theta); rope_hot(k, m->n_kv_heads, HD, (uint32_t)pos, m->rope_theta);
        size_t kvo = ((size_t)l * MS + (size_t)pos) * KVD; memcpy(kc + kvo, k, KVD * 4U); memcpy(vc + kvo, v, KVD * 4U);
        const uint32_t gsz = m->n_heads / m->n_kv_heads; const float inv_sd = 1.0f / sqrtf((float)HD);
        #pragma omp parallel for schedule(static)
        for (uint32_t h = 0; h < m->n_heads; h++) {
            uint32_t kvh = h / gsz; const float *qh = q + h * HD; float *oh = att + h * HD, sc[4096], ms = -1e30f;
            for (int t = 0; t <= pos; t++) { float s = attn_dot_hot(qh, kc + (((size_t)l * MS + (size_t)t) * KVD) + kvh * HD, HD) * inv_sd; sc[t] = s; if (s > ms) ms = s; }
            float se = 0.0f; for (int t = 0; t <= pos; t++) { sc[t] = expf(sc[t] - ms); se += sc[t]; }
            memset(oh, 0, HD * 4U); float ise = 1.0f / se;
            for (int t = 0; t <= pos; t++) attn_axpy_hot(oh, vc + (((size_t)l * MS + (size_t)t) * KVD) + kvh * HD, sc[t] * ise, HD);
        }
        float sca = quant_q8_hot(att, q8, H);
        if (m->gpu.active && (int)l < m->gpu.gpu_layers) gpu_mv_hot(m, xb, m->gpu_off[l][3], H, H, q8, sca);
        else matvec_tokamak_hot(xb, m->w_base, lw->o, q8, sca, NULL);
        for (uint32_t i = 0; i < H; i++) x[i] += xb[i];
        rmsnorm_hot(xb, x, (const float*)(m->w_base + lw->ffn_norm.off), H, m->rms_eps); float scf = quant_q8_hot(xb, q8, H);
        if (m->gpu.active && (int)l < m->gpu.gpu_layers) {
            gpu_gu_hot(m, l, gb, ub, q8, scf);
            for (uint32_t r = 0; r < FFN; r++) { float sl = gb[r] / (1.0f + expf(-gb[r])); ub[r] = (fabsf(sl) > sparse_th) ? (sl * ub[r]) : 0.0f; }
            float scd = quant_q8_hot(ub, q8, FFN); gpu_mv_hot(m, xb, m->gpu_off[l][6], H, FFN, q8, scd);
        } else {
            matvec_tokamak_hot(gb, m->w_base, lw->gate, q8, scf, NULL); const uint8_t *upb = m->w_base + lw->up.off; size_t urb = (H / 32U) * 34U;
            #pragma omp parallel for schedule(static)
            for (uint32_t r = 0; r < FFN; r++) { float sl = gb[r] / (1.0f + expf(-gb[r])); ub[r] = (fabsf(sl) > sparse_th) ? (sl * dot_q8_row_hot(upb + (size_t)r * urb, H / 32U, q8, scf)) : 0.0f; }
            float scd = quant_q8_hot(ub, q8, FFN); matvec_tokamak_hot(xb, m->w_base, lw->down, q8, scd, NULL);
        }
        for (uint32_t i = 0; i < H; i++) x[i] += xb[i];
    }
    rmsnorm_hot(xb, x, (const float*)(m->w_base + m->out_norm.off), H, m->rms_eps);
    if (m->output.ttype == 8) { float sco = quant_q8_hot(xb, q8, H); matvec_tokamak_hot(logits, m->w_base, m->output, q8, sco, NULL); }
    else { const uint8_t *om = m->w_base + m->output.off; size_t orb = (H / 256U) * 210U;
        #pragma omp parallel for schedule(static)
        for (uint32_t r = 0; r < m->vocab_size; r++) logits[r] = zero_dot_q6k_row_l1(om + (size_t)r * orb, H, xb); }
}
