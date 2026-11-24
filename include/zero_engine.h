// ZEROMANI (zerokull1) | zero-llm Split L1i-Hot / Cold-Loader Architecture
#ifndef ZERO_LLM_ENGINE_H
#define ZERO_LLM_ENGINE_H

#include <stdint.h>
#include <stddef.h>
#include "auto_hw.h"
#include "zero_accel.h"
#include "zero_gpu_ptx.h"
#include "zero_tok.h"

#define MAX_LAYERS_CAP 128
#define VOCAB_MAX 262144

typedef struct { uint64_t off; uint32_t rows; uint32_t cols; uint32_t ttype; } TensorRef;

typedef struct {
    TensorRef attn_norm, q, q_bias, k, k_bias, v, v_bias, o;
    TensorRef ffn_norm, gate, up, down;
} LayerWeights;

typedef struct {
    char arch[64], tok_model[32];
    uint32_t n_layers, hidden_dim, ffn_dim, n_heads, n_kv_heads, head_dim, kv_dim, model_ctx_cap;
    float rope_theta, rms_eps;
    int chat_fmt, bos_id, eos_id, nl_id, im_start_id, im_end_id, cpu_threads;
    char *vocab[VOCAB_MAX];
    uint32_t vocab_len[VOCAB_MAX];
    uint32_t vocab_size;
    AutoHwPlan hw_plan;
    ZeroTokamakWxArena wx_arena;
    TensorRef tok_embd, out_norm, output;
    LayerWeights layers[MAX_LAYERS_CAP];
    const uint8_t *w_base;
    uint8_t *mmap_ptr;
    size_t mmap_sz;
    ZeroGpuBackend gpu;
    size_t gpu_off[MAX_LAYERS_CAP][7];
} ZeroLlmModel;

int  zero_load_gguf_cold(ZeroLlmModel *m, const char *path);
int  zero_encode_prompt_cold(const ZeroLlmModel *m, const char *text, int *out_ids, int max_ids);
void zero_print_token_cold(const char *s);
void zero_prefill_batch_cold(const ZeroLlmModel *m, const int *toks, int n_tok, float *kc, float *vc, float *logits, float sparse_th);
void zero_forward_token_hot(const ZeroLlmModel *m, int token_id, int pos, float *kc, float *vc, float *logits, float sparse_th);

#endif
