// ZEROMANI (zerokull1) | zero-llm Universal Multilingual GGUF BPE/SPM Tokenizer (O(1) Hash + Merges)
#ifndef ZERO_LLM_BPE_GGUF_H
#define ZERO_LLM_BPE_GGUF_H
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define ZK_HASH_CAP 524288U

typedef struct {
    int32_t *tok_ht;     // хэш-таблица строка -> token_id
    int32_t *merge_ht;   // хэш-таблица "tokA tokB" -> ранг слияния
    float   *scores;     // SPM scores (если есть)
    int      has_merges;
} ZkVocabIndex;

static inline uint32_t zk_fnv1a(const void *data, size_t len) {
    const uint8_t *p = (const uint8_t *)data;
    uint32_t h = 2166136261U;
    for (size_t i = 0; i < len; i++) { h ^= p[i]; h *= 16777619U; }
    return h;
}

static inline void zk_ht_insert(int32_t *ht, const char *s, size_t len, int32_t val, char **keys, const uint32_t *klens) {
    uint32_t slot = zk_fnv1a(s, len) & (ZK_HASH_CAP - 1U);
    while (ht[slot] >= 0) {
        int32_t cur = ht[slot];
        if (klens[cur] == len && memcmp(keys[cur], s, len) == 0) return;
        slot = (slot + 1U) & (ZK_HASH_CAP - 1U);
    }
    ht[slot] = val;
}

static inline int32_t zk_ht_find(const int32_t *ht, const char *s, size_t len, char **keys, const uint32_t *klens) {
    if (!ht) return -1;
    uint32_t slot = zk_fnv1a(s, len) & (ZK_HASH_CAP - 1U);
    while (ht[slot] >= 0) {
        int32_t cur = ht[slot];
        if (klens[cur] == len && memcmp(keys[cur], s, len) == 0) return cur;
        slot = (slot + 1U) & (ZK_HASH_CAP - 1U);
    }
    return -1;
}
#endif

static inline int zk_utf8_char_len(uint8_t c) {
    if (c < 0x80) return 1;
    if ((c & 0xE0) == 0xC0) return 2;
    if ((c & 0xF0) == 0xE0) return 3;
    if ((c & 0xF8) == 0xF0) return 4;
    return 1;
}

static inline int zk_bpe_encode_indexed(
    const ZkVocabIndex *idx, char **vocab, const uint32_t *vlen,
    char **merges, const uint32_t *mlen_arr,
    const char *mapped, int mlen, int *out_ids, int max_ids
) {
    int off[4096], len[4096], ns = 0, pos = 0;
    while (pos < mlen && ns < 4096) {
        int cl = zk_utf8_char_len((uint8_t)mapped[pos]);
        if (pos + cl > mlen) cl = mlen - pos;
        off[ns] = pos; len[ns] = cl; ns++; pos += cl;
    }
    char pair_buf[512];
    while (ns > 1) {
        int best_i = -1, best_rank = 0x7FFFFFFF;
        for (int i = 0; i < ns - 1; i++) {
            int l1 = len[i], l2 = len[i + 1];
            if (l1 + l2 + 1 >= (int)sizeof(pair_buf)) continue;
            if (idx->has_merges && idx->merge_ht) {
                memcpy(pair_buf, mapped + off[i], (size_t)l1);
                pair_buf[l1] = ' ';
                memcpy(pair_buf + l1 + 1, mapped + off[i + 1], (size_t)l2);
                int r = zk_ht_find(idx->merge_ht, pair_buf, (size_t)(l1 + l2 + 1), merges, mlen_arr);
                if (r >= 0 && r < best_rank) { best_rank = r; best_i = i; }
            } else {
                int id = zk_ht_find(idx->tok_ht, mapped + off[i], (size_t)(l1 + l2), vocab, vlen);
                if (id >= 0 && id < best_rank) { best_rank = id; best_i = i; }
            }
        }
        if (best_i < 0) break;
        len[best_i] += len[best_i + 1];
        for (int j = best_i + 1; j < ns - 1; j++) { off[j] = off[j + 1]; len[j] = len[j + 1]; }
        ns--;
    }
    int n_out = 0;
    for (int i = 0; i < ns && n_out < max_ids; i++) {
        int id = zk_ht_find(idx->tok_ht, mapped + off[i], (size_t)len[i], vocab, vlen);
        if (id >= 0) out_ids[n_out++] = id;
    }
    return n_out;
}
