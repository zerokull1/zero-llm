// ZEROMANI (zerokull1) | zero-llm Auto-Tokenizer & W^X Arena
#ifndef ZERO_LLM_TOK_H
#define ZERO_LLM_TOK_H
#include <stdint.h>
#include <string.h>

static inline void zk_detect_chat_tpl(const char *tpl, int *fmt) {
    if (strstr(tpl, "<|im_start|>")) *fmt = 1;
    else if (strstr(tpl, "<|start_header_id|>")) *fmt = 2;
    else if (strstr(tpl, "<start_of_turn>")) *fmt = 3;
    else if (strstr(tpl, "[INST]")) *fmt = 4;
}

static inline void zk_scan_special_tok(const char *sv, uint32_t sl, int id, int *fmt, int *rs, int *re, int *eos, int *nl) {
    if (!strcmp(sv, "<|im_start|>")) { *rs = id; if (!*fmt) *fmt = 1; }
    else if (!strcmp(sv, "<|im_end|>")) { *re = id; *eos = id; }
    else if (!strcmp(sv, "<|start_header_id|>")) { *rs = id; *fmt = 2; }
    else if (!strcmp(sv, "<|eot_id|>")) { *re = id; *eos = id; }
    else if (!strcmp(sv, "<start_of_turn>")) { *rs = id; *fmt = 3; }
    else if (!strcmp(sv, "<end_of_turn>")) { *re = id; *eos = id; }
    else if (((uint8_t)sv[0] == 0xC4 && (uint8_t)sv[1] == 0x8A && sl == 2) || !strcmp(sv, "\n") || !strcmp(sv, "<0x0A>")) *nl = id;
}

static inline int zk_map_input_bytes(const char *tok_model, const char *raw, char *out, int max_out) {
    int mlen = 0, is_spm = (!strcmp(tok_model, "llama") || !strcmp(tok_model, "spm"));
    for (const uint8_t *p = (const uint8_t *)raw; *p && mlen < max_out - 4; p++) {
        if (is_spm && *p == ' ') { out[mlen++] = (char)0xE2; out[mlen++] = (char)0x96; out[mlen++] = (char)0x81; }
        else if (is_spm) out[mlen++] = (char)*p;
        else {
            uint8_t b = *p; uint32_t cp = b;
            if (!((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174))) {
                uint32_t n = 0;
                for (uint32_t x = 0; x < b; x++) if (!((x >= 33 && x <= 126) || (x >= 161 && x <= 172) || (x >= 174 && x <= 255))) n++;
                cp = 256U + n;
            }
            if (cp < 0x80) out[mlen++] = (char)cp;
            else { out[mlen++] = (char)(0xC0 | (cp >> 6)); out[mlen++] = (char)(0x80 | (cp & 0x3F)); }
        }
    }
    out[mlen] = '\0';
    return mlen;
}
#endif
