#[derive(Clone, Copy)]
pub struct TensorEntry { pub hash: u64, pub offset: u64, pub gtype: u32 }

#[derive(Clone, Copy)]
pub struct LayerOffs {
    pub anorm: usize, pub q: usize, pub qt: u32, pub qb: usize, pub k: usize, pub kt: u32, pub kb: usize,
    pub v: usize, pub vt: u32, pub vb: usize, pub o: usize, pub ot: u32,
    pub fnorm: usize, pub gate: usize, pub gt: u32, pub up: usize, pub ut: u32, pub down: usize, pub dt: u32,
}

#[derive(Clone, Copy)]
pub struct ModelCfg {
    pub n_layers: usize,
    pub hidden: usize,
    pub ffn: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub vocab: usize,
    pub rope_theta: f32,
}

pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut i = 0;
    while i < bytes.len() { h ^= bytes[i] as u64; h = h.wrapping_mul(0x100000001b3); i += 1; }
    h
}

fn ends_with(k: &[u8], sfx: &[u8]) -> bool {
    if k.len() < sfx.len() { return false; }
    let off = k.len() - sfx.len();
    let mut i = 0;
    while i < sfx.len() { if k[off + i] != sfx[i] { return false; } i += 1; }
    true
}

unsafe fn read_kv_u32(p: *const u8, off: usize, vt: u32) -> usize {
    match vt {
        4 | 5 => (p.add(off) as *const u32).read_unaligned() as usize,
        10 | 11 => (p.add(off) as *const u64).read_unaligned() as usize,
        _ => 0,
    }
}

unsafe fn skip_kv_val(p: *const u8, mut off: usize, vtype: u32) -> usize {
    match vtype {
        0 | 1 | 7 => off + 1, 2 | 3 => off + 2, 4 | 5 | 6 => off + 4, 10 | 11 | 12 => off + 8,
        8 => { let l = (p.add(off) as *const u64).read_unaligned() as usize; off + 8 + l }
        9 => {
            let at = (p.add(off) as *const u32).read_unaligned();
            let al = (p.add(off + 4) as *const u64).read_unaligned() as usize;
            off += 12;
            let mut k = 0; while k < al { off = skip_kv_val(p, off, at); k += 1; }
            off
        }
        _ => off,
    }
}

pub unsafe fn parse_gguf_index(base: *const u8, table: &mut [TensorEntry; 768], cfg: &mut ModelCfg) -> (usize, usize, usize) {
    *cfg = ModelCfg { n_layers: 28, hidden: 1536, ffn: 8960, n_heads: 12, n_kv_heads: 2, vocab: 151936, rope_theta: 1_000_000.0 };
    let n_tensors = (base.add(8) as *const u64).read_unaligned() as usize;
    let n_kv = (base.add(16) as *const u64).read_unaligned() as usize;
    let mut off = 24usize;
    let mut tok_arr_off = 0usize;
    let th = fnv1a_64(b"tokenizer.ggml.tokens");
    let mut i = 0;
    while i < n_kv {
        let kl = (base.add(off) as *const u64).read_unaligned() as usize;
        let kslice = core::slice::from_raw_parts(base.add(off + 8), kl);
        let kh = fnv1a_64(kslice);
        off += 8 + kl;
        let vt = (base.add(off) as *const u32).read_unaligned();
        let voff = off + 4;
        if kh == th && vt == 9 {
            tok_arr_off = voff;
            cfg.vocab = (base.add(voff + 4) as *const u64).read_unaligned() as usize;
        } else if ends_with(kslice, b".block_count") { cfg.n_layers = read_kv_u32(base, voff, vt); }
        else if ends_with(kslice, b".embedding_length") { cfg.hidden = read_kv_u32(base, voff, vt); }
        else if ends_with(kslice, b".feed_forward_length") { cfg.ffn = read_kv_u32(base, voff, vt); }
        else if ends_with(kslice, b".attention.head_count") { cfg.n_heads = read_kv_u32(base, voff, vt); }
        else if ends_with(kslice, b".attention.head_count_kv") { cfg.n_kv_heads = read_kv_u32(base, voff, vt); }
        else if ends_with(kslice, b".rope.freq_base") && vt == 6 {
            cfg.rope_theta = f32::from_bits((base.add(voff) as *const u32).read_unaligned());
        }
        off = skip_kv_val(base, voff, vt);
        i += 1;
    }
    let lim = if n_tensors < 768 { n_tensors } else { 768 };
    let mut t = 0;
    while t < lim {
        let nl = (base.add(off) as *const u64).read_unaligned() as usize;
        table[t].hash = fnv1a_64(core::slice::from_raw_parts(base.add(off + 8), nl));
        off += 8 + nl;
        let ndims = (base.add(off) as *const u32).read_unaligned() as usize;
        off += 4 + ndims * 8;
        table[t].gtype = (base.add(off) as *const u32).read_unaligned();
        table[t].offset = (base.add(off + 4) as *const u64).read_unaligned();
        off += 12;
        t += 1;
    }
    (lim, (off + 31) & !31usize, tok_arr_off)
}

pub fn lookup_tensor(table: &[TensorEntry; 768], count: usize, name: &[u8]) -> (usize, u32) {
    let h = fnv1a_64(name);
    let mut i = 0;
    while i < count { if table[i].hash == h { return (table[i].offset as usize, table[i].gtype); } i += 1; }
    (0, 0)
}
pub fn lookup_offset(table: &[TensorEntry; 768], count: usize, name: &[u8]) -> usize { lookup_tensor(table, count, name).0 }

pub fn lookup_layer_t(table: &[TensorEntry; 768], count: usize, layer: usize, sfx: &[u8]) -> (usize, u32) {
    let mut b = [0u8; 64];
    b[0] = b'b'; b[1] = b'l'; b[2] = b'k'; b[3] = b'.';
    let mut n = 4usize;
    if layer >= 10 { b[n] = b'0' + (layer / 10) as u8; n += 1; }
    b[n] = b'0' + (layer % 10) as u8; n += 1;
    let mut k = 0; while k < sfx.len() { b[n] = sfx[k]; n += 1; k += 1; }
    lookup_tensor(table, count, &b[..n])
}
pub fn lookup_layer(table: &[TensorEntry; 768], count: usize, layer: usize, sfx: &[u8]) -> usize { lookup_layer_t(table, count, layer, sfx).0 }

pub fn build_layer_table(t: &[TensorEntry; 768], c: usize, n_layers: usize, l: &mut [LayerOffs; 64]) {
    let mut i = 0;
    while i < n_layers && i < 64 {
        let (q, qt) = lookup_layer_t(t, c, i, b".attn_q.weight");
        let (k, kt) = lookup_layer_t(t, c, i, b".attn_k.weight");
        let (v, vt) = lookup_layer_t(t, c, i, b".attn_v.weight");
        let (o, ot) = lookup_layer_t(t, c, i, b".attn_output.weight");
        let (gate, gt) = lookup_layer_t(t, c, i, b".ffn_gate.weight");
        let (up, ut) = lookup_layer_t(t, c, i, b".ffn_up.weight");
        let (down, dt) = lookup_layer_t(t, c, i, b".ffn_down.weight");
        l[i] = LayerOffs {
            anorm: lookup_layer(t, c, i, b".attn_norm.weight"),
            q, qt, qb: lookup_layer(t, c, i, b".attn_q.bias"),
            k, kt, kb: lookup_layer(t, c, i, b".attn_k.bias"),
            v, vt, vb: lookup_layer(t, c, i, b".attn_v.bias"),
            o, ot, fnorm: lookup_layer(t, c, i, b".ffn_norm.weight"),
            gate, gt, up, ut, down, dt,
        };
        i += 1;
    }
}

pub unsafe fn decode_token_clean(base: *const u8, tok_arr_off: usize, target_id: usize, out: &mut [u8; 64]) -> usize {
    if tok_arr_off == 0 { return 0; }
    let al = (base.add(tok_arr_off + 4) as *const u64).read_unaligned() as usize;
    if target_id >= al { return 0; }
    let mut off = tok_arr_off + 12;
    let mut k = 0;
    while k < target_id { off += 8 + (base.add(off) as *const u64).read_unaligned() as usize; k += 1; }
    let sl = (base.add(off) as *const u64).read_unaligned() as usize;
    let src = base.add(off + 8);
    let mut s = 0usize; let mut d = 0usize;
    while s < sl && d < 62 {
        let b0 = *src.add(s);
        if s + 1 < sl && (b0 == 0xC4 || b0 == 0xC5) {
            let b1 = *src.add(s + 1);
            let cp = (((b0 & 0x1F) as u32) << 6) | ((b1 & 0x3F) as u32);
            if cp == 0x120 { out[d] = b' '; d += 1; }
            else if cp == 0x10A { out[d] = 0x0A; d += 1; }
            else if cp >= 0x121 && cp <= 0x142 { out[d] = (cp - 0xA2) as u8; d += 1; }
            else if cp == 0x143 { out[d] = 0xAD; d += 1; }
            s += 2;
        } else if s + 1 < sl && b0 == 0xC2 { out[d] = *src.add(s + 1); d += 1; s += 2; }
        else if s + 1 < sl && b0 == 0xC3 { out[d] = *src.add(s + 1) + 0x40; d += 1; s += 2; }
        else { out[d] = b0; d += 1; s += 1; }
    }
    d
}

pub unsafe fn encode_greedy_bpe(base: *const u8, tok_arr_off: usize, text: &[u8], out_ids: &mut [usize; 128]) -> usize {
    let mut norm = [0u8; 256]; let mut nlen = 0usize; let mut i = 0usize;
    while i < text.len() && nlen < 250 {
        let b = text[i];
        if b == b' ' { norm[nlen] = 0xC4; norm[nlen + 1] = 0xA0; nlen += 2; }
        else if b == 0x0A { norm[nlen] = 0xC4; norm[nlen + 1] = 0x8A; nlen += 2; }
        else if b >= 0x7F && b <= 0xA0 {
            let cp = (b as u32) + 0xA2;
            norm[nlen] = (0xC0 | (cp >> 6)) as u8; norm[nlen + 1] = (0x80 | (cp & 0x3F)) as u8; nlen += 2;
        } else if b == 0xAD { norm[nlen] = 0xC5; norm[nlen + 1] = 0x83; nlen += 2; }
        else if b >= 0xA1 && b < 0xC0 { norm[nlen] = 0xC2; norm[nlen + 1] = b; nlen += 2; }
        else if b >= 0xC0 { norm[nlen] = 0xC3; norm[nlen + 1] = b - 0x40; nlen += 2; }
        else if b != 0x0D { norm[nlen] = b; nlen += 1; }
        i += 1;
    }
    let al = (base.add(tok_arr_off + 4) as *const u64).read_unaligned() as usize;
    let scan_lim = if al < 151643 { al } else { 151643 };
    let mut pos = 0usize; let mut cnt = 0usize;
    while pos < nlen && cnt < 128 {
        let mut best_id = 0usize; let mut best_len = 0usize;
        let mut off = tok_arr_off + 12; let mut tid = 0usize;
        while tid < scan_lim {
            let sl = (base.add(off) as *const u64).read_unaligned() as usize;
            if sl > best_len && pos + sl <= nlen {
                let sp = base.add(off + 8); let mut m = 0usize;
                while m < sl && *sp.add(m) == norm[pos + m] { m += 1; }
                if m == sl { best_len = sl; best_id = tid; }
            }
            off += 8 + sl; tid += 1;
        }
        if best_len == 0 { pos += 1; } else { out_ids[cnt] = best_id; cnt += 1; pos += best_len; }
    }
    cnt
}
