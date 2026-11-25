#[derive(Clone, Copy)]
pub struct TensorEntry { pub hash: u64, pub offset: u64 }

#[derive(Clone, Copy)]
pub struct LayerOffs {
    pub anorm: usize, pub q: usize, pub qb: usize, pub k: usize, pub kb: usize,
    pub v: usize, pub vb: usize, pub o: usize,
    pub fnorm: usize, pub gate: usize, pub up: usize, pub down: usize,
}

pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut i = 0;
    while i < bytes.len() { h ^= bytes[i] as u64; h = h.wrapping_mul(0x100000001b3); i += 1; }
    h
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

pub unsafe fn parse_gguf_index(base: *const u8, table: &mut [TensorEntry; 384]) -> (usize, usize, usize) {
    let n_tensors = (base.add(8) as *const u64).read_unaligned() as usize;
    let n_kv = (base.add(16) as *const u64).read_unaligned() as usize;
    let mut off = 24usize;
    let mut tok_arr_off = 0usize;
    let th = fnv1a_64(b"tokenizer.ggml.tokens");
    let mut i = 0;
    while i < n_kv {
        let kl = (base.add(off) as *const u64).read_unaligned() as usize;
        let kh = fnv1a_64(core::slice::from_raw_parts(base.add(off + 8), kl));
        off += 8 + kl;
        let vt = (base.add(off) as *const u32).read_unaligned();
        if kh == th && vt == 9 { tok_arr_off = off + 4; }
        off = skip_kv_val(base, off + 4, vt);
        i += 1;
    }
    let lim = if n_tensors < 384 { n_tensors } else { 384 };
    let mut t = 0;
    while t < lim {
        let nl = (base.add(off) as *const u64).read_unaligned() as usize;
        table[t].hash = fnv1a_64(core::slice::from_raw_parts(base.add(off + 8), nl));
        off += 8 + nl;
        let ndims = (base.add(off) as *const u32).read_unaligned() as usize;
        off += 4 + ndims * 8 + 4;
        table[t].offset = (base.add(off) as *const u64).read_unaligned();
        off += 8;
        t += 1;
    }
    (lim, (off + 31) & !31usize, tok_arr_off)
}

pub fn lookup_offset(table: &[TensorEntry; 384], count: usize, name: &[u8]) -> usize {
    let h = fnv1a_64(name);
    let mut i = 0;
    while i < count { if table[i].hash == h { return table[i].offset as usize; } i += 1; }
    0
}

pub fn lookup_layer(table: &[TensorEntry; 384], count: usize, layer: usize, sfx: &[u8]) -> usize {
    let mut b = [0u8; 64];
    b[0] = b'b'; b[1] = b'l'; b[2] = b'k'; b[3] = b'.';
    let mut n = 4usize;
    if layer >= 10 { b[n] = b'0' + (layer / 10) as u8; n += 1; }
    b[n] = b'0' + (layer % 10) as u8; n += 1;
    let mut k = 0; while k < sfx.len() { b[n] = sfx[k]; n += 1; k += 1; }
    lookup_offset(table, count, &b[..n])
}

pub fn build_layer_table(t: &[TensorEntry; 384], c: usize, l: &mut [LayerOffs; 28]) {
    let mut i = 0;
    while i < 28 {
        l[i] = LayerOffs {
            anorm: lookup_layer(t, c, i, b".attn_norm.weight"),
            q: lookup_layer(t, c, i, b".attn_q.weight"), qb: lookup_layer(t, c, i, b".attn_q.bias"),
            k: lookup_layer(t, c, i, b".attn_k.weight"), kb: lookup_layer(t, c, i, b".attn_k.bias"),
            v: lookup_layer(t, c, i, b".attn_v.weight"), vb: lookup_layer(t, c, i, b".attn_v.bias"),
            o: lookup_layer(t, c, i, b".attn_output.weight"),
            fnorm: lookup_layer(t, c, i, b".ffn_norm.weight"),
            gate: lookup_layer(t, c, i, b".ffn_gate.weight"),
            up: lookup_layer(t, c, i, b".ffn_up.weight"),
            down: lookup_layer(t, c, i, b".ffn_down.weight"),
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
        } else if s + 1 < sl && b0 == 0xC2 {
            out[d] = *src.add(s + 1);
            d += 1; s += 2;
        } else if s + 1 < sl && b0 == 0xC3 {
            out[d] = *src.add(s + 1) + 0x40;
            d += 1; s += 2;
        } else {
            out[d] = b0; d += 1; s += 1;
        }
    }
    d
}

pub unsafe fn encode_greedy_bpe(base: *const u8, tok_arr_off: usize, text: &[u8], out_ids: &mut [usize; 128]) -> usize {
    let mut norm = [0u8; 256];
    let mut nlen = 0usize;
    let mut i = 0usize;
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
    while pos < nlen && cnt < 120 {
        let mut best_id = 0usize; let mut best_len = 0usize;
        let mut off = tok_arr_off + 12;
        let mut id = 0usize;
        while id < scan_lim {
            let sl = (base.add(off) as *const u64).read_unaligned() as usize;
            if sl > best_len && pos + sl <= nlen {
                let sp = base.add(off + 8);
                let mut m = 0usize;
                while m < sl && *sp.add(m) == norm[pos + m] { m += 1; }
                if m == sl { best_len = sl; best_id = id; }
            }
            off += 8 + sl;
            id += 1;
        }
        if best_len == 0 { pos += 1; } else { out_ids[cnt] = best_id; cnt += 1; pos += best_len; }
    }
    cnt
}
