#[derive(Clone, Copy)]
pub struct TensorEntry { pub hash: u64, pub offset: u64 }

pub fn fnv1a_64(bytes: &[u8]) -> u64 {
    let mut h = 0xcbf29ce484222325u64;
    let mut i = 0;
    while i < bytes.len() { h ^= bytes[i] as u64; h = h.wrapping_mul(0x100000001b3); i += 1; }
    h
}

unsafe fn skip_kv_val(p: *const u8, mut off: usize, vtype: u32) -> usize {
    match vtype {
        0 | 1 | 7 => off + 1,
        2 | 3 => off + 2,
        4 | 5 | 6 => off + 4,
        10 | 11 | 12 => off + 8,
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

pub unsafe fn parse_gguf_index(base: *const u8, table: &mut [TensorEntry; 384]) -> (usize, usize) {
    let n_tensors = (base.add(8) as *const u64).read_unaligned() as usize;
    let n_kv = (base.add(16) as *const u64).read_unaligned() as usize;
    let mut off = 24usize;
    let mut i = 0;
    while i < n_kv {
        let kl = (base.add(off) as *const u64).read_unaligned() as usize;
        off += 8 + kl;
        let vt = (base.add(off) as *const u32).read_unaligned();
        off = skip_kv_val(base, off + 4, vt);
        i += 1;
    }
    let lim = if n_tensors < 384 { n_tensors } else { 384 };
    let mut t = 0;
    while t < lim {
        let nl = (base.add(off) as *const u64).read_unaligned() as usize;
        let name_slice = core::slice::from_raw_parts(base.add(off + 8), nl);
        table[t].hash = fnv1a_64(name_slice);
        off += 8 + nl;
        let ndims = (base.add(off) as *const u32).read_unaligned() as usize;
        off += 4 + ndims * 8 + 4;
        table[t].offset = (base.add(off) as *const u64).read_unaligned();
        off += 8;
        t += 1;
    }
    let data_base = (off + 31) & !31usize;
    (lim, data_base)
}

pub fn lookup_offset(table: &[TensorEntry; 384], count: usize, name: &[u8]) -> Option<usize> {
    let h = fnv1a_64(name);
    let mut i = 0;
    while i < count { if table[i].hash == h { return Some(table[i].offset as usize); } i += 1; }
    None
}

pub fn lookup_layer_offset(table: &[TensorEntry; 384], count: usize, layer: usize, suffix: &[u8]) -> Option<usize> {
    let mut buf = [0u8; 64];
    let pfx = b"blk.";
    let mut n = 0usize;
    while n < pfx.len() { buf[n] = pfx[n]; n += 1; }
    if layer >= 10 {
        buf[n] = b'0' + ((layer / 10) as u8); n += 1;
        buf[n] = b'0' + ((layer % 10) as u8); n += 1;
    } else {
        buf[n] = b'0' + (layer as u8); n += 1;
    }
    let mut k = 0usize;
    while k < suffix.len() && n < 64 { buf[n] = suffix[k]; n += 1; k += 1; }
    lookup_offset(table, count, &buf[..n])
}

pub unsafe fn decode_token_str(base: *const u8, target_id: usize, out: &mut [u8; 64]) -> usize {
    let n_kv = (base.add(16) as *const u64).read_unaligned() as usize;
    let mut off = 24usize;
    let target_h = fnv1a_64(b"tokenizer.ggml.tokens");
    let mut i = 0usize;
    while i < n_kv {
        let kl = (base.add(off) as *const u64).read_unaligned() as usize;
        let kh = fnv1a_64(core::slice::from_raw_parts(base.add(off + 8), kl));
        off += 8 + kl;
        let vt = (base.add(off) as *const u32).read_unaligned();
        off += 4;
        if kh == target_h && vt == 9 {
            let al = (base.add(off + 4) as *const u64).read_unaligned() as usize;
            off += 12;
            let lim = if target_id < al { target_id } else { return 0 };
            let mut k = 0usize;
            while k < lim {
                let sl = (base.add(off) as *const u64).read_unaligned() as usize;
                off += 8 + sl;
                k += 1;
            }
            let sl = (base.add(off) as *const u64).read_unaligned() as usize;
            let cp = if sl < 64 { sl } else { 64 };
            let mut c = 0usize;
            while c < cp { out[c] = *base.add(off + 8 + c); c += 1; }
            return cp;
        }
        off = skip_kv_val(base, off, vt);
        i += 1;
    }
    0
}

#[derive(Clone, Copy)]
pub struct LayerOffs {
    pub anorm: usize, pub q: usize, pub k: usize, pub v: usize, pub o: usize,
    pub fnorm: usize, pub gate: usize, pub up: usize, pub down: usize,
}

pub fn build_layer_table(tbl: &[TensorEntry; 384], cnt: usize, layers: &mut [LayerOffs; 28]) {
    let mut l = 0usize;
    while l < 28 {
        layers[l] = LayerOffs {
            anorm: lookup_layer_offset(tbl, cnt, l, b".attn_norm.weight").unwrap_or(0),
            q: lookup_layer_offset(tbl, cnt, l, b".attn_q.weight").unwrap_or(0),
            k: lookup_layer_offset(tbl, cnt, l, b".attn_k.weight").unwrap_or(0),
            v: lookup_layer_offset(tbl, cnt, l, b".attn_v.weight").unwrap_or(0),
            o: lookup_layer_offset(tbl, cnt, l, b".attn_output.weight").unwrap_or(0),
            fnorm: lookup_layer_offset(tbl, cnt, l, b".ffn_norm.weight").unwrap_or(0),
            gate: lookup_layer_offset(tbl, cnt, l, b".ffn_gate.weight").unwrap_or(0),
            up: lookup_layer_offset(tbl, cnt, l, b".ffn_up.weight").unwrap_or(0),
            down: lookup_layer_offset(tbl, cnt, l, b".ffn_down.weight").unwrap_or(0),
        };
        l += 1;
    }
}
