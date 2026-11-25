// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 zerokull1 (https://github.com)
#[inline(always)]
pub fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1F) as u32;
    let m = (h & 0x03FF) as u32;
    let bits = if e == 0 {
        if m == 0 { s } else { s | ((127 - 14) << 23) | (m << 13) }
    } else if e == 31 {
        s | 0x7F80_0000 | (m << 13)
    } else {
        s | ((e + (127 - 15)) << 23) | (m << 13)
    };
    f32::from_bits(bits)
}

#[inline(always)]
pub fn fast_rsqrt_f32(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let half = 0.5f32 * x;
    let mut y = f32::from_bits(0x5f3759dfu32.wrapping_sub(x.to_bits() >> 1));
    y = y * (1.5f32 - half * y * y);
    y = y * (1.5f32 - half * y * y);
    y
}

#[inline(always)]
pub fn fast_exp_neg(x: f32) -> f32 {
    // Быстрая аппроксимация exp(x) для x <= 0 без libm
    if x < -16.0 { return 0.0; }
    let mut v = 1.0f32 + x * 0.00390625f32; // (1 + x/256)^256
    if v < 0.0 { return 0.0; }
    v *= v; v *= v; v *= v; v *= v;
    v *= v; v *= v; v *= v; v *= v;
    v
}

pub unsafe fn rms_norm_f32(out: *mut f32, x: *const f32, w: *const f32, dim: usize) {
    let mut ss = 0.0f32;
    let mut i = 0usize;
    while i < dim { let v = *x.add(i); ss += v * v; i += 1; }
    let inv = fast_rsqrt_f32(ss / (dim as f32) + 1e-6f32);
    let mut j = 0usize;
    while j < dim { *out.add(j) = (*x.add(j) * inv) * (*w.add(j)); j += 1; }
}

#[allow(dead_code)]
pub unsafe fn quantize_f32_to_u8_shifted(x: *const f32, out_u8: *mut u8, dim: usize) -> (f32, i32) {
    let mut amax = 0.0f32;
    let mut i = 0usize;
    while i < dim {
        let v = *x.add(i);
        let a = if v < 0.0 { -v } else { v };
        if a > amax { amax = a; }
        i += 1;
    }
    if amax < 1e-9f32 {
        let mut z = 0usize;
        while z < dim { *out_u8.add(z) = 0u8; z += 1; }
        return (0.0f32, 0);
    }
    let scale = amax / 127.0f32;
    let inv_scale = 127.0f32 / amax;
    let mut sum_i8: i32 = 0;
    let mut j = 0usize;
    while j < dim {
        let scaled = *x.add(j) * inv_scale;
        let q = if scaled >= 0.0 { (scaled + 0.5f32) as i32 } else { (scaled - 0.5f32) as i32 };
        let qc = if q > 127 { 127 } else if q < -127 { -127 } else { q };
        sum_i8 += qc;
        *out_u8.add(j) = (qc as i8) as u8;
        j += 1;
    }
    (scale, sum_i8)
}

#[inline(always)]
pub fn silu_f32(x: f32) -> f32 {
    if x >= 0.0 { x / (1.0f32 + fast_exp_neg(-x)) }
    else { let e = fast_exp_neg(x); (x * e) / (1.0f32 + e) }
}

#[inline(always)]
pub fn sincos_f32(angle: f32) -> (f32, f32) {
    let pi = 3.14159265359f32;
    let two_pi = 6.28318530718f32;
    let mut a = angle - ((angle / two_pi) as i32 as f32) * two_pi;
    if a > pi { a -= two_pi; } else if a < -pi { a += two_pi; }
    let mut sign_c = 1.0f32;
    if a > 1.57079632679f32 { a = pi - a; sign_c = -1.0f32; }
    else if a < -1.57079632679f32 { a = -pi - a; sign_c = -1.0f32; }
    let a2 = a * a;
    let s = a * (1.0f32 - a2 * (0.16666667f32 - a2 * (0.0083333338f32 - a2 * 0.0001984127f32)));
    let c = (1.0f32 - a2 * (0.5f32 - a2 * (0.041666668f32 - a2 * 0.0013888889f32))) * sign_c;
    (s, c)
}

pub unsafe fn apply_rope_qwen2(vec: *mut f32, n_heads: usize, pos: usize) {
    // Qwen 2.5 uses NEOX split-half RoPE: pairs (i, i + 64), theta = 1_000_000.0
    let step = 0.80584218776f32;
    let mut h = 0usize;
    while h < n_heads {
        let hv = vec.add(h * 128);
        let mut freq = 1.0f32;
        let mut i = 0usize;
        while i < 64 {
            let (s, c) = sincos_f32((pos as f32) * freq);
            let x0 = *hv.add(i);
            let x1 = *hv.add(i + 64);
            *hv.add(i) = x0 * c - x1 * s;
            *hv.add(i + 64) = x0 * s + x1 * c;
            freq *= step;
            i += 1;
        }
        h += 1;
    }
}
#[inline(always)]
#[allow(dead_code)]
pub fn f32_to_f16(v: f32) -> u16 {
    let b = v.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xFF) as i32;
    let m = b & 0x007F_FFFF;
    if e == 255 { return s | 0x7C00 | if m != 0 { 1 } else { 0 }; }
    let ne = e - 127 + 15;
    if ne >= 31 { return s | 0x7C00; }
    if ne <= 0 { return s; }
    s | ((ne as u16) << 10) | ((m >> 13) as u16)
}

pub unsafe fn quantize_f32_to_q8_0(x: *const f32, out_q8: *mut u8, dim: usize) {
    let blocks = dim >> 5;
    let mut b = 0usize;
    while b < blocks {
        let xp = x.add(b << 5);
        let op = out_q8.add(b * 36);
        let mut amax = 0.0f32;
        let mut i = 0usize;
        while i < 32 {
            let v = *xp.add(i);
            let a = if v < 0.0 { -v } else { v };
            if a > amax { amax = a; }
            i += 1;
        }
        let sc = amax / 127.0f32;
        let inv = if amax > 1e-20f32 { 127.0f32 / amax } else { 0.0f32 };
        (op as *mut f32).write_unaligned(sc);
        let mut j = 0usize;
        while j < 32 {
            let s = *xp.add(j) * inv;
            let q = if s >= 0.0 { (s + 0.5f32) as i32 } else { (s - 0.5f32) as i32 };
            let qc = if q > 127 { 127 } else if q < -127 { -127 } else { q };
            *op.add(4 + j) = (qc as i8) as u8;
            j += 1;
        }
        b += 1;
    }
}

pub unsafe extern "C" fn gguf_q4_k_dot_f32(w: *const u8, x: *const u8, blocks32: usize) -> f32 {
    let sb_cnt = blocks32 >> 3;
    let mut acc = 0.0f32;
    let mut sb = 0usize;
    while sb < sb_cnt {
        let wp = w.add(sb * 144);
        let d = f16_to_f32(u16::from_le_bytes([*wp, *wp.add(1)]));
        let dmin = f16_to_f32(u16::from_le_bytes([*wp.add(2), *wp.add(3)]));
        let sc_p = wp.add(4);
        let qs = wp.add(16);
        let mut p = 0usize;
        while p < 4 {
            let q_chunk = qs.add(p * 32);
            let mut half = 0usize;
            while half < 2 {
                let j = p * 2 + half;
                let (sc, m) = if j < 4 {
                    ((*sc_p.add(j) & 63) as f32, (*sc_p.add(j + 4) & 63) as f32)
                } else {
                    (((*sc_p.add(j + 4) & 0xF) | ((*sc_p.add(j - 4) >> 6) << 4)) as f32,
                     ((*sc_p.add(j + 4) >> 4) | ((*sc_p.add(j) >> 6) << 4)) as f32)
                };
                let xp = x.add((sb * 8 + j) * 36);
                let xs = *(xp as *const f32);
                let mut sum_qx = 0i32;
                let mut sum_x = 0i32;
                let mut k = 0usize;
                while k < 32 {
                    let qv = if half == 0 { *q_chunk.add(k) & 0xF } else { *q_chunk.add(k) >> 4 } as i32;
                    let xv = *xp.add(4 + k) as i8 as i32;
                    sum_qx += qv * xv;
                    sum_x += xv;
                    k += 1;
                }
                acc += xs * (d * sc * (sum_qx as f32) - dmin * m * (sum_x as f32));
                half += 1;
            }
            p += 1;
        }
        sb += 1;
    }
    acc
}

pub unsafe extern "C" fn gguf_q6_k_dot_f32(w: *const u8, x: *const u8, blocks32: usize) -> f32 {
    let sb_cnt = blocks32 >> 3;
    let mut acc = 0.0f32;
    let mut sb = 0usize;
    while sb < sb_cnt {
        let wp = w.add(sb * 210);
        let d = f16_to_f32(u16::from_le_bytes([*wp.add(208), *wp.add(209)]));
        let mut h_idx = 0usize;
        while h_idx < 2 {
            let ql = wp.add(h_idx * 64);
            let qh = wp.add(128 + h_idx * 32);
            let sc = wp.add(192 + h_idx * 8);
            let xb = x.add((sb * 8 + h_idx * 4) * 36);
            let mut l = 0usize;
            while l < 32 {
                let is = l >> 4;
                let ql0 = *ql.add(l); let ql32 = *ql.add(l + 32); let qhv = *qh.add(l);
                let q1 = (((ql0 & 0xF) | ((qhv & 3) << 4)) as i32) - 32;
                let q2 = (((ql32 & 0xF) | (((qhv >> 2) & 3) << 4)) as i32) - 32;
                let q3 = (((ql0 >> 4) | (((qhv >> 4) & 3) << 4)) as i32) - 32;
                let q4 = (((ql32 >> 4) | (((qhv >> 6) & 3) << 4)) as i32) - 32;
                let (x0, x1, x2, x3) = (xb, xb.add(36), xb.add(72), xb.add(108));
                let s0 = *(x0 as *const f32) * (*x0.add(4 + l) as i8 as f32) * (*sc.add(is) as i8 as f32);
                let s1 = *(x1 as *const f32) * (*x1.add(4 + l) as i8 as f32) * (*sc.add(is + 2) as i8 as f32);
                let s2 = *(x2 as *const f32) * (*x2.add(4 + l) as i8 as f32) * (*sc.add(is + 4) as i8 as f32);
                let s3 = *(x3 as *const f32) * (*x3.add(4 + l) as i8 as f32) * (*sc.add(is + 6) as i8 as f32);
                acc += d * ((q1 as f32) * s0 + (q2 as f32) * s1 + (q3 as f32) * s2 + (q4 as f32) * s3);
                l += 1;
            }
            h_idx += 1;
        }
        sb += 1;
    }
    acc
}

pub unsafe fn dequant_embd_row(embd_base: *const u8, gtype: u32, tok: usize, h_dim: usize, out: *mut f32) {
    if gtype == 14 {
        let sb_per_row = h_dim >> 8;
        let row = embd_base.add(tok * sb_per_row * 210);
        let mut sb = 0usize;
        while sb < sb_per_row {
            let wp = row.add(sb * 210);
            let d = f16_to_f32(u16::from_le_bytes([*wp.add(208), *wp.add(209)]));
            let mut h_idx = 0usize;
            while h_idx < 2 {
                let (ql, qh, sc) = (wp.add(h_idx * 64), wp.add(128 + h_idx * 32), wp.add(192 + h_idx * 8));
                let dst = out.add(sb * 256 + h_idx * 128);
                let mut l = 0usize;
                while l < 32 {
                    let is = l >> 4;
                    let (ql0, ql32, qhv) = (*ql.add(l), *ql.add(l + 32), *qh.add(l));
                    *dst.add(l) = d * (*sc.add(is) as i8 as f32) * ((((ql0 & 0xF) | ((qhv & 3) << 4)) as i32 - 32) as f32);
                    *dst.add(l + 32) = d * (*sc.add(is + 2) as i8 as f32) * ((((ql32 & 0xF) | (((qhv >> 2) & 3) << 4)) as i32 - 32) as f32);
                    *dst.add(l + 64) = d * (*sc.add(is + 4) as i8 as f32) * ((((ql0 >> 4) | (((qhv >> 4) & 3) << 4)) as i32 - 32) as f32);
                    *dst.add(l + 96) = d * (*sc.add(is + 6) as i8 as f32) * ((((ql32 >> 4) | (((qhv >> 6) & 3) << 4)) as i32 - 32) as f32);
                    l += 1;
                }
                h_idx += 1;
            }
            sb += 1;
        }
    } else {
        let hb = h_dim >> 5;
        let row = embd_base.add(tok * hb * 34);
        let mut b = 0usize;
        while b < hb {
            let bp = row.add(b * 34);
            let sc = f16_to_f32(u16::from_le_bytes([*bp, *bp.add(1)]));
            let mut k = 0usize;
            while k < 32 { *out.add((b << 5) + k) = (*bp.add(2 + k) as i8 as f32) * sc; k += 1; }
            b += 1;
        }
    }
}
