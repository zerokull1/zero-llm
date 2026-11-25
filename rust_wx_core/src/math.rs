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
