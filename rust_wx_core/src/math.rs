#[inline(always)]
pub fn fast_rsqrt_f32(x: f32) -> f32 {
    if x <= 0.0 { return 0.0; }
    let half = 0.5f32 * x;
    let mut y = f32::from_bits(0x5f3759dfu32.wrapping_sub(x.to_bits() >> 1));
    y = y * (1.5f32 - half * y * y);
    y = y * (1.5f32 - half * y * y);
    y
}

pub unsafe fn rms_norm_f32(out: *mut f32, x: *const f32, w: *const f32, dim: usize) {
    let mut ss = 0.0f32;
    let mut i = 0usize;
    while i < dim { let v = *x.add(i); ss += v * v; i += 1; }
    let inv = fast_rsqrt_f32(ss / (dim as f32) + 1e-6f32);
    let mut j = 0usize;
    while j < dim { *out.add(j) = (*x.add(j) * inv) * (*w.add(j)); j += 1; }
}

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
        while z < dim { *out_u8.add(z) = 128u8; z += 1; }
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
        *out_u8.add(j) = (qc + 128) as u8;
        j += 1;
    }
    (scale, sum_i8)
}

#[inline(always)]
pub fn silu_f32(x: f32) -> f32 {
    let ax = if x < 0.0 { -x } else { x };
    let sig = 0.5f32 * (1.0f32 + x / (1.0f32 + ax + 0.25f32 * ax * ax));
    x * sig
}
