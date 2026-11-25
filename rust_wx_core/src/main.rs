#![allow(unsafe_op_in_unsafe_fn)]
#![no_std]
#![no_main]

mod sys;
mod gguf;
mod math;
use sys::{CTRL, read_ticks, syscall3, syscall6, verify_wx_sum, w_str, w_u64};
use gguf::{LayerOffs, TensorEntry, build_layer_table, decode_token_clean, encode_greedy_bpe, lookup_offset, parse_gguf_index};
use math::{apply_rope_qwen2, f16_to_f32, fast_exp_neg, quantize_f32_to_q8_0, rms_norm_f32, silu_f32};
use core::arch::{asm, global_asm};
use core::sync::atomic::{Ordering, fence};

#[unsafe(no_mangle)] pub extern "C" fn rust_eh_personality() {}
#[panic_handler] fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

global_asm!(
    ".globl _start", "_start:", "and rsp, -16", "call zero_engine_main", "mov rdi, rax", "mov eax, 60", "syscall",
    ".globl gguf_q8_0_dot_f32_asm", ".globl gguf_q8_0_dot_f32_asm_end", ".p2align 6",
    "gguf_q8_0_dot_f32_asm:",
    "vxorps xmm0, xmm0, xmm0", "vpcmpeqw ymm7, ymm7, ymm7", "vpsrlw ymm7, ymm7, 15",
    "test rdx, rdx", "jz 3f",
    "2:",
    "movzx eax, word ptr [rdi]", "vmovd xmm4, eax", "vcvtph2ps xmm4, xmm4",
    "vmulss xmm4, xmm4, dword ptr [rsi]",
    "vmovdqu ymm2, [rsi + 4]", "vmovdqu ymm3, [rdi + 2]",
    "vpabsb ymm8, ymm3", "vpsignb ymm9, ymm2, ymm3",
    "vpmaddubsw ymm5, ymm8, ymm9", "vpmaddwd ymm5, ymm5, ymm7",
    "vextracti128 xmm6, ymm5, 1", "vpaddd xmm5, xmm5, xmm6",
    "vpshufd xmm6, xmm5, 0xEE", "vpaddd xmm5, xmm5, xmm6",
    "vpshufd xmm6, xmm5, 0x55", "vpaddd xmm5, xmm5, xmm6",
    "vcvtdq2ps xmm5, xmm5", "vfmadd231ss xmm0, xmm4, xmm5",
    "add rdi, 34", "add rsi, 36", "dec rdx", "jnz 2b",
    "3:", "vzeroupper", "ret",
    "gguf_q8_0_dot_f32_asm_end:"
);

unsafe extern "C" {
    fn gguf_q8_0_dot_f32_asm(w: *const u8, x: *const u8, blocks: usize) -> f32;
    fn gguf_q8_0_dot_f32_asm_end();
}

static mut TENSOR_TABLE: [TensorEntry; 384] = [TensorEntry { hash: 0, offset: 0 }; 384];
static mut LAYERS: [LayerOffs; 28] = [LayerOffs { anorm:0, q:0, qb:0, k:0, kb:0, v:0, vb:0, o:0, fnorm:0, gate:0, up:0, down:0 }; 28];
static mut EPOCH_SEQ: u64 = 0;

extern "C" fn worker_loop(tid: usize) -> ! {
    let mut local_ep = 0u64;
    loop {
        let mut ep = unsafe { (*(&raw const CTRL)).epoch.load(Ordering::Acquire) };
        while ep == local_ep {
            if unsafe { (*(&raw const CTRL)).stop.load(Ordering::Acquire) } != 0 {
                unsafe { asm!("mov eax, 60", "xor edi, edi", "syscall", options(noreturn)); }
            }
            core::hint::spin_loop();
            ep = unsafe { (*(&raw const CTRL)).epoch.load(Ordering::Acquire) };
        }
        local_ep = ep;
        let c = unsafe { &*(&raw const CTRL) };
        let kern: extern "C" fn(*const u8, *const u8, usize) -> f32 = unsafe { core::mem::transmute(c.wx_fn) };
        let sc = f32::from_bits(c.scale_bits);
        let row_bytes = c.blocks_per_row * 34;
        let base_row = tid * c.rows_per_core;
        let mut r = 0usize; let mut bid = base_row; let mut blog = -1e30f32;
        while r < c.rows_per_core {
            let v = kern(unsafe { c.w_base.add((base_row + r) * row_bytes) }, c.x_base, c.blocks_per_row) * sc;
            if !c.out_f32.is_null() { unsafe { *c.out_f32.add(base_row + r) = v; } }
            else if v > blog { blog = v; bid = base_row + r; }
            r += 1;
        }
        if c.out_f32.is_null() { unsafe { (*(&raw mut CTRL)).best_id[tid] = bid; (*(&raw mut CTRL)).best_logit[tid] = blog; } }
        c.done_workers.fetch_add(1, Ordering::Release);
    }
}

unsafe fn spawn_worker(stack_top: *mut u8, tid: usize) {
    let sp = (stack_top as usize & !15) - 16;
    *(sp as *mut usize) = tid;
    asm!("mov eax, 56", "mov edi, 0x10F00", "xor edx, edx", "xor r10d, r10d", "xor r8d, r8d", "syscall",
         "test eax, eax", "jnz 2f", "mov rdi, [rsp]", "and rsp, -16", "call {entry}", "2:",
         entry = sym worker_loop, in("rsi") sp, out("rax") _, out("rdi") _, out("rdx") _, out("r10") _, out("r8") _, out("rcx") _, out("r11") _, options(nostack));
}

unsafe fn dispatch_gemv(cores: usize, w: *const u8, x_u8: *const u8, out: *mut f32, rows: usize, blocks: usize, scale: f32) -> usize {
    let rpc = rows / cores;
    let ctrl = &mut *(&raw mut CTRL);
    ctrl.w_base = w; ctrl.x_base = x_u8; ctrl.out_f32 = out;
    ctrl.rows_per_core = rpc; ctrl.blocks_per_row = blocks; ctrl.scale_bits = scale.to_bits();
    ctrl.done_workers.store(0, Ordering::Relaxed);
    let ep = *(&raw const EPOCH_SEQ) + 1;
    *(&raw mut EPOCH_SEQ) = ep;
    ctrl.epoch.store(ep, Ordering::Release);
    let kern: extern "C" fn(*const u8, *const u8, usize) -> f32 = core::mem::transmute(ctrl.wx_fn);
    let row_bytes = blocks * 34;
    let mut r = 0usize; let mut bid = 0usize; let mut blog = -1e30f32;
    while r < rpc {
        let v = kern(w.add(r * row_bytes), x_u8, blocks) * scale;
        if !out.is_null() { *out.add(r) = v; } else if v > blog { blog = v; bid = r; }
        r += 1;
    }
    while ctrl.done_workers.load(Ordering::Acquire) < ((cores - 1) as u32) { core::hint::spin_loop(); }
    if out.is_null() {
        let mut wid = 1usize;
        while wid < cores { if ctrl.best_logit[wid] > blog { blog = ctrl.best_logit[wid]; bid = ctrl.best_id[wid]; } wid += 1; }
    }
    bid
}

unsafe fn forward_token(
    gguf: *const u8, data_base: usize, embd_off: usize, onorm_off: usize,
    lays: &[LayerOffs; 28], cores: usize, work: *mut u8, kv_cache: *mut f32,
    tok: usize, pos: usize, do_logits: bool
) -> usize {
    let x_f32 = work as *mut f32;
    let norm_f32 = work.add(6144) as *mut f32;
    let q_f32 = work.add(12288) as *mut f32;
    let att_out = work.add(18432) as *mut f32;
    let gate_f32 = work.add(24576) as *mut f32;
    let up_f32 = work.add(61440) as *mut f32;
    let x_u8 = work.add(98304);
    let scores = work.add(114688) as *mut f32;

    let tok_row = gguf.add(data_base + embd_off + tok * 48 * 34);
    let mut b = 0usize;
    while b < 48 {
        let bp = tok_row.add(b * 34);
        let sc = f16_to_f32(u16::from_le_bytes([*bp, *bp.add(1)]));
        let mut k = 0usize;
        while k < 32 { *x_f32.add((b << 5) + k) = (*bp.add(2 + k) as i8 as f32) * sc; k += 1; }
        b += 1;
    }

    let mut l = 0usize;
    while l < 28 {
        let lw = &lays[l];
        rms_norm_f32(norm_f32, x_f32, gguf.add(data_base + lw.anorm) as *const f32, 1536);
        quantize_f32_to_q8_0(norm_f32, x_u8, 1536);
        let k_pos = kv_cache.add((l * 1024 + pos) * 512);
        let v_pos = k_pos.add(256);
        dispatch_gemv(cores, gguf.add(data_base + lw.q), x_u8, q_f32, 1536, 48, 1.0f32);
        dispatch_gemv(cores, gguf.add(data_base + lw.k), x_u8, k_pos, 256, 48, 1.0f32);
        dispatch_gemv(cores, gguf.add(data_base + lw.v), x_u8, v_pos, 256, 48, 1.0f32);

        let qb = gguf.add(data_base + lw.qb) as *const f32;
        let kb = gguf.add(data_base + lw.kb) as *const f32;
        let vb = gguf.add(data_base + lw.vb) as *const f32;
        let mut i = 0usize; while i < 1536 { *q_f32.add(i) += *qb.add(i); i += 1; }
        let mut j = 0usize; while j < 256 { *k_pos.add(j) += *kb.add(j); *v_pos.add(j) += *vb.add(j); j += 1; }

        apply_rope_qwen2(q_f32, 12, pos);
        apply_rope_qwen2(k_pos, 2, pos);

        let mut h = 0usize;
        while h < 12 {
            let kvh = h / 6;
            let qh = q_f32.add(h * 128);
            let mut max_s = -1e30f32;
            let mut t = 0usize;
            while t <= pos {
                let kh = kv_cache.add((l * 1024 + t) * 512 + kvh * 128);
                let mut dot = 0.0f32; let mut d = 0usize;
                while d < 128 { dot += (*qh.add(d)) * (*kh.add(d)); d += 1; }
                dot *= 0.08838835f32;
                *scores.add(t) = dot;
                if dot > max_s { max_s = dot; }
                t += 1;
            }
            let mut sum_e = 0.0f32;
            let mut t2 = 0usize;
            while t2 <= pos { let e = fast_exp_neg(*scores.add(t2) - max_s); *scores.add(t2) = e; sum_e += e; t2 += 1; }
            let inv_e = 1.0f32 / (sum_e + 1e-9f32);
            let oh = att_out.add(h * 128);
            let mut d0 = 0usize; while d0 < 128 { *oh.add(d0) = 0.0; d0 += 1; }
            let mut t3 = 0usize;
            while t3 <= pos {
                let w_att = *scores.add(t3) * inv_e;
                let vh = kv_cache.add((l * 1024 + t3) * 512 + 256 + kvh * 128);
                let mut d1 = 0usize; while d1 < 128 { *oh.add(d1) += w_att * (*vh.add(d1)); d1 += 1; }
                t3 += 1;
            }
            h += 1;
        }

        quantize_f32_to_q8_0(att_out, x_u8, 1536);
        dispatch_gemv(cores, gguf.add(data_base + lw.o), x_u8, q_f32, 1536, 48, 1.0f32);
        let mut r1 = 0usize; while r1 < 1536 { *x_f32.add(r1) += *q_f32.add(r1); r1 += 1; }

        rms_norm_f32(norm_f32, x_f32, gguf.add(data_base + lw.fnorm) as *const f32, 1536);
        quantize_f32_to_q8_0(norm_f32, x_u8, 1536);
        dispatch_gemv(cores, gguf.add(data_base + lw.gate), x_u8, gate_f32, 8960, 48, 1.0f32);
        dispatch_gemv(cores, gguf.add(data_base + lw.up), x_u8, up_f32, 8960, 48, 1.0f32);
        let mut g = 0usize; while g < 8960 { *gate_f32.add(g) = silu_f32(*gate_f32.add(g)) * (*up_f32.add(g)); g += 1; }
        quantize_f32_to_q8_0(gate_f32, x_u8, 8960);
        dispatch_gemv(cores, gguf.add(data_base + lw.down), x_u8, q_f32, 1536, 280, 1.0f32);
        let mut r2 = 0usize; while r2 < 1536 { *x_f32.add(r2) += *q_f32.add(r2); r2 += 1; }
        l += 1;
    }

    if !do_logits { return 0; }
    rms_norm_f32(norm_f32, x_f32, gguf.add(data_base + onorm_off) as *const f32, 1536);
    quantize_f32_to_q8_0(norm_f32, x_u8, 1536);
    dispatch_gemv(cores, gguf.add(data_base + embd_off), x_u8, core::ptr::null_mut(), 151936, 48, 1.0f32)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zero_engine_main() -> i32 {
    let mut mask = [0u64; 16];
    let nbytes = syscall3(204, 0, 128, mask.as_mut_ptr() as usize);
    let mut cores = 0usize;
    if nbytes > 0 { for w in 0..((nbytes as usize) / 8) { cores += mask[w].count_ones() as usize; } }
    if cores == 0 { cores = 1; }

    let src = gguf_q8_0_dot_f32_asm as *const () as *const u8;
    let len = (gguf_q8_0_dot_f32_asm_end as *const () as usize) - (src as usize);
    let sum = verify_wx_sum(src, len);
    let arena = syscall6(9, 0, 12288, 0, 0x22, (-1isize) as usize, 0) as *mut u8;
    let wx = arena.add(4096);
    let _ = syscall3(10, wx as usize, 4096, 3);
    let mut c = 0; while c < len { wx.add(c).write_volatile(src.add(c).read()); c += 1; }
    fence(Ordering::SeqCst);
    if verify_wx_sum(wx, len) != sum || syscall3(10, wx as usize, 4096, 5) != 0 { return 99; }

    let fd = syscall6(257, (-100isize) as usize, b"/home/zero/AI_Models/qwen2.5-coder-1.5b-instruct-q8_0.gguf\0".as_ptr() as usize, 0, 0, 0, 0);
    let fsize = syscall3(8, fd as usize, 0, 2) as usize;
    let gguf = syscall6(9, 0, fsize, 1, 0x02, fd as usize, 0) as *const u8;
    let _ = syscall3(3, fd as usize, 0, 0);
    let _ = syscall3(28, gguf as usize, fsize, 14);
    if (gguf as *const u32).read_unaligned() != 0x46554747 { return 6; }

    let tbl = &mut *(&raw mut TENSOR_TABLE);
    let (t_cnt, data_base, tok_arr_off) = parse_gguf_index(gguf, tbl);
    let embd_off = lookup_offset(tbl, t_cnt, b"token_embd.weight");
    let onorm_off = lookup_offset(tbl, t_cnt, b"output_norm.weight");
    let lays = &mut *(&raw mut LAYERS);
    build_layer_table(tbl, t_cnt, lays);

    let work = syscall6(9, 0, 131072, 3, 0x22, (-1isize) as usize, 0) as *mut u8;
    let kv_cache = syscall6(9, 0, 28 * 1024 * 512 * 4, 3, 0x22, (-1isize) as usize, 0) as *mut f32;
    (&mut *(&raw mut CTRL)).wx_fn = wx as usize;
    let stacks = syscall6(9, 0, cores * 16384, 3, 0x22, (-1isize) as usize, 0) as *mut u8;
    let mut t = 1usize; while t < cores { spawn_worker(stacks.add((t + 1) * 16384), t); t += 1; }

    w_str(b"=== ZERO-CPP 0-LIBC MULTI-TURN CHAT (Qwen 2.5 Coder 1.5B Q8_0 | 'exit' to quit) ===\n");
    let mut pos = 0usize;
    loop {
        w_str(b"\nYou: ");
        let mut in_buf = [0u8; 512];
        let n_in = syscall3(0, 0, in_buf.as_mut_ptr() as usize, 500);
        if n_in <= 0 { break; }
        let mut ulen = n_in as usize;
        while ulen > 0 && (in_buf[ulen - 1] == 0x0A || in_buf[ulen - 1] == 0x0D) { ulen -= 1; }
        if ulen == 0 { continue; }
        if (ulen == 4 && &in_buf[..4] == b"exit") || (ulen == 4 && &in_buf[..4] == b"quit") { break; }
        if pos > 880 { pos = 0; }

        let mut prompt_ids = [0usize; 256];
        let mut p_cnt = 0usize;
        if pos > 0 { prompt_ids[p_cnt] = 198; p_cnt += 1; }
        prompt_ids[p_cnt] = 151644; p_cnt += 1;
        prompt_ids[p_cnt] = 872;    p_cnt += 1;
        prompt_ids[p_cnt] = 198;    p_cnt += 1;
        let mut enc_ids = [0usize; 128];
        let n_enc = encode_greedy_bpe(gguf, tok_arr_off, &in_buf[..ulen], &mut enc_ids);
        let mut e = 0usize;
        while e < n_enc && p_cnt < 240 { prompt_ids[p_cnt] = enc_ids[e]; p_cnt += 1; e += 1; }
        prompt_ids[p_cnt] = 151645; p_cnt += 1;
        prompt_ids[p_cnt] = 198;    p_cnt += 1;
        prompt_ids[p_cnt] = 151644; p_cnt += 1;
        prompt_ids[p_cnt] = 77091;  p_cnt += 1;
        prompt_ids[p_cnt] = 198;    p_cnt += 1;

        w_str(b"AI: ");
        let mut pi = 0usize; let mut next_tok = 0usize;
        while pi < p_cnt && pos < 1020 {
            next_tok = forward_token(gguf, data_base, embd_off, onorm_off, lays, cores, work, kv_cache, prompt_ids[pi], pos, pi + 1 == p_cnt);
            pos += 1; pi += 1;
        }

        let t_gen0 = read_ticks();
        let mut g_cnt = 0usize;
        while g_cnt < 256 && pos < 1020 {
            if next_tok == 151645 || next_tok == 151643 { break; }
            let mut tbuf = [0u8; 64];
            let tl = decode_token_clean(gguf, tok_arr_off, next_tok, &mut tbuf);
            w_str(&tbuf[..tl]);
            next_tok = forward_token(gguf, data_base, embd_off, onorm_off, lays, cores, work, kv_cache, next_tok, pos, true);
            pos += 1; g_cnt += 1;
        }
        let t_gen1 = read_ticks();
        w_str(b"\n[TOKENS="); w_u64(g_cnt as u64);
        w_str(b" | CTX_POS="); w_u64(pos as u64);
        w_str(b" | CYCLES="); w_u64(t_gen1.wrapping_sub(t_gen0)); w_str(b"]\n");
    }
    (&mut *(&raw mut CTRL)).stop.store(1, Ordering::Release);


    if syscall3(10, wx as usize, 4096, 3) == 0 {
        let mut z = 0; while z < len { wx.add(z).write_volatile(0); z += 1; }
        fence(Ordering::SeqCst); let _ = syscall3(10, wx as usize, 4096, 0);
    }
    let _ = syscall3(11, kv_cache as usize, 28 * 1024 * 512 * 4, 0);
    let _ = syscall3(11, work as usize, 131072, 0);
    let _ = syscall3(11, gguf as usize, fsize, 0);
    let _ = syscall3(11, arena as usize, 12288, 0);
    0
}
