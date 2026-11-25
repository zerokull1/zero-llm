#![allow(unsafe_op_in_unsafe_fn)]
#![no_std]
#![no_main]

mod sys;
mod gguf;
mod math;
mod gpu;
use gpu::GpuCluster;
use sys::{CTRL, read_ticks, syscall3, syscall6, verify_wx_sum, w_str, w_u64};
use gguf::{LayerOffs, ModelCfg, TensorEntry, build_layer_table, decode_token_clean, encode_greedy_bpe, lookup_offset, parse_gguf_index};
use math::{apply_rope_qwen2, f16_to_f32, fast_exp_neg, quantize_f32_to_q8_0, rms_norm_f32, silu_f32};
use core::arch::{asm, global_asm};
use core::sync::atomic::{Ordering, fence};

#[unsafe(no_mangle)] pub extern "C" fn rust_eh_personality() {}
#[panic_handler] fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }

global_asm!(
    ".globl _start", "_start:",
    "mov rdi, [rsp]", "lea rsi, [rsp + 8]", "and rsp, -16",
    "call zero_engine_main", "mov rdi, rax", "mov eax, 60", "syscall",
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

#[inline(never)]
unsafe fn has_avx2_fma_f16c() -> bool {
    let (ml, c1, x0, b7): (u32, u32, u32, u32);
    asm!("push rbx", "mov eax, 0", "cpuid", "pop rbx", out("eax") ml, out("ecx") _, out("edx") _, options(nomem));
    if ml < 7 { return false; }
    asm!("push rbx", "mov eax, 1", "cpuid", "pop rbx", out("eax") _, out("ecx") c1, out("edx") _, options(nomem));
    if (c1 & ((1<<27)|(1<<12)|(1<<29))) != ((1<<27)|(1<<12)|(1<<29)) { return false; }
    asm!("xor ecx, ecx", "xgetbv", out("eax") x0, out("edx") _, out("ecx") _, options(nomem, nostack));
    if (x0 & 6) != 6 { return false; }
    asm!("push rbx", "mov eax, 7", "xor ecx, ecx", "cpuid", "mov esi, ebx", "pop rbx", out("eax") _, out("esi") b7, out("ecx") _, out("edx") _, options(nomem));
    (b7 & (1 << 5)) != 0
}

unsafe extern "C" fn gguf_q8_0_dot_f32_fb(w: *const u8, x: *const u8, blocks: usize) -> f32 {
    let mut acc = 0.0f32; let mut b = 0usize;
    while b < blocks {
        let wp = w.add(b * 34); let xp = x.add(b * 36);
        let sc = f16_to_f32(u16::from_le_bytes([*wp, *wp.add(1)])) * *(xp as *const f32);
        let mut s = 0i32; let mut k = 0usize;
        while k < 32 { s += (*wp.add(2 + k) as i8 as i32) * (*xp.add(4 + k) as i8 as i32); k += 1; }
        acc += sc * (s as f32); b += 1;
    }
    acc
}

static mut TENSOR_TABLE: [TensorEntry; 768] = [TensorEntry { hash: 0, offset: 0 }; 768];
static mut LAYERS: [LayerOffs; 64] = [LayerOffs { anorm:0, q:0, qb:0, k:0, kb:0, v:0, vb:0, o:0, fnorm:0, gate:0, up:0, down:0 }; 64];
static mut EPOCH_SEQ: u64 = 0;

extern "C" fn worker_loop(tid: usize) -> ! {
    let mut local_ep = 0u64;
    loop {
        let mut ep = unsafe { (*(&raw const CTRL)).epoch.load(Ordering::Acquire) };
        while ep == local_ep {
            if unsafe { (*(&raw const CTRL)).stop.load(Ordering::Acquire) } != 0 {
                unsafe { asm!("mov eax, 60", "xor edi, edi", "syscall", options(noreturn)); }
            }
            let mut sp = 0u32; while sp < 64 { core::hint::spin_loop(); sp += 1; }
            unsafe { let _ = syscall3(24, 0, 0, 0); }
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
    let rem = rows % cores;
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
    let tail_start = rpc * cores;
    let mut tr = 0usize;
    while tr < rem {
        let idx = tail_start + tr;
        let v = kern(w.add(idx * row_bytes), x_u8, blocks) * scale;
        if !out.is_null() { *out.add(idx) = v; } else if v > blog { blog = v; bid = idx; }
        tr += 1;
    }
    while ctrl.done_workers.load(Ordering::Acquire) < ((cores - 1) as u32) { core::hint::spin_loop(); }
    if out.is_null() {
        let mut wid = 1usize;
        while wid < cores { if ctrl.best_logit[wid] > blog { blog = ctrl.best_logit[wid]; bid = ctrl.best_id[wid]; } wid += 1; }
    }
    bid
}

unsafe fn forward_token(
    gguf: *const u8, data_base: usize, embd_off: usize, out_w_off: usize, onorm_off: usize,
    lays: &[LayerOffs; 64], cfg: &ModelCfg, gpu: &GpuCluster, cores: usize, work: *mut u8, kv_cache: *mut f32,
    tok: usize, pos: usize, do_logits: bool
) -> usize {
    let (h_dim, f_dim, hb, fb) = (cfg.hidden, cfg.ffn, cfg.hidden >> 5, cfg.ffn >> 5);
    let kv_dim = (h_dim / cfg.n_heads) * cfg.n_kv_heads;
    let gqa_ratio = cfg.n_heads / cfg.n_kv_heads;
    let x_f32 = work as *mut f32;
    let norm_f32 = x_f32.add(h_dim);
    let q_f32 = norm_f32.add(h_dim);
    let att_out = q_f32.add(h_dim);
    let gate_f32 = att_out.add(h_dim);
    let up_f32 = gate_f32.add(f_dim);
    let scores = up_f32.add(f_dim);
    let x_u8 = scores.add(1024) as *mut u8;

    let tok_row = gguf.add(data_base + embd_off + tok * hb * 34);
    let mut b = 0usize;
    while b < hb {
        let bp = tok_row.add(b * 34);
        let sc = f16_to_f32(u16::from_le_bytes([*bp, *bp.add(1)]));
        let mut k = 0usize;
        while k < 32 { *x_f32.add((b << 5) + k) = (*bp.add(2 + k) as i8 as f32) * sc; k += 1; }
        b += 1;
    }

    let mut l = 0usize;
    while l < cfg.n_layers {
        let lw = &lays[l];
        rms_norm_f32(norm_f32, x_f32, gguf.add(data_base + lw.anorm) as *const f32, h_dim);
        quantize_f32_to_q8_0(norm_f32, x_u8, h_dim);
        let k_pos = kv_cache.add((l * 1024 + pos) * (kv_dim * 2));
        let v_pos = k_pos.add(kv_dim);
        let dev_id = gpu.layer_dev(l);
        if dev_id >= 0 {
            let vp = gpu.route(l);
            gpu.proj1(dev_id, x_u8, vp.q, q_f32, h_dim as u32, hb as u32);
            gpu.proj2(dev_id, x_u8, vp.k, vp.v, k_pos, v_pos, kv_dim as u32, hb as u32);
        } else {
            dispatch_gemv(cores, gguf.add(data_base + lw.q), x_u8, q_f32, h_dim, hb, 1.0f32);
            dispatch_gemv(cores, gguf.add(data_base + lw.k), x_u8, k_pos, kv_dim, hb, 1.0f32);
            dispatch_gemv(cores, gguf.add(data_base + lw.v), x_u8, v_pos, kv_dim, hb, 1.0f32);
        }
        if lw.qb != 0 {
            let (qb, kb, vb) = (gguf.add(data_base + lw.qb) as *const f32, gguf.add(data_base + lw.kb) as *const f32, gguf.add(data_base + lw.vb) as *const f32);
            let mut i = 0usize; while i < h_dim { *q_f32.add(i) += *qb.add(i); i += 1; }
            let mut j = 0usize; while j < kv_dim { *k_pos.add(j) += *kb.add(j); *v_pos.add(j) += *vb.add(j); j += 1; }
        }
        apply_rope_qwen2(q_f32, cfg.n_heads, pos);
        apply_rope_qwen2(k_pos, cfg.n_kv_heads, pos);
        let mut h = 0usize;
        while h < cfg.n_heads {
            let (kvh, qh) = (h / gqa_ratio, q_f32.add(h * 128));
            let mut max_s = -1e30f32; let mut t = 0usize;
            while t <= pos {
                let kh = kv_cache.add((l * 1024 + t) * (kv_dim * 2) + kvh * 128);
                let mut dot = 0.0f32; let mut d = 0usize;
                while d < 128 { dot += (*qh.add(d)) * (*kh.add(d)); d += 1; }
                dot *= 0.08838835f32; *scores.add(t) = dot;
                if dot > max_s { max_s = dot; }
                t += 1;
            }
            let mut sum_e = 0.0f32; let mut t2 = 0usize;
            while t2 <= pos { let e = fast_exp_neg(*scores.add(t2) - max_s); *scores.add(t2) = e; sum_e += e; t2 += 1; }
            let inv_e = 1.0f32 / (sum_e + 1e-9f32);
            let oh = att_out.add(h * 128);
            let mut d0 = 0usize; while d0 < 128 { *oh.add(d0) = 0.0; d0 += 1; }
            let mut t3 = 0usize;
            while t3 <= pos {
                let w_att = *scores.add(t3) * inv_e;
                let vh = kv_cache.add((l * 1024 + t3) * (kv_dim * 2) + kv_dim + kvh * 128);
                let mut d1 = 0usize; while d1 < 128 { *oh.add(d1) += w_att * (*vh.add(d1)); d1 += 1; }
                t3 += 1;
            }
            h += 1;
        }
        quantize_f32_to_q8_0(att_out, x_u8, h_dim);
        if dev_id >= 0 { gpu.proj1(dev_id, x_u8, gpu.route(l).o, q_f32, h_dim as u32, hb as u32); }
        else { dispatch_gemv(cores, gguf.add(data_base + lw.o), x_u8, q_f32, h_dim, hb, 1.0f32); }
        let mut r1 = 0usize; while r1 < h_dim { *x_f32.add(r1) += *q_f32.add(r1); r1 += 1; }
        rms_norm_f32(norm_f32, x_f32, gguf.add(data_base + lw.fnorm) as *const f32, h_dim);
        quantize_f32_to_q8_0(norm_f32, x_u8, h_dim);
        if dev_id >= 0 { let vp = gpu.route(l); gpu.proj2(dev_id, x_u8, vp.gate, vp.up, gate_f32, up_f32, f_dim as u32, hb as u32); }
        else { dispatch_gemv(cores, gguf.add(data_base + lw.gate), x_u8, gate_f32, f_dim, hb, 1.0f32); dispatch_gemv(cores, gguf.add(data_base + lw.up), x_u8, up_f32, f_dim, hb, 1.0f32); }
        let mut g = 0usize; while g < f_dim { *gate_f32.add(g) = silu_f32(*gate_f32.add(g)) * (*up_f32.add(g)); g += 1; }
        quantize_f32_to_q8_0(gate_f32, x_u8, f_dim);
        if dev_id >= 0 { gpu.proj1(dev_id, x_u8, gpu.route(l).down, q_f32, h_dim as u32, fb as u32); }
        else { dispatch_gemv(cores, gguf.add(data_base + lw.down), x_u8, q_f32, h_dim, fb, 1.0f32); }
        let mut r2 = 0usize; while r2 < h_dim { *x_f32.add(r2) += *q_f32.add(r2); r2 += 1; }
        l += 1;
    }
    if !do_logits { return 0; }
    rms_norm_f32(norm_f32, x_f32, gguf.add(data_base + onorm_off) as *const f32, h_dim);
    quantize_f32_to_q8_0(norm_f32, x_u8, h_dim);
    dispatch_gemv(cores, gguf.add(data_base + out_w_off), x_u8, core::ptr::null_mut(), cfg.vocab, hb, 1.0f32)
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn zero_engine_main(argc: usize, argv: *const *const u8) -> i32 {
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
    let wx_fn = if has_avx2_fma_f16c() { wx as usize } else { gguf_q8_0_dot_f32_fb as *const () as usize };

    let model_ptr = if argc > 1 && !(*argv.add(1)).is_null() { *argv.add(1) }
                    else { b"/home/zero/AI_Models/qwen2.5-coder-1.5b-instruct-q8_0.gguf\0".as_ptr() };
    let fd = syscall6(257, (-100isize) as usize, model_ptr as usize, 0, 0, 0, 0);
    if fd < 0 { return 2; }
    let fsize = syscall3(8, fd as usize, 0, 2) as usize;
    let gguf = syscall6(9, 0, fsize, 1, 0x02, fd as usize, 0) as *const u8;
    let _ = syscall3(3, fd as usize, 0, 0);
    let _ = syscall3(28, gguf as usize, fsize, 14);
    if (gguf as *const u32).read_unaligned() != 0x46554747 { return 6; }

    let tbl = &mut *(&raw mut TENSOR_TABLE);
    let mut cfg = core::mem::zeroed::<ModelCfg>();
    let (t_cnt, data_base, tok_arr_off) = parse_gguf_index(gguf, tbl, &mut cfg);
    let embd_off = lookup_offset(tbl, t_cnt, b"token_embd.weight");
    let out_w_off = { let o = lookup_offset(tbl, t_cnt, b"output.weight"); if o != 0 { o } else { embd_off } };
    let onorm_off = lookup_offset(tbl, t_cnt, b"output_norm.weight");
    let lays = &mut *(&raw mut LAYERS);
    build_layer_table(tbl, t_cnt, cfg.n_layers, lays);

    let work_bytes = ((cfg.hidden * 4 + cfg.ffn * 2 + 1024) * 4 + (cfg.ffn >> 5) * 36 + 4095) & !4095;
    let kv_dim = (cfg.hidden / cfg.n_heads) * cfg.n_kv_heads;
    let kv_bytes = (cfg.n_layers * 1024 * kv_dim * 2 * 4 + 4095) & !4095;
    let work = syscall6(9, 0, work_bytes, 3, 0x22, (-1isize) as usize, 0) as *mut u8;
    let kv_cache = syscall6(9, 0, kv_bytes, 3, 0x22, (-1isize) as usize, 0) as *mut f32;
    (&mut *(&raw mut CTRL)).wx_fn = wx_fn;
    let stacks = syscall6(9, 0, cores * 16384, 3, 0x22, (-1isize) as usize, 0) as *mut u8;
    let mut t = 1usize; while t < cores { spawn_worker(stacks.add((t + 1) * 16384), t); t += 1; }
    let gpu = GpuCluster::init_cluster_auto_split(gguf, data_base, lays, &cfg);

    w_str(b"[MODEL-HAL] Layers="); w_u64(cfg.n_layers as u64);
    w_str(b" | Hidden="); w_u64(cfg.hidden as u64);
    w_str(b" | FFN="); w_u64(cfg.ffn as u64);
    w_str(b" | Vocab="); w_u64(cfg.vocab as u64);
    w_str(b"\n=== ZERO-CPP 0-LIBC UNIVERSAL CHAT ('exit' to quit) ===\n");

    let sys_prompt = b"You are a concise expert assistant.";
    let mut sys_ids = [0usize; 128];
    let n_sys = encode_greedy_bpe(gguf, tok_arr_off, sys_prompt, &mut sys_ids);
    let mut pos = 0usize;
    let mut sys_seq = [0usize; 140];
    sys_seq[0] = 151644; sys_seq[1] = 8948; sys_seq[2] = 198;
    let mut si = 0usize;
    while si < n_sys && si < 120 { sys_seq[3 + si] = sys_ids[si]; si += 1; }
    sys_seq[3 + si] = 151645; sys_seq[4 + si] = 198;
    let sys_len = 5 + si;
    let mut sp = 0usize;
    while sp < sys_len {
        forward_token(gguf, data_base, embd_off, out_w_off, onorm_off, lays, &cfg, &gpu, cores, work, kv_cache, sys_seq[sp], pos, false);
        pos += 1; sp += 1;
    }
    loop {
        w_str(b"\nYou: ");
        let mut in_buf = [0u8; 512];
        let n_in = syscall3(0, 0, in_buf.as_mut_ptr() as usize, 500);
        if n_in <= 0 { break; }
        let mut ulen = n_in as usize;
        while ulen > 0 && (in_buf[ulen - 1] == 0x0A || in_buf[ulen - 1] == 0x0D) { ulen -= 1; }
        let mut s_i = 0usize;
        while s_i < ulen { if in_buf[s_i] < 0x20 && in_buf[s_i] != 0x09 { in_buf[s_i] = b' '; } s_i += 1; }
        if ulen == 0 { continue; }
        if (ulen == 4 && &in_buf[..4] == b"exit") || (ulen == 4 && &in_buf[..4] == b"quit") { break; }
        if pos > 880 { pos = sys_len; }

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
            next_tok = forward_token(gguf, data_base, embd_off, out_w_off, onorm_off, lays, &cfg, &gpu, cores, work, kv_cache, prompt_ids[pi], pos, pi + 1 == p_cnt);
            pos += 1; pi += 1;
        }
        let t_gen0 = read_ticks();
        let mut g_cnt = 0usize;
        while g_cnt < 256 && pos < 1020 {
            if next_tok == 151645 || next_tok == 151643 { break; }
            let mut tbuf = [0u8; 64];
            let tl = decode_token_clean(gguf, tok_arr_off, next_tok, &mut tbuf);
            w_str(&tbuf[..tl]);
            next_tok = forward_token(gguf, data_base, embd_off, out_w_off, onorm_off, lays, &cfg, &gpu, cores, work, kv_cache, next_tok, pos, true);
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
    let _ = syscall3(11, kv_cache as usize, kv_bytes, 0);
    let _ = syscall3(11, work as usize, work_bytes, 0);
    let _ = syscall3(11, gguf as usize, fsize, 0);
    let _ = syscall3(11, arena as usize, 12288, 0);
    0
}
