// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (c) 2026 zerokull1 (https://github.com)
use core::arch::asm;
use core::sync::atomic::{AtomicU32, AtomicU64};

#[repr(C, align(64))]
pub struct SharedControl {
    pub epoch: AtomicU64, pub done_workers: AtomicU32, pub stop: AtomicU32, pub scale_bits: u32,
    pub wx_fn: usize, pub w_base: *const u8, pub x_base: *const u8, pub out_f32: *mut f32,
    pub rows_per_core: usize, pub blocks_per_row: usize,
    pub best_id: [usize; 8], pub best_logit: [f32; 8],
}

pub static mut CTRL: SharedControl = SharedControl {
    epoch: AtomicU64::new(0), done_workers: AtomicU32::new(0), stop: AtomicU32::new(0), scale_bits: 0,
    wx_fn: 0, w_base: core::ptr::null(), x_base: core::ptr::null(), out_f32: core::ptr::null_mut(),
    rows_per_core: 0, blocks_per_row: 0,
    best_id: [0; 8], best_logit: [-1e30; 8],
};

#[inline(always)]
pub unsafe fn syscall3(n: usize, a1: usize, a2: usize, a3: usize) -> isize {
    let r: isize;
    asm!("syscall", inout("rax") n => r, in("rdi") a1, in("rsi") a2, in("rdx") a3, out("rcx") _, out("r11") _, options(nostack));
    r
}

#[inline(always)]
pub unsafe fn syscall6(n: usize, a1: usize, a2: usize, a3: usize, a4: usize, a5: usize, a6: usize) -> isize {
    let r: isize;
    asm!("syscall", inout("rax") n => r, in("rdi") a1, in("rsi") a2, in("rdx") a3, in("r10") a4, in("r8") a5, in("r9") a6, out("rcx") _, out("r11") _, options(nostack));
    r
}

#[inline(always)]
pub unsafe fn read_ticks() -> u64 {
    let lo: u32; let hi: u32;
    asm!("lfence", "rdtsc", out("eax") lo, out("edx") hi, options(nomem, nostack));
    ((hi as u64) << 32) | (lo as u64)
}

pub unsafe fn verify_wx_sum(p: *const u8, len: usize) -> u64 {
    let mut h: u64 = 0xCBF29CE484222325;
    let mut i = 0;
    while i < len { h ^= p.add(i).read_volatile() as u64; h = h.wrapping_mul(0x100000001B3); i += 1; }
    h
}

pub unsafe fn w_str(s: &[u8]) { let _ = syscall3(1, 1, s.as_ptr() as usize, s.len()); }
pub unsafe fn w_u64(mut v: u64) {
    if v == 0 { w_str(b"0"); return; }
    let mut b = [0u8; 24]; let mut i = 23;
    while v > 0 { b[i] = b'0' + (v % 10) as u8; v /= 10; i -= 1; }
    w_str(&b[i + 1..]);
}

#[allow(suspicious_runtime_symbol_definitions)]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn memset(s: *mut u8, c: i32, n: usize) -> *mut u8 {
    let mut i = 0usize;
    while i < n { s.add(i).write_volatile(c as u8); i += 1; }
    s
}

#[repr(C, align(64))]
pub struct TokamakBus {
    pub magic: u64,          // 0x5A45524F4C4C4D31 ("ZEROLLM1")
    pub req_seq: AtomicU64,
    pub resp_seq: AtomicU64,
    pub rack_epoch: u64,
    pub req_len: u32,
    pub resp_len: u32,
    pub src_rack: u16,
    pub ethertype: u16,      // 0 = local shm pulse, 0x88B5 = inter-rack XDP wire frame
    pub crc64: u64,
    pub stop: AtomicU32,
    pub _pad: [u8; 4],
    pub req_buf: [u8; 4096],
    pub resp_buf: [u8; 16384],
}

#[inline(never)]
pub unsafe fn tokamak_bus_verify(bus: &mut TokamakBus) -> bool {
    let rlen = bus.req_len as usize;
    if rlen == 0 || rlen > 500 { return false; }
    if bus.ethertype == 0x88B5 {
        if bus.rack_epoch <= bus.resp_seq.load(core::sync::atomic::Ordering::Relaxed) { return false; }
        if verify_wx_sum(bus.req_buf.as_ptr(), rlen) != bus.crc64 { return false; }
    }
    true
}

#[used]
#[unsafe(link_section = ".rodata")]
pub static AUTHOR_WATERMARK: [u8; 111] = *b"ZERO-LLM-WATERMARK::AUTHOR=zerokull1::ARCH=TOKAMAK-L1I-WX-0LIBC::ETHER=0x88B5::YEAR=2026::SIG=5A45524F4C4C4D31\0";
