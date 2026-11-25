use crate::gguf::LayerOffs;
use crate::sys::{syscall6, w_str, w_u64};
use core::ffi::c_void;

#[link(name = "c")]
unsafe extern "C" {
    fn dlopen(filename: *const u8, flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void, symbol: *const u8) -> *mut c_void;
}

pub const MAX_GPUS: usize = 1024;

#[derive(Clone, Copy)]
pub struct LayerGpuRoute {
    pub dev_id: i32,
    pub q: u64, pub k: u64, pub v: u64, pub o: u64,
    pub gate: u64, pub up: u64, pub down: u64,
}

#[derive(Clone, Copy)]
pub struct GpuDeviceSlot {
    pub ctx: *mut c_void,
    pub kern_fn: *mut c_void,
    pub d_x_q8: u64,
    pub d_out_f32: u64,
}

#[allow(dead_code)]
pub struct GpuCluster {
    pub num_gpus: usize,
    pub gpu_layers: usize,
    pub total_layers: usize,
    pub total_vram_used_mb: usize,
    pub devs: *mut GpuDeviceSlot,
    pub routes: *mut LayerGpuRoute,
    cu_ctx_set: unsafe extern "C" fn(*mut c_void) -> i32,
    cu_memcpy_htod: unsafe extern "C" fn(u64, *const c_void, usize) -> i32,
    cu_memcpy_dtoh: unsafe extern "C" fn(*mut c_void, u64, usize) -> i32,
    cu_launch_kernel: unsafe extern "C" fn(*mut c_void, u32, u32, u32, u32, u32, u32, u32, *mut c_void, *mut *mut c_void, *mut *mut c_void) -> i32,
}

unsafe fn sym_v2(lib: *mut c_void, v2: &[u8], v1: &[u8]) -> *mut c_void {
    let p = dlsym(lib, v2.as_ptr());
    if !p.is_null() { p } else { dlsym(lib, v1.as_ptr()) }
}

const PTX_HEAD: &[u8] = b".version 6.0\n.target sm_61\n.address_size 64\n\
.visible .entry q8_gemv_dp4a(.param .u64 p_w,.param .u64 p_x,.param .u64 p_out,.param .u32 p_rows,.param .u32 p_blocks){\n\
.reg .pred %p<3>;.reg .b32 %r<12>;.reg .b64 %rd<12>;.reg .f32 %f<6>;.reg .b16 %h<4>;\n\
mov.u32 %r1,%ctaid.x;mov.u32 %r3,%tid.x;ld.param.u32 %r4,[p_rows];\n\
setp.ge.u32 %p1,%r1,%r4;@%p1 bra DONE;ld.param.u64 %rd1,[p_w];ld.param.u64 %rd2,[p_x];\n\
ld.param.u64 %rd3,[p_out];ld.param.u32 %r5,[p_blocks];mul.wide.u32 %rd4,%r5,34;\n\
cvt.u64.u32 %rd5,%r1;mul.lo.u64 %rd6,%rd5,%rd4;add.u64 %rd1,%rd1,%rd6;mov.f32 %f1,0f00000000;mov.u32 %r6,%r3;\n\
LOOP_B:setp.ge.u32 %p2,%r6,%r5;@%p2 bra END_B;mul.wide.u32 %rd7,%r6,34;add.u64 %rd8,%rd1,%rd7;\n\
mul.wide.u32 %rd9,%r6,36;add.u64 %rd10,%rd2,%rd9;ld.global.u16 %h1,[%rd8];cvt.f32.f16 %f2,%h1;\n\
ld.global.f32 %f3,[%rd10];mul.f32 %f2,%f2,%f3;mov.u32 %r7,0;mov.u32 %r11,0;\n\
LOOP_K:setp.ge.u32 %p2,%r11,8;@%p2 bra END_K;mul.wide.u32 %rd7,%r11,4;add.u64 %rd9,%rd8,%rd7;\n\
ld.global.u16 %h2,[%rd9+2];ld.global.u16 %h3,[%rd9+4];cvt.u32.u16 %r8,%h2;cvt.u32.u16 %r10,%h3;bfi.b32 %r8,%r10,%r8,16,16;\n\
add.u64 %rd9,%rd10,%rd7;ld.global.u32 %r9,[%rd9+4];dp4a.s32.s32 %r7,%r8,%r9,%r7;add.u32 %r11,%r11,1;bra LOOP_K;\n\
END_K:cvt.rn.f32.s32 %f4,%r7;fma.rn.f32 %f1,%f2,%f4,%f1;add.u32 %r6,%r6,32;bra LOOP_B;\n\
END_B:shfl.sync.down.b32 %f5,%f1,16,31,0xffffffff;add.f32 %f1,%f1,%f5;\n\
shfl.sync.down.b32 %f5,%f1,8,31,0xffffffff;add.f32 %f1,%f1,%f5;\n\
shfl.sync.down.b32 %f5,%f1,4,31,0xffffffff;add.f32 %f1,%f1,%f5;\n\
shfl.sync.down.b32 %f5,%f1,2,31,0xffffffff;add.f32 %f1,%f1,%f5;\n\
shfl.sync.down.b32 %f5,%f1,1,31,0xffffffff;add.f32 %f1,%f1,%f5;\n\
setp.eq.u32 %p1,%r3,0;@!%p1 bra DONE;mul.wide.u32 %rd11,%r1,4;add.u64 %rd3,%rd3,%rd11;st.global.f32 [%rd3],%f1;DONE:ret;}\0";

impl GpuCluster {
    pub unsafe fn init_cluster_auto_split(gguf: *const u8, data_base: usize, lays: &[LayerOffs; 28], num_layers: usize) -> Self {
        unsafe extern "C" fn d_set(_: *mut c_void) -> i32 { -1 }
        unsafe extern "C" fn d_h2d(_: u64, _: *const c_void, _: usize) -> i32 { -1 }
        unsafe extern "C" fn d_d2h(_: *mut c_void, _: u64, _: usize) -> i32 { -1 }
        unsafe extern "C" fn d_l(_: *mut c_void, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: u32, _: *mut c_void, _: *mut *mut c_void, _: *mut *mut c_void) -> i32 { -1 }

        let devs = syscall6(9, 0, MAX_GPUS * 32, 3, 0x22, (-1isize) as usize, 0) as *mut GpuDeviceSlot;
        let routes = syscall6(9, 0, num_layers * 64, 3, 0x22, (-1isize) as usize, 0) as *mut LayerGpuRoute;
        let mut i = 0usize;
        while i < num_layers {
            *routes.add(i) = LayerGpuRoute { dev_id: -1, q:0, k:0, v:0, o:0, gate:0, up:0, down:0 };
            i += 1;
        }

        let mut cl = Self {
            num_gpus: 0, gpu_layers: 0, total_layers: num_layers, total_vram_used_mb: 0,
            devs, routes, cu_ctx_set: d_set, cu_memcpy_htod: d_h2d,
            cu_memcpy_dtoh: d_d2h, cu_launch_kernel: d_l,
        };

        let lib = dlopen(b"libcuda.so.1\0".as_ptr(), 1);
        if lib.is_null() { return cl; }

        let cu_init: unsafe extern "C" fn(u32) -> i32 = core::mem::transmute(dlsym(lib, b"cuInit\0".as_ptr()));
        let cu_cnt: unsafe extern "C" fn(*mut i32) -> i32 = core::mem::transmute(dlsym(lib, b"cuDeviceGetCount\0".as_ptr()));
        let cu_dev_get: unsafe extern "C" fn(*mut i32, i32) -> i32 = core::mem::transmute(dlsym(lib, b"cuDeviceGet\0".as_ptr()));
        let cu_ctx_create: unsafe extern "C" fn(*mut *mut c_void, u32, i32) -> i32 = core::mem::transmute(sym_v2(lib, b"cuCtxCreate_v2\0", b"cuCtxCreate\0"));
        let cu_mem_info: unsafe extern "C" fn(*mut usize, *mut usize) -> i32 = core::mem::transmute(sym_v2(lib, b"cuMemGetInfo_v2\0", b"cuMemGetInfo\0"));
        let cu_mem_alloc: unsafe extern "C" fn(*mut u64, usize) -> i32 = core::mem::transmute(sym_v2(lib, b"cuMemAlloc_v2\0", b"cuMemAlloc\0"));
        let cu_mod_load: unsafe extern "C" fn(*mut *mut c_void, *const c_void) -> i32 = core::mem::transmute(dlsym(lib, b"cuModuleLoadData\0".as_ptr()));
        let cu_mod_fn: unsafe extern "C" fn(*mut *mut c_void, *mut c_void, *const u8) -> i32 = core::mem::transmute(dlsym(lib, b"cuModuleGetFunction\0".as_ptr()));

        cl.cu_ctx_set = core::mem::transmute(dlsym(lib, b"cuCtxSetCurrent\0".as_ptr()));
        cl.cu_memcpy_htod = core::mem::transmute(sym_v2(lib, b"cuMemcpyHtoD_v2\0", b"cuMemcpyHtoD\0"));
        cl.cu_memcpy_dtoh = core::mem::transmute(sym_v2(lib, b"cuMemcpyDtoH_v2\0", b"cuMemcpyDtoH\0"));
        cl.cu_launch_kernel = core::mem::transmute(dlsym(lib, b"cuLaunchKernel\0".as_ptr()));

        if cu_init(0) != 0 { return cl; }
        let mut raw_cnt = 0i32;
        if cu_cnt(&mut raw_cnt) != 0 || raw_cnt <= 0 { return cl; }
        let n_devs = if (raw_cnt as usize) > MAX_GPUS { MAX_GPUS } else { raw_cnt as usize };
        cl.num_gpus = n_devs;

        let sz_q = 1536 * 48 * 34; let sz_kv = 256 * 48 * 34; let sz_gu = 18000 * 48 * 34; let sz_d = 1536 * 280 * 34;
        let layer_bytes = sz_q + sz_kv + sz_kv + sz_q + sz_gu + sz_gu + sz_d;
        let reserve = 220usize << 20;

        let mut cur_layer = 0usize;
        let mut d_idx = 0usize;
        while d_idx < n_devs && cur_layer < num_layers {
            let mut dev = 0i32;
            if cu_dev_get(&mut dev, d_idx as i32) != 0 { d_idx += 1; continue; }
            let mut ctx: *mut c_void = core::ptr::null_mut();
            if cu_ctx_create(&mut ctx, 0, dev) != 0 { d_idx += 1; continue; }
            (cl.cu_ctx_set)(ctx);

            let mut mod_h: *mut c_void = core::ptr::null_mut();
            let mut kfn: *mut c_void = core::ptr::null_mut();
            if cu_mod_load(&mut mod_h, PTX_HEAD.as_ptr() as *const c_void) != 0 { d_idx += 1; continue; }
            if cu_mod_fn(&mut kfn, mod_h, b"q8_gemv_dp4a\0".as_ptr()) != 0 { d_idx += 1; continue; }

            let mut dx = 0u64; let mut dout = 0u64;
            if cu_mem_alloc(&mut dx, 280 * 36) != 0 || cu_mem_alloc(&mut dout, 18000 * 4) != 0 { d_idx += 1; continue; }
            *devs.add(d_idx) = GpuDeviceSlot { ctx, kern_fn: kfn, d_x_q8: dx, d_out_f32: dout };

            let mut free_b = 0usize; let mut total_b = 0usize;
            if cu_mem_info(&mut free_b, &mut total_b) != 0 { d_idx += 1; continue; }
            let avail = if free_b > reserve { free_b - reserve } else { 0 };
            let cap_layers = avail / layer_bytes;

            let mut placed = 0usize;
            while placed < cap_layers && cur_layer < num_layers {
                let lw = &lays[cur_layer];
                let mut r = LayerGpuRoute { dev_id: d_idx as i32, q:0, k:0, v:0, o:0, gate:0, up:0, down:0 };
                if cu_mem_alloc(&mut r.q, sz_q) != 0 || cu_mem_alloc(&mut r.k, sz_kv) != 0 ||
                   cu_mem_alloc(&mut r.v, sz_kv) != 0 || cu_mem_alloc(&mut r.o, sz_q) != 0 ||
                   cu_mem_alloc(&mut r.gate, sz_gu) != 0 || cu_mem_alloc(&mut r.up, sz_gu) != 0 ||
                   cu_mem_alloc(&mut r.down, sz_d) != 0 { break; }

                (cl.cu_memcpy_htod)(r.q, gguf.add(data_base + lw.q) as *const c_void, sz_q);
                (cl.cu_memcpy_htod)(r.k, gguf.add(data_base + lw.k) as *const c_void, sz_kv);
                (cl.cu_memcpy_htod)(r.v, gguf.add(data_base + lw.v) as *const c_void, sz_kv);
                (cl.cu_memcpy_htod)(r.o, gguf.add(data_base + lw.o) as *const c_void, sz_q);
                (cl.cu_memcpy_htod)(r.gate, gguf.add(data_base + lw.gate) as *const c_void, sz_gu);
                (cl.cu_memcpy_htod)(r.up, gguf.add(data_base + lw.up) as *const c_void, sz_gu);
                (cl.cu_memcpy_htod)(r.down, gguf.add(data_base + lw.down) as *const c_void, sz_d);

                *routes.add(cur_layer) = r;
                cur_layer += 1;
                placed += 1;
                cl.total_vram_used_mb += layer_bytes;
            }
            d_idx += 1;
        }

        cl.gpu_layers = cur_layer;
        w_str(b"[CLUSTER AUTO-SPLIT] GPUS="); w_u64(cl.num_gpus as u64);
        w_str(b" | VRAM_USED="); w_u64((cl.total_vram_used_mb >> 20) as u64);
        w_str(b" MiB | GPU_LAYERS="); w_u64(cur_layer as u64);
        w_str(b"/"); w_u64(num_layers as u64);
        w_str(b" | DDR_LAYERS="); w_u64((num_layers - cur_layer) as u64);
        w_str(b"/"); w_u64(num_layers as u64); w_str(b"\n");
        cl
    }

    #[inline(always)]
    pub unsafe fn layer_dev(&self, layer: usize) -> i32 {
        (*self.routes.add(layer)).dev_id
    }

    #[inline(always)]
    pub unsafe fn route(&self, layer: usize) -> &LayerGpuRoute {
        &*self.routes.add(layer)
    }

    #[inline(always)]
    pub unsafe fn upload_x(&self, dev_id: i32, x_q8: *const u8, blocks: usize) {
        let slot = &*self.devs.add(dev_id as usize);
        (self.cu_ctx_set)(slot.ctx);
        (self.cu_memcpy_htod)(slot.d_x_q8, x_q8 as *const c_void, blocks * 36);
    }

    pub unsafe fn gemv_off(&self, dev_id: i32, d_w: u64, dst_off_bytes: u64, rows: u32, blocks: u32) {
        let slot = &*self.devs.add(dev_id as usize);
        let mut p_w = d_w; let mut p_x = slot.d_x_q8; let mut p_o = slot.d_out_f32 + dst_off_bytes;
        let mut p_r = rows; let mut p_b = blocks;
        let mut params: [*mut c_void; 5] = [
            &mut p_w as *mut u64 as *mut c_void, &mut p_x as *mut u64 as *mut c_void,
            &mut p_o as *mut u64 as *mut c_void, &mut p_r as *mut u32 as *mut c_void,
            &mut p_b as *mut u32 as *mut c_void,
        ];
        (self.cu_launch_kernel)(slot.kern_fn, rows, 1, 1, 32, 1, 1, 0, core::ptr::null_mut(), params.as_mut_ptr(), core::ptr::null_mut());
    }

    pub unsafe fn fetch(&self, dev_id: i32, dst_f32: *mut f32, src_off_bytes: u64, bytes: usize) {
        let slot = &*self.devs.add(dev_id as usize);
        (self.cu_memcpy_dtoh)(dst_f32 as *mut c_void, slot.d_out_f32 + src_off_bytes, bytes);
    }
}
