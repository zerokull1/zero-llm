use crate::gguf::{LayerOffs, ModelCfg};
use crate::sys::{syscall6,w_str,w_u64};
use core::ffi::c_void;

#[link(name = "c")]
unsafe extern "C" {
    fn dlopen(filename: *const u8,flags: i32) -> *mut c_void;
    fn dlsym(handle: *mut c_void,symbol: *const u8) -> *mut c_void;
}

pub const MAX_GPUS: usize = 1024;

#[derive(Clone,Copy)]
pub struct LayerGpuRoute {
    pub dev_id: i32,
    pub q: u64,pub k: u64,pub v: u64,pub o: u64,
    pub gate: u64,pub up: u64,pub down: u64,
}

#[derive(Clone,Copy)]
pub struct GpuDeviceSlot {
    pub ctx: *mut c_void,
    pub kern_fn: *mut c_void,
    pub d_x_q8: u64,
    pub d_out_f32: u64,
}

pub struct GpuCluster {
    pub num_gpus: usize,
    pub gpu_layers: usize,
    pub total_vram_used_mb: usize,
    pub devs: *mut GpuDeviceSlot,
    pub routes: *mut LayerGpuRoute,
    cu_ctx_set: Option<unsafe extern "C" fn(*mut c_void) -> i32>,
    cu_ctx_sync: Option<unsafe extern "C" fn() -> i32>,
    cu_memcpy_htod: Option<unsafe extern "C" fn(u64,*const c_void,usize) -> i32>,
    cu_memcpy_dtoh: Option<unsafe extern "C" fn(*mut c_void,u64,usize) -> i32>,
    cu_launch_kernel: Option<unsafe extern "C" fn(*mut c_void,u32,u32,u32,u32,u32,u32,u32,*mut c_void,*mut *mut c_void,*mut *mut c_void) -> i32>,
}

unsafe fn sym_v2(lib: *mut c_void,v2: &[u8],v1: &[u8]) -> *mut c_void {
    let p = dlsym(lib,v2.as_ptr());
    if !p.is_null() { p } else { dlsym(lib,v1.as_ptr()) }
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

pub unsafe fn probe_vulkan_devices() -> u32 {
    let vk = dlopen(b"libvulkan.so.1\0".as_ptr(),1);
    if vk.is_null() { return 0;}
    let vk_create: unsafe extern "C" fn(*const u64,*const c_void,*mut *mut c_void) -> i32 = core::mem::transmute(dlsym(vk,b"vkCreateInstance\0".as_ptr()));
    let vk_enum: unsafe extern "C" fn(*mut c_void,*mut u32,*mut *mut c_void) -> i32 = core::mem::transmute(dlsym(vk,b"vkEnumeratePhysicalDevices\0".as_ptr()));
    let vk_destroy: unsafe extern "C" fn(*mut c_void,*const c_void) = core::mem::transmute(dlsym(vk,b"vkDestroyInstance\0".as_ptr()));
    let mut ci = [0u64;8];ci[0] = 1;
    let mut inst: *mut c_void = core::ptr::null_mut();
    if vk_create(ci.as_ptr(),core::ptr::null(),&mut inst) != 0 || inst.is_null() { return 0;}
    let mut cnt = 0u32;
    let _ = vk_enum(inst,&mut cnt,core::ptr::null_mut());
    vk_destroy(inst,core::ptr::null());
    cnt
}

impl GpuCluster {
    #[inline(never)]
    pub unsafe fn init_cluster_auto_split(gguf: *const u8,data_base: usize,lays: &[LayerOffs;64],cfg: &ModelCfg) -> Self {
        let total_layers = cfg.n_layers;
        let vk_cnt = probe_vulkan_devices();
        let routes_bytes = (total_layers * core::mem::size_of::<LayerGpuRoute>() + 4095) & !4095;
        let routes = syscall6(9,0,routes_bytes,3,0x22,(-1isize) as usize,0) as *mut LayerGpuRoute;
        let mut l0 = 0usize;
        while l0 < total_layers { (*routes.add(l0)).dev_id = -1;l0 += 1;}
        let mut cl = Self {
            num_gpus: 0,gpu_layers: 0,total_vram_used_mb: 0,
            devs: core::ptr::null_mut(),routes,
            cu_ctx_set: None,cu_ctx_sync: None,cu_memcpy_htod: None,cu_memcpy_dtoh: None,cu_launch_kernel: None,
        };
        let lib = dlopen(b"libcuda.so.1\0".as_ptr(),1);
        if lib.is_null() {
            w_str(b"[CLUSTER-HAL] Vulkan=");w_u64(vk_cnt as u64);w_str(b" | GPUs=0\n");
            return cl;
        }
        let cu_init: unsafe extern "C" fn(u32) -> i32 = core::mem::transmute(dlsym(lib,b"cuInit\0".as_ptr()));
        let cu_dev_cnt: unsafe extern "C" fn(*mut i32) -> i32 = core::mem::transmute(dlsym(lib,b"cuDeviceGetCount\0".as_ptr()));
        let cu_dev_get: unsafe extern "C" fn(*mut i32,i32) -> i32 = core::mem::transmute(dlsym(lib,b"cuDeviceGet\0".as_ptr()));
        let cu_ctx_create: unsafe extern "C" fn(*mut *mut c_void,u32,i32) -> i32 = core::mem::transmute(sym_v2(lib,b"cuCtxCreate_v2\0",b"cuCtxCreate\0"));
        let cu_mem_info: unsafe extern "C" fn(*mut usize,*mut usize) -> i32 = core::mem::transmute(sym_v2(lib,b"cuMemGetInfo_v2\0",b"cuMemGetInfo\0"));
        let cu_mem_alloc: unsafe extern "C" fn(*mut u64,usize) -> i32 = core::mem::transmute(sym_v2(lib,b"cuMemAlloc_v2\0",b"cuMemAlloc\0"));
        let cu_mod_load: unsafe extern "C" fn(*mut *mut c_void,*const c_void) -> i32 = core::mem::transmute(dlsym(lib,b"cuModuleLoadData\0".as_ptr()));
        let cu_mod_fn: unsafe extern "C" fn(*mut *mut c_void,*mut c_void,*const u8) -> i32 = core::mem::transmute(dlsym(lib,b"cuModuleGetFunction\0".as_ptr()));
        cl.cu_ctx_set = Some(core::mem::transmute(dlsym(lib,b"cuCtxSetCurrent\0".as_ptr())));
        cl.cu_ctx_sync = Some(core::mem::transmute(dlsym(lib,b"cuCtxSynchronize\0".as_ptr())));
        cl.cu_memcpy_htod = Some(core::mem::transmute(sym_v2(lib,b"cuMemcpyHtoD_v2\0",b"cuMemcpyHtoD\0")));
        cl.cu_memcpy_dtoh = Some(core::mem::transmute(sym_v2(lib,b"cuMemcpyDtoH_v2\0",b"cuMemcpyDtoH\0")));
        cl.cu_launch_kernel = Some(core::mem::transmute(dlsym(lib,b"cuLaunchKernel\0".as_ptr())));
        if cu_init(0) != 0 { return cl;}
        let mut cnt = 0i32;
        if cu_dev_cnt(&mut cnt) != 0 || cnt <= 0 { return cl;}
        let n_devs = if (cnt as usize) > MAX_GPUS { MAX_GPUS } else { cnt as usize };
        let devs_bytes = (n_devs * core::mem::size_of::<GpuDeviceSlot>() + 4095) & !4095;
        cl.devs = syscall6(9,0,devs_bytes,3,0x22,(-1isize) as usize,0) as *mut GpuDeviceSlot;
        let hb = cfg.hidden >> 5; let fb = cfg.ffn >> 5; let kv_dim = (cfg.hidden / cfg.n_heads) * cfg.n_kv_heads;
        let layer_bytes = (2 * cfg.hidden * hb + 2 * kv_dim * hb + 2 * cfg.ffn * hb + cfg.hidden * fb) * 34;
        let mut next_layer = 0usize;
        let mut g = 0usize;
        while g < n_devs && next_layer < total_layers {
            let mut dev = 0i32;let mut ctx: *mut c_void = core::ptr::null_mut();
            if cu_dev_get(&mut dev,g as i32) == 0 && cu_ctx_create(&mut ctx,0,dev) == 0 && !ctx.is_null() {
                let mut free_b = 0usize;let mut tot_b = 0usize;
                let _ = cu_mem_info(&mut free_b,&mut tot_b);
                let mut can_fit = free_b.saturating_sub(320 * 1024 * 1024) / layer_bytes;
                if can_fit > total_layers - next_layer { can_fit = total_layers - next_layer;}
                let mut ptx_mod: *mut c_void = core::ptr::null_mut();let mut kern_fn: *mut c_void = core::ptr::null_mut();
                let mut d_x = 0u64;let mut d_out = 0u64;
                if can_fit > 0
                    && cu_mod_load(&mut ptx_mod,PTX_HEAD.as_ptr() as *const c_void) == 0
                    && cu_mod_fn(&mut kern_fn,ptx_mod,b"q8_gemv_dp4a\0".as_ptr()) == 0
                    && cu_mem_alloc(&mut d_x,(if fb > hb { fb } else { hb }) * 36) == 0 && cu_mem_alloc(&mut d_out,(if cfg.ffn > cfg.hidden { cfg.ffn } else { cfg.hidden }) * 8) == 0
                {
                    *cl.devs.add(cl.num_gpus) = GpuDeviceSlot { ctx,kern_fn,d_x_q8: d_x,d_out_f32: d_out };
                    let gpu_idx = cl.num_gpus as i32;
                    let htod = cl.cu_memcpy_htod.unwrap_unchecked();
                    let mut k = 0usize;
                    while k < can_fit {
                        let lw = &lays[next_layer];
                        let mut r = LayerGpuRoute { dev_id: gpu_idx,q:0,k:0,v:0,o:0,gate:0,up:0,down:0 };
                        let pairs: [(&mut u64,usize,usize);7] = [
                            (&mut r.q,lw.q,cfg.hidden*hb*34),(&mut r.k,lw.k,kv_dim*hb*34),(&mut r.v,lw.v,kv_dim*hb*34),
                            (&mut r.o,lw.o,cfg.hidden*hb*34),(&mut r.gate,lw.gate,cfg.ffn*hb*34),(&mut r.up,lw.up,cfg.ffn*hb*34),
                            (&mut r.down,lw.down,cfg.hidden*fb*34),
                        ];
                        let mut ok = true;
                        for (dp,off,sz) in pairs {
                            if cu_mem_alloc(dp,sz) != 0 { ok = false;break;}
                            let src_p = gguf.add(data_base + off); let mut pg = 0usize; while pg < sz { let _ = src_p.add(pg).read_volatile(); pg += 4096; }
                            if htod(*dp,src_p as *const c_void,sz) != 0 { ok = false; break; }
                        }
                        if !ok { break;}
                        *routes.add(next_layer) = r;
                        next_layer += 1;cl.gpu_layers += 1;cl.total_vram_used_mb += layer_bytes / (1024 * 1024);k += 1;
                    }
                    cl.num_gpus += 1;
                }
            }
            g += 1;
        }
        w_str(b"[CLUSTER-HAL] Vulkan=");w_u64(vk_cnt as u64);
        w_str(b" | GPUs=");w_u64(cl.num_gpus as u64);
        w_str(b" | Offload=");w_u64(cl.gpu_layers as u64);
        w_str(b"/");w_u64(total_layers as u64);w_str(b"\n");
        cl
    }
    #[inline(always)] pub unsafe fn layer_dev(&self,layer: usize) -> i32 { (*self.routes.add(layer)).dev_id }
    #[inline(always)] pub unsafe fn route(&self,layer: usize) -> LayerGpuRoute { *self.routes.add(layer) }
    #[inline(never)] pub unsafe fn upload_x(&self,dev_id: i32,x_u8: *const u8,blocks: usize) {
        let s = &*self.devs.add(dev_id as usize);
        let _ = (self.cu_ctx_set.unwrap_unchecked())(s.ctx);
        let _ = (self.cu_memcpy_htod.unwrap_unchecked())(s.d_x_q8,x_u8 as *const c_void,blocks * 36);
    }
    #[inline(never)] pub unsafe fn gemv_off(&self,dev_id: i32,d_w: u64,out_byte_off: u64,rows: u32,blocks: u32) {
        let s = &*self.devs.add(dev_id as usize);
        let (mut pw,mut px,mut po,mut pr,mut pb) = (d_w,s.d_x_q8,s.d_out_f32 + out_byte_off,rows,blocks);
        let mut p: [*mut c_void;5] = [&mut pw as *mut _ as _,&mut px as *mut _ as _,&mut po as *mut _ as _,&mut pr as *mut _ as _,&mut pb as *mut _ as _];
        let _ = (self.cu_launch_kernel.unwrap_unchecked())(s.kern_fn,rows,1,1,32,1,1,0,core::ptr::null_mut(),p.as_mut_ptr(),core::ptr::null_mut());
    }

    #[inline(never)] pub unsafe fn proj1(&self,dev_id: i32,x_u8: *const u8,w: u64,dst: *mut f32,rows: u32,blocks: u32) {
        self.upload_x(dev_id,x_u8,blocks as usize);
        self.gemv_off(dev_id,w,0,rows,blocks);
        self.fetch(dev_id,dst,0,(rows as usize) * 4);
    }
    #[inline(never)] pub unsafe fn proj2(&self,dev_id: i32,x_u8: *const u8,w1: u64,w2: u64,d1: *mut f32,d2: *mut f32,rows: u32,blocks: u32) {
        let b = (rows as u64) * 4;
        self.upload_x(dev_id,x_u8,blocks as usize);
        self.gemv_off(dev_id,w1,0,rows,blocks);
        self.gemv_off(dev_id,w2,b,rows,blocks);
        self.fetch(dev_id,d1,0,b as usize);
        self.fetch(dev_id,d2,b,b as usize);
    }
    #[inline(never)] pub unsafe fn fetch(&self,dev_id: i32,dst_f32: *mut f32,out_byte_off: u64,bytes: usize) {
        let s = &*self.devs.add(dev_id as usize);
        let _ = (self.cu_ctx_sync.unwrap_unchecked())();
        let _ = (self.cu_memcpy_dtoh.unwrap_unchecked())(dst_f32 as *mut c_void,s.d_out_f32 + out_byte_off,bytes);
    }
}
