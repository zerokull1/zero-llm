# zero-llm

**Universal Hardware-Adaptive GGUF Inference Engine (Pure-Speed Tokamak Core + W^X Arena)**

`zero-llm` is a standalone, zero-dependency LLM inference engine built on the **pure-speed Tokamak compute core** — stripped of any security/filtering overhead and engineered to run anywhere: on any desktop, laptop, edge device (x86_64, ARM64, RISC-V, IBM Z), or rack server, with or without a GPU.

## Core Architecture
- **Tokamak Pure-Speed Compute Core (`9.4 KB` L1i Hot Path)**:
  - **4-Way L1d-Orbital Matrix Streaming**: Processes 4 weight rows simultaneously against a single hot 64-byte L1d activation slice (`Q8_0` / `Q6_K` / `F16`), cutting L1d cache evictions by 4x.
  - **L1i-Resident Split Build**: Separates the hot token generation loop (`9,635 bytes` in L1i, zero `.data`/`.bss`) from the cold GGUF/BPE initializer (`22.3 KB` total binary `.text`).
  - **GF(2^64) & SwiGLU Sparse Pruning**: Skips near-zero FFN neurons deterministically without branch mispredictions.
- **Isolated `W^X` Activation Arena**: Working tensor buffers execute inside a dedicated `mmap`/`mprotect` arena bounded by `PROT_NONE` guard pages with `MADV_HUGEPAGE`.
- **Universal Portability (Any PC, Device, or Rack Server)**: Pure ISO C11 + OpenMP + dynamic `dlopen("libcuda.so.1")` PTX `dp4a` backend (no CUDA SDK required; automatically falls back to 100% CPU on headless servers or non-NVIDIA hardware).
- **Full Auto-Hardware & Auto-GGUF Planner**:
  - Automatically detects model architecture (`n_layers`, `hidden_dim`, `ffn_dim`, `n_heads`, `n_kv_heads`, `rope_theta`), tokenizer (`gpt2` byte-BPE / `llama` SentencePiece), and chat template (`ChatML`, `Llama-3`, `Gemma`, `Mistral`, `Raw`).
  - Queries `MemAvailable` (`/proc/meminfo`) and free GPU VRAM (`cuMemGetInfo`) to split layers between VRAM and `W^X` DDR and size the KV cache (`max_seq`) dynamically.

## Quick Start
```bash
make
./zero-llm -m /path/to/model.gguf "Hello! Reply in 5 words:"
./zero-llm -m /path/to/model.gguf   # Interactive chat (/clear, /exit)
```

## Standalone 0-Dependency Rust + Raw ASM W^X Core (`rust_wx_core/`)

Zero-`libc` (`#![no_std]`, `#![no_main]`, `-nodefaultlibs`) Multi-ISA inference engine synthesizing a `102 B` `Q8_0` (`F16C + AVX2 + FMA3`) kernel inside a 3-page `[PROT_NONE][RX][PROT_NONE]` W^X arena, zero-copy `mmap` + `MADV_HUGEPAGE` GGUF indexer, Lock-Free `SYS_clone` + `PAUSE` thread pool (`0 futex`), and `tokenizer.ggml.tokens` UTF-8 decoder:

- **L1i Footprint:** `11 264 B` `.text` (`34.3%` of 32 KB L1i).
- **Throughput:** `284.6M` cycles (`~86.2 ms` / `~11.6 full tok/s` across all 28 layers x 7 projections + `151,936`-vocab `LM Head Argmax` = `1.54B` `Q8_0` params/token on 4-core Haswell `i5-4590` DDR3 — `4.66x` faster than the early `2.49 tok/s` probe).

## Standalone 0-Dependency Rust + Raw ASM W^X Chat Engine (`rust_wx_core/`)

Zero-`libc` (`#![no_std]`, `#![no_main]`, `-nodefaultlibs`, statically linked) Multi-ISA inference engine synthesizing a `Q8_0` (`F16C + AVX2 + FMA3`, `vpabsb + vpsignb + vpmaddubsw`) kernel inside a 3-page `[PROT_NONE][RX][PROT_NONE]` W^X arena, with zero-copy `mmap` + `MADV_HUGEPAGE` GGUF indexer, Lock-Free `SYS_clone` + `PAUSE` thread pool (`0 futex`), NEOX split-half RoPE, Causal GQA Attention (`12 Q : 2 KV` heads + KV-cache), exponential SwiGLU FFN (`8960`), and built-in `tokenizer.ggml.tokens` BPE tokenizer/decoder:

- **L1i Footprint:** `13 527 B` `.text` (`41.3%` of 32 KB L1i, strictly `<= 16 KB`).
- **End-to-End Inference:** `~94.6–99 ms/token` (`~10.1–10.6 full tok/s` across all 28 layers x 7 projections + `151,936`-vocab `LM Head Argmax` = `1.54B` `Q8_0` params/token on 4-core Haswell `i5-4590` DDR3).

- **Multi-GPU & Vulkan/CUDA Cluster HAL (`rust_wx_core/src/gpu.rs`):** Auto-probes Vulkan devices (`libvulkan.so.1`) and splits model layers across up to 1024 CUDA GPUs (`28/28` layers offloaded on GTX 1050 2GB) with dynamic `CPUID` (`AVX2+FMA3+F16C` -> scalar fallback) and thermal `SYS_sched_yield` worker backoff (`.text` = **11 394 B**, 34.7% L1i).
