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
