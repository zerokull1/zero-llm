# zero-llm

**Universal Hardware-Adaptive GGUF Inference Engine (Zero Dependencies, ISO C11 + OpenMP + Dynamic PTX)**

`zero-llm` is a standalone, zero-dependency LLM inference engine written in portable ISO C11. It automatically inspects any GGUF model and host hardware at startup, balancing layers between GPU VRAM (`dlopen` PTX `dp4a`, no CUDA SDK required) and an isolated `W^X` `2MB-HugePage` DDR arena with `PROT_NONE` guard pages.

## Key Features
- **Zero External Dependencies**: Pure C11 + OpenMP + `dlopen("libcuda.so.1")` (gracefully falls back to 100% CPU on headless servers, ARM64, RISC-V, or systems without NVIDIA GPUs).
- **Auto-GGUF Architecture & Tokenizer**: Automatically detects model dimensions (`n_layers`, `hidden_dim`, `ffn_dim`, `n_heads`, `n_kv_heads`, `rope_theta`), vocabulary, tokenizer type (`gpt2` byte-BPE / `llama` SentencePiece), and chat template (`ChatML`, `Llama-3`, `Gemma`, `Mistral`, `Raw`).
- **Auto-Hardware & Context Planner**: Queries `MemAvailable` (`/proc/meminfo`) and free GPU VRAM (`cuMemGetInfo`) to split transformer layers automatically and size the KV cache (`max_seq`).
- **Isolated `W^X` Activation Arena**: Working activation buffers run inside a dedicated `mmap`/`mprotect` arena bounded by `PROT_NONE` guard pages with `MADV_HUGEPAGE`.

## Quick Start
```bash
make
./zero-llm -m /path/to/model.gguf "Hello! Reply in 5 words:"
./zero-llm -m /path/to/model.gguf   # Interactive chat (/clear, /exit)
```
