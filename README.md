# zero-llm
Universal, zero-dependency, hardware-adaptive GGUF inference engine in pure C11 + PTX (`sm_61+`) + `AVX2` (`W^X` `2MB HugePages`).
- **Auto-Architecture GGUF Loader**: reads layers, hidden dimensions, GQA heads, RoPE base, and BPE vocabulary directly from any GGUF file.
- **Auto-GPU / DDR Split**: queries live free VRAM via `cuMemGetInfo_v2`, offloads maximum layers to GPU (`dp4a.s32.s32` + async `CUstream`), and seals remaining layers in a `2MB HugePage` Read-Only `W^X` DDR vault.
- **Auto-Context Planner**: dynamically calculates safe `max_seq` and `max_gen` token limits from live `/proc/meminfo` (`MemAvailable`).
