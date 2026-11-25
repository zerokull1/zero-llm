# Security & Integrity Architecture

`zero-llm` (`rust_wx_core`) enforces strict hardware-level isolation:
- **Zero-Libc / Zero-Network Attack Surface**: No `glibc`, no `malloc`/`free`, and zero open TCP/UDP sockets.
- **W^X Runtime Arena**: Synthesized AVX2/FMA3/F16C machine code executes inside a 3-page `[PROT_NONE][RX][PROT_NONE]` arena with `FNV-1a` integrity verification before execution and mandatory zero-wipe (`PROT_NONE`) on exit.
- **Zero-Socket Tokamak Bus & XDP `0x88B5` Verification**: Inter-process and inter-rack frames are validated via monotonic `rack_epoch` anti-replay counters, `FNV-1a` payload checksums, and control-character SWAR sanitization.
