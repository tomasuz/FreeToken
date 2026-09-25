# Vendored llama.cpp MMQ / MMVQ (ggml-cuda)

Unmodified copies of the ggml CUDA/HIP quantized matmul (MMQ) and mat-vec (MMVQ) kernels
and the headers they need, from llama.cpp commit `6fcaa16` (2026-09-21), MIT licensed (see
`LICENSE`).

FreeToken's older GGUF kernels (`../gguf/`, via vLLM) date from a 2024 llama.cpp: their MMQ
knows no i-quants (IQ3_XXS, IQ4_NL, IQ4_XS, ...) and has no grouped-by-expert MoE path, so a
GGUF MoE prefill computed every token row as its own GEMV; their MMVQ predates the RDNA
tuning and the fused gate/up (GLU) variant. These files provide llama.cpp's `mul_mat_id`
MMQ path (`mm_ids_helper` + `quantize_mmq_q8_1` + `mul_mat_q_case`) and its MMVQ
(`mul_mat_vec_q`, dense and per-expert, optionally fused with the gate projection).

Layout mirrors upstream (`include/`, `src/`, `src/ggml-cuda/`) so the files compile as-is.
Everything FreeToken adds lives in `ft_*`: `ft_compat.h` (force-included first),
`ft_mmq.cu` (the runtime symbols ggml-cuda.cu and ggml.c would otherwise provide, and the
MMQ entry point) and `ft_mmvq.cu` (the MMVQ entry point), loaded with ctypes by
`freetoken/kernel/ggml_mmq.py`. To update, copy the same files from a newer llama.cpp and
rebuild; the file list is the transitive `#include` closure of `mmq.cuh`, `mmvq.cu`,
`quantize.cu` and `mmid.cu`.
