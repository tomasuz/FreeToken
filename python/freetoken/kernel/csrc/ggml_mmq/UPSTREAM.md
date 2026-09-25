# Vendored llama.cpp MMQ (ggml-cuda)

Unmodified copies of the ggml CUDA/HIP quantized matmul (MMQ) kernels and the headers they
need, from llama.cpp commit `6fcaa16` (2026-09-21), MIT licensed (see `LICENSE`).

FreeToken's older GGUF kernels (`../gguf/`, via vLLM) have no MMQ for the i-quants
(IQ3_XXS, IQ4_NL, IQ4_XS, ...) and no grouped-by-expert MoE path, so a GGUF MoE prefill
computed every token row as its own GEMV. These files provide llama.cpp's
`mul_mat_id` MMQ path (`mm_ids_helper` + `quantize_mmq_q8_1` + `mul_mat_q_case`).

Layout mirrors upstream (`include/`, `src/`, `src/ggml-cuda/`) so the files compile as-is.
Everything FreeToken adds lives in `ft_mmq.cu`: the few runtime symbols ggml-cuda.cu
and ggml.c would otherwise provide, and a C entry point loaded with ctypes
(`freetoken/kernel/ggml_mmq.py`). To update, copy the same files from a newer llama.cpp
and rebuild; the file list is the transitive `#include` closure of `mmq.cuh`,
`quantize.cu` and `mmid.cu`.
