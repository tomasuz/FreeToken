"""ggml_mmq's CUDA branch builds: nvcc compiles and links llama.cpp's MMQ/MMVQ with the same
entry points as the ROCm build. A compile check, runnable on any host with nvcc (no NVIDIA
GPU needed: the library is loaded, nothing is called)."""

from __future__ import annotations

import shutil

import pytest

pytestmark = [pytest.mark.slow]


def test_cuda_build_compiles_and_links(monkeypatch, tmp_path):
    from freetoken.kernel import ggml_mmq

    if shutil.which(ggml_mmq._nvcc()) is None:
        pytest.skip("no nvcc")
    monkeypatch.setattr(ggml_mmq, "_platform", lambda: "cuda")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.6")
    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    ggml_mmq._lib.cache_clear()
    try:
        lib = ggml_mmq._lib()
        for sym in ("ft_mmq_init", "ft_mmq_supports", "ft_mmq_moe", "ft_mmq_moe_workspace",
                    "ft_mmvq", "ft_mmvq_workspace"):
            assert getattr(lib, sym) is not None
    finally:
        ggml_mmq._lib.cache_clear()
