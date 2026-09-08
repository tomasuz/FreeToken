"""NVFP4 grouped-expert GEMV, JIT-compiled through hipcc/nvcc.

A second implementation of what ``kernel/triton/nvfp4_fused_moe.py`` already does, for
devices Triton's AMD backend refuses to build for (GCN5-class parts: gfx900, gfx90c).
Triton stays the production path everywhere it compiles; this exists so that a device it
does not cover can still take a share of the MoE split instead of sitting idle.

Build policy, host-compiler selection and the per-architecture build directory are shared
with the GGUF extension -- see ``kernel/gguf.py``, where the reasoning for each lives.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_CSRC = pathlib.Path(__file__).parent / "csrc" / "nvfp4_moe"


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    from .gguf import _apply_arch_selection, _build_directory, _c_compiler_for, _host_compiler

    is_hip = bool(getattr(torch.version, "hip", None))
    extra_cuda_cflags = ["-O3"] if is_hip else ["-O3", "--expt-relaxed-constexpr"]
    archs = _apply_arch_selection(is_hip)
    if archs:
        logger.info(f"building NVFP4 MoE kernel for the devices present: {', '.join(archs)}")
    host_cxx = None if is_hip else _host_compiler()
    if host_cxx is not None:
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)
    return load(
        name="freetoken_nvfp4_moe",
        sources=[str(_CSRC / "nvfp4_moe_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=_build_directory(archs, "freetoken_nvfp4_moe"),
        verbose=True,
    )


def nvfp4_moe_vec(
    a: torch.Tensor,
    packed: torch.Tensor,
    scale: torch.Tensor,
    global_: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """``[tokens * top_k, row]``: each route's activation against its expert's weight.

    ``topk_ids`` is read flat, one entry per route, and names a row of the stacked
    weight banks (a cache slot, not an expert id, wherever the caller keeps a cache).
    """
    # The wave width is the device's, and asking torch for it here keeps the kernel free
    # of the ATen context header, which hipifies to one that pulls in hipsparse.
    warp = torch.cuda.get_device_properties(a.device).warp_size
    return _module().nvfp4_moe_vec(
        a, packed, scale, global_, topk_ids.reshape(-1).to(torch.int32), int(top_k),
        int(row), int(tokens), int(warp),
    )


__all__ = ["nvfp4_moe_vec"]
