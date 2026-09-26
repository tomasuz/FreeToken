"""llama.cpp's CPU dot products for ggml block types, built with the host C compiler.

The CPU MoE executor (``csrc/cpu_moe``) computes GGUF experts row by row with these: per
weight type a ``vec_dot`` over one row, the activation type it pairs with (Q8_0 or Q8_K)
and that type's row quantizer -- what ``ggml_get_type_traits_cpu()`` answers inside
llama.cpp. The sources are vendored with the MMQ ones (``csrc/ggml_mmq``, see UPSTREAM.md)
and compiled for this machine (``-march=native``: the library is built where it runs, so
it takes the widest SIMD the host has, and the extension itself stays portable).
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import pathlib
import shutil
import subprocess
from dataclasses import dataclass

from freetoken.utils import init_logger

logger = init_logger(__name__)

_SRC = pathlib.Path(__file__).parent / "csrc" / "ggml_mmq"
_UNITS = (
    "ft_ggml_cpu.c",
    "src/ggml-cpu/quants.c",
    "src/ggml-cpu/arch/x86/quants.c",
    "src/ggml-quants.c",
)
_FLAGS = ("-O3", "-march=native", "-fPIC", "-DNDEBUG", "-std=gnu11")


def _cc() -> str:
    return os.environ.get("FREETOKEN_CPU_CC", os.environ.get("CC", "cc"))


def available() -> bool:
    import platform

    return platform.machine() in ("x86_64", "AMD64") and shutil.which(_cc()) is not None


def _digest() -> str:
    h = hashlib.sha256()
    for rel in _UNITS:
        h.update((_SRC / rel).read_bytes())
    for p in sorted((_SRC / "src").glob("*.h")) + sorted((_SRC / "src" / "ggml-cpu").glob("*.h")):
        h.update(p.read_bytes())
    h.update(" ".join(_FLAGS).encode())
    return h.hexdigest()[:16]


@functools.cache
def _lib() -> ctypes.CDLL:
    from torch.utils.cpp_extension import _get_build_directory

    out = pathlib.Path(f"{_get_build_directory('freetoken_ggml_cpu', verbose=False)}-{_digest()}")
    out.mkdir(parents=True, exist_ok=True)
    so = out / "libft_ggml_cpu.so"
    if not so.exists():
        logger.info("ggml_cpu: building llama.cpp CPU dot products (once)")
        tmp = out / f"libft_ggml_cpu.so.tmp{os.getpid()}"
        cmd = [
            _cc(), *_FLAGS, "-shared", "-Wl,-z,defs",
            f"-I{_SRC / 'include'}", f"-I{_SRC / 'src'}", f"-I{_SRC / 'src' / 'ggml-cpu'}",
            *[str(_SRC / u) for u in _UNITS], "-lm", "-o", str(tmp),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ggml_cpu: build failed:\n{' '.join(cmd)}\n{res.stderr[-6000:]}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    vpp, ip, i64p = ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64)
    lib.ft_ggml_cpu_lookup.argtypes = [ctypes.c_int, vpp, ip, vpp, i64p, i64p]
    lib.ft_ggml_cpu_lookup.restype = ctypes.c_int
    return lib


@dataclass(frozen=True)
class RowDot:
    """How the CPU computes rows of one weight type: addresses for the C++ executor."""

    vec_dot: int  # void (*)(int n, float *s, size_t bs, const void *x, size_t bx, const void *y, size_t by, int nrc)
    quantize: int  # void (*)(const float *x, void *y, int64_t k): activations -> ``act_type``
    act_type: int
    act_block: int  # elements per activation block
    act_block_bytes: int

    def act_row_bytes(self, k: int) -> int:
        assert k % self.act_block == 0, (k, self.act_block)
        return k // self.act_block * self.act_block_bytes


@functools.cache
def row_dot(ggml_type: int) -> RowDot | None:
    """The CPU row dot of ``ggml_type``, or None when no vendored kernel serves it."""
    if not available():
        return None
    lib = _lib()
    dot, quant = ctypes.c_void_p(), ctypes.c_void_p()
    vdt, blk, blk_bytes = ctypes.c_int(), ctypes.c_int64(), ctypes.c_int64()
    rc = lib.ft_ggml_cpu_lookup(int(ggml_type), ctypes.byref(dot), ctypes.byref(vdt), ctypes.byref(quant),
                                ctypes.byref(blk), ctypes.byref(blk_bytes))
    if rc != 0:
        return None
    return RowDot(dot.value, quant.value, vdt.value, blk.value, blk_bytes.value)


__all__ = ["RowDot", "available", "row_dot"]
