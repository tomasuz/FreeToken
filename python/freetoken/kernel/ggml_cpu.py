"""llama.cpp's CPU dot products for ggml block types, built with the host C compiler.

The CPU MoE executor (``csrc/cpu_moe``) computes GGUF experts row by row with these: per
weight type a ``vec_dot`` over one row, the activation type it pairs with (Q8_0 or Q8_K)
and that type's row quantizer -- what ``ggml_get_type_traits_cpu()`` answers inside
llama.cpp. The sources are vendored with the MMQ ones (``csrc/ggml_mmq``, see UPSTREAM.md)
and compiled for this machine (``-march=native`` on x86-64, ``-mcpu=native`` on aarch64: the
library is built where it runs, so it takes the widest SIMD the host has, and the extension
itself stays portable).
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
# llama.cpp's SIMD dot products per CPU family (the portable C ones build everywhere)
_ARCH_UNITS = {"x86_64": "src/ggml-cpu/arch/x86/quants.c", "aarch64": "src/ggml-cpu/arch/arm/quants.c"}
_MACHINES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}
_FLAGS = ("-O3", "-fPIC", "-DNDEBUG", "-std=gnu11")


def _arch() -> str | None:
    import platform

    return _MACHINES.get(platform.machine().lower())


def _units() -> tuple[str, ...]:
    return ("ft_ggml_cpu.c", "src/ggml-cpu/quants.c", _ARCH_UNITS[_arch()], "src/ggml-quants.c")


def _arch_flags() -> tuple[str, ...]:
    # tune for the host it builds on: x86 takes -march, aarch64 gcc/clang -mcpu
    return ("-march=native",) if _arch() == "x86_64" else ("-mcpu=native",)


def _cc() -> str:
    return os.environ.get("FREETOKEN_CPU_CC", os.environ.get("CC", "cc"))


def available() -> bool:
    return _arch() is not None and shutil.which(_cc()) is not None


def _digest() -> str:
    h = hashlib.sha256()
    for rel in _units():
        h.update((_SRC / rel).read_bytes())
    for p in sorted((_SRC / "src").glob("*.h")) + sorted((_SRC / "src" / "ggml-cpu").glob("*.h")):
        h.update(p.read_bytes())
    h.update(" ".join(_FLAGS + _arch_flags()).encode())
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
            _cc(), *_FLAGS, *_arch_flags(), "-shared", "-Wl,-z,defs",
            f"-I{_SRC / 'include'}", f"-I{_SRC / 'src'}", f"-I{_SRC / 'src' / 'ggml-cpu'}",
            *[str(_SRC / u) for u in _units()], "-lm", "-o", str(tmp),
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"ggml_cpu: build failed:\n{' '.join(cmd)}\n{res.stderr[-6000:]}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    vpp, ip, i64p = ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int64)
    for fn in (lib.ft_ggml_cpu_lookup, lib.ft_ggml_cpu_lookup_generic):
        fn.argtypes = [ctypes.c_int, vpp, ip, vpp, i64p, i64p]
        fn.restype = ctypes.c_int
    lib.ft_ggml_quantize_ref.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64]
    lib.ft_ggml_quantize_ref.restype = ctypes.c_int
    return lib


def quantize_ref(ggml_type: int, x_ptr: int, y_ptr: int, k: int) -> bool:
    """llama.cpp's reference quantizer of ``ggml_type`` over ``k`` floats at ``x_ptr`` into
    ``y_ptr`` (for tests); False for a type it does not cover."""
    return _lib().ft_ggml_quantize_ref(int(ggml_type), x_ptr, y_ptr, int(k)) == 0


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
def row_dot(ggml_type: int, generic: bool = False) -> RowDot | None:
    """The CPU row dot of ``ggml_type``, or None when no vendored kernel serves it.
    ``generic``: llama.cpp's portable C version instead of the SIMD one (a reference)."""
    if not available():
        return None
    lib = _lib()
    dot, quant = ctypes.c_void_p(), ctypes.c_void_p()
    vdt, blk, blk_bytes = ctypes.c_int(), ctypes.c_int64(), ctypes.c_int64()
    lookup = lib.ft_ggml_cpu_lookup_generic if generic else lib.ft_ggml_cpu_lookup
    rc = lookup(int(ggml_type), ctypes.byref(dot), ctypes.byref(vdt), ctypes.byref(quant),
                ctypes.byref(blk), ctypes.byref(blk_bytes))
    if rc != 0:
        return None
    return RowDot(dot.value, quant.value, vdt.value, blk.value, blk_bytes.value)


__all__ = ["RowDot", "available", "quantize_ref", "row_dot"]
