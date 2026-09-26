"""llama.cpp's MMQ (quantized matmul) and MMVQ (mat-vec), built with the GPU toolchain and
called through ctypes.

The vendored sources (``csrc/ggml_mmq``, see its UPSTREAM.md) are llama.cpp's own: they are
CUDA, and speak HIP through ``GGML_USE_HIP`` (``vendors/hip.h``), so they are compiled as they
are -- with hipcc on ROCm, nvcc on CUDA -- rather than through torch's hipify, which would
rewrite them. The result is a plain shared library with C entry points (``ft_mmq_moe``,
``ft_mmvq``); it links nothing of torch, and every buffer -- activations, output, scratch --
is a torch tensor passed by pointer. Without the toolchain (or when the build fails) the
older GGUF kernels serve instead.
"""

from __future__ import annotations

import ctypes
import functools
import hashlib
import os
import pathlib
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_SRC = pathlib.Path(__file__).parent / "csrc" / "ggml_mmq"
# ggml type enum names of the instantiated MMQ cases (one translation unit each)
_TYPES = ("Q4_0", "Q8_0", "Q4_K", "Q5_K", "Q6_K", "IQ3_XXS", "IQ4_NL", "IQ4_XS")
_DEFINES = ("-DGGML_USE_HIP", "-DGGML_HIP_NO_VMM", "-DNDEBUG")
_CUDA_DEFINES = ("-DGGML_CUDA_NO_VMM", "-DNDEBUG")


def _platform() -> str | None:
    """"hip" or "cuda": the runtime torch was built for, when this process sees a GPU."""
    if getattr(torch.version, "hip", None):
        return "hip"
    if getattr(torch.version, "cuda", None):
        return "cuda"
    return None


def available() -> bool:
    plat = _platform()
    return plat is not None and shutil.which(_compiler(plat)) is not None


def _hipcc() -> str:
    return os.environ.get("FREETOKEN_MMQ_HIPCC", "hipcc")


def _nvcc() -> str:
    nvcc = os.environ.get("FREETOKEN_MMQ_NVCC")
    if nvcc:
        return nvcc
    from torch.utils.cpp_extension import CUDA_HOME

    cand = pathlib.Path(CUDA_HOME) / "bin" / "nvcc" if CUDA_HOME else None
    return str(cand) if cand is not None and cand.exists() else "nvcc"


def _compiler(plat: str) -> str:
    return _hipcc() if plat == "hip" else _nvcc()


def _hipcc_env() -> dict[str, str]:
    """The environment hipcc builds these sources in: its own installation's defaults.

    ROCM_HOME/ROCM_PATH/HIP_PATH are set for torch's extension builds and may point at a
    ROCm tree whose HIP headers the system clang's HIP wrapper cannot use (no device
    fabsf/powf/min once they win the lookup). llama.cpp's sources need nothing beyond what
    hipcc finds by itself, so its build does not inherit them.
    """
    env = dict(os.environ)
    for k in ("ROCM_HOME", "ROCM_PATH", "HIP_PATH"):
        env.pop(k, None)
    return env


def _archs() -> list[str]:
    """ROCm: gfx targets; CUDA: compute capabilities as ``86``, ``90``... -- the visible
    devices', or ``PYTORCH_ROCM_ARCH`` / ``TORCH_CUDA_ARCH_LIST``."""
    if _platform() == "cuda":
        env = os.environ.get("TORCH_CUDA_ARCH_LIST")
        if env:
            return sorted({a.replace(".", "").replace("+PTX", "") for a in env.replace(",", ";").replace(" ", ";").split(";") if a})
        return sorted({"%d%d" % torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())})
    from freetoken.kernel.gguf import visible_device_archs

    env = os.environ.get("PYTORCH_ROCM_ARCH")
    if env:
        return [a for a in env.replace(",", ";").split(";") if a]
    return visible_device_archs()


def _sources_digest() -> str:
    h = hashlib.sha256()
    for p in sorted(_SRC.rglob("*")):
        if p.is_file() and p.suffix in (".h", ".cuh", ".cu"):
            h.update(str(p.relative_to(_SRC)).encode())
            h.update(p.read_bytes())
    h.update(" ".join(_DEFINES + _TYPES).encode())
    return h.hexdigest()[:16]


def _build_dir(archs: list[str]) -> pathlib.Path:
    from torch.utils.cpp_extension import _get_build_directory

    base = pathlib.Path(_get_build_directory("freetoken_ggml_mmq", verbose=False))
    path = pathlib.Path(f"{base}-{_platform()}-{'-'.join(archs)}-{_sources_digest()}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _arch_flags(archs: list[str]) -> list[str]:
    if _platform() == "cuda":
        return [f"-gencode=arch=compute_{a},code=sm_{a}" for a in archs]
    return [f"--offload-arch={a}" for a in archs]


def _compile(cc: str, src: pathlib.Path, obj: pathlib.Path, archs: list[str]) -> None:
    includes = [f"-I{_SRC / 'include'}", f"-I{_SRC / 'src'}", f"-I{_SRC / 'src' / 'ggml-cuda'}"]
    compat = ["-include", str(_SRC / "ft_compat.h")]
    if _platform() == "cuda":
        cmd = [cc, "-x", "cu", "-std=c++17", "-O3", "-Xcompiler", "-fPIC", "--expt-relaxed-constexpr",
               *_arch_flags(archs), *_CUDA_DEFINES, *compat, *includes, "-c", str(src), "-o", str(obj)]
        env = None
    else:
        cmd = [cc, "-x", "hip", "-std=c++17", "-O3", "-fPIC", *_arch_flags(archs), *_DEFINES, *compat,
               *includes, "-c", str(src), "-o", str(obj)]
        env = _hipcc_env()
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if res.returncode != 0:
        raise RuntimeError(f"ggml_mmq: compiling {src.name} failed:\n{' '.join(cmd)}\n{res.stderr[-6000:]}")


@functools.cache
def _lib() -> ctypes.CDLL:
    archs = _archs()
    if not archs:
        raise RuntimeError("ggml_mmq: no visible GPU to build for")
    out = _build_dir(archs)
    so = out / "libft_mmq.so"
    if not so.exists():
        cc = _compiler(_platform())
        units = [(_SRC / "ft_mmq.cu", out / "ft_mmq.o"),
                 (_SRC / "ft_mmvq.cu", out / "ft_mmvq.o"),
                 (_SRC / "src" / "ggml-cuda" / "quantize.cu", out / "quantize.o"),
                 (_SRC / "src" / "ggml-cuda" / "mmid.cu", out / "mmid.o")]
        for t in _TYPES:
            src = out / f"ft_mmq_inst_{t.lower()}.cu"
            src.write_text(f'#include "ggml-cuda/mmq.cuh"\n\nDECL_MMQ_CASE(GGML_TYPE_{t});\n')
            units.append((src, out / f"ft_mmq_inst_{t.lower()}.o"))
        logger.info(f"ggml_mmq: building llama.cpp MMQ/MMVQ with {_platform()} for {', '.join(archs)} ({len(units)} units, once)")
        jobs = max(1, min(len(units), (os.cpu_count() or 4) // 2))
        with ThreadPoolExecutor(jobs) as ex:
            list(ex.map(lambda u: _compile(cc, u[0], u[1], archs), units))
        tmp = out / f"libft_mmq.so.tmp{os.getpid()}"
        if _platform() == "cuda":
            cmd = [cc, "-shared", "-Xcompiler", "-fPIC", *_arch_flags(archs),
                   *[str(o) for _, o in units], "-lcudart", "-lcublas", "-o", str(tmp)]
            env = None
        else:
            cmd = [cc, "-shared", "-fPIC", *_arch_flags(archs), *[str(o) for _, o in units], "-o", str(tmp)]
            env = _hipcc_env()
        res = subprocess.run(cmd, capture_output=True, text=True, env=env)
        if res.returncode != 0:
            raise RuntimeError(f"ggml_mmq: linking failed:\n{res.stderr[-6000:]}")
        os.replace(tmp, so)
    lib = ctypes.CDLL(str(so))
    i64, vp = ctypes.c_int64, ctypes.c_void_p
    lib.ft_mmq_init.argtypes = []
    lib.ft_mmq_init.restype = ctypes.c_int
    lib.ft_mmq_supports.argtypes = [ctypes.c_int]
    lib.ft_mmq_supports.restype = ctypes.c_int
    lib.ft_mmq_moe_workspace.argtypes = [ctypes.c_int, i64, i64, i64, i64, i64]
    lib.ft_mmq_moe_workspace.restype = ctypes.c_size_t
    lib.ft_mmq_moe.argtypes = [vp, ctypes.c_int, i64, i64, i64, i64, i64, i64,
                               vp, i64, i64, vp, i64, i64, vp, vp, ctypes.c_size_t, vp]
    lib.ft_mmq_moe.restype = ctypes.c_int
    lib.ft_mmvq_workspace.argtypes = [i64, i64]
    lib.ft_mmvq_workspace.restype = ctypes.c_size_t
    lib.ft_mmvq.argtypes = [vp, vp, ctypes.c_int, ctypes.c_int, i64, i64, i64, i64, i64,
                            vp, i64, vp, i64, i64, vp, vp, ctypes.c_size_t, vp]
    lib.ft_mmvq.restype = ctypes.c_int
    return lib


@functools.cache
def prepare() -> bool:
    """Build/load the library and query the device now, outside any graph capture.
    False (never raises) when the kernels are unavailable here."""
    if not available():
        return False
    try:
        return _lib().ft_mmq_init() > 0
    except Exception as exc:  # noqa: BLE001 -- the old kernels still serve
        logger.warning(f"ggml_mmq unavailable, keeping the older GGUF kernels: {exc}")
        return False


def supports(ggml_type: int) -> bool:
    return prepare() and bool(_lib().ft_mmq_supports(int(ggml_type)))


def mmq_moe(
    weight: torch.Tensor,
    ggml_type: int,
    k: int,
    x: torch.Tensor,
    ids: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """``out[t, u] = dequant(weight[ids[t, u]]) @ x[t, u % x.shape[1]]`` via llama.cpp MMQ.

    ``weight`` is ``[slots, rows, row_bytes]`` uint8 of ggml ``ggml_type`` blocks over ``k``
    inputs; slots and rows may be strided (a slot-cache region). ``x`` is
    ``[tokens, 1 | used, k]``: one row per token shared by every used slot (gate/up), or one
    per slot (down). ``ids`` is int32 ``[tokens, used]``. ``num_experts`` is how many
    distinct experts the routing can reach (tile sizing). Returns f32 ``[tokens, used, rows]``.
    """
    lib = _lib()
    assert weight.dtype == torch.uint8 and weight.dim() == 3 and weight.stride(2) == 1, weight.shape
    slots, rows = weight.shape[0], weight.shape[1]
    tokens, used = ids.shape
    x = x.to(torch.float32).contiguous()
    assert x.dim() == 3 and x.shape[0] == tokens and x.shape[2] == k and x.shape[1] in (1, used), x.shape
    ids = ids.to(torch.int32).contiguous()
    ne11 = x.shape[1]
    ws_bytes = lib.ft_mmq_moe_workspace(int(ggml_type), k, slots, tokens, used, ne11)
    ws = torch.empty((ws_bytes,), dtype=torch.uint8, device=x.device)
    out = torch.empty((tokens, used, rows), dtype=torch.float32, device=x.device)
    rc = lib.ft_mmq_moe(
        weight.data_ptr(), int(ggml_type), k, rows, weight.stride(1), weight.stride(0), slots, num_experts,
        x.data_ptr(), ne11, ne11 * k, ids.data_ptr(), tokens, used,
        out.data_ptr(), ws.data_ptr(), ws_bytes, torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"ft_mmq_moe failed ({rc}): type {ggml_type}, strides {weight.stride()}")
    return out


# ggml_glu_op values (ggml.h) for mmvq's fused gate
GLU_SWIGLU = 2
GLU_GEGLU = 1
# the most tokens one MMVQ launch takes (MMVQ_MAX_BATCH_SIZE)
MMVQ_MAX_TOKENS = 8


def mmvq(
    weight: torch.Tensor,
    ggml_type: int,
    k: int,
    x: torch.Tensor,
    ids: torch.Tensor | None = None,
    *,
    gate: torch.Tensor | None = None,
    glu_op: int = GLU_SWIGLU,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """llama.cpp's quantized mat-vec over at most 8 tokens.

    Dense (``ids`` None): ``weight`` ``[rows, row_bytes]``, ``x`` ``[tokens, k]`` ->
    f32 ``[tokens, rows]``. Experts: ``weight`` ``[slots, rows, row_bytes]`` (strided slots
    allowed), ``x`` ``[tokens, 1 | used, k]``, ``ids`` int32 ``[tokens, used]`` -> f32
    ``[tokens, used, rows]``. ``gate`` (same shape and strides as ``weight``) fuses a gate
    projection: the result is ``act(x @ gate) * (x @ weight)``. Graph-capturable: no host sync,
    every buffer a torch allocation on the current stream.
    """
    lib = _lib()
    assert weight.dtype == torch.uint8 and weight.stride(-1) == 1, weight.shape
    x = x.to(torch.float32).contiguous()
    tokens = x.shape[0]
    assert tokens <= MMVQ_MAX_TOKENS, tokens
    if gate is not None:
        assert gate.shape == weight.shape and gate.stride() == weight.stride()
    if ids is None:
        assert weight.dim() == 2 and x.dim() == 2 and x.shape[1] == k
        rows, slots, slot_stride, ne11, used = weight.shape[0], 1, weight.stride(0) * weight.shape[0], 1, 1
        shape = (tokens, rows)
        rows_y = tokens
    else:
        assert weight.dim() == 3 and x.dim() == 3 and x.shape[2] == k
        slots, rows = weight.shape[0], weight.shape[1]
        slot_stride = weight.stride(0)
        used = ids.shape[1]
        ne11 = x.shape[1]
        ids = ids.to(torch.int32).contiguous()
        shape = (tokens, used, rows)
        rows_y = tokens * ne11
    row_stride = weight.stride(-2)
    ws_bytes = lib.ft_mmvq_workspace(k, rows_y)
    ws = torch.empty((ws_bytes,), dtype=torch.uint8, device=x.device)
    if out is None:
        out = torch.empty(shape, dtype=torch.float32, device=x.device)
    rc = lib.ft_mmvq(
        weight.data_ptr(), gate.data_ptr() if gate is not None else None, int(glu_op), int(ggml_type),
        k, rows, row_stride, slot_stride, slots, x.data_ptr(), ne11,
        ids.data_ptr() if ids is not None else None, tokens, used,
        out.data_ptr(), ws.data_ptr(), ws_bytes, torch.cuda.current_stream(x.device).cuda_stream,
    )
    if rc != 0:
        raise RuntimeError(f"ft_mmvq failed ({rc}): type {ggml_type}, strides {weight.stride()}")
    return out


__all__ = ["GLU_GEGLU", "GLU_SWIGLU", "MMVQ_MAX_TOKENS", "available", "mmq_moe", "mmvq", "prepare", "supports"]
