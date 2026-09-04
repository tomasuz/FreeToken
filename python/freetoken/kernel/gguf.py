"""Borrowed llama.cpp GGUF dequant/GEMM CUDA kernels, JIT-compiled on first use.

The ``.cu``/``.cuh`` under ``csrc/gguf/`` are vendored verbatim from sgl-kernel
(``csrc/quantization/gguf/``), which are themselves ports of llama.cpp. We compile
them through ``torch.utils.cpp_extension.load`` (the same toolchain sglang/vllm use)
into a torch-op module and expose the handful of ops the GGUF path needs. This is a
separate, torch-native extension that sits alongside FreeToken's tvm-ffi kernels.

All ops keep the weight in its native GGUF block layout (packed ``uint8`` rows) and
dequantize *inside* the kernel -- no bf16 copy of the weight is ever materialized.
"""

from __future__ import annotations

import functools
import os
import pathlib
import shutil

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_CSRC = pathlib.Path(__file__).parent / "csrc" / "gguf"


def _host_compiler() -> str | None:
    """A host compiler nvcc + libtorch headers accept.

    The system default gcc can be too new for the torch headers (gcc 16 hard-errors),
    and on this toolchain even nvcc+gcc-13 trips a non-conformant ``typename
    decltype`` in ``List_inl.h`` once ``torch::Tensor`` is instantiated -- but nvcc
    with ``clang++`` as host compiles it cleanly. So prefer clang++, then fall back
    to an older gcc. Override with ``FREETOKEN_GGUF_HOST_CXX``.
    """
    override = os.environ.get("FREETOKEN_GGUF_HOST_CXX")
    if override:
        return override
    for cxx in ("clang++", "g++-13", "g++-14", "g++-15"):
        if shutil.which(cxx):
            return cxx
    return None


def _c_compiler_for(cxx: str) -> str:
    base = os.path.basename(cxx)
    if "clang" in base:
        return shutil.which("clang") or "clang"
    cc = base.replace("g++", "gcc")
    return shutil.which(cc) or cc

def device_arch(index: int) -> str:
    """Compile target of one visible device: a ROCm ``gfx`` name or a CUDA ``sm_XX``.

    ``gcnArchName`` carries target-ID features (``gfx1200:sramecc-:xnack-``). Those are kept
    out of the compile flag: they make the target more specific than it needs to be here,
    and a code object built for a bare ``gfx1200`` loads on any of its feature variants.
    """
    import torch

    props = torch.cuda.get_device_properties(index)
    arch = getattr(props, "gcnArchName", None)
    if arch:  # ROCm
        return arch.split(":", 1)[0]
    return f"sm_{props.major}{props.minor}"  # CUDA


def visible_device_archs() -> list[str]:
    """Distinct compile targets of the devices this process can actually see.

    Order is stable (first appearance) so the flag list -- and therefore the JIT build
    directory hash -- does not churn between runs on the same machine.
    """
    import torch

    if not torch.cuda.is_available():
        return []
    seen: list[str] = []
    for i in range(torch.cuda.device_count()):
        arch = device_arch(i)
        if arch not in seen:
            seen.append(arch)
    return seen


def _apply_arch_selection(is_hip: bool) -> list[str]:
    """Restrict the JIT build to the architectures of the devices actually present.

    The obvious route -- passing ``--offload-arch`` in ``extra_cuda_cflags`` -- does not
    work, and fails in a way that looks like success. torch computes its own arch flags
    from the flag list *before* ``extra_cuda_cflags`` is appended::

        cuda_flags += _get_rocm_arch_flags(cuda_flags)   # ours is not in here yet
        cuda_flags += extra_cuda_cflags                  # ours lands after

    so its "the caller supplied arch flags, step aside" check can never see ours, and clang
    is handed the union of both lists: every architecture this PyTorch was built for, plus
    ours. That compiles and runs -- while building a dozen code objects nothing on this
    machine can execute, at minutes apiece.

    ``PYTORCH_ROCM_ARCH`` is the lever torch does consult first, so set that instead.
    Anything already in the environment wins untouched: cross-building for a device that is
    not in this box is a legitimate thing to want, and only the caller knows they want it.

    Returns the architectures the build will target (empty when we left the choice alone).
    """
    if not is_hip:
        return []  # torch's CUDA path already derives its flags from the present devices
    if os.environ.get("PYTORCH_ROCM_ARCH"):
        return []  # an explicit choice; not ours to override
    archs = visible_device_archs()
    if not archs:
        return []  # no visible device: say nothing and let torch decide
    os.environ["PYTORCH_ROCM_ARCH"] = ";".join(archs)
    return archs


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    import torch as _t

    _is_hip = bool(getattr(_t.version, "hip", None))
    # --expt-relaxed-constexpr and -ccbin are nvcc-only; hipcc/clang rejects both.
    extra_cuda_cflags = ["-O3"] if _is_hip else ["-O3", "--expt-relaxed-constexpr"]
    archs = _apply_arch_selection(_is_hip)
    if archs:
        logger.info(f"building GGUF kernels for the devices present: {', '.join(archs)}")
    host_cxx = None if _is_hip else _host_compiler()
    if host_cxx is not None:
        # Point both nvcc's host pass (-ccbin) and torch's C++ compile (CXX) at a
        # libtorch/nvcc-compatible compiler. Force (not setdefault): the system
        # default (CXX unset -> g++) can be a gcc too new for the torch headers.
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)

    # gguf_kernel.cu carries its own PYBIND11_MODULE (appended at the end), so a
    # plain `load` of the single source compiles + binds the ggml_* ops.
    return load(
        name="freetoken_gguf_kernels",
        sources=[str(_CSRC / "gguf_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=_build_directory(archs),
        verbose=True,
    )


def _build_directory(archs: list[str]) -> str | None:
    """A build directory per set of target architectures.

    torch keys the JIT cache by extension name alone, so two processes on the same machine
    that target different devices -- which is the whole point of deriving the targets from
    what is visible -- share one directory and overwrite each other's build. The loser
    loads code compiled for a device it is not running on, and faults.

    ``None`` (no archs to key on) keeps torch's own choice.
    """
    if not archs:
        return None
    try:
        from torch.utils.cpp_extension import _get_build_directory
    except ImportError:  # private helper; if it moves, torch's default is still correct
        return None
    base = _get_build_directory("freetoken_gguf_kernels", verbose=False)
    path = f"{base}-{'-'.join(archs)}"
    os.makedirs(path, exist_ok=True)
    return path


# ---- thin typed wrappers (signatures mirror sgl_kernel.quantization.gguf) ----


def ggml_dequantize(
    weight: torch.Tensor, quant_type: int, m: int, n: int, dtype: torch.dtype | None = None
) -> torch.Tensor:
    """Dequantize a packed GGUF weight ``[m, row_bytes]`` to a dense ``[m, n]`` tensor."""
    return _module().ggml_dequantize(weight, quant_type, m, n, dtype)


def ggml_mul_mat_vec_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMVQ: small-batch GEMV with on-the-fly dequant. ``row`` = output features."""
    return _module().ggml_mul_mat_vec_a8(weight, x, quant_type, row)


def ggml_mul_mat_a8(
    weight: torch.Tensor, x: torch.Tensor, quant_type: int, row: int
) -> torch.Tensor:
    """MMQ: large-batch quantized matmul. ``row`` = output features."""
    return _module().ggml_mul_mat_a8(weight, x, quant_type, row)


def ggml_moe_a8(
    x: torch.Tensor,
    weight: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    quant_type: int,
    row: int,
    top_k: int,
    tokens: int,
) -> torch.Tensor:
    """MMQ grouped expert matmul over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8(
        x, weight, sorted_token_ids, expert_ids, num_tokens_post_padded,
        quant_type, row, top_k, tokens,
    )


def ggml_moe_a8_vec(
    x: torch.Tensor,
    weight: torch.Tensor,
    topk_ids: torch.Tensor,
    top_k: int,
    quant_type: int,
    row: int,
    tokens: int,
) -> torch.Tensor:
    """MMVQ grouped expert GEMV over stacked experts ``weight[E, row, *]``."""
    return _module().ggml_moe_a8_vec(x, weight, topk_ids, top_k, quant_type, row, tokens)


def ggml_moe_get_block_size(quant_type: int) -> int:
    return _module().ggml_moe_get_block_size(quant_type)


__all__ = [
    "device_arch",
    "visible_device_archs",
    "ggml_dequantize",
    "ggml_mul_mat_vec_a8",
    "ggml_mul_mat_a8",
    "ggml_moe_a8",
    "ggml_moe_a8_vec",
    "ggml_moe_get_block_size",
]
