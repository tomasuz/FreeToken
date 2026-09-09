"""The worker handshake as capturable kernels.

``kernel/csrc/handshake`` explains why this exists rather than the stream memory
operations it replaces: those are silently not recorded into a graph on this runtime.
Build policy is shared with the other JIT extensions -- see ``kernel/gguf.py``.
"""

from __future__ import annotations

import functools
import os
import shutil

import pathlib
import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

_CSRC = pathlib.Path(__file__).parent / "csrc" / "handshake"

# A wedge guard, not a timeout: a worker that has died must not leave a graph replay
# spinning with no way to interrupt it. Large enough that a slow step is never cut short.
DEFAULT_MAX_SPINS = 1 << 34


@functools.cache
def _module():
    from torch.utils.cpp_extension import load

    from .gguf import _apply_arch_selection, _build_directory, _c_compiler_for, _host_compiler

    is_hip = bool(getattr(torch.version, "hip", None))
    extra_cuda_cflags = ["-O3"] if is_hip else ["-O3", "--expt-relaxed-constexpr"]
    archs = _apply_arch_selection(is_hip)
    host_cxx = None if is_hip else _host_compiler()
    if host_cxx is not None:
        cxx_path = shutil.which(host_cxx) or host_cxx
        extra_cuda_cflags += ["-ccbin", cxx_path]
        os.environ["CXX"] = cxx_path
        os.environ["CC"] = _c_compiler_for(cxx_path)
    return load(
        name="freetoken_handshake",
        sources=[str(_CSRC / "handshake_kernel.cu")],
        extra_include_paths=[str(_CSRC)],
        extra_cuda_cflags=extra_cuda_cflags,
        build_directory=_build_directory(archs, "freetoken_handshake"),
        verbose=True,
    )


def doorbell(done_addr: int, ready_addr: int, slot: int) -> None:
    """Clear this slot's completion and raise its request, on the current stream."""
    _module().doorbell(int(done_addr), int(ready_addr), int(slot))


def wait(done_addr: int, slot: int, max_spins: int = DEFAULT_MAX_SPINS) -> None:
    """Hold the current stream until the worker reports this slot done."""
    _module().wait(int(done_addr), int(slot), int(max_spins))


def replays_correctly() -> bool:
    """Does a captured graph actually perform this handshake?

    Asked by capture, not by construction: the operations this replaces answered yes to
    every check available outside a capture and still recorded nothing inside one, which
    cost a long search. So the question is put the only way that cannot be wrong -- a
    capture, a replay, and a look at the word afterwards.
    """
    flag = torch.zeros(2, dtype=torch.int64).pin_memory()
    addr = flag.data_ptr()
    try:
        side = torch.cuda.Stream()
        with torch.cuda.stream(side):  # warm-up: torch requires one before capture
            doorbell(addr, addr + 8, 0)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            doorbell(addr, addr + 8, 0)
        flag.zero_()
        graph.replay()
        torch.cuda.synchronize()
        return bool(flag[1].item() == 1)
    except Exception as exc:
        logger.info(f"handshake kernels unavailable ({type(exc).__name__}: {exc}); "
                    "decode will not be captured with a worker attached")
        return False


__all__ = ["doorbell", "wait", "replays_correctly", "DEFAULT_MAX_SPINS"]
