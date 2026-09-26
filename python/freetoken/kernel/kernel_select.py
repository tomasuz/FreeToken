"""Pick the fastest of equivalent kernel paths by timing them on this GPU, once per shape.

Which GGUF matmul path wins at a few tokens -- llama.cpp's current MMVQ or the older
kernels, with the conversions each needs around it -- depends on the GPU, the ggml type
and the shape; a rule measured on one card is wrong on the next. So the first eager call
of a shape runs every candidate a few times under CUDA events and keeps the fastest; later
calls, and captured graphs (whose warm-up is eager), take the decision. A shape first seen
under capture, before any decision, falls back to the caller's default. The default also
stays unless another candidate beats it by ``_MARGIN``: paths within timing noise of each
other would otherwise flip from one start to the next, and with them the rounding of the
output.

``FREETOKEN_GGUF_KERNEL_SELECT``: ``default`` (the default) takes each caller's rule;
``auto`` measures; a candidate name (``old``, ``new``) forces it wherever it is offered.
Measuring is opt-in: it times eager launches, while decode mostly replays CUDA graphs,
where launch overhead is gone, so its picks are not guaranteed to win there; it also makes
the choice (and with it the output's rounding) depend on each start's timings. Use it on a
GPU the rules were not measured on, and compare end to end.
"""

from __future__ import annotations

import os
from typing import Callable, Hashable, TypeVar

import torch

from freetoken.utils import init_logger

logger = init_logger(__name__)

T = TypeVar("T")

_MODE = os.getenv("FREETOKEN_GGUF_KERNEL_SELECT", "default").strip().lower()
_REPS = 3
_MARGIN = 0.05  # a candidate must be this much faster than the default to replace it
_decisions: dict[Hashable, str] = {}


def decisions() -> dict[Hashable, str]:
    """The choices made so far (shape key -> candidate name)."""
    return dict(_decisions)


def _time(fn: Callable[[], T]) -> tuple[float, T]:
    out = fn()  # warm: first launch, lazy builds
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_REPS):
        out = fn()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) / _REPS, out


def select(key: Hashable, candidates: dict[str, Callable[[], T]], default: str) -> T:
    """Run the candidate chosen for ``key`` (measuring once if none is chosen yet) and
    return its result. Every candidate must compute the same result without side effects."""
    if _MODE in candidates:
        return candidates[_MODE]()
    if _MODE != "auto":
        return candidates[default if default in candidates else next(iter(candidates))]()
    name = _decisions.get(key)
    if name is None:
        if len(candidates) == 1 or torch.cuda.is_current_stream_capturing() or not torch.cuda.is_available():
            return candidates[default if default in candidates else next(iter(candidates))]()
        times, outs = {}, {}
        for cand, fn in candidates.items():
            times[cand], outs[cand] = _time(fn)
        name = min(times, key=times.get)
        if default in times and times[name] > times[default] * (1 - _MARGIN):
            name = default
        _decisions[key] = name
        logger.debug(f"kernel_select {key}: {name} ({', '.join(f'{c} {t * 1e3:.0f} us' for c, t in times.items())})")
        return outs[name]
    return candidates[name]()


__all__ = ["decisions", "select"]
