"""Where a decode step's time actually goes, asked with stream events.

A diagnostic, off unless ``FREETOKEN_PHASE_TIMING`` names a step interval. It exists
because an arithmetic estimate of the expert path can be an order of magnitude away from
the measured step and leave no way to tell which of the remaining phases is responsible.
Events, not a host clock: everything here is asynchronous, so a clock around a phase
measures the launch and nothing else.

Capture-unsafe by construction (recording an event is not a graph node this stack will
replay, and reading one is a host sync), so a captured run reports nothing and says so.
"""

from __future__ import annotations

import os
from collections import defaultdict

import torch

_INTERVAL = int(os.getenv("FREETOKEN_PHASE_TIMING", "0") or 0)
ENABLED = _INTERVAL > 0
# Which phases to instrument, empty meaning all of them. A pair of events is not free, so
# comparing two phases is only sound when both runs carry the same number of them: name
# the set, keep the count equal, and the two passes can be read side by side.
_ONLY = frozenset(n for n in os.getenv("FREETOKEN_PHASE_ONLY", "").split(",") if n)

_totals: dict[str, float] = defaultdict(float)
_pending: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []
_steps = 0
_sampled = 0
_arming = ENABLED  # only a sampled step carries events; see below


class phase:
    """Time one phase of the step, or do nothing at all when the flag is not set.

    Instrumented on one step in ``FREETOKEN_PHASE_TIMING``, never on the rest. A pair of
    events costs a few hundred microseconds on this stack -- with a phase inside every
    layer that is more than the phases being measured, and the first attempt at this
    doubled the step and reported the instrument rather than the model. Sampling keeps the
    cost off 49 steps in 50 and the numbers describe a step of ordinary speed.
    """

    __slots__ = ("_name", "_start")

    def __init__(self, name: str) -> None:
        self._name = name
        self._start = None

    def __enter__(self):
        if _ONLY and self._name not in _ONLY:
            return self
        if _arming and not torch.cuda.is_current_stream_capturing():
            self._start = torch.cuda.Event(enable_timing=True)
            self._start.record()
        return self

    def __exit__(self, *exc):
        if self._start is not None:
            end = torch.cuda.Event(enable_timing=True)
            end.record()
            _pending.append((self._name, self._start, end))
        return False


def step_done(logger=None) -> None:
    """Close a step: arm the next sample, and report what the samples so far say."""
    global _steps, _sampled, _arming
    if not ENABLED or torch.cuda.is_current_stream_capturing():
        return
    _steps += 1
    if not _arming:
        # The next sample is the one that lands on the interval.
        _arming = _steps % _INTERVAL == _INTERVAL - 1
        return
    _arming = False
    _sampled += 1
    torch.cuda.synchronize()
    for name, start, end in _pending:
        _totals[name] += start.elapsed_time(end)
    _pending.clear()
    if _sampled % 10:
        return
    total = sum(_totals.values())
    parts = ", ".join(
        f"{name}={ms / _sampled:.2f} ms ({100 * ms / total:.0f}%)"
        for name, ms in sorted(_totals.items(), key=lambda kv: -kv[1])
    )
    line = (f"decode phases over {_sampled} sampled steps (of {_steps}): {parts}; "
            f"sum={total / _sampled:.2f} ms/step")
    if logger is not None:
        logger.info_rank0(line)
    else:
        print(line, flush=True)


__all__ = ["ENABLED", "phase", "step_done"]
