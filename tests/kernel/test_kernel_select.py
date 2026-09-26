"""kernel_select: the fastest equivalent path per shape, measured once on this GPU."""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _paths(calls):
    a = torch.randn(1024, 1024, device="cuda")

    def slow():
        calls.append("slow")
        y = a
        for _ in range(20):
            y = y @ a
        return torch.ones(4, device="cuda")

    def fast():
        calls.append("fast")
        return torch.ones(4, device="cuda")

    return {"slow": slow, "fast": fast}


def test_measures_once_then_keeps_the_fastest(monkeypatch):
    from freetoken.kernel import kernel_select

    monkeypatch.setattr(kernel_select, "_MODE", "auto")
    monkeypatch.setattr(kernel_select, "_decisions", {})
    calls = []
    out = kernel_select.select(("t", 1), _paths(calls), default="slow")
    assert torch.equal(out, torch.ones(4, device="cuda"))
    assert kernel_select.decisions()[("t", 1)] == "fast"
    calls.clear()
    kernel_select.select(("t", 1), _paths(calls), default="slow")
    assert calls == ["fast"]  # decided: no more measuring


def test_forced_mode_and_capture_default(monkeypatch):
    from freetoken.kernel import kernel_select

    monkeypatch.setattr(kernel_select, "_decisions", {})
    monkeypatch.setattr(kernel_select, "_MODE", "slow")
    calls = []
    kernel_select.select(("t", 2), _paths(calls), default="fast")
    assert calls == ["slow"] and ("t", 2) not in kernel_select.decisions()

    monkeypatch.setattr(kernel_select, "_MODE", "auto")
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    calls.clear()
    kernel_select.select(("t", 3), _paths(calls), default="slow")
    assert calls == ["slow"] and ("t", 3) not in kernel_select.decisions()


def test_default_stays_within_noise(monkeypatch):
    from freetoken.kernel import kernel_select

    monkeypatch.setattr(kernel_select, "_MODE", "auto")
    monkeypatch.setattr(kernel_select, "_decisions", {})
    t = iter([1.0, 0.98])  # "b" 2 % faster than the default "a": noise, keep "a"
    monkeypatch.setattr(kernel_select, "_time", lambda fn: (next(t), fn()))
    kernel_select.select(("t", 4), {"a": lambda: 1, "b": lambda: 2}, default="a")
    assert kernel_select.decisions()[("t", 4)] == "a"
