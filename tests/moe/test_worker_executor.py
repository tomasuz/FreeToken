"""Experts computed by a worker process on another device.

The claim to defend is that moving the compute out of the engine's process changes nothing
a caller can observe: same banks, same routing, same numbers. If that holds, the placement
planner is free to send a layer wherever it is cheapest, and a device the engine's own
process cannot drive stops being unusable.
"""

from __future__ import annotations

import pytest
import torch

from freetoken.models.gguf.dequant import GGML_Q4_0, row_bytes


def _q4_0_bank(*shape: int, blocks: int) -> torch.Tensor:
    """Random but VALID Q4_0 bytes: each 18-byte block is ``half d`` + 16 nibble bytes.

    Randomising all 18 would put random bit patterns in the fp16 scale, and a good share of
    those decode to NaN -- the comparison would then be NaN against NaN and prove nothing.
    """
    buf = torch.randint(0, 256, (*shape, blocks, 18), dtype=torch.uint8)
    buf[..., :2] = torch.tensor([0.05], dtype=torch.float16).view(torch.uint8)
    return buf.reshape(*shape, blocks * 18)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_worker_matches_in_process_compute_bitwise():
    """A worker on the same device must be bit-identical to computing here.

    Same device deliberately: it isolates the process boundary as the only variable, so a
    mismatch means the shared buffers or the handoff are wrong, not the hardware.
    """
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.moe.fused_q4_0 import fused_experts_gguf
    from freetoken.moe.worker_executor import WorkerMoeExecutor

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    device = torch.device("cuda")
    torch.manual_seed(0)
    E, H, I, tokens, top_k = 4, 64, 64, 3, 2

    gate_up = _q4_0_bank(E, 2 * I, blocks=H // 32)
    down = _q4_0_bank(E, H, blocks=I // 32)

    x = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
    topk_w = torch.rand(tokens, top_k, dtype=torch.float32, device=device)
    topk_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32, device=device)

    here = fused_experts_gguf(
        x, gate_up.to(device), down.to(device), topk_w, topk_ids.clone(), "silu", GGML_Q4_0
    )

    with WorkerMoeExecutor(
        torch.cuda.current_device(),
        {"gate_up": gate_up, "down": down},
        ggml_type=GGML_Q4_0,
        activation="silu",
        max_batch=8,
        hidden_size=H,
        top_k=top_k,
    ) as worker:
        there = worker.decode(x, topk_w, topk_ids.clone())

    torch.testing.assert_close(there, here, rtol=0, atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_worker_reports_a_failed_start_instead_of_hanging():
    """A device that cannot run the kernel has to surface the child's error, not a timeout."""
    from freetoken.moe.worker_executor import WorkerMoeExecutor

    E, H, I = 2, 64, 64
    banks = {
        "gate_up": _q4_0_bank(E, 2 * I, blocks=H // 32),
        "down": _q4_0_bank(E, H, blocks=I // 32),
    }
    with pytest.raises((RuntimeError, TimeoutError)) as excinfo:
        WorkerMoeExecutor(
            999,  # no such device: the child must fail, and say so
            banks,
            ggml_type=GGML_Q4_0,
            max_batch=4,
            hidden_size=H,
            top_k=1,
            env={"FREETOKEN_WORKER_START_PROBE": "1"},
        ).close()
    assert "999" in str(excinfo.value)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs an accelerator")
def test_worker_rejects_a_batch_it_cannot_hold():
    """max_batch sizes the shared buffers, so an oversized step must fail loudly here
    rather than silently truncating in the worker."""
    from freetoken.moe.fused_q4_0 import fused_experts_gguf  # noqa: F401  (kernel warm)
    from freetoken.moe.worker_executor import WorkerMoeExecutor

    E, H, I, top_k = 2, 64, 64, 1
    banks = {
        "gate_up": _q4_0_bank(E, 2 * I, blocks=H // 32),
        "down": _q4_0_bank(E, H, blocks=I // 32),
    }
    device = torch.device("cuda")
    with WorkerMoeExecutor(
        torch.cuda.current_device(), banks, ggml_type=GGML_Q4_0,
        max_batch=2, hidden_size=H, top_k=top_k,
    ) as worker:
        big = torch.randn(4, H, dtype=torch.bfloat16, device=device)
        w = torch.rand(4, top_k, dtype=torch.float32, device=device)
        ids = torch.zeros(4, top_k, dtype=torch.int32, device=device)
        with pytest.raises(AssertionError, match="max_batch"):
            worker.decode(big, w, ids)
