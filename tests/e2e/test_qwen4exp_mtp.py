"""MTP self-speculative decode on a Qwen3.8 (qwen4exp) GGUF, end to end.

Gated behind ``needs_weights``: set ``FREETOKEN_QWEN4EXP_GGUF`` to the model's first GGUF shard
and ``FREETOKEN_QWEN4EXP_MTP`` to llama.cpp's separate MTP GGUF; run on an idle GPU (the model
needs the whole card).

Checks:
* greedy output with MTP matches plain greedy decoding. A verify scores several tokens at
  once and the batched kernels round differently from the one-token ones (logits move by a
  few tenths), so a near-tie argmax may flip: the runs must agree until plain greedy's
  top two are within ``FREETOKEN_TEST_MTP_TIE`` logits, and there MTP must pick one of the
  near-tied tokens;
* the head's drafts are accepted (the model's own MTP): over one short answer the rate
  depends on the text (0.4-0.75 seen), a head fed the wrong rows or weights stays near 0,
  so the bar is ``FREETOKEN_TEST_MTP_ACCEPT`` (default 0.35);
* captured verify graphs give the same tokens as the eager verify.

Env knobs:
  FREETOKEN_QWEN4EXP_GGUF, FREETOKEN_QWEN4EXP_MTP   model and MTP head (required)
  FREETOKEN_TEST_CTX          context window (default 131072)
  FREETOKEN_TEST_MTP_TOKENS   tokens generated per run (default 96)
  FREETOKEN_TEST_MTP_EXACT    leading tokens graphed and eager verify must share (default 24)
  FREETOKEN_TEST_MTP_ACCEPT   minimum draft acceptance per prompt (default 0.35)
  FREETOKEN_TEST_MTP_TIE      logit gap under which plain greedy's choice counts as a tie (default 1.0)
  FREETOKEN_GGUF_CPU_ASSIST   1 to run it with the CPU computing the non-resident experts
"""

from __future__ import annotations

import os

import pytest
import torch

MODEL = os.environ.get("FREETOKEN_QWEN4EXP_GGUF")
MTP = os.environ.get("FREETOKEN_QWEN4EXP_MTP")

pytestmark = [
    pytest.mark.needs_weights,
    pytest.mark.slow,
    pytest.mark.skipif(not (MODEL and MTP), reason="set FREETOKEN_QWEN4EXP_GGUF and FREETOKEN_QWEN4EXP_MTP"),
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU"),
]

PROMPTS = [
    "Write a Python class implementing an LRU cache with O(1) get and put, with docstrings.",
    "Explain in three sentences why the sky is blue.",
]


@pytest.fixture(scope="module")
def llm():
    from freetoken.llm import LLM

    ctx = int(os.environ.get("FREETOKEN_TEST_CTX", "131072"))
    return LLM(MODEL, max_running_req=1, moe_strategy="offload", moe_cache_auto=True,
               kv_reserve_tokens=ctx, max_seq_len_override=ctx, max_extend_tokens=2048,
               memory_ratio=0.85, mtp=True, mtp_model_path=MTP)


def _generate(llm, prompt: str, spec: bool, steps: list | None = None) -> list[int]:
    """Greedy tokens; ``steps`` collects each sampled step's top logits {token: logit}."""
    from freetoken.core import SamplingParams
    from freetoken.engine import sample

    os.environ["FREETOKEN_MTP_SPEC"] = "1" if spec else "0"
    orig = sample.Sampler.sample
    if steps is not None:
        def record(self, logits, args):
            v, i = logits[-1].float().topk(8)
            steps.append(dict(zip(i.tolist(), v.tolist())))
            return orig(self, logits, args)

        sample.Sampler.sample = record
    try:
        text = llm.tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                                 add_generation_prompt=True)
        n = int(os.environ.get("FREETOKEN_TEST_MTP_TOKENS", "96"))
        out = llm.generate([text], SamplingParams(max_tokens=n, ignore_eos=True))
        return out[0]["token_ids"]
    finally:
        sample.Sampler.sample = orig
        os.environ.pop("FREETOKEN_MTP_SPEC", None)


def _common_prefix(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


@pytest.mark.parametrize("prompt", PROMPTS)
def test_mtp_matches_greedy_and_drafts_are_accepted(llm, prompt):
    steps: list[dict[int, float]] = []
    plain = _generate(llm, prompt, spec=False, steps=steps)
    spec = _generate(llm, prompt, spec=True)
    d = _common_prefix(plain, spec)
    if d < len(plain):  # the first difference must be a near-tie of plain greedy
        tie = float(os.environ.get("FREETOKEN_TEST_MTP_TIE", "1.0"))
        top = steps[d]
        best = max(top.values())
        assert spec[d] in top and best - top[spec[d]] < tie, (d, plain[: d + 1], spec[: d + 1], top)
    stat = llm._mtp4_stat
    assert not getattr(llm, "_mtp4_disabled", False), "a speculative step failed"
    assert stat["forwards"] > 0, "the MTP loop never engaged"
    accept = float(os.environ.get("FREETOKEN_TEST_MTP_ACCEPT", "0.35"))
    assert stat["accepted"] / max(stat["verified"], 1) > accept, stat


def test_verify_graphs_match_the_eager_verify(llm):
    prompt = PROMPTS[0]
    graphed = _generate(llm, prompt, spec=True)
    assert llm._mtp4_vg, "the verify graphs were not captured"
    saved, llm._mtp4_vg = llm._mtp4_vg, False  # False: verify eagerly, do not capture again
    try:
        eager = _generate(llm, prompt, spec=True)
    finally:
        llm._mtp4_vg = saved
    # the eager verify writes the draft ids into the request's buffer: drafts near the end
    # of max_tokens must not run past it
    assert not getattr(llm, "_mtp4_disabled", False), "a speculative step failed"
    exact = int(os.environ.get("FREETOKEN_TEST_MTP_EXACT", "24"))
    assert _common_prefix(graphed, eager) >= min(exact, len(graphed)), (graphed[:exact], eager[:exact])
