"""FREETOKEN_API_LOG_DIR also logs each finished generation's reasoning (thinking),
content and tool calls, alongside the existing inbound-request log -- generation.py's
generate_events/generate_full call request_logger.log_response from the same place
they already log the request into the request ring, so every wire protocol converges
here without each adapter re-serializing its own wire response just to log it."""

from __future__ import annotations

import asyncio
import json
import os
import sys
from types import SimpleNamespace

import pytest

# Same shim as the sibling server tests: the venv may hold a non-editable install, and
# without this the file only tests the source tree when a test that does insert it
# collects first.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.message import UserReply  # noqa: E402
from freetoken.server import api_server, request_logger  # noqa: E402
from freetoken.server.generation import GenSpec, generate_events, generate_full  # noqa: E402


@pytest.fixture(autouse=True)
def served_model_name():
    """_record_generation stamps the ring row from api_server._served_model_name();
    pin it so it does not read a leftover from whichever test ran before."""
    prev = api_server._GLOBAL_STATE
    api_server._GLOBAL_STATE = SimpleNamespace(config=SimpleNamespace(served_model_name="unit-model"))
    yield
    api_server._GLOBAL_STATE = prev


@pytest.fixture
def log_dir(tmp_path, monkeypatch):
    """Point request_logger at a fresh directory and reset its once-only init latches, so
    each test gets its own log file instead of racing the process-wide writer thread/queue
    that a real server would only ever set up once."""
    monkeypatch.setattr(request_logger, "_LOG_DIR", str(tmp_path))
    monkeypatch.setattr(request_logger, "_init_done", False)
    monkeypatch.setattr(request_logger, "_init_failed", False)
    monkeypatch.setattr(request_logger, "_log_path", None)
    monkeypatch.setattr(request_logger, "_fh", None)
    monkeypatch.setattr(request_logger, "_worker", None)
    monkeypatch.setattr(request_logger, "_warned", False)
    yield tmp_path


def _records(log_dir) -> list[dict]:
    assert request_logger.flush(timeout=5.0)
    files = list(log_dir.glob("requests-*.jsonl"))
    assert len(files) == 1, f"expected exactly one log file, found {files}"
    return [json.loads(line) for line in files[0].read_text().splitlines()]


class FakeState:
    """Yields canned acks in place of the scheduler; carries only what the generation
    helpers read (`config.reasoning_parser`, `config.served_model_name`, `wait_for_ack`)."""

    def __init__(self, replies: list[UserReply], reasoning_parser: str | None = "qwen3") -> None:
        self.config = SimpleNamespace(
            model_path="/m",
            served_model_name="unit-model",
            tool_call_parser="llama3",
            reasoning_parser=reasoning_parser,
        )
        self._replies = replies

    def new_user(self) -> int:
        return 42

    async def wait_for_ack(self, uid: int):
        assert uid == 42
        for reply in self._replies:
            yield reply


def _ack(prompt: int = 0, completion: int = 0, out: str = "", finished: bool = False) -> UserReply:
    return UserReply(
        uid=42,
        incremental_output=out,
        finished=finished,
        prompt_tokens_delta=prompt,
        completion_tokens_delta=completion,
        finish_reason="stop" if finished else None,
    )


def _spec() -> GenSpec:
    return GenSpec(messages=[{"role": "user", "content": "hi"}], sampling_params=SimpleNamespace())


def test_non_stream_response_log_carries_reasoning_and_content(log_dir):
    st = FakeState([
        _ack(prompt=5, out="<think>because 2+2</think>"),
        _ack(completion=1, out="4", finished=True),
    ])
    result = asyncio.run(generate_full(42, _spec(), st, source="/v1/chat/completions"))
    assert result.reasoning == "because 2+2" and result.content == "4"

    resp = [r for r in _records(log_dir) if r["kind"] == "response"]
    assert len(resp) == 1
    r = resp[0]
    assert r["endpoint"] == "/v1/chat/completions" and r["uid"] == 42
    assert r["reasoning"] == "because 2+2"
    assert r["content"] == "4"
    assert r["finish_reason"] == "stop"
    assert r["tool_calls"] is None
    assert r["error"] is None
    assert (r["prompt_tokens"], r["completion_tokens"]) == (5, 1)


def test_stream_response_log_accumulates_deltas_across_the_whole_stream(log_dir):
    st = FakeState([
        _ack(prompt=3, out="<think>reasoning bit</think>"),
        _ack(completion=1, out="ans", finished=False),
        _ack(completion=1, out="wer", finished=True),
    ])

    async def drain():
        async for _ev in generate_events(42, _spec(), st, source="/v1/messages"):
            pass

    asyncio.run(drain())
    resp = [r for r in _records(log_dir) if r["kind"] == "response"]
    assert len(resp) == 1
    assert resp[0]["reasoning"] == "reasoning bit"
    assert resp[0]["content"] == "answer"  # both content deltas concatenated
    assert resp[0]["endpoint"] == "/v1/messages"


def test_request_and_response_records_both_land_in_the_same_file(log_dir):
    from freetoken.server.api_models import ChatCompletionRequest

    req = ChatCompletionRequest(model="m", messages=[{"role": "user", "content": "hi"}])
    request_logger.log_request("/v1/chat/completions", req)
    st = FakeState([_ack(prompt=1, completion=1, out="ok", finished=True)], reasoning_parser=None)
    asyncio.run(generate_full(42, _spec(), st, source="/v1/chat/completions"))

    kinds = [r["kind"] for r in _records(log_dir)]
    assert kinds == ["request", "response"]


def test_a_generation_error_is_logged_as_a_response_record_carrying_the_error(log_dir):
    st = FakeState([UserReply(uid=42, incremental_output="", finished=True, error="boom")])
    with pytest.raises(Exception):
        asyncio.run(generate_full(42, _spec(), st, source="/v1/chat/completions"))
    resp = [r for r in _records(log_dir) if r["kind"] == "response"]
    assert resp[-1]["error"] == "boom"


def test_no_source_logs_neither_request_nor_response(log_dir):
    st = FakeState([_ack(prompt=1, completion=1, out="z", finished=True)], reasoning_parser=None)
    asyncio.run(generate_full(42, _spec(), st))  # no source: opts out, as for the request ring
    assert request_logger.flush(timeout=5.0)
    assert not list(log_dir.glob("requests-*.jsonl"))


def test_disabled_by_default_generation_still_works(monkeypatch):
    monkeypatch.setattr(request_logger, "_LOG_DIR", None)
    # reasoning_parser=None: plain output, unrelated to what this test checks (logging).
    st = FakeState([_ack(prompt=1, completion=1, out="z", finished=True)], reasoning_parser=None)
    result = asyncio.run(generate_full(42, _spec(), st, source="/v1/chat/completions"))
    assert result.content == "z"  # logging silently skipped, generation unaffected
