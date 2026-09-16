"""/metrics renders the /v1/stats numbers for a scraper without adding instrumentation.

The cases worth pinning are the ones a scraper turns into wrong dashboards rather than into
errors: a pool that does not exist for this model must be *absent*, not zero (zero reads as
"empty pool" on a graph); a backend that is not up yet must still answer 200 so a restart is a
gap in the series instead of a collector error; and the exposition text has to stay parseable,
since a malformed line makes the whole scrape fail, not just that sample.
"""
from __future__ import annotations

import os
import re
import sys

import pytest

# Same shim as the sibling server tests: the venv may hold a non-editable install, and without
# this the file only tests the source tree when a test that does insert it collects first.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_PY = os.path.join(_ROOT, "python")
if _PY not in sys.path:
    sys.path.insert(0, _PY)

from freetoken.server.metrics import CONTENT_TYPE, render_prometheus  # noqa: E402

# A colon is legal in an exposition name -- llama.cpp's metrics use one, and this
# endpoint keeps those names verbatim.
_SAMPLE = re.compile(r'^[a-zA-Z_:][a-zA-Z0-9_:]*(\{[^}]*\})? -?[0-9.eE+-]+$')
_META = re.compile(r'^# (HELP|TYPE) [a-zA-Z_:][a-zA-Z0-9_:]* ')


def _doc(**over):
    doc = {
        "instance_id": "i",
        "uptime_s": 3600,
        "model": {"id": "ornith-1.5-35b", "ctx": 262144, "attn": "hybrid_linear", "moe": True},
        "kv": {"used_pages": 3000, "total_pages": 30000, "page_size": 2},
        "mamba": {"used_slots": 4, "total_slots": 8},
        "swa": None,
        "vram_bytes": 123,
        "gpus": [{"index": 0, "name": 'AMD "RX" 9060 XT', "total_bytes": 17095983104}],
        "throughput": {"decode_tps": 31.4, "prefill_tps": 52.8},
        "requests": {"active": 1, "completed": 17, "p95_ms": 2400, "ttft_mean_ms": 310,
                     "prompt_tokens_total": 12345, "completion_tokens_total": 67890},
    }
    doc.update(over)
    return doc


def _lines(text):
    return [ln for ln in text.splitlines() if ln]


def test_every_line_parses():
    for ln in _lines(render_prometheus(_doc())):
        assert _META.match(ln) if ln.startswith("#") else _SAMPLE.match(ln), ln


def test_help_and_type_precede_their_samples():
    text = render_prometheus(_doc())
    seen = set()
    for ln in _lines(text):
        if ln.startswith("# TYPE "):
            seen.add(ln.split()[2])
        elif not ln.startswith("#"):
            name = ln.split("{")[0].split(" ")[0]
            assert name in seen, f"{name} emitted before its # TYPE"


def test_counters_are_typed_and_suffixed():
    text = render_prometheus(_doc())
    counters = {ln.split()[2] for ln in _lines(text) if ln.startswith("# TYPE ") and ln.endswith(" counter")}
    assert counters == {
        "llamacpp:prompt_tokens_total",
        "llamacpp:tokens_predicted_total",
        "freetoken_requests_completed_total",
    }
    for name in counters:
        assert name.endswith("_total")


def test_llamacpp_names_are_kept_verbatim():
    """The point of the endpoint: an existing llama-server scrape keeps working untouched.

    telegraf turns a metric name straight into a field name, so renaming any of these
    breaks every dashboard built against llama-server. Pin the exact spellings.
    """
    text = render_prometheus(_doc())
    assert "llamacpp:prompt_tokens_total 12345" in text
    assert "llamacpp:tokens_predicted_total 67890" in text
    assert "llamacpp:predicted_tokens_seconds 31.4" in text
    assert "llamacpp:prompt_tokens_seconds 52.8" in text
    assert "llamacpp:requests_processing 1" in text


def test_unmeasured_llamacpp_metrics_are_absent_not_faked():
    """A panel pinned at 0 claims "measured, and idle"; an empty one says "not measured"."""
    from freetoken.server.metrics import _ABSENT

    text = render_prometheus(_doc())
    for name in _ABSENT:
        assert name not in text


def test_absent_pool_is_omitted_not_zeroed():
    """swa is None for a model without sliding-window attention; a 0 would graph as an empty pool."""
    text = render_prometheus(_doc())
    assert "freetoken_swa" not in text
    assert "freetoken_mamba_slots_used 4" in text

    none_pools = render_prometheus(_doc(kv=None, mamba=None))
    assert "freetoken_kv_" not in none_pools and "freetoken_mamba_" not in none_pools
    assert "freetoken_up 1" in none_pools  # the engine is still up, it just has no such pool


def test_kv_ratio_and_tokens_use_the_pool_page_size():
    text = render_prometheus(_doc())
    assert "freetoken_kv_usage_ratio 0.1" in text
    assert "freetoken_kv_tokens_used 6000" in text  # 3000 pages x page_size 2


def test_zero_total_does_not_divide_by_zero():
    text = render_prometheus(_doc(kv={"used_pages": 0, "total_pages": 0, "page_size": 1}))
    assert "freetoken_kv_usage_ratio 0.0" in text


def test_label_values_are_escaped():
    text = render_prometheus(_doc())
    assert r'name="AMD \"RX\" 9060 XT"' in text
    assert 'moe="true"' in text


@pytest.mark.parametrize("doc", [None, "not a dict", 42])
def test_backend_not_ready_reports_down(doc):
    text = render_prometheus(doc)
    assert text.strip().endswith("freetoken_up 0")
    assert "freetoken_kv_" not in text


def test_route_answers_200_even_when_stats_raise():
    """A scrape must never surface the engine's internal failure as a collector error."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from freetoken.server.control_api import register_control_routes

    def broken_state():
        raise RuntimeError("backend not up")

    app = FastAPI(version="test")
    register_control_routes(app, broken_state)
    resp = TestClient(app).get("/metrics")

    assert resp.status_code == 200
    assert resp.headers["content-type"] == CONTENT_TYPE
    assert "freetoken_up 0" in resp.text
