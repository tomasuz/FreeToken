"""Prometheus exposition for ``/metrics``.

A pure renderer over the ``/v1/stats`` document: every number here already exists in
:func:`freetoken.server.stats.build_stats`, so scraping adds no instrumentation to the
decode path -- it reads the same counters the desktop app polls.

Every metric llama.cpp served under the same name and meaning keeps that exact name here,
including the ``llamacpp:`` prefix. A colon is legal in an exposition name (the grammar is
``[a-zA-Z_:][a-zA-Z0-9_:]*``; the convention that reserves it for recording rules is a
style rule, not a syntax one), and telegraf turns a metric name straight into a field name
-- so renaming these would silently break every dashboard built against llama-server, for
no gain. This endpoint is a drop-in: the existing scrape config needs no edit.

Metrics with no llama.cpp counterpart are additive ``freetoken_`` ones. A scraper picks
them up as new fields without any configuration change, and they carry what actually
matters for this engine: KV pressure, recurrent-state slots, device memory. The model
identity rides a single ``freetoken_model_info`` metric instead of being repeated as a
label on every series, so a collector that turns labels into tags does not fan every
value out per model.

Three llama.cpp metrics are deliberately absent rather than faked -- see ``_ABSENT``.
"""

from __future__ import annotations

from typing import Any

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# name -> (type, help). Kept beside the renderer so HELP/TYPE and the emitted samples
# cannot drift apart.
_METRICS: tuple[tuple[str, str, str], ...] = (
    # --- llama.cpp's names, kept verbatim so an existing scrape keeps working -------------
    ("llamacpp:prompt_tokens_total", "counter", "Number of prompt tokens processed."),
    ("llamacpp:prompt_tokens_seconds", "gauge", "Average prompt throughput in tokens/s."),
    ("llamacpp:tokens_predicted_total", "counter", "Number of generation tokens processed."),
    ("llamacpp:predicted_tokens_seconds", "gauge", "Average generation throughput in tokens/s."),
    ("llamacpp:requests_processing", "gauge", "Number of requests processing."),
    # --- engine-specific, additive: new fields, no scrape-config change ------------------
    ("freetoken_up", "gauge", "1 when the engine is serving, 0 while it is still starting"),
    ("freetoken_uptime_seconds", "gauge", "Seconds since the engine became ready"),
    ("freetoken_requests_completed_total", "counter", "Requests finished since start"),
    ("freetoken_request_latency_p95_milliseconds", "gauge", "95th percentile end-to-end request latency"),
    ("freetoken_request_ttft_mean_milliseconds", "gauge", "Mean time to first token"),
    ("freetoken_kv_pages_used", "gauge", "KV cache pages in use"),
    ("freetoken_kv_pages_total", "gauge", "KV cache pages allocated"),
    ("freetoken_kv_usage_ratio", "gauge", "KV cache pages in use divided by pages allocated"),
    ("freetoken_kv_tokens_used", "gauge", "KV cache tokens in use (pages x page size)"),
    ("freetoken_mamba_slots_used", "gauge", "Linear-attention (recurrent) state slots in use"),
    ("freetoken_mamba_slots_total", "gauge", "Linear-attention (recurrent) state slots allocated"),
    ("freetoken_swa_pages_used", "gauge", "Sliding-window attention pages in use"),
    ("freetoken_swa_pages_total", "gauge", "Sliding-window attention pages allocated"),
    ("freetoken_vram_bytes", "gauge", "Device memory the engine holds"),
    ("freetoken_gpu_memory_total_bytes", "gauge", "Total memory of each GPU the engine uses"),
    ("freetoken_model_info", "gauge", "Served model identity; the value is always 1"),
)

# llama.cpp metrics this engine does not emit. Each would have to be invented rather than
# read, and a fabricated series is worse for an operator than a missing one: a panel that
# stays empty says "not measured", while a panel pinned at 0 says "measured, and idle".
#
#   llamacpp:prompt_seconds_total            cumulative prompt seconds -- the tracker keeps
#   llamacpp:tokens_predicted_seconds_total  a sliding window, not lifetime busy time
#   llamacpp:requests_deferred               the frontend counts admitted requests; the
#                                            processing/queued split lives in the scheduler
#   llamacpp:n_tokens_max                    context high-water mark, never recorded
#   llamacpp:n_decode_total                  llama_decode() call count: no such call here
#   llamacpp:n_busy_slots_per_decode         slots are a llama.cpp scheduling concept
_ABSENT = (
    "llamacpp:prompt_seconds_total",
    "llamacpp:tokens_predicted_seconds_total",
    "llamacpp:requests_deferred",
    "llamacpp:n_tokens_max",
    "llamacpp:n_decode_total",
    "llamacpp:n_busy_slots_per_decode",
)


def _esc(value: Any) -> str:
    """Escape a label value per the exposition format (backslash, quote, newline)."""
    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _num(value: Any) -> str:
    """Render a sample value; non-numeric or missing becomes 0 rather than breaking the scrape."""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    return "0"


class _Out:
    """Collects samples and emits HELP/TYPE once per metric, in _METRICS order."""

    def __init__(self) -> None:
        self._samples: dict[str, list[str]] = {}

    def add(self, name: str, value: Any, labels: str = "") -> None:
        self._samples.setdefault(name, []).append(f"{name}{labels} {_num(value)}")

    def render(self) -> str:
        lines: list[str] = []
        for name, kind, help_text in _METRICS:
            rows = self._samples.get(name)
            if not rows:
                continue
            lines.append(f"# HELP {name} {help_text}")
            lines.append(f"# TYPE {name} {kind}")
            lines.extend(rows)
        return "\n".join(lines) + "\n"


def render_prometheus(doc: dict | None) -> str:
    """Render a ``/v1/stats`` document as Prometheus text.

    ``doc`` is None (or unusable) while the backend is still coming up: emit ``freetoken_up 0``
    so the scrape still succeeds and a restart is visible as a gap in the series rather than as
    a scrape error.
    """
    out = _Out()
    if not isinstance(doc, dict):
        out.add("freetoken_up", 0)
        return out.render()

    out.add("freetoken_up", 1)
    out.add("freetoken_uptime_seconds", doc.get("uptime_s", 0))

    req = doc.get("requests") or {}
    out.add("llamacpp:requests_processing", req.get("active", 0))
    out.add("freetoken_requests_completed_total", req.get("completed", 0))
    out.add("llamacpp:prompt_tokens_total", req.get("prompt_tokens_total", 0))
    out.add("llamacpp:tokens_predicted_total", req.get("completion_tokens_total", 0))
    out.add("freetoken_request_latency_p95_milliseconds", req.get("p95_ms", 0))
    out.add("freetoken_request_ttft_mean_milliseconds", req.get("ttft_mean_ms", 0))

    thr = doc.get("throughput") or {}
    out.add("llamacpp:predicted_tokens_seconds", thr.get("decode_tps", 0))
    out.add("llamacpp:prompt_tokens_seconds", thr.get("prefill_tps", 0))

    # kv/mamba/swa are null when the pool does not exist for this model (non-hybrid, no SWA,
    # borrowed KV): skip them instead of reporting zeros that would read as "full pool, idle".
    kv = doc.get("kv")
    if kv:
        used, total = kv.get("used_pages", 0), kv.get("total_pages", 0)
        page = kv.get("page_size", 1) or 1
        out.add("freetoken_kv_pages_used", used)
        out.add("freetoken_kv_pages_total", total)
        out.add("freetoken_kv_usage_ratio", (used / total) if total else 0.0)
        out.add("freetoken_kv_tokens_used", used * page)

    mamba = doc.get("mamba")
    if mamba:
        out.add("freetoken_mamba_slots_used", mamba.get("used_slots", 0))
        out.add("freetoken_mamba_slots_total", mamba.get("total_slots", 0))

    swa = doc.get("swa")
    if swa:
        out.add("freetoken_swa_pages_used", swa.get("used_pages", 0))
        out.add("freetoken_swa_pages_total", swa.get("total_pages", 0))

    out.add("freetoken_vram_bytes", doc.get("vram_bytes", 0))

    for gpu in doc.get("gpus") or []:
        labels = f'{{index="{_esc(gpu.get("index", 0))}",name="{_esc(gpu.get("name", ""))}"}}'
        out.add("freetoken_gpu_memory_total_bytes", gpu.get("total_bytes", 0), labels)

    model = doc.get("model") or {}
    labels = (
        f'{{model="{_esc(model.get("id", ""))}"'
        f',attn="{_esc(model.get("attn", ""))}"'
        f',moe="{"true" if model.get("moe") else "false"}"'
        f',ctx="{_esc(model.get("ctx", 0))}"}}'
    )
    out.add("freetoken_model_info", 1, labels)
    return out.render()


__all__ = ["CONTENT_TYPE", "render_prometheus"]
