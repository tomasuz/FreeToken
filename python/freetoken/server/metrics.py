"""Prometheus exposition for ``/metrics``.

A pure renderer over the ``/v1/stats`` document: every number here already exists in
:func:`freetoken.server.stats.build_stats`, so scraping adds no instrumentation to the
decode path -- it reads the same counters the desktop app polls.

Names follow the Prometheus conventions rather than llama.cpp's ``llamacpp:`` prefix,
which is not a legal exposition name (a colon is reserved for recording rules). Counters
carry the ``_total`` suffix and everything else is a gauge; the model identity rides a
single ``freetoken_model_info`` info metric instead of being repeated as a label on every
series, so a scraper that turns labels into tags (telegraf's prometheus input with
``metric_version = 2``) does not fan every value out per model.
"""

from __future__ import annotations

from typing import Any

CONTENT_TYPE = "text/plain; version=0.0.4; charset=utf-8"

# name -> (type, help). Kept beside the renderer so HELP/TYPE and the emitted samples
# cannot drift apart.
_METRICS: tuple[tuple[str, str, str], ...] = (
    ("freetoken_up", "gauge", "1 when the engine is serving, 0 while it is still starting"),
    ("freetoken_uptime_seconds", "gauge", "Seconds since the engine became ready"),
    ("freetoken_requests_active", "gauge", "Requests admitted and not yet finished"),
    ("freetoken_requests_completed_total", "counter", "Requests finished since start"),
    ("freetoken_prompt_tokens_total", "counter", "Prompt tokens processed since start"),
    ("freetoken_completion_tokens_total", "counter", "Completion tokens generated since start"),
    ("freetoken_decode_tokens_per_second", "gauge", "Decode throughput over the tracker's sliding window"),
    ("freetoken_prefill_tokens_per_second", "gauge", "Prefill throughput over the tracker's sliding window"),
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
    out.add("freetoken_requests_active", req.get("active", 0))
    out.add("freetoken_requests_completed_total", req.get("completed", 0))
    out.add("freetoken_prompt_tokens_total", req.get("prompt_tokens_total", 0))
    out.add("freetoken_completion_tokens_total", req.get("completion_tokens_total", 0))
    out.add("freetoken_request_latency_p95_milliseconds", req.get("p95_ms", 0))
    out.add("freetoken_request_ttft_mean_milliseconds", req.get("ttft_mean_ms", 0))

    thr = doc.get("throughput") or {}
    out.add("freetoken_decode_tokens_per_second", thr.get("decode_tps", 0))
    out.add("freetoken_prefill_tokens_per_second", thr.get("prefill_tps", 0))

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
