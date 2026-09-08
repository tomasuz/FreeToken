"""--kv-cache-dtype resolution and what it does to the sizing arithmetic.

The engine solves the page count before any pool exists, so the storage width has to
reach ``spec_kv_bytes_per_token`` by the same route it later reaches ``torch.empty``.
These pins hold the two together: if they ever disagree the engine allocates a slab of
one width having budgeted for another, which shows up as an OOM at startup rather than
anything a kernel test would catch.
"""

from types import SimpleNamespace

import pytest
import torch

from freetoken.kvcache.base import kv_storage_dtype, spec_kv_bytes_per_token
from freetoken.models.config import KVCacheGroupSpec


def _config(kv_cache_dtype="auto", dtype=torch.bfloat16):
    return SimpleNamespace(
        dtype=dtype,
        kv_cache_dtype=kv_cache_dtype,
        tp_info=SimpleNamespace(size=1),
    )


def _spec(**kw):
    base = dict(
        name="full", layer_ids=(0, 1), num_kv_heads=2, head_dim=128,
        sliding_window=None, mla=False, index_head_dim=0, num_index_layers=0,
    )
    base.update(kw)
    return KVCacheGroupSpec(**base)


def test_auto_follows_the_activation_dtype():
    assert kv_storage_dtype(_config()) is torch.bfloat16
    assert kv_storage_dtype(_config(dtype=torch.float16)) is torch.float16


def test_fp8_names_the_e4m3_storage_type():
    assert kv_storage_dtype(_config("fp8_e4m3")) is torch.float8_e4m3fn


def test_unknown_name_is_rejected_by_name():
    with pytest.raises(ValueError, match="fp8_e5m2"):
        kv_storage_dtype(_config("fp8_e5m2"))


def test_a_config_without_the_field_still_resolves():
    """Duck-typed configs in older callers carry no kv_cache_dtype; they mean auto."""
    assert kv_storage_dtype(SimpleNamespace(dtype=torch.bfloat16)) is torch.bfloat16


def test_fp8_halves_the_per_token_cost():
    spec = _spec()
    bf16 = spec_kv_bytes_per_token(spec, _config())
    fp8 = spec_kv_bytes_per_token(spec, _config("fp8_e4m3"))
    assert bf16 == 2 * 128 * 2 * 2 * 2  # K+V x head_dim x heads x itemsize x layers
    assert fp8 * 2 == bf16


def test_the_index_key_slab_stays_16_bit():
    """DSA index keys are a bf16 slab of their own; an fp8 K/V pool must not reprice
    them (and never reaches this path anyway -- it is paged-MHA only)."""
    spec = _spec(index_head_dim=64, num_index_layers=2)
    index_bytes = 64 * 2 * 2
    assert spec_kv_bytes_per_token(spec, _config("fp8_e4m3")) == (
        2 * 128 * 2 * 1 * 2 + index_bytes
    )
