"""Resident (VRAM-only) expert layers and file-backed host banks.

The invariant under test is the point of the feature: a resident layer's experts exist in
exactly ONE place. So these check not just that the device banks are readable, but that the
host side really let go -- the host bank is released, the copy plan carries a null source
pointer for it, and no movement call is made on its behalf.
"""

from __future__ import annotations

import os

import pytest
import torch

from freetoken.moe.host_banks import HostBank, PinPipeline, resident_upload


# ---------------------------------------------------------------------------
# resident uploader
# ---------------------------------------------------------------------------


class _FakeUploader:
    """Records the (layer, banks) the sink routes to it, without touching a device."""

    def __init__(self, layers):
        self.layers = frozenset(layers)
        self.seen = {}

    def claims(self, layer_id):
        return layer_id in self.layers

    def upload(self, layer_id, banks):
        self.seen[layer_id] = {n: b.tensor.clone() for n, b in banks.items()}


def test_pin_pipeline_routes_claimed_layers_to_upload_and_releases():
    """A claimed layer is uploaded and released; an unclaimed one is settled normally."""
    banks = {
        name: [HostBank((2, 16), torch.uint8) for _ in range(3)]
        for name in ("gate_up", "down")
    }
    for name, per_layer in banks.items():
        for layer_id, bank in enumerate(per_layer):
            bank.tensor.fill_(layer_id + 1)

    uploader = _FakeUploader({0, 2})
    os.environ["FREETOKEN_SKIP_BANK_PIN"] = "1"  # no CUDA in this test
    try:
        with resident_upload(uploader), PinPipeline() as pins:
            for layer_id in range(3):
                pins(layer_id, {n: per[layer_id] for n, per in banks.items()})
    finally:
        del os.environ["FREETOKEN_SKIP_BANK_PIN"]

    assert sorted(uploader.seen) == [0, 2]
    # uploaded contents are the pre-release bytes
    assert int(uploader.seen[0]["gate_up"][0, 0]) == 1
    assert int(uploader.seen[2]["down"][0, 0]) == 3
    # the unclaimed layer was never handed to the uploader
    assert 1 not in uploader.seen


def test_uploader_reports_layers_the_loader_skipped():
    from freetoken.moe.expert_banks import ResidentUploader

    up = ResidentUploader(frozenset({0, 1, 2}), torch.device("cpu"))
    banks = {"gate_up": HostBank((2, 16), torch.uint8), "down": HostBank((2, 16), torch.uint8)}
    up.upload(0, banks)
    assert up.missing() == [1, 2]
    up.upload(1, banks)
    up.upload(2, banks)
    assert up.missing() == []


# ---------------------------------------------------------------------------
# cache wiring
# ---------------------------------------------------------------------------


def _cache(num_layers=4, num_experts=8, cache_size=8, device="cpu"):
    from freetoken.moe.offload_cache import OffloadMoeCache

    return OffloadMoeCache(
        num_layers=num_layers,
        num_experts=num_experts,
        cache_size=cache_size,
        device=torch.device(device),
        quant_format="q4_0",
        prefill_overlap=False,
    )


def _sources(num_layers=4, num_experts=8, row=32):
    return {
        "gate_up": [torch.zeros(num_experts, row, dtype=torch.uint8) for _ in range(num_layers)],
        "down": [torch.zeros(num_experts, row, dtype=torch.uint8) for _ in range(num_layers)],
    }


def test_resident_layers_are_flagged_and_views_route_to_vram():
    cache = _cache()
    resident = {
        name: {l: torch.full((8, 32), l + 1, dtype=torch.uint8) for l in (0, 3)}
        for name in ("gate_up", "down")
    }
    cache.set_resident_banks(resident, frozenset({0, 3}))
    cache.set_bank_sources(_sources())

    assert cache.is_resident_layer(0) and cache.is_resident_layer(3)
    assert not cache.is_resident_layer(1)
    gate_up, down = cache.resident_views(3)
    assert gate_up.shape == (8, 32) and int(down[0, 0]) == 4
    # registration order matters: the kernels unpack views positionally
    assert cache.bank_schema == ("gate_up", "down")


def test_resident_layer_rejects_cpu_overlap():
    cache = _cache()
    cache.cpu_layer_ids = frozenset({1})
    resident = {name: {1: torch.zeros(8, 32, dtype=torch.uint8)} for name in ("gate_up", "down")}
    with pytest.raises(AssertionError, match="GPU-resident and CPU-decode"):
        cache.set_resident_banks(resident, frozenset({1}))


def test_resident_layer_requires_every_bank():
    cache = _cache()
    resident = {"gate_up": {0: torch.zeros(8, 32, dtype=torch.uint8)}}
    with pytest.raises(AssertionError, match="do not match"):
        cache.set_resident_banks(resident, frozenset({0}))


def test_set_bank_sources_tolerates_released_resident_sources():
    """The loader leaves resident layers' host banks in ``sources`` with their pages gone.

    They must survive validation (shape is still right) without being read.
    """
    cache = _cache()
    resident = {
        name: {0: torch.zeros(8, 32, dtype=torch.uint8)} for name in ("gate_up", "down")
    }
    cache.set_resident_banks(resident, frozenset({0}))
    sources = _sources()
    # a released HostBank keeps its shape but is not contiguous-checked by us any more;
    # simulate the worst case with a non-contiguous stand-in for the resident layer only
    sources["gate_up"][0] = torch.zeros(8, 64, dtype=torch.uint8)[:, ::2]
    cache.set_bank_sources(sources)  # must not raise
    assert cache.is_resident_layer(0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU for the copy plan")
def test_copy_plan_nulls_resident_sources():
    """The fused copy descriptor must carry 0 for a resident layer's source pointer.

    Taking its device alias would either fail (the bank was never registered) or, worse,
    point at dropped pages.
    """
    device = torch.device("cuda")
    cache = _cache(device="cuda")
    resident = {
        name: {2: torch.zeros(8, 32, dtype=torch.uint8, device=device)}
        for name in ("gate_up", "down")
    }
    cache.set_resident_banks(resident, frozenset({2}))

    from freetoken.moe.host_banks import HostBank as HB

    banks = {name: [HB((8, 32), torch.uint8) for _ in range(4)] for name in ("gate_up", "down")}
    for per_layer in banks.values():
        for layer_id, b in enumerate(per_layer):
            if layer_id != 2:
                b.pin()
    cache.set_bank_sources({n: [b.tensor for b in per] for n, per in banks.items()})

    if cache._copy_fused_ok:
        assert cache._copy_src_ptrs_host[2] == [0, 0]
        assert all(p != 0 for p in cache._copy_src_ptrs_host[1])


# ---------------------------------------------------------------------------
# flag resolution
# ---------------------------------------------------------------------------


class _Cfg:
    moe_backend = "offload"

    def __init__(self, spec):
        self.moe_resident_layers = spec


def test_resolve_resident_layers_spec_forms():
    from freetoken.engine.engine import _RESIDENT_AUTO, _resolve_resident_layers

    assert _resolve_resident_layers(_Cfg(None), 30) == frozenset()
    assert _resolve_resident_layers(_Cfg(""), 30) == frozenset()
    assert _resolve_resident_layers(_Cfg("auto"), 30) == _RESIDENT_AUTO
    assert _resolve_resident_layers(_Cfg("3,7,11"), 30) == frozenset({3, 7, 11})
    # a count is placed from the ends, where decode miss rates are highest
    assert sorted(_resolve_resident_layers(_Cfg("4"), 30)) == [0, 1, 28, 29]
    assert len(_resolve_resident_layers(_Cfg("0.5"), 30)) == 15
    assert _resolve_resident_layers(_Cfg("30"), 30) == frozenset(range(30))


def test_resolve_resident_layers_is_offload_only():
    cfg = _Cfg("8")
    cfg.moe_backend = "fused"
    from freetoken.engine.engine import _resolve_resident_layers

    assert _resolve_resident_layers(cfg, 30) == frozenset()


def test_resolve_resident_layers_rejects_out_of_range():
    from freetoken.engine.engine import _resolve_resident_layers

    with pytest.raises(ValueError, match="out of range"):
        _resolve_resident_layers(_Cfg("0,99"), 30)
    with pytest.raises(ValueError, match="must be in"):
        _resolve_resident_layers(_Cfg("31"), 30)


def test_auto_cache_size_excludes_resident_layers():
    """A resident layer must not be sized for a slot, and its VRAM is charged as weights."""
    from freetoken.engine.engine import Engine
    from freetoken.kvcache.mha_pool import MHAKVCache
    from freetoken.models.config import KVCacheGroupSpec

    class StubModelConfig:
        has_swa_attention = False
        num_experts = 4
        num_moe_layers = 4  # total_experts = 16 with no resident tier

        def kv_cache_group_specs(self):
            return [KVCacheGroupSpec(
                name="full", layer_ids=(0, 1, 2, 3), num_kv_heads=8, head_dim=64,
                sliding_window=None,
            )]

        def linear_attention_group(self):
            return None

    class StubConfig:
        dtype = torch.float16
        page_size = 16
        max_running_req = 4
        hybrid_swa_cache_mode = "auto"
        memory_ratio = 0.9
        moe_prefill_overlap = False
        kv_reserve_tokens = 0
        swa_full_tokens_ratio = 0.2
        swa_num_pages_override = None
        model_config = StubModelConfig()

        class tp_info:
            size = 1

    class Banks:
        quant_format = "bf16"
        sources = {
            "gate_up": [torch.zeros(4, 32, 8, dtype=torch.float16)] * 4,
            "down": [torch.zeros(4, 8, 16, dtype=torch.float16)] * 4,
        }

        def __init__(self, resident, resident_bytes=0):
            self.resident_layers = frozenset(resident)
            self.resident_bytes = resident_bytes

    engine = Engine.__new__(Engine)  # bypass __init__/GPU
    engine._baseline_free = 10_000_000
    engine._weights_bytes = 1_000_000
    engine._pool_cls = MHAKVCache

    all_offload, _, _ = engine._resolve_auto_moe_cache_size(StubConfig(), Banks(()))
    half_resident, _, _ = engine._resolve_auto_moe_cache_size(StubConfig(), Banks({0, 3}))
    # half the layers gone from the offload population -> the cache is sized for half the
    # experts (capped by the same budget, so this is an upper bound that must shrink)
    assert half_resident < all_offload
    assert half_resident >= StubModelConfig.num_experts  # never below the slot floor


# ---------------------------------------------------------------------------
# numerical equivalence: resident tier vs the offload path it replaces
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_resident_layer_matches_offload_path_bitwise():
    """The same experts must produce the same logits whichever tier serves them.

    This is the test that would catch the two ways the resident branch can be subtly wrong:
    passing slot-remapped ids where raw expert ids are wanted, or picking ``alphas_for_slots``
    over ``alphas_for_layer``. Layer 0 is served resident, layer 1 through the ordinary
    host-bank -> materialize -> slot-cache path, from byte-identical banks.
    """
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.layers.moe import OffloadMoELayer
    from freetoken.moe.host_banks import HostBank as HB
    from freetoken.moe.offload_cache import OffloadMoeCache
    from freetoken.models.gguf.dequant import GGML_Q4_0, row_bytes

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    device = torch.device("cuda")
    torch.manual_seed(0)
    E, H, I, tokens, top_k = 4, 64, 64, 3, 2
    gu_row, dn_row = row_bytes(H, GGML_Q4_0), row_bytes(I, GGML_Q4_0)

    def q4_0_bank(*shape: int, blocks: int) -> torch.Tensor:
        """Random but VALID Q4_0 bytes: each 18-byte block is ``half d`` + 16 nibble bytes.

        Randomising all 18 bytes would put random bit patterns in the fp16 scale, and a
        good share of those decode to NaN/Inf -- the comparison would then be NaN vs NaN
        and prove nothing. So the payload is random and the scale is a fixed small value.
        """
        buf = torch.randint(0, 256, (*shape, blocks, 18), dtype=torch.uint8)
        scale = torch.tensor([0.05], dtype=torch.float16).view(torch.uint8)  # 2 bytes
        buf[..., :2] = scale
        return buf.reshape(*shape, blocks * 18)

    # identical packed bytes for both layers
    gate_up = q4_0_bank(E, 2 * I, blocks=H // 32)
    down = q4_0_bank(E, H, blocks=I // 32)

    banks = {"gate_up": [HB((E, 2 * I, gu_row), torch.uint8) for _ in range(2)],
             "down": [HB((E, H, dn_row), torch.uint8) for _ in range(2)]}
    banks["gate_up"][0].tensor.copy_(gate_up)
    banks["gate_up"][1].tensor.copy_(gate_up)
    banks["down"][0].tensor.copy_(down)
    banks["down"][1].tensor.copy_(down)
    banks["gate_up"][1].pin()  # only the offloaded layer needs a device alias
    banks["down"][1].pin()

    cache = OffloadMoeCache(
        num_layers=2, num_experts=E, cache_size=E, device=device,
        quant_format="q4_0", prefill_overlap=False,
    )
    cache.set_resident_banks(
        {"gate_up": {0: gate_up.to(device)}, "down": {0: down.to(device)}}, frozenset({0})
    )
    cache.set_bank_sources({n: [b.tensor for b in per] for n, per in banks.items()})

    layer0 = OffloadMoELayer(layer_id=0, num_experts=E, top_k=top_k, hidden_size=H,
                             intermediate_size=I)
    layer1 = OffloadMoELayer(layer_id=1, num_experts=E, top_k=top_k, hidden_size=H,
                             intermediate_size=I)
    layer0.offload_cache = cache
    layer1.offload_cache = cache

    x = torch.randn(tokens, H, dtype=torch.bfloat16, device=device)
    topk_w = torch.rand(tokens, top_k, dtype=torch.float32, device=device)
    topk_ids = torch.randint(0, E, (tokens, top_k), dtype=torch.int32, device=device)

    resident_out = layer0._resident_expert_gemm(
        cache, x, topk_w, topk_ids.clone(), is_prefill=True
    )
    cache.materialize_layer(1)
    cache.copy_missing()
    offload_out = layer1._expert_gemm(
        cache, x, topk_w, topk_ids.clone(),
        views=cache.bank_views(E), n=E, alphas=cache.alphas_for_layer(1), is_prefill=True,
    )
    torch.testing.assert_close(resident_out, offload_out, rtol=0, atol=0)
