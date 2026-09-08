"""An fp8 KV pool must change what the cache *costs*, never what attention *computes*.

Every e4m3 value is exact in bf16, so the two caches compared here -- one holding the
e4m3 bytes, one holding those same values already widened -- carry identical numbers.
The kernels are therefore expected to agree bit for bit, not merely closely: any drift
means the decode in ``_load_kv``, the dtype it lands in, or the branch that skips the
q-cast disagrees with the plain path, and a tolerance would hide exactly that.
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="triton attention kernels need a GPU"
)


def _quantized_pair(shape, device, generator):
    """The same K (or V) tensor as e4m3 bytes and as its widened bf16 twin."""
    from freetoken.kernel.triton.e4m3_compat import quantize_e4m3

    raw = torch.randn(shape, device=device, dtype=torch.bfloat16, generator=generator)
    fp8 = quantize_e4m3(raw)
    return fp8, fp8.view(torch.float8_e4m3fn).to(torch.bfloat16)


def _decode_scratch(batch, num_q_heads, head_dim, max_kv_splits, device):
    return (
        torch.empty((batch, num_q_heads, max_kv_splits, head_dim), dtype=torch.float32, device=device),
        torch.empty((batch, num_q_heads, max_kv_splits), dtype=torch.float32, device=device),
        torch.full((batch,), max_kv_splits, dtype=torch.int32, device=device),
    )


def test_decode_matches_widened_cache():
    from freetoken.kernel.triton.attention import decode_paged_attention

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(20260908)
    batch, num_q_heads, num_kv_heads, head_dim, kv_len = 3, 8, 2, 128, 300
    slots = batch * kv_len

    q = torch.randn((batch, num_q_heads, head_dim), device=device, dtype=torch.bfloat16, generator=gen)
    k8, k16 = _quantized_pair((slots, num_kv_heads, head_dim), device, gen)
    v8, v16 = _quantized_pair((slots, num_kv_heads, head_dim), device, gen)
    indptr = torch.arange(batch + 1, device=device, dtype=torch.int32) * kv_len
    indices = torch.arange(slots, device=device, dtype=torch.int32)
    q_positions = torch.full((batch,), kv_len - 1, device=device, dtype=torch.int32)

    outs = []
    for k, v in ((k8, v8), (k16, v16)):
        logits, lse, splits = _decode_scratch(batch, num_q_heads, head_dim, 8, device)
        outs.append(
            decode_paged_attention(
                q=q, k_cache=k, v_cache=v, indptr=indptr, indices=indices,
                q_positions=q_positions, attn_logits=logits, attn_lse=lse,
                num_kv_splits=splits, max_kv_splits=8, sm_scale=head_dim**-0.5,
            ).clone()
        )
    assert torch.equal(outs[0], outs[1])
    assert outs[0].isfinite().all()


def test_extend_matches_widened_cache():
    from freetoken.kernel.triton.attention import extend_paged_attention

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(1)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 128
    prefix, q_len = 64, 48
    kv_len = prefix + q_len

    q = torch.randn((q_len, num_q_heads, head_dim), device=device, dtype=torch.bfloat16, generator=gen)
    k8, k16 = _quantized_pair((kv_len, num_kv_heads, head_dim), device, gen)
    v8, v16 = _quantized_pair((kv_len, num_kv_heads, head_dim), device, gen)
    qo_indptr = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, kv_len], device=device, dtype=torch.int32)
    kv_indices = torch.arange(kv_len, device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device=device, dtype=torch.int32)

    outs = [
        extend_paged_attention(
            q=q, k_cache=k, v_cache=v, qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            kv_indices=kv_indices, prefix_lens=prefix_lens, max_q_len=q_len,
            sm_scale=head_dim**-0.5,
        ).clone()
        for k, v in ((k8, v8), (k16, v16))
    ]
    assert torch.equal(outs[0], outs[1])
    assert outs[0].isfinite().all()


def test_extend_split_keeps_unquantized_new_tokens():
    """The split kernel reads the prefix from the cache and this step's own tokens from
    k_extend/v_extend, which are never quantized -- only the first pair may be bytes."""
    from freetoken.kernel.triton.attention import extend_paged_attention

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(2)
    num_q_heads, num_kv_heads, head_dim = 8, 2, 128
    prefix, q_len = 96, 32
    kv_len = prefix + q_len

    q = torch.randn((q_len, num_q_heads, head_dim), device=device, dtype=torch.bfloat16, generator=gen)
    k8, k16 = _quantized_pair((kv_len, num_kv_heads, head_dim), device, gen)
    v8, v16 = _quantized_pair((kv_len, num_kv_heads, head_dim), device, gen)
    k_ext = torch.randn((q_len, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16, generator=gen)
    v_ext = torch.randn((q_len, num_kv_heads, head_dim), device=device, dtype=torch.bfloat16, generator=gen)
    qo_indptr = torch.tensor([0, q_len], device=device, dtype=torch.int32)
    kv_indptr = torch.tensor([0, kv_len], device=device, dtype=torch.int32)
    kv_indices = torch.arange(kv_len, device=device, dtype=torch.int32)
    prefix_lens = torch.tensor([prefix], device=device, dtype=torch.int32)

    outs = [
        extend_paged_attention(
            q=q, k_cache=k, v_cache=v, qo_indptr=qo_indptr, kv_indptr=kv_indptr,
            kv_indices=kv_indices, prefix_lens=prefix_lens, max_q_len=q_len,
            sm_scale=head_dim**-0.5, k_extend=k_ext, v_extend=v_ext,
        ).clone()
        for k, v in ((k8, v8), (k16, v16))
    ]
    assert torch.equal(outs[0], outs[1])
    assert outs[0].isfinite().all()


def test_paged_matches_widened_cache():
    from freetoken.kernel.triton.attention import paged_attention

    device = torch.device("cuda")
    gen = torch.Generator(device=device).manual_seed(3)
    num_q_heads, num_kv_heads, head_dim, kv_len = 8, 2, 64, 128

    q = torch.randn((4, num_q_heads, head_dim), device=device, dtype=torch.bfloat16, generator=gen)
    k8, k16 = _quantized_pair((kv_len, num_kv_heads, head_dim), device, gen)
    v8, v16 = _quantized_pair((kv_len, num_kv_heads, head_dim), device, gen)
    indptr = torch.tensor([0, kv_len], device=device, dtype=torch.int32)
    indices = torch.arange(kv_len, device=device, dtype=torch.int32)
    q_to_req = torch.zeros(4, device=device, dtype=torch.int32)
    q_positions = torch.arange(4, device=device, dtype=torch.int32) + 60

    outs = [
        paged_attention(
            q=q, k_cache=k, v_cache=v, indptr=indptr, indices=indices,
            q_to_req=q_to_req, q_positions=q_positions, sm_scale=head_dim**-0.5,
        ).clone()
        for k, v in ((k8, v8), (k16, v16))
    ]
    assert torch.equal(outs[0], outs[1])
    assert outs[0].isfinite().all()


def test_quantize_e4m3_saturates_instead_of_producing_nan():
    """torch's e4m3 cast is the non-saturating variant; without the clamp a large key
    would come back NaN and take a whole softmax row with it."""
    from freetoken.kernel.triton.e4m3_compat import quantize_e4m3

    x = torch.tensor([1e5, -1e5, 448.0, -448.0, 0.0], device="cuda", dtype=torch.bfloat16)
    back = quantize_e4m3(x).view(torch.float8_e4m3fn).to(torch.float32)
    assert back.isfinite().all()
    assert torch.equal(back, torch.tensor([448.0, -448.0, 448.0, -448.0, 0.0], device="cuda"))


def test_pool_round_trips_through_e4m3_storage():
    from freetoken.distributed import set_tp_info, try_get_tp_info
    from freetoken.kvcache.mha_pool import MHAKVCache

    if try_get_tp_info() is None:
        set_tp_info(rank=0, size=1)
    device = torch.device("cuda")
    pool = MHAKVCache(
        num_kv_heads=2, num_layers=4, head_dim=64, num_pages=32, page_size=1,
        dtype=torch.float8_e4m3fn, device=device, layer_ids=[1, 3],
    )
    assert pool.k_cache(1).dtype == torch.uint8  # kernels take the byte view
    # bytes/token: K+V slabs x 2 KV-bearing layers x 2 heads x 64 dims, one byte each
    assert pool.unit_bytes()[0] == 2 * 2 * 2 * 64

    # store_cache takes the flattened [tokens, heads*dim] rows its callers hand it.
    gen = torch.Generator(device=device).manual_seed(4)
    k = torch.randn((5, 2 * 64), device=device, dtype=torch.bfloat16, generator=gen)
    v = torch.randn((5, 2 * 64), device=device, dtype=torch.bfloat16, generator=gen)
    out_loc = torch.tensor([3, 7, 8, 20, 31], device=device, dtype=torch.int32)
    pool.store_kv(k, v, out_loc, layer_id=3)

    stored = pool.k_cache(3).view(-1, 2 * 64)[out_loc.long()]
    assert torch.equal(stored, k.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8))
    stored_v = pool.v_cache(3).view(-1, 2 * 64)[out_loc.long()]
    assert torch.equal(stored_v, v.clamp(-448, 448).to(torch.float8_e4m3fn).view(torch.uint8))
