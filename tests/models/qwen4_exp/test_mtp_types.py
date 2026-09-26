"""The MTP head takes each projection's ggml type from its GGUF (models/qwen4_exp/mtp.py):
a type the GGUF kernels serve stays packed, anything else (f16, an i-quant, parts of
different types) becomes a dense bf16 weight.

Checked on a small synthetic MTP GGUF: the per-projection types, and the state dict the
head loads -- packed bytes where packed, the dequantized values where dense.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

gguf = pytest.importorskip("gguf")

LAYER, ROWS, COLS = 0, 4, 64
NORMS = ["nextn.enorm.weight", "nextn.hnorm.weight", "attn_q_norm.weight", "attn_k_norm.weight",
         "nextn.hc_head_norm.weight", "hc_attn_norm.weight", "hc_ffn_norm.weight"]
DENSE_F32 = ["ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight"]
F16 = {"attn_output.weight", "hc_ffn_inject.weight"}  # o_proj dense; hc_ffn down+inject mixed


def _write(path):
    from freetoken.models.qwen4_exp.mtp import _SITES

    rng = np.random.default_rng(0)
    w = gguf.GGUFWriter(str(path), "qwen4exp")
    values = {}
    for n in NORMS + DENSE_F32:
        a = rng.standard_normal((ROWS, COLS)).astype(np.float32)
        w.add_tensor(f"blk.{LAYER}.{n}", a)
    for names in _SITES.values():
        for n in names:
            a = (rng.standard_normal((ROWS, COLS)) * 0.1).astype(np.float32)
            values[n] = a
            if n in F16:
                w.add_tensor(f"blk.{LAYER}.{n}", a.astype(np.float16))
            else:
                q = gguf.quants.quantize(a, gguf.GGMLQuantizationType.Q8_0)
                w.add_tensor(f"blk.{LAYER}.{n}", q, raw_dtype=gguf.GGMLQuantizationType.Q8_0)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return values


def test_site_types_follow_the_file(tmp_path):
    from freetoken.gguf_quant import GGML_Q8_0
    from freetoken.models.qwen4_exp.mtp import mtp_site_types, mtp_tensor_types

    path = tmp_path / "mtp.gguf"
    _write(path)
    types = mtp_site_types(mtp_tensor_types(str(path)), LAYER)
    assert types["eh_proj"] == GGML_Q8_0
    assert types["self_attn.qkv_proj"] == GGML_Q8_0
    assert types["self_attn.o_proj"] is None  # f16: dense
    assert types["mlp_hyper_connection.input_mix_weight_down_block_inject"] is None  # mixed parts
    assert types["attn_hyper_connection.input_mix_weight_down_block_inject"] == GGML_Q8_0


def test_state_dict_packs_or_dequantizes_per_projection(tmp_path):
    from freetoken.gguf_quant import row_bytes
    from freetoken.models.qwen4_exp.mtp import mtp_site_types, mtp_state_dict, mtp_tensor_types

    path = tmp_path / "mtp.gguf"
    values = _write(path)
    types = mtp_site_types(mtp_tensor_types(str(path)), LAYER)
    sd = mtp_state_dict(str(path), LAYER, types)
    # packed: the file's Q8_0 rows as they are, parts stacked in row order
    qkv = sd["self_attn.qkv_proj.qweight"]
    assert qkv.dtype == torch.uint8 and tuple(qkv.shape) == (3 * ROWS, row_bytes(COLS, types["eh_proj"]))
    # dense f16 projection: its values exactly
    o = sd["self_attn.o_proj.weight"]
    assert torch.equal(o, torch.from_numpy(values["attn_output.weight"].astype(np.float16).astype(np.float32)))
    # mixed parts: dense, the Q8_0 part dequantized, the f16 part exact
    mixed = sd["mlp_hyper_connection.input_mix_weight_down_block_inject.weight"]
    assert tuple(mixed.shape) == (2 * ROWS, COLS)
    down = torch.from_numpy(values["hc_ffn_down.weight"])
    assert ((mixed[:ROWS] - down).norm() / down.norm()).item() < 0.01
    inject = torch.from_numpy(values["hc_ffn_inject.weight"].astype(np.float16).astype(np.float32))
    assert torch.equal(mixed[ROWS:], inject)
    assert "self_attn.o_proj.qweight" not in sd


def test_dense_projection_is_a_plain_linear():
    from freetoken.models.qwen4_exp.mtp import _DenseLinear

    lin = _DenseLinear(COLS, ROWS)
    lin.weight = torch.randn(ROWS, COLS)
    x = torch.randn(3, COLS)
    assert torch.allclose(lin.forward(x), x @ lin.weight.T)
