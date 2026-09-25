"""ggml quant type constants, block geometry and the GGUF format registry.

Pure data with no FreeToken imports. It lives at the package root rather than under
``freetoken.models`` because the MoE offload cache needs it, and importing anything
from ``freetoken.models`` initialises the whole model stack -- which imports the MoE
layer, which imports the offload cache. ``models.gguf.dequant`` re-exports every name
here, so existing imports keep working.
"""

from __future__ import annotations

GGML_F32 = 0
GGML_F16 = 1
GGML_Q4_0 = 2
GGML_Q4_1 = 3
GGML_Q5_0 = 6
GGML_Q5_1 = 7
GGML_Q8_0 = 8
GGML_Q2_K = 10
GGML_Q3_K = 11
GGML_Q4_K = 12
GGML_Q5_K = 13
GGML_Q6_K = 14
# i-quants: dequantize and the MMVQ / MoE-vector kernels handle them, the MMQ kernels do not
GGML_IQ2_XXS = 16
GGML_IQ2_XS = 17
GGML_IQ3_XXS = 18
GGML_IQ1_S = 19
GGML_IQ4_NL = 20
GGML_IQ3_S = 21
GGML_IQ2_S = 22
GGML_IQ4_XS = 23
GGML_IQ1_M = 29
GGML_BF16 = 30

# (block numel, bytes per block) per ggml type.
BLOCK_SHAPE: dict[int, tuple[int, int]] = {
    GGML_F32: (1, 4),
    GGML_F16: (1, 2),
    GGML_BF16: (1, 2),
    GGML_Q4_0: (32, 18),
    GGML_Q4_1: (32, 20),
    GGML_Q5_0: (32, 22),
    GGML_Q5_1: (32, 24),
    GGML_Q8_0: (32, 34),
    GGML_Q2_K: (256, 84),
    GGML_Q3_K: (256, 110),
    GGML_Q4_K: (256, 144),
    GGML_Q5_K: (256, 176),
    GGML_Q6_K: (256, 210),
    GGML_IQ2_XXS: (256, 66),
    GGML_IQ2_XS: (256, 74),
    GGML_IQ3_XXS: (256, 98),
    GGML_IQ1_S: (256, 50),
    GGML_IQ4_NL: (32, 18),
    GGML_IQ3_S: (256, 110),
    GGML_IQ2_S: (256, 82),
    GGML_IQ4_XS: (256, 136),
    GGML_IQ1_M: (256, 56),
}

GGML_NAME = {
    GGML_F32: "F32",
    GGML_F16: "F16",
    GGML_BF16: "BF16",
    GGML_Q4_0: "Q4_0",
    GGML_Q4_1: "Q4_1",
    GGML_Q5_0: "Q5_0",
    GGML_Q5_1: "Q5_1",
    GGML_Q8_0: "Q8_0",
    GGML_Q2_K: "Q2_K",
    GGML_Q3_K: "Q3_K",
    GGML_Q4_K: "Q4_K",
    GGML_Q5_K: "Q5_K",
    GGML_Q6_K: "Q6_K",
    GGML_IQ2_XXS: "IQ2_XXS",
    GGML_IQ2_XS: "IQ2_XS",
    GGML_IQ3_XXS: "IQ3_XXS",
    GGML_IQ1_S: "IQ1_S",
    GGML_IQ4_NL: "IQ4_NL",
    GGML_IQ3_S: "IQ3_S",
    GGML_IQ2_S: "IQ2_S",
    GGML_IQ4_XS: "IQ4_XS",
    GGML_IQ1_M: "IQ1_M",
}

# Native GGUF routed-expert / weight formats: the ggml quants every entry point in
# csrc/gguf dispatches. Format tag <-> ggml type, lowercased so the pre-existing
# "q4_0" tag keeps its exact spelling.
GGUF_EXPERT_FORMATS: dict[str, int] = {
    GGML_NAME[t].lower(): t
    for t in (
        GGML_Q4_0,
        GGML_Q4_1,
        GGML_Q5_0,
        GGML_Q5_1,
        GGML_Q8_0,
        GGML_Q2_K,
        GGML_Q3_K,
        GGML_Q4_K,
        GGML_Q5_K,
        GGML_Q6_K,
    )
}


def row_bytes(numel: int, ggml_type: int) -> int:
    """Packed byte length of one row of ``numel`` elements in ``ggml_type`` blocks.

    Single source of truth for the ``numel // block * type_size`` math shared by the
    packed-weight ops (``GGUFLinear``/``GGUFEmbedding``) and the expert bank loaders.
    """
    block, type_size = BLOCK_SHAPE[ggml_type]
    assert numel % block == 0, (
        f"{numel} not a multiple of block {block} for {GGML_NAME.get(ggml_type, ggml_type)}"
    )
    return numel // block * type_size


__all__ = [
    "BLOCK_SHAPE",
    "GGML_NAME",
    "GGUF_EXPERT_FORMATS",
    "row_bytes",
] + [n for n in dir() if n.startswith("GGML_") and n.isupper()]
