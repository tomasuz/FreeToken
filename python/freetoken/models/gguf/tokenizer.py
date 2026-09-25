"""Build a HF fast tokenizer from a GGUF file's embedded tokenizer metadata.

transformers' ``AutoTokenizer.from_pretrained(gguf_file=...)`` first builds the HF
config, which the gemma4 strict dataclass rejects (per-layer ``num_key_value_heads``
array). So we call the GGUF->fast tokenizer converter directly on the
``tokenizer.ggml.*`` metadata, bypassing config entirely.
"""

from __future__ import annotations

from typing import Any

from .reader import gguf_architecture, load_gguf_metadata

# GGUF architecture -> transformers GGUF tokenizer-converter key.
# qwen4exp (Qwen3.8-Flash-Next) ships Qwen3.5's tokenizer: gpt2 BPE, pre-tokenizer "qwen35".
_TOKENIZER_ARCH = {"gemma4": "gemma4_text", "qwen4exp": "qwen3_5_text"}


def load_gguf_tokenizer(model_path: str):
    from transformers import PreTrainedTokenizerFast
    from transformers.integrations.ggml import convert_gguf_tokenizer

    meta = load_gguf_metadata(model_path)
    arch = gguf_architecture(model_path)
    conv_arch = _TOKENIZER_ARCH.get(arch, arch)
    tok_dict: dict[str, Any] = {
        k[len("tokenizer.ggml.") :]: v
        for k, v in meta.items()
        if k.startswith("tokenizer.ggml.")
    }
    fast, _extra = convert_gguf_tokenizer(conv_arch, tok_dict)

    tokens = tok_dict["tokens"]

    vocab = set(tokens)

    def tok_for(id_key: str, default: str) -> str | None:
        tid = meta.get(f"tokenizer.ggml.{id_key}")
        if tid is not None and int(tid) < len(tokens):
            return tokens[int(tid)]
        # a default the vocab lacks would be appended as a new id past the model's
        # embedding rows (Qwen has no <unk>), so leave that role unset instead
        return default if default in vocab else None

    # gemma4 chat turns end with <turn|>; prefer it as eos so chat generation halts
    # (the formal <eos> is also a stop id, see gguf_eos_token_ids).
    turn_end = "<turn|>" if "<turn|>" in tokens else None
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=fast,
        bos_token=tok_for("bos_token_id", "<bos>"),
        eos_token=turn_end or tok_for("eos_token_id", "<eos>"),
        unk_token=tok_for("unknown_token_id", "<unk>"),
        pad_token=tok_for("padding_token_id", "<pad>"),
    )
    _register_special_tokens(tokenizer, tokens, tok_dict.get("token_type"))
    chat_template = meta.get("tokenizer.chat_template")
    if chat_template:
        tokenizer.chat_template = chat_template
    return tokenizer


# llama.cpp token types (llama_token_type): tokens of these types are matched whole in the
# text, never split by BPE -- CONTROL (<|im_start|>, <|im_end|>) and USER_DEFINED (Qwen's
# <think>, </think>, <tool_call>, ...).
_TOKEN_TYPE_CONTROL = 3
_TOKEN_TYPE_USER_DEFINED = 4


def _register_special_tokens(tokenizer, tokens: list[str], token_types) -> None:
    """Make every CONTROL / USER_DEFINED vocab entry an added special token at its own id.

    transformers' GGUF converters register only some of them, so ``<think>`` came out as
    ``<th`` ``ink`` ``>`` where llama.cpp -- and the HF tokenizer.json -- keep one token.
    The chat template is built of these tokens, and ``</think>`` has to be one token for
    the reasoning split to find it."""
    if not token_types:
        return
    from tokenizers import AddedToken

    have = tokenizer.get_added_vocab()
    add = [
        AddedToken(tokens[i], special=True, normalized=False)
        for i, kind in enumerate(token_types)
        if int(kind) in (_TOKEN_TYPE_CONTROL, _TOKEN_TYPE_USER_DEFINED) and tokens[i] not in have
    ]
    if add:
        # already in the vocab, so each keeps its id rather than being appended
        tokenizer.add_tokens(add, special_tokens=True)


def gguf_eos_token_ids(model_path: str, tokenizer) -> set[int]:
    """Stop ids for GGUF generation: the formal eos plus the family's chat turn end."""
    meta = load_gguf_metadata(model_path)
    tokens = meta["tokenizer.ggml.tokens"]
    ids: set[int] = set()
    if tokenizer.eos_token_id is not None:
        ids.add(int(tokenizer.eos_token_id))
    eid = meta.get("tokenizer.ggml.eos_token_id")
    if eid is not None:
        ids.add(int(eid))
    # Look the stop tokens up in the vocab directly (convert_tokens_to_ids would map an
    # absent name to <unk>, wrongly adding it as a stop id). gemma4 ends a turn with
    # <turn|>; the Qwen families with <|im_end|> and a document with <|endoftext|>.
    for name in ("<eos>", "<turn|>", "<|im_end|>", "<|endoftext|>"):
        try:
            ids.add(tokens.index(name))
        except ValueError:
            pass
    return ids


__all__ = ["load_gguf_tokenizer", "gguf_eos_token_ids"]
