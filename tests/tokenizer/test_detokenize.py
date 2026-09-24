from __future__ import annotations

from freetoken.message import DetokenizeMsg
from freetoken.tokenizer.detokenize import DetokenizeManager

EOS = 0


class ByteTokenizer:
    """Each id > 0 is one byte, so a multi-byte char can split across tokens."""

    eos_token_id = EOS

    def decode(self, ids):
        return bytes(ids).decode("utf-8", errors="replace")


def _run(ids):
    m = DetokenizeManager(ByteTokenizer())
    out = []
    for i, tok in enumerate(ids):
        out.append(m.detokenize([DetokenizeMsg(uid=1, next_token=tok, finished=i == len(ids) - 1)])[0])
    return out


def test_text_without_space_is_emitted():
    # "55" then eos: the whole answer has no space to break at and must not be dropped
    out = _run([ord("5"), ord("5"), EOS])
    assert "".join(out) == "55"
    assert out[0] == "5"


def test_last_word_survives_length_finish():
    assert "".join(_run(list(b"hello world"))) == "hello world"


def test_split_utf8_char_is_held_until_complete():
    ids = list("ą".encode()) + [EOS]
    out = _run(ids)
    assert out[0] == ""
    assert "".join(out) == "ą"
