"""GPT-2 tokenizer wrapper.

We use the HuggingFace fast tokenizer because its byte-level BPE is fully
deterministic given a fixed `transformers` version and produces the same
output across CPU/GPU and Python releases.
"""
from __future__ import annotations

from typing import Iterable, List

from transformers import AutoTokenizer, PreTrainedTokenizerBase


def load_tokenizer(name: str = "gpt2") -> PreTrainedTokenizerBase:
    """Load the GPT-2 tokenizer with `pad_token = eos_token`.

    GPT-2 has no native pad token. Setting it to EOS is the standard idiom
    used by HuggingFace causal-LM training code; padded positions are masked
    out by the attention mask, so the choice is harmless.
    """
    tok = AutoTokenizer.from_pretrained(name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def encode(
    tokenizer: PreTrainedTokenizerBase,
    text: str,
    add_eos: bool = True,
) -> List[int]:
    """Encode a single document. Returns a list of token IDs."""
    ids = tokenizer.encode(text, add_special_tokens=False)
    if add_eos:
        ids.append(tokenizer.eos_token_id)
    return ids


def encode_stream(
    tokenizer: PreTrainedTokenizerBase,
    docs: Iterable[str],
    add_eos: bool = True,
) -> Iterable[List[int]]:
    """Encode an iterable of documents lazily, preserving order."""
    for doc in docs:
        yield encode(tokenizer, doc, add_eos=add_eos)
