"""Document loading.

Currently supports HuggingFace `datasets` (defaults to wikitext-2).
The single entrypoint is `iter_documents`, which always preserves the
original on-disk order so that two builds with the same config produce
byte-identical token streams.

Adding a new source = adding a new branch in `iter_documents` (or a new
`_load_*` helper). The rest of the pipeline does not need to change.
"""
from __future__ import annotations

from typing import Iterable, Mapping


def iter_documents(
    source: str,
    source_config: str | None,
    split: str,
    text_column: str = "text",
) -> Iterable[str]:
    """Yield non-empty documents from `source` in deterministic order.

    Empty / whitespace-only rows are filtered out. We do not strip otherwise
    because the GPT-2 BPE is whitespace-sensitive.
    """
    if source in {"wikitext", "huggingface"}:
        yield from _load_hf(source, source_config, split, text_column)
    else:
        raise ValueError(f"Unknown dataset source: {source!r}")


def _load_hf(
    source: str,
    source_config: str | None,
    split: str,
    text_column: str,
) -> Iterable[str]:
    # Imported lazily so that `import nfmamba.data` doesn't pull in `datasets`
    # for code paths that only need the dataloader.
    from datasets import load_dataset

    repo = "wikitext" if source == "wikitext" else source_config
    cfg = source_config if source == "wikitext" else None
    ds = load_dataset(repo, cfg, split=split)

    if text_column not in ds.column_names:
        raise KeyError(
            f"Column {text_column!r} not in dataset (have {ds.column_names})"
        )

    for row in ds:
        text = row[text_column]
        if text and text.strip():
            yield text


def summary(docs: Mapping[str, list]) -> dict:
    """Compact summary used by the build script."""
    return {split: {"num_documents": len(d)} for split, d in docs.items()}
