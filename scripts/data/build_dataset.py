"""Build a packed dataset shard from a YAML config.

Outputs (per split):
    data/processed/{id}_{split}.bin     uint16 packed tokens
    data/manifests/{id}.json            full build manifest with sha256
    data/debug/{id}_debug.bin           tiny shard for overfit tests

Run:
    python scripts/data/build_dataset.py --config configs/data/wikitext_smoke.yaml
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import yaml

# Make the src-layout package importable when running the script directly.
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from nfmamba.data.dataset_builder import iter_documents
from nfmamba.data.packing import pack_eos
from nfmamba.data.tokenizer import encode, load_tokenizer


def set_global_seed(seed: int) -> None:
    """Pin every RNG we touch. Tokenization itself is deterministic, but
    seeding still guards against any future shuffle/sampling step."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def build_split(
    cfg: Dict[str, Any],
    split_name: str,
    hf_split: str,
    tokenizer,
    out_dir: Path,
) -> Dict[str, Any]:
    seq_len = cfg["packing"]["sequence_length"]
    add_eos = cfg["tokenizer"]["add_eos_per_doc"]
    dtype = np.dtype(cfg["output"]["bin_dtype"])

    print(f"[{split_name}] loading documents ...")
    t0 = time.time()
    docs = list(
        iter_documents(
            source=cfg["dataset"]["source"],
            source_config=cfg["dataset"].get("source_config"),
            split=hf_split,
            text_column=cfg["dataset"].get("text_column", "text"),
        )
    )
    print(f"[{split_name}] {len(docs)} non-empty documents in {time.time()-t0:.1f}s")

    print(f"[{split_name}] tokenizing ...")
    t0 = time.time()
    token_lists = [encode(tokenizer, d, add_eos=add_eos) for d in docs]
    print(f"[{split_name}] tokenized in {time.time()-t0:.1f}s")

    flat, num_seqs = pack_eos(
        token_lists,
        sequence_length=seq_len,
        dtype=dtype,
        drop_last=cfg["packing"].get("drop_last", True),
    )

    bin_path = out_dir / f"{cfg['dataset']['id']}_{split_name}.bin"
    flat.tofile(bin_path)

    return {
        "split": split_name,
        "hf_split": hf_split,
        "path": str(bin_path.relative_to(ROOT)),
        "num_documents": len(docs),
        "num_sequences": int(num_seqs),
        "num_tokens": int(num_seqs * seq_len),
        "stored_tokens": int(flat.shape[0]),  # = num_sequences * seq_len + 1
        "dtype": str(dtype),
        "sha256": sha256_file(bin_path),
    }


def write_debug_shard(
    train_bin: Path,
    debug_bin: Path,
    seq_len: int,
    n_seqs: int,
    dtype: np.dtype,
) -> Dict[str, Any]:
    """Slice the first N sequences of the train shard into a debug file."""
    arr = np.memmap(train_bin, dtype=dtype, mode="r")
    needed = n_seqs * seq_len + 1
    if arr.shape[0] < needed:
        raise ValueError(
            f"train shard has {arr.shape[0]} tokens, debug needs {needed}"
        )
    arr[:needed].tofile(debug_bin)
    return {
        "path": str(debug_bin.relative_to(ROOT)),
        "num_sequences": n_seqs,
        "num_tokens": n_seqs * seq_len,
        "stored_tokens": int(needed),
        "sha256": sha256_file(debug_bin),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=Path)
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    seed = int(cfg["determinism"]["seed"])
    set_global_seed(seed)

    proc_dir = ROOT / cfg["output"]["processed_dir"]
    man_dir = ROOT / cfg["output"]["manifest_dir"]
    dbg_dir = ROOT / cfg["output"]["debug_dir"]
    for d in (proc_dir, man_dir, dbg_dir):
        d.mkdir(parents=True, exist_ok=True)

    tokenizer = load_tokenizer(cfg["tokenizer"]["name"])

    split_records: Dict[str, Dict[str, Any]] = {}
    for logical, hf_split in cfg["dataset"]["splits"].items():
        split_records[logical] = build_split(
            cfg, logical, hf_split, tokenizer, proc_dir
        )

    debug_record = None
    if "train" in split_records and cfg.get("debug", {}).get("num_sequences", 0) > 0:
        debug_record = write_debug_shard(
            train_bin=ROOT / split_records["train"]["path"],
            debug_bin=dbg_dir / f"{cfg['dataset']['id']}_debug.bin",
            seq_len=cfg["packing"]["sequence_length"],
            n_seqs=int(cfg["debug"]["num_sequences"]),
            dtype=np.dtype(cfg["output"]["bin_dtype"]),
        )

    manifest = {
        "dataset_name": cfg["dataset"]["id"],
        "tokenizer": cfg["tokenizer"]["name"],
        "tokenizer_vocab_size": tokenizer.vocab_size,
        "eos_token_id": tokenizer.eos_token_id,
        "sequence_length": cfg["packing"]["sequence_length"],
        "packing": cfg["packing"]["strategy"],
        "data_source": f"{cfg['dataset']['source']}/{cfg['dataset'].get('source_config','')}",
        "seed": seed,
        "shuffle_documents": cfg["determinism"]["shuffle_documents"],
        "shuffle_sequences": cfg["determinism"]["shuffle_sequences"],
        "splits": split_records,
        "num_tokens": sum(r["num_tokens"] for r in split_records.values()),
        "num_sequences": sum(r["num_sequences"] for r in split_records.values()),
        "debug": debug_record,
        "config_path": str(args.config),
    }

    manifest_path = man_dir / f"{cfg['dataset']['id']}.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n=== build summary ===")
    print(f"manifest: {manifest_path}")
    for split, rec in split_records.items():
        print(
            f"  {split:<5} docs={rec['num_documents']:>7} "
            f"seqs={rec['num_sequences']:>7} "
            f"tokens={rec['num_tokens']:>10} "
            f"sha256={rec['sha256'][:16]}…"
        )
    if debug_record:
        print(
            f"  debug seqs={debug_record['num_sequences']:>7} "
            f"sha256={debug_record['sha256'][:16]}…"
        )


if __name__ == "__main__":
    main()
