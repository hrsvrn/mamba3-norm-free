"""Sanity-check a built dataset against its manifest.

Checks:
  1. Every shard exists and has the expected SHA-256 (rebuild reproducibility).
  2. Stored token count == num_sequences * seq_len + 1 (no truncation).
  3. No empty/all-pad sequences.
  4. Train and val shards do not share any packed sequence (no leakage).
  5. Token ID range is within tokenizer vocab.
  6. EOS token actually appears (sanity for `add_eos_per_doc`).

Run:
    python scripts/data/verify_dataset.py --manifest data/manifests/wikitext_smoke.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while buf := f.read(chunk):
            h.update(buf)
    return h.hexdigest()


def _seq_view(arr: np.ndarray, seq_len: int, n_seqs: int) -> np.ndarray:
    """View the flat token array as (n_seqs, seq_len). Drops the +1 buffer."""
    return arr[: n_seqs * seq_len].reshape(n_seqs, seq_len)


def check_shard(rec: dict, seq_len: int, vocab_size: int, eos_id: int) -> list[str]:
    """Return a list of failure strings (empty == pass)."""
    fails: list[str] = []
    path = ROOT / rec["path"]

    if not path.exists():
        return [f"{rec['split']}: shard missing at {path}"]

    digest = sha256_file(path)
    if digest != rec["sha256"]:
        fails.append(
            f"{rec['split']}: sha256 mismatch — manifest {rec['sha256'][:16]}… "
            f"vs file {digest[:16]}… (rebuild produced different bytes)"
        )

    arr = np.memmap(path, dtype=np.dtype(rec["dtype"]), mode="r")

    expected = rec["num_sequences"] * seq_len + 1
    if arr.shape[0] != expected:
        fails.append(
            f"{rec['split']}: stored_tokens={arr.shape[0]} but manifest says {expected}"
        )

    seqs = _seq_view(arr, seq_len, rec["num_sequences"])

    # Empty sequence = all-zeros (or all eq to eos), which would mean a packing bug.
    all_eos = np.all(seqs == eos_id, axis=1)
    if all_eos.any():
        fails.append(
            f"{rec['split']}: {int(all_eos.sum())} sequences are all-EOS"
        )

    # Length sanity (trivially true given reshape, but check axis-1 size).
    if seqs.shape[1] != seq_len:
        fails.append(f"{rec['split']}: bad axis-1 length {seqs.shape[1]} != {seq_len}")

    if int(arr.max()) >= vocab_size:
        fails.append(
            f"{rec['split']}: token id {int(arr.max())} >= vocab_size {vocab_size}"
        )
    if int(arr.min()) < 0:
        fails.append(f"{rec['split']}: negative token id {int(arr.min())}")

    if not (arr == eos_id).any():
        fails.append(
            f"{rec['split']}: EOS id {eos_id} never appears — `add_eos_per_doc` likely off"
        )

    return fails


def check_no_overlap(splits: dict, seq_len: int) -> list[str]:
    """Hash every sequence in every split; any cross-split collision = leakage."""
    fails: list[str] = []
    sigs: dict[bytes, str] = {}
    for split_name, rec in splits.items():
        arr = np.memmap(ROOT / rec["path"], dtype=np.dtype(rec["dtype"]), mode="r")
        seqs = _seq_view(arr, seq_len, rec["num_sequences"])
        for i in range(rec["num_sequences"]):
            sig = bytes(seqs[i].tobytes())
            prev = sigs.get(sig)
            if prev is not None and prev != split_name:
                fails.append(
                    f"sequence overlap: {prev} and {split_name} share an identical "
                    f"{seq_len}-token chunk (sig={hashlib.sha1(sig).hexdigest()[:12]})"
                )
                # one example is enough; don't spam
                return fails
            sigs[sig] = split_name
    return fails


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument(
        "--skip-overlap",
        action="store_true",
        help="Skip the O(N) sequence-collision scan (useful on huge shards).",
    )
    args = p.parse_args()

    manifest = json.loads(args.manifest.read_text())
    seq_len = manifest["sequence_length"]
    vocab = manifest["tokenizer_vocab_size"]
    eos = manifest["eos_token_id"]

    print(f"verifying {manifest['dataset_name']} (seq_len={seq_len})")
    all_fails: list[str] = []

    for split_name, rec in manifest["splits"].items():
        fails = check_shard(rec, seq_len, vocab, eos)
        status = "OK" if not fails else "FAIL"
        print(f"  [{status}] {split_name}: {rec['num_sequences']} sequences")
        all_fails.extend(fails)

    if manifest.get("debug"):
        # Treat debug shard like a split for shard-level checks.
        dbg = dict(manifest["debug"])
        dbg.update({"split": "debug", "dtype": manifest["splits"]["train"]["dtype"]})
        fails = check_shard(dbg, seq_len, vocab, eos)
        status = "OK" if not fails else "FAIL"
        print(f"  [{status}] debug: {dbg['num_sequences']} sequences")
        all_fails.extend(fails)

    if not args.skip_overlap and len(manifest["splits"]) > 1:
        fails = check_no_overlap(manifest["splits"], seq_len)
        status = "OK" if not fails else "FAIL"
        print(f"  [{status}] cross-split overlap")
        all_fails.extend(fails)

    print()
    if all_fails:
        print("=== FAILURES ===")
        for f in all_fails:
            print(f"  - {f}")
        sys.exit(1)
    print("all checks passed.")


if __name__ == "__main__":
    main()
