"""Pretrain a 180M-parameter Mamba-3 baseline on 10B tokens of FineWebEdu.

Shape mirrors the Mamba-2 paper Table 9 "125M" scaling-law point, inflated
to ~185M by the Llama-3.1 tokenizer (vocab 128256). Training recipe follows
Mamba-2 Appendix D.2 (which the Mamba-3 paper defers to in its Appendix D):
AdamW + cosine to floor + weight decay 0.1 + grad-clip 1.0, bf16.

Differences vs train_50m.py:
    * defaults to configs/mamba3_180m.yaml (10B tokens, Llama-3.1 tokenizer)
    * pushes intermediate + final checkpoints to Hugging Face Hub when
      `hf_hub.enabled: true` is set in the config (auth via HF_TOKEN env)

Single-GPU:
    python pretraining/train_180m.py --config pretraining/configs/mamba3_180m.yaml

Multi-GPU (torchrun):
    torchrun --standalone --nproc_per_node=8 \\
        pretraining/train_180m.py --config pretraining/configs/mamba3_180m.yaml
"""

from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "pretraining"))

from data import build_dataloader  # noqa: E402
from model import build_model_from_config  # noqa: E402
from probes import BCNormProbe, format_probe_table  # noqa: E402


# ----------------------------------------------------------------------------
# Distributed helpers
# ----------------------------------------------------------------------------

def setup_distributed() -> tuple[int, int, int]:
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count()))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def log(rank: int, msg: str):
    if is_main(rank):
        print(msg, flush=True)


# ----------------------------------------------------------------------------
# LR schedule
# ----------------------------------------------------------------------------

def cosine_lr(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1.0 + math.cos(math.pi * progress))


def linear_lr(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    if step >= total:
        return floor
    progress = (step - warmup) / max(1, total - warmup)
    return peak - (peak - floor) * progress


def constant_lr(step: int, warmup: int, total: int, peak: float, floor: float) -> float:
    if step < warmup:
        return peak * (step + 1) / max(1, warmup)
    return peak


SCHEDULES = {"cosine": cosine_lr, "linear": linear_lr, "constant": constant_lr}


# ----------------------------------------------------------------------------
# Param groups (no weight decay on biases / norms / SSM-internal vectors)
# ----------------------------------------------------------------------------

def build_param_groups(model: torch.nn.Module, weight_decay: float):
    decay, no_decay = [], []
    seen = set()
    for name, p in model.named_parameters():
        if not p.requires_grad or id(p) in seen:
            continue
        seen.add(id(p))
        skip = (
            p.ndim < 2
            or name.endswith(".bias")
            or "norm" in name.lower()
            or getattr(p, "_no_weight_decay", False)
        )
        (no_decay if skip else decay).append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ----------------------------------------------------------------------------
# Hugging Face Hub upload
# ----------------------------------------------------------------------------

class HubPusher:
    """Lazy HF Hub uploader. Creates the repo on first push; uploads each
    checkpoint as a separate file under the repo root. Token auth via the
    HF_TOKEN env var or a pre-existing `huggingface-cli login` cache."""

    def __init__(self, cfg: dict, run_name: str, run_dir: Path):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.push_every = int(cfg.get("push_every", 0))
        self.push_intermediate = bool(cfg.get("push_intermediate", True))
        self.push_final = bool(cfg.get("push_final", True))
        self.commit_prefix = cfg.get("commit_message_prefix", "[mamba3]")
        self.run_name = run_name
        self.run_dir = run_dir
        self.repo_id = self._resolve_repo_id(cfg.get("repo_id"))
        self.private = bool(cfg.get("private", False))
        self._api = None
        self._repo_ready = False

        if self.enabled and self.repo_id is None:
            raise RuntimeError(
                "hf_hub.enabled but no repo_id resolved. Set HF_REPO_ID in env "
                "or hf_hub.repo_id in the config."
            )

    def _resolve_repo_id(self, repo_id_cfg):
        env_repo = os.environ.get("HF_REPO_ID")
        if repo_id_cfg:
            return repo_id_cfg
        if env_repo:
            return env_repo
        if not self.enabled:
            return None
        # Last-resort default: <username>/<run_name>. Only valid if whoami works.
        try:
            from huggingface_hub import HfApi

            who = HfApi().whoami(token=os.environ.get("HF_TOKEN"))
            user = who.get("name") or who.get("username")
            if user:
                return f"{user}/{self.run_name.replace('_', '-')}"
        except Exception:
            pass
        return None

    def _ensure_repo(self):
        if self._repo_ready:
            return
        from huggingface_hub import HfApi

        self._api = HfApi(token=os.environ.get("HF_TOKEN"))
        self._api.create_repo(
            self.repo_id,
            repo_type="model",
            private=self.private,
            exist_ok=True,
        )
        self._repo_ready = True

    def _push_text(self, content: str, path_in_repo: str, message: str):
        self._ensure_repo()
        self._api.upload_file(
            path_or_fileobj=content.encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=message,
        )

    def push_file(self, local_path: Path, path_in_repo: str, message: str):
        if not self.enabled:
            return
        self._ensure_repo()
        self._api.upload_file(
            path_or_fileobj=str(local_path),
            path_in_repo=path_in_repo,
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=f"{self.commit_prefix} {message}",
        )

    def push_initial_config(self, cfg_full: dict):
        if not self.enabled:
            return
        # Snapshot the resolved config under the repo for reproducibility.
        snapshot = yaml.safe_dump(cfg_full, sort_keys=False)
        self._push_text(
            snapshot,
            path_in_repo="config.yaml",
            message=f"{self.commit_prefix} initial config snapshot",
        )

    def push_readme(self, n_params: int, total_steps: int):
        if not self.enabled:
            return
        readme = _render_readme(self.run_name, self.repo_id, n_params, total_steps, self.cfg)
        self._push_text(readme, "README.md", f"{self.commit_prefix} README")

    def push_checkpoint(self, ckpt_path: Path, step: int, is_final: bool = False):
        if not self.enabled:
            return
        if is_final:
            if not self.push_final:
                return
        else:
            if not self.push_intermediate:
                return
            if self.push_every <= 0:
                return
            if step % self.push_every != 0:
                return
        tag = "final" if is_final else f"step{step:06d}"
        self.push_file(
            ckpt_path,
            path_in_repo=ckpt_path.name,
            message=f"checkpoint {tag} ({step} steps)",
        )


def _render_readme(run_name, repo_id, n_params, total_steps, hub_cfg):
    return f"""---
license: apache-2.0
library_name: pytorch
tags:
  - mamba
  - mamba-3
  - state-space-model
  - language-model
  - fineweb-edu
datasets:
  - HuggingFaceFW/fineweb-edu
---

# {run_name}

Mamba-3 (SISO) language model, ~{n_params/1e6:.1f}M parameters, pretrained on
**10B tokens of FineWeb-Edu** with the **Llama-3.1 tokenizer**.

Trained as the baseline run for the *normalization-free Mamba-3* research
project (ICLR 2026 thesis: replacing BCNorm with element-wise stabilizers).
This checkpoint keeps BCNorm + BC bias intact and is the reference against
which all DyT / Derf / DyISRU / DySoftSign ablations are compared.

## Architecture

| | Value |
|---|---|
| Family | Mamba-3 SISO (Llama-style alternating SSM + SwiGLU MLP) |
| Layers | 12 |
| Model dim | 768 |
| State size (d_state) | 128 |
| Head dim | 64 |
| Expand | 2 |
| MLP intermediate | 1500 |
| Vocab | 128256 (Llama-3.1) |
| Tied embeddings | yes |
| Sequence length | 2048 |

## Training recipe

| | Value |
|---|---|
| Optimizer | AdamW (β=0.9, 0.95), wd=0.1, grad-clip=1.0 |
| LR schedule | cosine, peak 6e-4 → 6e-5, 1000 warmup steps |
| Batch (tokens) | 524288 (256 sequences × 2048) |
| Total tokens | 10B |
| Total steps | {total_steps:,} |
| Precision | bfloat16 |

See `config.yaml` for the full, reproducible config snapshot.

## Loading

```python
import torch, yaml
from huggingface_hub import hf_hub_download

cfg = yaml.safe_load(open(hf_hub_download("{repo_id}", "config.yaml")))
ckpt = torch.load(hf_hub_download("{repo_id}", "final.pt"), map_location="cpu")
# Reconstruct via pretraining/model.py:build_model_from_config(cfg)
```
"""


# ----------------------------------------------------------------------------
# Train loop
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=str(REPO_ROOT / "pretraining/configs/mamba3_180m.yaml"))
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    seed = cfg["run"]["seed"]
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    if cfg["training"]["tf32"]:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    precision = cfg["training"]["precision"]
    if precision == "bf16":
        param_dtype = torch.bfloat16
    elif precision == "fp16":
        param_dtype = torch.float16
    else:
        param_dtype = torch.float32

    # ------------------------------------------------------------------ Model
    log(rank, f"[rank {rank}] building Mamba-3 180M (dtype={param_dtype})")
    model = build_model_from_config(cfg, device=device, dtype=param_dtype)
    n_params = model.num_params()
    log(rank, f"[rank {rank}] params = {n_params:,} ({n_params/1e6:.2f}M)")

    if cfg["training"]["gradient_checkpointing"]:
        from torch.utils.checkpoint import checkpoint as _ckpt
        for block in model.layers:
            orig = block.forward
            block.forward = lambda h, r, _o=orig: _ckpt(_o, h, r, use_reentrant=False)

    # ---------------------------------------------------------------- Probes
    probe_cfg = cfg.get("probes", {})
    probe = None
    if probe_cfg.get("enabled", False) and is_main(rank):
        out_dir_for_probe = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / probe_cfg.get("dump_dir", "probes")
        probe = BCNormProbe(
            dump_dir=out_dir_for_probe,
            dump_every=probe_cfg.get("dump_raw_every", 0),
            hist_subsample=probe_cfg.get("hist_subsample", 10000),
            per_layer_keys=probe_cfg.get("per_layer_keys"),
        )
        probe.attach(model)

    if cfg["distributed"]["fsdp"] and world_size > 1:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp import ShardingStrategy
        strat = {
            "full": ShardingStrategy.FULL_SHARD,
            "grad_op": ShardingStrategy.SHARD_GRAD_OP,
            "hybrid": ShardingStrategy.HYBRID_SHARD,
        }[cfg["distributed"]["fsdp_sharding"]]
        model = FSDP(model, sharding_strategy=strat, device_id=local_rank)
    elif world_size > 1:
        from torch.nn.parallel import DistributedDataParallel as DDP
        model = DDP(model, device_ids=[local_rank])

    if cfg["training"]["compile"]:
        log(rank, "[rank 0] torch.compile enabled — first step will be slow")
        model = torch.compile(model)

    # -------------------------------------------------------------- Optimizer
    opt_cfg = cfg["optimizer"]
    param_groups = build_param_groups(
        model.module if hasattr(model, "module") else model,
        weight_decay=opt_cfg["weight_decay"],
    )
    optimizer = torch.optim.AdamW(
        param_groups,
        lr=cfg["lr"]["peak"],
        betas=tuple(opt_cfg["betas"]),
        eps=opt_cfg["eps"],
        fused=opt_cfg["fused"] and torch.cuda.is_available(),
    )

    # ---------------------------------------------------------------- Schedule
    train = cfg["training"]
    seq_len = train["seq_len"]
    micro_bs = train["micro_batch_size"]
    global_batch = train["batch_tokens"] // seq_len
    micro_per_step_global = global_batch
    micro_per_step_local = micro_per_step_global // world_size
    grad_accum = micro_per_step_local // micro_bs

    total_steps = train["total_tokens"] // train["batch_tokens"]
    schedule_fn = SCHEDULES[cfg["lr"]["schedule"]]

    log(
        rank,
        f"[rank 0] tokens={train['total_tokens']:,}  batch_tokens={train['batch_tokens']:,}  "
        f"seq_len={seq_len}  global_batch={global_batch}  grad_accum={grad_accum}  "
        f"total_steps={total_steps}",
    )

    # ----------------------------------------------------------------- Data
    loader = build_dataloader(
        cfg,
        seq_len=seq_len,
        micro_batch_size=micro_bs,
        rank=rank,
        world_size=world_size,
        num_workers=cfg["data"]["num_workers"],
        seed=seed,
    )
    data_iter = iter(loader)

    # ----------------------------------------------------------------- wandb
    wb = None
    wb_cfg = cfg.get("wandb", {})
    if is_main(rank) and cfg["run"]["wandb_mode"] != "disabled":
        try:
            import wandb

            wb = wandb.init(
                project=cfg["run"]["wandb_project"],
                name=cfg["run"]["name"],
                config=cfg,
                mode=cfg["run"]["wandb_mode"],
                tags=wb_cfg.get("tags", []),
            )

            wandb.define_metric("step")
            wandb.define_metric("train/*", step_metric="step")
            wandb.define_metric("bcnorm/*", step_metric="step")
            wandb.define_metric("bcnorm_hist/*", step_metric="step")
            wandb.define_metric("depth/*", step_metric="step")
            wandb.define_metric("system/*", step_metric="step")

            wandb.run.summary["model/params_M"] = round(n_params / 1e6, 3)
            wandb.run.summary["model/n_params"] = n_params
            wandb.run.summary["model/d_model"] = cfg["model"]["d_model"]
            wandb.run.summary["model/n_layers"] = cfg["model"]["n_layers"]
            wandb.run.summary["model/d_state"] = cfg["model"]["d_state"]
            wandb.run.summary["model/headdim"] = cfg["model"]["head_dim"]
            wandb.run.summary["model/vocab_size"] = cfg["model"]["vocab_size"]
            wandb.run.summary["training/total_steps"] = total_steps
            wandb.run.summary["training/global_batch"] = global_batch
            wandb.run.summary["training/grad_accum"] = grad_accum

            if wb_cfg.get("watch_model", True):
                inner = model.module if hasattr(model, "module") else model
                wandb.watch(
                    inner,
                    log=wb_cfg.get("watch_log", "all"),
                    log_freq=wb_cfg.get("watch_log_freq", 200),
                    log_graph=False,
                )
        except ImportError:
            log(rank, "[rank 0] wandb not installed — continuing without it")

    # ----------------------------------------------------------------- Hub
    out_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    hub = None
    if is_main(rank):
        try:
            hub = HubPusher(cfg.get("hf_hub", {}), run_name=cfg["run"]["name"], run_dir=out_dir)
            if hub.enabled:
                log(rank, f"[rank 0] HF Hub push enabled -> {hub.repo_id} (private={hub.private})")
                # Stage README + config first so the repo is browsable from step 0.
                hub.push_initial_config(cfg)
                hub.push_readme(n_params=n_params, total_steps=total_steps)
                # Also drop the YAML next to the checkpoints locally for convenience.
                shutil.copy(args.config, out_dir / "config.yaml")
        except Exception as e:
            log(rank, f"[rank 0] HF Hub init failed: {e} — continuing without upload")
            hub = None

    # ----------------------------------------------------------------- Resume
    start_step = 0
    if args.resume:
        log(rank, f"[rank 0] resuming from {args.resume}")
        ckpt = torch.load(args.resume, map_location="cpu")
        (model.module if hasattr(model, "module") else model).load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]

    # ---------------------------------------------------------------- Training
    model.train()
    tokens_seen = start_step * train["batch_tokens"]
    t_start = time.time()
    last_log_t = t_start
    last_log_tokens = tokens_seen
    loss_accum = 0.0

    for step in range(start_step, total_steps):
        step_t0 = time.time()
        lr = schedule_fn(
            step,
            warmup=cfg["lr"]["warmup_steps"],
            total=total_steps,
            peak=cfg["lr"]["peak"],
            floor=cfg["lr"]["min"],
        )
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        optimizer.zero_grad(set_to_none=True)
        loss_accum = 0.0
        probe_this_step = (
            probe is not None
            and probe_cfg.get("log_every", 0) > 0
            and step % probe_cfg["log_every"] == 0
        )
        hist_this_step = (
            probe_this_step
            and probe_cfg.get("hist_every", 0) > 0
            and step % probe_cfg["hist_every"] == 0
        )
        depth_plot_this_step = (
            probe_this_step
            and probe_cfg.get("depth_plot_every", 0) > 0
            and step % probe_cfg["depth_plot_every"] == 0
        )
        scatter_this_step = (
            probe_this_step
            and probe_cfg.get("scatter_every", 0) > 0
            and step % probe_cfg["scatter_every"] == 0
            and bool(probe_cfg.get("scatter_layers"))
        )
        dump_this_step = (
            probe_this_step
            and probe_cfg.get("dump_raw_every", 0) > 0
            and step % probe_cfg["dump_raw_every"] == 0
        )
        if dump_this_step:
            probe.request_dump()
        if hist_this_step or scatter_this_step:
            probe.request_histograms()
        sample_micro = grad_accum - 1
        for micro_step in range(grad_accum):
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                batch = next(data_iter)
            batch = batch.to(device, non_blocking=True)
            input_ids = batch[:, :-1]
            labels = batch[:, 1:]

            sync_ctx = (
                model.no_sync()
                if hasattr(model, "no_sync") and micro_step < grad_accum - 1
                else _NullCtx()
            )
            if probe_this_step and micro_step == sample_micro:
                probe.enable()
            with sync_ctx:
                out = model(input_ids, labels=labels)
                loss = out.loss / grad_accum
                loss.backward()
            if probe is not None:
                probe.disable()
            loss_accum += loss.detach().float().item()

        grad_norm = torch.nn.utils.clip_grad_norm_(
            (model.module if hasattr(model, "module") else model).parameters(),
            max_norm=opt_cfg["grad_clip"],
        )

        probe_table_str = None
        if probe_this_step:
            probe_out = probe.flush(
                wb,
                step,
                log_histograms=hist_this_step,
                log_depth_plots=depth_plot_this_step,
                scatter_layers=(
                    probe_cfg.get("scatter_layers") if scatter_this_step else None
                ),
            )
            if (
                probe_cfg.get("print_table_every", 0) > 0
                and step % probe_cfg["print_table_every"] == 0
                and is_main(rank)
            ):
                probe_table_str = format_probe_table(probe_out)

        optimizer.step()

        tokens_seen += train["batch_tokens"]
        step_secs = time.time() - step_t0

        if step % cfg["run"]["log_every"] == 0 and is_main(rank):
            now = time.time()
            tok_per_sec = (tokens_seen - last_log_tokens) / max(1e-6, now - last_log_t)
            last_log_t = now
            last_log_tokens = tokens_seen
            ppl = math.exp(min(20.0, loss_accum))
            samples_per_sec = tok_per_sec / seq_len
            log(
                rank,
                f"step {step:>6d}/{total_steps}  loss={loss_accum:.4f}  ppl={ppl:.2f}  "
                f"lr={lr:.2e}  grad_norm={grad_norm:.2f}  "
                f"tok/s={tok_per_sec/1e3:.1f}k  tokens={tokens_seen/1e6:.1f}M  "
                f"step_t={step_secs*1000:.0f}ms",
            )
            if wb is not None:
                payload = {
                    "step": step,
                    "train/loss": loss_accum,
                    "train/ppl": ppl,
                    "train/lr": lr,
                    "train/grad_norm": float(grad_norm),
                    "train/tokens": tokens_seen,
                    "train/tokens_per_sec": tok_per_sec,
                    "train/samples_per_sec": samples_per_sec,
                    "train/step_secs": step_secs,
                    "train/epoch_progress": tokens_seen / train["total_tokens"],
                }
                if wb_cfg.get("log_system_metrics", True) and torch.cuda.is_available():
                    payload.update(
                        {
                            "system/cuda_mem_alloc_GB": torch.cuda.memory_allocated() / 1e9,
                            "system/cuda_mem_reserved_GB": torch.cuda.memory_reserved() / 1e9,
                            "system/cuda_max_mem_alloc_GB": torch.cuda.max_memory_allocated() / 1e9,
                        }
                    )
                for i, pg in enumerate(optimizer.param_groups):
                    payload[f"train/lr_group{i}"] = pg["lr"]
                wb.log(payload, step=step)

        if probe_table_str is not None:
            log(rank, "\n" + probe_table_str + "\n")

        if cfg["run"]["save_every"] > 0 and step > 0 and step % cfg["run"]["save_every"] == 0 and is_main(rank):
            ckpt_path = out_dir / f"step_{step:06d}.pt"
            state = (model.module if hasattr(model, "module") else model).state_dict()
            torch.save(
                {"model": state, "optimizer": optimizer.state_dict(), "step": step, "config": cfg},
                ckpt_path,
            )
            log(rank, f"[rank 0] saved {ckpt_path}")
            if hub is not None:
                try:
                    hub.push_checkpoint(ckpt_path, step=step, is_final=False)
                except Exception as e:
                    log(rank, f"[rank 0] HF Hub push error at step {step}: {e}")

    # Final checkpoint
    if is_main(rank):
        ckpt_path = out_dir / "final.pt"
        state = (model.module if hasattr(model, "module") else model).state_dict()
        torch.save({"model": state, "step": total_steps, "config": cfg}, ckpt_path)
        log(rank, f"[rank 0] done in {(time.time()-t_start)/60:.1f} min — saved {ckpt_path}")

        if hub is not None:
            try:
                hub.push_checkpoint(ckpt_path, step=total_steps, is_final=True)
                log(rank, f"[rank 0] final checkpoint pushed to https://huggingface.co/{hub.repo_id}")
            except Exception as e:
                log(rank, f"[rank 0] HF Hub final push error: {e}")

        if wb is not None:
            try:
                wb.summary["train/final_loss"] = loss_accum
                wb.summary["train/wall_time_min"] = (time.time() - t_start) / 60.0
                wb.summary["train/tokens_seen"] = tokens_seen
                if hub is not None and hub.enabled:
                    wb.summary["hf_hub/repo_id"] = hub.repo_id
                wb.finish()
            except Exception as e:
                log(rank, f"[rank 0] wandb cleanup error: {e}")

    if world_size > 1:
        dist.destroy_process_group()


class _NullCtx:
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False


if __name__ == "__main__":
    main()
