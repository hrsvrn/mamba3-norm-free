"""Pretrain a 50M-parameter Mamba-3 baseline on 1B tokens of FineWebEdu.

This is the *baseline* run for the normalization-free Mamba-3 thesis: stock
Mamba-3 mixer (BCNorm + BCBias intact), official Triton kernels via mamba-og,
no element-wise stabilizer swaps. The output ppl is what every ablation will
be compared against.

Single-GPU:
    python pretraining/train_50m.py --config pretraining/configs/mamba3_50m.yaml

Multi-GPU (torchrun):
    torchrun --standalone --nproc_per_node=8 \\
        pretraining/train_50m.py --config pretraining/configs/mamba3_50m.yaml
"""

from __future__ import annotations

import argparse
import math
import os
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
# Train loop
# ----------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rank, world_size, local_rank = setup_distributed()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")

    # Seeding (rank-different so dataloader shards diverge but model init agrees).
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
    log(rank, f"[rank {rank}] building Mamba-3 50M (dtype={param_dtype})")
    model = build_model_from_config(cfg, device=device, dtype=param_dtype)
    n_params = model.num_params()
    log(rank, f"[rank {rank}] params = {n_params:,} ({n_params/1e6:.2f}M)")

    if cfg["training"]["gradient_checkpointing"]:
        # Block-level checkpointing — wrap each Block.forward
        from torch.utils.checkpoint import checkpoint as _ckpt
        for block in model.layers:
            orig = block.forward
            block.forward = lambda h, r, _o=orig: _ckpt(_o, h, r, use_reentrant=False)

    # ---------------------------------------------------------------- Probes
    # Attach BEFORE DDP/FSDP wrapping so hooks see the unwrapped modules.
    probe_cfg = cfg.get("probes", {})
    probe = None
    if probe_cfg.get("enabled", False) and is_main(rank):
        out_dir_for_probe = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"] / probe_cfg.get("dump_dir", "probes")
        probe = BCNormProbe(
            dump_dir=out_dir_for_probe,
            dump_every=probe_cfg.get("dump_raw_every", 0),
            hist_subsample=probe_cfg.get("hist_subsample", 10000),
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

            # Declare metric axes so the wandb UI groups everything against `step`.
            wandb.define_metric("step")
            wandb.define_metric("train/*", step_metric="step")
            wandb.define_metric("bcnorm/*", step_metric="step")
            wandb.define_metric("bcnorm_hist/*", step_metric="step")
            wandb.define_metric("depth/*", step_metric="step")
            wandb.define_metric("system/*", step_metric="step")

            # Top-line model summary
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

            # Watch all parameters + gradients across the full model.
            # The probe is BCNorm-targeted; wandb.watch covers everything else
            # (in_proj, out_proj, embedding, lm_head, dt_bias, D, ...).
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

    # ----------------------------------------------------------------- Resume
    out_dir = Path(cfg["run"]["out_dir"]) / cfg["run"]["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
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
        # Decide whether THIS optimizer step should also produce probe stats.
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
        dump_this_step = (
            probe_this_step
            and probe_cfg.get("dump_raw_every", 0) > 0
            and step % probe_cfg["dump_raw_every"] == 0
        )
        if dump_this_step:
            probe.request_dump()
        if hist_this_step:
            probe.request_histograms()
        # The "sample" micro-step (we use the last one so gradient is fully populated).
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

        # Probe flush has to land AFTER backward (grads exist) and BEFORE
        # optimizer.step zeroes them implicitly via fused AdamW.
        probe_table_str = None
        if probe_this_step:
            probe_out = probe.flush(
                wb,
                step,
                log_histograms=hist_this_step,
                log_depth_plots=depth_plot_this_step,
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
                # Per-param-group LR (decay vs no-decay).
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
            if wb is not None and wb_cfg.get("save_checkpoints_as_artifacts", False):
                try:
                    import wandb
                    art = wandb.Artifact(
                        name=f"{cfg['run']['name']}-step{step:06d}",
                        type="model",
                        metadata={"step": step, "loss": loss_accum},
                    )
                    art.add_file(str(ckpt_path))
                    wb.log_artifact(art, aliases=[f"step-{step}", "latest"])
                except Exception as e:
                    log(rank, f"[rank 0] artifact upload error: {e}")

    # Final checkpoint
    if is_main(rank):
        ckpt_path = out_dir / "final.pt"
        state = (model.module if hasattr(model, "module") else model).state_dict()
        torch.save({"model": state, "step": total_steps, "config": cfg}, ckpt_path)
        log(rank, f"[rank 0] done in {(time.time()-t_start)/60:.1f} min — saved {ckpt_path}")

        if wb is not None:
            try:
                wb.summary["train/final_loss"] = loss_accum
                wb.summary["train/wall_time_min"] = (time.time() - t_start) / 60.0
                wb.summary["train/tokens_seen"] = tokens_seen
                if wb_cfg.get("save_checkpoints_as_artifacts", False):
                    import wandb
                    art = wandb.Artifact(
                        name=f"{cfg['run']['name']}-final",
                        type="model",
                        metadata={
                            "step": total_steps,
                            "n_params": n_params,
                            "config": cfg,
                        },
                    )
                    art.add_file(str(ckpt_path))
                    wb.log_artifact(art)
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
