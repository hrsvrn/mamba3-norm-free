"""BCNorm probes for the Mamba-3 baseline.

The thesis of this repository is that BCNorm (RMSNormGated on B and C inside the
SSM block) can be replaced by an element-wise squash. To know whether *any*
replacement is acceptable we first need to know what BCNorm is actually doing:
- How heavy-tailed are B and C *before* the norm? (kurtosis, max-abs, p99.9)
- How much variance does BCNorm collapse? (norm_ratio, post-RMS)
- Does the learnable scale drift over training? (weight mean/std/max)
- Do B_bias / C_bias do real work or stay near init? (bias mean/std)
- Where does the gradient go? (∂L/∂w on the norm scale)
- Are pre-norm distributions stable across layers, or does depth amplify
  outliers? (per-layer percentile fan-out)

The probe attaches forward hooks to every B_norm and C_norm in the model and
computes stats on-device in a single sweep. Stats are buffered as scalar
tensors so we never sync to CPU during the forward pass. A `flush(step)` call
copies everything to CPU once per log interval and pushes it to wandb under
keys of the form `bcnorm/L00/B/<stat>`.

To keep overhead bounded the probe is *gated*: the trainer calls `enable()`
for exactly one micro-step per logged training step, so the expensive stats
fire ~1× per `log_every`, not 1× per gradient-accumulation micro-batch."""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn


# Subsample size for quantile / percentile estimation. Quantile sort is
# O(N log N); 50k is enough for stable p99.9 and keeps the hook cost <1ms.
_QUANT_SUBSAMPLE = 50_000
_PERCENTILES = (0.5, 0.9, 0.99, 0.999)
# Tail thresholds (in units of pre-norm std). Used for "what fraction of values
# fall outside ±k·σ" — proxy for how heavy the tail is that BCNorm has to absorb.
_TAIL_SIGMAS = (3.0, 6.0, 12.0)


@torch.no_grad()
def _subsample(x: torch.Tensor, n: int) -> torch.Tensor:
    """Random subsample of a flat tensor down to `n` values (no-op if smaller)."""
    if x.numel() <= n:
        return x
    idx = torch.randint(0, x.numel(), (n,), device=x.device)
    return x[idx]


@torch.no_grad()
def _kurtosis(x: torch.Tensor) -> torch.Tensor:
    """Excess kurtosis. Gaussian → 0. Heavy tails → large positive."""
    mu = x.mean()
    diff = x - mu
    var = diff.pow(2).mean()
    return diff.pow(4).mean() / (var.pow(2) + 1e-12) - 3.0


@torch.no_grad()
def _quantiles_abs(x: torch.Tensor, qs: tuple[float, ...]) -> torch.Tensor:
    flat = x.abs().flatten()
    if flat.numel() > _QUANT_SUBSAMPLE:
        idx = torch.randint(0, flat.numel(), (_QUANT_SUBSAMPLE,), device=flat.device)
        flat = flat[idx]
    q = torch.tensor(qs, device=flat.device, dtype=flat.dtype)
    return torch.quantile(flat, q)


@torch.no_grad()
def _per_token_norm(x: torch.Tensor) -> torch.Tensor:
    """L2 norm along the last (channel) dim, averaged over everything else."""
    return x.float().norm(dim=-1).mean()


@torch.no_grad()
def _compute_stats(pre: torch.Tensor, post: torch.Tensor) -> dict[str, torch.Tensor]:
    """One sweep of stats on a (B, L, G, S) tensor. Stays on device."""
    pre_f = pre.float()
    post_f = post.float()
    pre_flat = pre_f.flatten()
    post_flat = post_f.flatten()

    pre_std = pre_flat.std()
    pre_abs = pre_flat.abs()

    stats: dict[str, torch.Tensor] = {
        "pre/mean": pre_flat.mean(),
        "pre/std": pre_std,
        "pre/max_abs": pre_abs.max(),
        "pre/min": pre_flat.min(),
        "pre/max": pre_flat.max(),
        "pre/kurtosis": _kurtosis(pre_flat),
        "pre/norm_per_token": _per_token_norm(pre_f),
        "post/mean": post_flat.mean(),
        "post/std": post_flat.std(),
        "post/max_abs": post_flat.abs().max(),
        "post/min": post_flat.min(),
        "post/max": post_flat.max(),
        "post/kurtosis": _kurtosis(post_flat),
        "post/norm_per_token": _per_token_norm(post_f),
        "post/rms": post_f.pow(2).mean(-1).sqrt().mean(),
    }

    # Norm-ratio: how much does BCNorm rescale each token vector?
    # >1 → BCNorm amplifies, <1 → BCNorm shrinks.
    pre_n = pre_f.float().norm(dim=-1)
    post_n = post_f.float().norm(dim=-1)
    ratio = post_n / (pre_n + 1e-8)
    stats["delta/norm_ratio_mean"] = ratio.mean()
    stats["delta/norm_ratio_std"] = ratio.std()
    stats["delta/norm_ratio_max"] = ratio.max()

    # Pre-norm percentiles (the tail BCNorm has to handle).
    qs = _quantiles_abs(pre_f, _PERCENTILES)
    for q_val, q_t in zip(_PERCENTILES, qs.unbind()):
        tag = f"p{int(q_val*1000):04d}" if q_val >= 0.99 else f"p{int(q_val*100):02d}"
        stats[f"pre/abs_{tag}"] = q_t

    # Tail-mass: fraction of pre-norm values above k·σ.
    for k in _TAIL_SIGMAS:
        stats[f"pre/frac_above_{int(k)}sigma"] = (pre_abs > k * pre_std).float().mean()

    return stats


class BCNormProbe:
    """Forward-hook probe for every B_norm/C_norm in a Mamba-3 stack."""

    def __init__(
        self,
        dump_dir: Path | None = None,
        dump_every: int = 0,
        hist_subsample: int = 10_000,
    ):
        self.enabled = False
        self.dump_dir = Path(dump_dir) if dump_dir is not None else None
        self.dump_every = dump_every
        self.hist_subsample = hist_subsample
        self._dump_pending = False
        self._hist_pending = False
        self._dump_buffer: list[dict] = []

        # buffer[(layer_idx, "B"|"C")] -> list[dict[name, tensor]]
        self._buffer: dict[tuple[int, str], list[dict[str, torch.Tensor]]] = defaultdict(list)
        # Per-(layer, tag) flat samples for histogram logging.
        self._hist_buffer: dict[tuple[int, str], dict[str, torch.Tensor]] = defaultdict(dict)
        # the module references — kept so we can also log norm-weight & gradient stats
        self._modules: dict[tuple[int, str], nn.Module] = {}
        self._mixers: list = []     # for B_bias / C_bias param tracking
        self._handles: list = []

    # ------------------------------------------------------------------ Hook
    def attach(self, model: nn.Module) -> None:
        layers = getattr(model, "layers", None)
        if layers is None and hasattr(model, "backbone"):
            layers = model.backbone.layers
        if layers is None:
            raise AttributeError("Could not find `.layers` on model for probe attachment")

        for layer_idx, block in enumerate(layers):
            mixer = block.mixer
            self._mixers.append((layer_idx, mixer))
            self._register(mixer.B_norm, layer_idx, "B")
            self._register(mixer.C_norm, layer_idx, "C")
        print(f"[probe] attached to {len(self._modules)} BCNorm modules across {len(layers)} layers")

    def _register(self, module: nn.Module, layer_idx: int, tag: str) -> None:
        self._modules[(layer_idx, tag)] = module

        def _hook(_mod, inp, out):
            if not self.enabled:
                return
            pre = inp[0].detach()
            post = out.detach()
            stats = _compute_stats(pre, post)
            self._buffer[(layer_idx, tag)].append(stats)
            if self._hist_pending:
                # Subsample flat values for histogram shipping — keeps wandb payload small.
                self._hist_buffer[(layer_idx, tag)] = {
                    "pre": _subsample(pre.float().flatten(), self.hist_subsample).cpu(),
                    "post": _subsample(post.float().flatten(), self.hist_subsample).cpu(),
                }
            if self._dump_pending:
                # Hold onto a small slice of the raw tensors for offline analysis.
                # Slicing keeps the dump file tractable (a few MB per step).
                self._dump_buffer.append(
                    {
                        "layer": layer_idx,
                        "tensor": tag,
                        "pre": pre[:1].detach().cpu().to(torch.float32).clone(),
                        "post": post[:1].detach().cpu().to(torch.float32).clone(),
                    }
                )

        self._handles.append(module.register_forward_hook(_hook))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles.clear()
        self._modules.clear()
        self._mixers.clear()

    # --------------------------------------------------------------- Control
    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def request_dump(self) -> None:
        """Set a flag so the next enabled hook call also records raw tensors."""
        self._dump_pending = True

    def request_histograms(self) -> None:
        """Set a flag so the next enabled hook call also stashes histogram samples."""
        self._hist_pending = True

    # --------------------------------------------------------------- Flush
    def flush(self, wb, step: int, log_histograms: bool = False, log_depth_plots: bool = False) -> dict[str, float]:
        """Average buffered stats, log to wandb, and return a plain-Python dict.

        Args:
            wb: a wandb run (or None to skip the network call)
            step: training step, used as the wandb x-axis
            log_histograms: if True, also flush wandb.Histogram payloads
                            (requires `request_histograms()` to have been set before forward)
            log_depth_plots: if True, also push wandb.plot.line_series depth-profile charts
        """
        out: dict[str, float] = {}

        # --- activation stats ----
        for (layer, tag), stats_list in self._buffer.items():
            if not stats_list:
                continue
            avg: dict[str, torch.Tensor] = {}
            for k in stats_list[0]:
                avg[k] = torch.stack([s[k] for s in stats_list]).mean()
            for k, v in avg.items():
                out[f"bcnorm/L{layer:02d}/{tag}/{k}"] = float(v.item())

        # --- learnable parameters: norm scale + B/C bias ----
        for (layer, tag), mod in self._modules.items():
            w = mod.weight.detach().float()
            out[f"bcnorm/L{layer:02d}/{tag}/weight_mean"] = float(w.mean().item())
            out[f"bcnorm/L{layer:02d}/{tag}/weight_std"] = float(w.std().item())
            out[f"bcnorm/L{layer:02d}/{tag}/weight_max_abs"] = float(w.abs().max().item())
            if w.grad is not None:
                g = w.grad.detach().float()
                out[f"bcnorm/L{layer:02d}/{tag}/weight_grad_norm"] = float(g.norm().item())
                out[f"bcnorm/L{layer:02d}/{tag}/weight_grad_max_abs"] = float(g.abs().max().item())

        for layer_idx, mixer in self._mixers:
            for bias_name in ("B_bias", "C_bias"):
                p = getattr(mixer, bias_name, None)
                if p is None:
                    continue
                pf = p.detach().float()
                out[f"bcnorm/L{layer_idx:02d}/{bias_name}/mean"] = float(pf.mean().item())
                out[f"bcnorm/L{layer_idx:02d}/{bias_name}/std"] = float(pf.std().item())
                out[f"bcnorm/L{layer_idx:02d}/{bias_name}/max_abs"] = float(pf.abs().max().item())
                # Drift from init (which is 1.0 for B/C bias).
                out[f"bcnorm/L{layer_idx:02d}/{bias_name}/drift_from_1"] = float((pf - 1.0).abs().mean().item())
                if p.grad is not None:
                    gf = p.grad.detach().float()
                    out[f"bcnorm/L{layer_idx:02d}/{bias_name}/grad_norm"] = float(gf.norm().item())

        # --- depth-level aggregates (across layers, for quick wandb panels) ----
        out.update(self._summarize_across_depth(out))

        # --- histogram payload ----
        hist_payload: dict[str, "wandb.Histogram"] = {}
        if log_histograms and self._hist_pending and wb is not None:
            try:
                import wandb as _wb  # noqa
            except ImportError:
                _wb = None
            if _wb is not None:
                # Activations: pre/post per (layer, tag)
                for (layer, tag), bufs in self._hist_buffer.items():
                    for which, t in bufs.items():
                        try:
                            hist_payload[f"bcnorm_hist/L{layer:02d}/{tag}/{which}"] = _wb.Histogram(
                                t.numpy(), num_bins=64
                            )
                        except Exception:
                            pass
                # Norm weights
                for (layer, tag), mod in self._modules.items():
                    w = mod.weight.detach().float().cpu().numpy()
                    hist_payload[f"bcnorm_hist/L{layer:02d}/{tag}/weight"] = _wb.Histogram(w, num_bins=64)
                    if mod.weight.grad is not None:
                        g = mod.weight.grad.detach().float().cpu().numpy()
                        hist_payload[f"bcnorm_hist/L{layer:02d}/{tag}/weight_grad"] = _wb.Histogram(g, num_bins=64)
                # Biases
                for layer_idx, mixer in self._mixers:
                    for bias_name in ("B_bias", "C_bias", "dt_bias", "D"):
                        p = getattr(mixer, bias_name, None)
                        if p is None:
                            continue
                        pf = p.detach().float().cpu().numpy().ravel()
                        hist_payload[f"bcnorm_hist/L{layer_idx:02d}/{bias_name}"] = _wb.Histogram(pf, num_bins=64)
                        if p.grad is not None:
                            gf = p.grad.detach().float().cpu().numpy().ravel()
                            hist_payload[f"bcnorm_hist/L{layer_idx:02d}/{bias_name}_grad"] = _wb.Histogram(gf, num_bins=64)

        # --- depth-profile line_series plots ----
        depth_plots: dict[str, object] = {}
        if log_depth_plots and wb is not None:
            depth_plots = self._build_depth_plots(out, step)

        if wb is not None and (out or hist_payload or depth_plots):
            merged = {**out, **hist_payload, **depth_plots}
            wb.log(merged, step=step)

        # --- handle pending raw-tensor dump ----
        if self._dump_pending and self.dump_dir is not None and self._dump_buffer:
            self.dump_dir.mkdir(parents=True, exist_ok=True)
            path = self.dump_dir / f"bcnorm_dump_step{step:06d}.pt"
            torch.save(
                {
                    "step": step,
                    "records": self._dump_buffer,
                    "norm_weights": {
                        f"L{l:02d}/{t}": m.weight.detach().cpu().clone()
                        for (l, t), m in self._modules.items()
                    },
                    "biases": {
                        f"L{l:02d}/{name}": getattr(m, name).detach().cpu().clone()
                        for l, m in self._mixers
                        for name in ("B_bias", "C_bias")
                        if hasattr(m, name)
                    },
                },
                path,
            )
            print(f"[probe] dumped raw BCNorm tensors to {path}")
            self._dump_buffer.clear()
            self._dump_pending = False

        self._buffer.clear()
        if log_histograms:
            self._hist_buffer.clear()
            self._hist_pending = False
        return out

    # ------------------------------------------------ Depth-profile plots
    def _build_depth_plots(self, flat: dict[str, float], step: int) -> dict[str, object]:
        """One wandb.plot.line_series per key metric: x=layer index, y=metric value,
        two series (B and C). Lets you see at a glance how the metric fans out
        with depth at this particular step."""
        try:
            import wandb as _wb  # noqa
        except ImportError:
            return {}

        # (metric_key_suffix, panel_title)
        wanted = [
            ("pre/std", "Pre-BCNorm std vs depth"),
            ("pre/max_abs", "Pre-BCNorm max-abs vs depth"),
            ("pre/kurtosis", "Pre-BCNorm kurtosis vs depth"),
            ("pre/abs_p0999", "Pre-BCNorm p99.9 vs depth"),
            ("pre/frac_above_6sigma", "Pre-BCNorm fraction > 6σ vs depth"),
            ("post/std", "Post-BCNorm std vs depth"),
            ("delta/norm_ratio_mean", "BCNorm rescaling factor vs depth"),
            ("weight_mean", "BCNorm scale weight mean vs depth"),
            ("weight_grad_norm", "BCNorm scale weight grad-norm vs depth"),
        ]

        # Collect per-layer values, sorted by layer index, for B and C.
        layers_sorted = sorted({int(k.split("/")[1][1:]) for k in flat if k.startswith("bcnorm/L")})
        if not layers_sorted:
            return {}
        xs = layers_sorted
        plots: dict[str, object] = {}
        for suffix, title in wanted:
            ys_B = [flat.get(f"bcnorm/L{layer:02d}/B/{suffix}") for layer in layers_sorted]
            ys_C = [flat.get(f"bcnorm/L{layer:02d}/C/{suffix}") for layer in layers_sorted]
            if any(v is None for v in ys_B) or any(v is None for v in ys_C):
                continue
            try:
                plot = _wb.plot.line_series(
                    xs=xs,
                    ys=[ys_B, ys_C],
                    keys=["B", "C"],
                    title=f"{title} (step {step})",
                    xname="layer",
                )
                plots[f"depth/{suffix.replace('/', '_')}"] = plot
            except Exception:
                pass
        return plots

    # -------------------------------------------------------- Depth summary
    def _summarize_across_depth(self, flat: dict[str, float]) -> dict[str, float]:
        """Reduce per-layer stats to depth-level mean/max for top-line dashboards."""
        groups: dict[str, list[float]] = defaultdict(list)
        for k, v in flat.items():
            # bcnorm/Lxx/{tag}/{stat}
            parts = k.split("/")
            if len(parts) < 4:
                continue
            tag = parts[2]
            stat = "/".join(parts[3:])
            groups[f"bcnorm/all/{tag}/{stat}"].append(v)

        agg: dict[str, float] = {}
        for k, vs in groups.items():
            agg[f"{k}_mean"] = float(sum(vs) / len(vs))
            agg[f"{k}_max"] = float(max(vs))
            agg[f"{k}_min"] = float(min(vs))
        return agg


# ============================================================================
# Pretty console table — same style as src/nfmamba/diagnostics/probe_bc.py
# ============================================================================

def format_probe_table(flat: dict[str, float], layers: Iterable[int] | None = None) -> str:
    """Render a one-shot summary table from a flush() return dict."""
    by_layer: dict[int, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
    for k, v in flat.items():
        parts = k.split("/")
        if len(parts) < 4 or parts[0] != "bcnorm" or not parts[1].startswith("L"):
            continue
        try:
            layer = int(parts[1][1:])
        except ValueError:
            continue
        tag = parts[2]
        stat = "/".join(parts[3:])
        by_layer[layer][tag][stat] = v

    cols = [
        ("pre/std", "pre_std", 9),
        ("pre/max_abs", "pre_max", 11),
        ("pre/kurtosis", "pre_kurt", 10),
        ("pre/abs_p0999", "p99.9", 9),
        ("pre/frac_above_6sigma", ">6σ", 9),
        ("post/std", "post_std", 9),
        ("post/rms", "rms", 7),
        ("delta/norm_ratio_mean", "ratio", 7),
        ("weight_mean", "w̄", 7),
        ("weight_grad_norm", "‖∇w‖", 9),
    ]

    header = f"{'L':>3} {'T':>1} " + " ".join(f"{label:>{w}}" for _, label, w in cols)
    sep = "-" * len(header)
    lines = [sep, header, sep]
    layer_iter = sorted(by_layer.keys()) if layers is None else list(layers)
    for layer in layer_iter:
        for tag in ("B", "C"):
            d = by_layer[layer].get(tag, {})
            row = f"{layer:>3} {tag:>1} "
            for stat_key, _, w in cols:
                v = d.get(stat_key)
                row += " " + (f"{v:>{w}.3g}" if v is not None and math.isfinite(v) else f"{'—':>{w}}")
            lines.append(row)
    lines.append(sep)
    return "\n".join(lines)
