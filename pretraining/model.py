"""Mamba-3 LM assembly that wires the official mamba-og Mamba3 mixer into a
prenorm Llama-style stack. The reason we don't reuse `mixer_seq_simple.py`
verbatim: its `create_block` only knows about Mamba1/Mamba2. We replicate the
same Block layout and weight init so the model is structurally identical to
what the Mamba-3 authors trained, just with Mamba3 as the mixer."""

from __future__ import annotations

import math
from collections import namedtuple
from functools import partial
from pathlib import Path
import sys

import torch
import torch.nn as nn

# Make mamba-og importable without installing it.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_MAMBA_OG = _REPO_ROOT / "mamba-og"
if str(_MAMBA_OG) not in sys.path:
    sys.path.insert(0, str(_MAMBA_OG))

from mamba_ssm.modules.block import Block
from mamba_ssm.modules.mamba3 import Mamba3
from mamba_ssm.modules.mlp import GatedMLP
from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn

CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "loss"])


def _init_weights(
    module: nn.Module,
    n_layer: int,
    initializer_range: float = 0.02,
    rescale_prenorm_residual: bool = True,
    n_residuals_per_layer: int = 1,
):
    if isinstance(module, nn.Linear):
        if module.bias is not None and not getattr(module.bias, "_no_reinit", False):
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.normal_(module.weight, std=initializer_range)

    if rescale_prenorm_residual:
        for name, p in module.named_parameters():
            if name in ("out_proj.weight", "fc2.weight"):
                nn.init.kaiming_uniform_(p, a=math.sqrt(5))
                with torch.no_grad():
                    p /= math.sqrt(n_residuals_per_layer * n_layer)


def _build_block(
    layer_idx: int,
    d_model: int,
    d_intermediate: int,
    ssm_cfg: dict,
    norm_epsilon: float,
    rms_norm: bool,
    residual_in_fp32: bool,
    fused_add_norm: bool,
    device,
    dtype,
):
    factory_kwargs = {"device": device, "dtype": dtype}

    mixer_cls = partial(Mamba3, layer_idx=layer_idx, **ssm_cfg, **factory_kwargs)

    norm_cls = partial(
        RMSNorm if rms_norm else nn.LayerNorm, eps=norm_epsilon, **factory_kwargs
    )

    if d_intermediate == 0:
        mlp_cls = nn.Identity
    else:
        mlp_cls = partial(
            GatedMLP,
            hidden_features=d_intermediate,
            out_features=d_model,
            **factory_kwargs,
        )

    block = Block(
        d_model,
        mixer_cls,
        mlp_cls,
        norm_cls=norm_cls,
        fused_add_norm=fused_add_norm,
        residual_in_fp32=residual_in_fp32,
    )
    block.layer_idx = layer_idx
    return block


class Mamba3LM(nn.Module):
    """Mamba-3 language model (prenorm stack + tied lm_head)."""

    def __init__(
        self,
        d_model: int,
        n_layer: int,
        d_intermediate: int,
        vocab_size: int,
        ssm_cfg: dict,
        rms_norm: bool = True,
        residual_in_fp32: bool = True,
        fused_add_norm: bool = True,
        norm_epsilon: float = 1e-5,
        pad_vocab_multiple: int = 8,
        tie_embeddings: bool = True,
        initializer_range: float = 0.02,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if vocab_size % pad_vocab_multiple != 0:
            vocab_size += pad_vocab_multiple - (vocab_size % pad_vocab_multiple)
        self.vocab_size = vocab_size
        self.tie_embeddings = tie_embeddings
        self.residual_in_fp32 = residual_in_fp32
        self.fused_add_norm = fused_add_norm

        factory_kwargs = {"device": device, "dtype": dtype}
        self.embedding = nn.Embedding(vocab_size, d_model, **factory_kwargs)
        self.layers = nn.ModuleList(
            [
                _build_block(
                    layer_idx=i,
                    d_model=d_model,
                    d_intermediate=d_intermediate,
                    ssm_cfg=ssm_cfg,
                    norm_epsilon=norm_epsilon,
                    rms_norm=rms_norm,
                    residual_in_fp32=residual_in_fp32,
                    fused_add_norm=fused_add_norm,
                    device=device,
                    dtype=dtype,
                )
                for i in range(n_layer)
            ]
        )
        self.norm_f = (RMSNorm if rms_norm else nn.LayerNorm)(
            d_model, eps=norm_epsilon, **factory_kwargs
        )
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False, **factory_kwargs)

        self.apply(
            partial(
                _init_weights,
                n_layer=n_layer,
                initializer_range=initializer_range,
                n_residuals_per_layer=1 if d_intermediate == 0 else 2,
            )
        )
        if tie_embeddings:
            self.lm_head.weight = self.embedding.weight

    def forward(self, input_ids, labels=None):
        hidden_states = self.embedding(input_ids)
        residual = None
        for layer in self.layers:
            hidden_states, residual = layer(hidden_states, residual)

        if not self.fused_add_norm:
            residual = (hidden_states + residual) if residual is not None else hidden_states
            hidden_states = self.norm_f(residual.to(self.norm_f.weight.dtype))
        else:
            hidden_states = layer_norm_fn(
                hidden_states,
                self.norm_f.weight,
                self.norm_f.bias,
                eps=self.norm_f.eps,
                residual=residual,
                prenorm=False,
                residual_in_fp32=self.residual_in_fp32,
                is_rms_norm=isinstance(self.norm_f, RMSNorm),
            )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)).float(),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return CausalLMOutput(logits=logits, loss=loss)

    def num_params(self, trainable_only: bool = True) -> int:
        params = (p for p in self.parameters() if (p.requires_grad or not trainable_only))
        # When embeddings are tied, the underlying tensor is shared — count once.
        seen = set()
        total = 0
        for p in params:
            if id(p) in seen:
                continue
            seen.add(id(p))
            total += p.numel()
        return total


def build_model_from_config(cfg: dict, device=None, dtype=None) -> Mamba3LM:
    """Translate a parsed YAML config into a Mamba3LM.

    If ``architecture.bc_stabilizer`` is present and not ``"bcnorm"``, the
    stock RMSNormGated B/C normalizers are swapped out for the named
    element-wise stabilizer (DySoftSign, DyT, DyISRU, Derf, ...). The swap
    happens *after* model construction so the rest of the mixer (in_proj,
    biases, RoPE, SSD kernel) is bit-identical to the BCNorm baseline.
    """
    m = cfg["model"]
    a = cfg["architecture"]
    k = cfg["kernels"]

    ssm_cfg = dict(
        d_state=m["d_state"],
        expand=m["expand"],
        headdim=m["head_dim"],
        ngroups=m["ngroups"],
        rope_fraction=a["rope_fraction"],
        is_outproj_norm=a["is_outproj_norm"],
        is_mimo=a["is_mimo"],
        mimo_rank=a["mimo_rank"],
        chunk_size=k["chunk_size"],
    )

    model = Mamba3LM(
        d_model=m["d_model"],
        n_layer=m["n_layers"],
        d_intermediate=m["d_intermediate"],
        vocab_size=m["vocab_size"],
        ssm_cfg=ssm_cfg,
        rms_norm=m["rms_norm"],
        residual_in_fp32=m["residual_in_fp32"],
        fused_add_norm=m["fused_add_norm"],
        norm_epsilon=m["norm_epsilon"],
        pad_vocab_multiple=m["pad_vocab_multiple"],
        tie_embeddings=m["tie_embeddings"],
        initializer_range=m["initializer_range"],
        device=device,
        dtype=dtype,
    )

    stabilizer = str(a.get("bc_stabilizer", "bcnorm")).lower()
    if stabilizer != "bcnorm":
        if str(_REPO_ROOT / "src") not in sys.path:
            sys.path.insert(0, str(_REPO_ROOT / "src"))
        from nfmamba.adapters.bc_stabilizer import install_bc_stabilizer

        report = install_bc_stabilizer(
            model,
            name=stabilizer,
            stabilize_b=bool(a.get("stabilize_b", True)),
            stabilize_c=bool(a.get("stabilize_c", True)),
            squash_before_bias=bool(a.get("squash_before_bias", False)),
        )
        print(
            f"[model] BC stabilizer = {report.name!r} "
            f"(replaced={report.replaced}, B={report.stabilize_b}, C={report.stabilize_c}, "
            f"squash_before_bias={report.squash_before_bias})",
            flush=True,
        )

        if dtype is not None:
            model.to(dtype=dtype)
        if device is not None:
            model.to(device=device)

    return model
