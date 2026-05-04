# Mamba-3 Minimal Architecture

This folder does not contain a module or class literally named `BCNorm`.
In this implementation, the BC normalization point is represented as QK-style
RMS normalization on the projected `B` and `C` tensors:

- `Mamba3.__init__`: `self.B_norm = RMSNorm(self.bc_dim)` and
  `self.C_norm = RMSNorm(self.bc_dim)` in `mamba3.py` lines 288-291.
- Full-sequence forward path: `B = self.B_norm(B)` and `C = self.C_norm(C)`
  in `mamba3.py` lines 354-356.
- Single-token decode path: `B = self.B_norm(B)` and `C = self.C_norm(C)`
  in `mamba3.py` lines 532-534.
- The learnable BC bias is separate from the normalization and is added after
  normalization, at lines 293-308 for parameters and lines 388-390 / 445-447
  in the full-sequence path.

## Whole Model

```mermaid
flowchart TD
    ids["input_ids<br/>(batch, seqlen)"]
    emb["Token embedding<br/>vocab -> d_model"]
    stack["Repeat n_layer times"]
    final_norm["Final RMSNorm"]
    lm["Tied LM head<br/>d_model -> vocab"]
    logits["logits"]

    ids --> emb --> stack --> final_norm --> lm --> logits

    subgraph layer["One Mamba-3 LM Layer"]
        x0["x"]
        mix_norm["mixer_norm<br/>RMSNorm(d_model)"]
        mixer["Mamba3 SSM mixer"]
        add1["Residual add"]
        mlp_norm["mlp_norm<br/>RMSNorm(d_model)"]
        mlp["SwiGLU MLP"]
        add2["Residual add"]

        x0 --> mix_norm --> mixer --> add1
        x0 --> add1
        add1 --> mlp_norm --> mlp --> add2
        add1 --> add2
    end
```

## Mamba3 SSM Mixer

```mermaid
flowchart TD
    u["pre-normalized u<br/>(b, l, d_model)"]
    inproj["in_proj"]
    split["split into<br/>z, x, B, C, dt, lambda, theta"]

    u --> inproj --> split

    split --> z["z gate<br/>(d_inner)"]
    split --> x["x value<br/>(d_inner)"]
    split --> Braw["B projection<br/>(bc_dim)"]
    split --> Craw["C projection<br/>(bc_dim)"]
    split --> dt["dt + dt_bias<br/>softplus"]
    split --> lam["lambda<br/>sigmoid"]
    split --> theta["theta<br/>RoPE angles"]

    Braw --> Bnorm["B_norm<br/>RMSNorm(bc_dim)"]
    Craw --> Cnorm["C_norm<br/>RMSNorm(bc_dim)"]
    Bnorm --> Bbias["add B_bias"]
    Cnorm --> Cbias["add C_bias"]

    dt --> angles["raw_angles = dt * theta<br/>cum_angles = -cumsum(raw_angles)"]
    theta --> angles
    Bbias --> Brope["apply_rope(B, cum_angles)"]
    Cbias --> Crope["apply_rope(C, cum_angles)"]
    angles --> Brope
    angles --> Crope

    dt --> coeffs["trapezoidal coefficients<br/>alpha = exp(dt * A)<br/>beta = (1-lambda) * dt * alpha<br/>gamma = lambda * dt"]
    lam --> coeffs
    x --> reshape["reshape x to heads<br/>(b, l, nheads, headdim)"]

    reshape --> gamma_path["gamma path<br/>ssd(x * gamma, A, B, C)"]
    coeffs --> gamma_path
    Brope --> gamma_path
    Crope --> gamma_path

    reshape --> shift["shift previous x and B"]
    Brope --> shift
    shift --> beta_path["beta path<br/>ssd(x_prev * beta, A, B_prev, C)"]
    coeffs --> beta_path
    Crope --> beta_path

    gamma_path --> sum["sum gamma + beta"]
    beta_path --> sum
    sum --> skip["add D * x skip"]
    reshape --> skip
    z --> gate["multiply by SiLU(z)"]
    skip --> gate
    gate --> outproj["out_proj"]
    outproj --> y["mixer output<br/>(b, l, d_model)"]
```

## Optional MIMO Branch

When `use_mimo=True`, the same BC normalization point is used, but `bc_dim`
becomes `d_state * mimo_rank`. After `B_norm`/`C_norm`, `B` and `C` are reshaped
to `(d_state, R)`, biased per head/rank, RoPE-rotated per rank, and passed to
`ssd_mimo`. The rank dimension is gated and then down-projected before
`out_proj`.

```mermaid
flowchart LR
    BCRaw["B, C<br/>(bc_dim = d_state * R)"]
    Norm["B_norm / C_norm"]
    Reshape["reshape<br/>(d_state, R)"]
    Bias["add per-head<br/>per-rank BC bias"]
    Rope["apply RoPE<br/>per rank"]
    SSD["ssd_mimo<br/>gamma + beta paths"]
    Gate["rank-space SiLU gate"]
    Down["mimo_down<br/>sum over R"]
    Out["out_proj"]

    BCRaw --> Norm --> Reshape --> Bias --> Rope --> SSD --> Gate --> Down --> Out
```

## Decode Step Recurrence

For autoregressive decoding, `Mamba3.step` applies the same normalization and
bias order, then updates the cached SSM state with the trapezoidal recurrence:

```text
h_t = alpha_t * h_{t-1}
    + beta_t  * B_{t-1} x_{t-1}
    + gamma_t * B_t x_t

y_t = C_t^T h_t
```

The cache stores `ssm_state`, `prev_Bx`, and `cum_angle`, so each decode step is
constant-time with respect to already-processed sequence length.
