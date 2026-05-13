# mamba-norm-free

> Do Mamba-3's B/C projections need cross-channel normalization, or is bounded
> element-wise magnitude control enough?

This is a research repo for BCNorm replacement experiments in Mamba-3. The
central rule is simple:

**`mamba3-minimal/` is the untouched reference implementation. All experimental
code lives in `src/nfmamba/`.**

The goal is to compare the original B/C RMSNorm path against cheaper
element-wise stabilizers such as DyISRU and DyT, then decide whether the useful
part of BCNorm is magnitude control or cross-channel geometry.

## Layout

```text
mamba-norm-free/
├── mamba3-minimal/              # read-only reference Mamba-3 implementation
├── src/
│   └── nfmamba/
│       ├── adapters/            # apply experiment choices to reference models
│       ├── modules/             # PyTorch reference stabilizers
│       ├── diagnostics/         # B/C probes and control experiments
│       ├── data/                # deterministic tokenization and packing
│       └── utils/               # manifests, git/env capture, determinism
├── scripts/
│   └── data/                    # thin runnable dataset entrypoints
├── configs/                     # dataset and experiment configs
├── tests/                       # repo-level correctness/smoke tests
├── docs/
│   ├── theory/                  # BCNorm notes and research framing
│   ├── plans/                   # 180M/350M execution plans
│   └── status/                  # checkpoint/status pages
└── experiments/                 # curated pilot logs and generated run outputs
```

## Stabilizers

PyTorch reference stabilizers live in `src/nfmamba/modules/`:

| Name | Module | Formula |
| --- | --- | --- |
| `rmsnorm` / `bcnorm` | `ExternalRMSNorm` | `x / rms(x) * weight` |
| `identity` | `IdentityStabilizer` | `x` |
| `dyisru` | `DyISRU` | `x / sqrt(1 + alpha * x^2)` |
| `dyt` | `DyT` | `alpha * tanh(x)` |

Adapters live in `src/nfmamba/adapters/`. They patch a constructed reference
model instance without changing `mamba3-minimal`:

```python
from nfmamba import install_bc_stabilizer

model = create_toy_model(...)
install_bc_stabilizer(model, "dyisru")
```

## Diagnostics

Diagnostics are hook-based and non-intrusive:

```bash
uv run python -m nfmamba.diagnostics.probe_bc
uv run python -m nfmamba.diagnostics.bc_cosine
uv run python -m nfmamba.diagnostics.compare_no_bcnorm
```

Current Week 1 finding: at random initialization, BCNorm primarily acts as a
scale-setting/unit-RMS layer for B and C. Identity/no-BCNorm changes logits
immediately, so it is a real architecture variant rather than a harmless null
control.

## Data

The smoke dataset pipeline is deterministic and manifest-driven:

```bash
uv run python scripts/data/build_dataset.py --config configs/data/wikitext_smoke.yaml
uv run python scripts/data/verify_dataset.py --manifest data/manifests/wikitext_smoke.json
```

The repo-root `data/` directory is generated and gitignored. Source code under
`src/nfmamba/data/` is tracked.

## Tests

Run the external stabilizer smoke test:

```bash
uv run python tests/test_external_stabilizers.py
```

This verifies:

- all supported stabilizer names install into a stock reference model;
- external `rmsnorm` matches the reference BCNorm path at initialization;
- `identity`, `dyisru`, and `dyt` run forward/backward without NaN/Inf.

## Docs

Start here:

- `docs/status/week1_checkpoint.md`
- `docs/status/week1_status_page.md`
- `docs/theory/bcnorm_interpretation.md`
- `docs/plans/next_6_weeks_180m_plan.txt`

## Current Boundary

Built:

- external B/C stabilizer adapter;
- PyTorch `rmsnorm`, `identity`, `dyisru`, and `dyt`;
- B/C magnitude and cosine probes;
- BCNorm-vs-identity control experiment;
- Wikitext smoke data build/verify pipeline.

Not built yet:

- tiny LM trainer;
- DySN / Derf;
- 180M training harness;
- Triton/CUDA kernels;
- full benchmark/eval suite.
