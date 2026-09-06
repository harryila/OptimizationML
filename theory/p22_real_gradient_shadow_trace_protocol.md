# P22 real-gradient shadow-trace protocol

## Status and claim boundary

P22 freezes the first real-gradient acquisition before any gradient, shield,
loss, or fidelity result is inspected. The acquisition is not present in this
commit. The companion harness fails closed when its trainer, FineWeb bytes,
instrumentation, accelerator, or full matrix-shape coverage is missing; it
does not generate a synthetic substitute.

A future passing trace is empirical evidence about one pinned 256-step run.
It cannot establish global passivity, global Muon fidelity, global PL for a
neural loss, or accelerator-independent behavior. In particular, an Apple
MPS run is MPS-specific evidence, not CUDA, tensor-core, native-kernel, or
distributed parity.

The baseline update remains unchanged. P20 is evaluated on detached copies
and is never returned to the optimizer. Three complete fresh-process runs,
trace-off-A, trace-off-B, and trace-on, are required. Off-A must first match
off-B, and off-A must then match trace-on, bitwise in every predeclared
training-state field before any shadow metric is interpreted.

## Frozen source and arithmetic identity

The model source is `karpathy/nanoGPT` at commit
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, tree
`ca93bcd9b9c9ff32d3016e1e2556644e68bef86a`. The frozen `train.py` and
`model.py` Git blobs are respectively
`de5785051050d64bbda37a4964c2c079f0739b2b` and
`c698f8b60129d793494de058b0dd4e318c0dcb5e`. The isolated P22 harness imports
the frozen `model.py`; it does not execute or silently modify NanoGPT's
`train.py`. The checkout must be clean and the record-only optimizer adapter
is carried as a separate, hashed source file.

The optimizer source is KellerJordan/Muon commit
`f98f1cacc0263b04290753e32be8d498c1efc806`. Its audited `muon.py` SHA-256
is `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
The `SingleDeviceMuonWithAuxAdam` path is used. For every hidden matrix the
Muon settings are

- momentum `beta=19/20`, with Nesterov enabled;
- constant `eta=1/120`;
- weight decay zero;
- additive normalization `X/(||X||_F+1e-7)` after the one-time orientation;
- five BF16 Jordan stages with exact decimal coefficients represented here
  by `6889/2000`, `-191/40`, and `4063/2000`; and
- aspect multiplier `sqrt(max(1, rows/columns))` after orientation is
  restored.

The auxiliary group contains token and position embeddings, the tied language
model head, layer-normalization gains, and every vector or scalar. It uses a
constant learning rate `3/5000`, betas `(9/10,19/20)`, epsilon `1e-10`, and
weight decay zero. This auxiliary choice is part of the empirical baseline,
not a consequence of C27.

## Model, batch, and schedule

The model is GPT-2 small initialized from scratch: 12 layers, 12 heads,
embedding dimension 768, architectural block size 1024, vocabulary size
50257, zero dropout, no biases, and tied input/output embeddings. The
M4-feasible acquisition uses training sequence length 128, micro-batch size
one, no gradient accumulation, and no gradient clipping. It therefore
consumes one fixed sequential 128-token window per optimizer step and 32,768
training tokens over 256 optimizer steps. It performs no evaluation or
checkpoint write during the three runs.

There is one seed, `1337`, and one explicitly selected BF16-capable
accelerator. Model parameters, forward, and backward remain FP32; only the
pinned Newton--Schulz state is BF16. Autocast, compilation, and TF32 are
disabled. Deterministic algorithms are enabled. CUDA additionally requires
`CUBLAS_WORKSPACE_CONFIG=:4096:8`; MPS requires
`PYTORCH_ENABLE_MPS_FALLBACK=0` and requires `PYTORCH_MPS_FAST_MATH` to be
unset or `0`.

The 24 capture steps are inherited unchanged from P21:

- early: `0,...,7`;
- middle: `124,...,131`;
- late: `248,...,255`.

The step index is the optimizer-step index, never wall-clock progress.

## FineWeb materialization

The source is `HuggingFaceFW/fineweb`, configuration `sample-10BT`, split
`train`, at revision `9bb295ddab0e05d785b879661af7260fed5140fc`.
Documents are consumed in increasing row index. Text must be complete: a
datasets-server response is admissible only when its `truncated_cells` list
is empty; otherwise the revision-pinned source shard must supply the cell.
Every response or consumed shard is hashed.

Tokenization uses `tiktoken==0.14.0`, encoding `gpt2`, vocabulary size 50257,
and end-of-text token 50256 appended once after every document. The upstream
assets are frozen by SHA-256:

- `vocab.bpe`:
  `1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5`;
- `encoder.json`:
  `196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783`.

The complete token stream is split without shuffling. The first 131,072
tokens form `train.bin`; the next 32,768 form `val.bin`. Both are
little-endian `uint16`. The boundary row and in-row token offset, consumed-row
digest, every source byte digest, preprocessor digest, and binary digest are
mandatory. Step `t` consumes deterministic sequential input offsets
`[128t,128(t+1))` and their one-token-shifted targets. Every batch is hashed
in the run manifest. Preparing 131,072 training tokens does not claim that all
are used: the 256 updates consume the first 32,768 input tokens plus the one
final shifted target.

The template
`experiments/training/p22_fineweb_manifest.template.json` deliberately
contains null materialization fields and is rejected by preflight.

## Exact parameter inventory and shape extension

All 48 transformer-block matrices are selected before observation: four
roles in each of 12 layers. Stored PyTorch shapes per layer are

- attention fused QKV: `2304 x 768`;
- attention projection: `768 x 768`;
- MLP expansion: `3072 x 768`;
- MLP projection: `768 x 3072`.

P20 permits transposition into a certified orientation, but its frozen table
does not contain `2304 x 768` or `768 x 2304`. P22 therefore adds the single
canonical orientation `768 x 2304` through the unchanged P20 arithmetic
recurrence without rewriting the frozen P20 artifact. Stored QKV matrices are
transposed into that orientation and back. The exact P22 extension and its
independent reconstruction must pass before acquisition. The real trace must
report 100% coverage by both name count and number of elements; it may not
silently omit any of the 12 fused-QKV matrices.

## Actual accelerator capture

At each capture step, instrumentation runs inside the pinned `muon_update`
path after gradient accumulation and before parameter subtraction. It clones
and hashes:

1. the stored Nesterov signal actually consumed by the accelerator
   orthogonalizer;
2. when the pinned call boundary exposes it, the actual accelerator BF16
   result after five Jordan stages and orientation restoration but before
   aspect scaling; and
3. the mandatory actual stored accelerator candidate after the aspect
   multiplication.

The third tensor, not a CPU recomputation, is the fidelity comparator. The
pre-aspect tensor is an optional diagnostic: an unavailable value is recorded
as null with a reason and is never reconstructed on CPU. A
detached contiguous CPU copy of it and the signal is supplied to the P20
proof-reference shadow. The trace records P20 action and the P21 metrics and
gates, but the output is not exposed to the optimizer. The post-aspect hash is
mandatory; any available pre-aspect hash is retained so the aspect boundary
can be inspected independently.

This capture is backend evidence only. P20's stored-output theorem continues
to concern its locked CPU arithmetic graph; it does not become a theorem
about MPS or CUDA merely because the candidate originated there.

## Noninterference gate

Two trace-off runs (`off-A` and `off-B`) and one trace-on run start in fresh
processes from byte-identical model, optimizer, deterministic data, and RNG
states. Off-A must first match off-B, separating baseline repeatability from
observer noninterference; only then is off-A compared with trace-on. The
observer is a function of detached copies only and has no return path into the
update. The harness enforces that structural separation in two ways.

First, every observer invocation snapshots Python, NumPy, Torch CPU, and the
selected CUDA or MPS RNG state. If any byte changes, the harness restores the
snapshot and aborts. Second, the three complete executions are compared with
no numeric tolerance. At every one of 256 steps the following digests must
match exactly:

- sampled input-token batch;
- stored loss tensor;
- RNG state before and after the step.

All model parameter/buffer bytes and the complete optimizer state are checked
initially, after each of the 24 capture-schedule updates (including step 255),
and finally. The final RNG digest must also match. The trace-on manifest must
contain exactly the 24 frozen captures and assert that every observer
invocation preserved RNG. Trace-off must contain none. A single different
byte fails the acquisition; an `allclose` result is insufficient.

This three-run replay establishes repeatability and noninterference only for
the pinned executions.
It is not a universal proof about arbitrary accelerators, drivers, compilers,
or future instrumentation.

## Scaffold and blocked state

The standard entry point is:

```bash
uv run --locked python experiments/training/p22_nanogpt_shadow_trace.py \
  preflight --accelerator mps \
  --nanogpt-root /path/to/nanoGPT \
  --muon-source /path/to/muon.py \
  --fineweb-manifest /path/to/materialized-fineweb.json \
  --instrumentation-patch experiments/training/p22_observed_muon.py \
  --output /path/to/preflight.json
```

Use `--accelerator cuda` only for a real CUDA run satisfying the CUDA-specific
guard. A blocked report exits nonzero. The command validates but never starts
training.

The three acquisition commands use the isolated runner in fresh processes.
The off-A command is:

```bash
uv run --locked python \
  experiments/training/run_p22_real_gradient_shadow_trace.py run \
  --trace-mode trace_off --accelerator mps \
  --nanogpt-root /path/to/nanoGPT \
  --muon-source /path/to/muon.py \
  --fineweb-manifest /path/to/materialized-fineweb.json \
  --instrumentation-patch experiments/training/p22_observed_muon.py \
  --output /path/to/trace-off-a-manifest.json
```

Repeat it for off-B, changing only the output path. Trace-on uses the same
arguments plus a raw-trace destination:

```bash
uv run --locked python \
  experiments/training/run_p22_real_gradient_shadow_trace.py run \
  --trace-mode trace_on --accelerator mps \
  --nanogpt-root /path/to/nanoGPT \
  --muon-source /path/to/muon.py \
  --fineweb-manifest /path/to/materialized-fineweb.json \
  --instrumentation-patch experiments/training/p22_observed_muon.py \
  --output /path/to/trace-on-manifest.json \
  --raw-trace-output /path/to/p22-raw-trace.json
```

The backend and every other frozen input must agree across all three runs.
Use `--accelerator cuda` throughout only when the CUDA guards pass.

After the three real runs, establish baseline repeatability first:

```bash
uv run --locked python experiments/training/p22_nanogpt_shadow_trace.py \
  verify-repeatability \
  --trace-off-a /path/to/trace-off-a-manifest.json \
  --trace-off-b /path/to/trace-off-b-manifest.json \
  --output /path/to/repeatability.json
```

Only if that passes, compare off-A with trace-on:

```bash
uv run --locked python experiments/training/p22_nanogpt_shadow_trace.py \
  verify-noninterference \
  --trace-off /path/to/trace-off-a-manifest.json \
  --trace-on /path/to/trace-on-manifest.json \
  --output /path/to/noninterference.json
```

The isolated runner and materializer are scaffolding around the pristine
NanoGPT checkout, not edits to upstream `train.py`. No materialized FineWeb
bundle or real three-run acquisition is committed. Until every prerequisite
and comparison closes, P22 has no observed activation, fidelity, loss, or
training result.
The machine-readable authority for this protocol is
`experiments/training/p22_real_gradient_shadow_trace_protocol.json`; every run
records the SHA-256 of those exact bytes.
