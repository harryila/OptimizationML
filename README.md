# Passivizing Practical Muon

Personal research workspace for **Passivizing Practical Muon: Normalization
Gaps and a Minimal Circuit Repair**.

The project asks a deliberately narrow question: when a finite polynomial
orthogonalizer is applied after normalization by the *current* Frobenius norm,
does the resulting matrix operator remain monotone (incrementally passive), and what is the
smallest certified linear correction

\[
T_\rho(M) = T(M) + \rho M
\]

that repairs a measured deficit without another matrix multiplication?

## Status

The first proof gate is implemented:

- the diagonal reduction gives an exact criterion for normalization-induced
  indefiniteness;
- a rational Jordan-quintic witness is checked exactly at the Jacobian level;
- a robust finite pair is certified exactly and reevaluated at high precision.

The CPU experiment gate is also complete for Jordan, classical NS, Taylor NS5,
the pinned Polar Express five-stage prefix, and clean-room CANS-5x4. The
gain-matched quadratic result is qualified rather than positive: repairs help
several intrinsically nonmonotone polynomial prefixes, but show no
grid-resolved upper-endpoint benefit for the normalization-only classical and
Taylor cases. See [`results/summaries/RESULTS.md`](results/summaries/RESULTS.md).

The repair claim is intentionally scoped. A constant `rho` is the exact minimal
linear shift for a **specified point, pair, sample set, or domain with a finite
certified deficit**. For exact scale-invariant normalization on every nonzero
input, any negative local eigenvalue scales like `1 / ||M||_F`; consequently no
finite constant `rho` gives an unrestricted global repair. See
[`theory/claims.md`](theory/claims.md) before using the word "global."

## Reproduce the first milestone

Install Python 3.12 (the version in `.python-version`) and
[`uv`](https://docs.astral.sh/uv/), then run:

```bash
uv sync --locked
uv run --locked python scripts/find_counterexample.py
uv run --locked pytest
```

The complete CPU result pipeline is:

```bash
uv run --locked python experiments/matrices/run_deficit_audit.py
uv run --locked python experiments/quadratics/run_lr_sweep.py
uv run --locked python experiments/quadratics/run_horizon_check.py
uv run --locked python scripts/make_figures.py
```

The counterexample command does not rely on floating-point autodiff. It uses
exact rational arithmetic for the derivative mismatch, an exact rational-surd
comparison for the finite pair, and 100-digit decimal arithmetic for readable
values. Jacobian theorems concern the real-arithmetic operator; the upstream
BF16 implementation is a discontinuous deployment check and is tested
pairwise.

## Scope

This repository stays focused on four technical goals:

1. a theorem for current-input Frobenius normalization;
2. passivity-deficit measurements for Jordan, classical Newton--Schulz, Polar
   Express, and CANS;
3. the smallest certified `+ rho M` correction on an explicitly stated
   certification domain;
4. matrix/quadratic evidence and one small matched NanoGPT learning-rate sweep.

General circuit-designed optimizers, a broad "Muon is a circuit" narrative,
and a large Adam-beating benchmark are out of scope.

## Layout

- `src/passive_muon/`: normalization, orthogonalizers, deficit metrics, repairs;
- `theory/`: claim ledger, analytic proof, and exact/numerical certificates;
- `experiments/`: matrix, quadratic, and eventual NanoGPT studies;
- `scripts/`: reproducible entry points;
- `results/`: committed manifests, summaries, and figures only;
- `third_party/`: upstream revisions and license notices.

Numerical audits explicitly use `float64`; public orthogonalizers preserve the
caller's dtype. Low-precision training kernels remain separate so that
implementation convenience cannot silently change a theorem.
