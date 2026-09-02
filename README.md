# Exact Current Frobenius Normalization Can Break Monotonicity in Finite-Step Muon

Personal research workspace for exact certificates and controlled CPU
experiments on normalization-induced nonmonotonicity in finite-step Muon.

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
- all four fixed-scale `2 x 2` Jacobian modes at the witness are certified
  positive, without making a global fixed-scale monotonicity claim;
- a robust finite pair is certified exactly and reevaluated at high precision.

The CPU experiment gate is also complete for Jordan, classical NS, Taylor NS5,
the pinned Polar Express five-stage prefix, and clean-room CANS-5x4. The
gain-matched quadratic result is qualified rather than positive: repairs help
several intrinsically nonmonotone polynomial prefixes, but show no
grid-resolved upper-endpoint benefit for the normalization-only classical and
Taylor cases. See [`results/summaries/RESULTS.md`](results/summaries/RESULTS.md).

The floored-normalizer proof gate is now complete for the five-step Jordan
map. Outward-rounded Arb arithmetic and a full rectangular tangent reduction
give the dimension-uniform bracket

\[
159.549525785 < \delta(F_{h,1}) \le 159.564501081071.
\]

Thus `rho = 159.564501081071 / c` is a rigorous global repair for
`F_h,c(M)=H_h(M/max(c, ||M||_F))`. The upper endpoint is within `0.009386%` of
an exact finite-pair lower witness. It is a certified near-minimal sufficient
conductance, not the exact minimum for every fixed matrix shape. See
[`theory/floored_normalizer_certificate.md`](theory/floored_normalizer_certificate.md).

On branch `p3`, the repaired floored map is first connected to a **stylized
real-arithmetic, non-Nesterov momentum loop** for deterministic strongly convex
quadratics. A dimension-independent `3 x 3` IQC gives the explicit sufficient
region

\[
0<\eta KL<
\frac{2(1-\beta)^2(1+\beta)\nu}
{(1+\beta)^2-4\beta\nu^2},
\qquad \nu=\frac{\mu\ell}{KL}.
\]

The result has a closed-form strict LMI and a separate exact rational rate
certificate. It is rigorous but extremely conservative for the linked Jordan
map: at the locked representative, its local linear threshold is about
`18,183x` larger than the global sector-certified endpoint. See
[`theory/momentum_iqc_certificate.md`](theory/momentum_iqc_certificate.md).

A second `p3` certificate matches the pinned upstream **EMA state and Nesterov
signal ordering** in real arithmetic after replacing the orthogonalizer by the
repaired floored map and omitting weight decay:

\[
m_{t+1}=\beta m_t+(1-\beta)g_t,\qquad
s_{t+1}=\beta m_{t+1}+(1-\beta)g_t,\qquad
W_{t+1}=W_t-\eta R(s_{t+1}).
\]

At the upstream default `beta=0.95`, a 60-digit stationarity solve corroborated
by a broad logarithmic scan locates a repair-margin design near `mu=648.024`;
the exact replay locks `mu=648`,
`alpha=eta*K*L=1/400`,
`eta=65065340/336372400608849 = 1.93432457e-7`, and
`tau^2=99999/100000`. Exact rational Sylvester checks certify its `3 x 3` LMI.
The locked-design zero-linearization threshold is `0.002081066457...`, about
`10,758.62x` the locked exact certificate. Re-evaluating both quantities at
the numerical stationary design gives a `9,923.44x` gap.
By the predeclared decision rule this is an appendix-level proof of principle,
not a practical-stability headline. See
[`theory/ema_nesterov_iqc_certificate.md`](theory/ema_nesterov_iqc_certificate.md).

On branch `p4-structure-aware-stability`, the same pinned EMA/Nesterov ordering
is certified for every deterministic quadratic with
`I <= H <= 10 I` at the much larger step `eta=1/32000`. The proof uses the
specific centered decomposition of the repaired floored Jordan map, rather
than reducing it to a generic strongly-monotone/Lipschitz sector. It is the
quadratic bridge to the nonlinear result.

Branch `p5-nonquadratic-stability` proves two complementary theorems for every
fixed differentiable globally `1`-strongly-convex, `10`-smooth objective on
every finite real matrix shape. With the exact max floor `c=1`, no additive
epsilon, five Jordan steps with coefficients `6889/2000`, `-191/40`, and
`4063/2000`, constant repair
`rho=210177835339081/260261360000`, and `beta=19/20`:

- at `eta=1/640000`, a common quadratic storage proves arbitrary-pair global
  incremental contraction;
- at the full p4 step `eta=1/32000`, objective-gap/interpolation storage proves
  global exponential convergence of each trajectory to the unique minimizer
  at rate `tau=2499/2500`.

The full-step result is the primary positive theorem. It permits changing
local Hessian orientations, but it is trajectory-to-minimizer convergence,
not arbitrary-pair incremental stability. See
[`theory/nonquadratic_convergence_certificate.md`](theory/nonquadratic_convergence_certificate.md).

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
uv run --locked python scripts/record_bf16_witness.py
uv run --locked python scripts/certify_floored_repair.py
uv run --locked python scripts/certify_momentum_stability.py
uv run --locked python scripts/certify_ema_nesterov_stability.py
uv run --locked python scripts/certify_structure_aware_stability.py
uv run --locked python scripts/certify_nonquadratic_stability.py
uv run --locked python scripts/certify_nonquadratic_convergence.py
uv run --locked python scripts/reconstruct_nonquadratic_convergence.py
uv run --locked pytest
```

The complete CPU result pipeline is:

```bash
uv run --locked python experiments/matrices/run_deficit_audit.py
uv run --locked python experiments/quadratics/run_lr_sweep.py
uv run --locked python experiments/quadratics/run_horizon_check.py
uv run --locked python experiments/quadratics/run_nonquadratic_falsification.py
uv run --locked python experiments/quadratics/run_nonquadratic_convergence_falsification.py
uv run --locked python scripts/make_figures.py
```

The counterexample command does not rely on floating-point autodiff. It uses
exact rational arithmetic for the derivative mismatch, an exact rational-surd
comparison for the finite pair, and 100-digit decimal arithmetic for readable
values. Jacobian theorems concern the real-arithmetic operator; the upstream
BF16 implementation is a discontinuous deployment check and is tested
pairwise. Its committed manifest records the concrete backend, PyTorch build,
CPU architecture, cast/normalization behavior, operation order, coefficients,
epsilon, returned storage words, and observable accumulation behavior; it does
not support a universal claim across BF16 backends.

## Scope

This repository stays focused on eight technical goals:

1. a theorem for current-input Frobenius normalization;
2. exact local and finite-pair controls for the five-step Jordan map;
3. passivity-deficit measurements for Jordan, classical Newton--Schulz, Polar
   Express, and CANS;
4. a rigorous full-matrix repair certificate for a fixed Frobenius floor;
5. sector-IQC certificates for a stylized non-Nesterov loop and the pinned
   EMA/Nesterov state-and-signal ordering in deterministic quadratics;
6. a structure-aware full-step theorem for the pinned loop on fixed
   quadratics;
7. incremental and full-step convergence theorems for globally
   strongly-convex/smooth nonlinear objectives;
8. qualified matrix, quadratic, and nonlinear falsification studies.

The P5 full-step theorem is the main constructive optimizer result; P4 is its
quadratic bridge, and P3 is the conservative generic-IQC baseline. The generic
one-step-memory IQC framework is prior art; the new ingredients are the
certified full-matrix Muon operator and the structure-aware interconnection.
PL/nonconvex objectives, stochastic gradients, BF16/quantized execution,
weight decay, aspect-ratio scaling, complete neural-network training, and
formal circuit ports remain open.

## Layout

- `src/passive_muon/`: normalization, orthogonalizers, deficit metrics, repairs;
- `theory/`: claim ledger, analytic proof, and exact/numerical certificates;
- `experiments/`: matrix, qualified quadratic, and nonlinear falsification studies;
- `scripts/`: reproducible entry points;
- `results/`: committed manifests, summaries, and figures only;
- `third_party/`: upstream revisions and license notices.

Numerical audits explicitly use `float64`; public orthogonalizers preserve the
caller's dtype. Low-precision training kernels remain separate so that
implementation convenience cannot silently change a theorem.
