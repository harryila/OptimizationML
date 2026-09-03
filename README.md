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

Branch `p6-pl-convergence` retains the full P5 step while dropping convexity.
For every fixed differentiable globally `10`-smooth objective with finite
infimum satisfying the global PL inequality with constant `1`, an exact
value--momentum storage certificate proves

\[
f(W_t)-f_\star\le C
\left(\frac{399960001}{400000000}\right)^t,
\qquad m_t\to0,
\]

at `eta=1/32000`. The objective may be nonconvex and its minimizer set may be
non-singleton. This is global function-value convergence, not arbitrary-pair
incremental contraction; no unique or preselected minimizer is claimed. The
proof is dimension independent and allows changing local Hessian orientations
where Hessians exist. See
[`theory/pl_convergence_certificate.md`](theory/pl_convergence_certificate.md).

The canonical P6 certificate was generated from source commit
`a8f650f6c60dcbc5d2f83647fd367348f4c67548`. The completed, documented P6
checkpoint is commit `ef88d8f5b26148af0ec1ca70b506048938bf9bef`, frozen by the
annotated tag `p6-checkpoint`. These are deliberately distinct provenance
roles. A standalone standard-library reconstruction rebuilds the interpolation
supplies, storage, complete `4 x 4` LMI, and exact Sylvester minors before it
reads and compares the canonical artifact.

Branch `p7-robust-dissipativity` keeps the same global smooth-PL class,
max-floor operator, full step, and storage while adding gradient error `xi_t`
and post-operator implementation error `e_t`. For the exact max floor `c=1`,
no additive epsilon, five Jordan steps with coefficients
`(6889/2000,-191/40,4063/2000)`, constant repair
`rho=210177835339081/260261360000`, `beta=19/20`, and `eta=1/32000`, an exact
dimension-independent `6 x 6` certificate proves the pathwise inequality

\[
V_{t+1}\le
\frac{399960001}{400000000}V_t
+\frac12\lVert\xi_t\rVert_F^2
+\frac1{2000000}\lVert e_t\rVert_F^2.
\]

This gives deterministic input-to-storage/output bounds and a rigorous
bounded-second-moment stochastic corollary for the objective gap, momentum,
and true gradient. It is not full-state ISS: a square-summable harmonic error
can drift forever along a flat nonunique minimizer set while `V_t=0`. The
certificate and 144-case, 17,280-update falsification snapshot are sourced at
commit `c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a`; all sampled cases had zero
candidate violations. See
[`theory/robust_dissipativity_certificate.md`](theory/robust_dissipativity_certificate.md)
and [`results/summaries/P7_RESULTS.md`](results/summaries/P7_RESULTS.md).

The completed P7 checkpoint is
`30b55e53f50525ea980dc41f3201fdd160d4a75e`, frozen by the annotated tag
`p7-checkpoint`. P7 is the submission cutoff; P6 is its zero-disturbance
corollary. The independent human audit of C11--C12 is still pending in
[`theory/audits/P7_HUMAN_PROOF_AUDIT.md`](theory/audits/P7_HUMAN_PROOF_AUDIT.md).

Branch `p8-certified-mixed-precision` supplies a deliberately narrow
constructive implementation result. For a proposed fixed-`2 x 2` kernel,
FP32 computes the scaled max-floor normalizer, each complete serial-Horner
stage, and the linear repair; the normalized stage input and each of the five
completed stage outputs are stored in BF16. Under the locked IEEE
round-to-nearest, gradual-underflow,
no-FTZ arithmetic contract, every finite FP32 input with maximum absolute
entry at most `2^116` satisfies

\[
\lVert\widehat R(s)-R(s)\rVert_F
\le \frac{11}{100000}\lVert s\rVert_F+\frac{347}{100}.
\]

For an arbitrary real input in the same shape and range, an entrywise FP32
input adapter and the exact Lipschitz bound for `R` give

\[
\lVert\widehat R_{\mathbb R}(s)-R(s)\rVert_F
\le \frac1{5000}\lVert s\rVert_F+\frac{347}{100}.
\]

Placing that adapter inside the otherwise exact-real P7 loop with zero
gradient noise gives the exact rate

\[
q_8=\frac{41597186684695561}{41601344000000000}<1
\]

provided `V_0 <= H_safe`, where

\[
H_{\rm safe}=\frac{2600084\,2^{232}}{1655544025}
\approx1.08394\times10^{67}.
\]

The certificate checks that this storage range is invariant and keeps every
operator signal below the `2^116` input limit. It then gives the worst-case
objective-gap neighborhood

\[
\limsup_t(f(W_t)-f_\star)
\le
\frac{462392438350000000}{207695315294468001}
=2.22630172325\ldots.
\]

This is not literal upstream Muon, not a native all-BF16-intermediate kernel,
not an arbitrary-shape certificate, and not a whole-FP32-optimizer theorem.
In particular, FP32 momentum/Nesterov and parameter-update rounding are not
covered by P7's single post-operator port; parameter subtraction has a genuine
large-binade stalling obstruction. A deterministic 82-case CPU falsification
grid found zero candidate violations of either affine bound; its float64
reference is not exact, so this is diagnostic rather than proof. See
[`theory/mixed_precision_certificate.md`](theory/mixed_precision_certificate.md)
and [`results/summaries/P8_RESULTS.md`](results/summaries/P8_RESULTS.md).

Branch `p9-scalable-mixed-precision` replaces the two parts of P8 that do not
scale. A fixed balanced FP32 tree replaces the long serial norm and dot
reductions, and every stage boundary is stored as a compensated BF16 pair
`(high, low)` before reconstruction in FP32. For each audited shape `(r,c)`,
with `rc<=2^52` and maximum input magnitude at most `2^116`, exact rational
propagation proves

\[
\lVert\widehat R_{r,c,\mathbb R}(s)-R_{r,c}(s)\rVert_F
\le A_{r,c}\lVert s\rVert_F+B_{r,c},
\qquad
A_{r,c}=\frac{102465557}{549755813888}.
\]

All seven predeclared representative Transformer shapes certify, including
`768 x 3072`, `3072 x 12288`, `4096 x 11008`, and `4096 x 14336`. The largest
listed intercept is `2179083213031/1099511627776 = 1.98186463697...`.
Absorbing this affine error through P7 at zero gradient noise gives the common
strict rate

\[
q_9=\frac{137425214491}{137438953472}<1,
\]

and the largest listed objective-gap neighborhood is
`798350562999/1099511627776 = 0.726095607205...`. The proof includes explicit
FP32/BF16 overflow guards and an invariant sufficient to keep every operator
signal inside the finite input domain.

P9 also records why the arithmetic had to change. On an all-ones
`4096 x 11008` input, P8-style serial FP32 square accumulation sticks at
`2^24`; the returned normalized rank-one singular value has exact square
`43/16>25/16`, outside the proof tube before BF16 is involved. Ordinary
one-term BF16 storage separately loses the generic normwise boundary proof at
rank 72; that second statement is a proof obstruction, not an impossibility
theorem. The compensated boundary raises its corresponding slope-only gate
from 71 to 4,656,751. The implemented balanced kernel is deliberately a slow
CPU proof reference, so no BLAS, GPU, throughput, upstream-Muon, or whole-FP32
optimizer claim follows. Its two BF16 buffers have the same nominal storage as
one FP32 buffer, so it does not claim compression. See
[`theory/scalable_mixed_precision_certificate.md`](theory/scalable_mixed_precision_certificate.md)
and [`results/summaries/P9_RESULTS.md`](results/summaries/P9_RESULTS.md).

Branch `p10-finite-precision-outer-loop` instantiates the outer arithmetic
ports left open by P9 for one proposed CPU proof-reference optimizer shell at
shape `4096 x 11008`. The shell uses separately rounded FP32 EMA/Nesterov
updates at `beta=19/20`, the full `eta=1/32000` step, the P9 repaired operator,
and a three-word FP32 master whose logical value is `high+middle+low`. Two
error-free `TwoSum` cascades accumulate updates that ordinary FP32 parameter
subtraction can lose at large binades. Exact rational envelopes are derived
for the momentum, signal, and logical parameter-update residuals and are
closed through P7 into a storage/function-value certificate.

At the locked shape and full step, exact rational absorption proves on the
guarded set `V<=1`

\[
V_{t+1}\le
\frac{549700907325}{549755813888}V_t
+\frac{2162331}{1099511627776},
\qquad q_{10}<1.
\]

The exact forcing fits within the one-step storage margin, so this set is
forward invariant apart from the separately conditional high-word guard. The
resulting certified objective-gap neighborhood is

\[
\limsup_t(f(W_t)-f_\star)
\le\frac{399957341889}{549755813888}
=0.727518166766\ldots<1.
\]

An exact executable witness also isolates the reason for the compensated
master: at `W=2^30`, the actual P9 repaired output generates a positive FP32
step that ordinary subtraction loses for all 64 tested repeats, while the
compensated logical master moves immediately and its high word moves on
repeat 20.

The P10 high-word magnitude guard is conditional, not a consequence of PL
storage: an objective with a flat, nonunique minimizer set can permit
parameter drift without changing function value, gradient, momentum, or the
certified storage. The result therefore controls storage/function value,
true-gradient norm, and momentum while all arithmetic guards hold; it is not
full-state ISS or parameter convergence. Its locked result casts an exact
gradient oracle once to FP32 and absorbs that final boundary cast; it does not
cover earlier gradient-computation error, model-forward consumption of the
three-word master, literal upstream `torch.lerp_` bit semantics, weight decay,
aspect scaling, or native accelerator kernels. See
[`theory/finite_precision_outer_loop_certificate.md`](theory/finite_precision_outer_loop_certificate.md)
and [`results/summaries/P10_RESULTS.md`](results/summaries/P10_RESULTS.md).

Branch `p11-certified-implementation-margin` keeps every P10 theorem and
implementation parameter fixed while exposing two additional interfaces: a
pre-cast total gradient discrepancy `zeta` and a deployed-output discrepancy
`nu` beyond the P9 reference kernel. For budgets

\[
\|\zeta_t\|_F\le a_g\sqrt{V_t}+b_g,\qquad
\|\nu_t\|_F\le a_R\sqrt{V_t}+b_R,
\]

the exact composition proves

\[
V_{t+1}\le q_{11}(a_g,a_R)V_t+D_{11}(b_g,b_R).
\]

Zero error reproduces P10 exactly. On the declared `2^-40` budget grid, the
one-axis maxima are `10815225547/2^40` for `a_g`,
`513245498810/2^40` for `a_R`, `10879487718/2^40` for `b_g`, and
`13351103462525/2^40` for `b_R`; one more grid unit on each axis rejects the
forward-invariance certificate. A jointly nonzero profile
`(1/4096,1/128,1/4096,1/8)` retains a subunit objective-gap bound of
`930325132219/2^40`. These are acceptance limits for the stated sufficient
certificate, not measured production errors or instability thresholds. The
deployed output must already be a finite contiguous FP32 tensor, the
`2^30` high-word guard remains conditional, and human proof review is still
pending. See
[`theory/implementation_margin_certificate.md`](theory/implementation_margin_certificate.md)
and [`results/summaries/P11_RESULTS.md`](results/summaries/P11_RESULTS.md).
The inherited P10 provenance roles remain distinct: theorem source
`2d62b566...`, exact-artifact commit `6e7ea000...`, and diagnostic/checkpoint
commit `3246972d...`.

Branch `p12-additive-epsilon-deficit` closes the corresponding exact-real
question for the additive denominator

\[
E_{h,\epsilon}(M)=\mathcal H_h
\left(\frac{M}{\lVert M\rVert_F+\epsilon}\right),
\qquad \epsilon>0.
\]

For each fixed finite matrix shape, the global pairwise deficit scales
exactly by the `1/epsilon` law

\[
\delta_{m,n}(E_{h,\epsilon})
=\frac{\delta_{m,n}(E_{h,1})}{\epsilon}.
\]

A four-band Arb certificate and full rectangular tangent reduction prove the
dimension-uniform upper

\[
\delta_{m,n}(E_{h,1})
\le\frac{6602082433275499863}{41641817600000000}
=158.544530805386839\ldots.
\]

An exact rational swapped-diagonal pair proves the strict lower
`98823281/625000 = 158.1172496` for `2 x 2` and every shape with
`min(m,n)>=2` by zero-padding. At the pinned additive value
`epsilon=1e-7`, any global constant repair in such a shape must exceed
`1,581,172,496`, while the certified sufficient repair is approximately
`1.585445308e9`. This makes the deployed-scale constant repair catastrophic
at ordinary raw signal scale. The theorem is for the continuous exact-real
surrogate only; the pinned cast-before-norm BF16 map is discontinuous and is
not certified. Human proof review is pending and unsigned. See
[`theory/additive_epsilon_deficit_certificate.md`](theory/additive_epsilon_deficit_certificate.md)
and
[`results/summaries/P12_ADDITIVE_EPSILON_RESULTS.md`](results/summaries/P12_ADDITIVE_EPSILON_RESULTS.md).

Branch `p13-radial-passivation-tradeoff` converts that negative result into a
constructive exact-real repair. With `z=||M||_F/epsilon`, it integrates a
positive nonincreasing majorant `d_hat(z)` of P12's full-matrix pointwise
deficit and defines

\[
G_\epsilon(M)=p(z)\frac{M}{\lVert M\rVert_F},
\qquad p(z)=\int_0^z\widehat d(u)\,du.
\]

The complete Frobenius derivative has radial eigenvalue
`d_hat(z)/epsilon` and tangential eigenvalue `p(z)/(epsilon*z)`, so
`E_(h,epsilon)+G_epsilon` is globally monotone in every finite rectangular
shape. The closed form grows only logarithmically: at `epsilon=1e-7`, its
norm is certified near `2571.857826` at unit raw norm and `3086.959580` at
P11's exact `25.233506...` signal guard, instead of the constant repair's
`1.585e9` and `4.001e10`.

The improvement is output magnitude, not differential stiffness. The
construction has exact `Lip(G_epsilon)=158.544530805386.../epsilon`, while
P12's pair proves that every globally Lipschitz passivating correction in a
shape with `min(m,n)>=2` has Lipschitz constant strictly above
`158.1172496/epsilon`. An exact scalar quadratic control shows that the radial
repair alone is locally unstable in the pinned EMA/Nesterov ordering at
`beta=0.95`, `eta=1/32000`, and `epsilon=1e-7`; the theorem therefore does
not reuse the max-floor stability claims. See
[`theory/radial_passivation_tradeoff.md`](theory/radial_passivation_tradeoff.md)
and
[`results/summaries/P13_RADIAL_PASSIVATION_RESULTS.md`](results/summaries/P13_RADIAL_PASSIVATION_RESULTS.md).

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
uv run --locked python scripts/certify_pl_convergence.py
uv run --locked python scripts/reconstruct_pl_convergence.py
uv run --locked python scripts/certify_robust_dissipativity.py
uv run --locked python scripts/reconstruct_robust_dissipativity.py
uv run --locked python scripts/certify_mixed_precision.py
uv run --locked python scripts/reconstruct_mixed_precision.py
uv run --locked python scripts/certify_scalable_mixed_precision.py
uv run --locked python scripts/reconstruct_scalable_mixed_precision.py
uv run --locked python scripts/certify_outer_loop_roundoff.py
uv run --locked python scripts/reconstruct_outer_loop_roundoff.py
uv run --locked python scripts/certify_implementation_margin.py
uv run --locked python scripts/reconstruct_implementation_margin.py
uv run --locked python scripts/certify_additive_epsilon_deficit.py
uv run --locked python scripts/reconstruct_additive_epsilon_deficit.py \
  --require-canonical
uv run --locked python scripts/certify_radial_passivation_tradeoff.py
uv run --locked python scripts/reconstruct_radial_passivation_tradeoff.py \
  --require-canonical
uv run --locked pytest
```

The complete CPU result pipeline is:

```bash
uv run --locked python experiments/matrices/run_deficit_audit.py
uv run --locked python experiments/quadratics/run_lr_sweep.py
uv run --locked python experiments/quadratics/run_horizon_check.py
uv run --locked python experiments/quadratics/run_nonquadratic_falsification.py
uv run --locked python experiments/quadratics/run_nonquadratic_convergence_falsification.py
uv run --locked python experiments/nonconvex/run_pl_falsification.py
uv run --locked python experiments/nonconvex/run_robust_dissipativity_falsification.py
uv run --locked python experiments/mixed_precision/run_mixed_precision_falsification.py
uv run --locked python experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py
uv run --locked python experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py
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

This repository stays focused on sixteen technical goals:

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
8. full-step function-value convergence for globally smooth PL objectives,
   including nonconvex objectives with nonunique minimizers;
9. exact robust dissipativity and bounded-second-moment guarantees under
   additive gradient and repaired-operator-output errors;
10. a fixed-`2 x 2` mixed-precision operator-error certificate that
    instantiates the repaired-operator-output port for one proposed kernel;
11. a shape-parameterized compensated-BF16 certificate for representative
    Transformer matrix shapes, with balanced reductions and overflow guards;
12. a port-augmented finite-precision outer-shell certificate using FP32
    EMA/Nesterov arithmetic and a compensated three-word FP32 master;
13. exact two-port implementation-acceptance margins above that locked shell,
    including coordinatewise boundary controls and Pareto slices;
14. an exact-real, shape-uniform additive-epsilon deficit upper certificate,
    an exact rank-two lower witness, and the resulting deployed-epsilon
    constant-repair obstruction;
15. a full-matrix nonlinear radial passivator with logarithmic output growth,
    together with a universal near-matching `1/epsilon` stiffness lower bound
    and explicit-step negative control;
16. qualified matrix, quadratic, nonlinear, and precision diagnostics.

P7 is the submission cutoff and broadest robustness theorem; P6 is its
zero-disturbance smooth-PL corollary. P8 and P9 instantiate one P7 disturbance
port for proposed fixed-shape kernels and do not replace that headline. P5
supplies the stronger strongly-convex conclusions, P4 is the quadratic bridge,
and P3 is the conservative generic-IQC baseline. The generic one-step-memory
IQC framework is prior art; the new ingredients are the certified full-matrix
Muon operator and the structure-aware interconnection. P10 closes one proposed
outer arithmetic shell, and P11 quantifies additional admissible discrepancies
without asserting that any deployed system meets them. P12 shows that the
exact-real additive-epsilon alternative has a finite but catastrophically
large global repair at the pinned `1e-7` scale; it does not certify the BF16
backend. P13 reduces that repair's output magnitude through an exact radial
construction but proves that the worst-case `1/epsilon` differential
stiffness is unavoidable; it does not inherit P7--P11 stability. Production-
gradient measurement, model-forward use of the logical
master, weight decay, aspect-ratio scaling, complete stochastic neural-network
training, production-kernel parity, and formal circuit ports remain open.

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
