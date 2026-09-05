# Experiment results

The spectral, quadratic, and nonlinear falsification results were generated in
float64 on an arm64 CPU; Section 2 is the separately scoped BF16 deployment check. JSON files are the
source-of-truth manifests; CSV files are flattened views. All grid extrema are
lower-bound witnesses, not continuous-domain upper certificates.

## 1. Canonical exact witness

For the five-step Jordan map on diagonal matrices,

- `A = diag(3, 4)` and `B = diag(5/2, 3)`;
- exact-current-normalized pair gap: `-0.0702543371557703288`;
- exact pair repair threshold: `rho = 0.0562034697246162631`;
- current-plus-`1e-7` pair gap: `-0.07025429353210752`;
- fixed-scale-five pair gap: `+0.3750465803803160`;
- fixed-scale local off-diagonal divided differences before the common
  chain-rule factor `1/5`: `+1.981638806494635...` and
  `+1.315771498952257...`.

The current-normalized local determinant, both finite-pair signs, and all four
fixed-scale `2 x 2` local Jacobian modes at `A` are certified exactly. The
fixed-scale Jacobian is positive definite at that point (and therefore on some
sufficiently small neighborhood by continuity), but no global fixed-scale
monotonicity is claimed. The readable decimals are high-precision evaluations.
See `canonical_witness.json`.

## 2. Backend-specific BF16 deployment witness

The same canonical pair was executed on a macOS 15.5 arm64 CPU with PyTorch
2.13.0 using the literal operation order in pinned KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`: cast once to BF16, orient once,
normalize with retained dimensions and Python-float `eps=1e-7`, then apply five
Jordan stages with `B = b*A + (c*A)@A`.

- returned pair gap: `-7/128 = -0.0546875`;
- returned pair ratio: `-7/160 = -0.04375`;
- interpretation: a violation for this recorded backend, PyTorch build, dtype,
  epsilon, coefficients, iteration count, and operation order only.

The earlier local-shadow value `-3/128` used the real-arithmetic-equivalent but
BF16-inequivalent grouping `c*(A@A)` and is not reported as a deployed-order
result. The committed `bf16_witness.json` records output storage words, every
stage, cast/normalization behavior, an observable accumulation probe, complete
software/hardware data, source hashes, and the upstream revision. It does not
support a universal statement about other BF16 backends.

## 3. Deterministic 2x2 spectral audit

The exact-current unit-spectrum deficit is positive in all 24 audited
polynomial-prefix configurations. The final configured stages are:

For every row, exact rational arithmetic at the unit diagonal spectrum
`(3/5, 4/5)` also verifies unequal composed derivatives. Combined with the
analytic scale obstruction, this establishes an infinite unrestricted deficit
for exact normalization independently of the floating-point grid values.

| Baseline | Stages | Matmuls | Exact-current unit deficit | Fixed-scale unit deficit | Normalization-clean deficit | eps=1 diagonal grid deficit |
|---|---:|---:|---:|---:|---:|---:|
| Jordan quintic | 5 | 15 | 158.675144 | 158.680182 | 0.405923 | 157.958941 |
| Classical cubic | 5 | 10 | 0.015338 | 0 | 0.015338 | 0.007474 |
| Taylor quintic | 5 | 15 | 0.005034 | 0 | 0.005034 | 0.003841 |
| Polar Express repo `71cc` | 5 | 15 | 218.071514 | 218.073003 | 13.567219 | 217.888813 |
| CANS-5x4, delta=0.3 | 4 | 12 | 130.309482 | 3146.191653 | 15.784223 | 123.261935 |

Classical and Taylor are the clean attribution cases on this grid: their
fixed-scale Jacobians are nonnegative while current normalization produces a
strict deficit. Jordan, Polar Express, and CANS also have normalization-clean
witnesses, but their total deficits are dominated by fixed-scale polynomial
nonmonotonicity on parts of the unit-spectrum grid.

Polar Express is pinned to the current repository configuration at commit
`71cc37943d99cae780024c1d198977f2f8795407`. CANS uses the published
degree-five, four-stage, `delta=0.3` coefficient table; no code was copied from
its unlicensed repository.

## 4. Controlled diagonal matrix quadratics

Design: 32 matched 2x2 diagonal SPD quadratics, condition numbers
`{1, 3, 10, 30}`, current-Frobenius-plus-`eps=1` normalization, 750 updates,
73 geometric learning rates in `[1e-5, 3]`, and target-and-hold success defined
as objective ratio at most `1e-4` for every one of the final 50 iterations on
at least 90% of problems.

Each polynomial prefix uses one primary gain-matched comparison and one
algebraically equivalent cross-check:

1. gain-matched repair versus unrepaired, both with the same Jacobian at zero;
2. raw `+rho M` repair versus a gain-only unrepaired control, also with matched
   zero-input gain. This is the same contrast under learning-rate rescaling,
   so it is not independent evidence.

At the primary criterion:

- gain-matched repair versus unrepaired: 16/24 upper endpoints unchanged,
  8/24 repaired-only passing bands, 0 right shifts, and 0 left shifts among
  pairs where both bands existed;
- raw repair versus gain-only control: the same 16 unchanged and 8
  repaired-only pattern;
- raw repair versus unrepaired, which is not gain matched: 11 unchanged,
  5 left shifts, and 8 repaired-only bands;
- the 10 normalization-only classical/Taylor configurations are unchanged in
  both gain-matched comparisons at the learning-rate-grid resolution;
- all repaired-only bands occur in Jordan, Polar Express, or CANS prefixes
  whose fixed-scale maps are already nonmonotone on the audit grid.

Thus this experiment does **not** show that repairing the normalization-only
deficit widens the observed upper target endpoint. It does show repaired-only
finite-horizon target-and-hold bands for several strongly nonmonotone
polynomial prefixes. The corresponding unrepaired trajectories remain bounded
on the sampled learning-rate grid but do not meet the aggregate target within
750 steps. This cannot be attributed to normalization alone, and it is not
evidence that the repair removes divergence.

After normalizing both quantities by the zero-input gain, the complete-case
exploratory association between diagonal deficit and upper target endpoint is
`Spearman r = -0.256` (`n=16`, descriptive `p=0.339`). The eight configurations
whose unrepaired target band is absent are excluded based on the outcome. The
designed configurations are also dependent, so this is neither a confirmatory
test nor a predictor analysis over all 24 configurations.

The repair size is only a sampled-grid correction, not a trajectory-wide
certificate. All 48 repaired configuration summaries observe positive gradient
norms below the audit's lower dimensionless radius of `1e-8` (minimum observed
about `2.22e-162`), and 26 also exceed its upper radius of 1000 (maximum
observed `17100.25`). The analytic check at exactly zero does not certify the
unsampled interval between zero and `1e-8`.

Across nine target/fraction sensitivity criteria, gain-matched repairs range
from 1 to 10 repaired-only bands, 14 to 18 unchanged endpoints, 0 to 2 right
shifts, and 0 to 5 left shifts. This confirms that finite target bands depend
on the operational success definition.

## 5. Horizon check

Jordan step 5 and classical step 5 were repeated at 750 and 3000 iterations
with target ratios `1e-4` and `1e-8`.

- Jordan, target `1e-4`: the unrepaired band is absent at 750 steps but appears
  by 3000 steps with upper endpoint `0.007775`; the gain-matched repaired upper
  endpoint is `0.009263` at both horizons.
- Jordan, target `1e-8`: the unrepaired band is absent at both horizons; the
  gain-matched repaired upper endpoint is `0.003858`.
- Classical: all four interventions share upper endpoint `0.258301` at both
  horizons and both target ratios.

The horizon drift is why these are labeled finite target bands rather than
stability regions. Normalized endpoints above the local linear ceiling of two
are likewise finite-horizon target outcomes; they are not evidence of
asymptotic convergence to zero or of any particular attractor.

## 6. Floored-normalizer full-matrix certificate

For the real-arithmetic five-step Jordan map, define
`F_h,c(M)=H_h(M/max(c, ||M||_F))`, with `c>0`, exact coefficients
`(6889/2000, -191/40, 4063/2000)`, five iterations, and no additive epsilon.

An adaptive dyadic interval proof using outward-rounded Arb balls certifies

- `-159.5496 < h'(s) < 484.8763` for every `s` in `[0,1]`;
- identical 25,370-leaf covers at 160 and 224 bits, maximum dyadic depth 32;
- all diagonal, off-diagonal, repeated/zero, and rectangular-null tangent
  modes lie in the same slope interval by exact secant-average formulas;
- pairwise line integration handles the nondifferentiable floor boundary.

The resulting dimension-uniform full-matrix bracket at `c=1` is

\[
159.549525785 < \delta(F_{h,1})
\le \frac{41528474059081}{260261360000}
=159.5645010810709665084\ldots.
\]

An exact rational finite pair embedded in a rank-one mode inside the floor
proves the displayed strict lower bound; its actual deficit is approximately
`159.54952578566153`. The upper endpoint comes from the projection-envelope
bound. Its gap above the displayed safe lower threshold is `0.0149752961`, or
`0.009386%` relative. It is about `32.91%` of the map's zero-input differential
gain and about `3.04x` smaller than using that full `484.8763` gain as a
generic Lipschitz repair.

Exact rescaling proves `delta(F_h,c)=delta(F_h,1)/c`; therefore
`rho=(41528474059081/260261360000)/c` is a globally sufficient constant
conductance for every finite matrix shape. This is a certified near-minimal
repair, not an exact fixed-shape minimum. Setting
`rho=(41528474059081/260261360000)/c+mu` for `mu>0` yields a
`mu`-strongly monotone map and an `exp(-mu*t)` contraction guarantee for the
explicitly simplified system `dot(M)=-(F_h,c(M)+rho*M-b)`. This does not
establish momentum-Muon training stability.

Related finite-step regularization work by Chang et al. gives global
Frobenius-Lipschitz bounds for an additive denominator regularizer. It does not
provide a monotonicity-deficit certificate for this max-floor architecture:
<https://arxiv.org/abs/2606.01720>.

## 7. Stylized non-Nesterov repaired momentum certificate

For the stylized deterministic real-arithmetic, non-Nesterov quadratic loop

\[
m_{t+1}=\beta m_t+H(W_t-W_\star),\qquad
W_{t+1}=W_t-\eta R_{\rho,c}(m_{t+1}),
\]

the repaired floored map is globally `mu`-strongly monotone and has the
rigorous Lipschitz bound `K=rho+484.8763/c`. After the quadratic conditioning
transform, a dimension-independent `3 x 3` IQC proves global incremental
exponential stability when

\[
0<\eta KL<
\frac{2(1-\beta)^2(1+\beta)\nu}
{(1+\beta)^2-4\beta\nu^2},
\qquad \nu=\frac{\mu\ell}{KL}.
\]

An exact closed-form factorization proves the whole strict region. A separate
rational representative uses `c=1`, `mu=bar_delta_1`, `rho=2*bar_delta_1`,
`ell=1`, `L=10`, and `beta=0.9`. Its values are

- `nu = 41528474059081 / 2092515133879300`;
- strict sector endpoint `eta_star = 2.599354207182316e-8`;
- locked `eta = 1.2437728921821615e-8`, about `47.85%` of that endpoint;
- exact rate certificate `tau^2 = 99999/100000`.

Exact rational Sylvester checks establish positivity of the storage and
negative definiteness of the LMI. A floating-point SDP was used to discover
the locked storage and multipliers, but solver status is not part of the
proof. The closed-form region itself requires no solver.

The matched rank-one mode of `H=diag(1,10)` uses the repository's floored
five-step Jordan operator and holds every configuration fixed except `eta`:

- the certified trajectory reaches position about `2.37e-38` after 100,000
  updates from position `1`;
- at `eta=0.0005`, an exact local Jury sign proves the optimizer unstable and
  the trajectory approaches a nonzero period-four orbit;
- at `eta=0.002`, magnitude exceeds `1e100` in 159 updates.

The linked zero-linearization loses local stability at
`eta = 0.000472633706612...`, about `18,183x` the global sector endpoint. This
is therefore a clean, dimension-uniform, but extremely conservative
sufficient theorem. The complex-skew witness makes the endpoint sharp only
for the reduced strongly-monotone/Lipschitz sector class, not for the linked
floored Jordan architecture. No claim is made for nonquadratic, stochastic,
BF16, or neural-network training.

This generic strongly-monotone/Lipschitz, one-step-memory IQC construction is
prior art; see Lessard--Recht--Packard and Zhang--Bao--Lessard--Grosse (JMLR
2021): <https://arxiv.org/abs/1408.3595> and
<https://jmlr.org/papers/v22/20-1068.html>. The new ingredient is the certified
full-matrix floored-Muon sector and its repaired interconnection.

## 8. Pinned EMA/Nesterov ordering certificate

The pinned upstream update instead uses the EMA state and Nesterov signal

\[
m_{t+1}=\beta m_t+(1-\beta)g_t,\qquad
s_{t+1}=\beta m_{t+1}+(1-\beta)g_t,\qquad
W_{t+1}=W_t-\eta R(s_{t+1}).
\]

For deterministic quadratic gradients, the conditioned recurrence is

\[
z_{t+1}=\beta z_t+(1-\beta)y_t,\qquad
p_{t+1}=\beta^2z_t+(1-\beta^2)y_t,\qquad
y_{t+1}=y_t-\alpha u(p_{t+1}).
\]

A second dimension-independent `3 x 3` IQC/LMI therefore matches this exact
state-and-signal ordering in real arithmetic after replacing the upstream
orthogonalizer by the repaired floored map and omitting weight decay. At the
pinned default `beta=0.95`, a 60-digit stationarity solve corroborated by a
broad logarithmic scan locates a design near `mu=648.024`. The exact replay uses the nearby rational design

- `mu=648`, `rho=bar_delta_1+648`;
- `nu=208209088000/4152745686529`;
- `alpha=1/400`;
- `eta=65065340/336372400608849 = 1.934324572474699e-7`;
- `tau^2=99999/100000`.

The rational storage and IQC multipliers pass exact Sylvester checks; no
floating-point solver status carries the claim. The complex-skew necessary
boundary for the reduced sector class at locked `mu` is
`eta approximately 2.09708214294e-7`, and the numerical stationary design over `mu` is
`approximately 2.09708214366e-7`.

Matched float64 rank-one controls hold the quadratic, operator, floor, repair,
initial state, and operation order fixed; only `eta` changes:

- at the exact locked `eta`, position falls from `1` to about `7.64e-113`
  after 100,000 updates;
- at `eta=0.0025`, an exact Jury sign proves local instability and the run
  settles into a recorded period-two orbit with zero float64 tail residual;
- at `eta=0.005`, position exceeds magnitude `1e100` in 264 updates.

The matched curvature-10 zero-linearization has the exact local threshold

\[
\eta_{\mathrm{local}}
=\frac{8120154432000000000000000}
{3901919808117690731741568607}
=0.002081066457364538\ldots.
\]

This is `10,758.62x` the exact locked point. At the numerical stationary
design, the matching local threshold is `0.002081027708...`, about `9,923.44x`
its complex-skew necessary boundary. Because the gap remains thousands-fold,
the predeclared hard rule classifies this
as an appendix/proof-of-principle result, not a practical-stability headline.
It proves an embedding of the certified repair in the pinned EMA/Nesterov
ordering; it does not establish BF16, nonquadratic, stochastic, or
neural-network convergence.

The pinned `muon.py` bytes have SHA-256
`2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
An automated regression checks the literal two-`lerp` sequence and in-place
gradient behavior against the algebraic recurrence. Independent parity and
human proof audits remain unchecked.

## 9. P4 structure-aware full-step quadratic certificate

For every finite real matrix shape, use the exact-real-arithmetic repaired map

\[
R(M)=\mathcal H_{q^{\circ5}}
\!\left(\frac{M}{\max\{1,\lVert M\rVert_F\}}\right)+\rho M,
\]

with no additive epsilon, five Jordan steps, exact coefficients
`(6889/2000,-191/40,4063/2000)`, and constant
`rho=210177835339081/260261360000`. For the pinned EMA/Nesterov ordering at
`beta=19/20`, an exact dimension-independent `4 x 4` certificate proves
arbitrary-pair global incremental exponential stability for every fixed
quadratic `I <= H <= 10 I` at

\[
\eta=\frac1{32000},\qquad \tau=\frac{99999}{100000}.
\]

The structure-aware centered residual description retains the full locked
step and is the quadratic bridge between the generic P3 sector baseline and
P5. See `structure_aware_stability_certificate.json`.

## 10. P5 nonlinear strongly-convex certificates

P5 keeps exactly the P4 operator, normalization, coefficients, five-step
count, constant repair, matrix domain, and pinned `beta=19/20`. It establishes
two different guarantees for every fixed differentiable globally
`1`-strongly-convex, `10`-smooth objective:

- At `eta=1/640000`, a common quadratic storage proves arbitrary-pair global
  incremental contraction at `tau=99999/100000`.
- At the full P4 step `eta=1/32000`, exact objective-gap/interpolation storage
  proves global exponential convergence of each trajectory to the unique
  minimizer at `tau=2499/2500`.

The full-step theorem is trajectory-to-minimizer convergence, not an
incremental-stability claim. Neither proof
freezes a Hessian basis, so local Hessian orientations may change over time.
The full-step exact replay verifies positive storage, exact objective-value
flow cancellation, and strict negativity of the `5 x 5` LMI using rational
Sylvester minors. A separate standard-library-only implementation reconstructs
the entire matrix and matches every exact canonical field without importing
the project certificate code.

The two deterministic CPU/float64 probes each contain 36 changing-orientation
cases. The full-step probe ran 500 updates per case and observed no candidate
implementation violation, Lyapunov-rate violation, divergence, or nonfinite
value. Its maximum resolved `V_(t+1)/V_t` was `0.9799999799424988`, below
`tau^2=0.99920016`. These sampled passes are falsification diagnostics, not the
proof. See `P5_RESULTS.md`, `P5_FULL_STEP_RESULTS.md`, and the corresponding
machine-readable manifests.

The P5 results are for the repaired max-floored exact-arithmetic optimizer.
They do not cover exact-current or additive-epsilon normalization, stochastic
or time-varying objectives, BF16, weight decay, aspect-ratio scaling, or a
complete neural-network training system.

## 11. P6 smooth-PL function-value convergence

P6 keeps the P5 repaired exact-real operator and pinned EMA/Nesterov ordering
but drops convexity. For every fixed differentiable globally `10`-smooth
objective with finite infimum satisfying the global PL inequality with
constant `1`, an exact dimension-independent value--momentum certificate at
the full P5 step proves

\[
f(W_t)-f_\star\le C
\left(\frac{399960001}{400000000}\right)^t,
\qquad m_t\to0,
\qquad \eta=\frac1{32000}.
\]

This is global function-value convergence, not arbitrary-pair incremental
stability. PL permits nonconvex objectives and nonunique minimizers; no unique
or preselected minimizer is claimed. Summability of the certified update
signal additionally makes each trajectory converge to some
trajectory-dependent global minimizer. The exact replay checks function-value
cancellation, positive storage, and strict negativity of the `4 x 4` rational
LMI.

The accompanying 72-case, 500-update CPU/float64 falsification grid uses an
analytic PL objective family with Hessian spectrum in `[-3/8,9]`. It includes
36 rank-deficient nonunique-minimizer cases; 59 cases sampled negative
curvature, and all 72 changed Hessian orientation. It found zero candidate,
Lyapunov-rate, nonpositive-storage, unresolved-resurgence, analytic-bound,
divergence, or nonfinite violations. All 72 had at least one individual
objective increase, which is allowed because the theorem contracts a
composite storage rather than objective value pointwise. These sampled passes
are diagnostics, not the proof. See `P6_RESULTS.md`,
`pl_convergence_certificate.json`, and `pl_falsification.json`.

P6 remains deterministic and exact-real-arithmetic. It does not cover
stochastic gradients, BF16, weight decay, aspect-ratio scaling, exact-current
or additive-epsilon normalization, an unrepaired upstream implementation, or
complete neural-network training.

## 12. P7 robust dissipativity

P7 keeps the P6 repaired exact-real operator, global smooth-PL class, pinned
EMA/Nesterov ordering, and full `eta=1/32000` step while adding gradient error
`xi_t` and post-operator output error `e_t`. An exact, dimension-independent
`6 x 6` rational certificate proves, for every disturbance realization,

\[
V_{t+1}\le
\frac{399960001}{400000000}V_t
+\frac12\lVert\xi_t\rVert_F^2
+\frac1{2000000}\lVert e_t\rVert_F^2.
\]

The exact convolution yields deterministic bounded-input neighborhoods,
storage/output convergence for square-summable disturbances, and expected
storage/function-gap neighborhoods under conditional second-moment bounds.
Unbiasedness is not needed for that expectation inequality because the base
certificate is pathwise. Square-summable disturbances do not ensure iterate
convergence on a nonunique minimizer set: the committed flat-direction
harmonic-drift example supplies an explicit counterexample. Absolute
summability is a sufficient stronger condition.

The 144-case, 17,280-update CPU/float64 falsification grid found zero candidate
violations. It covers bounded deterministic, seeded stochastic, and
implementation-only disturbances, but remains a diagnostic rather than the
proof. The exact certificate and diagnostic source revision is
`c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a`; the completed checkpoint is
`30b55e53f50525ea980dc41f3201fdd160d4a75e`, tagged `p7-checkpoint`. The human
audit of C11--C12 remains pending. P7 is the submission cutoff, and P6 is its
zero-disturbance corollary. See `P7_RESULTS.md` and the two
`robust_dissipativity_*.json` manifests.

## 13. P8 fixed-2x2 mixed-precision operator certificate

P8 instantiates P7's post-operator error port for one proposed, fixed-`2 x 2`
implementation. The max-floor normalizer, each complete fixed-order serial
Horner stage, and the repair use FP32; the normalized stage input and each of
the five completed stage outputs are stored in BF16. Under IEEE
round-to-nearest, ties-to-even arithmetic with
gradual underflow and no FTZ/DAZ, every finite FP32 input whose maximum
absolute entry is at most `2^116` satisfies the exact bound

\[
\lVert\widehat R(s)-R(s)\rVert_F
\le\frac{11}{100000}\lVert s\rVert_F+\frac{347}{100}.
\]

Entrywise FP32 conversion extends the interface to every real `2 x 2` input
in the same range. Using the exact global Lipschitz bound for the ideal
repaired map gives

\[
\lVert\widehat R_{\mathbb R}(s)-R(s)\rVert_F
\le\frac1{5000}\lVert s\rVert_F+\frac{347}{100}.
\]

An exact Sturm calculation and rational matrix-roundoff recurrence close the
full five-stage invariant. Placing the real-input adapter in the otherwise
exact-real P7 loop, setting gradient noise to zero, and using Young's
inequality with `theta=5124` gives

\[
V_{t+1}\le
\frac{41597186684695561}{41601344000000000}V_t
+\frac{4936769}{819840000000}.
\]

For

\[
H_{\rm safe}=\frac{2600084\,2^{232}}{1655544025}
\approx1.08394\times10^{67},
\]

the exact replay checks that `V_0<=H_safe` is invariant and implies every
signal has Frobenius norm at most `2^116`. Thus all adapter calls remain in the
certified range. The rate is strictly below one, and P7's exact storage
conversion gives

\[
\limsup_t(f(W_t)-f_\star)
\le
\frac{462392438350000000}{207695315294468001}
=2.22630172325\ldots.
\]

A deterministic 82-case CPU falsification grid found zero candidate bound
violations and zero nonfinite results. The maximum observed kernel-bound ratio
was `0.32618452101249257`, and the maximum all-real-adapter-bound ratio was
`0.18142092780684033`. Its ideal-formula comparator uses float64 rather than
exact arithmetic, so these are diagnostics rather than the proof.

This is a worst-case certificate, not a measured loss and not a tightness
claim. It is not literal upstream Muon, not an all-BF16-intermediate or native
accelerator kernel, not valid for arbitrary shapes, and not a whole-FP32-loop
theorem. FP32 EMA/Nesterov and parameter-update rounding remain outside the
single P7 output-error port; parameter subtraction can stall at large binades.
See `P8_RESULTS.md`, `mixed_precision_certificate.json`,
`mixed_precision_falsification.json`, and
`../../theory/mixed_precision_certificate.md`.

## 14. P9 scalable mixed-precision certificate

P9 replaces P8's fixed-`2 x 2`, one-term-BF16 result with an exact,
shape-parameterized certificate for a proposed proof-reference kernel. For
each audited shape `(r,c)` and every real input with
`max_ij |s_ij| <= 2^116`, it proves

\[
\lVert\widehat R_{r,c}(s)-R_{r,c}(s)\rVert_F
\le A_{r,c}\lVert s\rVert_F+B_{r,c}.
\]

The target is the repaired max-floor (`c=1`, no epsilon), five-step Jordan
operator with coefficients `(6889/2000,-191/40,4063/2000)` and
`rho=210177835339081/260261360000`. Inputs are finite 2-D matrices with at
most `2^52` entries. The specified arithmetic uses a scale-free balanced-FP32
normalizer, deterministic balanced FP32 dots, and two-term BF16 stage states
with an exact upward-`2^-40` recurrence.

All seven locked Transformer shapes
`768 x 768`, `768 x 3072`, `768 x 50257`, `3072 x 12288`,
`4096 x 4096`, `4096 x 11008`, and `4096 x 14336` certify with

\[
A_{r,c}=\frac{102465557}{549755813888},\qquad
q_{r,c}=\frac{137425214491}{137438953472}<1.
\]

The largest intercept is
`2179083213031/1099511627776 = 1.981864636974...`; after closing the P7
post-operator port at `eta=1/32000`, the largest certified objective-gap
neighborhood is `798350562999/1099511627776 = 0.726095607205...`.

The negative evidence is deliberately narrower. Serial FP32 normalization on
the `4096 x 11008` all-ones matrix returns a rank-one output with squared
singular value `43/16`, outside the proof tube. The generic one-term-BF16
boundary estimate closes only through short rank `71`; that is an obstruction
to this proof, not an executable impossibility theorem. In the full recurrence,
`4608 x 18432` is a qualified 4:1 out-of-envelope point failing only the
fifth-stage-input tube; `4608` is not a universal rank frontier.

The generator and independent standard-library reconstruction are the exact
evidence. The optional CPU native-matmul diagnostic uses modest shapes by
default and a float64, non-exact comparator; realistic shapes require
`--include-realistic-shapes`. It is falsification only. P9 is not literal
upstream Muon, a GPU/BLAS/tensor-core parity result, native BF16 matmul, a
whole-FP32-loop theorem, or a neural-training result. See `P9_RESULTS.md` and
`../../theory/scalable_mixed_precision_certificate.md`.

## 15. P10 finite-precision outer-loop certificate

P10 closes the outer FP32 arithmetic ports for one proposed proof-reference
shell around the P9 operator at `4096 x 11008`. The exact-real reference still
uses max-floor normalization `c=1`, no additive epsilon, five Jordan stages
with coefficients `(6889/2000,-191/40,4063/2000)`, and repair
`rho=210177835339081/260261360000`. The executable shell locks separately
rounded CPU FP32 EMA/Nesterov operations at `beta=19/20`, the full
`eta=1/32000` step, and a three-word FP32 master updated by two `TwoSum`
cascades.

The locked deterministic theorem evaluates the exact gradient at the logical
real master `high+middle+low`, casts it once entrywise to FP32, and absorbs
that boundary cast plus the state and master rounding residuals through P7.
For every differentiable globally `10`-smooth objective satisfying the global
PL inequality with constant `1`, it proves on the guarded set `V_t<=1`

\[
V_{t+1}\le
\frac{549700907325}{549755813888}V_t
+\frac{2162331}{1099511627776}.
\]

The exact rate is `0.9999001255437179...<1`; the forcing is at most the
one-step storage margin, so `V<=1` is invariant apart from the separately
conditional high-word bound. The storage-to-function result is

\[
\limsup_t(f(W_t)-f_\star)
\le\frac{399957341889}{549755813888}
=0.727518166766\ldots<1.
\]

At `V<=1`, exact range propagation keeps the signal norm below
`25.233459`, the P9 output norm below `32614.736<2^15`, and the rounded
parameter-step max-absolute entry below `1.019211<2`. The middle and low
master-word guards are
forward invariant. The high word must still be checked against `2^30` after
every update: global PL storage is not coercive along a flat nonunique
minimizer set.

The negative control uses the actual P9 repaired operator at a `2 x 2` signal.
At `W_(1,1)=2^30`, its positive rounded step
`13548867/8388608` is lost by ordinary FP32 subtraction for all 64 repeated
updates. The compensated logical master moves immediately and its high word
first moves on update 20. A separate CPU diagnostic covers `1 x 1`, `2 x 3`,
and `8 x 16`: 810 exact EMA residual-entry checks, 270 exact master-residual
checks, 544 `TwoSum` identities, and 270 logical-update identities have zero
candidate violations. These finite checks validate/falsify the operation
graph; the exact rational generator and standalone standard-library
reconstruction carry the theorem.

P10 is a storage/function-value, true-gradient, and momentum result under
conditional arithmetic guards. It is not full-state ISS or parameter
convergence, literal upstream `torch.lerp_` parity, a gradient-computation or
model-forward theorem, native accelerator parity, or a neural-training
result. See `P10_RESULTS.md`, `outer_loop_roundoff_certificate.json`,
`finite_precision_outer_loop_diagnostic.json`, and
`../../theory/finite_precision_outer_loop_certificate.md`.

## 16. P11 certified implementation margins

P11 preserves the complete P10 theorem and arithmetic shell at
`4096 x 11008`, `beta=19/20`, `eta=1/32000`, exact max floor `c=1`, no
additive epsilon, five Jordan stages with coefficients
`(6889/2000,-191/40,4063/2000)`, and constant repair
`rho=210177835339081/260261360000`. It introduces a total pre-cast gradient
discrepancy `zeta` and a deployed-output discrepancy `nu` beyond the P9
reference kernel:

\[
\|\zeta_t\|_F\le a_g\sqrt{V_t}+b_g,\qquad
\|\nu_t\|_F\le a_R\sqrt{V_t}+b_R.
\]

Exact composition gives

\[
V_{t+1}\le q_{11}(a_g,a_R)V_t+D_{11}(b_g,b_R).
\]

Zero external error reproduces the P10 fractions exactly. On the declared
`2^-40` budget grid, the exact one-axis maxima are

| axis | largest certified value | adjacent rejected value |
|---|---:|---:|
| `a_g` | `10815225547/2^40` | `10815225548/2^40` |
| `a_R` | `513245498810/2^40` | `513245498811/2^40` |
| `b_g` | `10879487718/2^40` | `10879487719/2^40` |
| `b_R` | `13351103462525/2^40` | `13351103462526/2^40` |

Every accepted endpoint has `q11+D11=1`; the adjacent point has
`q11+D11=1+2^-40`. These controls show rejection of the sufficient
certificate, not actual instability. Two nine-row coordinatewise-maximal
gradient/operator frontier slices are recorded exactly in the canonical JSON.

The all-positive profile
`(a_g,a_R,b_g,b_R)=(1/4096,1/128,1/4096,1/8)` certifies

\[
q_{11}=\frac{274850515349}{274877906944},\qquad
D_{11}=\frac{1254603}{549755813888},
\]

with objective-gap limsup at most
`930325132219/1099511627776 = 0.846125778679... < 1`. Its signal, deployed
output, rounded step, and middle/low master-word guards close; the high word
remains a conditional premise. The theorem requires the deployed output to
already be a finite contiguous FP32 tensor, and model-weight error is covered
only insofar as it induces the declared pre-cast gradient discrepancy.

Replay the generator and standalone standard-library reconstruction with:

```bash
uv run --locked python scripts/certify_implementation_margin.py \
  --output results/summaries/implementation_margin_certificate.json
uv run --locked python scripts/reconstruct_implementation_margin.py \
  --require-canonical
```

The P10 provenance roles are explicit: theorem source `2d62b566...`, exact
artifact commit `6e7ea000...`, and diagnostic/head checkpoint `3246972d...`.
The P10 and P11 human audits remain pending. See `P11_RESULTS.md`,
`implementation_margin_certificate.json`, and
`../../theory/implementation_margin_certificate.md`.

## 17. Unrun gate

No NanoGPT result is reported. This machine exposes neither CUDA nor an
available MPS device. The `rho` used in the existing quadratic study is only a
finite-grid sampled repair; those results do not retroactively test the new
floored architecture or its global certificate. A matched language-model
sweep remains gated on suitable compute, a predeclared comparison, and
implementation-level parity for the chosen floored design. P10 closes one
proposed outer shell but does not specify how a model forward/backward pass
consumes the logical three-word master, so it does not clear that training
gate. Remaining implementation work includes aspect-ratio scaling, weight
decay, gradient-evaluation and represented-model error, and optimized-kernel
parity. P11 supplies exact acceptance budgets for two aggregate discrepancy
ports, but no production error has yet been measured against those budgets.

## 18. P12 additive-epsilon deficit certificate

For every fixed finite real matrix shape and every `epsilon>0`, P12 studies

\[
E_{h,\epsilon}(M)=\mathcal H_h
\left(\frac{M}{\lVert M\rVert_F+\epsilon}\right),
\]

where `h` is exactly five compositions of the Jordan quintic with coefficients
`(6889/2000,-191/40,4063/2000)`. The unrestricted pairwise deficit obeys the
exact fixed-shape `1/epsilon` scaling law

\[
\delta_{m,n}(E_{h,\epsilon})
=\frac{\delta_{m,n}(E_{h,1})}{\epsilon}.
\]

A two-precision Arb proof, four normalized-radius bands, the full rectangular
spectral tangent decomposition, and the exact projection/anticommutator
envelope prove

\[
\delta_{m,n}(E_{h,1})
\le\frac{6602082433275499863}{41641817600000000}
=158.544530805386839\ldots
\]

uniformly over every finite positive `m,n`. The origin is handled by its
ordinary positive Frechet derivative; no sampled zero-radius surrogate is
used. Integrating the symmetric-Jacobian lower bound along line segments
promotes the complete tangent-space result to the authoritative global
pairwise inequality.

An exact rational swapped-diagonal `2 x 2` pair at normalized radius
`8974467/10^9`, with Pythagorean direction
`(803760/1136689,803761/1136689)`, proves

\[
\delta_{2,2}(E_{h,1})>
\frac{98823281}{625000}=158.1172496.
\]

Its exact reduced quotient has SHA-256
`de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336`.
The pair zero-pads only into shapes with `min(m,n)>=2`, so that qualification
applies to the lower endpoint. The dimension-uniform upper remains valid for
all finite rectangular shapes.

The sufficient global repair is

\[
\rho_\epsilon=
\frac1\epsilon
\frac{6602082433275499863}{41641817600000000}.
\]

At the pinned `epsilon=1e-7`, any global constant repair in a shape containing
the witness must exceed `1,581,172,496`; the certified sufficient repair is

\[
\frac{6602082433275499863}{4164181760}
=1,585,445,308.053868\ldots.
\]

This is catastrophically large at ordinary raw signal scale. For comparison,
the max-floor `c=1` sufficient upper is only
`41528474059081/260261360000 = 159.564501081071...`. P12's conclusion is a
global constant-repair obstruction, not evidence that every local trajectory
attains the worst case.

The theorem concerns the continuous exact-real surrogate. The pinned
upstream map casts to BF16 before its norm and stages, is backend-specific and
discontinuous, and is not covered by the Jacobian or repair certificate. The
upstream revision establishes formula provenance only. The human proof audit
is pending and unsigned. See `P12_ADDITIVE_EPSILON_RESULTS.md`,
`additive_epsilon_deficit_certificate.json`, and
`../../theory/additive_epsilon_deficit_certificate.md`.

## 19. P13 radial passivation tradeoff

P13 keeps P12's exact-real additive-epsilon operator and defines a radial
correction from a positive nonincreasing majorant of its certified pointwise
full-matrix deficit. With

\[
U=\frac{6602082433275499863}{41641817600000000},\qquad
z_0=\frac{63}{9937},
\]

the majorant is `U` through `z0` and
`-(a+Gamma*z)/(1+z)^2` afterward, using P12's exact final-band
`a=-199437/1250` and
`Gamma=-41528474059081/260261360000`. Its integral `p` has an exact
rational/logarithmic closed form. The radial map

\[
G_\epsilon(M)=p(\lVert M\rVert_F/\epsilon)
\frac{M}{\lVert M\rVert_F},\qquad G_\epsilon(0)=0,
\]

has full-space derivative eigenvalues `d_hat(z)/epsilon` in the radial
direction and `p(z)/(epsilon*z)` in every tangential direction. Since the
latter is the average of the decreasing majorant, both dominate P12's local
deficit. Line-segment integration proves
`E_(h,epsilon)+G_epsilon` globally monotone for every finite rectangular
matrix shape. This is a full-matrix result, not a diagonal certificate.

At `epsilon=1e-7`, two outward-rounded Arb passes and an independent exact
rational-log reconstruction certify

| Raw norm | `||G_epsilon(M)||_F` | P12 constant-repair output |
| ---: | ---: | ---: |
| `1` | `2571.857826470212...` | `1,585,445,308.053868...` |
| P11 exact guard `25.2335060804...` | `3086.959580254301...` | `40,006,343,820.950377...` |

The reduction does not eliminate stiffness. Exactly,

\[
\operatorname{Lip}(G_\epsilon)=\frac{U}{\epsilon},
\]

while P12's finite pair proves that every globally Lipschitz correction that
passivates that pair satisfies

\[
\operatorname{Lip}(C_\epsilon)>
\frac{158.1172496}{\epsilon}.
\]

The universal lower statement is qualified to shapes with `min(m,n)>=2`.
The constructed stiffness is within `0.270231%` of this strict witness lower.

An exact scalar curvature-one quadratic control uses the pinned real-arithmetic
EMA/Nesterov order with `beta=19/20`. The correction alone is locally Schur
stable only below
`1082687257600/63820130188329832009 = 1.69647e-8` at `epsilon=1e-7`;
the complete repaired operator lowers the threshold to approximately
`4.18024e-9`. Thus `eta=1/32000` fails both exact Jury checks. A separate
rational one-step witness expands the scalar objective by a factor strictly
above `23,325,554`. These are explicit instability controls for the old step,
not claims about every initialization or an implicit method.

P13 concerns only the continuous exact-real surrogate. It does not certify
the backend-specific discontinuous BF16 map, propagate the new repair through
P7--P11, or establish neural-network convergence. The human proof audit is
pending and unsigned. See `P13_RADIAL_PASSIVATION_RESULTS.md`,
`radial_passivation_tradeoff_certificate.json`, and
`../../theory/radial_passivation_tradeoff.md`.

## 20. P14 Yosida stability

P14 retains P13's exact-real full-matrix monotone map
`A_epsilon=E_(h,epsilon)+G_epsilon`, for every `epsilon>0`, and changes the
architecture rather than tightening its explicit stiffness estimate. Define

\[
B_{\epsilon,\mu}=A_\epsilon+\mu I,\qquad
J_{\lambda B}=(I+\lambda B)^{-1},\qquad
Y_{\lambda,\mu}=\lambda^{-1}(I-J_{\lambda B}).
\]

The inherited `E_(h,epsilon)` normalizes by
`M/(||M||_F+epsilon)` and applies exactly five Jordan stages with coefficients
`6889/2000`, `-191/40`, and `4063/2000`; `G_epsilon` is P13's locked radial
correction.

Continuity, full domain, and monotonicity of `A_epsilon` make `B` maximal and
`mu`-strongly monotone on every fixed finite Frobenius matrix space. Thus the
resolvent is globally defined and single-valued. For increments
`u=Delta J`, `v=Delta Y`, and `x=Delta input=u+lambda*v`, strong monotonicity
pulls back to the exact graph inequality

\[
\langle v-mx,\,Mx-v\rangle_F\ge0,
\qquad m=\frac{\mu}{1+\lambda\mu},\quad M=\frac1\lambda.
\]

It also gives zero preservation, `lambda`-cocoercivity,
`mu/(1+lambda*mu)`-strong monotonicity, and `1/lambda`-Lipschitzness. These are
incremental nonsymmetric statements, not a Loewner order on `Y`.

At `lambda=1/1000`, `mu=1000`, the sector is exactly `[500,1000]` and

\[
Y=750I+\mathcal E,\qquad \operatorname{Lip}(\mathcal E)\le250,
\]

independently of `epsilon`. An exact rational `4 x 4` value--momentum LMI uses
the pinned EMA/Nesterov ordering, `beta=19/20`, `eta=1/32000`, global
smoothness `L=10`, and global PL constant `1`. Its storage is

\[
P=10^{-6}\begin{bmatrix}674389&-73827\\-73827&12368\end{bmatrix},
\qquad c_F=\frac{313243}{10^6},
\]

with exact nonnegative interpolation and residual multipliers. Sylvester's
criterion verifies `P` positive definite and the complete LMI strictly
negative definite. Exact function-value coefficients cancel, yielding

\[
V_{t+1}\le q_{14}V_t,
\qquad q_{14}=\left(\frac{499}{500}\right)^2
=\frac{249001}{250000}<1,
\qquad D_{14}=0.
\]

In artifact notation, this is exactly `D14=0`; no solve-error term is hidden.

Consequently, the objective gap, gradient, and momentum converge
geometrically for every differentiable globally `10`-smooth objective with
finite infimum satisfying the global PL inequality with constant `1`, even
when the objective is nonconvex and its minimizer is nonunique. Summability of
the exact updates gives convergence to some trajectory-dependent global
minimizer; no unique-minimizer or arbitrary-pair contraction claim is made.

The controls distinguish regularization from relabeling. The direct
`lambda=0` P13 limit fails its exact scalar Jury test. A finite
`lambda=1/100000` resolvent also fails at scalar curvature `10`, whereas the
selected `lambda=1/1000` point passes that exact local control and the stronger
global LMI. Exact lower, skew-circle, and limiting-upper examples show the
Yosida sector constants cannot be uniformly improved over the stated monotone
operator class.

P14 specifies an exact implicit map only. It provides no finite-iteration
resolvent algorithm, tolerance, cost bound, mixed-precision implementation,
or literal BF16/upstream-Muon theorem; those are separate P15 questions. The
human proof audit is pending and unsigned. See
`P14_YOSIDA_STABILITY_RESULTS.md`, `yosida_stability_certificate.json`, and
`../../theory/yosida_stability_certificate.md`.

## 21. P15 inexact-Yosida robustness

P15 retains P14's exact-real base operator for every `epsilon>0` and every
fixed finite real rectangular matrix shape, but replaces its exact resolvent
evaluation by a candidate `u_hat` with computable graph residual

\[
r=s-\widehat u-\lambda B(\widehat u),
\qquad \lambda=\frac1{1000},
\]

and deployed approximate output

\[
\widehat Y(s)=\frac{s-\widehat u}{\lambda}.
\]

The output convention is part of the theorem: away from an exact solve it is
not `B(u_hat)`. Since `u_hat=J(s-r)` and P14 proves `Lip(J)<=1/2`, exact
full-matrix Hilbert-space inequalities give

\[
\|\widehat u-J(s)\|_F\le\frac12\|r\|_F,
\qquad
\|\widehat Y(s)-Y(s)\|_F\le500\|r\|_F.
\]

The factor `500` is sharp over the abstract admissible monotone-base class,
as witnessed by `A=0`, `B=1000I`. It is not an assertion that the locked P13
map equals zero.

P15 locks the a posteriori stopping rule

\[
\|r\|_F\le\frac1{250}\|s\|_F+\bar r
\]

at every oracle call (equivalently on `(s_(t+1),r_(t+1))` in the pinned
recurrence).

Thus the locked relative tolerance is exactly `kappa=1/250`.

The relative solve error contributes radius `2`, so the P14 centered Yosida
radius grows from `250` to `252`; the remaining absolute output port is at
most `500*rbar`. The same P14 storage, exact multipliers, and
`tau=499/500` satisfy a strict exact `4 x 4` LMI at radius `252`. The
port-augmented `5 x 5` LMI has exact physical output-error gain `1/100000`.
For the pinned `beta=19/20`, `eta=1/32000` recurrence and every
differentiable globally `10`-smooth, global-PL-`1` objective with finite
infimum,

\[
\boxed{
V_{t+1}\le\frac{249001}{250000}V_t+\frac52\bar r^2.}
\]

Thus `q15=249001/250000<1` and `C15=5/2`. Unrolling gives

\[
\limsup_tV_t\le\frac{625000}{999}\bar r^2,
\qquad
\limsup_t(f(W_t)-f^\star)
\le\frac{6250000000000}{312929757}\bar r^2
\approx19972.533324787\bar r^2.
\]

When `rbar=0`, the nonzero relative tolerance retains geometric convergence
of objective gap, gradient, and momentum, and the iterates converge to some
trajectory-dependent global minimizer. With persistent `rbar>0`, parameter
convergence is not claimed.

Exact controls distinguish theorem boundaries. Setting `kappa=rbar=0`
recovers the appropriate P14 fractions exactly. Increasing to
`kappa=3/500`, hence radius `253`, fails the unchanged frozen-storage
`4 x 4` certificate; this is only certificate rejection. In the abstract
boundary class, `kappa=1`, `r=-s`, `u_hat=s` yields `Y_hat=0` and can stall
away from stationarity. At `kappa=2`, `r=-2s`, `u_hat=3s/2` yields
`Y_hat=-500s`; on scalar curvature `1`, the pinned characteristic has
`p(1)=-1/1280<0`, a genuine loose-tolerance instability control.

P15 certifies a residual criterion, not a method for meeting it. It supplies
no finite-step solver, complexity bound, rounded residual evaluation,
BF16/FP32 implementation, accelerator or upstream parity, weight decay,
aspect scaling, stochastic-gradient theorem, or neural-network result. The
human proof audit is pending and unsigned. See
`P15_INEXACT_YOSIDA_ROBUSTNESS_RESULTS.md`,
`inexact_yosida_robustness_certificate.json`, and
`../../theory/inexact_yosida_robustness_certificate.md`.

## Reproduce

```bash
uv run --locked python scripts/find_counterexample.py \
  --output results/summaries/canonical_witness.json
uv run --locked python scripts/record_bf16_witness.py \
  --output results/summaries/bf16_witness.json
uv run --locked python scripts/certify_floored_repair.py \
  --output results/summaries/floored_repair_certificate.json
uv run --locked python scripts/certify_momentum_stability.py \
  --output results/summaries/momentum_iqc_certificate.json
uv run --locked python scripts/certify_ema_nesterov_stability.py \
  --output results/summaries/ema_nesterov_iqc_certificate.json
uv run --locked python scripts/certify_structure_aware_stability.py \
  --output results/summaries/structure_aware_stability_certificate.json
uv run --locked python scripts/certify_nonquadratic_stability.py \
  --output results/summaries/nonquadratic_stability_certificate.json
uv run --locked python scripts/certify_nonquadratic_convergence.py \
  --output results/summaries/nonquadratic_convergence_certificate.json
uv run --locked python scripts/reconstruct_nonquadratic_convergence.py
uv run --locked python scripts/certify_pl_convergence.py \
  --output results/summaries/pl_convergence_certificate.json
uv run --locked python scripts/reconstruct_pl_convergence.py
uv run --locked python scripts/certify_robust_dissipativity.py \
  --output results/summaries/robust_dissipativity_certificate.json
uv run --locked python scripts/reconstruct_robust_dissipativity.py
uv run --locked python scripts/certify_mixed_precision.py \
  --output results/summaries/mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_mixed_precision.py
uv run --locked python scripts/certify_scalable_mixed_precision.py \
  --output results/summaries/scalable_mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_scalable_mixed_precision.py \
  --require-canonical
uv run --locked python scripts/certify_outer_loop_roundoff.py \
  --output results/summaries/outer_loop_roundoff_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_roundoff.py \
  --require-canonical
uv run --locked python scripts/certify_additive_epsilon_deficit.py \
  --output results/summaries/additive_epsilon_deficit_certificate.json
uv run --locked python scripts/reconstruct_additive_epsilon_deficit.py \
  --require-canonical
uv run --locked python scripts/certify_radial_passivation_tradeoff.py \
  --output results/summaries/radial_passivation_tradeoff_certificate.json
uv run --locked python scripts/reconstruct_radial_passivation_tradeoff.py \
  --require-canonical
uv run --locked python scripts/certify_yosida_stability.py \
  --output results/summaries/yosida_stability_certificate.json
uv run --locked python scripts/reconstruct_yosida_stability.py \
  --require-canonical
uv run --locked python scripts/certify_inexact_yosida_robustness.py \
  --output results/summaries/inexact_yosida_robustness_certificate.json
uv run --locked python scripts/reconstruct_inexact_yosida_robustness.py \
  --require-canonical
uv run --locked python experiments/matrices/run_deficit_audit.py
uv run --locked python experiments/quadratics/run_lr_sweep.py
uv run --locked python experiments/quadratics/run_horizon_check.py
uv run --locked python experiments/quadratics/run_nonquadratic_falsification.py \
  --output results/summaries/nonquadratic_falsification.json
uv run --locked python \
  experiments/quadratics/run_nonquadratic_convergence_falsification.py \
  --output results/summaries/nonquadratic_convergence_falsification.json
uv run --locked python experiments/nonconvex/run_pl_falsification.py \
  --output results/summaries/pl_falsification.json
uv run --locked python \
  experiments/nonconvex/run_robust_dissipativity_falsification.py \
  --output results/summaries/robust_dissipativity_falsification.json
uv run --locked python \
  experiments/mixed_precision/run_mixed_precision_falsification.py \
  --output results/summaries/mixed_precision_falsification.json
uv run --locked python \
  experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py \
  --output results/summaries/scalable_mixed_precision_diagnostic.json
uv run --locked python \
  experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py \
  --output results/summaries/finite_precision_outer_loop_diagnostic.json
uv run --locked python scripts/make_figures.py
uv run --locked pytest -q
```
