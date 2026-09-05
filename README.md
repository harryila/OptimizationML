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

Branch `p14-yosida-stability` pivots from direct evaluation of that stiff
operator to an exact implicit construction. Write
`A_epsilon=E_(h,epsilon)+G_epsilon`, add the shunt
`B_(epsilon,mu)=A_epsilon+mu*I`, and define

\[
J_{\lambda B}=(I+\lambda B)^{-1},\qquad
Y_{\lambda,\mu}=\lambda^{-1}(I-J_{\lambda B}).
\]

Here `E_(h,epsilon)` retains the normalization
`M/(||M||_F+epsilon)`, five Jordan stages, and exact stage coefficients
`6889/2000`, `-191/40`, and `4063/2000`; `G_epsilon` is exactly P13's radial
repair.

For every `epsilon>0` and every fixed finite rectangular matrix space, the
P13 map is continuous, full-domain, monotone, and zero-preserving. Hence the
resolvent exists uniquely. At the exact choice `lambda=1/1000`, `mu=1000`,
its Yosida approximation is zero-preserving, `1/1000`-cocoercive,
`500`-strongly monotone, and `1000`-Lipschitz. The authoritative nonsymmetric
statement is the incremental sector IQC

\[
\langle\Delta Y-500\Delta X,\,1000\Delta X-\Delta Y\rangle_F\ge0,
\]

not a Loewner ordering. Equivalently, `Y=750*I+E` with `Lip(E)<=250`.

An exact dimension-independent value--momentum LMI then certifies the pinned
EMA/Nesterov loop at `beta=19/20`, the unchanged `eta=1/32000`, and every
differentiable globally `10`-smooth objective with finite infimum satisfying
the global PL inequality with constant `1`:

\[
V_{t+1}\le \frac{249001}{250000}V_t.
\]

Thus P14 has `D14=0` for the exact resolvent. Exact scalar controls show that
the direct P13 limit and a finite under-regularized resolvent fail, while the
selected point passes. This is an exact-real implicit optimizer theorem; it
does not provide a resolvent algorithm, finite solve tolerance, BF16 kernel,
or literal upstream-Muon guarantee. See
[`theory/yosida_stability_certificate.md`](theory/yosida_stability_certificate.md)
and
[`results/summaries/P14_YOSIDA_STABILITY_RESULTS.md`](results/summaries/P14_YOSIDA_STABILITY_RESULTS.md).

Branch `p15-inexact-yosida-robustness` replaces the exact P14 solve by an
exact-real residual oracle. For a returned candidate `u_hat`, define

\[
r=s-\widehat u-\lambda B(\widehat u),
\qquad
\widehat Y(s)=\frac{s-\widehat u}{\lambda}.
\]

The output convention matters: away from an exact solve, `Y_hat` is not
`B(u_hat)`. Since `u_hat=J(s-r)` and `Lip(J)<=1/2`, P15 proves

\[
\|\widehat u-J(s)\|_F\le\frac12\|r\|_F,
\qquad
\|\widehat Y(s)-Y(s)\|_F\le500\|r\|_F.
\]

Under the computable stopping rule

\[
\|r\|_F\le\frac1{250}\|s\|_F+\bar r,
\]

that is, `kappa=1/250`,

the relative error enlarges P14's centered radius only from `250` to `252`.
An exact dimension-independent `5 x 5` value--momentum LMI retains
`beta=19/20`, `eta=1/32000`, and the full globally `10`-smooth PL-`1` class:

\[
V_{t+1}\le\frac{249001}{250000}V_t+\frac52\bar r^2.
\]

Thus zero absolute tolerance retains P14's geometric convergence even with
the locked nonzero relative tolerance. For persistent `rbar`, the exact
objective neighborhood is
`limsup(f-f*) <= (6250000000000/312929757)*rbar^2`; iterate convergence is
not claimed. P15 certifies a stopping criterion, not a solver, complexity
bound, rounded residual evaluation, or BF16/upstream implementation. See
[`theory/inexact_yosida_robustness_certificate.md`](theory/inexact_yosida_robustness_certificate.md)
and
[`results/summaries/P15_INEXACT_YOSIDA_ROBUSTNESS_RESULTS.md`](results/summaries/P15_INEXACT_YOSIDA_ROBUSTNESS_RESULTS.md).

Branch `p16-equivariant-resolvent-solver` supplies a structured exact-real
algorithm and a guarded FP64 reference implementation for that P15 oracle.
For arbitrary fixed finite real rectangular matrix shapes, the P14 resolvent
is bi-orthogonally equivariant and preserves singular subspaces, including
repeated and zero singular values. Its graph equation reduces to a coupled
singular-value system whose Jacobian is diagonal plus rank one, so each Newton
step uses Sherman--Morrison algebra after one SVD rather than a dense solve in
the number of singular values. The dense reference path uses a second SVD to
reevaluate `B` on the stored reconstructed candidate for its literal P15
graph-residual postcheck. Exact strong-monotonicity and Armijo arguments
give global exact-real convergence and finite termination at every positive
P15 residual threshold. This is not a useful uniform a priori iteration-count
bound or a certificate for FP64 rounding.

At the locked `epsilon=1e-7`, five Jordan stages with coefficients
`6889/2000`, `-191/40`, and `4063/2000`, `lambda=1/1000`, `mu=1000`, and P15
relative tolerance `1/250`, all 13 declared computed-residual cases pass. The
polynomial and epsilon placement are traced to KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; upstream contains neither the
repair nor this solver. The FP64 study reports at most 8 Newton iterations, no
accepted-case backtracking, and a worst attained graph residual of about
`5.880e-14`; the four large Transformer cases solve only the complete reduced
spectra and are not dense runtime benchmarks.

P16 does **not** pass its overall acceptance gate. An exact/Arb `2 x 2`
enclosure proves unequal modal gains, but on the locked `diag(3,4)` comparator
the best-scalar departure is only about `5.879e-6`, versus about `6.997e-2`
for the five-step Jordan comparator, for retention about `8.403e-5`. Thus the
operator is algebraically noncollapsed but effectively scalar at the declared
meaningful-fidelity threshold. A frozen sampled six-point
`(lambda,mu)` frontier finds no point that jointly passes the frozen P14
certificate and the fidelity gates; this diagnostic is not a global
impossibility theorem. P16 is therefore retained as an informative reference
solver and fidelity obstruction, not a successful Muon-fidelity result. See
[`theory/equivariant_resolvent_solver.md`](theory/equivariant_resolvent_solver.md)
and
[`results/summaries/P16_EQUIVARIANT_RESOLVENT_SOLVER_RESULTS.md`](results/summaries/P16_EQUIVARIANT_RESOLVENT_SOLVER_RESULTS.md).

Branch `p17-shape-preserving-resolvent` resolves that fidelity failure by
keeping the P16 resolvent internally while exposing a smooth blend of its
passive Yosida output and the unshifted five-stage Jordan response. With
`q=||S||_F^2`, the exact C2 quintic gate is zero for `q<=1/4` and reaches a
locked ceiling for `q>=1`. The passive branch avoids exposing the raw
negative slope on the certified small-signal band, while the Jordan branch
restores spectral shaping away from the origin. That placement is a
local/incremental safeguard; the global one-trajectory PL proof itself uses
only `0<=theta<=bar_theta`.

An analytic singular-mode argument gives a dimension-uniform **pointwise**
sector on every finite rectangular shape. Exact `4 x 4` value--momentum LMIs
then prove global one-trajectory convergence for every differentiable
globally `10`-smooth, global-PL-`1` objective at two locked points:

- the primary high-fidelity design uses gate ceiling `3/4`, passive divisor
  `4096`, and `eta=1/128000`, with
  `tau^2=281474943156225/281474976710656<1`;
- the secondary full-step design uses ceiling `1/8`, divisor `8192`, and
  `eta=1/32000`, with
  `tau^2=281474741829681/281474976710656<1`.

Both rigorously pass the unchanged P16 fidelity thresholds on the canonical
`diag(3,4)` input after P15 residual inflation. The theorem uses the exact
resolvent in exact real arithmetic: it is not an incremental-sector theorem
and does not propagate P16 solver error or FP64/BF16 rounding. See
[`theory/shape_preserving_resolvent.md`](theory/shape_preserving_resolvent.md)
and
[`results/summaries/P17_SHAPE_PRESERVING_RESOLVENT_RESULTS.md`](results/summaries/P17_SHAPE_PRESERVING_RESOLVENT_RESULTS.md).

Branch `p18-sector-projected-useful-rate` controls P17's large pointwise
sector without erasing its spectral direction. It projects
`X(S)=E_(h,epsilon)(J(S))` along its own ray into the pointwise disk sector
`[0,1]`, then blends it with `Y(S)/1024` under the same ceiling-`3/4` gate.
For every finite real matrix shape, the exposed exact-real map has the global
origin-centred pointwise sector

\[
\left[\frac{125}{1024},\frac{509}{512}\right].
\]

The initially proposed `eta=1/50` is impossible for a theorem using only
this sector: an admissible complex-skew boundary map violates Schur--Cohn
exactly. The mandated rational frontier nevertheless yields an acceptance-
passing point at `eta=1/83`, with
`q=999598040401/1000000000000`. Its certified Lyapunov-rate half-life is
about `1724.07`, or `9.959x` P14's, and the exact comparison
`q^10<249001/250000` closes the predeclared rate gate. A smaller-step
`eta=1/120` point certifies a faster `666.31`-iteration Lyapunov-rate
half-life.

Outward-rounded canonical checks retain about `51.86%` of upstream shaping,
and all 2,176 informative points in the sampled P17 operating annulus pass
the unchanged fidelity gates. Only 192/327 informative broad-grid cases pass,
so fidelity remains an operating-domain diagnostic, not a global theorem.
The exact amplitude and effective-step guards prevent scale-invariant scores
from being obtained by collapsing or exploding the output. P18 still assumes
the exact resolvent and exact real arithmetic. See
[`theory/sector_projected_useful_rate.md`](theory/sector_projected_useful_rate.md)
and
[`results/summaries/P18_SECTOR_PROJECTED_USEFUL_RATE_RESULTS.md`](results/summaries/P18_SECTOR_PROJECTED_USEFUL_RATE_RESULTS.md).

Branch `p19-sector-shielded-inexact-resolvent` removes solver accuracy as a
premise for P18's stability theorem. For any finite approximate candidate
`C(S)`, it returns the metric projection onto

\[
\mathcal D_S=\left\{U:
\left\|U-\frac{1143}{2048}S\right\|_F
\le\frac{893}{2048}\|S\|_F\right\}.
\]

This ball is exactly the P18 pointwise sector
`[125/1024,509/512]`, including the singleton `{0}` at `S=0`. Every finite
candidate is therefore safe after shielding, while the exact P18 output is
fixed because it already lies in the disk. For each fixed signal, projection
is nonexpansive in its candidate and cannot increase an independently
established error to P18. This is not joint nonexpansiveness in signal and
candidate, and the P15 graph residual alone still does not bound the final
nonlinear P18 candidate error.

The exact P18 smooth-PL certificates replay unchanged for arbitrary
time-varying finite candidates after shielding: `eta=1/83` has
`q=999598040401/1000000000000`, while `eta=1/120` has the faster certified
rate `q=624350169/625000000`. Their certified Lyapunov-rate half-lives are
approximately `1724.0734` and `666.314` iterations. These are rates of the
geometric storage bounds, not promises about an observed objective halving
at a fixed iteration.

The locked binary64 reference uses scaled/balanced norm evaluation, projects
to an inward margin, and exactly checks the stored dyadic output against the
original P18 disk. Every successful return is thus certified. Nonfinite
candidates use an exact-checked interior fallback; nonfinite signals and
unrepresentable subnormal cases fail closed. On all 2,688 declared annulus
calls the shield is inactive and bitwise identical, preserving all 2,176
informative fidelity passes. This is a correctness-first CPU reference, not
yet a scalable BF16/GPU theorem. See
[`theory/sector_shielded_inexact_resolvent.md`](theory/sector_shielded_inexact_resolvent.md)
and
[`results/summaries/P19_SECTOR_SHIELDED_INEXACT_RESOLVENT_RESULTS.md`](results/summaries/P19_SECTOR_SHIELDED_INEXACT_RESOLVENT_RESULTS.md).

Branch `p20-scalable-mixed-precision-sector-shield` replaces P19's runtime
exact-rational postcheck with a statically certified, shape-locked arithmetic
graph. For stored FP32 or exactly widened BF16 signal/candidate tensors, the
CPU proof reference uses round-to-nearest-even FP32 vector arithmetic,
maximum-scaled fixed balanced reductions, an exact FP64 product of two FP32
norm factors, and directed-inward scalar comparisons. It returns an inward
candidate unchanged, contracts a rejected finite candidate along its computed
displacement ray, and reserves `fl32(S/2)` for exceptional arithmetic. Every
successful stored FP32 return satisfies the same full-matrix pointwise sector
`[125/1024,509/512]` without a runtime big-integer check.

The exact shape recurrence certifies all seven P9 Transformer shapes. The
tight `4096 x 14336` overall margin is exactly
`71710053325847/1152921504606846976` (about `6.220e-5`); its clip-only margin
is `53997772719001/576460752303423488` (about `9.367e-5`). Provided every
shield call along the trajectory succeeds and stored `S` is identified with
the optimizer signal at the abstract port, the P19 smooth-PL rates replay
conditionally at `eta=1/83`, `q=999598040401/1000000000000`, and at
`eta=1/120`, `q=624350169/625000000`. P20 does not yet compose BF16/FP32
signal casts or outer momentum, parameter, master-weight, aspect-scaling,
weight-decay, or distributed arithmetic into that interconnection.

The frozen study leaves `2671/2688` P18 annulus candidates bitwise unchanged,
radially clips the remaining `17`, uses no half fallback there, and retains all
`2176/2176` informative fidelity passes. All `14/14` declared operating
Transformer spectra are inactive; seven deliberately flat boundary stresses
clip. The pinned upstream candidate passes through on `761/2688` annulus
cases and clips on `1927`, which demonstrates a safeguarded candidate rather
than certifying unmodified upstream Muon. P20 is an accept/clip/fallback map,
not P19's metric projection: fixed-input nonexpansiveness and global identity
on exact P18 are not claimed. FTZ/DAZ, native GPU reductions, a BF16 output,
and all-subnormal signals without exactly representable halving remain outside
the positive result. See
[`theory/scalable_mixed_precision_sector_shield.md`](theory/scalable_mixed_precision_sector_shield.md)
and
[`results/summaries/P20_SCALABLE_MIXED_PRECISION_SECTOR_SHIELD_RESULTS.md`](results/summaries/P20_SCALABLE_MIXED_PRECISION_SECTOR_SHIELD_RESULTS.md).

Branch `p21-certified-outer-loop-composition` closes that stored-signal
boundary around a concrete CPU proof-reference optimizer step. The actual
order is stored BF16/FP32 gradient, FP32 EMA/Nesterov with reused `bg`, pinned
five-stage BF16 Muon candidate, aspect scaling, P20 shield, and compensated
three-word FP32 master update. The exact `7 x 7` LMI is written at the stored
FP32 Nesterov signal; it never assumes incremental Lipschitzness of P20 or
compares shield calls at rounded and ideal signals.

For differentiable globally `10`-smooth, global-PL-`1` objectives, its exact
pathwise port inequality has gains `(16384,1024,512)` at the primary
`eta=1/120`, `q=624350169/625000000`, and `(32768,2048,512)` at the secondary
`eta=1/83`, `q=999598040401/1000000000000`. Zero ports recover P18/P19 entry
for entry. Exact FP32 roundoff envelopes close all seven P20 shapes on
`V<=1` under the explicit source premise
`||zeta||_F <= (sqrt(V)+1)/4096`. On the tight `4096 x 14336` shape, the
outward objective-neighborhood bounds are about `4.993e-4` and `2.583e-3` at
the two rates. These are invariant-domain finite-precision bounds, not global
neural-training claims.

P21 also converts P20's unrepresentable all-subnormal case into a rigorously
bounded absolute update port by returning zero relative to the conceptual
sector point `S/2`. The main smooth-PL result sets weight decay to zero;
nonzero decay is a separately bounded conditional port, with a centered
strong-convexity corollary. A frozen shadow-mode protocol predeclares the real-
gradient intervention gates, but no real trace was run: this checkout lacks a
pinned trainer, data/tokenizer/checkpoint, accelerator, and a P20 certificate
for vanilla GPT-2's fused `768 x 2304` QKV shape. See
[`theory/certified_outer_loop_composition.md`](theory/certified_outer_loop_composition.md),
[`theory/p21_shadow_trace_protocol.md`](theory/p21_shadow_trace_protocol.md),
and
[`results/summaries/P21_CERTIFIED_OUTER_LOOP_COMPOSITION_RESULTS.md`](results/summaries/P21_CERTIFIED_OUTER_LOOP_COMPOSITION_RESULTS.md).

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
uv run --locked python scripts/certify_yosida_stability.py
uv run --locked python scripts/reconstruct_yosida_stability.py \
  --require-canonical
uv run --locked python scripts/certify_inexact_yosida_robustness.py
uv run --locked python scripts/reconstruct_inexact_yosida_robustness.py \
  --require-canonical
uv run --locked python scripts/certify_equivariant_resolvent_solver.py
uv run --locked python scripts/reconstruct_equivariant_resolvent_solver.py \
  --require-canonical
uv run --locked python scripts/certify_shape_preserving_resolvent.py
uv run --locked python scripts/reconstruct_shape_preserving_resolvent.py \
  --require-canonical
uv run --locked python scripts/certify_sector_projected_useful_rate.py
uv run --locked python scripts/reconstruct_sector_projected_useful_rate.py \
  --require-canonical
uv run --locked python scripts/certify_sector_shielded_inexact_resolvent.py
uv run --locked python scripts/reconstruct_sector_shielded_inexact_resolvent.py \
  --require-canonical
uv run --locked python scripts/certify_scalable_sector_shield.py
uv run --locked python scripts/reconstruct_scalable_sector_shield.py \
  --require-canonical
uv run --locked python scripts/certify_outer_loop_composition.py \
  --output results/summaries/certified_outer_loop_composition_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_composition.py \
  --canonical results/summaries/certified_outer_loop_composition_certificate.json \
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
uv run --locked python experiments/resolvent/run_p16_solver_study.py
uv run --locked python experiments/resolvent/run_p17_shape_preserving_study.py
uv run --locked python experiments/resolvent/run_p18_sector_projected_study.py
uv run --locked python experiments/resolvent/run_p19_sector_shielded_study.py
uv run --locked python \
  experiments/mixed_precision/run_p20_scalable_sector_shield_study.py
uv run --locked python experiments/training/run_p21_synthetic_shadow_trace.py \
  --output results/summaries/p21_synthetic_shadow_trace.json
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

This repository stays focused on twenty-four technical goals (P20 completed
the preceding twenty-three-goal ledger):

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
16. an epsilon-independent Yosida sector theorem and exact full-step
    smooth-PL certificate for the pinned implicit EMA/Nesterov loop;
17. an exact inexact-resolvent residual-to-output theorem and robust
    smooth-PL ultimate-bound certificate at the full pinned step;
18. an equivariant singular-value resolvent reduction, guarded reference
    solver, and an honestly failed meaningful-Muon-fidelity gate;
19. a C2-gated shape-preserving resolvent interface with a global pointwise
    sector, two exact smooth-PL stability points, and a recovered canonical
    meaningful-fidelity pass;
20. a ray-sector-projected interface with a tight global pointwise sector,
    an exact useful-rate smooth-PL certificate, nonvanishing amplitude and
    effective-step guards, and a rational step--rate frontier;
21. an exact final sector shield that preserves P18 on exact candidates,
    makes arbitrary finite approximate candidates pointwise safe, replays
    both P18 rates, and exact-checks every successful binary64 reference
    output;
22. a statically certified, shape-locked BF16/FP32-to-FP32 CPU shield with
    balanced reductions, positive inward margins on seven Transformer shapes,
    radial clipping, fail-closed near-zero guards, and no runtime big-integer
    postcheck;
23. an exact stored-signal `7 x 7` port certificate composed with FP32
    EMA/Nesterov, the aspect-scaled BF16 candidate, P20, compensated
    three-word parameters, all-subnormal and represented-model ports, and
    seven-shape invariant-domain roundoff bounds;
24. qualified matrix, quadratic, nonlinear, precision, and predeclared
    shadow-mode diagnostics.

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
stiffness is unavoidable; it does not inherit P7--P11 stability. P14 removes
that stiffness from the outer-loop interface through an exact resolvent and
recovers a full-step smooth-PL theorem. P15 retains that rate for a verifiable
relative graph-residual tolerance and gives an explicit neighborhood for an
absolute residual floor. P16 supplies an exact-real globally convergent
structured solver and a checked FP64 reference implementation, but its locked
operator fails the meaningful-fidelity gate and no sampled frontier point
jointly passes that gate and the frozen P14 certificate. P17 changes the
exposed interface, proves global exact-real one-trajectory smooth-PL
convergence for a primary `3/4` high-fidelity design at one-quarter step and a
secondary `1/8` full-step design, and restores the canonical fidelity pass.
Its pointwise sector is not incremental, and its theorem does not inherit
P15's inexact-solver robustness. P18 projects the shape branch into `[0,1]`,
reduces the global condition ratio to `1018/125`, and certifies the selected
`eta=1/83` point within ten times P14's certified Lyapunov-rate half-life.
The initial `eta=1/50` sector target has an exact generic complex-skew
obstruction; this does not prove instability of the more structured P18
operator. P19 projects every finite candidate back into P18's pointwise disk,
so both P18 rates survive without solver accuracy as a stability premise.
Its fixed-signal nonexpansiveness preserves, but does not itself derive, a
candidate-fidelity bound from the P15 graph residual. The locked binary64
reference exact-checks every successful stored output and explicitly fails
closed on unrepresentable subnormal cases. P20 replaces that runtime rational
check with a static proof for one locked CPU FP32 graph, optionally accepting
exactly widened BF16 inputs. Its pass-through/radial-clip/half-fallback map is
pointwise safe on seven frozen shapes but is neither P19's metric projection
nor a theorem for arbitrary GPU, FTZ, reduction, or compiler semantics. The
P19 rates carry over only along trajectories whose shield calls all succeed
and after identifying stored `S` with the abstract operator-port signal;
P21 removes that port-identification qualification by writing its LMI directly
at the actual stored signal and composing the proof-reference FP32
EMA/Nesterov and compensated master. Its pathwise affine source budget and
nonzero-decay bounds remain explicit premises; they are not derived from an
arbitrary training run. The predeclared shadow observer exists, but the real
gradient trace is externally blocked. Production-gradient evidence, native
accelerator/distributed parity, a complete supported training shape inventory,
generic stochastic-gradient guarantees, throughput, and formal circuit ports
remain open.

## Layout

- `src/passive_muon/`: normalization, orthogonalizers, deficit metrics, repairs;
- `theory/`: claim ledger, analytic proof, and exact/numerical certificates;
- `experiments/`: matrix, qualified quadratic, nonlinear, and resolvent studies;
- `scripts/`: reproducible entry points;
- `results/`: committed manifests, summaries, and figures only;
- `third_party/`: upstream revisions and license notices.

Numerical audits explicitly use `float64`; public orthogonalizers preserve the
caller's dtype. Low-precision training kernels remain separate so that
implementation convenience cannot silently change a theorem.
