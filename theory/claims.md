# Claim ledger

This file separates established statements from open research claims. The
ambient inner product is Frobenius, `<A, B> = tr(A^T B)`, and monotonicity means

\[
\langle T(A)-T(B), A-B\rangle \ge 0
\]

for every pair in the stated domain. This is the incremental/equilibrium-
relative inequality that appears in Boyd's circuit energy-dissipation
argument. It is stronger than ordinary origin-based static passivity
`<T(A), A> >= 0`. Use **incremental passivity (monotonicity)** in notes and
results until the port convention is fully fixed.

## C1. Normalization-induced local nonmonotonicity — proved

Let `H_h` be the matrix spectral map induced by an odd polynomial `h`. For
`X=M/||M||_F`, let `A_X = D H_h(X)` and let
`Q_X[Z]=Z-<X,Z>_F X`. The exact full-matrix derivative is

\[
D T(M)=\frac{1}{\lVert M\rVert_F}A_XQ_X,
\qquad
\operatorname{Sym}DT(M)=
\frac{A_XQ_X+Q_XA_X}{2\lVert M\rVert_F}.
\]

The polynomial map is a gradient, so `A_X` is self-adjoint even at repeated or
zero singular values. If `s_i` are the active singular values of `X`, then
`A_X X` has singular coefficients `s_i h'(s_i)`. Unequal active derivatives
therefore make `Q_X A_X X != 0`, while the quadratic form of the symmetric
Jacobian vanishes in radial direction `X`. The symmetric Jacobian is
indefinite.

The positive-diagonal restriction provides the simplest explicit form. With
`x` denoting the diagonal,

\[
F_h(x)_i = h\!\left(\frac{x_i}{\lVert x\rVert_2}\right).
\]

At a nonzero point, set `u = x / ||x||_2`,
`D = diag(h'(u_i))`, and `P = I - uu^T`. Then

\[
J F_h(x)=\frac{1}{\lVert x\rVert_2}DP,\qquad
\operatorname{Sym} JF_h(x)
=\frac{DP+PD}{2\lVert x\rVert_2}.
\]

If two active coordinates have different values of `h'`, then `D u` is not
parallel to `u`; the symmetric Jacobian is indefinite. In two dimensions its
determinant is exactly

\[
-\frac{u_1^2u_2^2}{4\lVert x\rVert_2^2}
\bigl(h'(u_1)-h'(u_2)\bigr)^2 < 0.
\]

A failure on the diagonal subspace is a failure of the full matrix operator.
For `min(m,n) >= 3`, every non-affine odd polynomial admits a unit singular
spectrum with unequal active derivatives and is therefore nonmonotone after
exact current normalization. The universal statement is false without the
dimension qualifier: rank-one inputs, equal-singular-value inputs, and an
exceptional class of rank-two polynomials do not exhibit the mismatch. The
sharp pointwise theorem is always the active-derivative condition.

## C2. Exact Jordan witness — proved and executable

Let

\[
q(s)=3.4445s-4.7750s^3+2.0315s^5,
\qquad h=q^{\circ 5}.
\]

At `u = (3/5, 4/5)`, exact rational evaluation gives unequal **positive**
values `h'(3/5)` and `h'(4/5)`. At `M_0=diag(3,4)` with the fixed-scale map
`M -> H_h(M/5)`, exact divided differences additionally give

\[
\frac{h(3/5)-h(4/5)}{3/5-4/5}>0,
\qquad
\frac{h(3/5)+h(4/5)}{3/5+4/5}>0.
\]

After the positive chain-rule factor `1/5`, these two off-diagonal modes and
the two diagonal derivative modes are all positive. Thus the full `2 x 2`
fixed-scale Jacobian is positive definite at `M_0`; continuity gives
monotonicity on some sufficiently small neighborhood. This is not global
fixed-scale monotonicity. Current Frobenius normalization makes the symmetric
Jacobian indefinite at the same point, isolating the radial/tangential
normalization coupling rather than merely reusing the known scalar
nonmonotonicity of the Jordan polynomial.

The canonical finite pair is
`A=diag(3,4)`, `B=diag(5/2,3)`. The command proves its normalized gap negative
exactly by a rational-plus-square-root comparison, proves the fixed-scale-5
control positive exactly, and reports 100-digit numerical values. Run:

```bash
uv run --locked python scripts/find_counterexample.py
```

## C3. Minimal constant linear repair — proved with a domain qualifier

For any domain `D`, the repair-aligned primary definition is

\[
\delta_{\mathrm{pair},D}(T)=
\sup_{A\ne B\in D}
\left[-\frac{\langle T(A)-T(B),A-B\rangle_F}
{\lVert A-B\rVert_F^2}\right]_+.
\]

Then `T_rho(M)=T(M)+rho M` is monotone on `D` exactly when
`rho >= delta_pair,D(T)`. For a continuously differentiable map on an open
convex domain, it equals the Jacobian form

\[
\delta_D(T)=\sup_{M\in D}
\max\{0,-\lambda_{\min}(\operatorname{Sym}JT(M))\}.
\]

When this supremum is finite, no smaller constant works. On an annulus,
trajectory, or other nonconvex set, the pointwise Jacobian maximum is only a
local diagnostic; the pairwise definition remains authoritative.

The same identity gives exact pairwise and sampled-set repairs, but those are
lower-bound diagnostics, not global upper certificates.

## C4. Unrestricted global repair for exact normalization — false

For exact current normalization, `T(alpha M) = T(M)` for `alpha > 0`. If one
pair `(A, B)` has negative monotonicity product, then

\[
-\frac{\langle T(\alpha A)-T(\alpha B),
\alpha A-\alpha B\rangle}{\lVert\alpha A-\alpha B\rVert_F^2}
=\frac{1}{\alpha}
\left(
-\frac{\langle T(A)-T(B),A-B\rangle}{\lVert A-B\rVert_F^2}
\right).
\]

The unrestricted deficit is therefore infinite as `alpha -> 0`. No finite
constant linear conductance repairs the exact-normalized map on all nonzero
inputs.

This does **not** kill the project, but it forces the practical theorem to use
one of the following explicit objects:

- the actual epsilon-regularized implementation;
- a fixed normalization floor;
- a certified operating domain;
- or a local/pairwise deficit, labeled as such.

For `M / (||M||_F + eps)`, rescaling `M = eps Z` shows that the global deficit,
if certified, scales as `1 / eps`. A mathematically valid bound at the deployed
`eps = 1e-7` may therefore be too large to be a useful constant repair. This is
the main open design question after the theorem gate.

The deployed Keller Jordan code also casts to `bfloat16` before normalization.
The Jacobian theorem is therefore about the corresponding real-arithmetic
surrogate. The quantized map is discontinuous and must be assessed pairwise;
no autograd Jacobian claim should be labeled as a theorem about the exact BF16
implementation.

## C5. Floored-normalizer full-matrix repair — proved with a tight bracket

For `c>0`, define

\[
N_c(M)=\frac{M}{\max\{c,\lVert M\rVert_F\}},
\qquad F_{h,c}=\mathcal H_h\circ N_c,
\]

where `h` is the five-step Jordan composition with exact coefficients
`6889/2000`, `-191/40`, and `4063/2000`.  This is a max floor, not additive
epsilon normalization.

The map `N_c` is projection of `M/c` onto the Frobenius unit ball and is
`1/c`-Lipschitz.  A full rectangular singular-vector tangent decomposition,
including diagonal, off-diagonal, repeated/zero-singular-value, and
rectangular null-side modes, reduces the derivative of `H_h` to secant
averages of `h'`.  A 160/224-bit outward-rounded Arb certificate proves

\[
-159.5496<h'(s)<484.8763\qquad(0\le s\le1).
\]

For any self-adjoint `A` in this slope interval and any orthogonal projection
`Q`, the sharp dimension-free anticommutator bound gives

\[
\lambda_{\min}\!\left(\frac{AQ+QA}{2}\right)
\ge -\frac{(b-a)^2}{8(a+b)}.
\]

Pairwise line integration handles the nondifferentiable switching sphere; a
Clarke-hull calculation gives the same boundary bound.  Therefore, for every
fixed finite matrix shape,

\[
159.549525785<\delta(F_{h,1})
\le\bar\delta_1
=\frac{41528474059081}{260261360000}
=159.564501081070\ldots.
\]

An exact rational finite-pair witness embedded in a rank-one interior mode
proves the displayed strict lower endpoint.  The bracket width is `0.009386%`
relative to that safe lower threshold.  Exact scaling gives

\[
\delta(F_{h,c})=\frac{\delta(F_{h,1})}{c},
\]

so `F_h,c(M)+(bar_delta_1/c)M` is globally monotone.  The upper certificate is
dimension-uniform and near-minimal relative to the exact witness; it is not
claimed to equal the minimal conductance for each fixed matrix shape.  See
`floored_normalizer_certificate.md` and the locked result manifest.

## C6. Simplified continuous-time stability — proved, training claim open

For any `mu>0`, adding

\[
\rho=\frac{\bar\delta_1}{c}+\mu
\]

makes the floored real-arithmetic map `mu`-strongly monotone.  Consequently the
explicitly specified system

\[
\dot M=-\bigl(F_{h,c}(M)+\rho M-b\bigr)
\]

has a unique equilibrium and contracts in Frobenius norm at rate `mu`.
Global Lipschitzness gives well-posed trajectories, and the pairwise strong-
monotonicity inequality gives the energy estimate directly.  This is a
simplified continuous-time integrator-feedback result, not momentum-Muon or
discrete training stability.

## C7. Stylized non-Nesterov quadratic momentum stability — proved, conservative

Let `R=F_h,c+rho*I` with

\[
\rho=\frac{\bar\delta_1}{c}+\mu,\qquad \mu>0.
\]

The floored-repair certificate makes `R` globally `mu`-strongly monotone and

\[
K=\rho+\frac{484.8763}{c}
\]

is a rigorous global Lipschitz bound. For the deterministic quadratic

\[
f(W)=\tfrac12\langle W-W_\star,H(W-W_\star)\rangle,
\qquad \ell I\preceq H\preceq LI,
\]

consider the stylized real-arithmetic, non-Nesterov recurrence

\[
m_{t+1}=\beta m_t+H(W_t-W_\star),\qquad
W_{t+1}=W_t-\eta R(m_{t+1}).
\]

Set `nu=mu*ell/(K*L)` and `alpha=eta*K*L`. For `0 <= beta < 1`, the loop is
globally incrementally exponentially stable whenever

\[
0<\alpha<
\frac{2(1-\beta)^2(1+\beta)\nu}
{(1+\beta)^2-4\beta\nu^2}.
\]

The proof uses separate strong-monotonicity and Lipschitz IQCs and an explicit
`3 x 3` strict LMI factorization. This generic one-step-memory IQC machinery is
prior art; see Lessard--Recht--Packard and Zhang--Bao--Lessard--Grosse. The
repository's contribution is the certified full-matrix floored-Muon sector and
its repaired interconnection. A machine-readable representative at
`c=1`, `mu=bar_delta_1`, `ell=1`, `L=10`, and `beta=0.9` replays a rational
storage, multipliers, and `tau^2=99999/100000` by exact Sylvester checks.

The boundary is sharp for the reduced class of arbitrary `nu`-strongly
monotone, one-Lipschitz operators because an admissible constant skew slope is
marginal there. It is only sufficient for the linked floored-Jordan/quadratic
loop. At the representative point its certified learning-rate supremum is
about `2.599e-8`, whereas the linked curvature-10 zero-linearization loses
local stability at about `4.726e-4`. The roughly `18,183x` gap is explicit:
this is a rigorous but extremely conservative stylized-loop result. It does
not establish nonquadratic, stochastic, BF16, or neural-network convergence.
See `momentum_iqc_certificate.md`. C9--C12 below record the later
structure-aware quadratic and nonlinear extensions.

Prior-art references for the generic IQC framework are Lessard, Recht, and
Packard (<https://arxiv.org/abs/1408.3595>) and Zhang, Bao, Lessard, and
Grosse, JMLR 22(103), 2021
(<https://jmlr.org/papers/v22/20-1068.html>).

## C8. Pinned EMA/Nesterov signal ordering — proved, appendix-level

The pinned upstream implementation does not use C7's recurrence. With
`a=1-beta`, it updates

\[
m_{t+1}=\beta m_t+a g_t,\qquad
s_{t+1}=\beta m_{t+1}+a g_t,\qquad
W_{t+1}=W_t-\eta R(s_{t+1}).
\]

For deterministic quadratic `g_t=H(W_t-W_star)`, conditioned differences obey

\[
z_{t+1}=\beta z_t+a y_t,\qquad
p_{t+1}=\beta z_{t+1}+a y_t
          =\beta^2z_t+(1-\beta^2)y_t,\qquad
y_{t+1}=y_t-\alpha u(p_{t+1}),
\]

where `alpha=eta*K*L` and the normalized transformed operator satisfies the
separate incremental inequalities

\[
\langle p,u\rangle\ge\nu\lVert p\rVert^2,
\qquad \lVert u\rVert^2\le\lVert p\rVert^2,
\qquad \nu=\frac{\mu\ell}{KL}.
\]

With `chi=(y,z,u)`,

\[
T=\begin{bmatrix}1&0&-\alpha\\1-\beta&\beta&0\end{bmatrix},
\qquad
p=(1-\beta^2)y+\beta^2z,
\]

the same two-IQC construction gives a dimension-independent `3 x 3` LMI. At
the pinned default `beta=19/20`, a 60-digit stationarity solve corroborated by
a broad logarithmic scan locates a complex-skew design near `mu=648.024`. The exact representative instead locks

\[
\mu=648,\quad
\nu=\frac{208209088000}{4152745686529},\quad
\alpha=\frac1{400},\quad
\eta=\frac{65065340}{336372400608849},\quad
\tau^2=\frac{99999}{100000}.
\]

Its rational storage, nonnegative multipliers, and strict LMI pass exact
Sylvester checks. This matches the pinned EMA state and Nesterov signal
ordering in real arithmetic after replacing the upstream orthogonalizer by the
repaired floored map and omitting weight decay. It does not claim equivalence
to the deployed BF16/current-normalized kernel, transpose/aspect scaling, or
its complete optimizer path.

For the nearby exact design `mu=648`, the complex-skew sector boundary is
`eta approximately 2.09708214294e-7`; the numerical stationary design over `mu` is
`approximately 2.09708214366e-7`. The matched curvature-10 scalar
zero-linearization has the exact local threshold

\[
\eta_{\mathrm{local}}
=\frac{8120154432000000000000000}
{3901919808117690731741568607}
=0.002081066457364538\ldots.
\]

That is `10,758.62x` the locked certified `eta`. At the numerical stationary
design, the corresponding local threshold is `0.002081027708...`, about
`9,923.44x` its complex-skew necessary boundary. The predeclared hard rule therefore classifies C8 as
an appendix/proof-of-principle result, not a headline practical-stability
claim. No nonquadratic, stochastic, BF16, or neural-network theorem follows
from C8 alone. See `ema_nesterov_iqc_certificate.md`. C9--C12 use additional
operator and objective structure.

## C9. Structure-aware pinned-loop quadratic stability — proved

Use the repaired real-arithmetic operator on every fixed finite matrix shape

\[
R(M)=\mathcal H_{q^{\circ5}}
\!\left(\frac{M}{\max\{1,\lVert M\rVert_F\}}\right)+\rho M,
\]

where there is no additive epsilon,

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3+\frac{4063}{2000}s^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

For every deterministic quadratic with `I <= H <= 10 I`, the pinned
EMA/Nesterov recurrence in C8 with `beta=19/20` is globally incrementally
exponentially stable at

\[
\eta=\frac1{32000},\qquad \tau=\frac{99999}{100000}.
\]

The exact `4 x 4` certificate uses the specific centered decomposition
`R(s)=gamma*s+E(s)` and `Lip(E)<=K_E`, not merely the generic sector used in
C8. Its `P tensor I` storage is dimension independent and therefore covers
arbitrary finite matrix shapes. This is the quadratic bridge, not a nonlinear,
stochastic, BF16, or upstream-training theorem. See
`structure_aware_stability_certificate.md`.

## C10. Pinned-loop nonlinear strongly-convex stability — proved

Keep exactly the C9 operator, max-floor normalization, lack of additive
epsilon, five polynomial steps, constant repair, and `beta=19/20`. For every
fixed differentiable objective on an arbitrary finite real matrix shape that
is globally `1`-strongly convex and `10`-smooth, two distinct certificates
hold:

1. At `eta=1/640000`, a common quadratic `P tensor I` storage proves global
   incremental exponential contraction between any two trajectories at
   `tau=99999/100000`.
2. At the full C9 step `eta=1/32000`, a storage containing the normalized
   objective gap and exact smooth/strongly-convex interpolation supplies proves
   global exponential convergence of every trajectory to the unique minimizer
   at `tau=2499/2500`.

The second result is trajectory-to-minimizer convergence, not an arbitrary-pair
incremental theorem. It is P5's primary positive result and supplies the
stronger unique-minimizer conclusion for its strongly-convex class; C11 later
retains the same step for the broader smooth-PL class. Neither proof fixes
or diagonalizes a Hessian; local Hessian orientations may change along the
trajectory. The claims remain deterministic and exact-real-arithmetic only:
they exclude time-varying or stochastic objectives, BF16, weight decay,
aspect-ratio scaling, additive-epsilon or exact-current normalization, and a
complete neural-network optimizer. See `nonquadratic_stability_certificate.md`
and `nonquadratic_convergence_certificate.md`.

## C11. Pinned-loop smooth-PL function-value convergence — proved

Keep exactly the C9 repaired real-arithmetic operator, exact max floor `c=1`,
no additive epsilon, five Jordan steps with exact coefficients
`(6889/2000,-191/40,4063/2000)`, constant
`rho=210177835339081/260261360000`, and the pinned EMA/Nesterov ordering with
`beta=19/20`. Let `f` be any fixed differentiable objective on an arbitrary
finite real matrix shape with finite `f_star=inf f`, globally `10`-Lipschitz
gradient, and global PL constant `1` under the convention

\[
\frac12\lVert\nabla f(W)\rVert_F^2
\ge f(W)-f_\star.
\]

At the full C9/C10 step `eta=1/32000`, an exact value--momentum storage and the
directed smooth nonconvex interpolation inequalities prove

\[
f(W_t)-f_\star\le C
\left(\frac{399960001}{400000000}\right)^t,
\qquad m_t\longrightarrow0.
\]

The objective may be nonconvex and the minimizer set may be non-singleton.
This is a global function-value convergence theorem, not arbitrary-pair
incremental contraction. It does not assume or conclude a unique or
preselected minimizer. Geometric decay of the certified signal makes the
updates summable, so each trajectory additionally converges to some
trajectory-dependent global minimizer. Local Hessian orientations may change
where Hessians exist. See `pl_convergence_certificate.md` and the exact result
manifest.

The canonical artifact was generated at source commit
`a8f650f6c60dcbc5d2f83647fd367348f4c67548`; the completed P6 checkpoint is
`ef88d8f5b26148af0ec1ca70b506048938bf9bef`, tagged `p6-checkpoint`.
`scripts/reconstruct_pl_convergence.py` independently rebuilds the complete
exact algebra before comparing the canonical artifact. This code-path replay
does not replace the still-pending human proof audit.

The proof is deterministic and exact-real-arithmetic only. It does not cover
stochastic or time-varying gradients, BF16, weight decay, aspect-ratio scaling,
additive-epsilon or exact-current normalization, an unrepaired upstream Muon
implementation, or complete neural-network training.

## C12. Robust dissipativity for disturbed smooth-PL dynamics — proved

Keep exactly the C11 repaired exact-real operator on every fixed finite real
matrix shape:

\[
R(M)=\mathcal H_{q^{\circ5}}
\!\left(\frac{M}{\max\{1,\lVert M\rVert_F\}}\right)+\rho M,
\]

where the normalizer is the exact max floor `c=1`, there is no additive
epsilon, the Jordan quintic is composed for exactly five steps with

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3+\frac{4063}{2000}s^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

Let `f` satisfy the C11 assumptions: it is a fixed differentiable scalar
objective with finite infimum on an arbitrary finite matrix shape, its gradient
is globally `10`-Lipschitz, and it satisfies the global PL inequality with
constant `1`. Reuse one noisy gradient in both occurrences of the pinned
EMA/Nesterov update:

\[
\begin{aligned}
g_t&=\nabla f(W_t), & \widetilde g_t&=g_t+\xi_t,\\
m_{t+1}&=\beta m_t+(1-\beta)\widetilde g_t,
&s_{t+1}&=\beta m_{t+1}+(1-\beta)\widetilde g_t,\\
W_{t+1}&=W_t-\eta\bigl(R(s_{t+1})+e_t\bigr),
&\beta&=\frac{19}{20},\quad \eta=\frac1{32000}.
\end{aligned}
\]

Here `xi_t` is additive gradient error, while `e_t` is additive error after
the repaired operator and before multiplication by `eta`. For every sequence
of finite same-shaped disturbance matrices and every finite initialization,
the exact rational `6 x 6` certificate proves, pathwise,

\[
V_{t+1}\le
\frac{399960001}{400000000}V_t
+\frac12\lVert\xi_t\rVert_F^2
+\frac1{2000000}\lVert e_t\rVert_F^2,
\]

where `V` is the C11 value--momentum storage. The proof is dimension
independent and uses true gradients in the smooth nonconvex interpolation
supplies; the noisy gradient appears only in the disturbed dynamics. The
certificate therefore gives input-to-storage and input-to-output stability for
the function gap, momentum, and true gradient. It is not full-state ISS in
`W`, because a PL objective may have an unbounded non-singleton minimizer set.

The pathwise inequality gives the exact deterministic convolution bound. If
`||xi_t||_F <= X` and `||e_t||_F <= E`, then

\[
\limsup_t V_t\le
\frac{200000000}{39999}X^2+
\frac{200}{39999}E^2.
\]

Square summability of both disturbance sequences implies `V_t -> 0`,
`f(W_t)-f_star -> 0`, `m_t -> 0`, and `grad f(W_t) -> 0`. It does not by itself
imply convergence of `W_t`: for `f(x,y)=x^2/2`, the output errors
`e_t=(0,1/(t+1))` have finite squared energy but drive harmonic motion along
the flat minimizer direction while `V_t=0`. Absolute summability is a
sufficient stronger condition for iterate convergence to some
trajectory-dependent global minimizer.

For a filtration with `E[V_0]<infinity`, an `F_t`-measurable current state,
and `F_(t+1)`-measurable disturbances satisfying conditional second-moment
bounds

\[
\mathbb E[\lVert\xi_t\rVert_F^2\mid\mathcal F_t]\le\sigma_g^2,
\qquad
\mathbb E[\lVert e_t\rVert_F^2\mid\mathcal F_t]\le\sigma_R^2,
\]

taking expectations yields

\[
\mathbb E V_t\le
\bar q^t\mathbb E V_0+
\frac{1-\bar q^t}{1-\bar q}
\left(\frac{\sigma_g^2}{2}+
\frac{\sigma_R^2}{2000000}\right),
\qquad
\bar q=\frac{399960001}{400000000}.
\]

Conditional unbiasedness can be added for the standard stochastic-gradient
interpretation, but is not required for this expected energy bound because the
underlying inequality is pathwise. This is a bounded-second-moment
storage/function-gap result, not almost-sure convergence of the iterates under
persistent noise.

The authoritative certificate and diagnostic source revision is
`c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a`. A 144-case, 17,280-update
CPU/float64 falsification grid covering deterministic bounded, seeded
stochastic, and implementation-only disturbances found zero candidate
violations. These sampled passes are not the proof. See
`robust_dissipativity_certificate.md` and the exact machine-readable artifact.

C12 by itself does not establish full-state ISS, a BF16 error bound, or a
theorem for unrepaired upstream Muon, additive-epsilon or exact-current
normalization, weight decay, aspect-ratio scaling, time-varying objectives,
complete stochastic neural-network training, or errors injected at
unspecified internal locations. The two disturbance gains are sufficient and
are not claimed minimal.

## C13. Fixed-2x2 mixed-precision operator error — proved for a proposed kernel

Fix shape `2 x 2` and retain the exact-real C12 reference operator, exact max
floor `c=1`, no additive epsilon, five Jordan stages, exact coefficients, and
constant repair. The proposed implementation computes a scaled max-floor
normalizer and the linear repair in FP32. It stores the normalized initial
stage input and each of five completed stage outputs in BF16; within each
stage it evaluates the fixed Horner form in FP32 using serial length-two,
separately rounded multiply/add dot products. FMA contraction is excluded. It
is not a native all-BF16-intermediate kernel.

Under IEEE round-to-nearest, ties-to-even arithmetic with gradual underflow,
no FTZ/DAZ, the locked FP32 coefficient encodings and operation order, and for
every finite FP32 input satisfying `max_ij |s_ij| <= 2^116`, exact rational
error propagation proves

\[
\lVert\widehat R(s)-R(s)\rVert_F
\le \frac{11}{100000}\lVert s\rVert_F+\frac{347}{100}.
\]

This is a global input-wise bound on that finite binary32 domain, not a sampled
maximum. An exact Sturm calculation and a five-stage norm invariant cover the
entire matrix input set. The fixed-shape qualification is essential: C13 is
not the dimension-uniform operator theorem of C5--C12.

For an arbitrary real `2 x 2` input with the same magnitude bound, first round
entries to binary32 and then call the locked kernel. The exact global
Frobenius Lipschitz bound

\[
\operatorname{Lip}(R)\le
\rho+\frac{4848763}{10000}
=\frac{336372400608849}{260261360000}
\]

absorbs this interface cast and proves

\[
\lVert\widehat R_{\mathbb R}(s)-R(s)\rVert_F
\le \frac1{5000}\lVert s\rVert_F+\frac{347}{100}.
\]

Embedding this operator in the otherwise exact-real C12 loop, setting gradient
noise to zero, and assuming every signal remains in the certified magnitude
range gives

\[
V_{t+1}\le
\frac{41597186684695561}{41601344000000000}V_t
+\frac{4936769}{819840000000}.
\]

The rate is strictly below one. The range premise has an explicit sufficient
invariant. With
`C_s=1655544025/2600084`, let

\[
H_{\rm safe}=\frac{2^{232}}{C_s}
=\frac{2600084\,2^{232}}{1655544025}.
\]

The exact certificate checks
`4936769/819840000000 <= (1-q_8)H_safe`. Hence, for zero gradient
noise, `V_0<=H_safe` implies `V_t<=H_safe` and
`||s_(t+1)||_F<=2^116` for all `t`; every adapter call is covered. With
gradient noise, the operator signal contains an additional direct noise term,
so this storage-only range invariant does not apply.

Using C12's storage-to-function conversion, the zero-gradient-noise
initial-storage condition consequently gives

\[
\limsup_t(f(W_t)-f_\star)
\le
\frac{462392438350000000}{207695315294468001}
=2.22630172325\ldots.
\]

This is an exact end-to-end operator-error-to-objective certificate for the
proposed fixed-`2 x 2` design, not literal upstream Muon. It does not cover
rounding in FP32 EMA/Nesterov state updates or parameter updates, native BF16
matmul, GPU kernels, other shapes, weight decay, aspect scaling, or
current-plus-epsilon normalization. In particular, finite-precision parameter
subtraction can stall at large binades, so C13 is not a whole-FP32-optimizer
convergence theorem. See `mixed_precision_certificate.md`.

## C14. Shape-scalable mixed-precision operator error — proved for audited shapes

Retain the C13 exact-real reference operator: exact max floor `c=1`, no
additive epsilon, five Jordan stages with coefficients
`(6889/2000,-191/40,4063/2000)`, and constant repair
`rho=210177835339081/260261360000`. For a fixed matrix shape `(r,c)`, orient
the input once to `p x d`, where `p=min(r,c)` and `d=max(r,c)`. The P9
arithmetic contract requires `rc<=2^52`, finite inputs with maximum absolute
entry at most `2^116`, IEEE round-to-nearest ties-to-even, gradual underflow,
and no FTZ/DAZ.

The proposed proof-reference kernel uses the scale-free identity

\[
N(s)=\frac{s/\sigma}
{\max\{1/\sigma,\lVert s/\sigma\rVert_F\}},
\qquad \sigma=\max\{1,\max_{ij}|s_{ij}|\},
\]

with a fixed balanced FP32 reduction. Each thick-Horner matrix product uses
separately rounded FP32 products and adjacent balanced FP32 additions. At the
normalized input and after each of five complete stages, it stores

\[
h=\operatorname{RN}_{b}(Y),\quad
r=\operatorname{fl}_{32}(Y-h),\quad
l=\operatorname{RN}_{b}(r),\quad
X_+=\operatorname{fl}_{32}(h+l).
\]

Exact error propagation, without a Sterbenz assumption, gives the boundary
bound

\[
\lVert X_+-Y\rVert_F
\le \omega\lVert Y\rVert_F+\lceil\sqrt{rc}\rceil\chi,
\qquad
\omega=\frac{282587406795009}{2^{64}}.
\]

The positive subnormal constant `chi` and all of its operation-level terms
are retained exactly in the certificate. Pairwise dot factors depend on the
reduction depth, and every propagated envelope is rounded upward by the
declared exact operator
`U(x)=ceil(2^40 x)/2^40`. Thus the recurrence is finite and independently
replayable rather than a sampled or machine-float estimate.

For every shape whose exact recurrence closes, the arbitrary-real adapter
obeys

\[
\lVert\widehat R_{r,c,\mathbb R}(s)-R_{r,c}(s)\rVert_F
\le A_{r,c}\lVert s\rVert_F+B_{r,c},
\qquad
A_{r,c}=\frac{102465557}{549755813888},
\]

where `B_(r,c)` is the exact shape recurrence reported in the machine-readable
certificate. All seven locked representative shapes pass, including
`768 x 3072`, `3072 x 12288`, `4096 x 4096`, `4096 x 11008`, and
`4096 x 14336`. The maximum listed value is

\[
B_{4096,11008}=B_{4096,14336}
=\frac{2179083213031}{1099511627776}
=1.981864636974\ldots.
\]

At zero gradient noise, Young absorption through C12 with `theta=3006`
proves, for every passing shape,

\[
V_{t+1}\le
\frac{137425214491}{137438953472}V_t+D_{r,c},
\]

with an exact shape-specific `D_(r,c)`. The rate is strictly below one. The
largest representative objective-gap neighborhood is

\[
\frac{798350562999}{1099511627776}
=0.726095607205\ldots<1.
\]

The finite signal premise is closed by the same sufficient initial-storage
invariant used in C13. With gradient noise, the direct noisy-signal term needs
a separate range argument and is not covered by this closure.

There are two different negative findings. First, P8-style serial FP32 norm
accumulation has a genuine executable obstruction: on an all-ones
`4096 x 11008` matrix it sticks at `2^24`, so the returned normalized
rank-one singular value has exact square `43/16>25/16`. Second, ordinary
one-term BF16 boundaries lose this generic Frobenius-to-spectral proof at
rank 72. The latter is only a proof obstruction; it is not a universal
one-term-kernel impossibility result. The compensated slope-only boundary gate
is 4,656,751, while the full recurrence currently stops before
`4608 x 18432` because its fifth stage input leaves the `5/4` proof tube.
That frontier is likewise not an executable instability claim.

C14 is a theorem for the stated slow balanced CPU reference semantics and the
specific shapes whose audit passes. It is not native BLAS/tensor-core parity,
an upstream Muon theorem, a throughput claim, or a whole finite-precision
optimizer theorem. FP32 EMA/Nesterov rounding, parameter storage/update
rounding, weight decay, aspect scaling, and the additive-epsilon upstream
normalizer remain outside its scope. See
`scalable_mixed_precision_certificate.md`.

## C15. Finite-precision outer-loop storage certificate — proved for a proposed shell

Retain the C14 P9 repaired max-floor operator at the audited
`4096 x 11008` shape, with exact floor `c=1`, no additive epsilon, exactly
five Jordan stages, coefficients `(6889/2000,-191/40,4063/2000)`, and
constant repair `rho=210177835339081/260261360000`. P10 surrounds it with a
locked, separately rounded CPU FP32 arithmetic graph. For
`beta=19/20`, `a=1/20`, and `eta=1/32000`, it casts the exact gradient at the
logical master once to FP32, reuses the rounded `a32*g32` product in both
outer additions, and defines total residuals so that

\[
\begin{aligned}
m_{t+1}&=\beta m_t+a\nabla f(W_t)+r_t^m,\\
s_{t+1}&=\beta m_{t+1}+a\nabla f(W_t)+r_t^s,\\
W_{t+1}&=W_t-\eta\widehat R(s_{t+1})+r_t^W.
\end{aligned}
\]

Here `W=H+M+L` is the exact real sum of three finite FP32 master words. The
update forms a rounded FP32 operator step and `L-step`, then uses
`TwoSum(M,pending)` followed by `TwoSum(H,middle_candidate)`. Under the locked
IEEE round-to-nearest, ties-to-even, gradual-underflow, no-FTZ/DAZ,
no-overflow, non-fused operation graph, both `TwoSum` calls are error free.
Thus the logical sum changes by the pending term exactly, and the
parameter-update envelope has no term proportional to the high word.

At `4096 x 11008`, exact rational propagation gives the executable-graph
envelopes

\[
\begin{aligned}
\lVert\epsilon_t^m\rVert_F
&\le C_m\lVert m_t\rVert_F+C_g\lVert\widehat g_t\rVert_F+b_{\rm ema},\\
\lVert\epsilon_t^s\rVert_F
&\le C_m\lVert m_{t+1}\rVert_F+C_g\lVert\widehat g_t\rVert_F+b_{\rm ema},\\
\lVert r_t^W\rVert_F
&\le C_U\lVert\widehat R(s_{t+1})\rVert_F
 +2^{-24}\lVert L_t\rVert_F+b_W,
\end{aligned}
\]

where `epsilon^m,epsilon^s` are relative to the represented gradient and the
last gradient cast is additionally included in `r^m,r^s`. Their approximate
magnitudes are

\[
C_m=1.251697559823128\ldots\times10^{-7},\quad
C_g=6.705522803684974\ldots\times10^{-9},\quad
C_U=5.209585939719435\ldots\times10^{-12}.
\]

These are the displayed approximate magnitudes; their exact rational values
and positive gradual-underflow crumbs are recorded in the theorem note and
canonical artifact. Substitution of P7's storage supplies sharpens the
compensated master envelope to

\[
\lVert r_t^W\rVert_F
\le\frac{93403}{549755813888}\sqrt{V_t}
+\frac{6727}{1099511627776}.
\]

For every differentiable globally `10`-smooth objective satisfying the global
PL inequality with constant `1`, with zero gradient error before the locked
final cast, the exact port-augmented P7 certificate proves on the guarded set
`V_t<=1`

\[
V_{t+1}\le
\frac{549700907325}{549755813888}V_t
+\frac{2162331}{1099511627776}.
\]

The rate is strictly below one and the exact forcing is at most the one-step
storage margin, so `V<=1` is forward invariant subject to the high-word
premise. The storage-to-function consequence is

\[
\limsup_t(f(W_t)-f_\star)
\le\frac{399957341889}{549755813888}
=0.727518166766\ldots<1.
\]

The arithmetic domain requires the P9 `2^116` signal guard, high/middle/low
word bounds `2^30`, `2^7`, and `2^-16`, and the displayed operator-output and
step guards. Storage closes every named guard except the high-word bound,
which is checked after each step as a conditional premise. A global PL
objective can have flat nonunique minimizer directions, so C15 is a
storage/function-value, true-gradient, and momentum result, not full-state ISS
or parameter convergence.

An exact executable `2 x 2` witness uses the actual P9 repaired output at
`m=g=s=diag(64,0)`. At `W_(1,1)=2^30`, ordinary FP32 subtraction loses the
positive rounded step on all 64 repeats. The compensated logical master moves
immediately, and its high word first moves on update 20. This proves a failure
of the specified raw subtraction graph, not a universal impossibility for
every finite-precision update.

C15 does not cover noisy or inexact gradient evaluation before the final
cast, model-forward consumption of the three-word master, literal upstream
`torch.lerp_` bit parity, native BLAS/GPU/tensor-core execution, stochastic
rounding, FTZ/DAZ, weight decay, aspect scaling, additive-epsilon/current
normalization, or neural-network convergence. See
`finite_precision_outer_loop_certificate.md`. Independent human review is
still pending.

## C16. Certified implementation margins above P10 — proved on a locked grid

Retain C15 unchanged at shape `4096 x 11008`, `beta=19/20`,
`eta=1/32000`, exact max floor `c=1`, no additive epsilon, five Jordan
stages with coefficients `(6889/2000,-191/40,4063/2000)`, and constant
repair `rho=210177835339081/260261360000`. Let the real input to the final
gradient cast and the finite represented FP32 output consumed by the master
update satisfy

\[
y_t=\nabla f(W_t)+\zeta_t,\qquad
U_t^{\rm dep}=\widehat R_{9}(s_{t+1})+\nu_t,
\]

where the second equality defines `nu` over the reals rather than executing
another rounded addition. Under the affine Frobenius budgets

\[
\lVert\zeta_t\rVert_F\le a_g\sqrt{V_t}+b_g,\qquad
\lVert\nu_t\rVert_F\le a_R\sqrt{V_t}+b_R,
\]

exact rational propagation proves

\[
V_{t+1}\le q_{11}(a_g,a_R)V_t+D_{11}(b_g,b_R).
\]

The common explicit `zeta` term is placed in P7's effective gradient port and
cancels exactly from the implemented-versus-nominal Nesterov signal mismatch.
The proof nevertheless includes its effects on the final cast, FP32 EMA,
actual signal, P9 error and magnitude, and master-update rounding. The `nu`
port enters both directly and through the output-dependent master residual.
All augmented squares retain their cross terms.

On the declared nonnegative `2^-40` budget grid, the exact one-axis maxima
that preserve `V<=1` are

\[
\begin{array}{c|c}
a_g & 10815225547/2^{40}\\
a_R & 513245498810/2^{40}\\
b_g & 10879487718/2^{40}\\
b_R & 13351103462525/2^{40}.
\end{array}
\]

At every maximum, `q_11+D_11=1`; increasing that coordinate by one grid unit
makes the sum `1+2^-40`. These are exact maxima on the declared grid, not
claims about an unrestricted irrational supremum. The outside points reject
this sufficient certificate; they do not prove actual instability.

A jointly nonzero interior profile

\[
(a_g,a_R,b_g,b_R)=(1/4096,1/128,1/4096,1/8)
\]

gives

\[
q_{11}=\frac{274850515349}{274877906944},\qquad
D_{11}=\frac{1254603}{549755813888},
\]

and the subunit consequence

\[
\limsup_t(f(W_t)-f_\star)
\le\frac{930325132219}{1099511627776}
=0.846125778679\ldots.
\]

The `2^116` signal, `2^15` deployed-output, step-below-two, and middle/low
master-word guards close at this profile. The `2^30` high-word guard remains
conditional exactly as in C15. A model-weight reconstruction error is covered
only after it is converted into the declared total pre-cast gradient
discrepancy; global `10`-smoothness supplies the bound
`||Delta g||_F<=10||Delta W||_F` for that component. C16 measures an
acceptance margin for future implementations. It does not measure or certify
a production model, gradient implementation, CUDA kernel, or literal upstream
Muon. See `implementation_margin_certificate.md`; independent human review is
pending.

## C17. Circuit and broader optimization consequences — open

Boyd's framework models `y in partial f(x)` as a grounded multi-terminal
device and its energy argument uses
`<x-x_star, y-y_star> >= 0`. Thus monotonicity is the relevant circuit
inequality. If `M` is the voltage-like input and `T(M)` the current-like output,
adding `rho M` is a parallel linear **conductance** `rho` (a resistor of value
`1/rho` when `rho>0`). A full port mapping and sign convention still need to be
written before this becomes a formal circuit theorem.

The formal port mapping and the claim that lower deficit predicts a wider
training learning-rate interval remain open. They require:

1. a circuit sign convention and explicit interconnection model;
2. shape-scalable backend-parity quantization bounds, including production
      outer-loop semantics not covered by the proposed C13--C16 certificates;
3. implementation-level parity including weight decay and aspect scaling;
4. only then, a controlled small neural-training sweep.

## C18. Additive-epsilon full-matrix deficit — proved for the exact-real surrogate

Fix any finite real matrix shape `m x n`. Let `h=q composed 5 times`, where

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5,
\]

and define, for every real `epsilon>0`,

\[
E_{h,\epsilon}(M)=\mathcal H_h
\left(\frac{M}{\lVert M\rVert_F+\epsilon}\right).
\]

The arithmetic model is exact real arithmetic. The normalizer is the current
Frobenius norm plus additive epsilon, with no max floor. For the unrestricted
pairwise deficit on each fixed shape, the exact change of variables
`M=epsilon X` proves

\[
\delta_{m,n}(E_{h,\epsilon})
=\frac{\delta_{m,n}(E_{h,1})}{\epsilon}.
\]

This exact `1/epsilon` equality is shape by shape. A restricted-domain
statement would also have to scale its domain.

At nonzero input, with
`t=||M||_F/(epsilon+||M||_F)`, the normalizer derivative is

\[
DN_\epsilon(M)=\frac{1-t}{\epsilon}(I-tP_M)
=\frac{1-t}{\epsilon}\bigl((1-t)I+tQ_M\bigr).
\]

At zero, the quotient is Frechet differentiable and

\[
DE_{h,\epsilon}(0)=\frac1\epsilon
\left(\frac{6889}{2000}\right)^5I\succ0.
\]

The derivative of the outer rectangular spectral map is self-adjoint. Its
active, difference, sum, repeated/zero-singular-value, and rectangular
null-side modes are all secant averages of the even function `h'` on
`[-t,t]`. Three prefix Arb covers plus the inherited global cover give four
exact radius-band slope envelopes. Combining each with the dimension-free
projection/anticommutator lemma and maximizing the resulting rational
quadratic on that band proves

\[
\delta_{m,n}(E_{h,1})
\le
\bar\delta_1
=\frac{6602082433275499863}{41641817600000000}
=158.544530805386839\ldots
\]

for every finite positive `m,n`. Integration of the symmetric-Jacobian bound
along line segments promotes the pointwise result to the authoritative global
pairwise inequality. The proof covers the complete rectangular tangent space;
it is not a diagonal certificate.

For a lower witness, set

\[
\tau=\frac{8974467}{10^9},\qquad
u_-=\frac{803760}{1136689},\qquad
u_+=\frac{803761}{1136689},
\]

so `u_-^2+u_+^2=1`, and use the two matrices obtained by swapping the
diagonal entries of
`tau/(1-tau) * diag(u_-,u_+)`. Exact five-stage rational evaluation proves

\[
\delta_{2,2}(E_{h,1})>
\frac{98823281}{625000}=158.1172496.
\]

The exact quotient has canonical SHA-256
`de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336`.
Zero-padding transfers this strict lower bound only to shapes with
`min(m,n)>=2`; it is not claimed for scalar-only shapes.

Consequently,

\[
E_{h,\epsilon}(M)+\frac{\bar\delta_1}{\epsilon}M
\]

is globally monotone in every finite matrix shape. The upper endpoint is
sufficient, not claimed minimal. At the pinned value `epsilon=10^-7`, the
exact witness forces every globally valid constant repair in a shape
containing it to exceed `1,581,172,496`, while the certified sufficient value
is

\[
\frac{6602082433275499863}{4164181760}
=1,585,445,308.053868\ldots.
\]

That correction is catastrophic at ordinary raw signal scale. It is also
about `9.94` million times the prior max-floor `c=1` sufficient upper
`41528474059081/260261360000 = 159.564501081071...`. This is a design
obstruction for a global constant repair, not a claim that every trajectory
attains the worst case.

C18 certifies only the continuous exact-real surrogate. The pinned upstream
implementation casts to BF16 before its norm, uses BF16 stage arithmetic, and
is backend-specific and discontinuous. Formula provenance from pinned
KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806` does not transfer C18's Jacobian,
pairwise, or repair theorem to literal BF16 Muon. See
`additive_epsilon_deficit_certificate.md`; the human audit packet is pending
and unsigned.

## C19. Nonlinear radial passivation and unavoidable stiffness — proved for the exact-real surrogate

Retain C18's five-stage Jordan operator, additive normalization, exact
rational coefficients, and domain of every fixed finite real rectangular
matrix space. Put

\[
U=\frac{6602082433275499863}{41641817600000000},\quad
a=-\frac{199437}{1250},\quad
\Gamma=-\frac{41528474059081}{260261360000},\quad
z_0=\frac{63}{9937}.
\]

For `z>=0`, define

\[
\widehat d(z)=
\begin{cases}
U,&z\le z_0,\\
-\dfrac{a+\Gamma z}{(1+z)^2},&z\ge z_0,
\end{cases}
\qquad p(z)=\int_0^z\widehat d(u)\,du,
\]

and set `G_epsilon(0)=0` and

\[
G_\epsilon(M)=p(\lVert M\rVert_F/\epsilon)
\frac{M}{\lVert M\rVert_F}\quad(M\ne0).
\]

The two branches join exactly at `z0`, and the tail derivative is
`(2a-Gamma+Gamma*z)/(1+z)^3<0`. Thus `d_hat` is positive and
nonincreasing. It majorizes C18's complete pointwise full-matrix deficit
envelope. The primitive is linear below `z0` and, above it, equals

\[
Uz_0-\Gamma\log\!\frac{1+z}{1+z_0}
+(a-\Gamma)\left(\frac1{1+z}-\frac1{1+z_0}\right).
\]

On the full Frobenius tangent space, `DG_epsilon` has radial eigenvalue
`d_hat(z)/epsilon` and tangential eigenvalue `p(z)/(epsilon*z)`. Since
`d_hat(z)<=p(z)/z<=U`, its smallest derivative cancels C18's pointwise
deficit everywhere. The map is `C1` at zero and at the switch. Integrating
the complete symmetric-Jacobian inequality along line segments proves

\[
E_{h,\epsilon}+G_\epsilon\quad\text{is globally monotone}
\]

for every finite shape. No diagonal certificate is promoted to this claim.

The correction has logarithmic rather than linear output growth and exact
Lipschitz constant `U/epsilon`. At `epsilon=1e-7`, two Arb precisions and an
independent exact rational-log reconstruction enclose its magnitude between
`2571.857826470212145` and `2571.857826470212146` at unit raw norm, and
between `3086.959580254301425` and `3086.959580254301426` at P11's exact
signal guard `13872266672489/549755813888`. The corresponding P12 constant
repair outputs are about `1.585e9` and `4.001e10`.

The improvement cannot remove worst-case differential stiffness. Applying
monotonicity and Cauchy--Schwarz to C18's scaled rank-two pair proves, for
every globally Lipschitz correction `C` that passivates it,

\[
\operatorname{Lip}(C)>
\frac1\epsilon\frac{98823281}{625000}.
\]

This statement applies to shapes with `min(m,n)>=2` and assumes neither a
radial nor an odd correction. The proposed `G_epsilon` is within
`0.270231%` of that universal strict witness lower in Lipschitz stiffness;
this is not a proof of an exact fixed-shape optimum.

Finally, a scalar curvature-one quadratic under the pinned exact-real
EMA/Nesterov order, `beta=19/20`, provides a genuine negative control. At
`epsilon=1e-7`, the correction-only Jury threshold is about `1.69647e-8` and
the complete repaired-operator threshold is about `4.18024e-9`, so the prior
`eta=1/32000` is locally unstable. A finite exact one-step witness gives
objective growth strictly above `23,325,554`. Hence global monotonicity does
not justify reusing the max-floor explicit stability theorem.

C19 is exact-real only. Literal upstream BF16 is backend-specific and
discontinuous, and neither C18 nor C19 certifies it. C19 also does not
propagate the radial repair through the P7--P11 stack, prove strong
monotonicity, or establish neural-network convergence. See
`radial_passivation_tradeoff.md`; independent human review is pending.

## C20. Yosida regularization and full-step PL stability — proved for an exact implicit map

For every fixed finite real rectangular matrix space and every `epsilon>0`,
let `A_epsilon=E_(h,epsilon)+G_epsilon` be C19's continuous, full-domain,
zero-preserving monotone operator. For `lambda,mu>0`, define

\[
B_{\epsilon,\mu}=A_\epsilon+\mu I,\qquad
J_{\lambda B}=(I+\lambda B)^{-1},\qquad
Y_{\lambda,\mu}=\lambda^{-1}(I-J_{\lambda B}).
\]

Explicitly, `E_(h,epsilon)` uses the current additive normalization
`M/(||M||_F+epsilon)` and five Jordan polynomial stages with exact
coefficients `6889/2000`, `-191/40`, and `4063/2000`; `G_epsilon` is C19's
exact radial correction. The stability artifact pins `epsilon=1e-7`, while
the operator theorem and its sector constants hold for every `epsilon>0`.
The practical formula is traced to KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; that upstream revision does not
contain the radial passivator or the Yosida architecture.

The full-domain continuous monotone map `A_epsilon` is maximal monotone in the
finite-dimensional Frobenius space. Hence `B` is maximal and
`mu`-strongly monotone, `I+lambda*B` is bijective, and the resolvent is
globally single-valued. Since `B(0)=0`, both `J` and `Y` preserve zero.

For arbitrary input increments `x`, put `u=Delta J` and `v=Delta Y`, so
`x=u+lambda*v`. Pulling back strong monotonicity of `B` gives

\[
\langle v-mx,\,Mx-v\rangle_F
=\frac{\langle v,u\rangle_F-\mu\lVert u\rVert_F^2}
{\lambda(1+\lambda\mu)}\ge0,
\qquad
m=\frac{\mu}{1+\lambda\mu},\quad M=\frac1\lambda.
\]

This exact full-matrix incremental sector implies that `Y` is
`m`-strongly monotone and `M`-Lipschitz and that

\[
\left\lVert\Delta Y-\frac{M+m}{2}\Delta X\right\rVert_F
\le\frac{M-m}{2}\lVert\Delta X\rVert_F.
\]

The ordinary resolvent identity also gives
`<Delta Y,Delta X>_F>=lambda*||Delta Y||_F^2`, so `Y` is
`lambda`-cocoercive. These claims allow nonsymmetric increments; the shorthand
sector `[m,M]` is not a Loewner ordering or a claim that `Y` is a gradient.

At the locked exact choice `lambda=1/1000`, `mu=1000`, one has
`m=500`, `M=1000`, and the centered decomposition
`Y=750*I+mathcal E` with `Lip(mathcal E)<=250`, uniformly in `epsilon` and
matrix shape. A two-dimensional exact skew example saturates the centered
radius, `A=1000*S` with `S^T=-S` and `S^T*S=I`, for which
`Y=600*I+200*S`. The choice `A=0` attains the lower endpoint, and positive
linear gains tending to infinity approach the upper endpoint.

Use this exact `Y` in the pinned real-arithmetic EMA/Nesterov ordering

\[
m_{t+1}=\frac{19}{20}m_t+\frac1{20}\nabla f(W_t),\qquad
s_{t+1}=\frac{19}{20}m_{t+1}+\frac1{20}\nabla f(W_t),
\qquad W_{t+1}=W_t-\frac1{32000}Y(s_{t+1}).
\]

For every differentiable globally `10`-smooth objective with finite infimum
that satisfies the global PL inequality with constant `1`, the exact rational
value--momentum certificate with

\[
P=10^{-6}\begin{bmatrix}674389&-73827\\-73827&12368\end{bmatrix},
\quad c_F=\frac{313243}{10^6},\quad \tau=\frac{499}{500}
\]

has positive storage, a strictly negative `4 x 4` LMI, and exact objective-
value cancellation. Therefore

\[
V_{t+1}\le\frac{249001}{250000}V_t,
\]

so C20 has `D14=0`: objective gap, gradient, and momentum converge
geometrically, and the iterates converge to some trajectory-dependent global
minimizer. Convexity and uniqueness are not assumed. This is not an
arbitrary-pair incremental-stability claim.

The regularization is essential. At the pinned `epsilon=1e-7`, direct
evaluation (`lambda=0`) inherits C19's exact Jury failure. Even the finite
under-regularized choice `lambda=1/100000` fails the scalar curvature-`10`
Jury condition, while `lambda=1/1000` passes that local control and the global
PL LMI above.

C20 is an existence and stability theorem for an exact-real implicit
resolvent optimizer. It does not supply a numerical resolvent algorithm,
approximate-solve error bound, complexity guarantee, BF16 implementation,
weight decay, aspect scaling, or literal upstream-Muon theorem. See
`yosida_stability_certificate.md`; independent human review is pending.

## C21. Inexact-Yosida residual robustness — proved for an exact-real oracle criterion

Retain C20's operator, for every `epsilon>0` and every fixed finite real
rectangular matrix space,

\[
B=A_\epsilon+1000I,
\qquad
J=(I+B/1000)^{-1},
\qquad
Y=1000(I-J).
\]

Here `A_epsilon` is C19's continuous full-domain monotone radial repair of the
five-stage additive-epsilon Jordan map. The stage coefficients are exactly
`6889/2000`, `-191/40`, and `4063/2000`, and the normalization is
`M/(||M||_F+epsilon)`. The formula is traced to KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`, but upstream does not contain
the radial correction or the P14--P15 resolvent architecture.

At input `s`, let an exact-real oracle return `u_hat` and define

\[
r=s-\widehat u-\frac1{1000}B(\widehat u),
\qquad
\widehat Y(s)=1000(s-\widehat u).
\]

The approximate output is part of the claim and is not interchangeable with
`B(u_hat)` away from an exact solve. From
`u_hat=J(s-r)` and C20's `Lip(J)<=1/2`,

\[
\|\widehat u-J(s)\|_F\le\frac12\|r\|_F,
\qquad
\|\widehat Y(s)-Y(s)\|_F\le500\|r\|_F.
\]

The output gain is uniformly sharp over the abstract admissible monotone-base
class: `A=0`, `B=1000I` attains it. This boundary statement does not identify
the locked P13 map with the zero operator.

Suppose the oracle passes the computable graph-residual rule

\[
\|r\|_F\le\frac1{250}\|s\|_F+\bar r
\qquad \bar r\ge0.
\]

Equivalently, the locked relative tolerance is `kappa=1/250`.

at every oracle call. The output error admits a pointwise split into a
relative part bounded by `2*||s||_F` and an absolute part bounded by
`500*rbar`. Combining only the
relative part with C20's centered residual gives radius `252`. This is a
pointwise one-trajectory supply: an arbitrary oracle meeting the rule is not
claimed to define a monotone or incrementally Lipschitz approximate map.

Use `Y_hat` in the pinned exact-real EMA/Nesterov recurrence with
`beta=19/20` and `eta=1/32000`. For every differentiable globally
`10`-smooth objective with finite infimum satisfying the global PL inequality
with constant `1`, the same storage matrix and exact multipliers as C20
certify a strict `4 x 4` relative-error LMI at radius `252`. A strict exact
`5 x 5` port-augmented LMI has physical absolute-output-error gain
`1/100000`. Consequently,

\[
V_{t+1}\le\frac{249001}{250000}V_t+\frac52\bar r^2.
\]

Thus C21 has `q15=249001/250000<1` and `C15=5/2`. In particular,

\[
\limsup_tV_t\le\frac{625000}{999}\bar r^2,
\qquad
\limsup_t(f(W_t)-f^\star)
\le\frac{6250000000000}{312929757}\bar r^2.
\]

At `rbar=0`, objective gap, gradient, and momentum converge geometrically and
the iterates converge to some trajectory-dependent global minimizer. No
unique-minimizer or arbitrary-pair contraction claim is made. For persistent
positive `rbar`, only storage and objective ultimate neighborhoods are
claimed, not parameter convergence.

The exact controls have different logical force. `kappa=rbar=0` recovers the
appropriate C20 fractions. At `kappa=3/500`, centered radius `253` fails the
unchanged frozen-storage `4 x 4` certificate; this is not a proof of
instability. Loose rules do admit true abstract-class failures: with `A=0`
and `B=1000I`, `kappa=1`, `r=-s`, `u_hat=s` gives `Y_hat=0` and permits
stalling away from stationarity, while `kappa=2`, `r=-2s`, `u_hat=3s/2`
gives `Y_hat=-500s`; for scalar curvature `1`, the pinned characteristic has
`p(1)=-1/1280<0`.

C21 certifies an a posteriori exact-real residual criterion, not an algorithm
for satisfying it. It does not establish a finite iteration count,
complexity, finite-precision evaluation of `B` or the residual, BF16/FP32
kernel, accelerator or literal upstream parity, weight decay, aspect scaling,
stochastic gradients, or neural-network convergence. See
`inexact_yosida_robustness_certificate.md`; independent human review is
pending.

## C22. Equivariant structured resolvent solve and fidelity obstruction — solver theorem proved; meaningful-fidelity gate fails

Retain C19--C21's exact-real additive-epsilon architecture on an arbitrary
fixed finite real `m x n` matrix space. The P16 artifact locks
`epsilon=1/10000000`, five Jordan stages with exact coefficients `6889/2000`,
`-191/40`, and `4063/2000`, the P13 radial repair, `mu=1000`,
`lambda=1/1000`, and the P15 relative residual tolerance `kappa=1/250`.
The practical polynomial and epsilon placement are traced to
KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; upstream contains neither the
radial repair nor the resolvent solver.

Let `B=A_epsilon+mu*I`, `F=I+lambda*B`, and `J=F^(-1)`. For every pair of
orthogonal matrices of compatible sizes,

\[
J(QSR^\top)=QJ(S)R^\top.
\]

This follows from bi-orthogonal equivariance of Frobenius normalization, every
odd rectangular spectral-polynomial stage, the radial repair, and the shunt,
together with uniqueness of the resolvent. A stabilizer argument then proves
that `J(S)` preserves the singular subspaces and rank of `S`; repeated input
singular values yield repeated output values, zero modes remain zero, and the
claim is independent of basis choices inside repeated and null subspaces.

For singular values `sigma_i` of `S`, write the unknown resolvent singular
values as signed coordinates `x_i`, with `r=||x||_2`, `z=r/epsilon`, and
`w_i=x_i/(r+epsilon)`. The equation `F(J(S))=S` is exactly

\[
\Phi_i(x)=\left[1+\lambda\left(\mu+\frac{p(z)}r\right)\right]x_i
            +\lambda h(w_i)-\sigma_i=0.
\]

For `r>0`, its Jacobian is a diagonal matrix plus one rank-one update. The
three exact P12/P13 radius-band margins are positive, the diagonal entries are
at least `2`, and the Sherman--Morrison denominator is positive. Thus a Newton
direction can be computed in linear work per singular value after the scalar
five-stage evaluations, without a dense solve in the singular-value count.
The origin and every band switch are handled separately.

For the exact-real safeguarded method, the merit function
`||Phi(x)||_2^2/2` has strict Newton descent, and Armijo backtracking terminates
at every nonroot iterate. Strong monotonicity makes the level sets bounded,
gives a unique root, and proves global convergence from every finite start.
Consequently the exact-real loop terminates after finitely many iterations for
each fixed input whenever its P15 residual threshold is positive. This is not
a useful uniform a priori iteration-count bound.

The guarded FP64 reference implementation performs one SVD for the reduced
solve, reconstructs the matrix candidate, reevaluates `B` on that stored
candidate with a second SVD, recomputes the literal graph residual, and
returns success only when

\[
\|S-\widehat U-\lambda B(\widehat U)\|_F
\le \frac1{250}\|S\|_F+\bar r_{\rm fp64}.
\]

Its deployed output is `Y_hat=1000*(S-U_hat)`, never `B(U_hat)`. All 13
declared computed-residual cases pass, including complete reduced spectra for
`768 x 768`, `768 x 3072`, `3072 x 12288`, and `4096 x 11008`. The default
study attains at most 8 Newton iterations, no accepted-case backtracking, and
a worst computed residual of approximately `5.880e-14`. These are
deterministic FP64 diagnostics, not a proof that rounded residual evaluation
upper-bounds the exact residual, and the large-shape cases do not execute a
dense SVD or provide accelerator timing.

P16 separately evaluates whether the locked stable operator remains
meaningfully Muon-like. Exact rational algebra and outward-rounded Arb
enclosures prove algebraic noncollapse: on the canonical `diag(3,4)` input,
the two modal gains are distinct. Quantitatively, however, the exact-output
enclosure gives best-scalar departure about `5.879e-6`, while the exact-real
five-stage Jordan comparator has departure about `6.997e-2`. The retained
shaping fraction is only about `8.403e-5`, far below the frozen `1/10`
gate (and the absolute departure is below its `1/1000` gate). The selected
Yosida map is therefore distinct but effectively scalar on the locked
comparator.

A frozen six-point `(lambda,mu)` diagnostic finds no candidate that both
passes the unchanged frozen P14 certificate and the meaningful-fidelity
gates. This sampled frontier is not a global impossibility theorem, and
failure of a frozen certificate is not instability. Thus C22 is a positive
equivariance, reduction, and exact-real solver theorem plus a locked-parameter
fidelity obstruction; it does not satisfy the overall P16 acceptance gate.
It is not an FP64 rounding theorem, a BF16 solver, a literal upstream-Muon
implementation, or evidence for neural-network convergence. See
`equivariant_resolvent_solver.md`; independent human review is pending.

## C23. Shape-preserving gated resolvent — two global pointwise-sector smooth-PL designs certified

Retain C19--C22's exact-real additive-epsilon architecture on an arbitrary
fixed finite real `m x n` matrix space. Fix `epsilon=1/10000000`, five Jordan
stages with exact coefficients `6889/2000`, `-191/40`, and `4063/2000`, the
P13 radial repair, `mu=1000`, and `lambda=1/1000`. The practical polynomial
and epsilon placement are traced to KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; upstream contains neither the
radial repair, resolvent, nor gate below.

Let `A_epsilon=E_(h,epsilon)+G_epsilon`, `B=A_epsilon+1000I`,
`J=(I+B/1000)^(-1)`, `Y=1000(I-J)`, and
`X(S)=E_(h,epsilon)(J(S))`. For `q=||S||_F^2`, define the `C^2` gate

\[
\theta(q)=
\begin{cases}
0,&q\le 1/4,\\
\bar\theta\,\psi((4q-1)/3),&1/4<q<1,\\
\bar\theta,&q\ge1,
\end{cases}
\qquad
\psi(t)=6t^5-15t^4+10t^3,
\]

and expose

\[
T_{\bar\theta,c}(S)
=\bigl(1-\theta(\|S\|_F^2)\bigr)Y(S)/c
 +\theta(\|S\|_F^2)X(S).
\]

Bi-orthogonal equivariance and rank preservation reduce the origin-centered
bound to nonnegative singular modes, including repeated and zero modes. Write
the singular-mode values of `X(S)` as `e_i(S)`. Exact P12--P13 inequalities
give the dimension-uniform analytic gain bound

\[
0\le \frac{e_i(S)}{\sigma_i(S)}\le
M_X:=\frac{20191130443162880000000}{26793221204801899863}
<754.
\]

Together with `500<=y_i(S)/sigma_i(S)<=1000`, this proves

\[
m=\frac{(1-\bar\theta)500}{c},\qquad
M=\frac{(1-\bar\theta)1000}{c}+\bar\theta M_X,
\qquad
\|T(S)-\gamma S\|_F\le K\|S\|_F,
\]

where `gamma=(M+m)/2` and `K=(M-m)/2`. This is a global
**origin-centered pointwise sector**, not an incremental sector, Lipschitz
bound, or Jacobian certificate. Differentiating the gate would add a radial
cross term; the P6 value--momentum proof does not differentiate `T` or compare
two arbitrary inputs and therefore needs only the displayed pointwise
residual supply.

For the pinned exact-real EMA/Nesterov ordering with `beta=19/20`, two exact
`4 x 4` value--momentum LMIs prove global one-trajectory convergence for every
differentiable globally `10`-smooth objective with finite infimum satisfying
the global PL inequality with constant `1`:

- the primary high-fidelity design has `bar_theta=3/4`, `c=4096`,
  `eta=1/128000`, and
  `tau^2=281474943156225/281474976710656<1`;
- the secondary full-step design has `bar_theta=1/8`, `c=8192`,
  `eta=1/32000`, and
  `tau^2=281474741829681/281474976710656<1`.

In both cases, objective gap, gradient, and momentum converge geometrically,
and summable parameter updates imply convergence to some trajectory-dependent
global minimizer. PL does not imply uniqueness, and no arbitrary-pair
incremental contraction is claimed.

The passive region is necessary only for the claimed local
incremental/passivity safeguard, not for the pointwise-sector PL proof. At
the exact scalar control `t=63/10000`, raw `X` has derivative strictly between
`-147000` and `-146000` at a source strictly between `19/10000` and `1/500`.
Both locked gates are zero there. An under-sized passive-region control
instead exposes a derivative between `-19000` and `-18000` and has a strictly
negative frozen scalar EMA/Nesterov Jury margin. This is a local linearization
control, not a claimed divergent smooth-PL trajectory; the global PL proof
uses only `0<=theta<=bar_theta`.

Outward-rounded Arb enclosures on the canonical `diag(3,4)` input, including
P15 residual inflation, prove that both designs pass the unchanged P16
best-scalar-departure `1/1000` and shaping-retention `1/10` thresholds. The
declared spectrum-grid and realistic-rank study is deterministic FP64
diagnostic evidence, not a global fidelity extremum or rounding certificate.

C23 assumes the exact resolvent in exact real arithmetic. It does not
propagate P16 approximate-solver error through `X` and the gate; certify FP64
or BF16 arithmetic or rounded residual evaluation; establish literal
upstream parity; cover weight decay, aspect scaling, stochastic gradients, or
model-state reconstruction; or prove neural-network convergence. See
`shape_preserving_resolvent.md`; independent human review is pending.

## C24. Sector-projected shape interface and useful-rate PL convergence — proved in exact real arithmetic

Retain C23's matrix domain, additive `epsilon=1/10000000`, five Jordan
stages with exact coefficients `6889/2000`, `-191/40`, and `4063/2000`, P13
radial repair, `mu=1000`, `lambda=1/1000`, exact resolvent `J`, Yosida map
`Y=1000(I-J)`, and C2 gate on `q=||S||_F^2`. Put
`X(S)=E_(h,epsilon)(J(S))` and define

\[
\alpha_K(S)=\min\left\{1,
 \frac{K\langle X(S),S\rangle_F}{\|X(S)\|_F^2}\right\},
\qquad Z_K(S)=\alpha_K(S)X(S),
\]

with `Z_K=0` when `X=0`. P16's common singular-vector theorem and strict
sign preservation of every locked Jordan stage give `<X(S),S>_F>=0`.
Checking the active and inactive branches exactly proves

\[
\|Z_K(S)\|_F^2\le K\langle Z_K(S),S\rangle_F.
\]

Thus the ray scaling preserves the spectral direction of every nonzero `X`
while putting it in the full-matrix pointwise disk sector `[0,K]`.

Lock `K=1`, passive divisor `c=1024`, and gate ceiling `3/4`, and expose

\[
T(S)=(1-\theta)Y(S)/1024+\theta Z_1(S).
\]

Convexity of origin-centred sector disks, together with C20's `[500,1000]`
Yosida sector, gives the exact dimension-uniform pointwise sector

\[
\boxed{\left[\frac{125}{1024},\frac{509}{512}\right]},
\qquad
\gamma=\frac{1143}{2048},\quad K_T=\frac{893}{2048}.
\]

This is not an incremental sector or derivative bound. The ray projection
also has a kink at its active-set boundary. C24's P6 value--momentum argument
uses only the origin-centred pointwise residual supply and therefore does not
differentiate the projection or gate.

For `beta=19/20`, every differentiable globally `10`-smooth, global-PL-`1`
objective satisfies the exact one-trajectory certificate

\[
\mathcal V_{t+1}\le
\frac{999598040401}{1000000000000}\mathcal V_t
\]

at `eta=1/83`. The locked storage is
`P=((97/125,-151/500),(-151/500,17/100))`; the function storage is one,
the reverse-interpolation multiplier is `3459/500`, and the residual
multiplier is `9/250`. Exact rational Sylvester checks prove strict storage
positivity and LMI negativity. Consequently objective gap, gradient, and
momentum converge geometrically, and the iterates converge to some
trajectory-dependent global minimizer. Neither uniqueness nor arbitrary-pair
contraction is claimed.

The exact comparison

\[
\left(\frac{999598040401}{10^{12}}\right)^{10}
<\frac{249001}{250000}
\]

proves that the certified objective half-life is below ten times P14's. Its
descriptive value is `1724.0734` iterations, about `9.9592x` P14. Exact
amplitude and selected-step guards are

\[
\frac1{10}<\frac{\|T(S)\|_F}{\|S\|_F}<1,
\qquad
\frac1{1000}<\eta\frac{\|T(S)\|_F}{\|S\|_F}<\frac1{80},
\]

with the precise effective endpoints `125/84992` and `509/42496`.

The predeclared `eta=1/50` sector-IQC route is obstructed exactly. The
admissible boundary map `T_0=(1143/2048)I+(893/2048)Q`, with
`Q^T=-Q` and `Q^TQ=I`, on the valid quadratic `f(W)=5||W||_F^2` has second
Schur--Cohn margin

\[
-\frac{3768360579178620269}{1759218604441600000000000}<0.
\]

This excludes a global theorem over the entire sector class at that step; it
does not prove the structured P18 map itself is unstable. A finite exact
rational frontier records passing points from `eta=1/75` through `1/150`.
The selected `1/83` point is the declared maximum-step rate-gate pass, while
the `eta=1/120`, `q=624350169/625000000` point has a faster certified
half-life of about `666.31` iterations. No global optimality over unsearched
storages or stronger structural descriptions is claimed.

Outward-rounded Arb arithmetic on `diag(3,4)` certifies the frozen P16
departure and retention gates and the new amplitude gate. The separate FP64
study records complete computed P15 graph-residual checks and passes all
2,176 informative points on the sampled `3/4<=||S||_F<=25` annulus, but only
192/327 informative broad-grid cases. Therefore meaningful fidelity is not
global. Removing the projection restores a huge pointwise upper gain; using
`K=1/100` fails the fidelity gate.

C24 remains an exact-resolvent, exact-real theorem. It does not propagate
solver error through the nonlinear projection, certify FP64/BF16 rounding,
establish upstream learning-rate or update-magnitude parity, cover weight
decay, aspect scaling, stochastic gradients, represented model state, or
neural-network training. See `sector_projected_useful_rate.md`; independent
human review is pending.
