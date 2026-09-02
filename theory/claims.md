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
See `momentum_iqc_certificate.md`. C9 and C10 below record the later
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
from C8 alone. See `ema_nesterov_iqc_certificate.md`. C9 and C10 use additional
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
incremental theorem. It is the main positive stability result because it retains the full structure-aware
quadratic step for the complete nonlinear function class. Neither proof fixes
or diagonalizes a Hessian; local Hessian orientations may change along the
trajectory. The claims remain deterministic and exact-real-arithmetic only:
they exclude time-varying or stochastic objectives, BF16, weight decay,
aspect-ratio scaling, additive-epsilon or exact-current normalization, and a
complete neural-network optimizer. See `nonquadratic_stability_certificate.md`
and `nonquadratic_convergence_certificate.md`.

## C11. Circuit and broader optimization consequences — open

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
2. PL/nonconvex, stochastic, and quantized robustness extensions;
3. implementation-level parity including weight decay and aspect scaling;
4. only then, a controlled small neural-training sweep.
