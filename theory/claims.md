# Claim ledger

This file separates established statements from the intended paper claims. The
ambient inner product is Frobenius, `<A, B> = tr(A^T B)`, and monotonicity means

\[
\langle T(A)-T(B), A-B\rangle \ge 0
\]

for every pair in the stated domain. This is the incremental/equilibrium-
relative inequality that appears in Boyd's circuit energy-dissipation
argument. It is stronger than ordinary origin-based static passivity
`<T(A), A> >= 0`; the paper should say **incremental passivity (monotonicity)**
until its port convention is fully fixed.

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
values `h'(3/5)` and `h'(4/5)`. Thus the fixed-scale diagonal Jacobian is
positive at this point, while current Frobenius normalization makes the
symmetric Jacobian indefinite. This isolates normalization rather than merely
reusing the known scalar nonmonotonicity of the Jordan polynomial.

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

## C5. Circuit and optimization consequences — open

Boyd's framework models `y in partial f(x)` as a grounded multi-terminal
device and its energy argument uses
`<x-x_star, y-y_star> >= 0`. Thus monotonicity is the relevant circuit
inequality. If `M` is the voltage-like input and `T(M)` the current-like output,
adding `rho M` is a parallel linear **conductance** `rho` (a resistor of value
`1/rho` when `rho>0`). A full port mapping and sign convention still need to be
written before this becomes a formal circuit theorem.

The full-matrix upper certificate and the claim that lower deficit predicts a
wider stable learning-rate interval remain open. They require:

1. interval/SOS or another rigorous global upper bound on a stated domain;
2. controlled matrix-quadratic and small NanoGPT sweeps.
