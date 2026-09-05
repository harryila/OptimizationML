# P16 equivariant resolvent-solver theorem and acceptance contract

## Scope

This note derives a computable reference solver for the exact-real P14--P15
resolvent.  The ambient space is an arbitrary fixed finite real `m x n`
matrix space with the Frobenius inner product.  The additive-normalization
parameter satisfies `epsilon>0`.

The locked operator is

\[
\begin{aligned}
E_{h,\epsilon}(M)
  &=\mathcal H_h\!\left(\frac{M}{\lVert M\rVert_F+\epsilon}\right),\\
A_\epsilon(M)&=E_{h,\epsilon}(M)+G_\epsilon(M),\\
B(M)&=A_\epsilon(M)+\mu M,
\qquad \mu=1000,\\
J&=(I+\lambda B)^{-1},
\qquad \lambda=\frac1{1000},\\
Y(S)&=\frac{S-J(S)}{\lambda}.
\end{aligned}
\tag{P16.1}
\]

Here `h` is five repetitions of

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5,
\tag{P16.2}
\]

and `G_epsilon` is P13's exact radial passivator.  The practical polynomial
formula is traced to KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; that upstream code does not
contain the P13 correction or the P14--P16 resolvent architecture.

P13 proves that `A_epsilon` is continuous, full-domain, zero-preserving, and
monotone.  Hence `B` is `1000`-strongly monotone, and
`F=I+lambda*B` is `2`-strongly monotone.  P14 proves that `F` is bijective and
that `J` is single-valued.  P15 proves stability for any returned candidate
whose *actual* graph residual satisfies its stopping rule.  P16 supplies the
missing structured exact-arithmetic algorithm and a guarded FP64 reference
implementation contract; it does not yet certify FP64 rounding error.

## 1. Bi-orthogonal equivariance

Let `Q in O(m)` and `R in O(n)`.  Frobenius normalization is invariant under
the action `M -> Q M R^T`.  Each quintic stage can be written as an odd
rectangular spectral polynomial, and therefore

\[
\mathcal H_h(QMR^\top)=Q\mathcal H_h(M)R^\top.
\tag{P16.3}
\]

The P13 term has the radial form

\[
G_\epsilon(0)=0,
\qquad
G_\epsilon(M)=p\!\left(\frac{\lVert M\rVert_F}{\epsilon}\right)
                  \frac{M}{\lVert M\rVert_F}
\quad(M\ne0),
\tag{P16.4}
\]

so it has the same equivariance.  The identity shunt does as well.  Thus

\[
F(QMR^\top)=QF(M)R^\top.
\tag{P16.5}
\]

If `U=J(S)`, then `F(U)=S`, and (P16.5) gives
`F(QUR^T)=QSR^T`.  Uniqueness of the inverse consequently proves

\[
\boxed{J(QSR^\top)=QJ(S)R^\top.}
\tag{P16.6}
\]

The same identity holds for `Y`.  This is a mathematical
bi-orthogonal-equivariance theorem, not a physical-circuit claim.

## 2. Singular-vector, multiplicity, and rank preservation

Let `k=min(m,n)` and take a full SVD

\[
S=U\Sigma V^\top,
\qquad
\Sigma_{ii}=\sigma_i\ge0\quad(1\le i\le k).
\tag{P16.7}
\]

Apply (P16.6) to the complete stabilizer of `Sigma`.  Simultaneous rotations
on a repeated positive-singular-value block leave `Sigma` fixed and force the
corresponding block of `J(Sigma)` to be a scalar multiple of the identity.
Independent rotations and sign changes on the left and right null spaces
force every null-space and cross-block component to vanish.  Sign changes on
one positive block eliminate cross terms between distinct blocks.  Therefore

\[
J(S)=U\widehat\Sigma V^\top,
\qquad
\widehat\Sigma_{ii}=x_i,
\tag{P16.8}
\]

with equal `x_i` on every repeated `sigma_i` block and zeros on all
rectangular null-side modes.  This statement is independent of the chosen
bases inside repeated and null singular subspaces.

The signs can also be fixed.  One stage in (P16.2) is

\[
q(x)=x\left(\frac{4063}{2000}x^4-\frac{191}{40}x^2
             +\frac{6889}{2000}\right).
\tag{P16.9}
\]

The quadratic factor in `x^2` has positive leading coefficient and exact
discriminant

\[
\left(-\frac{191}{40}\right)^2
-4\frac{4063}{2000}\frac{6889}{2000}
=-\frac{2594691}{500000}<0.
\tag{P16.10}
\]

Thus every stage, and hence `h`, strictly preserves the sign of every
nonzero real scalar.  Because `p(z)>0` for `z>0` and `mu>0`, each diagonal
component of `F` has the sign of the corresponding component of its input.
It follows that

\[
x_i\ge0,
\qquad
x_i=0\Longleftrightarrow\sigma_i=0.
\tag{P16.11}
\]

In particular, the resolvent preserves rank.  Its output `Y` has the same
singular subspaces and support.  Repeated input singular values produce equal
output singular values; no differentiability of the SVD is used in this
argument.

## 3. Exact singular-value system

Write `x=(x_1,...,x_k)`,

\[
r=\lVert x\rVert_2,
\qquad z=\frac r\epsilon,
\qquad w_i=\frac{x_i}{r+\epsilon}.
\tag{P16.12}
\]

For `r>0`, define

\[
c(r)=\mu+\frac{p(z)}r.
\tag{P16.13}
\]

Then `F(J(S))=S` is exactly equivalent to the coupled `k`-scalar system

\[
\boxed{
\Phi_i(x)
=\left(1+\lambda c(r)\right)x_i
 +\lambda h(w_i)-\sigma_i=0,
\qquad 1\le i\le k.}
\tag{P16.14}
\]

The only cross-coordinate coupling is through `r`.  At an exact root the
singular values of the Yosida output are

\[
y_i=\frac{\sigma_i-x_i}{\lambda}.
\tag{P16.15}
\]

For an approximate root, (P16.15), rather than `B_i(x)`, remains the deployed
P15 output convention.  The graph residual in P15's sign convention is

\[
r_{\rm graph}=S-\widehat U-\lambda B(\widehat U),
\qquad
\lVert r_{\rm graph}\rVert_F=\lVert\Phi(x)\rVert_2.
\tag{P16.16}
\]

The equality in (P16.16) is exact because every term has the same singular
bases.  It permits the structured solver to check the actual mathematical
P15 residual without evaluating a dense matrix polynomial.

For completeness, P13 locks

\[
z_0=\frac{63}{9937},
\qquad
p'(z)=\widehat d(z),
\tag{P16.17}
\]

where

\[
\widehat d(z)=
\begin{cases}
U_0,&0\le z\le z_0,\\[2pt]
-\dfrac{a+\Gamma z}{(1+z)^2},&z\ge z_0,
\end{cases}
\tag{P16.18}
\]

with

\[
U_0=\frac{6602082433275499863}{41641817600000000},
\quad
a=-\frac{199437}{1250},
\quad
\Gamma=-\frac{41528474059081}{260261360000}.
\tag{P16.19}
\]

The closed form of `p` is the one recorded in P13; implementations should use
that locked expression and `log1p`-style evaluation in its tail.

## 4. Diagonal-plus-rank-one Jacobian

For `r>0`, put `v=x/r` and

\[
c'(r)=\frac{z\widehat d(z)-p(z)}{\epsilon^2z^2}.
\tag{P16.20}
\]

Direct differentiation of (P16.14) gives

\[
D\Phi(x)=D+a_{\rm lr}v^\top,
\tag{P16.21}
\]

where `D` is diagonal and

\[
\begin{aligned}
D_{ii}
 &=1+\lambda c(r)+\frac{\lambda h'(w_i)}{r+\epsilon},\\
(a_{\rm lr})_i
 &=\lambda x_i\left(
      c'(r)-\frac{h'(w_i)}{(r+\epsilon)^2}
    \right).
\end{aligned}
\tag{P16.22}
\]

Thus a Newton system needs only diagonal solves and one dot product.  For a
right-hand side `b`, Sherman--Morrison gives

\[
(D+a_{\rm lr}v^\top)^{-1}b
=D^{-1}b-D^{-1}a_{\rm lr}
 \frac{v^\top D^{-1}b}{1+v^\top D^{-1}a_{\rm lr}}.
\tag{P16.23}
\]

The diagonal inverses in (P16.23) are theorem-safe.  Indeed,

\[
\frac{p(z)}r+\frac{h'(w_i)}{r+\epsilon}
=\frac1\epsilon\left(\frac{p(z)}z+\frac{h'(w_i)}{1+z}\right)\ge0.
\tag{P16.24}
\]

Here `|w_i|<=z/(1+z)`, and `h'` is even because `h` is odd, so P12's
prefix derivative bounds apply also to signed Newton coordinates.  On P12's
first three radius bands, `p(z)/z=U_0`; the exact worst lower
margins in (P16.24) are

\[
\begin{aligned}
U_0-\frac{504}{5}
 &=\frac{2404587219195499863}{41641817600000000}>0,\\
U_0-\frac{77877}{500}\frac{199}{200}
 &=\frac{148632173097451863}{41641817600000000}>0,\\
U_0-\frac{6379}{40}\frac{497}{500}
 &=\frac{1098544686059863}{41641817600000000}>0.
\end{aligned}
\tag{P16.25}
\]

On the tail, monotonicity of `d_hat` gives `p(z)/z>=d_hat(z)`, and

\[
\widehat d(z)+\frac{a}{1+z}
=\frac{(a-\Gamma)z}{(1+z)^2}\ge0,
\qquad
a-\Gamma=\frac{6205081}{416418176}>0.
\tag{P16.26}
\]

Consequently

\[
D_{ii}\ge1+\lambda\mu=2.
\tag{P16.27}
\]

The full Jacobian is also nonsingular.  P13 monotonicity gives

\[
\operatorname{Sym}D\Phi(x)\succeq(1+\lambda\mu)I=2I.
\tag{P16.28}
\]

Every real matrix whose symmetric part is positive definite has positive
determinant.  Since `det(D)>0`, the matrix-determinant lemma and (P16.28)
show

\[
1+v^\top D^{-1}a_{\rm lr}
=\frac{\det D\Phi(x)}{\det D}>0.
\tag{P16.29}
\]

Therefore (P16.23) is an exact identity at every nonzero iterate.  A floating
implementation must nevertheless treat a nonpositive or nonfinite computed
diagonal or denominator as a detected numerical failure, not silently use
the theorem to override the computed value.

## 5. Zero, switch, repeated-value, and ordering branches

The following cases are part of the algorithm contract.

1. If `S=0`, return `J(S)=0` and `Y(S)=0` exactly.
2. At `x=0`, do not evaluate `p(z)/r` or (P16.20).  The ordinary derivative is

   \[
   D\Phi(0)=\left[
   1+\lambda\left(
     \mu+\frac{U_0+(6889/2000)^5}{\epsilon}
   \right)\right]I.
   \tag{P16.30}
   \]

3. For `0<z<=z0`, evaluate `p(z)/r=U_0/epsilon` and `c'(r)=0`
   directly.  This avoids a removable `0/0` and catastrophic cancellation.
4. At `z=z0`, P13 proves exact continuity of `d_hat` and `p'`; moreover
   `z*d_hat(z)-p(z)=0`.  Hence (P16.20) has the common value zero.  The map is
   `C1` there, although its second derivative has a branch kink.
5. If some `x_i=0` while `r>0`, then `v_i=(a_lr)_i=0`; its coordinate is
   decoupled and `D_ii` remains strictly positive.  No division by an
   individual singular value is permitted.
6. Repeated singular values are kept in a permutation-symmetric solve.
   Their exact roots agree.  Zero singular values return exact zeros, making
   reconstruction invariant to arbitrary null-space bases.

For fixed `r`, the scalar right side in (P16.14) has derivative `D_ii>=2`.
It is therefore strictly increasing.  Besides reproving equality on repeated
blocks, this shows that sorted input singular values produce sorted resolvent
singular values.

## 6. Safeguarded Newton convergence in exact arithmetic

The reduced equations may be solved on the signed coordinate space
`x in R^k`; intermediate Newton iterates need not be constrained to be
nonnegative.  The unique root is nonnegative by Section 2.

Let

\[
\Psi(x)=\frac12\lVert\Phi(x)\rVert_2^2.
\tag{P16.31}
\]

At a nonroot, compute the Newton direction from (P16.23),

\[
d=-D\Phi(x)^{-1}\Phi(x).
\tag{P16.32}
\]

It is always a strict merit-function descent direction:

\[
\nabla\Psi(x)^\top d
=\Phi(x)^\top D\Phi(x)d
=-\lVert\Phi(x)\rVert_2^2<0.
\tag{P16.33}
\]

Fix Armijo parameters `0<c_A<1/2` and `0<theta<1`.  Starting with step one,
backtrack by powers of `theta` until

\[
\Psi(x+td)\le\Psi(x)-c_A t\lVert\Phi(x)\rVert_2^2.
\tag{P16.34}
\]

This exact-arithmetic iteration converges globally from every finite start.
To see this, (P16.28) implies

\[
\lVert D\Phi(x)^{-1}\rVert_2\le\frac12
\tag{P16.35}
\]

and strong monotonicity makes each level set of `Psi` bounded.  Moreover,
`||d||_2<=||Phi||_2/2`; hence every rejected trial segment with `0<=t<=1`
lies in one fixed bounded enlargement of the initial level set.  The locked
normalizer is `C1` with locally Lipschitz derivative at zero; P13's `p` is
`C1` with locally Lipschitz derivative across `z0`; elsewhere the equations
are smooth.  Thus `D Phi` is Lipschitz and bounded on that compact enlarged
set.  Equations (P16.33)--(P16.35) give a uniform positive lower bound on
accepted Armijo steps.  Summing the decreases in (P16.34) forces
`||Phi(x_j)||_2 -> 0`.  Strong monotonicity then forces `x_j` to the unique
root.

For every nonzero `S`, the P15 threshold

\[
\frac1{250}\lVert S\rVert_F+\bar r_{\rm fp64}
\tag{P16.36}
\]

is positive.  Exact convergence therefore implies finite termination of the
loop once (P16.16) is below that threshold.  This is a reliable guarded
termination result, not a useful uniform iteration-count bound.

An FP64 implementation must additionally set iteration and backtracking caps,
check all values for finiteness, reject a failed SVD or line search, and
return a failure status if the recomputed residual does not pass (P16.36).
Only successful calls may emit

\[
\boxed{\widehat Y(S)=1000(S-\widehat U).}
\tag{P16.37}
\]

P16 may report attained FP64 residuals.  A rigorous rounding envelope for
`rbar_fp64` is deferred to P17.

## 7. Fidelity is a separate acceptance gate

An interval proof that two gains differ establishes **algebraic spectral
noncollapse**, but it does not by itself establish meaningful Muon behavior:
an operator can have distinct gains while being arbitrarily close to `cS`.

For `S!=0` and any output `T`, define its best scalar fit and scale-invariant
nonscalar fraction by

\[
\alpha_T=\frac{\langle T,S\rangle_F}{\lVert S\rVert_F^2},
\qquad
\chi(T;S)=\frac{\lVert T-\alpha_T S\rVert_F}{\lVert T\rVert_F}.
\tag{P16.38}
\]

Let

\[
T_{\rm Muon}(S)=\mathcal H_h\!\left(
  \frac{S}{\lVert S\rVert_F+\epsilon}\right)
\tag{P16.39}
\]

be the exact-real upstream-polynomial comparator.  On the pre-existing,
balanced canonical input `S_star=diag(3,4)` at `epsilon=10^-7`, P16 should
report

\[
\mathcal R_{\rm shape}
=\frac{\chi(Y(S_\star);S_\star)}
       {\chi(T_{\rm Muon}(S_\star);S_\star)}.
\tag{P16.40}
\]

The two fidelity decisions are intentionally separate:

1. **Algebraic gate:** an exact or outward-rounded interval proves
   `y_1/3 != y_2/4`.
2. **Meaningful-direction gate:** require
   `R_shape>=1/10`, meaning that the proposed operator retains at least ten
   percent of Muon's departure from its best scalar fit on the locked
   comparator.

The second threshold is benchmark-relative and invariant to a global output
rescaling.  P16 must also report distances to `500*S`, `750*S`, `1000*S`, the
best scalar fit, and the gain spread.  Passing only the first gate must be
reported as “distinct but effectively scalar,” and triggers the requested
`(lambda,mu)` stability--fidelity frontier rather than a success claim.

Pre-certificate FP64 diagnostics make this distinction material.  At the
locked `lambda=10^-3`, `mu=1000`, `epsilon=10^-7`, and `S_star`, they give
approximately

\[
\frac{y_1}{3}=760.207725,
\qquad
\frac{y_2}{4}=760.217036,
\tag{P16.41}
\]

so the gains appear unequal, but

\[
\chi(Y;S_\star)\approx5.88\times10^{-6},
\quad
\chi(T_{\rm Muon};S_\star)\approx6.997\times10^{-2},
\quad
\mathcal R_{\rm shape}\approx8.4\times10^{-5}.
\tag{P16.42}
\]

These decimals are diagnostic; the canonical P16 artifact separately encloses
the exact output by evaluating an exact-binary candidate's graph residual with
Arb and applying P15's factor-`500` residual-to-output bound.  The locked point
computes reliably but fails the meaningful-fidelity gate.  That is a valid P16
finding rather than a reason to weaken the gate.

## 8. Cost and claim boundary

After one solve SVD, every Newton iteration evaluates five scalar quintic
stages per active singular value and performs `O(k)` diagonal/rank-one
algebra.  It does not solve a dense `k x k` system.  Reconstruction requires
scaling the singular vectors and multiplying them once.  The guarded FP64
reference then reevaluates `B` on that stored reconstructed matrix, requiring
a second SVD and a second reconstruction multiplication, so its literal
full-coordinate P15 residual is
`S-U_hat-lambda B(U_hat)`.  The implementation report must separate these
SVD/reconstruction costs from the scalar iteration cost and compare the
operation structure against the fifteen dense matrix multiplications used by
five implemented Jordan quintic stages.

The final artifact must report the complete iteration-count distribution,
backtracking counts, worst accepted residual, zero and near-zero behavior,
the P11 signal guard, and every declared failure.  Sampling validates the
implementation only; it does not prove the equivariance, residual, or global
convergence theorems above.

P16 does not cover BF16, certified FP64 rounding, stochastic gradients,
weight decay, aspect scaling, neural-network losses, or literal upstream
Muon.  Its positive theorem is an exact-real, structured, globally convergent
reference solve.  Its fidelity result must be classified independently as a
pass or a locked-parameter obstruction.
