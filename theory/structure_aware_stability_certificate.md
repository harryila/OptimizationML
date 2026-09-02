# Structure-aware EMA/Nesterov stability certificate

This is an artifact proof/result note, not a paper draft.  All theorem claims
below are in real arithmetic.

## Statement and exact scope

Let the ambient space be `R^(m x n)`, for arbitrary fixed finite positive
`m,n`, with the Frobenius inner product.  Define

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5,
\qquad h=q^{\circ 5},
\]

and let `H_h` be the rectangular singular-value map induced by the odd
polynomial `h`.  This note uses the max-floored normalizer

\[
N_c(M)=\frac{M}{\max\{c,\lVert M\rVert_F\}},
\qquad F_{h,c}=\mathcal H_h\circ N_c,
\]

not additive-epsilon normalization.  In particular, there is no epsilon
parameter in the theorem.  The repaired operator is

\[
R(M)=F_{h,c}(M)+\rho M,
\qquad
\rho=\frac{\bar\delta_1}{c}+\mu,
\]

where

\[
\bar\delta_1=\frac{41528474059081}{260261360000},
\qquad \mu>0.
\]

The locked result takes

\[
c=1,\quad \mu=648,\quad
\rho=\frac{210177835339081}{260261360000},
\quad \ell=1,\quad L=10,\quad
\beta=\frac{19}{20},
\]

and analyzes the deterministic quadratic

\[
f(W)=\frac12\langle W-W_\star,
H(W-W_\star)\rangle_F,
\qquad I\preceq H\preceq 10I.
\]

Here `H` is an arbitrary fixed Frobenius-self-adjoint Hessian on the
vectorized matrix space; it need not preserve singular-vector modes.  For the
pinned EMA/Nesterov ordering

\[
\begin{aligned}
m_{t+1}&=\beta m_t+(1-\beta)H(W_t-W_\star),\\
s_{t+1}&=\beta m_{t+1}+(1-\beta)H(W_t-W_\star),\\
W_{t+1}&=W_t-\eta R(s_{t+1}),
\end{aligned}
\]

the exact rational replay at

\[
\eta=\frac1{32000},\qquad \tau=\frac{99999}{100000}
\]

proves that, for every such fixed finite matrix shape and Hessian, there is a
finite `C_H` such that any two trajectories obey

\[
\lVert (\Delta W_t,\Delta m_t)\rVert_F
\le C_H\tau^t
\lVert (\Delta W_0,\Delta m_0)\rVert_F.
\]

Thus the loop is globally incrementally exponentially stable and converges to
its unique equilibrium `(W_star,0)`.  The constant `C_H` may depend on the
fixed Hessian and on the equivalent finite-dimensional storage norm; the
certified rate `tau` and learning rate are common to the whole stated class.

## 1. Exact singular-vector tangent family

Put

\[
a=-\frac{199437}{1250}=-159.5496,
\qquad b=\frac{4848763}{10000}=484.8763.
\]

The earlier outward-rounded Arb proof establishes

\[
a<h'(s)<b\qquad(0\le s\le1)
\]

for the displayed exact five-stage Jordan composition.  Let
`Z=U diag(sigma) V^T`, with `p=min(m,n)`, and write
$A_Z=D\mathcal H_h(Z)$.  In the standard orthonormal singular-vector tangent basis,
`A_Z` is self-adjoint and has the following modes:

\[
\begin{array}{c|c}
\text{mode}&\text{slope}\\ \hline
u_i v_i^\top&h'(\sigma_i)\\[2mm]
(u_i v_j^\top+u_j v_i^\top)/\sqrt2&
\displaystyle\frac{h(\sigma_i)-h(\sigma_j)}
{\sigma_i-\sigma_j}\\[4mm]
(u_i v_j^\top-u_j v_i^\top)/\sqrt2&
\displaystyle\frac{h(\sigma_i)+h(\sigma_j)}
{\sigma_i+\sigma_j}\\[4mm]
u_\alpha^\perp v_i^\top\ \text{or}\
u_i(v_\alpha^\perp)^\top&
\displaystyle\frac{h(\sigma_i)}{\sigma_i}.
\end{array}
\]

The last row is present on the longer rectangular side.  Continuous values
are used at ties and zero: the difference quotient tends to `h'(sigma)`, the
sum quotient at two zeros and the null-side quotient at zero both tend to
`h'(0)`, and if exactly one of a pair is zero both off-diagonal quotients equal
`h(sigma)/sigma`.  These conventions also cover repeated and rank-deficient
matrices; the polynomial spectral map itself is smooth there.

Every displayed slope is a secant average of the even function `h'` on
`[-1,1]`.  Hence, on the full rectangular tangent space,

\[
aI\preceq A_Z\preceq bI,
\qquad \lVert A_Z\rVert\le b.
\]

This is a full-matrix statement.  It is not inferred from a diagonal-only
certificate.

## 2. Floor, radial coupling, and the Clarke boundary

Let `r=||M||_F`.  Away from the switching sphere,

\[
DN_c(M)=
\begin{cases}
c^{-1}I,&r<c,\\[1mm]
r^{-1}Q_Z,&r>c,
\end{cases}
\qquad
Q_Z=I-Z\otimes Z,
\]

where `Z=M/r` in the second line and
`(Z tensor Z)[E]=<Z,E>_F Z`.  At `r=c`, with `Z=M/c`, the exact Clarke family
is

\[
\partial_C N_c(M)=
\left\{\frac1c(I-\theta Z\otimes Z):0\le\theta\le1\right\}.
\]

Accordingly, at `r<c` the repaired derivative is simply
`rho*I+A_Z/c`; every tangent in Section 1 is an independent real scalar mode
there.

Since the outer polynomial map is continuously differentiable, the repaired
generalized Jacobians at the boundary are exactly

\[
G_\theta=\rho I+\frac1c A_Z(I-\theta Z\otimes Z),
\qquad 0\le\theta\le1.
\]

The endpoint `theta=0` is the inside limit and `theta=1` is the outside
limit.  Outside the floor the same formula holds with `theta=1` and `1/c`
replaced by `1/r`.

Only the active singular-value modes participate in the radial coupling.
Indeed, write

\[
Z=\sum_{i=1}^p\sigma_i u_i v_i^\top,
\qquad D=\operatorname{diag}(h'(\sigma_i)),
\qquad \sigma^\top\sigma=1
\]

on the switching sphere or outside it.  All symmetric, skew, and rectangular
null-side modes in the preceding table are orthogonal to `Z`; they remain
scalar modes of `G_theta`, with repaired slopes `rho+lambda*phi` for their
respective divided-difference or null-side slope `phi`.  On the active
diagonal block,

\[
G_\theta=\rho I+\lambda D(I-\theta\sigma\sigma^\top),
\]

where `lambda=1/c` on the boundary and `lambda=1/r` outside.  If

\[
\bar d=\sigma^\top D\sigma,
\qquad v=(I-\sigma\sigma^\top)D\sigma,
\qquad C=Q_\sigma DQ_\sigma|_{\sigma^\perp},
\]

then in the radial/tangential splitting
`span{sigma} direct-sum sigma-perp` this block is

\[
G_\theta=
\begin{bmatrix}
\rho+\lambda(1-\theta)\bar d&\lambda v^\top\\
\lambda(1-\theta)v&\rho I+\lambda C
\end{bmatrix}.
\]

For `theta<1`, the polynomial factor
`D(I-theta*sigma*sigma^T)` is similar to the self-adjoint matrix obtained by
putting a square root of `I-theta*sigma*sigma^T` on both sides of `D`.  At
`theta=1` its spectrum is `{0}` together with the spectrum of the
self-adjoint compression `C`.  Thus `G_theta` itself has only real
eigenvalues despite being nonsymmetric.

Thus the only nonsymmetric part is a rank-at-most-two radial/tangential
coupling.  At the outside endpoint the polynomial part vanishes on radial
inputs, while tangent inputs can still produce a radial output.  The coupling
vanishes for rank-one spectra and whenever the active values `h'(sigma_i)`
are equal.  For the locked constants, the generic sector extremizer would
have skew magnitude
`sqrt((rho+b)^2-648^2)`, which is strictly larger than the
structure-specific bound `(b-a)/4`; it is therefore not a Jacobian mode of
this operator.

## 3. Symmetric and skew derivative bounds

The projection/anticommutator lemma from the floored-normalizer certificate
gives, for every self-adjoint `A` with `aI <= A <= bI` and every orthogonal
projection `Q`,

\[
\Gamma I\preceq\frac{AQ+QA}{2}\preceq bI,
\qquad
\Gamma=-\frac{(b-a)^2}{8(a+b)}=-\bar\delta_1.
\]

The exact rational endpoints satisfy `a+b>0` and `b+3a>0`, the conditions
under which this is the minimizing branch of the projection envelope.

For completeness, if `x` is unit and `omega=||Qx||`, the only nonzero
eigenvalues of `Sym((Qx) tensor x)` are
`omega(omega+1)/2` and `omega(omega-1)/2`.  Assigning `a` to its positive
eigenspace and `b` to its negative eigenspace yields the lower envelope used
above.  Reversing those assignments gives

\[
\left\langle x,\frac{AQ+QA}{2}x\right\rangle
\le\frac{(a+b)\omega^2+(b-a)\omega}{2}\le b,
\]

which proves the upper bound.  The boundary family is the convex
interpolation between `A` and `AQ`, so it has the same symmetric bounds.

There is also a structure-specific skew bound.  With `P=Z tensor Z` and
`Q=I-P`,

\[
\operatorname{Skew}(A(I-\theta P))
=\frac{\theta}{2}(PA-AP).
\]

In the splitting `span{Z} direct-sum Z-perp`, put `v=QAZ`.  The commutator
has the block form

\[
PA-AP=\begin{bmatrix}0&v^\top\\-v&0\end{bmatrix},
\]

and hence has norm `||v||`.  If `alpha=<Z,AZ>`, then

\[
\begin{aligned}
\lVert v\rVert^2
&=\langle Z,A^2Z\rangle-\alpha^2\\
&\le(a+b)\alpha-ab-\alpha^2\\
&=(b-\alpha)(\alpha-a)
\le\frac{(b-a)^2}{4}.
\end{aligned}
\]

The first inequality follows by applying
`(A-aI)(bI-A) >= 0` to `Z`.  Consequently every ordinary or Clarke
generalized Jacobian `G` of `R` satisfies

\[
\begin{aligned}
\left(\rho+\frac{\Gamma}{c}\right)I
&\preceq\operatorname{Sym}G
\preceq\left(\rho+\frac bc\right)I,\\
\lVert\operatorname{Skew}G\rVert
&\le\frac{b-a}{4c}.
\end{aligned}
\]

For the locked repair these become

\[
648I\preceq\operatorname{Sym}G\preceq
\frac{336372400608849}{260261360000}I,
\qquad
\lVert\operatorname{Skew}G\rVert
\le\frac{6444259}{40000}.
\]

The skew estimate is where the radial spectral structure is retained; the
generic `(mu,K)` IQCs used on branch `p3` discard it.

## 4. A rigorously centered residual

Let

\[
M_R=\rho+\frac bc,
\qquad
\gamma=\frac{\mu+M_R}{2},
\qquad
d=\frac{M_R-\mu}{2},
\qquad
\kappa=\frac{b-a}{4c}.
\]

For every generalized Jacobian,

\[
\lVert G-\gamma I\rVert
\le \lVert\operatorname{Sym}G-\gamma I\rVert
   +\lVert\operatorname{Skew}G\rVert
\le d+\kappa.
\]

Define the centered residual only for analysis,

\[
\mathcal E(M)=R(M)-\gamma M.
\]

The locked exact center and residual bound are

\[
\gamma=\frac{505021761888849}{520522720000},
\qquad
K_E=d+\kappa
=\frac{251582619905461}{520522720000}.
\]

The map `R` is globally Lipschitz.  On every line segment it is absolutely
continuous; the switching sphere is harmless either by almost-everywhere
line integration or by the displayed Clarke bound.  Integrating the uniform
derivative estimate therefore proves the full pairwise inequality

\[
\lVert\mathcal E(X)-\mathcal E(Y)\rVert_F
\le K_E\lVert X-Y\rVert_F
\]

for all matrices `X,Y`.  This is a global upper certificate, not a sampled
lower-bound diagnostic.  The center `gamma` is an algebraic loop
decomposition; the implemented fixed repair remains `rho`, and no
input-dependent resistor is introduced.

## 5. Unconditioned difference loop and arbitrary Hessian orientation

For two trajectories put `w=Delta W`, `m=Delta m`, and `a_beta=1-beta`.
Their Nesterov-signal difference is

\[
s=\beta^2m+(1-\beta^2)Hw.
\]

Writing

\[
e=\mathcal E(s^{(1)})-\mathcal E(s^{(2)}),
\qquad \lVert e\rVert_F\le K_E\lVert s\rVert_F,
\]

gives the exact unconditioned Lur'e form

\[
\begin{aligned}
m_+&=\beta m+(1-\beta)Hw,\\
w_+&=w-\eta\gamma s-\eta e.
\end{aligned}
\]

Vectorize the Frobenius space and diagonalize the fixed Hessian as
`H=U Lambda U^T`.  The orthogonal change of variables by `U` leaves the
Euclidean/Frobenius norm unchanged, so the transformed residual remains a
single full-block `K_E`-Lipschitz uncertainty.  It may mix different
curvature eigenspaces; the proof does not assume otherwise.

The nominal linear plant is block diagonal.  On a curvature eigenmode
`lambda in [ell,L]`, with state `(w,m)`, it has

\[
A_\lambda=
\begin{bmatrix}
1-\eta\gamma(1-\beta^2)\lambda&-\eta\gamma\beta^2\\
(1-\beta)\lambda&\beta
\end{bmatrix},
\quad
B=\begin{bmatrix}-\eta\\0\end{bmatrix},
\quad
C_\lambda=\begin{bmatrix}(1-\beta^2)\lambda&\beta^2\end{bmatrix}.
\]

There is no direct feedthrough from `e_t` to `s_t`, so there is no algebraic
loop.  The transfer from `e` to `s` is

\[
G_\lambda(z)=
-\frac{\eta(1-\beta)\lambda((1+\beta)z-\beta)}
{z^2-[1+\beta-\eta\gamma(1-\beta^2)\lambda]z
+\beta[1-\eta\gamma(1-\beta)\lambda]}.
\]

Although the nonlinear residual can mix Hessian modes, the induced norm of
the nominal direct sum is exactly

\[
\lVert G_H\rVert_\infty
=\max_{\lambda\in\operatorname{spec}H}
\lVert G_\lambda\rVert_\infty.
\]

Full-block small gain therefore handles arbitrary Hessian orientation and
arbitrary cross-mode mixing by the transformed residual.  Treating the
residual as independent scalar uncertainties in each curvature mode would
not justify the full-matrix theorem.

## 6. Rate-scaled Jury and frequency certificate

To certify the rate `tau`, scale all difference signals by `tau^(-t)`.  The
scaled nominal plant has state matrices `A_lambda/tau`, `B/tau`, and transfer

\[
G_{\lambda,\tau}(z)=G_\lambda(\tau z).
\]

Although $\mathcal E$ is not homogeneous, its pairwise Lipschitz inequality
is preserved by this scaling: with
$\widehat e_t=\tau^{-t}e_t$ and $\widehat s_t=\tau^{-t}s_t$, the induced
time-varying incremental uncertainty still obeys
$\lVert\widehat e_t\rVert_F\le K_E\lVert\widehat s_t\rVert_F$ at every time.

Set `xi=eta*lambda` and define the two real coefficient polynomials

\[
A(\xi)=(1+\beta)[\gamma(1-\beta)\xi-1],
\qquad
B(\xi)=\beta[1-\gamma(1-\beta)\xi].
\]

Thus the transfer denominator and numerator are

\[
D(z,\xi)=z^2+A(\xi)z+B(\xi),
\qquad
N(z,\xi)=-(1-\beta)\xi[(1+\beta)z-\beta].
\]

The roots of `D` lie strictly inside the circle of radius `tau` exactly when
the three scaled Jury polynomials

\[
\begin{aligned}
J_+(\xi)&=D(\tau,\xi)=\tau^2+\tau A(\xi)+B(\xi),\\
J_-(\xi)&=D(-\tau,\xi)=\tau^2-\tau A(\xi)+B(\xi),\\
J_0(\xi)&=\tau^2-B(\xi)
\end{aligned}
\]

are all positive.

For `z=tau*exp(i omega)` and `u=1-cos(omega)`, the exact small-gain
gap is a quadratic in `u`:

\[
Q(u,\xi)
=|D(\tau e^{i\omega},\xi)|^2
-K_E^2|N(\tau e^{i\omega},\xi)|^2
=q_0(\xi)+q_1(\xi)u+q_2(\xi)u^2,
\]

where

\[
\begin{aligned}
q_0(\xi)={}&[\tau^2+\tau A(\xi)+B(\xi)]^2\\
&-K_E^2(1-\beta)^2\xi^2[(1+\beta)\tau-\beta]^2,\\[1mm]
q_1(\xi)={}&-2\tau A(\xi)[\tau^2+B(\xi)]-8\tau^2B(\xi)\\
&-2K_E^2(1-\beta)^2\beta(1+\beta)\tau\xi^2,\\[1mm]
q_2(\xi)={}&4\tau^2B(\xi).
\end{aligned}
\]

If

\[
q_2(\xi)>0,
\qquad
V(\xi)=4q_2(\xi)q_0(\xi)-q_1(\xi)^2>0,
\]

then `Q` has positive leading coefficient and negative discriminant, so it is
positive for every real `u`, in particular for `0<=u<=2`.  Once the scaled
Jury test holds, this proves the strict small-gain condition
`K_E ||G_(lambda,tau)||_infinity < 1`.

At the locked rational parameters,

\[
\xi=\eta\lambda\in
\left[\frac1{32000},\frac1{3200}\right].
\]

The executable audit checks the three affine Jury polynomials at exact
rational endpoints.  It converts the affine polynomial `q2` and the quartic
polynomial `V` to exact univariate Bernstein form on the displayed `xi`
interval.  Every Bernstein coefficient is strictly positive.  Since a
Bernstein polynomial lies in the convex hull of its coefficients, this proves
`q2>0` and `V>0` throughout the complete curvature interval.  Floating-point
optimizer status and frequency sampling are not proof inputs; the
machine-readable result records the exact power coefficients, Bernstein
coefficients, and Jury margins.

For each fixed finite `H`, choose one gain strictly between the certified
maximum nominal gain and `1/K_E`.  The strict discrete-time bounded-real
lemma supplies a storage for every scalar curvature block with that common
supply rate `gamma_BR<1/K_E`.  Taking their direct sum is valid even though
`e` mixes the blocks.  For some `epsilon_BR>0`, its strict dissipation
inequality has the form

\[
\widehat V_{t+1}-\widehat V_t
+\lVert\widehat s_t\rVert_F^2
-\gamma_{\rm BR}^2\lVert\widehat e_t\rVert_F^2
\le-\epsilon_{\rm BR}
\lVert(\widehat w_t,\widehat m_t,\widehat e_t)\rVert_F^2.
\]

Combining this with
$\lVert\widehat e_t\rVert_F\le K_E\lVert\widehat s_t\rVert_F$ makes the
direct-sum storage nonincreasing, with a strict margin.  Undoing the
`tau^(-t)` scaling gives the claimed `C_H tau^t` incremental bound.

## 7. Conservatism metrics and attainable local ceiling

The branch-`p3` generic `(mu,K)` EMA/Nesterov certificate used

\[
\eta_{p3}=\frac{65065340}{336372400608849}
=1.9343245724746989\times10^{-7}.
\]

The structure-aware locked value satisfies the exact ratio

\[
\frac{1/32000}{\eta_{p3}}
=\frac{336372400608849}{2082090880000}
=161.555100135\ldots.
\]

The improvement comes from retaining a known scalar center and bounding only
the residual full block, together with the rank-two radial skew bound.  It is
not evidence that the small-gain boundary is necessary for the Jordan map.

At the equilibrium the signal is zero, the floor is active, and every
operator tangent has the exact scalar gain

\[
g_0=\rho+\frac{h'(0)}c,
\qquad
h'(0)=\left(\frac{6889}{2000}\right)^5
=\frac{15516041187205853449}{32000000000000000}.
\]

The curvature-`L` rank-one subspace is an actual invariant subspace of the
full matrix loop.  Its EMA/Nesterov linearization loses Schur stability at

\[
\eta_{\rm local}
=\frac{2(1+\beta)}
{(1-\beta)(1+2\beta)Lg_0}
=\frac{8120154432000000000000000}
{3901919808117690731741568607}
=0.00208106645736\ldots.
\]

This is `66.5941266...` times the new locked learning rate.  It is an
attainable local upper ceiling for any global theorem covering the stated
class, whereas the complex-skew extremizer behind the earlier generic IQC is
not a tangent mode at zero.  Away from the floor, complex eigenvalues of the
effective spatial slope $H^{1/2}GH^{1/2}$ can only be generated by the
displayed radial/tangential coupling together with a Hessian that does not
commute with it; all off-diagonal divided-difference and rectangular-null
slopes of `G` remain real scalar modes.  (The two-state EMA recurrence can of
course have complex temporal roots even for a real scalar slope.)  Pointwise
mode stability alone still does not prove global incremental stability,
which is why the full-block small-gain argument is retained.

## 8. Limitations and nonclaims

- The theorem concerns the real-arithmetic repaired max-floored five-step
  Jordan map with the exact coefficients, iteration count, floor, and fixed
  repair stated above.  It is not a theorem for current-plus-epsilon
  normalization or for the upstream BF16 kernel.
- It matches the pinned EMA state and Nesterov signal ordering after replacing
  the upstream orthogonalizer by `R`.  It omits weight decay, transpose/aspect
  multipliers, distributed execution, and other optimizer-system details.
- The objective is a deterministic quadratic with a fixed self-adjoint
  Hessian in `[1,10]`.  No stochastic-gradient, nonquadratic, neural-network,
  or training-loss convergence claim follows.
- The Bernstein and Jury replay is a global sufficient upper certificate.
  Sampling could find a violation but would not certify the result.  The
  norm-bounded residual class contains time-varying, cross-mode uncertainties
  that need not be realizable by the linked Jordan map, so its boundary is not
  claimed to be exact.
- The derivative endpoints `a,b` are certified rational enclosures, not
  claimed exact extrema.  Likewise, the dimension-uniform radial variance
  bound need not be attained for a particular fixed matrix shape.
- `rho` and `gamma` are fixed constants.  `gamma` is only an analysis center;
  neither is fitted from the current input, so no omitted derivative-through-
  a-data-dependent-repair term is present.

## Reproduce

```bash
uv run --locked python scripts/certify_structure_aware_stability.py
uv run --locked pytest -q tests/test_structure_aware_stability.py
```
