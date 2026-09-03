# Robust-dissipativity certificate for disturbed smooth-PL dynamics

This is an artifact proof/result note, not a paper draft. All operator
evaluations are in exact real arithmetic. The exact certificate is a pathwise
input--output inequality; stochastic conclusions follow by taking conditional
expectations and do not replace that deterministic statement.

## Statement and scope

Keep the P6 repaired max-floored five-step Jordan operator

\[
R(M)=\mathcal H_{q^{\circ5}}
\!\left(\frac{M}{\max\{1,\lVert M\rVert_F\}}\right)+\rho M,
\]

where

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

The normalizer is the exact Frobenius max floor with `c=1`; there is no
additive epsilon. The displayed polynomial is composed for exactly five
Newton--Schulz steps. The constant shift equals the certified deficit upper
bound plus repair margin `648`.

Let `f:R^(m x n)->R` be fixed and differentiable, have finite infimum, have
globally `L=10`-Lipschitz gradient, and satisfy the global PL inequality with
constant `ell_PL=1`:

\[
\frac12\lVert\nabla f(W)\rVert_F^2
\ge f(W)-f_\star
\quad\text{for every }W.
\]

For every finite initialization `(W_0,m_0)` and every sequence of finite,
same-shaped disturbance matrices, use the same noisy gradient in both places
in the pinned EMA/Nesterov ordering:

\[
\begin{aligned}
g_t&=\nabla f(W_t), & \widetilde g_t&=g_t+\xi_t,\\
m_{t+1}&=\beta m_t+(1-\beta)\widetilde g_t,
&s_{t+1}&=\beta m_{t+1}+(1-\beta)\widetilde g_t,\\
W_{t+1}&=W_t-\eta\bigl(R(s_{t+1})+e_t\bigr),
&\beta&=\frac{19}{20},\quad \eta=\frac1{32000}.
\end{aligned}
\]

Here `xi_t` is additive gradient-oracle error and `e_t` is additive error at
the output of the repaired orthogonalizer. The latter does not yet constitute
a BF16 theorem: a concrete backend would additionally need a uniform or
probabilistic bound relating its complete implementation to this exact `R`.

With the same storage `V` as P6, the exact `6 x 6` rational certificate proves
for every finite matrix shape and every sample path

\[
\boxed{
V_{t+1}\le
\frac{399960001}{400000000}V_t
+\frac12\lVert\xi_t\rVert_F^2
+\frac1{2000000}\lVert e_t\rVert_F^2.}
\]

This is the primary P7 result. It is a robust dissipation inequality and gives
input-to-storage and input-to-output stability. Because the PL class may have
an unbounded non-singleton minimizer set, it is not full-state ISS in `W`.

## 1. Disturbed normalized dynamics

Use the P6 decomposition

\[
R(s)=\gamma s+\mathcal E(s),\qquad
\gamma=\frac{505021761888849}{520522720000},\qquad
\operatorname{Lip}(\mathcal E)\le
K_E=\frac{251582619905461}{520522720000}.
\]

Since `R(0)=0`, also `E(0)=0`. Normalize

\[
F=\frac{f-f_\star}{L},\quad
u=\frac{\nabla f(W)}L,\quad z=\frac mL,\quad
w=\frac\xi L,\quad h=\frac e{\gamma L},
\]

and set

\[
v=\frac{\mathcal E(Lp)}{K_EL},\qquad
r=\frac{K_E}{\gamma},\qquad
\alpha=\eta\gamma L.
\]

Then `||v||_F<=||p||_F` and the exact disturbed dynamics are

\[
\begin{aligned}
z_+&=\beta z+(1-\beta)(u+w),\\
p&=\frac{s_+}{L}
=\beta^2z+(1-\beta^2)(u+w),\\
d&=W_+-W=-\alpha(p+rv+h).
\end{aligned}
\]

The interpolation supplies below use the **true** gradients `u` and `u_+`.
Noise occurs only in the dynamics selectors for `z_+`, `p`, and `d`; treating
`u+w` as the gradient of `F` would be invalid.

## 2. Lifted exact LMI

Let

\[
\chi=(z,u,v,u_+,w,h).
\]

Retain exactly the P6 quadratic storage `P`, function-value coefficient,
directed smooth-nonconvex interpolation multipliers, next-state PL multiplier,
residual-Lipschitz multiplier, and rate

\[
\bar q=\left(\frac{19999}{20000}\right)^2
=\frac{399960001}{400000000}.
\]

Rebuild those supplies using the disturbed selectors above, and subtract the
two input penalties

\[
50\lVert w\rVert_F^2
+\frac{\gamma^2L^2}{2000000}\lVert h\rVert_F^2.
\]

The second normalized penalty is exactly

\[
\frac{\gamma^2L^2}{2000000}
=\frac{255046979981317296276230544801}
{5418878040723968000000000000}.
\]

The resulting symmetric rational matrix `M_6` is strictly negative definite.
The leading principal minors of `-M_6` are positive; for readability they are
approximately

\[
1.6567568\!\times10^{-2},\quad
7.8599846\!\times10^{-5},\quad
4.2583205\!\times10^{-7},\quad
3.7443246\!\times10^{-11},\quad
6.3159413\!\times10^{-10},\quad
8.1705846\!\times10^{-9}.
\]

The committed rational numerators and denominators, not these decimals, are
authoritative. Exact function-value cancellation is unchanged from P6. Thus

\[
\begin{aligned}
V_+-\bar qV
&+\lambda_{12}I_{12}+\lambda_{21}I_{21}
+\lambda_{\rm PL,+}J_{\rm PL,+}+\lambda_EJ_E\\
&-50\lVert w\rVert_F^2
-\frac{\gamma^2L^2}{2000000}\lVert h\rVert_F^2
=\langle\chi,(M_6\otimes I)\chi\rangle_F\le0.
\end{aligned}
\]

All four supplies are nonnegative. Finally,
`50||w||^2=||xi||^2/2` because `L=10`, and the normalized `h` term equals
`||e||^2/2000000`, proving the boxed physical-units inequality.

## 3. Deterministic consequences

For arbitrary inputs, iteration gives the convolution bound

\[
V_t\le \bar q^tV_0+
\sum_{k=0}^{t-1}\bar q^{t-1-k}
\left(\frac12\lVert\xi_k\rVert_F^2
+\frac1{2000000}\lVert e_k\rVert_F^2\right).
\]

If `||xi_t||_F<=X` and `||e_t||_F<=E`, then

\[
\limsup_{t\to\infty}V_t
\le \frac{200000000}{39999}X^2
+\frac{200}{39999}E^2.
\]

Since the P6 storage obeys

\[
V_t\ge \frac{13533}{50000}
\frac{f(W_t)-f_\star}{10},
\]

this also gives an explicit ultimate function-value neighborhood. Storage
coercivity similarly bounds normalized momentum and true-gradient energy.

If

\[
\sum_t\bigl(\lVert\xi_t\rVert_F^2+
\lVert e_t\rVert_F^2\bigr)<\infty,
\]

then `V_t->0`, `sum_t V_t<infinity`, the function gap vanishes, and both
`m_t` and `grad f(W_t)` tend to zero. This assumption alone does **not** imply
that `W_t` converges to one point. At a flat minimizer of
`f(x,y)=x^2/2`, the square-summable but not absolutely summable output errors
`e_t=(0,1/(t+1))` produce harmonic drift in `y` while `V_t=0`.

If instead both disturbance sequences are absolutely summable, taking square
roots in the one-step inequality and summing the resulting stable convolution
shows that the updates are absolutely summable. Then `W_t` converges to some
trajectory-dependent global minimizer. No uniqueness follows.

## 4. Stochastic corollary

Let `(W_0,m_0)` be `F_0`-measurable with `E[V_0]<infinity`.  Let the current
state be measurable with respect to a filtration `F_t`, let `xi_t,e_t` be
`F_(t+1)`-measurable, and assume

\[
\mathbb E[\lVert\xi_t\rVert_F^2\mid\mathcal F_t]
\le\sigma_g^2,
\qquad
\mathbb E[\lVert e_t\rVert_F^2\mid\mathcal F_t]
\le\sigma_R^2.
\]

Taking conditional expectations of the pathwise certificate gives

\[
\mathbb EV_t\le \bar q^t\mathbb EV_0
+\frac{1-\bar q^t}{1-\bar q}
\left(\frac{\sigma_g^2}{2}
+\frac{\sigma_R^2}{2000000}\right).
\]

Consequently the expected objective gap has the same geometric transient and
an explicit bounded-second-moment neighborhood after multiplication by
`10/(13533/50000)`. The conventional oracle condition
`E[xi_t|F_t]=0` may additionally be imposed to interpret `sigma_g^2` as a
conditional variance bound, but is not needed for this bound: the underlying
certificate is pathwise and uses only second moments. Time-varying conditional
second-moment bounds can replace the constants; if their sum is finite, then
`E[V_t]->0` and `sum_t E[V_t]<infinity`.

## 5. Necessary qualifications

This certificate covers the specified repaired max-floor operator, exact
five-step rational polynomial, exact real arithmetic, fixed globally smooth
PL objective, pinned EMA/Nesterov ordering, gradient error reused in both
gradient occurrences, and additive post-operator output error. It is uniform
over arbitrary finite matrix shape.

It does not certify an additive-epsilon or exact-current normalizer, unrepaired
upstream Muon, BF16 error magnitude, errors internal to the EMA state update,
weight decay, aspect-ratio scaling, time-varying objectives, arbitrary biased
oracle models beyond the displayed energy bounds, almost-sure pointwise
iterate convergence under persistent noise, or complete neural-network
training. Sampled disturbed trajectories can falsify an implementation, but
the exact rational `6 x 6` replay is the proof.
