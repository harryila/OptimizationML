# Radial passivation tradeoff (artifact note)

This is a proof and result record for the repository, not a paper draft.  It
constructs a nonlinear correction for the exact-real additive-epsilon
operator certified in P12 and records the explicit-step stiffness that the
construction cannot avoid.  Nothing in this note is a theorem for the
discontinuous BF16 implementation.

## Statement and scope

Fix a finite real matrix space with the Frobenius inner product and let

\[
E_{h,\epsilon}(M)=\mathcal H_h\!\left(
  \frac{M}{\lVert M\rVert_F+\epsilon}\right),
\qquad \epsilon>0,
\tag{P13.1}
\]

where `h` is five exact repetitions of

\[
q(x)=\frac{6889}{2000}x-\frac{191}{40}x^3
     +\frac{4063}{2000}x^5.
\tag{P13.2}
\]

The arithmetic is exact real arithmetic, the additive epsilon is outside the
Frobenius norm exactly as displayed, and there is no max floor.  Put

\[
\begin{aligned}
a&=-\frac{199437}{1250},\\
\Gamma&=-\frac{41528474059081}{260261360000},\\
\overline\delta_1=:U
  &=\frac{6602082433275499863}{41641817600000000},\\
t_0&=\frac{63}{10000},
&z_0&=\frac{t_0}{1-t_0}=\frac{63}{9937}.
\end{aligned}
\tag{P13.3}
\]

These are the exact active fourth-band constants from the frozen P12
full-rectangular certificate.  For `z>=0`, define

\[
\widehat d(z)=
\begin{cases}
U,&0\le z\le z_0,\\[3pt]
-\dfrac{a+\Gamma z}{(1+z)^2},&z\ge z_0,
\end{cases}
\qquad
p(z)=\int_0^z\widehat d(u)\,du.
\tag{P13.4}
\]

Finally, set

\[
G_\epsilon(0)=0,
\qquad
G_\epsilon(M)=p\!\left(\frac{\lVert M\rVert_F}{\epsilon}\right)
                 \frac{M}{\lVert M\rVert_F}
\quad(M\ne0).
\tag{P13.5}
\]

The main constructive conclusion is

\[
\boxed{E_{h,\epsilon}+G_\epsilon\text{ is globally monotone}}
\tag{P13.6}
\]

for every `epsilon>0` and every finite matrix shape.  The complementary
impossibility conclusion is that, in every shape containing the locked P12
rank-two pair, every globally Lipschitz correction `C` that passivates that
pair must satisfy

\[
\operatorname{Lip}(C)
\ge \frac{d_{\rm pair}}{\epsilon}
>\frac1\epsilon\frac{98823281}{625000}.
\tag{P13.7}
\]

Thus the radial correction removes the catastrophic *output magnitude* of a
constant conductance at ordinary signal norms, but it cannot remove the
`1/epsilon` worst-case differential stiffness.

## 1. P12's pointwise deficit as a radial majorant

At nonzero `M`, write

\[
r=\lVert M\rVert_F,
\qquad z=\frac r\epsilon,
\qquad t=\frac r{r+\epsilon}=\frac z{1+z}.
\tag{P13.8}
\]

P12 proves the global pointwise bound

\[
\operatorname{Sym}DE_{h,\epsilon}(M)
\succeq-\frac U\epsilon I
\tag{P13.9}
\]

at every radius.  On its fourth radius band, `t>=t0`, the stronger retained
bound is

\[
\operatorname{Sym}DE_{h,\epsilon}(M)
\succeq
\frac{1-t}{\epsilon}\bigl((1-t)a+t\Gamma\bigr)I
=\frac{a+\Gamma z}{\epsilon(1+z)^2}I.
\tag{P13.10}
\]

The first three P12 bands lie in `z<=z0`; using (P13.9) there and (P13.10)
afterward gives, for every nonzero input,

\[
\operatorname{Sym}DE_{h,\epsilon}(M)
\succeq-\frac{\widehat d(z)}\epsilon I.
\tag{P13.11}
\]

The two branches in (P13.4) join exactly because

\[
-\frac{a+\Gamma z_0}{(1+z_0)^2}=U.
\tag{P13.12}
\]

Moreover, `a<0`, `Gamma<0`, and

\[
\frac{d}{dz}\left[-\frac{a+\Gamma z}{(1+z)^2}\right]
=\frac{2a-\Gamma+\Gamma z}{(1+z)^3}<0
\quad(z\ge z_0),
\tag{P13.13}
\]

since

\[
2a-\Gamma=-\frac{41520717707831}{260261360000}<0.
\]

Therefore `d_hat` is positive, continuous, and nonincreasing on
`[0,infinity)`.  This is a majorant of the certified P12 pointwise deficit;
it is not asserted to equal the exact pointwise deficit.

## 2. Closed form of the radial correction

The primitive in (P13.4) is

\[
p(z)=
\begin{cases}
Uz,&0\le z\le z_0,\\[4pt]
Uz_0-\Gamma\log\!\left(\dfrac{1+z}{1+z_0}\right)
 +(a-\Gamma)\left(\dfrac1{1+z}-\dfrac1{1+z_0}\right),
 &z\ge z_0.
\end{cases}
\tag{P13.14}
\]

All algebraic coefficients are exact rationals.  Useful simplifications are

\[
Uz_0=\frac{41856817278490137}{41641817600000000},
\quad
a-\Gamma=\frac{6205081}{416418176},
\quad
\frac{1+z}{1+z_0}=\frac{9937(1+z)}{10000}.
\tag{P13.15}
\]

The logarithm is the only transcendental operation.  Formula (P13.14) is
continuous with derivative `d_hat` at `z0`.  Its asymptotic growth is

\[
p(z)=-\Gamma\log z+O(1),
\tag{P13.16}
\]

so the correction magnitude grows logarithmically rather than linearly.

## 3. Full-matrix derivative and global monotonicity

Regard the matrix space as a finite-dimensional real Hilbert space.  For
`M!=0`, let `P_M` be the orthogonal projector onto the radial direction
`M/||M||_F`.  Differentiating (P13.5) gives the complete tangent-space
decomposition

\[
DG_\epsilon(M)
=\lambda_{\rm radial}(z)P_M
 +\lambda_{\rm tangential}(z)(I-P_M),
\tag{P13.17}
\]

where

\[
\lambda_{\rm radial}(z)=\frac{\widehat d(z)}\epsilon,
\qquad
\lambda_{\rm tangential}(z)=\frac{p(z)}{\epsilon z}.
\tag{P13.18}
\]

Because `d_hat` is nonincreasing,

\[
\widehat d(z)
\le \frac1z\int_0^z\widehat d(u)\,du
=\frac{p(z)}z
\le U.
\tag{P13.19}
\]

Hence

\[
\frac{\widehat d(z)}\epsilon I
\preceq DG_\epsilon(M)
\preceq\frac U\epsilon I.
\tag{P13.20}
\]

For `z<=z0`, (P13.14) makes `G_epsilon(M)=(U/epsilon)M` exactly.  Thus the
map is Frechet differentiable at zero, is continuously differentiable across
`z=z0`, and satisfies

\[
DG_\epsilon(0)=\frac U\epsilon I,
\qquad
\operatorname{Lip}(G_\epsilon)=\frac U\epsilon.
\tag{P13.21}
\]

Combining (P13.11) and the lower half of (P13.20) yields

\[
\operatorname{Sym}D(E_{h,\epsilon}+G_\epsilon)(M)\succeq0
\tag{P13.22}
\]

at every input.  Integrating this symmetric-Jacobian inequality along an
arbitrary line segment proves (P13.6).  This is a full-matrix theorem: radial
and all tangential Frobenius directions are included, and the P12 spectral
bound already covers every active, divided-difference, and rectangular
null-side mode.

## 4. Certified magnitude reduction

The radial construction gives the exact identity

\[
\lVert G_\epsilon(M)\rVert_F
=p\!\left(\frac{\lVert M\rVert_F}{\epsilon}\right).
\tag{P13.23}
\]

Outward-rounded Arb evaluation of (P13.14) gives, at the pinned deployed
`epsilon=10^-7`,

\[
\begin{array}{c|c|c}
\lVert M\rVert_F & \lVert G_\epsilon(M)\rVert_F
 & (U/\epsilon)\lVert M\rVert_F\\ \hline
1
&[2571.857826470212145,\ 2571.857826470212146]
&1585445308.053868\ldots\\
\dfrac{13872266672489}{549755813888}
&[3086.959580254301425,\ 3086.959580254301426]
&40006343820.950377\ldots
\end{array}
\tag{P13.24}
\]

The second radius is the exact P11 signal guard
`25.233506080419829...`; `25.2335` is only its rounded display.  These are
norm evaluations of the proposed exact-real correction, not bounds on a BF16
training run.

## 5. Universal Lipschitz lower bound

Let `X1,Y1` be P12's exact swapped-diagonal pair at unit epsilon, put
`D1=X1-Y1`, and denote its exact pair deficit by

\[
d_{\rm pair}
=-\frac{\langle E_{h,1}(X_1)-E_{h,1}(Y_1),D_1\rangle_F}
        {\lVert D_1\rVert_F^2}
>\frac{98823281}{625000}.
\tag{P13.25}
\]

Its exact quotient is the frozen P12 fraction with SHA-256
`de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336`.
For general epsilon set `X_epsilon=epsilon X1` and
`Y_epsilon=epsilon Y1`.  Then

\[
E_{h,\epsilon}(X_\epsilon)=E_{h,1}(X_1),
\qquad
X_\epsilon-Y_\epsilon=\epsilon D_1,
\tag{P13.26}
\]

so this pair's deficit is exactly `d_pair/epsilon`.

Now let `C` be any globally `L_C`-Lipschitz correction for which
`E_(h,epsilon)+C` is monotone even just on this pair.  Monotonicity and
Cauchy--Schwarz give

\[
\begin{aligned}
\frac{d_{\rm pair}}\epsilon\lVert X_\epsilon-Y_\epsilon\rVert_F^2
&\le
\langle C(X_\epsilon)-C(Y_\epsilon),
        X_\epsilon-Y_\epsilon\rangle_F\\
&\le L_C\lVert X_\epsilon-Y_\epsilon\rVert_F^2.
\end{aligned}
\tag{P13.27}
\]

Canceling the nonzero squared norm proves (P13.7).  This argument assumes
neither radiality nor oddness of `C`.  Zero-padding transfers it to every
shape with `min(m,n)>=2`; it is not a scalar-shape lower bound.

Together, (P13.7) and (P13.21) show that the proposed repair's exact
Lipschitz constant is within

\[
\frac{U}{98823281/625000}-1
=0.270230608278\ldots\%
\tag{P13.28}
\]

of the universal strict lower witness.  “Near-minimal” here refers only to
worst-case Lipschitz stiffness, not pointwise output magnitude or a proved
exact optimum.

## 6. Scalar-quadratic explicit-step obstruction

The small output in (P13.24) does not make the correction compatible with the
previous explicit step.  Consider the scalar quadratic

\[
f(w)=\frac12w^2
\tag{P13.29}
\]

and the pinned exact-real EMA/Nesterov ordering

\[
\begin{aligned}
m_+&=\beta m+(1-\beta)w,\\
s_+&=\beta m_++(1-\beta)w,\\
w_+&=w-\eta R_\epsilon(s_+),
\end{aligned}
\qquad
\beta=\frac{19}{20},\quad \eta=\frac1{32000},
\tag{P13.30}
\]

where `R_epsilon=E_(h,epsilon)+G_epsilon`.  This objective is one-strongly
convex and one-smooth, hence lies inside the earlier globally ten-smooth,
PL-one objective class.

### 6.1 Exact local Jury calculation

The correction alone has origin slope `U/epsilon`.  If `R_epsilon` in
(P13.30) is first replaced by `G_epsilon`, set
`theta_G=eta U/epsilon`.  In state order `(w,m)`, the exact Jacobian is

\[
A_G=\begin{bmatrix}
1-\theta_G(1-\beta)(1+\beta)&
 -\eta(U/\epsilon)\beta^2\\
1-\beta&\beta
\end{bmatrix}.
\tag{P13.31}
\]

Its trace and determinant satisfy

\[
1+\operatorname{tr}A_G+\det A_G
=2(1+\beta)-\theta_G(1-\beta)(1+2\beta).
\tag{P13.32}
\]

The other two strict Jury expressions are positive for every `eta>0`.
Consequently the correction-only linearization is Schur stable exactly when

\[
0<\eta<\eta_G^{\rm crit}
:=\frac{2(1+\beta)\epsilon}
        {(1-\beta)(1+2\beta)U}.
\tag{P13.33}
\]

At `epsilon=10^-7`,

\[
\eta_G^{\rm crit}
=\frac{1082687257600}{63820130188329832009}
=1.696466701658312\ldots\times10^{-8}.
\tag{P13.34}
\]

The locked step `1/32000` is `1842.0638595...` times larger, and its exact
boundary expression is

\[
1+\operatorname{tr}A_G+\det A_G
=-\frac{191356452588259896027}{26650763264000000}<0.
\tag{P13.35}
\]

The monic characteristic polynomial is negative at `-1` by (P13.35) and
tends to positive infinity on the negative real axis.  It therefore has a
real eigenvalue below `-1`, proving local instability.  This isolates the
stiffness of the radial correction itself.

For the complete repaired operator,

\[
h'(0)=\left(\frac{6889}{2000}\right)^5
=\frac{15516041187205853449}{32000000000000000},
\quad
k_0=h'(0)+U
=\frac{66983030848166374889967883}
       {104104544000000000000000}.
\tag{P13.36}
\]

Replacing `U` by `k0` in (P13.33) gives

\[
\eta_R^{\rm crit}
=\frac{8120154432000000000}{1942507894596824871809068607}
=4.180242692751253\ldots\times10^{-9}.
\tag{P13.37}
\]

The locked step exceeds this threshold by `7475.6425157...`; its exact Jury
expression is

\[
1+\operatorname{tr}A_R+\det A_R
=-\frac{1942248049655000871809068607}
       {66626908160000000000000}<0.
\tag{P13.38}
\]

The positive origin slope of `E_(h,epsilon)` contributes to this full-operator
threshold, so (P13.37) must not be attributed solely to the correction.

### 6.2 Exact one-step witness

There is also a fully rational, noninfinitesimal one-step control.  At
`epsilon=10^-7`, take

\[
m_0=0,
\qquad w_0=\frac1{389025000}.
\tag{P13.39}
\]

Then the pinned ordering gives

\[
s_1=(1-\beta^2)w_0=\frac\epsilon{399},
\qquad
\frac{s_1}{s_1+\epsilon}=\frac1{400},
\tag{P13.40}
\]

which lies strictly inside the repair's linear region.  Therefore
`G_epsilon(s1)=U/399`.  Also `h(1/400)>0` exactly: writing
`q(x)=x(6889/2000-(191/40)x^2+(4063/2000)x^4)`, the quadratic in `x^2` has positive leading
coefficient and discriminant

\[
B^2-4AC=-\frac{2594691}{500000}<0,
\tag{P13.41}
\]

so every Jordan stage preserves the sign of a nonzero input.  It follows
without decimal arithmetic that

\[
\begin{aligned}
w_1
&=w_0-\eta\left(h(1/400)+U/399\right)\\
&<\left(1-\chi\right)w_0,\\
\chi
&:=\frac{\eta U(1-\beta^2)}\epsilon
=\frac{257481214897744494657}{53301526528000000}
=4830.653672976\ldots.
\end{aligned}
\tag{P13.42}
\]

Thus `w1<-(chi-1)w0` and

\[
\frac{f(w_1)}{f(w_0)}
>(\chi-1)^2>23325554.
\tag{P13.43}
\]

This is an explicit failure of the old step on the proposed repaired
additive-epsilon operator, not a claim that every initialization diverges or
that an implicit/resolvent update is unstable.

## 7. Interpretation and strict implementation boundary

P13 establishes two simultaneous facts.

1. A nonlinear radial correction makes the exact-real additive-epsilon map
   globally monotone and replaces linear-in-norm output growth by logarithmic
   growth.
2. Every Lipschitz correction capable of repairing the locked P12 witness
   retains `Omega(1/epsilon)` stiffness.  The proposed correction realizes
   essentially that lower endpoint, and the old explicit EMA/Nesterov step is
   consequently locally unstable even on a scalar quadratic.

The second fact motivates a smaller explicit step, an implicit/resolvent
treatment, or an operating-domain certificate.  It does not invalidate the
global monotonicity theorem.

All conclusions above concern (P13.1) with exact rational polynomial
coefficients and exact real arithmetic.  The pinned upstream implementation
at revision `f98f1cacc0263b04290753e32be8d498c1efc806` casts to BF16 before
the norm, uses backend-dependent BF16 stage arithmetic, and has a
discontinuous executable map.  P13 neither implements nor certifies that
backend.  In particular, its Jacobian proof, logarithmic correction, and
explicit-step thresholds cannot be transferred to literal BF16 Muon without
a separately specified kernel and rounding analysis.

The canonical machine artifact and its independent reconstruction replay the
exact algebra, the locked P12 witness hash, the Arb logarithm enclosures, the
Jury boundaries, and the one-step witness.  The accompanying human-audit
packet remains unsigned until an independent reviewer completes it.
