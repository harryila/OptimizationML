# Sector-projected useful-rate resolvent

## Locked operator and scope

Fix a finite real matrix space \(\mathbb R^{m\times n}\), with arbitrary
positive finite \(m,n\).  The additive normalizer is

\[
N_\epsilon(U)=\frac{U}{\lVert U\rVert_F+\epsilon},
\qquad \epsilon=10^{-7},
\]

and \(E_{h,\epsilon}=\mathcal H_h\circ N_\epsilon\) applies five stages of

\[
x\longmapsto x\left(\frac{6889}{2000}-\frac{191}{40}x^2
                 +\frac{4063}{2000}x^4\right).
\]

Let \(A_\epsilon\) be the exact-real P13 monotone radial repair, set
\(B_{\epsilon,1000}=A_\epsilon+1000I\), and retain the P14/P16 resolvent

\[
J=(I+10^{-3}B_{\epsilon,1000})^{-1},
\qquad Y(S)=1000(S-J(S)).
\]

Put \(X(S)=E_{h,\epsilon}(J(S))\).  P18 projects this shape channel along its
own ray:

\[
\alpha_K(S)=
\begin{cases}
\displaystyle\min\!\left\{1,
  \frac{K\langle X(S),S\rangle_F}{\lVert X(S)\rVert_F^2}\right\},
  &X(S)\ne0,\\[6pt]
0,&X(S)=0,
\end{cases}
\qquad Z_K(S)=\alpha_K(S)X(S).
\]

The locked design uses

\[
K=1,\qquad c=1024,\qquad \bar\theta=\frac34.
\]

For \(q=\lVert S\rVert_F^2\), \(\theta(q)\) is the P17 \(C^2\) quintic
smootherstep: it is zero for \(q\le1/4\), equals \(\bar\theta\) for
\(q\ge1\), and interpolates between them.  The optimizer sees

\[
T(S)=(1-\theta(q))\frac{Y(S)}{c}+\theta(q)Z_K(S).
\]

This section concerns the exact resolvent and exact real arithmetic.  The
sector below is origin-centred and pointwise, not incremental.  Neither the
gate derivative nor the derivative of \(\alpha_K\) is discarded; neither is
needed by the one-trajectory value--momentum certificate.

## Full-matrix sector projection

The P16 equivariance and rank-preservation theorem gives a common singular
basis for \(S\), \(J(S)\), and \(X(S)\).  Each positive singular mode of
\(X(S)\) has the sign of the corresponding mode of \(S\), because the locked
quintic factor is strictly positive and all five stages preserve sign.
Consequently

\[
\langle X(S),S\rangle_F\ge0.
\]

If the projection is inactive, its definition gives
\(\lVert X\rVert_F^2\le K\langle X,S\rangle_F\).  If it is active, then
\(\alpha_K=K\langle X,S\rangle_F/\lVert X\rVert_F^2\), and hence

\[
\lVert Z_K\rVert_F^2
=\alpha_K^2\lVert X\rVert_F^2
=K\alpha_K\langle X,S\rangle_F
=K\langle Z_K,S\rangle_F.
\]

The zero case is immediate.  Therefore, for every finite matrix shape,

\[
\boxed{\ \lVert Z_K(S)\rVert_F^2
       \le K\langle Z_K(S),S\rangle_F\ },
\]

which is exactly the pointwise disk sector \([0,K]\).  Since \(\alpha_K\)
is a nonnegative scalar, the operation preserves the spectral direction of
\(X\) whenever \(X\ne0\).  In particular, scale-invariant pure-shape scores
of the isolated shape channel are unchanged.

## Sector of the exposed interface

P14 gives \(Y\) the incremental sector \([500,1000]\) and \(Y(0)=0\), so
the pointwise sector of \(Y/c\) is

\[
[a,b]=\left[\frac{125}{256},\frac{125}{128}\right].
\]

For a fixed \(0\le\theta\le\bar\theta\), convexity of origin-centred sector
disks places the blend in

\[
[(1-\theta)a,\ (1-\theta)b+\theta K].
\]

Taking the interval hull over the complete gate range gives

\[
\boxed{
  m=\frac{125}{1024}
  \le T \le
  \frac{509}{512}=M
},
\]

where the notation denotes

\[
\langle T(S)-mS,\,MS-T(S)\rangle_F\ge0.
\]

Equivalently,

\[
\gamma=\frac{m+M}{2}=\frac{1143}{2048},\qquad
K_T=\frac{M-m}{2}=\frac{893}{2048},\qquad
\lVert T(S)-\gamma S\rVert_F\le K_T\lVert S\rVert_F.
\]

The exact condition ratio is \(M/m=1018/125=8.144\).  This improves the
P17 high-fidelity sector ratio by more than three orders of magnitude.

## Useful-rate smooth--PL certificate

Consider the pinned exact-real EMA/Nesterov recurrence

\[
\begin{aligned}
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}\nabla f(W_t),\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}\nabla f(W_t),\\
W_{t+1}&=W_t-\eta T(s_{t+1}).
\end{aligned}
\]

For every differentiable globally \(10\)-smooth objective satisfying the
global PL inequality with constant \(1\), the exact P6 value--momentum LMI
passes at

\[
\eta=\frac1{83},\qquad
\tau=\frac{999799}{1000000},\qquad
q_{18}=\tau^2=\frac{999598040401}{10^{12}}<1.
\]

One exact certificate is

\[
P=\begin{bmatrix}
97/125&-151/500\\
-151/500&17/100
\end{bmatrix},\qquad
c_F=1,\qquad
\omega=\frac{3459}{500},\qquad
\lambda_R=\frac9{250}.
\]

Exact rational Sylvester checks prove \(P\succ0\) and the \(4\times4\)
dissipation LMI is strictly negative definite.  Thus the same global
one-trajectory conclusion as P6 holds:

\[
\mathcal V_{t+1}\le q_{18}\mathcal V_t.
\]

The objective gap, gradient norm, and momentum converge geometrically, and
the iterates converge to some trajectory-dependent global minimizer.  PL
does not imply a unique minimizer, and this is not arbitrary-pair incremental
contraction.

The P14 exact-resolvent rate is \(q_{14}=249001/250000\).  The exact integer
comparison

\[
q_{18}^{10}<q_{14}
\]

proves that P18's objective half-life is less than ten times P14's.  The
readable values are approximately \(1724.0734\) iterations versus
\(173.1135\), a ratio of approximately \(9.9592\).  This narrowly passes the
predeclared useful-rate gate.

The global sector also supplies non-scale-invariant guards:

\[
\frac1{10}<m\le\frac{\lVert T(S)\rVert_F}{\lVert S\rVert_F}
\le M<1
\quad(S\ne0),
\]

and, at the selected step,

\[
\frac1{1000}<
\frac{125}{84992}
\le\eta\frac{\lVert T(S)\rVert_F}{\lVert S\rVert_F}
\le\frac{509}{42496}
<\frac1{80}.
\]

These gates prevent a nominal fidelity score from being obtained by driving
the exposed output or effective update to zero or to an unbounded scale.

## Predeclared \(1/50\) control and exact frontier

The initially requested \(\eta=1/50\) point cannot be justified from this
pointwise sector alone.  Let \(Q^\top=-Q\), \(Q^\top Q=I\) on a real
two-dimensional subspace and define the admissible boundary map

\[
T_0=\gamma I+K_TQ.
\]

It exactly saturates the centred sector disk.  On the valid test objective
\(f(W)=5\lVert W\rVert_F^2\), the EMA/Nesterov recurrence reduces to a
complex quadratic.  Its second Schur--Cohn margin is exactly

\[
-\frac{3768360579178620269}
       {1759218604441600000000000}<0,
\]

so one eigenvalue has modulus greater than one.  This is an impossibility
result for a theorem that uses only the locked pointwise sector.  It is not a
counterexample to the more structured spectral P18 map itself.

Following the predeclared fallback rule, an exact rational step--rate frontier
was replayed.  Every passing row has its own strictly feasible rational LMI;
the decimals are descriptive only.

| \(\eta\) | \(\tau\) | certified half-life | interpretation |
|---:|---:|---:|---|
| \(1/75\) | \(99995/100000\) | 6931.30 | larger step; misses rate gate |
| \(1/83\) | \(999799/1000000\) | 1724.07 | selected maximum-step gate pass |
| \(1/90\) | \(3999/4000\) | 1386.12 | exact frontier |
| \(1/95\) | \(9997/10000\) | 1155.07 | exact frontier |
| \(1/120\) | \(24987/25000\) | 666.31 | faster-rate alternative |
| \(1/150\) | \(4997/5000\) | 577.45 | smallest-step declared point |

This is a finite declared Pareto study, not a proof that no untested storage,
multiplier, or stronger structural constraint can improve a listed point.

## Fidelity and negative controls

The canonical `diag(3,4)` point is certified with outward-rounded Arb
arithmetic, including the P16 computed graph-residual allowance.  The sector
projection is inactive there.  Its best-scalar departure is approximately
`0.0362830`, upstream-shaping retention approximately `0.518570`, and output
norm approximately `0.385218` times the input norm.  Thus it passes the
unchanged P16 thresholds `chi >= 1e-3` and retention `>= 0.1` as well as the
new amplitude gates.

The deterministic operating-annulus and spectrum studies are falsification
and fidelity diagnostics, not continuous-domain certificates.  On the frozen
sampled annulus `3/4 <= ||S||_F <= 25`, all 2,176 informative cases pass both
P16 fidelity thresholds.  The minimum sampled departure is about `0.001859`
and the minimum sampled retention about `0.13025`.  Across the broad grid,
only 192 of 327 informative cases pass, because the passive fallback is
intentionally radial near the origin.  No global fidelity claim is made.

Two controls prevent misclassification:

1. Removing the sector projection restores a pointwise upper gain of about
   `565.44`, violating the amplitude and effective-update gates.
2. Setting `K=1/100` drives canonical retention below `0.04` and fails the
   declared annulus fidelity gate, even though a scale-invariant departure
   score alone can remain nonzero.

The computed FP64 solver calls check the P15 graph residual and are useful
implementation diagnostics.  The exact theorem still assumes the exact
resolvent.  Because the ray projection is nonlinear, P15's residual-to-Yosida
bound does not by itself certify the projected output.

## Not established

P18 does not establish an incremental sector for the projected interface,
global Muon fidelity, stability under an approximate resolvent, an FP64 or
BF16 rounding theorem, literal upstream-Muon parity, upstream learning-rate
or update-magnitude parity, weight decay, aspect scaling, stochastic-gradient
convergence, represented model-state semantics, accelerator throughput, or
neural-network training convergence.  Those require separate results.
