# Sector-shielded inexact resolvent

## Locked operator and scope

Fix any positive finite matrix dimensions \(m,n\) and equip
\(\mathbb R^{m\times n}\) with the Frobenius inner product. P19 retains the
complete P18 exact-real interface. In particular, it uses additive
normalization

\[
N_\epsilon(U)=\frac{U}{\lVert U\rVert_F+\epsilon},
\qquad \epsilon=10^{-7},
\]

and five stages of the Jordan polynomial

\[
x\longmapsto x\left(\frac{6889}{2000}-\frac{191}{40}x^2
                 +\frac{4063}{2000}x^4\right).
\]

Thus the locked textual values are `epsilon=1/10000000` (`1e-7`), five
stages, and coefficients `6889/2000`, `-191/40`, and `4063/2000`.

The exact P13 radial passivator, shunt \(\mu=1000\), resolvent parameter
\(\lambda=10^{-3}\), P17 quintic gate with ceiling \(3/4\), P18 ray
projection with \(K=1\), and passive divisor \(c=1024\) are unchanged.
P19 changes only the final interface: an arbitrary finite candidate
\(C(S)\) is projected into the exact P18 pointwise sector before it is
exposed to the optimizer.

This theorem permits an approximate, corrupted, history-dependent, or
prematurely terminated candidate. It does not require a graph-residual bound
for stability. It is a pointwise safety theorem, not an incremental-sector
or arbitrary-pair contraction theorem.

## Exact sector shield

The P18 sector endpoints are

\[
m_T=\frac{125}{1024},\qquad M_T=\frac{509}{512}.
\]

Write

\[
\gamma=\frac{m_T+M_T}{2}=\frac{1143}{2048},\qquad
r=\frac{M_T-m_T}{2}=\frac{893}{2048},
\]

and, for each signal \(S\), define the closed Frobenius ball

\[
\mathcal D_S=
\left\{U:\lVert U-\gamma S\rVert_F\le r\lVert S\rVert_F\right\}.
\]

For every finite candidate \(C\), the metric projection is

\[
\Pi_{\mathcal D_S}(C)=
\begin{cases}
0,&S=0,\\[3pt]
\gamma S+\min\!\left\{1,
\dfrac{r\lVert S\rVert_F}{\lVert C-\gamma S\rVert_F}\right\}
(C-\gamma S),&S\ne0,
\end{cases}
\]

where the multiplier is one when \(C=\gamma S\). P19 exposes

\[
T_{19}(S)=\Pi_{\mathcal D_S}(C(S)).
\]

The formula is total in exact real arithmetic. At \(S=0\), the feasible set
is the singleton \(\{0\}\), so every finite candidate maps to zero.

The ball slack is identically the P18 pointwise-sector supply:

\[
\begin{aligned}
&\left\langle U-m_TS,\,M_TS-U\right\rangle_F\\
&\quad=r^2\lVert S\rVert_F^2-
       \lVert U-\gamma S\rVert_F^2.
\end{aligned}
\]

Consequently, for every finite matrix shape, every signal, and every finite
candidate,

\[
\boxed{
\left\langle T_{19}(S)-\frac{125}{1024}S,\,
\frac{509}{512}S-T_{19}(S)\right\rangle_F\ge0.}
\]

This conclusion is independent of how \(C(S)\) was generated. In
particular, a bad approximate solve cannot escape the certified pointwise
sector after the shield.

## Identity and modular fidelity lemma

P18 proves globally that its exact output \(T_{18}(S)\) belongs to
\(\mathcal D_S\). Projection fixes every point in its set, hence

\[
\boxed{\Pi_{\mathcal D_S}(T_{18}(S))=T_{18}(S).}
\]

Thus P19 is exactly identical to P18 when the exact P18 candidate is
provided; it does not perturb the intended operator.

For each fixed \(S\), projection onto the nonempty closed convex set
\(\mathcal D_S\) is nonexpansive in the candidate:

\[
\lVert\Pi_{\mathcal D_S}(C_1)-\Pi_{\mathcal D_S}(C_2)\rVert_F
\le\lVert C_1-C_2\rVert_F.
\]

Taking \(C_2=T_{18}(S)\) gives the modular error statement

\[
\lVert T_{19}(S)-T_{18}(S)\rVert_F
\le\lVert C(S)-T_{18}(S)\rVert_F.
\]

The set itself changes with \(S\), so this is not joint nonexpansiveness in
\((S,C)\), nor an incremental property of the composite map. P15's graph
residual controls the resolvent root and Yosida output, but it does not alone
bound the final nonlinear P18 ray-projected candidate. Such an intervening
candidate-error bound is still required for a formal fidelity certificate;
the shield then cannot enlarge it.

## Recovered smooth--PL certificates

Consider the pinned real-arithmetic EMA/Nesterov recurrence

\[
\begin{aligned}
m_{t+1}&=\frac{19}{20}m_t+\frac1{20}\nabla f(W_t),\\
s_{t+1}&=\frac{19}{20}m_{t+1}+\frac1{20}\nabla f(W_t),\\
W_{t+1}&=W_t-\eta T_{19}(s_{t+1}).
\end{aligned}
\]

Let \(f\) be differentiable, globally \(10\)-smooth, bounded below, and
satisfy the global PL inequality with constant \(1\). The P18
value--momentum proof uses only the pointwise sector supply above. Therefore
it remains valid for every sequence of finite candidates, even if the
candidate rule varies with time.

At the maximum-step operating point,

\[
\eta=\frac1{83},\qquad
\mathcal V_{t+1}\le
\frac{999598040401}{10^{12}}\mathcal V_t.
\]

The replayed exact certificate uses

\[
P=\begin{bmatrix}
97/125&-151/500\\
-151/500&17/100
\end{bmatrix},\quad
c_F=1,\quad \omega=\frac{3459}{500},\quad
\lambda_R=\frac9{250}.
\]

Its certified Lyapunov-rate half-life is approximately \(1724.0734\)
iterations. This phrase refers to the geometric storage upper bound; it does
not assert that every observed objective value halves after exactly that
many optimizer steps.

P19 also replays the faster-rate P18 operating point

\[
\eta=\frac1{120},\qquad
\mathcal V_{t+1}\le
\frac{624350169}{625000000}\mathcal V_t,
\]

with

\[
P=\begin{bmatrix}
599/1000&-231/625\\
-231/625&657/2500
\end{bmatrix},\quad
c_F=1,\quad \omega=\frac{2177}{200},\quad
\lambda_R=\frac{53}{2500}.
\]

Its certified Lyapunov-rate half-life is approximately \(666.314\)
iterations. Exact rational Sylvester checks prove storage positivity and
strict negativity of both \(4\times4\) LMIs.

For either operating point, the objective gap, gradient norm, and momentum
converge geometrically, and the iterates converge to some
trajectory-dependent global minimizer. PL does not imply uniqueness, and no
arbitrary-pair contraction is claimed.

## Fail-closed binary64 reference shield

The reference implementation accepts binary64 NumPy arrays and applies the
same disk after an arbitrary candidate has been computed.

1. It evaluates norms with maximum scaling and balanced summation so that
   the projection construction does not square unscaled extreme entries.
2. If the stored candidate already belongs to the disk, exact dyadic
   arithmetic verifies that fact and the candidate is returned bit for bit.
3. An outside candidate is projected toward a radius rounded 32 binary64
   ULPs inward.
4. Every successful return is checked again by converting the stored
   binary64 entries to exact rational dyadics and evaluating the original
   P18 disk slack exactly.
5. If the rounded projection misses the disk, or if the candidate is
   nonfinite, the routine attempts the exact-checked interior update
   \(S/2\).
6. A nonfinite signal is rejected without an update. If no certified
   binary64 interior value is representable, the call also raises rather
   than returning an uncertified update.

The final clause is necessary. For example, at the smallest positive
binary64 subnormal signal, neither zero nor the same subnormal need lie in
the strict positive-gain sector, while halving underflows to zero. P19 does
not hide this representation obstruction behind a numerical tolerance.

This is a correctness-first CPU reference, not a portable theorem about all
FP64, BLAS, GPU, or compiler implementations. The exact stored-value
postcheck is intentionally expensive; a scalable shield belongs to P20.

The P18 FP64 ray projection now clips a rounded negative inner product to its
positive part, yielding the zero shape branch instead of raising. The final
P19 disk shield remains the authoritative safety check.

## Diagnostics and controls

The deterministic study evaluates the guarded P16/P18 reference path on the
canonical `diag(3,4)` case and the frozen P18 sampled operating annulus. On
all 2,688 annulus calls the shield is inactive and returns the P18 candidate
bit for bit. All 2,176 informative annulus cases retain the frozen fidelity
gates. The canonical best-scalar departure remains about `0.0362830`, and
upstream shaping retention remains about `0.518570`.

All 2,690 computed P15 graph-residual checks pass, with a worst residual of
about `1.356e-11` times its allowed threshold. These residuals quantify
root/Yosida fidelity ingredients; they are not promoted into a final
nonlinear P18 candidate-error theorem.

Exact rational radial and tangential corruption controls begin outside the
sector and project to its boundary. The numerical study adds zero, outward,
anti-aligned, large-skew, and nonfinite candidate controls. Every successful
shield return passes the exact stored-value disk check. Nonfinite signals
and unrepresentable subnormal outputs fail closed.

The annulus and fidelity numbers are deterministic sampled diagnostics, not
global fidelity extrema. Global safety and stability come from the exact
disk identity and rational LMIs, not from those samples.

## Not established

P19 does not establish an incremental sector, joint nonexpansiveness in the
signal and candidate, a final graph-residual-to-P18 fidelity bound, a
portable finite-precision proof for arbitrary backends, BF16 behavior,
literal upstream-Muon parity, aspect scaling, weight decay, represented
model-state semantics, stochastic-gradient convergence, accelerator
throughput, or neural-network training convergence. The original unrepaired
upstream Muon operator is not globally certified.
