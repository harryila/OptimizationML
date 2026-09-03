# P6 PL-convergence results

This is a research-artifact results note, not a paper draft. The exact rational
certificate supplies the theorem. The CPU/float64 trajectories are sampled
falsification and implementation diagnostics only; passing them cannot prove a
global result.

## Outcome

For every fixed differentiable scalar objective on any fixed finite real matrix
shape with finite infimum, globally `10`-Lipschitz gradient, and global PL
constant `1` under

\[
\frac12\|\nabla f(W)\|_F^2 \ge f(W)-f^\star,
\]

the P6 certificate proves global linear function-value convergence and
geometric momentum decay for the repaired exact-real optimizer at

\[
\eta=\frac1{32000},\qquad
\tau=\frac{19999}{20000},\qquad
q=\tau^2=\frac{399960001}{400000000}.
\]

Specifically,

\[
f(W_t)-f^\star \le \frac{L V_0}{c_F}q^t,
\qquad m_t\to0.
\]

The step `1/32000` is the complete P5 full step, so this meets the declared
“major P6 result” gate. PL permits nonconvex objectives and non-singleton
minimizer sets. This is not an arbitrary-pair incremental-stability result and
does not assume or conclude a unique minimizer. Geometric summability of the
updates additionally makes each iterate trajectory converge to some
trajectory-dependent global minimizer.

## Locked optimizer and proof

The deterministic, exact-real update is the pinned EMA/Nesterov ordering

\[
\begin{aligned}
g_t&=\nabla f(W_t),\\
m_{t+1}&=\beta m_t+(1-\beta)g_t,\\
s_{t+1}&=\beta m_{t+1}+(1-\beta)g_t,\\
W_{t+1}&=W_t-\eta R(s_{t+1}),
\end{aligned}
\qquad \beta=\frac{19}{20},
\]

with

\[
R(M)=H_h\!\left(\frac{M}{\max\{1,\|M\|_F\}}\right)+\rho M,
\quad
\rho=\frac{210177835339081}{260261360000}.
\]

There is no additive epsilon. `H_h` uses exactly five Newton–Schulz steps with
the Jordan quintic coefficients

\[
(a,b,c)=\left(\frac{6889}{2000},-\frac{191}{40},\frac{4063}{2000}\right).
\]

The value storage combines a quadratic form in normalized momentum and
gradient with the objective gap:

\[
V_t=
\left\langle [m_t/L,\nabla f(W_t)/L],
(P\otimes I)[m_t/L,\nabla f(W_t)/L]\right\rangle_F
+\frac{13533}{50000}\frac{f(W_t)-f^\star}{L},
\]

where

\[
P=\begin{bmatrix}
14487/20000 & -637/20000\\
-637/20000 & 499/100000
\end{bmatrix}.
\]

The replay checks exact cancellation of the function-value terms, positivity
of both Sylvester minors of `P`, and positivity of all four Sylvester minors of
the negative `4 x 4` LMI. Thus `P` is positive definite and the LMI is strictly
negative definite by exact rational arithmetic. The argument is dimension
independent and uses smooth nonconvex interpolation plus the PL supply; it does
not infer the theorem from a numerical solver or sampled trajectories.

The exact proof artifact records generation/source commit
`a8f650f6c60dcbc5d2f83647fd367348f4c67548`. The completed P6 result and
documentation are frozen at checkpoint commit
`ef88d8f5b26148af0ec1ca70b506048938bf9bef`, annotated tag `p6-checkpoint`.
The source SHA inside the immutable artifact is not presented as the later
checkpoint SHA. P5 remains frozen at annotated tag `p5-checkpoint`, peeled
commit `a549fb4c206335ef9ec264524e0f581216b250d4`.

`scripts/reconstruct_pl_convergence.py` supplies a separately committed,
standard-library-only code path. It rebuilds both directed interpolation
matrices, the dynamics, storage, function-value cancellation, complete `4 x
4` LMI, and exact Sylvester minors before reading the canonical JSON, then
cross-checks every published exact field. This is an exact replay with unit
cross-checks, not a human proof audit.

## Sampled falsification

The locked diagnostic ran 72 deterministic CPU/float64 trajectories at seed
`20260902` for an analytic projected radial PL family. Its complete factorial
coverage was:

- shapes: `2 x 2` and `3 x 3`;
- projector ranks: full rank and codimension one;
- transition scales: `1e-4`, `1`, and `100`;
- radius multipliers: `0.5`, `sqrt(2)`, and `4`;
- initial momentum: zero and seeded random;
- updates per trajectory: `500`.

The objective family is globally PL with lower bound `1`, has gradient
Lipschitz upper bound `9` (and hence lies in the theorem's `10`-smooth class),
and admits Hessian eigenvalues down to `-3/8`. The codimension-one cases have a
nonunique minimizer set.

All 72 sampled cases changed Hessian orientation; 59 encountered negative
curvature and 36 used rank-deficient projectors. Sampled Hessian eigenvalues
ranged from approximately `-0.3750000000000017` to `9.00000000000002`, the
minimum sampled PL ratio was approximately `1.06023`, and the largest
normalized Hessian commutator was approximately `0.28324`.

No candidate implementation violation, analytic-objective-bound violation,
Lyapunov-rate violation, unresolved-state Lyapunov resurgence, nonpositive
storage value, divergence, or nonfinite value was observed. The maximum
sampled Lyapunov ratio was approximately `0.993884`, below the exact bound `q`,
and its maximum normalized rate excess was approximately `-0.00601609`.
Individual objective increases occurred in all 72 cases and are permitted: the
theorem contracts the composite storage, not the objective at every individual
step.

These observations stress the nonconvex, changing-orientation, and nonunique
regimes, but they are not global evidence. The exact interpolation/LMI replay
is authoritative for the theorem.

## Scope limits

The result does not establish arbitrary-pair incremental contraction or cover
local-only PL assumptions, stochastic gradients, time-varying objectives,
BF16 arithmetic, weight decay, aspect-ratio scaling, additive-epsilon or
exact-current normalization, unrepaired upstream Muon, or complete neural
network training. The independent code-path reconstruction and unit
cross-checks pass; an independent human proof audit remains pending.

## Replay

```bash
uv run --locked python scripts/certify_pl_convergence.py
uv run --locked python scripts/reconstruct_pl_convergence.py
uv run --locked python experiments/nonconvex/run_pl_falsification.py
uv run --locked pytest -q tests/test_pl_convergence.py \
  tests/test_pl_convergence_cli.py \
  tests/test_pl_convergence_reconstruction.py \
  tests/test_pl_experiment.py \
  tests/test_pl_results.py
```

Machine-readable artifacts:

- `results/summaries/pl_convergence_certificate.json`
- `results/summaries/pl_falsification.json`
