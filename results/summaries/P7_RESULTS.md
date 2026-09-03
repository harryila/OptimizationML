# P7 robust-dissipativity results

This is a research-artifact results note, not a paper draft. The exact
rational certificate supplies the theorem. The CPU/float64 disturbed
trajectories are sampled falsification and implementation diagnostics only;
passing them cannot prove the global inequality.

## Outcome

P7 retains the P6 storage, objective class, contraction factor, and full step,
and adds arbitrary additive gradient and repaired-operator-output errors. For
every fixed differentiable scalar objective on any fixed finite real matrix
shape with finite infimum, globally `10`-Lipschitz gradient, and global PL
constant `1`, the exact certificate proves the pathwise inequality

\[
V_{t+1}\le
\frac{399960001}{400000000}V_t
+\frac12\lVert\xi_t\rVert_F^2
+\frac1{2000000}\lVert e_t\rVert_F^2.
\]

It holds at

\[
\beta=\frac{19}{20},\qquad \eta=\frac1{32000}
\]

for every finite disturbance realization. This meets the P7 full-step
deterministic robust-dissipativity gate. It is input-to-storage and
input-to-output stability for the objective gap, momentum, and true gradient;
it is not full-state ISS in `W` when the PL minimizer set is non-singleton.

## Locked operator and disturbed update

The exact-real repaired operator is

\[
R(M)=\mathcal H_{q^{\circ5}}
\!\left(\frac{M}{\max\{1,\lVert M\rVert_F\}}\right)+\rho M,
\]

with exact Frobenius max floor `c=1`, no additive epsilon, exactly five
Jordan-quintic steps,

\[
q(s)=\frac{6889}{2000}s-\frac{191}{40}s^3
     +\frac{4063}{2000}s^5,
\qquad
\rho=\frac{210177835339081}{260261360000}.
\]

The same noisy gradient is deliberately reused in both occurrences of the
pinned EMA/Nesterov ordering:

\[
\begin{aligned}
g_t&=\nabla f(W_t), & \widetilde g_t&=g_t+\xi_t,\\
m_{t+1}&=\beta m_t+(1-\beta)\widetilde g_t,
&s_{t+1}&=\beta m_{t+1}+(1-\beta)\widetilde g_t,\\
W_{t+1}&=W_t-\eta\bigl(R(s_{t+1})+e_t\bigr).
\end{aligned}
\]

Thus `xi_t` is physical-unit gradient-oracle error and `e_t` is physical-unit
additive error after `R` and before multiplication by `eta`. The exact proof
normalizes these as `w=xi/L` and `h=e/(gamma L)`, while its interpolation
supplies always use the true gradients rather than the noisy measurements.

The authoritative source/provenance commit for the exact certificate and
falsification snapshot is
`c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a`. The rational `6 x 6` LMI,
positive Sylvester minors of its negative, exact function-value cancellation,
and operator/source hashes are recorded in the machine-readable certificate.
A standard-library reconstruction rebuilds the complete algebra before
comparing that artifact.

## Deterministic and stochastic consequences

Iteration of the pathwise inequality gives

\[
V_t\le \bar q^tV_0+
\sum_{k=0}^{t-1}\bar q^{t-1-k}
\left(\frac12\lVert\xi_k\rVert_F^2
+\frac1{2000000}\lVert e_k\rVert_F^2\right),
\quad
\bar q=\frac{399960001}{400000000}.
\]

If `||xi_t||_F <= X` and `||e_t||_F <= E`, then

\[
\limsup_t V_t\le
\frac{200000000}{39999}X^2+
\frac{200}{39999}E^2.
\]

If the two squared disturbance norms are summable, then `V_t -> 0`, the
function gap tends to zero, and both momentum and the true gradient tend to
zero. This alone does not imply convergence of `W_t`; absolute summability of
the disturbances is a sufficient stronger condition for summable parameter
increments and convergence to some trajectory-dependent global minimizer.

For an adapted stochastic process with finite initial expected storage and
conditional second-moment bounds

\[
\mathbb E[\lVert\xi_t\rVert_F^2\mid\mathcal F_t]\le\sigma_g^2,
\qquad
\mathbb E[\lVert e_t\rVert_F^2\mid\mathcal F_t]\le\sigma_R^2,
\]

the pathwise result immediately yields

\[
\mathbb E V_t\le \bar q^t\mathbb E V_0+
\frac{1-\bar q^t}{1-\bar q}
\left(\frac{\sigma_g^2}{2}+
\frac{\sigma_R^2}{2000000}\right).
\]

This is the rigorous P7 bounded-second-moment stochastic corollary. Conditional
unbiasedness may be imposed for the usual stochastic-gradient interpretation,
but it is not needed for this energy bound because the underlying inequality
is pathwise. The result is an expected storage/function-gap neighborhood, not
almost-sure iterate convergence under persistent noise.

## Sampled falsification and the iterate boundary

At seed `20260903`, the CPU/float64 diagnostic ran `144` trajectories and
`17,280` disturbed updates on the analytic P6 projected-radial nonconvex PL
family. The grid crossed:

- matrix shapes `2 x 2` and `2 x 3`;
- full-rank and codimension-one projectors;
- transition scales `1e-3`, `1`, and `100`;
- radii `sqrt(2)` and `4` times the transition scale;
- zero and seeded-random initial momentum;
- deterministic bounded, seeded stochastic, and implementation-only errors.

All `144` cases completed with zero candidate inequality violations, zero
nonfinite values, and zero divergence flags. The largest sampled dissipation
excess was `-1.232679510606715e-12`. These observations test update ordering,
units, and storage accounting; they do not certify any untested objective,
dimension, disturbance, or arithmetic backend.

The same artifact includes a separate exact flat-minimizer construction. For
`f(x,y)=x^2/2`, start at a minimizer and inject only
`e_t=(0,1/(t+1))` along the flat direction. Then

\[
\sum_t\lVert e_t\rVert^2<\infty,
\qquad
V_t=0,
\qquad
y_t=-\eta\sum_{k<t}\frac1{k+1},
\]

so the objective, gradient, momentum, and storage remain zero while the
iterate drifts harmonically. This rules out a full-state or iterate-convergence
claim from square summability alone.

## Scope limits

P7 covers the repaired max-floor operator above in exact real arithmetic, a
fixed global smooth-PL objective, the displayed pinned EMA/Nesterov ordering,
arbitrary finite additive gradient errors reused in both gradient occurrences,
and additive post-operator output errors. It is dimension independent over
fixed finite matrix shapes.

It does not cover the unrepaired or exact-current-normalized operator,
additive-epsilon normalization, weight decay, aspect-ratio scaling,
time-varying objectives, errors inserted elsewhere in the optimizer, complete
neural-network training, or upstream BF16 execution. In particular, no bound
has yet been proved that embeds all errors from a concrete BF16 backend into
`e_t`; P7 is not itself a BF16 theorem. The disturbance gains are sufficient,
not claimed necessary or optimal. Independent human proof review remains
pending.

## Replay

```bash
uv run --locked python scripts/certify_robust_dissipativity.py
uv run --locked python scripts/reconstruct_robust_dissipativity.py --require-canonical
uv run --locked python experiments/nonconvex/run_robust_dissipativity_falsification.py
uv run --locked pytest -q tests/test_robust_dissipativity.py \
  tests/test_robust_dissipativity_cli.py \
  tests/test_robust_dissipativity_reconstruction.py \
  tests/test_robust_dissipativity_experiment.py \
  tests/test_robust_dissipativity_results.py
```

Machine-readable artifacts:

- `results/summaries/robust_dissipativity_certificate.json`
- `results/summaries/robust_dissipativity_falsification.json`
