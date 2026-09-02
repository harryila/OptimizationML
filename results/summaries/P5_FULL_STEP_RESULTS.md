# P5 full-step nonlinear stability results

This is a research-artifact result summary, not a paper draft. The exact proof
replay is authoritative; the float64 trajectories are falsification and
implementation-parity checks only.

## Outcome

The `p5-nonquadratic-stability` branch extends the p4 fixed-quadratic result to
every fixed differentiable globally `1`-strongly-convex, `10`-smooth scalar
objective on every finite real matrix shape. It retains the complete pinned
EMA/Nesterov update and the repaired max-floored five-step Jordan operator.

The new objective-interpolation storage certificate proves global exponential
convergence of each trajectory to the objective's unique minimizer at

\[
\eta=\frac1{32000}=\eta_{p4},
\qquad
\tau=\frac{2499}{2500}.
\]

This is the predeclared excellent-result case: the nonlinear theorem preserves
the full p4 step, rather than merely reaching `0.1 eta_p4`.

## Exact certificate

The proof writes the repaired operator as

\[
R(s)=\gamma s+\mathcal E(s),
\qquad \operatorname{Lip}(\mathcal E)\le K_E,
\]

and combines the full-matrix residual norm constraint with exact
smooth/strongly-convex interpolation inequalities between the minimizer,
current iterate, and next iterate. Its storage is

\[
V(w,z)=
\left\langle [w,z],(P\otimes I)[w,z]\right\rangle_F
+c_F\frac{f(W)-f(W_\star)}{L}.
\]

All entries, interpolation multipliers, objective-value flow cancellation,
and Sylvester minors of the `5 x 5` LMI replay with exact rational arithmetic.
The certificate is dimension independent and makes no common-Hessian-basis
assumption, so local Hessian orientations may change along the trajectory.
An independent automated exact-algebra reconstruction passed; an independent
human proof audit remains pending.

The scope distinction is essential:

- Full step, `eta=1/32000`: trajectory-to-minimizer global exponential
  convergence, using objective-gap/interpolation storage.
- Five percent of p4, `eta=1/640000`: the earlier common-quadratic certificate
  retains the stronger arbitrary-pair incremental contraction claim.

The quadratic edge supply in the new proof is not claimed as a standalone hard
IQC after its function-value terms are discarded.

## Falsification result

The locked CPU/float64 probe ran 36 deterministic trajectories:

- matrix shapes: `2 x 2`, `3 x 3`;
- log-cosh transition scales: `1e-4`, `1`, `100`;
- initial-radius multipliers: `0.25`, `1`, `4`;
- initial momentum: zero and seeded random;
- updates per trajectory: `500`.

All 36 cases exhibited changing sampled Hessian orientations. The maximum
normalized Hessian commutator was `0.20325940253416244`. The observed Hessian
eigenvalue range was `[1.0002533514820302, 10.000000000000021]`.

No candidate implementation violation, Lyapunov-rate violation, divergence, or
nonfinite value was observed. The maximum resolved one-step ratio was
`0.9799999799424988`, versus the certified bound
`tau^2=0.99920016`; the maximum excess was `-0.019200180057501237`.
Passing these samples is not evidence for global validity; the exact
interpolation/LMI replay supplies the theorem.

## Preserved checkpoints and limits

- P4 fallback: commit `c9636358de2d3d17bf5e62b0f03c7aff12da97bd`.
- First p5 arbitrary-pair certificate: code commit
  `f366af4345328fe66ffb269a4fe13ffe6d55d583`.
- Full-step p5 code/proof checkpoint:
  `7db9896c55bd131369e5e314a3e0175a9cf18c85`.
- Frozen final P5 checkpoint/tag target:
  `a549fb4c206335ef9ec264524e0f581216b250d4`.

The result does not cover stochastic or time-varying objectives, BF16,
weight decay, aspect-ratio scaling, additive-epsilon or exact-current
normalization, or complete neural-network training.

## Replay

```bash
uv run --locked python scripts/certify_nonquadratic_convergence.py
uv run --locked python experiments/quadratics/run_nonquadratic_convergence_falsification.py
uv run --locked pytest -q tests/test_nonquadratic_convergence.py \
  tests/test_nonquadratic_convergence_cli.py \
  tests/test_nonquadratic_convergence_experiment.py \
  tests/test_nonquadratic_convergence_results.py
```

Machine-readable artifacts:

- `results/summaries/nonquadratic_convergence_certificate.json`
- `results/summaries/nonquadratic_convergence_falsification.json`
