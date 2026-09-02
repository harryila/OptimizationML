# P5 first-stage incremental stability result

`p5-nonquadratic-stability` preserves p4 commit
`c9636358de2d3d17bf5e62b0f03c7aff12da97bd` as the fixed-quadratic fallback.

## Exact result

For every fixed finite matrix shape and every fixed differentiable objective
that is globally `1`-strongly convex and `10`-smooth, the deterministic
real-arithmetic EMA/Nesterov loop using

- the exact max floor `M/max(1, ||M||_F)`;
- five Jordan polynomial steps with exact coefficients
  `6889/2000`, `-191/40`, and `4063/2000`;
- constant repair
  `rho=210177835339081/260261360000`;
- `beta=19/20`; and
- `eta=1/640000`

is globally incrementally exponentially stable. A common `P tensor I`
storage contracts at rate `tau=99999/100000`. The proof uses the exact
gradient interpolation IQC and the full-matrix p4 decomposition
`R(s)=gamma*s+E(s)`, `Lip(E)<=K_E`; it does not freeze or diagonalize a
Hessian. Local Hessian orientations may therefore change along a trajectory.

The rational storage, multipliers, `4 x 4` LMI, and strict Sylvester replay are
in `nonquadratic_stability_certificate.json`. The selected learning rate is
exactly `0.05 eta_p4` and about `8.077755 eta_p3`, placing the result in the
predeclared **strong nonlinear extension** band.

## Search boundary and limitation

The discovery-only static-IQC search found a rate-one feasibility boundary
near `eta=1.69390537084e-6`, or `0.05420497 eta_p4`. The valid centered
residual inner-product IQCs were inactive and did not materially move that
boundary. This is not an impossibility theorem: dynamic or cyclic IQCs may
certify larger steps. This common-quadratic incremental certificate does not
retain the p4 step `1/32000`; the later objective-gap/interpolation certificate
does retain it for trajectory-to-minimizer convergence. See
`P5_FULL_STEP_RESULTS.md`.

## Falsification result

The deterministic CPU-float64 probe used an analytic rotating log-cosh
objective family with global Hessian bounds `[1,10]`. Across 36 paired runs:

- all 36 exhibited noncommuting, changing secant-Hessian orientations;
- no locked-storage rate violation was observed;
- no instability candidate, divergence, or nonfinite value was observed;
- the maximum sampled normalized secant commutator was
  `0.023782041249600067`; and
- the largest observed `V_(t+1)/V_t` was `0.9950025625800366`, below the
  locked `tau^2` by `0.004977437519963379`.

These samples audit the implementation and attempt falsification. They do not
prove stability; the exact common-storage LMI does.

## Excluded scope

No claim is made for exact-current or additive-epsilon normalization, BF16,
stochastic gradients, time-varying objectives, weight decay, aspect-ratio
scaling, or complete neural-network training.
