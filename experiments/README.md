# Experiments

Experiment order is gated:

1. exact and backend-specific executable witnesses;
2. spectral/matrix audit;
3. qualified diagonal matrix quadratics;
4. the completed full-matrix floored certificate;
5. the repaired stylized non-Nesterov quadratic IQC and matched rank-one
   boundary checks;
6. the pinned EMA/Nesterov state-and-signal ordering IQC, exact replay, and
   parity regression;
7. the P4 structure-aware full-step quadratic certificate;
8. the two P5 nonlinear certificates and changing-orientation falsification
   probes;
9. the P6 smooth-PL value certificate and nonconvex, rank-deficient,
   changing-orientation falsification probe;
10. the P7 robust dissipativity certificate and disturbed nonconvex-PL
    falsification probe;
11. the P8 exact fixed-`2 x 2` mixed-precision certificate and CPU diagnostic;
12. the P9 shape-parameterized exact mixed-precision replay and CPU
    native-matmul diagnostic, with realistic shapes enabled only explicitly;
13. the P10 finite-precision outer-shell replay, three-word compensated-master
    checks, raw-FP32 subtraction witness, and modest CPU diagnostic;
14. the P11 exact two-port implementation-margin replay, grid boundaries,
    coordinatewise-maximal frontier slices, and adjacent rejection controls;
15. the P12 exact-real additive-epsilon full-rectangular deficit certificate,
    exact rank-two lower witness, boundary controls, and independent directed-
    interval replay;
16. the P13 exact-real radial-passivation theorem, two-precision logarithmic
    magnitude evaluation, independent rational-log reconstruction, universal
    stiffness lower bound, and explicit-step negative controls;
17. the P14 exact-real Yosida theorem, exact smooth-PL LMI replay, independent
    rational reconstruction, sector-boundary witnesses, and direct versus
    under-regularized versus selected scalar controls;
18. the P15 exact-real inexact-resolvent theorem, sharp graph-residual error
    gains, relative-plus-absolute stopping rule, robust smooth-PL replay, and
    loose-tolerance controls;
19. the P16 exact-real equivariant singular-value solver theorem, guarded
    FP64 computed-residual study, exact/Arb noncollapse witness, and sampled
    stability--fidelity frontier;
20. the P17 exact-real shape-preserving gated-resolvent pointwise-sector
    theorem, two smooth-PL certificates, exact/Arb unsafe-band and canonical
    fidelity controls, and separate FP64 spectrum diagnostic;
21. the P18 exact-real ray-sector projection, useful-rate smooth-PL
    certificate, rational step--rate frontier, canonical Arb gates, and
    separate FP64 fidelity/amplitude diagnostic;
22. only after model-forward use of the logical master, aspect scaling, weight
    decay, and implementation-parity gates, a small matched neural-training
    sweep.

The floored Jordan architecture now has a valid global constant-`rho`
certificate, a stylized non-Nesterov quadratic theorem, and an exact rational
certificate for the pinned EMA/Nesterov ordering in real arithmetic. Even at
the numerical stationary repair-margin design, that generic P3 result's
complex-skew necessary boundary is about `9,923.44x` below its matched local
threshold. It is therefore retained as an appendix/proof of principle, not a
practical-stability headline. P4 and P5 subsequently give structure-aware
full-step quadratic and strongly-convex nonlinear results, while P6 retains
the full step for globally smooth PL objectives that may be nonconvex and have
nonunique minimizers. GPU work remains gated on the remaining robustness and
implementation-parity results and a predeclared use of the certified
architecture. P7 supplies the robust post-operator port; P8 and P9 instantiate
that port for proposed proof-reference kernels. P10 closes exact FP32
EMA/Nesterov and compensated logical-master residuals for one guarded
`4096 x 11008` CPU shell, but it does not specify how a model forward pass
consumes the three-word master or establish native accelerator/upstream
parity. P11 then computes exact additional gradient and deployed-operator
acceptance margins without asserting that a production implementation meets
them. P12 separately proves exact `1/epsilon` deficit scaling and a
dimension-uniform four-band upper for the continuous additive-epsilon
surrogate. At the pinned `1e-7` scale its required constant repair is
catastrophically large; this is not a certificate for the discontinuous BF16
backend. P13 proves a nonlinear radial correction can shrink output magnitude
to logarithmic growth, while a pair lower bound and exact scalar controls show
the `1/epsilon` stiffness and explicit-step obstruction remain. P14 replaces
direct evaluation by an exact resolvent: its epsilon-independent incremental
sector restores the full `eta=1/32000` smooth-PL guarantee. P15 permits an
oracle meeting the exact-real graph-residual rule at relative tolerance
`1/250`, retains the P14 rate, and gives an explicit ultimate bound for an
absolute residual floor. P16 supplies the missing exact-real structured
solver and a fail-closed FP64 reference implementation. Every declared case
passes the computed residual check, but the locked operator retains only
about `8.403e-5` of the upstream comparator's spectral shaping and therefore
fails the frozen pre-certificate meaningful-fidelity gate. Its six-point sampled
`(lambda,mu)` frontier finds no joint frozen-P14/fidelity pass; this is not a
global impossibility theorem. P16 provides neither a useful uniform
iteration-count bound nor certified FP64 rounding. P17 retains that resolvent
internally and gates between scaled `Y` and `E_(h,epsilon)(J(S))`. A global
origin-centered pointwise sector yields exact one-trajectory smooth-PL
convergence for a primary ceiling-`3/4`, divisor-`4096`, `eta=1/128000`
high-fidelity point and a secondary ceiling-`1/8`, divisor-`8192`,
full-`eta=1/32000` point. Both pass the unchanged canonical P16 fidelity
thresholds. This is not an incremental theorem and does not propagate P16
solver or finite-precision error through the gate. Each run
writes a self-contained JSON manifest and compact CSV
tables under `results/summaries/`; the JSON records inputs, operator details,
dtype, seed, software, hardware, and Git state.

Replay the CPU-only momentum result with:

```bash
uv run --locked python scripts/certify_momentum_stability.py \
  --output results/summaries/momentum_iqc_certificate.json
uv run --locked python scripts/certify_ema_nesterov_stability.py \
  --output results/summaries/ema_nesterov_iqc_certificate.json
```

The exact rational LMI and Jury signs carry the claims. The committed float64
trajectories are matched diagnostics, not sampled global certificates.
The first command studies the explicitly stylized non-Nesterov recurrence;
the second matches the pinned EMA state and Nesterov signal ordering after
substituting the repaired floored map and omitting weight decay. Neither P3
certificate is a BF16, nonquadratic, stochastic, or neural-network theorem;
the separate P5 and P6 certificates establish deterministic exact-arithmetic
nonlinear strongly-convex and smooth-PL cases, respectively.

Replay the P6 proof and its separate sampled diagnostic with:

```bash
uv run --locked python scripts/certify_pl_convergence.py \
  --output results/summaries/pl_convergence_certificate.json
uv run --locked python experiments/nonconvex/run_pl_falsification.py \
  --output results/summaries/pl_falsification.json
```

The first command replays the exact rational certificate. The second runs a
72-case CPU/float64 falsification grid containing negative curvature,
changing Hessian orientations, and nonunique minimizers. A sampled failure can
falsify the implementation; sampled passes do not prove the theorem.

Replay the P7 robust theorem, P8 fixed-shape theorem, P9 scalable theorem, and
P10 outer-shell theorem with their independent reconstructions:

```bash
uv run --locked python scripts/certify_robust_dissipativity.py \
  --output results/summaries/robust_dissipativity_certificate.json
uv run --locked python scripts/reconstruct_robust_dissipativity.py
uv run --locked python scripts/certify_mixed_precision.py \
  --output results/summaries/mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_mixed_precision.py
uv run --locked python scripts/certify_scalable_mixed_precision.py \
  --output results/summaries/scalable_mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_scalable_mixed_precision.py \
  --require-canonical
uv run --locked python scripts/certify_outer_loop_roundoff.py \
  --output results/summaries/outer_loop_roundoff_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_roundoff.py \
  --require-canonical
uv run --locked python scripts/certify_implementation_margin.py \
  --output results/summaries/implementation_margin_certificate.json
uv run --locked python scripts/reconstruct_implementation_margin.py \
  --require-canonical
uv run --locked python scripts/certify_additive_epsilon_deficit.py \
  --output results/summaries/additive_epsilon_deficit_certificate.json
uv run --locked python scripts/reconstruct_additive_epsilon_deficit.py \
  --require-canonical
uv run --locked python scripts/certify_radial_passivation_tradeoff.py \
  --output results/summaries/radial_passivation_tradeoff_certificate.json
uv run --locked python scripts/reconstruct_radial_passivation_tradeoff.py \
  --require-canonical
uv run --locked python scripts/certify_yosida_stability.py \
  --output results/summaries/yosida_stability_certificate.json
uv run --locked python scripts/reconstruct_yosida_stability.py \
  --require-canonical
uv run --locked python scripts/certify_inexact_yosida_robustness.py \
  --output results/summaries/inexact_yosida_robustness_certificate.json
uv run --locked python scripts/reconstruct_inexact_yosida_robustness.py \
  --require-canonical
uv run --locked python scripts/certify_equivariant_resolvent_solver.py \
  --output results/summaries/equivariant_resolvent_solver_certificate.json
uv run --locked python scripts/reconstruct_equivariant_resolvent_solver.py \
  --require-canonical
uv run --locked python scripts/certify_shape_preserving_resolvent.py \
  --output results/summaries/shape_preserving_resolvent_certificate.json
uv run --locked python scripts/reconstruct_shape_preserving_resolvent.py \
  --require-canonical
uv run --locked python scripts/certify_sector_projected_useful_rate.py \
  --output results/summaries/sector_projected_useful_rate_certificate.json
uv run --locked python scripts/reconstruct_sector_projected_useful_rate.py \
  --require-canonical
```

The P9--P18 generators and standard-library-only reconstructions carry their
exact claims. P11's adjacent controls establish rejection of the sufficient
certificate, not actual closed-loop instability. P12's older sampled epsilon
grid is discovery evidence only; its exact/interval generator carries the
global upper. P15's residual and base-operator evaluations are exact real;
its stopping rule is not a finite-precision solver claim. P16's exact/Arb
artifact proves algebraic noncollapse, while its FP64 solver and sampled
frontier are diagnostics and do not certify rounding error or a global
fidelity impossibility. P17's exact sector is pointwise and origin-centered,
not incremental; its exact-real stability theorem assumes the exact
resolvent, while its separate FP64 spectrum grid is diagnostic evidence only.
P18's ray projection is likewise pointwise rather than incremental. Its
selected useful-rate theorem assumes the exact resolvent, while the
computed-residual-checked FP64 diagnostics do not propagate solver or
rounding error through the nonlinear projection.

Replay the P16 guarded reference solver and fidelity diagnostic separately:

```bash
uv run --locked python experiments/resolvent/run_p16_solver_study.py \
  --output results/summaries/p16_solver_study.json
```

The default study runs nine modest dense cases and complete singular-value
vectors for four representative Transformer shapes through `4096 x 11008`.
It locks additive `epsilon=1e-7`, the five Jordan stages with coefficients
`6889/2000`, `-191/40`, and `4063/2000`, `lambda=1/1000`, `mu=1000`, and
KellerJordan/Muon source revision
`f98f1cacc0263b04290753e32be8d498c1efc806`.
The large cases do not allocate dense matrices, perform the corresponding
SVD, or benchmark an accelerator. A returned success proves only that the
computed FP64 candidate passes the computed P15 residual test. The exact-real
global convergence theorem and Arb fidelity enclosure are separate from this
diagnostic.

Replay the P17 shape-preserving spectrum and fidelity diagnostic separately:

```bash
uv run --locked python experiments/resolvent/run_p17_shape_preserving_study.py \
  --output results/summaries/p17_shape_preserving_study.json
```

The study evaluates both locked gates on the declared spectrum grid and
complete reduced singular-value vectors for representative Transformer
shapes. It records computed P15 residual checks, fidelity, amplitude, and
rank accumulation. These deterministic FP64 results are not a proof of the
dimension-uniform pointwise sector, a global fidelity extremum, or an exact
rounding envelope. The exact P17 generator and independent reconstruction
carry the smooth-PL theorem and canonical gate decisions.

Replay the P18 sector-projected fidelity and amplitude diagnostic separately:

```bash
uv run --locked python experiments/resolvent/run_p18_sector_projected_study.py \
  --output results/summaries/p18_sector_projected_study.json
```

The study retains the P17 broad grid, sampled operating annulus, realistic
reduced spectra, frozen departure/retention thresholds, and actual computed
P15 residual postcheck. It additionally records projection activity, raw
amplitude, effective-update gates, the unprojected control, and the
`K=1/100` control. These are deterministic FP64 diagnostics, not global
fidelity, inexact-solver, or finite-precision certificates.

The P9 separate CPU diagnostic is a native-matmul
falsification probe against a float64, non-exact target:

```bash
uv run --locked python \
  experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py \
  --output results/summaries/scalable_mixed_precision_diagnostic.json
```

That default diagnostic deliberately uses modest shapes. Pass
`--include-realistic-shapes` only for the explicitly requested, substantially
more expensive Transformer-shape run. Neither diagnostic mode proves the
theorem or establishes GPU/BLAS parity.

Replay the P10 CPU outer-shell diagnostic separately with:

```bash
uv run --locked python \
  experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py \
  --output results/summaries/finite_precision_outer_loop_diagnostic.json
```

The deterministic run covers modest `1 x 1`, `2 x 3`, and `8 x 16` shapes
and checks exact finite residual and `TwoSum` identities. It also replays the
actual-P9 ordinary-subtraction stalling witness. These checks can falsify the
operation graph; only the exact P10 generator and independent reconstruction
carry the full-shape storage claim.

The deployed BF16 pair is a separate, backend-specific executable check. It
does not use a Jacobian and does not support a universal BF16 claim. Record it
on each target backend with:

```bash
uv run --locked python scripts/record_bf16_witness.py \
  --output results/summaries/bf16_witness.json
```

The manifest includes the exact operation order, runtime coefficient and
epsilon representations, returned BF16 storage words, an observable matmul
rounding probe, upstream revision, and complete run provenance.
