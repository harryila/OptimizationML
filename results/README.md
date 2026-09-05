# Results policy

Commit manifests, compact JSON/CSV summaries, and final figures. Keep raw
checkpoints, W&B caches, and large tensors out of Git. Every result must name
the operator, normalizer, epsilon, coefficients, steps, dtype, seed, hardware,
software versions, and Git SHA.

Current outputs:

- `summaries/canonical_witness.json`: exact/high-precision Jordan witness;
- `summaries/bf16_witness.json`: backend-specific executable Keller--Jordan
  BF16 pairwise witness with full run provenance (record separately on every
  claimed backend; returned values are not universal);
- `summaries/floored_repair_certificate.json`: two-precision Arb scalar
  enclosure, exact finite-pair lower witness, and dimension-uniform
  full-matrix floored-repair certificate;
- `summaries/momentum_iqc_certificate.json`: exact closed-form sector region,
  rational rate-LMI replay, exact local instability control, and matched
  deterministic rank-one trajectories for the stylized non-Nesterov loop;
- `summaries/ema_nesterov_iqc_certificate.json`: exact rational LMI replay for
  the pinned EMA/Nesterov state-and-signal ordering, numerical repair-margin
  design scan, exact local threshold, and matched rank-one diagnostics;
- `summaries/structure_aware_stability_certificate.json`: full-step,
  dimension-independent P4 certificate for the repaired floored operator in
  the pinned deterministic quadratic loop;
- `summaries/nonquadratic_stability_certificate.json`: P5 arbitrary-pair
  incremental certificate for globally `1`-strongly-convex, `10`-smooth
  objectives at `eta=1/640000`;
- `summaries/nonquadratic_convergence_certificate.json`: P5 objective-gap and
  interpolation-storage certificate for trajectory-to-minimizer convergence
  at the full P4 step `eta=1/32000`;
- `summaries/nonquadratic_falsification.json` and
  `summaries/nonquadratic_convergence_falsification.json`: qualified float64
  changing-orientation implementation probes for the two P5 certificates;
- `summaries/P5_RESULTS.md` and `summaries/P5_FULL_STEP_RESULTS.md`: scoped
  human-readable summaries of the incremental and full-step P5 results;
- `summaries/pl_convergence_certificate.json`: exact dimension-independent P6
  value--momentum certificate for global function-gap convergence and momentum
  decay on differentiable globally `10`-smooth objectives satisfying the
  global PL inequality with constant `1`, at `eta=1/32000`;
- `summaries/pl_falsification.json`: 72-case CPU/float64 diagnostic on an
  analytic nonconvex PL family, including negative curvature, changing Hessian
  orientations, and nonunique minimizers;
- `summaries/P6_RESULTS.md`: scoped human-readable P6 theorem and diagnostic
  readout, including the distinct certificate-source and final-checkpoint
  revisions; the standalone reconstruction entry point is
  `scripts/reconstruct_pl_convergence.py`;
- `summaries/robust_dissipativity_certificate.json`: exact
  dimension-independent P7 `6 x 6` pathwise certificate at `beta=19/20` and
  `eta=1/32000`, proving
  `V_next <= (399960001/400000000)*V + ||xi||_F^2/2 + ||e||_F^2/2000000`
  for the repaired exact-real max-floor five-step Jordan operator; its source
  revision is `c55d3e65fa2220f6a9e91c1a3d29b0cff04e3b8a`;
- `summaries/robust_dissipativity_falsification.json`: 144-case,
  17,280-update CPU/float64 disturbed nonconvex-PL diagnostic with zero
  candidate violations, plus the flat-minimizer harmonic-drift construction
  showing that square-summable errors alone do not imply iterate convergence;
- `summaries/P7_RESULTS.md`: scoped human-readable P7 theorem, deterministic
  and bounded-second-moment stochastic consequences, diagnostic readout, and
  strict exclusion of full-state ISS and a BF16 deployment claim;
- `summaries/mixed_precision_certificate.json`: exact fixed-`2 x 2` P8
  forward-error certificate for the proposed FP32 serial-Horner kernel with
  BF16 stage-boundary storage, including the affine operator-error bound,
  arbitrary-real input adapter, five-stage invariant, locked finite-input
  domain, and exact P7 error-to-objective closure; the independent
  reconstruction entry point is `scripts/reconstruct_mixed_precision.py`;
- `summaries/mixed_precision_falsification.json`: deterministic 82-case P8
  fixed-`2 x 2` CPU diagnostic against a float64, non-exact evaluation of the
  ideal formula; it is falsification evidence, not the proof;
- `summaries/P8_RESULTS.md`: concise scoped P8 theorem, exact replay, and
  diagnostic readout;
- `summaries/scalable_mixed_precision_certificate.json`: canonical replay
  target for the exact P9 shape-parameterized recurrence, seven locked
  Transformer shapes, affine operator-error bounds, and the resulting P7
  rates; generate it with `scripts/certify_scalable_mixed_precision.py` and
  independently check it with the standard-library-only
  `scripts/reconstruct_scalable_mixed_precision.py`;
- `summaries/scalable_mixed_precision_diagnostic.json`: output target for the
  P9 CPU/native-matmul falsification probe. Its default run uses modest shapes;
  realistic shapes are an explicit expensive opt-in, and neither mode is the
  proof reference;
- `summaries/P9_RESULTS.md`: scoped P9 theorem, exact recurrence/replay,
  serial-normalizer and one-term-proof obstructions, qualified frontier
  evidence, and deployment exclusions;
- `summaries/outer_loop_roundoff_certificate.json`: canonical P10
  exact replay for the proposed `4096 x 11008` FP32 EMA/Nesterov shell,
  concrete momentum/signal/master residual envelopes, three-word compensated
  master guards, port-augmented P7 closure, ordinary-FP32 stalling witness,
  and the strict rate `549700907325/549755813888`; generate it with
  `scripts/certify_outer_loop_roundoff.py` and independently check it
  with the standard-library-only
  `scripts/reconstruct_outer_loop_roundoff.py`;
- `summaries/finite_precision_outer_loop_diagnostic.json`: deterministic P10
  CPU operation-graph diagnostic over three modest shapes, including exact
  finite-entry residual, `TwoSum`, and logical-update identities plus the
  actual-P9 raw-subtraction stalling witness; it is falsification/parity
  evidence, not the global certificate;
- `summaries/P10_RESULTS.md`: scoped P10 rounding-envelope, compensated-master,
  port-closure, objective-neighborhood, stalling-obstruction, and deployment-
  exclusion summary;
- `summaries/implementation_margin_certificate.json`: canonical exact P11
  replay for the two affine discrepancy ports above the frozen P10 shell,
  including zero-error identity, four exact `2^-40`-grid axis maxima,
  coordinatewise-maximal slope/intercept frontier slices, adjacent rejection
  controls, and a jointly nonzero subunit-objective profile; generate it with
  `scripts/certify_implementation_margin.py` and independently check it with
  the standard-library-only `scripts/reconstruct_implementation_margin.py`;
- `summaries/P11_RESULTS.md`: scoped P11 margin table, guard closure, negative-
  control interpretation, P10 provenance roles, and deployment exclusions;
- `summaries/additive_epsilon_deficit_certificate.json`: canonical P12
  exact-real additive-Frobenius-epsilon certificate, including exact
  fixed-shape `1/epsilon` scaling, the four-band dimension-uniform full-
  rectangular upper, the exact rank-two lower witness, deployed-epsilon
  repair obstruction, upstream formula provenance, and exact-real/BF16 scope
  separation; generate it with
  `scripts/certify_additive_epsilon_deficit.py` and independently check it
  with the standard-library-only
  `scripts/reconstruct_additive_epsilon_deficit.py`;
- `summaries/P12_ADDITIVE_EPSILON_RESULTS.md`: scoped human-readable P12
  upper/lower bracket, repair consequence, boundary controls, max-floor
  comparison, and BF16 exclusions; the human proof audit remains pending and
  unsigned;
- `summaries/radial_passivation_tradeoff_certificate.json`: canonical P13
  exact-real radial repair certificate, including the closed-form logarithmic
  primitive, dimension-uniform full-matrix monotonicity argument, two-
  precision Arb magnitudes, universal rank-two Lipschitz lower bound, and
  exact explicit-step controls; generate it with
  `scripts/certify_radial_passivation_tradeoff.py` and independently check it
  with the standard-library-only
  `scripts/reconstruct_radial_passivation_tradeoff.py`;
- `summaries/P13_RADIAL_PASSIVATION_RESULTS.md`: scoped P13 constructive
  passivation, magnitude/stiffness tradeoff, scalar negative controls, and
  exact-real/BF16 exclusions; the human proof audit remains pending and
  unsigned;
- `summaries/yosida_stability_certificate.json`: canonical P14 exact-real
  implicit-operator certificate, including the full-matrix resolvent/Yosida
  identities, epsilon-independent `[500,1000]` incremental sector, exact
  smooth-PL value--momentum LMI at `eta=1/32000`, and scalar failure/pass
  controls; generate it with `scripts/certify_yosida_stability.py` and
  independently check it with the standard-library-only
  `scripts/reconstruct_yosida_stability.py`;
- `summaries/P14_YOSIDA_STABILITY_RESULTS.md`: scoped P14 existence, sector,
  exact convergence, boundary-control, and implicit exact-real exclusions;
  the human proof audit remains pending and unsigned;
- `summaries/inexact_yosida_robustness_certificate.json`: canonical P15
  exact-real residual-oracle certificate, including the sharp resolvent- and
  output-error gains, the locked relative-plus-absolute graph-residual rule,
  exact robust smooth-PL LMI, and loose-tolerance controls; generate it with
  `scripts/certify_inexact_yosida_robustness.py` and independently check it
  with the standard-library-only
  `scripts/reconstruct_inexact_yosida_robustness.py`;
- `summaries/P15_INEXACT_YOSIDA_ROBUSTNESS_RESULTS.md`: scoped P15 stopping-
  criterion theorem, retained full-step rate, absolute-residual objective
  neighborhood, controls, and solver/BF16 exclusions; the human proof audit
  remains pending and unsigned;
- `summaries/equivariant_resolvent_solver_certificate.json`: canonical P16
  exact/Arb artifact for the bi-orthogonal-equivariance and structured
  singular-value solver theorem, exact Jacobian margins, unequal-mode witness,
  canonical fidelity enclosure, and honest failed meaningful-fidelity gate;
  generate it with `scripts/certify_equivariant_resolvent_solver.py` and
  independently check it with the standard-library-only
  `scripts/reconstruct_equivariant_resolvent_solver.py`;
- `summaries/p16_solver_study.json`: deterministic guarded-FP64 solver,
  computed-residual, cost-accounting, failure-control, realistic-spectrum,
  and sampled six-point stability--fidelity-frontier diagnostic; it is not an
  FP64 rounding certificate or global frontier theorem;
- `summaries/P16_EQUIVARIANT_RESOLVENT_SOLVER_RESULTS.md`: scoped P16 exact-
  real solver theorem and diagnostic readout, including algebraic noncollapse,
  the failed meaningful-fidelity gate, and deployment exclusions; the human
  proof audit remains pending and unsigned;
- `summaries/shape_preserving_resolvent_certificate.json`: canonical P17
  exact/Arb artifact for the dimension-uniform origin-centered pointwise
  sector, the primary high-fidelity reduced-step and secondary full-step
  smooth-PL certificates, the unsafe raw-shape derivative band, canonical
  fidelity decisions, and the under-sized-passive-region control; generate it
  with `scripts/certify_shape_preserving_resolvent.py` and independently check
  its exact fields with the standard-library-only
  `scripts/reconstruct_shape_preserving_resolvent.py`;
- `summaries/p17_shape_preserving_study.json`: deterministic computed-residual-
  checked FP64 spectrum-grid and realistic-rank fidelity diagnostic for both
  locked P17 designs; it is not a global fidelity extremum, an inexact-solver
  theorem, or a rounding certificate;
- `summaries/P17_SHAPE_PRESERVING_RESOLVENT_RESULTS.md`: scoped P17 global
  exact-real one-trajectory smooth-PL result, canonical fidelity pass, unsafe-
  band and gate controls, and explicit incremental/inexact-solver/deployment
  exclusions; the human proof audit remains pending and unsigned;
- `summaries/sector_projected_useful_rate_certificate.json`: canonical P18
  exact/Arb artifact for the ray-sector projection, global pointwise sector,
  exact rational smooth-PL frontier, generic-sector `eta=1/50` obstruction,
  selected useful-rate certificate, canonical fidelity and amplitude gates,
  and negative controls; generate it with
  `scripts/certify_sector_projected_useful_rate.py` and independently replay
  its exact fields with the standard-library-only
  `scripts/reconstruct_sector_projected_useful_rate.py`;
- `summaries/p18_sector_projected_study.json`: deterministic computed-
  residual-checked FP64 canonical, spectrum-grid, operating-annulus, and
  realistic-rank diagnostic for the P18 interface; it is not a global
  fidelity extremum, an approximate-resolvent theorem, or a rounding
  certificate;
- `summaries/P18_SECTOR_PROJECTED_USEFUL_RATE_RESULTS.md`: scoped P18 global
  exact-real useful-rate smooth-PL result, exact amplitude/effective-step
  gates, finite step--rate frontier, fidelity diagnostics, and explicit
  incremental/inexact-solver/deployment exclusions; the human proof audit
  remains pending and unsigned;
- `summaries/sector_shielded_inexact_resolvent_certificate.json`: canonical
  P19 exact artifact for the moving-ball shield, disk/sector identity,
  zero-input rule, exact-P18 fixed-point property, arbitrary finite-candidate
  containment, fixed-input nonexpansiveness, both exact P18 smooth-PL rate
  replays, corruption controls, and binary64 reference scope; generate it
  with `scripts/certify_sector_shielded_inexact_resolvent.py` and independently
  replay its exact fields with the standard-library-only
  `scripts/reconstruct_sector_shielded_inexact_resolvent.py`;
- `summaries/p19_sector_shielded_study.json`: deterministic P19 binary64
  canonical, sampled-annulus, solver-residual, bitwise-inactivity, and
  deliberately corrupted-candidate diagnostics; it is not a global fidelity
  extremum, a graph-residual-to-final-candidate theorem, or a portable
  rounding certificate;
- `summaries/P19_SECTOR_SHIELDED_INEXACT_RESOLVENT_RESULTS.md`: scoped P19
  exact-real arbitrary-finite-candidate safety theorem, replayed useful rates,
  exact-as-stored NumPy FP64 reference guarantee, sampled fidelity evidence,
  representability obstruction, and explicit deployment exclusions; the
  human proof audit remains pending and unsigned;
- `summaries/scalable_sector_shield_certificate.json`: canonical P20 exact
  artifact for the locked BF16/FP32-to-FP32 CPU arithmetic graph, scaled
  balanced-norm enclosure, pass-through/radial-clip/half-fallback rounding
  ledgers, all-subnormal representability guard, seven exact Transformer-shape
  margins, and conditional replay of both P19 sector rates; generate it with
  `scripts/certify_scalable_sector_shield.py` and independently check its
  exact fields with the standard-library-only
  `scripts/reconstruct_scalable_sector_shield.py`;
- `summaries/p20_scalable_sector_shield_study.json`: deterministic P20 CPU
  proof-reference diagnostic covering P18 and pinned-upstream candidates,
  sampled annulus activation/fidelity, packed Transformer spectra, a sparse
  full-`768 x 768` arithmetic smoke test, adversarial and one-ULP controls,
  near-zero/FTZ behavior, and a discrete cross-platform decision digest; its
  offline exact disk checks are diagnostics, not runtime operations or a
  native accelerator-parity theorem;
- `summaries/P20_SCALABLE_MIXED_PRECISION_SECTOR_SHIELD_RESULTS.md`: scoped
  P20 static containment theorem, exact seven-shape margins, conditional
  abstract-port rate consequence, candidate diagnostics, near-zero
  fail-closed boundary, and explicit outer-arithmetic/GPU exclusions; the
  human proof audit remains pending and unsigned;
- `summaries/certified_outer_loop_composition_certificate.json`: canonical
  P21 exact artifact for the stored-signal `7 x 7` LMIs, zero-port P18/P19
  recovery, FP32 EMA/Nesterov/master roundoff envelopes, all-subnormal
  absolute port, seven-shape invariant-domain results, weight-decay scope
  controls, and frozen operator provenance; generate it with
  `scripts/certify_outer_loop_composition.py` and independently check its
  exact fields with the standard-library-only
  `scripts/reconstruct_outer_loop_composition.py`;
- `summaries/P21_CERTIFIED_OUTER_LOOP_COMPOSITION_RESULTS.md`: scoped P21
  stored-computation theorem, exact primary/secondary port gains, concrete
  roundoff neighborhoods, all-subnormal completion, conditional nonzero-
  decay result, and explicit neural-training/GPU/distributed exclusions; the
  human proof audit remains pending and unsigned;
- `summaries/p21_synthetic_shadow_trace.json`: deterministic 144-observation
  CPU diagnostic for the frozen P21 shadow observer, schedule, metrics, zero
  handling, and empirical gate implementation; its 126 pass-throughs and 18
  activations pass the synthetic checks, but it contains no model, dataset,
  backward pass, accelerator, or real gradient and is not training evidence;
- `../experiments/training/p21_shadow_trace_protocol.json`: predeclared P21
  real-gradient shadow schedule, metrics, zero handling, intervention gates,
  and provenance requirements; no real-gradient result is present;
- `summaries/deficit_audit.{json,csv}`: 24-prefix spectral audit;
- `summaries/quadratic_lr_sweep.json` plus CSV tables: gain-matched quadratic
  sweep and sensitivity classifications;
- `summaries/quadratic_horizon_check.{json,csv}`: 750/3000-step robustness;
- `figures/*.svg`: source-hashed static figures;
- `summaries/RESULTS.md`: concise experiment-only readout.
