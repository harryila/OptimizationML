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
14. only after model-forward use of the logical master, aspect scaling, weight
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
parity. Each run writes a self-contained JSON manifest and compact CSV tables
under `results/summaries/`; the JSON records inputs, operator details, dtype,
seed, software, hardware, and Git state.

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
```

The P9 and P10 generators and standard-library-only reconstructions carry
their exact claims. P9's separate CPU diagnostic is a native-matmul
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
