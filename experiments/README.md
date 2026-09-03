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
10. only after stochastic, quantized, and implementation-parity gates, a
   small matched neural-training sweep.

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
architecture. Each run writes a self-contained JSON
manifest and compact CSV tables under `results/summaries/`; the JSON records
inputs, operator details, dtype, seed, software, hardware, and Git state.

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
