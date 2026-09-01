# Experiments

Experiment order is gated:

1. exact and backend-specific executable witnesses;
2. spectral/matrix audit;
3. qualified diagonal matrix quadratics;
4. the completed full-matrix floored certificate;
5. the repaired deterministic quadratic momentum IQC and matched rank-one
   boundary checks;
6. only then, a small matched NanoGPT learning-rate sweep.

The floored Jordan architecture now has a valid global constant-`rho`
certificate and a correctly scoped deterministic quadratic momentum theorem.
The global sector step bound is extremely conservative, so GPU work remains
gated on a stronger architecture-aware certificate and a predeclared use of
that architecture. Each run writes a self-contained JSON manifest and compact
CSV tables under `results/summaries/`; the JSON records inputs, operator
details, dtype, seed, software, hardware, and Git state.

Replay the CPU-only momentum result with:

```bash
uv run --locked python scripts/certify_momentum_stability.py \
  --output results/summaries/momentum_iqc_certificate.json
```

The exact rational LMI and Jury signs carry the claims. The committed float64
trajectories are matched diagnostics, not sampled global certificates.

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
