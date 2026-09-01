# Experiments

Experiment order is gated:

1. exact and backend-specific executable witnesses;
2. spectral/matrix audit;
3. qualified diagonal matrix quadratics;
4. a full-matrix certificate and matched momentum quadratics;
5. only then, a small matched NanoGPT learning-rate sweep.

No GPU experiment should begin until a repair domain and a valid constant
`rho` certificate are specified. Each run writes a self-contained JSON manifest
and compact CSV tables under `results/summaries/`; the JSON records inputs,
operator details, dtype, seed, software, hardware, and Git state.

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
