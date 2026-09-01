# Experiments

Experiment order is gated:

1. spectral/matrix audit;
2. controlled matrix quadratics;
3. one small matched NanoGPT learning-rate sweep.

No GPU experiment should begin until a repair domain and a valid constant
`rho` certificate are specified. Each run writes a self-contained JSON manifest
and compact CSV tables under `results/summaries/`; the JSON records inputs,
operator details, dtype, seed, software, hardware, and Git state.
