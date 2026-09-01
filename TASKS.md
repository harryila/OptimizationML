# Task ledger

## Gate 1: theorem and witness

- [x] Separate current and fixed Frobenius normalization.
- [x] Derive the diagonal normalized-map Jacobian.
- [x] Construct an exact Jordan witness with positive fixed-scale derivatives.
- [x] Verify a finite pairwise violation in high precision.
- [x] Independent proof review.
- [ ] Decide the certification domain for the practical repair theorem.

## Gate 2: deficit audit

- [x] Implement Jordan and classical Newton--Schulz in float64.
- [x] Implement local-Jacobian and pairwise deficits.
- [x] Check the robust witness in the upstream BF16 operation order.
- [x] Add Polar Express from a pinned upstream revision.
- [x] Add clean-room CANS from its published coefficient table.
- [ ] Add interval and/or SOS upper certificates.

## Gate 3: evidence

- [x] Matrix-spectrum audit for Jordan, classical, Taylor, Polar Express, CANS.
- [x] Gain-matched controlled matrix quadratics for all five families.
- [x] Long-horizon sensitivity check for Jordan and classical controls.
- [x] Matrix-multiplication accounting for every audited prefix.
- [ ] Matched small NanoGPT learning-rate sweep.
- [ ] Accelerator throughput benchmark.

## Explicitly deferred

- Lean formalization;
- Gram Newton--Schulz implementation audit;
- large language-model benchmark;
- general circuit-designed optimizers.
