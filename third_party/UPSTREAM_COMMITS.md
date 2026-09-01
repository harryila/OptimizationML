# Upstream provenance

Revisions are pinned before equations or implementation details are copied.
Public availability is not treated as licensing permission.

## KellerJordan/Muon

- Repository: <https://github.com/KellerJordan/Muon>
- Revision: `f98f1cacc0263b04290753e32be8d498c1efc806`
- Revision date: 2026-05-24
- Audited file: `muon.py`
- Audited file SHA-256:
  `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`
- License: MIT
- Used facts: Jordan coefficients `(3.4445, -4.7750, 2.0315)`, BF16 cast,
  one-time orientation before normalization, normalization through
  `X.norm(dim=(-2,-1), keepdim=True) + 1e-7`, the literal BF16 expression
  `B = b*A + (c*A)@A`, and the five-step default call path. The parentheses
  make explicit Python's left-associative parsing of upstream `c*A @ A`.
- Momentum facts: `muon_update` defaults to `beta=0.95`, `ns_steps=5`, and
  `nesterov=True`. Its literal calls
  `momentum.lerp_(grad, 1 - beta)` followed by
  `grad.lerp_(momentum, beta)` implement
  `m_next=beta*m+(1-beta)*g` and
  `s_next=beta*m_next+(1-beta)*g`. The second call mutates the gradient buffer.
  With a zero initial momentum buffer and `beta=0.95`, this gives
  `m_1=0.05*g` and `s_1=0.0975*g`.
- Local implementation: the smooth mathematical equation is independently
  reimplemented in `src/passive_muon/polynomials.py`. The backend-specific
  executable shadow in `src/passive_muon/deployed.py` separately preserves the
  pinned operation order because algebraically equivalent grouping can change
  BF16 results. A clean local regression separately checks the two-`lerp`
  EMA/Nesterov algebra and mutation semantics. No upstream training or
  distributed-optimizer code is copied.
- Exact-update scope: the branch-`p3` EMA/Nesterov theorem matches this
  state-and-signal ordering in real arithmetic after replacing the upstream
  orthogonalizer by the certified repaired floored map and omitting weight
  decay. It is not a parity claim for the complete BF16/current-normalized
  optimizer path.

## Polar Express

- Repository: <https://github.com/NoahAmsel/PolarExpress>
- Revision: `71cc37943d99cae780024c1d198977f2f8795407`
- Revision date: 2026-08-14
- Audited file: `polar_express.py`
- Paper: *The Polar Express: Optimal Matrix Sign Methods and Their Application
  to the Muon Algorithm*, arXiv:2505.16932v5
- License: MIT
- Frozen configuration: initial lower bound `1e-3`, ten generated stages,
  degree five, coefficient safety factor `1.01`, and cushion `0.02`. The local
  baseline freezes the first five generated coefficient triples, which are
  the stages used by the paper's five-step language-model comparison.
- Generation environment: NumPy 2.5.2, Python 3.12, arm64 macOS, Apple
  Accelerate BLAS/LAPACK. The exact resulting decimal strings are committed in
  `src/passive_muon/specs.py`; they are not regenerated at import time.
- Local implementation: the MIT-licensed recurrence is independently written
  as a normalization-free, float64-compatible mathematical shadow. The
  upstream BF16 cast and rule `X/(1.01*||X||_F + 1e-7)` are intentionally not
  hidden inside the polynomial baseline.
- Version caveat: arXiv v5's pseudocode, its Appendix A code, and the current
  repository differ in cushioning, normalization, and sequence length. Claims
  using this baseline must identify it as the pinned repository configuration,
  not as an undifferentiated reproduction of every paper variant.

## CANS

- Repository: <https://github.com/GrishKate/accelerating_orthogonalization>
- Revision: `3eca916ffa6f46ab3b313dd34c48235322747565`
- Revision date: 2025-06-13
- Audited files: `polynomials.py` and `nanoGPT/backends.py`
- Paper: *Accelerating Newton-Schulz Iteration for Orthogonalization via
  Chebyshev-type Polynomials*, arXiv:2506.10935v2
- License: none found. The repository has no `LICENSE` or `NOTICE`, and the
  GitHub API reports no detected license.
- Used published facts: Appendix J's CANS degree-five, four-stage,
  `delta=0.3` coefficient table and the paper's odd-polynomial matrix
  recurrence. This configuration costs twelve dense matrix multiplications.
- Local implementation: clean-room implementation from the paper only. No
  source code from the unlicensed repository is copied. The repository was
  inspected solely to corroborate the published table and identify deployment
  normalization choices.
- Normalization caveat: the paper's NanoGPT CANS result used an input-dependent
  Gelfand normalizer, whereas its original-Muon control used Frobenius
  normalization. The local baseline contains neither; controlled comparisons
  apply one explicitly selected normalizer to every orthogonalizer.
- Naming caveat: Appendix J's `eps=0.3` is the approximation deviation
  `delta`, not the numerical normalization epsilon.

## Modded-NanoGPT

Deferred until the theorem, repair domain, and matrix-quadratic gate pass. Pin
the experiment revision in its own manifest before running GPU work.
