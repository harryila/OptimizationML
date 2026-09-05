# P16 equivariant resolvent-solver results

## Verdict

P16 is an informative partial success. It proves the exact-real
bi-orthogonal-equivariance, singular-value reduction, and safeguarded Newton
termination theorem requested after P15, and it supplies a guarded FP64
reference implementation whose declared calls all pass the computed P15
residual check. It does **not** pass the separate meaningful-Muon-fidelity
gate: the locked stable Yosida operator is algebraically noncollapsed but
effectively a scalar map on the canonical comparator.

The FP64 results below are deterministic implementation evidence, not a
rounding certificate. The exact/Arb artifact and the theorem note, rather than
sampled solver cases, carry the algebraic claims.

| Item | P16 result |
| --- | --- |
| Matrix domain | every fixed finite real rectangular shape |
| Locked epsilon | `1/10000000` |
| Jordan map | coefficients `6889/2000`, `-191/40`, `4063/2000`; five stages |
| Formula provenance | KellerJordan/Muon `f98f1cacc0263b04290753e32be8d498c1efc806` |
| Resolvent parameters | `lambda=1/1000`, `mu=1000` |
| P15 relative tolerance | `kappa=1/250` |
| Structured solve | one solve SVD plus a diagonal-plus-rank-one scalar Newton system |
| Literal FP64 graph postcheck | second SVD reevaluates `B(U_hat)` on the stored matrix |
| Exact-real termination | global convergence and finite termination for each positive P15 threshold |
| Declared computed-residual cases | `13/13` pass |
| Largest observed Newton count | `8` |
| Worst observed graph residual | approximately `5.880e-14` |
| Canonical best-scalar departure | approximately `5.879e-6` |
| Canonical upstream departure | approximately `6.997e-2` |
| Canonical shaping retention | approximately `8.403e-5` |
| Meaningful-fidelity threshold | departure at least `1/1000` and retention at least `1/10` |
| Overall P16 acceptance gate | **fails meaningful fidelity** |

## Exact equivariance and singular reduction

For the P13 repaired additive-epsilon map `A_epsilon`, define

\[
B=A_\epsilon+1000I,
\qquad
J=(I+B/1000)^{-1},
\qquad
Y=1000(I-J).
\]

Frobenius normalization, the five odd rectangular spectral-polynomial stages,
the radial correction, and the identity shunt are all bi-orthogonally
equivariant. Uniqueness of the resolvent consequently gives

\[
J(QSR^\top)=QJ(S)R^\top.
\]

The stabilizer of an SVD then establishes singular-vector preservation without
differentiating the SVD: repeated singular values remain repeated, zero modes
remain zero, rectangular null-side components vanish, and the statement is
independent of the bases chosen within repeated or null subspaces.

If `sigma_i` are the singular values of `S`, the unknown resolvent values
`x_i` solve

\[
\Phi_i(x)=
\left[1+\lambda\left(\mu+\frac{p(\|x\|/\epsilon)}{\|x\|}\right)\right]x_i
+\lambda h\!\left(\frac{x_i}{\|x\|+\epsilon}\right)-\sigma_i=0.
\]

The cross-mode coupling enters only through the Euclidean radius. Its
Jacobian is diagonal plus rank one. Exact radius-band margins keep the
diagonal and Sherman--Morrison denominator positive, so a Newton direction
uses linear algebra proportional to the number of singular values rather than
a dense solve of that dimension.

For exact arithmetic, the Newton direction strictly decreases
`||Phi||_2^2/2`; Armijo backtracking terminates at each nonroot iterate.
Strong monotonicity makes the relevant level set bounded and the root unique,
yielding global convergence from every finite start. For each fixed positive
P15 threshold, convergence implies finite stopping. No useful input-uniform
iteration count is claimed.

## Guarded FP64 reference study

The reference path uses one solve SVD, the reduced safeguarded solve, and a
full matrix reconstruction. It then uses a second SVD to reevaluate `B` on
that stored reconstructed candidate. It emits only

\[
\widehat Y=1000(S-\widehat U)
\]

after recomputing the actual candidate graph residual and verifying

\[
\|S-\widehat U-B(\widehat U)/1000\|_F
\le \|S\|_F/250+\bar r_{\rm fp64}.
\]

The canonical deterministic study has nine dense calls and four complete
reduced-spectrum calls, so all `13/13` declared cases pass the computed rule.
The reduced large-shape cases are `768 x 768`, `768 x 3072`,
`3072 x 12288`, and `4096 x 11008`. They allocate the complete singular-value
vectors but not dense matrices, and therefore do not exercise a large SVD or
measure accelerator runtime. Across the declared cases, the study reports a
maximum of 8 Newton iterations, no accepted-case backtracking, a worst
computed graph residual near `5.880e-14`, and a worst residual-to-threshold
ratio near `1.061e-12`.

The operation comparison is structural. P16 uses one solve SVD, five scalar
quintic stages per singular value per Newton iteration, `O(k)`
Sherman--Morrison work, matrix reconstruction, and a second SVD plus
reconstruction for the literal full-matrix graph postcheck. The locked
five-stage Jordan implementation uses 15 dense matrix multiplications. This
accounting does not claim that the reference solver is faster.

Failure controls reject a zero iteration cap, nonfinite data, and signals
beyond the locked guard; they also distinguish the required resolvent-form
output from `B(U_hat)` and remove the Jordan term as a no-shaping control. A
successful return always passes the computed P15 residual rule.

## Fidelity result: distinct but effectively scalar

Algebraic noncollapse and meaningful fidelity are intentionally different
questions. Exact rational cancellation gives a two-mode witness with unequal
gains, and the canonical `diag(3,4)` result is enclosed with outward-rounded
Arb arithmetic after inflating the FP64 root by P15's exact residual-to-output
gain. Its modal gains are approximately `760.207725` and `760.217036`, so
they are rigorously distinct.

Nevertheless, for

\[
\chi(T;S)=
\frac{\|T-\langle T,S\rangle_F S/\|S\|_F^2\|_F}{\|T\|_F},
\]

the locked output has `chi` about `5.879e-6`. The exact-real upstream
five-stage Jordan comparator has `chi` about `6.997e-2`, leaving a shaping
retention of only about `8.403e-5`. This misses both declared gates:
`chi>=1/1000` and at least `1/10` of upstream shaping. The exact result is
therefore **algebraically noncollapsed but effectively scalar**, not evidence
that the `1000I`-shunted design retains meaningful Muon behavior.

The required follow-up frontier evaluates six frozen `(lambda,mu)`
points over a deterministic two-mode input grid. None jointly passes the
unchanged frozen P14 certificate and both fidelity gates. The grid does not
certify a global stability--fidelity impossibility, and rejection by the
frozen storage is not proof of instability under some newly optimized
certificate.

## Scope

P16 supplies a useful exact-real reference solver and exposes a fidelity
obstruction at the locked P14 design. It does not establish:

- a certified FP64 error envelope or an exact bound for rounded residual
  evaluation;
- a useful uniform a priori iteration count;
- a successful meaningful-Muon-fidelity design;
- BF16, accelerator, or literal upstream-Muon parity;
- weight decay, aspect scaling, stochastic-gradient, or neural-network
  guarantees.

## Reproduce

Generate the exact/Arb artifact:

```bash
uv run --locked python scripts/certify_equivariant_resolvent_solver.py \
  --output results/summaries/equivariant_resolvent_solver_certificate.json
```

Independently reconstruct its exact fields with the Python standard library:

```bash
uv run --locked python scripts/reconstruct_equivariant_resolvent_solver.py \
  --require-canonical
```

Run the deterministic FP64 solver and frontier study separately:

```bash
uv run --locked python experiments/resolvent/run_p16_solver_study.py \
  --output results/summaries/p16_solver_study.json
```

The committed JSON manifests are authoritative for exact fractions, interval
balls, source hashes, hardware/software provenance, and the realized
diagnostic values. Automated generation and reconstruction do not replace the
pending independent human proof audit.
