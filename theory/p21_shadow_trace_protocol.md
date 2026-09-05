# P21 shadow-trace protocol

## Status and evidence boundary

This protocol is frozen before any real training-gradient result is inspected.
It defines a CPU observer and empirical decision rules for the proposed P21
composition.  It is not a stability certificate and cannot turn sampled
passes into a global theorem.

The companion synthetic runner exercises deterministic matrices only.  It
uses no language model, dataset, backward pass, checkpoint, accelerator, or
real gradient.  A real-gradient run remains blocked until:

1. the P21 analytic certificate, independent reconstruction, and boundary
   controls pass;
2. a trainer, exact revision, and instrumentation patch are pinned;
3. every intended Muon parameter shape is covered by P20; and
4. a BF16-capable accelerator and hashed data/tokenizer artifacts are
   available.

The machine on which this protocol was prepared has neither CUDA nor MPS.  No
GPU experiment was run while preparing it.

## Frozen acquisition schedule

The first shadow study is a single-seed, single-device, 256-optimizer-step,
step-scheduled diagnostic.  Wall-clock schedules are excluded because trace
overhead would change the baseline path.  Every intended Muon hidden matrix
is observed at the following 24 steps, with no post-hoc layer selection:

- early: `0,...,7`;
- middle: `124,...,131`;
- late: `248,...,255`.

The observer runs after gradient accumulation and, in later distributed work,
after the fully specified gradient reduction.  It copies the stored
Nesterov signal and the candidate actually emitted by the backend before
aspect scaling.  Recomputing a candidate on CPU is permitted only for unit
tests and the synthetic diagnostic.

Training remains unchanged.  The frozen aspect multiplier

\[
  \sqrt{\max(1,\operatorname{rows}/\operatorname{columns})}
\]

is applied in the candidate's stored dtype before both orientation and the
P20 shield.  If only the reversed shape is certified, signal and candidate
are transposed together, copied contiguously without numerical conversion,
shielded, and transposed back.  The P20 output is recorded but never applied
to the baseline model.

## Metrics and zero handling

Each observation records the step, phase, tokens seen, parameter/layer/role,
raw and shield orientations, shapes, dtypes, hashes, optimizer scalars, aspect
factor, shield action, guard, norms, correction, cosine, amplitudes, spectral
departure, retention, and effective gains at both `eta=1/120` and `eta=1/83`.

Nearest-rank quantiles are used throughout: after sorting `n` finite values,
the `p` quantile is `x[ceil(p*n)-1]`.  Only mathematically undefined values
are absent.  A zero denominator records either `both_zero` or
`nonzero_over_zero`; it never silently inserts zero into a quantile.
Nonfinite values are failures, not missing data.

For a signal `S` and target `T`, best-scalar departure is

\[
 \frac{\sqrt{\max(0,\|T\|_F^2-
        \langle S,T\rangle_F^2/\|S\|_F^2)}}{\|T\|_F}.
\]

A zero target has departure zero.  A zero signal paired with a nonzero target
has departure one.

## Predeclared gates

Hard integration gates require the complete 24-step schedule, all three
phases, 100% intended-Muon shape coverage by both parameter count and number
of elements, no nonfinite signal or candidate, no unexpected shield failure, and no
all-subnormal dead-zone event.  Every ordinary event must be returned by a
successful P20 call.  All output/signal amplitudes must lie in `[1/10,1]`, and
the `eta=1/120` effective gains must lie in `[1/1000,1/80]`.

The unchanged P16/P18 gates are:

- a candidate is informative when its best-scalar departure is at least
  `1/100`;
- every informative output must have departure at least `1/1000` and retain
  at least `1/10` of candidate shaping;
- on `3/4 <= ||S||_F <= 25`, `||T||_F/||C||_F` must lie in `[1/4,8]`, with at
  least one annulus observation in every phase.

The newly frozen mild-intervention gates are:

- activation rate at most `1/4` overall and at most `1/2` in every
  phase/role cell containing at least 20 observations;
- relative correction median at most `1/20`, p95 at most `1/4`, and maximum
  at most one;
- candidate/output cosine p05 at least `19/20` and minimum at least `4/5`;
- output/candidate amplitude p05 at least `3/4` and p95 at most `5/4`.

Failure of an empirical gate does not invalidate P20 safety.  It means the
candidate/shield interface must be adapted in the next design branch; the
gate may not be weakened after observing the trace.

## Current external blockers

This repository contains no NanoGPT checkout or submodule, dataset,
tokenizer, checkpoint, or trace fixture.  Vanilla GPT-2-style NanoGPT also
contains a fused QKV parameter whose canonical shape is `768 x 2304`, absent
from P20's frozen shape table.  A real study must either certify that shape or
freeze an architecture whose complete Muon shape inventory is already
covered.  The exact trainer revision, training configuration, data hashes,
accelerator stack, and instrumentation patch must be committed to the result
manifest before execution.

The authoritative machine-readable schedule, gate values, metric semantics,
and required provenance fields are in
`experiments/training/p21_shadow_trace_protocol.json`.  Each result records
the SHA-256 of those exact bytes.
