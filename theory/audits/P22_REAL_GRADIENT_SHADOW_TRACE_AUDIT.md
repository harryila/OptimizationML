# P22 real-gradient shadow-trace audit

## Status

**The acquisition evidence has received an internal technical audit; independent
human review is pending.** This is an unsigned external-review packet, not an
approval record. The checked findings below record the internal audit of the
two completed trace-off runs. Unchecked reviewer items must not be cited as
completed independent review.

P22 stopped at its predeclared baseline-repeatability gate. No trace-on run was
performed, so this packet contains no observer-noninterference, shield-activity,
candidate-fidelity, or training-quality result.

## Materials under review

The frozen protocol and implementation are:

- `experiments/training/p22_real_gradient_shadow_trace_protocol.json`;
- `theory/p22_real_gradient_shadow_trace_protocol.md`;
- `experiments/training/run_p22_real_gradient_shadow_trace.py`;
- `experiments/training/p22_observed_muon.py`; and
- `experiments/training/p22_nanogpt_shadow_trace.py`.

The path-sanitized canonical evidence is:

- [`p22_repeatability_failure_evidence.json`](../../results/summaries/p22_repeatability_failure_evidence.json);
- [`p22_real_gradient_preflight_evidence.json`](../../results/summaries/p22_real_gradient_preflight_evidence.json);
  and
- [`p22_fineweb_materialization_evidence.json`](../../results/summaries/p22_fineweb_materialization_evidence.json).

The scoped narrative is
[`P22_REAL_GRADIENT_SHADOW_TRACE_RESULTS.md`](../../results/summaries/P22_REAL_GRADIENT_SHADOW_TRACE_RESULTS.md).
The native
manifests and comparison report remain external because they contain local
absolute paths and redundant per-step data. Their complete files are locked by
the SHA-256 values below, and the canonical repeatability artifact preserves
every unequal checked value plus hashes of both complete comparison
projections.

| native artifact | bytes | SHA-256 |
| --- | ---: | --- |
| `trace-off-a-manifest.json` | 197869 | `325c0e03fdfc7c5cd249b9c10cbe0a3b0e8c94b4b953a3eea3bf17586b497e98` |
| `trace-off-b-manifest.json` | 197868 | `49e902dad2b7b5fdc8c2f97ed8498c79a743c1cbaf8eba6798fd80cc5e3346f8` |
| `repeatability-report.json` | 28774 | `1b6522653780cec8092e0907be672bb21897ce847c01baa3922233ed3c597923` |
| `p22_preflight_after_shape_commit.json` | 4422 | `81ea8900f66cef389217dee04146120774d7cfc5326229d202b606b3133e7aef` |
| `p22_fineweb_manifest.json` | 4210 | `1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d` |

## Exact claim under review

Two fresh 256-step trace-off executions of the frozen GPT-2-small computation
started from byte-identical model, optimizer, and recorded RNG states and used
identical sequential FineWeb batches. Their model and optimizer states differed
at the first post-update checkpoint, after step 0. Their stored loss tensors
first differed at step 2. The exact verifier reported 303 mismatches, so the
frozen protocol correctly stopped before trace-on acquisition.

The defensible conclusion is limited to this execution pair:

> The pinned Apple-MPS/PyTorch 2.13.0 computation did not provide the bitwise
> repeatable trace-off baseline required to distinguish observer effects from
> ordinary execution variability.

This does not identify a responsible operation, prove backend-wide Apple-MPS
nondeterminism, or weaken the P18--P21 analytic certificates.

## Completed internal acquisition audit

### 1. Frozen inputs and provenance

- [x] Both manifests have the same run-identity digest
      `4dd1f079de75d45ef65ac5e8a43622e71fa511dd7c686ee37fc470832615c92e`.
- [x] Both have the same combined initial-state digest
      `c04abf549b837bb84cc6bb27cbf1191ec016665abfd4957deaff27c9232749f8`.
- [x] Initial model, optimizer, and RNG digests match separately.
- [x] The complete source-snapshot mappings and runtime-provenance mappings
      match between off-A and off-B.
- [x] Every project-source hash that the manifests record matches the
      corresponding file at OptimizationML commit `79f33ec` exactly.
- [x] nanoGPT is pinned at
      `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`; KellerJordan/Muon is pinned
      at `f98f1cacc0263b04290753e32be8d498c1efc806`, with `muon.py` SHA-256
      `2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d`.
- [x] The native manifests themselves do not contain an OptimizationML Git
      SHA or repository dirty-state field. The `79f33ec` attribution is a
      retrospective reconstruction from the exact per-file source hashes,
      not a contemporaneous Git-state record. The snapshot also omits four
      imported project dependencies: `p22_nanogpt_shadow_trace.py`, package
      `__init__.py`, `p21_shadow_trace.py`, and
      `scalable_sector_shield.py`. Their commit blobs are recorded for review,
      but no native hash proves those were the executed worktree bytes. These
      are provenance caveats under repository rule 7 and must remain
      disclosed.

### 2. Frozen arithmetic and schedule

- [x] Both runs report macOS 15.5 on arm64, Python 3.12.11, NumPy 2.5.2,
      PyTorch 2.13.0 at Git revision
      `cf30153c4c131c8164ee7798e5022d810682e2cb`, and the Apple-MPS backend.
- [x] The runtime manifest does not record the Apple chip identity or exact
      OS/Metal build. The evidence is scoped to this hash-locked execution
      pair and is not device-model or Metal-version provenance.
- [x] Both report `PYTORCH_ENABLE_MPS_FALLBACK=0` and
      `PYTORCH_MPS_FAST_MATH=0`.
- [x] The runner seeds Python, NumPy, Torch CPU, and MPS, enables
      `torch.use_deterministic_algorithms(True)`, disables the available TF32
      paths, and synchronizes MPS before state hashing.
- [x] Model parameters and forward/backward arithmetic are FP32 with no
      autocast or gradient scaler. Gradient and momentum storage is FP32. The
      candidate casts to BF16 before orientation, norm, epsilon addition,
      division, and the five Newton--Schulz stages on MPS.
- [x] The frozen protocol JSON's narrower "only the stages" wording is a
      post-acquisition description error. The machine-readable bytes remain
      unchanged and `p22_real_gradient_shadow_trace_protocol_erratum.md`
      records the correction.
- [x] Both executions use one seed, batch size one, sequence length 128, no
      accumulation, 256 optimizer steps, zero weight decay, Muon
      `eta=1/120`, `beta=19/20`, and 32,768 sequential training tokens.
- [x] Token offsets and batch hashes agree at all 256 steps. Recorded Python,
      NumPy, Torch CPU, and MPS RNG hashes agree before and after every step.

### 3. Trace-off execution path

- [x] Both executions use `trace_mode="trace_off"`, contain no capture
      records, and apply no shadow output to training.
- [x] The observation adapter wraps the pinned `muon_update` in both runs, but
      with `hook=None` it does not clone the signal or candidate and returns
      the exact object produced by the original function.
- [x] P20 is not evaluated in trace-off. Therefore these runs cannot measure
      shield activation, correction magnitude, or fidelity.
- [x] The common trace-off wrapper cannot by itself explain a difference
      between off-A and off-B. This check is not evidence that a future
      trace-on observer is noninterfering.

### 4. Hashing and verifier logic

- [x] Tensor hashes are formed from synchronized, detached, contiguous CPU
      copies with dtype, shape, and stored bytes included.
- [x] Mapping keys are canonically sorted, and type/length separators prevent
      ambiguous tree encodings.
- [x] The verifier first validates schema, trace mode, protocol hash, backend,
      shadow-update status, step count/order, offsets, digest syntax, and the
      state-checkpoint schedule.
- [x] The repeatability comparison uses exact equality with no tolerance for
      run identity, all six per-step fields, both state fields at all 24
      scheduled checkpoints, and all three final-state fields.
- [x] Independently deriving mismatches from the two native manifests
      reproduces the committed comparison report exactly.

### 5. Failure chronology

- [x] Model-state and optimizer-state hashes differ at every one of the 24
      scheduled post-update checkpoints, beginning at step 0.
- [x] Loss-tensor bytes first differ at step 2 and differ on 253 of 256 steps;
      only steps 0, 1, and 22 have equal stored loss bytes.
- [x] Final model and optimizer hashes differ, while the final recorded RNG
      hash agrees.
- [x] The 303 mismatches decompose into 253 loss mismatches, 24 model-state
      mismatches, 24 optimizer-state mismatches, and two final-state
      mismatches.
- [x] As a secondary diagnostic only, the maximum absolute difference between
      the stored scalar losses is `0.013334274291992188` at step 210, and the
      mean absolute difference over all 256 steps is
      `0.0013705622404813766`. The acceptance decision is byte-exact and does
      not depend on these floating-point summaries.

### 6. Internal diagnosis boundary

- [x] Equal input and recorded RNG schedules rule out those recorded sources
      as explanations; they do not prove deterministic execution of the
      accelerator operation graph.
- [x] Equal losses at steps 0 and 1 do not imply equal gradients. A scalar
      FP32 loss can remain byte-equal while unrecorded gradient tensors or
      parameter updates differ.
- [x] The whole-state hashes do not localize the first discrepancy to forward,
      backward, auxiliary Adam, Muon momentum, BF16 Newton--Schulz, or the
      parameter update.
- [x] Plausible MPS localization targets include scaled-dot-product-attention
      backward, repeated-index embedding-gradient accumulation, other parallel
      backward reductions, BF16 normalization/Newton--Schulz reductions and
      matrix products, and optimizer arithmetic. These are hypotheses, not
      results.
- [x] `PYTHONHASHSEED` and a one-thread CPU setting were not frozen. The
      audited parameter traversal is insertion ordered, the initial state is
      equal, and the material computation is on MPS, so neither omission is a
      demonstrated cause or a sufficient repair.
- [x] No defect affecting the observed 303-mismatch comparison, stale
      asynchronous hash, mismatched data schedule, RNG drift, or asymmetric
      trace-off observer path was found. The native verifier does have a
      provenance-enforcement gap: it trusts an opaque run-identity digest and
      does not separately compare or recompute every source/runtime mapping.
      The canonical evidence builder compensates for this execution pair by
      requiring those mappings to be equal and storing their canonical hashes;
      the native verifier must be hardened before a future acceptance run.

### 7. Stop decision and claim boundary

- [x] The machine-readable protocol requires off-A to match off-B before any
      off-A/trace-on comparison.
- [x] The written protocol says that one unequal byte fails acquisition and
      that `allclose` is insufficient.
- [x] Trace-on therefore remains barred under the frozen protocol.
- [x] The failure is not observer-interference evidence because no observer
      ran, and baseline variation is already present.
- [x] It is not evidence about P20 activation/fidelity, validation loss,
      convergence, throughput, CUDA, distributed execution, kernel parity, or
      stability of unshielded upstream Muon.
- [x] It does not modify the exact P22 `768 x 2304` static shape certificate
      or any P18--P21 mathematical theorem.

## Independent external-review checklist

### A. Artifact integrity and reconstruction

- [ ] Fetch the five native artifacts named above and verify their SHA-256 and
      byte counts before opening them.
- [ ] Run `scripts/build_p22_repeatability_failure_evidence.py` against those
      native files and require byte-identical canonical outputs.
- [ ] Recompute both complete comparison-projection hashes and all 303
      mismatch witnesses independently of project comparison functions.
- [ ] Confirm the path-sanitized artifacts contain no personal home or native
      temporary paths and retain enough hashes to identify all external data.
- [ ] Verify that the canonical evidence contains no trace-on,
      noninterference, or aggregate-shadow artifact and records why each is
      absent.

Reviewer notes:

> Pending.

### B. Source and data provenance

- [ ] Verify each recorded project source hash against commit `79f33ec` and
      record that this neither covers the four omitted imported dependencies
      nor replaces the missing contemporaneous OptimizationML
      Git-SHA/dirty-state field.
- [ ] Verify the nanoGPT revision, tree, tracked blobs, origin, and clean state
      from the preflight evidence.
- [ ] Verify the Muon revision and complete source hash, including additive
      epsilon placement, five Jordan stages, BF16 cast, transpose rule, and
      post-orientation aspect scaling.
- [ ] Verify the FineWeb revision, deterministic row selection, response
      hashes, tokenizer hashes, preprocessing hash, and both token-binary
      hashes.
- [ ] Confirm the data materialization is adequate for this repeatability
      diagnostic and is not presented as a representative training corpus or
      quality evaluation.

Reviewer notes:

> Pending.

### C. Runner and baseline equivalence

- [ ] Audit model construction, tied weights, parameter grouping, the exact
      48-tensor Muon inventory, auxiliary-Adam inventory, learning rates,
      beta, epsilon, weight decay, batch schedule, and checkpoint schedule.
- [ ] Verify the two trace-off runs execute the same source and operation graph
      and that no environment or runtime field differs.
- [ ] Confirm `hook=None` takes no clone, CPU-copy, P20-shadow, metric, or raw
      capture path, and that the original Muon update is returned unchanged.
- [ ] Check synchronization occurs before every stored state hash and before
      loss bytes are copied to CPU.
- [ ] Confirm no unrecorded random data sampling, dropout, autocast, scaler,
      compilation, CPU fallback, or weight decay is active.

Reviewer notes:

> Pending.

### D. Exact verifier

- [ ] Independently audit `canonical_state_sha256` for tensors, nested
      mappings, optimizer parameter identifiers, sequences, and RNG states.
- [ ] Check all accepted manifest fields and reject a manifest with a changed
      protocol hash, incomplete step order, undeclared checkpoint, trace-off
      capture, or missing final digest.
- [ ] Re-run the exact comparator and recover 303 mismatches with the same
      first state and loss discrepancies.
- [ ] Mutate one equal field and one unequal field in disposable copies and
      confirm the verifier respectively adds and removes exactly the expected
      witness.
- [ ] Confirm no `loss_float64`, tolerance, rounding, or `allclose` decision
      enters the acceptance gate.

Reviewer notes:

> Pending.

### E. Interpretation and next experiment

- [ ] Confirm the evidence establishes failure of exact baseline repeatability
      only for the two pinned Apple-MPS executions.
- [ ] Confirm it does not identify a particular operation or justify a
      backend-wide MPS nondeterminism claim.
- [ ] Confirm trace-on is prohibited by the frozen gate and cannot be used to
      infer noninterference from this acquisition.
- [ ] Confirm changing to tolerance-based equivalence, manual attention,
      CPU-hosted embeddings, a different optimizer split, or another backend
      would constitute a new or amended protocol rather than a pass of this
      one.
- [ ] For a localization-only diagnostic, predeclare per-parameter hashes for
      gradients before the optimizer, Muon momentum/candidate/output, auxiliary
      Adam state/output, and parameters after update. Keep this evidence
      explicitly separate from the frozen acceptance run.
- [ ] For the next acceptance attempt, use one pinned BF16-capable CUDA device,
      record the exact GPU/driver/runtime identity, satisfy its deterministic
      workspace requirements, repeat off-A/off-B exactly, and run trace-on
      only if that prerequisite passes.

Reviewer notes:

> Pending.

## External reviewer sign-off

Reviewer name:

Affiliation or role:

Date:

Reviewed commit:

Native evidence hashes verified:

Decision: `approve` / `approve with qualifications` / `reject`

Qualifications or required corrections:

> Pending.
