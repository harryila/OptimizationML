# P25 CUDA diagnostic determinism and redaction audit

## Status

**Unsigned preregistration audit packet.** P25 has not yet executed. This
packet reviews a one-shot correction to the P24 diagnostic and sanitizer,
followed only on complete success by the unchanged P23 CUDA acquisition. A
checked box must reflect independently inspected evidence, not the existence
of a code path or CI fixture.

P24 remains terminal. Its baseline diagnostic may not be rerun. Offline
ingestion of the retained P24 native bytes preserves the failed disposition
and is not a new diagnostic attempt.

## Materials under review

- `experiments/training/p25_cuda_diagnostic_contract.json`;
- `scripts/reconstruct_p25_cuda_diagnostic_contract.py`;
- `scripts/finalize_p25_cuda_diagnostic_contract.py`;
- `experiments/training/run_p25_executable_origin_diagnostic.py`;
- `scripts/sanitize_p25_executable_origin_diagnostic.py`;
- `scripts/ingest_p24_native_outcome_for_p25.py`;
- `scripts/run_p25_cuda_attempt.sh` and the P25 runbook;
- the byte-identical P24 image helper and Dockerfile;
- the unchanged P23 acquisition runner and frozen P21/P22 gates;
- after execution, every native and sanitized artifact, BuildKit record,
  image inspection, P25-named lock/attestation, process stream, and outcome
  record; and
- the external P24 native diagnostic with SHA-256
  `ce44c53c9f274cef3eda7b3adea755ce6b1313773c8dfe00fc9f194b74fb2bd2`
  and byte count `3,760,729`.

## A. Parent outcome and no-favorable-rerun boundary

- [ ] Verify the P24 outcome commit, tree, annotated tag object, and tag target
      against Git objects independently.
- [ ] Rehash the P24 compact outcome and result summary.
- [ ] Rehash the external P24 native artifact and match its exact byte count.
- [ ] Reconstruct all 36 P24 checks, with exactly 35 true and only
      `torch_determinism_matches_lock` false.
- [ ] Reconstruct exactly the six declared live-versus-lock state differences
      and no seventh difference.
- [ ] Confirm the exact generated-module bytes were observed under writable
      `/tmp`, without inferring a CUDA gradient or model execution.
- [ ] Confirm P24 was never rerun and its failed disposition was not changed by
      historical ingestion or new prose.

Reviewer notes:

> Pending.

## B. Contract and source freeze

- [ ] Run the independent P25 reconstruction and recover every exact check.
- [ ] Confirm no execution-source `sha256` field remains pending before the
      preregistration commit, image build, or execution.
- [ ] Rehash every P25 execution source and every inherited immutable
      authority from clean repository bytes.
- [ ] Confirm the P24 helper, Dockerfile, P23 acquisition runner, P21 shadow
      semantics, and P21/P22 thresholds are byte-identical to their declared
      authority.
- [ ] Confirm the preregistration commit predates image construction and all
      P25 diagnostic or gradient observations.
- [ ] Confirm the preregistration commit is one direct child of the
      source-freeze bootstrap and adds only the historical P24 wrapper, which
      was absent from the bootstrap tree.

Reviewer notes:

> Pending.

## C. Historical P24 ingestion and corrected redaction

- [ ] Confirm the ingester accepts only the exact retained P24 SHA-256 and byte
      count and rejects duplicate JSON keys.
- [ ] Confirm historical ingestion ran exactly once in the pinned old image
      with no network or GPU, a read-only root, exact original logical-root
      destinations, and a read-only overlay on the retained P24 native file.
- [ ] Confirm it reconstructs the old schema, all 36 checks, exact six state
      mismatches, exact generated module, and exactly two `/tmp` mapping-key
      locations.
- [ ] Confirm the wrapper labels P24 failed and grants no authority to build or
      acquire.
- [ ] Confirm the P25 sanitizer applies one reviewed logical-root transform to
      mapping keys and values, including nested mappings and lists.
- [ ] Confirm transformed-key collisions, duplicate native keys, undeclared
      absolute paths, root overlap, missing roots, aliasing, and preexisting
      outputs all fail closed.
- [ ] For a claimed pass, reconstruct the exact 40-check and closed top-level
      inventories independently: contract sources/artifacts, lock--attestation
      backlink, Python/Torch determinism, live CUDA/GPU state, pinned Torch
      sources, process/event/snapshots, and immutable generated-module closure.
      Confirm failed/error manifests cannot authorize acquisition.
- [ ] Recompute key/value replacement counts and native byte hash from the
      sanitized wrapper.

Reviewer notes:

> Pending.

## D. Replacement image, container, and runtime

- [ ] Confirm the P24 image recipe was built with `--no-cache` from the clean
      P25 preregistration tree and exact base digest.
- [ ] Independently select the immutable repository digest from retained
      BuildKit metadata; do not trust a mutable image tag.
- [ ] Confirm a fresh attempt root was empty and no historical P23/P24 artifact
      was overwritten, truncated, or mounted writable.
- [ ] Confirm the replacement container has a read-only root, network mode
      `none`, exactly ten bind mounts, exactly one locked `/tmp` tmpfs, and one
      pinned full non-MIG GPU UUID.
- [ ] Confirm all host evidence files existed before container launch, and the
      running-container inspection and namespace-entered mountinfo were
      updated in place rather than by rename.
- [ ] Recompute the new P25-named lock and attestation, their backlink, image,
      container, mount namespace, interpreter, environment, GPU, driver,
      CUDA/PyTorch, and deterministic-state fields.
- [ ] Confirm the new lock/attestation and image digest were reviewed and
      committed before the corrected diagnostic.
- [ ] Confirm the diagnostic commit is one direct child of the preregistration
      commit and adds only the new P25 lock and attestation; confirm both paths
      were absent from the preregistration tree.

Reviewer notes:

> Pending.

## E. One corrected diagnostic attempt

- [ ] Confirm standard-library preflight completed while Torch was absent from
      `sys.modules`.
- [ ] Confirm Torch import did not initialize CUDA before deterministic state
      configuration.
- [ ] Reconstruct the exact ordered configuration and verification of all
      deterministic algorithms, debug, thread, cuDNN, TF32, reduced-reduction,
      matmul-precision, and SDPA fields.
- [ ] Confirm CUDA remained uninitialized after configuration.
- [ ] Confirm the CPU FP32 `Parameter` was constructed before `torch.optim.SGD`
      and no model, data, CUDA forward/backward, Muon candidate, or optimizer
      step ran.
- [ ] Confirm the observed generated module has the exact immutable image path,
      SHA-256, byte count, and read-only effective mount.
- [ ] Confirm every pre/post source and runtime binding agrees exactly.
- [ ] Confirm exactly one P25 diagnostic was attempted and its native and
      sanitized outputs were fresh, nonaliasing, no-overwrite paths.
- [ ] On any failed check, confirm execution stopped before gradient
      acquisition and retained the first failure rather than retrying.

Reviewer notes:

> Pending.

## F. Conditional unchanged P23 acquisition

- [ ] Confirm `trace_off_a` began only after a passing native P25 diagnostic, a
      passing P25 sanitized wrapper, and the reviewed diagnostic-to-acquisition
      commit bridge.
- [ ] Confirm the bridge is one direct child of the diagnostic commit and adds
      only the sanitized P25 wrapper. Reconstruct the contract, rehash every
      frozen source, strictly parse the wrapper, and rehash the retained native
      bytes against its SHA-256 and byte-count binding before gradients.
- [ ] Confirm trace-off A/B are fresh processes and every run has a distinct
      required failure-output path.
- [ ] Recover zero exact off-A/off-B mismatches without a tolerance or
      `allclose`.
- [ ] Confirm trace-on was mechanically barred until that exact report passed.
- [ ] Recover zero exact off-A/trace-on state, loss, data, and RNG mismatches.
- [ ] Confirm trace-on contains exactly `48 * 24 = 1,152` actual post-aspect
      CUDA candidate observations over all `84,934,656` Muon elements.
- [ ] Confirm aggregation ran only after exact noninterference passed and used
      every unchanged P21/P22 activation, correction, cosine, amplitude, P16,
      P18, and mild-intervention threshold.
- [ ] Confirm every success or first-failure artifact was retained and
      sanitized without removing unequal witnesses.
- [ ] Confirm no sanitizer subprocess ran between successful acquisition
      stages: sanitization begins only after the first terminal failure or
      after successful aggregation.

Reviewer notes:

> Pending.

## G. Claim boundary

- [ ] Confirm CPU CI is described only as contract/source reconstruction and
      does not authenticate remote A100 execution.
- [ ] Confirm a passing minimal diagnostic is not called CUDA repeatability,
      real-gradient fidelity, native shield correctness, or training evidence.
- [ ] Confirm the P18--P21 theorems are unchanged and neither the neural loss
      nor unrepaired upstream Muon is claimed globally stable.
- [ ] Confirm the next branch is selected mechanically from the first terminal
      P25/P23 outcome, without threshold tuning or favorable reruns.

Reviewer notes:

> Pending.

## Sign-off

Reviewer name:

Affiliation or relationship:

Review date:

Verdict: `approve` / `approve with corrections` / `reject`

Signature or verifiable review reference:
