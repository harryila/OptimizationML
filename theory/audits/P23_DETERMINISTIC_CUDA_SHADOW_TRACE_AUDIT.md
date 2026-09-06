# P23 deterministic-CUDA shadow-trace audit

## Status

**Unsigned pre-acquisition review packet.**  P23 is protocol-ready, but its
host-generated CUDA runtime lock is pending and no P23 CUDA execution exists.
Unchecked acquisition items must not be cited as completed review or empirical
evidence.

## Materials under review

- `experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json`;
- `theory/p23_deterministic_cuda_shadow_trace_addendum.md`;
- `experiments/training/p23_cuda_runtime_lock.template.json`;
- `experiments/training/p23_host_attestation.template.json`;
- `experiments/training/p23_runtime.Dockerfile`, its exact-version
  requirements, and its path-only repository hook;
- the P23 runner, provenance, verifier, and evidence code named by the final
  source snapshot;
- the unchanged P22 protocol and arithmetic erratum; and
- after acquisition, the complete sanitized run manifests, comparison
  reports, trace record, aggregate, and scoped P23 results summary.

The P22 Apple-MPS diagnostic is historical input, not P23 CUDA evidence.  P23
does not modify any P18--P21 theorem or certify the unshielded baseline.

## A. Frozen inheritance and preregistration

- [ ] Recompute the P22 protocol and erratum SHA-256 values and match the P23
      addendum exactly.
- [ ] Confirm the P22 protocol and erratum bytes are unchanged from diagnostic
      commit `10d3c8becb98267135c29cb504a0ab5d6e5ff886`.
- [ ] Confirm seed, model, data, optimizer, 256-step schedule, 24 captures, and
      every P21/P22 metric threshold are unchanged.
- [ ] Confirm the P23 addendum and populated runtime lock were committed before
      the first CUDA gradient or candidate was inspected.
- [ ] Confirm no result, C29 claim, diagnostic tag, or checkpoint tag predates
      acquisition.

Reviewer notes:

> Pending.

## B. CUDA and container identity

- [ ] Verify exactly one full, non-MIG, BF16-capable NVIDIA GPU is visible by
      UUID and matches the populated lock's PCI identity, model, VBIOS,
      capability, multiprocessor count, and memory.
- [ ] Independently verify driver, compiled/runtime CUDA, cuDNN, PyTorch build
      SHA/configuration, Python, NumPy, OS, kernel, CPU, and architecture.
- [ ] Verify every freeze/acquisition/verifier/aggregation command directly used
      `/opt/p23-venv/bin/python`, that invocation path and the resolved target's
      SHA-256 match the runtime lock, and the clean repository contains no `.venv`.
      Confirm exact Python major/minor `3.12`, the frozen safe
      `PATH=/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`,
      `VIRTUAL_ENV=/opt/p23-venv`, `PYTHONNOUSERSITE=1`, and absent
      `PYTHONPATH`/`PYTHONHOME`/`PYTHONOPTIMIZE` were enforced before Torch
      import. Reconstruct and match the complete frozen Python 3.12 `sys.flags`
      map; version-divergent, PATH-divergent, optimized, isolated, or
      environment-ignoring invocations must fail.
      Confirm no `uv`, `pip`, dependency sync, or package installation ran
      during the CUDA evidence chain. Treat sanitization only as an offline,
      byte-bound transformation; do not infer a live CUDA measurement from it.
- [ ] Verify the running OCI image against the host-attested immutable
      repository digest; an image tag or in-container assertion alone is
      insufficient.
- [ ] Verify the reviewed `linux/amd64` runtime recipe was built without build-argument
      overrides, both pip operations admitted binary wheels only and resolved no
      undeclared dependencies, and `pip check` passed. Treat the final published
      repository digest as authoritative; do not infer byte-reproducible rebuilds
      from the recipe's unhashed apt indexes or wheel downloads.
- [ ] Confirm the root-run image has exact system Git `safe.directory` entries
      for only the three frozen read-only source roots. Reject a wildcard or an
      entry covering the writable evidence, data, or temporary roots.
- [ ] Confirm the retained running-container inspection has
      `HostConfig.NetworkMode=none` and the sanitized container identity binds
      the same disabled-network mode.
- [ ] Confirm the retained inspection has exactly one GPU
      `HostConfig.DeviceRequests` entry: empty driver, zero count, only the
      locked full-GPU UUID in `DeviceIDs`, capabilities exactly `[["gpu"]]`,
      and empty options. Confirm `Config.Env` contains exactly one matching
      UUID-valued `CUDA_VISIBLE_DEVICES` and `NVIDIA_VISIBLE_DEVICES` entry,
      and every fresh evidence process sees those same values. Cross-check the
      requested UUID against the sole live CUDA device and the non-MIG
      `nvidia-smi` identity; do not treat the long-lived shell PID 1 as an
      evidence process.
- [ ] Re-obtain the running-container init PID from the retained Docker
      `State.Pid`; independently repeat the host-side `/usr/bin/nsenter` into
      that PID's mount, PID, and cgroup namespaces; rehash the resulting
      `/proc/self/mountinfo` bytes; and confirm every freeze, acquisition,
      verifier, and aggregation process reports that exact digest for its live
      `/proc/self/mountinfo`. Do not substitute a direct host read of
      `/proc/<State.Pid>/mountinfo`, whose private-cgroup root presentation may
      differ. Confirm the live root and declared source/data/evidence paths
      have their required read-only or read-write modes independently of
      Docker's host inspection metadata.
- [ ] Confirm the math SDPA backend alone is enabled and no flash,
      memory-efficient, or cuDNN SDPA path is admissible.
- [ ] Confirm every frozen deterministic environment and Torch setting was
      applied before its required initialization boundary.
- [ ] Confirm no runtime-lock field is null, inferred after acquisition, or
      unequal across the three processes.

Reviewer notes:

> Pending.

## C. Repository and complete loaded-file closure

- [ ] Verify contemporaneous `HEAD`, tree, origin, submodules, exact status
      bytes, and `dirty=false` independently for each run.
- [ ] Recompute every tracked-file size/hash and the canonical tracked-tree
      digest from the recorded commit.
- [ ] Recompute every initialized/completed file-backed `sys.modules` artifact:
      `__file__`, spec origin, existing cached bytecode, and recoverable source,
      including stdlib, site-package/Torch, OptimizationML, nanoGPT, and Muon.
- [ ] Recompute every regular file-backed `/proc/self/maps` hash, including
      mapped native and driver libraries. Confirm deleted and unrooted paths
      fail closed and anonymous mappings are explicitly outside the claim.
- [ ] Confirm the completed closure removes or changes no initialized entry.
      Reconstruct every recorded lazy-load addition and require the complete
      initialized/completed/additions object to agree across all three runs;
      do not describe the two snapshots as a temporal import trace.
- [ ] Confirm no module origin or executable file mapping comes from the data,
      preprocessor-alias, writable native-evidence, or `/tmp` roots. Confirm the
      same for writable `/dev/shm`, while the non-executable frozen `train.bin`
      memmap is retained and hashed.
- [ ] Confirm the claims distinguish the three experiment-controlled source
      trees from Torch and other third-party image dependencies. Their loaded
      bytes are bound by the closure but are not individually called
      Git-controlled sources.
- [ ] Recompute each run identity from its constituent maps rather than
      trusting the stored digest, then require exact field-by-field equality
      of all stable source and runtime maps.
- [ ] Audit sanitization: the declared logical roots include the historical
      preprocessor alias and image-resident Python environment, no private
      absolute path remains, and no semantic field or unequal witness was
      removed. Confirm the exact safe colon-separated `PATH` is retained
      verbatim rather than treated as one absolute path. Match every sanitized
      wrapper to the retained native byte count and SHA-256.

Reviewer notes:

> Pending.

## D. Model/optimizer construction and inventory

- [ ] Confirm the model moves to selected CUDA device 0 before optimizer
      grouping or construction.
- [ ] Confirm optimizer parameters are the exact model `Parameter` objects,
      each occurring once, and all are FP32 on the selected device.
- [ ] Confirm the exact 48-name, 84,934,656-element Muon inventory and its four
      per-layer roles across 12 layers.
- [ ] Confirm the auxiliary group is the exact complement and the only storage
      alias is the tied token embedding/language-model head.
- [ ] Independently reconstruct the named parameter/group/device/dtype/shape
      digest recorded in every run.

Reviewer notes:

> Pending.

## E. Capture truthfulness and observer isolation

- [ ] Confirm each trace-off manifest says `not_observed`, with zero capture
      steps, observations, and captured candidates.
- [ ] Confirm trace-on says `observed` and contains exactly 1,152 records:
      every one of 48 parameters at every one of 24 frozen steps.
- [ ] Confirm every fidelity comparator is the actual stored BF16 CUDA
      post-aspect candidate; no CPU candidate or aspect recomputation enters.
- [ ] Confirm P20 receives detached CPU copies, its output has no return path
      to training, and each observer invocation preserves recorded RNG bytes.
- [ ] Confirm nonfinite values, wrong devices/dtypes, incomplete inventory, or
      forbidden execution origins or loaded-file removals/changes abort and
      discard the acquisition.

Reviewer notes:

> Pending.

## F. Exact sequencing and comparisons

- [ ] Verify trace-off A and B were separate fresh Python processes in the
      same inspected container ID, with the same runtime lock, source, data,
      and initial-state identity.
- [ ] Independently compare every predeclared off-A/off-B field and recover
      exactly zero mismatches without a tolerance or `allclose`.
- [ ] Verify trace-on could not start without the hash-bound passing
      repeatability report.
- [ ] Independently compare off-A and trace-on and recover exactly zero state,
      loss, data, and RNG mismatches.
- [ ] Mutate one source field, runtime field, state digest, and capture count
      in disposable manifests and confirm each fails closed.
- [ ] Confirm fidelity aggregation could not run before both equality reports
      passed.

Reviewer notes:

> Pending.

## G. Frozen fidelity and interpretation

- [ ] Recompute the aggregate from all 1,152 observation records and recover
      the unchanged hard, P16/P18, and mild-intervention decisions.
- [ ] Verify activation, relative-correction, cosine, candidate/output
      amplitude, output/signal amplitude, effective gain, shaping departure,
      and annulus handling use the frozen definitions and nearest-rank
      quantiles.
- [ ] Confirm undefined zero-denominator values are classified rather than
      silently replaced or dropped, and nonfinite values are failures.
- [ ] Confirm the final result follows the preregistered route without changing
      a threshold after observation.
- [ ] Confirm the paper-facing claim is limited to the exact pinned CUDA
      execution and does not assert neural-loss PL, global fidelity,
      unmodified-Muon stability, native-shield correctness, distributed
      parity, throughput, or training quality.

Reviewer notes:

> Pending.

## External reviewer sign-off

Reviewer name:

Affiliation or role:

Date:

Decision: pending

Notes:

> Pending.
