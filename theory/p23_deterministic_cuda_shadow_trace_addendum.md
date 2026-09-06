# P23 deterministic-CUDA shadow-trace addendum

## Status and authority

P23 is **protocol-ready and acquisition-blocked**.  No P23 CUDA run, shadow
observation, fidelity statistic, or training-quality result is present.  The
machine-readable authority is
`experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json`.
Its exact bytes must be hashed by each preflight and run; documentation must
not substitute a manually copied digest.

This addendum narrows, but does not rewrite, the frozen P22 acquisition.  It
inherits the P22 protocol with SHA-256
`f88eda60b366b561d277e41335b02c3fb9861a2604a947f8dd5c28fb54bedeac`,
the P22 arithmetic erratum with SHA-256
`42dc57bd8fb03b619c0de75c58570991ca520131ab7d1660b4f9373fd756a2cd`,
and P22 diagnostic commit
`10d3c8becb98267135c29cb504a0ab5d6e5ff886` is the branch point.  Those
inherited files remain unchanged.  In particular, P23 retains the pinned nanoGPT and Muon revisions,
FineWeb bytes, GPT-2-small configuration, seed `1337`, 256 optimizer steps,
24 capture steps, `eta=1/120`, `beta=19/20`, Nesterov momentum, and zero
weight decay.  The P21 activation, correction, cosine, amplitude, P16, P18,
and mild-intervention gates are unchanged.

P23 adds a CUDA-only execution lock, stronger provenance, and a versioned
capture manifest.  A populated runtime lock generated on the selected host
must be reviewed and committed before acquisition.  The checked-in
`p23_cuda_runtime_lock.template.json` contains null identity fields and is
deliberately invalid for a run.  Its software contract fixes
`python_executable` to `/opt/p23-venv/bin/python` and requires a populated
`python_executable_sha256`, exact Python major/minor `3.12`, and the executable
search path
`/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`.
This exact-field tightening occurred before any valid P23 runtime lock or
acquisition existed, so there is no earlier populated v2 artifact with which
compatibility is claimed.
The separate
`p23_host_attestation.template.json` is likewise non-executable: a populated,
sanitized host artifact and its populated runtime lock must both be reviewed,
tracked, and committed before acquisition.  Every acquisition, verifier, and
aggregation command requires both `--runtime-lock` and `--host-attestation`;
their exact tracked bytes enter the run identity.  The `freeze-runtime`
command creates those artifacts, while offline sanitization validates and
transforms an already retained, byte-bound native artifact without claiming a
new live CUDA attestation.

## Locked CUDA execution

The acceptance execution uses exactly one explicitly selected, non-MIG,
BF16-capable NVIDIA GPU.  CPU, Apple MPS, a second device, and an ordinal-only
device selection are not substitutes.  The populated lock records the full
GPU UUID and PCI identity, model, VBIOS, compute capability, multiprocessor
count, memory, driver, Python and PyTorch builds, compiled CUDA and cuDNN
versions, platform, and an OCI image selected by an immutable `sha256`
repository digest.  Host image inspection is linked to inspection of the
actual running container by immutable image ID.  The lock also records its
canonical 64-hex container ID, default hostname, disabled network mode, and
exact allowlisted bind mount modes.  The root filesystem is read-only, with only the declared writable
evidence bind and an exact one-entry `/tmp` tmpfs
(`rw,noexec,nosuid,nodev,size=1073741824`).

Every freeze, acquisition, verifier, and aggregation process in the CUDA
evidence chain invokes the image-resident `/opt/p23-venv/bin/python` directly.
Sanitization is an offline, byte-bound transformation of already retained
artifacts; it does not assert a fresh CUDA/runtime measurement.  The Python
environment is built before the OCI repository digest is frozen. The checked-in
`experiments/training/p23_runtime.Dockerfile`, exact-version requirements, and
path-only repository hook are the reviewed bootstrap recipe. They pin the base
image reference, Python `3.12.14`, Torch `2.7.0+cu128`, and the declared binary
wheel closure, but do not claim byte-identical rebuilds: Debian indexes and
downloaded wheels are not separately content-hash locked. The built and
published `repo@sha256:...` image digest is authoritative, and any rebuild
requires a new runtime lock and host attestation. Its complete
Python 3.12 `sys.flags` map and major/minor version are frozen to the ordinary
nonoptimized, nonisolated, environment-honoring invocation. A repository
`.venv`, any invocation through `uv`, runtime dependency synchronization, and
package installation are forbidden on the read-only acquisition mount.  The OCI
digest identifies the image layers, while the host-inspected mount table and
frozen process environment delimit allowed external influence.  The populated
runtime lock additionally records the interpreter invocation path and the
resolved target's SHA-256.
The acquisition container intentionally runs as root. The image admits the
three possibly host-UID-owned read-only source binds through exact system Git
`safe.directory` entries for their frozen container paths; no wildcard
exception is permitted.
This is procedural host inspection, not cryptographic remote attestation and
not protection against a malicious or compromised host.

The operation graph selects only the math scaled-dot-product-attention
backend.  Flash, memory-efficient, and cuDNN SDPA are disabled.  The following
settings are mandatory before Torch import or CUDA initialization where the
runtime requires that timing:

- `CUDA_VISIBLE_DEVICES` names one full-GPU UUID;
- `NVIDIA_VISIBLE_DEVICES` names that identical full-GPU UUID;
- `CUBLAS_WORKSPACE_CONFIG=:4096:8`;
- `PYTHONHASHSEED=1337`;
- `PATH=/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`,
  so a writable evidence, data, or temporary directory cannot directly shadow
  `git`, `nvidia-smi`, or another bare executable name through search order;
- `VIRTUAL_ENV=/opt/p23-venv`, `PYTHONNOUSERSITE=1`, and
  `PYTHONDONTWRITEBYTECODE=1`;
- `PYTHONOPTIMIZE` is absent and the complete frozen interpreter-flag map is
  checked before Torch import;
- `PYTHONPATH` and `PYTHONHOME` are absent, so arbitrary mounted or
  user-controlled packages cannot shadow the image-resident environment; the
  image-pinned path-only `.pth` adds only the declared read-only OptimizationML
  `src` root;
- `LD_PRELOAD`, `LD_LIBRARY_PATH`, and `LD_AUDIT` are absent, so the declared
  image-library boundary is not silently replaced by loader overrides;
- deterministic algorithms and deterministic debug mode `error`;
- TF32 and reduced-precision FP16/BF16 reductions disabled;
- cuDNN benchmarking disabled and deterministic mode enabled;
- FP32 matmul precision `highest`, no autocast, and no compilation; and
- one CPU intra-op and one inter-op thread.

The host-side running-container inspection also has to show exactly one
`HostConfig.DeviceRequests` entry with empty `Driver`, zero `Count`, that same
single full-GPU UUID in `DeviceIDs`, capabilities exactly `[["gpu"]]`, and an
empty `Options` map. The inspected `Config.Env` must retain exactly one copy of
each UUID-valued visibility variable. This is the locked Docker 29 native-CDI
selection path; an ordinal, `all`, a second request, a MIG identifier, or an
environment-only selection fails closed. The detached shell used to keep the
container alive is not evidence. Every freeze/acquisition/verifier/aggregation
role is a fresh `docker exec` process and must observe both visibility variables
as the UUID before importing Torch; its one-device CUDA and `nvidia-smi`
identities are then cross-checked against the host request.

A missing or unequal runtime-lock field blocks before model allocation.  A
passing run is evidence only for that exact inspected container, device,
driver, build,
source tree, and data identity; it is not an all-CUDA determinism or native
shield theorem.

The lock distinguishes PyTorch's compiled CUDA version from the actual
`cudaRuntimeGetVersion` result.  It also binds the complete Torch configuration
digest, OS and kernel identity, libc and architecture, and a canonical stable
CPU record (vendor, family, model, stepping, microcode, logical count, and a
sorted feature-flags digest).  Volatile `/proc/cpuinfo` fields such as current
clock frequency are deliberately excluded.  The host attestation binds hashes
of the raw image inspection, actual-running-container inspection, live
mountinfo snapshot, and `nvidia-smi` evidence without retaining machine-local
source paths. The live
process hostname must equal the inspected container's canonical default
hostname (the first 12 characters of its full ID).

The supported `freeze-runtime` command in
`run_p23_deterministic_cuda_shadow_trace.py` creates both populated artifacts;
they are not hand-authored. It consumes one structured host `docker image
inspect` object whose `RepoDigests` contains the exact requested image, one
`docker inspect` object for the actual running container, an exact host-side
`nsenter` snapshot obtained by entering the mount, PID, and cgroup namespaces
of the init PID named by that object and reading `/proc/self/mountinfo`, and a
single-row CSV from the frozen seven-field `nvidia-smi` query (including
`mig.mode.current`). A direct host read of `/proc/<State.Pid>/mountinfo` is not
substitutable because Linux may render the same private cgroup mount root
differently across the host and container cgroup namespace views.
It requires the running record's image ID to equal the inspected image ID, its
launch reference to equal the pinned digest reference, its default hostname to
match the live process, its network mode to equal `none`, and its mount
destinations/types/read-only modes to
equal the frozen ten-entry allowlist. The host-side namespace-entered mountinfo
digest must also match `/proc/self/mountinfo` in every evidence process byte
for byte, so Docker's host metadata is not the sole evidence for the
read-only/read-write view. It also
requires `ReadonlyRootfs=true`
and the exact `/tmp` tmpfs map. It also requires the exact singleton full-GPU
Docker `DeviceRequests`/`Config.Env` binding described above, records that
binding in both sanitized artifacts, and cross-checks the GPU row against the
live CUDA device, requires `Disabled`, calls `cudaRuntimeGetVersion`, validates
the completed artifacts, and refuses existing or `.template.json` outputs. The
runtime lock binds the exact host-attestation bytes; no reverse hash is claimed
or needed.
The complete executable freeze, mount/copy, acquisition, verification,
aggregation, and sanitization runbook is in `experiments/README.md`.  The
external nanoGPT checkout, Muon source, FineWeb manifest and every referenced
data file, clean repository, and native evidence directory must retain the
same absolute container paths across all fresh processes.  In particular, the
immutable P22 data tree must be mounted at
`/private/tmp/optimizationml-p22-data`, placing the manifest at
`/private/tmp/optimizationml-p22-data/materialized/p22_fineweb_manifest.json`.
That manifest also records the preprocessor at
`/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py`;
the selected image and acquisition mount layout must expose a read-only bind
of that one file at the recorded alias.  The clean primary repository remains
the distinct `/workspace/OptimizationML` mount, so satisfying the historical
manifest path does not create a second Git worktree or change repository
provenance.

## Repository and source provenance

Every run records the contemporaneous OptimizationML `HEAD`, tree, origin,
submodule state, exact porcelain-v1-z bytes and digest, and requires a clean
tree.  It retains the size and SHA-256 of every tracked file.  After the frozen
data memmap, model, optimizer, and observer are initialized, it records two
complementary views:

- the compatibility source map of every file-backed module under the
  OptimizationML, pinned nanoGPT, and pinned Muon roots; and
- a complete loaded-file snapshot of every file-backed `sys.modules` entry,
  including its `__file__`, `__spec__.origin`, existing `__cached__` bytecode,
  and recoverable source file, plus every regular file-backed pathname in
  `/proc/self/maps`.

The complete snapshot is repeated after training.  An initialized module or
mapped file may not disappear, relocate, or change bytes.  Legitimate lazy
loads are retained as explicit module/mapped-file additions; those additions
enter run identity and must agree exactly across off-A, off-B, and trace-on.
This is a pair of synchronized boundary snapshots, not a temporal import/event
trace.  `PYTHONDONTWRITEBYTECODE=1` and the pre-import controlled-root scan
additionally reject stale `.pyc`/`__pycache__` execution artifacts in the three
mutable source mounts.

All records use stable logical roots, and every retained byte is rehashed by
the independent verifier.  The verifier also reproduces the frozen
data/model/optimizer setup and requires exact initialized membership.  A
historical completed membership set cannot be inferred without rerunning the
256 training steps; instead it is corroborated by exact equality across the
three acquisition identities while every recorded completed artifact is
independently rehashed.  Equality of an opaque manifest-supplied digest alone
is insufficient.

Module origins and executable file mappings are permitted only from the
read-only OptimizationML, nanoGPT, Muon, image-resident Python-environment, or
image-layer roots.  Execution from the data tree, historical preprocessor
alias, writable native-evidence mount, `/tmp` tmpfs, or writable `/dev/shm`
aborts.  Non-executable mapped data such as the frozen `train.bin` memmap is
retained and hashed.
Anonymous mappings have no file bytes to hash and are outside this file-backed
closure; compilation is disabled, so an unexpected file-backed JIT artifact
from a writable/tmpfs origin fails closed.  The host NVIDIA driver and any
injected regular shared objects actually mapped into the process are covered by
the `/proc/self/maps` hashes.  This remains procedural process evidence rather
than cryptographic remote attestation or protection against a compromised
host.

The OptimizationML, nanoGPT, and Muon trees are the experiment-controlled
sources. Torch, the standard library, and other third-party image files are
not individually described as Git-controlled by this experiment; their loaded
bytes are instead covered by the complete module/mapping closure and the
pinned image and runtime evidence above.

The run-identity digest is derived from the actual protocol/addendum/erratum,
repository, initialized/completed loaded-file closure and additions, run, data,
and runtime maps.  The verifier recomputes the digest and compares the complete
maps field by field.

Machine-local absolute paths may be replaced only by deterministic logical
root references in the committed copies.  Sanitization must retain every
semantic field needed to replay the exact decisions.  The historical
preprocessor alias is declared as its own logical sanitization root because it
is outside both the clean `/workspace/OptimizationML` repository mount and the
`/private/tmp/optimizationml-p22-data` tree.  Raw local artifacts remain
separately hash-bound. The exact safe `PATH` is a frozen semantic value, not a
machine-local path, and is therefore retained verbatim by sanitization.

## Model and optimizer binding

P23 constructs the GPT model on the host, moves it to selected `cuda:0`, and
only then enumerates aliases and parameter groups and constructs the
optimizer.  The runner subsequently requires:

- every unique model parameter is the same Python `Parameter` object held by
  exactly one optimizer group;
- every parameter is FP32 and on the selected CUDA device;
- the Muon group contains exactly the frozen 48 names and 84,934,656
  elements, with the complete frozen shape inventory;
- the auxiliary group is the exact complement; and
- the tied token-embedding/language-model-head storage relationship is the
  only declared alias.

The resulting named parameter-to-group, object-identity, device, dtype,
shape, element-count, and alias inventory is part of run identity.  This
removes the P22 portability assumption created by constructing the optimizer
before `model.to(...)`.

## Capture semantics

P23 separates candidate execution from candidate observation.  The optimizer
executes the pinned CUDA candidate in every run, but a trace-off manifest says
`not_observed`, has zero capture steps, zero observations, and zero captured
post-aspect candidates.  It must not repeat P22's ambiguous
`actual_accelerator_candidates: true` wording.

A trace-on manifest says `observed` and must contain exactly

\[
  24\ \text{steps}\times48\ \text{matrices}=1152
\]

observations and 1,152 actual post-aspect candidate captures.  Each comparator
is the stored BF16 CUDA value returned by the pinned Muon call.  A CPU
reconstruction is forbidden.  The P20 shield remains a detached CPU shadow
and is never returned to the baseline optimizer.

## Mechanical run order and exact gates

The only permitted order is:

1. fresh-process `trace_off_a`;
2. fresh-process `trace_off_b`;
3. exact repeatability verification;
4. fresh-process `trace_on`, only when the hash-bound repeatability report
   records zero mismatches;
5. exact off-A/trace-on noninterference verification; and
6. fidelity aggregation, only when both exact verifiers record zero
   mismatches.

The two comparisons use no numeric tolerance and never call `allclose`.
Off-A/off-B must match exactly in the inherited state, loss, data, and RNG
schedule and in the newly bound repository/source/runtime maps.  Off-A and
trace-on must then match exactly in every training-state field.  The trace-on
observer must preserve all recorded RNG states.  The complete 48-name and
84,934,656-element inventory must appear at every one of the 24 capture
steps.

Only after both equality gates pass may the existing frozen hard, fidelity,
and mild-intervention rules be interpreted.  Passing them is empirical
evidence about the pinned execution, not a proof of neural-loss PL or global
Muon fidelity.

## Outcome routing and present blocker

- Exact CUDA baseline-repeatability failure routes P24 to operation-level
  first-difference localization.
- Exact observer-noninterference failure routes P24 to observer redesign.
- A valid trace that fails a frozen fidelity or intervention gate routes P24
  to a preregistered adapter evaluated on a held-out seed; the observed gate
  is not weakened.
- Passing every gate routes P24 to a native CUDA sector shield and then a
  matched shielded-training study.

The current checkout has no CUDA device and the runtime-lock template is not
populated.  Consequently P23 can presently validate only the contract,
fail-closed verifier, provenance logic, and CPU fixtures.  No diagnostic or
checkpoint tag is warranted until an actual pinned CUDA acquisition reaches
an outcome.  P18--P21 remain unchanged.
