# Experiments

Experiment order is gated:

1. exact and backend-specific executable witnesses;
2. spectral/matrix audit;
3. qualified diagonal matrix quadratics;
4. the completed full-matrix floored certificate;
5. the repaired stylized non-Nesterov quadratic IQC and matched rank-one
   boundary checks;
6. the pinned EMA/Nesterov state-and-signal ordering IQC, exact replay, and
   parity regression;
7. the P4 structure-aware full-step quadratic certificate;
8. the two P5 nonlinear certificates and changing-orientation falsification
   probes;
9. the P6 smooth-PL value certificate and nonconvex, rank-deficient,
   changing-orientation falsification probe;
10. the P7 robust dissipativity certificate and disturbed nonconvex-PL
    falsification probe;
11. the P8 exact fixed-`2 x 2` mixed-precision certificate and CPU diagnostic;
12. the P9 shape-parameterized exact mixed-precision replay and CPU
    native-matmul diagnostic, with realistic shapes enabled only explicitly;
13. the P10 finite-precision outer-shell replay, three-word compensated-master
    checks, raw-FP32 subtraction witness, and modest CPU diagnostic;
14. the P11 exact two-port implementation-margin replay, grid boundaries,
    coordinatewise-maximal frontier slices, and adjacent rejection controls;
15. the P12 exact-real additive-epsilon full-rectangular deficit certificate,
    exact rank-two lower witness, boundary controls, and independent directed-
    interval replay;
16. the P13 exact-real radial-passivation theorem, two-precision logarithmic
    magnitude evaluation, independent rational-log reconstruction, universal
    stiffness lower bound, and explicit-step negative controls;
17. the P14 exact-real Yosida theorem, exact smooth-PL LMI replay, independent
    rational reconstruction, sector-boundary witnesses, and direct versus
    under-regularized versus selected scalar controls;
18. the P15 exact-real inexact-resolvent theorem, sharp graph-residual error
    gains, relative-plus-absolute stopping rule, robust smooth-PL replay, and
    loose-tolerance controls;
19. the P16 exact-real equivariant singular-value solver theorem, guarded
    FP64 computed-residual study, exact/Arb noncollapse witness, and sampled
    stability--fidelity frontier;
20. the P17 exact-real shape-preserving gated-resolvent pointwise-sector
    theorem, two smooth-PL certificates, exact/Arb unsafe-band and canonical
    fidelity controls, and separate FP64 spectrum diagnostic;
21. the P18 exact-real ray-sector projection, useful-rate smooth-PL
    certificate, rational step--rate frontier, canonical Arb gates, and
    separate FP64 fidelity/amplitude diagnostic;
22. the P19 exact-real sector shield, both useful-rate replays, exact-as-stored
    binary64 reference postcondition, corrupted-candidate controls, and
    separate bitwise-inactivity/fidelity diagnostic;
23. the P20 static seven-shape mixed-precision shield certificate, locked CPU
    BF16/FP32 arithmetic graph, radial-clip and near-zero controls, P18 and
    pinned-upstream candidate diagnostics, and cross-platform decision replay;
24. the P21 stored-signal `7 x 7` port certificate, exact seven-shape FP32
    EMA/Nesterov/master roundoff absorption, all-subnormal wrapper, conditional
    decay treatment, and frozen synthetic shadow-observer diagnostic;
25. the frozen P22 real-gradient protocol, exact `768 x 2304` shape extension,
    and blocked Apple-MPS off-A/off-B baseline-repeatability diagnostic;
26. the P23 CUDA-only addendum, null-invalid runtime-lock and host-attestation
    templates, hardened model/optimizer and source/runtime binding, exact
    fresh-process sequencing, and a preregistered 1,152-observation acceptance
    gate with no CUDA result yet;
27. only after a fully supported model shape inventory, real-gradient shadow
    gates, model-forward use of the logical master, aspect scaling, weight
    decay, and implementation-parity gates, a small matched neural-training
    sweep.

The floored Jordan architecture now has a valid global constant-`rho`
certificate, a stylized non-Nesterov quadratic theorem, and an exact rational
certificate for the pinned EMA/Nesterov ordering in real arithmetic. Even at
the numerical stationary repair-margin design, that generic P3 result's
complex-skew necessary boundary is about `9,923.44x` below its matched local
threshold. It is therefore retained as an appendix/proof of principle, not a
practical-stability headline. P4 and P5 subsequently give structure-aware
full-step quadratic and strongly-convex nonlinear results, while P6 retains
the full step for globally smooth PL objectives that may be nonconvex and have
nonunique minimizers. GPU work remains gated on the remaining robustness and
implementation-parity results and a predeclared use of the certified
architecture. P7 supplies the robust post-operator port; P8 and P9 instantiate
that port for proposed proof-reference kernels. P10 closes exact FP32
EMA/Nesterov and compensated logical-master residuals for one guarded
`4096 x 11008` CPU shell, but it does not specify how a model forward pass
consumes the three-word master or establish native accelerator/upstream
parity. P11 then computes exact additional gradient and deployed-operator
acceptance margins without asserting that a production implementation meets
them. P12 separately proves exact `1/epsilon` deficit scaling and a
dimension-uniform four-band upper for the continuous additive-epsilon
surrogate. At the pinned `1e-7` scale its required constant repair is
catastrophically large; this is not a certificate for the discontinuous BF16
backend. P13 proves a nonlinear radial correction can shrink output magnitude
to logarithmic growth, while a pair lower bound and exact scalar controls show
the `1/epsilon` stiffness and explicit-step obstruction remain. P14 replaces
direct evaluation by an exact resolvent: its epsilon-independent incremental
sector restores the full `eta=1/32000` smooth-PL guarantee. P15 permits an
oracle meeting the exact-real graph-residual rule at relative tolerance
`1/250`, retains the P14 rate, and gives an explicit ultimate bound for an
absolute residual floor. P16 supplies the missing exact-real structured
solver and a fail-closed FP64 reference implementation. Every declared case
passes the computed residual check, but the locked operator retains only
about `8.403e-5` of the upstream comparator's spectral shaping and therefore
fails the frozen pre-certificate meaningful-fidelity gate. Its six-point sampled
`(lambda,mu)` frontier finds no joint frozen-P14/fidelity pass; this is not a
global impossibility theorem. P16 provides neither a useful uniform
iteration-count bound nor certified FP64 rounding. P17 retains that resolvent
internally and gates between scaled `Y` and `E_(h,epsilon)(J(S))`. A global
origin-centered pointwise sector yields exact one-trajectory smooth-PL
convergence for a primary ceiling-`3/4`, divisor-`4096`, `eta=1/128000`
high-fidelity point and a secondary ceiling-`1/8`, divisor-`8192`,
full-`eta=1/32000` point. Both pass the unchanged canonical P16 fidelity
thresholds. This is not an incremental theorem and does not propagate P16
solver or finite-precision error through the gate. P18 then obtains the
useful-rate sector, and P19 makes arbitrary candidates safe with an exact-
checked binary64 projection. P20 replaces that runtime exact check for seven
fixed shapes with a static rounding certificate around a locked CPU
BF16/FP32-to-FP32 pass-through/radial-clip/half-fallback graph. Its conditional
rate inheritance requires every call along a trajectory to succeed and treats
stored `S` as the abstract port signal; it does not yet compose the outer
signal cast or optimizer arithmetic. Each run
writes a self-contained JSON manifest and compact CSV
tables under `results/summaries/`; the JSON records inputs, operator details,
dtype, seed, software, hardware, and Git state.

Replay the CPU-only momentum result with:

```bash
uv run --locked python scripts/certify_momentum_stability.py \
  --output results/summaries/momentum_iqc_certificate.json
uv run --locked python scripts/certify_ema_nesterov_stability.py \
  --output results/summaries/ema_nesterov_iqc_certificate.json
```

The exact rational LMI and Jury signs carry the claims. The committed float64
trajectories are matched diagnostics, not sampled global certificates.
The first command studies the explicitly stylized non-Nesterov recurrence;
the second matches the pinned EMA state and Nesterov signal ordering after
substituting the repaired floored map and omitting weight decay. Neither P3
certificate is a BF16, nonquadratic, stochastic, or neural-network theorem;
the separate P5 and P6 certificates establish deterministic exact-arithmetic
nonlinear strongly-convex and smooth-PL cases, respectively.

Replay the P6 proof and its separate sampled diagnostic with:

```bash
uv run --locked python scripts/certify_pl_convergence.py \
  --output results/summaries/pl_convergence_certificate.json
uv run --locked python experiments/nonconvex/run_pl_falsification.py \
  --output results/summaries/pl_falsification.json
```

The first command replays the exact rational certificate. The second runs a
72-case CPU/float64 falsification grid containing negative curvature,
changing Hessian orientations, and nonunique minimizers. A sampled failure can
falsify the implementation; sampled passes do not prove the theorem.

Replay the P7 robust theorem, P8 fixed-shape theorem, P9 scalable theorem, and
P10 outer-shell theorem with their independent reconstructions:

```bash
uv run --locked python scripts/certify_robust_dissipativity.py \
  --output results/summaries/robust_dissipativity_certificate.json
uv run --locked python scripts/reconstruct_robust_dissipativity.py
uv run --locked python scripts/certify_mixed_precision.py \
  --output results/summaries/mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_mixed_precision.py
uv run --locked python scripts/certify_scalable_mixed_precision.py \
  --output results/summaries/scalable_mixed_precision_certificate.json
uv run --locked python scripts/reconstruct_scalable_mixed_precision.py \
  --require-canonical
uv run --locked python scripts/certify_outer_loop_roundoff.py \
  --output results/summaries/outer_loop_roundoff_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_roundoff.py \
  --require-canonical
uv run --locked python scripts/certify_implementation_margin.py \
  --output results/summaries/implementation_margin_certificate.json
uv run --locked python scripts/reconstruct_implementation_margin.py \
  --require-canonical
uv run --locked python scripts/certify_additive_epsilon_deficit.py \
  --output results/summaries/additive_epsilon_deficit_certificate.json
uv run --locked python scripts/reconstruct_additive_epsilon_deficit.py \
  --require-canonical
uv run --locked python scripts/certify_radial_passivation_tradeoff.py \
  --output results/summaries/radial_passivation_tradeoff_certificate.json
uv run --locked python scripts/reconstruct_radial_passivation_tradeoff.py \
  --require-canonical
uv run --locked python scripts/certify_yosida_stability.py \
  --output results/summaries/yosida_stability_certificate.json
uv run --locked python scripts/reconstruct_yosida_stability.py \
  --require-canonical
uv run --locked python scripts/certify_inexact_yosida_robustness.py \
  --output results/summaries/inexact_yosida_robustness_certificate.json
uv run --locked python scripts/reconstruct_inexact_yosida_robustness.py \
  --require-canonical
uv run --locked python scripts/certify_equivariant_resolvent_solver.py \
  --output results/summaries/equivariant_resolvent_solver_certificate.json
uv run --locked python scripts/reconstruct_equivariant_resolvent_solver.py \
  --require-canonical
uv run --locked python scripts/certify_shape_preserving_resolvent.py \
  --output results/summaries/shape_preserving_resolvent_certificate.json
uv run --locked python scripts/reconstruct_shape_preserving_resolvent.py \
  --require-canonical
uv run --locked python scripts/certify_sector_projected_useful_rate.py \
  --output results/summaries/sector_projected_useful_rate_certificate.json
uv run --locked python scripts/reconstruct_sector_projected_useful_rate.py \
  --require-canonical
uv run --locked python scripts/certify_sector_shielded_inexact_resolvent.py \
  --output results/summaries/sector_shielded_inexact_resolvent_certificate.json
uv run --locked python scripts/reconstruct_sector_shielded_inexact_resolvent.py \
  --require-canonical
uv run --locked python scripts/certify_scalable_sector_shield.py \
  --output results/summaries/scalable_sector_shield_certificate.json
uv run --locked python scripts/reconstruct_scalable_sector_shield.py \
  --require-canonical
```

The P9--P20 generators and standard-library-only reconstructions carry their
exact claims. P11's adjacent controls establish rejection of the sufficient
certificate, not actual closed-loop instability. P12's older sampled epsilon
grid is discovery evidence only; its exact/interval generator carries the
global upper. P15's residual and base-operator evaluations are exact real;
its stopping rule is not a finite-precision solver claim. P16's exact/Arb
artifact proves algebraic noncollapse, while its FP64 solver and sampled
frontier are diagnostics and do not certify rounding error or a global
fidelity impossibility. P17's exact sector is pointwise and origin-centered,
not incremental; its exact-real stability theorem assumes the exact
resolvent, while its separate FP64 spectrum grid is diagnostic evidence only.
P18's ray projection is likewise pointwise rather than incremental. Its
selected useful-rate theorem assumes the exact resolvent, while the
computed-residual-checked FP64 diagnostics do not propagate solver or
rounding error through the nonlinear projection. P19 makes arbitrary finite
candidates pointwise safe by final projection into P18's sector. Its fixed-
signal nonexpansiveness does not turn the P15 graph residual alone into a
final nonlinear P18 candidate-error bound. The locked NumPy binary64 shield
exact-checks every successful stored result and may fail closed when no
certified representable result exists; it is not a portable backend theorem.
P20's exact shape recurrence instead pays statically for the locked scaled
balanced norm, inward comparison, radial clip, and half fallback. It proves
pointwise containment for successful stored results on seven fixed shapes,
not incremental passivity or correctness of an arbitrary CPU/GPU graph. Its
outer-loop rate statement is conditional on every call succeeding and on
identifying stored `S` with the abstract operator-port signal. P21 closes that
identification gap for the locked CPU proof-reference outer loop by placing
the LMI at the actual stored signal and absorbing gradient cast, reused `bg`,
EMA/Nesterov, step, compensated-master, and all-subnormal errors through exact
ports. Its `1/4096` source envelope and nonzero-decay bounds are explicit
premises, not measurements of a production training run.

Replay the P16 guarded reference solver and fidelity diagnostic separately:

```bash
uv run --locked python experiments/resolvent/run_p16_solver_study.py \
  --output results/summaries/p16_solver_study.json
```

The default study runs nine modest dense cases and complete singular-value
vectors for four representative Transformer shapes through `4096 x 11008`.
It locks additive `epsilon=1e-7`, the five Jordan stages with coefficients
`6889/2000`, `-191/40`, and `4063/2000`, `lambda=1/1000`, `mu=1000`, and
KellerJordan/Muon source revision
`f98f1cacc0263b04290753e32be8d498c1efc806`.
The large cases do not allocate dense matrices, perform the corresponding
SVD, or benchmark an accelerator. A returned success proves only that the
computed FP64 candidate passes the computed P15 residual test. The exact-real
global convergence theorem and Arb fidelity enclosure are separate from this
diagnostic.

Replay the P17 shape-preserving spectrum and fidelity diagnostic separately:

```bash
uv run --locked python experiments/resolvent/run_p17_shape_preserving_study.py \
  --output results/summaries/p17_shape_preserving_study.json
```

The study evaluates both locked gates on the declared spectrum grid and
complete reduced singular-value vectors for representative Transformer
shapes. It records computed P15 residual checks, fidelity, amplitude, and
rank accumulation. These deterministic FP64 results are not a proof of the
dimension-uniform pointwise sector, a global fidelity extremum, or an exact
rounding envelope. The exact P17 generator and independent reconstruction
carry the smooth-PL theorem and canonical gate decisions.

Replay the P18 sector-projected fidelity and amplitude diagnostic separately:

```bash
uv run --locked python experiments/resolvent/run_p18_sector_projected_study.py \
  --output results/summaries/p18_sector_projected_study.json
```

The study retains the P17 broad grid, sampled operating annulus, realistic
reduced spectra, frozen departure/retention thresholds, and actual computed
P15 residual postcheck. It additionally records projection activity, raw
amplitude, effective-update gates, the unprojected control, and the
`K=1/100` control. These are deterministic FP64 diagnostics, not global
fidelity, inexact-solver, or finite-precision certificates.

Replay the P19 shield inactivity, fidelity, and corruption study separately:

```bash
uv run --locked python experiments/resolvent/run_p19_sector_shielded_study.py \
  --output results/summaries/p19_sector_shielded_study.json
```

The study uses the same additive `epsilon=1e-7`, five Jordan stages with
coefficients `6889/2000`, `-191/40`, and `4063/2000`, `lambda=1/1000`,
`mu=1000`, P18 gate/projection design, and sampled annulus. It records exact-
as-stored disk checks, bitwise identity on normal guarded candidates, both
P18 operating-point replays, computed P15 graph residuals, and deliberately
corrupted candidates. The exact disk theorem and rational smooth-PL LMIs are
separate from these sampled diagnostics. A successful reference-shield return
is exact-checked, but the study is not a general IEEE-754 or accelerator
rounding theorem.

Replay the P20 proof-reference shield study separately:

```bash
uv run --locked python \
  experiments/mixed_precision/run_p20_scalable_sector_shield_study.py \
  --output results/summaries/p20_scalable_sector_shield_study.json
```

The study fixes additive `epsilon=1e-7`, five Jordan stages with coefficients
`6889/2000`, `-191/40`, and `4063/2000`, the P18 interface parameters, the
pinned KellerJordan/Muon revision, and the P20 CPU arithmetic graph. It tests
P18 and literal-upstream candidate families. Only the canonical `2 x 2`
upstream case executes the full matrix graph; the seven Transformer-shape
records are packed singular-coordinate diagnostics. The annulus records
`2671/2688` P18 pass-throughs, `17` radial clips, no half fallbacks, and
`2176/2176` informative fidelity passes; the upstream candidate records
`761` pass-throughs and `1927` clips. Offline exact disk checks validate the
study output but are not part of the runtime. Sampled activation/fidelity and
a matching cross-platform decision digest do not prove global fidelity or
extend the static certificate to another backend.

Replay the P21 exact stored-signal composition and its independent
standard-library reconstruction with:

```bash
uv run --locked python scripts/certify_outer_loop_composition.py \
  --output results/summaries/certified_outer_loop_composition_certificate.json
uv run --locked python scripts/reconstruct_outer_loop_composition.py \
  --canonical results/summaries/certified_outer_loop_composition_certificate.json \
  --require-canonical
```

The core `7 x 7` LMI is dimension independent. The concrete FP32 port
absorption is an invariant-domain result for the seven P20 shapes, the locked
CPU arithmetic graph, and explicitly bounded pre-cast source and decay ports.
It is not an incremental P20 theorem, a generic stochastic-gradient result,
or accelerator parity.

The frozen shadow observer can be exercised without a model using:

```bash
uv run --locked python experiments/training/run_p21_synthetic_shadow_trace.py \
  --output results/summaries/p21_synthetic_shadow_trace.json
```

That command is an infrastructure diagnostic only. Its synthetic output
cannot satisfy or predict the real-gradient empirical gate. The frozen
fixture's 144 observations, 126 pass-throughs, and 18 activations all pass its
synthetic checks; those counts only validate acquisition and aggregation
logic. P22 subsequently added the isolated trainer/data path, native MPS
preflight, and separate fused-QKV `768 x 2304` CPU shield certificate. Its
real-gradient trace-on acquisition nevertheless remains absent because the
two native trace-off baselines failed exact repeatability.

P22 freezes the first real-gradient acquisition separately. Its protocol pins
NanoGPT commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, the audited Muon
commit, GPT-2 small, FineWeb `sample-10BT` revision
`9bb295ddab0e05d785b879661af7260fed5140fc`, `tiktoken==0.14.0`, one
seed, 256 optimizer steps, and P21's existing 24 capture steps. The feasible
pilot uses sequence length 128, batch one, no accumulation, weight decay zero,
and constant Muon `eta=1/120`. The model's architectural block size remains
1024. Read `theory/p22_real_gradient_shadow_trace_protocol_erratum.md`
alongside the frozen protocol: the candidate casts to BF16 before its norm and
additive-epsilon normalization, and trace-on must not run before the exact
off-A/off-B verifier passes.

Run the prerequisite checker with an explicitly selected backend:

```bash
uv run --locked python experiments/training/p22_nanogpt_shadow_trace.py \
  preflight --accelerator mps \
  --nanogpt-root /path/to/nanoGPT \
  --muon-source /path/to/muon.py \
  --fineweb-manifest /path/to/materialized-fineweb.json \
  --instrumentation-patch experiments/training/p22_observed_muon.py \
  --output /path/to/p22-preflight.json
```

Use `--accelerator cuda` only under the CUDA-specific deterministic guard.
MPS output is explicitly Apple-MPS evidence, not CUDA, tensor-core, native-
kernel, or distributed parity. A blocked preflight exits nonzero and never
starts training. The unmaterialized data template is intentionally rejected.

The isolated trainer instrumentation must copy the actual stored signal and
the mandatory actual post-aspect accelerator candidate. It also records an
actual pre-aspect value where the pinned call boundary exposes one, but never
reconstructs that value on CPU. It may not return the P20 shadow output to the
optimizer. The isolated runner imports the pinned NanoGPT `model.py`; it does
not execute or modify upstream `train.py`. Run it in three fresh processes,
using the same explicit backend and frozen prerequisites:

```bash
uv run --locked python \
  experiments/training/run_p22_real_gradient_shadow_trace.py run \
  --trace-mode trace_off --accelerator mps \
  --nanogpt-root /path/to/nanoGPT \
  --muon-source /path/to/muon.py \
  --fineweb-manifest /path/to/materialized-fineweb.json \
  --instrumentation-patch experiments/training/p22_observed_muon.py \
  --output /path/to/trace-off-a-manifest.json
```

Repeat that command in a fresh process for `trace-off-b-manifest.json`, then
enforce the exact baseline gate immediately:

```bash
uv run --locked python experiments/training/p22_nanogpt_shadow_trace.py \
  verify-repeatability \
  --trace-off-a /path/to/trace-off-a-manifest.json \
  --trace-off-b /path/to/trace-off-b-manifest.json \
  --output /path/to/p22-repeatability.json
```

Only if that command passes may a third fresh process run trace-on with the
required raw-capture output:

```bash
uv run --locked python \
  experiments/training/run_p22_real_gradient_shadow_trace.py run \
  --trace-mode trace_on --accelerator mps \
  --nanogpt-root /path/to/nanoGPT \
  --muon-source /path/to/muon.py \
  --fineweb-manifest /path/to/materialized-fineweb.json \
  --instrumentation-patch experiments/training/p22_observed_muon.py \
  --output /path/to/trace-on-manifest.json \
  --raw-trace-output /path/to/p22-raw-trace.json
```

Use `--accelerator cuda` throughout instead only for the frozen one-CUDA-device
profile. Never mix backends across the three runs. After the permitted
trace-on run, enforce noninterference:

```bash
uv run --locked python experiments/training/p22_nanogpt_shadow_trace.py \
  verify-noninterference \
  --trace-off /path/to/trace-off-a-manifest.json \
  --trace-on /path/to/trace-on-manifest.json \
  --output /path/to/p22-noninterference.json
```

That comparison first checks baseline repeatability (off-A against off-B) and
then observer noninterference (off-A against trace-on). It compares all 256
batches, stored losses, and host/selected-accelerator RNG states, plus exact
model/optimizer states initially, at all 24 capture-schedule updates, and
finally. No tolerance is used.

The first Apple-MPS acquisition stopped at the first gate. Off-A and off-B
started from the same exact state and retained identical data-window and RNG
schedules, but their first post-update model and optimizer hashes differed at
step 0; both hashes differed at all 24 scheduled checkpoints and finally. Of
256 loss hashes, 253 differed (only steps 0, 1, and 22 matched), giving 303
total verifier mismatches. The required trace-on run was therefore not
executed. This is a blocked diagnostic for the pinned Apple-MPS/PyTorch
`2.13.0` baseline, not real-gradient shield/fidelity evidence. The next
acquisition gate is the unchanged three-run protocol on one BF16-capable CUDA
device, with exact off-A/off-B equality required before trace-on. See
`theory/p22_real_gradient_shadow_trace_protocol.md` and
`results/summaries/P22_REAL_GRADIENT_SHADOW_TRACE_RESULTS.md`. The compact
path-sanitized record is
`results/summaries/p22_repeatability_failure_evidence.json`; its native
off-A/off-B manifests remain external and hash-locked.

With those native files available, reconstruct the committed sanitized
evidence with:

```bash
uv run --locked python scripts/build_p22_repeatability_failure_evidence.py \
  --native-root /path/to/p22-native-evidence \
  --output-root results/summaries
```

P23 does not rerun or amend that failed MPS acquisition. Its machine-readable
CUDA addendum is
`experiments/training/p23_deterministic_cuda_shadow_trace_addendum.json`; read
`theory/p23_deterministic_cuda_shadow_trace_addendum.md` alongside it. The
checked-in `p23_cuda_runtime_lock.template.json` is intentionally invalid for
acquisition: every host, container, GPU, driver, CUDA/PyTorch, and
deterministic-runtime field must be populated and committed from the selected
CUDA host before model allocation or data collection.  The interpreter is
fixed at `/opt/p23-venv/bin/python`; its resolved path, executable SHA-256,
and complete Python 3.12 `sys.flags` map are runtime-lock fields.  Optimized,
isolated, environment-ignoring, and otherwise flag-divergent invocations are
rejected.  The separate
`p23_host_attestation.template.json` is also intentionally invalid. Every P23
acquisition, verifier, and aggregation command requires a populated
`--host-attestation`; both artifacts must be tracked and committed before
acquisition. Offline sanitization consumes only a retained native artifact and
its declared path roots. The runtime lock binds the
attestation's exact bytes; the validator also compares their container and GPU
maps field by field.

There is no hand-authored acquisition JSON. P23 uses one long-lived,
host-inspected CUDA container and launches every role as a separate fresh
Python process inside it. This is a procedural inspection chain, not
cryptographic remote attestation or protection against a malicious host.
First retain raw host-side image inspection and `nvidia-smi` query output, and
create the files that will receive the actual running-container inspection and
its live mount-namespace snapshot:

```bash
set -euo pipefail
mkdir -p /secure/p23/evidence
docker image inspect registry.example/project@sha256:... \
  > /secure/p23-image-inspect.json
nvidia-smi --id=GPU-... \
  --query-gpu=uuid,pci.bus_id,name,driver_version,vbios_version,memory.total,mig.mode.current \
  --format=csv,noheader,nounits > /secure/p23-nvidia-smi.csv
touch /secure/p23-running-container-inspect.json
touch /secure/p23-running-mountinfo.txt
```

Launch the selected digest-pinned image with `--network none`, `--read-only`, the exact `/tmp`
tmpfs, and exactly the ten bind mounts below. The five source/data mounts and
four host-evidence files are read-only; only the native evidence directory is
writable. The image must already contain every destination parent and the four
host-evidence file mount points. This is the complete launch command; do not add a writable
root, another bind, or another tmpfs:

```bash
set -euo pipefail
export P23_IMAGE=registry.example/project@sha256:...
export P23_GPU_UUID=GPU-00000000-0000-0000-0000-000000000000

docker run --detach --name p23-acquisition \
  --gpus "device=$P23_GPU_UUID" \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,noexec,nosuid,nodev,size=1073741824 \
  --env VIRTUAL_ENV=/opt/p23-venv \
  --env PYTHONNOUSERSITE=1 \
  --env PYTHONDONTWRITEBYTECODE=1 \
  --env CUDA_VISIBLE_DEVICES="$P23_GPU_UUID" \
  --env NVIDIA_VISIBLE_DEVICES="$P23_GPU_UUID" \
  --env CUBLAS_WORKSPACE_CONFIG=:4096:8 \
  --env PYTHONHASHSEED=1337 \
  --env NVIDIA_TF32_OVERRIDE=0 \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  --env P23_REPO=/workspace/OptimizationML \
  --env P23_NANOGPT=/workspace/inputs/nanoGPT \
  --env P23_MUON_ROOT=/workspace/inputs/muon \
  --mount type=bind,src=/secure/p23/OptimizationML,dst=/workspace/OptimizationML,readonly \
  --mount type=bind,src=/secure/p23/nanoGPT,dst=/workspace/inputs/nanoGPT,readonly \
  --mount type=bind,src=/secure/p23/Muon,dst=/workspace/inputs/muon,readonly \
  --mount type=bind,src=/secure/p23/optimizationml-p22-data,dst=/private/tmp/optimizationml-p22-data,readonly \
  --mount type=bind,src=/secure/p23/OptimizationML/experiments/training/materialize_p22_fineweb.py,dst=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py,readonly \
  --mount type=bind,src=/secure/p23/evidence,dst=/workspace/evidence/p23 \
  --mount type=bind,src=/secure/p23-image-inspect.json,dst=/mounted-host-evidence/image-inspect.json,readonly \
  --mount type=bind,src=/secure/p23-running-container-inspect.json,dst=/mounted-host-evidence/running-container-inspect.json,readonly \
  --mount type=bind,src=/secure/p23-running-mountinfo.txt,dst=/mounted-host-evidence/running-mountinfo.txt,readonly \
  --mount type=bind,src=/secure/p23-nvidia-smi.csv,dst=/mounted-host-evidence/nvidia-smi.csv,readonly \
  --entrypoint /bin/sh \
  "$P23_IMAGE" -c 'while :; do sleep 3600; done'

docker inspect p23-acquisition \
  > /secure/p23-running-container-inspect.json
P23_CONTAINER_INIT_PID="$(
  docker inspect --format '{{.State.Pid}}' p23-acquisition
)"
if ! [[ "$P23_CONTAINER_INIT_PID" =~ ^[1-9][0-9]*$ ]]; then
  echo "invalid running-container State.Pid: $P23_CONTAINER_INIT_PID" >&2
  exit 1
fi
cat "/proc/$P23_CONTAINER_INIT_PID/mountinfo" \
  > /secure/p23-running-mountinfo.txt
```

The `docker inspect` command targets the actual running container, not the
image or an unrelated container.  The host obtains the same container's init
PID from Docker's `State.Pid` and retains that process's mount table directly
from host `/proc`; it does not obtain this evidence through `docker exec`.
The pre-mounted read-only file reflects the host-side byte update without
granting the container write access.

The inspected record must be running, use network mode `none`, use the default hostname (the first 12
characters of its full 64-hex ID), link by immutable image ID to the separately
inspected image, and contain exactly the ten frozen mounts. The retained host
`/proc/<State.Pid>/mountinfo` bytes must equal `/proc/self/mountinfo` in every
evidence process; this independently checks the live read-only/read-write view
rather than trusting Docker's host metadata alone. Freeze,
acquisition, verification, and aggregation execute in this same inspected
container; process isolation comes from separate direct Python invocations,
not from changing container IDs. Sanitization may run later as an offline
byte-bound transformation and is not a new CUDA/runtime attestation.

Run the remainder from one host-side Bash session.  The array below makes each
invocation an explicit `docker exec` of the image-resident interpreter: every
role is therefore a fresh Python process while retaining the one inspected
container ID.  The required variables were fixed in the `docker run` command.
Because there is no intervening shell that repairs the environment,
`PYTHONPATH`, `PYTHONHOME`, `PYTHONOPTIMIZE`, `LD_PRELOAD`, `LD_LIBRARY_PATH`,
and `LD_AUDIT` must also be absent from the selected image's `Config.Env`; the
pre-Torch bootstrap checks that absence and the complete frozen Python 3.12
interpreter-flag map on every invocation.

Replace both placeholders below with the attested digest and full GPU UUID:

```bash
set -euo pipefail
export P23_IMAGE=registry.example/project@sha256:...
export P23_GPU_UUID=GPU-00000000-0000-0000-0000-000000000000
export P23_HOST_REPO=/secure/p23/OptimizationML
export P23_HOST_NATIVE=/secure/p23/evidence

export P23_REPO=/workspace/OptimizationML
export P23_NANOGPT=/workspace/inputs/nanoGPT
export P23_MUON_ROOT=/workspace/inputs/muon
export P23_MUON="$P23_MUON_ROOT/muon.py"
export P23_DATA_ROOT=/private/tmp/optimizationml-p22-data
export P23_FINEWEB="$P23_DATA_ROOT/materialized/p22_fineweb_manifest.json"
export P23_PREPROCESSOR_ALIAS=/Users/harry/Desktop/temp/OptimizationML
export P23_NATIVE=/workspace/evidence/p23
export P23_RUNNER="$P23_REPO/experiments/training/run_p23_deterministic_cuda_shadow_trace.py"
export P23_OBSERVER="$P23_REPO/experiments/training/p22_observed_muon.py"
export P23_LOCK="$P23_REPO/experiments/training/p23_cuda_runtime_lock.json"
export P23_ATTESTATION="$P23_REPO/experiments/training/p23_host_attestation.json"
export P23_PYTHON_ENVIRONMENT=/opt/p23-venv
P23_PYTHON=(docker exec p23-acquisition /opt/p23-venv/bin/python)

test -d "$P23_HOST_REPO/.git"
test -d "$P23_HOST_NATIVE"
docker exec p23-acquisition test ! -e "$P23_REPO/.venv"

"${P23_PYTHON[@]}" "$P23_RUNNER" \
  freeze-runtime \
  --container-image "$P23_IMAGE" \
  --container-repository-digest "${P23_IMAGE##*@}" \
  --host-image-inspection /mounted-host-evidence/image-inspect.json \
  --host-running-container-inspection /mounted-host-evidence/running-container-inspect.json \
  --host-running-mountinfo /mounted-host-evidence/running-mountinfo.txt \
  --host-nvidia-smi-query /mounted-host-evidence/nvidia-smi.csv \
  --host-attestation-output /workspace/evidence/p23/p23_host_attestation.json \
  --runtime-lock-output /workspace/evidence/p23/p23_cuda_runtime_lock.json
```

Copy both generated files from the host evidence directory into
`experiments/training/`, review them, and commit them before running trace-off
A. The read-only repository bind reflects that host-side commit without
changing the inspected container or its mount contract. The generator refuses
existing outputs and both checked-in `.template.json` targets. For example,
after review, the host-side copy and commit are:

```bash
cp "$P23_HOST_NATIVE/p23_host_attestation.json" \
  "$P23_HOST_REPO/experiments/training/p23_host_attestation.json"
cp "$P23_HOST_NATIVE/p23_cuda_runtime_lock.json" \
  "$P23_HOST_REPO/experiments/training/p23_cuda_runtime_lock.json"
git -C "$P23_HOST_REPO" add \
  experiments/training/p23_host_attestation.json \
  experiments/training/p23_cuda_runtime_lock.json
git -C "$P23_HOST_REPO" commit -m "Freeze P23 CUDA runtime"
```

The acquisition commands below are the complete post-freeze runbook. Each
host-side `"${P23_PYTHON[@]}"` expansion starts the image-resident interpreter
as a new Python process in the same inspected container. That environment must be built into the image
before its repository digest is frozen.  The repository mount must contain no
`.venv`, and neither `uv`, `pip`, nor any dependency-sync or installation
command may run during `freeze-runtime`, acquisition, verification, or
aggregation. Sanitization is a later offline byte transformation, not another
runtime measurement. The OCI digest identifies the image layers, the
actual-running-container record binds the immutable image ID and mount modes,
and the runtime lock separately binds the resolved interpreter path and
executable SHA-256. Use the same absolute container paths and mounts
for every invocation; do not rewrite the frozen manifests between roles.  In
particular:

- mount the clean P23 repository, including its `.git` directory, at
  `P23_REPO`;
- mount the complete clean pinned nanoGPT checkout at `P23_NANOGPT`;
- mount the pinned Muon source under `P23_MUON_ROOT` and name the exact source
  file with `P23_MUON`;
- mount the complete frozen FineWeb bundle at the manifest's literal absolute
  root, `/private/tmp/optimizationml-p22-data`; its three captured row shards
  and two tokenizer files use absolute paths under that root, while
  `train.bin` and `val.bin` are relative to the `materialized/` manifest
  directory;
- expose the one preprocessor file at the other literal path recorded by the
  manifest,
  `/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py`.
  This is a read-only file bind from the clean primary repository, not a
  second repository checkout; `P23_REPO` remains `/workspace/OptimizationML`;
  and
- mount one external writable evidence directory at `P23_NATIVE`.  All native
  outputs stay outside the repository so its worktree remains clean, and that
  directory must have the identical absolute path in verifier processes
  because manifests and gate reports bind exact paths and bytes.

The frozen FineWeb manifest is not relocatable: its validator uses absolute
paths verbatim and resolves only its two relative output paths against the
manifest's parent.  A CUDA container therefore needs the following bind-mount
layout.  Replace the host-side `src` values with the immutable local assets,
but do not change any `dst` value:

```text
--mount type=bind,src=/secure/p23/OptimizationML,dst=/workspace/OptimizationML,readonly
--mount type=bind,src=/secure/p23/nanoGPT,dst=/workspace/inputs/nanoGPT,readonly
--mount type=bind,src=/secure/p23/Muon,dst=/workspace/inputs/muon,readonly
--mount type=bind,src=/secure/p23/optimizationml-p22-data,dst=/private/tmp/optimizationml-p22-data,readonly
--mount type=bind,src=/secure/p23/OptimizationML/experiments/training/materialize_p22_fineweb.py,dst=/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py,readonly
--mount type=bind,src=/secure/p23/evidence,dst=/workspace/evidence/p23
```

The digest-pinned image must contain the parent directory and file mount point
for the `/Users/harry/Desktop/temp/OptimizationML/...` alias before it is
frozen.  The alias exists solely because the immutable P22 manifest hashes
that preprocessor at that absolute path; it is neither the primary repository
nor an import root. Keep the primary repository read-only inside the inspected
container even during `freeze-runtime`; write generated artifacts to the
external evidence mount, then copy, review, and commit them from the host.
Every acquisition and verifier process uses that committed clean repository
through the same read-only mount. The complete data-tree mount must place the manifest
at exactly
`/private/tmp/optimizationml-p22-data/materialized/p22_fineweb_manifest.json`.

The UUID and digest above are placeholders and must be replaced by the same
full, non-MIG GPU UUID and immutable image reference frozen into both
artifacts. After reviewing and committing the generated lock and attestation,
assemble the shared immutable arguments in the same host Bash session:

```bash
P23_COMMON=(
  --nanogpt-root "$P23_NANOGPT"
  --muon-source "$P23_MUON"
  --fineweb-manifest "$P23_FINEWEB"
  --instrumentation-patch "$P23_OBSERVER"
  --runtime-lock "$P23_LOCK"
  --host-attestation "$P23_ATTESTATION"
)
```

Run trace-off A and trace-off B in two fresh processes:

```bash
"${P23_PYTHON[@]}" "$P23_RUNNER" run \
  --role trace_off_a "${P23_COMMON[@]}" \
  --output "$P23_NATIVE/trace-off-a.json"

"${P23_PYTHON[@]}" "$P23_RUNNER" run \
  --role trace_off_b "${P23_COMMON[@]}" \
  --output "$P23_NATIVE/trace-off-b.json"
```

Verify exact baseline repeatability before creating any trace-on process:

```bash
"${P23_PYTHON[@]}" "$P23_RUNNER" verify-repeatability \
  "${P23_COMMON[@]}" \
  --trace-off-a "$P23_NATIVE/trace-off-a.json" \
  --trace-off-b "$P23_NATIVE/trace-off-b.json" \
  --output "$P23_NATIVE/repeatability.json"
```

Only a zero-mismatch report permits the trace-on command.  Its prerequisite
arguments are revalidated before model allocation and again before step zero:

```bash
"${P23_PYTHON[@]}" "$P23_RUNNER" run \
  --role trace_on "${P23_COMMON[@]}" \
  --trace-off-a "$P23_NATIVE/trace-off-a.json" \
  --trace-off-b "$P23_NATIVE/trace-off-b.json" \
  --repeatability-report "$P23_NATIVE/repeatability.json" \
  --raw-trace-output "$P23_NATIVE/raw-trace.json" \
  --output "$P23_NATIVE/trace-on.json"
```

Then verify observer noninterference and aggregate the unchanged frozen gates:

```bash
"${P23_PYTHON[@]}" "$P23_RUNNER" verify-noninterference \
  "${P23_COMMON[@]}" \
  --trace-off-a "$P23_NATIVE/trace-off-a.json" \
  --trace-off-b "$P23_NATIVE/trace-off-b.json" \
  --trace-on "$P23_NATIVE/trace-on.json" \
  --output "$P23_NATIVE/noninterference.json"

"${P23_PYTHON[@]}" "$P23_RUNNER" aggregate \
  "${P23_COMMON[@]}" \
  --trace-off-a "$P23_NATIVE/trace-off-a.json" \
  --trace-off-b "$P23_NATIVE/trace-off-b.json" \
  --trace-on "$P23_NATIVE/trace-on.json" \
  --repeatability-report "$P23_NATIVE/repeatability.json" \
  --noninterference-report "$P23_NATIVE/noninterference.json" \
  --output "$P23_NATIVE/aggregate.json"
```

Finally retain every native artifact and create a path-sanitized complete copy
of each JSON.  This loop names every required artifact explicitly rather than
using a filesystem glob:

```bash
for P23_JSON in \
  trace-off-a.json trace-off-b.json repeatability.json trace-on.json \
  raw-trace.json noninterference.json aggregate.json
do
  "${P23_PYTHON[@]}" "$P23_RUNNER" sanitize \
    --manifest "$P23_NATIVE/$P23_JSON" \
    --path-root "repository=$P23_REPO" \
    --path-root "nanogpt=$P23_NANOGPT" \
    --path-root "muon=$P23_MUON_ROOT" \
    --path-root "data=$P23_DATA_ROOT" \
    --path-root "preprocessor_alias=$P23_PREPROCESSOR_ALIAS" \
    --path-root "python_environment=$P23_PYTHON_ENVIRONMENT" \
    --path-root "native=$P23_NATIVE" \
    --output "$P23_NATIVE/sanitized-$P23_JSON"
done
```

Any nonzero command stops the sequence.  A failed repeatability gate routes to
P24 localization; it never authorizes trace-on.  These commands create no P23
result in the current checkout because its runtime lock is still the invalid
null template and it has no CUDA device.

The CUDA runner must construct and move the model before creating optimizer
groups, then prove exact parameter-object, device, dtype, alias, name, shape,
and element-count coverage. P23 also replaces P22's ambiguous trace-off
candidate flag with an explicit `not_observed` state and zero capture count.
Only a trace-on run may say `observed`, and it must retain exactly
`48 x 24 = 1152` actual post-aspect BF16 CUDA candidates.

The acquisition order is fixed: trace-off A, trace-off B, exact repeatability
verification, trace-on, exact noninterference verification, then aggregation.
The trace-on entry point must consume a hash-bound passing repeatability
report; a human promise to run commands in order is insufficient. Both
comparators independently rebuild run identity and compare the complete
repository, initialized/completed loaded-file closure, recorded lazy-load
additions, and stable runtime maps in addition to the inherited
state/loss/data/RNG schedule. Every file-backed loaded module and every regular
file-backed `/proc/self/maps` pathname is hash-bound; module records include
their declared file, spec origin, existing cached bytecode, and recoverable
source file. No tolerance or
`allclose` is permitted.

The current machine has no CUDA device and the runtime lock remains a null
template. Therefore repository CI can replay only schema, provenance,
fail-closed comparison, sanitization, and CPU fixture checks. Such a replay is
not a CUDA acquisition, candidate-fidelity observation, or training result.
An actual P23 result requires access to one preferably exclusive non-MIG,
BF16-capable NVIDIA GPU and permission to run the digest-pinned OCI image.

The P9 separate CPU diagnostic is a native-matmul
falsification probe against a float64, non-exact target:

```bash
uv run --locked python \
  experiments/mixed_precision/run_scalable_mixed_precision_diagnostic.py \
  --output results/summaries/scalable_mixed_precision_diagnostic.json
```

That default diagnostic deliberately uses modest shapes. Pass
`--include-realistic-shapes` only for the explicitly requested, substantially
more expensive Transformer-shape run. Neither diagnostic mode proves the
theorem or establishes GPU/BLAS parity.

Replay the P10 CPU outer-shell diagnostic separately with:

```bash
uv run --locked python \
  experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py \
  --output results/summaries/finite_precision_outer_loop_diagnostic.json
```

The deterministic run covers modest `1 x 1`, `2 x 3`, and `8 x 16` shapes
and checks exact finite residual and `TwoSum` identities. It also replays the
actual-P9 ordinary-subtraction stalling witness. These checks can falsify the
operation graph; only the exact P10 generator and independent reconstruction
carry the full-shape storage claim.

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
