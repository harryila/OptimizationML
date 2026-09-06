# P22 real-gradient shadow-trace results

## Verdict

P22 is **blocked at the baseline-repeatability gate**. Two complete 256-step
trace-off runs began from the same exact state and had identical source,
runtime, data, and recorded RNG identities, but their stored training states
diverged immediately after optimizer step 0. The predeclared protocol
therefore forbade the trace-on run. There is no P22 shield-activation,
candidate-fidelity, or training-quality result.

This is a useful diagnostic for the pinned Apple-MPS/PyTorch `2.13.0`
execution, not real-gradient fidelity evidence. It shows that this frozen MPS
execution did not provide the exact repeatable baseline required to
distinguish observer effects from execution variability. It does not identify
the responsible MPS operation, establish that all MPS executions are
nondeterministic, or weaken any P18--P21 theorem.

## Frozen computation

Both trace-off runs used the same isolated GPT-2-small trainer and the exact
frozen schedule: one seed, 256 optimizer steps, batch size one, training
sequence length 128, no gradient accumulation, and 32,768 sequential FineWeb
training tokens. Model parameters, forward, backward, gradients, EMA, and
Nesterov state were FP32 without autocast. The candidate path cast the
Nesterov signal to BF16 before its orientation, norm, additive-epsilon
normalization, division, and five Newton--Schulz stages on the selected
Apple-MPS accelerator. Weight decay was zero and Muon used constant
`eta=1/120`, `beta=19/20`, and Nesterov momentum.

For every hidden matrix, the pinned Keller--Jordan/Muon candidate used
additive normalization

`X/(||X||_F+1e-7)`,

five Jordan stages with coefficients `6889/2000`, `-191/40`, and `4063/2000`,
and the upstream aspect multiplier after restoring orientation. The source was
KellerJordan/Muon revision
`f98f1cacc0263b04290753e32be8d498c1efc806`; the isolated model source was
karpathy/nanoGPT revision
`3adf61e154c3fe3fca428ad6bc3818b27a3b8291`. P20 was not evaluated during
trace-off, and no shielded value was applied to training.

The frozen machine-readable protocol's statement that "only" the five stages
use BF16 is too narrow. The executed pinned source follows the operation order
above. The protocol bytes and hash remain unchanged; the correction is
recorded in `../../theory/p22_real_gradient_shadow_trace_protocol_erratum.md`.

## Exact repeatability failure

The two raw trace-off manifest SHA-256 digests are:

- off-A: `325c0e03fdfc7c5cd249b9c10cbe0a3b0e8c94b4b953a3eea3bf17586b497e98`;
- off-B: `49e902dad2b7b5fdc8c2f97ed8498c79a743c1cbaf8eba6798fd80cc5e3346f8`.

The native exact repeatability report has SHA-256
`1b6522653780cec8092e0907be672bb21897ce847c01baa3922233ed3c597923`.

The exact no-tolerance verifier reported:

| check | outcome |
| --- | --- |
| initial combined state hash | equal |
| optimizer steps in each run | 256 |
| data-window offsets and batch hashes | equal at all 256 steps |
| recorded host/selected-accelerator RNG schedule | equal at all 256 steps |
| first post-update model-state mismatch | step 0 |
| first post-update optimizer-state mismatch | step 0 |
| first stored-loss mismatch | step 2 |
| loss-hash mismatches | 253/256; only steps 0, 1, and 22 match |
| scheduled model-state mismatches | 24/24 |
| scheduled optimizer-state mismatches | 24/24 |
| final state mismatches | model and optimizer, 2 total |
| total verifier mismatches | 303 |
| trace-on acquisition | not run, as required by the frozen gate |

The 303 mismatches are exactly 253 loss hashes, both model and optimizer hashes
at all 24 scheduled post-update checkpoints, and both final-state hashes.
State hashes were predeclared initially, after each of the 24 capture-schedule
updates, and finally; loss, batch, token-offset, and RNG hashes were recorded
at all 256 steps. Thus the earliest stored-state discrepancy is the first
post-update checkpoint, while the first loss discrepancy follows at step 2.
Exact source/runtime/data identities and data/RNG schedules exclude differences
in those recorded fields; attributing the divergence to a particular MPS
kernel would require a separate operation-level study.

The native manifests did not record the OptimizationML Git SHA or worktree
status directly. Every repository-owned source hash that they did record
retrospectively matches commit `79f33ec`. Their source snapshot also omitted
four imported project dependencies: the P22 preflight/verifier, package
initializer, P21 shadow observer, and P20 scalable shield. Consequently the
reconstruction neither proves the complete executed project graph nor replaces
the contemporaneous Git-state record required by repository rule 7. These
omissions do not change the exact pairwise mismatch verdict and remain a
disclosed provenance caveat.

The runtime record identifies macOS `15.5`, `arm64`, PyTorch `2.13.0`, and a
generic Apple-MPS device, but not the Apple chip model or the OS/Metal build.
The result must therefore remain scoped to the hash-locked execution pair,
not advertised as device-model or Metal-version evidence.

The native verifier also trusted the run-identity digest instead of separately
binding every source/runtime mapping. The committed evidence builder closes
that gap for this pair by requiring those mappings to be exactly equal and
storing their canonical hashes. This independently supports the negative
verdict; the native verifier should be hardened before a future acceptance
run.

## Claim boundary

These two trace-off runs contain real model gradients, but they contain no
shadow observations. In particular, they provide no evidence about:

- P20 activation frequency or correction magnitude;
- cosine, amplitude, effective-update, P16, or P18 fidelity gates;
- the actual accelerator post-aspect candidate at the 24 capture steps;
- observer noninterference;
- neural-loss PL assumptions, convergence, validation quality, or throughput;
- CUDA, tensor-core, distributed, or cross-platform parity; or
- stability of unshielded upstream Muon.

The exact P22 `768 x 2304` shield-shape extension remains a separate analytic
artifact. It certifies the locked CPU proof-reference containment recurrence;
it does not make the MPS baseline deterministic and was not exercised by these
trace-off runs.

## Next acquisition gate

Do not compare off-A with a trace-on MPS run and do not replace exact equality
with `allclose`. The next acquisition should preserve the same gates and
replay the three-run protocol on one BF16-capable CUDA device under the frozen
CUDA determinism guards. Before data collection, the native manifest/verifier
should record the repository SHA and dirty state, hash the complete imported
project graph, and bind the explicit source/runtime maps:

1. run fresh-process CUDA trace-off-A and trace-off-B;
2. require their exact repeatability report to pass;
3. only then run CUDA trace-on and evaluate off-A/trace-on noninterference;
4. interpret shadow fidelity metrics only after both gates pass.

If the CUDA baseline also fails exact repeatability, P22 remains blocked and
the mismatch must be localized or the intervention design revised before any
fidelity claim.

## Committed evidence

The path-sanitized canonical record is
`p22_repeatability_failure_evidence.json`. It retains every unequal checked
value, exact comparison-projection hashes, selected anchor records, the raw
external artifact byte counts and SHA-256 values, and records that the
trace-on, noninterference, and aggregate outputs were absent from the frozen
native run directory after the failed gate. The full raw off-A/off-B manifests
and native repeatability report remain external; their hashes above bind the
canonical record without publishing machine-local paths.

The supporting records are:

- `p22_real_gradient_preflight_evidence.json`, which records a ready frozen
  MPS preflight, complete 48-matrix shape coverage, and pinned source hashes;
- `p22_fineweb_materialization_evidence.json`, which records the revision,
  complete-row selection, tokenizer assets, token boundaries, and external
  binary hashes without committing dataset text or token binaries; and
- `p22_scalable_shield_shape_extension_certificate.json`, the separate exact
  CPU proof-reference extension for fused QKV.

The internal technical audit and unsigned external-review checklist are in
`../../theory/audits/P22_REAL_GRADIENT_SHADOW_TRACE_AUDIT.md`. Independent
human review is pending.

Given the locked external native files, rebuild the sanitized records with:

```bash
uv run --locked python scripts/build_p22_repeatability_failure_evidence.py \
  --native-root /path/to/p22-native-evidence \
  --output-root results/summaries
```
