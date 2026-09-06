# P26 permission-safe CUDA acquisition result

## Verdict

Attempt `20260906-02` is **terminal during `trace_off_a`, before step zero**.
P26 fixed the P25 host-permission defect: both independent contract
reconstructions passed, the root-container bridge authenticated the retained
native diagnostic and the two sanitized wrappers without changing their
permissions, and the first acquisition process was then invoked exactly once.

That process initialized the GPT-2-small model and optimizer on the locked
A100, printed `number of parameters: 123.55M`, and failed closed while taking
the initialized loaded-file snapshot.  The exact retained error is:

```text
deleted file-backed mapping is forbidden at /proc/self/maps line 17
```

The failure occurred before the first training step.  It is not an exact-CUDA
repeatability result, an observer-noninterference result, or real-gradient
candidate-fidelity evidence.  No second trace-off run, trace-on run, candidate
capture, aggregate, shielded update, or training result exists.

The frozen P23 failure artifact did not retain the raw pathname from line 17,
and failed process PID `137` no longer exists.  The mapping identity is
therefore irrecoverable from this attempt and is deliberately not guessed.

## Gates reached

| gate | result |
| --- | --- |
| fresh no-cache image and locked A100 runtime | passed and reviewed |
| corrected P25 diagnostic | passed `40/40` |
| P26 independent contract reconstruction | passed `19/19` |
| frozen P25 contract reconstruction | passed `23/23` |
| root-container permission-safe bridge | passed |
| trace-off A process | invoked once; failed during initialized provenance closure |
| optimizer step zero | not reached |
| trace-off B / exact repeatability | not run |
| trace-on / observer noninterference | not run |
| candidate observations | `0` |
| fidelity aggregation / shielded training | not run |

The permission-safe bridge performed no mutation.  It authenticated a
`root:root`, mode-`0600`, 4,115,671-byte native diagnostic with SHA-256
`b9cc8dad897e27a36414a320cb9aa2f446b292c45e9a170ddb9dae1ac8ea96a9`
and two exactly equal 5,099,937-byte sanitized wrappers with SHA-256
`ad33cb3e2e1e86f182496b8e3b73987bf4995290c3d90ed4342971e2a44507dd`.
The retained wrapper remained `root:root 0600`; the repository copy was
`0644`.  Native semantics and all 15 frozen P25 sources reconstructed.

## Locked runtime and source chain

The final P26 control commit is
`ae16c1a4429a5363c7a6168d19ca2089ecddee35`, tree
`a2537867260f322d9db5e1f48013f72846b7aa18`; it is a direct child of the
pre-runtime source freeze `bc2c84880a7e6f3e762d7a05dfbd5048772b2c69`.
The separate acquisition-source chain was:

| stage | commit | tree | parent |
| --- | --- | --- | --- |
| P25 preregistration | `e76ab62f92c95e6f0716cf2f1ed38a583cadfe56` | `4fe0f57d50a8fa136bb192ec3fbac95b1e746aa2` | `5814621979ffedf370f04f3d99d925d8584b640f` |
| P26 runtime review | `ba93225f3ef1abf5dbde71950c1eaa2e384a5dc0` | `e7cdf25c86360ecdd42f2dc188ec416744321fe1` | `e76ab62f92c95e6f0716cf2f1ed38a583cadfe56` |
| diagnostic bridge | `185e444afc0b44ca0a09b1bde49a6b6fa3973355` | `24f4bdac331a57bd7c1b807747d7c7fba253ee5a` | `ba93225f3ef1abf5dbde71950c1eaa2e384a5dc0` |

The runtime used image digest
`sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0`,
container ID
`e6682d8f09b9dc4be342354d520a8f6f8766a8e232840d1d7beb5e00e1fced6c`,
container init PID `39227`, restart count `0`, and the pinned non-MIG GPU
`GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d`.  The root filesystem was
read-only, network mode was `none`, and all ten mounts were locked.

## Native trace-off-A failure

The retained native failure artifact has schema
`passive-muon-p23-native-failure-manifest-v1`, status
`blocked_before_success_artifact`, exit code `2`, and acquisition role
`trace_off_a`.  It is 66,283 bytes, `root:root 0600`, with SHA-256
`b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91`.
It records failure UTC `2026-09-06T12:35:38Z`, phase `p22_initialized`, and
the call chain

```text
run -> P22 lifecycle("initialized") -> _collect_loaded_files
    -> collect_loaded_file_snapshot -> _mapped_regular_files
```

Its prepared and failure-time repository snapshots are clean and identical at
execution HEAD `185e444afc0b44ca0a09b1bde49a6b6fa3973355`, tree
`24f4bdac331a57bd7c1b807747d7c7fba253ee5a`. The prepared-runtime and
prepared-static-contract canonical hashes are respectively
`c40abffcc32b2596350ae94e11020ccea49801c32d19d940dbc9b86aafa45e0f`
and `1c7be31c697e443dad70d0c05e9b8f070c85f1d5b3ed2a64b2399c56690d567d`.

The corresponding process streams are independently retained:

| stream | bytes | SHA-256 |
| --- | ---: | --- |
| `p23-trace-off-a.stdout.log` | 30 | `694e39d99abd6f53579f3b9d24f769e7c01d28c8b0176a33baa2ffc758713c60` |
| `p23-trace-off-a.stderr.log` | 231 | `954ee85a7a152e540ca382d2def2128d02cbcec94912183094ad4162f5ab65e0` |

Both are owned by `ubuntu:ubuntu` with mode `0664`.  The failure manifest is
external retained evidence, not committed unsanitized repository content.

## Independent sanitizer failure

After preserving the native failure, the frozen P23 offline sanitizer also
failed closed.  Its exact stderr is:

```text
P23 blocked: undeclared absolute path cannot be sanitized: /opt/p23-venv/bin/python
```

The stderr is 84 bytes with SHA-256
`7818b2041d0fc8b9a0a12c150d68eda089bb0ceebd7fdc47a30e0a43fc0f798f`;
stdout is empty.  Consequently this attempt has no sanitized failure wrapper.

This is a separate, localized provenance bug.  P23 explicitly declares
`/opt/p23-venv/bin/python`, but its sanitizer physically resolves that venv
symlink before checking containment.  In the locked image the symlink resolves
to `/usr/local/bin/python3.12`, outside the declared
`python_environment=/opt/p23-venv` root.  The existing unit fixture did not
construct the live symlink and therefore missed this case.  A valid correction
must special-case only the exact already-validated pinned executable; it must
not weaken rejection of arbitrary symlink escapes or undeclared paths.

## Scope and next route

Attempt `20260906-02` must never be resumed or rerun.  The exact identity of
the deleted mapping must be established in a fresh, preregistered localization
process that records structured process-map fields at initialization stages
without running forward/backward or an optimizer step.  Executable or
unclassifiable deleted mappings must remain fatal; no blanket `(deleted)`
exception is justified.

The next branch is `p27-cuda-deleted-mapping-localization`.  It must also
repair failure-manifest sanitization narrowly and ingest the retained P26
native artifact through a hash-bound, one-read root bridge.  Only after the
mapping is exactly classified and independently reconstructed may a new
attempt `20260906-03` run the unchanged P23 scientific sequence.

P18--P21's analytic and finite-precision results are unaffected.  P26 proves
that the permission-safe bridge works and that the acquisition remains
fail-closed; it proves no CUDA repeatability, candidate fidelity, native shield
safety, or neural-training claim.

## Machine-readable record

The compact
[`p26_permission_safe_cuda_acquisition_outcome.json`](p26_permission_safe_cuda_acquisition_outcome.json)
binds the repository-authenticated source/runtime chain and explicitly labels
which terminal facts remain external retained evidence. Its canonical SHA-256
is
`9be5727213f1a23f00b05524940d88c2d02461510fb61316b29dacc6ee8ead47`.
Reconstruct it with:

```bash
python scripts/reconstruct_p26_permission_safe_cuda_acquisition_outcome.py
```
