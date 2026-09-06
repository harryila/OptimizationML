# P30 UID-1000 to root handoff engineering rehearsal

This is a logged engineering rehearsal, not a frozen scientific acquisition
and not another research phase. It exists to exercise the real Linux
checkout/source-receipt/root-seal boundary before a confirmatory P30 attempt is
frozen or consumed.

The workflow
`.github/workflows/p30-uid1000-root-engineering-rehearsal.yml` runs on a
disposable Ubuntu runner. A push to the temporary
`p30-engineering-rehearsal` branch tests `github.sha`; a manual dispatch may
instead provide an explicit reviewed P30 source commit. In either case that
commit must satisfy the source verifier's real history rule.

The rehearsal performs the following operations in order:

1. build a closed bundle advertising the exact nine P30/P29/P28/P27/P26 refs;
2. provision the real `/secure/p30` and root-only ledger authorities;
3. stage the bundle and bootstrap verifier as numeric UID/GID `1000:1000`;
4. execute the unmodified bootstrap verifier under `/usr/bin/python3 -I -S`
   as UID/GID 1000;
5. run all six real reconstruction paths and create the source receipt;
6. independently review and then replay that receipt;
7. extract the exact marked seal program from the receipt-bound source commit,
   never from the UID-1000-owned checkout, and execute it as UID/GID `0:0`; and
8. verify the resulting `root:root` mode-`0555` file against the
   receipt-bound mode-`0600` source;
9. create a synthetic direct-child runtime-review commit whose only additions
   are `p30_cuda_runtime_lock.json` and `p30_host_attestation.json`, both
   tracked as mode `100644`;
10. check that commit out as UID/GID 1000 with an explicit `umask 077`, proving
    both runtime-control files materialize as distinct regular files physically UID/GID `1000:1000` mode `0600`; and
11. hand the materialized checkout to root for a no-follow inode, byte, owner,
    mode, link-count, ACL, commit-parent, and Git-tree-mode verification, then
    pass all 37 localization variable names through a clean root `env -i`
    boundary and reject any missing, duplicate, or ambient `P30_*` name.

The two runtime-control files contain synthetic, credential-free placeholders
that identify themselves as engineering-rehearsal artifacts. They never enter
the source verifier's authoritative checkout, are never pushed as a runtime
review, and make no claim about an A100, CUDA state, or the real P30 attempt.
Values unavailable without a real runtime are conspicuously synthetic; this
phase tests the complete handoff name inventory, not their scientific values.
The actual verifier runtime-review materialization path, including both
physical modes and forbidden-ACL checks, is separately exercised by
`test_runtime_create_and_reviewed_receipt_replay`.

There is no monkeypatch, reconstruction stub, GPU command, CUDA
initialization, localization dispatch, candidate observation, optimizer
update, or training step.

The disposable filesystem deliberately mirrors P30's frozen `/secure/p30`
and `/var/lib/optimizationml-p30-20260906-03` path strings because those paths
are part of the code under test. It never creates the frozen attempt root and
cannot consume the distinct real-host attempt identity. The runner is
destroyed after the rehearsal.

## Logs and retention

Each numbered rehearsal phase, including initial provisioning, receipt
creation/review, and the root seal, receives a separate stdout, stderr, and
exit-status file. These native client streams live only under `${RUNNER_TEMP}`
and are uploaded with `if: always()` as a 90-day engineering artifact. They
never enter `/secure/p30`, the root ledger, `experiments/training` result
directories, or any directory whose exact scientific inventory is frozen.

The uploaded `rehearsal-manifest.json` hashes each retained stream and states
the evidence boundary explicitly: no GPU, no CUDA, no candidate observations,
and no training. It also records the synthetic runtime-review commit/tree and
the tested `100644`-tracked to `1000:1000 0600`-physical handoff. A later
durable record may cite the workflow run and these hashes, but must not relabel
the rehearsal as scientific acquisition evidence.

Historical failed rehearsals are retained rather than overwritten. A fixed
rehearsal may be rerun on a fresh disposable runner without allocating a new
P-number. Only a separately reviewed, frozen confirmatory acquisition may
support the P30 mapping result.
