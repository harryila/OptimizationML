# P29 permission-safe bundle localization bridge outcome

## Verdict

P29 attempt `20260906-02` is **terminal after source-receipt creation and
before reviewed-orchestrator sealing**. The single source verifier invocation
passed, authenticated the complete seven-ref bundle, constructed the clean
control checkout, completed the `15/15`, `12/12`, `10/10`, and `23/23`
reconstruction barriers, and created the external source receipt.

The subsequent root seal then failed closed with exact operator-captured
stderr:

```text
P29 reviewed orchestrator source authority differs
```

The exact one-LF representation is 51 bytes with SHA-256
`6862765b73c5e2df1bc16b28d9def47b1824f9295835f8d605a017edd879f92b`.
No native seal stdout or stderr file was retained, so this text, its exit status,
and the post-failure inventory are explicitly operator-recorded rather than
presented as independently authenticated stream artifacts. No invocation time
is invented.

## What passed before the stop

The repository-authenticated P29 source freeze is commit
`ab62f683bf41ebb593a46267501e6aa6ba8e04c6`, tree
`3a6f210b5ed4767cbfe0ac354e2f58031242064f`, directly on terminal P28 commit
`7274367f2b05fb3c9ed9f876a59be647703bbdc2`.

The transported source bundle is 3,503,837 bytes with SHA-256
`5546ab07222ca5e9b1fd1959a23fd7830f68333ba073386a302343b3fe5817a9`.
It has no prerequisites and advertises exactly the frozen P29 branch, P28 and
P27 branches and diagnostic annotated tags, and both required P26 annotated
tags. The mode-`0600` bootstrap verifier is 138,874 bytes with SHA-256
`4e1c8d23a0b770282b90ad200d5323d3d281c57bc302a6a21cd84fef636f17b9`.

The no-overwrite native source receipt remains external at
`/secure/p29/transport-20260906-02/p29_source_bundle_receipt.json`. It is 11,678
bytes, `ubuntu:ubuntu` mode `0600`, with SHA-256
`7e4b80c03ccc59049b78a1a553c087ec670ce5c4cbc5a3db6b1bda8db2b45065`.
The receipt bytes are intentionally not committed, so this outcome binds their
reviewed path, stat, and digest without claiming local reconstruction of the
external file.

## Exact cause and retained boundary

The verifier intentionally installed umask `077` before constructing the
control checkout. The orchestrator is a non-executable `100644` Git blob, so
its authenticated checkout copy was physically `ubuntu:ubuntu` mode `0600`.
The frozen root seal required the source to be `ubuntu:ubuntu` mode `0644`.
Its expected SHA-256
`1653a12c2a86d307ddbff11cc5e48701e4817ee99e4674eaf6b426a8b2302cc5`
and 206,145-byte size matched, but the mode predicate did not. The observed
mode was more restrictive than the required one; no authority was broadened.
The seal stopped before opening its root-owned destination.

The retained root ledger still contains exactly its empty `deferred` journal.
The sealed launcher, authority checkout, prepare token, localization token,
attempt root, immutable execution snapshot, container, runtime lock, host
attestation, and localization artifacts do not exist.
No GPU or CUDA operation ran. No model, optimizer, forward/backward operation, candidate observation,
optimizer update, or training operation ran. P18--P21 and all earlier terminal
results are unaffected.

## Terminal policy and route

This is the frozen P29 `layout_bundle_or_reconstruction_failure` route. Do not
chmod, repair, clean, resume, or reuse attempt `20260906-02`, its namespace,
transport, control checkout, closure, receipt, or ledger.

The next branch is `p30-umask-bound-control-seal` with fresh attempt
`20260906-03`, a fresh namespace, bundle, closure, control checkout, and source
receipt. Its contract should align the seal with the verifier-created physical
mode before execution. Accepting a stably authenticated uid/gid-1000 mode-`0600`
source is the minimal correction because it is stricter than `0644`; an
end-to-end preflight must start from an actual verifier-created checkout. This
outcome does not authorize scientific acquisition.

## Machine-readable record

The canonical
[`p29_permission_safe_bundle_localization_bridge_outcome.json`](p29_permission_safe_bundle_localization_bridge_outcome.json)
has SHA-256
`6d29601a71ce6e0c99bcc35497d328c8fdd2a3cfc135351c19876507b00a0f64`.
Reconstruct it with:

```bash
python3 scripts/reconstruct_p29_permission_safe_bundle_localization_bridge_outcome.py
```
