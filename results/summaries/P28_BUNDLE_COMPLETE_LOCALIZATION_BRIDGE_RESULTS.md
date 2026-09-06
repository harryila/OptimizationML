# P28 bundle-complete localization bridge outcome

## Verdict

P28 attempt `20260906-01` is **terminal before control-checkout and attempt-root
creation**. The single source-verifier invocation authenticated the staged
verifier and complete five-ref bundle, built and verified the fresh bare source
closure, and then failed closed while trying to create
`/secure/p28/control-source`:

```text
P28 control-bundle verification failed: Git command failed (128): init --initial-branch=p28-verifier-unborn /secure/p28/control-source: fatal: cannot mkdir /secure/p28/control-source: Permission denied
```

The exact one-LF stderr representation is 202 bytes with SHA-256
`3d85310ab120816ed22e1d78295b2384ff2e0e2147d5700dbb93abe1bb28b355`.
No native stdout or stderr file was retained, so this text and the external
permission inventory are explicitly operator-recorded facts rather than a
claim of independently authenticated native evidence. No invocation UTC is
invented.

## What passed before the stop

The repository-authenticated source freeze is commit
`003b6ee3f0ee2d1c269cac787c0281bf3f976ac4`, tree
`11e471539a8f422d74728548a0ee8ac19ffea1c0`, directly on terminal P27 commit
`ec63550331925ded158e3f389e294e4d1f12db3a`.

The transported source bundle is 3,362,894 bytes with SHA-256
`54461db0d3a2e07870d9f43cf9877da0cb4f28c13ad1c5f1cc32b44ff3e7945b`.
It has no prerequisites and advertises exactly the five frozen refs:

- the P28 source branch at `003b6ee3f0ee2d1c269cac787c0281bf3f976ac4`;
- the P27 branch at `ec63550331925ded158e3f389e294e4d1f12db3a`;
- the P27 diagnostic annotated tag object `c88b98eae9be2458abde45b05d3dccfe09c0c7ed`;
- the P26 checkpoint annotated tag object `bacad707d3779bfa10957e18cb4c69b1a7f0cbce`;
- the P26 acquisition-source annotated tag object `f35a7dca8f6bc39e9748e79b6712fab4a203396b`.

The verifier authenticated a read-only private copy, explicitly fetched every
ref into a fresh empty bare repository, checked tag types and peels, ran strict
full `git fsck`, and revalidated the private copy afterward. The retained source
closure contains all five refs. This corrects P27's transfer omission, but it is
not a completed source-receipt gate because the control checkout and receipt do
not exist.

## Exact cause and retained boundary

The runbook provisioned the mode-`0700`, `ubuntu:ubuntu` transport directory
beneath `/secure/p28`, but did not provision a writable parent for the sibling
control checkout. The parent remained mode `0755`, `root:root`; an unprivileged
`ubuntu` verifier therefore could not create `/secure/p28/control-source`.
This is a P28 runbook/host-layout preflight defect. The verifier itself behaved
correctly by failing closed.

Retained external state consists of the source bundle, mode-`0600` bootstrap
verifier, and mode-`0775` source closure below the terminal transport directory.
The following do not exist:

- control checkout and source receipt;
- P28 attempt root and container;
- runtime lock or host attestation;
- native or sanitized localization output.

Consequently, image inspection, GPU access, CUDA initialization, P27
reconstruction, mapping localization, model/optimizer construction,
forward/backward work, candidate observation, optimizer updates, and training
were not run. P18--P21 and all prior terminal results are unaffected.

## Terminal policy and route

Do not reuse, clean up, resume, or rerun P28 attempt `20260906-01`, its transport,
or its closure. The next step is a separately frozen branch, transport, and
attempt identity that changes only the reviewed control-parent placement or
ownership. A safe layout should put the absent control child beneath a
preregistered process-owned mode-`0700` parent. It must still produce and review
a fresh external source receipt before creating any attempt root.

This outcome does not authorize scientific acquisition.

## Machine-readable record

The canonical
[`p28_bundle_complete_localization_bridge_outcome.json`](p28_bundle_complete_localization_bridge_outcome.json)
has SHA-256
`298324d163aa12c1cb64bfdb912ae846152d0980ae3c981a54ead75a019305c9`.
Reconstruct it with:

```bash
python3 scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py
```
