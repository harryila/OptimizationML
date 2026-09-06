# P28 bundle-complete localization bridge runbook

## Evidence boundary

P28 corrects one operator-transfer defect and then runs the **unchanged P27
no-training localizer** once. P27 attempt `20260906-01` is terminal: its focused
bundle omitted the annotated P26 checkpoint tag, so reconstruction stopped
before image inspection, GPU access, container creation, or CUDA. P28 does not
reinterpret that stop and never reuses or cleans its attempt state.

This runbook does not authorize a P23 scientific acquisition. It permits only
the four P27 diagnostic stages: `pre_cuda`, `post_cuda_init`,
`post_model_move`, and `post_optimizer`. Data loading, forward, backward,
candidate evaluation, optimizer steps, parameter updates, and training remain
forbidden. The prospective acquisition identifier `20260906-03` remains
unauthorized.

Read and independently reconstruct these packets first:

```bash
python3 scripts/reconstruct_p27_cuda_deleted_mapping_localization.py
python3 scripts/reconstruct_p27_cuda_deleted_mapping_localization_outcome.py
python3 scripts/reconstruct_p28_bundle_complete_localization_bridge.py
```

All three commands must succeed. In particular, the P27 contract
reconstruction must report exactly 23 checks, 23 true, and zero false.

## 1. Freeze the P28 source package

Before creating a bundle, transport directory, control checkout, attempt root,
or container, run:

```bash
uv run --locked pytest -q \
  tests/test_p28_bundle_complete_localization_bridge_contract.py \
  tests/test_p28_control_bundle_verifier.py \
  tests/test_p28_bundle_complete_localization_bridge_orchestration.py \
  tests/test_p27_cuda_deleted_mapping_localization_contract.py \
  tests/test_p27_cuda_deleted_mapping_localization.py \
  tests/test_p27_cuda_deleted_mapping_localization_orchestration.py \
  tests/test_p27_cuda_deleted_mapping_localization_sanitizer.py \
  tests/test_p27_p26_failure_ingestion.py \
  tests/test_p27_cuda_deleted_mapping_localization_outcome.py
bash -n scripts/run_p28_bundle_complete_localization_bridge.sh
uv run --locked ruff format --check experiments/training scripts tests
uv run --locked ruff check experiments/training scripts tests
git diff --check
```

Commit and review the complete P28 package. Record its commit and tree as
`P28_SOURCE_FREEZE_COMMIT` and `P28_SOURCE_FREEZE_TREE`. The contract does not
embed those values because that would be self-referential.

The following refs must all exist locally with these meanings before bundle
creation:

| Ref | Required object | Required peel |
| --- | --- | --- |
| `refs/heads/p28-bundle-complete-localization-bridge` | source-freeze commit selected after review | same commit |
| `refs/heads/p27-cuda-deleted-mapping-localization` | `ec63550331925ded158e3f389e294e4d1f12db3a` | same commit |
| `refs/tags/p27-cuda-deleted-mapping-localization-diagnostic` | annotated tag `c88b98eae9be2458abde45b05d3dccfe09c0c7ed` | `ec63550331925ded158e3f389e294e4d1f12db3a` |
| `refs/tags/p26-permission-safe-acquisition-checkpoint` | annotated tag `bacad707d3779bfa10957e18cb4c69b1a7f0cbce` | `5429da23ff18888daa2312c530a4587780484d8b` |
| `refs/tags/p26-attempt02-acquisition-source` | annotated tag `f35a7dca8f6bc39e9748e79b6712fab4a203396b` | `185e444afc0b44ca0a09b1bde49a6b6fa3973355` |

Do not substitute a peeled commit for an annotated tag object.

Create the source bundle locally in a fresh private directory, and inspect it
before transport:

```bash
export P28_LOCAL_TRANSPORT="$(mktemp -d /private/tmp/p28-transfer.XXXXXX)"
export P28_LOCAL_SOURCE_BUNDLE="$P28_LOCAL_TRANSPORT/p28_source.bundle"
git bundle create "$P28_LOCAL_SOURCE_BUNDLE" \
  refs/heads/p28-bundle-complete-localization-bridge \
  refs/heads/p27-cuda-deleted-mapping-localization \
  refs/tags/p27-cuda-deleted-mapping-localization-diagnostic \
  refs/tags/p26-permission-safe-acquisition-checkpoint \
  refs/tags/p26-attempt02-acquisition-source
git bundle list-heads "$P28_LOCAL_SOURCE_BUNDLE"
git bundle verify "$P28_LOCAL_SOURCE_BUNDLE"
shasum -a 256 "$P28_LOCAL_SOURCE_BUNDLE"
wc -c "$P28_LOCAL_SOURCE_BUNDLE"
```

`list-heads` must contain exactly the five rows in the contract, with the
three tag rows naming the annotated tag objects rather than their peels.

## 2. Build and verify the full source bundle before creating the attempt root

The fixed remote identities are:

```bash
export P28_ATTEMPT_ID=20260906-01
export P28_ATTEMPT_ROOT=/secure/p28/attempt-20260906-01
export P28_CONTAINER=p28-localization-20260906-01
export P28_TRANSPORT_ROOT=/secure/p28/transport-20260906-01
export P28_CONTROL_REPO=/secure/p28/control-source
export P28_SOURCE_BUNDLE=/secure/p28/transport-20260906-01/p28_source.bundle
export P28_SOURCE_CLOSURE=/secure/p28/transport-20260906-01/p28_source_closure.git
export P28_SOURCE_RECEIPT=/secure/p28/transport-20260906-01/p28_source_bundle_receipt.json
export P28_BOOTSTRAP_VERIFIER=/secure/p28/transport-20260906-01/verify_p28_control_bundle.py
export P28_SSH_TARGET=ubuntu@129.146.177.214
```

Create one credential-free full bundle from the reviewed source freeze by
naming exactly the five refs above. Do not use exclusions, revision ranges, or
a shallow source. Record its SHA-256 and byte count before transfer. Transfer
it to the exact `P28_SOURCE_BUNDLE` path with no overwrite.

Create only the transport directory, then use this `O_EXCL|O_NOFOLLOW`
uploader for both the source bundle and bootstrap verifier. It never overwrites
a retained byte:

```bash
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P28_SSH_TARGET" \
  sudo /usr/bin/install -d -m 0700 -o ubuntu -g ubuntu "$P28_TRANSPORT_ROOT"
p28_o_excl_upload() {
  local source_path="$1"
  local destination_path="$2"
  local destination_mode="$3"
  test -f "$source_path"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P28_SSH_TARGET" \
    "/usr/bin/python3 -c 'import os,shutil,sys; p=sys.argv[1]; m=int(sys.argv[2],8); f=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,\"O_CLOEXEC\",0)|getattr(os,\"O_NOFOLLOW\",0); d=os.open(p,f,m); h=os.fdopen(d,\"wb\"); shutil.copyfileobj(sys.stdin.buffer,h); h.flush(); os.fsync(h.fileno()); h.close()' '$destination_path' '$destination_mode'" \
    <"$source_path"
}
p28_o_excl_upload "$P28_LOCAL_SOURCE_BUNDLE" "$P28_SOURCE_BUNDLE" 0600
p28_o_excl_upload \
  scripts/verify_p28_control_bundle.py "$P28_BOOTSTRAP_VERIFIER" 0600
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P28_SSH_TARGET" \
  sha256sum "$P28_SOURCE_BUNDLE" "$P28_BOOTSTRAP_VERIFIER"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P28_SSH_TARGET" \
  stat -c '%a %s %n' "$P28_SOURCE_BUNDLE" "$P28_BOOTSTRAP_VERIFIER"
```

If upload or verification fails, leave every partial or completed path as
evidence and stop. Do not delete, replace, or restage it under the same P28
identifier.

The upload above stages the separately reviewed verifier bytes exactly once at
`P28_BOOTSTRAP_VERIFIER`, without overwrite and at mode `0600`. The verifier
then proves that it is a regular nonsymlink file and that its SHA-256 equals
the final hash in the P28 contract. The mandatory `--bootstrap-verifier` and
`--expected-verifier-sha256` arguments make the verifier authenticate its
executing `__file__` by a stable `O_NOFOLLOW` read and record that
self-authentication in each receipt. The receipt binds the canonical
checked-in verifier path and SHA-256, not the transport file's owner or mode;
the staging guard enforces mode `0600` separately. After it constructs the
control checkout, it proves the checked-in verifier bytes are identical. Do
not execute an unbound copy from `/tmp` or a shell stream.

The sole verifier is `scripts/verify_p28_control_bundle.py`. Its source phase
must:

1. prove the P28 attempt root, control checkout, receipt, and closure repository
   are absent;
2. open the bundle once with `O_NOFOLLOW`, verify its operator-supplied SHA-256
   and byte count, and work from a private authenticated read-only copy whose
   hash and stable file identity are revalidated after all Git use;
3. initialize a fresh empty bare repository at `P28_SOURCE_CLOSURE`;
4. run Git bundle verification, reject every prerequisite, require exactly the
   five advertised refs, explicitly fetch each same-named ref, and run full
   object verification;
5. verify every commit, annotated tag object, and peel exactly;
6. construct the previously absent `P28_CONTROL_REPO` only from that verified
   closure, with no shallow state, alternates, grafts, or replace refs;
7. require a detached, clean checkout at the reviewed P28 source commit/tree;
8. run the frozen P27 independent reconstruction and require 23/23 checks; and
9. write `P28_SOURCE_RECEIPT` with `O_EXCL`.

Use the verifier's checked-in CLI exactly. The reviewed receipt is external
evidence and must never be committed into the history it authenticates. If
verification fails, stop: the attempt root must remain absent. Never rerun the
same transfer phase to seek a favorable result. A later attempt requires a
separately frozen identifier and contract regardless of why this phase failed.

The exact source receipt-creation form is:

```bash
/usr/bin/python3 "$P28_BOOTSTRAP_VERIFIER" \
  --phase source \
  --transport-root "$P28_TRANSPORT_ROOT" \
  --bundle "$P28_SOURCE_BUNDLE" \
  --expected-bundle-sha256 "$P28_EXPECTED_SOURCE_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P28_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" \
  --closure-repository "$P28_SOURCE_CLOSURE" \
  --control-checkout "$P28_CONTROL_REPO" \
  --attempt-root "$P28_ATTEMPT_ROOT" \
  --expected-p28-commit "$P28_SOURCE_FREEZE_COMMIT" \
  --expected-p28-tree "$P28_SOURCE_FREEZE_TREE" \
  --bootstrap-verifier "$P28_BOOTSTRAP_VERIFIER" \
  --expected-verifier-sha256 "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --receipt-output "$P28_SOURCE_RECEIPT"
```

Review the receipt and record its external SHA-256. Then, and only then,
construct the authority checkout from the verified source closure. It must be
absent first:

```bash
export P28_AUTHORITY_REPO=/secure/p28/authority-185e444
test ! -e "$P28_AUTHORITY_REPO"
git clone --no-checkout "$P28_SOURCE_CLOSURE" "$P28_AUTHORITY_REPO"
git -C "$P28_AUTHORITY_REPO" checkout --detach \
  185e444afc0b44ca0a09b1bde49a6b6fa3973355
git -C "$P28_AUTHORITY_REPO" remote remove origin
test "$(git -C "$P28_AUTHORITY_REPO" rev-parse HEAD)" = \
  185e444afc0b44ca0a09b1bde49a6b6fa3973355
test "$(git -C "$P28_AUTHORITY_REPO" rev-parse HEAD^{tree})" = \
  24f4bdac331a57bd7c1b807747d7c7fba253ee5a
test -z "$(git -C "$P28_AUTHORITY_REPO" status --porcelain=v1 --untracked-files=all)"
test ! -e "$P28_AUTHORITY_REPO/.git/shallow"
test ! -e "$P28_AUTHORITY_REPO/.git/objects/info/alternates"
test ! -e "$P28_AUTHORITY_REPO/.git/info/grafts"
test -z "$(git -C "$P28_AUTHORITY_REPO" for-each-ref --format='%(refname)' refs/replace)"
test -z "$(git -C "$P28_AUTHORITY_REPO" config --local --get extensions.partialClone)"
test -z "$(git -C "$P28_AUTHORITY_REPO" config --local --get-regexp '^remote\..*\.promisor$')"
```

Do not clone the authority from an ambient repository or the terminal P27
transport. This checkout is administrative source state, not an acquisition
attempt, and must still precede `prepare-runtime`.

## 3. Prepare the fresh runtime only after replaying the source receipt

Export the exact P28 source commit/tree, bundle SHA-256/byte count, receipt
SHA-256, and every expected source hash required by the host orchestrator.
Also pin:

```bash
export P28_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
export P28_NANOGPT_HOST=/secure/p23/nanoGPT
export P28_MUON_HOST=/secure/p23/Muon
export P28_DATA_HOST=/secure/p23/optimizationml-p22-data
```

The authority checkout mounted in the container remains the detached clean
P26 acquisition-source checkout at commit
`185e444afc0b44ca0a09b1bde49a6b6fa3973355`, tree
`24f4bdac331a57bd7c1b807747d7c7fba253ee5a`. It is not the P28 control
checkout.

Replay of the reviewed source receipt is a read-only pre-root check; it is not
the state-changing prepare attempt. After that replay passes, run exactly one
state-changing preparation:

```bash
bash "$P28_CONTROL_REPO/scripts/run_p28_bundle_complete_localization_bridge.sh" prepare-runtime
```

The orchestrator must replay the source receipt and re-prove its bundle,
closure, control commit/tree, five refs, and 23/23 P27 reconstruction **before
creating** `/secure/p28/attempt-20260906-01`. It then creates one fresh
`p28-localization-20260906-01` container from image digest
`sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0`
using network `none`, read-only rootfs, the locked `/tmp` tmpfs, the full
non-MIG A100 UUID, and exactly the ten P27 bind-mount roles. It freezes but does
not yet use:

```text
experiments/training/p28_cuda_runtime_lock.json
experiments/training/p28_host_attestation.json
```

Review both artifacts. No localizer process, model, optimizer, mapping
observation, gradient, or candidate may exist yet.

## 4. Freeze and independently transfer the runtime-review commit

Copy the two reviewed runtime artifacts into the local source-freeze checkout
at the exact paths above using this no-overwrite downloader. `pipefail` makes a
remote read failure fatal, while the local `O_EXCL|O_NOFOLLOW` open prevents
replacement:

```bash
set -o pipefail
p28_o_excl_download() {
  local retained_path="$1"
  local checkout_path="$2"
  test ! -e "$checkout_path"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P28_SSH_TARGET" \
    sudo /usr/bin/cat -- "$retained_path" | \
    /usr/bin/python3 -c \
    'import os,shutil,sys; p=sys.argv[1]; f=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0); d=os.open(p,f,0o600); h=os.fdopen(d,"wb"); shutil.copyfileobj(sys.stdin.buffer,h); h.flush(); os.fsync(h.fileno()); h.close()' \
    "$checkout_path"
}
p28_o_excl_download \
  /secure/p28/attempt-20260906-01/evidence/p28_cuda_runtime_lock.json \
  experiments/training/p28_cuda_runtime_lock.json
p28_o_excl_download \
  /secure/p28/attempt-20260906-01/evidence/p28_host_attestation.json \
  experiments/training/p28_host_attestation.json
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P28_SSH_TARGET" \
  sudo sha256sum \
  /secure/p28/attempt-20260906-01/evidence/p28_cuda_runtime_lock.json \
  /secure/p28/attempt-20260906-01/evidence/p28_host_attestation.json
shasum -a 256 \
  experiments/training/p28_cuda_runtime_lock.json \
  experiments/training/p28_host_attestation.json
```

The two local hashes must equal the corresponding retained hashes. Commit only
those files. The runtime-review commit
must be one direct child of `P28_SOURCE_FREEZE_COMMIT`, and its complete
name-status delta must be:

```text
A	experiments/training/p28_cuda_runtime_lock.json
A	experiments/training/p28_host_attestation.json
```

Record `P28_RUNTIME_REVIEW_COMMIT` and `P28_RUNTIME_REVIEW_TREE`. Create a
second complete bundle naming the same five refs, with only the P28 branch ref
advanced to the runtime-review commit. Transfer it without overwrite to:

```text
/secure/p28/transport-20260906-01/p28_runtime_review.bundle
```

Use the same exact ref list and uploader:

```bash
export P28_LOCAL_RUNTIME_BUNDLE="$P28_LOCAL_TRANSPORT/p28_runtime_review.bundle"
export P28_RUNTIME_REVIEW_BUNDLE=/secure/p28/transport-20260906-01/p28_runtime_review.bundle
export P28_RUNTIME_REVIEW_CLOSURE=/secure/p28/transport-20260906-01/p28_runtime_review_closure.git
export P28_RUNTIME_REVIEW_RECEIPT=/secure/p28/transport-20260906-01/p28_runtime_review_bundle_receipt.json
git bundle create "$P28_LOCAL_RUNTIME_BUNDLE" \
  refs/heads/p28-bundle-complete-localization-bridge \
  refs/heads/p27-cuda-deleted-mapping-localization \
  refs/tags/p27-cuda-deleted-mapping-localization-diagnostic \
  refs/tags/p26-permission-safe-acquisition-checkpoint \
  refs/tags/p26-attempt02-acquisition-source
git bundle list-heads "$P28_LOCAL_RUNTIME_BUNDLE"
git bundle verify "$P28_LOCAL_RUNTIME_BUNDLE"
shasum -a 256 "$P28_LOCAL_RUNTIME_BUNDLE"
wc -c "$P28_LOCAL_RUNTIME_BUNDLE"
p28_o_excl_upload \
  "$P28_LOCAL_RUNTIME_BUNDLE" "$P28_RUNTIME_REVIEW_BUNDLE" 0600
```

The verifier's `runtime-review` phase must use a **second fresh empty bare
repository** at:

```text
/secure/p28/transport-20260906-01/p28_runtime_review_closure.git
```

It repeats full closure and object verification, proves the exact direct-child
two-file delta, advances the clean control checkout only after success,
reconstructs P27 23/23 again, and writes this external `O_EXCL` receipt:

```text
/secure/p28/transport-20260906-01/p28_runtime_review_bundle_receipt.json
```

The second receipt is not committed. The P28 branch object must differ between
the source and runtime-review receipts; the other four ref objects must remain
exactly unchanged.

Its exact receipt-creation form is:

```bash
/usr/bin/python3 "$P28_BOOTSTRAP_VERIFIER" \
  --phase runtime-review \
  --transport-root "$P28_TRANSPORT_ROOT" \
  --bundle "$P28_RUNTIME_REVIEW_BUNDLE" \
  --expected-bundle-sha256 "$P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P28_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" \
  --closure-repository "$P28_RUNTIME_REVIEW_CLOSURE" \
  --control-checkout "$P28_CONTROL_REPO" \
  --attempt-root "$P28_ATTEMPT_ROOT" \
  --expected-p28-commit "$P28_RUNTIME_REVIEW_COMMIT" \
  --expected-p28-tree "$P28_RUNTIME_REVIEW_TREE" \
  --expected-source-commit "$P28_SOURCE_FREEZE_COMMIT" \
  --expected-source-tree "$P28_SOURCE_FREEZE_TREE" \
  --bootstrap-verifier "$P28_BOOTSTRAP_VERIFIER" \
  --expected-verifier-sha256 "$P28_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --receipt-output "$P28_RUNTIME_REVIEW_RECEIPT"
```

## 5. Run the sole unchanged P27 localization

Replay the runtime-review receipt and validate the live container ID, init PID,
restart count, image, GPU, mounts, runtime lock, host attestation, authority
checkout, and reviewed P28 control checkout. The localization marker may be
created only after all checks succeed.

Run exactly once:

```bash
bash "$P28_CONTROL_REPO/scripts/run_p28_bundle_complete_localization_bridge.sh" run-localization
```

The orchestrator stages the unchanged P27 localizer bytes, ingests the terminal
P26 failure with the unchanged P27 ingester and sanitizer authorities, executes
the no-training localizer once, and sanitizes its native output. It must retain
no-overwrite stdout, stderr, native, sanitized, and manifest evidence even on a
nonzero exit. Do not invoke the inner container command directly and do not
retry.

## 6. Freeze the result and route mechanically

Record one compact machine-readable P28 outcome plus native and sanitized
artifact hashes. Route in the contract's precedence order:

- bundle or reconstruction failure: stop at the next state-changing boundary;
- sanitizer or localizer failure: retain evidence and stop;
- deleted executable or writable mapping: design an image/loader correction;
- deleted unclassifiable or identity-unstable mapping: retain probes and keep
  rejection strict;
- conclusively read-only, nonexecutable deleted mapping set: preregister a
  narrow provenance/device policy;
- no deleted mapping: record only nonreproduction and do not rerun.

No route in P28 authorizes scientific acquisition attempt `20260906-03`.
