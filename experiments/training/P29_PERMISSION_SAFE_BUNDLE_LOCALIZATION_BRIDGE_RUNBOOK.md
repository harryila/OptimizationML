# P29 permission-safe bundle localization bridge runbook

## Evidence boundary

P29 corrects the host-directory permission defect recorded by terminal P28 and
closes the mutable staged-source race with one root-owned immutable execution
snapshot, then runs the unchanged P27 no-training deleted-mapping localizer
once.
P28 attempt `20260906-01` and everything beneath `/secure/p28` are retained
terminal evidence. They must not be cleaned, changed, mounted into P29, or
reused.

P29 permits only P27's four diagnostic stages: `pre_cuda`, `post_cuda_init`,
`post_model_move`, and `post_optimizer`. It forbids data loading, forward,
backward, candidate evaluation, optimizer steps, parameter updates, and
training. It does not authorize prospective scientific acquisition
`20260906-03`.

Before any P29 host operation, independently run:

```bash
uv run --locked python -I -S \
  scripts/reconstruct_p29_permission_safe_bundle_localization_bridge.py
```

The P29 reconstructor launches the P28 contract, terminal P28 outcome, and
unchanged P27 reconstructors as three isolated children. Their results must be
12/12, 10/10, and 23/23, respectively.

The locked local Python and the CUDA-host Python must be Python 3.10 or newer.
On the CUDA host, first run
`/usr/bin/python3 -I -S -c 'import sys; assert sys.version_info >= (3, 10)'`;
every later bootstrap or reconstructor entry uses exactly
`/usr/bin/python3 -I -S`. The verifier supplies unchanged nested authorities a
private temporary `python3` shim that execs that same absolute interpreter
with both flags. The independent P29 reconstructor likewise creates a fresh
private `python3` and `git` shim for each child, invokes the child with its own
resolved interpreter and `-I -S`, and makes every nested `python3` invocation
use that same interpreter and flags. The host environment removes `PYTHONHOME`, `PYTHONPATH`,
`PYTHONSTARTUP`, `PYTHONINSPECT`, and `PYTHONUSERBASE`; sets
`PYTHONNOUSERSITE=1`, `PYTHONDONTWRITEBYTECODE=1`, and `PYTHONSAFEPATH=1`; and
uses only `/usr/sbin:/usr/bin:/sbin:/bin`. Git trust operations use
`/usr/bin/git`, remove ambient `GIT_*` inputs, disable system/global config and
their includes, set `GIT_ATTR_NOSYSTEM=1` to disable system gitattributes,
disable hooks, replacement objects, and prompts, reject local checkout include
directives, and reject dirty tracked, untracked, or ignored files in either
checkout.

## 1. Freeze P29 before provisioning its host namespace

Run all portable checks locally:

```bash
uv run --locked pytest -q \
  tests/test_p29_permission_safe_bundle_localization_bridge_contract.py \
  tests/test_p29_control_bundle_verifier.py \
  tests/test_p29_permission_safe_bundle_localization_bridge_orchestration.py \
  tests/test_p28_bundle_complete_localization_bridge_contract.py \
  tests/test_p28_bundle_complete_localization_bridge_outcome.py \
  tests/test_p27_cuda_deleted_mapping_localization_contract.py \
  tests/test_p27_cuda_deleted_mapping_localization.py \
  tests/test_p27_cuda_deleted_mapping_localization_orchestration.py \
  tests/test_p27_cuda_deleted_mapping_localization_sanitizer.py \
  tests/test_p27_p26_failure_ingestion.py
bash -n scripts/run_p29_permission_safe_bundle_localization_bridge.sh
uv run --locked ruff format --check experiments/training scripts tests
uv run --locked ruff check experiments/training scripts tests
/usr/bin/git diff --check
```

Commit and independently review the complete P29 package as one direct child
of terminal P28 commit
`7274367f2b05fb3c9ed9f876a59be647703bbdc2`. Record the resulting commit and
tree as `P29_SOURCE_FREEZE_COMMIT` and `P29_SOURCE_FREEZE_TREE`. Those values
are selected after the commit and therefore are not embedded in the contract.

The source and runtime-review bundles must each advertise exactly these seven
refs:

| Ref | Required object | Required peel |
| --- | --- | --- |
| `refs/heads/p29-permission-safe-bundle-localization-bridge` | phase-specific reviewed P29 commit | same commit |
| `refs/heads/p28-bundle-complete-localization-bridge` | `7274367f2b05fb3c9ed9f876a59be647703bbdc2` | same commit |
| `refs/tags/p28-control-parent-permission-diagnostic` | annotated tag `f3c81a2f14f853f369015a4f81b6695b52eec74e` | `7274367f2b05fb3c9ed9f876a59be647703bbdc2` |
| `refs/heads/p27-cuda-deleted-mapping-localization` | `ec63550331925ded158e3f389e294e4d1f12db3a` | same commit |
| `refs/tags/p27-cuda-deleted-mapping-localization-diagnostic` | annotated tag `c88b98eae9be2458abde45b05d3dccfe09c0c7ed` | `ec63550331925ded158e3f389e294e4d1f12db3a` |
| `refs/tags/p26-permission-safe-acquisition-checkpoint` | annotated tag `bacad707d3779bfa10957e18cb4c69b1a7f0cbce` | `5429da23ff18888daa2312c530a4587780484d8b` |
| `refs/tags/p26-attempt02-acquisition-source` | annotated tag `f35a7dca8f6bc39e9748e79b6712fab4a203396b` | `185e444afc0b44ca0a09b1bde49a6b6fa3973355` |

Never substitute a peeled commit for an annotated tag object.

Create and review the complete source bundle locally:

```bash
export P29_LOCAL_TRANSPORT="$(mktemp -d /private/tmp/p29-transfer.XXXXXX)"
export P29_LOCAL_SOURCE_BUNDLE="$P29_LOCAL_TRANSPORT/p29_source.bundle"
/usr/bin/git bundle create "$P29_LOCAL_SOURCE_BUNDLE" \
  refs/heads/p29-permission-safe-bundle-localization-bridge \
  refs/heads/p28-bundle-complete-localization-bridge \
  refs/tags/p28-control-parent-permission-diagnostic \
  refs/heads/p27-cuda-deleted-mapping-localization \
  refs/tags/p27-cuda-deleted-mapping-localization-diagnostic \
  refs/tags/p26-permission-safe-acquisition-checkpoint \
  refs/tags/p26-attempt02-acquisition-source
/usr/bin/git bundle list-heads "$P29_LOCAL_SOURCE_BUNDLE"
/usr/bin/git bundle verify "$P29_LOCAL_SOURCE_BUNDLE"
shasum -a 256 "$P29_LOCAL_SOURCE_BUNDLE"
wc -c "$P29_LOCAL_SOURCE_BUNDLE"
```

`list-heads` must have exactly seven rows and the four tag rows must name tag
objects, not peeled commits.

## 2. Provision the exact permission-safe namespace once

Use these identities:

```bash
export P29_SSH_TARGET=ubuntu@129.146.177.214
export P29_ATTEMPT_ID=20260906-02
export P29_ATTEMPT_ROOT=/secure/p29/attempt-20260906-02
export P29_CONTAINER=p29-localization-20260906-02
export P29_SECURE_ROOT=/secure/p29
export P29_TRANSPORT_ROOT=/secure/p29/transport-20260906-02
export P29_EXECUTION_ROOT=/secure/p29/execution-20260906-02
export P29_EXECUTION_MANIFEST=/secure/p29/execution-20260906-02/snapshot-manifest.json
export P29_CONTROL_REPO=/secure/p29/transport-20260906-02/control-source
export P29_AUTHORITY_REPO=/secure/p29/transport-20260906-02/authority-185e444
export P29_LEDGER_ROOT=/var/lib/optimizationml-p29-20260906-02
export P29_PREPARE_INVOCATION=/var/lib/optimizationml-p29-20260906-02/prepare-runtime.invoked
export P29_LOCALIZATION_INVOCATION=/var/lib/optimizationml-p29-20260906-02/run-localization.invoked
export P29_LOCALIZATION_AUTHORIZATION=/var/lib/optimizationml-p29-20260906-02/localization.authorized
export P29_LOCALIZATION_EXIT_STATUS=/var/lib/optimizationml-p29-20260906-02/run-localization.exit-status.txt
export P29_SOURCE_BUNDLE=/secure/p29/transport-20260906-02/p29_source.bundle
export P29_SOURCE_CLOSURE=/secure/p29/transport-20260906-02/p29_source_closure.git
export P29_SOURCE_RECEIPT=/secure/p29/transport-20260906-02/p29_source_bundle_receipt.json
export P29_BOOTSTRAP_VERIFIER=/secure/p29/transport-20260906-02/verify_p29_control_bundle.py
```

Within `/secure/p29`, the only directories created before transfer are the
root-owned P29 namespace and its one process-owned transport capability root.
A separate non-capability ledger root is created empty under `/var/lib`.
All three new paths must initially be absent. `/secure` itself must already be
a root-owned, nonsymlink, nonmountpoint directory at mode `0755`, with its base
access ACL equal to those mode bits and no named/default or frozen-xattr ACL.
Do not repair `/secure` as part of P29: reject a mismatch before creating the
P29 namespace. Run these once on the CUDA host; if any command fails or any new
path already exists, retain the state and stop permanently under this
identifier:

```bash
set -euo pipefail
sudo /usr/bin/python3 -I -S - <<'PY'
import os
import stat

flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
root = os.open("/", flags)
secure = None
try:
    secure = os.open("secure", flags, dir_fd=root)
    before = os.fstat(secure)
    by_name = os.stat("secure", dir_fd=root, follow_symlinks=False)
    after = os.fstat(secure)
    fields = lambda value: (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
    )
    assert fields(before) == fields(after) == fields(by_name)
    assert stat.S_ISDIR(after.st_mode)
    assert (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode)) == (0, 0, 0o755)
    assert not os.path.ismount("/secure")
    forbidden = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    names = {name.decode() if isinstance(name, bytes) else name for name in os.listxattr(secure)}
    assert not names & forbidden
finally:
    if secure is not None:
        os.close(secure)
    os.close(root)
PY
sudo /usr/bin/test ! -e /secure/p29
sudo /usr/bin/test ! -L /secure/p29
sudo /usr/bin/mkdir --mode=0755 /secure/p29
sudo /usr/bin/chown root:root /secure/p29
sudo /usr/bin/test ! -e /secure/p29/transport-20260906-02
sudo /usr/bin/test ! -L /secure/p29/transport-20260906-02
sudo /usr/bin/mkdir --mode=0700 /secure/p29/transport-20260906-02
sudo /usr/bin/chown ubuntu:ubuntu /secure/p29/transport-20260906-02
sudo /usr/bin/test ! -e /var/lib/optimizationml-p29-20260906-02
sudo /usr/bin/test ! -L /var/lib/optimizationml-p29-20260906-02
sudo /usr/bin/mkdir --mode=0700 /var/lib/optimizationml-p29-20260906-02
sudo /usr/bin/chown root:root /var/lib/optimizationml-p29-20260906-02
sudo /usr/bin/mkdir --mode=0700 /var/lib/optimizationml-p29-20260906-02/deferred
sudo /usr/bin/chown root:root /var/lib/optimizationml-p29-20260906-02/deferred
```

The ledger root is walked componentwise without following symlinks. It remains
`root:root` mode `0700`, has a base access ACL equal to those mode bits, no
named/default or frozen-xattr ACL, and has exactly one initially empty child,
the root-owned mode-`0700` `deferred` journal, through source receipt creation.
Dispatcher entry for the sole `prepare-runtime` call first uses the supplied
expected sealed-orchestrator digest to authenticate its token transaction,
then O_EXCL-creates root-owned mode-`0400` `prepare-runtime.invoked` with exact
bytes `prepare-runtime\n`, before general environment validation or source
receipt replay. A missing or malformed supplied digest can therefore fail
before token creation. After successful token creation, a failed replay is
terminal while attempt-root, execution-root, container, and GPU mutation all
remain behind that replay gate. The prepare token persists through
runtime-review receipt creation and replay and every later localization state.
Dispatcher entry for the sole localization call likewise consumes the
supplied expected sealed-orchestrator digest before general environment,
evidence, or marker validation, then O_EXCL-creates root-owned mode-`0400`
`run-localization.invoked` with exact bytes `run-localization\n`. Runtime
receipt replay requires both invocation tokens present and requires
`localization.authorized` and the terminal status absent. Only after receipt
replay and every reviewed pre-marker validator pass
may the host O_EXCL-create root-owned mode-`0400` `localization.authorized`
with exact bytes `localization-authorized\n`. The retained host
`p29-post-authorization-ledger-validation` then requires the exact
prepare-invocation, localization-invocation, and authorization files before
container dispatch. The containerized P27 localizer does not read or receive
the root ledger. The dispatcher executes the localization body in a subshell
and, when that subshell returns, attempts exactly one O_EXCL transaction for
root-owned mode-`0400` `run-localization.exit-status.txt`. When that
transaction succeeds, the complete contents are one canonical ASCII decimal
integer in `0..255` followed by one newline; this includes an early
post-invocation predicate or evidence-logger failure. A terminal-status
transaction failure is itself terminal and nonrunnable and may leave the
status absent or created but unvalidated. Therefore the complete final ledger
inventory is guaranteed only after a successful status transaction: in
addition to the journal and sealed orchestrator, it has both invocation tokens
plus exit status if authorization was never reached, or those three plus
authorization otherwise. None of those validated files may be removed,
replaced, rewritten, or created twice. Process destruction that prevents the
parent dispatcher from running likewise cannot be mistaken for a completed
invocation: the invocation token remains while the missing or unvalidated
terminal status makes the state explicitly incomplete and permanently
non-rerunnable.

Immediately before each prepare-invocation, localization-invocation,
authorization-token, or terminal-status O_EXCL, the creation transaction
atomically reauthenticates the deferred directory as stable `root:root` mode
`0700` with no ACL and the sealed orchestrator as stable `root:root` mode
`0555` with no ACL and the reviewed SHA-256. No token or terminal status is
created from a by-path check that can race a replacement of either authority.

Do not precreate `control-source`, `authority-185e444`, the attempt root, or the
execution snapshot.
Do not make `/secure/p29` writable by `ubuntu`. Do not bind, copy, symlink, or
privileged-move either checkout to a sibling of the transport root.
After direct creation, control and authority are nonsymlink directories owned
by `ubuntu:ubuntu` (`1000:1000`) at mode `0700`, share the transport device and
mount ID, are not mountpoints, and have no extended or named access ACL, no
default ACL, and no frozen ACL xattr. Their base access ACL exactly matches the
mode bits. Those facts are revalidated rather than inferred from the parent.
Before source receipt creation or replay, `/secure/p29` has exactly one child:
`transport-20260906-02`. After the sole prepare phase, it has exactly three
sorted children: `attempt-20260906-02`, `execution-20260906-02`, and
`transport-20260906-02`. Any extra file, directory, symlink, mount, or other
capability sibling is terminal; it is never ignored or cleaned.

Before and after every source-phase transition, the verifier must walk
`/secure`, `/secure/p29`, and the transport root component by component using
`O_DIRECTORY|O_NOFOLLOW`. For `/`, it records and stabilizes device and inode
only. From `/secure` downward it additionally
records and revalidates uid, gid, mode, lexical identity, resolved identity,
ACL, and mount identity. The required POSIX ACLs, shown without comments or
numeric-name translation, are:

```text
/secure
user::rwx
group::r-x
other::r-x

/secure/p29
user::rwx
group::r-x
other::r-x

/secure/p29/transport-20260906-02
user::rwx
group::---
other::---
```

Named user/group ACL entries and every default ACL entry are forbidden. The
four ACL xattrs `system.posix_acl_access`, `system.posix_acl_default`,
`system.nfs4_acl`, and `system.richacl` must all be absent. The `/secure`
parent, namespace, transport, execution, attempt, and evidence directories
must share one device and mount ID and none may be a mountpoint.
Any identity, permission, or ACL change consumes P29; do not repair and retry.

## 3. Transfer and verify the source bundle before authority or attempt state

Create no additional directory. From the local workstation, stage the bundle
and separately reviewed verifier with `O_EXCL|O_NOFOLLOW`:

```bash
p29_o_excl_upload() {
  local source_path="$1"
  local destination_path="$2"
  local destination_mode="$3"
  test -f "$source_path"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P29_SSH_TARGET" \
    "/usr/bin/python3 -I -S -c 'import os,shutil,sys; p=sys.argv[1]; m=int(sys.argv[2],8); f=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,\"O_CLOEXEC\",0)|getattr(os,\"O_NOFOLLOW\",0); d=os.open(p,f,m); h=os.fdopen(d,\"wb\"); shutil.copyfileobj(sys.stdin.buffer,h); h.flush(); os.fsync(h.fileno()); h.close()' '$destination_path' '$destination_mode'" \
    <"$source_path"
}
p29_o_excl_upload "$P29_LOCAL_SOURCE_BUNDLE" "$P29_SOURCE_BUNDLE" 0600
p29_o_excl_upload \
  scripts/verify_p29_control_bundle.py "$P29_BOOTSTRAP_VERIFIER" 0600
```

Authenticate the transferred SHA-256 values, byte counts, owners, and modes.
The source verifier must then:

1. authenticate its exact bootstrap bytes and the permission-safe layout;
2. require control, authority, attempt, closure, and receipt paths absent;
3. copy the bundle once with `O_NOFOLLOW` to a private mode-`0400` file;
4. reject prerequisites and require exactly the seven advertised refs;
5. create a fresh empty bare closure, explicitly fetch all seven same-named
   refs, verify tag types and peels, and run full strict `git fsck`;
6. construct the absent canonical control checkout directly at
   `/secure/p29/transport-20260906-02/control-source` without a bind, copy,
   symlink, relocation, or privileged operation;
7. prove the control is detached, clean, closed, and free of Git object
   indirections;
8. execute the P29 reconstructor and its P28-contract 12/12, terminal-P28
   outcome 10/10, and unchanged-P27 23/23 children only from authenticated
   private bundle bytes, never through mutable checkout paths; and
9. revalidate every layout and bundle identity before writing the external
   source receipt with `O_EXCL`.

In an `ubuntu` shell on the CUDA host, export the same fixed paths plus the
independently reviewed source commit, tree, bundle, and verifier values. Run
the reviewed bootstrap verifier exactly once in source creation mode. The
root-ledger state with only its empty deferred journal is a provisioning and
operator precondition here; the UID-1000 verifier cannot and does not inspect
that root-only ledger:

```bash
/usr/bin/python3 -I -S "$P29_BOOTSTRAP_VERIFIER" \
  --phase source \
  --transport-root "$P29_TRANSPORT_ROOT" \
  --bundle "$P29_SOURCE_BUNDLE" \
  --expected-bundle-sha256 "$P29_EXPECTED_SOURCE_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P29_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" \
  --closure-repository "$P29_SOURCE_CLOSURE" \
  --control-checkout "$P29_CONTROL_REPO" \
  --attempt-root "$P29_ATTEMPT_ROOT" \
  --execution-root "$P29_EXECUTION_ROOT" \
  --expected-p29-commit "$P29_SOURCE_FREEZE_COMMIT" \
  --expected-p29-tree "$P29_SOURCE_FREEZE_TREE" \
  --bootstrap-verifier "$P29_BOOTSTRAP_VERIFIER" \
  --expected-verifier-sha256 "$P29_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --receipt-output "$P29_SOURCE_RECEIPT"
```

Review the external receipt and record its SHA-256. A failed verifier leaves
all partial state in place and permanently ends P29 attempt `20260906-02`.
There is no cleanup, repair, resume, or second source-verifier invocation.

The source-receipt creation/replay distinction is exact. Creation requires the
authority checkout to remain absent for the entire invocation. A later
read-only replay may observe either an absent authority or the exact canonical
`authority-185e444` checkout. If present, that checkout must independently
match the locked owner, mode, ACL, device/mount identity, detached commit
`185e444afc0b44ca0a09b1bde49a6b6fa3973355`, tree
`24f4bdac331a57bd7c1b807747d7c7fba253ee5a`, and all Git-indirection guards.
Replay never creates, mutates, relocates, or repairs authority state. The
attempt and execution roots remain absent throughout source-receipt creation
and replay. The root ledger has no invocation files throughout source-receipt
creation. The sole prepare dispatcher burns `prepare-runtime.invoked` before
the one source-receipt replay; the localization invocation, authorization, and
terminal status remain absent through that replay and runtime-review receipt
creation.

Immediately after source-receipt review, trusted root provisioning must
component-walk the reviewed control checkout and ledger with no-follow opens,
stable-read the exact reviewed orchestration bytes, verify their preregistered
SHA-256, and O_EXCL-create
`/var/lib/optimizationml-p29-20260906-02/run_p29_permission_safe_bundle_localization_bridge.sh`
as `root:root` mode `0555`. Its source and destination identities, hash, size,
mode, link count, and absence of ACL/xattr authority are rechecked before the
source descriptor closes. No phase may later execute the process-writable
control-checkout path. Both phase invocations use only
`/bin/bash -p /var/lib/optimizationml-p29-20260906-02/run_p29_permission_safe_bundle_localization_bridge.sh ...`;
privileged Bash mode prevents `BASH_ENV`, exported shell functions, `SHELLOPTS`,
and related ambient startup controls from replacing the reviewed code.

Use this no-follow, no-overwrite seal after exporting the independently
reviewed `P29_EXPECTED_ORCHESTRATOR_SHA256`:

```bash
sudo /usr/bin/python3 -I -S - \
  "$P29_CONTROL_REPO/scripts/run_p29_permission_safe_bundle_localization_bridge.sh" \
  "$P29_LEDGER_ROOT" "$P29_EXPECTED_ORCHESTRATOR_SHA256" <<'PY'
import hashlib
import os
import pathlib
import stat
import sys

source = pathlib.PurePosixPath(sys.argv[1])
ledger = pathlib.PurePosixPath(sys.argv[2])
expected = sys.argv[3]
destination_name = "run_p29_permission_safe_bundle_localization_bridge.sh"
if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
    raise SystemExit("P29 reviewed orchestrator hash is malformed")
directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW


def open_directory(path):
    descriptor = os.open("/", directory_flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, directory_flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def fields(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
        value.st_nlink,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


source_parent = open_directory(source.parent)
ledger_fd = open_directory(ledger)
source_fd = destination_fd = None
try:
    ledger_info = os.fstat(ledger_fd)
    if (
        not stat.S_ISDIR(ledger_info.st_mode)
        or (ledger_info.st_uid, ledger_info.st_gid, stat.S_IMODE(ledger_info.st_mode))
        != (0, 0, 0o700)
        or sorted(os.listdir(ledger_fd)) != ["deferred"]
    ):
        raise SystemExit("P29 pre-seal ledger authority or inventory differs")
    source_fd = os.open(
        source.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=source_parent
    )
    before = os.fstat(source_fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (1000, 1000, 0o644)
    ):
        raise SystemExit("P29 reviewed orchestrator source authority differs")
    forbidden_acl = {
        "system.posix_acl_access",
        "system.posix_acl_default",
        "system.nfs4_acl",
        "system.richacl",
    }
    if {
        value.decode() if isinstance(value, bytes) else value
        for value in os.listxattr(source_fd)
    } & forbidden_acl:
        raise SystemExit("P29 reviewed orchestrator source has a forbidden ACL")
    raw = bytearray()
    while chunk := os.read(source_fd, 1024 * 1024):
        raw.extend(chunk)
    after = os.fstat(source_fd)
    by_name = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
    if fields(before) != fields(after) or fields(after) != fields(by_name):
        raise SystemExit("P29 reviewed orchestrator changed during stable read")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit("P29 reviewed orchestrator hash differs")
    destination_fd = os.open(
        destination_name,
        os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o555,
        dir_fd=ledger_fd,
    )
    view = memoryview(raw)
    while view:
        written = os.write(destination_fd, view)
        if written <= 0:
            raise SystemExit("P29 sealed orchestrator write made no progress")
        view = view[written:]
    os.fchmod(destination_fd, 0o555)
    os.fsync(destination_fd)
    sealed = os.fstat(destination_fd)
    sealed_by_name = os.stat(destination_name, dir_fd=ledger_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(sealed.st_mode)
        or sealed.st_nlink != 1
        or (sealed.st_uid, sealed.st_gid, stat.S_IMODE(sealed.st_mode))
        != (0, 0, 0o555)
        or sealed.st_size != len(raw)
        or fields(sealed) != fields(sealed_by_name)
        or hashlib.sha256(os.pread(destination_fd, sealed.st_size, 0)).hexdigest()
        != expected
        or {
            value.decode() if isinstance(value, bytes) else value
            for value in os.listxattr(destination_fd)
        }
        & forbidden_acl
        or sorted(os.listdir(ledger_fd)) != ["deferred", destination_name]
    ):
        raise SystemExit("P29 sealed orchestrator authority differs")
    os.fsync(ledger_fd)
finally:
    if destination_fd is not None:
        os.close(destination_fd)
    if source_fd is not None:
        os.close(source_fd)
    os.close(ledger_fd)
    os.close(source_parent)
PY
```

## 4. Construct the authority and prepare one runtime

Only after source-receipt review, create the authority in that same CUDA-host
`ubuntu` shell directly beneath the transport root. A direct construction
failure is terminal and is never cleaned or retried:

```bash
set -euo pipefail
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
for p29_git_variable in $(compgen -A variable GIT_ || true); do
  unset "$p29_git_variable"
done
unset p29_git_variable
export GIT_CONFIG_NOSYSTEM=1
export GIT_ATTR_NOSYSTEM=1
export GIT_CONFIG_SYSTEM=/dev/null
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_NO_REPLACE_OBJECTS=1
export GIT_OPTIONAL_LOCKS=0
export GIT_TERMINAL_PROMPT=0
test ! -e "$P29_AUTHORITY_REPO"
/usr/bin/git init --initial-branch=p29-authority-unborn "$P29_AUTHORITY_REPO"
/usr/bin/git -C "$P29_AUTHORITY_REPO" config core.autocrlf false
/usr/bin/git -C "$P29_AUTHORITY_REPO" config core.filemode true
/usr/bin/git -C "$P29_AUTHORITY_REPO" config core.hooksPath /dev/null
/usr/bin/git -C "$P29_AUTHORITY_REPO" config core.fsmonitor false
/usr/bin/git -C "$P29_AUTHORITY_REPO" fetch --no-tags --no-write-fetch-head \
  "$P29_SOURCE_CLOSURE" \
  +refs/tags/p26-attempt02-acquisition-source:refs/tags/p26-attempt02-acquisition-source
/usr/bin/git -C "$P29_AUTHORITY_REPO" checkout --detach \
  185e444afc0b44ca0a09b1bde49a6b6fa3973355
chmod 0700 "$P29_AUTHORITY_REPO"
p29_authority_head=$(/usr/bin/git -C "$P29_AUTHORITY_REPO" rev-parse HEAD)
test "$p29_authority_head" = \
  185e444afc0b44ca0a09b1bde49a6b6fa3973355
p29_authority_tree=$(/usr/bin/git -C "$P29_AUTHORITY_REPO" rev-parse HEAD^{tree})
test "$p29_authority_tree" = \
  24f4bdac331a57bd7c1b807747d7c7fba253ee5a
p29_authority_status=$(
  /usr/bin/git -C "$P29_AUTHORITY_REPO" status --porcelain=v1 --untracked-files=all
)
test -z "$p29_authority_status"
p29_authority_ignored=$(
  /usr/bin/git -C "$P29_AUTHORITY_REPO" ls-files --others --ignored --exclude-standard
)
test -z "$p29_authority_ignored"
test ! -e "$P29_AUTHORITY_REPO/.git/shallow"
test ! -e "$P29_AUTHORITY_REPO/.git/objects/info/alternates"
test ! -e "$P29_AUTHORITY_REPO/.git/info/grafts"
p29_authority_replacements=$(
  /usr/bin/git -C "$P29_AUTHORITY_REPO" for-each-ref --format='%(refname)' refs/replace
)
test -z "$p29_authority_replacements"
if p29_partial_clone=$(
  /usr/bin/git -C "$P29_AUTHORITY_REPO" config --local --get extensions.partialClone
); then
  test -z "$p29_partial_clone"
else
  test "$?" -eq 1
fi
if p29_promisor=$(
  /usr/bin/git -C "$P29_AUTHORITY_REPO" config --local --get-regexp '^remote\..*\.promisor$'
); then
  test -z "$p29_promisor"
else
  test "$?" -eq 1
fi
if p29_local_includes=$(
  /usr/bin/git -C "$P29_AUTHORITY_REPO" config --local --get-regexp '^include.*\.'
); then
  test -z "$p29_local_includes"
else
  test "$?" -eq 1
fi
/usr/bin/git -C "$P29_AUTHORITY_REPO" fsck --full --strict --no-dangling
```

Export every reviewed P29 source, receipt, verifier, orchestrator,
reconstructor, terminal-P28, and unchanged-P27 hash required by the host
orchestrator, plus the pinned inputs:

```bash
export P29_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
export P29_NANOGPT_HOST=/secure/p23/nanoGPT
export P29_MUON_HOST=/secure/p23/Muon
export P29_DATA_HOST=/secure/p23/optimizationml-p22-data
```

The sealed host orchestrator must run as numeric UID/GID `0:0`; the bootstrap
bundle verifier separately remains UID/GID `1000:1000` for transport/control
mutations. The orchestrator binds the
nanoGPT checkout to commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`,
tree `ca93bcd9b9c9ff32d3016e1e2556644e68bef86a`, and the Muon checkout to
commit `f98f1cacc0263b04290753e32be8d498c1efc806`, tree
`4ea5cd8ab6ebd56a18536f06453619efcd636da0`; clean means no tracked,
untracked, or ignored dirt. Do not rely on `sudo` environment inheritance or
`--preserve-env`: default `sudoers` policy commonly drops the required `P29_*`
values. Instead, pass exactly the preregistered source variables through a
clean `/usr/bin/env -i` argument vector:

```bash
p29_source_root_environment=(
  "P29_ATTEMPT_ID=$P29_ATTEMPT_ID"
  "P29_GPU_UUID=$P29_GPU_UUID"
  "P29_AUTHORITY_REPO=$P29_AUTHORITY_REPO"
  "P29_NANOGPT_HOST=$P29_NANOGPT_HOST"
  "P29_MUON_HOST=$P29_MUON_HOST"
  "P29_DATA_HOST=$P29_DATA_HOST"
  "P29_SOURCE_FREEZE_COMMIT=$P29_SOURCE_FREEZE_COMMIT"
  "P29_SOURCE_FREEZE_TREE=$P29_SOURCE_FREEZE_TREE"
  "P29_EXPECTED_SOURCE_BUNDLE_SHA256=$P29_EXPECTED_SOURCE_BUNDLE_SHA256"
  "P29_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT=$P29_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT"
  "P29_EXPECTED_SOURCE_RECEIPT_SHA256=$P29_EXPECTED_SOURCE_RECEIPT_SHA256"
  "P29_EXPECTED_CONTRACT_SHA256=$P29_EXPECTED_CONTRACT_SHA256"
  "P29_EXPECTED_ORCHESTRATOR_SHA256=$P29_EXPECTED_ORCHESTRATOR_SHA256"
  "P29_EXPECTED_RECONSTRUCTOR_SHA256=$P29_EXPECTED_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_BUNDLE_VERIFIER_SHA256=$P29_EXPECTED_BUNDLE_VERIFIER_SHA256"
  "P29_EXPECTED_P28_CONTRACT_SHA256=$P29_EXPECTED_P28_CONTRACT_SHA256"
  "P29_EXPECTED_P28_RECONSTRUCTOR_SHA256=$P29_EXPECTED_P28_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_P28_OUTCOME_SHA256=$P29_EXPECTED_P28_OUTCOME_SHA256"
  "P29_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256=$P29_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_P27_CONTRACT_SHA256=$P29_EXPECTED_P27_CONTRACT_SHA256"
  "P29_EXPECTED_P27_RECONSTRUCTOR_SHA256=$P29_EXPECTED_P27_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_P27_LOCALIZER_SHA256=$P29_EXPECTED_P27_LOCALIZER_SHA256"
  "P29_EXPECTED_INGESTER_SHA256=$P29_EXPECTED_INGESTER_SHA256"
  "P29_EXPECTED_P23_CORE_SHA256=$P29_EXPECTED_P23_CORE_SHA256"
  "P29_EXPECTED_P27_SANITIZER_SHA256=$P29_EXPECTED_P27_SANITIZER_SHA256"
)
p29_verify_root_environment_handoff() {
  local expected_count="$1"
  shift
  test "$#" -eq "$expected_count"
  sudo /usr/bin/env -i \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    "$@" \
    /usr/bin/python3 -I -S -c \
    'import os,sys
count=int(sys.argv[1]); pairs=sys.argv[2:]
if len(pairs) != count: raise SystemExit("P29 handoff count differs")
expected={}
for pair in pairs:
    name, separator, value = pair.partition("=")
    if separator != "=" or not name.startswith("P29_") or not value or name in expected:
        raise SystemExit("P29 handoff assignment differs")
    expected[name]=value
actual={name:value for name,value in os.environ.items() if name.startswith("P29_")}
if actual != expected or os.geteuid() != 0 or os.getegid() != 0:
    raise SystemExit("P29 root handoff differs")' \
    "$expected_count" "$@"
}
p29_verify_root_environment_handoff 25 "${p29_source_root_environment[@]}"
```

Source-receipt replay is read-only, but the fail-closed prepare admission that
contains it is not: a successful dispatcher token transaction burns
`prepare-runtime.invoked` before the replay. If that transaction creates the
token but fails an internal post-create check, the token remains without an
admission journal as an explicit, permanently non-rerunnable incomplete
state. In the intended run the replay sees and authenticates the canonical
authority just constructed; that is deliberately different from the
authority-absent creation invocation. Run the sealed dispatcher exactly once;
it must complete the sole replay before it may create the attempt root,
execution snapshot, container, or CUDA state:

```bash
sudo /usr/bin/env -i \
  PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  "${p29_source_root_environment[@]}" \
  /bin/bash -p \
  /var/lib/optimizationml-p29-20260906-02/run_p29_permission_safe_bundle_localization_bridge.sh \
  prepare-runtime
```

These assignments precede `/bin/bash -p` and the sealed script in the root
process argument vector. The script revalidates all 25 values after its
one-shot token admission and before the prepare body can create the attempt,
execution snapshot, container, or CUDA state. The executing script path and
digest are also checked after admission. Operators must invoke the sealed path;
a different root-dispatched copy can consume the token and then only fail
closed, so it is not a retryable setup mistake.

Because `evidence` does not exist before attempt construction, after the
prepare-token transaction succeeds the complete pre-attempt
admission—including source-receipt replay and every
namespace, control, history, ledger, absence, input, container-name, and image
validator—runs inside one root-journal-only retained boundary. Its exact files
are `.deferred-p29-prepare-pre-attempt-admission.stdout.log`,
`.stderr.log`, and `.exit-status.txt`; all are O_EXCL/no-follow, root-owned,
sealed mode `0400`, and never published into the later evidence directory
after successful artifact finalization. Finalization failure best-effort seals
every already-closed member and returns terminal 125. A
nonzero admission status is terminal. The immediate `after-admission`
validator is parameterized by that exact producer status, so a producer exit
such as 37 must retain exact `37\n` and return 37 after successful
finalization; it is not rewritten to zero. Progression to `after-prepare` or
`after-runtime` requires the retained admission status to be exactly `0\n`.

The following
`p29-prepare-attempt-layout` command writes its three O_EXCL transcripts to the
root-only deferred journal. The subsequent publication into the newly created
evidence directory best-effort creates and seals its own journal stdout,
stderr, and exit status. On successful finalization both stdout/stderr are
empty and the status is exactly
`0\n`; the journal then has the three admission files plus exactly the six
preregistered
`.deferred-p29-prepare-attempt-layout{,.publication}.{stdout.log,stderr.log,exit-status.txt}`
files (using the literal filename combinations listed by the contract) and no
failure sentinel. Publication failure best-effort adds the preregistered
`.publication-failure.exit-status.txt` sentinel and permanently ends P29; a
status or seal failure may leave an incomplete journal and returns terminal
125 rather than claiming the complete inventory.

The orchestrator creates `/secure/p29/attempt-20260906-02` with privilege only
at this point. Its final authority is `root:root` mode `0555`; its direct
`evidence` child is `root:root` mode `0555`. The root orchestrator and the
UID-0 container create only preregistered children with `O_EXCL`; once each
producer closes a child, the host seals it `root:root` mode `0444` before any
review or next boundary. UID 1000 can traverse and read the credential-free
evidence for receipt verification but cannot create, rewrite, rename, or
replace evidence or its children.
Both are nonsymlink nonmountpoint directories. If creation is partial or any
later preparation fails, the attempt is consumed. They are validated
componentwise with `O_DIRECTORY|O_NOFOLLOW`. The host reopens them at each
logged boundary and compares the new records; it does not claim one descriptor
remains open across the entire marker-to-localizer process span. The receipt
verifier separately holds its descriptors throughout each receipt create or
replay invocation. Their device, inode, UID/GID, mode, base access ACL matching
the mode bits, absence of extended/named or default ACLs, absence of all four
frozen ACL xattrs, and mount ID must remain unchanged. They share the
`/secure` device and mount ID. A failed post-transition revalidation is
terminal evidence, never a repair opportunity.

During that same sole prepare invocation, the orchestrator derives a separate
execution authority at `/secure/p29/execution-20260906-02`. This is not a
relocation, bind, or replacement of canonical transport authority
`authority-185e444`. Its four root-owned immutable directory snapshots are:

| Snapshot | Exact authority |
| --- | --- |
| `repository` | commit `185e444afc0b44ca0a09b1bde49a6b6fa3973355`, tree `24f4bdac331a57bd7c1b807747d7c7fba253ee5a` |
| `nanogpt` | commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`, tree `ca93bcd9b9c9ff32d3016e1e2556644e68bef86a` |
| `muon` | commit `f98f1cacc0263b04290753e32be8d498c1efc806`, tree `4ea5cd8ab6ebd56a18536f06453619efcd636da0` |
| `data` | original `materialized/p22_fineweb_manifest.json`, 4,210 bytes, SHA-256 `1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d` |

The execution root and every directory are `root:root` mode `0555`; regular
files are `root:root` mode `0444` or `0555`. Recursively reject symlinks,
hardlinked regular files, special files, nested mounts, ACLs, and xattrs.
`snapshot-manifest.json` is `root:root` mode `0444`, uses schema
`passive-muon-p29-immutable-execution-snapshot-v1`, and is canonical UTF-8 JSON
(`sort_keys=True`, compact separators, one trailing newline). Its sorted
`entries` array covers every relative path other than the manifest itself.
Every entry binds relative path, kind, mode, UID, and GID; file entries also
bind byte count and SHA-256. Its `source_authorities` bind all three commit/tree
pairs, the original data-manifest hash/size, every helper hash, and the P26
native hash/size.

The read-only adjunct `data/.p29-tools/` contains exactly the unchanged P27
localizer, P27 ingester, P23 core, P27 sanitizer, and authenticated P26 input:

```text
run_p27_cuda_deleted_mapping_localization.py
ingest_p26_trace_off_a_failure.py
p23_deterministic_cuda_shadow_trace.py
sanitize_p27_cuda_deleted_mapping_localization.py
p26-trace-off-a-failure.authenticated.json
```

The P26 input is exactly 66,283 bytes with SHA-256
`b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91`.
The complete manifest is verified at the retained prepare snapshot boundary,
during runtime-receipt creation, and during runtime-receipt replay. Later
pre-marker checks authenticate the top-level snapshot inventory and manifest
authority without claiming another full-manifest replay.

After the attempt root first exists, every substantive command or validator is
invoked only after its retained no-overwrite logger has admitted the declared
stdout/stderr paths. The same rule applies after the localization marker.
Logger admission or artifact-finalization failure is terminal and may lack a
complete transcript triplet; it never licenses an unlogged substantive
validator or a retry.
It creates one `p29-localization-20260906-02` container using the unchanged
image, GPU, network-none, read-only-rootfs, tmpfs, deterministic environment,
and exact ten P27 destinations. The four directory sources are the immutable
execution `repository`, `nanogpt`, `muon`, and `data` snapshots; the repository
snapshot also supplies the read-only materializer alias; only the evidence
mount is writable; and the four retained inspection files remain read-only.
The canonical transport authority and all mutable input sources are never
mounted. No localizer may run yet.

The container launch is pinned to
`--user 0:0 --userns host --cap-drop ALL --cap-add DAC_OVERRIDE`. The retained
Docker inspection must show `Config.User == "0:0"`,
`HostConfig.UsernsMode == "host"`, `HostConfig.CapDrop == ["ALL"]`, and
`HostConfig.CapAdd == ["DAC_OVERRIDE"]`. Immediately after launch, the retained
`p29-fresh-container-id-validation.{stdout.log,stderr.log,exit-status.txt}`
triplet binds the returned ID to the running container and rejects the
terminal P26 ID before later inspection or freeze steps.

The frozen root container and its `DAC_OVERRIDE` capability are explicitly
inside the runtime trusted computing base. Root ownership and mode `0444`
prevent mutation by the UID-1000 transport/verifier process; they are not
claimed to defend against a malicious root container. At each declared
producer boundary the host seals and authenticates the newly produced
artifact, and the explicit receipt/namespace validators reauthenticate their
declared identities. This is not a claim that every historical evidence byte
or the complete execution manifest is reread after every logger boundary.

The exact mount table is:

| Host source | Frozen container destination | Access |
| --- | --- | --- |
| `/secure/p29/execution-20260906-02/repository` | `/workspace/OptimizationML` | read-only |
| `/secure/p29/execution-20260906-02/nanogpt` | `/workspace/inputs/nanoGPT` | read-only |
| `/secure/p29/execution-20260906-02/muon` | `/workspace/inputs/muon` | read-only |
| `/secure/p29/execution-20260906-02/data` | `/private/tmp/optimizationml-p22-data` | read-only |
| `/secure/p29/execution-20260906-02/repository/experiments/training/materialize_p22_fineweb.py` | `/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py` | read-only |
| `/secure/p29/attempt-20260906-02/evidence` | `/workspace/evidence/p23` | read-write |
| `/secure/p29/attempt-20260906-02/p29-image-inspect.json` | `/mounted-host-evidence/image-inspect.json` | read-only |
| `/secure/p29/attempt-20260906-02/p29-running-container-inspect.json` | `/mounted-host-evidence/running-container-inspect.json` | read-only |
| `/secure/p29/attempt-20260906-02/p29-running-mountinfo.txt` | `/mounted-host-evidence/running-mountinfo.txt` | read-only |
| `/secure/p29/attempt-20260906-02/p29-nvidia-smi.csv` | `/mounted-host-evidence/nvidia-smi.csv` | read-only |

Review these two new retained artifacts:

```text
experiments/training/p29_cuda_runtime_lock.json
experiments/training/p29_host_attestation.json
```

The attempt root itself has exactly `evidence`, `p29-image-inspect.json`,
`p29-nvidia-smi.csv`, `p29-running-container-inspect.json`,
`p29-running-mountinfo.txt`, and `p29-container-id.txt`; no other top-level
entry is permitted before runtime-review receipt publication or replay.

The exact pre-localization runtime-receipt evidence allowlist is these 27
files—no more and no fewer:

```text
p29-prepare-attempt-layout.stdout.log
p29-prepare-attempt-layout.stderr.log
p29-prepare-attempt-layout.exit-status.txt
p29-prepare-execution-snapshot-verification.stdout.log
p29-prepare-execution-snapshot-verification.stderr.log
p29-prepare-execution-snapshot-verification.exit-status.txt
p29-image-inspection.stderr.log
p29-image-inspection.exit-status.txt
p29-nvidia-smi.stderr.log
p29-nvidia-smi.exit-status.txt
p29-container-launch.stderr.log
p29-container-launch.exit-status.txt
p29-fresh-container-id-validation.stdout.log
p29-fresh-container-id-validation.stderr.log
p29-fresh-container-id-validation.exit-status.txt
p29-running-container-inspection.stderr.log
p29-running-container-inspection.exit-status.txt
p29-running-mountinfo.stderr.log
p29-running-mountinfo.exit-status.txt
p29-freeze-runtime.stdout.log
p29-freeze-runtime.stderr.log
p29-freeze-runtime.exit-status.txt
p29-post-freeze-attempt-layout.stdout.log
p29-post-freeze-attempt-layout.stderr.log
p29-post-freeze-attempt-layout.exit-status.txt
p29_cuda_runtime_lock.json
p29_host_attestation.json
```

Every one of the ten listed `exit-status.txt` files contains exactly the two
UTF-8 bytes `0\n`. Every one of these 27 evidence children is sealed
`root:root` mode `0444` after its producer closes it. The verifier rejects a
missing, extra, writable, nonzero, or noncanonical status before publishing
the runtime receipt.

## 5. Freeze and verify the two-file runtime review

Back on the local workstation, copy the two reviewed runtime artifacts into
the source-freeze checkout using this no-overwrite, fail-closed downloader:

```bash
set -o pipefail
p29_o_excl_download() {
  local retained_path="$1"
  local checkout_path="$2"
  test ! -e "$checkout_path"
  test ! -L "$checkout_path"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P29_SSH_TARGET" \
    sudo /usr/bin/cat -- "$retained_path" | \
    /usr/bin/python3 -I -S -c \
    'import os,shutil,sys; p=sys.argv[1]; f=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0); d=os.open(p,f,0o600); h=os.fdopen(d,"wb"); shutil.copyfileobj(sys.stdin.buffer,h); h.flush(); os.fsync(h.fileno()); h.close()' \
    "$checkout_path"
}
p29_o_excl_download \
  /secure/p29/attempt-20260906-02/evidence/p29_cuda_runtime_lock.json \
  experiments/training/p29_cuda_runtime_lock.json
p29_o_excl_download \
  /secure/p29/attempt-20260906-02/evidence/p29_host_attestation.json \
  experiments/training/p29_host_attestation.json
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes "$P29_SSH_TARGET" \
  sudo sha256sum \
  /secure/p29/attempt-20260906-02/evidence/p29_cuda_runtime_lock.json \
  /secure/p29/attempt-20260906-02/evidence/p29_host_attestation.json
shasum -a 256 \
  experiments/training/p29_cuda_runtime_lock.json \
  experiments/training/p29_host_attestation.json
```

Confirm local and retained SHA-256 values match, then commit only those two
files. The runtime-review commit must be one direct child of
`P29_SOURCE_FREEZE_COMMIT`, with exactly:

```text
A	experiments/training/p29_cuda_runtime_lock.json
A	experiments/training/p29_host_attestation.json
```

Record `P29_RUNTIME_REVIEW_COMMIT` and `P29_RUNTIME_REVIEW_TREE`. Create the
second full bundle with the same seven refs and only the P29 branch advanced:

Before the run phase, also export the reviewed runtime-only values, including
`P29_EXPECTED_RUNTIME_LOCK_SHA256`,
`P29_EXPECTED_HOST_ATTESTATION_SHA256`, and
`P29_EXPECTED_CONTAINER_ID`. The latter is the exact 64-lowercase-hex value
from the sealed `p29-container-id.txt`; it must independently match the
runtime lock and live Docker inspection. It is not inferred from a truncated
Docker name or supplied before the fresh-container-ID validation succeeds.

```bash
export P29_LOCAL_RUNTIME_BUNDLE="$P29_LOCAL_TRANSPORT/p29_runtime_review.bundle"
export P29_RUNTIME_REVIEW_BUNDLE=/secure/p29/transport-20260906-02/p29_runtime_review.bundle
export P29_RUNTIME_REVIEW_CLOSURE=/secure/p29/transport-20260906-02/p29_runtime_review_closure.git
export P29_RUNTIME_REVIEW_RECEIPT=/secure/p29/transport-20260906-02/p29_runtime_review_bundle_receipt.json
/usr/bin/git bundle create "$P29_LOCAL_RUNTIME_BUNDLE" \
  refs/heads/p29-permission-safe-bundle-localization-bridge \
  refs/heads/p28-bundle-complete-localization-bridge \
  refs/tags/p28-control-parent-permission-diagnostic \
  refs/heads/p27-cuda-deleted-mapping-localization \
  refs/tags/p27-cuda-deleted-mapping-localization-diagnostic \
  refs/tags/p26-permission-safe-acquisition-checkpoint \
  refs/tags/p26-attempt02-acquisition-source
/usr/bin/git bundle list-heads "$P29_LOCAL_RUNTIME_BUNDLE"
/usr/bin/git bundle verify "$P29_LOCAL_RUNTIME_BUNDLE"
shasum -a 256 "$P29_LOCAL_RUNTIME_BUNDLE"
wc -c "$P29_LOCAL_RUNTIME_BUNDLE"
p29_o_excl_upload \
  "$P29_LOCAL_RUNTIME_BUNDLE" "$P29_RUNTIME_REVIEW_BUNDLE" 0600
```

The runtime-review verifier repeats full closure in a second fresh empty bare
repository, proves the exact direct-child delta and retained-artifact byte
equality, advances the control checkout, executes P29 and all three nested
reconstructors only from authenticated private bundle bytes,
proves the immutable execution snapshot and exact 27-file preparation evidence,
proves no localization evidence exists, and writes a second external O_EXCL
receipt. The UID-1000 verifier deliberately makes no claim that it can inspect
the root-only ledger or deferred journal. The expected prepare-token inventory
is an operator precondition for this external receipt-creation invocation; the
sealed root orchestrator mechanically binds the token inventories around its
later verifier replay invocation. After the bundle
upload, run this in the CUDA-host `ubuntu` shell:

It fd-anchors the attempt and evidence roots with componentwise no-follow
opens before inspecting inventory, then revalidates their complete
owner/mode/ACL/device/inode/mount identity after Git, byte-comparison, and
receipt handling. Receipt replay repeats the same checks and does not mutate
either directory.

```bash
set -euo pipefail
p29_runtime_receipt_creation_logged() {
  /usr/bin/python3 -I -S - \
    "$P29_TRANSPORT_ROOT/p29-runtime-review-receipt-creation.stdout.log" \
    "$P29_TRANSPORT_ROOT/p29-runtime-review-receipt-creation.stderr.log" \
    "$P29_TRANSPORT_ROOT/p29-runtime-review-receipt-creation.exit-status.txt" \
    "$@" <<'PY'
import os
import stat
import subprocess
import sys

paths = sys.argv[1:4]
command = sys.argv[4:]
flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
flags |= getattr(os, "O_NOFOLLOW", 0)
descriptors = []
records = []
try:
    for path in paths:
        descriptor = os.open(path, flags, 0o600)
        os.fchmod(descriptor, 0o600)
        descriptors.append(descriptor)
        records.append(os.fstat(descriptor))
    environment = {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }
    try:
        result = subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=descriptors[0],
            stderr=descriptors[1],
            env=environment,
            check=False,
        )
        result_code = result.returncode
    except OSError as exc:
        os.write(descriptors[1], f"receipt wrapper failed: {exc}\n".encode())
        result_code = 125
    for path, record in zip(paths, records, strict=True):
        current = os.lstat(path)
        if (
            not stat.S_ISREG(current.st_mode)
            or stat.S_IMODE(current.st_mode) != 0o600
            or (current.st_dev, current.st_ino) != (record.st_dev, record.st_ino)
        ):
            result_code = 126
    os.write(descriptors[2], f"{result_code}\n".encode("ascii"))
    for descriptor in descriptors:
        os.fsync(descriptor)
finally:
    for descriptor in descriptors:
        os.close(descriptor)
raise SystemExit(result_code)
PY
}

p29_runtime_receipt_creation_logged \
  /usr/bin/python3 -I -S "$P29_BOOTSTRAP_VERIFIER" \
  --phase runtime-review \
  --transport-root "$P29_TRANSPORT_ROOT" \
  --bundle "$P29_RUNTIME_REVIEW_BUNDLE" \
  --expected-bundle-sha256 "$P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" \
  --closure-repository "$P29_RUNTIME_REVIEW_CLOSURE" \
  --control-checkout "$P29_CONTROL_REPO" \
  --attempt-root "$P29_ATTEMPT_ROOT" \
  --execution-root "$P29_EXECUTION_ROOT" \
  --expected-p29-commit "$P29_RUNTIME_REVIEW_COMMIT" \
  --expected-p29-tree "$P29_RUNTIME_REVIEW_TREE" \
  --expected-source-commit "$P29_SOURCE_FREEZE_COMMIT" \
  --expected-source-tree "$P29_SOURCE_FREEZE_TREE" \
  --bootstrap-verifier "$P29_BOOTSTRAP_VERIFIER" \
  --expected-verifier-sha256 "$P29_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --receipt-output "$P29_RUNTIME_REVIEW_RECEIPT"
```

When wrapper setup and finalization succeed, these three transcripts retain
either verifier success or verifier failure under the transport root; they are
never copied into `evidence`. A wrapper O_EXCL, write, fsync, or close failure
may leave a partial transport transcript set and is terminal. Thus the later
runtime-receipt replay still observes the exact frozen 27-file evidence
inventory. A nonzero or incomplete logged invocation may not be rerun.

The other six ref objects must remain unchanged between receipts. Never commit
either receipt into the history it authenticates.

## 6. Run the sole unchanged P27 localization

After independent runtime-receipt review, extend the exact clean root handoff
with the eight runtime-only values, then run exactly once:

```bash
p29_verify_root_environment_handoff() {
  local expected_count="$1"
  shift
  test "$#" -eq "$expected_count"
  sudo /usr/bin/env -i \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    "$@" \
    /usr/bin/python3 -I -S -c \
    'import os,sys
count=int(sys.argv[1]); pairs=sys.argv[2:]
if len(pairs) != count: raise SystemExit("P29 handoff count differs")
expected={}
for pair in pairs:
    name, separator, value = pair.partition("=")
    if separator != "=" or not name.startswith("P29_") or not value or name in expected:
        raise SystemExit("P29 handoff assignment differs")
    expected[name]=value
actual={name:value for name,value in os.environ.items() if name.startswith("P29_")}
if actual != expected or os.geteuid() != 0 or os.getegid() != 0:
    raise SystemExit("P29 root handoff differs")' \
    "$expected_count" "$@"
}
p29_localization_root_environment=(
  "P29_ATTEMPT_ID=$P29_ATTEMPT_ID"
  "P29_GPU_UUID=$P29_GPU_UUID"
  "P29_AUTHORITY_REPO=$P29_AUTHORITY_REPO"
  "P29_NANOGPT_HOST=$P29_NANOGPT_HOST"
  "P29_MUON_HOST=$P29_MUON_HOST"
  "P29_DATA_HOST=$P29_DATA_HOST"
  "P29_SOURCE_FREEZE_COMMIT=$P29_SOURCE_FREEZE_COMMIT"
  "P29_SOURCE_FREEZE_TREE=$P29_SOURCE_FREEZE_TREE"
  "P29_EXPECTED_SOURCE_BUNDLE_SHA256=$P29_EXPECTED_SOURCE_BUNDLE_SHA256"
  "P29_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT=$P29_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT"
  "P29_EXPECTED_SOURCE_RECEIPT_SHA256=$P29_EXPECTED_SOURCE_RECEIPT_SHA256"
  "P29_EXPECTED_CONTRACT_SHA256=$P29_EXPECTED_CONTRACT_SHA256"
  "P29_EXPECTED_ORCHESTRATOR_SHA256=$P29_EXPECTED_ORCHESTRATOR_SHA256"
  "P29_EXPECTED_RECONSTRUCTOR_SHA256=$P29_EXPECTED_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_BUNDLE_VERIFIER_SHA256=$P29_EXPECTED_BUNDLE_VERIFIER_SHA256"
  "P29_EXPECTED_P28_CONTRACT_SHA256=$P29_EXPECTED_P28_CONTRACT_SHA256"
  "P29_EXPECTED_P28_RECONSTRUCTOR_SHA256=$P29_EXPECTED_P28_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_P28_OUTCOME_SHA256=$P29_EXPECTED_P28_OUTCOME_SHA256"
  "P29_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256=$P29_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_P27_CONTRACT_SHA256=$P29_EXPECTED_P27_CONTRACT_SHA256"
  "P29_EXPECTED_P27_RECONSTRUCTOR_SHA256=$P29_EXPECTED_P27_RECONSTRUCTOR_SHA256"
  "P29_EXPECTED_P27_LOCALIZER_SHA256=$P29_EXPECTED_P27_LOCALIZER_SHA256"
  "P29_EXPECTED_INGESTER_SHA256=$P29_EXPECTED_INGESTER_SHA256"
  "P29_EXPECTED_P23_CORE_SHA256=$P29_EXPECTED_P23_CORE_SHA256"
  "P29_EXPECTED_P27_SANITIZER_SHA256=$P29_EXPECTED_P27_SANITIZER_SHA256"
  "P29_RUNTIME_REVIEW_COMMIT=$P29_RUNTIME_REVIEW_COMMIT"
  "P29_RUNTIME_REVIEW_TREE=$P29_RUNTIME_REVIEW_TREE"
  "P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256=$P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256"
  "P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT=$P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT"
  "P29_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256=$P29_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256"
  "P29_EXPECTED_RUNTIME_LOCK_SHA256=$P29_EXPECTED_RUNTIME_LOCK_SHA256"
  "P29_EXPECTED_HOST_ATTESTATION_SHA256=$P29_EXPECTED_HOST_ATTESTATION_SHA256"
  "P29_EXPECTED_CONTAINER_ID=$P29_EXPECTED_CONTAINER_ID"
)
p29_verify_root_environment_handoff 33 "${p29_localization_root_environment[@]}"
sudo /usr/bin/env -i \
  PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  "${p29_localization_root_environment[@]}" \
  /bin/bash -p \
  /var/lib/optimizationml-p29-20260906-02/run_p29_permission_safe_bundle_localization_bridge.sh \
  run-localization
```

This passes exactly all 33 contract variables and no ambient user environment.
The script revalidates them after its one-shot token admission and before the
localization body performs runtime receipt replay or any later phase work.

Phase dispatch first enters one Python transaction that uses the supplied
expected sealed-orchestrator digest, validates that transaction's root-ledger,
deferred-journal, and sealed-source authorities, and O_EXCL-creates the
root-owned mode-`0400`
ledger `/var/lib/optimizationml-p29-20260906-02/run-localization.invoked`
before general environment validation or any evidence or later-marker
predicate. A missing or malformed supplied digest can fail before token
creation. The transaction returns success only to the process that created
and postvalidated the token. Its exact bytes are `run-localization\n`. At this
point the ledger already contains only the sealed orchestrator, deferred
journal, and exact `prepare-runtime.invoked` token. A premature localization
dispatch after that prepare token exists, or a repeated localization dispatch,
consumes P29. A pre-prepare localization dispatch cannot satisfy the required
ledger pre-inventory, creates no new ledger state, and is retryable after the
sole successful prepare admission. This unlink-resistant root-owned token is
the irreversible invocation record and remains after every success or failure. At this boundary
`localization.authorized` must still be absent. The later evidence marker
records a reached boundary only; it cannot authorize a retry. Failure before
evidence logging is therefore still terminal and visible outside the
process-writable transport directory. A replay that finds an existing token
must not create or replace the original invocation's terminal status. A
partial internal token transaction leaves token-without-status as an explicit,
permanently non-rerunnable incomplete state.

Only after that token transaction succeeds does the localization body run in a
subshell. After this creator-owned subshell returns for any ordinary success
or failure, the dispatcher attempts one O_EXCL transaction for
`/var/lib/optimizationml-p29-20260906-02/run-localization.exit-status.txt` as
`root:root` mode `0400`. If the transaction succeeds, it contains the
canonical decimal exit status and one newline. Besides the sealed orchestrator
and deferred journal, that successful final root-ledger inventory is exactly `prepare-runtime.invoked`,
`run-localization.invoked`, and `run-localization.exit-status.txt` when the
attempt fails before authorization, and exactly those three files plus
`localization.authorized` after authorization. Failure to retain that terminal
status is itself terminal and may leave the file absent or created but not
fully validated; neither that incomplete state, a missing status after process
destruction, nor any nonzero validated status authorizes another invocation.

Runtime-receipt replay is the first operation in the localization body; no
ordinary evidence logger or entry-preflight transcript may precede its exact
27-file inventory check. Inside that deferred producer, all 33 runtime
environment values are validated before transport validation or verifier
dispatch. It uses a specialized deferred-publication logger. It
creates root-owned no-overwrite stdout and stderr transcripts beneath
`/var/lib/optimizationml-p29-20260906-02/deferred` as
`.deferred-p29-pre-marker-runtime-receipt-replay.stdout.log`,
`.deferred-p29-pre-marker-runtime-receipt-replay.stderr.log` before invoking
the verifier while `evidence` still has exactly the frozen 27-file
pre-localization inventory. After the verifier exits—on success or failure—it
O_EXCL-creates
`.deferred-p29-pre-marker-runtime-receipt-replay.exit-status.txt`; only then
does the orchestrator reauthenticate the reviewed attempt/evidence identity.
Only after those steps may it stably reread each nonsymlink journal file and
publish
byte-identical `p29-pre-marker-runtime-receipt-replay.stdout.log`,
`p29-pre-marker-runtime-receipt-replay.stderr.log`, and
`p29-pre-marker-runtime-receipt-replay.exit-status.txt` into `evidence` with
`O_EXCL|O_NOFOLLOW`, then seal them `root:root` mode `0444`, before any other
pre-marker log. The original journal copies remain. An ordinary verifier failure with intact identity is retained
in both locations, creates no marker, and terminates the attempt.

Only after those replay transcripts have been published does the retained
`p29-localization-entry-preflight` run. That preflight requires the deferred
journal's exact after-runtime stage; the next retained boundary is
`p29-pre-marker-receipt-namespace-binding`.

The publication operation best-effort creates and seals the preregistered
journal triple `.publication.stdout.log`, `.publication.stderr.log`, and
`.publication.exit-status.txt` for this prefix. A complete triplet is
guaranteed only when artifact finalization succeeds; otherwise the logger
attempts every remaining close/seal/status action and returns terminal status
125. Exact success requires empty stdout and stderr and exact status bytes
`0\n`. All successfully finalized journal files are root-owned,
O_EXCL/no-follow, and sealed mode `0400`. A publication failure also
best-effort retains
`.deferred-p29-pre-marker-runtime-receipt-replay.publication-failure.exit-status.txt`;
that sentinel is forbidden on success.

Thus the successful exact deferred-journal inventories have 0 files before
prepare admission, 3 after admission, 9 after attempt-layout publication, and
15 after runtime-receipt replay publication. Extra, missing, writable, or
misowned journal children are terminal.

If identity reauthentication or evidence publication itself fails, the host
must not write through the suspect evidence path. With the root journal still
intact, it best-effort O_EXCL-creates the root-journal status
`.deferred-p29-pre-marker-runtime-receipt-replay.publication-failure.exit-status.txt`.
Failure to finalize that status returns terminal 125. Any successfully sealed
original journal transcripts remain and no marker is created.
The exact-27 receipt gate is never weakened to accommodate observer output.
The verifier holds its own directory descriptors during the replay process;
the host compares fresh no-follow identity records before and after that
external process and does not claim its descriptors span the process. Evidence
publication after either success or failure is permitted only after the fresh
identity equals the reviewed runtime-attempt binding; whenever publication
succeeds, its three files byte-match the retained journal originals.

The host does not get a blanket identity or full-manifest check merely because
an operation is logged. The complete execution snapshot is verified at the
retained `p29-prepare-execution-snapshot-verification` boundary and again by
the bundle verifier during runtime-receipt creation and the retained
`p29-pre-marker-runtime-receipt-replay`. The later
`p29-pre-marker-snapshot-layout` check authenticates the guarded directories,
top-level execution inventory, and manifest authority; it does not replay the
full manifest.

Attempt/evidence identity is compared with the reviewed receipt at the
explicit `*-receipt-namespace-binding` labels after receipt replay, ingestion,
authorization, localization, sanitization, and final localization, and the
fresh layout is checked by `p29-localization-entry-preflight` and
`p29-pre-marker-snapshot-layout`. Each validator reopens its inputs
componentwise with no-follow semantics. This is deliberately neither a claim
that one host descriptor stays open across the entire marker-to-localizer
process span nor a claim that every logger implicitly repeats those checks.
New scientific artifacts are sealed and authenticated at their declared
producer or binding boundaries; historical evidence bytes are not reread
after every unrelated log. A failed explicit identity, authority, ACL, device,
mount, manifest, or artifact check stops the attempt and preserves its partial
state.

Four explicit ledger/preflight checks are retained in this exact chronological
order: `p29-localization-entry-preflight`,
`p29-post-authorization-ledger-validation`,
`p29-post-localizer-ledger-validation`, and
`p29-post-sanitizer-ledger-validation`. Each label uses an O_EXCL/no-follow
stdout, stderr, and exit-status triplet. When finalization succeeds, the full
triplet is sealed `root:root` mode `0444` before the next boundary; on
finalization failure, the root host best-effort seals every already-closed
member and returns 125.

All retained logger classes (`run_logged`, `capture_new`,
`refresh_bound_file`, and deferred external/publication status creation) use
best-effort finalization: if any required close/seal/status operation fails,
they still attempt to seal every other already-closed artifact and return the
terminal code 125. The original producer status is preserved only when every
required artifact finalization succeeds. The evidence-inventory sealer walks
every child and collects all child failures before returning terminal failure.
The `freeze-runtime` logged call is run with errexit temporarily disabled, its
status is captured, errexit is restored, and that captured status is then
checked so its failure evidence is not lost to an early shell exit.

The orchestrator must replay the runtime receipt while the invocation token is
present and the authorization token absent; authenticate the live container,
the immutable execution manifest at the explicit verifier boundaries, and the
source/runtime/layout identities at their declared validators. The
bundle verifier is the sole host reconstruction authority: its authenticated
private payload must report P29 plus nested P28 contract 12/12, terminal P28
outcome 10/10, and P27 23/23. No mutable control-checkout reconstructor is
executed by path. After every other retained pre-marker validator passes, the
host O_EXCL-creates root-owned mode-`0400`
`/var/lib/optimizationml-p29-20260906-02/localization.authorized` with exact
bytes `localization-authorized\n`. This authorization is created exactly once,
only after all reviewed gates pass. The container-side ingester, localizer,
P23 core, and sanitizer execute from the read-only
`/private/tmp/optimizationml-p22-data/.p29-tools/` adjunct. Before dispatch,
the host's explicit ledger validator requires all three root-owned tokens to
remain present and exact; the containerized tools do not read that ledger.
They ingest the already authenticated P26 failure, run the no-training
localizer once, and retain no-overwrite
stdout, stderr, native, sanitized, and manifest evidence. Never invoke an
inner command directly and never retry.

## 7. Freeze and route the result mechanically

Record a compact machine-readable P29 outcome and every retained hash. Use the
contract's precedence order:

- layout, bundle, receipt, or reconstruction failure: stop and retain all
  partial state;
- sanitizer or localizer failure: retain evidence and stop;
- executable or writable deleted mapping: design an image/loader correction;
- unclassifiable or identity-unstable mapping: retain probes and keep strict
  rejection;
- conclusively read-only nonexecutable deleted mappings: preregister a narrow
  provenance/device policy;
- no deleted mapping: record only nonreproduction.

No result permits cleanup or a favorable rerun under this P29 identifier, and
no P29 route authorizes scientific acquisition `20260906-03`.
