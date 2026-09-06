# P30 umask-bound control seal runbook

## Evidence boundary

P30 corrects the reviewed-orchestrator source-mode defect recorded by terminal
P29. The verifier's umask `077` creates a tracked `100644` orchestrator as a
physical uid/gid-1000 mode-`0600` file; P30 binds that exact authority into the
receipt and root seal without chmod or chown. It then runs the unchanged P27
no-training deleted-mapping localizer once. P29 attempt `20260906-02` and
everything beneath `/secure/p29` are retained terminal evidence. They must not
be cleaned, changed, mounted into P30, or reused.

P30 permits only P27's four diagnostic stages: `pre_cuda`, `post_cuda_init`,
`post_model_move`, and `post_optimizer`. It forbids data loading, forward,
backward, candidate evaluation, optimizer steps, parameter updates, and
training. It does not authorize prospective scientific acquisition
`20260906-04`.

The immutable tags through
`p30-umask-bound-control-seal-source-freeze-v5` retain the earlier source
freezes. This pre-execution v6 revision keeps v5's corrected
receipt-review-seal order and closes only pre-launch engineering defects found
before any P30 host state was created. In addition to v5's full 37-variable
localization handoff and verifier-umask-bound runtime-file authority, v6 pins
the local Python and host key, authenticates every materialized client tool,
reserves and finalizes client log triplets before mutation, binds downloaded
receipts to their one-shot creation summaries, mechanically checks both Git
bundles, and rehearses the real UID-1000-to-root handoff. Removing the remote
receipt-reviewer upload/hash operations yields the exact 19 logged boundary
labels `00` through `18`; both receipt reviews instead execute locally from
source-frozen mode-`0400` bytes. It does not alter P30's four diagnostic
stages, mapping classification, terminal routing, or no-training boundary.
Every historical tag and P23--P29 terminal record remains unchanged.

## 0. Rehearse engineering handoffs before the revised freeze

This section is deliberately split by the source-freeze dependency. First
perform only **0A**. Then perform Section 1 through the source-commit, tree,
and control-digest derivation. Only then return to **0B**, whose first command
mechanically rejects an unset source commit. After 0B's read-only host
preflight passes, resume Section 1 at local bundle creation and continue
top-to-bottom through Sections 2--7. Every client command belongs to one
uninterrupted local Bash session on the macOS workstation; none of these
client blocks is pasted at the Ubuntu prompt.

### 0A. Repeatable disposable handoff rehearsal

Follow
[`ENGINEERING_REHEARSAL_POLICY.md`](ENGINEERING_REHEARSAL_POLICY.md). Run the
dedicated disposable Linux rehearsal before making the revised P30 source
commit authoritative. The rehearsal crosses the real numeric UID/GID
`1000:1000` verifier to `0:0` seal boundary, executes the unmodified bundle
verifier and all six reconstructions, reviews and replays the receipt, checks
the root-owned mode-`0555` sealed bytes, then rehearses the runtime-review
checkout and its full root environment handoff. It proves that both new
tracked-`100644` runtime-control files materialize as uid/gid-1000 mode-`0600`
regular single-link files under verifier umask `077`. It may mirror P30's
absolute path strings on the disposable runner, but it creates no real attempt
root and cannot consume the CUDA host's frozen identity. It does not initialize
CUDA or produce a mapping or optimizer observation. Its failures are
repeatable engineering failures, not new research phases.

Retain the rehearsal's native stdout, stderr, and exit status only as
engineering artifacts. They are not P30 scientific evidence and must not be
placed beneath `/secure/p30`, the P30 attempt evidence directory, or another
frozen exact-content directory.

Before any real-host state creation, perform and retain a read-only freshness
check for all three fixed authorities:

- `/secure/p30`;
- `/var/lib/optimizationml-p30-20260906-03`; and
- container name `p30-localization-20260906-03`.

Any existing path or container consumes that identity for this protocol. Do
not clean or reuse it; revise every hard-coded identity and path before another
rehearsal and source freeze.

### 0B. Post-freeze client tooling and actual-host read-only preflight

Do not enter this subsection until Section 1 has produced and independently
reviewed the direct-child source-freeze commit and derived all source bindings.
Create an absolute external client-log directory owned by the workstation
operator and mode `0700`. After the revised source commit in Section 1 exists,
materialize the client logger, preflight, receipt reviewer, local bundle
verifier, authenticated-transfer helper, and remote bundle verifier
bytes from that exact commit, not from the mutable working tree. Each
destination is a new external
single-link mode-`0400` file created with `O_EXCL`; a stable no-follow read must
match the corresponding SHA-256 embedded in the frozen contract before use:

```bash
set -euo pipefail
: "${P30_SOURCE_FREEZE_COMMIT:?complete Section 1 source freeze before Section 0B}"
: "${P30_SOURCE_FREEZE_TREE:?complete Section 1 source freeze before Section 0B}"
export P30_LOCAL_REPOSITORY=/Users/harry/Desktop/temp/OptimizationML
export P30_CLIENT_SESSION_ROOT="$({
  /usr/bin/mktemp -d /private/tmp/p30-v6-client-session.XXXXXX
})"
export P30_CLIENT_UID="$({ /usr/bin/id -u; })"
export P30_CLIENT_GID="$({ /usr/bin/id -g; })"
/usr/bin/chgrp "$P30_CLIENT_GID" "$P30_CLIENT_SESSION_ROOT"
/bin/chmod 0700 "$P30_CLIENT_SESSION_ROOT"
test "$({
  /usr/bin/stat -f '%u:%g:%Lp' "$P30_CLIENT_SESSION_ROOT"
})" = "$P30_CLIENT_UID:$P30_CLIENT_GID:700"
export P30_CLIENT_LOG_ROOT="$P30_CLIENT_SESSION_ROOT/logs"
export P30_FROZEN_CLIENT_TOOL_ROOT="$P30_CLIENT_SESSION_ROOT/tools"
export P30_CLIENT_TRANSFER_ROOT="$P30_CLIENT_SESSION_ROOT/transfers"
export P30_CLIENT_PYTHON=/opt/homebrew/Cellar/python@3.14/3.14.3_1/Frameworks/Python.framework/Versions/3.14/bin/python3.14
export P30_EXPECTED_CLIENT_PYTHON_SHA256=f13707ff725eb3675d57e0f04527a4dada0511cebfb7cbd80f5cab72733a2a9d
export P30_FROZEN_CLIENT_SOURCE_ROOT="$P30_FROZEN_CLIENT_TOOL_ROOT/source"
export P30_FROZEN_CLIENT_LOGGER="$P30_FROZEN_CLIENT_SOURCE_ROOT/run_logged_client_command.py"
export P30_FROZEN_HOST_PREFLIGHT="$P30_FROZEN_CLIENT_SOURCE_ROOT/run_p30_actual_host_prerequisite_preflight.py"
export P30_FROZEN_RECEIPT_REVIEWER="$P30_FROZEN_CLIENT_SOURCE_ROOT/review_p30_control_bundle_receipt.py"
export P30_FROZEN_LOCAL_BUNDLE_VERIFIER="$P30_FROZEN_CLIENT_SOURCE_ROOT/verify_p30_local_bundle.py"
export P30_FROZEN_BUNDLE_VERIFIER="$P30_FROZEN_CLIENT_SOURCE_ROOT/verify_p30_control_bundle.py"
export P30_FROZEN_AUTHENTICATED_TRANSFER="$P30_FROZEN_CLIENT_SOURCE_ROOT/run_p30_authenticated_client_transfer.py"
export P30_EXPECTED_CLIENT_LOGGER_SOURCE_SHA256=e6255126d1f0f36101acab15db849f4b914eb4c97faf5713d6de943048efc8da
export P30_EXPECTED_HOST_PREFLIGHT_SOURCE_SHA256=15fa3ad26b6078caba40eed32cf51dd61ae53c8aa6446ceb6dcf764bba6b93dc
export P30_EXPECTED_RECEIPT_REVIEWER_SOURCE_SHA256=01e64c79234eb2cf6665f3aae781b6b8f7740e5b536fa3ede1e10063b85301a2
export P30_EXPECTED_LOCAL_BUNDLE_VERIFIER_SOURCE_SHA256=eb81e46024517e27667a282167bbbcb866672fdca7c751bd5abaef01d4a7d7dc
export P30_EXPECTED_BUNDLE_VERIFIER_SHA256=9f3c13c37bed37833528c1e3d9f5b833cbbe17263e1a7f3cb3488ea48f59492c
export P30_EXPECTED_AUTHENTICATED_TRANSFER_SOURCE_SHA256=0a21d8a525886c02c66c7ac6c13c2c2977ee110def29054be6efed340407b470
case "$P30_CLIENT_PYTHON" in /*) ;; *) exit 1 ;; esac
test -f "$P30_CLIENT_PYTHON"
test ! -L "$P30_CLIENT_PYTHON"
test -x "$P30_CLIENT_PYTHON"
test "${#P30_EXPECTED_CLIENT_PYTHON_SHA256}" -eq 64
case "$P30_EXPECTED_CLIENT_PYTHON_SHA256" in *[!0-9a-f]*) exit 1 ;; esac
test "$({
  /usr/bin/shasum -a 256 "$P30_CLIENT_PYTHON" | /usr/bin/awk '{print $1}'
})" = "$P30_EXPECTED_CLIENT_PYTHON_SHA256"
"$P30_CLIENT_PYTHON" -I -S -c \
  'import os,stat,sys
p=sys.argv[1]
s=os.lstat(p)
assert os.path.isabs(p) and os.path.realpath(p)==p
assert os.path.realpath(sys.executable)==p
assert stat.S_ISREG(s.st_mode) and s.st_nlink==1
assert stat.S_IMODE(s.st_mode)&0o022==0
assert sys.version_info[:3] == (3,14,3)' \
  "$P30_CLIENT_PYTHON"
umask 077
/bin/mkdir "$P30_CLIENT_LOG_ROOT"
/bin/chmod 0700 "$P30_CLIENT_LOG_ROOT"
/bin/mkdir "$P30_FROZEN_CLIENT_TOOL_ROOT"
/bin/chmod 0700 "$P30_FROZEN_CLIENT_TOOL_ROOT"
/bin/mkdir "$P30_FROZEN_CLIENT_SOURCE_ROOT"
/bin/chmod 0700 "$P30_FROZEN_CLIENT_SOURCE_ROOT"
/bin/mkdir "$P30_CLIENT_TRANSFER_ROOT"
/bin/chmod 0700 "$P30_CLIENT_TRANSFER_ROOT"

p30_materialize_frozen_client_source() {
  local relative_path="$1"
  local destination="$2"
  local expected_sha256="$3"
  set -o pipefail
  /usr/bin/env -i \
    HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    GIT_CONFIG_GLOBAL=/dev/null GIT_CONFIG_NOSYSTEM=1 \
    GIT_CONFIG_SYSTEM=/dev/null GIT_NO_REPLACE_OBJECTS=1 \
    GIT_OPTIONAL_LOCKS=0 \
    /usr/bin/git --no-replace-objects \
      -c "safe.directory=$P30_LOCAL_REPOSITORY" \
      -C "$P30_LOCAL_REPOSITORY" cat-file blob \
      "$P30_SOURCE_FREEZE_COMMIT:$relative_path" | \
    "$P30_CLIENT_PYTHON" -I -S -c \
      'import os,shutil,sys
p=sys.argv[1]
f=os.O_WRONLY|os.O_CREAT|os.O_EXCL|getattr(os,"O_CLOEXEC",0)|getattr(os,"O_NOFOLLOW",0)
d=os.open(p,f,0o400)
with os.fdopen(d,"wb") as h:
    shutil.copyfileobj(sys.stdin.buffer,h)
    h.flush(); os.fsync(h.fileno())' \
      "$destination"
  "$P30_CLIENT_PYTHON" -I -S - "$destination" "$expected_sha256" <<'PY'
import hashlib, os, stat, sys
p, expected = sys.argv[1:]
before = os.lstat(p)
flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
fd = os.open(p, flags)
try:
    opened = os.fstat(fd)
    raw = bytearray()
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        raw.extend(chunk)
    after = os.fstat(fd)
finally:
    os.close(fd)
by_name = os.stat(p, follow_symlinks=False)
stable = lambda s: (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
if not (stable(before) == stable(opened) == stable(after) == stable(by_name)):
    raise SystemExit("frozen client source changed during stable read")
if not stat.S_ISREG(after.st_mode) or after.st_uid != os.getuid() or after.st_nlink != 1 or stat.S_IMODE(after.st_mode) != 0o400:
    raise SystemExit("frozen client source authority differs")
if hashlib.sha256(raw).hexdigest() != expected:
    raise SystemExit("frozen client source SHA-256 differs")
PY
}

p30_materialize_frozen_client_source \
  scripts/run_logged_client_command.py \
  "$P30_FROZEN_CLIENT_LOGGER" \
  "$P30_EXPECTED_CLIENT_LOGGER_SOURCE_SHA256"
p30_materialize_frozen_client_source \
  scripts/run_p30_actual_host_prerequisite_preflight.py \
  "$P30_FROZEN_HOST_PREFLIGHT" \
  "$P30_EXPECTED_HOST_PREFLIGHT_SOURCE_SHA256"
p30_materialize_frozen_client_source \
  scripts/review_p30_control_bundle_receipt.py \
  "$P30_FROZEN_RECEIPT_REVIEWER" \
  "$P30_EXPECTED_RECEIPT_REVIEWER_SOURCE_SHA256"
p30_materialize_frozen_client_source \
  scripts/verify_p30_local_bundle.py \
  "$P30_FROZEN_LOCAL_BUNDLE_VERIFIER" \
  "$P30_EXPECTED_LOCAL_BUNDLE_VERIFIER_SOURCE_SHA256"
p30_materialize_frozen_client_source \
  scripts/verify_p30_control_bundle.py \
  "$P30_FROZEN_BUNDLE_VERIFIER" \
  "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256"
p30_materialize_frozen_client_source \
  scripts/run_p30_authenticated_client_transfer.py \
  "$P30_FROZEN_AUTHENTICATED_TRANSFER" \
  "$P30_EXPECTED_AUTHENTICATED_TRANSFER_SOURCE_SHA256"

p30_client_forbidden_roots=(
  "$P30_LOCAL_REPOSITORY"
  "$P30_FROZEN_CLIENT_TOOL_ROOT"
  "$P30_CLIENT_TRANSFER_ROOT"
)
p30_run_logged_client_command() {
  test "$#" -ge 2
  local label="$1"
  local forbidden_root
  local -a forbidden_arguments=()
  shift
  for forbidden_root in "${p30_client_forbidden_roots[@]}"; do
    forbidden_arguments+=(--forbidden-root "$forbidden_root")
  done
  "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_CLIENT_LOGGER" \
    --log-root "$P30_CLIENT_LOG_ROOT" \
    "${forbidden_arguments[@]}" \
    --expected-logger-source-sha256 \
      "$P30_EXPECTED_CLIENT_LOGGER_SOURCE_SHA256" \
    --label "$label" -- "$@"
}
```

Creation or validation failure leaves the external engineering record in
place and requires a new external tool root before retry. The tool root and
client-log root must be disjoint siblings: neither may equal or contain the
other. None of these files is remote scientific evidence. Run each earliest client-side provisioning,
source verifier, receipt-review, and reviewed-seal command through:

```bash
set -euo pipefail
p30_run_logged_client_command \
  <unique-lowercase-label> <command> <arguments...>
```

The logger mirrors output while retaining no-overwrite `.stdout.log`,
`.stderr.log`, and `.exit-status.txt` files. It deliberately does not retain
the command line, which may contain operator credentials. The log root must be
outside this repository and outside every frozen remote evidence directory.

Pin the single operator-confirmed ED25519 key in an absolute caller-owned
single-link known-hosts file. The operator-confirmed authority is the host-side
fingerprint obtained out of band, not the unauthenticated output of
`ssh-keyscan`. If an already existing operator-provided file is used, do not
rewrite it: the frozen preflight below stably authenticates its canonical path,
owner, mode, link count, exact one-entry inventory, key type, host name, and
fingerprint. If the file is absent, create it once with the following
engineering-only acquisition. The client logger retains the native keyscan
streams and exit status; the isolated writer accepts exactly one ED25519 key,
requires its fingerprint to equal the out-of-band value, normalizes the host
field to the exact default-port IP, and creates the final file with
`O_EXCL|O_NOFOLLOW`. A failure leaves all engineering artifacts in place and
requires a new log root and destination rather than cleanup or overwrite. The
unnumbered `engineering-host-keyscan` label is deliberately outside P30's
scientific boundary-label sequence:

```bash
export P30_PINNED_KNOWN_HOSTS="$P30_CLIENT_SESSION_ROOT/p30_known_hosts"
export P30_EXPECTED_HOST_KEY_FINGERPRINT=SHA256:t7nLeL/gt2qBu7e+I0yb8xICPtX8WcP8RMn7yEkXrJU
test ! -e "$P30_PINNED_KNOWN_HOSTS"
test ! -L "$P30_PINNED_KNOWN_HOSTS"
p30_run_logged_client_command engineering-host-keyscan \
  /usr/bin/ssh-keyscan -T 10 -p 22 -t ed25519 129.146.177.214
"$P30_CLIENT_PYTHON" -I -S - \
  "$P30_CLIENT_LOG_ROOT/engineering-host-keyscan.stdout.log" \
  "$P30_CLIENT_LOG_ROOT/engineering-host-keyscan.exit-status.txt" \
  "$P30_PINNED_KNOWN_HOSTS" \
  "$P30_EXPECTED_HOST_KEY_FINGERPRINT" <<'P30_PIN_CONFIRMED_HOST_KEY'
import base64
import binascii
import hashlib
import os
import pathlib
import stat
import sys

scan_path = pathlib.Path(sys.argv[1])
status_path = pathlib.Path(sys.argv[2])
destination = pathlib.Path(sys.argv[3])
expected_fingerprint = sys.argv[4]
caller = (os.getuid(), os.getgid())


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


def directory_authority(value):
    return (
        value.st_dev,
        value.st_ino,
        value.st_mode,
        value.st_uid,
        value.st_gid,
    )


def stable_read(path, expected_mode):
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise SystemExit("host-key engineering artifact path is not absolute")
    parent = path.parent
    if parent.resolve(strict=True) != parent:
        raise SystemExit("host-key engineering artifact parent is not canonical")
    parent_initial = parent.lstat()
    if (
        not stat.S_ISDIR(parent_initial.st_mode)
        or (parent_initial.st_uid, parent_initial.st_gid) != caller
        or stat.S_IMODE(parent_initial.st_mode) & 0o077
    ):
        raise SystemExit("host-key engineering artifact parent authority differs")
    parent_fd = os.open(
        parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    )
    descriptor = None
    try:
        parent_before = os.fstat(parent_fd)
        descriptor = os.open(
            path.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent_fd
        )
        before = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        parent_by_name = parent.lstat()
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
    if not (
        fields(parent_initial)
        == fields(parent_before)
        == fields(parent_after)
        == fields(parent_by_name)
    ):
        raise SystemExit("host-key engineering artifact parent changed")
    if not (fields(before) == fields(after) == fields(by_name)):
        raise SystemExit("host-key engineering artifact changed during read")
    if (
        not stat.S_ISREG(after.st_mode)
        or (after.st_uid, after.st_gid) != caller
        or after.st_nlink != 1
        or stat.S_IMODE(after.st_mode) != expected_mode
    ):
        raise SystemExit("host-key engineering artifact authority differs")
    return b"".join(chunks)


if stable_read(status_path, 0o600) != b"0\n":
    raise SystemExit("ssh-keyscan did not exit successfully")
try:
    text = stable_read(scan_path, 0o600).decode("ascii")
except UnicodeDecodeError as exc:
    raise SystemExit("ssh-keyscan output is not ASCII") from exc
lines = [line for line in text.splitlines() if line and not line.startswith("#")]
if len(lines) != 1:
    raise SystemExit("ssh-keyscan did not return exactly one host key")
parts = lines[0].split()
if (
    len(parts) != 3
    or parts[0] not in {"129.146.177.214", "[129.146.177.214]:22"}
    or parts[1] != "ssh-ed25519"
):
    raise SystemExit("ssh-keyscan host or key type differs")
try:
    key_blob = base64.b64decode(parts[2], validate=True)
except (binascii.Error, ValueError) as exc:
    raise SystemExit("ssh-keyscan key blob is malformed") from exc
fingerprint = (
    "SHA256:"
    + base64.b64encode(hashlib.sha256(key_blob).digest()).decode("ascii").rstrip("=")
)
if fingerprint != expected_fingerprint:
    raise SystemExit("ssh-keyscan key does not match the out-of-band fingerprint")
if not destination.is_absolute() or destination.name in {"", ".", ".."}:
    raise SystemExit("pinned known-hosts destination is not absolute")
parent = destination.parent
if parent.resolve(strict=True) != parent:
    raise SystemExit("pinned known-hosts parent is not canonical")
parent_initial = parent.lstat()
if (
    not stat.S_ISDIR(parent_initial.st_mode)
    or (parent_initial.st_uid, parent_initial.st_gid) != caller
    or stat.S_IMODE(parent_initial.st_mode) & 0o022
):
    raise SystemExit("pinned known-hosts parent authority differs")
parent_fd = os.open(
    parent, os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
)
destination_fd = None
try:
    parent_before = os.fstat(parent_fd)
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    destination_fd = os.open(destination.name, flags, 0o400, dir_fd=parent_fd)
    canonical = (
        "129.146.177.214 ssh-ed25519 "
        + base64.b64encode(key_blob).decode("ascii")
        + "\n"
    ).encode("ascii")
    view = memoryview(canonical)
    while view:
        written = os.write(destination_fd, view)
        if written <= 0:
            raise SystemExit("pinned known-hosts write made no progress")
        view = view[written:]
    os.fchmod(destination_fd, 0o400)
    os.fsync(destination_fd)
    created = os.fstat(destination_fd)
    stored = os.pread(destination_fd, len(canonical) + 1, 0)
    by_name = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
    os.fsync(parent_fd)
    parent_after = os.fstat(parent_fd)
    parent_by_name = parent.lstat()
finally:
    if destination_fd is not None:
        os.close(destination_fd)
    os.close(parent_fd)
if not (
    directory_authority(parent_initial)
    == directory_authority(parent_before)
    == directory_authority(parent_after)
    == directory_authority(parent_by_name)
):
    raise SystemExit("pinned known-hosts parent changed during creation")
if (
    fields(created) != fields(by_name)
    or not stat.S_ISREG(created.st_mode)
    or (created.st_uid, created.st_gid) != caller
    or created.st_nlink != 1
    or stat.S_IMODE(created.st_mode) != 0o400
    or stored != canonical
):
    raise SystemExit("pinned known-hosts stored authority differs")
print(fingerprint)
P30_PIN_CONFIRMED_HOST_KEY
```

Then run the frozen credential-free preflight through the logger before
provisioning. The preflight independently reauthenticates the key file and also
pins the GPU, image, source checkouts, data manifest, P26 evidence, and all
freshness identities; it uses only read-only Git, Docker, filesystem, and
`nvidia-smi` operations and never imports Torch or initializes CUDA:

```bash
export P30_SSH_IDENTITY_FILE=/Users/harry/.ssh/id_ed25519
export P30_SSH_TARGET=ubuntu@129.146.177.214
p30_strict_ssh=(
  /usr/bin/ssh -T -F /dev/null -p 22
  -o BatchMode=yes
  -o StrictHostKeyChecking=yes
  -o "UserKnownHostsFile=$P30_PINNED_KNOWN_HOSTS"
  -o GlobalKnownHostsFile=/dev/null
  -o HostKeyAlgorithms=ssh-ed25519
  -o UpdateHostKeys=no
  -o CheckHostIP=yes
  -o ClearAllForwardings=yes
  -o ProxyCommand=none
  -o ProxyJump=none
  -o ControlMaster=no
  -o ControlPath=none
  -o PermitLocalCommand=no
  -o PasswordAuthentication=no
  -o KbdInteractiveAuthentication=no
  -o PreferredAuthentications=publickey
  -o IdentitiesOnly=yes
  -i "$P30_SSH_IDENTITY_FILE"
  "$P30_SSH_TARGET"
)
p30_run_logged_client_command 00-host-freshness \
  "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_HOST_PREFLIGHT" \
  --known-hosts "$P30_PINNED_KNOWN_HOSTS" \
  --identity-file "$P30_SSH_IDENTITY_FILE" \
  --expected-client-source-sha256 "$P30_EXPECTED_HOST_PREFLIGHT_SOURCE_SHA256"
```

Its sole success record has status
`all_read_only_prerequisites_passed_before_p30_state_creation`. A failed
read-only preflight may be diagnosed and rerun before provisioning; it is an
engineering check, not a consumed scientific attempt. Once provisioning
begins, no freshness mismatch is repairable under this P30 identity.

Before any P30 host operation, independently run:

```bash
"$P30_CLIENT_PYTHON" -I -S \
  scripts/reconstruct_p30_umask_bound_control_seal.py
```

The P30 reconstructor launches the P29 contract, terminal P29 outcome, P28
contract, terminal P28 outcome, and unchanged P27 reconstructors as five
isolated children. Their results must be 15/15, 11/11, 12/12, 10/10, and 23/23,
respectively.

The SHA-256-authenticated, canonical, nonsymlink local
`P30_CLIENT_PYTHON` and the CUDA-host Python must be Python 3.10 or newer.
On the CUDA host, first run
`/usr/bin/python3 -I -S -c 'import sys; assert sys.version_info >= (3, 10)'`;
every later bootstrap or reconstructor entry uses exactly
`/usr/bin/python3 -I -S`. The verifier supplies unchanged nested authorities a
private temporary `python3` shim that execs that same absolute interpreter
with both flags. The independent P30 reconstructor likewise creates a fresh
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

## 1. Freeze revised P30 before provisioning its host namespace

Run all portable checks locally:

```bash
uv run --locked pytest -q \
  tests/test_p30_umask_bound_control_seal_contract.py \
  tests/test_p30_control_bundle_verifier.py \
  tests/test_p30_umask_bound_control_seal_orchestration.py \
  tests/test_run_logged_client_command.py \
  tests/test_p30_uid1000_root_handoff_rehearsal.py \
  tests/test_p30_actual_host_prerequisite_preflight.py \
  tests/test_p30_control_bundle_receipt_review.py \
  tests/test_p30_local_bundle_verifier.py \
  tests/test_p30_authenticated_client_transfer.py \
  tests/test_p28_bundle_complete_localization_bridge_contract.py \
  tests/test_p28_bundle_complete_localization_bridge_outcome.py \
  tests/test_p27_cuda_deleted_mapping_localization_contract.py \
  tests/test_p27_cuda_deleted_mapping_localization.py \
  tests/test_p27_cuda_deleted_mapping_localization_orchestration.py \
  tests/test_p27_cuda_deleted_mapping_localization_sanitizer.py \
  tests/test_p27_p26_failure_ingestion.py
bash -n scripts/run_p30_umask_bound_control_seal.sh
uv run --locked ruff format --check experiments/training scripts tests
uv run --locked ruff check experiments/training scripts tests
/usr/bin/git diff --check
```

Commit and independently review the complete P30 package as one direct child
of terminal P29 commit
`8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf`. The checked-out v5 source freeze
already occupies that direct-child position, so the block below authenticates
its annotated tag, stages exactly the complete 26-path v6 package, checks the
prospective index delta and modes, and amends the branch tip. An ordinary child
commit is forbidden. The immutable v5 tag remains attached to the old commit.
Record the resulting replacement commit and tree as
`P30_SOURCE_FREEZE_COMMIT` and `P30_SOURCE_FREEZE_TREE`. Those values are
selected after the commit and therefore are not embedded in the contract.

In the same local Bash session, resolve the reviewed branch tip, prove that it
is one direct child of terminal P29, and derive every control-source digest
from Git object bytes rather than from the mutable worktree:

```bash
set -euo pipefail
export P30_LOCAL_REPOSITORY=/Users/harry/Desktop/temp/OptimizationML
test "$({ /bin/pwd -P; })" = "$P30_LOCAL_REPOSITORY"
p30_local_git() {
  /usr/bin/env -i \
    HOME=/nonexistent LANG=C LC_ALL=C PATH=/usr/bin:/bin \
    GIT_ATTR_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
    GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
    GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 \
    GIT_TERMINAL_PROMPT=0 \
    "GIT_AUTHOR_NAME=OptimizationML P30" \
    GIT_AUTHOR_EMAIL=p30@invalid.example \
    "GIT_COMMITTER_NAME=OptimizationML P30" \
    GIT_COMMITTER_EMAIL=p30@invalid.example \
    /usr/bin/git --no-replace-objects \
      -c core.hooksPath=/dev/null -c commit.gpgSign=false \
      -c "safe.directory=$P30_LOCAL_REPOSITORY" \
      -C "$P30_LOCAL_REPOSITORY" "$@"
}
p30_phase_one_paths=(
  .github/compat/p29/stat
  .github/workflows/ci.yml
  .github/workflows/p30-uid1000-root-engineering-rehearsal.yml
  .github/workflows/p30-umask-bound-control-seal.yml
  experiments/training/ENGINEERING_REHEARSAL_POLICY.md
  experiments/training/P30_UID1000_ROOT_HANDOFF_REHEARSAL.md
  experiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md
  experiments/training/p30_umask_bound_control_seal_contract.json
  scripts/reconstruct_p30_umask_bound_control_seal.py
  scripts/rehearse_p30_uid1000_root_handoff.sh
  scripts/review_p30_control_bundle_receipt.py
  scripts/run_logged_client_command.py
  scripts/run_p30_actual_host_prerequisite_preflight.py
  scripts/run_p30_authenticated_client_transfer.py
  scripts/run_p30_umask_bound_control_seal.sh
  scripts/verify_p30_control_bundle.py
  scripts/verify_p30_local_bundle.py
  tests/test_p30_actual_host_prerequisite_preflight.py
  tests/test_p30_authenticated_client_transfer.py
  tests/test_p30_control_bundle_receipt_review.py
  tests/test_p30_control_bundle_verifier.py
  tests/test_p30_local_bundle_verifier.py
  tests/test_p30_uid1000_root_handoff_rehearsal.py
  tests/test_p30_umask_bound_control_seal_contract.py
  tests/test_p30_umask_bound_control_seal_orchestration.py
  tests/test_run_logged_client_command.py
)
test "${#p30_phase_one_paths[@]}" -eq 26

# v5 already occupies the sole direct-child position in the checked-out
# branch. Preserve its annotated tag on the old object and replace only the
# branch tip by amending that commit; an ordinary child commit is forbidden.
export P30_HISTORICAL_V5_COMMIT=8286d45db45c27e219724cfb9ace2108a2823301
export P30_HISTORICAL_V5_TAG_OBJECT=4c0b848bee12efb27dc93276cd1866c8ddc2d0b3
test "$({ p30_local_git symbolic-ref --short HEAD; })" = \
  p30-umask-bound-control-seal
test "$({ p30_local_git rev-parse HEAD; })" = "$P30_HISTORICAL_V5_COMMIT"
test "$({ p30_local_git cat-file -t \
  refs/tags/p30-umask-bound-control-seal-source-freeze-v5; })" = tag
test "$({ p30_local_git rev-parse \
  refs/tags/p30-umask-bound-control-seal-source-freeze-v5; })" = \
  "$P30_HISTORICAL_V5_TAG_OBJECT"
test "$({ p30_local_git rev-parse \
  refs/tags/p30-umask-bound-control-seal-source-freeze-v5^{commit}; })" = \
  "$P30_HISTORICAL_V5_COMMIT"
test "$({ p30_local_git rev-list --parents -n 1 HEAD; })" = \
  "$P30_HISTORICAL_V5_COMMIT 8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"
p30_local_git add -- "${p30_phase_one_paths[@]}"
test "$({
  p30_local_git diff --cached --no-renames --name-status \
    8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf
})" = "$({
  /usr/bin/printf '%s\n' \
    $'A\t.github/compat/p29/stat' \
    $'M\t.github/workflows/ci.yml' \
    $'A\t.github/workflows/p30-uid1000-root-engineering-rehearsal.yml' \
    $'A\t.github/workflows/p30-umask-bound-control-seal.yml' \
    $'A\texperiments/training/ENGINEERING_REHEARSAL_POLICY.md' \
    $'A\texperiments/training/P30_UID1000_ROOT_HANDOFF_REHEARSAL.md' \
    $'A\texperiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md' \
    $'A\texperiments/training/p30_umask_bound_control_seal_contract.json' \
    $'A\tscripts/reconstruct_p30_umask_bound_control_seal.py' \
    $'A\tscripts/rehearse_p30_uid1000_root_handoff.sh' \
    $'A\tscripts/review_p30_control_bundle_receipt.py' \
    $'A\tscripts/run_logged_client_command.py' \
    $'A\tscripts/run_p30_actual_host_prerequisite_preflight.py' \
    $'A\tscripts/run_p30_authenticated_client_transfer.py' \
    $'A\tscripts/run_p30_umask_bound_control_seal.sh' \
    $'A\tscripts/verify_p30_control_bundle.py' \
    $'A\tscripts/verify_p30_local_bundle.py' \
    $'A\ttests/test_p30_actual_host_prerequisite_preflight.py' \
    $'A\ttests/test_p30_authenticated_client_transfer.py' \
    $'A\ttests/test_p30_control_bundle_receipt_review.py' \
    $'A\ttests/test_p30_control_bundle_verifier.py' \
    $'A\ttests/test_p30_local_bundle_verifier.py' \
    $'A\ttests/test_p30_uid1000_root_handoff_rehearsal.py' \
    $'A\ttests/test_p30_umask_bound_control_seal_contract.py' \
    $'A\ttests/test_p30_umask_bound_control_seal_orchestration.py' \
    $'A\ttests/test_run_logged_client_command.py'
})"
test "$({
  p30_local_git ls-files -s -- "${p30_phase_one_paths[@]}" | \
    /usr/bin/awk '{print $1 "\t" $4}'
})" = "$({
  /usr/bin/printf '%s\n' \
    $'100755\t.github/compat/p29/stat' \
    $'100644\t.github/workflows/ci.yml' \
    $'100644\t.github/workflows/p30-uid1000-root-engineering-rehearsal.yml' \
    $'100644\t.github/workflows/p30-umask-bound-control-seal.yml' \
    $'100644\texperiments/training/ENGINEERING_REHEARSAL_POLICY.md' \
    $'100644\texperiments/training/P30_UID1000_ROOT_HANDOFF_REHEARSAL.md' \
    $'100644\texperiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md' \
    $'100644\texperiments/training/p30_umask_bound_control_seal_contract.json' \
    $'100755\tscripts/reconstruct_p30_umask_bound_control_seal.py' \
    $'100755\tscripts/rehearse_p30_uid1000_root_handoff.sh' \
    $'100644\tscripts/review_p30_control_bundle_receipt.py' \
    $'100755\tscripts/run_logged_client_command.py' \
    $'100644\tscripts/run_p30_actual_host_prerequisite_preflight.py' \
    $'100755\tscripts/run_p30_authenticated_client_transfer.py' \
    $'100644\tscripts/run_p30_umask_bound_control_seal.sh' \
    $'100755\tscripts/verify_p30_control_bundle.py' \
    $'100644\tscripts/verify_p30_local_bundle.py' \
    $'100644\ttests/test_p30_actual_host_prerequisite_preflight.py' \
    $'100644\ttests/test_p30_authenticated_client_transfer.py' \
    $'100644\ttests/test_p30_control_bundle_receipt_review.py' \
    $'100644\ttests/test_p30_control_bundle_verifier.py' \
    $'100644\ttests/test_p30_local_bundle_verifier.py' \
    $'100644\ttests/test_p30_uid1000_root_handoff_rehearsal.py' \
    $'100644\ttests/test_p30_umask_bound_control_seal_contract.py' \
    $'100644\ttests/test_p30_umask_bound_control_seal_orchestration.py' \
    $'100644\ttests/test_run_logged_client_command.py'
})"
p30_local_git diff --cached --check
test -z "$({ p30_local_git diff --no-ext-diff --name-only; })"
test -z "$({ p30_local_git ls-files --others --exclude-standard; })"
p30_local_git commit --amend -m \
  "Freeze P30 v6 one-shot localization package"
test "$({ p30_local_git rev-parse HEAD; })" != "$P30_HISTORICAL_V5_COMMIT"
test "$({ p30_local_git rev-parse \
  refs/tags/p30-umask-bound-control-seal-source-freeze-v5; })" = \
  "$P30_HISTORICAL_V5_TAG_OBJECT"
test "$({ p30_local_git rev-parse \
  refs/tags/p30-umask-bound-control-seal-source-freeze-v5^{commit}; })" = \
  "$P30_HISTORICAL_V5_COMMIT"
export P30_SOURCE_FREEZE_COMMIT="$({
  p30_local_git rev-parse --verify \
    refs/heads/p30-umask-bound-control-seal^{commit}
})"
test "$({ p30_local_git rev-list --parents -n 1 \
  "$P30_SOURCE_FREEZE_COMMIT"; })" = \
  "$P30_SOURCE_FREEZE_COMMIT 8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"
export P30_SOURCE_FREEZE_TREE="$({
  p30_local_git rev-parse --verify "$P30_SOURCE_FREEZE_COMMIT^{tree}"
})"
test "$({
  p30_local_git diff-tree --no-commit-id --name-status -r \
    8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf \
    "$P30_SOURCE_FREEZE_COMMIT"
})" = "$({
  /usr/bin/printf '%s\n' \
    $'A\t.github/compat/p29/stat' \
    $'M\t.github/workflows/ci.yml' \
    $'A\t.github/workflows/p30-uid1000-root-engineering-rehearsal.yml' \
    $'A\t.github/workflows/p30-umask-bound-control-seal.yml' \
    $'A\texperiments/training/ENGINEERING_REHEARSAL_POLICY.md' \
    $'A\texperiments/training/P30_UID1000_ROOT_HANDOFF_REHEARSAL.md' \
    $'A\texperiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md' \
    $'A\texperiments/training/p30_umask_bound_control_seal_contract.json' \
    $'A\tscripts/reconstruct_p30_umask_bound_control_seal.py' \
    $'A\tscripts/rehearse_p30_uid1000_root_handoff.sh' \
    $'A\tscripts/review_p30_control_bundle_receipt.py' \
    $'A\tscripts/run_logged_client_command.py' \
    $'A\tscripts/run_p30_actual_host_prerequisite_preflight.py' \
    $'A\tscripts/run_p30_authenticated_client_transfer.py' \
    $'A\tscripts/run_p30_umask_bound_control_seal.sh' \
    $'A\tscripts/verify_p30_control_bundle.py' \
    $'A\tscripts/verify_p30_local_bundle.py' \
    $'A\ttests/test_p30_actual_host_prerequisite_preflight.py' \
    $'A\ttests/test_p30_authenticated_client_transfer.py' \
    $'A\ttests/test_p30_control_bundle_receipt_review.py' \
    $'A\ttests/test_p30_control_bundle_verifier.py' \
    $'A\ttests/test_p30_local_bundle_verifier.py' \
    $'A\ttests/test_p30_uid1000_root_handoff_rehearsal.py' \
    $'A\ttests/test_p30_umask_bound_control_seal_contract.py' \
    $'A\ttests/test_p30_umask_bound_control_seal_orchestration.py' \
    $'A\ttests/test_run_logged_client_command.py'
})"
test "$({
  p30_local_git ls-tree -r --full-tree "$P30_SOURCE_FREEZE_COMMIT" -- \
    "${p30_phase_one_paths[@]}" | \
    /usr/bin/awk '{print $1 "\t" $4}'
})" = "$({
  /usr/bin/printf '%s\n' \
    $'100755\t.github/compat/p29/stat' \
    $'100644\t.github/workflows/ci.yml' \
    $'100644\t.github/workflows/p30-uid1000-root-engineering-rehearsal.yml' \
    $'100644\t.github/workflows/p30-umask-bound-control-seal.yml' \
    $'100644\texperiments/training/ENGINEERING_REHEARSAL_POLICY.md' \
    $'100644\texperiments/training/P30_UID1000_ROOT_HANDOFF_REHEARSAL.md' \
    $'100644\texperiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md' \
    $'100644\texperiments/training/p30_umask_bound_control_seal_contract.json' \
    $'100755\tscripts/reconstruct_p30_umask_bound_control_seal.py' \
    $'100755\tscripts/rehearse_p30_uid1000_root_handoff.sh' \
    $'100644\tscripts/review_p30_control_bundle_receipt.py' \
    $'100755\tscripts/run_logged_client_command.py' \
    $'100644\tscripts/run_p30_actual_host_prerequisite_preflight.py' \
    $'100755\tscripts/run_p30_authenticated_client_transfer.py' \
    $'100644\tscripts/run_p30_umask_bound_control_seal.sh' \
    $'100755\tscripts/verify_p30_control_bundle.py' \
    $'100644\tscripts/verify_p30_local_bundle.py' \
    $'100644\ttests/test_p30_actual_host_prerequisite_preflight.py' \
    $'100644\ttests/test_p30_authenticated_client_transfer.py' \
    $'100644\ttests/test_p30_control_bundle_receipt_review.py' \
    $'100644\ttests/test_p30_control_bundle_verifier.py' \
    $'100644\ttests/test_p30_local_bundle_verifier.py' \
    $'100644\ttests/test_p30_uid1000_root_handoff_rehearsal.py' \
    $'100644\ttests/test_p30_umask_bound_control_seal_contract.py' \
    $'100644\ttests/test_p30_umask_bound_control_seal_orchestration.py' \
    $'100644\ttests/test_run_logged_client_command.py'
})"
test -z "$({ p30_local_git status --porcelain=v1 --untracked-files=all; })"

p30_export_source_blob_sha256() {
  local variable_name="$1"
  local relative_path="$2"
  local value
  case "$variable_name" in P30_[A-Z0-9_]*) ;; *) return 1 ;; esac
  value="$({
    set -o pipefail
    p30_local_git cat-file blob \
      "$P30_SOURCE_FREEZE_COMMIT:$relative_path" | \
      /usr/bin/shasum -a 256 | /usr/bin/awk '{print $1}'
  })"
  test "${#value}" -eq 64
  case "$value" in *[!0-9a-f]*) return 1 ;; esac
  printf -v "$variable_name" '%s' "$value"
  export "$variable_name"
}

p30_export_source_blob_sha256 P30_EXPECTED_CONTRACT_SHA256 \
  experiments/training/p30_umask_bound_control_seal_contract.json
p30_export_source_blob_sha256 P30_EXPECTED_ORCHESTRATOR_SHA256 \
  scripts/run_p30_umask_bound_control_seal.sh
p30_export_source_blob_sha256 P30_EXPECTED_RECONSTRUCTOR_SHA256 \
  scripts/reconstruct_p30_umask_bound_control_seal.py
p30_export_source_blob_sha256 P30_EXPECTED_LOCAL_BUNDLE_VERIFIER_SOURCE_SHA256 \
  scripts/verify_p30_local_bundle.py
p30_export_source_blob_sha256 P30_EXPECTED_AUTHENTICATED_TRANSFER_SOURCE_SHA256 \
  scripts/run_p30_authenticated_client_transfer.py
p30_export_source_blob_sha256 P30_EXPECTED_P29_CONTRACT_SHA256 \
  experiments/training/p29_permission_safe_bundle_localization_bridge_contract.json
p30_export_source_blob_sha256 P30_EXPECTED_P29_RECONSTRUCTOR_SHA256 \
  scripts/reconstruct_p29_permission_safe_bundle_localization_bridge.py
p30_export_source_blob_sha256 P30_EXPECTED_P29_OUTCOME_SHA256 \
  results/summaries/p29_permission_safe_bundle_localization_bridge_outcome.json
p30_export_source_blob_sha256 P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256 \
  scripts/reconstruct_p29_permission_safe_bundle_localization_bridge_outcome.py
p30_export_source_blob_sha256 P30_EXPECTED_P28_CONTRACT_SHA256 \
  experiments/training/p28_bundle_complete_localization_bridge_contract.json
p30_export_source_blob_sha256 P30_EXPECTED_P28_RECONSTRUCTOR_SHA256 \
  scripts/reconstruct_p28_bundle_complete_localization_bridge.py
p30_export_source_blob_sha256 P30_EXPECTED_P28_OUTCOME_SHA256 \
  results/summaries/p28_bundle_complete_localization_bridge_outcome.json
p30_export_source_blob_sha256 P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256 \
  scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py
p30_export_source_blob_sha256 P30_EXPECTED_P27_CONTRACT_SHA256 \
  experiments/training/p27_cuda_deleted_mapping_localization_contract.json
p30_export_source_blob_sha256 P30_EXPECTED_P27_RECONSTRUCTOR_SHA256 \
  scripts/reconstruct_p27_cuda_deleted_mapping_localization.py
p30_export_source_blob_sha256 P30_EXPECTED_P27_LOCALIZER_SHA256 \
  experiments/training/run_p27_cuda_deleted_mapping_localization.py
p30_export_source_blob_sha256 P30_EXPECTED_INGESTER_SHA256 \
  scripts/ingest_p26_trace_off_a_failure.py
p30_export_source_blob_sha256 P30_EXPECTED_P23_CORE_SHA256 \
  experiments/training/p23_deterministic_cuda_shadow_trace.py
p30_export_source_blob_sha256 P30_EXPECTED_P27_SANITIZER_SHA256 \
  scripts/sanitize_p27_cuda_deleted_mapping_localization.py

test "$P30_EXPECTED_P29_CONTRACT_SHA256" = \
  54d10ab01b7d453d01931459b314fc4491b6e4317a02fb763ca9872686311b0a
test "$P30_EXPECTED_P29_RECONSTRUCTOR_SHA256" = \
  a384bfcbc25fe9148a158a5f524170557808f8e68d07cb99474b64bbe3c10aea
test "$P30_EXPECTED_P29_OUTCOME_SHA256" = \
  6d29601a71ce6e0c99bcc35497d328c8fdd2a3cfc135351c19876507b00a0f64
test "$P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256" = \
  b850a9423b42aa2026db8505b7fdd3767d7954e94283cc96ebdb87cba289b33f
test "$P30_EXPECTED_P28_CONTRACT_SHA256" = \
  6a5005f902d937f0edbec3394e6205e482fe6f6fea45f986ac873fd6ab2e8718
test "$P30_EXPECTED_P28_RECONSTRUCTOR_SHA256" = \
  c8b5c853a5635f277f7d1d4c0185a199c6658c575a35c088c7cca51d36563b62
test "$P30_EXPECTED_P28_OUTCOME_SHA256" = \
  298324d163aa12c1cb64bfdb912ae846152d0980ae3c981a54ead75a019305c9
test "$P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256" = \
  d736909181eaa75298851b7fc65f9c0352fb32c3d07e40eb4118eb10c1898631
test "$P30_EXPECTED_P27_CONTRACT_SHA256" = \
  02000debea28f499ddeb3b8c6c8b0615786cd2e7858cc8a1605b04d638da4b7a
test "$P30_EXPECTED_P27_RECONSTRUCTOR_SHA256" = \
  0eedffa5442ce7fc3950ae74d169fa652c70794935c541028eba3d0bda20e033
test "$P30_EXPECTED_P27_LOCALIZER_SHA256" = \
  86ba200d69a029c64b2218581d6b92245930d5f1e60d0176aca22ad4d6ddf65a
test "$P30_EXPECTED_INGESTER_SHA256" = \
  3e7769d668a8756aea7359bef4d9b8ccfc8db66da102edfbc6c5de385654cdeb
test "$P30_EXPECTED_P23_CORE_SHA256" = \
  ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab
test "$P30_EXPECTED_P27_SANITIZER_SHA256" = \
  3ec64cdc3681d41d70c624c6ffd26dde1a7e498bda13ddafaaced271a43dd6c4
```

Now return to Section 0B and execute its complete client-tool materialization,
host-key pin, and label-`00` read-only host preflight. Resume immediately below
only after the frozen logger status for that preflight is exactly `0\n` and its
canonical result reports
`all_read_only_prerequisites_passed_before_p30_state_creation`. This is the
only deliberate dependency jump in the runbook.

The source and runtime-review bundles must each advertise exactly these nine
refs:

| Ref | Required object | Required peel |
| --- | --- | --- |
| `refs/heads/p30-umask-bound-control-seal` | phase-specific reviewed P30 commit | same commit |
| `refs/heads/p29-permission-safe-bundle-localization-bridge` | `8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf` | same commit |
| `refs/tags/p29-orchestrator-source-mode-diagnostic` | annotated tag `cf3c60b1d5c004b9e601a6afbf630358f80f1d79` | `8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf` |
| `refs/heads/p28-bundle-complete-localization-bridge` | `7274367f2b05fb3c9ed9f876a59be647703bbdc2` | same commit |
| `refs/tags/p28-control-parent-permission-diagnostic` | annotated tag `f3c81a2f14f853f369015a4f81b6695b52eec74e` | `7274367f2b05fb3c9ed9f876a59be647703bbdc2` |
| `refs/heads/p27-cuda-deleted-mapping-localization` | `ec63550331925ded158e3f389e294e4d1f12db3a` | same commit |
| `refs/tags/p27-cuda-deleted-mapping-localization-diagnostic` | annotated tag `c88b98eae9be2458abde45b05d3dccfe09c0c7ed` | `ec63550331925ded158e3f389e294e4d1f12db3a` |
| `refs/tags/p26-permission-safe-acquisition-checkpoint` | annotated tag `bacad707d3779bfa10957e18cb4c69b1a7f0cbce` | `5429da23ff18888daa2312c530a4587780484d8b` |
| `refs/tags/p26-attempt02-acquisition-source` | annotated tag `f35a7dca8f6bc39e9748e79b6712fab4a203396b` | `185e444afc0b44ca0a09b1bde49a6b6fa3973355` |

Never substitute a peeled commit for an annotated tag object.

Create and review the complete source bundle locally:

```bash
export P30_LOCAL_TRANSPORT="$(/usr/bin/mktemp -d /private/tmp/p30-transfer.XXXXXX)"
/usr/bin/chgrp "$P30_CLIENT_GID" "$P30_LOCAL_TRANSPORT"
/bin/chmod 0700 "$P30_LOCAL_TRANSPORT"
test "$({
  /usr/bin/stat -f '%u:%g:%Lp' "$P30_LOCAL_TRANSPORT"
})" = "$P30_CLIENT_UID:$P30_CLIENT_GID:700"
export P30_LOCAL_SOURCE_BUNDLE="$P30_LOCAL_TRANSPORT/p30_source.bundle"
p30_client_forbidden_roots+=("$P30_LOCAL_TRANSPORT")
p30_local_git bundle create "$P30_LOCAL_SOURCE_BUNDLE" \
  refs/heads/p30-umask-bound-control-seal \
  refs/heads/p29-permission-safe-bundle-localization-bridge \
  refs/tags/p29-orchestrator-source-mode-diagnostic \
  refs/heads/p28-bundle-complete-localization-bridge \
  refs/tags/p28-control-parent-permission-diagnostic \
  refs/heads/p27-cuda-deleted-mapping-localization \
  refs/tags/p27-cuda-deleted-mapping-localization-diagnostic \
  refs/tags/p26-permission-safe-acquisition-checkpoint \
  refs/tags/p26-attempt02-acquisition-source
P30_LOCAL_SOURCE_BUNDLE_BINDING="$({
  "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_LOCAL_BUNDLE_VERIFIER" \
    --phase source \
    --bundle "$P30_LOCAL_SOURCE_BUNDLE" \
    --git-executable /usr/bin/git \
    --expected-p30-commit "$P30_SOURCE_FREEZE_COMMIT" \
    --expected-p30-tree "$P30_SOURCE_FREEZE_TREE" \
    --expected-p30-parent 8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf \
    --expected-verifier-source-sha256 \
      "$P30_EXPECTED_LOCAL_BUNDLE_VERIFIER_SOURCE_SHA256"
})"
p30_local_bundle_binding_field() {
  "$P30_CLIENT_PYTHON" -I -S - "$1" "$2" "$3" "$4" "$5" <<'PY'
import json, sys

raw, phase, commit, tree, field = sys.argv[1:]


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise SystemExit("duplicate local bundle binding key")
        result[key] = value
    return result


payload = json.loads(raw, object_pairs_hook=reject_duplicates)
if set(payload) != {
    "bundle_byte_count",
    "bundle_sha256",
    "p30_commit",
    "p30_tree",
    "phase",
    "schema",
    "verified_ref_count",
}:
    raise SystemExit("local bundle binding field inventory differs")
if (
    payload["schema"] != "passive-muon-p30-local-bundle-verification-v1"
    or payload["phase"] != phase
    or payload["p30_commit"] != commit
    or payload["p30_tree"] != tree
    or payload["verified_ref_count"] != 9
):
    raise SystemExit("local bundle binding authority differs")
digest = payload["bundle_sha256"]
count = payload["bundle_byte_count"]
if (
    not isinstance(digest, str)
    or len(digest) != 64
    or any(character not in "0123456789abcdef" for character in digest)
    or isinstance(count, bool)
    or not isinstance(count, int)
    or count <= 0
):
    raise SystemExit("local bundle digest or size is malformed")
if field not in {"bundle_sha256", "bundle_byte_count"}:
    raise SystemExit("unapproved local bundle binding field")
print(payload[field])
PY
}
export P30_EXPECTED_SOURCE_BUNDLE_SHA256="$({
  p30_local_bundle_binding_field \
    "$P30_LOCAL_SOURCE_BUNDLE_BINDING" source \
    "$P30_SOURCE_FREEZE_COMMIT" "$P30_SOURCE_FREEZE_TREE" bundle_sha256
})"
export P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT="$({
  p30_local_bundle_binding_field \
    "$P30_LOCAL_SOURCE_BUNDLE_BINDING" source \
    "$P30_SOURCE_FREEZE_COMMIT" "$P30_SOURCE_FREEZE_TREE" bundle_byte_count
})"
```

The frozen local verifier reads the bundle through a componentwise no-follow
single-link guard, revalidates the bytes and path identity, runs full
`git bundle verify` and strict closure checking on a private copy, and checks
the exact nine ref objects, all object types, all five annotated-tag peels, and
the phase-specific P30 tree and direct parent. Its canonical output is the sole
source of the exported source-bundle SHA-256 and byte count.

## 2. Provision the exact permission-safe namespace once

Use these identities:

```bash
export P30_ATTEMPT_ID=20260906-03
export P30_ATTEMPT_ROOT=/secure/p30/attempt-20260906-03
export P30_CONTAINER=p30-localization-20260906-03
export P30_SECURE_ROOT=/secure/p30
export P30_TRANSPORT_ROOT=/secure/p30/transport-20260906-03
export P30_EXECUTION_ROOT=/secure/p30/execution-20260906-03
export P30_EXECUTION_MANIFEST=/secure/p30/execution-20260906-03/snapshot-manifest.json
export P30_CONTROL_REPO=/secure/p30/transport-20260906-03/control-source
export P30_AUTHORITY_REPO=/secure/p30/transport-20260906-03/authority-185e444
export P30_LEDGER_ROOT=/var/lib/optimizationml-p30-20260906-03
export P30_PREPARE_INVOCATION=/var/lib/optimizationml-p30-20260906-03/prepare-runtime.invoked
export P30_LOCALIZATION_INVOCATION=/var/lib/optimizationml-p30-20260906-03/run-localization.invoked
export P30_LOCALIZATION_AUTHORIZATION=/var/lib/optimizationml-p30-20260906-03/localization.authorized
export P30_LOCALIZATION_EXIT_STATUS=/var/lib/optimizationml-p30-20260906-03/run-localization.exit-status.txt
export P30_SOURCE_BUNDLE=/secure/p30/transport-20260906-03/p30_source.bundle
export P30_SOURCE_CLOSURE=/secure/p30/transport-20260906-03/p30_source_closure.git
export P30_SOURCE_RECEIPT=/secure/p30/transport-20260906-03/p30_source_bundle_receipt.json
export P30_BOOTSTRAP_VERIFIER=/secure/p30/transport-20260906-03/verify_p30_control_bundle.py
```

Within `/secure/p30`, the only directories created before transfer are the
root-owned P30 namespace and its one process-owned transport capability root.
A separate non-capability ledger root is created empty under `/var/lib`.
All three new paths must initially be absent. `/secure` itself must already be
a root-owned, nonsymlink, nonmountpoint directory at mode `0755`, with its base
access ACL equal to those mode bits and no named/default or frozen-xattr ACL.
Do not repair `/secure` as part of P30: reject a mismatch before creating the
P30 namespace. Run this once from the client through the frozen logger and the
single strict SSH vector; if any command fails or any new path already exists,
retain the state and stop permanently under this identifier:

```bash
set -euo pipefail
p30_run_logged_client_command 01-provision-namespace \
  "${p30_strict_ssh[@]}" \
  /usr/bin/sudo -n /usr/bin/env -i \
  HOME=/root LANG=C LC_ALL=C PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash -p -s <<'P30_PROVISION_NAMESPACE'
set -euo pipefail
/usr/bin/python3 -I -S - <<'PY'
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
/usr/bin/test ! -e /secure/p30
/usr/bin/test ! -L /secure/p30
/usr/bin/mkdir --mode=0755 /secure/p30
/usr/bin/chown root:root /secure/p30
/usr/bin/test ! -e /secure/p30/transport-20260906-03
/usr/bin/test ! -L /secure/p30/transport-20260906-03
/usr/bin/mkdir --mode=0700 /secure/p30/transport-20260906-03
/usr/bin/chown ubuntu:ubuntu /secure/p30/transport-20260906-03
/usr/bin/test ! -e /var/lib/optimizationml-p30-20260906-03
/usr/bin/test ! -L /var/lib/optimizationml-p30-20260906-03
/usr/bin/mkdir --mode=0700 /var/lib/optimizationml-p30-20260906-03
/usr/bin/chown root:root /var/lib/optimizationml-p30-20260906-03
/usr/bin/mkdir --mode=0700 /var/lib/optimizationml-p30-20260906-03/deferred
/usr/bin/chown root:root /var/lib/optimizationml-p30-20260906-03/deferred
P30_PROVISION_NAMESPACE
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
`p30-post-authorization-ledger-validation` then requires the exact
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
Do not make `/secure/p30` writable by `ubuntu`. Do not bind, copy, symlink, or
privileged-move either checkout to a sibling of the transport root.
After direct creation, control and authority are nonsymlink directories owned
by `ubuntu:ubuntu` (`1000:1000`) at mode `0700`, share the transport device and
mount ID, are not mountpoints, and have no extended or named access ACL, no
default ACL, and no frozen ACL xattr. Their base access ACL exactly matches the
mode bits. Those facts are revalidated rather than inferred from the parent.
Before source receipt creation or replay, `/secure/p30` has exactly one child:
`transport-20260906-03`. After the sole prepare phase, it has exactly three
sorted children: `attempt-20260906-03`, `execution-20260906-03`, and
`transport-20260906-03`. Any extra file, directory, symlink, mount, or other
capability sibling is terminal; it is never ignored or cleaned.

Before and after every source-phase transition, the verifier must walk
`/secure`, `/secure/p30`, and the transport root component by component using
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

/secure/p30
user::rwx
group::r-x
other::r-x

/secure/p30/transport-20260906-03
user::rwx
group::---
other::---
```

Named user/group ACL entries and every default ACL entry are forbidden. The
four ACL xattrs `system.posix_acl_access`, `system.posix_acl_default`,
`system.nfs4_acl`, and `system.richacl` must all be absent. The `/secure`
parent, namespace, transport, execution, attempt, and evidence directories
must share one device and mount ID and none may be a mountpoint.
Any identity, permission, or ACL change consumes P30; do not repair and retry.

## 3. Transfer and verify the source bundle before authority or attempt state

Create no additional directory. From the local workstation, stage the bundle
and separately reviewed verifier with `O_EXCL|O_NOFOLLOW`:

```bash
p30_o_excl_upload() {
  local label="$1"
  local source_path="$2"
  local destination_path="$3"
  local destination_mode="$4"
  local expected_sha256="$5"
  local expected_byte_count="$6"
  test "$#" -eq 6
  p30_run_logged_client_command "$label" \
    "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_AUTHENTICATED_TRANSFER" \
    --expected-transfer-source-sha256 \
      "$P30_EXPECTED_AUTHENTICATED_TRANSFER_SOURCE_SHA256" \
    --direction upload \
    --source "$source_path" \
    --staging-root "$P30_CLIENT_TRANSFER_ROOT" \
    --label "$label" \
    --remote-path "$destination_path" \
    --remote-mode "$destination_mode" \
    --expected-source-sha256 "$expected_sha256" \
    --expected-source-byte-count "$expected_byte_count" \
    -- "${p30_strict_ssh[@]}"
}
p30_o_excl_upload \
  02-source-bundle-upload "$P30_LOCAL_SOURCE_BUNDLE" "$P30_SOURCE_BUNDLE" 0600 \
  "$P30_EXPECTED_SOURCE_BUNDLE_SHA256" "$P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT"
p30_o_excl_upload \
  03-bootstrap-verifier-upload \
  "$P30_FROZEN_BUNDLE_VERIFIER" "$P30_BOOTSTRAP_VERIFIER" 0600 \
  "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" -
p30_o_excl_download() {
  local label="$1"
  local retained_path="$2"
  local local_path="$3"
  p30_run_logged_client_command "$label" \
    "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_AUTHENTICATED_TRANSFER" \
    --expected-transfer-source-sha256 \
      "$P30_EXPECTED_AUTHENTICATED_TRANSFER_SOURCE_SHA256" \
    --direction download \
    --destination "$local_path" \
    --remote-path "$retained_path" \
    -- "${p30_strict_ssh[@]}"
  /usr/bin/cmp "$P30_CLIENT_LOG_ROOT/$label.stdout.log" "$local_path"
  /usr/bin/test "$({ /usr/bin/cat "$P30_CLIENT_LOG_ROOT/$label.exit-status.txt"; })" = 0
}
```

For each upload, the source-frozen helper performs a fresh no-follow stable
read, checks the reviewed SHA-256 and byte count, and writes those exact bytes
to a private O_EXCL mode-`0400` staging file. Only after the staging bytes,
identity, digest, size, directory, and by-name binding pass does that same
logged helper launch SSH with the held staging descriptor as stdin. It
revalidates the stage after SSH returns. Thus a source or staging failure before
SSH cannot create remote state, and a later local or remote failure produces
one nonzero retained wrapper status rather than a misleading SSH-only success.
The download direction likewise reserves and finalizes its local O_EXCL file
inside the same logged helper that launches SSH; its retained status is zero
only after both the remote stable read and local stored-byte validation pass.

Authenticate the transferred SHA-256 values, byte counts, owners, and modes.
The source verifier must then:

1. authenticate its exact bootstrap bytes and the permission-safe layout;
2. require control, authority, attempt, closure, and receipt paths absent;
3. copy the bundle once with `O_NOFOLLOW` to a private mode-`0400` file;
4. reject prerequisites and require exactly the nine advertised refs;
5. create a fresh empty bare closure, explicitly fetch all nine same-named
   refs, verify tag types and peels, and run full strict `git fsck`;
6. construct the absent canonical control checkout directly at
   `/secure/p30/transport-20260906-03/control-source` without a bind, copy,
   symlink, relocation, or privileged operation;
7. prove the control is detached, clean, closed, and free of Git object
   indirections;
8. execute the P30 reconstructor and its P29-contract 15/15, terminal-P29
   outcome 11/11, P28-contract 12/12, terminal-P28 outcome 10/10, and
   unchanged-P27 23/23 children only from authenticated private bundle bytes,
   never through mutable checkout paths;
9. require the orchestrator's exact Git mode `100644`, physical authority
   `ubuntu:ubuntu 0600`, one link, contract-bound digest, stable bytes, and no
   ACL xattrs, and bind that record into the receipt; and
10. revalidate every layout and bundle identity before writing the external
   source receipt with `O_EXCL`.

From the client, pass the same fixed paths plus the independently reviewed
source commit, tree, bundle, and verifier values as arguments to the reviewed
bootstrap verifier. Run it exactly once in source creation mode through the
frozen client logger and strict SSH vector. The root-ledger state with only its
empty deferred journal is a provisioning and operator precondition here; the
UID-1000 verifier cannot and does not inspect that root-only ledger:

```bash
set -euo pipefail
p30_run_logged_client_command 04-source-receipt-create \
  "${p30_strict_ssh[@]}" /usr/bin/env -i \
  HOME=/home/ubuntu LANG=C LC_ALL=C PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  GIT_ATTR_NOSYSTEM=1 GIT_CONFIG_GLOBAL=/dev/null \
  GIT_CONFIG_NOSYSTEM=1 GIT_CONFIG_SYSTEM=/dev/null \
  GIT_NO_REPLACE_OBJECTS=1 GIT_OPTIONAL_LOCKS=0 GIT_TERMINAL_PROMPT=0 \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /usr/bin/python3 -I -S "$P30_BOOTSTRAP_VERIFIER" \
  --phase source \
  --transport-root "$P30_TRANSPORT_ROOT" \
  --bundle "$P30_SOURCE_BUNDLE" \
  --expected-bundle-sha256 "$P30_EXPECTED_SOURCE_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" \
  --closure-repository "$P30_SOURCE_CLOSURE" \
  --control-checkout "$P30_CONTROL_REPO" \
  --attempt-root "$P30_ATTEMPT_ROOT" \
  --execution-root "$P30_EXECUTION_ROOT" \
  --expected-p30-commit "$P30_SOURCE_FREEZE_COMMIT" \
  --expected-p30-tree "$P30_SOURCE_FREEZE_TREE" \
  --bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER" \
  --expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --expected-orchestrator-sha256 "$P30_EXPECTED_ORCHESTRATOR_SHA256" \
  --receipt-output "$P30_SOURCE_RECEIPT"
```

Review the external receipt and record its SHA-256. A failed verifier leaves
all partial state in place and permanently ends P30 attempt `20260906-03`.
There is no cleanup, repair, resume, or second source-verifier invocation.

The review is not an informal JSON inspection. Download the exact remote
receipt with `O_EXCL`, require it to byte-match the receipt-download command's
frozen logger stdout, and review that client-owned mode-`0600` copy locally with the
source-frozen mode-`0400` independent reviewer. No reviewer is executed from
the process-writable remote transport namespace:

```bash
set -euo pipefail
export P30_LOCAL_SOURCE_RECEIPT="$P30_LOCAL_TRANSPORT/p30_source_bundle_receipt.json"
p30_o_excl_download \
  05-source-receipt-download "$P30_SOURCE_RECEIPT" "$P30_LOCAL_SOURCE_RECEIPT"
export P30_EXPECTED_SOURCE_RECEIPT_SHA256="$({
  /usr/bin/shasum -a 256 "$P30_LOCAL_SOURCE_RECEIPT" | /usr/bin/awk '{print $1}'
})"
test "${#P30_EXPECTED_SOURCE_RECEIPT_SHA256}" -eq 64
export P30_CLIENT_UID="$({ /usr/bin/id -u; })"
export P30_CLIENT_GID="$({ /usr/bin/id -g; })"
p30_run_logged_client_command 06-source-receipt-review \
  "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_RECEIPT_REVIEWER" \
  --receipt "$P30_LOCAL_SOURCE_RECEIPT" --phase source \
  --creation-summary \
  "$P30_CLIENT_LOG_ROOT/04-source-receipt-create.stdout.log" \
  --creation-status \
  "$P30_CLIENT_LOG_ROOT/04-source-receipt-create.exit-status.txt" \
  --expected-reviewer-source-sha256 "$P30_EXPECTED_RECEIPT_REVIEWER_SOURCE_SHA256" \
  --expected-reviewer-source-uid "$P30_CLIENT_UID" \
  --expected-reviewer-source-gid "$P30_CLIENT_GID" \
  --expected-receipt-uid "$P30_CLIENT_UID" \
  --expected-receipt-gid "$P30_CLIENT_GID" \
  --expected-receipt-sha256 "$P30_EXPECTED_SOURCE_RECEIPT_SHA256" \
  --expected-p30-commit "$P30_SOURCE_FREEZE_COMMIT" \
  --expected-p30-tree "$P30_SOURCE_FREEZE_TREE" \
  --expected-bundle-sha256 "$P30_EXPECTED_SOURCE_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" \
  --expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --expected-orchestrator-sha256 "$P30_EXPECTED_ORCHESTRATOR_SHA256" \
  --expected-p30-contract-sha256 "$P30_EXPECTED_CONTRACT_SHA256" \
  --expected-p30-reconstructor-sha256 "$P30_EXPECTED_RECONSTRUCTOR_SHA256"
"$P30_CLIENT_PYTHON" -I -S - \
  "$P30_CLIENT_LOG_ROOT/06-source-receipt-review.stdout.log" \
  "$P30_CLIENT_LOG_ROOT/06-source-receipt-review.exit-status.txt" \
  source "$P30_EXPECTED_RECEIPT_REVIEWER_SOURCE_SHA256" <<'PY'
import json, pathlib, sys
stdout_path, status_path = map(pathlib.Path, sys.argv[1:3])
phase, reviewer_sha256 = sys.argv[3:5]
if status_path.read_bytes() != b"0\n":
    raise SystemExit("receipt-review client status differs")
payload = json.loads(stdout_path.read_bytes())
if payload.get("status") != "passed" or payload.get("passes") is not True:
    raise SystemExit("receipt review did not pass")
if payload.get("phase") != phase or payload.get("safe_to_cross_next_boundary") is not True:
    raise SystemExit("receipt review did not authorize the exact next boundary")
if payload.get("reviewer_source_sha256") != reviewer_sha256:
    raise SystemExit("receipt review source binding differs")
PY
```

The review must exit zero and emit `"safe_to_cross_next_boundary": true`
before the root seal. Its source bytes and tests are hash-bound by the P30
contract and source commit.

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
`/var/lib/optimizationml-p30-20260906-03/run_p30_umask_bound_control_seal.sh`
as `root:root` mode `0555`. Its source and destination identities, hash, size,
mode, link count, and absence of ACL/xattr authority are rechecked before the
source descriptor closes. No phase may later execute the process-writable
control-checkout path. Both phase invocations use only
`/bin/bash -p /var/lib/optimizationml-p30-20260906-03/run_p30_umask_bound_control_seal.sh ...`;
privileged Bash mode prevents `BASH_ENV`, exported shell functions, `SHELLOPTS`,
and related ambient startup controls from replacing the reviewed code.

Use this no-follow, no-overwrite seal after exporting the independently
reviewed `P30_EXPECTED_ORCHESTRATOR_SHA256`:

```bash
set -euo pipefail
p30_run_logged_client_command 07-reviewed-root-seal \
  "${p30_strict_ssh[@]}" \
  /usr/bin/sudo -n /usr/bin/python3 -I -S - \
  "$P30_CONTROL_REPO/scripts/run_p30_umask_bound_control_seal.sh" \
  "$P30_LEDGER_ROOT" "$P30_EXPECTED_ORCHESTRATOR_SHA256" \
  1000 1000 0 0 <<'PY'
# P30_REVIEWED_ORCHESTRATOR_SEAL_BEGIN
import hashlib
import os
import pathlib
import stat
import sys

source = pathlib.PurePosixPath(sys.argv[1])
ledger = pathlib.PurePosixPath(sys.argv[2])
expected = sys.argv[3]
source_uid = int(sys.argv[4])
source_gid = int(sys.argv[5])
destination_uid = int(sys.argv[6])
destination_gid = int(sys.argv[7])
destination_name = "run_p30_umask_bound_control_seal.sh"
if len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected):
    raise SystemExit("P30 reviewed orchestrator hash is malformed")
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
        != (destination_uid, destination_gid, 0o700)
        or sorted(os.listdir(ledger_fd)) != ["deferred"]
    ):
        raise SystemExit("P30 pre-seal ledger authority or inventory differs")
    source_fd = os.open(
        source.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=source_parent
    )
    before = os.fstat(source_fd)
    if (
        not stat.S_ISREG(before.st_mode)
        or before.st_nlink != 1
        or (before.st_uid, before.st_gid, stat.S_IMODE(before.st_mode))
        != (source_uid, source_gid, 0o600)
    ):
        raise SystemExit("P30 reviewed orchestrator source authority differs")
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
        raise SystemExit("P30 reviewed orchestrator source has a forbidden ACL")
    raw = bytearray()
    while chunk := os.read(source_fd, 1024 * 1024):
        raw.extend(chunk)
    after = os.fstat(source_fd)
    by_name = os.stat(source.name, dir_fd=source_parent, follow_symlinks=False)
    if fields(before) != fields(after) or fields(after) != fields(by_name):
        raise SystemExit("P30 reviewed orchestrator changed during stable read")
    if hashlib.sha256(raw).hexdigest() != expected:
        raise SystemExit("P30 reviewed orchestrator hash differs")
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
            raise SystemExit("P30 sealed orchestrator write made no progress")
        view = view[written:]
    os.fchmod(destination_fd, 0o555)
    os.fsync(destination_fd)
    sealed = os.fstat(destination_fd)
    sealed_by_name = os.stat(destination_name, dir_fd=ledger_fd, follow_symlinks=False)
    if (
        not stat.S_ISREG(sealed.st_mode)
        or sealed.st_nlink != 1
        or (sealed.st_uid, sealed.st_gid, stat.S_IMODE(sealed.st_mode))
        != (destination_uid, destination_gid, 0o555)
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
        raise SystemExit("P30 sealed orchestrator authority differs")
    source_final = os.fstat(source_fd)
    source_by_name_final = os.stat(
        source.name, dir_fd=source_parent, follow_symlinks=False
    )
    if fields(after) != fields(source_final) or fields(source_final) != fields(
        source_by_name_final
    ):
        raise SystemExit("P30 reviewed orchestrator changed while sealing")
    os.fsync(ledger_fd)
finally:
    if destination_fd is not None:
        os.close(destination_fd)
    if source_fd is not None:
        os.close(source_fd)
    os.close(ledger_fd)
    os.close(source_parent)
# P30_REVIEWED_ORCHESTRATOR_SEAL_END
PY
```

Before host transfer, the portable orchestration test must create a control
checkout through the actual P30 verifier under umask `077`, prove that Git's
tracked `100644` launcher is physically mode `0600`, and execute this exact
marked seal program against it. The preflight must prove the positive
root-owned mode-`0555` copy and negative `0644`, symlink, hardlink, ACL, digest,
stable-stat, and preexisting-destination/O_EXCL cases. A synthetic file that did
not pass through the verifier checkout transition is not a sufficient
preflight.

## 4. Construct the authority and prepare one runtime

Only after source-receipt review, create the authority directly beneath the
transport root through the frozen client logger and strict SSH vector. A
direct construction failure is terminal and is never cleaned or retried:

```bash
set -euo pipefail
p30_run_logged_client_command 08-authority-checkout \
  "${p30_strict_ssh[@]}" /usr/bin/env -i \
  HOME=/home/ubuntu LANG=C LC_ALL=C PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  /bin/bash -s -- "$P30_AUTHORITY_REPO" "$P30_SOURCE_CLOSURE" \
  <<'P30_AUTHORITY_CHECKOUT'
set -euo pipefail
P30_AUTHORITY_REPO="$1"
P30_SOURCE_CLOSURE="$2"
umask 077
export PATH=/usr/sbin:/usr/bin:/sbin:/bin
unset PYTHONHOME PYTHONPATH PYTHONSTARTUP PYTHONINSPECT PYTHONUSERBASE
export PYTHONNOUSERSITE=1
export PYTHONDONTWRITEBYTECODE=1
export PYTHONSAFEPATH=1
for p30_git_variable in $(compgen -A variable GIT_ || true); do
  unset "$p30_git_variable"
done
unset p30_git_variable
export GIT_CONFIG_NOSYSTEM=1
export GIT_ATTR_NOSYSTEM=1
export GIT_CONFIG_SYSTEM=/dev/null
export GIT_CONFIG_GLOBAL=/dev/null
export GIT_NO_REPLACE_OBJECTS=1
export GIT_OPTIONAL_LOCKS=0
export GIT_TERMINAL_PROMPT=0
test ! -e "$P30_AUTHORITY_REPO"
/usr/bin/git init --initial-branch=p30-authority-unborn "$P30_AUTHORITY_REPO"
/usr/bin/git -C "$P30_AUTHORITY_REPO" config core.autocrlf false
/usr/bin/git -C "$P30_AUTHORITY_REPO" config core.filemode true
/usr/bin/git -C "$P30_AUTHORITY_REPO" config core.hooksPath /dev/null
/usr/bin/git -C "$P30_AUTHORITY_REPO" config core.fsmonitor false
/usr/bin/git -C "$P30_AUTHORITY_REPO" fetch --no-tags --no-write-fetch-head \
  "$P30_SOURCE_CLOSURE" \
  +refs/tags/p26-attempt02-acquisition-source:refs/tags/p26-attempt02-acquisition-source
/usr/bin/git -C "$P30_AUTHORITY_REPO" checkout --detach \
  185e444afc0b44ca0a09b1bde49a6b6fa3973355
/usr/bin/chmod 0700 "$P30_AUTHORITY_REPO"
p30_authority_head=$(/usr/bin/git -C "$P30_AUTHORITY_REPO" rev-parse HEAD)
test "$p30_authority_head" = \
  185e444afc0b44ca0a09b1bde49a6b6fa3973355
p30_authority_tree=$(/usr/bin/git -C "$P30_AUTHORITY_REPO" rev-parse HEAD^{tree})
test "$p30_authority_tree" = \
  24f4bdac331a57bd7c1b807747d7c7fba253ee5a
p30_authority_status=$(
  /usr/bin/git -C "$P30_AUTHORITY_REPO" status --porcelain=v1 --untracked-files=all
)
test -z "$p30_authority_status"
p30_authority_ignored=$(
  /usr/bin/git -C "$P30_AUTHORITY_REPO" ls-files --others --ignored --exclude-standard
)
test -z "$p30_authority_ignored"
test ! -e "$P30_AUTHORITY_REPO/.git/shallow"
test ! -e "$P30_AUTHORITY_REPO/.git/objects/info/alternates"
test ! -e "$P30_AUTHORITY_REPO/.git/info/grafts"
p30_authority_replacements=$(
  /usr/bin/git -C "$P30_AUTHORITY_REPO" for-each-ref --format='%(refname)' refs/replace
)
test -z "$p30_authority_replacements"
if p30_partial_clone=$(
  /usr/bin/git -C "$P30_AUTHORITY_REPO" config --local --get extensions.partialClone
); then
  test -z "$p30_partial_clone"
else
  test "$?" -eq 1
fi
if p30_promisor=$(
  /usr/bin/git -C "$P30_AUTHORITY_REPO" config --local --get-regexp '^remote\..*\.promisor$'
); then
  test -z "$p30_promisor"
else
  test "$?" -eq 1
fi
if p30_local_includes=$(
  /usr/bin/git -C "$P30_AUTHORITY_REPO" config --local --get-regexp '^include.*\.'
); then
  test -z "$p30_local_includes"
else
  test "$?" -eq 1
fi
/usr/bin/git -C "$P30_AUTHORITY_REPO" fsck --full --strict --no-dangling
P30_AUTHORITY_CHECKOUT
```

Export every reviewed P30 source, receipt, verifier, orchestrator,
reconstructor, terminal-P29, unchanged-P28, and unchanged-P27 hash required by the host
orchestrator, plus the pinned inputs:

```bash
export P30_GPU_UUID=GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d
export P30_NANOGPT_HOST=/secure/p23/nanoGPT
export P30_MUON_HOST=/secure/p23/Muon
export P30_DATA_HOST=/secure/p23/optimizationml-p22-data
```

The sealed host orchestrator must run as numeric UID/GID `0:0`; the bootstrap
bundle verifier separately remains UID/GID `1000:1000` for transport/control
mutations. The orchestrator binds the
nanoGPT checkout to commit `3adf61e154c3fe3fca428ad6bc3818b27a3b8291`,
tree `ca93bcd9b9c9ff32d3016e1e2556644e68bef86a`, and the Muon checkout to
commit `f98f1cacc0263b04290753e32be8d498c1efc806`, tree
`4ea5cd8ab6ebd56a18536f06453619efcd636da0`; clean means no tracked,
untracked, or ignored dirt. Do not rely on `sudo` environment inheritance or
`--preserve-env`: default `sudoers` policy commonly drops the required `P30_*`
values. Instead, pass exactly the preregistered source variables through a
clean `/usr/bin/env -i` argument vector:

```bash
p30_source_root_environment=(
  "P30_ATTEMPT_ID=$P30_ATTEMPT_ID"
  "P30_GPU_UUID=$P30_GPU_UUID"
  "P30_AUTHORITY_REPO=$P30_AUTHORITY_REPO"
  "P30_NANOGPT_HOST=$P30_NANOGPT_HOST"
  "P30_MUON_HOST=$P30_MUON_HOST"
  "P30_DATA_HOST=$P30_DATA_HOST"
  "P30_SOURCE_FREEZE_COMMIT=$P30_SOURCE_FREEZE_COMMIT"
  "P30_SOURCE_FREEZE_TREE=$P30_SOURCE_FREEZE_TREE"
  "P30_EXPECTED_SOURCE_BUNDLE_SHA256=$P30_EXPECTED_SOURCE_BUNDLE_SHA256"
  "P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT=$P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT"
  "P30_EXPECTED_SOURCE_RECEIPT_SHA256=$P30_EXPECTED_SOURCE_RECEIPT_SHA256"
  "P30_EXPECTED_CONTRACT_SHA256=$P30_EXPECTED_CONTRACT_SHA256"
  "P30_EXPECTED_ORCHESTRATOR_SHA256=$P30_EXPECTED_ORCHESTRATOR_SHA256"
  "P30_EXPECTED_RECONSTRUCTOR_SHA256=$P30_EXPECTED_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_BUNDLE_VERIFIER_SHA256=$P30_EXPECTED_BUNDLE_VERIFIER_SHA256"
  "P30_EXPECTED_P29_CONTRACT_SHA256=$P30_EXPECTED_P29_CONTRACT_SHA256"
  "P30_EXPECTED_P29_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P29_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P29_OUTCOME_SHA256=$P30_EXPECTED_P29_OUTCOME_SHA256"
  "P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P28_CONTRACT_SHA256=$P30_EXPECTED_P28_CONTRACT_SHA256"
  "P30_EXPECTED_P28_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P28_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P28_OUTCOME_SHA256=$P30_EXPECTED_P28_OUTCOME_SHA256"
  "P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P27_CONTRACT_SHA256=$P30_EXPECTED_P27_CONTRACT_SHA256"
  "P30_EXPECTED_P27_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P27_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P27_LOCALIZER_SHA256=$P30_EXPECTED_P27_LOCALIZER_SHA256"
  "P30_EXPECTED_INGESTER_SHA256=$P30_EXPECTED_INGESTER_SHA256"
  "P30_EXPECTED_P23_CORE_SHA256=$P30_EXPECTED_P23_CORE_SHA256"
  "P30_EXPECTED_P27_SANITIZER_SHA256=$P30_EXPECTED_P27_SANITIZER_SHA256"
)
p30_verify_root_environment_handoff() {
  local label="$1"
  local expected_count="$2"
  shift 2
  test "$#" -eq "$expected_count"
  p30_run_logged_client_command "$label" \
    "${p30_strict_ssh[@]}" /usr/bin/sudo -n /usr/bin/env -i \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    "$@" \
    /usr/bin/python3 -I -S - "$expected_count" "$@" \
    <<'P30_ROOT_ENVIRONMENT_CHECK'
import os,sys
count=int(sys.argv[1]); pairs=sys.argv[2:]
if len(pairs) != count: raise SystemExit("P30 handoff count differs")
expected={}
for pair in pairs:
    name, separator, value = pair.partition("=")
    if separator != "=" or not name.startswith("P30_") or not value or name in expected:
        raise SystemExit("P30 handoff assignment differs")
    expected[name]=value
actual={name:value for name,value in os.environ.items() if name.startswith("P30_")}
if actual != expected or os.geteuid() != 0 or os.getegid() != 0:
    raise SystemExit("P30 root handoff differs")
P30_ROOT_ENVIRONMENT_CHECK
}
p30_verify_root_environment_handoff \
  09-source-root-handoff 29 "${p30_source_root_environment[@]}"
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
set -euo pipefail
p30_run_logged_client_command 10-prepare-runtime \
  "${p30_strict_ssh[@]}" /usr/bin/sudo -n /usr/bin/env -i \
  PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  "${p30_source_root_environment[@]}" \
  /bin/bash -p \
  /var/lib/optimizationml-p30-20260906-03/run_p30_umask_bound_control_seal.sh \
  prepare-runtime
```

These assignments precede `/bin/bash -p` and the sealed script in the root
process argument vector. The script revalidates all 29 values after its
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
are `.deferred-p30-prepare-pre-attempt-admission.stdout.log`,
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
`p30-prepare-attempt-layout` command writes its three O_EXCL transcripts to the
root-only deferred journal. The subsequent publication into the newly created
evidence directory best-effort creates and seals its own journal stdout,
stderr, and exit status. On successful finalization both stdout/stderr are
empty and the status is exactly
`0\n`; the journal then has the three admission files plus exactly the six
preregistered
`.deferred-p30-prepare-attempt-layout{,.publication}.{stdout.log,stderr.log,exit-status.txt}`
files (using the literal filename combinations listed by the contract) and no
failure sentinel. Publication failure best-effort adds the preregistered
`.publication-failure.exit-status.txt` sentinel and permanently ends P30; a
status or seal failure may leave an incomplete journal and returns terminal
125 rather than claiming the complete inventory.

The orchestrator creates `/secure/p30/attempt-20260906-03` with privilege only
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
execution authority at `/secure/p30/execution-20260906-03`. This is not a
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
`passive-muon-p30-immutable-execution-snapshot-v1`, and is canonical UTF-8 JSON
(`sort_keys=True`, compact separators, one trailing newline). Its sorted
`entries` array covers every relative path other than the manifest itself.
Every entry binds relative path, kind, mode, UID, and GID; file entries also
bind byte count and SHA-256. Its `source_authorities` bind all three commit/tree
pairs, the original data-manifest hash/size, every helper hash, and the P26
native hash/size.

The read-only adjunct `data/.p30-tools/` contains exactly the unchanged P27
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
It creates one `p30-localization-20260906-03` container using the unchanged
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
`p30-fresh-container-id-validation.{stdout.log,stderr.log,exit-status.txt}`
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
| `/secure/p30/execution-20260906-03/repository` | `/workspace/OptimizationML` | read-only |
| `/secure/p30/execution-20260906-03/nanogpt` | `/workspace/inputs/nanoGPT` | read-only |
| `/secure/p30/execution-20260906-03/muon` | `/workspace/inputs/muon` | read-only |
| `/secure/p30/execution-20260906-03/data` | `/private/tmp/optimizationml-p22-data` | read-only |
| `/secure/p30/execution-20260906-03/repository/experiments/training/materialize_p22_fineweb.py` | `/Users/harry/Desktop/temp/OptimizationML/experiments/training/materialize_p22_fineweb.py` | read-only |
| `/secure/p30/attempt-20260906-03/evidence` | `/workspace/evidence/p23` | read-write |
| `/secure/p30/attempt-20260906-03/p30-image-inspect.json` | `/mounted-host-evidence/image-inspect.json` | read-only |
| `/secure/p30/attempt-20260906-03/p30-running-container-inspect.json` | `/mounted-host-evidence/running-container-inspect.json` | read-only |
| `/secure/p30/attempt-20260906-03/p30-running-mountinfo.txt` | `/mounted-host-evidence/running-mountinfo.txt` | read-only |
| `/secure/p30/attempt-20260906-03/p30-nvidia-smi.csv` | `/mounted-host-evidence/nvidia-smi.csv` | read-only |

Review these two new retained artifacts:

```text
experiments/training/p30_cuda_runtime_lock.json
experiments/training/p30_host_attestation.json
```

The attempt root itself has exactly `evidence`, `p30-image-inspect.json`,
`p30-nvidia-smi.csv`, `p30-running-container-inspect.json`,
`p30-running-mountinfo.txt`, and `p30-container-id.txt`; no other top-level
entry is permitted before runtime-review receipt publication or replay.

The exact pre-localization runtime-receipt evidence allowlist is these 27
files—no more and no fewer:

```text
p30-prepare-attempt-layout.stdout.log
p30-prepare-attempt-layout.stderr.log
p30-prepare-attempt-layout.exit-status.txt
p30-prepare-execution-snapshot-verification.stdout.log
p30-prepare-execution-snapshot-verification.stderr.log
p30-prepare-execution-snapshot-verification.exit-status.txt
p30-image-inspection.stderr.log
p30-image-inspection.exit-status.txt
p30-nvidia-smi.stderr.log
p30-nvidia-smi.exit-status.txt
p30-container-launch.stderr.log
p30-container-launch.exit-status.txt
p30-fresh-container-id-validation.stdout.log
p30-fresh-container-id-validation.stderr.log
p30-fresh-container-id-validation.exit-status.txt
p30-running-container-inspection.stderr.log
p30-running-container-inspection.exit-status.txt
p30-running-mountinfo.stderr.log
p30-running-mountinfo.exit-status.txt
p30-freeze-runtime.stdout.log
p30-freeze-runtime.stderr.log
p30-freeze-runtime.exit-status.txt
p30-post-freeze-attempt-layout.stdout.log
p30-post-freeze-attempt-layout.stderr.log
p30-post-freeze-attempt-layout.exit-status.txt
p30_cuda_runtime_lock.json
p30_host_attestation.json
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
set -euo pipefail
export P30_RUNTIME_REVIEW_WORKTREE="$P30_LOCAL_REPOSITORY"
test "$P30_RUNTIME_REVIEW_WORKTREE" = "$P30_LOCAL_REPOSITORY"
cd "$P30_RUNTIME_REVIEW_WORKTREE"
test "$({ /bin/pwd -P; })" = "$P30_RUNTIME_REVIEW_WORKTREE"
test "$({ /usr/bin/git symbolic-ref --short HEAD; })" = \
  p30-umask-bound-control-seal
test "$({ /usr/bin/git rev-parse HEAD; })" = "$P30_SOURCE_FREEZE_COMMIT"
test -z "$({ /usr/bin/git status --porcelain=v1 --untracked-files=all; })"
p30_o_excl_download \
  11-runtime-lock-download \
  /secure/p30/attempt-20260906-03/evidence/p30_cuda_runtime_lock.json \
  "$P30_RUNTIME_REVIEW_WORKTREE/experiments/training/p30_cuda_runtime_lock.json"
p30_o_excl_download \
  12-host-attestation-download \
  /secure/p30/attempt-20260906-03/evidence/p30_host_attestation.json \
  "$P30_RUNTIME_REVIEW_WORKTREE/experiments/training/p30_host_attestation.json"
/usr/bin/cmp \
  "$P30_CLIENT_LOG_ROOT/11-runtime-lock-download.stdout.log" \
  experiments/training/p30_cuda_runtime_lock.json
/usr/bin/cmp \
  "$P30_CLIENT_LOG_ROOT/12-host-attestation-download.stdout.log" \
  experiments/training/p30_host_attestation.json
/usr/bin/shasum -a 256 \
  experiments/training/p30_cuda_runtime_lock.json \
  experiments/training/p30_host_attestation.json
```

Confirm local and retained SHA-256 values match, then commit only those two
files. The commit and every postcondition are fail-closed. The runtime-review
commit must be one direct child of `P30_SOURCE_FREEZE_COMMIT`, contain exactly
the two declared additions at Git mode `100644`, and leave the checkout clean:

```bash
set -euo pipefail
p30_local_git add -- \
  experiments/training/p30_cuda_runtime_lock.json \
  experiments/training/p30_host_attestation.json
test "$({ p30_local_git diff --cached --name-status; })" = "$({
  /usr/bin/printf '%s\n' \
    $'A\texperiments/training/p30_cuda_runtime_lock.json' \
    $'A\texperiments/training/p30_host_attestation.json'
})"
p30_local_git diff --cached --check
test "$({
  p30_local_git ls-files -s -- \
    experiments/training/p30_cuda_runtime_lock.json \
    experiments/training/p30_host_attestation.json | \
    /usr/bin/awk '{print $1}' | /usr/bin/sort -u
})" = 100644
p30_local_git commit -m \
  "Freeze reviewed P30 CUDA runtime identity before localization"
export P30_RUNTIME_REVIEW_COMMIT="$({ p30_local_git rev-parse HEAD; })"
export P30_RUNTIME_REVIEW_TREE="$({ p30_local_git rev-parse 'HEAD^{tree}'; })"
test "$({ p30_local_git rev-parse 'HEAD^'; })" = "$P30_SOURCE_FREEZE_COMMIT"
test "$({ p30_local_git rev-list --parents -n 1 HEAD | /usr/bin/awk '{print NF}'; })" = 2
test "$({ p30_local_git diff-tree --no-commit-id --name-status -r HEAD^ HEAD; })" = "$({
  /usr/bin/printf '%s\n' \
    $'A\texperiments/training/p30_cuda_runtime_lock.json' \
    $'A\texperiments/training/p30_host_attestation.json'
})"
test -z "$({ p30_local_git status --porcelain=v1 --untracked-files=all; })"
set -o pipefail
p30_local_git cat-file blob \
  "$P30_RUNTIME_REVIEW_COMMIT:experiments/training/p30_cuda_runtime_lock.json" | \
  /usr/bin/cmp - experiments/training/p30_cuda_runtime_lock.json
p30_local_git cat-file blob \
  "$P30_RUNTIME_REVIEW_COMMIT:experiments/training/p30_host_attestation.json" | \
  /usr/bin/cmp - experiments/training/p30_host_attestation.json
```

Record the exact artifact bindings after those checks pass:

```bash
export P30_EXPECTED_RUNTIME_LOCK_SHA256="$({
  /usr/bin/shasum -a 256 \
    experiments/training/p30_cuda_runtime_lock.json | /usr/bin/awk '{print $1}'
})"
export P30_EXPECTED_HOST_ATTESTATION_SHA256="$({
  /usr/bin/shasum -a 256 \
    experiments/training/p30_host_attestation.json | /usr/bin/awk '{print $1}'
})"
export P30_RUNTIME_LOCK_BYTE_COUNT="$({
  /usr/bin/wc -c < experiments/training/p30_cuda_runtime_lock.json | /usr/bin/tr -d ' '
})"
export P30_HOST_ATTESTATION_BYTE_COUNT="$({
  /usr/bin/wc -c < experiments/training/p30_host_attestation.json | /usr/bin/tr -d ' '
})"
test "${#P30_EXPECTED_RUNTIME_LOCK_SHA256}" -eq 64
test "${#P30_EXPECTED_HOST_ATTESTATION_SHA256}" -eq 64
test "$P30_RUNTIME_LOCK_BYTE_COUNT" -gt 0
test "$P30_HOST_ATTESTATION_BYTE_COUNT" -gt 0
export P30_EXPECTED_CONTAINER_ID="$({
  "$P30_CLIENT_PYTHON" -I -S - \
    experiments/training/p30_cuda_runtime_lock.json \
    experiments/training/p30_host_attestation.json \
    "$P30_EXPECTED_RUNTIME_LOCK_SHA256" \
    "$P30_EXPECTED_HOST_ATTESTATION_SHA256" <<'PY'
import hashlib
import json
import os
import pathlib
import re
import stat
import sys

lock_path = pathlib.Path(sys.argv[1])
attestation_path = pathlib.Path(sys.argv[2])
expected_lock_sha256 = sys.argv[3]
expected_attestation_sha256 = sys.argv[4]


def reject_constant(value):
    raise ValueError(f"non-finite JSON constant: {value}")


def reject_duplicates(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def stable_json(path):
    before = os.lstat(path)
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    try:
        opened = os.fstat(descriptor)
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    by_name = os.stat(path, follow_symlinks=False)
    fields = lambda value: (
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
    if fields(before) != fields(opened) or fields(opened) != fields(after) or fields(after) != fields(by_name):
        raise SystemExit("runtime JSON changed during stable read")
    if not stat.S_ISREG(after.st_mode) or after.st_nlink != 1 or stat.S_IMODE(after.st_mode) != 0o600:
        raise SystemExit("runtime JSON authority differs")
    if (after.st_uid, after.st_gid) != (os.getuid(), os.getgid()):
        raise SystemExit("runtime JSON owner differs")
    raw = b"".join(chunks)
    parsed = json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_constant,
    )
    return raw, parsed


lock_raw, lock = stable_json(lock_path)
attestation_raw, attestation = stable_json(attestation_path)
if hashlib.sha256(lock_raw).hexdigest() != expected_lock_sha256:
    raise SystemExit("runtime-lock digest differs")
if hashlib.sha256(attestation_raw).hexdigest() != expected_attestation_sha256:
    raise SystemExit("host-attestation digest differs")
if lock.get("schema_version") != "passive-muon-p23-cuda-runtime-lock-v3":
    raise SystemExit("runtime-lock schema differs")
if lock.get("status") != "pinned_for_acquisition":
    raise SystemExit("runtime-lock status differs")
if attestation.get("schema_version") != "passive-muon-p23-host-attestation-v3":
    raise SystemExit("host-attestation schema differs")
if attestation.get("status") != "procedurally_host_attested":
    raise SystemExit("host-attestation status differs")
lock_container = lock.get("container")
attestation_container = attestation.get("container")
if not isinstance(lock_container, dict) or not isinstance(attestation_container, dict):
    raise SystemExit("runtime container record differs")
container_id = lock_container.get("container_id")
if (
    not isinstance(container_id, str)
    or re.fullmatch(r"[0-9a-f]{64}", container_id) is None
    or attestation_container.get("container_id") != container_id
    or lock_container.get("host_attestation_sha256") != expected_attestation_sha256
):
    raise SystemExit("runtime container identity cross-binding differs")
print(container_id)
PY
})"
test "${#P30_EXPECTED_CONTAINER_ID}" -eq 64
case "$P30_EXPECTED_CONTAINER_ID" in *[!0-9a-f]*) exit 1 ;; esac
```

```text
A	experiments/training/p30_cuda_runtime_lock.json
A	experiments/training/p30_host_attestation.json
```

Record `P30_RUNTIME_REVIEW_COMMIT` and `P30_RUNTIME_REVIEW_TREE`. Create the
second full bundle with the same nine refs and only the P30 branch advanced:

Before the run phase, also export the reviewed runtime-only values, including
`P30_EXPECTED_RUNTIME_LOCK_SHA256`,
`P30_EXPECTED_HOST_ATTESTATION_SHA256`, and
`P30_EXPECTED_CONTAINER_ID`. The latter is cross-derived from the two exact
downloaded runtime JSONs only after their schemas, statuses, owners, modes,
stable identities, mutual container ID, and host-attestation digest binding
pass. The sealed host orchestrator later independently requires that exact
64-lowercase-hex value to match the root-owned `p30-container-id.txt`, both
reviewed JSONs, and live Docker inspection. It is never inferred from a
truncated Docker name or supplied before fresh-container validation succeeds.

```bash
export P30_LOCAL_RUNTIME_BUNDLE="$P30_LOCAL_TRANSPORT/p30_runtime_review.bundle"
export P30_RUNTIME_REVIEW_BUNDLE=/secure/p30/transport-20260906-03/p30_runtime_review.bundle
export P30_RUNTIME_REVIEW_CLOSURE=/secure/p30/transport-20260906-03/p30_runtime_review_closure.git
export P30_RUNTIME_REVIEW_RECEIPT=/secure/p30/transport-20260906-03/p30_runtime_review_bundle_receipt.json
p30_local_git bundle create "$P30_LOCAL_RUNTIME_BUNDLE" \
  refs/heads/p30-umask-bound-control-seal \
  refs/heads/p29-permission-safe-bundle-localization-bridge \
  refs/tags/p29-orchestrator-source-mode-diagnostic \
  refs/heads/p28-bundle-complete-localization-bridge \
  refs/tags/p28-control-parent-permission-diagnostic \
  refs/heads/p27-cuda-deleted-mapping-localization \
  refs/tags/p27-cuda-deleted-mapping-localization-diagnostic \
  refs/tags/p26-permission-safe-acquisition-checkpoint \
  refs/tags/p26-attempt02-acquisition-source
P30_LOCAL_RUNTIME_BUNDLE_BINDING="$({
  "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_LOCAL_BUNDLE_VERIFIER" \
    --phase runtime-review \
    --bundle "$P30_LOCAL_RUNTIME_BUNDLE" \
    --git-executable /usr/bin/git \
    --expected-p30-commit "$P30_RUNTIME_REVIEW_COMMIT" \
    --expected-p30-tree "$P30_RUNTIME_REVIEW_TREE" \
    --expected-p30-parent "$P30_SOURCE_FREEZE_COMMIT" \
    --expected-verifier-source-sha256 \
      "$P30_EXPECTED_LOCAL_BUNDLE_VERIFIER_SOURCE_SHA256"
})"
export P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256="$({
  p30_local_bundle_binding_field \
    "$P30_LOCAL_RUNTIME_BUNDLE_BINDING" runtime-review \
    "$P30_RUNTIME_REVIEW_COMMIT" "$P30_RUNTIME_REVIEW_TREE" bundle_sha256
})"
export P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT="$({
  p30_local_bundle_binding_field \
    "$P30_LOCAL_RUNTIME_BUNDLE_BINDING" runtime-review \
    "$P30_RUNTIME_REVIEW_COMMIT" "$P30_RUNTIME_REVIEW_TREE" bundle_byte_count
})"
p30_o_excl_upload \
  13-runtime-review-bundle-upload \
  "$P30_LOCAL_RUNTIME_BUNDLE" "$P30_RUNTIME_REVIEW_BUNDLE" 0600 \
  "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" \
  "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT"
```

The runtime-review verifier repeats full closure in a second fresh empty bare
repository, proves the exact direct-child delta and retained-artifact byte
equality, advances the control checkout, executes P30 and all five nested
reconstructors only from authenticated private bundle bytes,
proves the immutable execution snapshot and exact 27-file preparation evidence,
proves no localization evidence exists, and writes a second external O_EXCL
receipt. The UID-1000 verifier deliberately makes no claim that it can inspect
the root-only ledger or deferred journal. The expected prepare-token inventory
is an operator precondition for this external receipt-creation invocation; the
sealed root orchestrator mechanically binds the token inventories around its
later verifier replay invocation. After the bundle
upload, run this once from the client through the frozen logger and strict SSH
vector:

It fd-anchors the attempt and evidence roots with componentwise no-follow
opens before inspecting inventory, then revalidates their complete
owner/mode/ACL/device/inode/mount identity after Git, byte-comparison, and
receipt handling. Receipt replay repeats the same checks and does not mutate
either directory.

```bash
set -euo pipefail
p30_run_logged_client_command 14-runtime-receipt-create \
  "${p30_strict_ssh[@]}" /usr/bin/env -i \
  HOME=/home/ubuntu LANG=C LC_ALL=C PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 \
  /usr/bin/python3 -I -S - \
  "$P30_TRANSPORT_ROOT/p30-runtime-review-receipt-creation.stdout.log" \
  "$P30_TRANSPORT_ROOT/p30-runtime-review-receipt-creation.stderr.log" \
  "$P30_TRANSPORT_ROOT/p30-runtime-review-receipt-creation.exit-status.txt" \
  /usr/bin/python3 -I -S "$P30_BOOTSTRAP_VERIFIER" \
  --phase runtime-review \
  --transport-root "$P30_TRANSPORT_ROOT" \
  --bundle "$P30_RUNTIME_REVIEW_BUNDLE" \
  --expected-bundle-sha256 "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" \
  --closure-repository "$P30_RUNTIME_REVIEW_CLOSURE" \
  --control-checkout "$P30_CONTROL_REPO" \
  --attempt-root "$P30_ATTEMPT_ROOT" \
  --execution-root "$P30_EXECUTION_ROOT" \
  --expected-p30-commit "$P30_RUNTIME_REVIEW_COMMIT" \
  --expected-p30-tree "$P30_RUNTIME_REVIEW_TREE" \
  --expected-source-commit "$P30_SOURCE_FREEZE_COMMIT" \
  --expected-source-tree "$P30_SOURCE_FREEZE_TREE" \
  --bootstrap-verifier "$P30_BOOTSTRAP_VERIFIER" \
  --expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --expected-orchestrator-sha256 "$P30_EXPECTED_ORCHESTRATOR_SHA256" \
  --receipt-output "$P30_RUNTIME_REVIEW_RECEIPT" <<'PY'
import os
import stat
import subprocess
import sys

paths = sys.argv[1:4]
command = sys.argv[4:]
flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
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
    if result_code == 0:
        summary_size = os.fstat(descriptors[0]).st_size
        summary = os.pread(descriptors[0], summary_size, 0)
        if len(summary) != summary_size or not summary:
            raise SystemExit("runtime receipt summary could not be replayed")
        sys.stdout.buffer.write(summary)
        sys.stdout.buffer.flush()
finally:
    for descriptor in descriptors:
        os.close(descriptor)
raise SystemExit(result_code)
PY
```

When wrapper setup and finalization succeed, these three transcripts retain
either verifier success or verifier failure under the transport root; they are
never copied into `evidence`. A wrapper O_EXCL, write, fsync, or close failure
may leave a partial transport transcript set and is terminal. Thus the later
runtime-receipt replay still observes the exact frozen 27-file evidence
inventory. A nonzero or incomplete logged invocation may not be rerun.

Run the same frozen independent reviewer before constructing the 37-value
localization handoff. Download the exact remote receipt with `O_EXCL`, require
it to byte-match the receipt-download command's frozen logger stdout, and review that
client-owned mode-`0600` copy locally. The artifact byte counts are the
independently retained native file sizes:

```bash
set -euo pipefail
export P30_LOCAL_RUNTIME_REVIEW_RECEIPT="$P30_LOCAL_TRANSPORT/p30_runtime_review_bundle_receipt.json"
p30_o_excl_download \
  15-runtime-receipt-download \
  "$P30_RUNTIME_REVIEW_RECEIPT" "$P30_LOCAL_RUNTIME_REVIEW_RECEIPT"
export P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256="$({
  /usr/bin/shasum -a 256 "$P30_LOCAL_RUNTIME_REVIEW_RECEIPT" | \
    /usr/bin/awk '{print $1}'
})"
test "${#P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256}" -eq 64
p30_run_logged_client_command 16-runtime-receipt-review \
  "$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_RECEIPT_REVIEWER" \
  --receipt "$P30_LOCAL_RUNTIME_REVIEW_RECEIPT" --phase runtime-review \
  --creation-summary \
  "$P30_CLIENT_LOG_ROOT/14-runtime-receipt-create.stdout.log" \
  --creation-status \
  "$P30_CLIENT_LOG_ROOT/14-runtime-receipt-create.exit-status.txt" \
  --expected-reviewer-source-sha256 "$P30_EXPECTED_RECEIPT_REVIEWER_SOURCE_SHA256" \
  --expected-reviewer-source-uid "$P30_CLIENT_UID" \
  --expected-reviewer-source-gid "$P30_CLIENT_GID" \
  --expected-receipt-uid "$P30_CLIENT_UID" \
  --expected-receipt-gid "$P30_CLIENT_GID" \
  --expected-receipt-sha256 "$P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256" \
  --expected-p30-commit "$P30_RUNTIME_REVIEW_COMMIT" \
  --expected-p30-tree "$P30_RUNTIME_REVIEW_TREE" \
  --expected-bundle-sha256 "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" \
  --expected-bundle-byte-count "$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" \
  --expected-verifier-sha256 "$P30_EXPECTED_BUNDLE_VERIFIER_SHA256" \
  --expected-orchestrator-sha256 "$P30_EXPECTED_ORCHESTRATOR_SHA256" \
  --expected-p30-contract-sha256 "$P30_EXPECTED_CONTRACT_SHA256" \
  --expected-p30-reconstructor-sha256 "$P30_EXPECTED_RECONSTRUCTOR_SHA256" \
  --expected-source-commit "$P30_SOURCE_FREEZE_COMMIT" \
  --expected-source-tree "$P30_SOURCE_FREEZE_TREE" \
  --expected-runtime-lock-sha256 "$P30_EXPECTED_RUNTIME_LOCK_SHA256" \
  --expected-runtime-lock-byte-count "$P30_RUNTIME_LOCK_BYTE_COUNT" \
  --expected-host-attestation-sha256 "$P30_EXPECTED_HOST_ATTESTATION_SHA256" \
  --expected-host-attestation-byte-count "$P30_HOST_ATTESTATION_BYTE_COUNT"
"$P30_CLIENT_PYTHON" -I -S - \
  "$P30_CLIENT_LOG_ROOT/16-runtime-receipt-review.stdout.log" \
  "$P30_CLIENT_LOG_ROOT/16-runtime-receipt-review.exit-status.txt" \
  runtime-review "$P30_EXPECTED_RECEIPT_REVIEWER_SOURCE_SHA256" <<'PY'
import json, pathlib, sys
stdout_path, status_path = map(pathlib.Path, sys.argv[1:3])
phase, reviewer_sha256 = sys.argv[3:5]
if status_path.read_bytes() != b"0\n":
    raise SystemExit("receipt-review client status differs")
payload = json.loads(stdout_path.read_bytes())
if payload.get("status") != "passed" or payload.get("passes") is not True:
    raise SystemExit("receipt review did not pass")
if payload.get("phase") != phase or payload.get("safe_to_cross_next_boundary") is not True:
    raise SystemExit("receipt review did not authorize the exact next boundary")
if payload.get("reviewer_source_sha256") != reviewer_sha256:
    raise SystemExit("receipt review source binding differs")
PY
```

This review must also exit zero and emit the explicit safe-to-cross result.

The other eight ref objects must remain unchanged between receipts. Never commit
either receipt into the history it authenticates.

## 6. Run the sole unchanged P27 localization

Pre-execution review found that P30 seals the localizer's retained native JSON
as `root:root` mode `0444`, while the unchanged P27 sanitizer deliberately
accepts only a private `root:root` mode-`0600` input. The reviewed P30
orchestrator therefore performs one narrow, mode-only compatibility bridge
inside the sanitizer producer: it binds the sealed native file by nofollow
identity, SHA-256, and byte count; requires the initially bound device and
inode; changes only its mode to `0600` (with the expected `ctime` change but no
payload, inode, owner, link-count, size, or `mtime` change); invokes the
unchanged hash-pinned sanitizer; and then attempts to reseal the same inode to
`0444` before honoring the sanitizer status. Success additionally requires the
resealed SHA-256 and byte count to match. No P27 source, schema, validation, or
classification rule is changed. This recorded correction supersedes the
freeze-time assumption that a sealed `0444` input could be passed directly to
the P27 sanitizer; all earlier source-freeze tags remain historical.

After independent runtime-receipt review, extend the exact clean root handoff
with the eight runtime-only values, then run exactly once:

```bash
p30_verify_root_environment_handoff() {
  local label="$1"
  local expected_count="$2"
  shift 2
  test "$#" -eq "$expected_count"
  p30_run_logged_client_command "$label" \
    "${p30_strict_ssh[@]}" /usr/bin/sudo -n /usr/bin/env -i \
    PATH=/usr/sbin:/usr/bin:/sbin:/bin \
    "$@" \
    /usr/bin/python3 -I -S - "$expected_count" "$@" \
    <<'P30_ROOT_ENVIRONMENT_CHECK'
import os,sys
count=int(sys.argv[1]); pairs=sys.argv[2:]
if len(pairs) != count: raise SystemExit("P30 handoff count differs")
expected={}
for pair in pairs:
    name, separator, value = pair.partition("=")
    if separator != "=" or not name.startswith("P30_") or not value or name in expected:
        raise SystemExit("P30 handoff assignment differs")
    expected[name]=value
actual={name:value for name,value in os.environ.items() if name.startswith("P30_")}
if actual != expected or os.geteuid() != 0 or os.getegid() != 0:
    raise SystemExit("P30 root handoff differs")
P30_ROOT_ENVIRONMENT_CHECK
}
p30_localization_root_environment=(
  "P30_ATTEMPT_ID=$P30_ATTEMPT_ID"
  "P30_GPU_UUID=$P30_GPU_UUID"
  "P30_AUTHORITY_REPO=$P30_AUTHORITY_REPO"
  "P30_NANOGPT_HOST=$P30_NANOGPT_HOST"
  "P30_MUON_HOST=$P30_MUON_HOST"
  "P30_DATA_HOST=$P30_DATA_HOST"
  "P30_SOURCE_FREEZE_COMMIT=$P30_SOURCE_FREEZE_COMMIT"
  "P30_SOURCE_FREEZE_TREE=$P30_SOURCE_FREEZE_TREE"
  "P30_EXPECTED_SOURCE_BUNDLE_SHA256=$P30_EXPECTED_SOURCE_BUNDLE_SHA256"
  "P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT=$P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT"
  "P30_EXPECTED_SOURCE_RECEIPT_SHA256=$P30_EXPECTED_SOURCE_RECEIPT_SHA256"
  "P30_EXPECTED_CONTRACT_SHA256=$P30_EXPECTED_CONTRACT_SHA256"
  "P30_EXPECTED_ORCHESTRATOR_SHA256=$P30_EXPECTED_ORCHESTRATOR_SHA256"
  "P30_EXPECTED_RECONSTRUCTOR_SHA256=$P30_EXPECTED_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_BUNDLE_VERIFIER_SHA256=$P30_EXPECTED_BUNDLE_VERIFIER_SHA256"
  "P30_EXPECTED_P29_CONTRACT_SHA256=$P30_EXPECTED_P29_CONTRACT_SHA256"
  "P30_EXPECTED_P29_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P29_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P29_OUTCOME_SHA256=$P30_EXPECTED_P29_OUTCOME_SHA256"
  "P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P29_OUTCOME_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P28_CONTRACT_SHA256=$P30_EXPECTED_P28_CONTRACT_SHA256"
  "P30_EXPECTED_P28_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P28_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P28_OUTCOME_SHA256=$P30_EXPECTED_P28_OUTCOME_SHA256"
  "P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P27_CONTRACT_SHA256=$P30_EXPECTED_P27_CONTRACT_SHA256"
  "P30_EXPECTED_P27_RECONSTRUCTOR_SHA256=$P30_EXPECTED_P27_RECONSTRUCTOR_SHA256"
  "P30_EXPECTED_P27_LOCALIZER_SHA256=$P30_EXPECTED_P27_LOCALIZER_SHA256"
  "P30_EXPECTED_INGESTER_SHA256=$P30_EXPECTED_INGESTER_SHA256"
  "P30_EXPECTED_P23_CORE_SHA256=$P30_EXPECTED_P23_CORE_SHA256"
  "P30_EXPECTED_P27_SANITIZER_SHA256=$P30_EXPECTED_P27_SANITIZER_SHA256"
  "P30_RUNTIME_REVIEW_COMMIT=$P30_RUNTIME_REVIEW_COMMIT"
  "P30_RUNTIME_REVIEW_TREE=$P30_RUNTIME_REVIEW_TREE"
  "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256=$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256"
  "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT=$P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT"
  "P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256=$P30_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256"
  "P30_EXPECTED_RUNTIME_LOCK_SHA256=$P30_EXPECTED_RUNTIME_LOCK_SHA256"
  "P30_EXPECTED_HOST_ATTESTATION_SHA256=$P30_EXPECTED_HOST_ATTESTATION_SHA256"
  "P30_EXPECTED_CONTAINER_ID=$P30_EXPECTED_CONTAINER_ID"
)
p30_verify_root_environment_handoff \
  17-localization-root-handoff 37 "${p30_localization_root_environment[@]}"
p30_run_logged_client_command 18-run-localization \
  "${p30_strict_ssh[@]}" /usr/bin/sudo -n /usr/bin/env -i \
  PATH=/usr/sbin:/usr/bin:/sbin:/bin \
  "${p30_localization_root_environment[@]}" \
  /bin/bash -p \
  /var/lib/optimizationml-p30-20260906-03/run_p30_umask_bound_control_seal.sh \
  run-localization
```

This passes exactly all 37 contract variables and no ambient user environment.
The script revalidates them after its one-shot token admission and before the
localization body performs runtime receipt replay or any later phase work.

Phase dispatch first enters one Python transaction that uses the supplied
expected sealed-orchestrator digest, validates that transaction's root-ledger,
deferred-journal, and sealed-source authorities, and O_EXCL-creates the
root-owned mode-`0400`
ledger `/var/lib/optimizationml-p30-20260906-03/run-localization.invoked`
before general environment validation or any evidence or later-marker
predicate. A missing or malformed supplied digest can fail before token
creation. The transaction returns success only to the process that created
and postvalidated the token. Its exact bytes are `run-localization\n`. At this
point the ledger already contains only the sealed orchestrator, deferred
journal, and exact `prepare-runtime.invoked` token. A premature localization
dispatch after that prepare token exists, or a repeated localization dispatch,
consumes P30. A pre-prepare localization dispatch cannot satisfy the required
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
`/var/lib/optimizationml-p30-20260906-03/run-localization.exit-status.txt` as
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
27-file inventory check. Inside that deferred producer, all 37 runtime
environment values are validated before transport validation or verifier
dispatch. It uses a specialized deferred-publication logger. It
creates root-owned no-overwrite stdout and stderr transcripts beneath
`/var/lib/optimizationml-p30-20260906-03/deferred` as
`.deferred-p30-pre-marker-runtime-receipt-replay.stdout.log`,
`.deferred-p30-pre-marker-runtime-receipt-replay.stderr.log` before invoking
the verifier while `evidence` still has exactly the frozen 27-file
pre-localization inventory. After the verifier exits—on success or failure—it
O_EXCL-creates
`.deferred-p30-pre-marker-runtime-receipt-replay.exit-status.txt`; only then
does the orchestrator reauthenticate the reviewed attempt/evidence identity.
Only after those steps may it stably reread each nonsymlink journal file and
publish
byte-identical `p30-pre-marker-runtime-receipt-replay.stdout.log`,
`p30-pre-marker-runtime-receipt-replay.stderr.log`, and
`p30-pre-marker-runtime-receipt-replay.exit-status.txt` into `evidence` with
`O_EXCL|O_NOFOLLOW`, then seal them `root:root` mode `0444`, before any other
pre-marker log. The original journal copies remain. An ordinary verifier failure with intact identity is retained
in both locations, creates no marker, and terminates the attempt.

Only after those replay transcripts have been published does the retained
`p30-localization-entry-preflight` run. That preflight requires the deferred
journal's exact after-runtime stage; the next retained boundary is
`p30-pre-marker-receipt-namespace-binding`.

The publication operation best-effort creates and seals the preregistered
journal triple `.publication.stdout.log`, `.publication.stderr.log`, and
`.publication.exit-status.txt` for this prefix. A complete triplet is
guaranteed only when artifact finalization succeeds; otherwise the logger
attempts every remaining close/seal/status action and returns terminal status
125. Exact success requires empty stdout and stderr and exact status bytes
`0\n`. All successfully finalized journal files are root-owned,
O_EXCL/no-follow, and sealed mode `0400`. A publication failure also
best-effort retains
`.deferred-p30-pre-marker-runtime-receipt-replay.publication-failure.exit-status.txt`;
that sentinel is forbidden on success.

Thus the successful exact deferred-journal inventories have 0 files before
prepare admission, 3 after admission, 9 after attempt-layout publication, and
15 after runtime-receipt replay publication. Extra, missing, writable, or
misowned journal children are terminal.

If identity reauthentication or evidence publication itself fails, the host
must not write through the suspect evidence path. With the root journal still
intact, it best-effort O_EXCL-creates the root-journal status
`.deferred-p30-pre-marker-runtime-receipt-replay.publication-failure.exit-status.txt`.
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
retained `p30-prepare-execution-snapshot-verification` boundary and again by
the bundle verifier during runtime-receipt creation and the retained
`p30-pre-marker-runtime-receipt-replay`. The later
`p30-pre-marker-snapshot-layout` check authenticates the guarded directories,
top-level execution inventory, and manifest authority; it does not replay the
full manifest.

Attempt/evidence identity is compared with the reviewed receipt at the
explicit `*-receipt-namespace-binding` labels after receipt replay, ingestion,
authorization, localization, sanitization, and final localization, and the
fresh layout is checked by `p30-localization-entry-preflight` and
`p30-pre-marker-snapshot-layout`. Each validator reopens its inputs
componentwise with no-follow semantics. This is deliberately neither a claim
that one host descriptor stays open across the entire marker-to-localizer
process span nor a claim that every logger implicitly repeats those checks.
New scientific artifacts are sealed and authenticated at their declared
producer or binding boundaries; historical evidence bytes are not reread
after every unrelated log. A failed explicit identity, authority, ACL, device,
mount, manifest, or artifact check stops the attempt and preserves its partial
state.

Four explicit ledger/preflight checks are retained in this exact chronological
order: `p30-localization-entry-preflight`,
`p30-post-authorization-ledger-validation`,
`p30-post-localizer-ledger-validation`, and
`p30-post-sanitizer-ledger-validation`. Each label uses an O_EXCL/no-follow
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
private payload must report P30 plus nested P29 contract 15/15, terminal P29
outcome 11/11, P28 contract 12/12, terminal P28 outcome 10/10, and P27 23/23.
No mutable control-checkout reconstructor is
executed by path. After every other retained pre-marker validator passes, the
host O_EXCL-creates root-owned mode-`0400`
`/var/lib/optimizationml-p30-20260906-03/localization.authorized` with exact
bytes `localization-authorized\n`. This authorization is created exactly once,
only after all reviewed gates pass. The container-side ingester, localizer,
P23 core, and sanitizer execute from the read-only
`/private/tmp/optimizationml-p22-data/.p30-tools/` adjunct. Before dispatch,
the host's explicit ledger validator requires all three root-owned tokens to
remain present and exact; the containerized tools do not read that ledger.
They ingest the already authenticated P26 failure, run the no-training
localizer once, and retain no-overwrite
stdout, stderr, native, sanitized, and manifest evidence. Never invoke an
inner command directly and never retry.

## 7. Freeze and route the result mechanically

Record a compact machine-readable P30 outcome and every retained hash. Use the
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

No result permits cleanup or a favorable rerun under this P30 identifier, and
no P30 route authorizes scientific acquisition `20260906-04`.
