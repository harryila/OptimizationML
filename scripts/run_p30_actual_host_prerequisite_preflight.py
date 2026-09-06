#!/usr/bin/env python3
"""Run P30's credential-free, read-only actual-host prerequisite preflight.

The normal client invocation is intentionally suitable for wrapping with
``scripts/run_logged_client_command.py``::

    python3 -I -S scripts/run_logged_client_command.py \
      --log-root "$P30_CLIENT_LOG_ROOT" --label 00-host-freshness -- \
      python3 -I -S "$P30_PREFLIGHT_SOURCE" \
      --known-hosts /absolute/path/to/p30_known_hosts \
      --identity-file /absolute/reviewed/path/to/p30_identity \
      --expected-client-source-sha256 "$P30_PREFLIGHT_SOURCE_SHA256"

The known-hosts file must contain exactly the confirmed ED25519 key for the
locked host. ``P30_PREFLIGHT_SOURCE`` must be an absolute, independently
materialized source-commit blob, and the expected digest must be computed from
that reviewed blob outside this process. The client stably reads that exact
mode-0400 regular file without following links, rejects an authority or digest
mismatch, and streams the held bytes to an isolated root Python process over
strict public-key SSH using only the explicitly reviewed identity and no
ambient SSH configuration. The remote half only reads host, Git, Docker,
NVIDIA, and frozen-input state. It does not create the P30 namespace, start a
container, initialize CUDA, import torch, or run training.
"""

from __future__ import annotations

import argparse
import base64
import binascii
import grp
import hashlib
import json
import os
import pathlib
import pwd
import re
import stat
import subprocess
import sys
from collections.abc import Sequence
from typing import NoReturn

SSH_TARGET = "ubuntu@129.146.177.214"
SSH_HOST = "129.146.177.214"
SSH_PORT = 22
EXPECTED_HOST_KEY_TYPE = "ssh-ed25519"
EXPECTED_HOST_KEY_FINGERPRINT = "SHA256:t7nLeL/gt2qBu7e+I0yb8xICPtX8WcP8RMn7yEkXrJU"

P30_NAMESPACE = pathlib.PurePosixPath("/secure/p30")
P30_LEDGER = pathlib.PurePosixPath("/var/lib/optimizationml-p30-20260906-03")
P30_CONTAINER = "p30-localization-20260906-03"

GPU_UUID = "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d"
GPU_NAME = "NVIDIA A100-SXM4-40GB"
IMAGE_ID = "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0"
IMAGE_REFERENCE = f"localhost:5000/p25-runtime@{IMAGE_ID}"

NANOGPT = pathlib.PurePosixPath("/secure/p23/nanoGPT")
NANOGPT_COMMIT = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
NANOGPT_TREE = "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
MUON = pathlib.PurePosixPath("/secure/p23/Muon")
MUON_COMMIT = "f98f1cacc0263b04290753e32be8d498c1efc806"
MUON_TREE = "4ea5cd8ab6ebd56a18536f06453619efcd636da0"

DATA_MANIFEST = pathlib.PurePosixPath(
    "/secure/p23/optimizationml-p22-data/materialized/p22_fineweb_manifest.json"
)
DATA_MANIFEST_SHA256 = "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d"
DATA_MANIFEST_BYTES = 4210
P26_EVIDENCE = pathlib.PurePosixPath(
    "/secure/p25/attempt-20260906-02/evidence/trace-off-a-failure.json"
)
P26_EVIDENCE_SHA256 = "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91"
P26_EVIDENCE_BYTES = 66283

REQUIRED_EXECUTABLES = (
    "/bin/bash",
    "/bin/sh",
    "/usr/bin/cat",
    "/usr/bin/chmod",
    "/usr/bin/chown",
    "/usr/bin/docker",
    "/usr/bin/env",
    "/usr/bin/find",
    "/usr/bin/git",
    "/usr/bin/head",
    "/usr/bin/id",
    "/usr/bin/mkdir",
    "/usr/bin/nsenter",
    "/usr/bin/nvidia-smi",
    "/usr/bin/python3",
    "/usr/bin/realpath",
    "/usr/bin/setpriv",
    "/usr/bin/sha256sum",
    "/usr/bin/sort",
    "/usr/bin/stat",
    "/usr/bin/sudo",
    "/usr/bin/test",
)

ACL_XATTRS = {
    "system.posix_acl_access",
    "system.posix_acl_default",
    "system.nfs4_acl",
    "system.richacl",
}
GIT_ENVIRONMENT = {
    "GIT_ATTR_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_NO_REPLACE_OBJECTS": "1",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "HOME": "/nonexistent",
    "LANG": "C",
    "LC_ALL": "C",
    "PATH": "/usr/sbin:/usr/bin:/sbin:/bin",
}


class PreflightError(RuntimeError):
    """A prerequisite differs from the frozen P30 expectation."""


def _fail(message: str) -> NoReturn:
    raise PreflightError(message)


def _stable_fields(value: os.stat_result) -> tuple[int, ...]:
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


def _require_kernel_flags() -> None:
    for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
        value = getattr(os, name, None)
        if not isinstance(value, int) or value == 0:
            _fail(f"platform lacks mandatory {name}")


def _open_directory(path: pathlib.PurePosixPath) -> int:
    if not path.is_absolute() or str(path) != os.path.normpath(path):
        _fail(f"directory path is not canonical: {path}")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open("/", flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _assert_absent(path: pathlib.PurePosixPath, label: str) -> dict[str, object]:
    try:
        os.lstat(path)
    except FileNotFoundError:
        return {"absent": True, "path": str(path)}
    except OSError as exc:
        raise PreflightError(f"cannot establish {label} absence: {exc}") from exc
    _fail(f"{label} already exists and consumes the frozen identity: {path}")


def _decode_mount_field(value: str) -> str:
    return (
        value.replace("\\040", " ")
        .replace("\\011", "\t")
        .replace("\\012", "\n")
        .replace("\\134", "\\")
    )


def _secure_authority() -> dict[str, object]:
    descriptor = _open_directory(pathlib.PurePosixPath("/secure"))
    try:
        before = os.fstat(descriptor)
        by_name = os.stat("/secure", follow_symlinks=False)
        names = {
            item.decode() if isinstance(item, bytes) else item for item in os.listxattr(descriptor)
        }
        mount_points = {
            _decode_mount_field(line.split()[4])
            for line in pathlib.Path("/proc/self/mountinfo")
            .read_text(encoding="utf-8")
            .splitlines()
            if len(line.split()) >= 5
        }
        after = os.fstat(descriptor)
        if _stable_fields(before) != _stable_fields(after) or _stable_fields(
            after
        ) != _stable_fields(by_name):
            _fail("/secure changed during prerequisite inspection")
        if not stat.S_ISDIR(after.st_mode):
            _fail("/secure is not a directory")
        if (after.st_uid, after.st_gid, stat.S_IMODE(after.st_mode)) != (0, 0, 0o755):
            _fail("/secure owner or mode differs from root:root 0755")
        if names & ACL_XATTRS:
            _fail("/secure has a forbidden access/default/frozen ACL xattr")
        if "/secure" in mount_points or after.st_dev != os.stat("/").st_dev:
            _fail("/secure is a mount point or crosses the root filesystem")
        if pathlib.Path("/secure").resolve(strict=True) != pathlib.Path("/secure"):
            _fail("/secure lexical and resolved identities differ")
        return {
            "acl_xattrs_absent": True,
            "gid": after.st_gid,
            "mode_octal": "0755",
            "mountpoint": False,
            "uid": after.st_uid,
        }
    finally:
        os.close(descriptor)


def _account_authority() -> dict[str, object]:
    try:
        account = pwd.getpwnam("ubuntu")
        by_uid = pwd.getpwuid(1000)
        group = grp.getgrgid(1000)
    except KeyError as exc:
        raise PreflightError("ubuntu numeric account/group authority is absent") from exc
    if (account.pw_uid, account.pw_gid, by_uid.pw_name, group.gr_name) != (
        1000,
        1000,
        "ubuntu",
        "ubuntu",
    ):
        _fail("ubuntu account does not bind exactly to uid/gid 1000:1000")
    if os.geteuid() != 0 or os.getegid() != 0:
        _fail("remote preflight did not enter through noninteractive root sudo")
    return {"group": group.gr_name, "name": account.pw_name, "uid": 1000, "gid": 1000}


def _executable_authority() -> dict[str, object]:
    missing: list[str] = []
    for raw in REQUIRED_EXECUTABLES:
        path = pathlib.Path(raw)
        try:
            resolved = path.resolve(strict=True)
            info = resolved.stat()
        except OSError:
            missing.append(raw)
            continue
        if not stat.S_ISREG(info.st_mode) or not os.access(path, os.X_OK):
            missing.append(raw)
    if missing:
        _fail("required host executables are absent or nonexecutable: " + ", ".join(missing))
    # The streamed remote half intentionally supports the host contract's 3.10 floor,
    # independently of this package's newer development-environment floor.
    if sys.version_info < (3, 10):  # noqa: UP036
        _fail("/usr/bin/python3 is older than Python 3.10")
    return {
        "count": len(REQUIRED_EXECUTABLES),
        "paths": list(REQUIRED_EXECUTABLES),
        "python_version": ".".join(str(value) for value in sys.version_info[:3]),
        "sudo_noninteractive_root_entry": True,
    }


def _run_read_only(command: Sequence[str], *, environment: dict[str, str] | None = None) -> str:
    try:
        completed = subprocess.run(
            list(command),
            check=False,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            env=environment,
            timeout=120,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise PreflightError(f"read-only command could not complete: {command[0]}: {exc}") from exc
    if completed.returncode != 0:
        stderr = completed.stderr.decode("utf-8", errors="backslashreplace").strip()
        _fail(f"read-only command failed ({completed.returncode}): {' '.join(command)}: {stderr}")
    try:
        return completed.stdout.decode("utf-8").rstrip("\n")
    except UnicodeDecodeError as exc:
        raise PreflightError(f"read-only command output is not UTF-8: {command[0]}") from exc


def _git(repository: pathlib.PurePosixPath, *arguments: str) -> str:
    return _run_read_only(
        [
            "/usr/bin/git",
            "-c",
            f"safe.directory={repository}",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.attributesFile=/dev/null",
            "-c",
            "core.excludesFile=/dev/null",
            "-C",
            str(repository),
            *arguments,
        ],
        environment=GIT_ENVIRONMENT,
    )


def _git_repository(
    repository: pathlib.PurePosixPath,
    *,
    expected_commit: str,
    expected_tree: str,
    label: str,
) -> dict[str, object]:
    path = pathlib.Path(repository)
    git_dir = path / ".git"
    if not path.is_dir() or path.is_symlink() or not git_dir.is_dir() or git_dir.is_symlink():
        _fail(f"{label} is not one plain Git worktree")
    if path.resolve(strict=True) != path or git_dir.resolve(strict=True) != git_dir:
        _fail(f"{label} lexical and resolved identities differ")
    forbidden_paths = (
        git_dir / "commondir",
        git_dir / "shallow",
        git_dir / "objects/info/alternates",
        git_dir / "objects/info/http-alternates",
        git_dir / "info/grafts",
    )
    if any(item.exists() or item.is_symlink() for item in forbidden_paths):
        _fail(f"{label} has a shallow checkout or forbidden Git object indirection")
    config_names = _git(repository, "config", "--local", "--no-includes", "--name-only", "--list")
    forbidden_config = re.compile(
        r"^(?:include\.path|includeif\..*\.path|core\.(?:worktree|fsmonitor|hookspath|"
        r"sshcommand|attributesfile|editor|pager)|credential\.helper|diff\.(?:external|.*\.command)|"
        r"filter\..*\.(?:clean|smudge|process)|merge\..*\.driver|pager\..*|"
        r"interactive\.difffilter|extensions\.(?:worktreeconfig|partialclone)|"
        r"remote\..*\.(?:promisor|partialclonefilter))$",
        flags=re.IGNORECASE,
    )
    rejected = [name for name in config_names.splitlines() if forbidden_config.fullmatch(name)]
    if rejected:
        _fail(f"{label} has forbidden local Git configuration: {rejected[0]}")
    if _git(repository, "rev-parse", "--is-bare-repository") != "false":
        _fail(f"{label} is bare")
    if _git(repository, "rev-parse", "--is-shallow-repository") != "false":
        _fail(f"{label} is shallow")
    if _git(repository, "rev-parse", "--show-toplevel") != str(repository):
        _fail(f"{label} top-level path differs")
    if _git(repository, "rev-parse", "--absolute-git-dir") != str(repository / ".git"):
        _fail(f"{label} Git-directory path differs")
    actual_commit = _git(repository, "rev-parse", "HEAD")
    actual_tree = _git(repository, "rev-parse", "HEAD^{tree}")
    if actual_commit != expected_commit or actual_tree != expected_tree:
        _fail(f"{label} commit or tree differs")
    if _git(repository, "for-each-ref", "--format=%(refname)", "refs/replace/"):
        _fail(f"{label} has replacement refs")
    if _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignored=matching",
    ):
        _fail(f"{label} contains a tracked change, untracked file, or ignored file")
    return {
        "clean_including_ignored": True,
        "commit": actual_commit,
        "object_indirections_absent": True,
        "path": str(repository),
        "tree": actual_tree,
    }


def _stable_file(
    path: pathlib.PurePosixPath,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> dict[str, object]:
    parent = _open_directory(path.parent)
    descriptor: int | None = None
    try:
        descriptor = os.open(path.name, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW, dir_fd=parent)
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
            _fail(f"{label} is not one regular single-link file")
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        by_name = os.stat(path.name, dir_fd=parent, follow_symlinks=False)
        if _stable_fields(before) != _stable_fields(after) or _stable_fields(
            after
        ) != _stable_fields(by_name):
            _fail(f"{label} changed during stable read")
        actual_sha256 = digest.hexdigest()
        if actual_sha256 != expected_sha256 or byte_count != expected_bytes:
            _fail(f"{label} SHA-256 or byte count differs")
        return {
            "byte_count": byte_count,
            "path": str(path),
            "sha256": actual_sha256,
            "stable_nofollow_read": True,
        }
    except OSError as exc:
        raise PreflightError(f"cannot read {label} without following links: {exc}") from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent)


def _gpu_authority() -> dict[str, object]:
    output = _run_read_only(
        [
            "/usr/bin/nvidia-smi",
            f"--id={GPU_UUID}",
            "--query-gpu=uuid,name",
            "--format=csv,noheader,nounits",
        ],
        environment={"LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    rows = [row for row in output.splitlines() if row]
    if rows != [f"{GPU_UUID}, {GPU_NAME}"]:
        _fail("NVIDIA GPU UUID/name query differs from the frozen A100")
    return {"cuda_initialized": False, "name": GPU_NAME, "query_rows": 1, "uuid": GPU_UUID}


def _docker_authority() -> dict[str, object]:
    base = [
        "/usr/bin/docker",
        "--host",
        "unix:///var/run/docker.sock",
        "--config",
        "/nonexistent/p30-docker-config",
    ]
    names = _run_read_only(
        [*base, "container", "ls", "--all", "--no-trunc", "--format={{.Names}}"],
        environment={"HOME": "/nonexistent", "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    ).splitlines()
    if P30_CONTAINER in names:
        _fail(f"frozen P30 container name already exists: {P30_CONTAINER}")
    actual_image_id = _run_read_only(
        [*base, "image", "inspect", "--format={{.Id}}", IMAGE_REFERENCE],
        environment={"HOME": "/nonexistent", "LANG": "C", "LC_ALL": "C", "PATH": "/usr/bin:/bin"},
    )
    if actual_image_id != IMAGE_ID:
        _fail("pinned P30 image reference resolves to a different image ID")
    return {
        "container_name_absent": True,
        "image_id": actual_image_id,
        "image_reference": IMAGE_REFERENCE,
    }


def run_remote_preflight() -> dict[str, object]:
    _require_kernel_flags()
    namespace = _assert_absent(P30_NAMESPACE, "P30 namespace")
    ledger = _assert_absent(P30_LEDGER, "P30 ledger")
    secure = _secure_authority()
    account = _account_authority()
    executables = _executable_authority()
    docker = _docker_authority()
    gpu = _gpu_authority()
    nanogpt = _git_repository(
        NANOGPT,
        expected_commit=NANOGPT_COMMIT,
        expected_tree=NANOGPT_TREE,
        label="pinned nanoGPT checkout",
    )
    muon = _git_repository(
        MUON,
        expected_commit=MUON_COMMIT,
        expected_tree=MUON_TREE,
        label="pinned Muon checkout",
    )
    data_manifest = _stable_file(
        DATA_MANIFEST,
        expected_sha256=DATA_MANIFEST_SHA256,
        expected_bytes=DATA_MANIFEST_BYTES,
        label="FineWeb materialization manifest",
    )
    p26_evidence = _stable_file(
        P26_EVIDENCE,
        expected_sha256=P26_EVIDENCE_SHA256,
        expected_bytes=P26_EVIDENCE_BYTES,
        label="terminal P26 evidence",
    )
    return {
        "account": account,
        "claim_boundary": {
            "container_created_or_started": False,
            "cuda_initialized": False,
            "gpu_queried_with_nvidia_smi_only": True,
            "p30_namespace_ledger_or_container_state_created": False,
            "scientific_observation": False,
            "training_run": False,
        },
        "data_manifest": data_manifest,
        "docker": docker,
        "executables": executables,
        "gpu": gpu,
        "ledger": ledger,
        "muon": muon,
        "nanogpt": nanogpt,
        "namespace": namespace,
        "p26_evidence": p26_evidence,
        "schema_version": "passive-muon-p30-actual-host-prerequisite-preflight-v1",
        "secure_authority": secure,
        "status": "all_read_only_prerequisites_passed_before_p30_state_creation",
    }


def _fingerprint(key_blob: bytes) -> str:
    return "SHA256:" + base64.b64encode(hashlib.sha256(key_blob).digest()).decode("ascii").rstrip(
        "="
    )


def _require_sha256(value: str, label: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        _fail(f"{label} is not one lowercase SHA-256")
    return value


def _stable_client_source(
    path: pathlib.Path, expected_sha256: str
) -> tuple[bytes, dict[str, object]]:
    expected_sha256 = _require_sha256(expected_sha256, "expected client-source digest")
    _require_kernel_flags()
    if not path.is_absolute():
        _fail("client preflight source path must be absolute")
    if path.name in {"", ".", ".."}:
        _fail("client preflight source basename is malformed")
    source_fd: int | None = None
    parent_fd: int | None = None
    try:
        parent_info = path.parent.lstat()
        source_info = path.lstat()
    except OSError as exc:
        raise PreflightError(f"cannot inspect materialized client preflight source: {exc}") from exc
    if path.parent.resolve(strict=True) != path.parent:
        _fail("client preflight source parent lexical and resolved identities differ")
    if (
        not stat.S_ISDIR(parent_info.st_mode)
        or parent_info.st_uid != os.getuid()
        or stat.S_IMODE(parent_info.st_mode) & 0o022
    ):
        _fail("client preflight source parent must be caller-owned and not group/world writable")
    if (
        not stat.S_ISREG(source_info.st_mode)
        or source_info.st_uid != os.getuid()
        or source_info.st_gid != os.getgid()
        or source_info.st_nlink != 1
        or stat.S_IMODE(source_info.st_mode) != 0o400
    ):
        _fail(
            "client preflight source must have the exact caller UID/GID, be regular and "
            "single-link, and have physical mode 0400"
        )
    try:
        parent_fd = os.open(
            path.parent,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
        parent_before = os.fstat(parent_fd)
        source_fd = os.open(
            path.name,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW,
            dir_fd=parent_fd,
        )
        source_before = os.fstat(source_fd)
        chunks: list[bytes] = []
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_fd, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
            digest.update(chunk)
        source_after = os.fstat(source_fd)
        source_by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        parent_by_name = path.parent.lstat()
    except OSError as exc:
        raise PreflightError(
            f"cannot stably read materialized client source without following links: {exc}"
        ) from exc
    finally:
        if source_fd is not None:
            os.close(source_fd)
        if parent_fd is not None:
            os.close(parent_fd)
    if (
        _stable_fields(parent_info) != _stable_fields(parent_before)
        or _stable_fields(parent_before) != _stable_fields(parent_after)
        or _stable_fields(parent_after) != _stable_fields(parent_by_name)
    ):
        _fail("client preflight source parent changed during stable read")
    if (
        _stable_fields(source_info) != _stable_fields(source_before)
        or _stable_fields(source_before) != _stable_fields(source_after)
        or _stable_fields(source_after) != _stable_fields(source_by_name)
    ):
        _fail("client preflight source changed during stable read")
    actual_sha256 = digest.hexdigest()
    if actual_sha256 != expected_sha256:
        _fail("materialized client preflight source SHA-256 differs")
    raw = b"".join(chunks)
    if not raw:
        _fail("materialized client preflight source is empty")
    return raw, {
        "byte_count": len(raw),
        "gid": source_after.st_gid,
        "link_count": source_after.st_nlink,
        "mode_octal": format(stat.S_IMODE(source_after.st_mode), "04o"),
        "path": str(path),
        "sha256": actual_sha256,
        "stable_nofollow_read": True,
        "uid": source_after.st_uid,
    }


def _validate_known_hosts(path: pathlib.Path, expected_fingerprint: str) -> None:
    if not path.is_absolute():
        _fail("known-hosts path must be absolute")
    descriptor: int | None = None
    try:
        info = path.lstat()
    except OSError as exc:
        raise PreflightError(f"cannot inspect pinned known-hosts file: {exc}") from exc
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) != 0o400
    ):
        _fail("known-hosts file must be caller-owned, single-link, and exact mode 0400")
    if path.resolve(strict=True) != path:
        _fail("known-hosts lexical and resolved paths differ")
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        before = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 64 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        by_name = path.lstat()
    except OSError as exc:
        raise PreflightError(
            f"cannot read pinned known-hosts file without following links: {exc}"
        ) from exc
    finally:
        if descriptor is not None:
            os.close(descriptor)
    if (
        _stable_fields(info) != _stable_fields(before)
        or _stable_fields(before) != _stable_fields(after)
        or _stable_fields(after) != _stable_fields(by_name)
    ):
        _fail("known-hosts file changed during stable read")
    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as exc:
        raise PreflightError("known-hosts file is not UTF-8") from exc
    lines = [line for line in text.splitlines() if line and not line.startswith("#")]
    if len(lines) != 1:
        _fail("known-hosts file must contain exactly one noncomment key")
    fields = lines[0].split()
    if len(fields) != 3 or fields[0] not in {SSH_HOST, f"[{SSH_HOST}]:{SSH_PORT}"}:
        _fail("known-hosts entry does not name the exact locked host and port")
    if fields[1] != EXPECTED_HOST_KEY_TYPE:
        _fail("known-hosts entry is not the locked ED25519 host-key type")
    try:
        key_blob = base64.b64decode(fields[2], validate=True)
    except (binascii.Error, ValueError) as exc:
        raise PreflightError("known-hosts key blob is malformed") from exc
    if _fingerprint(key_blob) != expected_fingerprint:
        _fail("known-hosts ED25519 fingerprint differs from the operator-confirmed value")


def _validate_identity_file(path: pathlib.Path) -> None:
    if not path.is_absolute():
        _fail("identity-file path must be absolute")
    try:
        info = path.lstat()
    except OSError as exc:
        raise PreflightError(f"cannot inspect SSH identity file: {exc}") from exc
    if path.resolve(strict=True) != path:
        _fail("identity-file lexical and resolved paths differ")
    if (
        not stat.S_ISREG(info.st_mode)
        or info.st_uid != os.getuid()
        or info.st_gid != os.getgid()
        or info.st_nlink != 1
        or stat.S_IMODE(info.st_mode) not in {0o400, 0o600}
    ):
        _fail(
            "identity file must have the exact caller UID/GID, be single-link, and have "
            "physical mode 0400 or 0600"
        )


def build_ssh_command(
    known_hosts: pathlib.Path,
    identity_file: pathlib.Path,
    client_source_sha256: str,
) -> list[str]:
    client_source_sha256 = _require_sha256(client_source_sha256, "client-source digest")
    command = [
        "/usr/bin/ssh",
        "-F",
        "/dev/null",
        "-T",
        "-p",
        str(SSH_PORT),
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=yes",
        "-o",
        f"UserKnownHostsFile={known_hosts}",
        "-o",
        "GlobalKnownHostsFile=/dev/null",
        "-o",
        "HostKeyAlgorithms=ssh-ed25519",
        "-o",
        "UpdateHostKeys=no",
        "-o",
        "CheckHostIP=yes",
        "-o",
        "ClearAllForwardings=yes",
        "-o",
        "ProxyCommand=none",
        "-o",
        "ProxyJump=none",
        "-o",
        "ControlMaster=no",
        "-o",
        "ControlPath=none",
        "-o",
        "PermitLocalCommand=no",
        "-o",
        "PasswordAuthentication=no",
        "-o",
        "KbdInteractiveAuthentication=no",
        "-o",
        "PreferredAuthentications=publickey",
        "-o",
        "IdentitiesOnly=yes",
        "-i",
        str(identity_file),
    ]
    command.extend(
        [
            SSH_TARGET,
            "/usr/bin/sudo",
            "-n",
            "/usr/bin/env",
            "-i",
            "HOME=/root",
            "LANG=C",
            "LC_ALL=C",
            "PATH=/usr/sbin:/usr/bin:/sbin:/bin",
            "PYTHONDONTWRITEBYTECODE=1",
            "PYTHONNOUSERSITE=1",
            "PYTHONSAFEPATH=1",
            "/usr/bin/python3",
            "-I",
            "-S",
            "-",
            "--remote",
            "--expected-client-source-sha256",
            client_source_sha256,
        ]
    )
    return command


def _remote_main(client_source_sha256: str) -> int:
    try:
        client_source_sha256 = _require_sha256(client_source_sha256, "remote client-source digest")
        result = run_remote_preflight()
    except (PreflightError, OSError) as exc:
        print(f"P30 actual-host prerequisite preflight failed: {exc}", file=sys.stderr)
        return 1
    result["client_source"] = {
        "sha256": client_source_sha256,
        "streamed_over_pinned_ssh": True,
    }
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n")
    return 0


def _client_main(argv: Sequence[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--known-hosts", required=True, type=pathlib.Path)
    parser.add_argument("--identity-file", required=True, type=pathlib.Path)
    parser.add_argument("--expected-client-source-sha256", required=True)
    arguments = parser.parse_args(argv)
    try:
        _require_kernel_flags()
        _validate_known_hosts(arguments.known_hosts, EXPECTED_HOST_KEY_FINGERPRINT)
        _validate_identity_file(arguments.identity_file)
        source, _source_authority = _stable_client_source(
            pathlib.Path(__file__), arguments.expected_client_source_sha256
        )
        completed = subprocess.run(
            build_ssh_command(
                arguments.known_hosts,
                arguments.identity_file,
                arguments.expected_client_source_sha256,
            ),
            check=False,
            input=source,
        )
    except (PreflightError, OSError) as exc:
        print(f"P30 actual-host prerequisite preflight client failed: {exc}", file=sys.stderr)
        return 125
    return completed.returncode if 0 <= completed.returncode <= 255 else 1


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["--remote"]:
        remote = argparse.ArgumentParser(add_help=False)
        remote.add_argument("--remote", action="store_true")
        remote.add_argument("--expected-client-source-sha256", required=True)
        parsed = remote.parse_args(arguments)
        return _remote_main(parsed.expected_client_source_sha256)
    return _client_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
