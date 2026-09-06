#!/usr/bin/env python3
"""Fail-closed verification for P28's two offline control-bundle transfers.

The source phase runs before an acquisition attempt root may exist; the runtime
phase runs after preparation but before localization. The verifier copies the
transport bundle with one stable ``O_NOFOLLOW`` read, verifies that private copy
in a newly initialized empty bare repository, explicitly fetches the closed ref
inventory, and only then creates or advances the detached control checkout. A
deterministic external receipt can subsequently be authenticated and replayed
without overwriting it.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCHEMA = "passive-muon-p28-control-bundle-receipt-v1"
STATUS_BY_PHASE = {
    "source": "source_bundle_verified_before_attempt_root_creation",
    "runtime-review": "runtime_review_bundle_verified_before_localization",
}
BRANCH_REF = "refs/heads/p28-bundle-complete-localization-bridge"
P27_BRANCH_REF = "refs/heads/p27-cuda-deleted-mapping-localization"
P27_TAG_REF = "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic"
P26_CHECKPOINT_REF = "refs/tags/p26-permission-safe-acquisition-checkpoint"
P26_SOURCE_REF = "refs/tags/p26-attempt02-acquisition-source"

P27_COMMIT = "ec63550331925ded158e3f389e294e4d1f12db3a"
P27_TREE = "d8efa72fda9ee41fde0b5d15b126aa87397cd48d"
P27_TAG_OBJECT = "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
P26_CHECKPOINT_OBJECT = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
P26_CHECKPOINT_COMMIT = "5429da23ff18888daa2312c530a4587780484d8b"
P26_SOURCE_OBJECT = "f35a7dca8f6bc39e9748e79b6712fab4a203396b"
P26_SOURCE_COMMIT = "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
P27_RECONSTRUCTOR = "scripts/reconstruct_p27_cuda_deleted_mapping_localization.py"
VERIFIER_SOURCE = "scripts/verify_p28_control_bundle.py"
P27_RECONSTRUCTOR_SHA256 = "0eedffa5442ce7fc3950ae74d169fa652c70794935c541028eba3d0bda20e033"
P27_RECONSTRUCTION_SCHEMA = "passive-muon-p27-cuda-deleted-mapping-localization-reconstruction-v1"
P27_CONTRACT_SHA256 = "02000debea28f499ddeb3b8c6c8b0615786cd2e7858cc8a1605b04d638da4b7a"

PHASE_LAYOUT = {
    "source": {
        "bundle": "p28_source.bundle",
        "closure": "p28_source_closure.git",
        "receipt": "p28_source_bundle_receipt.json",
    },
    "runtime-review": {
        "bundle": "p28_runtime_review.bundle",
        "closure": "p28_runtime_review_closure.git",
        "receipt": "p28_runtime_review_bundle_receipt.json",
    },
}
RUNTIME_DELTA = [
    "A\texperiments/training/p28_cuda_runtime_lock.json",
    "A\texperiments/training/p28_host_attestation.json",
]
RUNTIME_EVIDENCE_FILES = {
    "runtime_lock": "p28_cuda_runtime_lock.json",
    "host_attestation": "p28_host_attestation.json",
}
RUNTIME_FORBIDDEN_LOCALIZATION_FILES = [
    "p28-run-localization.invoked",
    "p28_cuda_deleted_mapping_localization.native.json",
    "p28_cuda_deleted_mapping_localization.sanitized.json",
    "p28-localization.exit-status.txt",
    "p28-localization-sanitizer.exit-status.txt",
    "p28-p26-failure-ingestion.exit-status.txt",
    "run_p27_cuda_deleted_mapping_localization.py",
    "ingest_p26_trace_off_a_failure.py",
    "p23_deterministic_cuda_shadow_trace.py",
    "sanitize_p27_cuda_deleted_mapping_localization.py",
]
RUNTIME_FORBIDDEN_PREFIXES = [
    "p28-run-contract-reconstruction.",
    "p28-pre-ingestion-",
    "p28-p26-failure-ingestion.",
    "p28-pre-localization-",
    "p28-localization.",
    "p28-localization-sanitizer.",
]
RUNTIME_ATTEMPT_TOP_LEVEL = [
    "evidence",
    "p28-image-inspect.json",
    "p28-nvidia-smi.csv",
    "p28-running-container-inspect.json",
    "p28-running-mountinfo.txt",
    "p28-container-id.txt",
]
RUNTIME_EVIDENCE_ALLOWLIST = [
    "p28-prepare-contract-reconstruction.stdout.log",
    "p28-prepare-contract-reconstruction.stderr.log",
    "p28-image-inspection.stderr.log",
    "p28-nvidia-smi.stderr.log",
    "p28-container-launch.stderr.log",
    "p28-running-container-inspection.stderr.log",
    "p28-running-mountinfo.stderr.log",
    "p28-freeze-runtime.stdout.log",
    "p28-freeze-runtime.stderr.log",
    "p28_cuda_runtime_lock.json",
    "p28_host_attestation.json",
]
SHA1_RE = re.compile(r"[0-9a-f]{40}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")


class VerificationError(RuntimeError):
    """A fail-closed P28 transport-verification error."""


@dataclass(frozen=True)
class StableFile:
    sha256: str
    byte_count: int
    mode_octal: str
    uid: int
    gid: int
    device: int
    inode: int
    link_count: int


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise VerificationError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> None:
    raise VerificationError(f"nonfinite JSON constant: {token}")


def strict_json_loads(raw: bytes) -> Any:
    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=_reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise VerificationError(f"invalid strict JSON: {exc}") from exc


def canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _stable_fields(info: os.stat_result) -> tuple[int, ...]:
    return (
        info.st_dev,
        info.st_ino,
        info.st_mode,
        info.st_uid,
        info.st_gid,
        info.st_nlink,
        info.st_size,
        info.st_mtime_ns,
        info.st_ctime_ns,
    )


def stable_copy_nofollow(source: Path, destination: Path) -> StableFile:
    """Copy ``source`` exactly once through a no-follow descriptor.

    All later Git operations consume ``destination``, never the transport path.
    The source descriptor is checked before and after the read so size or inode
    mutation cannot silently change the authenticated byte stream.
    """

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        source_fd = os.open(source, flags)
    except OSError as exc:
        raise VerificationError(f"cannot open bundle with O_NOFOLLOW: {exc}") from exc
    destination_fd: int | None = None
    try:
        before = os.fstat(source_fd)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("transport bundle is not a regular file")
        destination_fd = os.open(
            destination,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                if written <= 0:
                    raise VerificationError("short write while privatizing bundle")
                view = view[written:]
        os.fsync(destination_fd)
        after = os.fstat(source_fd)
        if _stable_fields(before) != _stable_fields(after):
            raise VerificationError("transport bundle changed during its stable read")
        if byte_count != before.st_size:
            raise VerificationError("transport bundle size disagrees with bytes read")
        return StableFile(
            sha256=digest.hexdigest(),
            byte_count=byte_count,
            mode_octal=format(stat.S_IMODE(before.st_mode), "04o"),
            uid=before.st_uid,
            gid=before.st_gid,
            device=before.st_dev,
            inode=before.st_ino,
            link_count=before.st_nlink,
        )
    finally:
        if destination_fd is not None:
            os.close(destination_fd)
        os.close(source_fd)


def stable_read_nofollow(source: Path) -> tuple[bytes, StableFile]:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(source, flags)
    except OSError as exc:
        raise VerificationError(f"cannot open receipt with O_NOFOLLOW: {exc}") from exc
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise VerificationError("reviewed receipt is not a regular file")
        pieces: list[bytes] = []
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            pieces.append(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        if _stable_fields(before) != _stable_fields(after):
            raise VerificationError("reviewed receipt changed during its stable read")
        if byte_count != before.st_size:
            raise VerificationError("reviewed receipt size disagrees with bytes read")
        raw = b"".join(pieces)
        return raw, StableFile(
            sha256=sha256_bytes(raw),
            byte_count=byte_count,
            mode_octal=format(stat.S_IMODE(before.st_mode), "04o"),
            uid=before.st_uid,
            gid=before.st_gid,
            device=before.st_dev,
            inode=before.st_ino,
            link_count=before.st_nlink,
        )
    finally:
        os.close(descriptor)


def authenticate_executing_verifier(expected_sha256: str, declared_source: Path) -> StableFile:
    executing = Path(__file__).absolute()
    if declared_source.absolute() != executing:
        raise VerificationError("declared bootstrap verifier is not the executing source")
    if executing.is_symlink():
        raise VerificationError("executing P28 verifier is a symlink")
    _raw, record = stable_read_nofollow(executing)
    if record.sha256 != expected_sha256:
        raise VerificationError("executing P28 verifier bytes differ")
    if record.mode_octal != "0600":
        raise VerificationError("executing P28 bootstrap verifier mode is not 0600")
    return record


def seal_private_bundle(path: Path, expected: StableFile) -> StableFile:
    """Make the authenticated copy read-only and bind its pre-Git identity."""

    if path.is_symlink():
        raise VerificationError("private authenticated bundle copy is a symlink")
    try:
        os.chmod(path, 0o400, follow_symlinks=False)
    except OSError as exc:
        raise VerificationError(f"cannot seal private bundle copy read-only: {exc}") from exc
    _raw, record = stable_read_nofollow(path)
    if record.mode_octal != "0400":
        raise VerificationError("private authenticated bundle copy mode is not 0400")
    if record.sha256 != expected.sha256 or record.byte_count != expected.byte_count:
        raise VerificationError("private bundle copy differs before Git verification")
    return record


def revalidate_private_bundle(path: Path, expected: StableFile) -> None:
    """Reauthenticate private bundle identity, bytes, size, and mode after Git."""

    if path.is_symlink():
        raise VerificationError("private authenticated bundle copy became a symlink")
    _raw, actual = stable_read_nofollow(path)
    if actual != expected:
        raise VerificationError("private authenticated bundle copy changed during Git use")
    if actual.mode_octal != "0400":
        raise VerificationError("private authenticated bundle copy is no longer read-only")


def _clean_git_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for key in (
        "GIT_ALTERNATE_OBJECT_DIRECTORIES",
        "GIT_COMMON_DIR",
        "GIT_DIR",
        "GIT_INDEX_FILE",
        "GIT_OBJECT_DIRECTORY",
        "GIT_REPLACE_REF_BASE",
        "GIT_SHALLOW_FILE",
        "GIT_WORK_TREE",
    ):
        environment.pop(key, None)
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    return environment


def run_git(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess[bytes]:
    command = [
        "git",
        "-c",
        "core.hooksPath=/dev/null",
        "-c",
        "protocol.file.allow=always",
        *arguments,
    ]
    process = subprocess.run(
        command,
        cwd=cwd,
        env=_clean_git_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if check and process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise VerificationError(
            f"Git command failed ({process.returncode}): {' '.join(arguments)}: {stderr}"
        )
    return process


def require_sha1(value: str, label: str) -> str:
    if SHA1_RE.fullmatch(value) is None:
        raise VerificationError(f"{label} is not one lowercase SHA-1 object ID")
    return value


def require_sha256(value: str, label: str) -> str:
    if SHA256_RE.fullmatch(value) is None:
        raise VerificationError(f"{label} is not one lowercase SHA-256 digest")
    return value


def expected_refs(p28_commit: str) -> dict[str, str]:
    return {
        BRANCH_REF: p28_commit,
        P27_BRANCH_REF: P27_COMMIT,
        P27_TAG_REF: P27_TAG_OBJECT,
        P26_CHECKPOINT_REF: P26_CHECKPOINT_OBJECT,
        P26_SOURCE_REF: P26_SOURCE_OBJECT,
    }


def parse_v2_bundle_header(bundle: Path) -> tuple[dict[str, str], list[str]]:
    """Read only the textual header of the already-private bundle copy."""

    refs: dict[str, str] = {}
    prerequisites: list[str] = []
    with bundle.open("rb") as handle:
        signature = handle.readline()
        if signature != b"# v2 git bundle\n":
            raise VerificationError("control bundle must use the closed v2 bundle format")
        while True:
            line = handle.readline()
            if line == b"":
                raise VerificationError("bundle header ended before its required blank line")
            if line == b"\n":
                break
            if b"\x00" in line or not line.endswith(b"\n"):
                raise VerificationError("malformed bundle header line")
            try:
                text = line[:-1].decode("ascii")
            except UnicodeDecodeError as exc:
                raise VerificationError("bundle header is not ASCII") from exc
            if text.startswith("-"):
                prerequisite = text[1:].split(" ", 1)[0]
                require_sha1(prerequisite, "bundle prerequisite")
                prerequisites.append(prerequisite)
                continue
            parts = text.split(" ")
            if len(parts) != 2:
                raise VerificationError("bundle advertised-ref line is malformed")
            object_id, refname = parts
            require_sha1(object_id, "advertised object")
            if refname in refs:
                raise VerificationError(f"duplicate advertised ref: {refname}")
            refs[refname] = object_id
    return refs, prerequisites


def list_refs(repository: Path) -> dict[str, str]:
    output = run_git(
        ["-C", str(repository), "for-each-ref", "--format=%(objectname) %(refname)"],
    ).stdout
    refs: dict[str, str] = {}
    for raw_line in output.splitlines():
        try:
            object_raw, ref_raw = raw_line.split(b" ", 1)
            object_id = object_raw.decode("ascii")
            refname = ref_raw.decode("ascii")
        except (ValueError, UnicodeDecodeError) as exc:
            raise VerificationError("repository ref inventory is malformed") from exc
        require_sha1(object_id, "repository ref object")
        if refname in refs:
            raise VerificationError(f"duplicate repository ref: {refname}")
        refs[refname] = object_id
    return refs


def git_text(repository: Path, arguments: Sequence[str]) -> str:
    raw = run_git(["-C", str(repository), *arguments]).stdout
    try:
        return raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise VerificationError(f"non-ASCII Git output for {' '.join(arguments)}") from exc


def assert_attempt_absent(attempt_root: Path, boundary: str) -> None:
    if attempt_root.exists() or attempt_root.is_symlink():
        raise VerificationError(f"attempt root exists {boundary}: {attempt_root}")


def validate_runtime_attempt(
    attempt_root: Path,
    *,
    control: Path | None = None,
) -> dict[str, object]:
    """Validate phase two's frozen runtime without authorizing localization."""

    if not attempt_root.is_dir() or attempt_root.is_symlink():
        raise VerificationError("runtime-review requires one existing nonsymlink attempt root")
    evidence = attempt_root / "evidence"
    if not evidence.is_dir() or evidence.is_symlink():
        raise VerificationError("runtime-review requires one existing nonsymlink evidence root")
    top_level = sorted(child.name for child in attempt_root.iterdir())
    if top_level != sorted(RUNTIME_ATTEMPT_TOP_LEVEL):
        raise VerificationError(
            "runtime-review attempt-root inventory differs: "
            f"expected {sorted(RUNTIME_ATTEMPT_TOP_LEVEL)!r}, got {top_level!r}"
        )
    evidence_inventory = sorted(child.name for child in evidence.iterdir())
    if evidence_inventory != sorted(RUNTIME_EVIDENCE_ALLOWLIST):
        raise VerificationError(
            "runtime-review evidence inventory differs: "
            f"expected {sorted(RUNTIME_EVIDENCE_ALLOWLIST)!r}, got {evidence_inventory!r}"
        )
    records: dict[str, object] = {}
    for label, basename in RUNTIME_EVIDENCE_FILES.items():
        path = evidence / basename
        raw, file_record = stable_read_nofollow(path)
        if not raw:
            raise VerificationError(f"runtime-review {label} artifact is empty")
        if control is not None:
            committed = control / "experiments/training" / basename
            if not committed.is_file() or committed.is_symlink():
                raise VerificationError(f"committed runtime-review {label} is absent")
            committed_raw, _committed_record = stable_read_nofollow(committed)
            if committed_raw != raw:
                raise VerificationError(
                    f"committed runtime-review {label} differs from native evidence"
                )
        records[label] = {
            "evidence_relative_path": f"evidence/{basename}",
            "committed_relative_path": f"experiments/training/{basename}",
            "sha256": file_record.sha256,
            "byte_count": file_record.byte_count,
            "mode_octal": file_record.mode_octal,
            "uid": file_record.uid,
            "gid": file_record.gid,
        }
    for basename in RUNTIME_FORBIDDEN_LOCALIZATION_FILES:
        path = evidence / basename
        if path.exists() or path.is_symlink():
            raise VerificationError(
                f"localization evidence exists before runtime receipt replay: {basename}"
            )
    for child in evidence.iterdir():
        if any(child.name.startswith(prefix) for prefix in RUNTIME_FORBIDDEN_PREFIXES):
            raise VerificationError(
                "run-phase evidence exists before runtime receipt replay: " + child.name
            )
    return {
        "mode": "existing_frozen_runtime_before_localization",
        "attempt_root_existing": True,
        "evidence_root_existing": True,
        "runtime_artifacts": records,
        "exact_attempt_top_level": RUNTIME_ATTEMPT_TOP_LEVEL,
        "exact_prepare_evidence_inventory": RUNTIME_EVIDENCE_ALLOWLIST,
        "forbidden_localization_files": RUNTIME_FORBIDDEN_LOCALIZATION_FILES,
        "forbidden_run_phase_prefixes": RUNTIME_FORBIDDEN_PREFIXES,
        "all_forbidden_localization_files_absent": True,
        "attempt_root_created_or_mutated_by_verifier": False,
    }


def assert_no_object_indirections(repository: Path, *, bare: bool) -> None:
    git_dir = repository if bare else repository / ".git"
    forbidden_files = [
        git_dir / "shallow",
        git_dir / "objects/info/alternates",
        git_dir / "info/grafts",
    ]
    for candidate in forbidden_files:
        if candidate.exists() or candidate.is_symlink():
            raise VerificationError(f"forbidden Git object indirection exists: {candidate}")
    shallow = git_text(repository, ["rev-parse", "--is-shallow-repository"])
    if shallow != "false":
        raise VerificationError("control repository is shallow")
    replace = run_git(
        ["-C", str(repository), "for-each-ref", "--format=%(refname)", "refs/replace"],
    ).stdout
    if replace.strip():
        raise VerificationError("control repository contains replacement refs")


def assert_object_contract(repository: Path, refs: Mapping[str, str]) -> None:
    expected_types = {
        BRANCH_REF: "commit",
        P27_BRANCH_REF: "commit",
        P27_TAG_REF: "tag",
        P26_CHECKPOINT_REF: "tag",
        P26_SOURCE_REF: "tag",
    }
    for refname, expected_type in expected_types.items():
        actual_type = git_text(repository, ["cat-file", "-t", refs[refname]])
        if actual_type != expected_type:
            raise VerificationError(
                f"{refname} has type {actual_type!r}, expected {expected_type!r}"
            )
    peeled = {
        P27_TAG_REF: P27_COMMIT,
        P26_CHECKPOINT_REF: P26_CHECKPOINT_COMMIT,
        P26_SOURCE_REF: P26_SOURCE_COMMIT,
    }
    for refname, expected_commit in peeled.items():
        actual = git_text(repository, ["rev-parse", f"{refname}^{{}}"])
        if actual != expected_commit:
            raise VerificationError(f"{refname} peels to {actual}, expected {expected_commit}")
        if git_text(repository, ["cat-file", "-t", actual]) != "commit":
            raise VerificationError(f"peeled target for {refname} is not a commit")
    if git_text(repository, ["rev-parse", f"{P27_BRANCH_REF}^{{tree}}"]) != P27_TREE:
        raise VerificationError("P27 branch tree differs")


def assert_phase_history(
    repository: Path,
    *,
    phase: str,
    p28_commit: str,
    p28_tree: str,
    source_commit: str | None,
    source_tree: str | None,
) -> dict[str, object]:
    if git_text(repository, ["rev-parse", f"{p28_commit}^{{tree}}"]) != p28_tree:
        raise VerificationError("P28 commit/tree binding differs")
    parents = git_text(repository, ["rev-list", "--parents", "-n", "1", p28_commit]).split()
    if phase == "source":
        if parents != [p28_commit, P27_COMMIT]:
            raise VerificationError("P28 source freeze is not one direct child of P27")
        return {
            "source_commit": p28_commit,
            "source_tree": p28_tree,
            "runtime_review_direct_child": False,
            "runtime_review_exact_delta": [],
        }
    if source_commit is None or source_tree is None:
        raise VerificationError("runtime-review phase requires source commit and tree")
    require_sha1(source_commit, "source commit")
    require_sha1(source_tree, "source tree")
    if parents != [p28_commit, source_commit]:
        raise VerificationError("P28 runtime review is not one direct child of source freeze")
    if git_text(repository, ["rev-parse", f"{source_commit}^{{tree}}"]) != source_tree:
        raise VerificationError("P28 source commit/tree binding differs")
    source_parents = git_text(
        repository, ["rev-list", "--parents", "-n", "1", source_commit]
    ).split()
    if source_parents != [source_commit, P27_COMMIT]:
        raise VerificationError("P28 source freeze history differs")
    delta_raw = run_git(
        [
            "-C",
            str(repository),
            "diff",
            "--name-status",
            source_commit,
            p28_commit,
            "--",
        ]
    ).stdout
    try:
        delta = delta_raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise VerificationError("runtime-review delta is not UTF-8") from exc
    if delta != RUNTIME_DELTA:
        raise VerificationError(
            f"runtime-review delta differs: expected {RUNTIME_DELTA!r}, got {delta!r}"
        )
    return {
        "source_commit": source_commit,
        "source_tree": source_tree,
        "runtime_review_direct_child": True,
        "runtime_review_exact_delta": RUNTIME_DELTA,
    }


def initialize_empty_bare(closure: Path) -> None:
    if closure.exists() or closure.is_symlink():
        raise VerificationError(f"phase closure must be absent: {closure}")
    run_git(["init", "--bare", str(closure)])
    if list_refs(closure):
        raise VerificationError("new bare closure unexpectedly contains refs")
    count = git_text(closure, ["count-objects", "-v"])
    parsed = dict(line.split(": ", 1) for line in count.splitlines() if ": " in line)
    if parsed.get("count") != "0" or parsed.get("packs") != "0":
        raise VerificationError("new bare closure was not object-empty")
    assert_no_object_indirections(closure, bare=True)


def verify_and_fill_closure(
    closure: Path,
    private_bundle: Path,
    refs: Mapping[str, str],
) -> None:
    run_git(["-C", str(closure), "bundle", "verify", str(private_bundle)])
    for refname in refs:
        run_git(
            [
                "-C",
                str(closure),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(private_bundle),
                f"+{refname}:{refname}",
            ]
        )
    if list_refs(closure) != dict(refs):
        raise VerificationError("fresh closure ref inventory differs from advertised refs")
    assert_no_object_indirections(closure, bare=True)
    assert_object_contract(closure, refs)
    run_git(["-C", str(closure), "fsck", "--full", "--strict", "--no-dangling"])


def validate_existing_closure(closure: Path, refs: Mapping[str, str]) -> None:
    if not closure.is_dir() or closure.is_symlink():
        raise VerificationError("reviewed phase closure is absent or a symlink")
    if git_text(closure, ["rev-parse", "--is-bare-repository"]) != "true":
        raise VerificationError("reviewed phase closure is not bare")
    if list_refs(closure) != dict(refs):
        raise VerificationError("reviewed phase closure ref inventory differs")
    assert_no_object_indirections(closure, bare=True)
    assert_object_contract(closure, refs)
    run_git(["-C", str(closure), "fsck", "--full", "--strict", "--no-dangling"])


def _fetch_closed_refs(source: Path, target: Path, refs: Mapping[str, str]) -> None:
    for refname in refs:
        run_git(
            [
                "-C",
                str(target),
                "fetch",
                "--no-tags",
                "--no-write-fetch-head",
                str(source),
                f"+{refname}:{refname}",
            ]
        )


def create_source_control(
    control: Path, closure: Path, refs: Mapping[str, str], p28_commit: str
) -> None:
    if control.exists() or control.is_symlink():
        raise VerificationError("source control checkout must be absent")
    # Keep HEAD on a distinct unborn name while the closed P28 branch ref is
    # fetched. Git refuses to fetch into a checked-out branch, even before its
    # first checkout.
    run_git(["init", "--initial-branch=p28-verifier-unborn", str(control)])
    run_git(["-C", str(control), "config", "core.autocrlf", "false"])
    run_git(["-C", str(control), "config", "core.filemode", "true"])
    _fetch_closed_refs(closure, control, refs)
    run_git(["-C", str(control), "checkout", "--detach", p28_commit])


def advance_runtime_control(
    control: Path,
    closure: Path,
    refs: Mapping[str, str],
    *,
    source_commit: str,
    source_tree: str,
    runtime_commit: str,
    expected_verifier_sha256: str,
) -> None:
    validate_control_checkout(
        control,
        expected_head=source_commit,
        expected_tree=source_tree,
        refs=expected_refs(source_commit),
        run_reconstruction=True,
        expected_verifier_sha256=expected_verifier_sha256,
    )
    _fetch_closed_refs(closure, control, refs)
    run_git(["-C", str(control), "checkout", "--detach", runtime_commit])


def run_p27_reconstruction(control: Path) -> dict[str, object]:
    reconstructor = control / P27_RECONSTRUCTOR
    if not reconstructor.is_file() or reconstructor.is_symlink():
        raise VerificationError("frozen P27 reconstructor is absent or a symlink")
    _reconstructor_raw, reconstructor_record = stable_read_nofollow(reconstructor)
    if reconstructor_record.sha256 != P27_RECONSTRUCTOR_SHA256:
        raise VerificationError("frozen P27 reconstructor bytes differ")
    process = subprocess.run(
        [sys.executable, str(reconstructor)],
        cwd=control,
        env=_clean_git_environment(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        check=False,
    )
    if process.returncode != 0:
        stderr = process.stderr.decode("utf-8", errors="replace").strip()
        raise VerificationError(f"P27 reconstruction failed: {stderr}")
    payload = strict_json_loads(process.stdout)
    if not isinstance(payload, Mapping):
        raise VerificationError("P27 reconstruction output is not an object")
    checks = payload.get("checks")
    if not isinstance(checks, Mapping) or len(checks) != 23:
        raise VerificationError("P27 reconstruction does not contain exactly 23 checks")
    if any(value is not True for value in checks.values()):
        raise VerificationError("P27 reconstruction is not 23/23 true")
    if payload.get("internally_consistent") is not True:
        raise VerificationError("P27 reconstruction is not internally consistent")
    if payload.get("schema_version") != P27_RECONSTRUCTION_SCHEMA:
        raise VerificationError("P27 reconstruction schema differs")
    if payload.get("canonical_sha256") != P27_CONTRACT_SHA256:
        raise VerificationError("P27 canonical contract digest differs")
    return {
        "reconstructor_path": P27_RECONSTRUCTOR,
        "reconstructor_sha256": P27_RECONSTRUCTOR_SHA256,
        "schema_version": P27_RECONSTRUCTION_SCHEMA,
        "canonical_sha256": P27_CONTRACT_SHA256,
        "check_count": 23,
        "true_check_count": 23,
        "internally_consistent": True,
    }


def validate_control_checkout(
    control: Path,
    *,
    expected_head: str,
    expected_tree: str,
    refs: Mapping[str, str],
    run_reconstruction: bool,
    expected_verifier_sha256: str,
) -> dict[str, object] | None:
    if not control.is_dir() or control.is_symlink():
        raise VerificationError("control checkout is absent or a symlink")
    if git_text(control, ["rev-parse", "--is-bare-repository"]) != "false":
        raise VerificationError("control checkout is bare")
    if run_git(["-C", str(control), "symbolic-ref", "-q", "HEAD"], check=False).returncode == 0:
        raise VerificationError("control checkout HEAD is not detached")
    if git_text(control, ["rev-parse", "HEAD"]) != expected_head:
        raise VerificationError("control checkout HEAD differs")
    if git_text(control, ["rev-parse", "HEAD^{tree}"]) != expected_tree:
        raise VerificationError("control checkout tree differs")
    if run_git(
        ["-C", str(control), "status", "--porcelain=v1", "-z", "--untracked-files=all"]
    ).stdout:
        raise VerificationError("control checkout is not exactly clean")
    if list_refs(control) != dict(refs):
        raise VerificationError("control checkout ref inventory differs")
    assert_no_object_indirections(control, bare=False)
    assert_object_contract(control, refs)
    run_git(["-C", str(control), "fsck", "--full", "--strict", "--no-dangling"])
    checked_in_verifier = control / VERIFIER_SOURCE
    if not checked_in_verifier.is_file() or checked_in_verifier.is_symlink():
        raise VerificationError("checked-in P28 verifier is absent or a symlink")
    _checked_in_raw, checked_in_record = stable_read_nofollow(checked_in_verifier)
    if checked_in_record.sha256 != expected_verifier_sha256:
        raise VerificationError("checked-in P28 verifier bytes differ from executing verifier")
    if run_reconstruction:
        return run_p27_reconstruction(control)
    return None


def _path_is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def validate_layout(args: argparse.Namespace) -> dict[str, Path]:
    phase_layout = PHASE_LAYOUT[args.phase]
    paths = {
        "transport": Path(args.transport_root).absolute(),
        "bundle": Path(args.bundle).absolute(),
        "closure": Path(args.closure_repository).absolute(),
        "control": Path(args.control_checkout).absolute(),
        "attempt": Path(args.attempt_root).absolute(),
        "receipt": Path(args.receipt_output or args.verify_existing_receipt).absolute(),
    }
    transport = paths["transport"]
    if not transport.is_dir() or transport.is_symlink():
        raise VerificationError("transport root is absent or a symlink")
    resolved_transport = transport.resolve(strict=True)
    if resolved_transport != transport:
        raise VerificationError("transport root contains a symlink or lexical alias")
    expected = {
        "bundle": transport / phase_layout["bundle"],
        "closure": transport / phase_layout["closure"],
        "receipt": transport / phase_layout["receipt"],
        "control": transport.parent / "control-source",
        "attempt": transport.parent / "attempt-20260906-01",
    }
    for label, expected_path in expected.items():
        if paths[label] != expected_path:
            raise VerificationError(
                f"{label} path differs: expected {expected_path}, got {paths[label]}"
            )
    if not _path_is_within(paths["bundle"], transport):
        raise VerificationError("bundle is outside the transport root")
    if paths["receipt"] == paths["bundle"]:
        raise VerificationError("receipt aliases its input bundle")
    if _path_is_within(paths["receipt"], paths["control"]):
        raise VerificationError("receipt would be committed inside the control checkout")
    return paths


def build_receipt(
    *,
    phase: str,
    paths: Mapping[str, Path],
    bundle_file: StableFile,
    expected_bundle_sha256: str,
    expected_bundle_byte_count: int,
    p28_commit: str,
    p28_tree: str,
    refs: Mapping[str, str],
    history: Mapping[str, object],
    reconstruction: Mapping[str, object],
    attempt_guard: Mapping[str, object],
    verifier_sha256: str,
) -> dict[str, object]:
    claim_boundary = (
        "Offline Git object/ref/control-checkout provenance only. The source receipt "
        "records no attempt-root creation, container, GPU, CUDA, mapping, gradient, "
        "candidate, optimizer update, or training observation."
        if phase == "source"
        else "Offline Git object/ref/control-checkout provenance plus identity of the "
        "already prepared frozen runtime. The runtime-review receipt records no "
        "localization invocation, mapping observation, gradient, candidate, optimizer "
        "update, or training observation."
    )
    ref_records = [
        {
            "refname": refname,
            "object_id": object_id,
            "object_type": "commit" if refname in (BRANCH_REF, P27_BRANCH_REF) else "tag",
            "peeled_commit": {
                P27_TAG_REF: P27_COMMIT,
                P26_CHECKPOINT_REF: P26_CHECKPOINT_COMMIT,
                P26_SOURCE_REF: P26_SOURCE_COMMIT,
            }.get(refname),
        }
        for refname, object_id in refs.items()
    ]
    return {
        "schema_version": SCHEMA,
        "status": STATUS_BY_PHASE[phase],
        "phase": phase,
        "claim_boundary": claim_boundary,
        "transport": {
            "root": str(paths["transport"]),
            "bundle_path": str(paths["bundle"]),
            "bundle_format": "v2",
            "bundle_sha256": expected_bundle_sha256,
            "bundle_byte_count": expected_bundle_byte_count,
            "bundle_mode_octal": bundle_file.mode_octal,
            "bundle_uid": bundle_file.uid,
            "bundle_gid": bundle_file.gid,
            "bundle_device": bundle_file.device,
            "bundle_inode": bundle_file.inode,
            "bundle_link_count": bundle_file.link_count,
            "stable_nofollow_read_count": 1,
            "private_copy_mode_octal_before_git_use": "0400",
            "all_git_verification_used_private_authenticated_read_only_copy": True,
            "private_copy_identity_hash_size_mode_revalidated_after_git_use": True,
            "network_used": False,
        },
        "verifier": {
            "canonical_source_path": VERIFIER_SOURCE,
            "source_sha256": verifier_sha256,
            "executing_source_authenticated_by_stable_nofollow_read": True,
            "executing_bootstrap_mode_octal": "0600",
            "checked_in_control_source_matches": True,
            "executing_absolute_path_excluded_for_deterministic_replay": True,
        },
        "closed_bundle": {
            "advertised_ref_count": 5,
            "advertised_refs": ref_records,
            "prerequisite_count": 0,
            "prerequisites": [],
            "verified_in_new_empty_bare_repository": True,
            "each_ref_fetched_explicitly": True,
            "fsck_full_strict_passed": True,
            "tag_types_and_peels_verified": True,
        },
        "closure": {
            "path": str(paths["closure"]),
            "created_from_absent_empty_bare_repository": True,
            "exact_closed_ref_inventory": True,
            "shallow": False,
            "alternates": False,
            "grafts": False,
            "replace_refs": False,
        },
        "control_checkout": {
            "path": str(paths["control"]),
            "transition": (
                "created_from_verified_source_closure"
                if phase == "source"
                else "advanced_from_verified_source_to_runtime_review_closure"
            ),
            "head": p28_commit,
            "tree": p28_tree,
            "detached": True,
            "clean": True,
            "exact_closed_ref_inventory": True,
            "shallow": False,
            "alternates": False,
            "grafts": False,
            "replace_refs": False,
        },
        "history": dict(history),
        "p27_reconstruction": dict(reconstruction),
        "attempt_guard": {
            "attempt_root": str(paths["attempt"]),
            **dict(attempt_guard),
        },
        "receipt_integrity": {
            "receipt_path": str(paths["receipt"]),
            "external_to_control_checkout": True,
            "created_with_no_overwrite": True,
            "receipt_sha256_field_present": False,
            "self_reference_excluded_by_design": True,
            "review_requires_external_sha256": True,
        },
    }


def write_receipt_exclusive(path: Path, raw: bytes) -> None:
    try:
        descriptor = os.open(
            path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0),
            0o600,
        )
    except OSError as exc:
        raise VerificationError(f"cannot create receipt without overwrite: {exc}") from exc
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise VerificationError("short receipt write")
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=sorted(PHASE_LAYOUT), required=True)
    parser.add_argument("--transport-root", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--expected-bundle-sha256", required=True)
    parser.add_argument("--expected-bundle-byte-count", required=True, type=int)
    parser.add_argument("--closure-repository", required=True)
    parser.add_argument("--control-checkout", required=True)
    parser.add_argument("--attempt-root", required=True)
    parser.add_argument("--expected-p28-commit", required=True)
    parser.add_argument("--expected-p28-tree", required=True)
    parser.add_argument("--bootstrap-verifier", required=True)
    parser.add_argument("--expected-verifier-sha256", required=True)
    parser.add_argument("--expected-source-commit")
    parser.add_argument("--expected-source-tree")
    receipts = parser.add_mutually_exclusive_group(required=True)
    receipts.add_argument("--receipt-output")
    receipts.add_argument("--verify-existing-receipt")
    parser.add_argument("--expected-receipt-sha256")
    return parser


def verify(args: argparse.Namespace) -> dict[str, object]:
    expected_verifier_sha256 = require_sha256(
        args.expected_verifier_sha256, "expected verifier digest"
    )
    authenticate_executing_verifier(expected_verifier_sha256, Path(args.bootstrap_verifier))
    expected_bundle_sha256 = require_sha256(args.expected_bundle_sha256, "expected bundle digest")
    if args.expected_bundle_byte_count <= 0:
        raise VerificationError("expected bundle byte count must be positive")
    p28_commit = require_sha1(args.expected_p28_commit, "expected P28 commit")
    p28_tree = require_sha1(args.expected_p28_tree, "expected P28 tree")
    if args.phase == "source":
        if args.expected_source_commit is not None or args.expected_source_tree is not None:
            raise VerificationError("source phase forbids runtime source-history arguments")
    elif args.expected_source_commit is None or args.expected_source_tree is None:
        raise VerificationError("runtime-review phase requires source commit and tree")
    if args.receipt_output is not None:
        if args.expected_receipt_sha256 is not None:
            raise VerificationError("receipt creation forbids an expected receipt digest")
    else:
        if args.expected_receipt_sha256 is None:
            raise VerificationError("receipt replay requires its reviewed SHA-256")
        require_sha256(args.expected_receipt_sha256, "expected receipt digest")

    paths = validate_layout(args)
    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "before bundle read")
        initial_attempt_guard: dict[str, object] = {
            "mode": "attempt_root_absent_through_source_verification",
            "absent_before_bundle_read": True,
            "absent_before_closure_creation": True,
            "absent_before_control_transition": True,
            "absent_at_validation_end": True,
            "attempt_root_created_or_mutated_by_verifier": False,
        }
    else:
        initial_attempt_guard = validate_runtime_attempt(paths["attempt"])
    refs = expected_refs(p28_commit)
    replay = args.verify_existing_receipt is not None

    if not paths["bundle"].is_file() or paths["bundle"].is_symlink():
        raise VerificationError("phase bundle is absent or a symlink")
    if replay:
        if not paths["closure"].is_dir() or paths["closure"].is_symlink():
            raise VerificationError("reviewed phase closure is absent or a symlink")
        if not paths["control"].is_dir() or paths["control"].is_symlink():
            raise VerificationError("reviewed control checkout is absent or a symlink")
    else:
        if paths["closure"].exists() or paths["closure"].is_symlink():
            raise VerificationError("new phase closure path already exists")
        if paths["receipt"].exists() or paths["receipt"].is_symlink():
            raise VerificationError("new receipt path already exists")
        if args.phase == "source" and (paths["control"].exists() or paths["control"].is_symlink()):
            raise VerificationError("source control checkout path already exists")
        if args.phase == "runtime-review" and (
            not paths["control"].is_dir() or paths["control"].is_symlink()
        ):
            raise VerificationError("runtime-review requires the source control checkout")

    with tempfile.TemporaryDirectory(prefix="p28-bundle-verification-") as temporary:
        private_bundle = Path(temporary) / "authenticated.bundle"
        bundle_file = stable_copy_nofollow(paths["bundle"], private_bundle)
        if bundle_file.sha256 != expected_bundle_sha256:
            raise VerificationError("transport bundle SHA-256 differs")
        if bundle_file.byte_count != args.expected_bundle_byte_count:
            raise VerificationError("transport bundle byte count differs")
        private_bundle_file = seal_private_bundle(private_bundle, bundle_file)
        advertised, prerequisites = parse_v2_bundle_header(private_bundle)
        if prerequisites:
            raise VerificationError("bundle contains forbidden prerequisite lines")
        if advertised != refs:
            raise VerificationError(
                f"bundle advertised refs differ: expected {refs!r}, got {advertised!r}"
            )
        if args.phase == "source":
            assert_attempt_absent(paths["attempt"], "before closure validation")
        else:
            validate_runtime_attempt(paths["attempt"])
        if replay:
            # The reviewed receipt authenticates that this durable closure was born empty;
            # replay validates its current closed state and the bundle again in a fresh,
            # ephemeral empty bare repository so ambient objects still cannot mask gaps.
            validate_existing_closure(paths["closure"], refs)
            ephemeral_closure = Path(temporary) / "replay-empty.git"
            initialize_empty_bare(ephemeral_closure)
            verify_and_fill_closure(ephemeral_closure, private_bundle, refs)
            assert_phase_history(
                ephemeral_closure,
                phase=args.phase,
                p28_commit=p28_commit,
                p28_tree=p28_tree,
                source_commit=args.expected_source_commit,
                source_tree=args.expected_source_tree,
            )
        else:
            initialize_empty_bare(paths["closure"])
            if args.phase == "source":
                assert_attempt_absent(paths["attempt"], "after empty closure creation")
            else:
                validate_runtime_attempt(paths["attempt"])
            verify_and_fill_closure(paths["closure"], private_bundle, refs)
        revalidate_private_bundle(private_bundle, private_bundle_file)

    history = assert_phase_history(
        paths["closure"],
        phase=args.phase,
        p28_commit=p28_commit,
        p28_tree=p28_tree,
        source_commit=args.expected_source_commit,
        source_tree=args.expected_source_tree,
    )
    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "before control checkout transition")
    else:
        validate_runtime_attempt(paths["attempt"])
    if not replay:
        if args.phase == "source":
            create_source_control(paths["control"], paths["closure"], refs, p28_commit)
        else:
            advance_runtime_control(
                paths["control"],
                paths["closure"],
                refs,
                source_commit=args.expected_source_commit,
                source_tree=args.expected_source_tree,
                runtime_commit=p28_commit,
                expected_verifier_sha256=expected_verifier_sha256,
            )
    reconstruction = validate_control_checkout(
        paths["control"],
        expected_head=p28_commit,
        expected_tree=p28_tree,
        refs=refs,
        run_reconstruction=True,
        expected_verifier_sha256=expected_verifier_sha256,
    )
    assert reconstruction is not None
    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "at validation end")
        final_attempt_guard = initial_attempt_guard
    else:
        final_attempt_guard = validate_runtime_attempt(paths["attempt"], control=paths["control"])
        if final_attempt_guard != initial_attempt_guard:
            raise VerificationError("frozen runtime evidence changed during bundle verification")
    receipt = build_receipt(
        phase=args.phase,
        paths=paths,
        bundle_file=bundle_file,
        expected_bundle_sha256=expected_bundle_sha256,
        expected_bundle_byte_count=args.expected_bundle_byte_count,
        p28_commit=p28_commit,
        p28_tree=p28_tree,
        refs=refs,
        history=history,
        reconstruction=reconstruction,
        attempt_guard=final_attempt_guard,
        verifier_sha256=expected_verifier_sha256,
    )
    expected_receipt_raw = canonical_json_bytes(receipt)
    if replay:
        reviewed_raw, reviewed_file = stable_read_nofollow(paths["receipt"])
        if reviewed_file.sha256 != args.expected_receipt_sha256:
            raise VerificationError("reviewed receipt SHA-256 differs")
        reviewed = strict_json_loads(reviewed_raw)
        if reviewed != receipt:
            raise VerificationError("reviewed receipt does not reconstruct from live state")
        if reviewed_raw != expected_receipt_raw:
            raise VerificationError("reviewed receipt is not canonical JSON")
    else:
        write_receipt_exclusive(paths["receipt"], expected_receipt_raw)
    if args.phase == "source":
        assert_attempt_absent(paths["attempt"], "after receipt handling")
    elif (
        validate_runtime_attempt(paths["attempt"], control=paths["control"]) != final_attempt_guard
    ):
        raise VerificationError("frozen runtime evidence changed during receipt handling")
    return {
        "schema_version": SCHEMA,
        "status": "receipt_created" if not replay else "reviewed_receipt_replayed",
        "phase": args.phase,
        "bundle_sha256": expected_bundle_sha256,
        "bundle_byte_count": args.expected_bundle_byte_count,
        "p28_commit": p28_commit,
        "p28_tree": p28_tree,
        "receipt_path": str(paths["receipt"]),
        "receipt_sha256": sha256_bytes(expected_receipt_raw),
        "attempt_boundary": (
            "absent" if args.phase == "source" else "existing_frozen_runtime_no_localization"
        ),
        "p27_reconstruction": "23/23",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        summary = verify(args)
    except VerificationError as exc:
        print(f"P28 control-bundle verification failed: {exc}", file=sys.stderr)
        return 1
    sys.stdout.buffer.write(canonical_json_bytes(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
