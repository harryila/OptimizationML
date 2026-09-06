#!/usr/bin/env python3
"""Fail-closed local verifier for the two P30 transfer bundles.

This verifier runs on the client *before* either bundle is uploaded.  It reads
the operator-created bundle through a no-follow descriptor, copies those exact
bytes into a private directory, asks Git to verify and unpack that private
copy, and then checks the complete advertised ref/object/type/peel map.  The
only successful stdout is one canonical JSON record containing the SHA-256 and
byte count that the later transport receipt must bind.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

SCHEMA = "passive-muon-p30-local-bundle-verification-v1"
P30_REF = "refs/heads/p30-umask-bound-control-seal"
P29_COMMIT = "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"


class VerificationError(RuntimeError):
    """The local bundle failed a frozen P30 condition."""


@dataclass(frozen=True)
class RefSpec:
    object_id: str
    object_type: str
    peeled_commit: str


FIXED_REFS: Mapping[str, RefSpec] = {
    "refs/heads/p29-permission-safe-bundle-localization-bridge": RefSpec(
        P29_COMMIT, "commit", P29_COMMIT
    ),
    "refs/tags/p29-orchestrator-source-mode-diagnostic": RefSpec(
        "cf3c60b1d5c004b9e601a6afbf630358f80f1d79", "tag", P29_COMMIT
    ),
    "refs/heads/p28-bundle-complete-localization-bridge": RefSpec(
        "7274367f2b05fb3c9ed9f876a59be647703bbdc2",
        "commit",
        "7274367f2b05fb3c9ed9f876a59be647703bbdc2",
    ),
    "refs/tags/p28-control-parent-permission-diagnostic": RefSpec(
        "f3c81a2f14f853f369015a4f81b6695b52eec74e",
        "tag",
        "7274367f2b05fb3c9ed9f876a59be647703bbdc2",
    ),
    "refs/heads/p27-cuda-deleted-mapping-localization": RefSpec(
        "ec63550331925ded158e3f389e294e4d1f12db3a",
        "commit",
        "ec63550331925ded158e3f389e294e4d1f12db3a",
    ),
    "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic": RefSpec(
        "c88b98eae9be2458abde45b05d3dccfe09c0c7ed",
        "tag",
        "ec63550331925ded158e3f389e294e4d1f12db3a",
    ),
    "refs/tags/p26-permission-safe-acquisition-checkpoint": RefSpec(
        "bacad707d3779bfa10957e18cb4c69b1a7f0cbce",
        "tag",
        "5429da23ff18888daa2312c530a4587780484d8b",
    ),
    "refs/tags/p26-attempt02-acquisition-source": RefSpec(
        "f35a7dca8f6bc39e9748e79b6712fab4a203396b",
        "tag",
        "185e444afc0b44ca0a09b1bde49a6b6fa3973355",
    ),
}


def _require_sha1(value: str, label: str) -> None:
    if len(value) != 40 or any(character not in "0123456789abcdef" for character in value):
        raise VerificationError(f"{label} is not one lowercase SHA-1 object ID")


def _require_sha256(value: str, label: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise VerificationError(f"{label} is not one lowercase SHA-256 digest")


def _stat_identity(value: os.stat_result) -> tuple[int, ...]:
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


def _directory_identity(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)


class StableFile:
    """Componentwise no-follow descriptor guard for one absolute file."""

    def __init__(self, path: Path, *, require_single_link: bool = True) -> None:
        raw = os.fspath(path)
        if (
            not os.path.isabs(raw)
            or PurePosixPath(raw).as_posix() != raw
            or os.path.normpath(raw) != raw
        ):
            raise VerificationError("bundle and verifier paths must be canonical absolute paths")
        components = [item for item in PurePosixPath(raw).parts if item != "/"]
        if not components:
            raise VerificationError("a file path cannot name the filesystem root")

        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        self.path = path
        self._fds: list[int] = []
        self._parts: list[str] = []
        self._initial: list[os.stat_result] = []
        try:
            current = os.open("/", directory_flags)
            self._fds.append(current)
            self._parts.append("/")
            self._initial.append(os.fstat(current))
            for component in components[:-1]:
                current = os.open(component, directory_flags, dir_fd=current)
                opened = os.fstat(current)
                if not stat.S_ISDIR(opened.st_mode):
                    raise VerificationError(f"non-directory path component: {component}")
                self._fds.append(current)
                self._parts.append(component)
                self._initial.append(opened)
            self.fd = os.open(components[-1], file_flags, dir_fd=current)
            opened = os.fstat(self.fd)
            if not stat.S_ISREG(opened.st_mode):
                raise VerificationError("guarded path is not a regular file")
            if require_single_link and opened.st_nlink != 1:
                raise VerificationError("guarded file must have exactly one hard link")
            self._fds.append(self.fd)
            self._parts.append(components[-1])
            self._initial.append(opened)
        except (OSError, VerificationError) as exc:
            self.close()
            if isinstance(exc, VerificationError):
                raise
            raise VerificationError(f"cannot no-follow open {path}: {exc}") from exc

    def close(self) -> None:
        while getattr(self, "_fds", []):
            os.close(self._fds.pop())

    def __enter__(self) -> StableFile:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def revalidate(self) -> None:
        final_index = len(self._fds) - 1
        for index, (descriptor, initial) in enumerate(zip(self._fds, self._initial, strict=True)):
            identity = _stat_identity if index == final_index else _directory_identity
            if identity(os.fstat(descriptor)) != identity(initial):
                raise VerificationError("an open path component changed during verification")

        # Rewalk by name so a rename/replacement cannot hide behind the held FDs.
        directory_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW
        file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        reopened: list[int] = []
        try:
            current = os.open("/", directory_flags)
            reopened.append(current)
            if _directory_identity(os.fstat(current)) != _directory_identity(self._initial[0]):
                raise VerificationError("filesystem root identity changed")
            for index, component in enumerate(self._parts[1:], start=1):
                flags = file_flags if index == len(self._parts) - 1 else directory_flags
                current = os.open(component, flags, dir_fd=current)
                reopened.append(current)
                identity = _stat_identity if index == final_index else _directory_identity
                if identity(os.fstat(current)) != identity(self._initial[index]):
                    raise VerificationError("guarded path identity changed by name")
        except OSError as exc:
            raise VerificationError(f"cannot revalidate guarded path {self.path}: {exc}") from exc
        finally:
            while reopened:
                os.close(reopened.pop())


def _copy_and_digest(source: StableFile, destination: Path) -> tuple[str, int]:
    output_flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    output_fd = os.open(destination, output_flags, 0o400)
    digest = hashlib.sha256()
    byte_count = 0
    try:
        os.lseek(source.fd, 0, os.SEEK_SET)
        while True:
            chunk = os.read(source.fd, 1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
            byte_count += len(chunk)
            view = memoryview(chunk)
            while view:
                written = os.write(output_fd, view)
                if written <= 0:
                    raise VerificationError("private bundle copy made no write progress")
                view = view[written:]
        os.fsync(output_fd)
    finally:
        os.close(output_fd)
    source.revalidate()
    return digest.hexdigest(), byte_count


def _digest_again(source: StableFile) -> tuple[str, int]:
    digest = hashlib.sha256()
    byte_count = 0
    os.lseek(source.fd, 0, os.SEEK_SET)
    while True:
        chunk = os.read(source.fd, 1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
        byte_count += len(chunk)
    source.revalidate()
    return digest.hexdigest(), byte_count


def _parse_v2_header(bundle: Path) -> tuple[dict[str, str], tuple[str, ...]]:
    refs: dict[str, str] = {}
    prerequisites: list[str] = []
    with bundle.open("rb") as handle:
        if handle.readline() != b"# v2 git bundle\n":
            raise VerificationError("P30 local bundle must use the closed v2 format")
        while True:
            raw = handle.readline()
            if raw == b"":
                raise VerificationError("bundle header ended before its blank line")
            if raw == b"\n":
                break
            if b"\x00" in raw or not raw.endswith(b"\n"):
                raise VerificationError("bundle header line is malformed")
            try:
                text = raw[:-1].decode("ascii")
            except UnicodeDecodeError as exc:
                raise VerificationError("bundle header is not ASCII") from exc
            if text.startswith("-"):
                object_id = text[1:].split(" ", 1)[0]
                _require_sha1(object_id, "bundle prerequisite")
                prerequisites.append(object_id)
                continue
            pieces = text.split(" ")
            if len(pieces) != 2:
                raise VerificationError("advertised-ref line is malformed")
            object_id, refname = pieces
            _require_sha1(object_id, "advertised object")
            if refname in refs:
                raise VerificationError(f"duplicate advertised ref: {refname}")
            refs[refname] = object_id
    return refs, tuple(prerequisites)


def _git_environment() -> dict[str, str]:
    return {
        "HOME": "/nonexistent",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": "/usr/bin:/bin",
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": "/dev/null",
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_TERMINAL_PROMPT": "0",
    }


def _run_git(git: Path, arguments: Sequence[str]) -> bytes:
    try:
        completed = subprocess.run(
            [os.fspath(git), "--no-replace-objects", *arguments],
            check=True,
            env=_git_environment(),
            stdin=subprocess.DEVNULL,
            capture_output=True,
            timeout=300,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = ""
        if isinstance(exc, subprocess.CalledProcessError):
            detail = exc.stderr.decode("utf-8", errors="replace").strip()
        raise VerificationError(f"Git verification failed: {detail or exc}") from exc
    return completed.stdout


def _decode_one_line(raw: bytes, label: str) -> str:
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError as exc:
        raise VerificationError(f"{label} output is not ASCII") from exc
    if not text or "\n" in text:
        raise VerificationError(f"{label} did not produce exactly one line")
    return text


def _expected_refs(p30_commit: str, fixed_refs: Mapping[str, RefSpec]) -> dict[str, RefSpec]:
    return {
        P30_REF: RefSpec(p30_commit, "commit", p30_commit),
        **fixed_refs,
    }


def verify_local_bundle(
    *,
    bundle: Path,
    git: Path,
    phase: str,
    expected_p30_commit: str,
    expected_p30_tree: str,
    expected_p30_parent: str,
    fixed_refs: Mapping[str, RefSpec] = FIXED_REFS,
) -> dict[str, object]:
    """Verify one source/runtime bundle and return its canonical payload."""

    if phase not in {"source", "runtime-review"}:
        raise VerificationError("phase must be source or runtime-review")
    for value, label in (
        (expected_p30_commit, "expected P30 commit"),
        (expected_p30_tree, "expected P30 tree"),
        (expected_p30_parent, "expected P30 parent"),
    ):
        _require_sha1(value, label)
    locked_p29 = fixed_refs.get("refs/heads/p29-permission-safe-bundle-localization-bridge")
    if locked_p29 is None:
        raise VerificationError("terminal P29 branch lock is absent")
    if phase == "source" and expected_p30_parent != locked_p29.object_id:
        raise VerificationError("source P30 commit must directly descend from terminal P29")
    if not git.is_absolute() or git != Path(os.path.realpath(git)):
        raise VerificationError("Git executable must be an absolute canonical path")
    git_stat = os.lstat(git)
    if not stat.S_ISREG(git_stat.st_mode) or stat.S_IMODE(git_stat.st_mode) & 0o022:
        raise VerificationError("Git executable authority is unsafe")

    expected = _expected_refs(expected_p30_commit, fixed_refs)
    if len(expected) != 9:
        raise VerificationError("P30 verifier must lock exactly nine refs")
    for refname, spec in expected.items():
        if not refname.startswith("refs/") or any(character.isspace() for character in refname):
            raise VerificationError("locked refname is malformed")
        _require_sha1(spec.object_id, f"locked object for {refname}")
        _require_sha1(spec.peeled_commit, f"locked peel for {refname}")
        if spec.object_type not in {"commit", "tag"}:
            raise VerificationError("locked ref type must be commit or tag")

    with StableFile(bundle) as guarded_bundle:
        temporary_root = Path(tempfile.mkdtemp(prefix=".p30-local-bundle-", dir=bundle.parent))
        try:
            os.chmod(temporary_root, 0o700)
            private_bundle = temporary_root / "authenticated.bundle"
            digest, byte_count = _copy_and_digest(guarded_bundle, private_bundle)
            if byte_count <= 0:
                raise VerificationError("bundle is empty")
            if stat.S_IMODE(os.lstat(private_bundle).st_mode) != 0o400:
                raise VerificationError("private bundle copy mode differs")

            header_refs, prerequisites = _parse_v2_header(private_bundle)
            expected_objects = {name: spec.object_id for name, spec in expected.items()}
            if prerequisites:
                raise VerificationError("P30 bundle must have zero prerequisites")
            if header_refs != expected_objects:
                raise VerificationError("bundle does not advertise the exact nine-ref object map")

            closure = temporary_root / "closure.git"
            _run_git(git, ["init", "--bare", "--quiet", os.fspath(closure)])
            # This command is deliberately required even though the textual
            # header was already parsed: Git validates pack closure and format.
            _run_git(git, ["-C", os.fspath(closure), "bundle", "verify", os.fspath(private_bundle)])
            unbundled = _run_git(
                git,
                ["-C", os.fspath(closure), "bundle", "unbundle", os.fspath(private_bundle)],
            )
            unbundled_refs: dict[str, str] = {}
            for line in unbundled.splitlines():
                try:
                    object_raw, ref_raw = line.split(b" ", 1)
                    object_id = object_raw.decode("ascii")
                    refname = ref_raw.decode("ascii")
                except (ValueError, UnicodeDecodeError) as exc:
                    raise VerificationError("git bundle unbundle output is malformed") from exc
                _require_sha1(object_id, "unbundled object")
                if refname in unbundled_refs:
                    raise VerificationError("git unbundle returned a duplicate ref")
                unbundled_refs[refname] = object_id
            if unbundled_refs != expected_objects:
                raise VerificationError("Git did not unbundle the exact advertised ref map")

            for refname, spec in expected.items():
                _run_git(
                    git,
                    ["-C", os.fspath(closure), "update-ref", refname, spec.object_id, "0" * 40],
                )
            inventory = _run_git(
                git,
                ["-C", os.fspath(closure), "for-each-ref", "--format=%(objectname) %(refname)"],
            )
            actual_inventory: dict[str, str] = {}
            for line in inventory.splitlines():
                object_raw, ref_raw = line.split(b" ", 1)
                actual_inventory[ref_raw.decode("ascii")] = object_raw.decode("ascii")
            if actual_inventory != expected_objects:
                raise VerificationError("private closure ref inventory differs")

            for refname, spec in expected.items():
                actual_type = _decode_one_line(
                    _run_git(git, ["-C", os.fspath(closure), "cat-file", "-t", spec.object_id]),
                    f"object type for {refname}",
                )
                if actual_type != spec.object_type:
                    raise VerificationError(f"object type differs for {refname}")
                peeled = _decode_one_line(
                    _run_git(
                        git,
                        ["-C", os.fspath(closure), "rev-parse", f"{refname}^{{commit}}"],
                    ),
                    f"peeled commit for {refname}",
                )
                if peeled != spec.peeled_commit:
                    raise VerificationError(f"peeled commit differs for {refname}")

            actual_tree = _decode_one_line(
                _run_git(
                    git,
                    ["-C", os.fspath(closure), "rev-parse", f"{P30_REF}^{{tree}}"],
                ),
                "P30 tree",
            )
            if actual_tree != expected_p30_tree:
                raise VerificationError("phase-specific P30 tree differs")
            parent_line = _decode_one_line(
                _run_git(
                    git,
                    ["-C", os.fspath(closure), "rev-list", "--parents", "-n", "1", P30_REF],
                ),
                "P30 parent line",
            )
            if parent_line != f"{expected_p30_commit} {expected_p30_parent}":
                raise VerificationError("phase-specific P30 commit is not the exact direct child")
            _run_git(git, ["-C", os.fspath(closure), "fsck", "--full", "--strict"])

            digest_after, count_after = _digest_again(guarded_bundle)
            if (digest_after, count_after) != (digest, byte_count):
                raise VerificationError("bundle bytes changed during local verification")
        finally:
            shutil.rmtree(temporary_root)

    return {
        "bundle_byte_count": byte_count,
        "bundle_sha256": digest,
        "p30_commit": expected_p30_commit,
        "p30_tree": expected_p30_tree,
        "phase": phase,
        "schema": SCHEMA,
        "verified_ref_count": 9,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=("source", "runtime-review"))
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--git-executable", required=True, type=Path)
    parser.add_argument("--expected-p30-commit", required=True)
    parser.add_argument("--expected-p30-tree", required=True)
    parser.add_argument("--expected-p30-parent", required=True)
    parser.add_argument("--expected-verifier-source-sha256", required=True)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    fixed_refs: Mapping[str, RefSpec] = FIXED_REFS,
) -> int:
    arguments = _build_parser().parse_args(argv)
    try:
        _require_sha256(
            arguments.expected_verifier_source_sha256,
            "expected local-verifier source digest",
        )
        with StableFile(Path(os.path.abspath(__file__))) as source:
            source_digest, _source_count = _digest_again(source)
        if source_digest != arguments.expected_verifier_source_sha256:
            raise VerificationError("local bundle verifier source digest differs")
        result = verify_local_bundle(
            bundle=arguments.bundle,
            git=arguments.git_executable,
            phase=arguments.phase,
            expected_p30_commit=arguments.expected_p30_commit,
            expected_p30_tree=arguments.expected_p30_tree,
            expected_p30_parent=arguments.expected_p30_parent,
            fixed_refs=fixed_refs,
        )
    except (OSError, VerificationError) as exc:
        print(f"P30 local bundle verification failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
