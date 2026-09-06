#!/usr/bin/env python3
"""Run one client command while retaining exact no-overwrite stream triplets."""

from __future__ import annotations

import argparse
import contextlib
import errno
import hashlib
import os
import re
import selectors
import signal
import stat
import subprocess
import sys
from pathlib import Path
from typing import BinaryIO

LABEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
CHILD_TERMINATION_GRACE_SECONDS = 1.0


class LogError(RuntimeError):
    """Raised when a client stream bundle cannot be retained safely."""


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise LogError("client stream write made no progress")
        view = view[written:]


def _mirror_chunk(mirror: BinaryIO, data: bytes) -> bool:
    try:
        mirror.write(data)
        mirror.flush()
    except OSError as exc:
        if exc.errno != errno.EPIPE:
            raise
        # Prevent Python's final stdio flush from turning a completely retained
        # child result into exit 120 after the downstream consumer disappears.
        null_fd: int | None = None
        try:
            null_fd = os.open(os.devnull, os.O_WRONLY | os.O_CLOEXEC)
            os.dup2(null_fd, mirror.fileno())
        except OSError:
            pass
        finally:
            if null_fd is not None:
                with contextlib.suppress(OSError):
                    os.close(null_fd)
        return False
    return True


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


def _authenticate_logger_source(path: Path, expected_sha256: str) -> None:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise LogError("expected logger-source digest is not one lowercase SHA-256")
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise LogError("client logger source path must be absolute")
    for name in ("O_CLOEXEC", "O_DIRECTORY", "O_NOFOLLOW"):
        if not getattr(os, name, 0):
            raise LogError(f"platform lacks mandatory {name}")

    parent_fd = source_fd = None
    try:
        parent_initial = path.parent.lstat()
        source_initial = path.lstat()
        if path.parent.resolve(strict=True) != path.parent:
            raise LogError("client logger source parent is not canonical")
        if (
            not stat.S_ISDIR(parent_initial.st_mode)
            or parent_initial.st_uid != os.getuid()
            or stat.S_IMODE(parent_initial.st_mode) & 0o022
        ):
            raise LogError(
                "client logger source parent must be caller-owned and not group/world writable"
            )
        if (
            not stat.S_ISREG(source_initial.st_mode)
            or source_initial.st_uid != os.getuid()
            or source_initial.st_gid != os.getgid()
            or source_initial.st_nlink != 1
            or stat.S_IMODE(source_initial.st_mode) & 0o022
        ):
            raise LogError(
                "client logger source must be caller-owned, single-link, regular, and not "
                "group/world writable"
            )

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
        digest = hashlib.sha256()
        while True:
            chunk = os.read(source_fd, 64 * 1024)
            if not chunk:
                break
            digest.update(chunk)
        source_after = os.fstat(source_fd)
        source_by_name = os.stat(path.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_after = os.fstat(parent_fd)
        parent_by_name = path.parent.lstat()
    except OSError as exc:
        raise LogError(f"cannot stably authenticate client logger source: {exc}") from exc
    finally:
        for descriptor in (source_fd, parent_fd):
            if descriptor is not None:
                with contextlib.suppress(OSError):
                    os.close(descriptor)

    if not (
        _stable_fields(parent_initial)
        == _stable_fields(parent_before)
        == _stable_fields(parent_after)
        == _stable_fields(parent_by_name)
    ):
        raise LogError("client logger source parent changed during authentication")
    if not (
        _stable_fields(source_initial)
        == _stable_fields(source_before)
        == _stable_fields(source_after)
        == _stable_fields(source_by_name)
    ):
        raise LogError("client logger source changed during authentication")
    if digest.hexdigest() != expected_sha256:
        raise LogError("client logger source SHA-256 differs")


def _canonical_directory(path: Path, label: str) -> Path:
    if not path.is_absolute():
        raise LogError(f"{label} must be absolute")
    resolved = path.resolve(strict=True)
    if resolved != path:
        raise LogError(f"{label} must be canonical")
    info = resolved.stat()
    if not stat.S_ISDIR(info.st_mode):
        raise LogError(f"{label} must be a directory")
    return resolved


def _open_log_root(path: Path, forbidden_roots: list[Path]) -> int:
    resolved = _canonical_directory(path, "client log root")
    if not forbidden_roots:
        raise LogError("at least one forbidden root is required")
    for forbidden_path in forbidden_roots:
        forbidden = _canonical_directory(forbidden_path, "forbidden root")
        if resolved == forbidden or forbidden in resolved.parents or resolved in forbidden.parents:
            raise LogError("client log root and every forbidden root must be disjoint")
    flags = os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(resolved, flags)
    info = os.fstat(descriptor)
    if (
        not stat.S_ISDIR(info.st_mode)
        or info.st_uid != os.getuid()
        or stat.S_IMODE(info.st_mode) & 0o077
    ):
        os.close(descriptor)
        raise LogError("client log root must be owned by the caller and mode 0700 or stricter")
    return descriptor


def _open_exclusive(directory: int, name: str) -> int:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    return os.open(name, flags, 0o600, dir_fd=directory)


def _close_best_effort(descriptors: list[int | None]) -> OSError | None:
    first_error: OSError | None = None
    for descriptor in descriptors:
        if descriptor is None:
            continue
        try:
            os.close(descriptor)
        except OSError as exc:
            if first_error is None:
                first_error = exc
    return first_error


def _reserve_triplet(directory: int, label: str) -> tuple[int, int, int]:
    names = (
        f"{label}.stdout.log",
        f"{label}.stderr.log",
        f"{label}.exit-status.txt",
    )
    # This precheck avoids leaving partial reservations for ordinary collisions.
    # O_EXCL remains the authority if a name appears between this check and open.
    for name in names:
        try:
            os.stat(name, dir_fd=directory, follow_symlinks=False)
        except FileNotFoundError:
            continue
        raise LogError(f"client log artifact already exists: {name}")

    opened: list[tuple[str, int, os.stat_result]] = []
    try:
        for name in names:
            descriptor = _open_exclusive(directory, name)
            opened.append((name, descriptor, os.fstat(descriptor)))
        os.fsync(directory)
    except BaseException:
        _close_best_effort([descriptor for _name, descriptor, _info in opened])
        # Remove only names that still identify the inode this process created.
        for name, _descriptor, created in reversed(opened):
            with contextlib.suppress(OSError):
                current = os.stat(name, dir_fd=directory, follow_symlinks=False)
                if (current.st_dev, current.st_ino) == (created.st_dev, created.st_ino):
                    os.unlink(name, dir_fd=directory)
        raise
    return opened[0][1], opened[1][1], opened[2][1]


def _canonical_status(returncode: int) -> int:
    if returncode < 0:
        return min(255, 128 - returncode)
    return min(255, returncode)


def _terminate_process_group_and_reap(process: subprocess.Popen[bytes]) -> None:
    """Stop a partially observed command and every child in its private session."""

    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
    with contextlib.suppress(subprocess.TimeoutExpired):
        process.wait(timeout=CHILD_TERMINATION_GRACE_SECONDS)

    # The group can outlive its leader.  Kill the original private process group
    # even when wait() already observed the direct child exit after SIGTERM.
    with contextlib.suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGKILL)
    if process.poll() is None:
        process.wait()


def run_logged(
    *,
    log_root: Path,
    forbidden_roots: list[Path],
    expected_logger_source_sha256: str,
    label: str,
    command: list[str],
    tee: bool = True,
) -> int:
    if LABEL_RE.fullmatch(label) is None:
        raise LogError("client log label is malformed")
    if not command:
        raise LogError("client command is empty")

    _authenticate_logger_source(Path(__file__), expected_logger_source_sha256)
    directory = _open_log_root(log_root, forbidden_roots)
    stdout_fd = stderr_fd = status_fd = None
    body_succeeded = False
    try:
        stdout_fd, stderr_fd, status_fd = _reserve_triplet(directory, label)
        try:
            process = subprocess.Popen(
                command,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
                start_new_session=True,
            )
        except OSError as exc:
            message = f"client command launch failed: {exc}\n".encode(
                "utf-8", errors="backslashreplace"
            )
            _write_all(stderr_fd, message)
            if tee:
                sys.stderr.buffer.write(message)
                sys.stderr.buffer.flush()
            status = 127
        else:
            try:
                assert process.stdout is not None and process.stderr is not None
                streams = {
                    process.stdout.fileno(): (process.stdout, stdout_fd, sys.stdout.buffer),
                    process.stderr.fileno(): (process.stderr, stderr_fd, sys.stderr.buffer),
                }
                live_mirrors = set(streams)
                selector = selectors.DefaultSelector()
                try:
                    for stream, _destination, _mirror in streams.values():
                        selector.register(stream, selectors.EVENT_READ)
                    while selector.get_map():
                        for key, _events in selector.select():
                            stream, destination, mirror = streams[key.fd]
                            chunk = os.read(stream.fileno(), 64 * 1024)
                            if not chunk:
                                selector.unregister(stream)
                                stream.close()
                                continue
                            _write_all(destination, chunk)
                            if tee and key.fd in live_mirrors and not _mirror_chunk(mirror, chunk):
                                live_mirrors.remove(key.fd)
                finally:
                    selector.close()
                status = _canonical_status(process.wait())
            except BaseException:
                _terminate_process_group_and_reap(process)
                raise

        os.fsync(stdout_fd)
        os.fsync(stderr_fd)
        _write_all(status_fd, f"{status}\n".encode("ascii"))
        os.fsync(status_fd)
        os.fsync(directory)
        body_succeeded = True
        return status
    finally:
        close_error = _close_best_effort([status_fd, stderr_fd, stdout_fd, directory])
        if body_succeeded and close_error is not None:
            raise LogError(
                f"cannot close retained client-log artifact: {close_error}"
            ) from close_error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", required=True, type=Path)
    parser.add_argument("--forbidden-root", required=True, action="append", type=Path)
    parser.add_argument("--expected-logger-source-sha256", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--no-tee", action="store_true")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = arguments.command
    if command[:1] == ["--"]:
        command = command[1:]
    try:
        return run_logged(
            log_root=arguments.log_root,
            forbidden_roots=arguments.forbidden_root,
            expected_logger_source_sha256=arguments.expected_logger_source_sha256,
            label=arguments.label,
            command=command,
            tee=not arguments.no_tee,
        )
    except (LogError, OSError) as exc:
        print(f"client logger failed: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
