#!/usr/bin/env python3
"""Run one client command while retaining exact no-overwrite stream triplets."""

from __future__ import annotations

import argparse
import os
import re
import selectors
import stat
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")


class LogError(RuntimeError):
    """Raised when a client stream bundle cannot be retained safely."""


def _write_all(descriptor: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise LogError("client stream write made no progress")
        view = view[written:]


def _open_log_root(path: Path) -> int:
    if not path.is_absolute():
        raise LogError("client log root must be absolute")
    resolved = path.resolve(strict=True)
    if resolved == ROOT or ROOT in resolved.parents:
        raise LogError("client log root must be outside the repository")
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


def _canonical_status(returncode: int) -> int:
    if returncode < 0:
        return min(255, 128 - returncode)
    return min(255, returncode)


def run_logged(*, log_root: Path, label: str, command: list[str], tee: bool = True) -> int:
    if LABEL_RE.fullmatch(label) is None:
        raise LogError("client log label is malformed")
    if not command:
        raise LogError("client command is empty")

    directory = _open_log_root(log_root)
    stdout_fd = stderr_fd = status_fd = None
    try:
        stdout_fd = _open_exclusive(directory, f"{label}.stdout.log")
        stderr_fd = _open_exclusive(directory, f"{label}.stderr.log")
        try:
            process = subprocess.Popen(
                command,
                stdin=None,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                close_fds=True,
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
            assert process.stdout is not None and process.stderr is not None
            streams = {
                process.stdout.fileno(): (process.stdout, stdout_fd, sys.stdout.buffer),
                process.stderr.fileno(): (process.stderr, stderr_fd, sys.stderr.buffer),
            }
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
                        if tee:
                            mirror.write(chunk)
                            mirror.flush()
            finally:
                selector.close()
            status = _canonical_status(process.wait())

        os.fsync(stdout_fd)
        os.fsync(stderr_fd)
        os.close(stdout_fd)
        stdout_fd = None
        os.close(stderr_fd)
        stderr_fd = None
        status_fd = _open_exclusive(directory, f"{label}.exit-status.txt")
        _write_all(status_fd, f"{status}\n".encode("ascii"))
        os.fsync(status_fd)
        os.close(status_fd)
        status_fd = None
        os.fsync(directory)
        return status
    finally:
        for descriptor in (status_fd, stderr_fd, stdout_fd):
            if descriptor is not None:
                os.close(descriptor)
        os.close(directory)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", required=True, type=Path)
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
            label=arguments.label,
            command=command,
            tee=not arguments.no_tee,
        )
    except (LogError, OSError) as exc:
        print(f"client logger failed: {exc}", file=sys.stderr)
        return 125


if __name__ == "__main__":
    raise SystemExit(main())
