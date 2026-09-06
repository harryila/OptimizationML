#!/usr/bin/env python3
"""Run one authenticated P30 upload or download as one logged client command.

Uploads are fully authenticated into a private, no-overwrite staging file before
SSH is launched.  Downloads reserve and finalize their no-overwrite destination
inside the same process that launches SSH.  The enclosing client logger therefore
records the status of the complete transfer boundary rather than only SSH's status.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shlex
import stat
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

LABEL_RE = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TRANSFER_FAILURE_STATUS = 125


class TransferError(RuntimeError):
    """The complete authenticated transfer boundary failed."""


@dataclass(frozen=True)
class StableBytes:
    raw: bytes
    sha256: str
    byte_count: int
    identity: tuple[int, ...]


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


def _directory_fields(value: os.stat_result) -> tuple[int, ...]:
    return (value.st_dev, value.st_ino, value.st_mode, value.st_uid, value.st_gid)


def _canonical_owned_directory(
    path: Path, *, required_mode: int | None = None
) -> tuple[int, tuple[int, ...]]:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise TransferError("transfer directory must be absolute")
    try:
        if path.resolve(strict=True) != path:
            raise TransferError("transfer directory must be canonical")
        initial = path.lstat()
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_DIRECTORY | os.O_NOFOLLOW,
        )
    except OSError as exc:
        raise TransferError(f"cannot open transfer directory: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        by_name = path.lstat()
        mode = stat.S_IMODE(opened.st_mode)
        if not (
            _directory_fields(initial) == _directory_fields(opened) == _directory_fields(by_name)
        ):
            raise TransferError("transfer directory identity changed")
        if (
            not stat.S_ISDIR(opened.st_mode)
            or (opened.st_uid, opened.st_gid) != (os.getuid(), os.getgid())
            or mode & 0o022
            or (required_mode is not None and mode != required_mode)
        ):
            raise TransferError("transfer directory authority differs")
        return descriptor, _directory_fields(opened)
    except BaseException:
        os.close(descriptor)
        raise


def _read_regular_file_stably(path: Path) -> StableBytes:
    if not path.is_absolute() or path.name in {"", ".", ".."}:
        raise TransferError("upload source must be absolute")
    try:
        if path.resolve(strict=True) != path:
            raise TransferError("upload source must be canonical")
        before = path.lstat()
        descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
    except OSError as exc:
        raise TransferError(f"cannot open upload source: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        pieces: list[bytes] = []
        digest = hashlib.sha256()
        byte_count = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            pieces.append(chunk)
            digest.update(chunk)
            byte_count += len(chunk)
        after = os.fstat(descriptor)
        by_name = path.lstat()
    finally:
        os.close(descriptor)
    if not (
        _stable_fields(before)
        == _stable_fields(opened)
        == _stable_fields(after)
        == _stable_fields(by_name)
    ):
        raise TransferError("upload source changed during authentication")
    if (
        not stat.S_ISREG(after.st_mode)
        or (after.st_uid, after.st_gid) != (os.getuid(), os.getgid())
        or after.st_nlink != 1
        or stat.S_IMODE(after.st_mode) & 0o022
        or byte_count != after.st_size
    ):
        raise TransferError("upload source authority differs")
    return StableBytes(
        raw=b"".join(pieces),
        sha256=digest.hexdigest(),
        byte_count=byte_count,
        identity=_stable_fields(after),
    )


def _authenticate_executing_source(expected_sha256: str) -> None:
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise TransferError("expected transfer-helper source digest is malformed")
    source = _read_regular_file_stably(Path(os.path.abspath(__file__)))
    mode = stat.S_IMODE(source.identity[2])
    if mode != 0o400:
        raise TransferError("transfer-helper source mode is not exactly 0400")
    if source.sha256 != expected_sha256:
        raise TransferError("transfer-helper source SHA-256 differs")


def _write_all(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    while view:
        written = os.write(descriptor, view)
        if written <= 0:
            raise TransferError("authenticated transfer write made no progress")
        view = view[written:]


def _create_staging_file(
    staging_root_fd: int,
    name: str,
    authenticated: StableBytes,
) -> tuple[int, tuple[int, ...]]:
    flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
    try:
        descriptor = os.open(name, flags, 0o400, dir_fd=staging_root_fd)
    except OSError as exc:
        raise TransferError(f"cannot reserve authenticated upload staging file: {exc}") from exc
    try:
        _write_all(descriptor, authenticated.raw)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        by_name = os.stat(name, dir_fd=staging_root_fd, follow_symlinks=False)
        identity = _stable_fields(opened)
        stored = os.pread(descriptor, opened.st_size + 1, 0)
        if (
            identity != _stable_fields(by_name)
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_uid, opened.st_gid) != (os.getuid(), os.getgid())
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o400
            or stored != authenticated.raw
            or hashlib.sha256(stored).hexdigest() != authenticated.sha256
        ):
            raise TransferError("authenticated upload staging authority differs")
        os.fsync(staging_root_fd)
        return descriptor, identity
    except BaseException:
        os.close(descriptor)
        raise


def _revalidate_named_descriptor(
    descriptor: int,
    *,
    parent_fd: int,
    name: str,
    expected_identity: tuple[int, ...],
    label: str,
) -> None:
    opened = os.fstat(descriptor)
    by_name = os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
    if not (_stable_fields(opened) == expected_identity == _stable_fields(by_name)):
        raise TransferError(f"{label} changed during transfer")


def _revalidate_directory(
    descriptor: int,
    *,
    path: Path,
    expected_identity: tuple[int, ...],
    label: str,
) -> None:
    opened = os.fstat(descriptor)
    by_name = path.lstat()
    if not (_directory_fields(opened) == expected_identity == _directory_fields(by_name)):
        raise TransferError(f"{label} changed during transfer")


def _canonical_status(returncode: int) -> int:
    if returncode < 0:
        return min(255, 128 - returncode)
    return min(255, returncode)


def _ssh_arguments(arguments: Sequence[str]) -> list[str]:
    values = list(arguments)
    if values and values[0] == "--":
        values.pop(0)
    if not values or any(not value or "\x00" in value for value in values):
        raise TransferError("SSH argument vector is absent or malformed")
    return values


def _remote_upload_command(
    *, remote_path: str, remote_mode: str, sha256: str, byte_count: int
) -> str:
    program = (
        "import hashlib,os,stat,sys;"
        "p,m,e,c=sys.argv[1:];"
        "f=os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_CLOEXEC|os.O_NOFOLLOW;"
        "d=os.open(p,f,int(m,8));h=os.fdopen(d,'wb');q=hashlib.sha256();n=0;"
        "\nwhile True:\n"
        " b=sys.stdin.buffer.read(1048576)\n"
        " if not b: break\n"
        " h.write(b);q.update(b);n+=len(b)\n"
        "\nh.flush();os.fsync(h.fileno());s=os.fstat(h.fileno());"
        "o=os.stat(p,follow_symlinks=False);"
        "ok=(q.hexdigest()==e and n==int(c) and stat.S_ISREG(s.st_mode) "
        "and s.st_nlink==1 and stat.S_IMODE(s.st_mode)==int(m,8) "
        "and (s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,s.st_size,"
        "s.st_mtime_ns,s.st_ctime_ns)=="
        "(o.st_dev,o.st_ino,o.st_mode,o.st_uid,o.st_gid,o.st_nlink,o.st_size,"
        "o.st_mtime_ns,o.st_ctime_ns));"
        "h.close();"
        "\nif not ok: raise SystemExit('remote upload authority or content differs')"
    )
    return shlex.join(
        [
            "/usr/bin/python3",
            "-I",
            "-S",
            "-c",
            program,
            remote_path,
            remote_mode,
            sha256,
            str(byte_count),
        ]
    )


def _remote_download_command(remote_path: str) -> str:
    program = (
        "import os,stat,sys;"
        "p=sys.argv[1];b=os.lstat(p);d=os.open(p,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW);"
        "o=os.fstat(d);x=bytearray();"
        "\nwhile True:\n"
        " q=os.read(d,1048576)\n"
        " if not q: break\n"
        " x.extend(q)\n"
        "\na=os.fstat(d);n=os.stat(p,follow_symlinks=False);os.close(d);"
        "f=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_uid,s.st_gid,s.st_nlink,"
        "s.st_size,s.st_mtime_ns,s.st_ctime_ns);"
        "\nif not (f(b)==f(o)==f(a)==f(n) and stat.S_ISREG(a.st_mode) "
        "and a.st_nlink==1 and len(x)==a.st_size):"
        " raise SystemExit('remote download source authority differs')\n"
        "sys.stdout.buffer.write(x);sys.stdout.buffer.flush()"
    )
    return shlex.join(["/usr/bin/python3", "-I", "-S", "-c", program, remote_path])


def upload(
    *,
    source: Path,
    staging_root: Path,
    label: str,
    remote_path: str,
    remote_mode: str,
    expected_sha256: str,
    expected_byte_count: int | None,
    ssh_arguments: Sequence[str],
) -> int:
    if LABEL_RE.fullmatch(label) is None:
        raise TransferError("upload label is malformed")
    if SHA256_RE.fullmatch(expected_sha256) is None:
        raise TransferError("expected upload digest is malformed")
    if expected_byte_count is not None and expected_byte_count <= 0:
        raise TransferError("expected upload byte count is malformed")
    if remote_mode not in {"0400", "0600"} or not remote_path.startswith("/"):
        raise TransferError("remote upload path or mode is malformed")
    ssh = _ssh_arguments(ssh_arguments)
    authenticated = _read_regular_file_stably(source)
    if authenticated.sha256 != expected_sha256:
        raise TransferError("upload source SHA-256 differs")
    if expected_byte_count is not None and authenticated.byte_count != expected_byte_count:
        raise TransferError("upload source byte count differs")

    staging_root_fd, staging_root_identity = _canonical_owned_directory(
        staging_root, required_mode=0o700
    )
    staging_fd: int | None = None
    try:
        staging_name = f"{label}.upload-stage"
        staging_fd, staging_identity = _create_staging_file(
            staging_root_fd, staging_name, authenticated
        )
        os.lseek(staging_fd, 0, os.SEEK_SET)
        command = [
            *ssh,
            _remote_upload_command(
                remote_path=remote_path,
                remote_mode=remote_mode,
                sha256=authenticated.sha256,
                byte_count=authenticated.byte_count,
            ),
        ]
        try:
            result = subprocess.run(
                command,
                stdin=staging_fd,
                check=False,
                close_fds=True,
            )
        except OSError as exc:
            raise TransferError(f"cannot launch authenticated upload: {exc}") from exc
        _revalidate_named_descriptor(
            staging_fd,
            parent_fd=staging_root_fd,
            name=staging_name,
            expected_identity=staging_identity,
            label="authenticated upload staging file",
        )
        _revalidate_directory(
            staging_root_fd,
            path=staging_root,
            expected_identity=staging_root_identity,
            label="authenticated upload staging directory",
        )
        return _canonical_status(result.returncode)
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        os.close(staging_root_fd)


def _reserve_download(destination: Path) -> tuple[int, int, tuple[int, ...]]:
    if not destination.is_absolute() or destination.name in {"", ".", ".."}:
        raise TransferError("download destination must be absolute")
    parent_fd, parent_identity = _canonical_owned_directory(destination.parent)
    descriptor: int | None = None
    try:
        flags = os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW
        try:
            descriptor = os.open(destination.name, flags, 0o600, dir_fd=parent_fd)
        except OSError as exc:
            raise TransferError(f"cannot reserve download destination: {exc}") from exc
        os.fchmod(descriptor, 0o600)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        by_name = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        if (
            _stable_fields(opened) != _stable_fields(by_name)
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_uid, opened.st_gid) != (os.getuid(), os.getgid())
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or opened.st_size != 0
        ):
            raise TransferError("reserved download destination authority differs")
        os.fsync(parent_fd)
        return parent_fd, descriptor, parent_identity
    except BaseException:
        if descriptor is not None:
            os.close(descriptor)
        os.close(parent_fd)
        raise


def _finalize_download(
    destination: Path,
    *,
    parent_fd: int,
    descriptor: int,
    expected_parent_identity: tuple[int, ...],
    raw: bytes,
) -> None:
    try:
        _write_all(descriptor, raw)
        os.fsync(descriptor)
        opened = os.fstat(descriptor)
        by_name = os.stat(destination.name, dir_fd=parent_fd, follow_symlinks=False)
        parent_opened = os.fstat(parent_fd)
        parent_by_name = destination.parent.lstat()
        if (
            _stable_fields(opened) != _stable_fields(by_name)
            or _directory_fields(parent_opened) != expected_parent_identity
            or _directory_fields(parent_by_name) != expected_parent_identity
            or not stat.S_ISREG(opened.st_mode)
            or (opened.st_uid, opened.st_gid) != (os.getuid(), os.getgid())
            or opened.st_nlink != 1
            or stat.S_IMODE(opened.st_mode) != 0o600
            or os.pread(descriptor, opened.st_size + 1, 0) != raw
        ):
            raise TransferError("download destination authority or content differs")
        os.fsync(parent_fd)
    except OSError as exc:
        raise TransferError(f"cannot finalize download destination: {exc}") from exc


def download(
    *,
    destination: Path,
    remote_path: str,
    ssh_arguments: Sequence[str],
) -> int:
    if not remote_path.startswith("/"):
        raise TransferError("remote download path is malformed")
    ssh = _ssh_arguments(ssh_arguments)
    parent_fd, destination_fd, parent_identity = _reserve_download(destination)
    try:
        try:
            result = subprocess.run(
                [*ssh, _remote_download_command(remote_path)],
                check=False,
                capture_output=True,
            )
        except OSError as exc:
            raise TransferError(f"cannot launch authenticated download: {exc}") from exc
        if result.stderr:
            sys.stderr.buffer.write(result.stderr)
            sys.stderr.buffer.flush()
        status = _canonical_status(result.returncode)
        if status != 0:
            if result.stdout:
                sys.stdout.buffer.write(result.stdout)
                sys.stdout.buffer.flush()
            return status
        _finalize_download(
            destination,
            parent_fd=parent_fd,
            descriptor=destination_fd,
            expected_parent_identity=parent_identity,
            raw=result.stdout,
        )
        sys.stdout.buffer.write(result.stdout)
        sys.stdout.buffer.flush()
        return 0
    finally:
        os.close(destination_fd)
        os.close(parent_fd)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-transfer-source-sha256", required=True)
    parser.add_argument("--direction", required=True, choices=("upload", "download"))
    parser.add_argument("--remote-path", required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--staging-root", type=Path)
    parser.add_argument("--label")
    parser.add_argument("--remote-mode")
    parser.add_argument("--expected-source-sha256")
    parser.add_argument("--expected-source-byte-count")
    parser.add_argument("ssh_arguments", nargs=argparse.REMAINDER)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        _authenticate_executing_source(arguments.expected_transfer_source_sha256)
        if arguments.direction == "upload":
            required = (
                arguments.source,
                arguments.staging_root,
                arguments.label,
                arguments.remote_mode,
                arguments.expected_source_sha256,
                arguments.expected_source_byte_count,
            )
            if any(value is None for value in required) or arguments.destination is not None:
                raise TransferError("upload arguments are incomplete or contain download state")
            count_text = arguments.expected_source_byte_count
            assert count_text is not None
            if count_text == "-":
                expected_count = None
            elif count_text.isdecimal() and int(count_text) > 0:
                expected_count = int(count_text)
            else:
                raise TransferError("expected upload byte count is malformed")
            return upload(
                source=arguments.source,
                staging_root=arguments.staging_root,
                label=arguments.label,
                remote_path=arguments.remote_path,
                remote_mode=arguments.remote_mode,
                expected_sha256=arguments.expected_source_sha256,
                expected_byte_count=expected_count,
                ssh_arguments=arguments.ssh_arguments,
            )
        forbidden = (
            arguments.source,
            arguments.staging_root,
            arguments.label,
            arguments.remote_mode,
            arguments.expected_source_sha256,
            arguments.expected_source_byte_count,
        )
        if arguments.destination is None or any(value is not None for value in forbidden):
            raise TransferError("download arguments are incomplete or contain upload state")
        return download(
            destination=arguments.destination,
            remote_path=arguments.remote_path,
            ssh_arguments=arguments.ssh_arguments,
        )
    except (OSError, TransferError) as exc:
        print(f"P30 authenticated client transfer failed: {exc}", file=sys.stderr)
        return TRANSFER_FAILURE_STATUS


if __name__ == "__main__":
    raise SystemExit(main())
