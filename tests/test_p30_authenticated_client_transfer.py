from __future__ import annotations

import hashlib
import importlib.util
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/run_p30_authenticated_client_transfer.py"
LOGGER = ROOT / "scripts/run_logged_client_command.py"
SPEC = importlib.util.spec_from_file_location("p30_authenticated_transfer", HELPER)
assert SPEC is not None and SPEC.loader is not None
transfer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = transfer
SPEC.loader.exec_module(transfer)


def _logged(
    log_root: Path,
    label: str,
    helper_arguments: list[str],
) -> subprocess.CompletedProcess[bytes]:
    tool_root = log_root.parent / f"{label}-tools"
    tool_root.mkdir(mode=0o700)
    frozen_helper = tool_root / HELPER.name
    shutil.copyfile(HELPER, frozen_helper)
    frozen_helper.chmod(0o400)
    helper_digest = hashlib.sha256(frozen_helper.read_bytes()).hexdigest()
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(LOGGER),
            "--log-root",
            str(log_root),
            "--forbidden-root",
            str(ROOT),
            "--expected-logger-source-sha256",
            hashlib.sha256(LOGGER.read_bytes()).hexdigest(),
            "--label",
            label,
            "--no-tee",
            "--",
            sys.executable,
            "-I",
            "-S",
            str(frozen_helper),
            "--expected-transfer-source-sha256",
            helper_digest,
            *helper_arguments,
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )


def _upload_arguments(
    *,
    source: Path,
    staging_root: Path,
    label: str,
    remote: Path,
    digest: str | None = None,
) -> list[str]:
    if digest is None:
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return [
        "--direction",
        "upload",
        "--source",
        str(source),
        "--staging-root",
        str(staging_root),
        "--label",
        label,
        "--remote-path",
        str(remote),
        "--remote-mode",
        "0600",
        "--expected-source-sha256",
        digest,
        "--expected-source-byte-count",
        str(source.stat().st_size),
        "--",
        "/bin/sh",
        "-c",
    ]


def test_upload_is_staged_before_remote_launch_and_whole_boundary_is_logged(
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    staging_root = tmp_path / "staging"
    log_root.mkdir(mode=0o700)
    staging_root.mkdir(mode=0o700)
    source = tmp_path / "source.bundle"
    source.write_bytes(b"authenticated bundle bytes\n")
    source.chmod(0o600)
    remote = tmp_path / "remote.bundle"

    result = _logged(
        log_root,
        "upload",
        _upload_arguments(
            source=source,
            staging_root=staging_root,
            label="upload",
            remote=remote,
        ),
    )

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert remote.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(remote.stat().st_mode) == 0o600
    staged = staging_root / "upload.upload-stage"
    assert staged.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(staged.stat().st_mode) == 0o400
    assert (log_root / "upload.stdout.log").read_bytes() == b""
    assert (log_root / "upload.exit-status.txt").read_bytes() == b"0\n"


def test_bad_upload_binding_is_logged_nonzero_before_remote_creation(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    staging_root = tmp_path / "staging"
    log_root.mkdir(mode=0o700)
    staging_root.mkdir(mode=0o700)
    source = tmp_path / "source.bundle"
    source.write_bytes(b"wrong bytes\n")
    source.chmod(0o600)
    remote = tmp_path / "must-not-exist.bundle"

    result = _logged(
        log_root,
        "bad-binding",
        _upload_arguments(
            source=source,
            staging_root=staging_root,
            label="bad-binding",
            remote=remote,
            digest="0" * 64,
        ),
    )

    assert result.returncode == transfer.TRANSFER_FAILURE_STATUS
    assert not remote.exists()
    assert not any(staging_root.iterdir())
    assert (log_root / "bad-binding.exit-status.txt").read_bytes() == b"125\n"
    assert b"upload source SHA-256 differs" in (log_root / "bad-binding.stderr.log").read_bytes()


def test_staging_collision_is_logged_nonzero_before_remote_creation(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    staging_root = tmp_path / "staging"
    log_root.mkdir(mode=0o700)
    staging_root.mkdir(mode=0o700)
    source = tmp_path / "source.bundle"
    source.write_bytes(b"bundle\n")
    source.chmod(0o600)
    collision = staging_root / "collision.upload-stage"
    collision.write_bytes(b"preexisting\n")
    collision.chmod(0o400)
    remote = tmp_path / "must-not-exist.bundle"

    result = _logged(
        log_root,
        "collision",
        _upload_arguments(
            source=source,
            staging_root=staging_root,
            label="collision",
            remote=remote,
        ),
    )

    assert result.returncode == transfer.TRANSFER_FAILURE_STATUS
    assert not remote.exists()
    assert collision.read_bytes() == b"preexisting\n"
    assert (log_root / "collision.exit-status.txt").read_bytes() == b"125\n"


def test_download_reservation_finalization_and_output_share_logged_status(
    tmp_path: Path,
) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir(mode=0o700)
    remote = tmp_path / "remote.json"
    remote.write_bytes(b'{"real":"receipt"}\n')
    remote.chmod(0o600)
    destination = tmp_path / "downloaded.json"
    arguments = [
        "--direction",
        "download",
        "--destination",
        str(destination),
        "--remote-path",
        str(remote),
        "--",
        "/bin/sh",
        "-c",
    ]

    result = _logged(log_root, "download", arguments)

    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert destination.read_bytes() == remote.read_bytes()
    assert stat.S_IMODE(destination.stat().st_mode) == 0o600
    assert (log_root / "download.stdout.log").read_bytes() == remote.read_bytes()
    assert (log_root / "download.exit-status.txt").read_bytes() == b"0\n"


def test_download_collision_is_one_logged_nonzero_boundary(tmp_path: Path) -> None:
    log_root = tmp_path / "logs"
    log_root.mkdir(mode=0o700)
    remote = tmp_path / "remote.json"
    remote.write_bytes(b"remote\n")
    remote.chmod(0o600)
    destination = tmp_path / "downloaded.json"
    destination.write_bytes(b"preexisting\n")
    destination.chmod(0o600)
    arguments = [
        "--direction",
        "download",
        "--destination",
        str(destination),
        "--remote-path",
        str(remote),
        "--",
        "/bin/sh",
        "-c",
    ]

    result = _logged(log_root, "download-collision", arguments)

    assert result.returncode == transfer.TRANSFER_FAILURE_STATUS
    assert destination.read_bytes() == b"preexisting\n"
    assert (log_root / "download-collision.exit-status.txt").read_bytes() == b"125\n"


def test_download_finalization_failure_cannot_return_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    remote = tmp_path / "remote.json"
    remote.write_bytes(b"remote\n")
    remote.chmod(0o600)
    destination = tmp_path / "downloaded.json"

    def fail_finalization(*_args: object, **_kwargs: object) -> None:
        raise transfer.TransferError("injected download finalization failure")

    monkeypatch.setattr(transfer, "_authenticate_executing_source", lambda _digest: None)
    monkeypatch.setattr(transfer, "_finalize_download", fail_finalization)
    status = transfer.main(
        [
            "--expected-transfer-source-sha256",
            hashlib.sha256(HELPER.read_bytes()).hexdigest(),
            "--direction",
            "download",
            "--destination",
            str(destination),
            "--remote-path",
            str(remote),
            "--",
            "/bin/sh",
            "-c",
        ]
    )

    assert status == transfer.TRANSFER_FAILURE_STATUS
    assert destination.exists()


def test_upload_staging_replacement_after_ssh_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    source = tmp_path / "source.bundle"
    source.write_bytes(b"bundle\n")
    source.chmod(0o600)
    remote = tmp_path / "remote.bundle"
    real_run = transfer.subprocess.run

    def replace_after_remote(arguments: list[str], **kwargs: object) -> object:
        result = real_run(arguments, **kwargs)
        staged = staging_root / "replace.upload-stage"
        moved = staging_root / "opened-stage"
        staged.rename(moved)
        staged.write_bytes(b"replacement\n")
        staged.chmod(0o400)
        return result

    monkeypatch.setattr(transfer.subprocess, "run", replace_after_remote)

    with pytest.raises(transfer.TransferError, match="changed during transfer"):
        transfer.upload(
            source=source,
            staging_root=staging_root,
            label="replace",
            remote_path=str(remote),
            remote_mode="0600",
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_byte_count=source.stat().st_size,
            ssh_arguments=["/bin/sh", "-c"],
        )

    assert remote.read_bytes() == source.read_bytes()


def test_upload_staging_directory_replacement_after_ssh_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    staging_root = tmp_path / "staging"
    staging_root.mkdir(mode=0o700)
    source = tmp_path / "source.bundle"
    source.write_bytes(b"bundle\n")
    source.chmod(0o600)
    remote = tmp_path / "remote.bundle"
    real_run = transfer.subprocess.run

    def replace_directory_after_remote(arguments: list[str], **kwargs: object) -> object:
        result = real_run(arguments, **kwargs)
        moved = tmp_path / "opened-staging"
        staging_root.rename(moved)
        staging_root.mkdir(mode=0o700)
        return result

    monkeypatch.setattr(transfer.subprocess, "run", replace_directory_after_remote)

    with pytest.raises(transfer.TransferError, match="staging directory changed"):
        transfer.upload(
            source=source,
            staging_root=staging_root,
            label="replace-directory",
            remote_path=str(remote),
            remote_mode="0600",
            expected_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
            expected_byte_count=source.stat().st_size,
            ssh_arguments=["/bin/sh", "-c"],
        )

    assert remote.read_bytes() == source.read_bytes()
