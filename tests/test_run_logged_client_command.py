from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOGGER = ROOT / "scripts/run_logged_client_command.py"


def _invoke(log_root: Path, label: str, command: list[str]) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(LOGGER),
            "--log-root",
            str(log_root),
            "--label",
            label,
            "--no-tee",
            "--",
            *command,
        ],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )


def test_retains_exact_streams_and_status_without_command_metadata(tmp_path: Path) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    secret_argument = "not-recorded-credential"
    result = _invoke(
        log_root,
        "source-verifier",
        [
            sys.executable,
            "-c",
            "import os,sys; os.write(1,b'out\\x00bytes\\n'); "
            "os.write(2,b'err\\xffbytes\\n'); raise SystemExit(7)",
            secret_argument,
        ],
    )

    assert result.returncode == 7
    assert result.stdout == b""
    assert result.stderr == b""
    assert (log_root / "source-verifier.stdout.log").read_bytes() == b"out\x00bytes\n"
    assert (log_root / "source-verifier.stderr.log").read_bytes() == b"err\xffbytes\n"
    assert (log_root / "source-verifier.exit-status.txt").read_bytes() == b"7\n"
    retained = b"".join(path.read_bytes() for path in log_root.iterdir())
    assert secret_argument.encode() not in retained


def test_replay_fails_closed_without_overwriting_triplet(tmp_path: Path) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    command = [sys.executable, "-c", "print('first')"]
    first = _invoke(log_root, "provision", command)
    before = {path.name: path.read_bytes() for path in log_root.iterdir()}
    second = _invoke(log_root, "provision", command)

    assert first.returncode == 0
    assert second.returncode == 125
    assert {path.name: path.read_bytes() for path in log_root.iterdir()} == before


def test_rejects_repository_log_root_and_malformed_label(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "raise SystemExit(0)"]
    inside_repository = _invoke(ROOT, "provision", command)
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    malformed = _invoke(log_root, "../escape", command)

    assert inside_repository.returncode == 125
    assert malformed.returncode == 125
    assert not any(log_root.iterdir())
