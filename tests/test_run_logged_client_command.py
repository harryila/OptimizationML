from __future__ import annotations

import errno
import hashlib
import importlib.util
import os
import shutil
import signal
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LOGGER = ROOT / "scripts/run_logged_client_command.py"


def _load_logger_module():
    spec = importlib.util.spec_from_file_location("p30_client_logger_under_test", LOGGER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _invoke(
    log_root: Path,
    label: str,
    command: list[str],
    *,
    logger: Path = LOGGER,
    forbidden_roots: tuple[Path, ...] = (ROOT,),
    expected_source_sha256: str | None = None,
) -> subprocess.CompletedProcess[bytes]:
    if expected_source_sha256 is None:
        expected_source_sha256 = hashlib.sha256(logger.read_bytes()).hexdigest()
    forbidden_arguments = [
        argument for root in forbidden_roots for argument in ("--forbidden-root", str(root))
    ]
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(logger),
            "--log-root",
            str(log_root),
            *forbidden_arguments,
            "--expected-logger-source-sha256",
            expected_source_sha256,
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


@pytest.mark.parametrize(
    "colliding_suffix",
    ("stdout.log", "stderr.log", "exit-status.txt"),
)
def test_each_artifact_collision_blocks_launch_before_any_side_effect(
    tmp_path: Path, colliding_suffix: str
) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    collision = log_root / f"collision.{colliding_suffix}"
    collision.write_bytes(b"preexisting\n")
    marker = tmp_path / "command-launched"

    result = _invoke(
        log_root,
        "collision",
        [
            sys.executable,
            "-c",
            "from pathlib import Path; import sys; Path(sys.argv[1]).write_text('launched')",
            str(marker),
        ],
    )

    assert result.returncode == 125
    assert not marker.exists()
    assert collision.read_bytes() == b"preexisting\n"
    assert sorted(path.name for path in log_root.iterdir()) == [collision.name]


def test_status_is_reserved_before_launch_and_finalized_canonically(tmp_path: Path) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    status_path = log_root / "reservation.exit-status.txt"

    result = _invoke(
        log_root,
        "reservation",
        [
            sys.executable,
            "-c",
            "import os,sys; print(int(os.path.isfile(sys.argv[1]))); raise SystemExit(23)",
            str(status_path),
        ],
    )

    assert result.returncode == 23
    assert (log_root / "reservation.stdout.log").read_bytes() == b"1\n"
    assert status_path.read_bytes() == b"23\n"


def test_launch_failure_still_finalizes_reserved_triplet(tmp_path: Path) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)

    result = _invoke(log_root, "launch-failure", [str(tmp_path / "does-not-exist")])

    assert result.returncode == 127
    assert (log_root / "launch-failure.stdout.log").read_bytes() == b""
    assert b"client command launch failed:" in (log_root / "launch-failure.stderr.log").read_bytes()
    assert (log_root / "launch-failure.exit-status.txt").read_bytes() == b"127\n"


def test_signal_status_is_canonicalized_and_finalized(tmp_path: Path) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)

    result = _invoke(
        log_root,
        "signal",
        [
            sys.executable,
            "-c",
            f"import os; os.kill(os.getpid(), {int(signal.SIGTERM)})",
        ],
    )

    expected = 128 + int(signal.SIGTERM)
    assert result.returncode == expected
    assert (log_root / "signal.exit-status.txt").read_bytes() == f"{expected}\n".encode()


def test_broken_stdout_mirror_keeps_capturing_and_reaps_child(tmp_path: Path) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    marker = tmp_path / "child-finished"
    child_pid_path = tmp_path / "child.pid"
    digest = hashlib.sha256(LOGGER.read_bytes()).hexdigest()
    payload_size = 4 * 1024 * 1024
    process = subprocess.Popen(
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
            digest,
            "--label",
            "broken-mirror",
            "--",
            sys.executable,
            "-c",
            (
                "import os,pathlib,sys; "
                "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                "os.write(1,b'x'*int(sys.argv[3])); "
                "pathlib.Path(sys.argv[2]).write_text('finished'); "
                "raise SystemExit(37)"
            ),
            str(child_pid_path),
            str(marker),
            str(payload_size),
        ],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert process.stdout is not None and process.stderr is not None
    assert process.stdout.read(1) == b"x"
    process.stdout.close()
    stderr = process.stderr.read()
    returncode = process.wait(timeout=10)

    assert returncode == 37, stderr.decode(errors="replace")
    assert marker.read_text() == "finished"
    assert (log_root / "broken-mirror.stdout.log").stat().st_size == payload_size
    assert (log_root / "broken-mirror.exit-status.txt").read_bytes() == b"37\n"
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid_path.read_text()), 0)


def test_capture_failure_kills_private_process_group_and_reaps_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_logger_module()
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    child_pid_path = tmp_path / "child.pid"
    digest = hashlib.sha256(LOGGER.read_bytes()).hexdigest()

    def fail_capture(_descriptor: int, _data: bytes) -> None:
        raise OSError(errno.ENOSPC, "injected capture failure")

    monkeypatch.setattr(module, "_write_all", fail_capture)
    with pytest.raises(OSError, match="injected capture failure"):
        module.run_logged(
            log_root=log_root,
            forbidden_roots=[ROOT],
            expected_logger_source_sha256=digest,
            label="capture-failure",
            command=[
                sys.executable,
                "-c",
                (
                    "import os,pathlib,signal,sys,time; "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "pathlib.Path(sys.argv[1]).write_text(str(os.getpid())); "
                    "os.write(1,b'trigger'); time.sleep(60)"
                ),
                str(child_pid_path),
            ],
            tee=False,
        )

    assert child_pid_path.exists()
    with pytest.raises(ProcessLookupError):
        os.kill(int(child_pid_path.read_text()), 0)
    assert (log_root / "capture-failure.exit-status.txt").read_bytes() == b""


def test_materialized_logger_authenticates_itself_and_disjoint_roots(tmp_path: Path) -> None:
    tool_root = tmp_path / "frozen-tools"
    source_root = tool_root / "source"
    source_root.mkdir(parents=True, mode=0o700)
    materialized = source_root / LOGGER.name
    shutil.copyfile(LOGGER, materialized)
    materialized.chmod(0o400)
    digest = hashlib.sha256(materialized.read_bytes()).hexdigest()
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)

    result = _invoke(
        log_root,
        "materialized",
        [sys.executable, "-c", "print('authenticated')"],
        logger=materialized,
        forbidden_roots=(ROOT, tool_root),
        expected_source_sha256=digest,
    )

    assert result.returncode == 0
    assert (log_root / "materialized.stdout.log").read_bytes() == b"authenticated\n"


@pytest.mark.parametrize("relationship", ("equal", "log-inside", "forbidden-inside"))
def test_forbidden_root_rejects_both_containment_directions(
    tmp_path: Path, relationship: str
) -> None:
    common = tmp_path / "common"
    common.mkdir(mode=0o700)
    if relationship == "equal":
        log_root = forbidden = common
    elif relationship == "log-inside":
        forbidden = common
        log_root = common / "logs"
        log_root.mkdir(mode=0o700)
    else:
        log_root = common
        forbidden = common / "frozen-tools"
        forbidden.mkdir(mode=0o700)

    marker = tmp_path / f"{relationship}-launched"
    result = _invoke(
        log_root,
        relationship,
        [sys.executable, "-c", "open(__import__('sys').argv[1], 'w').close()", str(marker)],
        forbidden_roots=(forbidden,),
    )

    assert result.returncode == 125
    assert not marker.exists()


def test_bad_materialized_source_digest_blocks_launch(tmp_path: Path) -> None:
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    marker = tmp_path / "launched"

    result = _invoke(
        log_root,
        "bad-source",
        [sys.executable, "-c", "open(__import__('sys').argv[1], 'w').close()", str(marker)],
        expected_source_sha256="0" * 64,
    )

    assert result.returncode == 125
    assert not marker.exists()
    assert not any(log_root.iterdir())


def test_close_helper_attempts_every_descriptor_after_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _load_logger_module()
    attempted: list[int] = []

    def fake_close(descriptor: int) -> None:
        attempted.append(descriptor)
        if descriptor in {11, 13}:
            raise OSError(f"close failed for {descriptor}")

    monkeypatch.setattr(module.os, "close", fake_close)
    error = module._close_best_effort([11, None, 12, 13])

    assert attempted == [11, 12, 13]
    assert isinstance(error, OSError)
    assert str(error) == "close failed for 11"


def test_rejects_repository_log_root_and_malformed_label(tmp_path: Path) -> None:
    command = [sys.executable, "-c", "raise SystemExit(0)"]
    inside_repository = _invoke(ROOT, "provision", command)
    log_root = tmp_path / "client-logs"
    log_root.mkdir(mode=0o700)
    malformed = _invoke(log_root, "../escape", command)

    assert inside_repository.returncode == 125
    assert malformed.returncode == 125
    assert not any(log_root.iterdir())
