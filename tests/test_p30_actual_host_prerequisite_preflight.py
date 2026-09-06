from __future__ import annotations

import ast
import base64
import hashlib
import importlib.util
import json
import pathlib
import stat
import subprocess
from types import ModuleType, SimpleNamespace

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/run_p30_actual_host_prerequisite_preflight.py"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location("p30_actual_host_preflight", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def module() -> ModuleType:
    return _load()


def _write_known_hosts(module: ModuleType, path: pathlib.Path) -> str:
    key_blob = b"unit-test-p30-ed25519-host-key"
    encoded = base64.b64encode(key_blob).decode("ascii")
    if path.exists():
        path.chmod(0o600)
    path.write_text(f"{module.SSH_HOST} ssh-ed25519 {encoded}\n", encoding="utf-8")
    path.chmod(0o400)
    return module._fingerprint(key_blob)


def _write_identity(path: pathlib.Path) -> None:
    path.write_bytes(b"unit-test-reviewed-private-identity\n")
    path.chmod(0o600)


def _write_frozen_client_source(path: pathlib.Path) -> str:
    raw = SCRIPT.read_bytes()
    path.write_bytes(raw)
    path.chmod(0o400)
    return hashlib.sha256(raw).hexdigest()


def _git(command: list[str], *, cwd: pathlib.Path) -> str:
    return subprocess.run(
        ["/usr/bin/git", *command],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def test_constants_lock_the_actual_p30_host_and_inputs(module: ModuleType) -> None:
    assert module.SSH_TARGET == "ubuntu@129.146.177.214"
    assert module.EXPECTED_HOST_KEY_FINGERPRINT == (
        "SHA256:t7nLeL/gt2qBu7e+I0yb8xICPtX8WcP8RMn7yEkXrJU"
    )
    assert str(module.P30_NAMESPACE) == "/secure/p30"
    assert str(module.P30_LEDGER) == "/var/lib/optimizationml-p30-20260906-03"
    assert module.P30_CONTAINER == "p30-localization-20260906-03"
    assert module.GPU_UUID == "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d"
    assert module.GPU_NAME == "NVIDIA A100-SXM4-40GB"
    assert module.IMAGE_REFERENCE == (
        "localhost:5000/p25-runtime@"
        "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0"
    )
    assert module.NANOGPT_COMMIT == "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
    assert module.NANOGPT_TREE == "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
    assert module.MUON_COMMIT == "f98f1cacc0263b04290753e32be8d498c1efc806"
    assert module.MUON_TREE == "4ea5cd8ab6ebd56a18536f06453619efcd636da0"
    assert module.DATA_MANIFEST_SHA256 == (
        "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d"
    )
    assert module.DATA_MANIFEST_BYTES == 4210
    assert module.P26_EVIDENCE_SHA256 == (
        "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91"
    )
    assert module.P26_EVIDENCE_BYTES == 66283


def test_known_hosts_requires_the_exact_single_ed25519_key(
    module: ModuleType, tmp_path: pathlib.Path
) -> None:
    known_hosts = tmp_path / "known_hosts"
    fingerprint = _write_known_hosts(module, known_hosts)
    module._validate_known_hosts(known_hosts, fingerprint)

    with pytest.raises(module.PreflightError, match="fingerprint differs"):
        module._validate_known_hosts(known_hosts, "SHA256:" + "A" * 43)

    doubled = known_hosts.read_text() + known_hosts.read_text()
    known_hosts.chmod(0o600)
    known_hosts.write_text(doubled, encoding="utf-8")
    known_hosts.chmod(0o400)
    with pytest.raises(module.PreflightError, match="exactly one"):
        module._validate_known_hosts(known_hosts, fingerprint)

    _write_known_hosts(module, known_hosts)
    with_comment = known_hosts.read_text().rstrip("\n") + " trailing-comment\n"
    known_hosts.chmod(0o600)
    known_hosts.write_text(with_comment)
    known_hosts.chmod(0o400)
    with pytest.raises(module.PreflightError, match="exact locked host and port"):
        module._validate_known_hosts(known_hosts, fingerprint)

    _write_known_hosts(module, known_hosts)
    known_hosts.chmod(0o600)
    with pytest.raises(module.PreflightError, match="exact mode 0400"):
        module._validate_known_hosts(known_hosts, fingerprint)


def test_client_uses_strict_pinned_noninteractive_ssh_and_streams_its_source(
    module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known_hosts = tmp_path / "known_hosts"
    fingerprint = _write_known_hosts(module, known_hosts)
    monkeypatch.setattr(module, "EXPECTED_HOST_KEY_FINGERPRINT", fingerprint)
    identity_file = tmp_path / "p30_identity"
    _write_identity(identity_file)
    frozen_source = tmp_path / "run_p30_actual_host_prerequisite_preflight.py"
    source_sha256 = _write_frozen_client_source(frozen_source)
    monkeypatch.setattr(module, "__file__", str(frozen_source))
    observed: dict[str, object] = {}

    def fake_run(command: list[str], **kwargs: object) -> SimpleNamespace:
        observed["command"] = command
        observed.update(kwargs)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(module.subprocess, "run", fake_run)
    assert (
        module.main(
            [
                "--known-hosts",
                str(known_hosts),
                "--identity-file",
                str(identity_file),
                "--expected-client-source-sha256",
                source_sha256,
            ]
        )
        == 0
    )

    command = observed["command"]
    assert isinstance(command, list)
    assert command[0] == "/usr/bin/ssh"
    assert command[1:3] == ["-F", "/dev/null"]
    assert module.SSH_TARGET in command
    for locked_option in (
        "BatchMode=yes",
        "StrictHostKeyChecking=yes",
        f"UserKnownHostsFile={known_hosts}",
        "GlobalKnownHostsFile=/dev/null",
        "HostKeyAlgorithms=ssh-ed25519",
        "UpdateHostKeys=no",
        "ClearAllForwardings=yes",
        "ProxyCommand=none",
        "ProxyJump=none",
        "PasswordAuthentication=no",
        "KbdInteractiveAuthentication=no",
        "PreferredAuthentications=publickey",
        "IdentitiesOnly=yes",
    ):
        assert locked_option in command
    ssh_options = command[: command.index(module.SSH_TARGET)]
    assert command.count("-F") == 1
    assert ssh_options.count("-i") == 1
    assert ssh_options[ssh_options.index("-i") + 1] == str(identity_file)
    assert command[-6:] == [
        "-I",
        "-S",
        "-",
        "--remote",
        "--expected-client-source-sha256",
        source_sha256,
    ]
    assert command[command.index(module.SSH_TARGET) + 1 : command.index(module.SSH_TARGET) + 5] == [
        "/usr/bin/sudo",
        "-n",
        "/usr/bin/env",
        "-i",
    ]
    assert observed["input"] == frozen_source.read_bytes()
    assert observed["check"] is False


def test_client_rejects_wrong_source_digest_before_opening_ssh(
    module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    known_hosts = tmp_path / "known_hosts"
    fingerprint = _write_known_hosts(module, known_hosts)
    monkeypatch.setattr(module, "EXPECTED_HOST_KEY_FINGERPRINT", fingerprint)
    identity_file = tmp_path / "p30_identity"
    _write_identity(identity_file)
    frozen_source = tmp_path / "run_p30_actual_host_prerequisite_preflight.py"
    _write_frozen_client_source(frozen_source)
    monkeypatch.setattr(module, "__file__", str(frozen_source))

    def forbidden_run(*_args: object, **_kwargs: object) -> None:
        pytest.fail("SSH must not run after a client-source digest mismatch")

    monkeypatch.setattr(module.subprocess, "run", forbidden_run)
    assert (
        module.main(
            [
                "--known-hosts",
                str(known_hosts),
                "--identity-file",
                str(identity_file),
                "--expected-client-source-sha256",
                "0" * 64,
            ]
        )
        == 125
    )


def test_client_source_requires_exact_digest_and_stable_nofollow_authority(
    module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"reviewed source-commit blob\n"
    source = tmp_path / "p30_preflight.py"
    source.write_bytes(payload)
    source.chmod(0o400)
    expected = hashlib.sha256(payload).hexdigest()

    raw, authority = module._stable_client_source(source, expected)
    assert raw == payload
    assert authority == {
        "byte_count": len(payload),
        "gid": source.stat().st_gid,
        "link_count": 1,
        "mode_octal": "0400",
        "path": str(source),
        "sha256": expected,
        "stable_nofollow_read": True,
        "uid": source.stat().st_uid,
    }

    with pytest.raises(module.PreflightError, match="SHA-256 differs"):
        module._stable_client_source(source, "0" * 64)
    with pytest.raises(module.PreflightError, match="lowercase SHA-256"):
        module._stable_client_source(source, expected.upper())

    link = tmp_path / "p30_preflight_link.py"
    link.symlink_to(source)
    with pytest.raises(module.PreflightError, match="exact caller UID/GID"):
        module._stable_client_source(link, expected)

    for wrong_mode in (0o000, 0o600, 0o640, 0o644):
        source.chmod(wrong_mode)
        with pytest.raises(module.PreflightError, match="physical mode 0400"):
            module._stable_client_source(source, expected)

    source.chmod(0o400)
    monkeypatch.setattr(module.os, "getgid", lambda: source.stat().st_gid + 1)
    with pytest.raises(module.PreflightError, match="exact caller UID/GID"):
        module._stable_client_source(source, expected)


def test_client_requires_reviewed_identity_and_disables_ambient_ssh_config(
    module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ambient_home = tmp_path / "ambient-home"
    (ambient_home / ".ssh").mkdir(parents=True)
    (ambient_home / ".ssh/config").write_text(
        "Host *\n  ProxyCommand false\n  IdentityFile /tmp/unreviewed\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("HOME", str(ambient_home))
    known_hosts = tmp_path / "known_hosts"
    identity_file = tmp_path / "reviewed_identity"
    _write_identity(identity_file)
    digest = "a" * 64

    command = module.build_ssh_command(known_hosts, identity_file, digest)
    ssh_options = command[: command.index(module.SSH_TARGET)]
    assert command[:3] == ["/usr/bin/ssh", "-F", "/dev/null"]
    assert command.count("-F") == 1
    assert ssh_options.count("-i") == 1
    assert ssh_options[ssh_options.index("-i") + 1] == str(identity_file)
    assert "IdentitiesOnly=yes" in command
    assert "ProxyCommand=none" in command
    assert "ProxyJump=none" in command
    assert str(ambient_home / ".ssh/config") not in command
    assert "/tmp/unreviewed" not in command

    with pytest.raises(SystemExit) as missing_identity:
        module.main(
            [
                "--known-hosts",
                str(known_hosts),
                "--expected-client-source-sha256",
                digest,
            ]
        )
    assert missing_identity.value.code == 2


def test_client_source_detects_a_source_identity_race(
    module: ModuleType,
    tmp_path: pathlib.Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"reviewed source-commit blob\n"
    source = tmp_path / "p30_preflight.py"
    source.write_bytes(payload)
    source.chmod(0o400)
    expected = hashlib.sha256(payload).hexdigest()
    original = module._stable_fields
    calls = 0

    def inject_identity_change(value: object) -> tuple[int, ...]:
        nonlocal calls
        calls += 1
        fields = original(value)
        if calls == 12:
            return (*fields[:-1], fields[-1] + 1)
        return fields

    monkeypatch.setattr(module, "_stable_fields", inject_identity_change)
    with pytest.raises(module.PreflightError, match="changed during stable read"):
        module._stable_client_source(source, expected)


def test_remote_preflight_calls_every_required_read_only_gate(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []

    monkeypatch.setattr(
        module,
        "_assert_absent",
        lambda path, label: calls.append(("absent", str(path), label)) or {"absent": True},
    )
    monkeypatch.setattr(module, "_secure_authority", lambda: calls.append("secure") or {})
    monkeypatch.setattr(module, "_account_authority", lambda: calls.append("account") or {})
    monkeypatch.setattr(module, "_executable_authority", lambda: calls.append("executables") or {})
    monkeypatch.setattr(module, "_docker_authority", lambda: calls.append("docker") or {})
    monkeypatch.setattr(module, "_gpu_authority", lambda: calls.append("gpu") or {})
    monkeypatch.setattr(
        module,
        "_git_repository",
        lambda path, **kwargs: calls.append(("git", str(path), kwargs)) or {},
    )
    monkeypatch.setattr(
        module,
        "_stable_file",
        lambda path, **kwargs: calls.append(("file", str(path), kwargs)) or {},
    )

    result = module.run_remote_preflight()

    assert calls[:2] == [
        ("absent", "/secure/p30", "P30 namespace"),
        (
            "absent",
            "/var/lib/optimizationml-p30-20260906-03",
            "P30 ledger",
        ),
    ]
    assert calls[2:6] == ["secure", "account", "executables", "docker"]
    assert calls[6] == "gpu"
    assert [entry[0] for entry in calls[7:]] == ["git", "git", "file", "file"]
    assert result["status"] == "all_read_only_prerequisites_passed_before_p30_state_creation"
    assert result["claim_boundary"] == {
        "container_created_or_started": False,
        "cuda_initialized": False,
        "gpu_queried_with_nvidia_smi_only": True,
        "p30_namespace_ledger_or_container_state_created": False,
        "scientific_observation": False,
        "training_run": False,
    }


def test_git_commands_bind_the_exact_uid1000_repository_as_safe(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: dict[str, object] = {}

    def fake_read_only(command: list[str], **kwargs: object) -> str:
        observed["command"] = command
        observed.update(kwargs)
        return ""

    monkeypatch.setattr(module, "_run_read_only", fake_read_only)
    repository = pathlib.PurePosixPath("/secure/p23/nanoGPT")
    module._git(repository, "status", "--porcelain=v1")

    command = observed["command"]
    assert isinstance(command, list)
    assert command[:3] == [
        "/usr/bin/git",
        "-c",
        "safe.directory=/secure/p23/nanoGPT",
    ]
    assert observed["environment"] == module.GIT_ENVIRONMENT


def test_absence_gate_rejects_files_directories_and_dangling_symlinks(
    module: ModuleType, tmp_path: pathlib.Path
) -> None:
    absent = tmp_path / "absent"
    assert module._assert_absent(pathlib.PurePosixPath(absent), "fixture")["absent"] is True

    absent.write_text("occupied", encoding="utf-8")
    with pytest.raises(module.PreflightError, match="already exists"):
        module._assert_absent(pathlib.PurePosixPath(absent), "fixture")
    absent.unlink()
    absent.symlink_to(tmp_path / "missing-target")
    with pytest.raises(module.PreflightError, match="already exists"):
        module._assert_absent(pathlib.PurePosixPath(absent), "fixture")


def test_stable_file_gate_checks_hash_size_and_rejects_symlinks(
    module: ModuleType, tmp_path: pathlib.Path
) -> None:
    payload = b"p30-read-only-input\n"
    source = tmp_path / "input.json"
    source.write_bytes(payload)
    expected = hashlib.sha256(payload).hexdigest()
    record = module._stable_file(
        pathlib.PurePosixPath(source),
        expected_sha256=expected,
        expected_bytes=len(payload),
        label="fixture",
    )
    assert record["sha256"] == expected
    assert record["byte_count"] == len(payload)

    with pytest.raises(module.PreflightError, match="SHA-256 or byte count differs"):
        module._stable_file(
            pathlib.PurePosixPath(source),
            expected_sha256="0" * 64,
            expected_bytes=len(payload),
            label="fixture",
        )
    link = tmp_path / "link.json"
    link.symlink_to(source)
    with pytest.raises(module.PreflightError, match="without following links"):
        module._stable_file(
            pathlib.PurePosixPath(link),
            expected_sha256=expected,
            expected_bytes=len(payload),
            label="fixture link",
        )


def test_git_gate_accepts_exact_clean_checkout_and_rejects_dirt_and_indirection(
    module: ModuleType, tmp_path: pathlib.Path
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(["init"], cwd=repository)
    _git(["config", "user.name", "P30 Test"], cwd=repository)
    _git(["config", "user.email", "p30@example.invalid"], cwd=repository)
    (repository / "tracked.txt").write_text("frozen\n", encoding="utf-8")
    _git(["add", "tracked.txt"], cwd=repository)
    _git(["commit", "-m", "frozen"], cwd=repository)
    commit = _git(["rev-parse", "HEAD"], cwd=repository)
    tree = _git(["rev-parse", "HEAD^{tree}"], cwd=repository)

    record = module._git_repository(
        pathlib.PurePosixPath(repository),
        expected_commit=commit,
        expected_tree=tree,
        label="fixture checkout",
    )
    assert record["clean_including_ignored"] is True
    assert record["object_indirections_absent"] is True

    untracked = repository / "untracked.txt"
    untracked.write_text("dirt\n", encoding="utf-8")
    with pytest.raises(module.PreflightError, match="tracked change, untracked file, or ignored"):
        module._git_repository(
            pathlib.PurePosixPath(repository),
            expected_commit=commit,
            expected_tree=tree,
            label="fixture checkout",
        )
    untracked.unlink()
    alternates = repository / ".git/objects/info/alternates"
    alternates.write_text("/tmp/forbidden\n", encoding="utf-8")
    with pytest.raises(module.PreflightError, match="object indirection"):
        module._git_repository(
            pathlib.PurePosixPath(repository),
            expected_commit=commit,
            expected_tree=tree,
            label="fixture checkout",
        )


def test_gpu_and_docker_gates_use_queries_only(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[list[str]] = []

    def fake_run(command: list[str], **_kwargs: object) -> str:
        observed.append(command)
        if command[0] == "/usr/bin/nvidia-smi":
            return f"{module.GPU_UUID}, {module.GPU_NAME}"
        if "container" in command:
            return "unrelated-container"
        return module.IMAGE_ID

    monkeypatch.setattr(module, "_run_read_only", fake_run)
    assert module._gpu_authority()["cuda_initialized"] is False
    assert module._docker_authority()["container_name_absent"] is True

    flattened = [token for command in observed for token in command]
    assert "run" not in flattened
    assert "start" not in flattened
    assert "create" not in flattened
    assert "pull" not in flattened
    assert "exec" not in flattened
    assert "container" in flattened and "ls" in flattened
    assert "image" in flattened and "inspect" in flattened


def test_script_contains_no_state_creation_cuda_or_training_primitive() -> None:
    source = SCRIPT.read_text(encoding="utf-8")
    syntax = ast.parse(source)
    imported_roots = {
        alias.name.partition(".")[0]
        for node in ast.walk(syntax)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported_roots.update(
        node.module.partition(".")[0]
        for node in ast.walk(syntax)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    )
    assert "torch" not in imported_roots
    for forbidden in (
        "O_CREAT",
        "O_WRONLY",
        "os.mkdir(",
        "os.makedirs(",
        "os.chmod(",
        "os.chown(",
        "os.unlink(",
        "os.remove(",
        "os.rename(",
        "os.replace(",
        '"run",',
        '"start",',
        '"create",',
        '"pull",',
        '"checkout",',
        '"fetch",',
    ):
        assert forbidden not in source
    assert "--query-gpu=uuid,name" in source
    assert '"status",' in source
    assert '"container", "ls"' in source
    assert '"image", "inspect"' in source


def test_remote_main_emits_one_canonical_json_record(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = {"status": "passed", "z": 1, "a": False}
    monkeypatch.setattr(module, "run_remote_preflight", lambda: payload)
    digest = "a" * 64
    assert module.main(["--remote", "--expected-client-source-sha256", digest]) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    expected = {
        **payload,
        "client_source": {"sha256": digest, "streamed_over_pinned_ssh": True},
    }
    assert captured.out == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"


def test_remote_failure_is_fail_closed_without_success_json(
    module: ModuleType, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fail() -> dict[str, object]:
        raise module.PreflightError("injected mismatch")

    monkeypatch.setattr(module, "run_remote_preflight", fail)
    assert module.main(["--remote", "--expected-client-source-sha256", "a" * 64]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "injected mismatch" in captured.err


def test_required_executable_inventory_covers_prepare_and_localization_tools(
    module: ModuleType,
) -> None:
    required = set(module.REQUIRED_EXECUTABLES)
    assert {
        "/bin/bash",
        "/usr/bin/docker",
        "/usr/bin/git",
        "/usr/bin/nsenter",
        "/usr/bin/nvidia-smi",
        "/usr/bin/python3",
        "/usr/bin/setpriv",
        "/usr/bin/sudo",
    } <= required
    assert all(pathlib.PurePosixPath(path).is_absolute() for path in required)
    assert stat.S_IMODE(SCRIPT.stat().st_mode) in {0o644, 0o755}
