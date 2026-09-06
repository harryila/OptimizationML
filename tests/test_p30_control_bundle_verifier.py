from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/verify_p30_control_bundle.py"
RUNBOOK = ROOT / "experiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md"
SPEC = importlib.util.spec_from_file_location("p30_bundle_verifier", SOURCE)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)


def _git(*arguments: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> str:
    process_env = os.environ.copy()
    process_env.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    if env:
        process_env.update(env)
    process = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ],
        cwd=cwd,
        env=process_env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
    return process.stdout.strip()


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _seal_program() -> str:
    source = RUNBOOK.read_text(encoding="utf-8")
    start = "# P30_REVIEWED_ORCHESTRATOR_SEAL_BEGIN\n"
    end = "# P30_REVIEWED_ORCHESTRATOR_SEAL_END\n"
    assert source.count(start) == source.count(end) == 1
    return source.split(start, 1)[1].split(end, 1)[0]


def _run_seal(
    source: Path,
    ledger: Path,
    expected_digest: str,
    *,
    program: str | None = None,
) -> subprocess.CompletedProcess[str]:
    seal_program = program or _seal_program()
    # The production seal deliberately requires descriptor-based ACL-xattr
    # inspection on Linux.  Darwin's Python build does not expose
    # ``os.listxattr`` at all, so provide a test-only syscall stand-in for the
    # portable positive/control cases.  ACL-negative tests replace this shim
    # with their hostile result below; no production source is changed.
    seal_program = (
        "import os\n"
        "if not hasattr(os, 'listxattr'):\n"
        "    os.listxattr = lambda _descriptor: []\n" + seal_program
    )
    return subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            seal_program,
            str(source),
            str(ledger),
            expected_digest,
            str(os.getuid()),
            str(os.getgid()),
            str(os.getuid()),
            str(os.getgid()),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _commit_tree(repository: Path, tree: str, parent: str, message: str) -> str:
    identity = {
        "GIT_AUTHOR_NAME": "P30 Test",
        "GIT_AUTHOR_EMAIL": "p30@example.invalid",
        "GIT_AUTHOR_DATE": "2026-09-06T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "P30 Test",
        "GIT_COMMITTER_EMAIL": "p30@example.invalid",
        "GIT_COMMITTER_DATE": "2026-09-06T00:00:00+00:00",
    }
    process = subprocess.run(
        ["git", "commit-tree", tree, "-p", parent],
        cwd=repository,
        env={**os.environ, **identity, "GIT_CONFIG_NOSYSTEM": "1"},
        input=f"{message}\n",
        capture_output=True,
        text=True,
        check=True,
    )
    return process.stdout.strip()


@pytest.fixture
def transfer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    secure_parent = tmp_path / "secure"
    p30_root = secure_parent / "p30"
    transport = p30_root / "transport-20260906-03"
    secure_parent.mkdir(mode=0o755)
    p30_root.mkdir(mode=0o755)
    transport.mkdir(mode=0o700)
    secure_parent.chmod(0o755)
    p30_root.chmod(0o755)
    transport.chmod(0o700)
    monkeypatch.setattr(verifier, "SECURE_PARENT", secure_parent)
    monkeypatch.setattr(verifier, "SECURE_PARENT_UID", os.getuid())
    monkeypatch.setattr(verifier, "SECURE_PARENT_GID", os.getgid())
    monkeypatch.setattr(verifier, "NAMESPACE_ROOT", p30_root)
    monkeypatch.setattr(verifier, "NAMESPACE_UID", os.getuid())
    monkeypatch.setattr(verifier, "NAMESPACE_GID", os.getgid())
    monkeypatch.setattr(verifier, "TRANSPORT_UID", os.getuid())
    monkeypatch.setattr(verifier, "TRANSPORT_GID", os.getgid())
    monkeypatch.setattr(verifier, "VERIFIER_UID", os.geteuid())
    monkeypatch.setattr(verifier, "VERIFIER_GID", os.getegid())
    monkeypatch.setattr(verifier, "ATTEMPT_UID", os.getuid())
    monkeypatch.setattr(verifier, "ATTEMPT_GID", os.getgid())
    monkeypatch.setattr(verifier, "EVIDENCE_UID", os.getuid())
    monkeypatch.setattr(verifier, "EVIDENCE_GID", os.getgid())
    monkeypatch.setattr(verifier, "EXECUTION_UID", os.getuid())
    monkeypatch.setattr(verifier, "EXECUTION_GID", os.getgid())
    if getattr(verifier.os, "listxattr", None) is None:
        monkeypatch.setattr(verifier.os, "listxattr", lambda _descriptor: [], raising=False)
    monkeypatch.setattr(verifier, "REQUIRE_EXECUTING_PYTHON_ISOLATION", False)
    real_reconstruction = verifier.run_frozen_reconstruction

    def run_fixture_reconstruction(control: Path, **kwargs: object) -> dict[str, object]:
        if kwargs.get("label") == "P30 contract":
            return {
                "label": "P30 contract",
                "reconstructor_path": verifier.P30_RECONSTRUCTOR,
                "reconstructor_sha256": "0" * 64,
                "schema_version": verifier.P30_RECONSTRUCTION_SCHEMA,
                "canonical_sha256": "1" * 64,
                "check_count": 15,
                "true_check_count": 15,
                "internally_consistent": True,
            }
        return real_reconstruction(control, **kwargs)

    monkeypatch.setattr(verifier, "run_frozen_reconstruction", run_fixture_reconstruction)
    bootstrap = transport / "verify_p30_control_bundle.py"
    bootstrap.write_bytes(SOURCE.read_bytes())
    bootstrap.chmod(0o600)
    monkeypatch.setattr(verifier, "__file__", str(bootstrap))
    seed = tmp_path / "seed.git"
    _git("clone", "--mirror", str(ROOT), str(seed))
    index = tmp_path / "source.index"
    index_environment = {"GIT_INDEX_FILE": str(index)}
    _git("-C", str(seed), "read-tree", verifier.P29_COMMIT, env=index_environment)
    verifier_blob = _git("-C", str(seed), "hash-object", "-w", str(SOURCE))
    _git(
        "-C",
        str(seed),
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{verifier_blob},{verifier.VERIFIER_SOURCE}",
        env=index_environment,
    )
    orchestrator_source = ROOT / verifier.ORCHESTRATOR_SOURCE
    orchestrator_digest = _digest(orchestrator_source)
    orchestrator_blob = _git("-C", str(seed), "hash-object", "-w", str(orchestrator_source))
    _git(
        "-C",
        str(seed),
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{orchestrator_blob},{verifier.ORCHESTRATOR_SOURCE}",
        env=index_environment,
    )
    fixture_contract = json.loads((ROOT / verifier.P30_CONTRACT).read_text(encoding="utf-8"))
    fixture_contract["host_orchestration"]["source_sha256"] = orchestrator_digest
    fixture_contract_path = tmp_path / "fixture-p30-contract.json"
    fixture_contract_path.write_text(
        json.dumps(fixture_contract, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    contract_blob = _git("-C", str(seed), "hash-object", "-w", str(fixture_contract_path))
    _git(
        "-C",
        str(seed),
        "update-index",
        "--add",
        "--cacheinfo",
        f"100644,{contract_blob},{verifier.P30_CONTRACT}",
        env=index_environment,
    )
    source_tree = _git("-C", str(seed), "write-tree", env=index_environment)
    source_commit = _commit_tree(seed, source_tree, verifier.P29_COMMIT, "P30 source")
    source_tree = _git("-C", str(seed), "rev-parse", f"{source_commit}^{{tree}}")
    _git("-C", str(seed), "update-ref", verifier.BRANCH_REF, source_commit)

    source_bundle = transport / "p30_source.bundle"
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(source_bundle),
        verifier.BRANCH_REF,
        verifier.P29_BRANCH_REF,
        verifier.P29_TAG_REF,
        verifier.P28_BRANCH_REF,
        verifier.P28_TAG_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_CHECKPOINT_REF,
        verifier.P26_SOURCE_REF,
    )
    source_bundle.chmod(0o600)
    return {
        "monkeypatch": monkeypatch,
        "seed": seed,
        "bootstrap": bootstrap,
        "root": p30_root,
        "transport": transport,
        "control": transport / "control-source",
        "attempt": p30_root / "attempt-20260906-03",
        "execution": p30_root / "execution-20260906-03",
        "source_bundle": source_bundle,
        "source_commit": source_commit,
        "source_tree": source_tree,
        "orchestrator_digest": orchestrator_digest,
    }


def _source_args(transfer: dict[str, object], *, replay: bool = False) -> list[str]:
    transport = transfer["transport"]
    bundle = transfer["source_bundle"]
    assert isinstance(transport, Path) and isinstance(bundle, Path)
    args = [
        "--phase",
        "source",
        "--transport-root",
        str(transport),
        "--bundle",
        str(bundle),
        "--expected-bundle-sha256",
        _digest(bundle),
        "--expected-bundle-byte-count",
        str(bundle.stat().st_size),
        "--closure-repository",
        str(transport / "p30_source_closure.git"),
        "--control-checkout",
        str(transfer["control"]),
        "--attempt-root",
        str(transfer["attempt"]),
        "--execution-root",
        str(transfer["execution"]),
        "--expected-p30-commit",
        str(transfer["source_commit"]),
        "--expected-p30-tree",
        str(transfer["source_tree"]),
        "--bootstrap-verifier",
        str(transfer["bootstrap"]),
        "--expected-verifier-sha256",
        _digest(SOURCE),
        "--expected-orchestrator-sha256",
        str(transfer["orchestrator_digest"]),
    ]
    receipt = transport / "p30_source_bundle_receipt.json"
    if replay:
        args.extend(
            [
                "--verify-existing-receipt",
                str(receipt),
                "--expected-receipt-sha256",
                _digest(receipt),
            ]
        )
    else:
        args.extend(["--receipt-output", str(receipt)])
    return args


def _make_runtime_bundle(transfer: dict[str, object]) -> tuple[Path, str, str]:
    seed = transfer["seed"]
    transport = transfer["transport"]
    assert isinstance(seed, Path) and isinstance(transport, Path)
    work = transport.parent.parent / "runtime-work"
    _git("clone", str(seed), str(work))
    _git("-C", str(work), "checkout", "--detach", str(transfer["source_commit"]))
    for relative in (
        "experiments/training/p30_cuda_runtime_lock.json",
        "experiments/training/p30_host_attestation.json",
    ):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"test":true}\n', encoding="utf-8")
        _git("-C", str(work), "add", "--", relative)
    identity = {
        "GIT_AUTHOR_NAME": "P30 Test",
        "GIT_AUTHOR_EMAIL": "p30@example.invalid",
        "GIT_AUTHOR_DATE": "2026-09-06T00:01:00+00:00",
        "GIT_COMMITTER_NAME": "P30 Test",
        "GIT_COMMITTER_EMAIL": "p30@example.invalid",
        "GIT_COMMITTER_DATE": "2026-09-06T00:01:00+00:00",
    }
    _git("-C", str(work), "commit", "-m", "P30 runtime review", env=identity)
    runtime_commit = _git("-C", str(work), "rev-parse", "HEAD")
    runtime_tree = _git("-C", str(work), "rev-parse", "HEAD^{tree}")
    _git(
        "-C",
        str(work),
        "push",
        str(seed),
        f"HEAD:{verifier.BRANCH_REF}",
    )
    runtime_bundle = transport / "p30_runtime_review.bundle"
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(runtime_bundle),
        verifier.BRANCH_REF,
        verifier.P29_BRANCH_REF,
        verifier.P29_TAG_REF,
        verifier.P28_BRANCH_REF,
        verifier.P28_TAG_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_CHECKPOINT_REF,
        verifier.P26_SOURCE_REF,
    )
    runtime_bundle.chmod(0o600)
    return runtime_bundle, runtime_commit, runtime_tree


@pytest.fixture
def runtime_control_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, dict[str, bytes]]:
    control = tmp_path / "runtime-control"
    _git("init", "--initial-branch=runtime-control", str(control))
    monkeypatch.setattr(verifier, "TRANSPORT_UID", os.getuid())
    monkeypatch.setattr(verifier, "TRANSPORT_GID", os.getgid())
    if getattr(verifier.os, "listxattr", None) is None:
        monkeypatch.setattr(verifier.os, "listxattr", lambda _descriptor: [], raising=False)
    payloads = {
        "experiments/training/p30_cuda_runtime_lock.json": b'{"runtime":true}\n',
        "experiments/training/p30_host_attestation.json": b'{"attestation":true}\n',
    }
    for relative, raw in payloads.items():
        path = control / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        path.chmod(0o600)
    _git("-C", str(control), "add", "--", *payloads)
    identity = {
        "GIT_AUTHOR_NAME": "P30 Runtime Authority Test",
        "GIT_AUTHOR_EMAIL": "p30-runtime-authority@example.invalid",
        "GIT_AUTHOR_DATE": "2026-09-06T00:03:00+00:00",
        "GIT_COMMITTER_NAME": "P30 Runtime Authority Test",
        "GIT_COMMITTER_EMAIL": "p30-runtime-authority@example.invalid",
        "GIT_COMMITTER_DATE": "2026-09-06T00:03:00+00:00",
    }
    _git("-C", str(control), "commit", "-m", "Runtime controls", env=identity)
    for directory in (control, control / "experiments", control / "experiments/training"):
        directory.chmod(verifier.TRANSPORT_MODE)
    return control, payloads


def _validate_runtime_control_fixture(
    control: Path, *, relative_path: str, expected_native_raw: bytes
) -> dict[str, object]:
    control_fd = os.open(control, verifier._directory_open_flags())
    try:
        return verifier.validate_runtime_control_artifact(
            control,
            control_fd=control_fd,
            relative_path=relative_path,
            expected_native_raw=expected_native_raw,
        )
    finally:
        os.close(control_fd)


def _git_bytes(repository: Path, *arguments: str) -> bytes:
    environment = os.environ.copy()
    environment.update(
        {
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_CONFIG_SYSTEM": os.devnull,
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    process = subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "core.fsmonitor=false",
            *arguments,
        ],
        cwd=repository,
        env=environment,
        capture_output=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr.decode(errors="replace")
    return process.stdout


def _make_snapshot_git_checkout(path: Path, label: str) -> tuple[str, str]:
    _git("init", "--initial-branch=snapshot-test", str(path))
    (path / "tracked.txt").write_text(f"{label} snapshot\n", encoding="utf-8")
    executable = path / "executable.py"
    executable.write_text("#!/usr/bin/env python3\n", encoding="utf-8")
    executable.chmod(0o755)
    _git("-C", str(path), "add", "--", "tracked.txt", "executable.py")
    identity = {
        "GIT_AUTHOR_NAME": "P30 Snapshot Test",
        "GIT_AUTHOR_EMAIL": "p30-snapshot@example.invalid",
        "GIT_AUTHOR_DATE": "2026-09-06T00:02:00+00:00",
        "GIT_COMMITTER_NAME": "P30 Snapshot Test",
        "GIT_COMMITTER_EMAIL": "p30-snapshot@example.invalid",
        "GIT_COMMITTER_DATE": "2026-09-06T00:02:00+00:00",
    }
    _git("-C", str(path), "commit", "-m", f"{label} snapshot", env=identity)
    head = _git("-C", str(path), "rev-parse", "HEAD")
    tree = _git("-C", str(path), "rev-parse", "HEAD^{tree}")
    for key in ("core.ignorecase", "core.precomposeunicode"):
        subprocess.run(
            ["git", "-C", str(path), "config", "--local", "--unset-all", key],
            env={**os.environ, "GIT_CONFIG_NOSYSTEM": "1"},
            capture_output=True,
            check=False,
        )
    for key, value in verifier.SNAPSHOT_GIT_CONFIG.items():
        _git("-C", str(path), "config", "--local", key, value)
    _git("-C", str(path), "checkout", "--detach", head)
    return head, tree


def _snapshot_entry(path: Path, root: Path) -> dict[str, object]:
    info = path.lstat()
    relative = path.relative_to(root).as_posix()
    if stat.S_ISDIR(info.st_mode):
        return {
            "relative_path": relative,
            "kind": "directory",
            "mode_octal": format(stat.S_IMODE(info.st_mode), "04o"),
            "uid": info.st_uid,
            "gid": info.st_gid,
        }
    raw = path.read_bytes()
    return {
        "relative_path": relative,
        "kind": "file",
        "mode_octal": format(stat.S_IMODE(info.st_mode), "04o"),
        "uid": info.st_uid,
        "gid": info.st_gid,
        "byte_count": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _materialize_execution_snapshot(transfer: dict[str, object]) -> None:
    execution = transfer["execution"]
    control = transfer["control"]
    source_commit = str(transfer["source_commit"])
    monkeypatch = transfer["monkeypatch"]
    assert isinstance(execution, Path) and isinstance(control, Path)
    assert isinstance(monkeypatch, pytest.MonkeyPatch)
    execution.mkdir(mode=0o700)
    git_authorities: dict[str, dict[str, str]] = {}
    for label in ("repository", "nanogpt", "muon"):
        head, tree = _make_snapshot_git_checkout(execution / label, label)
        git_authorities[label] = {"commit": head, "tree": tree}
    monkeypatch.setattr(
        verifier,
        "SNAPSHOT_REPOSITORY_COMMIT",
        git_authorities["repository"]["commit"],
    )
    monkeypatch.setattr(
        verifier,
        "SNAPSHOT_REPOSITORY_TREE",
        git_authorities["repository"]["tree"],
    )
    monkeypatch.setattr(verifier, "NANOGPT_COMMIT", git_authorities["nanogpt"]["commit"])
    monkeypatch.setattr(verifier, "NANOGPT_TREE", git_authorities["nanogpt"]["tree"])
    monkeypatch.setattr(verifier, "MUON_COMMIT", git_authorities["muon"]["commit"])
    monkeypatch.setattr(verifier, "MUON_TREE", git_authorities["muon"]["tree"])

    data = execution / "data"
    materialized = data / "materialized"
    tools = data / ".p30-tools"
    materialized.mkdir(parents=True)
    tools.mkdir()
    fineweb_raw = b'{"test":"fineweb"}\n'
    (materialized / "p22_fineweb_manifest.json").write_bytes(fineweb_raw)
    monkeypatch.setattr(
        verifier,
        "SNAPSHOT_FINEWEB_MANIFEST_SHA256",
        hashlib.sha256(fineweb_raw).hexdigest(),
    )
    monkeypatch.setattr(verifier, "SNAPSHOT_FINEWEB_MANIFEST_BYTE_COUNT", len(fineweb_raw))
    helpers: dict[str, dict[str, object]] = {}
    for basename, source_path in verifier.SNAPSHOT_TOOL_SOURCE_PATHS.items():
        raw = _git_bytes(control, "show", f"{source_commit}:{source_path}")
        destination = tools / basename
        destination.write_bytes(raw)
        destination.chmod(int(verifier.SNAPSHOT_TOOL_MODES[basename], 8))
        helpers[basename] = {"sha256": hashlib.sha256(raw).hexdigest(), "byte_count": len(raw)}
    p26_raw = b'{"authenticated":"p26-test"}\n'
    p26_name = "p26-trace-off-a-failure.authenticated.json"
    (tools / p26_name).write_bytes(p26_raw)
    helpers[p26_name] = {
        "sha256": hashlib.sha256(p26_raw).hexdigest(),
        "byte_count": len(p26_raw),
    }
    monkeypatch.setattr(
        verifier,
        "SNAPSHOT_P26_FAILURE_SHA256",
        hashlib.sha256(p26_raw).hexdigest(),
    )
    monkeypatch.setattr(verifier, "SNAPSHOT_P26_FAILURE_BYTE_COUNT", len(p26_raw))

    for directory, _subdirectories, filenames in os.walk(execution, topdown=False):
        directory_path = Path(directory)
        for filename in filenames:
            path = directory_path / filename
            path.chmod(0o555 if stat.S_IMODE(path.stat().st_mode) & 0o111 else 0o444)
        if directory_path != execution:
            directory_path.chmod(0o555)

    entries = sorted(
        (
            _snapshot_entry(path, execution)
            for path in execution.rglob("*")
            if path != execution / verifier.SNAPSHOT_MANIFEST_BASENAME
        ),
        key=lambda entry: str(entry["relative_path"]),
    )
    manifest = {
        "schema_version": verifier.SNAPSHOT_SCHEMA,
        "source_authorities": {
            **git_authorities,
            "data": {
                "original_manifest_relative_path": "materialized/p22_fineweb_manifest.json",
                "original_manifest_sha256": hashlib.sha256(fineweb_raw).hexdigest(),
                "original_manifest_byte_count": len(fineweb_raw),
            },
            "helpers": {
                name: str(record["sha256"]) for name, record in helpers.items() if name != p26_name
            },
            "p26_authenticated_input": helpers[p26_name],
        },
        "entries": entries,
    }
    manifest_path = execution / verifier.SNAPSHOT_MANIFEST_BASENAME
    manifest_path.write_bytes(verifier.compact_canonical_json_bytes(manifest))
    manifest_path.chmod(0o444)
    execution.chmod(0o555)


def _rewrite_snapshot_manifest(execution: Path) -> None:
    manifest_path = execution / verifier.SNAPSHOT_MANIFEST_BASENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"] = sorted(
        (
            _snapshot_entry(path, execution)
            for path in execution.rglob("*")
            if path != manifest_path
        ),
        key=lambda entry: str(entry["relative_path"]),
    )
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(verifier.compact_canonical_json_bytes(manifest))
    manifest_path.chmod(0o444)


def _verify_snapshot_direct(transfer: dict[str, object]) -> dict[str, object]:
    execution = transfer["execution"]
    control = transfer["control"]
    assert isinstance(execution, Path) and isinstance(control, Path)
    descriptor = os.open(execution, verifier._directory_open_flags())
    try:
        return verifier.verify_immutable_execution_snapshot(
            execution,
            execution_fd=descriptor,
            control=control,
            source_commit=str(transfer["source_commit"]),
        )
    finally:
        os.close(descriptor)


def _materialize_runtime_attempt(transfer: dict[str, object]) -> None:
    attempt = transfer["attempt"]
    assert isinstance(attempt, Path)
    evidence = attempt / "evidence"
    evidence.mkdir(parents=True)
    attempt.chmod(0o700)
    evidence.chmod(0o700)
    for basename in verifier.RUNTIME_ATTEMPT_TOP_LEVEL:
        if basename == "evidence":
            continue
        (attempt / basename).write_text("test\n", encoding="utf-8")
    for basename in verifier.RUNTIME_EVIDENCE_ALLOWLIST:
        if basename in verifier.RUNTIME_EVIDENCE_FILES.values():
            content = '{"test":true}\n'
        elif basename in verifier.RUNTIME_SUCCESS_STATUS_FILES:
            content = "0\n"
        else:
            content = "test\n"
        (evidence / basename).write_text(content, encoding="utf-8")
    for basename in verifier.RUNTIME_ATTEMPT_TOP_LEVEL:
        if basename != "evidence":
            (attempt / basename).chmod(0o444)
    for basename in verifier.RUNTIME_EVIDENCE_ALLOWLIST:
        (evidence / basename).chmod(0o444)
    evidence.chmod(verifier.EVIDENCE_MODE)
    attempt.chmod(verifier.ATTEMPT_MODE)
    _materialize_execution_snapshot(transfer)


def _materialize_authority(transfer: dict[str, object]) -> Path:
    seed = transfer["seed"]
    transport = transfer["transport"]
    assert isinstance(seed, Path) and isinstance(transport, Path)
    authority = transport / verifier.AUTHORITY_BASENAME
    _git("init", "--initial-branch=p30-authority-unborn", str(authority))
    _git("-C", str(authority), "config", "core.autocrlf", "false")
    _git("-C", str(authority), "config", "core.filemode", "true")
    _git(
        "-C",
        str(authority),
        "fetch",
        "--no-tags",
        "--no-write-fetch-head",
        str(seed),
        f"+{verifier.P26_SOURCE_REF}:{verifier.P26_SOURCE_REF}",
    )
    _git("-C", str(authority), "checkout", "--detach", verifier.P26_SOURCE_COMMIT)
    authority.chmod(0o700)
    return authority


def _runtime_args(
    transfer: dict[str, object],
    bundle: Path,
    runtime_commit: str,
    runtime_tree: str,
    *,
    replay: bool = False,
) -> list[str]:
    transport = transfer["transport"]
    assert isinstance(transport, Path)
    args = [
        "--phase",
        "runtime-review",
        "--transport-root",
        str(transport),
        "--bundle",
        str(bundle),
        "--expected-bundle-sha256",
        _digest(bundle),
        "--expected-bundle-byte-count",
        str(bundle.stat().st_size),
        "--closure-repository",
        str(transport / "p30_runtime_review_closure.git"),
        "--control-checkout",
        str(transfer["control"]),
        "--attempt-root",
        str(transfer["attempt"]),
        "--execution-root",
        str(transfer["execution"]),
        "--expected-p30-commit",
        runtime_commit,
        "--expected-p30-tree",
        runtime_tree,
        "--expected-source-commit",
        str(transfer["source_commit"]),
        "--expected-source-tree",
        str(transfer["source_tree"]),
        "--bootstrap-verifier",
        str(transfer["bootstrap"]),
        "--expected-verifier-sha256",
        _digest(SOURCE),
        "--expected-orchestrator-sha256",
        str(transfer["orchestrator_digest"]),
    ]
    receipt = transport / "p30_runtime_review_bundle_receipt.json"
    if replay:
        args.extend(
            [
                "--verify-existing-receipt",
                str(receipt),
                "--expected-receipt-sha256",
                _digest(receipt),
            ]
        )
    else:
        args.extend(["--receipt-output", str(receipt)])
    return args


def _snapshot_only_args(transfer: dict[str, object]) -> list[str]:
    return [
        "--verify-execution-snapshot-only",
        "--transport-root",
        str(transfer["transport"]),
        "--control-checkout",
        str(transfer["control"]),
        "--execution-root",
        str(transfer["execution"]),
        "--bootstrap-verifier",
        str(transfer["bootstrap"]),
        "--expected-verifier-sha256",
        _digest(SOURCE),
        "--expected-source-commit",
        str(transfer["source_commit"]),
        "--expected-source-tree",
        str(transfer["source_tree"]),
    ]


def test_source_create_and_reviewed_receipt_replay(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    transport = transfer["transport"]
    control = transfer["control"]
    assert isinstance(transport, Path) and isinstance(control, Path)
    receipt_path = transport / "p30_source_bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == verifier.SCHEMA
    assert receipt["status"] == verifier.STATUS_BY_PHASE["source"]
    assert receipt["phase"] == "source"
    assert receipt["closed_bundle"]["advertised_ref_count"] == 9
    assert receipt["closed_bundle"]["prerequisite_count"] == 0
    barrier = receipt["reconstruction_publication_barrier"]
    assert barrier["p30_contract"]["true_check_count"] == 15
    assert barrier["p29_terminal_outcome"]["true_check_count"] == 11
    assert barrier["p29_contract"]["true_check_count"] == 15
    assert barrier["p28_terminal_outcome"]["true_check_count"] == 10
    assert barrier["p28_contract"]["true_check_count"] == 12
    assert barrier["p27_contract"]["true_check_count"] == 23
    assert barrier["receipt_publication_blocked_unless_all_pass"] is True
    assert receipt["permission_safe_namespace"]["namespace"]["mode_octal"] == "0755"
    assert receipt["permission_safe_namespace"]["transport"]["mode_octal"] == "0700"
    assert receipt["permission_safe_namespace"]["exact_namespace_child_inventory"] == {
        "names": [verifier.TRANSPORT_BASENAME],
        "types": {verifier.TRANSPORT_BASENAME: "directory"},
        "extra_siblings_and_capabilities_rejected": True,
        "inventory_revalidated_through_receipt_handling": True,
    }
    assert "receipt_sha256" not in receipt
    assert receipt["receipt_integrity"]["receipt_sha256_field_present"] is False
    reviewed = receipt["reviewed_orchestrator_source"]
    assert reviewed["relative_path"] == verifier.ORCHESTRATOR_SOURCE
    assert reviewed["git_mode"] == "100644"
    assert reviewed["physical_mode_octal"] == "0600"
    assert reviewed["sha256"] == transfer["orchestrator_digest"]
    assert reviewed["verifier_umask_octal"] == "0077"
    assert reviewed["contract_digest_binding"] is True
    assert (control / verifier.ORCHESTRATOR_SOURCE).stat().st_mode & 0o777 == 0o600
    assert _git("-C", str(control), "rev-parse", "HEAD") == transfer["source_commit"]
    assert verifier.main(_source_args(transfer, replay=True)) == 0


def test_actual_verifier_created_checkout_seals_exact_mode_0600_source(
    transfer: dict[str, object], tmp_path: Path
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    control = transfer["control"]
    assert isinstance(control, Path)
    source = control / verifier.ORCHESTRATOR_SOURCE
    assert stat.S_IMODE(source.stat().st_mode) == 0o600
    ledger = tmp_path / "seal-ledger"
    ledger.mkdir(mode=0o700)
    ledger.chmod(0o700)
    (ledger / "deferred").mkdir(mode=0o700)
    result = _run_seal(source, ledger, str(transfer["orchestrator_digest"]))
    assert result.returncode == 0, result.stderr
    sealed = ledger / "run_p30_umask_bound_control_seal.sh"
    assert sealed.read_bytes() == source.read_bytes()
    assert stat.S_IMODE(sealed.stat().st_mode) == 0o555
    assert sealed.stat().st_nlink == 1
    assert sorted(path.name for path in ledger.iterdir()) == [
        "deferred",
        "run_p30_umask_bound_control_seal.sh",
    ]
    second = _run_seal(source, ledger, str(transfer["orchestrator_digest"]))
    assert second.returncode != 0
    assert sealed.read_bytes() == source.read_bytes()


@pytest.mark.parametrize(
    "mutation, expected_error",
    [
        ("old_0644", "source authority differs"),
        ("symlink", "Too many levels of symbolic links"),
        ("hardlink", "source authority differs"),
        ("acl", "forbidden ACL"),
        ("hash", "hash differs"),
        ("stat", "changed during stable read"),
        ("o_excl", "File exists"),
    ],
)
def test_exact_seal_rejects_old_mode_alias_acl_hash_stat_and_o_excl_controls(
    tmp_path: Path, mutation: str, expected_error: str
) -> None:
    source_dir = tmp_path / "control" / "scripts"
    source_dir.mkdir(parents=True)
    source = source_dir / "run_p30_umask_bound_control_seal.sh"
    raw = b"#!/bin/bash\nexit 0\n"
    source.write_bytes(raw)
    source.chmod(0o600)
    if mutation == "old_0644":
        source.chmod(0o644)
    elif mutation == "symlink":
        source.unlink()
        target = tmp_path / "source-target"
        target.write_bytes(raw)
        target.chmod(0o600)
        source.symlink_to(target)
    elif mutation == "hardlink":
        os.link(source, tmp_path / "second-link")
    ledger = tmp_path / "ledger"
    ledger.mkdir(mode=0o700)
    ledger.chmod(0o700)
    (ledger / "deferred").mkdir(mode=0o700)
    digest = hashlib.sha256(raw).hexdigest()
    if mutation == "hash":
        digest = "0" * 64
    program = _seal_program()
    if mutation == "acl":
        program = program.replace(
            "source_parent = open_directory(source.parent)",
            'os.listxattr = lambda _descriptor: ["system.posix_acl_access"]\n'
            "source_parent = open_directory(source.parent)",
        )
    elif mutation == "stat":
        marker = "    after = os.fstat(source_fd)\n"
        assert program.count(marker) == 1
        program = program.replace(
            marker,
            marker
            + "    os.utime(source.name, ns=(after.st_atime_ns, after.st_mtime_ns + 1), "
            + "dir_fd=source_parent, follow_symlinks=False)\n",
        )
    elif mutation == "o_excl":
        marker = "    destination_fd = os.open(\n"
        assert program.count(marker) == 1
        program = program.replace(
            marker,
            "    collision_fd = os.open(\n"
            "        destination_name,\n"
            "        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC\n"
            "        | os.O_NOFOLLOW,\n"
            "        0o555,\n"
            "        dir_fd=ledger_fd,\n"
            "    )\n"
            '    os.write(collision_fd, b"retained\\n")\n'
            "    os.close(collision_fd)\n" + marker,
        )
    result = _run_seal(source, ledger, digest, program=program)
    assert result.returncode != 0
    assert expected_error in result.stderr
    destination = ledger / "run_p30_umask_bound_control_seal.sh"
    if mutation == "o_excl":
        assert destination.read_bytes() == b"retained\n"
    else:
        assert not destination.exists()


def test_orchestrator_receipt_binding_rejects_mode_0644_and_wrong_digest(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    control = transfer["control"]
    assert isinstance(control, Path)
    source = control / verifier.ORCHESTRATOR_SOURCE
    source.chmod(0o644)
    assert verifier.main(_source_args(transfer, replay=True)) == 1
    source.chmod(0o600)
    args = _source_args(transfer, replay=True)
    position = args.index("--expected-orchestrator-sha256") + 1
    args[position] = "0" * 64
    assert verifier.main(args) == 1


def test_runtime_create_and_reviewed_receipt_replay(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    create_args = _runtime_args(transfer, bundle, runtime_commit, runtime_tree)
    assert verifier.main(create_args) == 0
    control = transfer["control"]
    transport = transfer["transport"]
    assert isinstance(control, Path) and isinstance(transport, Path)
    assert _git("-C", str(control), "rev-parse", "HEAD") == runtime_commit
    for delta in verifier.RUNTIME_DELTA:
        status, relative = delta.split("\t", 1)
        assert status == "A"
        runtime_control = control / relative
        assert runtime_control.is_file()
        assert not runtime_control.is_symlink()
        runtime_stat = runtime_control.stat()
        assert stat.S_IMODE(runtime_stat.st_mode) == 0o600
        assert runtime_stat.st_nlink == 1
        assert not {
            "system.posix_acl_access",
            "system.posix_acl_default",
            "system.nfs4_acl",
            "system.richacl",
        }.intersection(os.listxattr(runtime_control))
    receipt = json.loads(
        (transport / "p30_runtime_review_bundle_receipt.json").read_text(encoding="utf-8")
    )
    control_authority = receipt["attempt_guard"]["runtime_control_checkout_authority"]
    assert control_authority["expected_uid"] == os.getuid()
    assert control_authority["expected_gid"] == os.getgid()
    assert control_authority["expected_physical_mode_octal"] == "0600"
    assert control_authority["expected_git_mode"] == "100644"
    assert control_authority["expected_link_count"] == 1
    assert control_authority["verifier_umask_octal"] == "0077"
    assert control_authority["all_regular_nonsymlink_single_link"] is True
    assert control_authority["all_acl_xattrs_absent"] is True
    assert control_authority["all_native_evidence_bytes_equal"] is True
    assert set(control_authority["artifacts"]) == {"runtime_lock", "host_attestation"}
    for label, basename in verifier.RUNTIME_EVIDENCE_FILES.items():
        runtime_control = control / "experiments/training" / basename
        authority = control_authority["artifacts"][label]
        runtime_stat = runtime_control.stat()
        assert authority["relative_path"] == f"experiments/training/{basename}"
        assert authority["git_mode"] == "100644"
        assert authority["physical_mode_octal"] == "0600"
        assert authority["uid"] == runtime_stat.st_uid == os.getuid()
        assert authority["gid"] == runtime_stat.st_gid == os.getgid()
        assert authority["device"] == runtime_stat.st_dev
        assert authority["inode"] == runtime_stat.st_ino
        assert authority["link_count"] == runtime_stat.st_nlink == 1
        assert authority["byte_count"] == runtime_stat.st_size
        assert authority["sha256"] == _digest(runtime_control)
        assert authority["verifier_umask_octal"] == "0077"
        assert authority["regular_file"] is True
        assert authority["nonsymlink_nofollow"] is True
        assert authority["stable_nofollow_read"] is True
        assert authority["acl_xattrs_absent"] is True
        assert authority["native_evidence_bytes_equal"] is True
        assert authority["git_blob_bytes_equal"] is True
        assert authority["componentwise_parent_nofollow"] is True
        assert [
            component["relative_path"]
            for component in authority["intermediate_directory_authority"]
        ] == ["experiments", "experiments/training"]
    assert set(receipt["permission_safe_namespace"]["guarded_child_directories"]) == {
        "attempt_evidence",
        "attempt_root",
        "control_checkout",
        "execution_root",
        "runtime-review_closure",
    }
    assert receipt["permission_safe_namespace"]["exact_namespace_child_inventory"][
        "names"
    ] == sorted(
        [verifier.TRANSPORT_BASENAME, verifier.ATTEMPT_BASENAME, verifier.EXECUTION_BASENAME]
    )
    snapshot = receipt["immutable_execution_snapshot"]
    assert snapshot["schema_version"] == verifier.SNAPSHOT_SCHEMA
    assert snapshot["manifest_sha256"] == _digest(
        Path(transfer["execution"]) / verifier.SNAPSHOT_MANIFEST_BASENAME
    )
    assert snapshot["git_checkouts_verified"] is True
    assert snapshot["tool_inventory_and_source_bindings_verified"] is True
    assert (
        verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree, replay=True))
        == 0
    )


def test_runtime_control_authority_record_binds_git_physical_and_native_bytes(
    runtime_control_checkout: tuple[Path, dict[str, bytes]],
) -> None:
    control, payloads = runtime_control_checkout
    for relative, raw in payloads.items():
        record = _validate_runtime_control_fixture(
            control,
            relative_path=relative,
            expected_native_raw=raw,
        )
        assert record["git_mode"] == "100644"
        assert record["physical_mode_octal"] == "0600"
        assert record["uid"] == os.getuid()
        assert record["gid"] == os.getgid()
        assert record["link_count"] == 1
        assert record["regular_file"] is True
        assert record["nonsymlink_nofollow"] is True
        assert record["acl_xattrs_absent"] is True
        assert record["native_evidence_bytes_equal"] is True
        assert record["git_blob_bytes_equal"] is True
        assert record["git_index_and_head_stable"] is True
        assert record["parent_directory_stable"] is True
        assert record["componentwise_parent_nofollow"] is True
        assert record["same_device_and_mount_as_control"] is True
        assert record["not_mountpoint"] is True
        assert [
            component["relative_path"] for component in record["intermediate_directory_authority"]
        ] == ["experiments", "experiments/training"]
        for component in record["intermediate_directory_authority"]:
            assert component["mode_octal"] == "0700"
            assert component["uid"] == os.getuid()
            assert component["gid"] == os.getgid()
            assert component["same_device_and_mount_as_control"] is True
            assert component["not_mountpoint"] is True
            assert component["acl_check"]["access_acl_absent"] is True
            assert component["acl_check"]["default_acl_absent"] is True


def test_production_runtime_control_modes_are_exactly_pinned() -> None:
    assert verifier.TRANSPORT_UID == 1000
    assert verifier.TRANSPORT_GID == 1000
    assert verifier.RUNTIME_CONTROL_PHYSICAL_MODE == 0o600
    assert verifier.RUNTIME_CONTROL_GIT_MODE == "100644"


@pytest.mark.parametrize(
    ("mutation", "expected_error"),
    [
        ("mode", "physical authority differs"),
        ("uid", "physical authority differs"),
        ("gid", "physical authority differs"),
        ("hardlink", "physical authority differs"),
        ("symlink", "O_NOFOLLOW"),
        ("directory", "physical authority differs"),
    ],
)
def test_runtime_control_authority_rejects_hostile_filesystem_mutations(
    runtime_control_checkout: tuple[Path, dict[str, bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    expected_error: str,
) -> None:
    control, payloads = runtime_control_checkout
    relative, raw = next(iter(payloads.items()))
    path = control / relative
    if mutation == "mode":
        path.chmod(0o644)
    elif mutation in {"uid", "gid"}:
        target = path.stat()
        real_fstat = os.fstat

        def hostile_file_owner(descriptor: int) -> os.stat_result:
            info = real_fstat(descriptor)
            if (info.st_dev, info.st_ino) != (target.st_dev, target.st_ino):
                return info
            fields = list(info)
            field = stat.ST_UID if mutation == "uid" else stat.ST_GID
            fields[field] += 1
            return os.stat_result(fields)

        monkeypatch.setattr(verifier.os, "fstat", hostile_file_owner)
    elif mutation == "hardlink":
        os.link(path, tmp_path / "hostile-hardlink")
    elif mutation == "symlink":
        target = tmp_path / "symlink-target"
        target.write_bytes(raw)
        path.unlink()
        path.symlink_to(target)
    elif mutation == "directory":
        path.unlink()
        path.mkdir()
    else:  # pragma: no cover - parametrization is exhaustive
        raise AssertionError(mutation)

    with pytest.raises(verifier.VerificationError, match=expected_error):
        _validate_runtime_control_fixture(
            control,
            relative_path=relative,
            expected_native_raw=raw,
        )


def test_runtime_control_authority_rejects_acl_xattr(
    runtime_control_checkout: tuple[Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, payloads = runtime_control_checkout
    relative, raw = next(iter(payloads.items()))

    def hostile_file_acl(descriptor: int) -> list[str]:
        return ["system.posix_acl_access"] if stat.S_ISREG(os.fstat(descriptor).st_mode) else []

    monkeypatch.setattr(verifier.os, "listxattr", hostile_file_acl)
    with pytest.raises(verifier.VerificationError, match="forbidden ACL xattr"):
        _validate_runtime_control_fixture(
            control,
            relative_path=relative,
            expected_native_raw=raw,
        )


def test_runtime_control_authority_rejects_intermediate_directory_acl(
    runtime_control_checkout: tuple[Path, dict[str, bytes]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, payloads = runtime_control_checkout
    relative, raw = next(iter(payloads.items()))
    experiments_inode = (control / "experiments").stat().st_ino

    def hostile_intermediate_acl(descriptor: int) -> list[str]:
        info = os.fstat(descriptor)
        return (
            ["system.posix_acl_access"]
            if stat.S_ISDIR(info.st_mode) and info.st_ino == experiments_inode
            else []
        )

    monkeypatch.setattr(verifier.os, "listxattr", hostile_intermediate_acl)
    with pytest.raises(verifier.VerificationError, match="ACL xattr present"):
        _validate_runtime_control_fixture(
            control,
            relative_path=relative,
            expected_native_raw=raw,
        )


def test_runtime_control_authority_rejects_intermediate_directory_symlink(
    runtime_control_checkout: tuple[Path, dict[str, bytes]],
) -> None:
    control, payloads = runtime_control_checkout
    relative, raw = next(iter(payloads.items()))
    experiments = control / "experiments"
    moved = control / "experiments-authentic"
    experiments.rename(moved)
    experiments.symlink_to(moved.name, target_is_directory=True)

    with pytest.raises(
        verifier.VerificationError,
        match="intermediate directory with O_NOFOLLOW",
    ):
        _validate_runtime_control_fixture(
            control,
            relative_path=relative,
            expected_native_raw=raw,
        )


def test_runtime_control_authority_rejects_intermediate_directory_race(
    runtime_control_checkout: tuple[Path, dict[str, bytes]],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    control, payloads = runtime_control_checkout
    relative, raw = next(iter(payloads.items()))
    experiments = control / "experiments"
    training = experiments / "training"
    moved = experiments / "training-authentic"
    hostile = tmp_path / "hostile-training"
    hostile.mkdir(mode=0o700)
    real_run_git = verifier.run_git
    raced = False

    def swap_intermediate_before_git_blob(
        arguments: list[str] | tuple[str, ...],
        *,
        cwd: Path | None = None,
        check: bool = True,
    ) -> subprocess.CompletedProcess[bytes]:
        nonlocal raced
        if not raced and "cat-file" in arguments and "blob" in arguments:
            raced = True
            training.rename(moved)
            training.symlink_to(hostile, target_is_directory=True)
        return real_run_git(arguments, cwd=cwd, check=check)

    monkeypatch.setattr(verifier, "run_git", swap_intermediate_before_git_blob)
    with pytest.raises(
        verifier.VerificationError,
        match="runtime-control intermediate directory changed",
    ):
        _validate_runtime_control_fixture(
            control,
            relative_path=relative,
            expected_native_raw=raw,
        )
    assert raced is True


def test_standalone_snapshot_verification_is_read_only_and_compact(
    transfer: dict[str, object], capfd: pytest.CaptureFixture[str]
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    assert verifier.main(_snapshot_only_args(transfer)) == 0
    output = capfd.readouterr().out.splitlines()[-1]
    assert " " not in output
    record = json.loads(output)
    assert record["status"] == "immutable_execution_snapshot_verified_before_container_launch"
    assert record["bundle_or_control_transition_performed"] is False
    assert record["immutable_execution_snapshot"]["git_checkouts_verified"] is True


def test_snapshot_manifest_must_cover_every_entry(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    manifest_path = execution / verifier.SNAPSHOT_MANIFEST_BASENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["entries"].pop()
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(verifier.compact_canonical_json_bytes(manifest))
    manifest_path.chmod(0o444)
    with pytest.raises(verifier.VerificationError, match="manifest differs from live contents"):
        _verify_snapshot_direct(transfer)


def test_snapshot_manifest_must_be_root_readable_immutable_mode(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    manifest_path = execution / verifier.SNAPSHOT_MANIFEST_BASENAME
    manifest_path.chmod(0o400)
    with pytest.raises(verifier.VerificationError, match="mode is not 0444"):
        _verify_snapshot_direct(transfer)


def test_snapshot_tracked_bytes_are_checked_against_index(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    tracked = execution / "repository/tracked.txt"
    tracked.chmod(0o644)
    tracked.write_text("manifest-authenticated but not indexed\n", encoding="utf-8")
    tracked.chmod(0o444)
    _rewrite_snapshot_manifest(execution)
    with pytest.raises(verifier.VerificationError, match="bytes differ from its index"):
        _verify_snapshot_direct(transfer)


def test_snapshot_tracked_mode_is_checked_against_index(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    tracked = execution / "repository/tracked.txt"
    tracked.chmod(0o555)
    _rewrite_snapshot_manifest(execution)
    with pytest.raises(verifier.VerificationError, match="mode differs from its index"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_noncanonical_git_config_value(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    git_directory = execution / "repository/.git"
    config = git_directory / "config"
    git_directory.chmod(0o755)
    config.chmod(0o644)
    _git("-C", str(execution / "repository"), "config", "core.filemode", "false")
    config.chmod(0o444)
    git_directory.chmod(0o555)
    _rewrite_snapshot_manifest(execution)
    with pytest.raises(verifier.VerificationError, match="local Git config differs"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_hardlinks(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    tools = execution / "data/.p30-tools"
    tools.chmod(0o755)
    os.link(
        tools / "p26-trace-off-a-failure.authenticated.json",
        tools / "unexpected-hardlink",
    )
    tools.chmod(0o555)
    with pytest.raises(verifier.VerificationError, match="hard linked"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_symlinks(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    tools = execution / "data/.p30-tools"
    tools.chmod(0o755)
    (tools / "unexpected-symlink").symlink_to("p26-trace-off-a-failure.authenticated.json")
    tools.chmod(0o555)
    with pytest.raises(verifier.VerificationError, match="contains a symlink"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_special_files(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    tools = execution / "data/.p30-tools"
    tools.chmod(0o755)
    os.mkfifo(tools / "unexpected-fifo", 0o444)
    tools.chmod(0o555)
    with pytest.raises(verifier.VerificationError, match="contains a special file"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_manifest_authenticated_extra_tool(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    tools = execution / "data/.p30-tools"
    tools.chmod(0o755)
    extra = tools / "unexpected-helper.py"
    extra.write_text("raise SystemExit('unexpected')\n", encoding="utf-8")
    extra.chmod(0o444)
    tools.chmod(0o555)
    _rewrite_snapshot_manifest(execution)
    with pytest.raises(verifier.VerificationError, match="tool inventory differs"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_http_alternates_even_when_manifested(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    info = execution / "repository/.git/objects/info"
    info.chmod(0o755)
    indirection = info / "http-alternates"
    indirection.write_text("https://example.invalid/objects\n", encoding="utf-8")
    indirection.chmod(0o444)
    info.chmod(0o555)
    _rewrite_snapshot_manifest(execution)
    with pytest.raises(verifier.VerificationError, match="object indirection"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_manifest_source_authority_tamper(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    manifest_path = execution / verifier.SNAPSHOT_MANIFEST_BASENAME
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_authorities"]["helpers"]["run_p27_cuda_deleted_mapping_localization.py"] = (
        "0" * 64
    )
    manifest_path.chmod(0o600)
    manifest_path.write_bytes(verifier.compact_canonical_json_bytes(manifest))
    manifest_path.chmod(verifier.SNAPSHOT_MANIFEST_MODE)
    with pytest.raises(verifier.VerificationError, match="helper binding differs"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_unexpected_xattrs(
    transfer: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    monkeypatch.setattr(verifier.os, "listxattr", lambda _descriptor: ["user.hostile"])
    with pytest.raises(verifier.VerificationError, match="unexpected extended attributes"):
        _verify_snapshot_direct(transfer)


def test_snapshot_rejects_nested_mount_inventory(
    transfer: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_runtime_attempt(transfer)
    execution = transfer["execution"]
    assert isinstance(execution, Path)
    monkeypatch.setattr(
        verifier,
        "_mount_inventory",
        lambda: [("1", "1:1", "/"), ("2", "1:2", str(execution / "data"))],
    )
    with pytest.raises(verifier.VerificationError, match="contains a mountpoint"):
        _verify_snapshot_direct(transfer)


def test_attempt_root_blocks_before_closure_creation(transfer: dict[str, object]) -> None:
    attempt = transfer["attempt"]
    transport = transfer["transport"]
    assert isinstance(attempt, Path) and isinstance(transport, Path)
    attempt.mkdir()
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_closure.git").exists()
    assert not (transport / "p30_source_bundle_receipt.json").exists()


@pytest.mark.parametrize("entry_type", ["directory", "file", "symlink"])
def test_extra_namespace_child_blocks_source_receipt(
    transfer: dict[str, object], entry_type: str
) -> None:
    namespace = transfer["root"]
    transport = transfer["transport"]
    assert isinstance(namespace, Path) and isinstance(transport, Path)
    extra = namespace / "undeclared-capability"
    if entry_type == "directory":
        extra.mkdir()
    elif entry_type == "file":
        extra.write_text("capability\n", encoding="utf-8")
    else:
        extra.symlink_to(transport.name, target_is_directory=True)
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_bundle_receipt.json").exists()


def test_extra_namespace_child_blocks_source_receipt_replay(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    namespace = transfer["root"]
    assert isinstance(namespace, Path)
    (namespace / "undeclared-capability").mkdir()
    assert verifier.main(_source_args(transfer, replay=True)) == 1


def test_symlink_bundle_is_rejected(transfer: dict[str, object]) -> None:
    transport = transfer["transport"]
    source_bundle = transfer["source_bundle"]
    assert isinstance(transport, Path) and isinstance(source_bundle, Path)
    actual = transport / "actual.bundle"
    source_bundle.rename(actual)
    source_bundle.symlink_to(actual.name)
    args = _source_args(transfer)
    assert verifier.main(args) == 1
    assert not (transport / "p30_source_closure.git").exists()


@pytest.mark.parametrize("operation", ("copy", "read"))
def test_stable_bundle_and_receipt_reads_reject_by_name_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    parent = tmp_path / "transport"
    parent.mkdir()
    source = parent / "authority.json"
    source.write_bytes(b"original authority bytes\n")
    source.chmod(0o600)
    moved = parent / "renamed-open-object.json"
    destination = tmp_path / "private-copy"
    real_read = verifier.os.read
    replacement_done = False

    def replace_name_after_first_read(descriptor: int, byte_count: int) -> bytes:
        nonlocal replacement_done
        chunk = real_read(descriptor, byte_count)
        if chunk and not replacement_done:
            replacement_done = True
            source.rename(moved)
            source.write_bytes(b"replacement authority\n")
            source.chmod(0o600)
        return chunk

    monkeypatch.setattr(verifier.os, "read", replace_name_after_first_read)
    parent_fd = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        with pytest.raises(verifier.VerificationError, match="changed during its stable read"):
            if operation == "copy":
                verifier.stable_copy_nofollow(source, destination, source_dir_fd=parent_fd)
            else:
                verifier.stable_read_nofollow(source, source_dir_fd=parent_fd)
    finally:
        os.close(parent_fd)

    assert replacement_done is True
    assert moved.read_bytes() == b"original authority bytes\n"
    assert source.read_bytes() == b"replacement authority\n"


def test_missing_ref_is_rejected_before_closure(transfer: dict[str, object]) -> None:
    seed = transfer["seed"]
    transport = transfer["transport"]
    bundle = transfer["source_bundle"]
    assert isinstance(seed, Path) and isinstance(transport, Path) and isinstance(bundle, Path)
    bundle.unlink()
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(bundle),
        verifier.BRANCH_REF,
        verifier.P28_BRANCH_REF,
        verifier.P28_TAG_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_SOURCE_REF,
    )
    bundle.chmod(0o600)
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_closure.git").exists()


def test_extra_advertised_ref_is_rejected_before_closure(
    transfer: dict[str, object],
) -> None:
    seed = transfer["seed"]
    transport = transfer["transport"]
    bundle = transfer["source_bundle"]
    assert isinstance(seed, Path) and isinstance(transport, Path) and isinstance(bundle, Path)
    _git("-C", str(seed), "update-ref", "refs/heads/unexpected-p30-ref", verifier.P28_COMMIT)
    bundle.unlink()
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(bundle),
        verifier.BRANCH_REF,
        verifier.P28_BRANCH_REF,
        verifier.P28_TAG_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_CHECKPOINT_REF,
        verifier.P26_SOURCE_REF,
        "refs/heads/unexpected-p30-ref",
    )
    bundle.chmod(0o600)
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_closure.git").exists()


def test_source_freeze_with_p27_instead_of_terminal_p28_parent_is_rejected(
    transfer: dict[str, object],
) -> None:
    seed = transfer["seed"]
    transport = transfer["transport"]
    bundle = transfer["source_bundle"]
    assert isinstance(seed, Path) and isinstance(transport, Path) and isinstance(bundle, Path)
    wrong_commit = _commit_tree(
        seed,
        str(transfer["source_tree"]),
        verifier.P27_COMMIT,
        "wrong P30 parent",
    )
    _git("-C", str(seed), "update-ref", verifier.BRANCH_REF, wrong_commit)
    bundle.unlink()
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(bundle),
        *verifier.expected_refs(wrong_commit),
    )
    bundle.chmod(0o600)
    args = _source_args(transfer)
    args[args.index("--expected-p30-commit") + 1] = wrong_commit
    assert verifier.main(args) == 1
    assert not Path(transfer["control"]).exists()
    assert not (transport / "p30_source_bundle_receipt.json").exists()


def test_prerequisite_header_is_rejected(tmp_path: Path) -> None:
    bundle = tmp_path / "test.bundle"
    bundle.write_bytes(
        b"# v2 git bundle\n-" + verifier.P27_COMMIT.encode("ascii") + b" prerequisite\n\nPACK"
    )
    refs, prerequisites = verifier.parse_v2_bundle_header(bundle)
    assert refs == {}
    assert prerequisites == [verifier.P27_COMMIT]


def test_receipt_hash_tamper_is_rejected(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    args = _source_args(transfer, replay=True)
    receipt = Path(args[args.index("--verify-existing-receipt") + 1])
    receipt.write_bytes(receipt.read_bytes() + b" ")
    assert verifier.main(args) == 1


def test_source_replay_rejects_wrong_mode_control_directory(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    control = transfer["control"]
    assert isinstance(control, Path)
    control.chmod(0o755)
    assert verifier.main(_source_args(transfer, replay=True)) == 1


def test_source_replay_rejects_ignored_untracked_control_file(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    control = transfer["control"]
    assert isinstance(control, Path)
    (control / ".DS_Store").write_bytes(b"ignored but stateful")
    assert verifier.main(_source_args(transfer, replay=True)) == 1


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("core.fsmonitor", "/bin/false"),
        ("core.worktree", "/tmp/p30-wrong-worktree"),
        ("filter.p30.process", "/bin/false"),
    ],
)
def test_source_replay_rejects_noncanonical_control_config_before_status(
    transfer: dict[str, object], key: str, value: str
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    control = transfer["control"]
    assert isinstance(control, Path)
    _git("-C", str(control), "config", key, value)
    assert verifier.main(_source_args(transfer, replay=True)) == 1


def test_source_replay_rejects_noncanonical_later_authority_config(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    authority = _materialize_authority(transfer)
    _git("-C", str(authority), "config", "core.fsmonitor", "/bin/false")
    assert verifier.main(_source_args(transfer, replay=True)) == 1


def test_runtime_wrong_attempt_mode_fails_before_runtime_closure(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    attempt = transfer["attempt"]
    transport = transfer["transport"]
    assert isinstance(attempt, Path) and isinstance(transport, Path)
    attempt.chmod(0o755)
    assert verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree)) == 1
    assert not (transport / "p30_runtime_review_closure.git").exists()


def test_runtime_writable_retained_file_fails_before_runtime_closure(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    attempt = transfer["attempt"]
    transport = transfer["transport"]
    assert isinstance(attempt, Path) and isinstance(transport, Path)
    (attempt / "evidence/p30-container-launch.exit-status.txt").chmod(0o644)
    assert verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree)) == 1
    assert not (transport / "p30_runtime_review_closure.git").exists()


def test_runtime_localization_marker_blocks_receipt(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    attempt = transfer["attempt"]
    transport = transfer["transport"]
    assert isinstance(attempt, Path) and isinstance(transport, Path)
    evidence = attempt / "evidence"
    evidence.chmod(0o700)
    (evidence / "p30-run-localization.invoked").write_text("forbidden\n", encoding="utf-8")
    evidence.chmod(verifier.EVIDENCE_MODE)
    assert verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree)) == 1
    assert not (transport / "p30_runtime_review_bundle_receipt.json").exists()


def test_runtime_nonzero_prepare_status_blocks_receipt(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    attempt = transfer["attempt"]
    transport = transfer["transport"]
    assert isinstance(attempt, Path) and isinstance(transport, Path)
    status_file = attempt / "evidence" / "p30-container-launch.exit-status.txt"
    status_file.chmod(0o644)
    status_file.write_text("1\n", encoding="utf-8")
    status_file.chmod(0o444)
    assert verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree)) == 1
    assert not (transport / "p30_runtime_review_bundle_receipt.json").exists()


def test_extra_namespace_child_blocks_runtime_receipt(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    namespace = transfer["root"]
    transport = transfer["transport"]
    assert isinstance(namespace, Path) and isinstance(transport, Path)
    (namespace / "undeclared-capability").mkdir()
    assert verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree)) == 1
    assert not (transport / "p30_runtime_review_bundle_receipt.json").exists()


def test_extra_namespace_child_blocks_runtime_receipt_replay(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    assert verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree)) == 0
    namespace = transfer["root"]
    assert isinstance(namespace, Path)
    (namespace / "undeclared-capability").mkdir()
    assert (
        verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree, replay=True))
        == 1
    )


def test_runtime_symlinked_evidence_directory_is_rejected(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    attempt = transfer["attempt"]
    transport = transfer["transport"]
    assert isinstance(attempt, Path) and isinstance(transport, Path)
    evidence = attempt / "evidence"
    actual = attempt / "evidence-actual"
    attempt.chmod(0o700)
    evidence.chmod(0o700)
    evidence.rename(actual)
    evidence.symlink_to(actual.name, target_is_directory=True)
    attempt.chmod(verifier.ATTEMPT_MODE)
    assert verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree)) == 1
    assert not (transport / "p30_runtime_review_closure.git").exists()


def test_runtime_wrong_delta_fails_without_advancing_control(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    seed = transfer["seed"]
    transport = transfer["transport"]
    assert isinstance(seed, Path) and isinstance(transport, Path)
    wrong_tree = verifier.P27_TREE
    runtime_commit = _commit_tree(seed, wrong_tree, str(transfer["source_commit"]), "wrong runtime")
    _git("-C", str(seed), "update-ref", verifier.BRANCH_REF, runtime_commit)
    bundle = transport / "p30_runtime_review.bundle"
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(bundle),
        verifier.BRANCH_REF,
        verifier.P28_BRANCH_REF,
        verifier.P28_TAG_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_CHECKPOINT_REF,
        verifier.P26_SOURCE_REF,
    )
    bundle.chmod(0o600)
    args = _runtime_args(transfer, bundle, runtime_commit, wrong_tree)
    assert verifier.main(args) == 1
    control = transfer["control"]
    assert isinstance(control, Path)
    assert _git("-C", str(control), "rev-parse", "HEAD") == transfer["source_commit"]


def test_strict_json_rejects_duplicates_and_nonfinite() -> None:
    with pytest.raises(verifier.VerificationError, match="duplicate JSON key"):
        verifier.strict_json_loads(b'{"a":1,"a":2}')
    with pytest.raises(verifier.VerificationError, match="nonfinite JSON constant"):
        verifier.strict_json_loads(b'{"a":NaN}')


def test_exact_constants_and_phase_layout() -> None:
    assert verifier.BRANCH_REF == "refs/heads/p30-umask-bound-control-seal"
    assert verifier.P29_COMMIT == "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf"
    assert verifier.P29_TREE == "6ee56f8c218eab1d25047f6f1046b97f7529d996"
    assert verifier.P28_COMMIT == "7274367f2b05fb3c9ed9f876a59be647703bbdc2"
    assert verifier.P28_TREE == "8d971e176e664ea47d8f769ae54f8cf349ceb3d5"
    assert verifier.expected_refs(verifier.P27_COMMIT) == {
        verifier.BRANCH_REF: verifier.P27_COMMIT,
        verifier.P29_BRANCH_REF: "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf",
        verifier.P29_TAG_REF: "cf3c60b1d5c004b9e601a6afbf630358f80f1d79",
        verifier.P28_BRANCH_REF: "7274367f2b05fb3c9ed9f876a59be647703bbdc2",
        verifier.P28_TAG_REF: "f3c81a2f14f853f369015a4f81b6695b52eec74e",
        verifier.P27_BRANCH_REF: verifier.P27_COMMIT,
        verifier.P27_TAG_REF: "c88b98eae9be2458abde45b05d3dccfe09c0c7ed",
        verifier.P26_CHECKPOINT_REF: "bacad707d3779bfa10957e18cb4c69b1a7f0cbce",
        verifier.P26_SOURCE_REF: "f35a7dca8f6bc39e9748e79b6712fab4a203396b",
    }
    assert verifier.PHASE_LAYOUT == {
        "source": {
            "bundle": "p30_source.bundle",
            "closure": "p30_source_closure.git",
            "receipt": "p30_source_bundle_receipt.json",
        },
        "runtime-review": {
            "bundle": "p30_runtime_review.bundle",
            "closure": "p30_runtime_review_closure.git",
            "receipt": "p30_runtime_review_bundle_receipt.json",
        },
    }
    assert len(verifier.RUNTIME_EVIDENCE_ALLOWLIST) == 27
    assert len(verifier.RUNTIME_SUCCESS_STATUS_FILES) == 10
    assert all(name.endswith(".exit-status.txt") for name in verifier.RUNTIME_SUCCESS_STATUS_FILES)
    assert {
        "p30-prepare-execution-snapshot-verification.stdout.log",
        "p30-prepare-execution-snapshot-verification.stderr.log",
        "p30-prepare-execution-snapshot-verification.exit-status.txt",
    }.issubset(verifier.RUNTIME_EVIDENCE_ALLOWLIST)
    assert {
        "p30-fresh-container-id-validation.stdout.log",
        "p30-fresh-container-id-validation.stderr.log",
        "p30-fresh-container-id-validation.exit-status.txt",
    }.issubset(verifier.RUNTIME_EVIDENCE_ALLOWLIST)
    assert not any(
        "prepare-contract-reconstruction" in name for name in verifier.RUNTIME_EVIDENCE_ALLOWLIST
    )
    assert verifier.SECURE_PARENT_UID == 0
    assert verifier.SECURE_PARENT_GID == 0
    assert verifier.VERIFIER_UID == 1000
    assert verifier.VERIFIER_GID == 1000
    assert verifier.ATTEMPT_MODE == 0o555
    assert verifier.EVIDENCE_UID == 0
    assert verifier.EVIDENCE_GID == 0
    assert verifier.EVIDENCE_MODE == 0o555
    assert verifier.EXECUTION_MODE == 0o555
    assert verifier.SNAPSHOT_SCHEMA == "passive-muon-p30-immutable-execution-snapshot-v1"
    assert verifier.SNAPSHOT_MANIFEST_MODE == 0o444
    assert tuple(sorted(verifier.SNAPSHOT_TOOL_BASENAMES)) == verifier.SNAPSHOT_TOOL_BASENAMES


def test_wrong_transport_mode_fails_before_bundle_consumption(
    transfer: dict[str, object],
) -> None:
    transport = transfer["transport"]
    assert isinstance(transport, Path)
    transport.chmod(0o755)
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_closure.git").exists()
    assert not (transport / "p30_source_bundle_receipt.json").exists()


def test_wrong_verifier_execution_identity_fails_before_bundle_consumption(
    transfer: dict[str, object], monkeypatch: pytest.MonkeyPatch
) -> None:
    transport = transfer["transport"]
    assert isinstance(transport, Path)
    monkeypatch.setattr(verifier, "VERIFIER_UID", os.geteuid() + 1)
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_closure.git").exists()
    assert not (transport / "p30_source_bundle_receipt.json").exists()


def test_symlinked_namespace_component_is_rejected(
    transfer: dict[str, object],
) -> None:
    namespace = transfer["root"]
    transport = transfer["transport"]
    assert isinstance(namespace, Path) and isinstance(transport, Path)
    actual = namespace.with_name(namespace.name + "-actual")
    namespace.rename(actual)
    namespace.symlink_to(actual.name, target_is_directory=True)
    assert verifier.main(_source_args(transfer)) == 1
    assert not (actual / transport.name / "p30_source_bundle_receipt.json").exists()


def test_preexisting_authority_checkout_blocks_source_receipt(
    transfer: dict[str, object],
) -> None:
    transport = transfer["transport"]
    assert isinstance(transport, Path)
    (transport / verifier.AUTHORITY_BASENAME).mkdir()
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_bundle_receipt.json").exists()


def test_reviewed_source_receipt_replays_with_exact_later_authority(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    _materialize_authority(transfer)
    assert verifier.main(_source_args(transfer, replay=True)) == 0


def test_reviewed_source_receipt_rejects_wrong_mode_later_authority(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    authority = _materialize_authority(transfer)
    authority.chmod(0o755)
    assert verifier.main(_source_args(transfer, replay=True)) == 1


def test_reviewed_source_receipt_rejects_symlinked_later_authority(
    transfer: dict[str, object],
) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    authority = _materialize_authority(transfer)
    actual = authority.with_name(authority.name + "-actual")
    authority.rename(actual)
    authority.symlink_to(actual.name, target_is_directory=True)
    assert verifier.main(_source_args(transfer, replay=True)) == 1


def test_hard_linked_bundle_is_rejected(transfer: dict[str, object]) -> None:
    bundle = transfer["source_bundle"]
    transport = transfer["transport"]
    assert isinstance(bundle, Path) and isinstance(transport, Path)
    os.link(bundle, transport / "second-link.bundle")
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p30_source_closure.git").exists()


def test_acl_or_default_acl_xattr_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        verifier.os,
        "listxattr",
        lambda _descriptor: ["system.posix_acl_default"],
        raising=False,
    )
    with pytest.raises(verifier.VerificationError, match="ACL xattr present"):
        verifier._validate_acl_absence({"namespace": 1, "transport": 2})


def test_subprocess_environment_strips_git_and_python_injection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GIT_DIR", "/hostile/git")
    monkeypatch.setenv("GIT_CONFIG_COUNT", "99")
    monkeypatch.setenv("PYTHONPATH", "/hostile/python")
    monkeypatch.setenv("PYTHONINSPECT", "1")
    environment = verifier._clean_git_environment()
    assert environment["GIT_OPTIONAL_LOCKS"] == "0"
    assert environment["GIT_NO_REPLACE_OBJECTS"] == "1"
    assert environment["GIT_ATTR_NOSYSTEM"] == "1"
    assert environment["PYTHONNOUSERSITE"] == "1"
    assert environment["PYTHONSAFEPATH"] == "1"
    assert "GIT_DIR" not in environment
    assert "GIT_CONFIG_COUNT" not in environment
    assert "PYTHONPATH" not in environment
    assert "PYTHONINSPECT" not in environment


def test_git_trusts_only_the_explicit_repository_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[str] = []

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        captured.extend(command)
        return subprocess.CompletedProcess(command, 0, b"", b"")

    monkeypatch.setattr(verifier.subprocess, "run", fake_run)
    verifier.run_git(["-C", "/secure/p30/execution-20260906-03/repository", "status"])
    assert "safe.directory=/secure/p30/execution-20260906-03/repository" in captured
    assert "safe.directory=*" not in captured


def test_exact_namespace_mountpoint_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    namespace = tmp_path / "p30"
    transport = namespace / verifier.TRANSPORT_BASENAME
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        f"41 1 8:1 / {namespace} rw - ext4 /dev/test rw\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(verifier, "MOUNTINFO_PATH", mountinfo)
    monkeypatch.setattr(verifier.platform, "system", lambda: "Linux")
    with pytest.raises(verifier.VerificationError, match="unexpected mountpoint"):
        verifier._validate_mount_layout(namespace, transport)
