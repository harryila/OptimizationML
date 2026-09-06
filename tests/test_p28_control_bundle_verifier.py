from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/verify_p28_control_bundle.py"
SPEC = importlib.util.spec_from_file_location("p28_bundle_verifier", SOURCE)
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
        ["git", "-c", "core.hooksPath=/dev/null", *arguments],
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


def _commit_tree(repository: Path, tree: str, parent: str, message: str) -> str:
    identity = {
        "GIT_AUTHOR_NAME": "P28 Test",
        "GIT_AUTHOR_EMAIL": "p28@example.invalid",
        "GIT_AUTHOR_DATE": "2026-09-06T00:00:00+00:00",
        "GIT_COMMITTER_NAME": "P28 Test",
        "GIT_COMMITTER_EMAIL": "p28@example.invalid",
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
def transfer(tmp_path: Path) -> dict[str, object]:
    bootstrap = tmp_path / "verify_p28_control_bundle.py"
    bootstrap.write_bytes(SOURCE.read_bytes())
    bootstrap.chmod(0o600)
    verifier.__file__ = str(bootstrap)
    seed = tmp_path / "seed.git"
    _git("clone", "--mirror", str(ROOT), str(seed))
    index = tmp_path / "source.index"
    index_environment = {"GIT_INDEX_FILE": str(index)}
    _git("-C", str(seed), "read-tree", verifier.P27_COMMIT, env=index_environment)
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
    source_tree = _git("-C", str(seed), "write-tree", env=index_environment)
    source_commit = _commit_tree(seed, source_tree, verifier.P27_COMMIT, "P28 source")
    source_tree = _git("-C", str(seed), "rev-parse", f"{source_commit}^{{tree}}")
    _git("-C", str(seed), "update-ref", verifier.BRANCH_REF, source_commit)

    p28_root = tmp_path / "p28"
    transport = p28_root / "transport-20260906-01"
    transport.mkdir(parents=True)
    source_bundle = transport / "p28_source.bundle"
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(source_bundle),
        verifier.BRANCH_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_CHECKPOINT_REF,
        verifier.P26_SOURCE_REF,
    )
    return {
        "seed": seed,
        "bootstrap": bootstrap,
        "root": p28_root,
        "transport": transport,
        "control": p28_root / "control-source",
        "attempt": p28_root / "attempt-20260906-01",
        "source_bundle": source_bundle,
        "source_commit": source_commit,
        "source_tree": source_tree,
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
        str(transport / "p28_source_closure.git"),
        "--control-checkout",
        str(transfer["control"]),
        "--attempt-root",
        str(transfer["attempt"]),
        "--expected-p28-commit",
        str(transfer["source_commit"]),
        "--expected-p28-tree",
        str(transfer["source_tree"]),
        "--bootstrap-verifier",
        str(transfer["bootstrap"]),
        "--expected-verifier-sha256",
        _digest(SOURCE),
    ]
    receipt = transport / "p28_source_bundle_receipt.json"
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
    work = transport.parent / "runtime-work"
    _git("clone", str(seed), str(work))
    _git("-C", str(work), "checkout", "--detach", str(transfer["source_commit"]))
    for relative in (
        "experiments/training/p28_cuda_runtime_lock.json",
        "experiments/training/p28_host_attestation.json",
    ):
        path = work / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text('{"test":true}\n', encoding="utf-8")
        _git("-C", str(work), "add", "--", relative)
    identity = {
        "GIT_AUTHOR_NAME": "P28 Test",
        "GIT_AUTHOR_EMAIL": "p28@example.invalid",
        "GIT_AUTHOR_DATE": "2026-09-06T00:01:00+00:00",
        "GIT_COMMITTER_NAME": "P28 Test",
        "GIT_COMMITTER_EMAIL": "p28@example.invalid",
        "GIT_COMMITTER_DATE": "2026-09-06T00:01:00+00:00",
    }
    _git("-C", str(work), "commit", "-m", "P28 runtime review", env=identity)
    runtime_commit = _git("-C", str(work), "rev-parse", "HEAD")
    runtime_tree = _git("-C", str(work), "rev-parse", "HEAD^{tree}")
    _git(
        "-C",
        str(work),
        "push",
        str(seed),
        f"HEAD:{verifier.BRANCH_REF}",
    )
    runtime_bundle = transport / "p28_runtime_review.bundle"
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(runtime_bundle),
        verifier.BRANCH_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_CHECKPOINT_REF,
        verifier.P26_SOURCE_REF,
    )
    return runtime_bundle, runtime_commit, runtime_tree


def _materialize_runtime_attempt(transfer: dict[str, object]) -> None:
    attempt = transfer["attempt"]
    assert isinstance(attempt, Path)
    evidence = attempt / "evidence"
    evidence.mkdir(parents=True)
    for basename in verifier.RUNTIME_ATTEMPT_TOP_LEVEL:
        if basename == "evidence":
            continue
        (attempt / basename).write_text("test\n", encoding="utf-8")
    for basename in verifier.RUNTIME_EVIDENCE_ALLOWLIST:
        content = (
            '{"test":true}\n' if basename in verifier.RUNTIME_EVIDENCE_FILES.values() else "test\n"
        )
        (evidence / basename).write_text(content, encoding="utf-8")


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
        str(transport / "p28_runtime_review_closure.git"),
        "--control-checkout",
        str(transfer["control"]),
        "--attempt-root",
        str(transfer["attempt"]),
        "--expected-p28-commit",
        runtime_commit,
        "--expected-p28-tree",
        runtime_tree,
        "--expected-source-commit",
        str(transfer["source_commit"]),
        "--expected-source-tree",
        str(transfer["source_tree"]),
        "--bootstrap-verifier",
        str(transfer["bootstrap"]),
        "--expected-verifier-sha256",
        _digest(SOURCE),
    ]
    receipt = transport / "p28_runtime_review_bundle_receipt.json"
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


def test_source_create_and_reviewed_receipt_replay(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    transport = transfer["transport"]
    control = transfer["control"]
    assert isinstance(transport, Path) and isinstance(control, Path)
    receipt_path = transport / "p28_source_bundle_receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == verifier.SCHEMA
    assert receipt["status"] == verifier.STATUS_BY_PHASE["source"]
    assert receipt["phase"] == "source"
    assert receipt["closed_bundle"]["advertised_ref_count"] == 5
    assert receipt["closed_bundle"]["prerequisite_count"] == 0
    assert receipt["p27_reconstruction"]["true_check_count"] == 23
    assert "receipt_sha256" not in receipt
    assert receipt["receipt_integrity"]["receipt_sha256_field_present"] is False
    assert _git("-C", str(control), "rev-parse", "HEAD") == transfer["source_commit"]
    assert verifier.main(_source_args(transfer, replay=True)) == 0


def test_runtime_create_and_reviewed_receipt_replay(transfer: dict[str, object]) -> None:
    assert verifier.main(_source_args(transfer)) == 0
    bundle, runtime_commit, runtime_tree = _make_runtime_bundle(transfer)
    _materialize_runtime_attempt(transfer)
    create_args = _runtime_args(transfer, bundle, runtime_commit, runtime_tree)
    assert verifier.main(create_args) == 0
    control = transfer["control"]
    assert isinstance(control, Path)
    assert _git("-C", str(control), "rev-parse", "HEAD") == runtime_commit
    assert (
        verifier.main(_runtime_args(transfer, bundle, runtime_commit, runtime_tree, replay=True))
        == 0
    )


def test_attempt_root_blocks_before_closure_creation(transfer: dict[str, object]) -> None:
    attempt = transfer["attempt"]
    transport = transfer["transport"]
    assert isinstance(attempt, Path) and isinstance(transport, Path)
    attempt.mkdir()
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p28_source_closure.git").exists()
    assert not (transport / "p28_source_bundle_receipt.json").exists()


def test_symlink_bundle_is_rejected(transfer: dict[str, object]) -> None:
    transport = transfer["transport"]
    source_bundle = transfer["source_bundle"]
    assert isinstance(transport, Path) and isinstance(source_bundle, Path)
    actual = transport / "actual.bundle"
    source_bundle.rename(actual)
    source_bundle.symlink_to(actual.name)
    args = _source_args(transfer)
    assert verifier.main(args) == 1
    assert not (transport / "p28_source_closure.git").exists()


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
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_SOURCE_REF,
    )
    assert verifier.main(_source_args(transfer)) == 1
    assert not (transport / "p28_source_closure.git").exists()


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
    bundle = transport / "p28_runtime_review.bundle"
    _git(
        "-C",
        str(seed),
        "bundle",
        "create",
        str(bundle),
        verifier.BRANCH_REF,
        verifier.P27_BRANCH_REF,
        verifier.P27_TAG_REF,
        verifier.P26_CHECKPOINT_REF,
        verifier.P26_SOURCE_REF,
    )
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
    assert verifier.expected_refs(verifier.P27_COMMIT) == {
        verifier.BRANCH_REF: verifier.P27_COMMIT,
        verifier.P27_BRANCH_REF: verifier.P27_COMMIT,
        verifier.P27_TAG_REF: "c88b98eae9be2458abde45b05d3dccfe09c0c7ed",
        verifier.P26_CHECKPOINT_REF: "bacad707d3779bfa10957e18cb4c69b1a7f0cbce",
        verifier.P26_SOURCE_REF: "f35a7dca8f6bc39e9748e79b6712fab4a203396b",
    }
    assert verifier.PHASE_LAYOUT == {
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
