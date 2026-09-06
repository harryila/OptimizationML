#!/usr/bin/env python3
"""Independently reconstruct P29's permission-safe localization bridge contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import subprocess
import sys
import tempfile
from collections.abc import Mapping
from functools import cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = (
    ROOT / "experiments/training/p29_permission_safe_bundle_localization_bridge_contract.json"
)
EXPECTED_CONTRACT_SHA256 = "54d10ab01b7d453d01931459b314fc4491b6e4317a02fb763ca9872686311b0a"
EXPECTED_SCHEMA = "passive-muon-p29-permission-safe-bundle-localization-bridge-contract-v1"
EXPECTED_STATUS = "frozen_pre_transfer_pre_container_no_training"
EXPECTED_BRANCH = "p29-permission-safe-bundle-localization-bridge"

P28_OUTCOME_PATH = "results/summaries/p28_bundle_complete_localization_bridge_outcome.json"
P28_OUTCOME_SHA256 = "298324d163aa12c1cb64bfdb912ae846152d0980ae3c981a54ead75a019305c9"
P28_OUTCOME_COMMIT = "7274367f2b05fb3c9ed9f876a59be647703bbdc2"
P28_OUTCOME_TREE = "8d971e176e664ea47d8f769ae54f8cf349ceb3d5"
P28_TAG_OBJECT = "f3c81a2f14f853f369015a4f81b6695b52eec74e"
P27_OUTCOME_COMMIT = "ec63550331925ded158e3f389e294e4d1f12db3a"
P27_TAG_OBJECT = "c88b98eae9be2458abde45b05d3dccfe09c0c7ed"
P26_CHECKPOINT_OBJECT = "bacad707d3779bfa10957e18cb4c69b1a7f0cbce"
P26_CHECKPOINT_COMMIT = "5429da23ff18888daa2312c530a4587780484d8b"
P26_SOURCE_OBJECT = "f35a7dca8f6bc39e9748e79b6712fab4a203396b"
P26_SOURCE_COMMIT = "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
SHA256_RE = re.compile(r"[0-9a-f]{64}\Z")
TRUSTED_GIT = Path("/usr/bin/git")
TRUSTED_GIT_OPTIONS = (
    "-c",
    "core.hooksPath=/dev/null",
    "-c",
    "core.fsmonitor=false",
    "-c",
    "protocol.file.allow=never",
)

EXPECTED_TOP_LEVEL_ORDER = [
    "schema_version",
    "status",
    "branch",
    "purpose",
    "claim_boundary",
    "terminal_parent",
    "unchanged_p28_authorities",
    "unchanged_p27_authorities",
    "required_bundle_refs",
    "permission_safe_layout",
    "bundle_closure",
    "bundle_verifier",
    "host_orchestration",
    "fresh_localization_attempt",
    "immutable_execution_snapshot",
    "locked_base_runtime",
    "two_phase_freeze",
    "no_training_boundary",
    "execution_order",
    "terminal_routing",
    "forbidden_changes",
]

EXPECTED_REQUIRED_REFS = [
    "refs/heads/p29-permission-safe-bundle-localization-bridge",
    "refs/heads/p28-bundle-complete-localization-bridge",
    "refs/tags/p28-control-parent-permission-diagnostic",
    "refs/heads/p27-cuda-deleted-mapping-localization",
    "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic",
    "refs/tags/p26-permission-safe-acquisition-checkpoint",
    "refs/tags/p26-attempt02-acquisition-source",
]
EXPECTED_ORCHESTRATOR_ENVIRONMENT = [
    "P29_ATTEMPT_ID",
    "P29_GPU_UUID",
    "P29_AUTHORITY_REPO",
    "P29_NANOGPT_HOST",
    "P29_MUON_HOST",
    "P29_DATA_HOST",
    "P29_SOURCE_FREEZE_COMMIT",
    "P29_SOURCE_FREEZE_TREE",
    "P29_EXPECTED_SOURCE_BUNDLE_SHA256",
    "P29_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT",
    "P29_EXPECTED_SOURCE_RECEIPT_SHA256",
    "P29_EXPECTED_CONTRACT_SHA256",
    "P29_EXPECTED_ORCHESTRATOR_SHA256",
    "P29_EXPECTED_RECONSTRUCTOR_SHA256",
    "P29_EXPECTED_BUNDLE_VERIFIER_SHA256",
    "P29_EXPECTED_P28_CONTRACT_SHA256",
    "P29_EXPECTED_P28_RECONSTRUCTOR_SHA256",
    "P29_EXPECTED_P28_OUTCOME_SHA256",
    "P29_EXPECTED_P28_OUTCOME_RECONSTRUCTOR_SHA256",
    "P29_EXPECTED_P27_CONTRACT_SHA256",
    "P29_EXPECTED_P27_RECONSTRUCTOR_SHA256",
    "P29_EXPECTED_P27_LOCALIZER_SHA256",
    "P29_EXPECTED_INGESTER_SHA256",
    "P29_EXPECTED_P23_CORE_SHA256",
    "P29_EXPECTED_P27_SANITIZER_SHA256",
    "P29_RUNTIME_REVIEW_COMMIT",
    "P29_RUNTIME_REVIEW_TREE",
    "P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256",
    "P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT",
    "P29_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256",
    "P29_EXPECTED_RUNTIME_LOCK_SHA256",
    "P29_EXPECTED_HOST_ATTESTATION_SHA256",
    "P29_EXPECTED_CONTAINER_ID",
]


class DuplicateKeyError(ValueError):
    """Raised when purported canonical JSON repeats a key."""


def _strict_json(data: bytes) -> object:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    def nonfinite(token: str) -> object:
        raise ValueError(f"nonfinite JSON constant: {token}")

    return json.loads(data, object_pairs_hook=pairs, parse_constant=nonfinite)


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _mapping(value: object) -> Mapping[str, object]:
    return value if isinstance(value, Mapping) else {}


def _isolated_environment(shim_directory: Path | None = None) -> dict[str, str]:
    path = "/usr/bin:/bin"
    if shim_directory is not None:
        path = f"{shim_directory}:{path}"
    return {
        "GIT_ATTR_NOSYSTEM": "1",
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_NO_REPLACE_OBJECTS": "1",
        "GIT_OPTIONAL_LOCKS": "0",
        "GIT_PAGER": "cat",
        "GIT_TERMINAL_PROMPT": "0",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": path,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONSAFEPATH": "1",
    }


def _write_private_shim(path: Path, body: str) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o700)
    try:
        os.fchmod(descriptor, 0o700)
        with os.fdopen(descriptor, "wb", closefd=False) as stream:
            stream.write(body.encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
    finally:
        os.close(descriptor)
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o700:
        raise OSError(f"unsafe private shim: {path}")


def _resolved_interpreter() -> Path:
    if not sys.executable:
        raise OSError("Python did not report its executable")
    interpreter = Path(sys.executable).resolve(strict=True)
    if not stat.S_ISREG(interpreter.stat().st_mode):
        raise OSError("resolved Python interpreter is not a regular file")
    return interpreter


@cache
def _local_git_trust_is_closed() -> bool:
    command = [str(TRUSTED_GIT), *TRUSTED_GIT_OPTIONS]
    options = {
        "cwd": ROOT,
        "env": _isolated_environment(),
        "stdin": subprocess.DEVNULL,
        "capture_output": True,
        "check": False,
    }
    try:
        config = subprocess.run([*command, "config", "--local", "--name-only", "--list"], **options)
        if config.returncode != 0:
            return False
        keys = config.stdout.decode("utf-8", errors="strict").splitlines()
        if any(key.lower().startswith(("include.", "includeif.")) for key in keys):
            return False
        git_dir_process = subprocess.run([*command, "rev-parse", "--absolute-git-dir"], **options)
        if git_dir_process.returncode != 0:
            return False
        git_dir = Path(git_dir_process.stdout.decode("utf-8", errors="strict").strip())
        if any(
            (git_dir / relative).exists()
            for relative in ("shallow", "objects/info/alternates", "info/grafts")
        ):
            return False
        replacements = subprocess.run(
            [*command, "for-each-ref", "--format=%(refname)", "refs/replace"], **options
        )
        return replacements.returncode == 0 and not replacements.stdout.strip()
    except (OSError, UnicodeError):
        return False


@cache
def _git(*arguments: str) -> bytes | None:
    try:
        return subprocess.run(
            [str(TRUSTED_GIT), *TRUSTED_GIT_OPTIONS, *arguments],
            cwd=ROOT,
            check=True,
            capture_output=True,
            env=_isolated_environment(),
            stdin=subprocess.DEVNULL,
        ).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def _git_text(*arguments: str) -> str | None:
    value = _git(*arguments)
    return value.decode().strip() if value is not None else None


def _file_hashes_exact(authorities: Mapping[str, object]) -> bool:
    for item in authorities.values():
        entry = _mapping(item)
        path_value = entry.get("path")
        expected = entry.get("sha256")
        if not isinstance(path_value, str) or not isinstance(expected, str):
            continue
        if SHA256_RE.fullmatch(expected) is None:
            return False
        try:
            actual = _sha256((ROOT / path_value).read_bytes())
        except OSError:
            return False
        if actual != expected:
            return False
    return True


def _run_reconstruction(path: str) -> tuple[bool, int, int, int]:
    try:
        interpreter = _resolved_interpreter()
        with tempfile.TemporaryDirectory(prefix="p29-reconstruction-shims-") as directory:
            shim_directory = Path(directory)
            shim_directory.chmod(0o700)
            _write_private_shim(
                shim_directory / "python3",
                f'#!/bin/sh\nexec {shlex.quote(str(interpreter))} -I -S "$@"\n',
            )
            _write_private_shim(
                shim_directory / "git",
                "#!/bin/sh\n"
                f"exec {shlex.quote(str(TRUSTED_GIT))} "
                "-c core.hooksPath=/dev/null -c core.fsmonitor=false "
                '-c protocol.file.allow=never "$@"\n',
            )
            process = subprocess.run(
                [str(interpreter), "-I", "-S", path],
                cwd=ROOT,
                check=False,
                capture_output=True,
                env=_isolated_environment(shim_directory),
                stdin=subprocess.DEVNULL,
            )
        payload = _mapping(_strict_json(process.stdout))
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        return False, 0, 0, 0
    checks = _mapping(payload.get("checks"))
    true_count = sum(value is True for value in checks.values())
    false_count = sum(value is False for value in checks.values())
    return (
        process.returncode == 0 and payload.get("internally_consistent") is True,
        len(checks),
        true_count,
        false_count,
    )


def reconstruct(contract_path: Path = DEFAULT_CONTRACT) -> dict[str, object]:
    """Rebuild repository facts and fail closed on any contract-byte change."""

    try:
        contract_bytes = contract_path.read_bytes()
        parsed = _strict_json(contract_bytes)
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError, DuplicateKeyError):
        contract_bytes = b""
        parsed = None
    contract = _mapping(parsed)
    terminal = _mapping(contract.get("terminal_parent"))
    p28_authorities = _mapping(contract.get("unchanged_p28_authorities"))
    p27_authorities = _mapping(contract.get("unchanged_p27_authorities"))
    layout = _mapping(contract.get("permission_safe_layout"))
    closure = _mapping(contract.get("bundle_closure"))
    verifier = _mapping(contract.get("bundle_verifier"))
    orchestration = _mapping(contract.get("host_orchestration"))
    attempt = _mapping(contract.get("fresh_localization_attempt"))
    execution = _mapping(contract.get("immutable_execution_snapshot"))
    runtime = _mapping(contract.get("locked_base_runtime"))
    freeze = _mapping(contract.get("two_phase_freeze"))
    boundary = _mapping(contract.get("no_training_boundary"))
    routing = _mapping(contract.get("terminal_routing"))

    required_refs = contract.get("required_bundle_refs")
    ref_entries = required_refs if isinstance(required_refs, list) else []
    ref_names = [_mapping(entry).get("ref") for entry in ref_entries]
    p28_blob = _git("show", f"{P28_OUTCOME_COMMIT}:{P28_OUTCOME_PATH}")
    p28_contract_ok, p28_contract_count, p28_contract_true, p28_contract_false = (
        _run_reconstruction("scripts/reconstruct_p28_bundle_complete_localization_bridge.py")
    )
    p28_ok, p28_count, p28_true, p28_false = _run_reconstruction(
        "scripts/reconstruct_p28_bundle_complete_localization_bridge_outcome.py"
    )
    p27_ok, p27_count, p27_true, p27_false = _run_reconstruction(
        "scripts/reconstruct_p27_cuda_deleted_mapping_localization.py"
    )

    prepare_journal = [
        ".deferred-p29-prepare-attempt-layout.exit-status.txt",
        ".deferred-p29-prepare-attempt-layout.publication.exit-status.txt",
        ".deferred-p29-prepare-attempt-layout.publication.stderr.log",
        ".deferred-p29-prepare-attempt-layout.publication.stdout.log",
        ".deferred-p29-prepare-attempt-layout.stderr.log",
        ".deferred-p29-prepare-attempt-layout.stdout.log",
        ".deferred-p29-prepare-pre-attempt-admission.exit-status.txt",
        ".deferred-p29-prepare-pre-attempt-admission.stderr.log",
        ".deferred-p29-prepare-pre-attempt-admission.stdout.log",
    ]
    prepare_admission_journal = [
        ".deferred-p29-prepare-pre-attempt-admission.exit-status.txt",
        ".deferred-p29-prepare-pre-attempt-admission.stderr.log",
        ".deferred-p29-prepare-pre-attempt-admission.stdout.log",
    ]
    runtime_replay_journal = [
        ".deferred-p29-pre-marker-runtime-receipt-replay.exit-status.txt",
        ".deferred-p29-pre-marker-runtime-receipt-replay.publication.exit-status.txt",
        ".deferred-p29-pre-marker-runtime-receipt-replay.publication.stderr.log",
        ".deferred-p29-pre-marker-runtime-receipt-replay.publication.stdout.log",
        ".deferred-p29-pre-marker-runtime-receipt-replay.stderr.log",
        ".deferred-p29-pre-marker-runtime-receipt-replay.stdout.log",
        *prepare_journal,
    ]

    verifier_hashes_final = (
        verifier.get("placeholders_must_be_replaced_before_source_freeze") is False
        and isinstance(verifier.get("source_sha256"), str)
        and SHA256_RE.fullmatch(str(verifier.get("source_sha256"))) is not None
        and isinstance(verifier.get("test_sha256"), str)
        and SHA256_RE.fullmatch(str(verifier.get("test_sha256"))) is not None
        and _file_hashes_exact(
            {
                "source": {
                    "path": verifier.get("source_path"),
                    "sha256": verifier.get("source_sha256"),
                },
                "test": {
                    "path": verifier.get("test_path"),
                    "sha256": verifier.get("test_sha256"),
                },
            }
        )
    )
    orchestrator_hashes_final = (
        orchestration.get("placeholders_must_be_replaced_before_source_freeze") is False
        and isinstance(orchestration.get("source_sha256"), str)
        and SHA256_RE.fullmatch(str(orchestration.get("source_sha256"))) is not None
        and isinstance(orchestration.get("test_sha256"), str)
        and SHA256_RE.fullmatch(str(orchestration.get("test_sha256"))) is not None
        and _file_hashes_exact(
            {
                "source": {
                    "path": orchestration.get("source_path"),
                    "sha256": orchestration.get("source_sha256"),
                },
                "test": {
                    "path": orchestration.get("test_path"),
                    "sha256": orchestration.get("test_sha256"),
                },
            }
        )
    )

    expected_execution_tools = {
        "run_p27_cuda_deleted_mapping_localization.py": {
            "sha256": "86ba200d69a029c64b2218581d6b92245930d5f1e60d0176aca22ad4d6ddf65a",
            "mode": "0555",
        },
        "ingest_p26_trace_off_a_failure.py": {
            "sha256": "3e7769d668a8756aea7359bef4d9b8ccfc8db66da102edfbc6c5de385654cdeb",
            "mode": "0555",
        },
        "p23_deterministic_cuda_shadow_trace.py": {
            "sha256": "ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab",
            "mode": "0444",
        },
        "sanitize_p27_cuda_deleted_mapping_localization.py": {
            "sha256": "3ec64cdc3681d41d70c624c6ffd26dde1a7e498bda13ddafaaced271a43dd6c4",
            "mode": "0555",
        },
        "p26-trace-off-a-failure.authenticated.json": {
            "sha256": "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91",
            "byte_count": 66283,
            "mode": "0444",
        },
    }
    expected_manifest_source_authorities = {
        "repository": {
            "commit": "185e444afc0b44ca0a09b1bde49a6b6fa3973355",
            "tree": "24f4bdac331a57bd7c1b807747d7c7fba253ee5a",
        },
        "nanogpt": {
            "commit": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
            "tree": "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a",
        },
        "muon": {
            "commit": "f98f1cacc0263b04290753e32be8d498c1efc806",
            "tree": "4ea5cd8ab6ebd56a18536f06453619efcd636da0",
        },
        "data": {
            "original_manifest_relative_path": "materialized/p22_fineweb_manifest.json",
            "original_manifest_sha256": (
                "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d"
            ),
            "original_manifest_byte_count": 4210,
        },
        "helpers": {
            "run_p27_cuda_deleted_mapping_localization.py": (
                "86ba200d69a029c64b2218581d6b92245930d5f1e60d0176aca22ad4d6ddf65a"
            ),
            "ingest_p26_trace_off_a_failure.py": (
                "3e7769d668a8756aea7359bef4d9b8ccfc8db66da102edfbc6c5de385654cdeb"
            ),
            "p23_deterministic_cuda_shadow_trace.py": (
                "ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab"
            ),
            "sanitize_p27_cuda_deleted_mapping_localization.py": (
                "3ec64cdc3681d41d70c624c6ffd26dde1a7e498bda13ddafaaced271a43dd6c4"
            ),
        },
        "p26_authenticated_input": {
            "sha256": "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91",
            "byte_count": 66283,
        },
    }
    expected_mounts = [
        {
            "source": "/secure/p29/execution-20260906-02/repository",
            "destination": "/workspace/OptimizationML",
            "read_only": True,
        },
        {
            "source": "/secure/p29/execution-20260906-02/nanogpt",
            "destination": "/workspace/inputs/nanoGPT",
            "read_only": True,
        },
        {
            "source": "/secure/p29/execution-20260906-02/muon",
            "destination": "/workspace/inputs/muon",
            "read_only": True,
        },
        {
            "source": "/secure/p29/execution-20260906-02/data",
            "destination": "/private/tmp/optimizationml-p22-data",
            "read_only": True,
        },
        {
            "source": (
                "/secure/p29/execution-20260906-02/repository/experiments/training/"
                "materialize_p22_fineweb.py"
            ),
            "destination": (
                "/Users/harry/Desktop/temp/OptimizationML/experiments/training/"
                "materialize_p22_fineweb.py"
            ),
            "read_only": True,
        },
        {
            "source": "/secure/p29/attempt-20260906-02/evidence",
            "destination": "/workspace/evidence/p23",
            "read_only": False,
        },
        {
            "source": "/secure/p29/attempt-20260906-02/p29-image-inspect.json",
            "destination": "/mounted-host-evidence/image-inspect.json",
            "read_only": True,
        },
        {
            "source": ("/secure/p29/attempt-20260906-02/p29-running-container-inspect.json"),
            "destination": "/mounted-host-evidence/running-container-inspect.json",
            "read_only": True,
        },
        {
            "source": "/secure/p29/attempt-20260906-02/p29-running-mountinfo.txt",
            "destination": "/mounted-host-evidence/running-mountinfo.txt",
            "read_only": True,
        },
        {
            "source": "/secure/p29/attempt-20260906-02/p29-nvidia-smi.csv",
            "destination": "/mounted-host-evidence/nvidia-smi.csv",
            "read_only": True,
        },
    ]

    checks = {
        "canonical_contract_bytes_schema_status_and_closed_top_level": (
            isinstance(parsed, Mapping)
            and list(parsed) == EXPECTED_TOP_LEVEL_ORDER
            and contract_bytes == (json.dumps(parsed, indent=2) + "\n").encode()
            and _sha256(contract_bytes) == EXPECTED_CONTRACT_SHA256
            and contract.get("schema_version") == EXPECTED_SCHEMA
            and contract.get("status") == EXPECTED_STATUS
            and contract.get("branch") == EXPECTED_BRANCH
            and _local_git_trust_is_closed()
        ),
        "terminal_p28_commit_tree_tag_outcome_and_permission_stop_exact": (
            _git_text("rev-parse", f"{P28_OUTCOME_COMMIT}^{{tree}}") == P28_OUTCOME_TREE
            and _git_text("rev-parse", "refs/heads/p28-bundle-complete-localization-bridge")
            == P28_OUTCOME_COMMIT
            and _git_text("cat-file", "-t", P28_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", "refs/tags/p28-control-parent-permission-diagnostic")
            == P28_TAG_OBJECT
            and _git_text("rev-parse", "refs/tags/p28-control-parent-permission-diagnostic^{}")
            == P28_OUTCOME_COMMIT
            and p28_blob is not None
            and _sha256(p28_blob) == P28_OUTCOME_SHA256
            and terminal.get("outcome_commit") == P28_OUTCOME_COMMIT
            and terminal.get("outcome_tree") == P28_OUTCOME_TREE
            and terminal.get("attempt_is_terminal") is True
            and terminal.get("attempt_reuse_allowed") is False
            and terminal.get("source_verifier_invocations_run") == 1
            and terminal.get("source_receipt_created") is False
            and terminal.get("control_checkout_created") is False
            and terminal.get("attempt_root_created") is False
            and terminal.get("container_created") is False
            and terminal.get("failed_operation")
            == "git init --initial-branch=p28-verifier-unborn /secure/p28/control-source"
            and terminal.get("exact_terminal_stderr")
            == (
                "P28 control-bundle verification failed: Git command failed (128): init "
                "--initial-branch=p28-verifier-unborn /secure/p28/control-source: fatal: "
                "cannot mkdir /secure/p28/control-source: Permission denied"
            )
            and isinstance(terminal.get("exact_terminal_stderr"), str)
            and _sha256((str(terminal.get("exact_terminal_stderr")) + "\n").encode())
            == terminal.get("exact_terminal_stderr_sha256")
            and terminal.get("retained_source_closure_created") is True
            and terminal.get("retained_transport_or_closure_reuse_allowed") is False
            and terminal.get("root_cause") == "unprivileged_control_checkout_parent_permission"
        ),
        "unchanged_p28_and_p27_source_hashes_exact": (
            len(p28_authorities) == 6
            and len(p27_authorities) == 10
            and _file_hashes_exact(p28_authorities)
            and _file_hashes_exact(p27_authorities)
        ),
        "p26_and_p27_annotated_tag_objects_and_peels_exact": (
            _git_text("rev-parse", "refs/heads/p27-cuda-deleted-mapping-localization")
            == P27_OUTCOME_COMMIT
            and _git_text("cat-file", "-t", P27_TAG_OBJECT) == "tag"
            and _git_text("rev-parse", "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic")
            == P27_TAG_OBJECT
            and _git_text(
                "rev-parse", "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic^{}"
            )
            == P27_OUTCOME_COMMIT
            and _git_text("cat-file", "-t", P26_CHECKPOINT_OBJECT) == "tag"
            and _git_text("rev-parse", "refs/tags/p26-permission-safe-acquisition-checkpoint")
            == P26_CHECKPOINT_OBJECT
            and _git_text("rev-parse", "refs/tags/p26-permission-safe-acquisition-checkpoint^{}")
            == P26_CHECKPOINT_COMMIT
            and _git_text("cat-file", "-t", P26_SOURCE_OBJECT) == "tag"
            and _git_text("rev-parse", "refs/tags/p26-attempt02-acquisition-source")
            == P26_SOURCE_OBJECT
            and _git_text("rev-parse", "refs/tags/p26-attempt02-acquisition-source^{}")
            == P26_SOURCE_COMMIT
        ),
        "exact_seven_required_bundle_refs_objects_and_peels": (
            ref_names == EXPECTED_REQUIRED_REFS
            and len(ref_entries) == 7
            and _mapping(ref_entries[1]).get("object") == P28_OUTCOME_COMMIT
            and _mapping(ref_entries[2]).get("object") == P28_TAG_OBJECT
            and _mapping(ref_entries[2]).get("peeled_commit") == P28_OUTCOME_COMMIT
            and _mapping(ref_entries[3]).get("object") == P27_OUTCOME_COMMIT
            and _mapping(ref_entries[4]).get("object") == P27_TAG_OBJECT
            and _mapping(ref_entries[5]).get("object") == P26_CHECKPOINT_OBJECT
            and _mapping(ref_entries[6]).get("object") == P26_SOURCE_OBJECT
        ),
        "permission_safe_root_transport_control_authority_layout_exact": (
            layout.get("secure_parent") == "/secure"
            and layout.get("secure_parent_owner") == "root:root"
            and layout.get("secure_parent_uid") == 0
            and layout.get("secure_parent_gid") == 0
            and layout.get("secure_parent_mode") == "0755"
            and layout.get("secure_parent_process_writable") is False
            and layout.get("secure_root") == "/secure/p29"
            and layout.get("secure_root_owner") == "root:root"
            and layout.get("secure_root_uid") == 0
            and layout.get("secure_root_gid") == 0
            and layout.get("secure_root_mode") == "0755"
            and layout.get("secure_root_process_writable") is False
            and layout.get("transport_root") == "/secure/p29/transport-20260906-02"
            and layout.get("transport_owner") == "ubuntu:ubuntu"
            and layout.get("transport_uid") == 1000
            and layout.get("transport_gid") == 1000
            and layout.get("transport_mode") == "0700"
            and layout.get("transport_is_only_precreated_process_writable_directory") is True
            and layout.get("localization_ledger_root") == "/var/lib/optimizationml-p29-20260906-02"
            and layout.get("localization_ledger_root_owner") == "root:root"
            and layout.get("localization_ledger_root_uid") == 0
            and layout.get("localization_ledger_root_gid") == 0
            and layout.get("localization_ledger_root_mode") == "0700"
            and layout.get(
                "localization_ledger_root_is_provisioned_with_only_deferred_journal_before_source_phase"
            )
            is True
            and layout.get(
                "localization_ledger_root_is_not_a_checkout_mount_or_transfer_capability"
            )
            is True
            and layout.get(
                "localization_ledger_root_is_componentwise_nofollow_with_mode_only_access_acl_and_no_default_named_or_frozen_xattr_acl"
            )
            is True
            and layout.get("deferred_journal_root")
            == "/var/lib/optimizationml-p29-20260906-02/deferred"
            and layout.get("deferred_journal_root_owner") == "root:root"
            and layout.get("deferred_journal_root_mode") == "0700"
            and layout.get("deferred_journal_is_not_accessible_to_the_uid1000_bundle_verifier")
            is True
            and layout.get(
                "deferred_journal_files_are_root_owned_o_excl_nofollow_and_preregistered"
            )
            is True
            and layout.get("deferred_journal_file_mode") == "0400"
            and layout.get(
                "deferred_publication_best_effort_attempts_to_retain_stdout_stderr_and_exit_status"
            )
            is True
            and layout.get(
                "deferred_publication_complete_triplet_is_guaranteed_only_when_artifact_finalization_succeeds"
            )
            is True
            and layout.get("deferred_publication_success_stdout_stderr_exact_bytes_utf8") == ""
            and layout.get("deferred_publication_success_exit_status_exact_bytes_utf8") == "0\n"
            and layout.get("deferred_publication_failure_sentinel_suffix")
            == ".publication-failure.exit-status.txt"
            and layout.get("deferred_publication_failure_sentinel_exists_only_on_failure") is True
            and layout.get("deferred_journal_exact_before_prepare_admission") == []
            and layout.get("deferred_journal_exact_after_successful_prepare_pre_attempt_admission")
            == prepare_admission_journal
            and layout.get(
                "after_admission_journal_validation_accepts_only_the_exact_producer_status_argument"
            )
            is True
            and layout.get(
                "failed_prepare_admission_retains_and_returns_its_original_status_when_finalization_succeeds"
            )
            is True
            and layout.get(
                "after_prepare_and_after_runtime_journal_stages_require_prepare_admission_status_zero"
            )
            is True
            and layout.get("deferred_journal_exact_after_successful_prepare_layout_publication")
            == prepare_journal
            and layout.get(
                "deferred_journal_exact_after_successful_runtime_receipt_replay_publication"
            )
            == runtime_replay_journal
            and layout.get("sealed_orchestrator")
            == (
                "/var/lib/optimizationml-p29-20260906-02/"
                "run_p29_permission_safe_bundle_localization_bridge.sh"
            )
            and layout.get("sealed_orchestrator_owner") == "root:root"
            and layout.get("sealed_orchestrator_mode") == "0555"
            and layout.get(
                "sealed_orchestrator_is_o_excl_nofollow_stable_copy_of_reviewed_control_bytes"
            )
            is True
            and layout.get(
                "sealed_orchestrator_is_created_only_after_source_receipt_and_control_checkout_review"
            )
            is True
            and layout.get(
                "prepare_and_localization_invoke_only_sealed_orchestrator_with_bin_bash_p"
            )
            is True
            and layout.get("localization_ledger_root_exact_before_prepare_inventory")
            == ["deferred", "run_p29_permission_safe_bundle_localization_bridge.sh"]
            and layout.get("prepare_runtime_invocation_token")
            == "/var/lib/optimizationml-p29-20260906-02/prepare-runtime.invoked"
            and layout.get("prepare_runtime_invocation_token_owner") == "root:root"
            and layout.get("prepare_runtime_invocation_token_mode") == "0400"
            and layout.get("prepare_runtime_invocation_token_exact_bytes_utf8")
            == "prepare-runtime\n"
            and layout.get("prepare_runtime_invocation_token_is_o_excl_nofollow") is True
            and layout.get(
                "prepare_runtime_invocation_token_transaction_uses_supplied_expected_orchestrator_digest_before_general_environment_validation_or_source_receipt_replay"
            )
            is True
            and layout.get(
                "prepare_pre_attempt_admission_is_one_root_journal_triplet_covering_source_receipt_replay_and_every_pre_attempt_validator_after_a_successful_prepare_token_transaction"
            )
            is True
            and layout.get(
                "partial_internal_prepare_token_transaction_leaves_explicit_incomplete_nonrunnable_state"
            )
            is True
            and layout.get(
                "localization_ledger_root_exact_after_prepare_dispatch_before_localization_inventory"
            )
            == [
                "deferred",
                "prepare-runtime.invoked",
                "run_p29_permission_safe_bundle_localization_bridge.sh",
            ]
            and layout.get(
                "localization_ledger_root_exact_during_source_replay_and_runtime_receipt_creation_inventory"
            )
            == [
                "deferred",
                "prepare-runtime.invoked",
                "run_p29_permission_safe_bundle_localization_bridge.sh",
            ]
            and layout.get("localization_invocation_token")
            == "/var/lib/optimizationml-p29-20260906-02/run-localization.invoked"
            and layout.get("localization_invocation_token_owner") == "root:root"
            and layout.get("localization_invocation_token_mode") == "0400"
            and layout.get("localization_invocation_token_exact_bytes_utf8") == "run-localization\n"
            and layout.get("localization_authorization_token")
            == "/var/lib/optimizationml-p29-20260906-02/localization.authorized"
            and layout.get("localization_authorization_token_owner") == "root:root"
            and layout.get("localization_authorization_token_mode") == "0400"
            and layout.get("localization_authorization_token_exact_bytes_utf8")
            == "localization-authorized\n"
            and layout.get("localization_terminal_status")
            == "/var/lib/optimizationml-p29-20260906-02/run-localization.exit-status.txt"
            and layout.get("localization_terminal_status_owner") == "root:root"
            and layout.get("localization_terminal_status_mode") == "0400"
            and layout.get("localization_terminal_status_exact_bytes_rule")
            == "one canonical ASCII decimal integer from 0 through 255 followed by newline"
            and layout.get("localization_terminal_status_is_o_excl_nofollow") is True
            and layout.get(
                "localization_ledger_root_exact_after_dispatch_before_authorization_inventory"
            )
            == [
                "deferred",
                "prepare-runtime.invoked",
                "run-localization.invoked",
                "run_p29_permission_safe_bundle_localization_bridge.sh",
            ]
            and layout.get("localization_ledger_root_exact_authorized_inventory")
            == [
                "deferred",
                "localization.authorized",
                "prepare-runtime.invoked",
                "run-localization.invoked",
                "run_p29_permission_safe_bundle_localization_bridge.sh",
            ]
            and layout.get(
                "localization_ledger_root_exact_after_successful_terminal_status_preauthorization_failure_inventory"
            )
            == [
                "deferred",
                "prepare-runtime.invoked",
                "run-localization.exit-status.txt",
                "run-localization.invoked",
                "run_p29_permission_safe_bundle_localization_bridge.sh",
            ]
            and layout.get(
                "localization_ledger_root_exact_after_successful_terminal_status_authorized_inventory"
            )
            == [
                "deferred",
                "localization.authorized",
                "prepare-runtime.invoked",
                "run-localization.exit-status.txt",
                "run-localization.invoked",
                "run_p29_permission_safe_bundle_localization_bridge.sh",
            ]
            and layout.get(
                "source_receipt_creation_operator_precondition_is_only_empty_deferred_journal_in_root_ledger"
            )
            is True
            and layout.get(
                "source_receipt_creation_uid1000_verifier_mechanically_inspects_root_ledger"
            )
            is False
            and layout.get(
                "source_receipt_replay_requires_prepare_invocation_present_and_localization_files_absent"
            )
            is True
            and layout.get(
                "runtime_receipt_creation_operator_precondition_is_prepare_invocation_present_and_localization_files_absent"
            )
            is True
            and layout.get(
                "runtime_receipt_creation_uid1000_verifier_mechanically_inspects_root_ledger"
            )
            is False
            and layout.get(
                "runtime_receipt_replay_requires_prepare_and_localization_invocations_present_and_authorization_and_status_absent"
            )
            is True
            and layout.get(
                "authorization_is_created_once_only_after_reviewed_runtime_receipt_and_all_pre_marker_checks"
            )
            is True
            and layout.get(
                "host_post_authorization_boundary_validates_prepare_invocation_localization_invocation_and_authorization_present_and_exact_before_dispatch"
            )
            is True
            and layout.get("containerized_p27_localizer_reads_root_ledger") is False
            and layout.get(
                "each_localization_token_creation_atomically_revalidates_deferred_root_and_sealed_orchestrator_before_o_excl"
            )
            is True
            and layout.get(
                "terminal_status_creation_atomically_revalidates_deferred_root_and_sealed_orchestrator_before_o_excl"
            )
            is True
            and layout.get(
                "token_creation_requires_deferred_root_root_root_0700_no_acl_and_stable_identity"
            )
            is True
            and layout.get(
                "token_creation_requires_sealed_orchestrator_root_root_0555_no_acl_stable_identity_and_reviewed_digest"
            )
            is True
            and layout.get(
                "after_successful_invocation_token_transaction_dispatcher_attempts_one_terminal_status_transaction_even_if_evidence_logging_or_early_predicates_fail"
            )
            is True
            and layout.get(
                "complete_validated_terminal_status_is_guaranteed_only_when_terminal_status_transaction_succeeds"
            )
            is True
            and layout.get(
                "failed_terminal_status_transaction_is_terminal_nonrunnable_and_may_leave_status_absent_or_created_but_unvalidated"
            )
            is True
            and layout.get("second_authorization_creation_or_localization_rerun_allowed") is False
            and layout.get("source_phase_exact_secure_root_child_inventory")
            == ["transport-20260906-02"]
            and layout.get("runtime_phase_exact_secure_root_child_inventory")
            == [
                "attempt-20260906-02",
                "execution-20260906-02",
                "transport-20260906-02",
            ]
            and layout.get("extra_capability_or_namespace_sibling_allowed") is False
            and layout.get("control_checkout") == "/secure/p29/transport-20260906-02/control-source"
            and layout.get("authority_checkout")
            == "/secure/p29/transport-20260906-02/authority-185e444"
            and layout.get("control_and_authority_must_be_absent_before_source_receipt_creation")
            is True
            and layout.get("source_receipt_creation_requires_authority_absent_for_entire_creation")
            is True
            and layout.get("source_receipt_replay_authority_states_allowed")
            == ["absent", "exact_canonical_locked_authority"]
            and layout.get("source_receipt_replay_requires_exact_existing_control_checkout") is True
            and layout.get(
                "source_receipt_replay_existing_authority_requires_layout_commit_tree_and_indirection_validation"
            )
            is True
            and layout.get("source_receipt_replay_may_not_create_mutate_or_relocate_authority")
            is True
            and layout.get(
                "attempt_and_execution_roots_must_be_absent_during_source_receipt_creation_and_replay"
            )
            is True
            and layout.get("precreate_empty_control_or_authority_directory_allowed") is False
            and layout.get("bind_copy_symlink_or_privileged_relocation_allowed") is False
            and layout.get("control_and_authority_post_creation_owner") == "ubuntu:ubuntu"
            and layout.get("control_and_authority_post_creation_uid") == 1000
            and layout.get("control_and_authority_post_creation_gid") == 1000
            and layout.get("control_and_authority_post_creation_mode") == "0700"
            and layout.get(
                "control_and_authority_are_nonsymlink_directories_on_transport_device_and_mount_id"
            )
            is True
            and layout.get(
                "control_and_authority_have_no_extended_or_named_access_no_default_and_no_frozen_xattr_acl"
            )
            is True
            and layout.get("componentwise_o_directory_o_nofollow_required") is True
            and layout.get("ancestor_nofollow_components_with_stable_device_and_inode_only")
            == ["/"]
            and layout.get("full_owner_mode_acl_mount_authority_boundary_begins_at") == "/secure"
            and layout.get("ancestor_owner_mode_acl_or_mount_authority_claimed") is False
            and layout.get("access_acl_must_equal_mode_bits_without_named_entries") is True
            and layout.get("default_acl_entries_allowed") is False
            and layout.get("forbidden_acl_xattrs")
            == [
                "system.posix_acl_access",
                "system.posix_acl_default",
                "system.nfs4_acl",
                "system.richacl",
            ]
            and layout.get(
                "secure_parent_namespace_transport_execution_attempt_and_evidence_must_match_expected_identity_authority_acl_device_and_mount_at_each_explicit_validator"
            )
            is True
            and layout.get(
                "secure_parent_namespace_transport_execution_attempt_and_evidence_must_not_be_mountpoints"
            )
            is True
            and layout.get(
                "secure_parent_namespace_transport_execution_attempt_and_evidence_must_share_device_and_mount_id"
            )
            is True
            and layout.get(
                "layout_identity_revalidated_before_and_after_each_bundle_phase_transition"
            )
            is True
            and layout.get("cleanup_repair_or_retry_under_same_identifier_allowed") is False
        ),
        "two_fresh_closures_external_receipts_and_reconstructions_exact": (
            closure.get("transport_root") == layout.get("transport_root")
            and closure.get("control_checkout") == layout.get("control_checkout")
            and closure.get("authority_checkout") == layout.get("authority_checkout")
            and closure.get("runtime_review_receipt_creation_transport_only_transcripts")
            == [
                "p29-runtime-review-receipt-creation.stdout.log",
                "p29-runtime-review-receipt-creation.stderr.log",
                "p29-runtime-review-receipt-creation.exit-status.txt",
            ]
            and closure.get(
                "runtime_review_receipt_creation_transcripts_are_o_excl_nofollow_and_retain_verifier_success_or_failure_when_wrapper_setup_and_finalization_succeed"
            )
            is True
            and closure.get(
                "runtime_review_receipt_creation_wrapper_failure_may_leave_partial_transport_transcripts_and_is_terminal"
            )
            is True
            and closure.get(
                "runtime_review_receipt_creation_transcripts_may_be_published_under_evidence"
            )
            is False
            and closure.get(
                "runtime_review_receipt_creation_preserves_exact_27_file_evidence_for_later_replay"
            )
            is True
            and closure.get("receipt_schema") == "passive-muon-p29-control-bundle-receipt-v1"
            and closure.get("fresh_empty_bare_repository_required_for_each_phase") is True
            and closure.get("exact_advertised_ref_count") == 7
            and closure.get("bundle_prerequisite_count") == 0
            and closure.get("receipt_files_are_external_and_not_committed") is True
            and closure.get("receipt_outputs_are_o_excl_no_overwrite") is True
            and closure.get("private_copy_required_mode") == "0400"
            and closure.get(
                "source_receipt_required_before_authority_attempt_or_execution_root_creation"
            )
            is True
            and closure.get("p28_contract_reconstruction_required_check_count") == 12
            and closure.get("p28_contract_reconstruction_required_true_count") == 12
            and closure.get("p28_contract_reconstruction_required_false_count") == 0
            and closure.get("p28_outcome_reconstruction_required_check_count") == 10
            and closure.get("p27_reconstruction_required_check_count") == 23
        ),
        "bundle_verifier_final_hashes_layout_and_phases_exact": (
            verifier_hashes_final
            and verifier.get("source_path") == "scripts/verify_p29_control_bundle.py"
            and verifier.get("required_process_uid") == 1000
            and verifier.get("required_process_gid") == 1000
            and verifier.get(
                "uid1000_ownership_is_required_for_transport_closure_control_and_receipt_mutations"
            )
            is True
            and verifier.get("bootstrap_execution_path")
            == "/secure/p29/transport-20260906-02/verify_p29_control_bundle.py"
            and verifier.get("checked_in_control_verifier_path")
            == (
                "/secure/p29/transport-20260906-02/control-source/"
                "scripts/verify_p29_control_bundle.py"
            )
            and verifier.get("trusted_python_executable") == "/usr/bin/python3"
            and verifier.get("trusted_python_flags") == ["-I", "-S"]
            and verifier.get("bootstrap_and_all_reconstructors_use_trusted_python_with_both_flags")
            is True
            and verifier.get(
                "reconstructors_execute_only_from_the_verifiers_authenticated_private_bundle_bytes"
            )
            is True
            and verifier.get(
                "verifier_is_the_sole_host_reconstruction_authority_in_source_and_runtime_receipts"
            )
            is True
            and verifier.get("nested_python3_shim_executes_the_same_trusted_python_with_both_flags")
            is True
            and verifier.get(
                "nested_reconstruction_environment_scrubs_python_and_git_control_inputs"
            )
            is True
            and verifier.get("absolute_git_executable") == "/usr/bin/git"
            and verifier.get("git_system_attributes_disabled_with_git_attr_nosystem") is True
            and verifier.get(
                "git_system_global_includes_hooks_replacements_and_ambient_control_are_disabled"
            )
            is True
            and verifier.get(
                "control_and_authority_cleanliness_includes_tracked_untracked_and_ignored_files"
            )
            is True
            and verifier.get("allowed_phases") == ["source", "runtime-review"]
            and verifier.get("phase_receipt_statuses")
            == {
                "source": (
                    "source_bundle_and_permission_layout_verified_before_attempt_root_creation"
                ),
                "runtime-review": (
                    "runtime_review_bundle_and_permission_layout_verified_before_localization"
                ),
            }
            and verifier.get("receipt_binds_permission_layout_identity_and_acl") is True
            and verifier.get("source_phase_exact_namespace_child_inventory")
            == ["transport-20260906-02"]
            and verifier.get("runtime_review_phase_exact_namespace_child_inventory")
            == [
                "attempt-20260906-02",
                "execution-20260906-02",
                "transport-20260906-02",
            ]
            and verifier.get(
                "phase_specific_namespace_inventory_revalidated_before_and_after_transitions"
            )
            is True
            and verifier.get("source_phase_constructs_absent_control_checkout") is True
            and verifier.get("source_phase_reconstructs_p28_contract_12_of_12") is True
            and verifier.get("source_phase_reconstructs_terminal_p28_10_of_10") is True
            and verifier.get("source_phase_reconstructs_p27_23_of_23") is True
            and verifier.get("source_receipt_creation_requires_authority_absent") is True
            and verifier.get(
                "uid1000_bundle_verifier_does_not_claim_access_to_root_only_ledger_or_deferred_journal"
            )
            is True
            and verifier.get(
                "source_receipt_replay_allows_authority_absent_or_exact_canonical_locked_authority"
            )
            is True
            and verifier.get(
                "source_receipt_replay_existing_authority_must_match_locked_commit_tree_layout_acl_and_indirection_guards"
            )
            is True
            and verifier.get("source_receipt_replay_never_creates_mutates_or_relocates_authority")
            is True
            and verifier.get("runtime_review_exact_attempt_root_inventory")
            == [
                "evidence",
                "p29-image-inspect.json",
                "p29-nvidia-smi.csv",
                "p29-running-container-inspect.json",
                "p29-running-mountinfo.txt",
                "p29-container-id.txt",
            ]
            and verifier.get("runtime_review_required_prepare_layout_transcripts")
            == [
                "p29-prepare-attempt-layout.stdout.log",
                "p29-prepare-attempt-layout.stderr.log",
                "p29-post-freeze-attempt-layout.stdout.log",
                "p29-post-freeze-attempt-layout.stderr.log",
            ]
            and verifier.get("runtime_review_exact_prelocalization_evidence_inventory")
            == [
                "p29-prepare-attempt-layout.stdout.log",
                "p29-prepare-attempt-layout.stderr.log",
                "p29-prepare-attempt-layout.exit-status.txt",
                "p29-prepare-execution-snapshot-verification.stdout.log",
                "p29-prepare-execution-snapshot-verification.stderr.log",
                "p29-prepare-execution-snapshot-verification.exit-status.txt",
                "p29-image-inspection.stderr.log",
                "p29-image-inspection.exit-status.txt",
                "p29-nvidia-smi.stderr.log",
                "p29-nvidia-smi.exit-status.txt",
                "p29-container-launch.stderr.log",
                "p29-container-launch.exit-status.txt",
                "p29-fresh-container-id-validation.stdout.log",
                "p29-fresh-container-id-validation.stderr.log",
                "p29-fresh-container-id-validation.exit-status.txt",
                "p29-running-container-inspection.stderr.log",
                "p29-running-container-inspection.exit-status.txt",
                "p29-running-mountinfo.stderr.log",
                "p29-running-mountinfo.exit-status.txt",
                "p29-freeze-runtime.stdout.log",
                "p29-freeze-runtime.stderr.log",
                "p29-freeze-runtime.exit-status.txt",
                "p29-post-freeze-attempt-layout.stdout.log",
                "p29-post-freeze-attempt-layout.stderr.log",
                "p29-post-freeze-attempt-layout.exit-status.txt",
                "p29_cuda_runtime_lock.json",
                "p29_host_attestation.json",
            ]
            and verifier.get("runtime_review_required_zero_exit_status_files")
            == [
                "p29-prepare-attempt-layout.exit-status.txt",
                "p29-prepare-execution-snapshot-verification.exit-status.txt",
                "p29-image-inspection.exit-status.txt",
                "p29-nvidia-smi.exit-status.txt",
                "p29-container-launch.exit-status.txt",
                "p29-fresh-container-id-validation.exit-status.txt",
                "p29-running-container-inspection.exit-status.txt",
                "p29-running-mountinfo.exit-status.txt",
                "p29-freeze-runtime.exit-status.txt",
                "p29-post-freeze-attempt-layout.exit-status.txt",
            ]
            and verifier.get("runtime_review_success_exit_status_exact_bytes_utf8") == "0\n"
            and verifier.get("runtime_review_phase_exact_delta")
            == [
                "A\texperiments/training/p29_cuda_runtime_lock.json",
                "A\texperiments/training/p29_host_attestation.json",
            ]
            and verifier.get("runtime_review_attempt_and_evidence_are_fd_anchored_nofollow") is True
            and verifier.get(
                "runtime_review_attempt_and_evidence_identity_authority_acl_device_and_mount_are_stable"
            )
            is True
            and verifier.get(
                "runtime_review_execution_snapshot_is_fd_anchored_nofollow_and_manifest_revalidated_before_and_after_receipt"
            )
            is True
            and verifier.get("runtime_review_phase_requires_exact_immutable_execution_snapshot")
            is True
            and verifier.get("runtime_review_attempt_root_owner_mode") == "root:root 0555"
            and verifier.get("runtime_review_evidence_root_owner_mode") == "root:root 0555"
            and verifier.get("runtime_review_evidence_children_owner_mode") == "root:root 0444"
            and verifier.get(
                "runtime_review_evidence_children_are_o_excl_and_sealed_after_producer_close"
            )
            is True
            and verifier.get(
                "runtime_review_evidence_is_world_readable_by_design_and_contains_no_credentials"
            )
            is True
            and verifier.get("runtime_review_execution_root_owner_mode") == "root:root 0555"
            and verifier.get("runtime_review_execution_manifest_owner_mode") == "root:root 0444"
            and verifier.get(
                "attempt_and_execution_roots_must_remain_absent_during_source_receipt_creation_and_replay"
            )
            is True
            and verifier.get("source_receipt_creation_invocations_allowed") == 1
            and verifier.get("source_receipt_replay_invocations_allowed") == 1
            and verifier.get("runtime_review_receipt_creation_invocations_allowed") == 1
            and verifier.get("runtime_review_receipt_replay_invocations_allowed") == 1
        ),
        "host_orchestrator_final_hashes_and_one_shot_order_exact": (
            orchestrator_hashes_final
            and orchestration.get("source_path")
            == "scripts/run_p29_permission_safe_bundle_localization_bridge.sh"
            and orchestration.get("allowed_phases")
            == [
                "prepare-runtime",
                "run-localization",
            ]
            and orchestration.get("required_environment") == EXPECTED_ORCHESTRATOR_ENVIRONMENT
            and orchestration.get("runtime_only_required_environment")
            == [
                "P29_RUNTIME_REVIEW_COMMIT",
                "P29_RUNTIME_REVIEW_TREE",
                "P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256",
                "P29_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT",
                "P29_EXPECTED_RUNTIME_REVIEW_RECEIPT_SHA256",
                "P29_EXPECTED_RUNTIME_LOCK_SHA256",
                "P29_EXPECTED_HOST_ATTESTATION_SHA256",
                "P29_EXPECTED_CONTAINER_ID",
            ]
            and orchestration.get("source_required_environment")
            == EXPECTED_ORCHESTRATOR_ENVIRONMENT[:-8]
            and orchestration.get("root_environment_handoff_executable") == "/usr/bin/env"
            and orchestration.get("root_environment_handoff_clear_environment_flag") == "-i"
            and orchestration.get("root_environment_handoff_exact_path")
            == "/usr/sbin:/usr/bin:/sbin:/bin"
            and orchestration.get(
                "prepare_root_environment_handoff_exact_names_equal_source_required_environment"
            )
            is True
            and orchestration.get(
                "localization_root_environment_handoff_exact_names_equal_required_environment"
            )
            is True
            and orchestration.get("ambient_sudo_environment_inheritance_or_preserve_env_is_allowed")
            is False
            and orchestration.get(
                "root_environment_handoff_assignments_precede_bin_bash_p_and_the_sealed_orchestrator"
            )
            is True
            and orchestration.get(
                "root_environment_handoff_read_only_preflight_compares_exact_nonempty_name_value_map_count_and_root_identity_before_each_one_shot_dispatch"
            )
            is True
            and orchestration.get(
                "root_environment_handoff_read_only_preflight_creates_no_p29_state"
            )
            is True
            and orchestration.get(
                "sealed_orchestrator_revalidates_every_handed_off_value_inside_the_first_retained_producer_before_substantive_phase_work"
            )
            is True
            and orchestration.get("trusted_python_executable") == "/usr/bin/python3"
            and orchestration.get("trusted_python_flags") == ["-I", "-S"]
            and orchestration.get(
                "bootstrap_and_reconstructor_invocations_use_trusted_python_with_both_flags"
            )
            is True
            and orchestration.get("nested_python_and_git_environment_is_scrubbed") is True
            and orchestration.get(
                "independent_p29_reconstructor_creates_private_python3_and_git_shims_per_child"
            )
            is True
            and orchestration.get(
                "independent_p29_python3_shim_executes_the_same_resolved_interpreter_with_isolated_no_site_flags"
            )
            is True
            and orchestration.get(
                "independent_p29_git_shim_executes_absolute_git_with_frozen_trust_controls"
            )
            is True
            and orchestration.get("host_direct_reconstructor_invocations_allowed") is False
            and orchestration.get(
                "bundle_verifier_private_payload_must_report_p29_plus_nested_p28_contract_12_p28_outcome_10_and_p27_23"
            )
            is True
            and orchestration.get("absolute_git_executable") == "/usr/bin/git"
            and orchestration.get("git_system_attributes_disabled_with_git_attr_nosystem") is True
            and orchestration.get("clean_checkouts_include_tracked_untracked_and_ignored_files")
            is True
            and orchestration.get(
                "every_substantive_post_state_and_post_marker_command_or_validator_is_invoked_only_after_successful_no_overwrite_logger_admission"
            )
            is True
            and orchestration.get(
                "complete_validator_transcript_triplet_is_guaranteed_only_when_artifact_finalization_succeeds"
            )
            is True
            and orchestration.get(
                "logger_admission_or_finalization_fault_is_terminal_and_may_lack_a_complete_transcript_triplet"
            )
            is True
            and orchestration.get("state_changing_prepare_runtime_invocations_allowed") == 1
            and orchestration.get("sealed_orchestrator")
            == (
                "/var/lib/optimizationml-p29-20260906-02/"
                "run_p29_permission_safe_bundle_localization_bridge.sh"
            )
            and orchestration.get("sealed_orchestrator_owner") == "root:root"
            and orchestration.get("sealed_orchestrator_mode") == "0555"
            and orchestration.get(
                "sealed_orchestrator_is_created_o_excl_from_stably_read_reviewed_control_bytes_after_source_receipt"
            )
            is True
            and orchestration.get(
                "sealed_orchestrator_is_the_required_operator_prepare_and_localization_entrypoint"
            )
            is True
            and orchestration.get(
                "executing_orchestrator_path_and_digest_are_revalidated_after_token_admission_before_stateful_phase_work"
            )
            is True
            and orchestration.get(
                "wrong_root_dispatched_script_path_may_consume_the_one_shot_token_then_must_fail_closed"
            )
            is True
            and orchestration.get("sealed_orchestrator_is_invoked_with_bin_bash_privileged_mode")
            is True
            and orchestration.get("prepare_runtime_root_owned_invocation_ledger")
            == "/var/lib/optimizationml-p29-20260906-02/prepare-runtime.invoked"
            and orchestration.get("prepare_runtime_root_owned_invocation_ledger_owner")
            == "root:root"
            and orchestration.get("prepare_runtime_root_owned_invocation_ledger_mode") == "0400"
            and orchestration.get("prepare_runtime_root_owned_invocation_ledger_exact_bytes_utf8")
            == "prepare-runtime\n"
            and orchestration.get("prepare_runtime_root_owned_invocation_ledger_is_o_excl_nofollow")
            is True
            and orchestration.get(
                "prepare_runtime_root_owned_invocation_transaction_uses_supplied_expected_orchestrator_digest_before_general_environment_validation_or_source_receipt_replay"
            )
            is True
            and orchestration.get(
                "source_receipt_replay_is_read_only_but_occurs_after_irreversible_prepare_admission_and_before_attempt_execution_or_container_mutation"
            )
            is True
            and orchestration.get("localization_invocations_allowed") == 1
            and orchestration.get("run_localization_root_owned_invocation_ledger")
            == "/var/lib/optimizationml-p29-20260906-02/run-localization.invoked"
            and orchestration.get("run_localization_root_owned_invocation_ledger_owner")
            == "root:root"
            and orchestration.get("run_localization_root_owned_invocation_ledger_mode") == "0400"
            and orchestration.get("run_localization_root_owned_invocation_ledger_exact_bytes_utf8")
            == "run-localization\n"
            and orchestration.get(
                "run_localization_root_owned_invocation_ledger_is_o_excl_nofollow"
            )
            is True
            and orchestration.get("run_localization_root_owned_authorization")
            == "/var/lib/optimizationml-p29-20260906-02/localization.authorized"
            and orchestration.get("run_localization_root_owned_authorization_owner") == "root:root"
            and orchestration.get("run_localization_root_owned_authorization_mode") == "0400"
            and orchestration.get("run_localization_root_owned_authorization_exact_bytes_utf8")
            == "localization-authorized\n"
            and orchestration.get("run_localization_root_owned_authorization_is_o_excl_nofollow")
            is True
            and orchestration.get(
                "run_localization_authorization_is_created_only_after_reviewed_runtime_receipt_and_all_pre_marker_validators"
            )
            is True
            and orchestration.get(
                "run_localization_host_post_authorization_boundary_validates_prepare_invocation_localization_invocation_and_authorization_tokens_before_dispatch"
            )
            is True
            and orchestration.get("run_localization_containerized_p27_localizer_reads_root_ledger")
            is False
            and orchestration.get("run_localization_root_owned_terminal_status")
            == "/var/lib/optimizationml-p29-20260906-02/run-localization.exit-status.txt"
            and orchestration.get("run_localization_root_owned_terminal_status_owner")
            == "root:root"
            and orchestration.get("run_localization_root_owned_terminal_status_mode") == "0400"
            and orchestration.get("run_localization_root_owned_terminal_status_exact_bytes_rule")
            == "one canonical ASCII decimal integer from 0 through 255 followed by newline"
            and orchestration.get("run_localization_root_owned_terminal_status_is_o_excl_nofollow")
            is True
            and orchestration.get(
                "localization_invocation_token_transaction_includes_all_pre_and_post_validation_and_returns_success_only_to_creator"
            )
            is True
            and orchestration.get(
                "run_localization_dispatcher_runs_body_in_subshell_then_attempts_one_o_excl_terminal_status_transaction_only_after_successful_invocation_token_transaction"
            )
            is True
            and orchestration.get(
                "run_localization_complete_validated_terminal_status_is_guaranteed_only_when_terminal_status_transaction_succeeds"
            )
            is True
            and orchestration.get(
                "run_localization_failed_terminal_status_transaction_is_terminal_nonrunnable_and_may_leave_status_absent_or_created_but_unvalidated"
            )
            is True
            and orchestration.get(
                "existing_localization_invocation_token_replay_may_create_or_replace_terminal_status"
            )
            is False
            and orchestration.get(
                "partial_internal_invocation_token_transaction_leaves_explicit_incomplete_nonrunnable_state"
            )
            is True
            and orchestration.get(
                "partial_internal_prepare_token_transaction_leaves_explicit_incomplete_nonrunnable_state"
            )
            is True
            and orchestration.get(
                "prepare_runtime_dispatch_requires_exact_ledger_direct_child_inventory_plus_deferred_authority_and_sealed_orchestrator_identity_before_burning_prepare_invocation"
            )
            is True
            and orchestration.get(
                "premature_localization_after_successful_prepare_token_exists_consumes_p29"
            )
            is True
            and orchestration.get(
                "pre_prepare_localization_dispatch_creates_no_new_ledger_state_and_is_retryable"
            )
            is True
            and orchestration.get(
                "run_localization_root_owned_invocation_transaction_uses_supplied_expected_orchestrator_digest_before_general_environment_validation_evidence_or_marker_predicates"
            )
            is True
            and orchestration.get(
                "missing_or_malformed_supplied_expected_orchestrator_digest_may_fail_before_invocation_token_creation"
            )
            is True
            and orchestration.get(
                "run_localization_root_owned_invocation_ledger_once_created_remains_after_every_subsequent_success_or_failure"
            )
            is True
            and orchestration.get("late_evidence_marker_is_not_the_one_shot_admission_authority")
            is True
            and orchestration.get("cleanup_resume_or_retry_phase_exists") is False
            and orchestration.get(
                "runtime_receipt_replay_observes_untouched_exact_27_file_evidence_inventory"
            )
            is True
            and orchestration.get("runtime_receipt_replay_is_first_localization_body_operation")
            is True
            and orchestration.get(
                "runtime_environment_is_validated_inside_the_deferred_replay_producer_before_transport_validation_or_verifier_dispatch"
            )
            is True
            and orchestration.get("initial_localization_boundary_order")
            == [
                "p29-pre-marker-runtime-receipt-replay",
                "p29-localization-entry-preflight",
                "p29-pre-marker-receipt-namespace-binding",
            ]
            and orchestration.get(
                "localization_entry_preflight_runs_only_after_runtime_replay_transcript_publication"
            )
            is True
            and orchestration.get(
                "localization_entry_preflight_requires_after_runtime_deferred_journal_stage"
            )
            is True
            and orchestration.get(
                "runtime_receipt_replay_root_journal_stdout_and_stderr_are_created_o_excl_before_verifier"
            )
            is True
            and orchestration.get(
                "runtime_receipt_replay_root_journal_exit_status_is_created_o_excl_after_verifier_before_publication"
            )
            is True
            and orchestration.get("runtime_receipt_replay_deferred_journal_transcripts")
            == [
                ".deferred-p29-pre-marker-runtime-receipt-replay.stdout.log",
                ".deferred-p29-pre-marker-runtime-receipt-replay.stderr.log",
                ".deferred-p29-pre-marker-runtime-receipt-replay.exit-status.txt",
            ]
            and orchestration.get("runtime_receipt_replay_published_evidence_transcripts")
            == [
                "p29-pre-marker-runtime-receipt-replay.stdout.log",
                "p29-pre-marker-runtime-receipt-replay.stderr.log",
                "p29-pre-marker-runtime-receipt-replay.exit-status.txt",
            ]
            and orchestration.get("runtime_receipt_replay_publication_journal_transcripts")
            == [
                ".deferred-p29-pre-marker-runtime-receipt-replay.publication.stdout.log",
                ".deferred-p29-pre-marker-runtime-receipt-replay.publication.stderr.log",
                ".deferred-p29-pre-marker-runtime-receipt-replay.publication.exit-status.txt",
            ]
            and orchestration.get(
                "runtime_receipt_replay_publication_success_requires_empty_stdout_stderr_and_zero_status"
            )
            is True
            and orchestration.get(
                "runtime_receipt_replay_evidence_publication_requires_stable_reviewed_identity_reauthentication"
            )
            is True
            and orchestration.get(
                "runtime_receipt_replay_transcripts_are_published_to_evidence_after_verifier_exit_for_success_or_failure_only_when_identity_reauthentication_succeeds_by_stable_nofollow_o_excl_copy"
            )
            is True
            and orchestration.get(
                "when_evidence_publication_succeeds_journal_and_evidence_transcript_bytes_must_match"
            )
            is True
            and orchestration.get("successful_replay_exit_status_exact_bytes_utf8") == "0\n"
            and orchestration.get(
                "runtime_receipt_replay_failure_transcripts_remain_under_root_journal"
            )
            is True
            and orchestration.get(
                "runtime_receipt_replay_failure_transcripts_are_published_under_evidence_only_when_identity_reauthentication_succeeds"
            )
            is True
            and orchestration.get("runtime_receipt_replay_publication_failure_journal_status")
            == (
                ".deferred-p29-pre-marker-runtime-receipt-replay."
                "publication-failure.exit-status.txt"
            )
            and orchestration.get(
                "runtime_receipt_replay_publication_failure_status_is_o_excl_best_effort_when_evidence_cannot_be_reauthenticated_or_published_and_root_journal_is_intact"
            )
            is True
            and orchestration.get(
                "runtime_receipt_replay_publication_finalization_failure_returns_terminal_125"
            )
            is True
            and orchestration.get(
                "runtime_receipt_replay_may_publish_into_identity_compromised_evidence"
            )
            is False
            and orchestration.get("runtime_receipt_replay_failure_creates_localization_marker")
            is False
            and orchestration.get(
                "runtime_receipt_replay_inventory_weakening_or_premature_evidence_logging_allowed"
            )
            is False
            and orchestration.get(
                "phase_specific_namespace_child_inventory_is_exact_and_extra_siblings_are_rejected"
            )
            is True
            and orchestration.get(
                "source_bundle_and_layout_verification_precede_authority_attempt_and_execution_root_creation"
            )
            is True
            and orchestration.get(
                "runtime_receipt_verifier_holds_fd_anchors_during_each_invocation"
            )
            is True
            and orchestration.get("localization_attempt_evidence_receipt_binding_labels")
            == [
                "p29-pre-marker-receipt-namespace-binding",
                "p29-post-ingestion-receipt-namespace-binding",
                "p29-pre-localization-receipt-namespace-binding",
                "p29-post-authorization-receipt-namespace-binding",
                "p29-post-localizer-receipt-namespace-binding",
                "p29-post-sanitizer-receipt-namespace-binding",
                "p29-post-localization-receipt-namespace-binding",
            ]
            and orchestration.get("localization_attempt_evidence_layout_labels")
            == [
                "p29-localization-entry-preflight",
                "p29-pre-marker-snapshot-layout",
            ]
            and orchestration.get("execution_snapshot_full_manifest_verification_boundaries")
            == [
                "p29-prepare-execution-snapshot-verification",
                "runtime-review receipt creation",
                "p29-pre-marker-runtime-receipt-replay",
            ]
            and orchestration.get(
                "pre_marker_snapshot_layout_checks_top_level_inventory_and_manifest_authority_not_full_manifest_bytes"
            )
            is True
            and orchestration.get(
                "ordinary_logged_boundaries_do_not_implicitly_revalidate_attempt_evidence_or_full_execution_manifest"
            )
            is True
            and orchestration.get(
                "localization_host_claims_one_continuous_marker_to_process_descriptor_guard"
            )
            is False
            and orchestration.get("required_host_uid") == 0
            and orchestration.get("required_host_gid") == 0
            and orchestration.get("container_process_uid") == 0
            and orchestration.get("container_process_gid") == 0
            and orchestration.get("container_run_identity_and_capability_arguments")
            == [
                "--user",
                "0:0",
                "--userns",
                "host",
                "--cap-drop",
                "ALL",
                "--cap-add",
                "DAC_OVERRIDE",
            ]
            and orchestration.get("container_inspect_config_user") == "0:0"
            and orchestration.get("container_inspect_userns_mode") == "host"
            and orchestration.get("container_user_namespace_remapping_allowed") is False
            and orchestration.get("container_inspect_cap_drop") == ["ALL"]
            and orchestration.get("container_inspect_cap_add") == ["DAC_OVERRIDE"]
            and orchestration.get(
                "trusted_root_container_with_dac_override_is_inside_the_runtime_tcb"
            )
            is True
            and orchestration.get(
                "sealed_evidence_is_claimed_immutable_against_a_malicious_root_container"
            )
            is False
            and orchestration.get(
                "scientific_artifact_progression_requires_each_new_artifact_to_be_sealed_and_validated_at_its_declared_producer_or_binding_boundary"
            )
            is True
            and orchestration.get(
                "historical_evidence_bytes_are_not_implicitly_revalidated_after_every_logged_boundary"
            )
            is True
            and orchestration.get("localization_ledger_validation_labels_in_exact_order")
            == [
                "p29-localization-entry-preflight",
                "p29-post-authorization-ledger-validation",
                "p29-post-localizer-ledger-validation",
                "p29-post-sanitizer-ledger-validation",
            ]
            and orchestration.get(
                "localization_ledger_validation_triplets_after_successful_finalization_are_o_excl_nofollow_root_owned_and_sealed_0444"
            )
            is True
            and orchestration.get(
                "retained_logger_classes_requiring_best_effort_seal_of_every_already_closed_artifact"
            )
            == [
                "run_logged",
                "capture_new",
                "refresh_bound_file",
                "deferred_external_and_publication_status_creation",
            ]
            and orchestration.get("artifact_finalization_failure_terminal_exit_code") == 125
            and orchestration.get(
                "original_producer_exit_status_is_preserved_only_when_all_required_artifact_finalization_succeeds"
            )
            is True
            and orchestration.get(
                "artifact_finalization_failure_does_not_skip_sealing_other_already_closed_artifacts"
            )
            is True
            and orchestration.get(
                "evidence_inventory_sealing_collects_all_child_failures_before_returning_terminal_failure"
            )
            is True
            and orchestration.get(
                "freeze_runtime_logged_call_is_captured_with_errexit_disabled_then_checked_after_errexit_restoration"
            )
            is True
            and orchestration.get(
                "root_host_best_effort_seals_each_closed_evidence_child_to_root_root_0444_before_review_or_next_boundary"
            )
            is True
            and orchestration.get("canonical_authority_checkout_is_never_mounted_into_container")
            is True
            and orchestration.get("execution_repository_snapshot_is_only_repository_mount") is True
            and orchestration.get("exact_bind_mount_count") == 10
            and orchestration.get(
                "localizer_ingester_p23_core_sanitizer_and_authenticated_p26_input_enter_through_read_only_execution_data_tools_adjunct"
            )
            is True
        ),
        "immutable_execution_snapshot_manifest_sources_and_mounts_exact": (
            execution.get("root") == "/secure/p29/execution-20260906-02"
            and execution.get("root_must_be_absent_before_the_sole_prepare_runtime_invocation")
            is True
            and execution.get("created_once_during_the_sole_prepare_runtime_invocation") is True
            and execution.get("separate_from_canonical_transport_control_and_authority_checkouts")
            is True
            and execution.get("canonical_authority_is_not_relocated_rebound_or_mounted") is True
            and (execution.get("root_owner"), execution.get("root_mode")) == ("root:root", "0555")
            and (execution.get("directory_owner"), execution.get("directory_mode"))
            == ("root:root", "0555")
            and execution.get("regular_file_owner") == "root:root"
            and execution.get("regular_file_modes_allowed") == ["0444", "0555"]
            and execution.get("manifest_path")
            == "/secure/p29/execution-20260906-02/snapshot-manifest.json"
            and execution.get("manifest_schema")
            == "passive-muon-p29-immutable-execution-snapshot-v1"
            and (execution.get("manifest_owner"), execution.get("manifest_mode"))
            == ("root:root", "0444")
            and execution.get("manifest_exact_top_level_keys")
            == ["entries", "schema_version", "source_authorities"]
            and execution.get("manifest_entries_cover_every_relative_path_except_manifest") is True
            and execution.get(
                "manifest_entries_are_strictly_sorted_by_relative_path_without_duplicates"
            )
            is True
            and execution.get("manifest_directory_entry_keys")
            == ["relative_path", "kind", "mode_octal", "uid", "gid"]
            and execution.get("manifest_directory_kind_literal") == "directory"
            and execution.get("manifest_file_entry_keys")
            == [
                "relative_path",
                "kind",
                "mode_octal",
                "uid",
                "gid",
                "byte_count",
                "sha256",
            ]
            and execution.get("manifest_file_kind_literal") == "file"
            and execution.get("manifest_entry_uid") == 0
            and execution.get("manifest_entry_gid") == 0
            and execution.get("exact_top_level_children")
            == ["data", "muon", "nanogpt", "repository", "snapshot-manifest.json"]
            and _mapping(_mapping(execution.get("directory_sources")).get("repository")).get(
                "canonical_source"
            )
            == "/secure/p29/transport-20260906-02/authority-185e444"
            and _mapping(_mapping(execution.get("directory_sources")).get("repository")).get(
                "commit"
            )
            == "185e444afc0b44ca0a09b1bde49a6b6fa3973355"
            and _mapping(_mapping(execution.get("directory_sources")).get("repository")).get("tree")
            == "24f4bdac331a57bd7c1b807747d7c7fba253ee5a"
            and _mapping(_mapping(execution.get("directory_sources")).get("nanogpt")).get(
                "canonical_source"
            )
            == "/secure/p23/nanoGPT"
            and _mapping(_mapping(execution.get("directory_sources")).get("nanogpt")).get("commit")
            == "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
            and _mapping(_mapping(execution.get("directory_sources")).get("nanogpt")).get("tree")
            == "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
            and _mapping(_mapping(execution.get("directory_sources")).get("muon")).get("commit")
            == "f98f1cacc0263b04290753e32be8d498c1efc806"
            and _mapping(_mapping(execution.get("directory_sources")).get("muon")).get(
                "canonical_source"
            )
            == "/secure/p23/Muon"
            and _mapping(_mapping(execution.get("directory_sources")).get("muon")).get("tree")
            == "4ea5cd8ab6ebd56a18536f06453619efcd636da0"
            and _mapping(_mapping(execution.get("directory_sources")).get("data")).get(
                "canonical_source"
            )
            == "/secure/p23/optimizationml-p22-data"
            and _mapping(_mapping(execution.get("directory_sources")).get("data")).get(
                "original_manifest_sha256"
            )
            == "1dfbe2087fb61004c6c4ee938d1be53d769f2fa66303c2533e14acb9c1d5a96d"
            and _mapping(_mapping(execution.get("directory_sources")).get("data")).get(
                "original_manifest_byte_count"
            )
            == 4210
            and execution.get("tools_adjunct_root")
            == "/secure/p29/execution-20260906-02/data/.p29-tools"
            and execution.get("helpers_canonical_source_checkout")
            == "/secure/p29/transport-20260906-02/control-source"
            and execution.get("p26_authenticated_input_original_source")
            == "/secure/p25/attempt-20260906-02/evidence/trace-off-a-failure.json"
            and execution.get("tools_adjunct_exact_files") == expected_execution_tools
            and execution.get("manifest_source_authorities") == expected_manifest_source_authorities
            and execution.get(
                "componentwise_nofollow_identity_acl_device_mount_and_manifest_checks_required"
            )
            is True
            and execution.get(
                "recursive_symlinks_hardlinked_files_special_files_nested_mounts_acls_and_xattrs_allowed"
            )
            is False
            and execution.get("regular_file_link_count_required") == 1
            and execution.get(
                "manifest_is_verified_before_container_launch_runtime_receipt_creation_replay_and_localization"
            )
            is True
            and execution.get("exact_bind_mounts") == expected_mounts
            and execution.get("exact_bind_mount_count") == 10
            and execution.get("only_evidence_mount_is_writable") is True
            and execution.get("frozen_p23_container_destinations_are_unchanged") is True
        ),
        "fresh_attempt_locked_runtime_and_two_file_review_exact": (
            attempt.get("attempt_id") == "20260906-02"
            and attempt.get("attempt_root") == "/secure/p29/attempt-20260906-02"
            and attempt.get("container_name") == "p29-localization-20260906-02"
            and attempt.get("transport_root") == layout.get("transport_root")
            and attempt.get("root_created_once_with_privilege_after_receipt_replay") is True
            and attempt.get("root_post_creation_owner") == "root:root"
            and attempt.get("root_post_creation_uid") == 0
            and attempt.get("root_post_creation_gid") == 0
            and attempt.get("root_post_creation_mode") == "0555"
            and attempt.get("root_is_not_process_writable_after_atomic_preparation") is True
            and attempt.get("evidence_root") == "/secure/p29/attempt-20260906-02/evidence"
            and attempt.get("evidence_owner") == "root:root"
            and attempt.get("evidence_uid") == 0
            and attempt.get("evidence_gid") == 0
            and attempt.get("evidence_mode") == "0555"
            and attempt.get(
                "evidence_parent_and_evidence_root_are_root_owned_nonwritable_and_prevent_uid1000_transport_process_path_replacement"
            )
            is True
            and attempt.get(
                "root_owned_nonwritable_evidence_prevents_uid1000_transport_process_mutation_but_not_trusted_root_container_dac_override"
            )
            is True
            and attempt.get("evidence_children_owner") == "root:root"
            and attempt.get("evidence_children_sealed_mode") == "0444"
            and attempt.get(
                "evidence_children_are_created_o_excl_and_sealed_only_after_their_producer_closes"
            )
            is True
            and attempt.get(
                "evidence_world_readability_is_intentional_and_no_evidence_child_may_contain_credentials"
            )
            is True
            and attempt.get("attempt_and_evidence_componentwise_o_directory_o_nofollow_required")
            is True
            and attempt.get(
                "attempt_and_evidence_access_acl_must_equal_mode_bits_without_named_entries"
            )
            is True
            and attempt.get("attempt_and_evidence_default_acl_entries_allowed") is False
            and attempt.get("attempt_and_evidence_forbidden_acl_xattrs")
            == [
                "system.posix_acl_access",
                "system.posix_acl_default",
                "system.nfs4_acl",
                "system.richacl",
            ]
            and attempt.get("attempt_and_evidence_must_share_namespace_device_and_mount_id") is True
            and attempt.get("attempt_and_evidence_must_not_be_mountpoints") is True
            and attempt.get(
                "attempt_and_evidence_must_match_reviewed_identity_authority_acl_device_and_mount_id_at_runtime_receipt_creation_replay_and_explicit_localization_checks"
            )
            is True
            and attempt.get(
                "verifier_holds_open_attempt_and_evidence_descriptors_through_each_runtime_receipt_creation_or_replay"
            )
            is True
            and attempt.get(
                "host_reopens_attempt_and_evidence_componentwise_nofollow_at_the_explicit_layout_and_receipt_binding_checks"
            )
            is True
            and attempt.get(
                "host_descriptors_remain_open_across_entire_marker_to_localizer_process_transition"
            )
            is False
            and attempt.get("partial_root_creation_consumes_attempt") is True
            and attempt.get("future_scientific_acquisition_is_authorized") is False
            and runtime.get("image_digest")
            == "sha256:14dafde07ae578cc4725f452a51d4bc4c920b69f6af23125a72e27162e193ec0"
            and runtime.get("gpu_uuid") == "GPU-ee4c9bf9-42f5-7cdd-66d2-f638c2db549d"
            and runtime.get("network_mode") == "none"
            and runtime.get("container_run_user") == "0:0"
            and runtime.get("container_inspect_config_user") == "0:0"
            and runtime.get("container_userns_mode") == "host"
            and runtime.get("container_user_namespace_remapping_allowed") is False
            and runtime.get("container_cap_drop") == ["ALL"]
            and runtime.get("container_cap_add") == ["DAC_OVERRIDE"]
            and runtime.get("exact_bind_mount_count") == 10
            and runtime.get("mounted_repository_source")
            == "/secure/p29/execution-20260906-02/repository"
            and runtime.get("mounted_nanogpt_source") == "/secure/p29/execution-20260906-02/nanogpt"
            and runtime.get("mounted_muon_source") == "/secure/p29/execution-20260906-02/muon"
            and runtime.get("mounted_data_source") == "/secure/p29/execution-20260906-02/data"
            and runtime.get("pinned_nanogpt_commit") == "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
            and runtime.get("pinned_nanogpt_tree") == "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
            and runtime.get("pinned_muon_commit") == "f98f1cacc0263b04290753e32be8d498c1efc806"
            and runtime.get("pinned_muon_tree") == "4ea5cd8ab6ebd56a18536f06453619efcd636da0"
            and freeze.get("phase_one_parent_commit") == P28_OUTCOME_COMMIT
            and freeze.get("phase_one_must_be_direct_child_of_terminal_p28") is True
            and freeze.get("phase_two_exact_delta")
            == [
                "A\texperiments/training/p29_cuda_runtime_lock.json",
                "A\texperiments/training/p29_host_attestation.json",
            ]
            and freeze.get("phase_two_localization_started") is False
            and freeze.get(
                "canonical_authority_repository_remains_locked_at_185e444_but_is_not_mounted"
            )
            is True
            and freeze.get(
                "root_owned_immutable_execution_repository_is_derived_from_exact_185e444_tree_and_mounted"
            )
            is True
        ),
        "unchanged_no_training_boundary_and_future_acquisition_barred": (
            boundary.get("unchanged_p27_localizer_only") is True
            and boundary.get("diagnostic_stages")
            == ["pre_cuda", "post_cuda_init", "post_model_move", "post_optimizer"]
            and boundary.get("data_loading_allowed") is False
            and boundary.get("forward_allowed") is False
            and boundary.get("backward_allowed") is False
            and boundary.get("optimizer_step_allowed") is False
            and boundary.get("candidate_evaluation_allowed") is False
            and boundary.get("candidate_observation_count_required") == 0
            and routing.get("future_scientific_acquisition_authorized") is False
        ),
        "terminal_p28_outcome_independent_reconstruction_is_10_of_10": (
            p28_ok and p28_count == 10 and p28_true == 10 and p28_false == 0
        ),
        "p28_contract_independent_reconstruction_is_12_of_12": (
            p28_contract_ok
            and p28_contract_count == 12
            and p28_contract_true == 12
            and p28_contract_false == 0
        ),
        "unchanged_p27_independent_reconstruction_is_23_of_23": (
            p27_ok and p27_count == 23 and p27_true == 23 and p27_false == 0
        ),
    }
    return {
        "schema_version": (
            "passive-muon-p29-permission-safe-bundle-localization-bridge-reconstruction-v1"
        ),
        "internally_consistent": all(checks.values()),
        "checks": checks,
        "canonical_sha256": _sha256(contract_bytes),
        "p28_contract_reconstruction": {
            "check_count": p28_contract_count,
            "true_count": p28_contract_true,
            "false_count": p28_contract_false,
        },
        "p28_outcome_reconstruction": {
            "check_count": p28_count,
            "true_count": p28_true,
            "false_count": p28_false,
        },
        "p27_reconstruction": {
            "check_count": p27_count,
            "true_count": p27_true,
            "false_count": p27_false,
        },
        "claim_boundary": (
            "This reconstructs repository and preregistration facts only. It does not "
            "authenticate a host layout, transfer receipt, checkout, attempt root, container, "
            "CUDA or mapping observation, candidate, optimizer update, or training result."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CONTRACT)
    arguments = parser.parse_args()
    result = reconstruct(arguments.canonical)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["internally_consistent"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
