from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/sanitize_p27_cuda_deleted_mapping_localization.py"
RUNNER_SOURCE = ROOT / "experiments/training/run_p27_cuda_deleted_mapping_localization.py"


def _load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _modules():
    return (
        _load("p27_sanitizer_under_test", SOURCE),
        _load("p27_sanitizer_runner_fixture", RUNNER_SOURCE),
    )


def _empty_map_probe(runner) -> dict[str, object]:
    return {
        "attempted": False,
        "lstat_status": "not_applicable",
        "readlink_status": "not_applicable",
        "readlink_target_bytes_hex": None,
        "open_status": "not_applicable",
        "fstat_status": "not_applicable",
        "device_major_minor": None,
        "inode_decimal": None,
        "mode_octal": None,
        "file_type": None,
        "size_bytes": None,
        "errno": None,
        "error_message": None,
    }


def _mapping(runner) -> dict[str, object]:
    pathname = b"/workspace/evidence/p27/cuda-cache (deleted)"
    target = b"/workspace/evidence/p27/cuda-cache"
    map_probe = _empty_map_probe(runner)
    map_probe.update(
        {
            "attempted": True,
            "lstat_status": "error",
            "readlink_status": "ok",
            "readlink_target_bytes_hex": target.hex(),
            "open_status": "error",
            "fstat_status": "not_attempted",
            "errno": 2,
            "error_message": "open failed at /proc/self/map_files/1000-2000",
        }
    )
    fd_probe = {
        "fd_decimal": 7,
        "readlink_status": "ok",
        "readlink_target_bytes_hex": target.hex(),
        "fstat_status": "ok",
        "device_major_minor": "00:01",
        "inode_decimal": 17,
        "mode_octal": "100400",
        "file_type": "regular",
        "size_bytes": 4096,
        "fd_flags_hex": "0x0",
        "errno": None,
        "error_message": None,
    }
    return {
        "line_number": 17,
        "raw_line_sha256": "1" * 64,
        "raw_line_byte_count": 91,
        "address_start_hex": "1000",
        "address_end_hex": "2000",
        "permissions": "r--s",
        "offset_hex": "00000000",
        "device_major_minor": "00:01",
        "inode_decimal": 17,
        "pathname_present": True,
        "pathname_bytes_hex": pathname.hex(),
        "deleted_suffix_present": True,
        "pathname_without_deleted_bytes_hex": target.hex(),
        "classification": "deleted_regular_read_only",
        "map_files_probe": map_probe,
        "matching_fd_probes": [fd_probe],
    }


def _snapshots(runner) -> list[dict[str, object]]:
    snapshots = []
    for index, stage in enumerate(runner.STAGES):
        mappings = [_mapping(runner)] if index >= 1 else []
        snapshots.append(
            {
                "stage": stage,
                "time_ns": 100 + index,
                "process_pid": 19,
                "proc_maps_sha256": str(index) * 64,
                "proc_maps_byte_count": 4096 + index,
                "mapping_count": len(mappings),
                "post_probe_proc_maps_sha256": str(index) * 64,
                "post_probe_proc_maps_byte_count": 4096 + index,
                "post_probe_mapping_count": len(mappings),
                "maps_byte_stable_during_probe": True,
                "all_deleted_mapping_identities_stable_during_probe": True,
                "deleted_mapping_count": len(mappings),
                "mappings": mappings,
            }
        )
    return snapshots


def _success_payload(runner) -> dict[str, object]:
    snapshots = _snapshots(runner)
    payload = {
        "schema_version": runner.SCHEMA,
        "status": "complete",
        "passes": True,
        "scope": "four-stage CUDA deleted-mapping localization; no training computation",
        "seed": runner.SEED,
        "process": {
            "pid": 19,
            "started_time_ns": 1,
            "completed_time_ns": 2,
            "argv": [
                "/opt/p23-venv/bin/python",
                "/workspace/repository/experiments/training/run_p27_cuda_deleted_mapping_localization.py",
                "--output",
                "/workspace/evidence/p27/native.json",
            ],
        },
        "bindings": {
            "source": {
                "path": (
                    "/workspace/repository/experiments/training/"
                    "run_p27_cuda_deleted_mapping_localization.py"
                ),
                "sha256": "2" * 64,
                "byte_count": 12,
            },
            "nanogpt_origin": "https://github.com/karpathy/nanoGPT",
        },
        "live_runtime": {
            "torch_version": "2.7.0+cu128",
            "cuda_device_count": 1,
        },
        "stage_order": list(runner.STAGES),
        "stage_snapshots": snapshots,
        "construction": {
            "unique_parameter_count": 75,
            "all_gradients_absent": True,
        },
        "localization": runner._localization_summary(snapshots),
        "training_operations": {name: 0 for name in runner.TRAINING_OPERATION_FIELDS},
        "policy": {
            "classification_performed": False,
            "allowlist_changed": False,
            "statement": "No mapping was authorized.",
        },
        "checks": {"complete": True},
        "claim_boundary": (
            "This is a non-training localization diagnostic, not repeatability, "
            "noninterference, gradient, fidelity, or training evidence."
        ),
    }
    runner.validate_payload(payload)
    return payload


def _failure_payload(runner) -> dict[str, object]:
    snapshots = _snapshots(runner)[:2]
    return {
        "schema_version": runner.SCHEMA,
        "status": "blocked",
        "passes": False,
        "scope": "four-stage CUDA deleted-mapping localization; no training computation",
        "seed": runner.SEED,
        "process": {
            "pid": 19,
            "started_time_ns": 1,
            "failed_time_ns": 2,
            "argv": [
                "/opt/p23-venv/bin/python",
                "/workspace/repository/experiments/training/run_p27_cuda_deleted_mapping_localization.py",
            ],
        },
        "bindings": {"source": {"path": "/workspace/repository/experiments/training/runner.py"}},
        "stage_order": list(runner.STAGES),
        "stage_snapshots": snapshots,
        "training_operations": {name: 0 for name in runner.TRAINING_OPERATION_FIELDS},
        "policy": {
            "classification_performed": False,
            "allowlist_changed": False,
            "statement": "Failure evidence only; no mapping policy decision was made.",
        },
        "error": {
            "class": "builtins.RuntimeError",
            "message": "probe failed at /proc/self/map_files/1000-2000",
            "traceback": (
                '  File "/workspace/repository/experiments/training/runner.py", line 1\n'
                "RuntimeError: probe failed\n"
            ),
        },
        "claim_boundary": (
            "This blocked non-training diagnostic is not repeatability, noninterference, "
            "gradient, fidelity, or training evidence."
        ),
    }


def _roots() -> dict[str, Path]:
    return {
        "repository_mount": Path("/workspace/repository"),
        "native_evidence": Path("/workspace/evidence/p27"),
        "python_environment": Path("/opt/p23-venv"),
        "proc": Path("/proc"),
    }


def _write_native(path: Path, value: object) -> tuple[str, int]:
    raw = (json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    path.write_bytes(raw)
    path.chmod(0o600)
    return hashlib.sha256(raw).hexdigest(), len(raw)


def _sanitize(
    sanitizer,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: object,
    *,
    native_name: str = "native.json",
    output_name: str = "sanitized.json",
):
    native = tmp_path / native_name
    output = tmp_path / output_name
    digest, byte_count = _write_native(native, payload)
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_UID", os.getuid())
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_GID", os.getgid())
    result = sanitizer.sanitize(
        native=native,
        output=output,
        expected_native_sha256=digest,
        expected_native_byte_count=byte_count,
        path_roots=_roots(),
        output_root=tmp_path,
        runner_source_path=RUNNER_SOURCE,
        repository_root=ROOT,
    )
    return native, output, result


def test_success_payload_is_bound_path_free_and_preserves_nonpath_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sanitizer, runner = _modules()
    native, output, result = _sanitize(sanitizer, monkeypatch, tmp_path, _success_payload(runner))
    wrapper = json.loads(output.read_bytes())
    assert wrapper["schema_version"] == sanitizer.SCHEMA
    assert wrapper["native_artifact"] == {
        "schema_version": runner.SCHEMA,
        "payload_kind": "complete_payload",
        "sha256": hashlib.sha256(native.read_bytes()).hexdigest(),
        "byte_count": len(native.read_bytes()),
        "mode_octal": "0600",
        "uid": os.getuid(),
        "gid": os.getgid(),
    }
    assert wrapper["runner_authority_sha256"] == sanitizer.EXPECTED_RUNNER_SHA256
    manifest = wrapper["manifest"]
    mapping = manifest["stage_snapshots"][1]["mappings"][0]
    assert mapping["line_number"] == 17
    assert mapping["raw_line_sha256"] == "1" * 64
    assert mapping["raw_line_byte_count"] == 91
    assert mapping["classification"] == "deleted_regular_read_only"
    expected_path = b"/workspace/evidence/p27/cuda-cache (deleted)"
    assert mapping["pathname_bytes_hex"] == {
        "sha256": hashlib.sha256(expected_path).hexdigest(),
        "byte_count": len(expected_path),
        "logical_labels": ["native_evidence"],
    }
    assert mapping["map_files_probe"]["errno"] == 2
    assert mapping["map_files_probe"]["error_message"]["logical_labels"] == ["proc"]
    assert manifest["bindings"]["nanogpt_origin"] == ("https://github.com/karpathy/nanoGPT")
    assert all(isinstance(item, dict) for item in manifest["process"]["argv"])
    assert stat.S_IMODE(output.stat().st_mode) == 0o600
    assert result["sanitized_sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    assert b"/workspace/" not in output.read_bytes()
    assert b"/opt/p23-venv" not in output.read_bytes()
    assert b"/proc/self" not in output.read_bytes()


def test_exact_failure_schema_is_validated_and_sanitized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sanitizer, runner = _modules()
    _, output, _ = _sanitize(sanitizer, monkeypatch, tmp_path, _failure_payload(runner))
    wrapper = json.loads(output.read_bytes())
    assert wrapper["native_artifact"]["payload_kind"] == "failure_payload"
    error = wrapper["manifest"]["error"]
    assert error["class"] == "builtins.RuntimeError"
    assert error["message"]["logical_labels"] == ["proc"]
    assert error["traceback"]["logical_labels"] == ["repository_mount"]
    assert wrapper["manifest"]["training_operations"] == {
        name: 0 for name in runner.TRAINING_OPERATION_FIELDS
    }


@pytest.mark.parametrize(
    "unbound",
    [
        "/secret/unregistered/value",
        "prefix|/secret/unregistered/value",
        "file:///secret/unregistered/value",
    ],
)
def test_unknown_absolute_path_fails_closed_without_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, unbound: str
) -> None:
    sanitizer, runner = _modules()
    payload = _success_payload(runner)
    payload["live_runtime"]["unbound"] = unbound
    native = tmp_path / "native.json"
    output = tmp_path / "sanitized.json"
    digest, byte_count = _write_native(native, payload)
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_UID", os.getuid())
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_GID", os.getgid())
    with pytest.raises(
        sanitizer.P27SanitizationError,
        match=r"unknown absolute path|file URI",
    ):
        sanitizer.sanitize(
            native=native,
            output=output,
            expected_native_sha256=digest,
            expected_native_byte_count=byte_count,
            path_roots=_roots(),
            output_root=tmp_path,
            runner_source_path=RUNNER_SOURCE,
            repository_root=ROOT,
        )
    assert not output.exists()


@pytest.mark.parametrize("variant", ["duplicate", "nonfinite", "unknown_schema"])
def test_strict_json_and_schema_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, variant: str
) -> None:
    sanitizer, runner = _modules()
    if variant == "duplicate":
        raw = (
            b'{"schema_version":"'
            + runner.SCHEMA.encode()
            + b'","schema_version":"'
            + runner.SCHEMA.encode()
            + b'"}\n'
        )
    elif variant == "nonfinite":
        raw = b'{"value":NaN}\n'
    else:
        raw = b'{"schema_version":"unknown"}\n'
    native = tmp_path / "native.json"
    native.write_bytes(raw)
    native.chmod(0o600)
    output = tmp_path / "sanitized.json"
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_UID", os.getuid())
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_GID", os.getgid())
    with pytest.raises((ValueError, sanitizer.P27SanitizationError)):
        sanitizer.sanitize(
            native=native,
            output=output,
            expected_native_sha256=hashlib.sha256(raw).hexdigest(),
            expected_native_byte_count=len(raw),
            path_roots=_roots(),
            output_root=tmp_path,
            runner_source_path=RUNNER_SOURCE,
            repository_root=ROOT,
        )
    assert not output.exists()


def test_native_requires_nofollow_root_owner_and_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sanitizer, runner = _modules()
    payload = _success_payload(runner)
    real = tmp_path / "real.json"
    digest, byte_count = _write_native(real, payload)
    link = tmp_path / "native.json"
    link.symlink_to(real)
    output = tmp_path / "sanitized.json"
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_UID", os.getuid())
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_GID", os.getgid())
    with pytest.raises(sanitizer.P27SanitizationError, match="nonsymlink"):
        sanitizer.sanitize(
            native=link,
            output=output,
            expected_native_sha256=digest,
            expected_native_byte_count=byte_count,
            path_roots=_roots(),
            output_root=tmp_path,
            runner_source_path=RUNNER_SOURCE,
            repository_root=ROOT,
        )
    real.chmod(0o644)
    with pytest.raises(sanitizer.P27SanitizationError, match="mode differs"):
        sanitizer.sanitize(
            native=real,
            output=output,
            expected_native_sha256=digest,
            expected_native_byte_count=byte_count,
            path_roots=_roots(),
            output_root=tmp_path,
            runner_source_path=RUNNER_SOURCE,
            repository_root=ROOT,
        )
    real.chmod(0o600)
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_UID", os.getuid() + 1)
    with pytest.raises(sanitizer.P27SanitizationError, match="UID differs"):
        sanitizer.sanitize(
            native=real,
            output=output,
            expected_native_sha256=digest,
            expected_native_byte_count=byte_count,
            path_roots=_roots(),
            output_root=tmp_path,
            runner_source_path=RUNNER_SOURCE,
            repository_root=ROOT,
        )
    assert not output.exists()


def test_output_is_fresh_external_nonaliasing_and_bound_to_declared_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sanitizer, runner = _modules()
    payload = _success_payload(runner)
    native = tmp_path / "native.json"
    digest, byte_count = _write_native(native, payload)
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_UID", os.getuid())
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_GID", os.getgid())
    common = {
        "native": native,
        "expected_native_sha256": digest,
        "expected_native_byte_count": byte_count,
        "path_roots": _roots(),
        "runner_source_path": RUNNER_SOURCE,
        "repository_root": ROOT,
    }
    with pytest.raises(sanitizer.P27SanitizationError, match="alias"):
        sanitizer.sanitize(output=native, output_root=tmp_path, **common)
    other = tmp_path / "other"
    other.mkdir()
    with pytest.raises(sanitizer.P27SanitizationError, match="declared root"):
        sanitizer.sanitize(output=other / "out.json", output_root=tmp_path, **common)
    existing = tmp_path / "existing.json"
    existing.write_text("retain", encoding="utf-8")
    with pytest.raises(sanitizer.P27SanitizationError, match="already exists"):
        sanitizer.sanitize(output=existing, output_root=tmp_path, **common)
    assert existing.read_text(encoding="utf-8") == "retain"
    inside_repo = ROOT / "results"
    with pytest.raises(sanitizer.P27SanitizationError, match="outside repo"):
        sanitizer.sanitize(
            output=inside_repo / "forbidden-p27-sanitized.json",
            output_root=inside_repo,
            **common,
        )


def test_validation_authority_hash_and_failure_semantics_are_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sanitizer, runner = _modules()
    payload = _failure_payload(runner)
    payload["training_operations"]["candidate_evaluations"] = 1
    native = tmp_path / "native.json"
    output = tmp_path / "sanitized.json"
    digest, byte_count = _write_native(native, payload)
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_UID", os.getuid())
    monkeypatch.setattr(sanitizer, "EXPECTED_NATIVE_GID", os.getgid())
    with pytest.raises(sanitizer.P27SanitizationError, match="training operation"):
        sanitizer.sanitize(
            native=native,
            output=output,
            expected_native_sha256=digest,
            expected_native_byte_count=byte_count,
            path_roots=_roots(),
            output_root=tmp_path,
            runner_source_path=RUNNER_SOURCE,
            repository_root=ROOT,
        )
    payload = _success_payload(runner)
    digest, byte_count = _write_native(native, payload)
    monkeypatch.setattr(sanitizer, "EXPECTED_RUNNER_SHA256", "0" * 64)
    with pytest.raises(sanitizer.P27SanitizationError, match="authority SHA-256"):
        sanitizer.sanitize(
            native=native,
            output=output,
            expected_native_sha256=digest,
            expected_native_byte_count=byte_count,
            path_roots=_roots(),
            output_root=tmp_path,
            runner_source_path=RUNNER_SOURCE,
            repository_root=ROOT,
        )
    assert not output.exists()


def test_input_objects_are_not_mutated(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sanitizer, runner = _modules()
    payload = _success_payload(runner)
    before = copy.deepcopy(payload)
    _sanitize(sanitizer, monkeypatch, tmp_path, payload)
    assert payload == before
