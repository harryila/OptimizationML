from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/ingest_p26_trace_off_a_failure.py"
P23_CORE_SOURCE = ROOT / "experiments/training/p23_deterministic_cuda_shadow_trace.py"


def _load(name: str, path: Path):
    specification = importlib.util.spec_from_file_location(name, path)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _module():
    return _load("p27_p26_failure_ingestion_under_test", SOURCE)


def _valid_payload(module, p23_core, *, native_path: Path) -> dict[str, object]:
    return {
        "schema_version": module.EXPECTED_NATIVE_SCHEMA,
        "status": "blocked_before_success_artifact",
        "acquisition_role": module.EXPECTED_ROLE,
        "exit_code": 2,
        "failure": {
            "phase": "p22_initialized",
            "error_message": module.EXPECTED_ERROR,
        },
        "runtime": {
            "python_executable": p23_core.PINNED_PYTHON_EXECUTABLE,
            "native_path": str(native_path),
        },
    }


def _write_source(
    module,
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    payload: dict[str, object],
) -> bytes:
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    source.write_bytes(raw)
    source.chmod(0o600)
    metadata = source.stat()
    monkeypatch.setattr(module, "EXPECTED_NATIVE_SHA256", hashlib.sha256(raw).hexdigest())
    monkeypatch.setattr(module, "EXPECTED_NATIVE_BYTE_COUNT", len(raw))
    monkeypatch.setattr(module, "EXPECTED_SOURCE_UID", metadata.st_uid)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_GID", metadata.st_gid)
    return raw


def _roots(p23_core, tmp_path: Path) -> dict[str, Path]:
    return {
        "repository": ROOT,
        "preprocessor_alias": tmp_path / "materialize.py",
        "nanogpt": tmp_path / "nanoGPT",
        "muon": tmp_path / "muon",
        "data": tmp_path / "data",
        "python_environment": Path(p23_core.PINNED_PYTHON_ENVIRONMENT),
        "native": Path("/workspace/evidence/p23"),
    }


def _install_reviewed_modules(module, monkeypatch: pytest.MonkeyPatch, p23_core):
    validated: list[object] = []

    class FakeRunnerError(RuntimeError):
        pass

    runner = SimpleNamespace(
        P23AcquisitionError=FakeRunnerError,
        _validate_sanitizable_artifact=lambda payload: validated.append(payload),
    )

    def load_module(_name: str, path: Path):
        return p23_core if path == module.P23_CORE_PATH else runner

    monkeypatch.setattr(module, "_load_module", load_module)
    return validated


def test_ingest_reads_once_and_publishes_exact_sanitized_wrapper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    p23_core = _load("p27_ingestion_real_p23_core", P23_CORE_SOURCE)
    p26_evidence = tmp_path / "p26-evidence"
    p27_evidence = tmp_path / "p27-evidence"
    p26_evidence.mkdir()
    p27_evidence.mkdir()
    source = p26_evidence / "trace-off-a.failure.native.json"
    payload = _valid_payload(
        module,
        p23_core,
        native_path=Path("/workspace/evidence/p23/trace-off-a.failure.native.json"),
    )
    raw = _write_source(module, monkeypatch, source, payload)
    validated = _install_reviewed_modules(module, monkeypatch, p23_core)
    native_copy = p27_evidence / "trace-off-a.failure.authenticated.json"
    sanitized_output = p27_evidence / "trace-off-a.failure.sanitized.json"

    result = module.ingest(
        source=source,
        native_copy=native_copy,
        sanitized_output=sanitized_output,
        path_roots=_roots(p23_core, tmp_path),
        output_root=p27_evidence,
    )

    assert validated == [payload]
    assert source.read_bytes() == raw
    assert native_copy.read_bytes() == raw
    assert stat.S_IMODE(native_copy.stat().st_mode) == 0o600
    assert stat.S_IMODE(sanitized_output.stat().st_mode) == 0o600
    sanitized = json.loads(sanitized_output.read_bytes())
    assert sanitized["native_artifact_sha256"] == hashlib.sha256(raw).hexdigest()
    assert sanitized["native_artifact_byte_count"] == len(raw)
    assert sanitized["p27_ingestion"] == {
        "schema_version": "passive-muon-p27-p26-failure-ingestion-v1",
        "source_read_count": 1,
        "source_mutations_performed": [],
        "source_mode_octal": "0600",
        "source_uid": source.stat().st_uid,
        "source_gid": source.stat().st_gid,
    }
    retained = sanitized["manifest"]
    assert retained["runtime"]["python_executable"] == "python_environment:bin/python"
    assert retained["runtime"]["native_path"].startswith("native:")
    assert str(source) not in sanitized_output.read_text(encoding="utf-8")
    assert result["passes"] is True
    assert result["source_read_count"] == 1
    assert result["native_copy_sha256"] == hashlib.sha256(raw).hexdigest()


def test_read_once_requires_nofollow_and_exact_root_owned_mode(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source = tmp_path / "native.json"
    source.write_bytes(b"evidence")
    source.chmod(0o600)
    metadata = source.stat()
    monkeypatch.setattr(module, "EXPECTED_SOURCE_UID", metadata.st_uid)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_GID", metadata.st_gid)
    original_open = os.open
    observed_flags: list[int] = []

    def recording_open(path, flags, *args, **kwargs):
        if Path(path) == source:
            observed_flags.append(flags)
        return original_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(module.os, "open", recording_open)
    raw, observed = module._read_once(source)
    assert raw == b"evidence"
    assert observed.st_ino == metadata.st_ino
    assert len(observed_flags) == 1
    assert observed_flags[0] & os.O_NOFOLLOW

    monkeypatch.setattr(module, "EXPECTED_SOURCE_UID", metadata.st_uid + 1)
    with pytest.raises(module.P27IngestionError, match="owner differs from root:root"):
        module._read_once(source)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_UID", metadata.st_uid)
    source.chmod(0o640)
    with pytest.raises(module.P27IngestionError, match="mode differs from 0600"):
        module._read_once(source)


def test_read_once_rejects_symlink_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    source = tmp_path / "native.json"
    source.write_bytes(b"evidence")
    source.chmod(0o600)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_UID", source.stat().st_uid)
    monkeypatch.setattr(module, "EXPECTED_SOURCE_GID", source.stat().st_gid)
    alias = tmp_path / "alias.json"
    alias.symlink_to(source)
    with pytest.raises(module.P27IngestionError, match="must not be a symlink"):
        module._read_once(alias)


def test_output_preflight_rejects_aliases_existing_paths_and_other_directories(
    tmp_path: Path,
) -> None:
    module = _module()
    source_evidence = tmp_path / "p26-evidence"
    evidence = tmp_path / "p27-evidence"
    other = tmp_path / "other"
    source_evidence.mkdir()
    evidence.mkdir()
    other.mkdir()
    source = source_evidence / "native.json"
    source.write_bytes(b"evidence")

    shared = evidence / "shared.json"
    with pytest.raises(module.P27IngestionError, match="must not alias"):
        module._preflight_paths(
            source=source,
            native_copy=shared,
            sanitized_output=shared,
            output_root=evidence,
            repository_root=ROOT,
        )
    existing = evidence / "existing.json"
    existing.write_bytes(b"do not overwrite")
    fresh = evidence / "fresh.json"
    with pytest.raises(module.P27IngestionError, match="already exists"):
        module._preflight_paths(
            source=source,
            native_copy=fresh,
            sanitized_output=existing,
            output_root=evidence,
            repository_root=ROOT,
        )
    assert not fresh.exists()
    with pytest.raises(module.P27IngestionError, match="one preexisting external"):
        module._preflight_paths(
            source=source,
            native_copy=evidence / "copy.json",
            sanitized_output=other / "sanitized.json",
            output_root=evidence,
            repository_root=ROOT,
        )
    with pytest.raises(module.P27IngestionError, match="terminal P26"):
        module._preflight_paths(
            source=source,
            native_copy=source_evidence / "copy.json",
            sanitized_output=source_evidence / "sanitized.json",
            output_root=source_evidence,
            repository_root=ROOT,
        )


def test_output_preflight_rejects_repository_paths(tmp_path: Path) -> None:
    module = _module()
    source_evidence = tmp_path / "p26-evidence"
    source_evidence.mkdir()
    source = source_evidence / "native.json"
    source.write_bytes(b"evidence")
    native_copy = SOURCE.parent / "must-not-create-native.json"
    sanitized_output = SOURCE.parent / "must-not-create-sanitized.json"
    with pytest.raises(module.P27IngestionError, match="outside repo"):
        module._preflight_paths(
            source=source,
            native_copy=native_copy,
            sanitized_output=sanitized_output,
            output_root=SOURCE.parent,
            repository_root=ROOT,
        )
    assert not native_copy.exists()
    assert not sanitized_output.exists()


def test_complete_validation_and_sanitization_precede_any_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _module()
    p23_core = _load("p27_ingestion_rejecting_p23_core", P23_CORE_SOURCE)
    p26_evidence = tmp_path / "p26-evidence"
    p27_evidence = tmp_path / "p27-evidence"
    p26_evidence.mkdir()
    p27_evidence.mkdir()
    source = p26_evidence / "trace-off-a.failure.native.json"
    payload = _valid_payload(
        module,
        p23_core,
        native_path=Path("/workspace/evidence/p23/trace-off-a.failure.native.json"),
    )
    payload["undeclared_path"] = "/not/a/declared/p23/root"
    _write_source(module, monkeypatch, source, payload)
    _install_reviewed_modules(module, monkeypatch, p23_core)
    native_copy = p27_evidence / "trace-off-a.failure.authenticated.json"
    sanitized_output = p27_evidence / "trace-off-a.failure.sanitized.json"

    with pytest.raises(module.P27IngestionError, match="sanitization blocked"):
        module.ingest(
            source=source,
            native_copy=native_copy,
            sanitized_output=sanitized_output,
            path_roots=_roots(p23_core, tmp_path),
            output_root=p27_evidence,
        )

    assert not native_copy.exists()
    assert not sanitized_output.exists()


def test_path_roots_are_exact_absolute_inventory(tmp_path: Path) -> None:
    module = _module()
    complete = {name: tmp_path / name for name in module._PATH_ROOT_LABELS}
    assert set(module._normalized_path_roots(complete)) == module._PATH_ROOT_LABELS
    incomplete = dict(complete)
    incomplete.pop("native")
    with pytest.raises(module.P27IngestionError, match="label inventory"):
        module._normalized_path_roots(incomplete)
    relative = dict(complete)
    relative["native"] = Path("relative")
    with pytest.raises(module.P27IngestionError, match="must be absolute"):
        module._normalized_path_roots(relative)
