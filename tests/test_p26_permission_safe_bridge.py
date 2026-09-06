from __future__ import annotations

import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERIFIER = ROOT / "scripts/verify_p26_permission_safe_bridge.py"
ORCHESTRATOR = ROOT / "scripts/run_p26_permission_safe_acquisition.sh"


def _module():
    specification = importlib.util.spec_from_file_location("p26_bridge_test", VERIFIER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _write_json(path: Path, value: object, *, mode: int = 0o644) -> bytes:
    raw = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    path.write_bytes(raw)
    path.chmod(mode)
    return raw


def _fixture(tmp_path: Path) -> dict[str, Path]:
    native = tmp_path / "diagnostic.native.json"
    native_raw = _write_json(
        native,
        {
            "schema_version": "passive-muon-p25-executable-origin-diagnostic-v1",
            "mode": "remediated",
            "status": "passes",
            "passes": True,
        },
        mode=0o600,
    )
    manifest = {
        "schema_version": "passive-muon-p25-executable-origin-diagnostic-v1",
        "mode": "remediated",
        "status": "passes",
        "passes": True,
    }
    wrapper = {
        "schema_version": ("passive-muon-p25-sanitized-executable-origin-diagnostic-v1"),
        "native_schema_version": "passive-muon-p25-executable-origin-diagnostic-v1",
        "native_artifact": {
            "byte_count": len(native_raw),
            "sha256": hashlib.sha256(native_raw).hexdigest(),
        },
        "mode": "remediated",
        "status": "passes",
        "passes": True,
        "manifest": manifest,
    }
    repository_wrapper = tmp_path / "repository-wrapper.json"
    retained_wrapper = tmp_path / "retained-wrapper.json"
    _write_json(repository_wrapper, wrapper)
    _write_json(retained_wrapper, wrapper, mode=0o600)
    contract = tmp_path / "p25-contract.json"
    _write_json(
        contract,
        {
            "schema_version": ("passive-muon-p25-cuda-diagnostic-correction-contract-v1"),
            "status": "frozen_pre_remediation_build_and_pre_acquisition",
        },
    )
    return {
        "repository_wrapper": repository_wrapper,
        "retained_wrapper": retained_wrapper,
        "retained_native": native,
        "contract_path": contract,
        "repository": tmp_path,
        "sanitizer_path": tmp_path / "unused-sanitizer.py",
    }


def _verify(module: object, paths: dict[str, Path]):
    return module._verify_bound_files(
        repository_wrapper=paths["repository_wrapper"],
        retained_wrapper=paths["retained_wrapper"],
        retained_native=paths["retained_native"],
        required_uid=os.getuid(),
        required_gid=os.getgid(),
    )


def test_exact_wrapper_copies_pass_while_retained_copy_stays_0600(tmp_path: Path) -> None:
    module = _module()
    paths = _fixture(tmp_path)

    repository, retained, native, native_payload = _verify(module, paths)

    assert repository.raw == retained.raw
    assert retained.mode == 0o600
    assert native.sha256 == hashlib.sha256(native.raw).hexdigest()
    assert native_payload["passes"] is True
    assert paths["retained_wrapper"].stat().st_mode & 0o777 == 0o600


def test_one_byte_wrapper_mutation_fails_closed(tmp_path: Path) -> None:
    module = _module()
    paths = _fixture(tmp_path)
    retained = paths["retained_wrapper"]
    raw = retained.read_bytes()
    retained.write_bytes(raw.replace(b'"passes": true', b'"passes":true ', 1))
    retained.chmod(0o600)

    with pytest.raises(module.BridgeVerificationError, match="wrappers differ"):
        _verify(module, paths)


@pytest.mark.parametrize("name", ["repository_wrapper", "retained_wrapper", "retained_native"])
def test_missing_critical_file_fails_closed(tmp_path: Path, name: str) -> None:
    module = _module()
    paths = _fixture(tmp_path)
    paths[name] = tmp_path / f"missing-{name}"

    with pytest.raises(module.BridgeVerificationError, match="missing or unreadable"):
        _verify(module, paths)


@pytest.mark.parametrize("name", ["repository_wrapper", "retained_wrapper", "retained_native"])
def test_symlink_critical_file_fails_closed(tmp_path: Path, name: str) -> None:
    module = _module()
    paths = _fixture(tmp_path)
    link = tmp_path / f"linked-{name}"
    link.symlink_to(paths[name])
    paths[name] = link

    with pytest.raises(module.BridgeVerificationError, match="must not be a symlink"):
        _verify(module, paths)


@pytest.mark.parametrize("name", ["repository_wrapper", "retained_wrapper", "retained_native"])
def test_nonregular_critical_file_fails_closed(tmp_path: Path, name: str) -> None:
    module = _module()
    paths = _fixture(tmp_path)
    directory = tmp_path / f"directory-{name}"
    directory.mkdir()
    paths[name] = directory

    with pytest.raises(module.BridgeVerificationError, match="must be a regular file"):
        _verify(module, paths)


def test_native_binding_mutation_fails_closed(tmp_path: Path) -> None:
    module = _module()
    paths = _fixture(tmp_path)
    wrapper = json.loads(paths["repository_wrapper"].read_text(encoding="utf-8"))
    wrapper["native_artifact"]["sha256"] = "0" * 64
    _write_json(paths["repository_wrapper"], wrapper)
    _write_json(paths["retained_wrapper"], wrapper, mode=0o600)

    with pytest.raises(module.BridgeVerificationError, match="native SHA-256 differs"):
        _verify(module, paths)


def test_duplicate_json_key_in_equal_copies_fails_closed(tmp_path: Path) -> None:
    module = _module()
    paths = _fixture(tmp_path)
    raw = (
        paths["repository_wrapper"]
        .read_bytes()
        .replace(b'  "passes": true,', b'  "passes": true,\n  "passes": true,', 1)
    )
    paths["repository_wrapper"].write_bytes(raw)
    paths["retained_wrapper"].write_bytes(raw)
    paths["retained_wrapper"].chmod(0o600)

    with pytest.raises(module.BridgeVerificationError, match="duplicate JSON key"):
        _verify(module, paths)


def test_retained_wrapper_permission_drift_fails_closed(tmp_path: Path) -> None:
    module = _module()
    paths = _fixture(tmp_path)
    paths["retained_wrapper"].chmod(0o644)

    with pytest.raises(module.BridgeVerificationError, match="mode 0600"):
        _verify(module, paths)


def test_production_entry_point_has_no_validation_bypass() -> None:
    module = _module()
    parameters = inspect.signature(module.verify_bridge).parameters

    assert set(parameters) == {
        "repository_wrapper",
        "retained_wrapper",
        "retained_native",
        "contract_path",
        "repository",
        "sanitizer_path",
    }
    assert all(
        parameter.kind is inspect.Parameter.KEYWORD_ONLY for parameter in parameters.values()
    )


def test_orchestrator_uses_logged_root_bridge_before_unchanged_p23_sequence() -> None:
    script = ORCHESTRATOR.read_text(encoding="utf-8")

    assert "P26_REQUIRED_ATTEMPT_ID=20260906-02" in script
    assert "P26_P25_PREREG_COMMIT=e76ab62f92c95e6f0716cf2f1ed38a583cadfe56" in script
    assert "P26_EXPECTED_ORCHESTRATOR_SHA256" in script
    assert "P26_EXPECTED_RECONSTRUCTOR_SHA256" in script
    assert "run_logged_hash_bound_stdin p26-permission-safe-bridge-verification" in script
    assert "raw=pathlib.Path(sys.argv[1]).read_bytes()" in script
    assert "sys.stdout.buffer.write(raw)" in script
    assert 'rev-list --parents -n 1 "$P26_CONTROL_HEAD"' in script
    assert "sudo docker exec -i" in script
    assert "--repository-wrapper" in script
    assert "--retained-wrapper" in script
    assert "assert_no_p23_acquisition_state" in script
    assert "p26-p25-contract-reconstruction" in script
    assert script.index("p26-permission-safe-bridge-verification") < script.index("p23-trace-off-a")
    assert "cmp -s" not in script
    assert "chmod" not in script
    assert "chown" not in script
    assert "--role trace_off_a" in script
    assert "verify-repeatability" in script
    assert "--role trace_on" in script
    assert "verify-noninterference" in script
    assert '"$runner" aggregate' in script
