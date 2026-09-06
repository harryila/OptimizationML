from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/sanitize_p25_executable_origin_diagnostic.py"
DIAGNOSTIC = ROOT / "experiments/training/run_p25_executable_origin_diagnostic.py"


def _module():
    specification = importlib.util.spec_from_file_location("p25_sanitizer_test", SCRIPT)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _roots(tmp_path: Path) -> dict[str, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    result = {name: tmp_path / name for name in _module().REQUIRED_LOGICAL_ROOTS}
    for path in result.values():
        path.mkdir()
    return result


def _passing_payload(module: object, artifact_root: Path) -> dict[str, object]:
    digest = "1" * 64
    artifact_root.mkdir(parents=True, exist_ok=True)
    diagnostic_source_raw = b"# synthetic diagnostic source\n"
    diagnostic_source_digest = hashlib.sha256(diagnostic_source_raw).hexdigest()
    snapshots = {
        name: {
            "file_backed_module_count": 1,
            "canonical_sha256": digest,
            "generated_module_present": present,
        }
        for name, present in module._EXPECTED_SNAPSHOT_PRESENCE.items()
    }
    generated = {
        "path": module._GENERATED_PATH,
        "sha256": module._GENERATED_SHA256,
        "byte_count": module._GENERATED_BYTE_COUNT,
        "effective_mount": {"mount_point": "/", "writable": False},
    }
    source_authorities = {
        "unchanged_authorities": {
            "p23_addendum": {
                "path": "experiments/p23.json",
                "sha256": digest,
                "mutation_allowed": False,
            }
        },
        "execution_sources": {
            "corrected_executable_origin_diagnostic": {
                "path": "experiments/diagnostic.py",
                "sha256": diagnostic_source_digest,
                "mutation_allowed_after_freeze": False,
            }
        },
    }
    remediation = {
        "build_patch": {"path": "experiments/patch.py", "sha256": digest},
        "image_recipe": {"path": "experiments/Dockerfile", "sha256": digest},
    }
    source_records = {
        f"{group}.{name}": {
            "logical_path": authority["path"],
            "path": f"/repository/{authority['path']}",
            "byte_count": 1,
            "sha256": authority["sha256"],
        }
        for group, authorities in source_authorities.items()
        for name, authority in authorities.items()
    }
    source_records.update(
        {
            f"remediation_sources.{name}": {
                "logical_path": authority["path"],
                "path": f"/repository/{authority['path']}",
                "byte_count": 1,
                "sha256": authority["sha256"],
            }
            for name, authority in remediation.items()
        }
    )
    gpu = {"uuid": "GPU-00000000-0000-0000-0000-000000000000", "name": "A100"}
    attested_container = {
        "hostname": "container",
        "mountinfo_sha256": digest,
        "network_mode": "none",
        "rootfs_read_only": True,
    }
    determinism = {
        "bf16_reduced_precision_reduction": False,
        "cpu_threads": 1,
        "cublas_workspace_config": ":4096:8",
        "cuda_visible_devices": gpu["uuid"],
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_debug_mode": "error",
        "float32_matmul_precision": "highest",
        "fp16_reduced_precision_reduction": False,
        "interop_threads": 1,
        "nvidia_tf32_override": "0",
        "nvidia_visible_devices": gpu["uuid"],
        "pythonhashseed": "1337",
        "sdpa_backend": "math",
        "tf32": False,
    }
    loader_environment = {
        "LD_AUDIT": None,
        "LD_LIBRARY_PATH": None,
        "LD_PRELOAD": None,
        "PATH": "/opt/p23-venv/bin:/usr/bin",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": None,
    }
    software = {
        "platform": "Linux-test",
        "python": "3.12.14",
        "python_executable": "/opt/p23-venv/bin/python",
        "python_flags": dict(module._FROZEN_PYTHON_FLAGS),
        "torch_version": "2.7.0+cu128",
        "torch_git_version": "a" * 40,
        "torch_cuda_compiled_version": "12.8",
        "cuda_runtime_version": 12080,
        "cudnn_version": 90701,
    }
    host_attestation = {
        "schema_version": module._HOST_ATTESTATION_SCHEMA,
        "status": "procedurally_host_attested",
        "container": attested_container,
        "gpu": gpu,
        "evidence": {},
    }
    host_attestation_raw = (json.dumps(host_attestation, indent=2) + "\n").encode()
    host_attestation_digest = hashlib.sha256(host_attestation_raw).hexdigest()
    container = {
        **attested_container,
        "host_attestation_sha256": host_attestation_digest,
    }
    runtime_lock = {
        "schema_version": module._RUNTIME_LOCK_SCHEMA,
        "status": "pinned_for_acquisition",
        "p23_addendum_sha256": digest,
        "container": container,
        "gpu": gpu,
        "software": software,
        "determinism": determinism,
        "loader_environment": loader_environment,
    }
    runtime_lock_raw = (json.dumps(runtime_lock, indent=2) + "\n").encode()
    contract_content = {
        "schema_version": ("passive-muon-p25-cuda-diagnostic-correction-contract-v1"),
        "status": "frozen_pre_remediation_build_and_pre_acquisition",
        **source_authorities,
        "remediation_design": remediation,
    }
    contract_raw = (json.dumps(contract_content, indent=2) + "\n").encode()
    raw_artifacts = {
        "contract": contract_raw,
        "runtime_lock": runtime_lock_raw,
        "host_attestation": host_attestation_raw,
        "diagnostic_source": diagnostic_source_raw,
    }
    artifacts: dict[str, dict[str, object]] = {}
    for name, content in raw_artifacts.items():
        path = artifact_root / f"{name}.fixture"
        path.write_bytes(content)
        artifacts[name] = {
            "path": str(path.resolve()),
            "sha256": hashlib.sha256(content).hexdigest(),
            "byte_count": len(content),
        }
    live_runtime = {
        "hostname": container["hostname"],
        "platform": software["platform"],
        "python": software["python"],
        "python_executable": software["python_executable"],
        "python_flags": software["python_flags"],
        "torch_version": software["torch_version"],
        "torch_git_version": software["torch_git_version"],
        "torch_cuda_compiled_version": software["torch_cuda_compiled_version"],
        "cuda_runtime_version": software["cuda_runtime_version"],
        "cudnn_version": software["cudnn_version"],
        "cuda_available": True,
        "cuda_device_count": 1,
        "cuda_device_name": gpu["name"],
        "cuda_device_uuid": gpu["uuid"],
        "environment": module._expected_environment(runtime_lock),
        "mountinfo_sha256": digest,
        "root_effective_mount": {"mount_point": "/", "writable": False},
        "cuda_initialized_after_import": False,
        "cuda_initialized_after_configuration": False,
        "configured_determinism": dict(module._EXPECTED_TORCH_DETERMINISM),
        "determinism_after_cpu_parameter": dict(module._EXPECTED_TORCH_DETERMINISM),
        "determinism_after_sgd_optimizer": dict(module._EXPECTED_TORCH_DETERMINISM),
    }
    payload = {
        "schema_version": module.NATIVE_SCHEMA,
        "mode": "remediated",
        "status": "passes",
        "passes": True,
        "scope": "minimal diagnostic",
        "seed": None,
        "seed_semantics": "not applicable: the diagnostic performs no random operation",
        "process": {
            "pid": 1,
            "nonce": digest,
            "started_time_ns": 1,
            "completed_time_ns": 2,
            "argv": ["diagnostic.py"],
        },
        "repository": {
            "expected_head": "a" * 40,
            "expected_tree": "b" * 40,
            "observed": {"head": "a" * 40, "tree": "b" * 40, "dirty": False},
        },
        "artifacts": artifacts,
        "contract_binding": {
            "expected_sha256": artifacts["contract"]["sha256"],
            "content": contract_content,
            "source_records": source_records,
        },
        "runtime_lock_binding": {
            "expected_sha256": artifacts["runtime_lock"]["sha256"],
            "content": runtime_lock,
        },
        "host_attestation_binding": {
            "expected_sha256": artifacts["host_attestation"]["sha256"],
            "content": host_attestation,
        },
        "live_runtime": live_runtime,
        "pinned_sources": {
            name: {
                "path": f"/opt/p23-venv/{module._PINNED_SOURCE_SUFFIX[name]}",
                "sha256": sha256,
                "byte_count": 1,
            }
            for name, sha256 in module._PINNED_SOURCE_SHA256.items()
        },
        "trigger": {
            "event_order": list(module._EXPECTED_EVENT_ORDER),
            "module_snapshots": snapshots,
            "generated_module": generated,
            "temporary_generated_module_names": [],
        },
        "file_backed_module_closure": {
            "count": 1,
            "canonical_sha256": hashlib.sha256(
                json.dumps(
                    {"_remote_module_non_scriptable": generated},
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "records": {"_remote_module_non_scriptable": generated},
        },
        "claim_boundary": "minimal diagnostic only",
        "checks": {name: True for name in module._PASS_CHECK_NAMES},
    }
    assert set(payload) == module._PASS_TOP_LEVEL_FIELDS
    return payload


def test_strict_loader_rejects_duplicate_keys_and_nonfinite_constants() -> None:
    module = _module()
    with pytest.raises(module.SanitizationError, match="duplicate JSON key"):
        module.load_json_strict(b'{"a": 1, "a": 2}')
    with pytest.raises(module.SanitizationError, match="nonfinite JSON constant"):
        module.load_json_strict(b'{"a": NaN}')


def test_one_rule_redacts_mapping_keys_and_values_with_separate_counts(
    tmp_path: Path,
) -> None:
    module = _module()
    roots = module.validate_path_roots(_roots(tmp_path))
    repository = roots["repository"]
    payload = {
        str(repository / "absolute-key"): str(repository / "absolute-value"),
        "embedded": f"failure at {repository}/module.py",
        "nested": [{"path": str(repository)}],
    }
    redacted, counts = module.redact_payload(payload, roots)
    assert redacted == {
        "repository:absolute-key": "repository:absolute-value",
        "embedded": "failure at repository:module.py",
        "nested": [{"path": "repository:"}],
    }
    assert counts == {
        "key_replacement_count": 1,
        "value_replacement_count": 3,
        "total_replacement_count": 4,
    }
    rendered = json.dumps(redacted, sort_keys=True)
    assert str(repository) not in rendered


def test_redactor_rejects_post_redaction_mapping_key_collision(tmp_path: Path) -> None:
    module = _module()
    roots = module.validate_path_roots(_roots(tmp_path))
    repository = roots["repository"]
    payload = {str(repository / "same"): 1, "repository:same": 2}
    with pytest.raises(module.SanitizationError, match="mapping-key collision"):
        module.redact_payload(payload, roots)


def test_redactor_eliminates_declared_and_undeclared_absolute_paths(tmp_path: Path) -> None:
    module = _module()
    roots = module.validate_path_roots(_roots(tmp_path))
    payload = {
        "filesystem_root": "/",
        "external": "/not-a-declared-root/file.py",
        "path_list": "/one/bin:/two/bin",
    }
    redacted, counts = module.redact_payload(payload, roots)
    assert counts == {
        "key_replacement_count": 0,
        "value_replacement_count": 4,
        "total_replacement_count": 4,
    }
    assert redacted["filesystem_root"].startswith("external_absolute_path_sha256:")
    assert redacted["external"].startswith("external_absolute_path_sha256:")
    assert "/" not in redacted["path_list"]
    module._assert_no_absolute_paths(redacted)


@pytest.mark.parametrize(
    "payload",
    [
        {"token_secret": "not-retained"},
        {"token": "not-retained"},
        {"nested": {"api-key": "not-retained"}},
        {"value": math.inf},
    ],
)
def test_redactor_rejects_credentials_and_nonfinite_values(
    tmp_path: Path, payload: dict[str, object]
) -> None:
    module = _module()
    roots = module.validate_path_roots(_roots(tmp_path))
    with pytest.raises(module.SanitizationError):
        module.redact_payload(payload, roots)


def test_sanitize_native_binds_bytes_and_publishes_no_overwrite(tmp_path: Path) -> None:
    module = _module()
    raw_roots = _roots(tmp_path)
    native = raw_roots["native"] / "p25.native.json"
    output = raw_roots["native"] / "p25.sanitized.json"
    payload = {
        "schema_version": module.NATIVE_SCHEMA,
        "mode": "remediated",
        "status": "fails",
        "passes": False,
        "path_map": {str(raw_roots["repository"]): str(raw_roots["python_environment"])},
    }
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    native.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    result = module.sanitize_native(
        native=native,
        expected_native_sha256=digest,
        output=output,
        path_roots=raw_roots,
    )
    assert result["native_artifact"] == {"byte_count": len(raw), "sha256": digest}
    assert result["redaction"] == {
        "key_replacement_count": 1,
        "value_replacement_count": 1,
        "total_replacement_count": 2,
    }
    assert result["manifest"]["path_map"] == {"repository:": "python_environment:"}
    assert json.loads(output.read_text(encoding="utf-8")) == result
    with pytest.raises(module.SanitizationError, match="fresh path"):
        module.sanitize_native(
            native=native,
            expected_native_sha256=digest,
            output=output,
            path_roots=raw_roots,
        )


def test_pass_authorization_requires_the_complete_exact_native_semantics(
    tmp_path: Path,
) -> None:
    module = _module()
    roots = _roots(tmp_path)
    native = roots["native"] / "passing.native.json"
    output = roots["native"] / "passing.sanitized.json"
    payload = _passing_payload(module, roots["repository"])
    raw = (json.dumps(payload, sort_keys=True) + "\n").encode()
    native.write_bytes(raw)
    result = module.sanitize_native(
        native=native,
        expected_native_sha256=hashlib.sha256(raw).hexdigest(),
        output=output,
        path_roots=roots,
    )
    assert result["passes"] is True

    minimal = roots["native"] / "minimal.native.json"
    minimal_payload = {
        "schema_version": module.NATIVE_SCHEMA,
        "mode": "remediated",
        "status": "passes",
        "passes": True,
    }
    minimal_raw = json.dumps(minimal_payload).encode()
    minimal.write_bytes(minimal_raw)
    with pytest.raises(module.SanitizationError, match="top-level schema"):
        module.sanitize_native(
            native=minimal,
            expected_native_sha256=hashlib.sha256(minimal_raw).hexdigest(),
            output=roots["native"] / "minimal.sanitized.json",
            path_roots=roots,
        )


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("runtime_lock_binding", "embedded runtime_lock content"),
        ("host_attestation_binding", "embedded host_attestation content"),
        ("live_runtime", "live-runtime schema"),
        ("pinned_sources", "pinned Torch-source inventory"),
        ("process", "process record"),
    ],
)
def test_pass_authorization_reconstructs_native_facts_instead_of_trusting_checks(
    tmp_path: Path, field: str, message: str
) -> None:
    module = _module()
    payload = _passing_payload(module, tmp_path / "artifacts")
    if field in {"runtime_lock_binding", "host_attestation_binding"}:
        payload[field]["content"] = {}
    else:
        payload[field] = {}
    assert all(payload["checks"].values())
    with pytest.raises(module.SanitizationError, match=message):
        module.validate_native_semantics(payload)


def test_joint_native_tamper_cannot_borrow_the_bound_authority_digest(tmp_path: Path) -> None:
    module = _module()
    payload = _passing_payload(module, tmp_path / "artifacts")
    payload["runtime_lock_binding"]["content"]["software"]["platform"] = "tampered"
    payload["live_runtime"]["platform"] = "tampered"
    assert all(payload["checks"].values())
    with pytest.raises(module.SanitizationError, match="embedded runtime_lock content"):
        module.validate_native_semantics(payload)


def test_pass_check_inventory_matches_the_diagnostic_source_exactly() -> None:
    module = _module()
    tree = ast.parse(DIAGNOSTIC.read_text(encoding="utf-8"))
    literal_names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or not any(
            isinstance(target, ast.Name) and target.id == "checks" for target in node.targets
        ):
            continue
        if not isinstance(node.value, ast.Dict):
            continue
        literal_names.update(
            key.value
            for key in node.value.keys
            if isinstance(key, ast.Constant) and isinstance(key.value, str)
        )
    assert literal_names == module._PASS_CHECK_NAMES
    assert len(literal_names) == 40


def test_sanitize_native_rejects_digest_mismatch_and_symlink(tmp_path: Path) -> None:
    module = _module()
    roots = _roots(tmp_path)
    native = roots["native"] / "native.json"
    native.write_text(
        json.dumps(
            {
                "schema_version": module.NATIVE_SCHEMA,
                "status": "fails",
                "passes": False,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(module.SanitizationError, match="expected SHA-256"):
        module.sanitize_native(
            native=native,
            expected_native_sha256="0" * 64,
            output=roots["native"] / "out.json",
            path_roots=roots,
        )
    link = roots["native"] / "link.json"
    try:
        os.symlink(native, link)
    except OSError:
        pytest.skip("symbolic links unavailable")
    with pytest.raises(module.SanitizationError, match="symbolic link"):
        module.sanitize_native(
            native=link,
            expected_native_sha256=hashlib.sha256(native.read_bytes()).hexdigest(),
            output=roots["native"] / "link-out.json",
            path_roots=roots,
        )


def test_root_validation_rejects_overlap_and_filesystem_root(tmp_path: Path) -> None:
    module = _module()
    roots = _roots(tmp_path)
    child = roots["native"] / "child"
    child.mkdir()
    roots["temporary"] = child
    with pytest.raises(module.SanitizationError, match="pairwise nonoverlapping"):
        module.validate_path_roots(roots)
    roots = _roots(tmp_path / "second")
    roots["native"] = Path("/")
    with pytest.raises(module.SanitizationError, match="non-root absolute"):
        module.validate_path_roots(roots)


def test_root_validation_rejects_missing_label(tmp_path: Path) -> None:
    module = _module()
    roots = _roots(tmp_path)
    del roots["data"]
    with pytest.raises(module.SanitizationError, match="nine-label inventory"):
        module.validate_path_roots(roots)
