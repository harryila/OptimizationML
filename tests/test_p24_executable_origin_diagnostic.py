from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/training/run_p24_executable_origin_diagnostic.py"
SANITIZER = ROOT / "scripts/sanitize_p24_executable_origin_diagnostic.py"


def _module():
    spec = importlib.util.spec_from_file_location("p24_origin_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sanitizer_module():
    spec = importlib.util.spec_from_file_location("p24_origin_sanitizer", SANITIZER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _write_bound(path: Path, payload: bytes) -> dict[str, object]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return {
        "path": str(path.resolve()),
        "byte_count": len(payload),
        "sha256": _digest(payload),
    }


def test_p24_diagnostic_constants_are_pinned() -> None:
    module = _module()
    assert module.GENERATED_SHA256 == (
        "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
    )
    assert module.ORIGINAL_INSTANTIATOR_SHA256 == (
        "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
    )
    assert module.PATCHED_INSTANTIATOR_SHA256 == (
        "1ecc8ae1ac4a517511914fff6e01c78ed3ad828bbe8a07b02079c206ee56bceb"
    )
    assert module.FIXED_GENERATED_PATH.endswith(
        "torch/_p24_generated_remote_modules/_remote_module_non_scriptable.py"
    )


def test_mountinfo_parser_uses_longest_effective_mount() -> None:
    module = _module()
    mounts = module.parse_mountinfo(
        "1 0 0:1 / / ro,relatime - overlay overlay rw,lowerdir=/x\n"
        "2 1 0:2 / /synthetic-tmp rw,nosuid,nodev - tmpfs tmpfs rw,size=1024\n"
    )
    root = module.effective_mount(Path("/opt/example.py"), mounts)
    temporary = module.effective_mount(Path("/synthetic-tmp/example.py"), mounts)
    assert root["mount_point"] == "/" and root["writable"] is False
    assert temporary["mount_point"] == "/synthetic-tmp" and temporary["writable"] is True


def test_mountinfo_parser_decodes_octal_escapes() -> None:
    module = _module()
    mounts = module.parse_mountinfo("1 0 0:1 / /path\\040with\\040spaces ro - ext4 /dev/x ro\n")
    assert mounts[0]["mount_point"] == "/path with spaces"


def test_mountinfo_parser_rejects_malformed_lines() -> None:
    module = _module()
    with pytest.raises(module.DiagnosticError, match="malformed"):
        module.parse_mountinfo("not mountinfo")


def test_atomic_writer_refuses_existing_output(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "diagnostic.json"
    output.write_text("existing", encoding="utf-8")
    with pytest.raises(module.DiagnosticError, match="already exists"):
        module._atomic_write_json(output, {"unexpected": True})
    assert output.read_text(encoding="utf-8") == "existing"


def test_atomic_writer_produces_canonical_complete_json(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "diagnostic.json"
    payload = {"schema_version": module.SCHEMA, "passes": False}
    module._atomic_write_json(output, payload)
    assert json.loads(output.read_text(encoding="utf-8")) == payload
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_atomic_writer_cannot_overwrite_a_racing_publisher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    output = tmp_path / "diagnostic.json"
    real_link = os.link

    def racing_link(
        source: str | Path,
        destination: str | Path,
        *,
        follow_symlinks: bool = True,
    ) -> None:
        Path(destination).write_text("racing publisher\n", encoding="utf-8")
        real_link(source, destination, follow_symlinks=follow_symlinks)

    monkeypatch.setattr(module.os, "link", racing_link)
    with pytest.raises(module.DiagnosticError, match="already exists"):
        module._atomic_write_json(output, {"unexpected": True})
    assert output.read_text(encoding="utf-8") == "racing publisher\n"
    assert [path.name for path in tmp_path.iterdir()] == ["diagnostic.json"]


def _preflight_fixture(
    module: object, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    contract_path = tmp_path / "contract.json"
    runtime_path = tmp_path / "runtime-lock.json"
    attestation_path = tmp_path / "host-attestation.json"
    mountinfo = b"fixture mountinfo\n"
    repository_head = "a" * 40
    repository_tree = "b" * 40
    contract = {
        "schema_version": module.CONTRACT_SCHEMA,
        "status": "frozen_pre_remediation_build_and_pre_acquisition",
        "unchanged_authorities": {
            "p23_addendum": {
                "path": "fixture",
                "sha256": "c" * 64,
                "mutation_allowed": False,
            }
        },
    }
    container_without_backlink = {
        "hostname": "fixture-container",
        "rootfs_read_only": True,
        "mountinfo_sha256": module.sha256_bytes(mountinfo),
    }
    gpu = {"uuid": "GPU-fixture", "name": "fixture GPU"}
    attestation = {
        "schema_version": module.HOST_ATTESTATION_SCHEMA,
        "status": "procedurally_host_attested",
        "container": container_without_backlink,
        "gpu": gpu,
        "evidence": {},
    }
    attestation_path.write_text(json.dumps(attestation) + "\n", encoding="utf-8")
    attestation_sha256 = module.sha256_file(attestation_path)
    runtime = {
        "schema_version": module.RUNTIME_LOCK_SCHEMA,
        "status": "pinned_for_acquisition",
        "p23_addendum_sha256": "c" * 64,
        "container": {
            **container_without_backlink,
            "host_attestation_sha256": attestation_sha256,
        },
        "gpu": gpu,
        "software": {
            "python": sys.version,
            "python_executable": sys.executable,
        },
        "determinism": {
            "cublas_workspace_config": ":4096:8",
            "cuda_visible_devices": "GPU-fixture",
            "cpu_threads": 1,
            "nvidia_tf32_override": "0",
            "nvidia_visible_devices": "GPU-fixture",
            "pythonhashseed": "1337",
        },
        "loader_environment": {
            "LD_AUDIT": None,
            "LD_LIBRARY_PATH": None,
            "LD_PRELOAD": None,
            "PATH": "/opt/p23-venv/bin:/usr/bin",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONOPTIMIZE": None,
        },
    }
    contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
    runtime_path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
    expected_environment = module._expected_process_environment(runtime)
    for name in module.DETERMINISM_ENVIRONMENT:
        monkeypatch.delenv(name, raising=False)
    for name, value in expected_environment.items():
        if value is not None:
            monkeypatch.setenv(name, value)
    monkeypatch.setattr(module.socket, "gethostname", lambda: "fixture-container")
    monkeypatch.setattr(
        module,
        "git_provenance",
        lambda repository: {
            "path": str(repository.resolve()),
            "head": repository_head,
            "tree": repository_tree,
            "dirty": False,
            "porcelain_v1_z_sha256": module.sha256_bytes(b""),
        },
    )
    monkeypatch.setattr(
        module,
        "_contract_source_records",
        lambda repository, value: {"unchanged_authorities.p23_addendum": {"sha256": "c" * 64}},
    )
    return {
        "mode": "remediated",
        "repository": ROOT,
        "runtime_lock_path": runtime_path,
        "host_attestation_path": attestation_path,
        "contract_path": contract_path,
        "expected_contract_sha256": module.sha256_file(contract_path),
        "expected_runtime_lock_sha256": module.sha256_file(runtime_path),
        "expected_host_attestation_sha256": attestation_sha256,
        "expected_repository_head": repository_head,
        "expected_repository_tree": repository_tree,
        "mountinfo": mountinfo,
        "mounts": [
            {
                "mount_id": 1,
                "mount_point": "/",
                "filesystem_type": "overlay",
                "mount_options": ["ro"],
                "super_options": ["rw"],
                "writable": False,
            }
        ],
    }


def test_preflight_binds_complete_lock_attestation_contract_repo_and_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    arguments = _preflight_fixture(module, tmp_path, monkeypatch)
    result = module.validate_preflight(**arguments)

    assert all(result["checks"].values())
    assert result["artifacts"]["contract"]["sha256"] == arguments["expected_contract_sha256"]
    assert (
        result["artifacts"]["runtime_lock"]["sha256"] == arguments["expected_runtime_lock_sha256"]
    )
    assert result["runtime_lock"]["container"]["rootfs_read_only"] is True
    assert result["host_attestation"]["container"] == {
        "hostname": "fixture-container",
        "rootfs_read_only": True,
        "mountinfo_sha256": module.sha256_bytes(arguments["mountinfo"]),
    }
    assert result["repository"]["head"] == arguments["expected_repository_head"]
    assert result["observed_environment"] == result["expected_environment"]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_contract_sha256", "0" * 64),
        ("expected_runtime_lock_sha256", "1" * 64),
        ("expected_host_attestation_sha256", "2" * 64),
        ("expected_repository_head", "3" * 40),
        ("expected_repository_tree", "4" * 40),
    ],
)
def test_preflight_rejects_every_external_expected_binding_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    module = _module()
    arguments = _preflight_fixture(module, tmp_path, monkeypatch)
    arguments[field] = value
    with pytest.raises(module.DiagnosticError, match="preflight checks failed"):
        module.validate_preflight(**arguments)


def test_preflight_rejects_writable_root_and_environment_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    arguments = _preflight_fixture(module, tmp_path, monkeypatch)
    arguments["mounts"][0]["writable"] = True
    with pytest.raises(module.DiagnosticError, match="rootfs_declared_and_observed_read_only"):
        module.validate_preflight(**arguments)

    arguments = _preflight_fixture(module, tmp_path, monkeypatch)
    monkeypatch.setenv("PYTHONHASHSEED", "mutated")
    with pytest.raises(module.DiagnosticError, match="process_environment_matches_lock_exactly"):
        module.validate_preflight(**arguments)


def test_error_payload_retains_available_rule7_provenance(tmp_path: Path) -> None:
    module = _module()
    runtime_lock = tmp_path / "runtime-lock.json"
    contract = tmp_path / "contract.json"
    host_attestation = tmp_path / "host-attestation.json"
    runtime_lock.write_text(
        json.dumps(
            {
                "container": {},
                "gpu": {},
                "software": {},
                "status": "fixture",
            }
        ),
        encoding="utf-8",
    )
    contract.write_text("{}\n", encoding="utf-8")
    host_attestation.write_text("{}\n", encoding="utf-8")
    payload = module.error_payload(
        mode="baseline",
        repository=ROOT,
        runtime_lock_path=runtime_lock,
        host_attestation_path=host_attestation,
        contract_path=contract,
        expected_contract_sha256="a" * 64,
        expected_runtime_lock_sha256="b" * 64,
        expected_host_attestation_sha256="c" * 64,
        expected_repository_head="d" * 40,
        expected_repository_tree="e" * 40,
        error=RuntimeError("fixture failure"),
    )
    assert payload["schema_version"] == module.SCHEMA
    assert payload["status"] == "error"
    assert payload["seed"] is None
    assert payload["repository"]["observed"]["head"]
    assert payload["artifacts"]["diagnostic_source"]["sha256"]
    assert payload["artifacts"]["runtime_lock"]["sha256"] == module.sha256_file(runtime_lock)
    assert payload["artifacts"]["host_attestation"]["sha256"] == module.sha256_file(
        host_attestation
    )
    assert payload["runtime_lock_binding"]["expected_sha256"] == "b" * 64
    assert payload["error"]["message"] == "fixture failure"
    assert payload["passes"] is False


def _sanitizer_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[object, dict[str, object], Path, Path, dict[str, Path]]:
    module = _sanitizer_module()
    roots = {
        name: tmp_path / name
        for name in (
            "repository",
            "nanogpt",
            "muon",
            "data",
            "preprocessor_alias",
            "python_environment",
            "python_standard_library",
            "temporary",
            "native",
        )
    }
    for root in roots.values():
        root.mkdir(parents=True)

    repository = roots["repository"]
    p23 = _write_bound(repository / "p23.json", b"p23 authority\n")
    diagnostic = _write_bound(repository / "diagnostic.py", b"diagnostic source\n")
    build_helper = _write_bound(repository / "build.py", b"build helper\n")
    image_recipe = _write_bound(repository / "Dockerfile", b"FROM scratch\n")
    contract = {
        "schema_version": module.DIAGNOSTIC.CONTRACT_SCHEMA,
        "status": "frozen_pre_remediation_build_and_pre_acquisition",
        "unchanged_authorities": {
            "p23_addendum": {
                "path": "p23.json",
                "sha256": p23["sha256"],
                "mutation_allowed": False,
            }
        },
        "execution_sources": {
            "native_executable_origin_diagnostic": {
                "path": "diagnostic.py",
                "sha256": diagnostic["sha256"],
                "mutation_allowed_after_freeze": False,
            }
        },
        "remediation_design": {
            "build_patch": {
                "path": "build.py",
                "sha256": build_helper["sha256"],
                "mutation_allowed_after_freeze": False,
            },
            "image_recipe": {
                "path": "Dockerfile",
                "sha256": image_recipe["sha256"],
                "mutation_allowed_after_freeze": False,
            },
        },
    }
    contract_path = repository / "contract.json"
    contract_path.write_text(json.dumps(contract) + "\n", encoding="utf-8")
    contract_binding = _write_bound(contract_path, contract_path.read_bytes())

    mountinfo_sha256 = "f" * 64
    container_without_backlink = {
        "hostname": "fixture-container",
        "rootfs_read_only": True,
        "mountinfo_sha256": mountinfo_sha256,
    }
    gpu = {"uuid": "GPU-fixture", "name": "fixture GPU"}
    attestation = {
        "schema_version": module.DIAGNOSTIC.HOST_ATTESTATION_SCHEMA,
        "status": "procedurally_host_attested",
        "container": container_without_backlink,
        "gpu": gpu,
        "evidence": {},
    }
    attestation_path = roots["native"] / "host-attestation.json"
    attestation_path.write_text(json.dumps(attestation) + "\n", encoding="utf-8")
    attestation_binding = _write_bound(attestation_path, attestation_path.read_bytes())
    determinism = {
        "cublas_workspace_config": ":4096:8",
        "cuda_visible_devices": "GPU-fixture",
        "cpu_threads": 1,
        "interop_threads": 1,
        "nvidia_tf32_override": "0",
        "nvidia_visible_devices": "GPU-fixture",
        "pythonhashseed": "1337",
        "deterministic_algorithms": True,
        "deterministic_debug_mode": "error",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "float32_matmul_precision": "highest",
        "tf32": False,
        "fp16_reduced_precision_reduction": False,
        "bf16_reduced_precision_reduction": False,
    }
    loader_environment = {
        "LD_AUDIT": None,
        "LD_LIBRARY_PATH": None,
        "LD_PRELOAD": None,
        "PATH": module.SAFE_PATH_VALUE,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": None,
    }
    python_executable = roots["python_environment"] / "bin/python"
    _write_bound(python_executable, b"fixture python\n")
    software = {
        "python": "fixture Python",
        "python_executable": str(python_executable.resolve()),
        "torch_version": "2.7.0+cu128",
        "torch_git_version": "torch-git",
        "torch_cuda_compiled_version": "12.8",
        "cuda_runtime_version": 12080,
        "cudnn_version": 90701,
    }
    runtime = {
        "schema_version": module.DIAGNOSTIC.RUNTIME_LOCK_SCHEMA,
        "status": "pinned_for_acquisition",
        "p23_addendum_sha256": p23["sha256"],
        "container": {
            **container_without_backlink,
            "host_attestation_sha256": attestation_binding["sha256"],
        },
        "gpu": gpu,
        "software": software,
        "determinism": determinism,
        "loader_environment": loader_environment,
    }
    runtime_path = roots["native"] / "runtime-lock.json"
    runtime_path.write_text(json.dumps(runtime) + "\n", encoding="utf-8")
    runtime_binding = _write_bound(runtime_path, runtime_path.read_bytes())

    pinned_sources: dict[str, dict[str, object]] = {}
    for name in ("instantiator", "remote_module", "template"):
        pinned_sources[name] = _write_bound(
            roots["python_environment"] / f"{name}.py", f"{name}\n".encode()
        )
    monkeypatch.setattr(
        module.DIAGNOSTIC,
        "PATCHED_INSTANTIATOR_SHA256",
        pinned_sources["instantiator"]["sha256"],
    )
    monkeypatch.setattr(
        module.DIAGNOSTIC,
        "REMOTE_MODULE_SHA256",
        pinned_sources["remote_module"]["sha256"],
    )
    monkeypatch.setattr(
        module.DIAGNOSTIC,
        "TEMPLATE_SHA256",
        pinned_sources["template"]["sha256"],
    )
    monkeypatch.setattr(
        module.DIAGNOSTIC,
        "PINNED_SOURCES",
        {name: Path(str(record["path"])) for name, record in pinned_sources.items()},
    )
    generated_path = roots["python_environment"] / "generated.py"
    generated = _write_bound(generated_path, b"generated module\n")
    monkeypatch.setattr(module.DIAGNOSTIC, "GENERATED_SHA256", generated["sha256"])
    monkeypatch.setattr(module.DIAGNOSTIC, "FIXED_GENERATED_PATH", str(generated_path.resolve()))

    mount = {
        "mount_id": 1,
        "mount_point": "/",
        "filesystem_type": "overlay",
        "mount_options": ["ro"],
        "super_options": ["rw"],
        "writable": False,
    }
    generated_record = {**generated, "effective_mount": mount}
    closure_records = {module.DIAGNOSTIC.GENERATED_MODULE: generated_record}
    source_records = {
        "unchanged_authorities.p23_addendum": {
            "logical_path": "p23.json",
            **p23,
        },
        "execution_sources.native_executable_origin_diagnostic": {
            "logical_path": "diagnostic.py",
            **diagnostic,
        },
        "remediation_sources.build_patch": {
            "logical_path": "build.py",
            **build_helper,
        },
        "remediation_sources.image_recipe": {
            "logical_path": "Dockerfile",
            **image_recipe,
        },
    }
    expected_environment = module.DIAGNOSTIC._expected_process_environment(runtime)
    expected_environment["VIRTUAL_ENV"] = str(roots["python_environment"].resolve())
    monkeypatch.setattr(
        module.DIAGNOSTIC,
        "_expected_process_environment",
        lambda value: expected_environment,
    )
    head = "a" * 40
    tree = "b" * 40
    payload: dict[str, object] = {
        "schema_version": module.DIAGNOSTIC.SCHEMA,
        "mode": "remediated",
        "status": "passes",
        "scope": "fixture diagnostic only",
        "seed": None,
        "seed_semantics": "not applicable: the diagnostic performs no random operation",
        "process": {
            "pid": 1,
            "nonce": "c" * 64,
            "started_time_ns": 1,
            "completed_time_ns": 2,
            "argv": [str(diagnostic["path"])],
        },
        "repository": {
            "expected_head": head,
            "expected_tree": tree,
            "observed": {
                "path": str(repository.resolve()),
                "head": head,
                "tree": tree,
                "dirty": False,
                "porcelain_v1_z_sha256": module._EMPTY_SHA256,
            },
        },
        "artifacts": {
            "diagnostic_source": diagnostic,
            "contract": contract_binding,
            "runtime_lock": runtime_binding,
            "host_attestation": attestation_binding,
        },
        "contract_binding": {
            "expected_sha256": contract_binding["sha256"],
            "content": contract,
            "source_records": source_records,
        },
        "runtime_lock_binding": {
            "expected_sha256": runtime_binding["sha256"],
            "content": runtime,
        },
        "host_attestation_binding": {
            "expected_sha256": attestation_binding["sha256"],
            "content": attestation,
        },
        "live_runtime": {
            "hostname": "fixture-container",
            "platform": "fixture platform",
            **software,
            "cuda_available": True,
            "cuda_device_count": 1,
            "cuda_device_name": gpu["name"],
            "cuda_device_uuid": gpu["uuid"],
            "environment": expected_environment,
            "mountinfo_sha256": mountinfo_sha256,
            "root_effective_mount": mount,
            "deterministic_algorithms": True,
            "deterministic_debug_mode": 2,
            "cpu_threads": 1,
            "interop_threads": 1,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "float32_matmul_precision": "highest",
            "tf32": False,
            "fp16_reduced_precision_reduction": False,
            "bf16_reduced_precision_reduction": False,
        },
        "pinned_sources": pinned_sources,
        "trigger": {
            "operation": (
                "import torch; create one zero CPU float32 Parameter; construct "
                "torch.optim.SGD([parameter], lr=1.0); perform no optimizer step"
            ),
            "generated_module": generated_record,
            "temporary_generated_module_names": [],
            "new_modules_after_torch_import": [],
            "new_modules_after_parameter": [],
            "new_modules_after_optimizer": [module.DIAGNOSTIC.GENERATED_MODULE],
        },
        "file_backed_module_closure": {
            "count": len(closure_records),
            "canonical_sha256": module.DIAGNOSTIC.canonical_sha256(closure_records),
            "records": closure_records,
        },
        "checks": {name: True for name in module.COMPLETED_CHECK_NAMES},
        "claim_boundary": "fixture diagnostic; no acquisition or training result",
        "passes": True,
    }
    native = roots["native"] / "diagnostic.native.json"
    native.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    output = roots["native"] / "diagnostic.sanitized.json"
    return module, payload, native, output, roots


def _rewrite_native(path: Path, payload: object) -> str:
    path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    return _digest(path.read_bytes())


def _as_error_payload(
    module: object, payload: dict[str, object], roots: dict[str, Path]
) -> dict[str, object]:
    artifacts = payload["artifacts"]
    assert isinstance(artifacts, dict)
    best_effort = {
        name: {
            "path": record["path"],
            "exists": True,
            "is_file": True,
            "byte_count": record["byte_count"],
            "sha256": record["sha256"],
        }
        for name, record in artifacts.items()
    }
    payload.update(
        {
            "status": "error",
            "process": {
                "pid": 1,
                "nonce": "c" * 64,
                "failed_time_ns": 2,
                "argv": [str(artifacts["diagnostic_source"]["path"])],
            },
            "repository": {
                "expected_head": "a" * 40,
                "expected_tree": "b" * 40,
                "observed": {
                    "path": str(roots["repository"].resolve()),
                    "status": "unavailable",
                    "error_class": "builtins.RuntimeError",
                    "error_message": "fixture collection failure",
                },
            },
            "artifacts": best_effort,
            "contract_binding": {
                "expected_sha256": artifacts["contract"]["sha256"],
                "content": None,
                "source_records": {},
            },
            "runtime_lock_binding": {
                "expected_sha256": artifacts["runtime_lock"]["sha256"],
                "content": None,
            },
            "host_attestation_binding": {
                "expected_sha256": artifacts["host_attestation"]["sha256"],
                "content": None,
            },
            "live_runtime": {
                "hostname": "fixture-container",
                "platform": "fixture platform",
                "python": "fixture Python",
                "python_executable": str((roots["python_environment"] / "bin/python").resolve()),
                "environment": {name: None for name in module.DIAGNOSTIC.DETERMINISM_ENVIRONMENT},
                "mountinfo_sha256": None,
            },
            "pinned_sources": {},
            "trigger": None,
            "file_backed_module_closure": None,
            "checks": {"diagnostic_completed": False},
            "error": {
                "class": "builtins.RuntimeError",
                "message": "fixture failure at /var/lib/docker/private",
                "traceback": (
                    "Traceback (most recent call last):\n"
                    f'  File "{roots["repository"] / "diagnostic.py"}", line 1\n'
                    "RuntimeError: fixture failure at /var/lib/docker/private\n"
                ),
            },
            "passes": False,
        }
    )
    return payload


def test_p24_offline_sanitizer_retains_full_manifest_and_native_byte_binding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    native_sha256 = _digest(native.read_bytes())
    result = module.sanitize_native(
        native=native,
        expected_native_sha256=native_sha256,
        output=output,
        path_roots=roots,
    )
    assert result["schema_version"] == module.SANITIZED_SCHEMA
    assert result["native_artifact_sha256"] == native_sha256
    assert result["native_artifact_byte_count"] == native.stat().st_size
    assert result["manifest"]["status"] == payload["status"]
    assert result["logical_root_labels"] == sorted(roots)
    rendered = output.read_text(encoding="utf-8")
    assert all(str(root.resolve()) not in rendered for root in roots.values())
    assert result["path_replacement_count"] > 0


def test_p24_offline_sanitizer_accepts_evidence_consistent_completed_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    payload["live_runtime"]["torch_version"] = "post-trigger-drift"  # type: ignore[index]
    payload["checks"]["torch_version_matches_lock"] = False  # type: ignore[index]
    payload["status"] = "fails"
    payload["passes"] = False
    expected = _rewrite_native(native, payload)

    result = module.sanitize_native(
        native=native,
        expected_native_sha256=expected,
        output=output,
        path_roots=roots,
    )

    assert result["status"] == "fails"
    assert result["manifest"]["checks"]["torch_version_matches_lock"] is False


def test_p24_offline_sanitizer_retains_missing_generated_module_as_completed_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    closure = payload["file_backed_module_closure"]  # type: ignore[assignment]
    generated = closure["records"].pop(module.DIAGNOSTIC.GENERATED_MODULE)
    closure["records"]["other_loaded_module"] = {
        **generated,
        "path": payload["pinned_sources"]["remote_module"]["path"],  # type: ignore[index]
    }
    closure["count"] = 1
    closure["canonical_sha256"] = module.DIAGNOSTIC.canonical_sha256(closure["records"])
    payload["trigger"]["generated_module"] = None  # type: ignore[index]
    payload["trigger"]["new_modules_after_optimizer"] = []  # type: ignore[index]
    for name in (
        "generated_module_loaded_only_after_optimizer",
        "generated_module_bytes_exact",
        "mode_origin_matches",
    ):
        payload["checks"][name] = False  # type: ignore[index]
    payload["status"] = "fails"
    payload["passes"] = False
    expected = _rewrite_native(native, payload)

    result = module.sanitize_native(
        native=native,
        expected_native_sha256=expected,
        output=output,
        path_roots=roots,
    )

    assert result["status"] == "fails"
    assert result["manifest"]["trigger"]["generated_module"] is None


def test_p24_offline_sanitizer_accepts_baseline_writable_origin_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(
        module.DIAGNOSTIC,
        "ORIGINAL_INSTANTIATOR_SHA256",
        payload["pinned_sources"]["instantiator"]["sha256"],  # type: ignore[index]
    )
    generated_name = module.DIAGNOSTIC.GENERATED_MODULE
    generated = payload["trigger"]["generated_module"]  # type: ignore[index]
    roots["temporary"] = Path("/tmp")
    generated["path"] = "/tmp/generated.py"
    generated["effective_mount"] = copy.deepcopy(generated["effective_mount"])
    generated["effective_mount"]["mount_options"] = ["rw"]
    generated["effective_mount"]["writable"] = True
    closure = payload["file_backed_module_closure"]  # type: ignore[assignment]
    closure["records"][generated_name] = copy.deepcopy(generated)
    closure["canonical_sha256"] = module.DIAGNOSTIC.canonical_sha256(closure["records"])
    payload["trigger"]["temporary_generated_module_names"] = [generated_name]  # type: ignore[index]
    payload["mode"] = "baseline"
    expected = _rewrite_native(native, payload)

    result = module.sanitize_native(
        native=native,
        expected_native_sha256=expected,
        output=output,
        path_roots=roots,
    )

    assert result["status"] == "passes"
    assert result["manifest"]["mode"] == "baseline"


def test_p24_offline_sanitizer_accepts_zero_byte_closure_modules(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    empty_path = roots["python_environment"] / "empty_module.py"
    empty_path.write_bytes(b"")
    empty_record = {
        "path": str(empty_path.resolve()),
        "byte_count": 0,
        "sha256": _digest(b""),
        "effective_mount": copy.deepcopy(
            payload["trigger"]["generated_module"]["effective_mount"]  # type: ignore[index]
        ),
    }
    closure = payload["file_backed_module_closure"]  # type: ignore[assignment]
    closure["records"]["empty_module"] = empty_record
    closure["count"] = len(closure["records"])
    closure["canonical_sha256"] = module.DIAGNOSTIC.canonical_sha256(closure["records"])
    expected = _rewrite_native(native, payload)

    module.sanitize_native(
        native=native,
        expected_native_sha256=expected,
        output=output,
        path_roots=roots,
    )


@pytest.mark.parametrize("mutation", ["missing_check", "lying_check", "contradictory_mount"])
def test_p24_offline_sanitizer_rejects_check_and_mount_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    if mutation == "missing_check":
        payload["checks"].pop("mode_origin_matches")  # type: ignore[index]
    elif mutation == "lying_check":
        payload["live_runtime"]["torch_version"] = "post-trigger-drift"  # type: ignore[index]
    else:
        generated_name = module.DIAGNOSTIC.GENERATED_MODULE
        closure = payload["file_backed_module_closure"]  # type: ignore[assignment]
        closure["records"][generated_name]["effective_mount"]["mount_options"] = ["rw"]
        closure["canonical_sha256"] = module.DIAGNOSTIC.canonical_sha256(closure["records"])
    expected = _rewrite_native(native, payload)

    with pytest.raises(module.SanitizationError):
        module.sanitize_native(
            native=native,
            expected_native_sha256=expected,
            output=output,
            path_roots=roots,
        )


def test_p24_offline_sanitizer_rejects_every_fabricated_check_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    for index, name in enumerate(sorted(module.COMPLETED_CHECK_NAMES)):
        mutated = copy.deepcopy(payload)
        mutated["checks"][name] = False
        mutated["status"] = "fails"
        mutated["passes"] = False
        expected = _rewrite_native(native, mutated)
        candidate_output = output.with_name(f"check-{index}.json")
        with pytest.raises(module.SanitizationError, match="checks differ from reconstruction"):
            module.sanitize_native(
                native=native,
                expected_native_sha256=expected,
                output=candidate_output,
                path_roots=roots,
            )


def test_p24_offline_sanitizer_redacts_embedded_external_mount_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    generated_name = module.DIAGNOSTIC.GENERATED_MODULE
    closure = payload["file_backed_module_closure"]  # type: ignore[assignment]
    external = "/var/lib/docker/overlay2/private-layer"
    for generated in (
        payload["trigger"]["generated_module"],  # type: ignore[index]
        closure["records"][generated_name],
    ):
        generated["effective_mount"]["super_options"] = [
            "rw",
            f"lowerdir={external}",
            f"upperdir={external}-upper",
        ]
    closure["canonical_sha256"] = module.DIAGNOSTIC.canonical_sha256(closure["records"])
    expected = _rewrite_native(native, payload)

    result = module.sanitize_native(
        native=native,
        expected_native_sha256=expected,
        output=output,
        path_roots=roots,
    )

    rendered = json.dumps(result, sort_keys=True)
    assert "/var/lib/docker" not in rendered
    assert "external_absolute_path_sha256:" in rendered


def test_p24_offline_sanitizer_accepts_and_rehashes_error_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    payload = _as_error_payload(module, payload, roots)
    expected = _rewrite_native(native, payload)

    result = module.sanitize_native(
        native=native,
        expected_native_sha256=expected,
        output=output,
        path_roots=roots,
    )

    assert result["status"] == "error"
    rendered = json.dumps(result, sort_keys=True)
    assert "/var/lib/docker" not in rendered
    assert str(roots["repository"]) not in rendered

    module, payload, native, output, roots = _sanitizer_fixture(tmp_path / "stale", monkeypatch)
    payload = _as_error_payload(module, payload, roots)
    expected = _rewrite_native(native, payload)
    Path(payload["artifacts"]["contract"]["path"]).write_text(  # type: ignore[index]
        "changed after failure\n", encoding="utf-8"
    )
    with pytest.raises(module.SanitizationError, match="changed after"):
        module.sanitize_native(
            native=native,
            expected_native_sha256=expected,
            output=output,
            path_roots=roots,
        )


def test_p24_offline_sanitizer_rejects_wrong_native_digest_and_existing_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, _, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    with pytest.raises(module.SanitizationError, match="expected digest"):
        module.sanitize_native(
            native=native,
            expected_native_sha256="0" * 64,
            output=output,
            path_roots=roots,
        )
    output.write_text("existing\n", encoding="utf-8")
    with pytest.raises(module.SanitizationError, match="fresh distinct"):
        module.sanitize_native(
            native=native,
            expected_native_sha256=_digest(native.read_bytes()),
            output=output,
            path_roots=roots,
        )
    assert output.read_text(encoding="utf-8") == "existing\n"


@pytest.mark.parametrize(
    "path",
    [
        ("runtime_lock_binding", "content", "p23_addendum_sha256"),
        (
            "contract_binding",
            "content",
            "unchanged_authorities",
            "p23_addendum",
            "sha256",
        ),
        (
            "contract_binding",
            "source_records",
            "unchanged_authorities.p23_addendum",
            "sha256",
        ),
    ],
)
def test_p24_offline_sanitizer_rejects_each_addendum_binding_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    path: tuple[str, ...],
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    mutated = copy.deepcopy(payload)
    target = mutated
    for item in path[:-1]:
        target = target[item]  # type: ignore[index]
    target[path[-1]] = "0" * 64  # type: ignore[index]
    expected = _rewrite_native(native, mutated)
    with pytest.raises(module.SanitizationError):
        module.sanitize_native(
            native=native,
            expected_native_sha256=expected,
            output=output,
            path_roots=roots,
        )


def test_p24_offline_sanitizer_rejects_mutated_closure_and_overlapping_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module, payload, native, output, roots = _sanitizer_fixture(tmp_path, monkeypatch)
    mutated = copy.deepcopy(payload)
    mutated["file_backed_module_closure"]["canonical_sha256"] = "0" * 64  # type: ignore[index]
    expected = _rewrite_native(native, mutated)
    with pytest.raises(module.SanitizationError, match="module-closure canonical"):
        module.sanitize_native(
            native=native,
            expected_native_sha256=expected,
            output=output,
            path_roots=roots,
        )

    _, _, native, output, roots = _sanitizer_fixture(tmp_path / "overlap", monkeypatch)
    roots["data"] = roots["repository"]
    with pytest.raises(module.SanitizationError, match="pairwise non-overlapping"):
        module.sanitize_native(
            native=native,
            expected_native_sha256=_digest(native.read_bytes()),
            output=output,
            path_roots=roots,
        )
