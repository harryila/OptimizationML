from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/training/p23_deterministic_cuda_shadow_trace.py"
SPEC = importlib.util.spec_from_file_location("p23_deterministic_cuda_shadow_trace", SCRIPT)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P23 provenance module")
P23 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = P23
SPEC.loader.exec_module(P23)


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


_READ_ONLY_MOUNTINFO = b"1 0 0:1 / / ro,relatime - overlay overlay ro\n"


def _mountinfo_with(*, read_only: tuple[Path, ...] = (), writable: tuple[Path, ...] = ()) -> bytes:
    lines = [_READ_ONLY_MOUNTINFO.decode("utf-8").rstrip("\n")]
    for mount_id, (path, mode) in enumerate(
        [*((path, "ro") for path in read_only), *((path, "rw") for path in writable)],
        start=2,
    ):
        lines.append(f"{mount_id} 1 0:{mount_id} / {path} {mode},relatime - tmpfs tmpfs {mode}")
    return ("\n".join(lines) + "\n").encode("utf-8")


def _cpu_identity() -> dict[str, object]:
    return {
        "vendor_id": "GenuineTest",
        "cpu_family": "6",
        "model": "1",
        "model_name": "Pinned CPU",
        "stepping": "1",
        "microcode": "0x1",
        "flags_sha256": "6" * 64,
        "logical_processor_count": 8,
    }


def _container_identity(*, include_attestation_hash: bool = True) -> dict[str, object]:
    identity: dict[str, object] = {
        "image": "registry.example/p23@sha256:" + "1" * 64,
        "repository_digest": "sha256:" + "1" * 64,
        "container_id": "a" * 64,
        "image_id": "sha256:" + "b" * 64,
        "container_init_pid": 1234,
        "hostname": "a" * 12,
        "default_hostname": True,
        "rootfs_read_only": True,
        "network_mode": P23.PINNED_CONTAINER_NETWORK_MODE,
        "mountinfo_sha256": hashlib.sha256(_READ_ONLY_MOUNTINFO).hexdigest(),
        "mount_contract": copy.deepcopy(P23._SANITIZED_MOUNT_CONTRACT),
        "tmpfs_contract": copy.deepcopy(P23._SANITIZED_TMPFS_CONTRACT),
        "attestation_scope": P23.CONTAINER_ATTESTATION_SCOPE,
    }
    if include_attestation_hash:
        identity["host_attestation_sha256"] = "2" * 64
    return identity


def _running_mounts() -> list[dict[str, object]]:
    return [
        {
            "Type": "bind",
            "Source": f"/secure/p23/{name}",
            "Destination": destination,
            "RW": not read_only,
            "Propagation": "rprivate",
        }
        for name, destination, read_only in P23.REQUIRED_CONTAINER_MOUNTS
    ]


def _valid_runtime_lock() -> dict[str, object]:
    return {
        "schema_version": P23.P23_RUNTIME_LOCK_SCHEMA,
        "status": "pinned_for_acquisition",
        "p23_addendum_sha256": P23.sha256_file(P23.DEFAULT_ADDENDUM_PATH),
        "container": _container_identity(),
        "gpu": {
            "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "pci_bus_id": "00000000:01:00.0",
            "name": "Pinned BF16 GPU",
            "vbios_version": "00.00.00.00.00",
            "driver_version": "999.1",
            "compute_capability": [9, 0],
            "multiprocessor_count": 100,
            "total_memory": 80_000_000_000,
            "nvidia_smi_memory_total_mib": 76_294,
            "mig_mode": "disabled",
        },
        "software": {
            "python": "3.12-test",
            "python_executable": P23.PINNED_PYTHON_EXECUTABLE,
            "python_executable_sha256": "8" * 64,
            "python_flags": copy.deepcopy(P23.FROZEN_PYTHON_FLAGS),
            "torch_version": "2.test",
            "torch_git_version": "3" * 40,
            "torch_config_sha256": "5" * 64,
            "torch_cuda_compiled_version": "12.test",
            "cuda_runtime_version": 12_080,
            "cudnn_version": 90_000,
            "numpy_version": "2.test",
            "platform": "Linux-test",
            "system": "Linux",
            "release": "kernel-test",
            "version": "kernel-version-test",
            "machine": "x86_64",
            "processor": "Pinned CPU",
            "libc": "glibc-test",
            "cpu_identity": _cpu_identity(),
            "proc_version_sha256": "7" * 64,
        },
        "determinism": {
            "cublas_workspace_config": ":4096:8",
            "pythonhashseed": "1337",
            "nvidia_tf32_override": "0",
            "cuda_visible_devices": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "nvidia_visible_devices": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "sdpa_backend": "math",
            "deterministic_algorithms": True,
            "deterministic_debug_mode": "error",
            "tf32": False,
            "cudnn_benchmark": False,
            "cudnn_deterministic": True,
            "float32_matmul_precision": "highest",
            "fp16_reduced_precision_reduction": False,
            "bf16_reduced_precision_reduction": False,
            "cpu_threads": 1,
            "interop_threads": 1,
        },
        "loader_environment": {
            "LD_PRELOAD": None,
            "LD_LIBRARY_PATH": None,
            "LD_AUDIT": None,
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONOPTIMIZE": None,
        },
        "runtime_lock_sha256": "4" * 64,
    }


def _valid_runtime() -> dict[str, object]:
    lock = _valid_runtime_lock()
    return {
        "backend": "cuda",
        "live_hostname": "a" * 12,
        "live_mountinfo_sha256": hashlib.sha256(_READ_ONLY_MOUNTINFO).hexdigest(),
        "python_version": "3.12-test",
        "python_executable": P23.PINNED_PYTHON_EXECUTABLE,
        "python_executable_sha256": "8" * 64,
        "python_flags": copy.deepcopy(P23.FROZEN_PYTHON_FLAGS),
        "platform": "Linux-test",
        "system": "Linux",
        "release": "kernel-test",
        "version": "kernel-version-test",
        "machine": "x86_64",
        "processor": "Pinned CPU",
        "libc": "glibc-test",
        "cpu_identity": _cpu_identity(),
        "proc_version_sha256": "7" * 64,
        "numpy_version": "2.test",
        "torch_version": "2.test",
        "torch_git_version": "3" * 40,
        "torch_cuda_compiled_version": "12.test",
        "cuda_runtime_version": 12_080,
        "cudnn_version": 90_000,
        "torch_config_sha256": "5" * 64,
        "gpu": {
            "logical_index": 0,
            "visible_device_count": 1,
            "uuid": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "name": "Pinned BF16 GPU",
            "pci_bus_id": "00000000:01:00.0",
            "driver_version": "999.1",
            "vbios_version": "00.00.00.00.00",
            "compute_capability": [9, 0],
            "multiprocessor_count": 100,
            "total_memory_bytes": 80_000_000_000,
            "nvidia_smi_memory_total_mib": 76_294,
            "mig_mode": "disabled",
            "bf16_supported": True,
        },
        "container": lock,
        "sdpa": {
            "selected_backend": "math",
            "math_enabled": True,
            "flash_enabled": False,
            "memory_efficient_enabled": False,
            "cudnn_enabled": False,
        },
        "determinism": {
            "deterministic_algorithms": True,
            "deterministic_debug_mode": 2,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "allow_tf32_matmul": False,
            "allow_tf32_cudnn": False,
            "float32_matmul_precision": "highest",
            "allow_fp16_reduced_precision_reduction": False,
            "allow_bf16_reduced_precision_reduction": False,
            "torch_num_threads": 1,
            "torch_num_interop_threads": 1,
        },
        "environment": {
            "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "NVIDIA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
            "PYTHONHASHSEED": "1337",
            "NVIDIA_TF32_OVERRIDE": "0",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "VIRTUAL_ENV": P23.PINNED_PYTHON_ENVIRONMENT,
            "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONOPTIMIZE": None,
            "PYTHONPATH": None,
            "PYTHONHOME": None,
            "LD_PRELOAD": None,
            "LD_LIBRARY_PATH": None,
            "LD_AUDIT": None,
        },
    }


def _write_bound_host_attestation(directory: Path, lock: dict[str, object]) -> Path:
    attestation = {
        "schema_version": P23.P23_HOST_ATTESTATION_SCHEMA,
        "status": "procedurally_host_attested",
        "container": _container_identity(include_attestation_hash=False),
        "gpu": copy.deepcopy(lock["gpu"]),
        "evidence": {
            "image_inspection_sha256": _digest("image-inspection"),
            "running_container_inspection_sha256": _digest("running-container-inspection"),
            "running_mountinfo_sha256": hashlib.sha256(_READ_ONLY_MOUNTINFO).hexdigest(),
            "nvidia_smi_query_sha256": _digest("nvidia-smi-query"),
        },
    }
    path = directory / "host-attestation.json"
    path.write_text(json.dumps(attestation, sort_keys=True))
    lock["container"]["host_attestation_sha256"] = P23.sha256_file(path)
    return path


def _identity() -> dict[str, object]:
    snapshot = P23.collect_loaded_file_snapshot(
        {"image": Path("/")},
        modules={},
        proc_maps_text="",
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )
    return P23.build_execution_identity(
        addendum_path=P23.DEFAULT_ADDENDUM_PATH,
        repository={
            "head": "a" * 40,
            "tree": "b" * 40,
            "origin": "https://example.invalid/repository.git",
            "clean": True,
            "tracked_files": {"x": {"sha256": "c" * 64, "byte_count": 1}},
        },
        source_modules={
            "runner": {
                "scope": "repository",
                "logical_path": "repository:runner.py",
                "sha256": "d" * 64,
                "byte_count": 1,
                "source_kind": "python",
            }
        },
        loaded_file_closure=P23.build_loaded_file_closure(snapshot, snapshot),
        runtime=_valid_runtime(),
        static_contract={"seed": 1337, "steps": 256},
    )


def _manifest(mode: str, identity: dict[str, object] | None = None) -> dict[str, object]:
    identity = _identity() if identity is None else identity
    on = mode == "trace_on"
    initial_state = {
        "model_state_sha256": _digest("initial-model"),
        "optimizer_state_sha256": _digest("initial-optimizer"),
        "rng_state_sha256": _digest("initial-rng"),
    }
    steps: list[dict[str, object]] = []
    for step in range(P23.EXPECTED_OPTIMIZER_STEPS):
        checkpoint = step in P23.EXPECTED_CAPTURE_STEPS
        capture = on and checkpoint
        steps.append(
            {
                "step": step,
                "tokens_seen": (step + 1) * 128,
                "data_token_offset": step * 128,
                "batch_sha256": _digest(f"batch:{step}"),
                "loss_tensor_sha256": _digest(f"loss:{step}"),
                "rng_before_step_sha256": _digest(f"rng-before:{step}"),
                "rng_after_step_sha256": _digest(f"rng-after:{step}"),
                "state_checkpoint_present": checkpoint,
                "model_state_sha256": _digest(f"model:{step}") if checkpoint else None,
                "optimizer_state_sha256": (_digest(f"optimizer:{step}") if checkpoint else None),
                "observation_count": 48 if capture else 0,
                "capture": (
                    {
                        "parameter_count": 48,
                        "actual_pre_aspect_candidate": False,
                        "pre_aspect_unavailable_reason": "pinned call returns post-aspect only",
                        "actual_post_aspect_candidate": True,
                        "actual_post_aspect_cuda_candidate": True,
                        "p20_shadow_only": True,
                        "record_sha256": _digest(f"capture:{step}"),
                    }
                    if capture
                    else None
                ),
                "observer_rng_unchanged": True if capture else None,
            }
        )
    return {
        "schema_version": P23.P23_RUN_MANIFEST_SCHEMA,
        "trace_mode": mode,
        "run_identity_sha256": P23.execution_identity_sha256(identity),
        "execution_identity": identity,
        "initial_state": initial_state,
        "initial_state_sha256": P23._p22_initial_state_sha256(initial_state),
        "shadow_update_applied": False,
        "candidate_execution": {
            "backend": "cuda",
            "actual_accelerator_candidate_computed_and_applied": True,
            "shield_shadow_only": True,
        },
        "candidate_observation": (
            copy.deepcopy(P23.TRACE_ON_OBSERVATION)
            if on
            else copy.deepcopy(P23.TRACE_OFF_OBSERVATION)
        ),
        "capture_steps": list(P23.EXPECTED_CAPTURE_STEPS) if on else [],
        "raw_trace": ({"observation_count": 1_152, "sha256": _digest("raw-trace")} if on else None),
        "steps": steps,
        "final_state": {
            "model_state_sha256": _digest("final-model"),
            "optimizer_state_sha256": _digest("final-optimizer"),
            "rng_state_sha256": _digest("final-rng"),
        },
        "artifact_path": "/workspace/results/run.json",
    }


def test_committed_addendum_binds_all_four_frozen_artifacts() -> None:
    addendum = P23.load_and_validate_addendum()
    assert addendum["schema_version"] == P23.P23_ADDENDUM_SCHEMA
    assert set(addendum["inherits"]) == {
        "p21_fidelity_gates",
        "p22_protocol",
        "p22_erratum",
        "p22_fused_qkv_certificate",
    }
    assert addendum["capture_semantics"]["trace_off"]["status"] == "not_observed"
    assert addendum["capture_semantics"]["trace_on"]["observation_count"] == 1_152
    assert "fresh Python process" in addendum["run_sequence"]["process_isolation"]
    assert "same host-inspected container ID" in addendum["run_sequence"]["process_isolation"]


def test_addendum_rejects_one_changed_inherited_hash(tmp_path: Path) -> None:
    payload = json.loads(P23.DEFAULT_ADDENDUM_PATH.read_text())
    payload["inherits"]["p22_protocol"]["sha256"] = "0" * 64
    changed = tmp_path / "changed-addendum.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(P23.P23ProvenanceError, match=r"p22_protocol.*mismatch"):
        P23.load_and_validate_addendum(changed)


def test_addendum_rejects_changed_process_isolation_contract(tmp_path: Path) -> None:
    payload = json.loads(P23.DEFAULT_ADDENDUM_PATH.read_text())
    payload["run_sequence"]["process_isolation"] = "fresh container process"
    changed = tmp_path / "changed-addendum.json"
    changed.write_text(json.dumps(payload))
    with pytest.raises(P23.P23ProvenanceError, match="process-isolation"):
        P23.load_and_validate_addendum(changed)


def test_git_provenance_hashes_every_tracked_file_and_rejects_dirty_tree(
    tmp_path: Path,
) -> None:
    (tmp_path / "a.txt").write_text("a")
    (tmp_path / "b.py").write_text("print('b')\n")

    def clean_runner(arguments: tuple[str, ...], cwd: Path | None) -> bytes:
        assert cwd == tmp_path.resolve()
        command = arguments[1:]
        values = {
            ("rev-parse", "--show-toplevel"): str(tmp_path.resolve()).encode() + b"\n",
            ("rev-parse", "HEAD"): b"a" * 40 + b"\n",
            ("rev-parse", "HEAD^{tree}"): b"b" * 40 + b"\n",
            ("remote", "get-url", "origin"): b"https://example.invalid/repo.git\n",
            ("status", "--porcelain=v1", "--untracked-files=all", "-z"): b"",
            ("submodule", "status", "--recursive"): b"",
            ("ls-files", "-z"): b"a.txt\0b.py\0",
        }
        return values[command]

    provenance = P23.collect_git_provenance(tmp_path, command_runner=clean_runner)
    assert provenance["clean"] is True
    assert provenance["tracked_file_count"] == 2
    assert provenance["tracked_files"]["a.txt"] == {
        "sha256": hashlib.sha256(b"a").hexdigest(),
        "byte_count": 1,
        "kind": "file",
    }

    def dirty_runner(arguments: tuple[str, ...], cwd: Path | None) -> bytes:
        if arguments[1] == "status":
            return b"?? local-output.json\0"
        return clean_runner(arguments, cwd)

    with pytest.raises(P23.P23ProvenanceError, match="clean worktree"):
        P23.collect_git_provenance(tmp_path, command_runner=dirty_runner)


def test_git_provenance_rejects_credentials_and_detects_post_hash_byte_drift(
    tmp_path: Path,
) -> None:
    tracked = tmp_path / "tracked.py"
    tracked.write_text("before\n")
    ls_files_calls = 0

    def runner(arguments: tuple[str, ...], cwd: Path | None) -> bytes:
        nonlocal ls_files_calls
        assert cwd == tmp_path.resolve()
        command = arguments[1:]
        if command == ("rev-parse", "--show-toplevel"):
            return str(tmp_path.resolve()).encode() + b"\n"
        if command == ("rev-parse", "HEAD"):
            return b"a" * 40 + b"\n"
        if command == ("rev-parse", "HEAD^{tree}"):
            return b"b" * 40 + b"\n"
        if command == ("remote", "get-url", "origin"):
            return b"https://example.invalid/repo.git\n"
        if command == ("status", "--porcelain=v1", "--untracked-files=all", "-z"):
            return b""
        if command == ("submodule", "status", "--recursive"):
            return b""
        if command == ("ls-files", "-z"):
            ls_files_calls += 1
            if ls_files_calls == 2:
                tracked.write_text("after\n")
            return b"tracked.py\0"
        raise AssertionError(command)

    with pytest.raises(P23.P23ProvenanceError, match="tracked repository bytes changed"):
        P23.collect_git_provenance(tmp_path, command_runner=runner)

    def credential_runner(arguments: tuple[str, ...], cwd: Path | None) -> bytes:
        if arguments[1:] == ("remote", "get-url", "origin"):
            return b"https://secret-token@example.invalid/repo.git\n"
        return runner(arguments, cwd)

    with pytest.raises(P23.P23ProvenanceError) as error:
        P23.collect_git_provenance(tmp_path, command_runner=credential_runner)
    assert "secret-token" not in str(error.value)


def test_loaded_module_map_is_complete_scoped_and_frozen_after_initialization(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    nanogpt = tmp_path / "nanogpt"
    muon = tmp_path / "muon.py"
    repository.mkdir()
    nanogpt.mkdir()
    paths = {
        "project.runner": repository / "runner.py",
        "pinned.nanogpt": nanogpt / "model.py",
        "pinned.muon": muon,
    }
    modules: dict[str, ModuleType] = {}
    for name, path in paths.items():
        path.write_text(f"# {name}\n")
        module = ModuleType(name)
        module.__file__ = str(path)
        modules[name] = module
    scopes = {"repository": repository, "nanogpt": nanogpt, "muon": muon}
    initialized = P23.collect_runtime_module_hashes(scopes, modules=modules)
    assert set(initialized) == set(paths)
    assert {record["scope"] for record in initialized.values()} == {
        "repository",
        "nanogpt",
        "muon",
    }
    P23.assert_runtime_module_map_unchanged(initialized, copy.deepcopy(initialized))

    late_path = repository / "late.py"
    late_path.write_text("# forbidden lazy project import\n")
    late = ModuleType("project.late")
    late.__file__ = str(late_path)
    modules[late.__name__] = late
    final = P23.collect_runtime_module_hashes(scopes, modules=modules)
    with pytest.raises(P23.P23ProvenanceError, match=r"project\.late"):
        P23.assert_runtime_module_map_unchanged(initialized, final)

    modules.pop("project.late")
    paths["project.runner"].write_text("# bytes changed after initialization\n")
    byte_changed = P23.collect_runtime_module_hashes(scopes, modules=modules)
    with pytest.raises(P23.P23ProvenanceError, match="runtime module map changed"):
        P23.assert_runtime_module_map_unchanged(initialized, byte_changed)


def test_loaded_module_map_skips_torch_proxies_and_external_image_environment(
    tmp_path: Path,
) -> None:
    import torch

    repository = tmp_path / "repository"
    nanogpt = tmp_path / "nanogpt"
    repository.mkdir()
    nanogpt.mkdir()
    runner_path = repository / "runner.py"
    runner_path.write_text("# tracked runner\n")
    ignored_path = tmp_path / "p23-venv" / "ignored.py"
    ignored_path.parent.mkdir()
    ignored_path.write_text("# image-bound dependency\n")
    nanogpt_path = nanogpt / "model.py"
    nanogpt_path.write_text("# pinned nanoGPT module\n")
    muon_path = tmp_path / "muon.py"
    muon_path.write_text("# pinned Muon module\n")

    modules = dict(sys.modules)
    for name, path in {
        "p23.test.runner": runner_path,
        "p23.test.ignored_environment": ignored_path,
        "p23.test.nanogpt": nanogpt_path,
        "p23.test.muon": muon_path,
    }.items():
        module = ModuleType(name)
        module.__file__ = str(path)
        modules[name] = module

    records = P23.collect_runtime_module_hashes(
        {"repository": repository, "nanogpt": nanogpt, "muon": muon_path},
        modules=modules,
        scope_file_allowlists={"repository": frozenset({"runner.py"})},
    )

    assert "p23.test.runner" in records
    assert "p23.test.nanogpt" in records
    assert "p23.test.muon" in records
    assert "p23.test.ignored_environment" not in records
    assert "torch.ops" not in records
    assert "torch.classes" not in records
    assert torch.ops.__file__ == "_ops.py"


def test_git_provenance_rejects_repository_virtual_environment(tmp_path: Path) -> None:
    (tmp_path / ".venv").mkdir()

    def must_not_run(arguments: tuple[str, ...], cwd: Path | None) -> bytes:
        raise AssertionError(f"Git must not run: {arguments}, {cwd}")

    with pytest.raises(P23.P23ProvenanceError, match=r"repository-local \.venv"):
        P23.collect_git_provenance(tmp_path, command_runner=must_not_run)


def test_loaded_module_map_rejects_missing_claimed_controlled_source(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    nanogpt = tmp_path / "nanogpt"
    repository.mkdir()
    nanogpt.mkdir()
    muon = tmp_path / "muon.py"
    muon.write_text("# pinned Muon module\n")
    missing = ModuleType("p23.test.missing")
    missing.__file__ = str(repository / "missing.py")

    with pytest.raises(P23.P23ProvenanceError, match="loaded module file is unavailable"):
        P23.collect_runtime_module_hashes(
            {"repository": repository, "nanogpt": nanogpt, "muon": muon},
            modules={missing.__name__: missing},
        )


def test_loaded_module_map_rejects_untracked_repository_shadow_code(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    nanogpt = tmp_path / "nanogpt"
    repository.mkdir()
    nanogpt.mkdir()
    runner_path = repository / "runner.py"
    runner_path.write_text("# tracked runner\n")
    shadow_path = repository / "ignored" / "shadow.py"
    shadow_path.parent.mkdir()
    shadow_path.write_text("# ignored but executable shadow code\n")
    nanogpt_path = nanogpt / "model.py"
    nanogpt_path.write_text("# pinned nanoGPT module\n")
    muon_path = tmp_path / "muon.py"
    muon_path.write_text("# pinned Muon module\n")

    modules: dict[str, ModuleType] = {}
    for name, path in {
        "p23.test.runner": runner_path,
        "p23.test.ignored_shadow": shadow_path,
        "p23.test.nanogpt": nanogpt_path,
        "p23.test.muon": muon_path,
    }.items():
        module = ModuleType(name)
        module.__file__ = str(path)
        modules[name] = module

    with pytest.raises(
        P23.P23ProvenanceError,
        match=r"ignored_shadow.*outside the repository source allowlist",
    ):
        P23.collect_runtime_module_hashes(
            {"repository": repository, "nanogpt": nanogpt, "muon": muon_path},
            modules=modules,
            scope_file_allowlists={"repository": frozenset({"runner.py"})},
        )


def test_runtime_lock_template_fails_and_populated_lock_binds_addendum(tmp_path: Path) -> None:
    with pytest.raises(P23.P23ProvenanceError, match="template"):
        P23.load_runtime_lock(ROOT / "experiments/training/p23_cuda_runtime_lock.template.json")
    lock = _valid_runtime_lock()
    lock.pop("runtime_lock_sha256")
    attestation_path = _write_bound_host_attestation(tmp_path, lock)
    path = tmp_path / "runtime-lock.json"
    path.write_text(json.dumps(lock))
    with pytest.raises(P23.P23ProvenanceError, match="host-attestation artifact"):
        P23.load_runtime_lock(path)
    loaded = P23.load_runtime_lock(path, host_attestation_path=attestation_path)
    assert loaded["container"]["repository_digest"] == "sha256:" + "1" * 64
    assert loaded["runtime_lock_sha256"] == P23.sha256_file(path)


def test_sanitized_host_attestation_is_hash_bound_to_runtime_lock(tmp_path: Path) -> None:
    lock = _valid_runtime_lock()
    lock.pop("runtime_lock_sha256")
    attestation_path = _write_bound_host_attestation(tmp_path, lock)
    lock_path = tmp_path / "runtime-lock.json"
    lock_path.write_text(json.dumps(lock, sort_keys=True))
    loaded = P23.load_runtime_lock(lock_path, host_attestation_path=attestation_path)
    assert loaded["container"]["host_attestation_sha256"] == P23.sha256_file(attestation_path)

    attestation = json.loads(attestation_path.read_text())
    attestation["gpu"]["driver_version"] = "different"
    attestation_path.write_text(json.dumps(attestation, sort_keys=True))
    with pytest.raises(P23.P23ProvenanceError, match="host-attestation bytes"):
        P23.load_runtime_lock(lock_path, host_attestation_path=attestation_path)


def test_cuda_runtime_validation_is_injectable_and_rejects_backend_drift() -> None:
    runtime = _valid_runtime()
    observed = P23.collect_cuda_runtime(
        container_identity=_valid_runtime_lock(), injected_probe=lambda: runtime
    )
    assert observed == runtime
    P23.validate_runtime_against_lock(runtime, _valid_runtime_lock())

    bad = copy.deepcopy(runtime)
    bad["sdpa"]["flash_enabled"] = True
    with pytest.raises(P23.P23ProvenanceError, match="math-only"):
        P23.validate_cuda_runtime_map(bad)
    bad = copy.deepcopy(runtime)
    bad["gpu"]["driver_version"] = "different"
    with pytest.raises(P23.P23ProvenanceError, match="driver_version"):
        P23.validate_runtime_against_lock(bad, _valid_runtime_lock())
    with pytest.raises(P23.P23ProvenanceError, match="driver_version"):
        P23.collect_cuda_runtime(
            container_identity=_valid_runtime_lock(), injected_probe=lambda: bad
        )
    bad = copy.deepcopy(runtime)
    bad["python_executable"] = "/usr/bin/python"
    with pytest.raises(P23.P23ProvenanceError, match="pinned Python executable"):
        P23.validate_cuda_runtime_map(bad)
    bad = copy.deepcopy(runtime)
    bad["python_executable_sha256"] = "invalid"
    with pytest.raises(P23.P23ProvenanceError, match="Python executable digest"):
        P23.validate_cuda_runtime_map(bad)
    bad = copy.deepcopy(runtime)
    bad["live_hostname"] = "different-host"
    with pytest.raises(P23.P23ProvenanceError, match="live process hostname"):
        P23.validate_cuda_runtime_map(bad)
    bad = copy.deepcopy(runtime)
    bad["live_mountinfo_sha256"] = _digest("different-mount-namespace")
    with pytest.raises(P23.P23ProvenanceError, match="live mount namespace"):
        P23.validate_cuda_runtime_map(bad)
    bad = copy.deepcopy(runtime)
    bad["environment"]["LD_LIBRARY_PATH"] = "/shadow"
    with pytest.raises(P23.P23ProvenanceError, match="environment map"):
        P23.validate_cuda_runtime_map(bad)


def test_cuda_runtime_version_uses_runtime_api(monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeVersionFunction:
        argtypes: object = None
        restype: object = None

        def __call__(self, pointer: object) -> int:
            pointer._obj.value = 12_080
            return 0

    fake_library = SimpleNamespace(cudaRuntimeGetVersion=FakeVersionFunction())
    monkeypatch.setattr(P23.ctypes.util, "find_library", lambda name: "libcudart-pinned.so")
    monkeypatch.setattr(P23.ctypes, "CDLL", lambda name: fake_library)
    assert P23._default_cuda_runtime_version_probe() == 12_080


def test_freeze_runtime_builds_both_valid_artifacts_without_hand_authored_json(
    tmp_path: Path,
) -> None:
    digest = "sha256:" + "1" * 64
    image = f"registry.example/p23@{digest}"
    image_id = "sha256:" + "b" * 64
    image_inspection = tmp_path / "host-image-inspection.json"
    image_inspection.write_text(json.dumps({"Id": image_id, "RepoDigests": [image]}))
    running_inspection = tmp_path / "host-running-container-inspection.json"
    running_inspection.write_text(
        json.dumps(
            {
                "Id": "a" * 64,
                "Image": image_id,
                "State": {"Running": True, "Pid": 1234},
                "Config": {"Hostname": "a" * 12, "Image": image},
                "HostConfig": {
                    "ReadonlyRootfs": True,
                    "NetworkMode": P23.PINNED_CONTAINER_NETWORK_MODE,
                    "Tmpfs": copy.deepcopy(P23.REQUIRED_CONTAINER_TMPFS),
                },
                "Mounts": _running_mounts(),
            }
        )
    )
    running_mountinfo = tmp_path / "host-running-mountinfo.txt"
    running_mountinfo.write_bytes(_READ_ONLY_MOUNTINFO)
    nvidia = tmp_path / "host-nvidia-smi.csv"
    nvidia.write_text(
        "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee, "
        "00000000:01:00.0, Pinned BF16 GPU, 999.1, 00.00.00.00.00, 76294, Disabled\n"
    )
    attestation_output = tmp_path / "p23-host.json"
    lock_output = tmp_path / "p23-lock.json"
    components = _valid_runtime()
    components.pop("container")
    environment = copy.deepcopy(_valid_runtime()["environment"])
    result = P23.freeze_runtime_artifacts(
        container_image=image,
        container_repository_digest=digest,
        host_image_inspection=image_inspection,
        host_running_container_inspection=running_inspection,
        host_running_mountinfo=running_mountinfo,
        host_nvidia_smi_query=nvidia,
        host_attestation_output=attestation_output,
        runtime_lock_output=lock_output,
        torch_module=_FakeTorch,
        environ=environment,
        python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
        runtime_component_probe=lambda: components,
    )
    assert result["next_action"] == "review_and_commit_both_artifacts_before_any_acquisition"
    assert result["host_attestation"]["sha256"] == P23.sha256_file(attestation_output)
    loaded = P23.load_runtime_lock(
        lock_output,
        host_attestation_path=attestation_output,
    )
    assert loaded["software"]["cuda_runtime_version"] == 12_080
    assert loaded["software"]["cpu_identity"] == _cpu_identity()

    original_running = json.loads(running_inspection.read_text())

    def reject_running_inspection(payload: dict[str, object], *, label: str, message: str) -> None:
        running_inspection.write_text(json.dumps(payload))
        with pytest.raises(P23.P23ProvenanceError, match=message):
            P23.freeze_runtime_artifacts(
                container_image=image,
                container_repository_digest=digest,
                host_image_inspection=image_inspection,
                host_running_container_inspection=running_inspection,
                host_running_mountinfo=running_mountinfo,
                host_nvidia_smi_query=nvidia,
                host_attestation_output=tmp_path / f"{label}-host.json",
                runtime_lock_output=tmp_path / f"{label}-lock.json",
                torch_module=_FakeTorch,
                environ=environment,
                runtime_component_probe=lambda: components,
            )

    wrong_image = copy.deepcopy(original_running)
    wrong_image["Image"] = "sha256:" + "c" * 64
    reject_running_inspection(wrong_image, label="wrong-image", message="immutable image ID")

    wrong_hostname = copy.deepcopy(original_running)
    wrong_hostname["Config"]["Hostname"] = "custom-host"
    reject_running_inspection(wrong_hostname, label="wrong-host", message="default hostname")

    writable_source = copy.deepcopy(original_running)
    writable_source["Mounts"][0]["RW"] = True
    reject_running_inspection(
        writable_source, label="writable-source", message="wrong type or mode"
    )

    writable_root = copy.deepcopy(original_running)
    writable_root["HostConfig"]["ReadonlyRootfs"] = False
    reject_running_inspection(writable_root, label="writable-root", message="root filesystem")

    wrong_tmpfs = copy.deepcopy(original_running)
    wrong_tmpfs["HostConfig"]["Tmpfs"] = {}
    reject_running_inspection(wrong_tmpfs, label="wrong-tmpfs", message="tmpfs map")

    networked = copy.deepcopy(original_running)
    networked["HostConfig"]["NetworkMode"] = "default"
    reject_running_inspection(networked, label="networked", message="network mode")

    running_inspection.write_text(json.dumps(original_running))
    wrong_namespace_components = copy.deepcopy(components)
    wrong_namespace_components["live_mountinfo_sha256"] = _digest("different-mount-namespace")
    with pytest.raises(P23.P23ProvenanceError, match="live mount namespace"):
        P23.freeze_runtime_artifacts(
            container_image=image,
            container_repository_digest=digest,
            host_image_inspection=image_inspection,
            host_running_container_inspection=running_inspection,
            host_running_mountinfo=running_mountinfo,
            host_nvidia_smi_query=nvidia,
            host_attestation_output=tmp_path / "wrong-namespace-host.json",
            runtime_lock_output=tmp_path / "wrong-namespace-lock.json",
            torch_module=_FakeTorch,
            environ=environment,
            python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
            runtime_component_probe=lambda: wrong_namespace_components,
        )

    with pytest.raises(P23.P23ProvenanceError, match="must not already exist"):
        P23.freeze_runtime_artifacts(
            container_image=image,
            container_repository_digest=digest,
            host_image_inspection=image_inspection,
            host_running_container_inspection=running_inspection,
            host_running_mountinfo=running_mountinfo,
            host_nvidia_smi_query=nvidia,
            host_attestation_output=attestation_output,
            runtime_lock_output=lock_output,
            torch_module=_FakeTorch,
            environ=environment,
            python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
            runtime_component_probe=lambda: components,
        )


class _FakeMatmul:
    allow_tf32 = True
    allow_fp16_reduced_precision_reduction = True
    allow_bf16_reduced_precision_reduction = True


class _FakeCudaBackend:
    matmul = _FakeMatmul()
    _math = False
    _flash = True
    _efficient = True
    _cudnn = True

    @classmethod
    def enable_math_sdp(cls, value: bool) -> None:
        cls._math = value

    @classmethod
    def enable_flash_sdp(cls, value: bool) -> None:
        cls._flash = value

    @classmethod
    def enable_mem_efficient_sdp(cls, value: bool) -> None:
        cls._efficient = value

    @classmethod
    def enable_cudnn_sdp(cls, value: bool) -> None:
        cls._cudnn = value

    math_sdp_enabled = classmethod(lambda cls: cls._math)
    flash_sdp_enabled = classmethod(lambda cls: cls._flash)
    mem_efficient_sdp_enabled = classmethod(lambda cls: cls._efficient)
    cudnn_sdp_enabled = classmethod(lambda cls: cls._cudnn)


class _FakeCudnn:
    allow_tf32 = True
    benchmark = True
    deterministic = False


class _FakeTorch:
    backends = SimpleNamespace(cuda=_FakeCudaBackend, cudnn=_FakeCudnn)
    cuda = SimpleNamespace(is_initialized=lambda: False)
    _algorithms = False
    _debug = 0
    _precision = "medium"
    _threads = 2
    _interop = 2

    @classmethod
    def use_deterministic_algorithms(cls, value: bool) -> None:
        cls._algorithms = value

    @classmethod
    def are_deterministic_algorithms_enabled(cls) -> bool:
        return cls._algorithms

    @classmethod
    def set_deterministic_debug_mode(cls, value: str) -> None:
        cls._debug = 2 if value == "error" else 0

    @classmethod
    def get_deterministic_debug_mode(cls) -> int:
        return cls._debug

    @classmethod
    def set_float32_matmul_precision(cls, value: str) -> None:
        cls._precision = value

    @classmethod
    def get_float32_matmul_precision(cls) -> str:
        return cls._precision

    @classmethod
    def set_num_threads(cls, value: int) -> None:
        cls._threads = value

    @classmethod
    def get_num_threads(cls) -> int:
        return cls._threads

    @classmethod
    def set_num_interop_threads(cls, value: int) -> None:
        cls._interop = value

    @classmethod
    def get_num_interop_threads(cls) -> int:
        return cls._interop


def test_configure_cuda_determinism_runs_before_cuda_initialization() -> None:
    environment = {
        "CUDA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "NVIDIA_VISIBLE_DEVICES": "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "1337",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "VIRTUAL_ENV": P23.PINNED_PYTHON_ENVIRONMENT,
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": None,
        "PYTHONPATH": None,
        "PYTHONHOME": None,
        "LD_PRELOAD": None,
        "LD_LIBRARY_PATH": None,
        "LD_AUDIT": None,
    }
    configured = P23.configure_cuda_determinism(
        torch_module=_FakeTorch,
        environ=environment,
        python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
    )
    assert configured == {
        "deterministic_algorithms": True,
        "deterministic_debug_mode": 2,
        "math_sdpa": True,
        "flash_sdpa": False,
        "memory_efficient_sdpa": False,
        "cudnn_sdpa": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "allow_tf32_matmul": False,
        "allow_tf32_cudnn": False,
        "float32_matmul_precision": "highest",
        "allow_fp16_reduced_precision_reduction": False,
        "allow_bf16_reduced_precision_reduction": False,
        "torch_num_threads": 1,
        "torch_num_interop_threads": 1,
        "cuda_initialized": False,
    }
    with pytest.raises(P23.P23ProvenanceError, match="PYTHONHASHSEED"):
        P23.configure_cuda_determinism(
            torch_module=_FakeTorch,
            environ={**environment, "PYTHONHASHSEED": "0"},
            python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
        )
    for invalid_uuid in ("GPU-a", "GPU-------------------------------------", "MIG-abc"):
        with pytest.raises(P23.P23ProvenanceError, match="full GPU UUID"):
            P23.configure_cuda_determinism(
                torch_module=_FakeTorch,
                environ={
                    **environment,
                    "CUDA_VISIBLE_DEVICES": invalid_uuid,
                    "NVIDIA_VISIBLE_DEVICES": invalid_uuid,
                },
                python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
            )
    with pytest.raises(P23.P23ProvenanceError, match="PYTHONPATH"):
        P23.configure_cuda_determinism(
            torch_module=_FakeTorch,
            environ={**environment, "PYTHONPATH": "/shadow"},
            python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
        )
    with pytest.raises(P23.P23ProvenanceError, match="LD_PRELOAD"):
        P23.configure_cuda_determinism(
            torch_module=_FakeTorch,
            environ={**environment, "LD_PRELOAD": "/shadow.so"},
            python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
        )
    with pytest.raises(P23.P23ProvenanceError, match="interpreter flags"):
        P23.configure_cuda_determinism(
            torch_module=_FakeTorch,
            environ=environment,
            python_flags=SimpleNamespace(**{**P23.FROZEN_PYTHON_FLAGS, "optimize": 1}),
        )
    with pytest.raises(P23.P23ProvenanceError, match="PYTHONOPTIMIZE"):
        P23.configure_cuda_determinism(
            torch_module=_FakeTorch,
            environ={**environment, "PYTHONOPTIMIZE": "1"},
            python_flags=SimpleNamespace(**P23.FROZEN_PYTHON_FLAGS),
        )


def test_execution_identity_is_recomputed_and_maps_are_checked_independently() -> None:
    identity = _identity()
    manifest = _manifest("trace_off", identity)
    report = P23.verify_execution_identity(
        manifest,
        addendum_path=P23.DEFAULT_ADDENDUM_PATH,
        repository_collector=lambda: identity["repository"],
        source_collector=lambda: identity["source_modules"],
        loaded_file_initialized_collector=lambda: identity["loaded_file_closure"]["initialized"],
        loaded_file_roots={"image": Path("/")},
        runtime_collector=lambda: identity["runtime"],
        expected_static_contract=identity["static_contract"],
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )
    assert report["passes"] is True

    forged = copy.deepcopy(manifest)
    forged["run_identity_sha256"] = "0" * 64
    with pytest.raises(P23.P23ProvenanceError, match="digest"):
        P23.verify_execution_identity(
            forged,
            addendum_path=P23.DEFAULT_ADDENDUM_PATH,
            repository_collector=lambda: identity["repository"],
            source_collector=lambda: identity["source_modules"],
            loaded_file_initialized_collector=lambda: identity["loaded_file_closure"][
                "initialized"
            ],
            loaded_file_roots={"image": Path("/")},
            runtime_collector=lambda: identity["runtime"],
            expected_static_contract=identity["static_contract"],
            mountinfo_text=_READ_ONLY_MOUNTINFO,
        )
    with pytest.raises(P23.P23ProvenanceError, match="source_modules"):
        P23.verify_execution_identity(
            manifest,
            addendum_path=P23.DEFAULT_ADDENDUM_PATH,
            repository_collector=lambda: identity["repository"],
            source_collector=lambda: {},
            loaded_file_initialized_collector=lambda: identity["loaded_file_closure"][
                "initialized"
            ],
            loaded_file_roots={"image": Path("/")},
            runtime_collector=lambda: identity["runtime"],
            expected_static_contract=identity["static_contract"],
            mountinfo_text=_READ_ONLY_MOUNTINFO,
        )


def test_cuda_only_manifest_capture_semantics_and_exact_pair_gates() -> None:
    off_a = _manifest("trace_off")
    off_b = copy.deepcopy(off_a)
    on = _manifest("trace_on", copy.deepcopy(off_a["execution_identity"]))
    P23.validate_run_manifest(off_a, "trace_off")
    P23.validate_run_manifest(on, "trace_on")
    assert P23.compare_repeatability_manifests(off_a, off_b)["passes"] is True
    noninterference = P23.compare_noninterference_manifests(off_a, on)
    assert noninterference["passes"] is True
    assert noninterference["observation_count"] == 48 * 24

    bad_off = copy.deepcopy(off_a)
    bad_off["candidate_observation"] = {
        "status": "observed_actual_post_aspect_cuda",
        "capture_step_count": 0,
        "observation_count": 0,
    }
    with pytest.raises(P23.P23ProvenanceError, match="observation semantics"):
        P23.validate_run_manifest(bad_off, "trace_off")

    changed_identity = copy.deepcopy(off_b["execution_identity"])
    changed_identity["source_modules"]["runner"]["sha256"] = "e" * 64
    off_b["execution_identity"] = changed_identity
    off_b["run_identity_sha256"] = P23.execution_identity_sha256(changed_identity)
    with pytest.raises(P23.P23ProvenanceError, match="source_modules"):
        P23.compare_repeatability_manifests(off_a, off_b)

    off_b = copy.deepcopy(off_a)
    changed_runtime = off_b["execution_identity"]["runtime"]
    changed_runtime["gpu"]["driver_version"] = "999.2"
    changed_runtime["container"]["gpu"]["driver_version"] = "999.2"
    off_b["run_identity_sha256"] = P23.execution_identity_sha256(off_b["execution_identity"])
    with pytest.raises(P23.P23ProvenanceError, match="runtime"):
        P23.compare_repeatability_manifests(off_a, off_b)

    extra_identity = copy.deepcopy(off_a)
    extra_identity["execution_identity"]["unverified_extra"] = True
    extra_identity["run_identity_sha256"] = P23.execution_identity_sha256(
        extra_identity["execution_identity"]
    )
    with pytest.raises(P23.P23ProvenanceError, match="unverified_extra"):
        P23.compare_repeatability_manifests(off_a, extra_identity)

    bad_capture = copy.deepcopy(on)
    bad_capture["steps"][0]["observer_rng_unchanged"] = False
    with pytest.raises(P23.P23ProvenanceError, match="RNG"):
        P23.validate_run_manifest(bad_capture, "trace_on")

    bad_initial = copy.deepcopy(off_a)
    bad_initial["initial_state"]["model_state_sha256"] = _digest("tampered-initial-model")
    with pytest.raises(P23.P23ProvenanceError, match="initial-state digest"):
        P23.validate_run_manifest(bad_initial, "trace_off")


def test_exact_comparison_reports_one_bitwise_state_mismatch() -> None:
    off_a = _manifest("trace_off")
    off_b = copy.deepcopy(off_a)
    off_b["steps"][124]["optimizer_state_sha256"] = _digest("one-bit-different")
    report = P23.compare_repeatability_manifests(off_a, off_b)
    assert report["passes"] is False
    assert report["mismatch_count"] == 1
    assert report["mismatches"] == [
        {"scope": "step", "step": 124, "field": "optimizer_state_sha256"}
    ]


def test_complete_manifest_sanitization_replaces_paths_without_dropping_fields(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    outputs = tmp_path / "outputs"
    repository.mkdir()
    outputs.mkdir()
    manifest = _manifest("trace_off")
    manifest["repository_path"] = str(repository / "runner.py")
    manifest["artifact_path"] = str(outputs / "trace-off-a.json")
    sanitized = P23.sanitize_complete_manifest(
        manifest,
        path_roots={
            "repository": repository,
            "python_environment": Path(P23.PINNED_PYTHON_ENVIRONMENT),
            "artifacts": outputs,
        },
    )
    retained = sanitized["manifest"]
    assert retained["repository_path"] == "repository:runner.py"
    assert retained["artifact_path"] == "artifacts:trace-off-a.json"
    assert retained["steps"] == manifest["steps"]
    assert retained["execution_identity"]["runtime"]["python_executable"] == (
        "python_environment:bin/python"
    )
    assert retained["execution_identity"]["runtime"]["environment"]["VIRTUAL_ENV"] == (
        "python_environment:."
    )
    assert sanitized["path_replacement_count"] == 5
    assert "/private/tmp/" not in json.dumps(sanitized)

    manifest["unknown_path"] = "/unknown-machine-root/value"
    with pytest.raises(P23.P23ProvenanceError, match="undeclared absolute path"):
        P23.sanitize_complete_manifest(
            manifest,
            path_roots={
                "repository": repository,
                "python_environment": Path(P23.PINNED_PYTHON_ENVIRONMENT),
                "artifacts": outputs,
            },
        )

    credentialed = _manifest("trace_off")
    credentialed["execution_identity"]["repository"]["origin"] = (
        "https://secret@example.invalid/repository.git"
    )
    with pytest.raises(P23.P23ProvenanceError) as error:
        P23.sanitize_complete_manifest(
            credentialed,
            path_roots={"repository": repository, "artifacts": outputs},
        )
    assert "secret" not in str(error.value)


def _module_with_files(source: Path, cache: Path | None = None) -> ModuleType:
    module = ModuleType(f"loaded_{source.stem}")
    module.__file__ = str(source)
    module.__spec__ = SimpleNamespace(origin=str(source))
    module.__cached__ = str(cache) if cache is not None else None
    return module


def _maps_line(path: Path, *, executable: bool = False) -> str:
    permissions = "r-xp" if executable else "r--s"
    return f"1000-2000 {permissions} 00000000 00:00 1 {path}\n"


def test_complete_loaded_file_snapshot_hashes_module_roles_and_proc_maps(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    python_environment = tmp_path / "python-environment"
    data = tmp_path / "data"
    for root in (repository, python_environment, data):
        root.mkdir()
    source = repository / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    cache = repository / "__pycache__" / "module.cpython-312.pyc"
    cache.parent.mkdir()
    cache.write_bytes(b"pinned-bytecode")
    native = python_environment / "libsolver.so"
    native.write_bytes(b"native-bytes")
    tokens = data / "train.bin"
    tokens.write_bytes(b"token-bytes")
    roots = {
        "repository": repository,
        "python_environment": python_environment,
        "data": data,
        "image": Path("/"),
    }

    snapshot = P23.collect_loaded_file_snapshot(
        roots,
        modules={"fixture.module": _module_with_files(source, cache)},
        proc_maps_text=_maps_line(native, executable=True) + _maps_line(tokens),
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )

    assert snapshot["module_count"] == 1
    assert snapshot["mapped_file_count"] == 2
    artifacts = snapshot["modules"]["fixture.module"]["artifacts"]
    assert artifacts["module_file"]["logical_path"] == "repository:module.py"
    assert artifacts["spec_origin"] == artifacts["module_file"]
    assert artifacts["source_file"] == artifacts["module_file"]
    assert artifacts["cached_bytecode"]["logical_path"] == (
        "repository:__pycache__/module.cpython-312.pyc"
    )
    assert snapshot["mapped_files"]["python_environment:libsolver.so"]["executable_origin"] is True
    assert snapshot["mapped_files"]["data:train.bin"]["executable_origin"] is False
    assert P23.validate_loaded_file_snapshot(snapshot) == snapshot


def test_loaded_file_mount_classification_rejects_rw_execution_and_allows_ro_image(
    tmp_path: Path,
) -> None:
    image_root = tmp_path / "image"
    rw_dev = image_root / "dev"
    ro_library = image_root / "usr" / "lib"
    data = tmp_path / "data"
    for root in (rw_dev, ro_library, data):
        root.mkdir(parents=True)
    injected = rw_dev / "injected.so"
    library = ro_library / "libpinned.so"
    tokens = data / "train.bin"
    for path in (injected, library, tokens):
        path.write_bytes(b"fixture-bytes")

    roots = {"image": image_root, "data": data}
    mountinfo = _mountinfo_with(read_only=(ro_library,), writable=(rw_dev, data))
    with pytest.raises(P23.P23ProvenanceError, match="writable mount"):
        P23.collect_loaded_file_snapshot(
            roots,
            modules={"fixture.injected": _module_with_files(injected)},
            proc_maps_text="",
            mountinfo_text=mountinfo,
        )

    snapshot = P23.collect_loaded_file_snapshot(
        roots,
        modules={"fixture.library": _module_with_files(library)},
        proc_maps_text=_maps_line(tokens, executable=False),
        mountinfo_text=mountinfo,
    )
    assert (
        snapshot["modules"]["fixture.library"]["artifacts"]["module_file"]["logical_path"]
        == "image:usr/lib/libpinned.so"
    )
    assert snapshot["mapped_files"]["data:train.bin"]["executable_origin"] is False


def test_loaded_file_closure_records_additions_and_rejects_removal_or_change(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    python_environment = tmp_path / "python-environment"
    repository.mkdir()
    python_environment.mkdir()
    first = repository / "first.py"
    second = repository / "second.py"
    native = python_environment / "libcuda-fixture.so"
    first.write_text("FIRST = 1\n", encoding="utf-8")
    second.write_text("SECOND = 2\n", encoding="utf-8")
    native.write_bytes(b"native")
    roots = {
        "repository": repository,
        "python_environment": python_environment,
        "image": Path("/"),
    }
    initialized = P23.collect_loaded_file_snapshot(
        roots,
        modules={"fixture.first": _module_with_files(first)},
        proc_maps_text="",
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )
    completed = P23.collect_loaded_file_snapshot(
        roots,
        modules={
            "fixture.first": _module_with_files(first),
            "fixture.second": _module_with_files(second),
        },
        proc_maps_text=_maps_line(native, executable=True),
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )
    closure = P23.build_loaded_file_closure(initialized, completed)
    assert set(closure["module_additions"]) == {"fixture.second"}
    assert set(closure["mapped_file_additions"]) == {"python_environment:libcuda-fixture.so"}
    assert P23.validate_loaded_file_closure(closure) == closure

    with pytest.raises(P23.P23ProvenanceError, match="removed or changed"):
        P23.build_loaded_file_closure(completed, initialized)
    first.write_text("FIRST = 9\n", encoding="utf-8")
    changed = P23.collect_loaded_file_snapshot(
        roots,
        modules={"fixture.first": _module_with_files(first)},
        proc_maps_text="",
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )
    with pytest.raises(P23.P23ProvenanceError, match="removed or changed"):
        P23.build_loaded_file_closure(initialized, changed)


def test_loaded_file_closure_rejects_deleted_unrooted_and_forbidden_execution(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    data = tmp_path / "data"
    native_evidence = tmp_path / "native"
    tmpfs = tmp_path / "tmpfs"
    dev_shm = tmp_path / "dev-shm"
    elsewhere = tmp_path / "elsewhere"
    for root in (repository, data, native_evidence, tmpfs, dev_shm, elsewhere):
        root.mkdir()
    data_file = data / "train.bin"
    native_file = native_evidence / "candidate.so"
    temporary_file = tmpfs / "generated.py"
    shared_memory_file = dev_shm / "generated.so"
    unrooted = elsewhere / "outside.so"
    for path in (data_file, native_file, temporary_file, shared_memory_file, unrooted):
        path.write_bytes(b"bytes")
    roots = {
        "repository": repository,
        "data": data,
        "native_evidence": native_evidence,
        "tmpfs": tmpfs,
        "dev_shm": dev_shm,
    }

    for forbidden in (data_file, native_file, temporary_file, shared_memory_file):
        with pytest.raises(P23.P23ProvenanceError, match="forbidden"):
            P23.collect_loaded_file_snapshot(
                roots,
                modules={"fixture.forbidden": _module_with_files(forbidden)},
                proc_maps_text="",
                mountinfo_text=_READ_ONLY_MOUNTINFO,
            )
        with pytest.raises(P23.P23ProvenanceError, match="forbidden"):
            P23.collect_loaded_file_snapshot(
                roots,
                modules={},
                proc_maps_text=_maps_line(forbidden, executable=True),
                mountinfo_text=_READ_ONLY_MOUNTINFO,
            )

    with pytest.raises(P23.P23ProvenanceError, match="outside every logical root"):
        P23.collect_loaded_file_snapshot(
            roots,
            modules={},
            proc_maps_text=_maps_line(unrooted, executable=True),
            mountinfo_text=_READ_ONLY_MOUNTINFO,
        )
    with pytest.raises(P23.P23ProvenanceError, match="deleted"):
        P23.collect_loaded_file_snapshot(
            roots,
            modules={},
            proc_maps_text=_maps_line(data_file).rstrip() + " (deleted)\n",
            mountinfo_text=_READ_ONLY_MOUNTINFO,
        )

    relative_module = ModuleType("fixture.relative")
    relative_module.__file__ = "missing-relative-module.py"
    relative_module.__spec__ = SimpleNamespace(origin=None)
    with pytest.raises(P23.P23ProvenanceError, match="relative executable path"):
        P23.collect_loaded_file_snapshot(
            roots,
            modules={"fixture.relative": relative_module},
            proc_maps_text="",
            mountinfo_text=_READ_ONLY_MOUNTINFO,
        )


def test_loaded_file_verifier_recollects_membership_and_rehashes_completed_bytes(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    source = repository / "module.py"
    source.write_text("VALUE = 1\n", encoding="utf-8")
    roots = {"repository": repository}
    snapshot = P23.collect_loaded_file_snapshot(
        roots,
        modules={"fixture.module": _module_with_files(source)},
        proc_maps_text="",
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )
    closure = P23.build_loaded_file_closure(snapshot, snapshot)
    report = P23.verify_loaded_file_closure(
        closure,
        logical_roots=roots,
        current_initialized=snapshot,
        mountinfo_text=_READ_ONLY_MOUNTINFO,
    )
    assert report["passes"] is True
    assert report["unique_file_count"] == 1

    source.write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(P23.P23ProvenanceError, match="bytes changed"):
        P23.verify_loaded_file_closure(
            closure,
            logical_roots=roots,
            current_initialized=snapshot,
            mountinfo_text=_READ_ONLY_MOUNTINFO,
        )
