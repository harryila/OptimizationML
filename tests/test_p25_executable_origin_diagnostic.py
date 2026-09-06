from __future__ import annotations

import ast
import importlib.util
import inspect
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/training/run_p25_executable_origin_diagnostic.py"


def _module():
    spec = importlib.util.spec_from_file_location("p25_origin_diagnostic", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeMatmul:
    def __init__(self) -> None:
        self.allow_tf32 = True
        self.allow_fp16_reduced_precision_reduction = True
        self.allow_bf16_reduced_precision_reduction = True


class _FakeCudaBackend:
    def __init__(self) -> None:
        self.matmul = _FakeMatmul()
        self._math = False
        self._flash = True
        self._efficient = True
        self._cudnn = True

    def enable_math_sdp(self, value: bool) -> None:
        self._math = value

    def enable_flash_sdp(self, value: bool) -> None:
        self._flash = value

    def enable_mem_efficient_sdp(self, value: bool) -> None:
        self._efficient = value

    def enable_cudnn_sdp(self, value: bool) -> None:
        self._cudnn = value

    def math_sdp_enabled(self) -> bool:
        return self._math

    def flash_sdp_enabled(self) -> bool:
        return self._flash

    def mem_efficient_sdp_enabled(self) -> bool:
        return self._efficient

    def cudnn_sdp_enabled(self) -> bool:
        return self._cudnn


class _FakeCudnn:
    allow_tf32 = True
    benchmark = True
    deterministic = False


class _FakeCuda:
    def __init__(self, initialized: bool = False) -> None:
        self.initialized = initialized

    def is_initialized(self) -> bool:
        return self.initialized


class _FakeOptimizer:
    def __init__(self) -> None:
        self.param_groups = [{}]


class _FakeTorch:
    def __init__(
        self,
        *,
        modules: dict[str, object] | None = None,
        generated_path: Path | None = None,
        cuda_initialized: bool = False,
    ) -> None:
        self.log: list[str] = []
        self.modules = modules
        self.generated_path = generated_path
        self.backends = SimpleNamespace(cuda=_FakeCudaBackend(), cudnn=_FakeCudnn())
        self.cuda = _FakeCuda(cuda_initialized)
        self.float32 = object()
        self._algorithms = False
        self._debug = 0
        self._precision = "medium"
        self._threads = 30
        self._interop = 30
        self.nn = SimpleNamespace(Parameter=self._parameter)
        self.optim = SimpleNamespace(SGD=self._sgd)

    def use_deterministic_algorithms(self, value: bool) -> None:
        self.log.append("use_deterministic_algorithms")
        self._algorithms = value

    def are_deterministic_algorithms_enabled(self) -> bool:
        return self._algorithms

    def set_deterministic_debug_mode(self, value: str) -> None:
        self.log.append("set_deterministic_debug_mode")
        self._debug = 2 if value == "error" else 0

    def get_deterministic_debug_mode(self) -> int:
        return self._debug

    def set_float32_matmul_precision(self, value: str) -> None:
        self.log.append("set_float32_matmul_precision")
        self._precision = value

    def get_float32_matmul_precision(self) -> str:
        return self._precision

    def set_num_threads(self, value: int) -> None:
        self.log.append("set_num_threads")
        self._threads = value

    def get_num_threads(self) -> int:
        return self._threads

    def set_num_interop_threads(self, value: int) -> None:
        self.log.append("set_num_interop_threads")
        self._interop = value

    def get_num_interop_threads(self) -> int:
        return self._interop

    def zeros(self, size: int, *, dtype: object, device: str) -> object:
        assert size == 1 and dtype is self.float32 and device == "cpu"
        assert self._algorithms and self._debug == 2
        assert not self.cuda.is_initialized()
        self.log.append("zeros")
        return object()

    def _parameter(self, value: object) -> object:
        del value
        self.log.append("parameter")
        return object()

    def _sgd(self, parameters: list[object], *, lr: float) -> _FakeOptimizer:
        assert len(parameters) == 1 and lr == 1.0
        self.log.append("sgd")
        if self.modules is not None and self.generated_path is not None:
            self.modules["_remote_module_non_scriptable"] = SimpleNamespace(
                __file__=str(self.generated_path)
            )
        return _FakeOptimizer()


def _locked_determinism() -> dict[str, object]:
    return {
        "bf16_reduced_precision_reduction": False,
        "cpu_threads": 1,
        "cublas_workspace_config": ":4096:8",
        "cuda_visible_devices": "GPU-fixture",
        "cudnn_benchmark": False,
        "cudnn_deterministic": True,
        "deterministic_algorithms": True,
        "deterministic_debug_mode": "error",
        "float32_matmul_precision": "highest",
        "fp16_reduced_precision_reduction": False,
        "interop_threads": 1,
        "nvidia_tf32_override": "0",
        "nvidia_visible_devices": "GPU-fixture",
        "pythonhashseed": "1337",
        "sdpa_backend": "math",
        "tf32": False,
    }


def _frozen_python_flags(module):
    return SimpleNamespace(**module.FROZEN_PYTHON_FLAGS)


def _read_only_root_mount() -> list[dict[str, object]]:
    return [
        {
            "mount_id": 1,
            "mount_point": "/",
            "filesystem_type": "overlay",
            "mount_options": ["ro"],
            "super_options": ["rw"],
            "writable": False,
        }
    ]


def test_p25_runner_is_self_contained_and_imports_only_stdlib_before_preflight() -> None:
    tree = ast.parse(SCRIPT.read_text(encoding="utf-8"))
    top_level_imports = [
        node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
    ]
    imported_roots: set[str] = set()
    for node in top_level_imports:
        if isinstance(node, ast.Import):
            imported_roots.update(alias.name.split(".")[0] for alias in node.names)
        elif node.module != "__future__":
            assert node.module is not None
            imported_roots.add(node.module.split(".")[0])
    assert imported_roots <= set(sys.stdlib_module_names)
    assert {"torch", "numpy", "random", "secrets"}.isdisjoint(imported_roots)
    assert "passive_muon" not in SCRIPT.read_text(encoding="utf-8")


def test_p25_run_diagnostic_preflights_before_importing_torch() -> None:
    module = _module()
    source = inspect.getsource(module.run_diagnostic)
    assert source.index("assert_torch_not_loaded(sys.modules)") < source.index(
        "preflight = validate_preflight("
    )
    assert source.index("preflight = validate_preflight(") < source.index("import torch")


def test_p25_rejects_any_preloaded_torch_module() -> None:
    module = _module()
    module.assert_torch_not_loaded({"json": object()})
    for name in ("torch", "torch.cuda", "torch.optim.sgd"):
        with pytest.raises(module.DiagnosticError, match="must not already be loaded"):
            module.assert_torch_not_loaded({name: object()})


def test_p25_python_flags_match_lock_and_pinned_major_minor_exactly() -> None:
    module = _module()
    locked = dict(module.FROZEN_PYTHON_FLAGS)
    flags = _frozen_python_flags(module)
    assert module.python_interpreter_matches_lock(locked, flags=flags, version_info=(3, 12, 14))

    wrong_value = dict(locked)
    wrong_value["optimize"] = 1
    assert not module.python_interpreter_matches_lock(
        wrong_value, flags=flags, version_info=(3, 12, 14)
    )

    wrong_type = dict(locked)
    wrong_type["debug"] = False
    assert not module.python_interpreter_matches_lock(
        wrong_type, flags=flags, version_info=(3, 12, 14)
    )

    assert not module.python_interpreter_matches_lock(locked, flags=flags, version_info=(3, 13, 0))


def test_p25_configures_exact_p23_state_without_initializing_cuda() -> None:
    module = _module()
    torch = _FakeTorch()
    observed = module.configure_torch_determinism(torch, _locked_determinism())
    assert observed == module.EXPECTED_TORCH_DETERMINISM
    assert torch.cuda.is_initialized() is False


@pytest.mark.parametrize(
    ("field", "mutated"),
    [
        ("deterministic_algorithms", False),
        ("deterministic_debug_mode", "warn"),
        ("sdpa_backend", "flash"),
        ("cudnn_deterministic", False),
        ("cudnn_benchmark", True),
        ("tf32", True),
        ("float32_matmul_precision", "high"),
        ("fp16_reduced_precision_reduction", True),
        ("bf16_reduced_precision_reduction", True),
        ("cpu_threads", 2),
        ("interop_threads", 2),
        ("cublas_workspace_config", ":16:8"),
        ("cuda_visible_devices", "0"),
        ("nvidia_tf32_override", "1"),
        ("pythonhashseed", "0"),
    ],
)
def test_p25_rejects_each_locked_torch_determinism_drift(field: str, mutated: object) -> None:
    module = _module()
    lock = _locked_determinism()
    lock[field] = mutated
    with pytest.raises(module.DiagnosticError, match="runtime lock"):
        module.configure_torch_determinism(_FakeTorch(), lock)


def test_p25_rejects_cuda_initialized_before_configuration() -> None:
    module = _module()
    with pytest.raises(module.DiagnosticError, match="CUDA was initialized"):
        module.configure_torch_determinism(_FakeTorch(cuda_initialized=True), _locked_determinism())


def test_p25_rejects_an_extra_locked_determinism_field() -> None:
    module = _module()
    lock = _locked_determinism()
    lock["unexpected"] = True
    with pytest.raises(module.DiagnosticError, match="runtime lock"):
        module.configure_torch_determinism(_FakeTorch(), lock)


def test_p25_fails_when_backend_refuses_a_required_setting() -> None:
    module = _module()
    torch = _FakeTorch()
    torch.backends.cuda.enable_math_sdp = lambda value: None
    with pytest.raises(module.DiagnosticError, match="failed to establish"):
        module.configure_torch_determinism(torch, _locked_determinism())


def test_p25_trigger_order_and_module_snapshots_are_exact(tmp_path: Path) -> None:
    module = _module()
    generated = tmp_path / "_remote_module_non_scriptable.py"
    generated.write_text("# generated fixture\n", encoding="utf-8")
    modules: dict[str, object] = {}
    torch = _FakeTorch(modules=modules, generated_path=generated)

    result = module.execute_minimal_trigger(
        torch_module=torch,
        locked_determinism=_locked_determinism(),
        mounts=_read_only_root_mount(),
        modules=modules,
    )

    assert result["events"] == [
        "torch_imported",
        "torch_determinism_configured",
        "cpu_fp32_parameter_constructed",
        "sgd_optimizer_constructed",
    ]
    assert torch.log[-3:] == ["zeros", "parameter", "sgd"]
    assert module.GENERATED_MODULE not in result["after_torch"]
    assert module.GENERATED_MODULE not in result["after_configuration"]
    assert module.GENERATED_MODULE not in result["after_parameter"]
    assert module.GENERATED_MODULE in result["after_optimizer"]
    assert result["state_after_parameter"] == module.EXPECTED_TORCH_DETERMINISM
    assert result["state_after_optimizer"] == module.EXPECTED_TORCH_DETERMINISM
    record = result["after_optimizer"][module.GENERATED_MODULE]
    assert record["sha256"] == module.sha256_file(generated)


def test_p25_trigger_rejects_generated_module_before_optimizer() -> None:
    module = _module()
    with pytest.raises(module.DiagnosticError, match="already loaded"):
        module.execute_minimal_trigger(
            torch_module=_FakeTorch(),
            locked_determinism=_locked_determinism(),
            mounts=_read_only_root_mount(),
            modules={module.GENERATED_MODULE: object()},
        )


def test_p25_trigger_rejects_cuda_initialization_after_bare_import() -> None:
    module = _module()
    with pytest.raises(module.DiagnosticError, match="bare PyTorch import"):
        module.execute_minimal_trigger(
            torch_module=_FakeTorch(cuda_initialized=True),
            locked_determinism=_locked_determinism(),
            mounts=_read_only_root_mount(),
            modules={},
        )


def test_p25_fixed_origin_is_the_p24_image_resident_module() -> None:
    module = _module()
    assert module.SCHEMA == "passive-muon-p25-executable-origin-diagnostic-v1"
    assert module.CONTRACT_SCHEMA == ("passive-muon-p25-cuda-diagnostic-correction-contract-v1")
    assert module.FIXED_GENERATED_PATH.endswith(
        "torch/_p24_generated_remote_modules/_remote_module_non_scriptable.py"
    )
    assert module.PATCHED_INSTANTIATOR_SHA256 == (
        "1ecc8ae1ac4a517511914fff6e01c78ed3ad828bbe8a07b02079c206ee56bceb"
    )


def test_p25_atomic_writer_is_no_overwrite(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "native.json"
    module._atomic_write_json(output, {"schema_version": module.SCHEMA})
    original = output.read_bytes()
    with pytest.raises(module.DiagnosticError, match="already exists"):
        module._atomic_write_json(output, {"mutated": True})
    assert output.read_bytes() == original
