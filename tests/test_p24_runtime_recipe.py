from __future__ import annotations

import importlib
import importlib.util
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RUNTIME_ROOT = ROOT / "experiments/training"
PREPARER = RUNTIME_ROOT / "prepare_p24_immutable_remote_module.py"
DOCKERFILE = RUNTIME_ROOT / "p24_runtime.Dockerfile"

P23_IMAGE = (
    "localhost:5000/p23-runtime@"
    "sha256:44ef23717780b1cbf112b183e7988b1319ddfed6b1d224efaa33e1e6d96de4c1"
)


def _module():
    specification = importlib.util.spec_from_file_location("p24_image_preparer", PREPARER)
    assert specification is not None and specification.loader is not None
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def test_p24_runtime_is_one_exact_layer_over_the_p23_image() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert dockerfile.startswith(f"FROM {P23_IMAGE}\nUSER root\n")
    assert re.findall(r"^FROM (.+)$", dockerfile, flags=re.MULTILINE) == [P23_IMAGE]
    assert not re.findall(r"^ARG ", dockerfile, flags=re.MULTILINE)
    assert dockerfile.count("\nCOPY ") == 1
    assert "\nADD " not in dockerfile
    assert "prepare_p24_immutable_remote_module.py --verify-only" in dockerfile
    assert dockerfile.count("/opt/p23-venv/bin/python") == 3
    assert "PYTHONDONTWRITEBYTECODE=1" in dockerfile
    assert 'sys.modules["_remote_module_non_scriptable"].__file__' in dockerfile
    assert 'sys.exit("P24 generated module did not load from the exact image path")' in dockerfile
    assert " assert " not in dockerfile
    assert "latest" not in dockerfile.lower()


def test_p24_preparer_locks_all_source_and_generated_hashes() -> None:
    module = _module()
    assert module.TORCH_VERSION == "2.7.0+cu128"
    assert module.ORIGINAL_INSTANTIATOR_SHA256 == (
        "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
    )
    assert module.PATCHED_INSTANTIATOR_SHA256 == (
        "1ecc8ae1ac4a517511914fff6e01c78ed3ad828bbe8a07b02079c206ee56bceb"
    )
    assert module.PATCHED_INSTANTIATOR_BYTES == 6438
    assert module.REMOTE_MODULE_SHA256 == (
        "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1"
    )
    assert module.TEMPLATE_SHA256 == (
        "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc"
    )
    assert module.GENERATED_MODULE_SHA256 == (
        "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
    )
    assert module.GENERATED_MODULE_BYTES == 2355
    assert str(module.FIXED_GENERATED_MODULE) == (
        "/opt/p23-venv/lib/python3.12/site-packages/torch/"
        "_p24_generated_remote_modules/_remote_module_non_scriptable.py"
    )


def test_p24_preparer_reconstructs_the_exact_generated_module() -> None:
    module = _module()
    template = (
        ROOT / ".venv/lib/python3.12/site-packages/torch/distributed/nn/jit/templates/"
        "remote_module_template.py"
    )
    if not template.is_file():
        pytest.skip("local PyTorch template is unavailable")
    if module.sha256_file(template) != module.TEMPLATE_SHA256:
        pytest.skip("local PyTorch template differs from the pinned template")
    generated = module.render_generated_module(template)
    assert len(generated) == module.GENERATED_MODULE_BYTES
    assert module.sha256_bytes(generated) == module.GENERATED_MODULE_SHA256


def test_p24_source_transform_removes_writes_and_fails_on_anchor_drift() -> None:
    module = _module()
    source = b"".join(
        (
            b"import importlib\n",
            b"import atexit\n",
            b"import tempfile\n",
            b"from typing import Optional\n",
            module._TEMP_DIRECTORY_BLOCK,
            module._WRITABLE_WRITE_FUNCTION,
            module._DYNAMIC_IMPORT_FUNCTION,
        )
    )
    transformed = module.transform_instantiator(source)
    assert b"TemporaryDirectory" not in transformed
    assert b'open(out_path, "w")' not in transformed
    assert b"_p24_generated_remote_modules" in transformed
    assert b"differs from the pinned template" in transformed
    assert b"sys.path.append" not in transformed
    assert b"importlib.import_module" not in transformed
    assert transformed.count(b"import importlib.util\n") == 1
    assert b"import importlib\n" not in transformed
    assert b"importlib.util.spec_from_file_location" in transformed
    assert b"sys.modules[generated_module_name] = generated_module" in transformed
    assert b"did not load from the exact image path" in transformed

    with pytest.raises(module.PreparationError, match="exactly once"):
        module.transform_instantiator(source.replace(b"import atexit\n", b""))
    with pytest.raises(module.PreparationError, match="importlib import"):
        module.transform_instantiator(source.replace(b"import importlib\n", b""))
    with pytest.raises(module.PreparationError, match="exactly once"):
        module.transform_instantiator(source + module._WRITABLE_WRITE_FUNCTION)


def test_p24_exact_path_loader_replaces_same_name_sys_path_module(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _module()
    generated_name = module.GENERATED_MODULE_NAME
    exact_directory = tmp_path / "exact"
    shadow_directory = tmp_path / "shadow"
    exact_directory.mkdir()
    shadow_directory.mkdir()
    exact_path = exact_directory / module.GENERATED_MODULE_FILENAME
    exact_path.write_text("ORIGIN = 'exact-image-path'\n", encoding="utf-8")
    (shadow_directory / module.GENERATED_MODULE_FILENAME).write_text(
        "ORIGIN = 'sys-path-shadow'\n", encoding="utf-8"
    )
    monkeypatch.syspath_prepend(str(shadow_directory))
    previous = sys.modules.pop(generated_name, None)
    try:
        shadow = importlib.import_module(generated_name)
        assert shadow.ORIGIN == "sys-path-shadow"
        namespace = {
            "importlib": importlib,
            "os": __import__("os"),
            "sys": sys,
            "INSTANTIATED_TEMPLATE_DIR_PATH": str(exact_directory),
            "get_remote_module_template": lambda enabled: "ORIGIN = 'exact-image-path'\n",
            "_write": lambda path, text: None,
        }
        exec(module._EXACT_PATH_IMPORT_FUNCTION, namespace)
        loaded = namespace["_do_instantiate_remote_module_template"](generated_name, {}, True)
        assert loaded.ORIGIN == "exact-image-path"
        assert Path(loaded.__file__).resolve() == exact_path.resolve()
        assert sys.modules[generated_name] is loaded
    finally:
        sys.modules.pop(generated_name, None)
        if previous is not None:
            sys.modules[generated_name] = previous


def test_p24_exact_file_guard_rejects_byte_count_and_hash_drift(tmp_path: Path) -> None:
    module = _module()
    source = tmp_path / "source.py"
    source.write_bytes(b"exact")
    module._require_file(
        source,
        expected_sha256=module.sha256_bytes(b"exact"),
        expected_bytes=5,
        label="fixture",
    )
    with pytest.raises(module.PreparationError, match="byte count mismatch"):
        module._require_file(
            source,
            expected_sha256=module.sha256_bytes(b"exact"),
            expected_bytes=6,
            label="fixture",
        )
    with pytest.raises(module.PreparationError, match="SHA-256 mismatch"):
        module._require_file(
            source,
            expected_sha256="0" * 64,
            expected_bytes=5,
            label="fixture",
        )


def test_p24_preparer_deletes_only_locked_stale_bytecode_patterns(tmp_path: Path) -> None:
    module = _module()
    instantiator_cache = tmp_path / module.INSTANTIATOR_RELATIVE.parent / "__pycache__"
    instantiator_cache.mkdir(parents=True)
    stale_instantiator = instantiator_cache / "instantiator.cpython-312.pyc"
    unrelated = instantiator_cache / "other.cpython-312.pyc"
    stale_instantiator.write_bytes(b"stale")
    unrelated.write_bytes(b"keep")
    generated_cache = tmp_path / "another/__pycache__"
    generated_cache.mkdir(parents=True)
    stale_generated = generated_cache / "_remote_module_non_scriptable.cpython-312.pyc"
    stale_generated.write_bytes(b"stale")

    module._remove_stale_bytecode(tmp_path)
    assert not stale_instantiator.exists()
    assert not stale_generated.exists()
    assert unrelated.read_bytes() == b"keep"
