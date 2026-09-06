#!/usr/bin/env python3
"""Prepare the exact PyTorch generated module as an immutable image artifact.

PyTorch 2.7.0 creates ``_remote_module_non_scriptable.py`` lazily in a
``TemporaryDirectory`` while constructing an optimizer.  P23 correctly rejects
that executable source because ``/tmp`` is writable.  This build-time helper
does not relax that check.  It verifies the three relevant wheel sources,
renders the exact generated source, and redirects the pinned instantiator to a
read-only, image-resident copy.

Every source transformation is guarded by exact pre- and postimage hashes.
Any wheel revision, partial prior patch, template drift, or unexpected generated
content therefore fails the image build closed.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import runpy
import stat
from pathlib import Path
from typing import Final

TORCH_VERSION: Final = "2.7.0+cu128"
SITE_PACKAGES: Final = Path("/opt/p23-venv/lib/python3.12/site-packages")
INSTANTIATOR_RELATIVE: Final = Path("torch/distributed/nn/jit/instantiator.py")
REMOTE_MODULE_RELATIVE: Final = Path("torch/distributed/nn/api/remote_module.py")
TEMPLATE_RELATIVE: Final = Path("torch/distributed/nn/jit/templates/remote_module_template.py")
FIXED_DIRECTORY_RELATIVE: Final = Path("torch/_p24_generated_remote_modules")
GENERATED_MODULE_NAME: Final = "_remote_module_non_scriptable"
GENERATED_MODULE_FILENAME: Final = f"{GENERATED_MODULE_NAME}.py"
FIXED_DIRECTORY: Final = SITE_PACKAGES / FIXED_DIRECTORY_RELATIVE
FIXED_GENERATED_MODULE: Final = FIXED_DIRECTORY / GENERATED_MODULE_FILENAME
MANIFEST_PATH: Final = Path("/opt/p24-immutable-remote-module-manifest.json")

ORIGINAL_INSTANTIATOR_SHA256: Final = (
    "567d1314ee27ff0b3bd22e7c4d1157246469de25e7a3183d96debe167b193615"
)
ORIGINAL_INSTANTIATOR_BYTES: Final = 5510
REMOTE_MODULE_SHA256: Final = "f9bb2f5c5438791581d399e38a27606e123bdbeb3c6cb53683318a06060439c1"
REMOTE_MODULE_BYTES: Final = 31251
TEMPLATE_SHA256: Final = "0ff1856bbd031b5298d46c06c0502abc20bd804f42c1949ed4127e8c773660cc"
TEMPLATE_BYTES: Final = 3463
GENERATED_MODULE_SHA256: Final = "8205b16956fb264841ecd8644784a0d157f87df79b17c16825dc1163433ce5d8"
GENERATED_MODULE_BYTES: Final = 2355

# Filled from the exact byte transformation below.  Keeping the value literal
# makes both the image recipe and the native P24 diagnostic independent checks.
PATCHED_INSTANTIATOR_SHA256: Final = (
    "1ecc8ae1ac4a517511914fff6e01c78ed3ad828bbe8a07b02079c206ee56bceb"
)
PATCHED_INSTANTIATOR_BYTES: Final = 6438

_TEMP_DIRECTORY_BLOCK: Final = b"""_FILE_PREFIX = "_remote_module_"
_TEMP_DIR = tempfile.TemporaryDirectory()
INSTANTIATED_TEMPLATE_DIR_PATH = _TEMP_DIR.name
atexit.register(_TEMP_DIR.cleanup)
logger.info("Created a temporary directory at %s", INSTANTIATED_TEMPLATE_DIR_PATH)
sys.path.append(INSTANTIATED_TEMPLATE_DIR_PATH)
"""

_IMMUTABLE_DIRECTORY_BLOCK: Final = b"""_FILE_PREFIX = "_remote_module_"
INSTANTIATED_TEMPLATE_DIR_PATH = (
    "/opt/p23-venv/lib/python3.12/site-packages/torch/_p24_generated_remote_modules"
)
logger.info(
    "Using immutable P24 generated-module directory at %s",
    INSTANTIATED_TEMPLATE_DIR_PATH,
)
"""

_WRITABLE_WRITE_FUNCTION: Final = b"""def _write(out_path, text):
    old_text: Optional[str]
    try:
        with open(out_path) as f:
            old_text = f.read()
    except OSError:
        old_text = None
    if old_text != text:
        with open(out_path, "w") as f:
            logger.info("Writing %s", out_path)
            f.write(text)
    else:
        logger.info("Skipped writing %s", out_path)
"""

_IMMUTABLE_VERIFY_FUNCTION: Final = b"""def _write(out_path, text):
    expected_path = os.path.join(
        INSTANTIATED_TEMPLATE_DIR_PATH, f"{_FILE_PREFIX}non_scriptable.py"
    )
    if os.path.realpath(out_path) != os.path.realpath(expected_path):
        raise RuntimeError(
            "P24 immutable remote-module image supports only the pinned "
            "non-scriptable template"
        )
    try:
        with open(out_path, encoding="utf-8") as handle:
            old_text = handle.read()
    except OSError as error:
        raise RuntimeError(
            "P24 immutable remote-module source is unavailable"
        ) from error
    if old_text != text:
        raise RuntimeError(
            "P24 immutable remote-module source differs from the pinned template"
        )
    logger.info("Verified immutable generated module %s", out_path)
"""

_DYNAMIC_IMPORT_FUNCTION: Final = b"""def _do_instantiate_remote_module_template(
    generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
):
    generated_code_text = get_remote_module_template(
        enable_moving_cpu_tensors_to_cuda
    ).format(**str_dict)
    out_path = os.path.join(
        INSTANTIATED_TEMPLATE_DIR_PATH, f"{generated_module_name}.py"
    )
    _write(out_path, generated_code_text)

    # From importlib doc,
    # > If you are dynamically importing a module that was created since
    # the interpreter began execution (e.g., created a Python source file),
    # you may need to call invalidate_caches() in order for the new module
    # to be noticed by the import system.
    importlib.invalidate_caches()
    generated_module = importlib.import_module(f"{generated_module_name}")
    return generated_module
"""

_EXACT_PATH_IMPORT_FUNCTION: Final = b"""def _do_instantiate_remote_module_template(
    generated_module_name, str_dict, enable_moving_cpu_tensors_to_cuda
):
    generated_code_text = get_remote_module_template(
        enable_moving_cpu_tensors_to_cuda
    ).format(**str_dict)
    out_path = os.path.join(
        INSTANTIATED_TEMPLATE_DIR_PATH, f"{generated_module_name}.py"
    )
    _write(out_path, generated_code_text)

    spec = importlib.util.spec_from_file_location(generated_module_name, out_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("P24 cannot construct the exact generated-module loader")
    generated_module = importlib.util.module_from_spec(spec)
    previous_module = sys.modules.get(generated_module_name)
    sys.modules[generated_module_name] = generated_module
    try:
        spec.loader.exec_module(generated_module)
        observed_path = getattr(generated_module, "__file__", None)
        if not isinstance(observed_path, str) or os.path.realpath(
            observed_path
        ) != os.path.realpath(out_path):
            raise RuntimeError("P24 generated module did not load from the exact image path")
    except BaseException:
        if previous_module is None:
            sys.modules.pop(generated_module_name, None)
        else:
            sys.modules[generated_module_name] = previous_module
        raise
    return generated_module
"""


class PreparationError(RuntimeError):
    """Raised when the exact build-time contract cannot be satisfied."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _require_file(
    path: Path,
    *,
    expected_sha256: str,
    expected_bytes: int,
    label: str,
) -> bytes:
    try:
        payload = path.read_bytes()
    except OSError as error:
        raise PreparationError(f"cannot read pinned {label}: {path}") from error
    if len(payload) != expected_bytes:
        raise PreparationError(
            f"{label} byte count mismatch: expected {expected_bytes}, got {len(payload)}"
        )
    digest = sha256_bytes(payload)
    if digest != expected_sha256:
        raise PreparationError(
            f"{label} SHA-256 mismatch: expected {expected_sha256}, got {digest}"
        )
    return payload


def _replace_once(source: bytes, old: bytes, new: bytes, *, label: str) -> bytes:
    occurrences = source.count(old)
    if occurrences != 1:
        raise PreparationError(
            f"{label} anchor must occur exactly once; observed {occurrences} occurrences"
        )
    return source.replace(old, new, 1)


def transform_instantiator(source: bytes) -> bytes:
    """Apply the only permitted byte-for-byte transformation."""

    transformed = source
    transformed = _replace_once(
        transformed,
        b"import importlib\n",
        b"import importlib.util\n",
        label="importlib import",
    )
    transformed = _replace_once(
        transformed,
        b"import atexit\n",
        b"",
        label="atexit import",
    )
    transformed = _replace_once(
        transformed,
        b"import tempfile\n",
        b"",
        label="tempfile import",
    )
    transformed = _replace_once(
        transformed,
        b"from typing import Optional\n",
        b"",
        label="Optional import",
    )
    transformed = _replace_once(
        transformed,
        _TEMP_DIRECTORY_BLOCK,
        _IMMUTABLE_DIRECTORY_BLOCK,
        label="temporary-directory initialization",
    )
    transformed = _replace_once(
        transformed,
        _WRITABLE_WRITE_FUNCTION,
        _IMMUTABLE_VERIFY_FUNCTION,
        label="generated-source writer",
    )
    transformed = _replace_once(
        transformed,
        _DYNAMIC_IMPORT_FUNCTION,
        _EXACT_PATH_IMPORT_FUNCTION,
        label="generated-module importer",
    )
    return transformed


def render_generated_module(template_path: Path) -> bytes:
    """Render the exact non-scriptable module without importing PyTorch."""

    namespace = runpy.run_path(str(template_path))
    template_factory = namespace.get("get_remote_module_template")
    if not callable(template_factory):
        raise PreparationError("pinned template does not expose get_remote_module_template")
    substitutions = {
        "assign_module_interface_cls": "module_interface_cls = None",
        "args": "*args",
        "kwargs": "**kwargs",
        "arg_types": "*args, **kwargs",
        "arrow_and_return_type": "",
        "arrow_and_future_return_type": "",
        "jit_script_decorator": "",
    }
    try:
        rendered = template_factory(True).format(**substitutions)
    except (KeyError, TypeError, ValueError) as error:
        raise PreparationError("cannot render the pinned non-scriptable template") from error
    return rendered.encode("utf-8")


def _matching_stale_bytecode(site_packages: Path) -> list[Path]:
    instantiator_cache = site_packages / INSTANTIATOR_RELATIVE.parent / "__pycache__"
    paths = list(instantiator_cache.glob("instantiator.*.pyc"))
    paths.extend(site_packages.rglob(f"{GENERATED_MODULE_NAME}.*.pyc"))
    return sorted(set(paths))


def _remove_stale_bytecode(site_packages: Path) -> None:
    for path in _matching_stale_bytecode(site_packages):
        try:
            path.unlink()
        except OSError as error:
            raise PreparationError(f"cannot remove stale bytecode: {path}") from error


def _write_new_exact(path: Path, payload: bytes, *, mode: int) -> None:
    if path.exists():
        raise PreparationError(f"refusing to overwrite existing image artifact: {path}")
    if not path.parent.is_dir():
        raise PreparationError(f"image artifact parent directory is absent: {path.parent}")
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(FileNotFoundError):
                path.unlink()
            raise
    except OSError as error:
        raise PreparationError(f"cannot create exact image artifact: {path}") from error


def _replace_exact(path: Path, payload: bytes, *, mode: int) -> None:
    temporary = path.with_name(f".{path.name}.p24-new")
    if temporary.exists():
        raise PreparationError(f"stale patch temporary exists: {temporary}")
    try:
        descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            temporary.unlink()
        raise


def _manifest() -> dict[str, object]:
    return {
        "schema_version": "passive-muon-p24-immutable-remote-module-image-v1",
        "torch_version": TORCH_VERSION,
        "site_packages": str(SITE_PACKAGES),
        "instantiator": {
            "path": str(SITE_PACKAGES / INSTANTIATOR_RELATIVE),
            "before_sha256": ORIGINAL_INSTANTIATOR_SHA256,
            "before_byte_count": ORIGINAL_INSTANTIATOR_BYTES,
            "after_sha256": PATCHED_INSTANTIATOR_SHA256,
        },
        "remote_module": {
            "path": str(SITE_PACKAGES / REMOTE_MODULE_RELATIVE),
            "sha256": REMOTE_MODULE_SHA256,
            "byte_count": REMOTE_MODULE_BYTES,
        },
        "template": {
            "path": str(SITE_PACKAGES / TEMPLATE_RELATIVE),
            "sha256": TEMPLATE_SHA256,
            "byte_count": TEMPLATE_BYTES,
        },
        "generated_module": {
            "path": str(FIXED_GENERATED_MODULE),
            "sha256": GENERATED_MODULE_SHA256,
            "byte_count": GENERATED_MODULE_BYTES,
            "file_mode": "0444",
            "directory_mode": "0555",
        },
        "runtime_generation_policy": "verify exact existing non-scriptable source or fail closed",
        "stale_instantiator_and_generated_bytecode_removed": True,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def verify_preimage(site_packages: Path) -> tuple[bytes, bytes]:
    if importlib.metadata.version("torch") != TORCH_VERSION:
        raise PreparationError(
            f"torch version mismatch: expected {TORCH_VERSION}, "
            f"got {importlib.metadata.version('torch')}"
        )
    instantiator = _require_file(
        site_packages / INSTANTIATOR_RELATIVE,
        expected_sha256=ORIGINAL_INSTANTIATOR_SHA256,
        expected_bytes=ORIGINAL_INSTANTIATOR_BYTES,
        label="instantiator preimage",
    )
    _require_file(
        site_packages / REMOTE_MODULE_RELATIVE,
        expected_sha256=REMOTE_MODULE_SHA256,
        expected_bytes=REMOTE_MODULE_BYTES,
        label="remote_module source",
    )
    template_path = site_packages / TEMPLATE_RELATIVE
    _require_file(
        template_path,
        expected_sha256=TEMPLATE_SHA256,
        expected_bytes=TEMPLATE_BYTES,
        label="remote-module template",
    )
    generated = render_generated_module(template_path)
    if len(generated) != GENERATED_MODULE_BYTES:
        raise PreparationError(
            "generated module byte count mismatch: "
            f"expected {GENERATED_MODULE_BYTES}, got {len(generated)}"
        )
    generated_digest = sha256_bytes(generated)
    if generated_digest != GENERATED_MODULE_SHA256:
        raise PreparationError(
            "generated module SHA-256 mismatch: "
            f"expected {GENERATED_MODULE_SHA256}, got {generated_digest}"
        )
    return instantiator, generated


def verify_postimage(site_packages: Path) -> None:
    _require_file(
        site_packages / INSTANTIATOR_RELATIVE,
        expected_sha256=PATCHED_INSTANTIATOR_SHA256,
        expected_bytes=PATCHED_INSTANTIATOR_BYTES,
        label="patched instantiator",
    )
    _require_file(
        site_packages / REMOTE_MODULE_RELATIVE,
        expected_sha256=REMOTE_MODULE_SHA256,
        expected_bytes=REMOTE_MODULE_BYTES,
        label="remote_module source",
    )
    _require_file(
        site_packages / TEMPLATE_RELATIVE,
        expected_sha256=TEMPLATE_SHA256,
        expected_bytes=TEMPLATE_BYTES,
        label="remote-module template",
    )
    _require_file(
        site_packages / FIXED_DIRECTORY_RELATIVE / GENERATED_MODULE_FILENAME,
        expected_sha256=GENERATED_MODULE_SHA256,
        expected_bytes=GENERATED_MODULE_BYTES,
        label="immutable generated module",
    )
    generated_directory = site_packages / FIXED_DIRECTORY_RELATIVE
    if stat.S_IMODE(generated_directory.stat().st_mode) != 0o555:
        raise PreparationError("immutable generated-module directory mode is not 0555")
    if stat.S_IMODE((generated_directory / GENERATED_MODULE_FILENAME).stat().st_mode) != 0o444:
        raise PreparationError("immutable generated-module file mode is not 0444")
    if _matching_stale_bytecode(site_packages):
        raise PreparationError("stale instantiator or generated-module bytecode remains")


def prepare() -> None:
    instantiator, generated = verify_preimage(SITE_PACKAGES)
    patched = transform_instantiator(instantiator)
    patched_digest = sha256_bytes(patched)
    if patched_digest != PATCHED_INSTANTIATOR_SHA256:
        raise PreparationError(
            "patched instantiator SHA-256 mismatch: "
            f"expected {PATCHED_INSTANTIATOR_SHA256}, got {patched_digest}"
        )
    compile(patched, str(SITE_PACKAGES / INSTANTIATOR_RELATIVE), "exec")
    _remove_stale_bytecode(SITE_PACKAGES)
    try:
        FIXED_DIRECTORY.mkdir(mode=0o755)
    except OSError as error:
        raise PreparationError(
            f"cannot create immutable generated-module directory: {FIXED_DIRECTORY}"
        ) from error
    _write_new_exact(FIXED_GENERATED_MODULE, generated, mode=0o644)
    _replace_exact(
        SITE_PACKAGES / INSTANTIATOR_RELATIVE,
        patched,
        mode=0o644,
    )
    os.chmod(FIXED_GENERATED_MODULE, 0o444)
    os.chmod(FIXED_DIRECTORY, 0o555)
    _write_new_exact(MANIFEST_PATH, _canonical_json_bytes(_manifest()), mode=0o444)
    verify_postimage(SITE_PACKAGES)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="verify an already prepared image without writing any file",
    )
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        if arguments.verify_only:
            verify_postimage(SITE_PACKAGES)
            expected_manifest = _canonical_json_bytes(_manifest())
            _require_file(
                MANIFEST_PATH,
                expected_sha256=sha256_bytes(expected_manifest),
                expected_bytes=len(expected_manifest),
                label="P24 image preparation manifest",
            )
        else:
            prepare()
    except (OSError, PreparationError, SyntaxError) as error:
        print(f"P24 immutable-module preparation blocked: {error}", file=os.sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
