#!/usr/bin/env python3
"""One-read, hash-bound P26 failure ingestion for the P27 evidence packet."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
P23_CORE_PATH: Final = ROOT / "experiments/training/p23_deterministic_cuda_shadow_trace.py"
P23_RUNNER_PATH: Final = ROOT / "experiments/training/run_p23_deterministic_cuda_shadow_trace.py"
EXPECTED_P23_CORE_SHA256: Final = "ec1037839a3de378ee7391ae815ffc29c184865c5216d30027edc460ba112fab"
EXPECTED_P23_RUNNER_SHA256: Final = (
    "a2b4bb5b686b7f681958d09be1d465917b40e34d45e4d3503efef0f35e7ae8cb"
)
EXPECTED_NATIVE_SHA256: Final = "b2a92b5062988534bc424d3ecadf3977ecedc0f80aa27628d4a0457858a64c91"
EXPECTED_NATIVE_BYTE_COUNT: Final = 66_283
EXPECTED_NATIVE_SCHEMA: Final = "passive-muon-p23-native-failure-manifest-v1"
EXPECTED_ROLE: Final = "trace_off_a"
EXPECTED_ERROR: Final = "deleted file-backed mapping is forbidden at /proc/self/maps line 17"
EXPECTED_SOURCE_UID: Final = 0
EXPECTED_SOURCE_GID: Final = 0
_PATH_ROOT_LABELS: Final = {
    "repository",
    "preprocessor_alias",
    "nanogpt",
    "muon",
    "data",
    "python_environment",
    "native",
}


class P27IngestionError(RuntimeError):
    """Raised when retained P26 evidence cannot be copied and sanitized exactly."""


class DuplicateKeyError(ValueError):
    """Raised when retained JSON repeats a mapping key."""


def _load_module(name: str, path: Path) -> ModuleType:
    specification = importlib.util.spec_from_file_location(name, path)
    if specification is None or specification.loader is None:
        raise P27IngestionError(f"cannot load reviewed source: {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[name] = module
    specification.loader.exec_module(module)
    return module


def _reviewed_source_sha256(path: Path, *, label: str) -> str:
    if not hasattr(os, "O_NOFOLLOW"):
        raise P27IngestionError(f"O_NOFOLLOW is required for {label}")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise P27IngestionError(f"{label} must be one nonsymlink regular file")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0))
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise P27IngestionError(f"{label} changed before it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(opened) != _file_identity(after):
        raise P27IngestionError(f"{label} changed while it was read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise P27IngestionError(f"{label} byte count changed while it was read")
    return hashlib.sha256(raw).hexdigest()


def _strict_json(raw: bytes) -> Mapping[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise DuplicateKeyError(key)
            result[key] = value
        return result

    def reject_nonfinite(token: str) -> object:
        raise ValueError(f"nonfinite JSON constant: {token}")

    parsed = json.loads(
        raw,
        object_pairs_hook=reject_duplicates,
        parse_constant=reject_nonfinite,
    )
    if not isinstance(parsed, Mapping):
        raise P27IngestionError("retained native failure root is not an object")
    return parsed


def _file_identity(metadata: os.stat_result) -> tuple[int, ...]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_mode,
        metadata.st_uid,
        metadata.st_gid,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def _read_once(path: Path) -> tuple[bytes, os.stat_result]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise P27IngestionError("O_NOFOLLOW is required for retained evidence ingestion")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode):
        raise P27IngestionError("retained native failure must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise P27IngestionError("retained native failure is not a regular file")
    flags = os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise P27IngestionError("retained native failure is not a regular file")
        if _file_identity(before) != _file_identity(opened):
            raise P27IngestionError("retained native failure changed before it was opened")
        if stat.S_IMODE(opened.st_mode) != 0o600:
            raise P27IngestionError("retained P26 native failure mode differs from 0600")
        if opened.st_uid != EXPECTED_SOURCE_UID or opened.st_gid != EXPECTED_SOURCE_GID:
            raise P27IngestionError("retained P26 native failure owner differs from root:root")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1 << 20)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    if _file_identity(opened) != _file_identity(after):
        raise P27IngestionError("retained native failure changed during its one read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise P27IngestionError("retained native failure byte count changed during its one read")
    return raw, after


def _write_new(path: Path, raw: bytes, *, mode: int) -> None:
    if path.is_symlink() or os.path.lexists(path):
        raise P27IngestionError(f"output already exists: {path}")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, mode)
    try:
        os.fchmod(descriptor, mode)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != mode:
            raise P27IngestionError(f"output mode differs from {mode:04o}: {path}")
    finally:
        os.close(descriptor)


def _normalized_path_roots(path_roots: Mapping[str, Path]) -> dict[str, Path]:
    if set(path_roots) != _PATH_ROOT_LABELS:
        raise P27IngestionError("path-root label inventory differs from frozen P23")
    result: dict[str, Path] = {}
    for name, raw_path in path_roots.items():
        path = Path(raw_path)
        if not path.is_absolute():
            raise P27IngestionError("path-root values must be absolute")
        result[name] = path.resolve()
    return result


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _preflight_paths(
    *,
    source: Path,
    native_copy: Path,
    sanitized_output: Path,
    output_root: Path,
    repository_root: Path,
) -> tuple[Path, Path]:
    repository = repository_root.resolve(strict=True)
    if not repository.is_dir():
        raise P27IngestionError("repository root is not a directory")
    source_parent = source.parent.resolve(strict=True)
    native_parent = native_copy.parent.resolve(strict=True)
    sanitized_parent = sanitized_output.parent.resolve(strict=True)
    if not source_parent.is_dir() or not native_parent.is_dir():
        raise P27IngestionError("P26 and P27 evidence directories must already exist")
    if native_parent != sanitized_parent:
        raise P27IngestionError("P27 outputs need one preexisting external evidence directory")
    if output_root.is_symlink():
        raise P27IngestionError("declared P27 output root must not be a symlink")
    resolved_output_root = output_root.resolve(strict=True)
    if not resolved_output_root.is_dir() or resolved_output_root != native_parent:
        raise P27IngestionError("declared output root differs from the P27 evidence directory")
    if source_parent == native_parent:
        raise P27IngestionError("P27 outputs must not modify the terminal P26 evidence directory")
    resolved_source = source.resolve(strict=True)
    resolved_native = native_parent / native_copy.name
    resolved_sanitized = sanitized_parent / sanitized_output.name
    if any(
        _is_within(path, repository)
        for path in (resolved_source, resolved_native, resolved_sanitized)
    ):
        raise P27IngestionError("retained source and ingestion outputs must remain outside repo")
    if len({resolved_source, resolved_native, resolved_sanitized}) != 3:
        raise P27IngestionError("retained source and ingestion outputs must not alias")
    for output in (resolved_native, resolved_sanitized):
        if output.is_symlink() or os.path.lexists(output):
            raise P27IngestionError(f"output already exists: {output}")
    return resolved_native, resolved_sanitized


def _path_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        path = Path(raw_path)
        if not separator or not name or name in result or not path.is_absolute():
            raise P27IngestionError("--path-root must be unique NAME=/absolute/path entries")
        result[name] = path
    return _normalized_path_roots(result)


def ingest(
    *,
    source: Path,
    native_copy: Path,
    sanitized_output: Path,
    path_roots: Mapping[str, Path],
    output_root: Path,
    p23_core_path: Path = P23_CORE_PATH,
    p23_runner_path: Path = P23_RUNNER_PATH,
    repository_root: Path = ROOT,
) -> dict[str, object]:
    """Read one retained P26 file once, validate it, and publish two new files."""

    roots = _normalized_path_roots(path_roots)
    resolved_native, resolved_sanitized = _preflight_paths(
        source=source,
        native_copy=native_copy,
        sanitized_output=sanitized_output,
        output_root=output_root,
        repository_root=repository_root,
    )
    raw, source_stat = _read_once(source)
    digest = hashlib.sha256(raw).hexdigest()
    if digest != EXPECTED_NATIVE_SHA256 or len(raw) != EXPECTED_NATIVE_BYTE_COUNT:
        raise P27IngestionError("retained P26 native failure binding differs")
    payload = _strict_json(raw)
    failure = payload.get("failure")
    if (
        payload.get("schema_version") != EXPECTED_NATIVE_SCHEMA
        or payload.get("status") != "blocked_before_success_artifact"
        or payload.get("acquisition_role") != EXPECTED_ROLE
        or payload.get("exit_code") != 2
        or not isinstance(failure, Mapping)
        or failure.get("phase") != "p22_initialized"
        or failure.get("error_message") != EXPECTED_ERROR
    ):
        raise P27IngestionError("retained P26 native failure semantics differ")

    source_bindings = (
        (
            p23_core_path,
            EXPECTED_P23_CORE_SHA256,
            "corrected P23 sanitizer source",
            "p27_ingestion_p23_core",
        ),
        (
            p23_runner_path,
            EXPECTED_P23_RUNNER_SHA256,
            "frozen P23 validation authority",
            "p27_ingestion_p23_runner",
        ),
    )
    reviewed_modules: list[ModuleType] = []
    for path, expected_sha256, label, module_name in source_bindings:
        if _reviewed_source_sha256(path, label=label) != expected_sha256:
            raise P27IngestionError(f"{label} SHA-256 differs")
        reviewed_modules.append(_load_module(module_name, path))
        if _reviewed_source_sha256(path, label=label) != expected_sha256:
            raise P27IngestionError(f"{label} changed while it was loaded")
    p23_core, runner = reviewed_modules
    try:
        runner._validate_sanitizable_artifact(payload)
        sanitized = p23_core.sanitize_complete_manifest(payload, path_roots=roots)
    except (p23_core.P23ProvenanceError, runner.P23AcquisitionError) as error:
        raise P27IngestionError(
            f"frozen P23 validation or sanitization blocked: {error}"
        ) from error
    sanitized["native_artifact_sha256"] = digest
    sanitized["native_artifact_byte_count"] = len(raw)
    sanitized["p27_ingestion"] = {
        "schema_version": "passive-muon-p27-p26-failure-ingestion-v1",
        "source_read_count": 1,
        "source_mutations_performed": [],
        "source_mode_octal": "0600",
        "source_uid": source_stat.st_uid,
        "source_gid": source_stat.st_gid,
    }
    sanitized_raw = (
        json.dumps(sanitized, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    # Repeat the complete two-output freshness preflight immediately before the
    # first write. O_EXCL remains the final race-resistant no-overwrite guard.
    resolved_native, resolved_sanitized = _preflight_paths(
        source=source,
        native_copy=resolved_native,
        sanitized_output=resolved_sanitized,
        output_root=output_root,
        repository_root=repository_root,
    )
    _write_new(resolved_native, raw, mode=0o600)
    _write_new(resolved_sanitized, sanitized_raw, mode=0o600)
    return {
        "schema_version": "passive-muon-p27-p26-failure-ingestion-result-v1",
        "passes": True,
        "source_read_count": 1,
        "source_mutations_performed": [],
        "native_sha256": digest,
        "native_byte_count": len(raw),
        "native_copy_sha256": hashlib.sha256(resolved_native.read_bytes()).hexdigest(),
        "sanitized_sha256": hashlib.sha256(sanitized_raw).hexdigest(),
        "sanitized_byte_count": len(sanitized_raw),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--native-copy", type=Path, required=True)
    parser.add_argument("--sanitized-output", type=Path, required=True)
    parser.add_argument("--path-root", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--p23-core", type=Path, default=P23_CORE_PATH)
    parser.add_argument("--p23-runner", type=Path, default=P23_RUNNER_PATH)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = ingest(
            source=args.source,
            native_copy=args.native_copy,
            sanitized_output=args.sanitized_output,
            path_roots=_path_roots(args.path_root),
            output_root=args.output_root,
            p23_core_path=args.p23_core,
            p23_runner_path=args.p23_runner,
            repository_root=args.repository_root,
        )
    except (OSError, UnicodeError, ValueError, P27IngestionError) as error:
        print(f"P27 ingestion blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
