#!/usr/bin/env python3
"""Sanitize one hash-bound P27 CUDA mapping-localization artifact."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Final

ROOT: Final = Path(__file__).resolve().parents[1]
RUNNER_PATH: Final = ROOT / "experiments/training/run_p27_cuda_deleted_mapping_localization.py"
EXPECTED_RUNNER_SHA256: Final = "86ba200d69a029c64b2218581d6b92245930d5f1e60d0176aca22ad4d6ddf65a"
SCHEMA: Final = "passive-muon-p27-sanitized-cuda-deleted-mapping-localization-v1"
EXPECTED_NATIVE_UID: Final = 0
EXPECTED_NATIVE_GID: Final = 0
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_ROOT_LABEL: Final = re.compile(r"^[a-z][a-z0-9_]*$")
_HEX_BYTES: Final = re.compile(r"^(?:[0-9a-f]{2})*$")
_SAFE_URI: Final = re.compile(r"^(?:https?|ssh|git)://")
_ABSOLUTE_FRAGMENT: Final = re.compile(rb'(?<![A-Za-z0-9._~:/-])(/[^\s"\x27<>\(\)\[\]\{\},;\|]+)')
_BYTE_HEX_FIELDS: Final = {
    "pathname_bytes_hex": "mapping_pathname",
    "pathname_without_deleted_bytes_hex": "mapping_pathname_without_deleted_suffix",
    "readlink_target_bytes_hex": "readlink_target",
}
_SUCCESS_FIELDS: Final = {
    "schema_version",
    "status",
    "passes",
    "scope",
    "seed",
    "process",
    "bindings",
    "live_runtime",
    "stage_order",
    "stage_snapshots",
    "construction",
    "localization",
    "training_operations",
    "policy",
    "checks",
    "claim_boundary",
}
_FAILURE_FIELDS: Final = {
    "schema_version",
    "status",
    "passes",
    "scope",
    "seed",
    "process",
    "bindings",
    "stage_order",
    "stage_snapshots",
    "training_operations",
    "policy",
    "error",
    "claim_boundary",
}


class P27SanitizationError(RuntimeError):
    """Raised when native P27 evidence cannot be validated and sanitized safely."""


class DuplicateKeyError(ValueError):
    """Raised when JSON repeats a mapping key."""


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


def _read_regular_once(
    path: Path,
    *,
    label: str,
    required_mode: int | None = None,
    required_uid: int | None = None,
    required_gid: int | None = None,
) -> tuple[bytes, os.stat_result]:
    if not hasattr(os, "O_NOFOLLOW"):
        raise P27SanitizationError(f"O_NOFOLLOW is required for {label}")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise P27SanitizationError(f"{label} must be one nonsymlink regular file")
    descriptor = os.open(
        path,
        os.O_RDONLY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if _file_identity(before) != _file_identity(opened):
            raise P27SanitizationError(f"{label} changed before it was opened")
        if required_mode is not None and stat.S_IMODE(opened.st_mode) != required_mode:
            raise P27SanitizationError(f"{label} mode differs from {required_mode:04o}")
        if required_uid is not None and opened.st_uid != required_uid:
            raise P27SanitizationError(f"{label} UID differs")
        if required_gid is not None and opened.st_gid != required_gid:
            raise P27SanitizationError(f"{label} GID differs")
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
        raise P27SanitizationError(f"{label} changed during its one read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise P27SanitizationError(f"{label} byte count changed during its one read")
    return raw, after


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
        raise P27SanitizationError("native P27 localization root is not an object")
    return parsed


def _load_runner(path: Path) -> ModuleType:
    raw, _ = _read_regular_once(path, label="P27 localization validation authority")
    if hashlib.sha256(raw).hexdigest() != EXPECTED_RUNNER_SHA256:
        raise P27SanitizationError("P27 localization validation authority SHA-256 differs")
    specification = importlib.util.spec_from_file_location(
        "p27_sanitizer_localization_authority", path
    )
    if specification is None or specification.loader is None:
        raise P27SanitizationError("cannot load P27 localization validation authority")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    code = compile(raw, str(path), "exec")
    exec(code, module.__dict__)
    return module


def _validate_snapshot_prefix(runner: ModuleType, value: object) -> None:
    if not isinstance(value, list):
        raise P27SanitizationError("failure stage snapshots are not a list")
    observed_stages: list[object] = []
    for entry in value:
        if not isinstance(entry, Mapping) or set(entry) != runner.SNAPSHOT_FIELDS:
            raise P27SanitizationError("failure stage snapshot schema differs")
        observed_stages.append(entry.get("stage"))
        mappings = entry.get("mappings")
        if not isinstance(mappings, list) or entry.get("mapping_count") != len(mappings):
            raise P27SanitizationError("failure snapshot mapping count differs")
        deleted_count = 0
        for record in mappings:
            if not isinstance(record, Mapping) or set(record) != runner.MAPPING_FIELDS:
                raise P27SanitizationError("failure mapping schema differs")
            if record.get("classification") not in runner._CLASSIFICATIONS:
                raise P27SanitizationError("failure mapping classification differs")
            if record.get("deleted_suffix_present") is True:
                deleted_count += 1
            map_probe = record.get("map_files_probe")
            fd_probes = record.get("matching_fd_probes")
            if not isinstance(map_probe, Mapping) or set(map_probe) != runner.MAP_FILES_FIELDS:
                raise P27SanitizationError("failure map_files probe schema differs")
            if not isinstance(fd_probes, list) or any(
                not isinstance(probe, Mapping) or set(probe) != runner.FD_PROBE_FIELDS
                for probe in fd_probes
            ):
                raise P27SanitizationError("failure matching-FD probe schema differs")
        if entry.get("deleted_mapping_count") != deleted_count:
            raise P27SanitizationError("failure deleted-mapping count differs")
    if observed_stages != list(runner.STAGES[: len(observed_stages)]):
        raise P27SanitizationError("failure stage snapshots are not an ordered prefix")


def _validate_failure_payload(runner: ModuleType, payload: Mapping[str, object]) -> None:
    if set(payload) != _FAILURE_FIELDS:
        raise P27SanitizationError("P27 failure field inventory differs")
    if (
        payload.get("schema_version") != runner.SCHEMA
        or payload.get("status") != "blocked"
        or payload.get("passes") is not False
        or payload.get("scope")
        != "four-stage CUDA deleted-mapping localization; no training computation"
        or payload.get("seed") != runner.SEED
        or payload.get("stage_order") != list(runner.STAGES)
    ):
        raise P27SanitizationError("P27 failure identity differs")
    process = payload.get("process")
    if (
        not isinstance(process, Mapping)
        or set(process) != {"pid", "started_time_ns", "failed_time_ns", "argv"}
        or type(process.get("pid")) is not int
        or (
            process.get("started_time_ns") is not None
            and type(process.get("started_time_ns")) is not int
        )
        or type(process.get("failed_time_ns")) is not int
        or not isinstance(process.get("argv"), list)
        or not all(isinstance(item, str) for item in process["argv"])
    ):
        raise P27SanitizationError("P27 failure process record differs")
    bindings = payload.get("bindings")
    if bindings is not None and not isinstance(bindings, Mapping):
        raise P27SanitizationError("P27 failure bindings are malformed")
    _validate_snapshot_prefix(runner, payload.get("stage_snapshots"))
    operations = payload.get("training_operations")
    if (
        not isinstance(operations, Mapping)
        or set(operations) != runner.TRAINING_OPERATION_FIELDS
        or any(type(value) is not int or value != 0 for value in operations.values())
    ):
        raise P27SanitizationError("P27 failure claims a training operation")
    policy = payload.get("policy")
    if (
        not isinstance(policy, Mapping)
        or set(policy) != {"classification_performed", "allowlist_changed", "statement"}
        or policy.get("classification_performed") is not False
        or policy.get("allowlist_changed") is not False
        or policy.get("statement") != "Failure evidence only; no mapping policy decision was made."
    ):
        raise P27SanitizationError("P27 failure policy record differs")
    error = payload.get("error")
    if (
        not isinstance(error, Mapping)
        or set(error) != {"class", "message", "traceback"}
        or not all(isinstance(error.get(name), str) for name in error)
    ):
        raise P27SanitizationError("P27 failure error record differs")
    if payload.get("claim_boundary") != (
        "This blocked non-training diagnostic is not repeatability, noninterference, "
        "gradient, fidelity, or training evidence."
    ):
        raise P27SanitizationError("P27 failure claim boundary differs")


def _validate_native(runner: ModuleType, payload: Mapping[str, object]) -> str:
    if payload.get("schema_version") != runner.SCHEMA:
        raise P27SanitizationError("unknown P27 native schema")
    if set(payload) == _SUCCESS_FIELDS:
        try:
            runner.validate_payload(payload)
        except runner.P27DiagnosticError as error:
            raise P27SanitizationError(f"P27 native validation failed: {error}") from error
        return "complete_payload"
    if set(payload) == _FAILURE_FIELDS:
        _validate_failure_payload(runner, payload)
        return "failure_payload"
    raise P27SanitizationError("unknown P27 native payload field inventory")


def _normalize_roots(path_roots: Mapping[str, Path]) -> dict[str, bytes]:
    if not path_roots:
        raise P27SanitizationError("at least one declared path root is required")
    result: dict[str, bytes] = {}
    for name, raw_path in path_roots.items():
        if _ROOT_LABEL.fullmatch(name) is None or name in result:
            raise P27SanitizationError("path-root labels must be unique lowercase identifiers")
        path = Path(raw_path)
        if not path.is_absolute():
            raise P27SanitizationError("path-root values must be absolute")
        encoded = os.fsencode(path.resolve(strict=False))
        if encoded == b"/":
            raise P27SanitizationError("filesystem root is too broad for path sanitization")
        normalized = encoded.rstrip(b"/")
        if normalized in result.values():
            raise P27SanitizationError("path-root values must be unique")
        result[name] = normalized
    return result


class _Redactor:
    def __init__(self, roots: Mapping[str, bytes]) -> None:
        self.roots = sorted(roots.items(), key=lambda item: len(item[1]), reverse=True)
        self.byte_field_replacement_count = 0
        self.text_replacement_count = 0
        self.key_replacement_count = 0
        self.argv_replacement_count = 0

    def _absolute_label(self, raw: bytes) -> str:
        for name, root in self.roots:
            if raw == root or raw.startswith(root + b"/"):
                return name
        raise P27SanitizationError(
            "unknown absolute path bytes cannot be included in the sanitized artifact"
        )

    def _embedded_labels(self, raw: bytes) -> list[str]:
        if b"file://" in raw:
            raise P27SanitizationError("file URI cannot be included in the sanitized artifact")
        labels: list[str] = []
        for match in _ABSOLUTE_FRAGMENT.finditer(raw):
            labels.append(self._absolute_label(match.group(1).rstrip(b".!?")))
        return sorted(set(labels))

    @staticmethod
    def _record(raw: bytes, labels: list[str]) -> dict[str, object]:
        return {
            "sha256": hashlib.sha256(raw).hexdigest(),
            "byte_count": len(raw),
            "logical_labels": labels,
        }

    def _redact_hex(self, value: object, *, fallback_label: str) -> object:
        if value is None:
            return None
        if not isinstance(value, str) or _HEX_BYTES.fullmatch(value) is None:
            raise P27SanitizationError(f"{fallback_label} is not canonical lowercase byte hex")
        raw = bytes.fromhex(value)
        labels = (
            [self._absolute_label(raw)]
            if raw.startswith(b"/")
            else [f"non_absolute_{fallback_label}"]
        )
        self.byte_field_replacement_count += 1
        return self._record(raw, labels)

    def _redact_text(self, value: str, *, fallback_label: str | None = None) -> object:
        raw = value.encode("utf-8", errors="surrogatepass")
        if _SAFE_URI.match(value):
            return value
        labels = self._embedded_labels(raw)
        if Path(value).is_absolute() and not labels:
            labels = [self._absolute_label(raw)]
        if fallback_label is None and not labels:
            return value
        if fallback_label is not None:
            labels = sorted(set([fallback_label, *labels]))
        self.text_replacement_count += 1
        return self._record(raw, labels)

    def _redact_key(self, key: str) -> str:
        transformed = self._redact_text(key)
        if isinstance(transformed, str):
            return transformed
        self.key_replacement_count += 1
        labels = "+".join(str(label) for label in transformed["logical_labels"])
        return f"redacted_absolute_key:{labels}:{transformed['byte_count']}:{transformed['sha256']}"

    def redact(self, value: object, *, field: str | None = None) -> object:
        if field in _BYTE_HEX_FIELDS:
            return self._redact_hex(value, fallback_label=_BYTE_HEX_FIELDS[field])
        if field == "argv":
            if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
                raise P27SanitizationError("process argv is malformed")
            result = []
            for item in value:
                result.append(self._redact_text(item, fallback_label="argv_argument"))
                self.argv_replacement_count += 1
            return result
        if isinstance(value, Mapping):
            result: dict[str, object] = {}
            for key, item in value.items():
                if not isinstance(key, str):
                    raise P27SanitizationError("native JSON mapping key is not text")
                redacted_key = self._redact_key(key)
                if redacted_key in result:
                    raise P27SanitizationError("path-key redaction created a duplicate key")
                result[redacted_key] = self.redact(item, field=key)
            return result
        if isinstance(value, list):
            return [self.redact(item) for item in value]
        if isinstance(value, tuple):
            return [self.redact(item) for item in value]
        if isinstance(value, str):
            return self._redact_text(value)
        return value


def _assert_no_absolute_path_strings(value: object) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _assert_no_absolute_path_strings(key)
            _assert_no_absolute_path_strings(item)
        return
    if isinstance(value, list):
        for item in value:
            _assert_no_absolute_path_strings(item)
        return
    if not isinstance(value, str) or _SAFE_URI.match(value):
        return
    raw = value.encode("utf-8", errors="surrogatepass")
    if b"file://" in raw or Path(value).is_absolute() or _ABSOLUTE_FRAGMENT.search(raw):
        raise P27SanitizationError("sanitized artifact retains absolute path bytes")


def _preflight_output(
    *, native: Path, output: Path, output_root: Path, repository_root: Path
) -> Path:
    repository = repository_root.resolve(strict=True)
    if not repository.is_dir():
        raise P27SanitizationError("repository root is not a directory")
    if output_root.is_symlink():
        raise P27SanitizationError("declared P27 output root must not be a symlink")
    resolved_output_root = output_root.resolve(strict=True)
    if not resolved_output_root.is_dir():
        raise P27SanitizationError("declared P27 output root is not a directory")
    output_parent = output.parent.resolve(strict=True)
    if output_parent != resolved_output_root:
        raise P27SanitizationError("sanitized output parent differs from its declared root")
    resolved_native = native.resolve(strict=True)
    resolved_output = output_parent / output.name
    if resolved_native == resolved_output:
        raise P27SanitizationError("sanitized output must not alias the native artifact")
    for path in (resolved_native, resolved_output):
        try:
            path.relative_to(repository)
        except ValueError:
            continue
        raise P27SanitizationError("native and sanitized output must remain outside repo")
    if output.is_symlink() or os.path.lexists(resolved_output):
        raise P27SanitizationError("sanitized output already exists or is a symlink")
    return resolved_output


def _write_new(path: Path, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    descriptor = os.open(path, flags, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise P27SanitizationError("sanitized output write made no progress")
            view = view[written:]
        os.fsync(descriptor)
        if stat.S_IMODE(os.fstat(descriptor).st_mode) != 0o600:
            raise P27SanitizationError("sanitized output mode differs from 0600")
    finally:
        os.close(descriptor)


def sanitize(
    *,
    native: Path,
    output: Path,
    expected_native_sha256: str,
    expected_native_byte_count: int,
    path_roots: Mapping[str, Path],
    output_root: Path,
    runner_source_path: Path = RUNNER_PATH,
    repository_root: Path = ROOT,
) -> dict[str, object]:
    """Authenticate one native artifact and publish one path-free wrapper."""

    if _SHA256.fullmatch(expected_native_sha256) is None:
        raise P27SanitizationError("expected native SHA-256 is malformed")
    if type(expected_native_byte_count) is not int or expected_native_byte_count <= 0:
        raise P27SanitizationError("expected native byte count is malformed")
    resolved_output = _preflight_output(
        native=native,
        output=output,
        output_root=output_root,
        repository_root=repository_root,
    )
    raw, metadata = _read_regular_once(
        native,
        label="native P27 localization artifact",
        required_mode=0o600,
        required_uid=EXPECTED_NATIVE_UID,
        required_gid=EXPECTED_NATIVE_GID,
    )
    digest = hashlib.sha256(raw).hexdigest()
    if digest != expected_native_sha256 or len(raw) != expected_native_byte_count:
        raise P27SanitizationError("native P27 localization SHA-256/size binding differs")
    payload = _strict_json(raw)
    runner = _load_runner(runner_source_path)
    payload_kind = _validate_native(runner, payload)
    roots = _normalize_roots(path_roots)
    redactor = _Redactor(roots)
    redacted = redactor.redact(payload)
    assert isinstance(redacted, dict)
    _assert_no_absolute_path_strings(redacted)
    wrapper = {
        "schema_version": SCHEMA,
        "status": "sanitized",
        "native_artifact": {
            "schema_version": runner.SCHEMA,
            "payload_kind": payload_kind,
            "sha256": digest,
            "byte_count": len(raw),
            "mode_octal": "0600",
            "uid": metadata.st_uid,
            "gid": metadata.st_gid,
        },
        "runner_authority_sha256": EXPECTED_RUNNER_SHA256,
        "logical_path_root_labels": sorted(roots),
        "redaction": {
            "byte_field_replacement_count": redactor.byte_field_replacement_count,
            "text_replacement_count": redactor.text_replacement_count,
            "key_replacement_count": redactor.key_replacement_count,
            "argv_replacement_count": redactor.argv_replacement_count,
            "unknown_absolute_paths_allowed": False,
            "nonpath_evidence_preserved": True,
        },
        "manifest": redacted,
        "claim_boundary": (
            "This wrapper sanitizes one native P27 non-training localization artifact. "
            "It is not repeatability, noninterference, gradient, fidelity, shield, "
            "optimizer-update, or training evidence."
        ),
    }
    _assert_no_absolute_path_strings(wrapper)
    rendered = (json.dumps(wrapper, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    resolved_output = _preflight_output(
        native=native,
        output=resolved_output,
        output_root=output_root,
        repository_root=repository_root,
    )
    _write_new(resolved_output, rendered)
    return {
        "schema_version": "passive-muon-p27-localization-sanitization-result-v1",
        "passes": True,
        "native_sha256": digest,
        "native_byte_count": len(raw),
        "sanitized_sha256": hashlib.sha256(rendered).hexdigest(),
        "sanitized_byte_count": len(rendered),
        "output_mode_octal": "0600",
    }


def _path_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        path = Path(raw_path)
        if not separator or not name or name in result or not path.is_absolute():
            raise P27SanitizationError(
                "--path-root must contain unique NAME=/absolute/path entries"
            )
        result[name] = path
    return result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-native-sha256", required=True)
    parser.add_argument("--expected-native-byte-count", type=int, required=True)
    parser.add_argument("--path-root", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--runner-source", type=Path, default=RUNNER_PATH)
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        result = sanitize(
            native=args.native,
            output=args.output,
            expected_native_sha256=args.expected_native_sha256,
            expected_native_byte_count=args.expected_native_byte_count,
            path_roots=_path_roots(args.path_root),
            output_root=args.output_root,
            runner_source_path=args.runner_source,
            repository_root=args.repository_root,
        )
    except (OSError, UnicodeError, ValueError, P27SanitizationError) as error:
        print(f"P27 localization sanitization blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
