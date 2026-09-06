#!/usr/bin/env python3
"""Verify the P26 diagnostic bridge from inside the locked root container.

The retained sanitized diagnostic is deliberately ``root:root 0600``.  This
verifier therefore runs as root in the already-attested acquisition container,
reads both sanitized copies exactly once, and authenticates their equality
before validating the retained native diagnostic and every P25 frozen source.
It never changes either file.
"""

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
from typing import Final

SANITIZED_SCHEMA: Final = "passive-muon-p25-sanitized-executable-origin-diagnostic-v1"
NATIVE_SCHEMA: Final = "passive-muon-p25-executable-origin-diagnostic-v1"
CONTRACT_SCHEMA: Final = "passive-muon-p25-cuda-diagnostic-correction-contract-v1"
CONTRACT_STATUS: Final = "frozen_pre_remediation_build_and_pre_acquisition"
TRANSCRIPT_SCHEMA: Final = "passive-muon-p26-permission-safe-bridge-verification-v1"
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")


class BridgeVerificationError(RuntimeError):
    """Raised when the permission-safe bridge must fail closed."""


class StableFile:
    """Bytes and metadata captured by one no-follow file read."""

    __slots__ = ("byte_count", "device", "gid", "inode", "mode", "raw", "sha256", "uid")

    def __init__(
        self,
        *,
        raw: bytes,
        sha256: str,
        byte_count: int,
        mode: int,
        uid: int,
        gid: int,
        device: int,
        inode: int,
    ) -> None:
        self.raw = raw
        self.sha256 = sha256
        self.byte_count = byte_count
        self.mode = mode
        self.uid = uid
        self.gid = gid
        self.device = device
        self.inode = inode


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise BridgeVerificationError(f"duplicate JSON key is forbidden: {key}")
        result[key] = value
    return result


def _reject_nonfinite(token: str) -> object:
    raise BridgeVerificationError(f"nonfinite JSON constant is forbidden: {token}")


def load_json_strict(raw: bytes, *, label: str) -> object:
    """Parse captured UTF-8 JSON, rejecting duplicate keys and nonfinite values."""

    try:
        return json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_nonfinite,
        )
    except UnicodeDecodeError as error:
        raise BridgeVerificationError(f"{label} is not UTF-8") from error
    except json.JSONDecodeError as error:
        raise BridgeVerificationError(f"{label} is not valid JSON") from error


def _identity(metadata: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        metadata.st_ctime_ns,
    )


def read_regular_once(path: Path, *, label: str) -> StableFile:
    """Read one regular, nonsymlink file once while detecting replacement."""

    try:
        before = path.lstat()
    except OSError as error:
        raise BridgeVerificationError(f"{label} is missing or unreadable") from error
    if stat.S_ISLNK(before.st_mode):
        raise BridgeVerificationError(f"{label} must not be a symlink")
    if not stat.S_ISREG(before.st_mode):
        raise BridgeVerificationError(f"{label} must be a regular file")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = -1
    try:
        descriptor = os.open(path, flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode) or (
            opened.st_dev,
            opened.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise BridgeVerificationError(f"{label} changed before it was opened")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
    except OSError as error:
        raise BridgeVerificationError(f"{label} could not be read safely") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)

    if _identity(before) != _identity(opened) or _identity(opened) != _identity(after):
        raise BridgeVerificationError(f"{label} changed while it was read")
    raw = b"".join(chunks)
    if len(raw) != after.st_size:
        raise BridgeVerificationError(f"{label} byte count changed while it was read")
    return StableFile(
        raw=raw,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_count=len(raw),
        mode=stat.S_IMODE(after.st_mode),
        uid=after.st_uid,
        gid=after.st_gid,
        device=after.st_dev,
        inode=after.st_ino,
    )


def _mapping(value: object, *, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise BridgeVerificationError(f"{label} must be a JSON object")
    return value


def _validate_wrapper(value: object, *, label: str) -> Mapping[str, object]:
    wrapper = _mapping(value, label=label)
    if wrapper.get("schema_version") != SANITIZED_SCHEMA:
        raise BridgeVerificationError(f"{label} schema differs")
    if wrapper.get("native_schema_version") != NATIVE_SCHEMA:
        raise BridgeVerificationError(f"{label} native schema binding differs")
    if (
        wrapper.get("mode") != "remediated"
        or wrapper.get("status") != "passes"
        or wrapper.get("passes") is not True
    ):
        raise BridgeVerificationError(f"{label} does not authorize acquisition")
    binding = _mapping(wrapper.get("native_artifact"), label=f"{label} native binding")
    if set(binding) != {"byte_count", "sha256"}:
        raise BridgeVerificationError(f"{label} native binding schema differs")
    if type(binding.get("byte_count")) is not int or binding.get("byte_count", 0) <= 0:
        raise BridgeVerificationError(f"{label} native byte count differs")
    if (
        not isinstance(binding.get("sha256"), str)
        or _SHA256.fullmatch(str(binding.get("sha256"))) is None
    ):
        raise BridgeVerificationError(f"{label} native SHA-256 differs")
    manifest = _mapping(wrapper.get("manifest"), label=f"{label} manifest")
    if (
        manifest.get("schema_version") != NATIVE_SCHEMA
        or manifest.get("mode") != "remediated"
        or manifest.get("status") != "passes"
        or manifest.get("passes") is not True
    ):
        raise BridgeVerificationError(f"{label} manifest does not record a pass")
    return wrapper


def _load_module(path: Path) -> object:
    specification = importlib.util.spec_from_file_location("p26_frozen_p25_sanitizer", path)
    if specification is None or specification.loader is None:
        raise BridgeVerificationError("cannot load the frozen P25 sanitizer")
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def _validate_contract_and_sources(
    *, contract_raw: bytes, repository: Path, sanitizer_path: Path
) -> tuple[Mapping[str, object], int]:
    contract = _mapping(load_json_strict(contract_raw, label="P25 contract"), label="P25 contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise BridgeVerificationError("P25 contract schema differs")
    if contract.get("status") != CONTRACT_STATUS:
        raise BridgeVerificationError("P25 contract status differs")

    records: list[Mapping[str, object]] = []
    for group_name in ("unchanged_authorities", "execution_sources"):
        group = _mapping(contract.get(group_name), label=f"P25 contract {group_name}")
        records.extend(_mapping(item, label="P25 source record") for item in group.values())
    remediation = _mapping(contract.get("remediation_design"), label="P25 remediation design")
    records.extend(
        _mapping(remediation.get(name), label=f"P25 remediation source {name}")
        for name in ("build_patch", "image_recipe")
    )

    resolved_repository = repository.resolve(strict=True)
    if not resolved_repository.is_dir():
        raise BridgeVerificationError("repository is not a directory")
    observed_sanitizer = False
    for record in records:
        relative = record.get("path")
        expected = record.get("sha256")
        if (
            not isinstance(relative, str)
            or Path(relative).is_absolute()
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
        ):
            raise BridgeVerificationError("P25 source binding is malformed")
        try:
            source = (resolved_repository / relative).resolve(strict=True)
        except OSError as error:
            raise BridgeVerificationError(f"P25 frozen source is absent: {relative}") from error
        if source != resolved_repository and resolved_repository not in source.parents:
            raise BridgeVerificationError(f"P25 source escapes the repository: {relative}")
        observed = read_regular_once(source, label=f"P25 frozen source {relative}")
        if observed.sha256 != expected:
            raise BridgeVerificationError(f"P25 frozen source differs: {relative}")
        if source == sanitizer_path.resolve(strict=True):
            observed_sanitizer = True
    if not observed_sanitizer:
        raise BridgeVerificationError("frozen P25 sanitizer is absent from contract sources")
    return contract, len(records)


def _verify_bound_files(
    *,
    repository_wrapper: Path,
    retained_wrapper: Path,
    retained_native: Path,
    required_uid: int,
    required_gid: int,
) -> tuple[StableFile, StableFile, StableFile, Mapping[str, object]]:
    """Verify the permission-sensitive files and return their captured bytes."""

    repository_record = read_regular_once(repository_wrapper, label="committed repository wrapper")
    retained_record = read_regular_once(retained_wrapper, label="retained sanitized wrapper")
    if repository_record.mode != 0o644:
        raise BridgeVerificationError("committed repository wrapper must be mode 0644")
    if retained_record.uid != required_uid or retained_record.gid != required_gid:
        raise BridgeVerificationError("retained sanitized wrapper must remain root:root")
    if retained_record.mode != 0o600:
        raise BridgeVerificationError("retained sanitized wrapper must remain mode 0600")
    if repository_record.raw != retained_record.raw:
        raise BridgeVerificationError("committed and retained diagnostic wrappers differ")

    # Strict-parse both captured byte strings independently.  Do not reopen
    # either path after the equality decision.
    repository_payload = _validate_wrapper(
        load_json_strict(repository_record.raw, label="committed repository wrapper"),
        label="committed repository wrapper",
    )
    retained_payload = _validate_wrapper(
        load_json_strict(retained_record.raw, label="retained sanitized wrapper"),
        label="retained sanitized wrapper",
    )
    if dict(repository_payload) != dict(retained_payload):
        raise BridgeVerificationError("strictly parsed diagnostic wrappers differ")

    native_record = read_regular_once(retained_native, label="retained native diagnostic")
    if native_record.uid != required_uid or native_record.gid != required_gid:
        raise BridgeVerificationError("retained native diagnostic must remain root:root")
    if native_record.mode != 0o600:
        raise BridgeVerificationError("retained native diagnostic must remain mode 0600")
    binding = _mapping(
        retained_payload.get("native_artifact"), label="retained native artifact binding"
    )
    if binding.get("byte_count") != native_record.byte_count:
        raise BridgeVerificationError("retained native byte count differs from wrapper")
    if binding.get("sha256") != native_record.sha256:
        raise BridgeVerificationError("retained native SHA-256 differs from wrapper")
    native_payload = _mapping(
        load_json_strict(native_record.raw, label="retained native diagnostic"),
        label="retained native diagnostic",
    )
    return repository_record, retained_record, native_record, native_payload


def verify_bridge(
    *,
    repository_wrapper: Path,
    retained_wrapper: Path,
    retained_native: Path,
    contract_path: Path,
    repository: Path,
    sanitizer_path: Path,
) -> dict[str, object]:
    """Authenticate both wrappers and reconstruct the full native diagnostic."""

    if os.geteuid() != 0:
        raise BridgeVerificationError("permission-safe verifier must run as root")

    repository_record, retained_record, native_record, native_payload = _verify_bound_files(
        repository_wrapper=repository_wrapper,
        retained_wrapper=retained_wrapper,
        retained_native=retained_native,
        required_uid=0,
        required_gid=0,
    )

    contract_record = read_regular_once(contract_path, label="P25 contract")
    _, source_count = _validate_contract_and_sources(
        contract_raw=contract_record.raw,
        repository=repository,
        sanitizer_path=sanitizer_path,
    )
    sanitizer = _load_module(sanitizer_path)
    sanitizer.reject_credentials_and_nonfinite(native_payload)
    sanitizer.validate_native_semantics(native_payload)

    return {
        "schema_version": TRANSCRIPT_SCHEMA,
        "status": "passes",
        "passes": True,
        "wrapper": {
            "byte_count": retained_record.byte_count,
            "sha256": retained_record.sha256,
            "repository_mode_octal": f"{repository_record.mode:04o}",
            "retained_mode_octal": f"{retained_record.mode:04o}",
            "retained_uid": retained_record.uid,
            "retained_gid": retained_record.gid,
            "exact_bytes_equal": True,
            "strictly_parsed_both_captured_copies": True,
        },
        "native_artifact": {
            "byte_count": native_record.byte_count,
            "sha256": native_record.sha256,
            "binding_matches": True,
            "strict_semantics_reconstructed": True,
            "mode_octal": f"{native_record.mode:04o}",
            "uid": native_record.uid,
            "gid": native_record.gid,
        },
        "frozen_p25_source_count": source_count,
        "mutations_performed": [],
        "claim_boundary": (
            "Permission-safe authentication of one fresh P25 diagnostic bridge; "
            "not a CUDA gradient, fidelity, or training result."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-wrapper", required=True, type=Path)
    parser.add_argument("--retained-wrapper", required=True, type=Path)
    parser.add_argument("--retained-native", required=True, type=Path)
    parser.add_argument("--contract", required=True, type=Path)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--sanitizer", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    arguments = _parse_args()
    try:
        result = verify_bridge(
            repository_wrapper=arguments.repository_wrapper,
            retained_wrapper=arguments.retained_wrapper,
            retained_native=arguments.retained_native,
            contract_path=arguments.contract,
            repository=arguments.repository,
            sanitizer_path=arguments.sanitizer,
        )
    except (BridgeVerificationError, OSError, TypeError, ValueError) as error:
        print(f"P26 permission-safe bridge verification blocked: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
