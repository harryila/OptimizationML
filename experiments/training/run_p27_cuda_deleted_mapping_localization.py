#!/usr/bin/env python3
"""Localize CUDA-created deleted mappings without running training.

The diagnostic takes four synchronized ``/proc/self/maps`` snapshots: after
the frozen Torch configuration but before CUDA initialization, after explicit
CUDA initialization, after moving the pinned GPT-2-small model to CUDA, and
after constructing the pinned Muon optimizer.  It never performs a forward or
backward pass, computes a loss, calls an optimizer step, or produces a Muon
candidate.

This file is intentionally an observer, not a policy.  It preserves every
deleted mapping's kernel-reported fields and records fail-closed
``/proc/self/map_files`` and open-FD probes.  A later reviewed contract may use
those facts to change policy; this diagnostic does not allowlist anything.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import importlib.util
import json
import os
import random
import re
import stat
import subprocess
import sys
import tempfile
import time
import traceback
from collections.abc import Mapping
from pathlib import Path
from types import ModuleType
from typing import Final

SCRIPT_DIR: Final = Path(__file__).resolve().parent

SCHEMA: Final = "passive-muon-p27-cuda-deleted-mapping-localization-v1"
STAGES: Final = (
    "pre_cuda",
    "post_cuda_init",
    "post_model_move",
    "post_optimizer",
)
SEED: Final = 1337
NANOGPT_REVISION: Final = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
NANOGPT_TREE: Final = "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
NANOGPT_MODEL_BLOB: Final = "c698f8b60129d793494de058b0dd4e318c0dcb5e"
NANOGPT_ORIGINS: Final = frozenset(
    {
        "https://github.com/karpathy/nanoGPT",
        "https://github.com/karpathy/nanoGPT.git",
        "git@github.com:karpathy/nanoGPT.git",
    }
)
MUON_REVISION: Final = "f98f1cacc0263b04290753e32be8d498c1efc806"
MUON_SOURCE_SHA256: Final = "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
P25_SOURCE_SHA256: Final = "6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770"
MODEL_CONFIGURATION: Final = {
    "block_size": 1024,
    "vocab_size": 50_257,
    "n_layer": 12,
    "n_head": 12,
    "n_embd": 768,
    "dropout": 0.0,
    "bias": False,
}
MUON_NAME: Final = re.compile(
    r"^transformer\.h\.[0-9]+\."
    r"(?:attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
_GIT_OBJECT: Final = re.compile(r"^[0-9a-f]{40}$")
_RANGE: Final = re.compile(r"^(?P<start>[0-9a-f]+)-(?P<end>[0-9a-f]+)$")
_PERMISSIONS: Final = re.compile(r"^[r-][w-][x-][ps]$")
_OFFSET: Final = re.compile(r"^[0-9a-f]+$")
_DEVICE: Final = re.compile(r"^(?P<major>[0-9a-f]+):(?P<minor>[0-9a-f]+)$")
_DELETED_SUFFIX: Final = b" (deleted)"
_CLASSIFICATIONS: Final = frozenset(
    {
        "not_file_backed",
        "present_regular_read_only",
        "present_regular_executable",
        "present_regular_writable",
        "present_character_device_read_only",
        "present_character_device_executable",
        "present_character_device_writable",
        "present_other_nonregular_read_only",
        "present_other_nonregular_executable",
        "present_other_nonregular_writable",
        "present_unclassifiable",
        "deleted_regular_read_only",
        "deleted_regular_executable",
        "deleted_regular_writable",
        "deleted_character_device_read_only",
        "deleted_character_device_executable",
        "deleted_character_device_writable",
        "deleted_other_nonregular_read_only",
        "deleted_other_nonregular_executable",
        "deleted_other_nonregular_writable",
        "deleted_unclassifiable",
    }
)
SNAPSHOT_FIELDS: Final = frozenset(
    {
        "stage",
        "time_ns",
        "process_pid",
        "proc_maps_sha256",
        "proc_maps_byte_count",
        "mapping_count",
        "post_probe_proc_maps_sha256",
        "post_probe_proc_maps_byte_count",
        "post_probe_mapping_count",
        "maps_byte_stable_during_probe",
        "all_deleted_mapping_identities_stable_during_probe",
        "deleted_mapping_count",
        "mappings",
    }
)
MAPPING_FIELDS: Final = frozenset(
    {
        "line_number",
        "raw_line_sha256",
        "raw_line_byte_count",
        "address_start_hex",
        "address_end_hex",
        "permissions",
        "offset_hex",
        "device_major_minor",
        "inode_decimal",
        "pathname_present",
        "pathname_bytes_hex",
        "deleted_suffix_present",
        "pathname_without_deleted_bytes_hex",
        "classification",
        "map_files_probe",
        "matching_fd_probes",
    }
)
MAP_FILES_FIELDS: Final = frozenset(
    {
        "attempted",
        "lstat_status",
        "readlink_status",
        "readlink_target_bytes_hex",
        "open_status",
        "fstat_status",
        "device_major_minor",
        "inode_decimal",
        "mode_octal",
        "file_type",
        "size_bytes",
        "errno",
        "error_message",
    }
)
FD_PROBE_FIELDS: Final = frozenset(
    {
        "fd_decimal",
        "readlink_status",
        "readlink_target_bytes_hex",
        "fstat_status",
        "device_major_minor",
        "inode_decimal",
        "mode_octal",
        "file_type",
        "size_bytes",
        "fd_flags_hex",
        "errno",
        "error_message",
    }
)
TRAINING_OPERATION_FIELDS: Final = frozenset(
    {
        "data_loads",
        "forward_calls",
        "backward_calls",
        "optimizer_steps",
        "candidate_evaluations",
        "candidate_observations",
        "gradient_observations",
        "parameter_updates",
    }
)


class P27DiagnosticError(RuntimeError):
    """Raised when the localization record cannot be completed safely."""


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: object) -> str:
    return sha256_bytes(json.dumps(value, sort_keys=True, separators=(",", ":")).encode())


def _ascii(value: bytes, *, line_number: int, label: str) -> str:
    try:
        return value.decode("ascii")
    except UnicodeDecodeError as error:
        raise P27DiagnosticError(
            f"non-ASCII {label} at /proc/self/maps line {line_number}"
        ) from error


def _empty_map_files_probe() -> dict[str, object]:
    return {
        "attempted": False,
        "lstat_status": "not_applicable",
        "readlink_status": "not_applicable",
        "readlink_target_bytes_hex": None,
        "open_status": "not_applicable",
        "fstat_status": "not_applicable",
        "device_major_minor": None,
        "inode_decimal": None,
        "mode_octal": None,
        "file_type": None,
        "size_bytes": None,
        "errno": None,
        "error_message": None,
    }


def _file_type(mode: int) -> str:
    if stat.S_ISREG(mode):
        return "regular"
    if stat.S_ISCHR(mode):
        return "character_device"
    if stat.S_ISBLK(mode):
        return "block_device"
    if stat.S_ISFIFO(mode):
        return "fifo"
    if stat.S_ISSOCK(mode):
        return "socket"
    if stat.S_ISDIR(mode):
        return "directory"
    return "other"


def _metadata_fields(metadata: os.stat_result) -> dict[str, object]:
    return {
        "device_major_minor": (f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x}"),
        "inode_decimal": metadata.st_ino,
        "mode_octal": format(metadata.st_mode, "o"),
        "file_type": _file_type(metadata.st_mode),
        "size_bytes": metadata.st_size,
    }


def _set_first_error(record: dict[str, object], operation: str, error: OSError) -> None:
    if record["errno"] is None:
        record["errno"] = error.errno
        record["error_message"] = f"{operation}: {error}"


def parse_proc_maps(payload: bytes | str) -> list[dict[str, object]]:
    """Parse Linux maps bytes without lossy pathname decoding."""

    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    if not isinstance(payload, bytes) or not payload:
        raise P27DiagnosticError("/proc/self/maps is empty")
    records: list[dict[str, object]] = []
    split_lines = payload.split(b"\n")
    raw_lines = [
        line + (b"\n" if index < len(split_lines) - 1 else b"")
        for index, line in enumerate(split_lines)
        if line or index < len(split_lines) - 1
    ]
    for line_number, raw_line in enumerate(raw_lines, start=1):
        parse_line = raw_line[:-1] if raw_line.endswith(b"\n") else raw_line
        fields = parse_line.split(maxsplit=5)
        if len(fields) < 5:
            raise P27DiagnosticError(f"malformed /proc/self/maps line {line_number}")
        address_range = _ascii(fields[0], line_number=line_number, label="address range")
        permissions = _ascii(fields[1], line_number=line_number, label="permissions")
        offset = _ascii(fields[2], line_number=line_number, label="offset")
        device = _ascii(fields[3], line_number=line_number, label="device")
        inode = _ascii(fields[4], line_number=line_number, label="inode")
        range_match = _RANGE.fullmatch(address_range)
        if (
            range_match is None
            or _PERMISSIONS.fullmatch(permissions) is None
            or _OFFSET.fullmatch(offset) is None
            or _DEVICE.fullmatch(device) is None
            or not inode.isdigit()
        ):
            raise P27DiagnosticError(f"invalid /proc/self/maps fields at line {line_number}")
        start = str(range_match.group("start"))
        end = str(range_match.group("end"))
        if int(start, 16) >= int(end, 16):
            raise P27DiagnosticError(f"nonpositive /proc/self/maps range at line {line_number}")
        pathname = fields[5] if len(fields) == 6 else None
        deleted = pathname is not None and pathname.endswith(_DELETED_SUFFIX)
        pathname_without_deleted = (
            pathname[: -len(_DELETED_SUFFIX)] if deleted and pathname is not None else pathname
        )
        file_backed = pathname is not None and not pathname.startswith(b"[")
        records.append(
            {
                "line_number": line_number,
                "raw_line_sha256": sha256_bytes(raw_line),
                "raw_line_byte_count": len(raw_line),
                "address_start_hex": start,
                "address_end_hex": end,
                "permissions": permissions,
                "offset_hex": offset,
                "device_major_minor": device,
                "inode_decimal": int(inode),
                "pathname_present": pathname is not None,
                "pathname_bytes_hex": pathname.hex() if pathname is not None else None,
                "deleted_suffix_present": deleted,
                "pathname_without_deleted_bytes_hex": (
                    pathname_without_deleted.hex() if pathname_without_deleted is not None else None
                ),
                "classification": (
                    ("deleted_unclassifiable" if deleted else "present_unclassifiable")
                    if file_backed
                    else "not_file_backed"
                ),
                "map_files_probe": _empty_map_files_probe(),
                "matching_fd_probes": [],
            }
        )
    return records


def _probe_map_file(
    record: Mapping[str, object], map_files_root: Path
) -> tuple[dict[str, object], int | None]:
    result = _empty_map_files_probe()
    result["attempted"] = True
    name = f"{record['address_start_hex']}-{record['address_end_hex']}"
    path_bytes = os.fsencode(map_files_root) + b"/" + name.encode("ascii")
    try:
        os.lstat(path_bytes)
        result["lstat_status"] = "ok"
    except OSError as error:
        result["lstat_status"] = "error"
        _set_first_error(result, "lstat", error)
    try:
        result["readlink_target_bytes_hex"] = os.readlink(path_bytes).hex()
        result["readlink_status"] = "ok"
    except OSError as error:
        result["readlink_status"] = "error"
        _set_first_error(result, "readlink", error)
    descriptor: int | None = None
    mode: int | None = None
    try:
        flags = getattr(os, "O_PATH", os.O_RDONLY) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(path_bytes, flags)
        result["open_status"] = "ok"
    except OSError as error:
        result["open_status"] = "error"
        result["fstat_status"] = "not_attempted"
        _set_first_error(result, "open", error)
    if descriptor is not None:
        try:
            metadata = os.fstat(descriptor)
            result.update(_metadata_fields(metadata))
            result["fstat_status"] = "ok"
            mode = metadata.st_mode
        except OSError as error:
            result["fstat_status"] = "error"
            _set_first_error(result, "fstat", error)
        finally:
            os.close(descriptor)
    return result, mode


def _fd_probe(fd: int, fd_root: Path) -> tuple[dict[str, object], int | None, bytes | None]:
    result: dict[str, object] = {
        "fd_decimal": fd,
        "readlink_status": "not_attempted",
        "readlink_target_bytes_hex": None,
        "fstat_status": "not_attempted",
        "device_major_minor": None,
        "inode_decimal": None,
        "mode_octal": None,
        "file_type": None,
        "size_bytes": None,
        "fd_flags_hex": None,
        "errno": None,
        "error_message": None,
    }
    path_bytes = os.fsencode(fd_root) + b"/" + str(fd).encode("ascii")
    target: bytes | None = None
    mode: int | None = None
    try:
        target = os.readlink(path_bytes)
        result["readlink_target_bytes_hex"] = target.hex()
        result["readlink_status"] = "ok"
    except OSError as error:
        result["readlink_status"] = "error"
        _set_first_error(result, "readlink", error)
    try:
        metadata = os.fstat(fd)
        result.update(_metadata_fields(metadata))
        result["fstat_status"] = "ok"
        mode = metadata.st_mode
    except OSError as error:
        result["fstat_status"] = "error"
        _set_first_error(result, "fstat", error)
    try:
        result["fd_flags_hex"] = hex(fcntl.fcntl(fd, fcntl.F_GETFL))
    except OSError as error:
        _set_first_error(result, "fcntl(F_GETFL)", error)
    return result, mode, target


def _fd_inventory(fd_root: Path) -> list[tuple[dict[str, object], int | None, bytes | None]]:
    """Probe one stable directory listing of every currently open descriptor."""

    directory: int | None = None
    try:
        directory = os.open(
            fd_root, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        )
        names = os.listdir(directory)
        if not all(name.isdigit() for name in names):
            raise P27DiagnosticError("/proc/self/fd contains a nonnumeric entry")
        return [_fd_probe(int(name), fd_root) for name in sorted(names, key=int)]
    except OSError as error:
        raise P27DiagnosticError(
            f"cannot enumerate /proc/self/fd: errno={error.errno}: {error}"
        ) from error
    finally:
        if directory is not None:
            os.close(directory)


def _same_mapped_identity(record: Mapping[str, object], probe: Mapping[str, object]) -> bool:
    return (
        probe.get("device_major_minor") == record.get("device_major_minor")
        and probe.get("inode_decimal") == record.get("inode_decimal")
        and int(record.get("inode_decimal", 0)) != 0
    )


def _decode_proc_path_bytes(value: bytes) -> bytes:
    """Decode only the kernel's three-digit octal path escapes, without text decoding."""

    return re.sub(rb"\\([0-7]{3})", lambda match: bytes([int(match.group(1), 8)]), value)


def _matching_fds(
    record: Mapping[str, object],
    inventory: list[tuple[dict[str, object], int | None, bytes | None]],
) -> tuple[list[dict[str, object]], list[int]]:
    pathname_hex = record.get("pathname_bytes_hex")
    pathname = bytes.fromhex(pathname_hex) if isinstance(pathname_hex, str) else None
    without_deleted_hex = record.get("pathname_without_deleted_bytes_hex")
    without_deleted = (
        bytes.fromhex(without_deleted_hex) if isinstance(without_deleted_hex, str) else None
    )
    exact_targets = {
        _decode_proc_path_bytes(value) for value in (pathname, without_deleted) if value is not None
    }
    matches: list[dict[str, object]] = []
    modes: list[int] = []
    for probe, mode, target in inventory:
        if (target is not None and target in exact_targets) or _same_mapped_identity(record, probe):
            matches.append(dict(probe))
            if mode is not None:
                modes.append(mode)
    return matches, modes


def _classification(record: Mapping[str, object], map_mode: int | None, fd_modes: list[int]) -> str:
    pathname_hex = record.get("pathname_bytes_hex")
    pathname = bytes.fromhex(pathname_hex) if isinstance(pathname_hex, str) else None
    if pathname is None or pathname.startswith(b"["):
        return "not_file_backed"
    modes = ([map_mode] if map_mode is not None else []) + fd_modes
    deleted = record.get("deleted_suffix_present") is True
    kinds = {_file_type(mode) for mode in modes}
    if not kinds or len(kinds) != 1:
        return f"{'deleted' if deleted else 'present'}_unclassifiable"
    kind = kinds.pop()
    kind_label = kind if kind in {"regular", "character_device"} else "other_nonregular"
    access = (
        "writable"
        if "w" in str(record["permissions"])
        else ("executable" if "x" in str(record["permissions"]) else "read_only")
    )
    return f"{'deleted' if deleted else 'present'}_{kind_label}_{access}"


def capture_stage(
    stage: str,
    *,
    maps_path: Path = Path("/proc/self/maps"),
    map_files_root: Path = Path("/proc/self/map_files"),
    fd_root: Path = Path("/proc/self/fd"),
) -> dict[str, object]:
    """Capture one byte-exact, stability-checked maps boundary."""

    if stage not in STAGES:
        raise P27DiagnosticError(f"unknown localization stage: {stage}")
    captured_ns = time.time_ns()
    before = maps_path.read_bytes()
    mappings = parse_proc_maps(before)
    fd_inventory = _fd_inventory(fd_root)
    for record in mappings:
        pathname_hex = record["pathname_bytes_hex"]
        pathname = bytes.fromhex(pathname_hex) if isinstance(pathname_hex, str) else None
        if pathname is None or pathname.startswith(b"["):
            continue
        map_probe, map_mode = _probe_map_file(record, map_files_root)
        matching_fds, fd_modes = _matching_fds(record, fd_inventory)
        record["map_files_probe"] = map_probe
        record["matching_fd_probes"] = matching_fds
        record["classification"] = _classification(record, map_mode, fd_modes)
    after = maps_path.read_bytes()
    after_mappings = parse_proc_maps(after)
    initial_deleted = {
        _mapping_identity(record) for record in mappings if record["deleted_suffix_present"] is True
    }
    final_deleted = {
        _mapping_identity(record)
        for record in after_mappings
        if record["deleted_suffix_present"] is True
    }
    deleted_identities_stable = initial_deleted.issubset(final_deleted)
    if any(record["classification"] not in _CLASSIFICATIONS for record in mappings):
        raise P27DiagnosticError("a mapping received an out-of-contract classification")
    return {
        "stage": stage,
        "time_ns": captured_ns,
        "process_pid": os.getpid(),
        "proc_maps_sha256": sha256_bytes(before),
        "proc_maps_byte_count": len(before),
        "mapping_count": len(mappings),
        "post_probe_proc_maps_sha256": sha256_bytes(after),
        "post_probe_proc_maps_byte_count": len(after),
        "post_probe_mapping_count": len(after_mappings),
        "maps_byte_stable_during_probe": after == before,
        "all_deleted_mapping_identities_stable_during_probe": deleted_identities_stable,
        "deleted_mapping_count": sum(
            record["deleted_suffix_present"] is True for record in mappings
        ),
        "mappings": mappings,
    }


def _mapping_identity(record: Mapping[str, object]) -> tuple[object, ...]:
    return tuple(
        record.get(name)
        for name in (
            "address_start_hex",
            "address_end_hex",
            "permissions",
            "offset_hex",
            "device_major_minor",
            "inode_decimal",
            "pathname_bytes_hex",
        )
    )


def _localization_summary(snapshots: list[dict[str, object]]) -> dict[str, object]:
    first: dict[str, object] | None = None
    previous: set[tuple[object, ...]] = set()
    introduced: dict[str, list[dict[str, object]]] = {}
    counts: dict[str, int] = {}
    for snapshot in snapshots:
        stage = str(snapshot["stage"])
        mappings = snapshot["mappings"]
        assert isinstance(mappings, list)
        deleted = [
            (index, record)
            for index, record in enumerate(mappings)
            if isinstance(record, Mapping) and record.get("deleted_suffix_present") is True
        ]
        counts[stage] = len(deleted)
        if first is None and deleted:
            index, record = deleted[0]
            first = {"stage": stage, "mapping_index": index, "mapping": dict(record)}
        current = {_mapping_identity(record) for _, record in deleted}
        introduced[stage] = [
            dict(record) for _, record in deleted if _mapping_identity(record) not in previous
        ]
        previous = current
    return {
        "deleted_mapping_counts_by_stage": counts,
        "first_stage_with_deleted_mapping": None if first is None else first["stage"],
        "first_deleted_mapping": first,
        "new_deleted_mappings_by_stage": introduced,
    }


def _stable_binding(path: Path, label: str) -> dict[str, object]:
    resolved = path.resolve(strict=True)
    before = resolved.stat()
    if not stat.S_ISREG(before.st_mode):
        raise P27DiagnosticError(f"{label} is not a regular file")
    payload = resolved.read_bytes()
    after = resolved.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after or len(payload) != after.st_size:
        raise P27DiagnosticError(f"{label} changed while hashing")
    return {"path": str(resolved), "byte_count": len(payload), "sha256": sha256_bytes(payload)}


def _git(repository: Path, *arguments: str) -> str:
    return subprocess.run(
        ["git", *arguments], cwd=repository, check=True, capture_output=True, text=True
    ).stdout.strip()


def nanogpt_binding(root: Path) -> dict[str, object]:
    root = root.resolve(strict=True)
    status = _git(root, "status", "--porcelain", "--untracked-files=all")
    record = {
        "path": str(root),
        "head": _git(root, "rev-parse", "HEAD"),
        "tree": _git(root, "rev-parse", "HEAD^{tree}"),
        "model_blob": _git(root, "rev-parse", "HEAD:model.py"),
        "origin": _git(root, "remote", "get-url", "origin"),
        "dirty": bool(status),
        "model_source": _stable_binding(root / "model.py", "nanoGPT model.py"),
    }
    if (
        record["head"] != NANOGPT_REVISION
        or record["tree"] != NANOGPT_TREE
        or record["model_blob"] != NANOGPT_MODEL_BLOB
        or record["origin"] not in NANOGPT_ORIGINS
        or record["dirty"] is not False
    ):
        raise P27DiagnosticError("nanoGPT checkout differs from the pinned P22 source")
    return record


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise P27DiagnosticError(f"cannot load pinned module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _construct_model_and_optimizer(
    torch_module: object,
    *,
    nanogpt_root: Path,
    muon_source: Path,
    snapshots: list[dict[str, object]],
) -> tuple[object, object, dict[str, object]]:
    model_module = _load_module("p27_pinned_nanogpt_model", nanogpt_root / "model.py")
    muon_module = _load_module("p27_pinned_kellerjordan_muon", muon_source)
    config = model_module.GPTConfig(**MODEL_CONFIGURATION)
    model = model_module.GPT(config)
    parameter_ids_before_move = {
        name: id(parameter) for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    model.to(device=torch_module.device("cuda"), dtype=torch_module.float32)
    torch_module.cuda.synchronize()
    parameter_ids_after_move = {
        name: id(parameter) for name, parameter in model.named_parameters(remove_duplicate=False)
    }
    if parameter_ids_after_move != parameter_ids_before_move:
        raise P27DiagnosticError("model CUDA move changed Parameter object identity")
    if not all(
        parameter.device.type == "cuda" and parameter.device.index == 0
        for parameter in model.parameters()
    ):
        raise P27DiagnosticError("not every model parameter is on the sole CUDA device")
    snapshots.append(capture_stage("post_model_move"))

    muon_parameters: list[object] = []
    auxiliary_parameters: list[object] = []
    seen: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        if id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (muon_parameters if MUON_NAME.fullmatch(name) else auxiliary_parameters).append(parameter)
    if len(muon_parameters) != 48 or len(auxiliary_parameters) != 27:
        raise P27DiagnosticError("GPT-2-small optimizer partition changed")
    optimizer = muon_module.SingleDeviceMuonWithAuxAdam(
        [
            {
                "params": auxiliary_parameters,
                "lr": 3.0 / 5_000.0,
                "betas": (0.9, 0.95),
                "eps": 1.0e-10,
                "weight_decay": 0.0,
                "use_muon": False,
            },
            {
                "params": muon_parameters,
                "lr": 1.0 / 120.0,
                "momentum": 19.0 / 20.0,
                "weight_decay": 0.0,
                "use_muon": True,
            },
        ]
    )
    optimizer_ids = [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    expected_optimizer_ids = [id(parameter) for parameter in auxiliary_parameters] + [
        id(parameter) for parameter in muon_parameters
    ]
    if optimizer_ids != expected_optimizer_ids or len(set(optimizer_ids)) != len(seen):
        raise P27DiagnosticError("optimizer parameters differ from the exact moved model objects")
    if not all(
        parameter.device.type == "cuda" and parameter.device.index == 0
        for group in optimizer.param_groups
        for parameter in group["params"]
    ):
        raise P27DiagnosticError("not every optimizer parameter is on the sole CUDA device")
    gradients_absent = all(parameter.grad is None for parameter in model.parameters())
    optimizer_state_empty = len(optimizer.state) == 0
    if not gradients_absent or not optimizer_state_empty:
        raise P27DiagnosticError("model or optimizer acquired training state during construction")
    snapshots.append(capture_stage("post_optimizer"))
    return (
        model,
        optimizer,
        {
            "unique_parameter_count": len(seen),
            "muon_parameter_count": len(muon_parameters),
            "auxiliary_parameter_count": len(auxiliary_parameters),
            "all_gradients_absent": gradients_absent,
            "optimizer_state_entry_count": len(optimizer.state),
            "optimizer_state_empty": optimizer_state_empty,
            "parameter_identity_preserved_by_cuda_move": True,
            "optimizer_parameter_identity_exact": True,
            "all_model_and_optimizer_parameters_on_cuda_zero": True,
        },
    )


def _validate_digest(value: str, label: str) -> None:
    if _SHA256.fullmatch(value) is None:
        raise P27DiagnosticError(f"{label} must be a lowercase SHA-256")


def _validate_object(value: str, label: str) -> None:
    if _GIT_OBJECT.fullmatch(value) is None:
        raise P27DiagnosticError(f"{label} must be a full Git object ID")


def run_diagnostic(
    *,
    repository: Path,
    nanogpt_root: Path,
    muon_source: Path,
    p25_source: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    p25_contract_path: Path,
    expected_source_sha256: str,
    expected_p25_source_sha256: str,
    expected_p25_contract_sha256: str,
    expected_runtime_lock_sha256: str,
    expected_host_attestation_sha256: str,
    expected_repository_head: str,
    expected_repository_tree: str,
    state: dict[str, object] | None = None,
) -> dict[str, object]:
    """Execute exactly the four-stage non-training localization protocol."""

    state = {} if state is None else state
    for value, label in (
        (expected_source_sha256, "expected source SHA-256"),
        (expected_p25_source_sha256, "expected P25 source SHA-256"),
        (expected_p25_contract_sha256, "expected P25 contract SHA-256"),
        (expected_runtime_lock_sha256, "expected runtime-lock SHA-256"),
        (expected_host_attestation_sha256, "expected host-attestation SHA-256"),
    ):
        _validate_digest(value, label)
    _validate_object(expected_repository_head, "expected repository HEAD")
    _validate_object(expected_repository_tree, "expected repository tree")
    if any(name == "torch" or name.startswith("torch.") for name in sys.modules):
        raise P27DiagnosticError("PyTorch was loaded before the P27 static preflight")
    started_ns = time.time_ns()
    state.update({"started_time_ns": started_ns, "stage_snapshots": []})
    source_binding = _stable_binding(Path(__file__), "P27 diagnostic source")
    if source_binding["sha256"] != expected_source_sha256:
        raise P27DiagnosticError("P27 diagnostic source differs from the expected SHA-256")
    muon_binding = _stable_binding(muon_source, "pinned Muon source")
    if muon_binding["sha256"] != MUON_SOURCE_SHA256:
        raise P27DiagnosticError("Muon source differs from the pinned P22 source")
    nanogpt_before = nanogpt_binding(nanogpt_root)
    p25_binding = _stable_binding(p25_source, "P25 diagnostic authority")
    if (
        expected_p25_source_sha256 != P25_SOURCE_SHA256
        or p25_binding["sha256"] != P25_SOURCE_SHA256
    ):
        raise P27DiagnosticError("P25 diagnostic authority differs from its pinned SHA-256")
    artifact_bindings = {
        "p25_contract": _stable_binding(p25_contract_path, "P25 contract"),
        "runtime_lock": _stable_binding(runtime_lock_path, "P27 runtime lock"),
        "host_attestation": _stable_binding(host_attestation_path, "P27 host attestation"),
    }
    for name, expected in (
        ("p25_contract", expected_p25_contract_sha256),
        ("runtime_lock", expected_runtime_lock_sha256),
        ("host_attestation", expected_host_attestation_sha256),
    ):
        if artifact_bindings[name]["sha256"] != expected:
            raise P27DiagnosticError(f"{name} differs from its expected SHA-256")
    P25 = _load_module("p27_pinned_p25_diagnostic_authority", p25_source.resolve())
    P25.assert_torch_not_loaded(sys.modules)
    mountinfo = Path("/proc/self/mountinfo").read_bytes()
    mounts = P25.parse_mountinfo(mountinfo.decode("utf-8"))
    preflight = P25.validate_preflight(
        mode="remediated",
        repository=repository,
        runtime_lock_path=runtime_lock_path,
        host_attestation_path=host_attestation_path,
        contract_path=p25_contract_path,
        expected_contract_sha256=expected_p25_contract_sha256,
        expected_runtime_lock_sha256=expected_runtime_lock_sha256,
        expected_host_attestation_sha256=expected_host_attestation_sha256,
        expected_repository_head=expected_repository_head,
        expected_repository_tree=expected_repository_tree,
        mountinfo=mountinfo,
        mounts=mounts,
    )
    state["bindings"] = {
        "source": source_binding,
        "nanogpt": nanogpt_before,
        "muon": {"revision": MUON_REVISION, **muon_binding},
        "p25_source": p25_binding,
        "runtime_artifacts": artifact_bindings,
        "p25_preflight": preflight,
    }

    import torch

    runtime_lock = preflight["runtime_lock"]
    if not isinstance(runtime_lock, Mapping) or not isinstance(
        runtime_lock.get("determinism"), Mapping
    ):
        raise P27DiagnosticError("validated runtime lock has no determinism mapping")
    configured = P25.configure_torch_determinism(torch, runtime_lock["determinism"])
    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_initialized():
        raise P27DiagnosticError("CUDA initialized before the pre-CUDA localization boundary")
    snapshots = state["stage_snapshots"]
    assert isinstance(snapshots, list)
    snapshots.append(capture_stage("pre_cuda"))

    torch.cuda.init()
    torch.cuda.synchronize()
    properties = torch.cuda.get_device_properties(0)
    snapshots.append(capture_stage("post_cuda_init"))
    model, optimizer, construction = _construct_model_and_optimizer(
        torch,
        nanogpt_root=nanogpt_root.resolve(),
        muon_source=muon_source.resolve(),
        snapshots=snapshots,
    )

    source_after = _stable_binding(Path(__file__), "P27 diagnostic source")
    p25_after = _stable_binding(p25_source, "P25 diagnostic authority")
    muon_after = _stable_binding(muon_source, "pinned Muon source")
    nanogpt_after = nanogpt_binding(nanogpt_root)
    artifact_bindings_after = {
        "p25_contract": _stable_binding(p25_contract_path, "P25 contract"),
        "runtime_lock": _stable_binding(runtime_lock_path, "P27 runtime lock"),
        "host_attestation": _stable_binding(host_attestation_path, "P27 host attestation"),
    }
    repository_after = P25.git_provenance(repository)
    current_mountinfo = Path("/proc/self/mountinfo").read_bytes()
    gpu = runtime_lock.get("gpu")
    software = runtime_lock.get("software")
    if not isinstance(gpu, Mapping) or not isinstance(software, Mapping):
        raise P27DiagnosticError("validated runtime lock omits GPU/software mappings")
    live_uuid = f"GPU-{properties.uuid}"
    snapshot_schema_exact = all(
        isinstance(entry, Mapping)
        and set(entry) == SNAPSHOT_FIELDS
        and isinstance(entry.get("mappings"), list)
        and entry.get("mapping_count") == len(entry["mappings"])
        and entry.get("deleted_mapping_count")
        == sum(
            record.get("deleted_suffix_present") is True
            for record in entry["mappings"]
            if isinstance(record, Mapping)
        )
        for entry in snapshots
    )
    mapping_schema_exact = all(
        isinstance(record, Mapping)
        and set(record) == MAPPING_FIELDS
        and record.get("classification") in _CLASSIFICATIONS
        and isinstance(record.get("map_files_probe"), Mapping)
        and set(record["map_files_probe"]) == MAP_FILES_FIELDS
        and isinstance(record.get("matching_fd_probes"), list)
        and all(
            isinstance(probe, Mapping) and set(probe) == FD_PROBE_FIELDS
            for probe in record["matching_fd_probes"]
        )
        for entry in snapshots
        for record in entry.get("mappings", [])
        if isinstance(entry, Mapping)
    )
    checks = {
        "all_four_stages_captured_in_order": [entry.get("stage") for entry in snapshots]
        == list(STAGES),
        "snapshot_schema_exact": snapshot_schema_exact,
        "deleted_mapping_identities_stable_during_each_probe": all(
            entry.get("all_deleted_mapping_identities_stable_during_probe") is True
            for entry in snapshots
        ),
        "mapping_and_probe_schema_exact": mapping_schema_exact,
        "every_deleted_mapping_has_explicit_probe_outcome": all(
            record.get("deleted_suffix_present") is not True
            or (
                isinstance(record.get("map_files_probe"), Mapping)
                and record["map_files_probe"].get("attempted") is True
                and str(record.get("classification")).startswith("deleted_")
            )
            for entry in snapshots
            for record in entry.get("mappings", [])
            if isinstance(record, Mapping)
        ),
        "source_unchanged": source_after == source_binding,
        "p25_source_unchanged": p25_after == p25_binding,
        "muon_source_unchanged": muon_after == muon_binding,
        "nanogpt_source_unchanged": nanogpt_after == nanogpt_before,
        "runtime_artifacts_unchanged": artifact_bindings_after == artifact_bindings,
        "repository_unchanged": repository_after == preflight["repository"],
        "mount_namespace_unchanged": current_mountinfo == mountinfo,
        "single_locked_gpu": torch.cuda.device_count() == 1
        and live_uuid == gpu.get("uuid")
        and properties.name == gpu.get("name"),
        "torch_runtime_matches_lock": torch.__version__ == software.get("torch_version")
        and torch.version.git_version == software.get("torch_git_version")
        and torch.version.cuda == software.get("torch_cuda_compiled_version")
        and P25.cuda_runtime_version() == software.get("cuda_runtime_version")
        and torch.backends.cudnn.version() == software.get("cudnn_version"),
        "determinism_preserved": P25.torch_determinism_snapshot(torch)
        == {**P25.EXPECTED_TORCH_DETERMINISM, "cuda_initialized": True},
        "no_forward_backward_or_step": construction["all_gradients_absent"] is True
        and construction["optimizer_state_empty"] is True
        and construction["parameter_identity_preserved_by_cuda_move"] is True
        and construction["optimizer_parameter_identity_exact"] is True
        and construction["all_model_and_optimizer_parameters_on_cuda_zero"] is True,
    }
    completed_ns = time.time_ns()
    del optimizer, model
    payload = {
        "schema_version": SCHEMA,
        "status": "complete" if all(checks.values()) else "blocked",
        "passes": all(checks.values()),
        "scope": "four-stage CUDA deleted-mapping localization; no training computation",
        "seed": SEED,
        "process": {
            "pid": os.getpid(),
            "started_time_ns": started_ns,
            "completed_time_ns": completed_ns,
            "argv": list(sys.argv),
        },
        "bindings": state["bindings"],
        "live_runtime": {
            "torch_version": torch.__version__,
            "torch_git_version": torch.version.git_version,
            "torch_cuda_compiled_version": torch.version.cuda,
            "cuda_runtime_version": P25.cuda_runtime_version(),
            "cudnn_version": torch.backends.cudnn.version(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_device_name": properties.name,
            "cuda_device_uuid": live_uuid,
            "configured_determinism": configured,
            "mountinfo_sha256": sha256_bytes(current_mountinfo),
        },
        "stage_order": list(STAGES),
        "stage_snapshots": snapshots,
        "construction": construction,
        "localization": _localization_summary(snapshots),
        "training_operations": {
            "data_loads": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "candidate_evaluations": 0,
            "candidate_observations": 0,
            "gradient_observations": 0,
            "parameter_updates": 0,
        },
        "policy": {
            "classification_performed": False,
            "allowlist_changed": False,
            "statement": (
                "Mappings receive only the frozen factual file/access classification. No "
                "mapping is classified as safe, allowlisted, or authorized by this artifact."
            ),
        },
        "checks": checks,
        "claim_boundary": (
            "This is a non-training localization diagnostic, not repeatability, "
            "noninterference, gradient, fidelity, or training evidence."
        ),
    }
    validate_payload(payload)
    return payload


def validate_payload(payload: Mapping[str, object]) -> None:
    expected = {
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
    if set(payload) != expected or payload.get("schema_version") != SCHEMA:
        raise P27DiagnosticError("P27 payload schema changed")
    stages = payload.get("stage_snapshots")
    if not isinstance(stages, list) or [entry.get("stage") for entry in stages] != list(STAGES):
        raise P27DiagnosticError("P27 payload does not contain the exact four stages")
    for entry in stages:
        if not isinstance(entry, Mapping) or set(entry) != SNAPSHOT_FIELDS:
            raise P27DiagnosticError("P27 stage record is malformed")
        mappings = entry.get("mappings")
        if not isinstance(mappings, list) or entry.get("mapping_count") != len(mappings):
            raise P27DiagnosticError("P27 mapping count is inconsistent")
        deleted_count = 0
        for record in mappings:
            if not isinstance(record, Mapping) or set(record) != MAPPING_FIELDS:
                raise P27DiagnosticError("P27 mapping record is malformed")
            if record.get("classification") not in _CLASSIFICATIONS:
                raise P27DiagnosticError("P27 mapping classification is outside the contract")
            if record.get("deleted_suffix_present") is True:
                deleted_count += 1
            map_probe = record.get("map_files_probe")
            fd_probes = record.get("matching_fd_probes")
            if not isinstance(map_probe, Mapping) or set(map_probe) != MAP_FILES_FIELDS:
                raise P27DiagnosticError("P27 map_files probe is malformed")
            if not isinstance(fd_probes, list) or any(
                not isinstance(probe, Mapping) or set(probe) != FD_PROBE_FIELDS
                for probe in fd_probes
            ):
                raise P27DiagnosticError("P27 matching-FD probe is malformed")
        if entry.get("deleted_mapping_count") != deleted_count:
            raise P27DiagnosticError("P27 deleted-mapping count is inconsistent")
    localization = payload.get("localization")
    if not isinstance(localization, Mapping) or dict(localization) != _localization_summary(stages):
        raise P27DiagnosticError("P27 localization summary does not reconstruct")
    operations = payload.get("training_operations")
    if (
        not isinstance(operations, Mapping)
        or set(operations) != TRAINING_OPERATION_FIELDS
        or any(type(value) is not int or value != 0 for value in operations.values())
    ):
        raise P27DiagnosticError("P27 payload claims a training operation")
    policy = payload.get("policy")
    if (
        not isinstance(policy, Mapping)
        or policy.get("classification_performed") is not False
        or policy.get("allowlist_changed") is not False
    ):
        raise P27DiagnosticError("P27 payload performs a forbidden policy classification")
    checks = payload.get("checks")
    if (
        not isinstance(checks, Mapping)
        or not checks
        or not all(isinstance(value, bool) for value in checks.values())
    ):
        raise P27DiagnosticError("P27 checks are malformed")
    passes = all(checks.values())
    if payload.get("passes") is not passes or payload.get("status") != (
        "complete" if passes else "blocked"
    ):
        raise P27DiagnosticError("P27 status differs from its checks")


def _atomic_write_new(path: Path, payload: object) -> None:
    parent = path.parent.resolve(strict=True)
    target = parent / path.name
    if path.is_symlink() or target.exists():
        raise P27DiagnosticError("P27 output already exists or is a symlink")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary_name, target, follow_symlinks=False)
        except FileExistsError as error:
            raise P27DiagnosticError("P27 output already exists") from error
        directory = os.open(parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary_name)


def _failure_payload(error: BaseException, state: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_version": SCHEMA,
        "status": "blocked",
        "passes": False,
        "scope": "four-stage CUDA deleted-mapping localization; no training computation",
        "seed": SEED,
        "process": {
            "pid": os.getpid(),
            "started_time_ns": state.get("started_time_ns"),
            "failed_time_ns": time.time_ns(),
            "argv": list(sys.argv),
        },
        "bindings": state.get("bindings"),
        "stage_order": list(STAGES),
        "stage_snapshots": state.get("stage_snapshots", []),
        "training_operations": {
            "data_loads": 0,
            "forward_calls": 0,
            "backward_calls": 0,
            "optimizer_steps": 0,
            "candidate_evaluations": 0,
            "candidate_observations": 0,
            "gradient_observations": 0,
            "parameter_updates": 0,
        },
        "policy": {
            "classification_performed": False,
            "allowlist_changed": False,
            "statement": "Failure evidence only; no mapping policy decision was made.",
        },
        "error": {
            "class": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
            "traceback": traceback.format_exc(),
        },
        "claim_boundary": (
            "This blocked non-training diagnostic is not repeatability, noninterference, "
            "gradient, fidelity, or training evidence."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--nanogpt-root", type=Path, required=True)
    parser.add_argument("--muon-source", type=Path, required=True)
    parser.add_argument("--p25-source", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--host-attestation", type=Path, required=True)
    parser.add_argument("--p25-contract", type=Path, required=True)
    parser.add_argument("--expected-source-sha256", required=True)
    parser.add_argument("--expected-p25-source-sha256", required=True)
    parser.add_argument("--expected-p25-contract-sha256", required=True)
    parser.add_argument("--expected-runtime-lock-sha256", required=True)
    parser.add_argument("--expected-host-attestation-sha256", required=True)
    parser.add_argument("--expected-repository-head", required=True)
    parser.add_argument("--expected-repository-tree", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    repository = args.repository.resolve()
    output = args.output
    try:
        parent = output.parent.resolve(strict=True)
    except OSError as error:
        print(f"P27 diagnostic blocked before output validation: {error}", file=sys.stderr)
        return 2
    target = parent / output.name
    if target == repository or repository in target.parents:
        print("P27 diagnostic blocked: output must remain outside the repository", file=sys.stderr)
        return 2
    if output.is_symlink() or target.exists():
        print("P27 diagnostic blocked: output already exists or is a symlink", file=sys.stderr)
        return 2
    state: dict[str, object] = {}
    try:
        payload = run_diagnostic(
            repository=repository,
            nanogpt_root=args.nanogpt_root.resolve(),
            muon_source=args.muon_source.resolve(),
            p25_source=args.p25_source.resolve(),
            runtime_lock_path=args.runtime_lock.resolve(),
            host_attestation_path=args.host_attestation.resolve(),
            p25_contract_path=args.p25_contract.resolve(),
            expected_source_sha256=args.expected_source_sha256,
            expected_p25_source_sha256=args.expected_p25_source_sha256,
            expected_p25_contract_sha256=args.expected_p25_contract_sha256,
            expected_runtime_lock_sha256=args.expected_runtime_lock_sha256,
            expected_host_attestation_sha256=args.expected_host_attestation_sha256,
            expected_repository_head=args.expected_repository_head,
            expected_repository_tree=args.expected_repository_tree,
            state=state,
        )
    except BaseException as error:
        payload = _failure_payload(error, state)
    try:
        _atomic_write_new(target, payload)
    except BaseException as error:
        print(f"P27 diagnostic could not retain native evidence: {error}", file=sys.stderr)
        return 2
    rendered = target.read_bytes()
    print(
        json.dumps(
            {
                "output": str(target),
                "output_sha256": sha256_bytes(rendered),
                "output_byte_count": len(rendered),
                "passes": payload.get("passes", False),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload.get("passes") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
