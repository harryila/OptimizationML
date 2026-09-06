from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import stat
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "experiments/training/run_p27_cuda_deleted_mapping_localization.py"


def _module():
    spec = importlib.util.spec_from_file_location("p27_localization_under_test", SOURCE)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _maps_line(
    path: bytes | None = None,
    *,
    metadata: os.stat_result | None = None,
    address: bytes = b"1000-2000",
    permissions: bytes = b"r--s",
) -> bytes:
    device = b"00:00"
    inode = 0
    if metadata is not None:
        device = f"{os.major(metadata.st_dev):02x}:{os.minor(metadata.st_dev):02x}".encode()
        inode = metadata.st_ino
    suffix = b" " + path if path is not None else b""
    return (
        address
        + b" "
        + permissions
        + b" 00000000 "
        + device
        + f" {inode}".encode()
        + suffix
        + b"\n"
    )


def _empty_mapping(module, *, deleted: bool = False) -> dict[str, object]:
    path = b"/dev/zero (deleted)" if deleted else None
    return {
        "line_number": 1,
        "raw_line_sha256": "0" * 64,
        "raw_line_byte_count": 1,
        "address_start_hex": "1000",
        "address_end_hex": "2000",
        "permissions": "r--s",
        "offset_hex": "00000000",
        "device_major_minor": "00:00",
        "inode_decimal": 0,
        "pathname_present": path is not None,
        "pathname_bytes_hex": None if path is None else path.hex(),
        "deleted_suffix_present": deleted,
        "pathname_without_deleted_bytes_hex": None if path is None else b"/dev/zero".hex(),
        "classification": "deleted_unclassifiable" if deleted else "not_file_backed",
        "map_files_probe": module._empty_map_files_probe(),
        "matching_fd_probes": [],
    }


def test_module_has_no_top_level_torch_or_p25_import() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    top_imports = [node for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))]
    names = {alias.name.split(".")[0] for node in top_imports for alias in node.names}
    assert "torch" not in names
    assert "run_p25_executable_origin_diagnostic" not in names
    assert SOURCE.read_text(encoding="utf-8").count("    import torch\n") == 1


def test_parse_proc_maps_preserves_line_and_path_bytes_losslessly() -> None:
    module = _module()
    raw = b"1000-2000 rw-s 00001000 00:01 77 /dev/shm/a\\040b\xff (deleted)\n"
    record = module.parse_proc_maps(raw)[0]
    assert set(record) == module.MAPPING_FIELDS
    assert record["raw_line_sha256"] == hashlib.sha256(raw).hexdigest()
    assert record["raw_line_byte_count"] == len(raw)
    assert record["pathname_bytes_hex"] == b"/dev/shm/a\\040b\xff (deleted)".hex()
    assert record["pathname_without_deleted_bytes_hex"] == b"/dev/shm/a\\040b\xff".hex()
    assert record["deleted_suffix_present"] is True
    assert record["classification"] == "deleted_unclassifiable"


def test_parse_proc_maps_does_not_split_on_non_newline_path_bytes() -> None:
    module = _module()
    record = module.parse_proc_maps(b"1000-2000 r--p 0 00:01 7 /tmp/a\vb\n")[0]
    assert record["pathname_bytes_hex"] == b"/tmp/a\vb".hex()


@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"bad\n",
        b"2000-1000 rw-p 0 00:00 0\n",
        b"1000-2000 rwqp 0 00:00 0\n",
        b"1000-2000 rw-p nothex 00:00 0\n",
        b"1000-2000 rw-p 0 bad 0\n",
        b"1000-2000 rw-p 0 00:00 nope\n",
    ],
)
def test_parse_proc_maps_fails_closed_on_malformed_input(payload: bytes) -> None:
    module = _module()
    with pytest.raises(module.P27DiagnosticError):
        module.parse_proc_maps(payload)


def test_map_files_probe_retains_exact_target_and_regular_metadata(tmp_path: Path) -> None:
    module = _module()
    backing = tmp_path / "backing.bin"
    backing.write_bytes(b"mapped bytes")
    metadata = backing.stat()
    record = module.parse_proc_maps(_maps_line(f"{backing} (deleted)".encode(), metadata=metadata))[
        0
    ]
    map_files = tmp_path / "map_files"
    map_files.mkdir()
    (map_files / "1000-2000").symlink_to(backing)
    probe, mode = module._probe_map_file(record, map_files)
    assert set(probe) == module.MAP_FILES_FIELDS
    assert probe["readlink_status"] == "ok"
    assert bytes.fromhex(probe["readlink_target_bytes_hex"]) == os.fsencode(backing)
    assert probe["open_status"] == probe["fstat_status"] == "ok"
    assert probe["file_type"] == "regular"
    assert stat.S_ISREG(mode)


def test_map_files_failure_is_structured_and_unclassifiable(tmp_path: Path) -> None:
    module = _module()
    record = module.parse_proc_maps(b"1000-2000 r--s 00000000 00:01 9 /dev/zero (deleted)\n")[0]
    probe, mode = module._probe_map_file(record, tmp_path / "absent")
    assert probe["lstat_status"] == "error"
    assert probe["readlink_status"] == "error"
    assert probe["open_status"] == "error"
    assert isinstance(probe["errno"], int)
    assert "lstat" in probe["error_message"]
    assert mode is None
    assert module._classification(record, mode, []) == "deleted_unclassifiable"


def test_fd_probe_and_matching_preserve_flags_and_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    backing = tmp_path / "fd.bin"
    backing.write_bytes(b"x")
    descriptor = os.open(backing, os.O_RDONLY)
    try:
        monkeypatch.setattr(module.os, "readlink", lambda _path: os.fsencode(backing))
        probe, mode, target = module._fd_probe(descriptor, Path("/proc/self/fd"))
        metadata = os.fstat(descriptor)
        record = module.parse_proc_maps(_maps_line(os.fsencode(backing), metadata=metadata))[0]
        matches, modes = module._matching_fds(record, [(probe, mode, target)])
    finally:
        os.close(descriptor)
    assert set(probe) == module.FD_PROBE_FIELDS
    assert probe["readlink_status"] == probe["fstat_status"] == "ok"
    assert probe["fd_flags_hex"].startswith("0x")
    assert matches == [probe]
    assert modes == [mode]


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (stat.S_IFREG | 0o400, "deleted_regular_read_only"),
        (stat.S_IFCHR | 0o600, "deleted_character_device_read_only"),
        (stat.S_IFBLK | 0o600, "deleted_other_nonregular_read_only"),
    ],
)
def test_deleted_classification_distinguishes_conclusive_file_type(
    mode: int, expected: str
) -> None:
    module = _module()
    record = module.parse_proc_maps(b"1000-2000 r--s 00000000 00:01 9 /dev/zero (deleted)\n")[0]
    assert module._classification(record, mode, []) == expected


def test_capture_stage_retains_all_mappings_and_unrelated_churn(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _module()
    maps = tmp_path / "maps"
    deleted = _maps_line(b"/dev/zero (deleted)")
    maps.write_bytes(deleted)
    map_files = tmp_path / "map_files"
    fd_root = tmp_path / "fd"
    map_files.mkdir()
    fd_root.mkdir()

    def inventory(_root):
        maps.write_bytes(deleted + _maps_line(b"[heap]", address=b"3000-4000"))
        return []

    monkeypatch.setattr(module, "_fd_inventory", inventory)
    snapshot = module.capture_stage(
        "pre_cuda", maps_path=maps, map_files_root=map_files, fd_root=fd_root
    )
    assert set(snapshot) == module.SNAPSHOT_FIELDS
    assert snapshot["mapping_count"] == snapshot["deleted_mapping_count"] == 1
    assert snapshot["post_probe_mapping_count"] == 2
    assert snapshot["maps_byte_stable_during_probe"] is False
    assert snapshot["all_deleted_mapping_identities_stable_during_probe"] is True
    assert snapshot["mappings"][0]["deleted_suffix_present"] is True


def _valid_payload(module) -> dict[str, object]:
    stages = []
    for stage in module.STAGES:
        mapping = _empty_mapping(module)
        stages.append(
            {
                "stage": stage,
                "time_ns": 1,
                "process_pid": 2,
                "proc_maps_sha256": "0" * 64,
                "proc_maps_byte_count": 1,
                "mapping_count": 1,
                "post_probe_proc_maps_sha256": "0" * 64,
                "post_probe_proc_maps_byte_count": 1,
                "post_probe_mapping_count": 1,
                "maps_byte_stable_during_probe": True,
                "all_deleted_mapping_identities_stable_during_probe": True,
                "deleted_mapping_count": 0,
                "mappings": [mapping],
            }
        )
    checks = {"complete": True}
    return {
        "schema_version": module.SCHEMA,
        "status": "complete",
        "passes": True,
        "scope": "test",
        "seed": module.SEED,
        "process": {},
        "bindings": {},
        "live_runtime": {},
        "stage_order": list(module.STAGES),
        "stage_snapshots": stages,
        "construction": {},
        "localization": module._localization_summary(stages),
        "training_operations": {name: 0 for name in module.TRAINING_OPERATION_FIELDS},
        "policy": {"classification_performed": False, "allowlist_changed": False},
        "checks": checks,
        "claim_boundary": "test",
    }


def test_payload_validation_forbids_training_and_policy_authorization() -> None:
    module = _module()
    payload = _valid_payload(module)
    module.validate_payload(payload)
    payload["training_operations"]["optimizer_steps"] = 1
    with pytest.raises(module.P27DiagnosticError, match="training operation"):
        module.validate_payload(payload)
    payload = _valid_payload(module)
    payload["policy"]["classification_performed"] = True
    with pytest.raises(module.P27DiagnosticError, match="policy classification"):
        module.validate_payload(payload)


def test_payload_validation_reconstructs_localization_summary() -> None:
    module = _module()
    payload = _valid_payload(module)
    payload["localization"]["first_stage_with_deleted_mapping"] = "post_cuda_init"
    with pytest.raises(module.P27DiagnosticError, match="summary"):
        module.validate_payload(payload)


def test_atomic_write_is_new_only(tmp_path: Path) -> None:
    module = _module()
    output = tmp_path / "native.json"
    payload = {"b": 2, "a": 1}
    module._atomic_write_new(output, payload)
    assert json.loads(output.read_bytes()) == payload
    with pytest.raises(module.P27DiagnosticError, match="already exists"):
        module._atomic_write_new(output, payload)


def test_source_contains_no_training_invocation() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    forbidden_attributes = {"backward", "step", "forward"}
    assert not any(
        isinstance(call.func, ast.Attribute) and call.func.attr in forbidden_attributes
        for call in calls
    )


def test_source_hash_and_p25_authority_are_explicit() -> None:
    source = SOURCE.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    text = source.decode("utf-8")
    assert digest not in text
    assert "--expected-source-sha256" in text
    assert "--p25-source" in text
    assert "--expected-p25-source-sha256" in text
    assert "6305fb9683503eb67e091cdfb0a1105628fcb4bec76fe5ef7fd23dd49ba9d770" in text
