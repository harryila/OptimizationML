#!/usr/bin/env python3
"""Fail-closed scaffolding for the frozen P22 real-gradient shadow trace.

This file does not synthesize a trace. It provides three reviewable pieces
used by the separate isolated runner around the pinned NanoGPT model:

* prerequisite and provenance validation before accelerator work;
* deterministic state/RNG hashing and an RNG-neutral observer guard; and
* exact off-A/off-B repeatability and off-A/trace-on noninterference checks.

The P22 evidence claim intentionally remains blocked until the hashed FineWeb
materialization, BF16 CUDA/MPS device, instrumentation, and complete three-run
comparison are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, TypeVar

import numpy as np
import torch
from torch import Tensor

from passive_muon.p21_shadow_trace import CAPTURE_STEPS
from passive_muon.p22_scalable_sector_shield_extension import (
    P22_CERTIFIED_SHIELD_SHAPES,
    p22_canonical_shield_orientation,
)

P22_PROTOCOL_SCHEMA: Final = "passive-muon-p22-real-gradient-shadow-trace-v1"
P22_DATA_SCHEMA: Final = "passive-muon-p22-fineweb-materialization-v1"
P22_LEGACY_RUN_MANIFEST_SCHEMA: Final = "passive-muon-p22-run-manifest-v1"
P22_RUN_MANIFEST_SCHEMA: Final = "passive-muon-p22-run-manifest-v2"
P22_COMPARISON_SCHEMA: Final = "passive-muon-p22-noninterference-v1"

NANOGPT_REVISION: Final = "3adf61e154c3fe3fca428ad6bc3818b27a3b8291"
NANOGPT_TREE: Final = "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a"
NANOGPT_TRAIN_BLOB: Final = "de5785051050d64bbda37a4964c2c079f0739b2b"
NANOGPT_MODEL_BLOB: Final = "c698f8b60129d793494de058b0dd4e318c0dcb5e"
MUON_REVISION: Final = "f98f1cacc0263b04290753e32be8d498c1efc806"
MUON_SOURCE_SHA256: Final = "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"

EXPECTED_OPTIMIZER_STEPS: Final = 256
EXPECTED_CAPTURE_STEPS: Final = tuple(CAPTURE_STEPS)
EXPECTED_CANDIDATE_OBSERVATIONS: Final = 48 * len(EXPECTED_CAPTURE_STEPS)
EXPECTED_MODEL_PARAMETER_COUNT: Final = 75
EXPECTED_MUON_PARAMETER_COUNT: Final = 48
EXPECTED_AUXILIARY_PARAMETER_COUNT: Final = 27
EXPECTED_TIED_PARAMETER_ALIASES: Final = [
    ["lm_head.weight", "transformer.wte.weight"],
]
EXPECTED_TRAINING_TOKENS_CONSUMED: Final = 32_768
EXPECTED_SEQUENCE_LENGTH: Final = EXPECTED_TRAINING_TOKENS_CONSUMED // EXPECTED_OPTIMIZER_STEPS
EXPECTED_TRAIN_POOL_TOKENS: Final = 131_072
EXPECTED_VALIDATION_TOKENS: Final = 32_768
EXPECTED_TRAIN_BYTES: Final = 2 * EXPECTED_TRAIN_POOL_TOKENS
EXPECTED_VALIDATION_BYTES: Final = 2 * EXPECTED_VALIDATION_TOKENS
READ_BLOCK_BYTES: Final = 2**20

PROTOCOL_PATH: Final = Path("experiments/training/p22_real_gradient_shadow_trace_protocol.json")
DATA_TEMPLATE_PATH: Final = Path("experiments/training/p22_fineweb_manifest.template.json")

NONINTERFERENCE_STEP_FIELDS: Final = (
    "tokens_seen",
    "data_token_offset",
    "batch_sha256",
    "loss_tensor_sha256",
    "rng_before_step_sha256",
    "rng_after_step_sha256",
)
NONINTERFERENCE_STATE_FIELDS: Final = (
    "model_state_sha256",
    "optimizer_state_sha256",
)
STATE_CHECKPOINT_STEPS: Final = frozenset((*EXPECTED_CAPTURE_STEPS, EXPECTED_OPTIMIZER_STEPS - 1))
NONINTERFERENCE_FINAL_FIELDS: Final = (
    "model_state_sha256",
    "optimizer_state_sha256",
    "rng_state_sha256",
)

T = TypeVar("T")


class P22ProtocolError(ValueError):
    """Raised when a purported P22 artifact violates the frozen protocol."""


def repository_root() -> Path:
    """Return this OptimizationML checkout's root."""

    return Path(__file__).resolve().parents[2]


def protocol_path() -> Path:
    """Return the frozen P22 protocol path."""

    return repository_root() / PROTOCOL_PATH


def data_template_path() -> Path:
    """Return the unmaterialized FineWeb manifest template path."""

    return repository_root() / DATA_TEMPLATE_PATH


def sha256_file(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(READ_BLOCK_BYTES):
            digest.update(block)
    return digest.hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def load_protocol() -> dict[str, Any]:
    """Load and internally check the frozen P22 protocol."""

    payload = json.loads(protocol_path().read_text(encoding="utf-8"))
    if payload.get("schema_version") != P22_PROTOCOL_SCHEMA:
        raise P22ProtocolError("P22 protocol schema mismatch")
    run = payload.get("run")
    if not isinstance(run, dict):
        raise P22ProtocolError("P22 protocol has no run mapping")
    if run.get("total_optimizer_steps") != EXPECTED_OPTIMIZER_STEPS:
        raise P22ProtocolError("P22 optimizer-step count changed")
    if run.get("capture_steps") != list(EXPECTED_CAPTURE_STEPS):
        raise P22ProtocolError("P22 capture schedule changed")
    if run.get("seed_count") != 1 or run.get("device_count") != 1:
        raise P22ProtocolError("P22 must remain one-seed and one-device")
    if (
        run.get("micro_batch_size"),
        run.get("sequence_length"),
        run.get("gradient_accumulation_steps"),
        run.get("total_training_tokens"),
    ) != (1, 128, 1, EXPECTED_TRAINING_TOKENS_CONSUMED):
        raise P22ProtocolError("P22 M4-feasible batch/token profile changed")
    if payload.get("optimizer", {}).get("muon_learning_rate") != "1/120":
        raise P22ProtocolError("P22 primary Muon learning rate changed")
    if payload.get("optimizer", {}).get("muon_weight_decay") != 0:
        raise P22ProtocolError("P22 Muon weight decay must remain zero")
    return payload


def protocol_sha256() -> str:
    """Hash the exact frozen protocol bytes."""

    return sha256_file(protocol_path())


def expected_gpt2_small_muon_inventory() -> list[dict[str, Any]]:
    """Return the predeclared 48-name hidden-matrix inventory.

    PyTorch Linear stores ``(out_features, in_features)``.  P20 accepts either
    the stored shape or its transpose, so both are reported explicitly.
    """

    roles = (
        ("attn.c_attn.weight", (2_304, 768)),
        ("attn.c_proj.weight", (768, 768)),
        ("mlp.c_fc.weight", (3_072, 768)),
        ("mlp.c_proj.weight", (768, 3_072)),
    )
    inventory: list[dict[str, Any]] = []
    for layer in range(12):
        for suffix, shape in roles:
            transpose = (shape[1], shape[0])
            try:
                shield_shape, transposed_for_shield = p22_canonical_shield_orientation(shape)
                covered = shield_shape in P22_CERTIFIED_SHIELD_SHAPES
            except ValueError:
                shield_shape = None
                transposed_for_shield = None
                covered = False
            inventory.append(
                {
                    "name": f"transformer.h.{layer}.{suffix}",
                    "stored_shape": list(shape),
                    "transpose_shape": list(transpose),
                    "shield_shape": list(shield_shape) if shield_shape is not None else None,
                    "transposed_for_shield": transposed_for_shield,
                    "numel": shape[0] * shape[1],
                    "p20_covered": covered,
                }
            )
    return inventory


def inventory_coverage() -> dict[str, Any]:
    """Summarize exact P20 coverage of the frozen GPT-2-small inventory."""

    inventory = expected_gpt2_small_muon_inventory()
    covered = [item for item in inventory if item["p20_covered"]]
    total_numel = sum(int(item["numel"]) for item in inventory)
    covered_numel = sum(int(item["numel"]) for item in covered)
    return {
        "intended_parameter_count": len(inventory),
        "covered_parameter_count": len(covered),
        "parameter_count_fraction": f"{len(covered)}/{len(inventory)}",
        "intended_numel": total_numel,
        "covered_numel": covered_numel,
        "numel_fraction": f"{covered_numel}/{total_numel}",
        "complete": len(covered) == len(inventory),
        "unsupported": [item for item in inventory if not item["p20_covered"]],
    }


def _run_git(checkout: Path, *arguments: str) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["git", "-C", str(checkout), *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return False, str(error)
    if completed.returncode != 0:
        detail = completed.stderr.strip() or completed.stdout.strip()
        return False, detail
    return True, completed.stdout.strip()


def validate_nanogpt_checkout(checkout: Path | None) -> dict[str, Any]:
    """Check the NanoGPT revision, tree, tracked blobs, origin, and cleanliness."""

    blockers: list[str] = []
    details: dict[str, Any] = {}
    if checkout is None:
        return {"ready": False, "blockers": ["NanoGPT checkout was not supplied"], "details": {}}
    checkout = checkout.resolve()
    if not checkout.is_dir():
        return {
            "ready": False,
            "blockers": [f"NanoGPT checkout is not a directory: {checkout}"],
            "details": {},
        }

    queries = {
        "revision": ("rev-parse", "HEAD"),
        "tree": ("rev-parse", "HEAD^{tree}"),
        "train_py_blob": ("rev-parse", "HEAD:train.py"),
        "model_py_blob": ("rev-parse", "HEAD:model.py"),
        "origin": ("remote", "get-url", "origin"),
        "dirty": ("status", "--porcelain", "--untracked-files=all"),
    }
    for key, arguments in queries.items():
        success, value = _run_git(checkout, *arguments)
        if not success:
            blockers.append(f"NanoGPT git query {key!r} failed: {value}")
        else:
            details[key] = value

    expected = {
        "revision": NANOGPT_REVISION,
        "tree": NANOGPT_TREE,
        "train_py_blob": NANOGPT_TRAIN_BLOB,
        "model_py_blob": NANOGPT_MODEL_BLOB,
    }
    for key, expected_value in expected.items():
        observed = details.get(key)
        if observed is not None and observed != expected_value:
            blockers.append(f"NanoGPT {key} is {observed}, expected {expected_value}")

    origin = details.get("origin")
    permitted_origins = {
        "https://github.com/karpathy/nanoGPT",
        "https://github.com/karpathy/nanoGPT.git",
        "git@github.com:karpathy/nanoGPT.git",
    }
    if origin is not None and origin not in permitted_origins:
        blockers.append(f"NanoGPT origin is not the pinned repository: {origin}")
    if details.get("dirty"):
        blockers.append("NanoGPT checkout is dirty; instrumentation must remain an external patch")
    return {"ready": not blockers, "blockers": blockers, "details": details}


def _checked_file(
    *,
    manifest_root: Path,
    label: str,
    entry: object,
    blockers: list[str],
) -> dict[str, Any] | None:
    if not isinstance(entry, dict):
        blockers.append(f"{label} entry is missing")
        return None
    raw_path = entry.get("path")
    expected_digest = entry.get("sha256")
    expected_bytes = entry.get("byte_count")
    if not isinstance(raw_path, str) or not raw_path:
        blockers.append(f"{label} path is not materialized")
        return None
    if not _is_sha256(expected_digest):
        blockers.append(f"{label} SHA-256 is not a lowercase 64-digit digest")
        return None
    if (
        not isinstance(expected_bytes, int)
        or isinstance(expected_bytes, bool)
        or expected_bytes < 0
    ):
        blockers.append(f"{label} byte_count is invalid")
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = manifest_root / path
    path = path.resolve()
    if not path.is_file():
        blockers.append(f"{label} file is missing: {path}")
        return None
    observed_bytes = path.stat().st_size
    if observed_bytes != expected_bytes:
        blockers.append(f"{label} byte count is {observed_bytes}, expected {expected_bytes}")
        return None
    observed_digest = sha256_file(path)
    if observed_digest != expected_digest:
        blockers.append(f"{label} SHA-256 mismatch")
        return None
    return {"path": str(path), "byte_count": observed_bytes, "sha256": observed_digest}


def validate_fineweb_manifest(manifest_path: Path | None) -> dict[str, Any]:
    """Validate every frozen FineWeb, tokenizer, preprocessor, and output hash."""

    if manifest_path is None:
        return {"ready": False, "blockers": ["FineWeb manifest was not supplied"], "artifacts": []}
    manifest_path = manifest_path.resolve()
    if not manifest_path.is_file():
        return {
            "ready": False,
            "blockers": [f"FineWeb manifest is missing: {manifest_path}"],
            "artifacts": [],
        }
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {
            "ready": False,
            "blockers": [f"FineWeb manifest cannot be read: {error}"],
            "artifacts": [],
        }

    blockers: list[str] = []
    artifacts: list[dict[str, Any]] = []
    if payload.get("schema_version") != P22_DATA_SCHEMA:
        blockers.append("FineWeb manifest schema mismatch")
    if payload.get("status") != "materialized":
        blockers.append("FineWeb manifest is not marked materialized")

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        blockers.append("FineWeb dataset mapping is missing")
        dataset = {}
    expected_dataset = {
        "repository": "HuggingFaceFW/fineweb",
        "config": "sample-10BT",
        "split": "train",
    }
    for key, expected in expected_dataset.items():
        if dataset.get(key) != expected:
            blockers.append(f"FineWeb dataset {key} must be {expected!r}")
    if dataset.get("revision") != "9bb295ddab0e05d785b879661af7260fed5140fc":
        blockers.append("FineWeb dataset revision does not match the frozen commit")
    source_shards = dataset.get("source_shards")
    observed_row_ids: list[int] = []
    if not isinstance(source_shards, list) or not source_shards:
        blockers.append("at least one consumed FineWeb source shard is required")
    else:
        for index, entry in enumerate(source_shards):
            if isinstance(entry, dict):
                source_kind = entry.get("source_kind")
                if source_kind != "captured datasets-server response":
                    blockers.append(f"source shard {index} has no recognized source_kind")
            checked = _checked_file(
                manifest_root=manifest_path.parent,
                label=f"source shard {index}",
                entry=entry,
                blockers=blockers,
            )
            if checked is not None:
                artifacts.append(checked)
                try:
                    response = json.loads(Path(checked["path"]).read_text(encoding="utf-8"))
                except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
                    blockers.append(f"source shard {index} cannot be decoded: {error}")
                    continue
                rows = response.get("rows") if isinstance(response, dict) else None
                if not isinstance(rows, list) or not rows:
                    blockers.append(f"source shard {index} has no nonempty rows list")
                    continue
                for row_position, row in enumerate(rows):
                    if not isinstance(row, dict) or row.get("truncated_cells") != []:
                        blockers.append(
                            f"source shard {index} row {row_position} is truncated or malformed"
                        )
                        break
                    row_id = row.get("row_idx")
                    row_value = row.get("row")
                    if (
                        not isinstance(row_id, int)
                        or isinstance(row_id, bool)
                        or not isinstance(row_value, dict)
                        or not isinstance(row_value.get("text"), str)
                    ):
                        blockers.append(
                            f"source shard {index} row {row_position} lacks exact row/text data"
                        )
                        break
                    observed_row_ids.append(row_id)
                if isinstance(entry, dict) and isinstance(rows, list) and rows:
                    first_row = rows[0].get("row_idx") if isinstance(rows[0], dict) else None
                    last_row = rows[-1].get("row_idx") if isinstance(rows[-1], dict) else None
                    expected_url = (
                        "https://datasets-server.huggingface.co/rows?"
                        "dataset=HuggingFaceFW%2Ffineweb&config=sample-10BT&split=train"
                        f"&offset={first_row}&length={len(rows)}"
                        "&revision=9bb295ddab0e05d785b879661af7260fed5140fc"
                    )
                    exact_fields = {
                        "row_start": first_row,
                        "row_end_inclusive": last_row,
                        "row_count": len(rows),
                        "request_url": expected_url,
                    }
                    for field, expected in exact_fields.items():
                        if entry.get(field) != expected:
                            blockers.append(
                                f"source shard {index} {field} does not match captured rows"
                            )

    selection = dataset.get("selection")
    if not isinstance(selection, dict):
        blockers.append("FineWeb selection mapping is missing")
        selection = {}
    if selection.get("ordering") != (
        "Increasing row index at the pinned revision; complete untruncated text only"
    ):
        blockers.append("FineWeb row ordering changed")
    if not _is_sha256(selection.get("canonical_consumed_rows_sha256")):
        blockers.append("canonical consumed-row SHA-256 is missing")
    for key in ("first_row_id", "last_row_id_inclusive", "row_count"):
        value = selection.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value < 0:
            blockers.append(f"FineWeb selection {key} is invalid")
    if observed_row_ids:
        expected_rows = list(range(len(observed_row_ids)))
        if observed_row_ids != expected_rows:
            blockers.append("FineWeb source rows are not contiguous from row zero")
        if selection.get("first_row_id") != observed_row_ids[0]:
            blockers.append("FineWeb first_row_id does not match captured rows")
        selected_last = selection.get("last_row_id_inclusive")
        selected_count = selection.get("row_count")
        if (
            not isinstance(selected_last, int)
            or selected_last < 0
            or selected_last > observed_row_ids[-1]
        ):
            blockers.append("FineWeb last_row_id lies outside captured rows")
        if isinstance(selected_last, int) and selected_count != selected_last + 1:
            blockers.append("FineWeb row_count does not match the selected inclusive range")
    for boundary_name in ("train_boundary", "validation_boundary"):
        boundary = selection.get(boundary_name)
        if not isinstance(boundary, dict):
            blockers.append(f"FineWeb {boundary_name} is missing")
            continue
        for key in ("row_id", "token_offset_exclusive"):
            value = boundary.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                blockers.append(f"FineWeb {boundary_name}.{key} is invalid")

    tokenizer = payload.get("tokenizer")
    if not isinstance(tokenizer, dict):
        blockers.append("tokenizer mapping is missing")
        tokenizer = {}
    frozen_tokenizer = {
        "package": "tiktoken",
        "version": "0.14.0",
        "encoding": "gpt2",
        "vocab_size": 50_257,
    }
    for key, expected in frozen_tokenizer.items():
        if tokenizer.get(key) != expected:
            blockers.append(f"tokenizer {key} must be {expected!r}")
    if tokenizer.get("end_of_text_token_id") != 50_256:
        blockers.append("GPT-2 end-of-text token changed")
    if tokenizer.get("append_end_of_text_after_each_document") is not True:
        blockers.append("every FineWeb document must end with GPT-2 end-of-text")
    tokenizer_files = tokenizer.get("files")
    if not isinstance(tokenizer_files, dict):
        blockers.append("tokenizer file mapping is missing")
        tokenizer_files = {}
    frozen_asset_hashes = {
        "vocab.bpe": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
        "encoder.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
    }
    for filename, expected_digest in frozen_asset_hashes.items():
        entry = tokenizer_files.get(filename)
        if isinstance(entry, dict) and entry.get("sha256") != expected_digest:
            blockers.append(f"tokenizer {filename} does not match the frozen asset hash")
        checked = _checked_file(
            manifest_root=manifest_path.parent,
            label=f"tokenizer {filename}",
            entry=entry,
            blockers=blockers,
        )
        if checked is not None:
            artifacts.append(checked)

    checked_preprocessor = _checked_file(
        manifest_root=manifest_path.parent,
        label="preprocessor",
        entry=payload.get("preprocessor"),
        blockers=blockers,
    )
    if checked_preprocessor is not None:
        artifacts.append(checked_preprocessor)

    outputs = payload.get("outputs")
    if not isinstance(outputs, dict):
        blockers.append("token output mapping is missing")
        outputs = {}
    expected_outputs = {
        "train.bin": (EXPECTED_TRAIN_POOL_TOKENS, EXPECTED_TRAIN_BYTES),
        "val.bin": (EXPECTED_VALIDATION_TOKENS, EXPECTED_VALIDATION_BYTES),
    }
    for filename, (expected_tokens, expected_bytes) in expected_outputs.items():
        entry = outputs.get(filename)
        if not isinstance(entry, dict):
            blockers.append(f"{filename} entry is missing")
            continue
        if entry.get("token_count") != expected_tokens:
            blockers.append(f"{filename} token_count changed")
        if entry.get("byte_count") != expected_bytes:
            blockers.append(f"{filename} byte_count changed")
        if entry.get("dtype") != "little-endian uint16":
            blockers.append(f"{filename} dtype changed")
        checked = _checked_file(
            manifest_root=manifest_path.parent,
            label=filename,
            entry=entry,
            blockers=blockers,
        )
        if checked is not None:
            artifacts.append(checked)

    return {
        "ready": not blockers,
        "blockers": blockers,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "artifacts": artifacts,
    }


def validate_muon_source(source_path: Path | None) -> dict[str, Any]:
    """Validate the exact audited KellerJordan/Muon source bytes."""

    if source_path is None:
        return {"ready": False, "blockers": ["pinned Muon source was not supplied"]}
    source_path = source_path.resolve()
    if not source_path.is_file():
        return {"ready": False, "blockers": [f"Muon source is missing: {source_path}"]}
    digest = sha256_file(source_path)
    blockers = [] if digest == MUON_SOURCE_SHA256 else ["Muon source SHA-256 mismatch"]
    return {
        "ready": not blockers,
        "blockers": blockers,
        "path": str(source_path),
        "sha256": digest,
        "revision": MUON_REVISION,
    }


def validate_instrumentation_patch(patch_path: Path | None) -> dict[str, Any]:
    """Require a nonempty, externally reviewable instrumentation patch."""

    if patch_path is None:
        return {"ready": False, "blockers": ["NanoGPT instrumentation patch was not supplied"]}
    patch_path = patch_path.resolve()
    if not patch_path.is_file() or patch_path.stat().st_size == 0:
        return {
            "ready": False,
            "blockers": [f"instrumentation patch is missing or empty: {patch_path}"],
        }
    return {
        "ready": True,
        "blockers": [],
        "path": str(patch_path),
        "byte_count": patch_path.stat().st_size,
        "sha256": sha256_file(patch_path),
    }


def validate_accelerator_runtime(backend: str) -> dict[str, Any]:
    """Probe exactly one selected BF16 CUDA or MPS accelerator, never CPU."""

    if backend not in {"cuda", "mps"}:
        return {
            "ready": False,
            "blockers": ["accelerator backend must be explicitly selected as cuda or mps"],
            "details": {"selected_backend": backend},
        }

    blockers: list[str] = []
    details: dict[str, Any] = {
        "selected_backend": backend,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "mps_built": torch.backends.mps.is_built(),
        "mps_available": torch.backends.mps.is_available(),
    }
    if backend == "cuda":
        details.update(
            {
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                "evidence_scope": "CUDA-specific shadow trace; not automatic kernel parity",
            }
        )
        if not torch.cuda.is_available():
            blockers.append("selected CUDA backend is unavailable; substitution is forbidden")
            details["visible_device_count"] = 0
            return {"ready": False, "blockers": blockers, "details": details}
        device_count = torch.cuda.device_count()
        details["visible_device_count"] = device_count
        if device_count != 1:
            blockers.append(f"exactly one visible CUDA device is required, found {device_count}")
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
            blockers.append("CUBLAS_WORKSPACE_CONFIG must equal :4096:8 for CUDA")
        if device_count:
            properties = torch.cuda.get_device_properties(0)
            details.update(
                {
                    "device_name": properties.name,
                    "compute_capability": [properties.major, properties.minor],
                    "total_memory": properties.total_memory,
                    "bf16_supported": torch.cuda.is_bf16_supported(),
                }
            )
            if not torch.cuda.is_bf16_supported():
                blockers.append("visible CUDA device does not report BF16 support")
        return {"ready": not blockers, "blockers": blockers, "details": details}

    details.update(
        {
            "visible_device_count": 1 if torch.backends.mps.is_available() else 0,
            "pytorch_enable_mps_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "evidence_scope": (
                "Apple-MPS-specific shadow trace; explicitly not CUDA, tensor-core, "
                "or native-kernel parity"
            ),
        }
    )
    if not torch.backends.mps.is_available():
        blockers.append("selected MPS backend is unavailable; substitution is forbidden")
        return {"ready": False, "blockers": blockers, "details": details}
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") != "0":
        blockers.append("PYTORCH_ENABLE_MPS_FALLBACK must equal 0 for the MPS trace")
    if os.environ.get("PYTORCH_MPS_FAST_MATH") not in {None, "0"}:
        blockers.append("PYTORCH_MPS_FAST_MATH must be unset or equal 0")
    try:
        probe = torch.ones((2, 2), dtype=torch.bfloat16, device="mps")
        product = probe @ probe
        torch.mps.synchronize()
        details["bf16_supported"] = product.dtype == torch.bfloat16
        details["bf16_probe_sha256"] = canonical_state_sha256(product)
        if product.dtype != torch.bfloat16:
            blockers.append("MPS BF16 matmul did not retain BF16 storage")
    except (RuntimeError, TypeError) as error:
        details["bf16_supported"] = False
        details["bf16_probe_error"] = str(error)
        blockers.append("MPS BF16 matmul probe failed")
    return {"ready": not blockers, "blockers": blockers, "details": details}


def build_preflight_report(
    *,
    nanogpt_root: Path | None,
    muon_source: Path | None,
    fineweb_manifest: Path | None,
    instrumentation_patch: Path | None,
    accelerator_backend: str,
) -> dict[str, Any]:
    """Return a complete preflight report and aggregate every blocker."""

    load_protocol()
    checks = {
        "nanogpt": validate_nanogpt_checkout(nanogpt_root),
        "muon": validate_muon_source(muon_source),
        "fineweb": validate_fineweb_manifest(fineweb_manifest),
        "instrumentation": validate_instrumentation_patch(instrumentation_patch),
        "accelerator": validate_accelerator_runtime(accelerator_backend),
    }
    coverage = inventory_coverage()
    blockers = [
        f"{name}: {blocker}" for name, result in checks.items() for blocker in result["blockers"]
    ]
    if not coverage["complete"]:
        blockers.append("shape_inventory: GPT-2-small inventory is not fully P20/P22-certified")
    return {
        "schema_version": P22_PROTOCOL_SCHEMA,
        "status": "ready" if not blockers else "blocked",
        "real_run_executed": False,
        "protocol_path": PROTOCOL_PATH.as_posix(),
        "protocol_sha256": protocol_sha256(),
        "checks": checks,
        "shape_inventory": coverage,
        "blockers": blockers,
        "claim_boundary": (
            "This is a prerequisite report, not a gradient trace, training result, or theorem."
        ),
    }


def _update_digest(digest: Any, value: object) -> None:
    """Encode a state tree with explicit type and length separators."""

    def emit(tag: bytes, payload: bytes = b"") -> None:
        digest.update(tag)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)

    if value is None:
        emit(b"none")
    elif isinstance(value, bool):
        emit(b"bool", b"1" if value else b"0")
    elif isinstance(value, int):
        emit(b"int", str(value).encode("ascii"))
    elif isinstance(value, float):
        if not math.isfinite(value):
            raise P22ProtocolError("state hash refuses nonfinite Python floats")
        emit(b"float64", struct.pack(">d", value))
    elif isinstance(value, str):
        emit(b"str", value.encode("utf-8"))
    elif isinstance(value, bytes):
        emit(b"bytes", value)
    elif isinstance(value, Tensor):
        if value.layout != torch.strided:
            raise P22ProtocolError("state hash supports only strided tensors")
        stored = value.detach().cpu().contiguous()
        emit(b"tensor-dtype", str(stored.dtype).encode("ascii"))
        emit(b"tensor-shape", json.dumps(list(stored.shape)).encode("ascii"))
        emit(b"tensor-bytes", stored.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    elif isinstance(value, np.ndarray):
        stored_array = np.ascontiguousarray(value)
        emit(b"ndarray-dtype", stored_array.dtype.str.encode("ascii"))
        emit(b"ndarray-shape", json.dumps(list(stored_array.shape)).encode("ascii"))
        emit(b"ndarray-bytes", stored_array.tobytes(order="C"))
    elif isinstance(value, Mapping):
        emit(b"mapping-size", str(len(value)).encode("ascii"))
        ordered: list[tuple[str, object, object]] = []
        for key, item in value.items():
            key_order = f"{type(key).__module__}.{type(key).__qualname__}:{key!r}"
            ordered.append((key_order, key, item))
        for _, key, item in sorted(ordered, key=lambda element: element[0]):
            _update_digest(digest, key)
            _update_digest(digest, item)
    elif isinstance(value, Sequence):
        emit(b"sequence-size", str(len(value)).encode("ascii"))
        for item in value:
            _update_digest(digest, item)
    else:
        raise P22ProtocolError(f"unsupported state-hash type: {type(value)!r}")


def canonical_state_sha256(value: object) -> str:
    """Hash nested model/optimizer/RNG state without pickle or tolerances."""

    digest = hashlib.sha256()
    _update_digest(digest, value)
    return digest.hexdigest()


def capture_rng_state(accelerator_backend: str | None = None) -> dict[str, object]:
    """Capture host RNGs and the explicitly selected accelerator RNG."""

    if accelerator_backend not in {None, "cuda", "mps"}:
        raise P22ProtocolError("RNG accelerator backend must be cuda, mps, or None")
    capture_cuda = accelerator_backend == "cuda"
    capture_mps = accelerator_backend == "mps"
    if capture_cuda and not torch.cuda.is_available():
        raise P22ProtocolError("selected CUDA RNG is unavailable")
    if capture_mps and not torch.backends.mps.is_available():
        raise P22ProtocolError("selected MPS RNG is unavailable")

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda_all": [state.clone() for state in torch.cuda.get_rng_state_all()]
        if capture_cuda
        else [],
        "torch_mps": torch.mps.get_rng_state().clone() if capture_mps else None,
    }


def restore_rng_state(snapshot: Mapping[str, object]) -> None:
    """Restore a snapshot returned by :func:`capture_rng_state`."""

    random.setstate(snapshot["python"])  # type: ignore[arg-type]
    np.random.set_state(snapshot["numpy"])  # type: ignore[arg-type]
    torch.set_rng_state(snapshot["torch_cpu"])  # type: ignore[arg-type]
    cuda_states = snapshot["torch_cuda_all"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise P22ProtocolError("cannot restore recorded CUDA RNG on a non-CUDA runtime")
        torch.cuda.set_rng_state_all(cuda_states)  # type: ignore[arg-type]
    mps_state = snapshot["torch_mps"]
    if mps_state is not None:
        if not torch.backends.mps.is_available():
            raise P22ProtocolError("cannot restore recorded MPS RNG on a non-MPS runtime")
        torch.mps.set_rng_state(mps_state)  # type: ignore[arg-type]


def run_rng_neutral_observer(
    observer: Callable[[], T], *, accelerator_backend: str | None = None
) -> tuple[T, str]:
    """Run an observer, restoring and rejecting any RNG-state mutation."""

    before = capture_rng_state(accelerator_backend)
    before_digest = canonical_state_sha256(before)
    try:
        result = observer()
    except BaseException:
        after = capture_rng_state(accelerator_backend)
        if canonical_state_sha256(after) != before_digest:
            restore_rng_state(before)
        raise
    after = capture_rng_state(accelerator_backend)
    if canonical_state_sha256(after) != before_digest:
        restore_rng_state(before)
        raise P22ProtocolError("shadow observer changed training RNG state; restored and aborted")
    return result, before_digest


def _validate_model_optimizer_binding(payload: Mapping[str, object], expected_mode: str) -> None:
    """Validate the stable record of the in-process post-move identity audit."""

    audit = payload.get("model_optimizer_binding")
    if not isinstance(audit, Mapping):
        raise P22ProtocolError(f"{expected_mode} has no model/optimizer binding audit")
    for field in (
        "verified",
        "complete_unique_identity_coverage",
        "no_duplicate_optimizer_parameters",
        "muon_metadata_exact",
    ):
        if audit.get(field) is not True:
            raise P22ProtocolError(f"{expected_mode} binding audit did not verify {field}")
    if audit.get("verification") != (
        "exact in-process Python object identity after accelerator move"
    ):
        raise P22ProtocolError(f"{expected_mode} binding-audit method changed")

    backend = payload["accelerator_backend"]
    if audit.get("selected_device_type") != backend:
        raise P22ProtocolError(f"{expected_mode} binding audit names the wrong device type")
    devices = audit.get("stored_parameter_devices")
    if (
        not isinstance(devices, list)
        or not devices
        or any(
            not isinstance(device, str)
            or (device != backend and not device.startswith(f"{backend}:"))
            for device in devices
        )
    ):
        raise P22ProtocolError(f"{expected_mode} binding audit has the wrong stored devices")
    if audit.get("stored_parameter_dtypes") != ["torch.float32"]:
        raise P22ProtocolError(f"{expected_mode} binding audit is not entirely FP32")

    expected_counts = {
        "model_unique_parameter_count": EXPECTED_MODEL_PARAMETER_COUNT,
        "optimizer_parameter_count": EXPECTED_MODEL_PARAMETER_COUNT,
        "muon_parameter_count": EXPECTED_MUON_PARAMETER_COUNT,
        "auxiliary_parameter_count": EXPECTED_AUXILIARY_PARAMETER_COUNT,
    }
    if any(audit.get(field) != value for field, value in expected_counts.items()):
        raise P22ProtocolError(f"{expected_mode} binding-audit parameter counts changed")
    if audit.get("tied_parameter_aliases") != EXPECTED_TIED_PARAMETER_ALIASES:
        raise P22ProtocolError(f"{expected_mode} binding-audit alias inventory changed")
    if audit.get("model_config") != {
        "block_size": 1024,
        "vocab_size": 50_257,
        "n_layer": 12,
        "n_head": 12,
        "n_embd": 768,
        "dropout": 0.0,
        "bias": False,
    }:
        raise P22ProtocolError(f"{expected_mode} binding-audit model configuration changed")

    groups = audit.get("optimizer_groups")
    if not isinstance(groups, list) or len(groups) != 2:
        raise P22ProtocolError(f"{expected_mode} binding audit has the wrong optimizer groups")
    expected_group_contract = (
        (
            0,
            False,
            EXPECTED_AUXILIARY_PARAMETER_COUNT,
            {
                "learning_rate": 3.0 / 5_000.0,
                "betas": [0.9, 0.95],
                "epsilon": 1.0e-10,
                "weight_decay": 0.0,
            },
        ),
        (
            1,
            True,
            48,
            {
                "learning_rate": 1.0 / 120.0,
                "momentum": 19.0 / 20.0,
                "weight_decay": 0.0,
            },
        ),
    )
    names: list[str] = []
    for group, (index, use_muon, count, hyperparameters) in zip(
        groups, expected_group_contract, strict=True
    ):
        if not isinstance(group, Mapping):
            raise P22ProtocolError(f"{expected_mode} binding audit has a malformed group")
        parameter_names = group.get("parameter_names")
        if (
            group.get("group_index") != index
            or group.get("use_muon") is not use_muon
            or group.get("hyperparameters") != hyperparameters
            or not isinstance(parameter_names, list)
            or len(parameter_names) != count
            or any(not isinstance(name, str) or name.startswith("<") for name in parameter_names)
        ):
            raise P22ProtocolError(f"{expected_mode} binding-audit group contract changed")
        names.extend(parameter_names)
    if len(names) != len(set(names)) or len(names) != EXPECTED_MODEL_PARAMETER_COUNT:
        raise P22ProtocolError(f"{expected_mode} binding audit repeats a parameter name")


def _load_run_manifest(path: Path, expected_mode: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise P22ProtocolError(f"cannot read {expected_mode} manifest: {error}") from error
    schema_version = payload.get("schema_version")
    if schema_version not in {P22_LEGACY_RUN_MANIFEST_SCHEMA, P22_RUN_MANIFEST_SCHEMA}:
        raise P22ProtocolError(f"{expected_mode} manifest schema mismatch")
    legacy_manifest = schema_version == P22_LEGACY_RUN_MANIFEST_SCHEMA
    if payload.get("trace_mode") != expected_mode:
        raise P22ProtocolError(f"expected trace_mode={expected_mode!r}")
    if payload.get("evidence_kind") != "real_gradient_shadow_trace":
        raise P22ProtocolError(f"{expected_mode} is not labeled real-gradient evidence")
    if payload.get("protocol_sha256") != protocol_sha256():
        raise P22ProtocolError(f"{expected_mode} protocol SHA-256 mismatch")
    if payload.get("shadow_update_applied") is not False:
        raise P22ProtocolError(f"{expected_mode} applied or ambiguously applied a shadow update")
    if payload.get("accelerator_backend") not in {"cuda", "mps"}:
        raise P22ProtocolError(f"{expected_mode} has no explicit CUDA/MPS backend")
    if legacy_manifest:
        # Historical P22 trace-off manifests used this ambiguous boolean.  Keep
        # them replayable without carrying that schema into new acquisitions.
        if payload.get("actual_accelerator_candidates") is not True:
            raise P22ProtocolError(f"{expected_mode} did not use actual accelerator candidates")
    else:
        accelerator_backend = payload["accelerator_backend"]
        execution = payload.get("candidate_execution")
        if execution != {
            "backend": accelerator_backend,
            "actual_accelerator_candidate_computed_and_applied": True,
            "shield_shadow_only": True,
        }:
            raise P22ProtocolError(f"{expected_mode} candidate-execution contract mismatch")
        observation = payload.get("candidate_observation")
        expected_observation = (
            {
                "status": f"observed_actual_post_aspect_{accelerator_backend}",
                "capture_step_count": len(EXPECTED_CAPTURE_STEPS),
                "observation_count": EXPECTED_CANDIDATE_OBSERVATIONS,
            }
            if expected_mode == "trace_on"
            else {
                "status": "not_observed",
                "capture_step_count": 0,
                "observation_count": 0,
            }
        )
        if observation != expected_observation:
            raise P22ProtocolError(f"{expected_mode} candidate-observation contract mismatch")
        _validate_model_optimizer_binding(payload, expected_mode)
    for field in ("run_identity_sha256", "initial_state_sha256"):
        if not _is_sha256(payload.get(field)):
            raise P22ProtocolError(f"{expected_mode} has an invalid {field}")
    expected_captures = list(EXPECTED_CAPTURE_STEPS) if expected_mode == "trace_on" else []
    if payload.get("capture_steps") != expected_captures:
        raise P22ProtocolError(f"{expected_mode} capture-step schedule mismatch")
    steps = payload.get("steps")
    if not isinstance(steps, list) or len(steps) != EXPECTED_OPTIMIZER_STEPS:
        raise P22ProtocolError(f"{expected_mode} must contain exactly 256 step records")
    if [record.get("step") for record in steps if isinstance(record, dict)] != list(
        range(EXPECTED_OPTIMIZER_STEPS)
    ):
        raise P22ProtocolError(f"{expected_mode} step indices are incomplete or out of order")
    for step, record in enumerate(steps):
        expected_offset = step * EXPECTED_SEQUENCE_LENGTH
        expected_tokens_seen = (step + 1) * EXPECTED_SEQUENCE_LENGTH
        if record.get("data_token_offset") != expected_offset:
            raise P22ProtocolError(f"{expected_mode} step {step} data offset changed")
        if record.get("tokens_seen") != expected_tokens_seen:
            raise P22ProtocolError(f"{expected_mode} step {step} token count changed")
        digest_fields = (
            "batch_sha256",
            "loss_tensor_sha256",
            "rng_before_step_sha256",
            "rng_after_step_sha256",
        )
        if not all(_is_sha256(record.get(field)) for field in digest_fields):
            raise P22ProtocolError(f"{expected_mode} step record has an invalid digest")
        expected_checkpoint = step in STATE_CHECKPOINT_STEPS
        if record.get("state_checkpoint_present") is not expected_checkpoint:
            raise P22ProtocolError(f"{expected_mode} step {step} state-checkpoint flag changed")
        for field in NONINTERFERENCE_STATE_FIELDS:
            value = record.get(field)
            if expected_checkpoint and not _is_sha256(value):
                raise P22ProtocolError(f"{expected_mode} step {step} has no exact state digest")
            if not expected_checkpoint and value is not None:
                raise P22ProtocolError(
                    f"{expected_mode} step {step} has an undeclared state checkpoint"
                )
        if expected_mode == "trace_off":
            if record.get("capture") is not None:
                raise P22ProtocolError(f"trace_off step {step} contains a shadow capture")
            if not legacy_manifest and record.get("observation_count") != 0:
                raise P22ProtocolError(f"trace_off step {step} has a candidate observation")
            observer_status = record.get("observer_rng_unchanged")
            if observer_status is not None and observer_status is not False:
                raise P22ProtocolError(f"trace_off step {step} claims an observer invocation")
    if expected_mode == "trace_on":
        for step in EXPECTED_CAPTURE_STEPS:
            capture = steps[step].get("capture")
            if not isinstance(capture, dict):
                raise P22ProtocolError(f"trace_on step {step} has no capture record")
            required_capture = {
                "parameter_count": 48,
                "actual_post_aspect_candidate": True,
                "actual_post_aspect_cuda_candidate": payload["accelerator_backend"] == "cuda",
                "p20_shadow_only": True,
            }
            if any(capture.get(key) != value for key, value in required_capture.items()):
                raise P22ProtocolError(f"trace_on step {step} capture contract mismatch")
            if not legacy_manifest and steps[step].get("observation_count") != 48:
                raise P22ProtocolError(f"trace_on step {step} observation count is not 48")
            if not isinstance(capture.get("actual_pre_aspect_candidate"), bool):
                raise P22ProtocolError(f"trace_on step {step} pre-aspect status is missing")
            if not _is_sha256(capture.get("record_sha256")):
                raise P22ProtocolError(f"trace_on step {step} capture digest is invalid")
            if steps[step].get("observer_rng_unchanged") is not True:
                raise P22ProtocolError(f"trace_on step {step} did not preserve observer RNG")
        for step in set(range(EXPECTED_OPTIMIZER_STEPS)) - set(EXPECTED_CAPTURE_STEPS):
            if steps[step].get("capture") is not None:
                raise P22ProtocolError(f"trace_on step {step} has an unscheduled capture")
            if not legacy_manifest and steps[step].get("observation_count") != 0:
                raise P22ProtocolError(
                    f"trace_on step {step} has an unscheduled candidate observation"
                )
        if not legacy_manifest:
            observed = sum(int(record["observation_count"]) for record in steps)
            if observed != EXPECTED_CANDIDATE_OBSERVATIONS:
                raise P22ProtocolError("trace_on total candidate-observation count changed")
            raw_trace = payload.get("raw_trace")
            if (
                not isinstance(raw_trace, dict)
                or raw_trace.get("observation_count") != EXPECTED_CANDIDATE_OBSERVATIONS
                or not _is_sha256(raw_trace.get("sha256"))
            ):
                raise P22ProtocolError("trace_on raw-trace observation contract mismatch")
    if (
        not legacy_manifest
        and expected_mode == "trace_off"
        and payload.get("raw_trace") is not None
    ):
        raise P22ProtocolError("trace_off manifest must not reference a raw candidate trace")
    final_state = payload.get("final_state")
    if not isinstance(final_state, dict):
        raise P22ProtocolError(f"{expected_mode} has no final state")
    for field in NONINTERFERENCE_FINAL_FIELDS:
        if not _is_sha256(final_state.get(field)):
            raise P22ProtocolError(f"{expected_mode} has an invalid final {field}")
    return payload


def compare_noninterference_manifests(trace_off_path: Path, trace_on_path: Path) -> dict[str, Any]:
    """Apply the off-A/trace-on leg of the frozen three-run gate."""

    trace_off_path = trace_off_path.resolve()
    trace_on_path = trace_on_path.resolve()
    off = _load_run_manifest(trace_off_path, "trace_off")
    on = _load_run_manifest(trace_on_path, "trace_on")
    mismatches: list[dict[str, Any]] = []

    for field in ("run_identity_sha256", "initial_state_sha256"):
        if off.get(field) != on.get(field):
            mismatches.append({"scope": "run", "field": field})

    off_steps = off["steps"]
    on_steps = on["steps"]
    for step, (off_record, on_record) in enumerate(zip(off_steps, on_steps, strict=True)):
        for field in NONINTERFERENCE_STEP_FIELDS:
            if off_record.get(field) != on_record.get(field):
                mismatches.append({"scope": "step", "step": step, "field": field})
        if step in STATE_CHECKPOINT_STEPS:
            for field in NONINTERFERENCE_STATE_FIELDS:
                if off_record.get(field) != on_record.get(field):
                    mismatches.append({"scope": "state", "step": step, "field": field})
        if step in EXPECTED_CAPTURE_STEPS and on_record.get("observer_rng_unchanged") is not True:
            mismatches.append({"scope": "step", "step": step, "field": "observer_rng_unchanged"})

    off_final = off.get("final_state")
    on_final = on.get("final_state")
    if not isinstance(off_final, dict) or not isinstance(on_final, dict):
        mismatches.append({"scope": "final", "field": "final_state"})
    else:
        for field in NONINTERFERENCE_FINAL_FIELDS:
            if off_final.get(field) != on_final.get(field):
                mismatches.append({"scope": "final", "field": field})

    captures = on.get("capture_steps")
    if captures != list(EXPECTED_CAPTURE_STEPS):
        mismatches.append({"scope": "trace_on", "field": "capture_steps"})
    if off.get("capture_steps") not in ([], None):
        mismatches.append({"scope": "trace_off", "field": "capture_steps"})

    return {
        "schema_version": P22_COMPARISON_SCHEMA,
        "passes": not mismatches,
        "comparison": "bitwise_exact_no_tolerances",
        "trace_off_path": str(trace_off_path),
        "trace_off_sha256": sha256_file(trace_off_path),
        "trace_on_path": str(trace_on_path),
        "trace_on_sha256": sha256_file(trace_on_path),
        "checked_step_count": EXPECTED_OPTIMIZER_STEPS,
        "checked_step_fields": list(NONINTERFERENCE_STEP_FIELDS),
        "checked_state_fields": list(NONINTERFERENCE_STATE_FIELDS),
        "state_checkpoint_steps": sorted(STATE_CHECKPOINT_STEPS),
        "checked_final_fields": list(NONINTERFERENCE_FINAL_FIELDS),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "claim_boundary": (
            "Passing establishes noninterference for these two pinned executions only; it is "
            "not a backend-independent proof and says nothing about training quality."
        ),
    }


def compare_repeatability_manifests(
    trace_off_a_path: Path,
    trace_off_b_path: Path,
) -> dict[str, Any]:
    """Require two fresh trace-off MPS/CUDA executions to match exactly."""

    trace_off_a_path = trace_off_a_path.resolve()
    trace_off_b_path = trace_off_b_path.resolve()
    left = _load_run_manifest(trace_off_a_path, "trace_off")
    right = _load_run_manifest(trace_off_b_path, "trace_off")
    mismatches: list[dict[str, Any]] = []
    for field in ("run_identity_sha256", "initial_state_sha256"):
        if left.get(field) != right.get(field):
            mismatches.append({"scope": "run", "field": field})
    for step, (left_record, right_record) in enumerate(
        zip(left["steps"], right["steps"], strict=True)
    ):
        for field in NONINTERFERENCE_STEP_FIELDS:
            if left_record.get(field) != right_record.get(field):
                mismatches.append({"scope": "step", "step": step, "field": field})
        if step in STATE_CHECKPOINT_STEPS:
            for field in NONINTERFERENCE_STATE_FIELDS:
                if left_record.get(field) != right_record.get(field):
                    mismatches.append({"scope": "state", "step": step, "field": field})
    for field in NONINTERFERENCE_FINAL_FIELDS:
        if left["final_state"].get(field) != right["final_state"].get(field):
            mismatches.append({"scope": "final", "field": field})
    return {
        "schema_version": "passive-muon-p22-baseline-repeatability-v1",
        "passes": not mismatches,
        "comparison": "bitwise_exact_no_tolerances",
        "trace_off_a_path": str(trace_off_a_path),
        "trace_off_a_sha256": sha256_file(trace_off_a_path),
        "trace_off_b_path": str(trace_off_b_path),
        "trace_off_b_sha256": sha256_file(trace_off_b_path),
        "checked_step_count": EXPECTED_OPTIMIZER_STEPS,
        "checked_step_fields": list(NONINTERFERENCE_STEP_FIELDS),
        "checked_state_fields": list(NONINTERFERENCE_STATE_FIELDS),
        "state_checkpoint_steps": sorted(STATE_CHECKPOINT_STEPS),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches,
        "claim_boundary": (
            "Passing establishes repeatability only for these two pinned executions; it is "
            "not a backend-wide determinism guarantee."
        ),
    }


def write_json_atomic(path: Path, payload: Mapping[str, object]) -> None:
    """Write one JSON artifact atomically with nonfinite numbers forbidden."""

    path = path.resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as handle:
        handle.write(rendered)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _emit(payload: Mapping[str, object], output: Path | None) -> None:
    if output is None:
        print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    else:
        write_json_atomic(output, payload)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="validate prerequisites; never train")
    preflight.add_argument("--nanogpt-root", type=Path)
    preflight.add_argument("--muon-source", type=Path)
    preflight.add_argument("--fineweb-manifest", type=Path)
    preflight.add_argument("--instrumentation-patch", type=Path)
    preflight.add_argument("--accelerator", choices=("cuda", "mps"), required=True)
    preflight.add_argument("--output", type=Path)

    compare = subparsers.add_parser(
        "verify-noninterference", help="compare complete trace-off/trace-on run manifests"
    )
    compare.add_argument("--trace-off", type=Path, required=True)
    compare.add_argument("--trace-on", type=Path, required=True)
    compare.add_argument("--output", type=Path)

    repeat = subparsers.add_parser(
        "verify-repeatability", help="compare two fresh trace-off run manifests"
    )
    repeat.add_argument("--trace-off-a", type=Path, required=True)
    repeat.add_argument("--trace-off-b", type=Path, required=True)
    repeat.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "preflight":
        report = build_preflight_report(
            nanogpt_root=args.nanogpt_root,
            muon_source=args.muon_source,
            fineweb_manifest=args.fineweb_manifest,
            instrumentation_patch=args.instrumentation_patch,
            accelerator_backend=args.accelerator,
        )
        _emit(report, args.output)
        return 0 if report["status"] == "ready" else 2
    if args.command == "verify-noninterference":
        try:
            report = compare_noninterference_manifests(args.trace_off, args.trace_on)
        except P22ProtocolError as error:
            report = {
                "schema_version": P22_COMPARISON_SCHEMA,
                "passes": False,
                "error": str(error),
            }
        _emit(report, args.output)
        return 0 if report["passes"] else 2
    if args.command == "verify-repeatability":
        try:
            report = compare_repeatability_manifests(args.trace_off_a, args.trace_off_b)
        except P22ProtocolError as error:
            report = {
                "schema_version": "passive-muon-p22-baseline-repeatability-v1",
                "passes": False,
                "error": str(error),
            }
        _emit(report, args.output)
        return 0 if report["passes"] else 2
    raise AssertionError(f"unreachable command: {args.command}")


if __name__ == "__main__":
    sys.exit(main())
