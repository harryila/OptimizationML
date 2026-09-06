#!/usr/bin/env python3
"""Run the frozen P22 NanoGPT real-gradient shadow trace.

The trainer is deliberately small in surface area: it imports the model from
the exact pinned nanoGPT checkout and the optimizer from the exact pinned
KellerJordan/Muon source, then executes the predeclared 256-step sequential
FineWeb run.  The P20/P22 shield is record-only; the baseline always applies
the unshielded candidate returned by the pinned optimizer.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
import torch
from torch import Tensor

from passive_muon.p21_shadow_trace import CAPTURE_STEPS, summarize_shadow_trace
from passive_muon.p22_real_gradient_shadow_trace import observe_post_aspect_candidate

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from p22_nanogpt_shadow_trace import (  # noqa: E402
    EXPECTED_CANDIDATE_OBSERVATIONS,
    EXPECTED_OPTIMIZER_STEPS,
    EXPECTED_TRAINING_TOKENS_CONSUMED,
    P22_RUN_MANIFEST_SCHEMA,
    build_preflight_report,
    canonical_state_sha256,
    capture_rng_state,
    protocol_sha256,
    run_rng_neutral_observer,
    sha256_file,
    validate_fineweb_manifest,
    write_json_atomic,
)
from p22_observed_muon import MuonParameterMetadata, ObservedMuonStep  # noqa: E402

SEED = 1337
SEQUENCE_LENGTH = 128
MICRO_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 1
MODEL_BLOCK_SIZE = 1024
VOCAB_SIZE = 50_257
MODEL_N_LAYER = 12
MODEL_N_HEAD = 12
MODEL_N_EMBD = 768
MODEL_DROPOUT = 0.0
MODEL_BIAS = False
MUON_LEARNING_RATE = 1.0 / 120.0
MUON_MOMENTUM = 19.0 / 20.0
MUON_WEIGHT_DECAY = 0.0
AUXILIARY_LEARNING_RATE = 3.0 / 5_000.0
AUXILIARY_BETAS = (0.9, 0.95)
AUXILIARY_EPSILON = 1.0e-10
AUXILIARY_WEIGHT_DECAY = 0.0
STATE_CHECKPOINT_STEPS = frozenset((*CAPTURE_STEPS, EXPECTED_OPTIMIZER_STEPS - 1))
MUON_NAME = re.compile(
    r"^transformer\.h\.(?P<layer>[0-9]+)\."
    r"(?P<role>attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
)
ROLE_NAMES = {
    "attn.c_attn": "fused_qkv",
    "attn.c_proj": "attention_output",
    "mlp.c_fc": "mlp_expand",
    "mlp.c_proj": "mlp_contract",
}


class P22RunError(RuntimeError):
    """Raised when the real trace cannot honor its frozen contract."""


EXPECTED_TIED_ALIAS_INVENTORY = (("lm_head.weight", "transformer.wte.weight"),)


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise P22RunError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _tensor_sha256(value: Tensor) -> str:
    return canonical_state_sha256(value)


def _synchronize(backend: str) -> None:
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()
    else:  # pragma: no cover - guarded by preflight
        raise P22RunError(f"unsupported accelerator {backend}")


def _set_determinism(backend: str) -> None:
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if backend == "cuda":
        torch.cuda.manual_seed_all(SEED)
    elif backend == "mps":
        torch.mps.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.set_float32_matmul_precision("highest")
    if hasattr(torch.backends, "cuda"):
        torch.backends.cuda.matmul.allow_tf32 = False
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False


def _audit_model_optimizer_binding(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    parameter_metadata: Mapping[int, MuonParameterMetadata],
    backend: str,
    expected_muon_parameter_count: int = 48,
    expected_alias_inventory: tuple[tuple[str, ...], ...] = EXPECTED_TIED_ALIAS_INVENTORY,
) -> dict[str, object]:
    """Fail unless the optimizer owns the exact post-move model parameters.

    Object identities are checked in-process but never serialized because
    Python object ids are not meaningful across the three fresh executions.
    The manifest instead retains stable parameter names and verified counts.
    """

    named_parameters = list(model.named_parameters(remove_duplicate=False))
    aliases_by_identity: dict[int, list[str]] = {}
    aliases_by_storage: dict[int, list[str]] = {}
    parameter_by_identity: dict[int, Tensor] = {}
    canonical_name_by_identity: dict[int, str] = {}
    for name, parameter in named_parameters:
        identity = id(parameter)
        aliases_by_identity.setdefault(identity, []).append(name)
        aliases_by_storage.setdefault(parameter.untyped_storage().data_ptr(), []).append(name)
        parameter_by_identity[identity] = parameter
        canonical_name_by_identity.setdefault(identity, name)

    identity_aliases = tuple(
        sorted(tuple(sorted(names)) for names in aliases_by_identity.values() if len(names) > 1)
    )
    storage_aliases = tuple(
        sorted(tuple(sorted(names)) for names in aliases_by_storage.values() if len(names) > 1)
    )
    normalized_expected_aliases = tuple(
        sorted(tuple(sorted(names)) for names in expected_alias_inventory)
    )
    if identity_aliases != normalized_expected_aliases:
        raise P22RunError(f"unexpected exact parameter alias inventory: {identity_aliases}")
    if storage_aliases != identity_aliases:
        raise P22RunError("parameter storage aliases do not match exact object aliases")

    config = getattr(model, "config", None)
    expected_model_config = {
        "block_size": MODEL_BLOCK_SIZE,
        "vocab_size": VOCAB_SIZE,
        "n_layer": MODEL_N_LAYER,
        "n_head": MODEL_N_HEAD,
        "n_embd": MODEL_N_EMBD,
        "dropout": MODEL_DROPOUT,
        "bias": MODEL_BIAS,
    }
    if config is None or any(
        type(getattr(config, field, None)) is not type(expected)
        or getattr(config, field, None) != expected
        for field, expected in expected_model_config.items()
    ):
        raise P22RunError("actual model configuration differs from the frozen GPT-2 contract")

    optimizer_parameters: list[Tensor] = []
    group_records: list[dict[str, object]] = []
    muon_parameters: list[Tensor] = []
    optimizer_membership: dict[int, tuple[int, bool]] = {}
    for group_index, group in enumerate(optimizer.param_groups):
        parameters = list(group["params"])
        optimizer_parameters.extend(parameters)
        use_muon = bool(group.get("use_muon"))
        for parameter in parameters:
            optimizer_membership[id(parameter)] = (group_index, use_muon)
        if use_muon:
            muon_parameters.extend(parameters)
        expected_hyperparameters = (
            {
                "learning_rate": MUON_LEARNING_RATE,
                "momentum": MUON_MOMENTUM,
                "weight_decay": MUON_WEIGHT_DECAY,
            }
            if use_muon
            else {
                "learning_rate": AUXILIARY_LEARNING_RATE,
                "betas": list(AUXILIARY_BETAS),
                "epsilon": AUXILIARY_EPSILON,
                "weight_decay": AUXILIARY_WEIGHT_DECAY,
            }
        )
        actual_hyperparameters = (
            {
                "learning_rate": group.get("lr"),
                "momentum": group.get("momentum"),
                "weight_decay": group.get("weight_decay"),
            }
            if use_muon
            else {
                "learning_rate": group.get("lr"),
                "betas": list(group.get("betas", ())),
                "epsilon": group.get("eps"),
                "weight_decay": group.get("weight_decay"),
            }
        )
        if actual_hyperparameters != expected_hyperparameters:
            raise P22RunError("actual optimizer hyperparameters differ from the frozen contract")
        group_records.append(
            {
                "group_index": group_index,
                "use_muon": use_muon,
                "hyperparameters": expected_hyperparameters,
                "parameter_names": [
                    canonical_name_by_identity.get(id(parameter), "<not-a-model-parameter>")
                    for parameter in parameters
                ],
            }
        )

    optimizer_identities = [id(parameter) for parameter in optimizer_parameters]
    model_identities = set(parameter_by_identity)
    if len(optimizer_identities) != len(set(optimizer_identities)):
        raise P22RunError("an exact model parameter occurs more than once in optimizer groups")
    if set(optimizer_identities) != model_identities:
        raise P22RunError("optimizer groups do not exactly cover the unique model parameters")
    if any(
        parameter_by_identity[id(parameter)] is not parameter for parameter in optimizer_parameters
    ):
        raise P22RunError("optimizer retained a nonidentical model parameter object")

    selected_device = torch.device(backend)
    if any(parameter.device.type != selected_device.type for parameter in optimizer_parameters):
        raise P22RunError("optimizer parameter is not on the selected accelerator")
    if selected_device.type == "cuda" and any(
        parameter.device.index not in {None, selected_device.index or 0}
        for parameter in optimizer_parameters
    ):
        raise P22RunError("optimizer parameter is on the wrong CUDA device")
    if any(parameter.dtype != torch.float32 for parameter in optimizer_parameters):
        raise P22RunError("model and optimizer parameters must remain FP32")

    muon_identities = {id(parameter) for parameter in muon_parameters}
    if len(muon_parameters) != expected_muon_parameter_count:
        raise P22RunError(
            f"optimizer has {len(muon_parameters)} Muon parameters, "
            f"expected {expected_muon_parameter_count}"
        )
    if muon_identities != set(parameter_metadata):
        raise P22RunError("Muon metadata does not exactly cover the optimizer Muon group")
    for identity, metadata in parameter_metadata.items():
        if canonical_name_by_identity.get(identity) != metadata.name:
            raise P22RunError("Muon metadata name does not match the exact model parameter")

    parameter_inventory: list[dict[str, object]] = []
    for identity, canonical_name in canonical_name_by_identity.items():
        parameter = parameter_by_identity[identity]
        membership = optimizer_membership.get(identity)
        if membership is None:  # pragma: no cover - guarded by exact coverage above
            raise P22RunError("model parameter has no optimizer-group membership")
        group_index, use_muon = membership
        parameter_inventory.append(
            {
                "name": canonical_name,
                "aliases": sorted(aliases_by_identity[identity]),
                "optimizer_group_index": group_index,
                "use_muon": use_muon,
                "device": str(parameter.device),
                "dtype": str(parameter.dtype),
                "shape": list(parameter.shape),
                "numel": parameter.numel(),
            }
        )
    parameter_inventory.sort(key=lambda record: str(record["name"]))

    return {
        "verified": True,
        "verification": "exact in-process Python object identity after accelerator move",
        "selected_device_type": selected_device.type,
        "stored_parameter_devices": sorted(
            {str(parameter.device) for parameter in optimizer_parameters}
        ),
        "stored_parameter_dtypes": sorted(
            {str(parameter.dtype) for parameter in optimizer_parameters}
        ),
        "model_unique_parameter_count": len(model_identities),
        "optimizer_parameter_count": len(optimizer_parameters),
        "muon_parameter_count": len(muon_parameters),
        "auxiliary_parameter_count": len(optimizer_parameters) - len(muon_parameters),
        "complete_unique_identity_coverage": True,
        "no_duplicate_optimizer_parameters": True,
        "muon_metadata_exact": True,
        "tied_parameter_aliases": [list(names) for names in identity_aliases],
        "model_config": expected_model_config,
        "optimizer_groups": group_records,
        "parameter_inventory": parameter_inventory,
    }


def _model_and_optimizer(
    *,
    nanogpt_root: Path,
    muon_source: Path,
    backend: str,
) -> tuple[
    torch.nn.Module,
    torch.optim.Optimizer,
    ObservedMuonStep,
    dict[str, int],
    dict[str, object],
]:
    model_module = _load_module("p22_pinned_nanogpt_model", nanogpt_root / "model.py")
    muon_module = _load_module("p22_pinned_kellerjordan_muon", muon_source)
    config = model_module.GPTConfig(
        block_size=MODEL_BLOCK_SIZE,
        vocab_size=VOCAB_SIZE,
        n_layer=MODEL_N_LAYER,
        n_head=MODEL_N_HEAD,
        n_embd=MODEL_N_EMBD,
        dropout=MODEL_DROPOUT,
        bias=MODEL_BIAS,
    )
    model = model_module.GPT(config)
    # Keep the frozen CPU initialization, but complete the only device move
    # before collecting Parameter objects or constructing either optimizer.
    model.to(device=torch.device(backend), dtype=torch.float32)

    muon_params: list[Tensor] = []
    auxiliary_params: list[Tensor] = []
    metadata: dict[int, MuonParameterMetadata] = {}
    intended_numel: dict[str, int] = {}
    seen_parameters: set[int] = set()
    for name, parameter in model.named_parameters(remove_duplicate=False):
        identity = id(parameter)
        if identity in seen_parameters:
            continue
        seen_parameters.add(identity)
        match = MUON_NAME.fullmatch(name)
        if match is None:
            auxiliary_params.append(parameter)
            continue
        layer = int(match.group("layer"))
        role = ROLE_NAMES[match.group("role")]
        muon_params.append(parameter)
        metadata[id(parameter)] = MuonParameterMetadata(name=name, layer=layer, role=role)
        intended_numel[name] = parameter.numel()

    if len(muon_params) != 48:
        raise P22RunError(f"found {len(muon_params)} Muon tensors, expected 48")
    if sum(intended_numel.values()) != 84_934_656:
        raise P22RunError("Muon tensor inventory has an unexpected element count")
    groups = [
        {
            "params": auxiliary_params,
            "lr": AUXILIARY_LEARNING_RATE,
            "betas": AUXILIARY_BETAS,
            "eps": AUXILIARY_EPSILON,
            "weight_decay": AUXILIARY_WEIGHT_DECAY,
            "use_muon": False,
        },
        {
            "params": muon_params,
            "lr": MUON_LEARNING_RATE,
            "momentum": MUON_MOMENTUM,
            "weight_decay": MUON_WEIGHT_DECAY,
            "use_muon": True,
        },
    ]
    optimizer = muon_module.SingleDeviceMuonWithAuxAdam(groups)
    observed_step = ObservedMuonStep(
        muon_module=muon_module,
        optimizer=optimizer,
        parameter_metadata=metadata,
    )
    binding_audit = _audit_model_optimizer_binding(
        model=model,
        optimizer=optimizer,
        parameter_metadata=metadata,
        backend=backend,
    )
    return model, optimizer, observed_step, intended_numel, binding_audit


def _load_train_tokens(manifest_path: Path) -> tuple[np.memmap, dict[str, Any]]:
    validation = validate_fineweb_manifest(manifest_path)
    if not validation["ready"]:
        raise P22RunError(f"FineWeb manifest failed validation: {validation['blockers']}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = payload["outputs"]["train.bin"]
    token_path = Path(entry["path"])
    if not token_path.is_absolute():
        token_path = manifest_path.parent / token_path
    tokens = np.memmap(token_path.resolve(), dtype="<u2", mode="r")
    if tokens.size < EXPECTED_TRAINING_TOKENS_CONSUMED + 1:
        raise P22RunError("train.bin lacks the shifted target token")
    return tokens, validation


def _batch(tokens: np.memmap, step: int, backend: str) -> tuple[Tensor, Tensor, int]:
    start = step * SEQUENCE_LENGTH
    stop = start + SEQUENCE_LENGTH
    x_host = torch.from_numpy(np.asarray(tokens[start:stop], dtype=np.int64).copy()).view(1, -1)
    y_host = torch.from_numpy(np.asarray(tokens[start + 1 : stop + 1], dtype=np.int64).copy()).view(
        1, -1
    )
    if tuple(x_host.shape) != (MICRO_BATCH_SIZE, SEQUENCE_LENGTH):
        raise P22RunError("frozen sequential batch is incomplete")
    return x_host.to(backend), y_host.to(backend), start


def _state_hashes(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    backend: str,
) -> dict[str, str]:
    _synchronize(backend)
    return {
        "model_state_sha256": canonical_state_sha256(model.state_dict()),
        "optimizer_state_sha256": canonical_state_sha256(optimizer.state_dict()),
    }


def _source_snapshot(
    *,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
) -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    paths = {
        "nanogpt_model.py": nanogpt_root / "model.py",
        "nanogpt_train.py": nanogpt_root / "train.py",
        "muon.py": muon_source,
        "p22_protocol": root / "experiments/training/p22_real_gradient_shadow_trace_protocol.json",
        "p22_trainer": Path(__file__).resolve(),
        "p22_observer_adapter": root / "experiments/training/p22_observed_muon.py",
        "p22_observer": root / "src/passive_muon/p22_real_gradient_shadow_trace.py",
        "p22_shape_extension": root / "src/passive_muon/p22_scalable_sector_shield_extension.py",
        "fineweb_manifest": data_manifest,
        "pyproject.toml": root / "pyproject.toml",
        "uv.lock": root / "uv.lock",
    }
    return {name: sha256_file(path.resolve()) for name, path in paths.items()}


def _run_identity(source_snapshot: Mapping[str, str], data_sha256: str, backend: str) -> str:
    return canonical_state_sha256(
        {
            "protocol_sha256": protocol_sha256(),
            "source_snapshot": dict(source_snapshot),
            "data_manifest_sha256": data_sha256,
            "backend": backend,
            "seed": SEED,
        }
    )


def _runtime_provenance(backend: str) -> dict[str, Any]:
    accelerator: dict[str, Any] = {
        "backend": backend,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
    }
    if backend == "cuda":
        properties = torch.cuda.get_device_properties(0)
        accelerator.update(
            {
                "name": properties.name,
                "total_memory": properties.total_memory,
                "compute_capability": [properties.major, properties.minor],
            }
        )
    else:
        accelerator.update(
            {
                "name": "Apple MPS device",
                "scope": "BF16-stored Apple-GPU shadow evidence; not CUDA or kernel parity",
            }
        )
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "torch_version": torch.__version__,
        "torch_git_version": torch.version.git_version,
        "torch_config": torch.__config__.show(),
        "numpy_version": np.__version__,
        "accelerator": accelerator,
        "environment": {
            "PYTORCH_ENABLE_MPS_FALLBACK": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
            "PYTORCH_MPS_FAST_MATH": os.environ.get("PYTORCH_MPS_FAST_MATH"),
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "arithmetic": {
            "model_parameters": "FP32",
            "forward_backward": "FP32; no autocast or scaler",
            "gradient_and_momentum": "FP32",
            "muon_newton_schulz_state": "BF16 on selected accelerator",
            "shield": "detached CPU P20 proof-reference plus P22 shape extension",
        },
    }


def run_trace(
    *,
    trace_mode: str,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
    instrumentation_patch: Path,
    backend: str,
    output: Path,
    raw_trace_output: Path | None,
    _lifecycle_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Execute one fresh trace-off or trace-on process."""

    if trace_mode not in {"trace_off", "trace_on"}:
        raise P22RunError("trace_mode must be trace_off or trace_on")
    if trace_mode == "trace_on" and raw_trace_output is None:
        raise P22RunError("trace_on requires --raw-trace-output")
    if trace_mode == "trace_off" and raw_trace_output is not None:
        raise P22RunError("trace_off forbids --raw-trace-output because no candidate is captured")
    expected_instrumentation = SCRIPT_DIR / "p22_observed_muon.py"
    if instrumentation_patch.resolve() != expected_instrumentation.resolve():
        raise P22RunError("--instrumentation-patch must be the frozen P22 observation adapter")
    preflight = build_preflight_report(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        fineweb_manifest=data_manifest,
        instrumentation_patch=instrumentation_patch,
        accelerator_backend=backend,
    )
    if preflight["status"] != "ready":
        raise P22RunError(f"P22 preflight is blocked: {preflight['blockers']}")

    _set_determinism(backend)
    tokens, data_validation = _load_train_tokens(data_manifest)
    model, optimizer, observed_step, intended_numel, binding_audit = _model_and_optimizer(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        backend=backend,
    )
    if _lifecycle_callback is not None:
        _lifecycle_callback("initialized")
    model.train()
    initial_hashes = _state_hashes(model, optimizer, backend=backend)
    initial_rng = canonical_state_sha256(capture_rng_state(backend))
    initial_state_sha256 = canonical_state_sha256({**initial_hashes, "rng": initial_rng})
    source_snapshot = _source_snapshot(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        data_manifest=data_manifest,
    )
    run_identity = _run_identity(
        source_snapshot,
        str(data_validation["manifest_sha256"]),
        backend,
    )

    observations: list[dict[str, object]] = []
    steps: list[dict[str, Any]] = []
    started = time.monotonic()
    for step in range(EXPECTED_OPTIMIZER_STEPS):
        x, y, token_offset = _batch(tokens, step, backend)
        rng_before = canonical_state_sha256(capture_rng_state(backend))
        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        if loss is None or not bool(torch.isfinite(loss)):
            raise P22RunError(f"nonfinite or missing loss at step {step}")
        loss.backward()
        del x, y

        capture = trace_mode == "trace_on" and step in CAPTURE_STEPS
        observation_start = len(observations)
        current_step = step

        def hook(
            metadata: MuonParameterMetadata,
            signal: Tensor,
            candidate: Tensor,
            capture_step: int = current_step,
        ) -> None:
            if signal.device.type != backend or candidate.device.type != backend:
                raise P22RunError("capture did not remain on the selected accelerator")
            if signal.dtype != torch.float32 or candidate.dtype != torch.bfloat16:
                raise P22RunError(
                    "stored signal/candidate dtypes are not FP32/BF16 at capture boundary"
                )
            _synchronize(backend)
            signal_cpu = signal.to(device="cpu", dtype=torch.float32).contiguous()
            candidate_cpu = candidate.to(device="cpu").contiguous()

            def observe() -> dict[str, object]:
                return observe_post_aspect_candidate(
                    signal_cpu,
                    candidate_cpu,
                    optimizer_step=capture_step,
                    tokens_seen=(capture_step + 1) * SEQUENCE_LENGTH,
                    seed=SEED,
                    parameter_name=metadata.name,
                    layer=metadata.layer,
                    role=metadata.role,
                    learning_rate=MUON_LEARNING_RATE,
                    beta=MUON_MOMENTUM,
                    weight_decay=MUON_WEIGHT_DECAY,
                    accelerator_backend=backend,
                )

            record, rng_digest = run_rng_neutral_observer(
                observe,
                accelerator_backend=backend,
            )
            record["observer_rng_sha256"] = rng_digest
            observations.append(record)

        observed_step.step(hook=hook if capture else None)
        _synchronize(backend)
        rng_after = canonical_state_sha256(capture_rng_state(backend))
        state_checkpoint = (
            _state_hashes(model, optimizer, backend=backend)
            if step in STATE_CHECKPOINT_STEPS
            else {"model_state_sha256": None, "optimizer_state_sha256": None}
        )
        if capture and len(observations) - observation_start != observed_step.parameter_count:
            raise P22RunError("capture did not cover every Muon parameter")
        captured_records = observations[observation_start:]
        capture_record = (
            {
                "parameter_count": len(captured_records),
                "actual_pre_aspect_candidate": False,
                "pre_aspect_unavailable_reason": (
                    "the pinned muon_update call boundary returns only the post-aspect value"
                ),
                "actual_post_aspect_candidate": True,
                "actual_post_aspect_cuda_candidate": backend == "cuda",
                "p20_shadow_only": True,
                "record_sha256": canonical_state_sha256(captured_records),
            }
            if capture
            else None
        )
        loss_cpu = loss.detach().to(device="cpu").contiguous()
        batch_host = np.asarray(
            tokens[token_offset : token_offset + SEQUENCE_LENGTH + 1],
            dtype="<u2",
        ).copy()
        steps.append(
            {
                "step": step,
                "tokens_seen": (step + 1) * SEQUENCE_LENGTH,
                "data_token_offset": token_offset,
                "batch_sha256": hashlib.sha256(batch_host.tobytes(order="C")).hexdigest(),
                "loss_tensor_sha256": _tensor_sha256(loss_cpu),
                "loss_float64": float(loss_cpu),
                **state_checkpoint,
                "state_checkpoint_present": step in STATE_CHECKPOINT_STEPS,
                "rng_before_step_sha256": rng_before,
                "rng_after_step_sha256": rng_after,
                "observer_rng_unchanged": True if capture else None,
                "observation_count": len(observations) - observation_start,
                "capture": capture_record,
            }
        )
        del loss, loss_cpu

    expected_observations = EXPECTED_CANDIDATE_OBSERVATIONS if trace_mode == "trace_on" else 0
    if len(observations) != expected_observations:
        raise P22RunError(
            f"captured {len(observations)} candidate observations, expected {expected_observations}"
        )

    final_hashes = _state_hashes(model, optimizer, backend=backend)
    final_rng = canonical_state_sha256(capture_rng_state(backend))
    if _lifecycle_callback is not None:
        _lifecycle_callback("completed")
    elapsed = time.monotonic() - started

    raw_trace: dict[str, Any] | None = None
    if trace_mode == "trace_on":
        assert raw_trace_output is not None
        raw_trace = {
            "schema_version": "passive-muon-p22-raw-real-gradient-trace-v1",
            "evidence_kind": "real_gradient_shadow_trace",
            "protocol_sha256": protocol_sha256(),
            "run_identity_sha256": run_identity,
            "capture_steps": list(CAPTURE_STEPS),
            "observation_count": len(observations),
            "observations": observations,
        }
        write_json_atomic(raw_trace_output, raw_trace)

    payload: dict[str, Any] = {
        "schema_version": P22_RUN_MANIFEST_SCHEMA,
        "trace_mode": trace_mode,
        "evidence_kind": "real_gradient_shadow_trace",
        "protocol_sha256": protocol_sha256(),
        "run_identity_sha256": run_identity,
        "initial_state_sha256": initial_state_sha256,
        "initial_state": {**initial_hashes, "rng_state_sha256": initial_rng},
        "accelerator_backend": backend,
        "candidate_execution": {
            "backend": backend,
            "actual_accelerator_candidate_computed_and_applied": True,
            "shield_shadow_only": True,
        },
        "candidate_observation": (
            {
                "status": f"observed_actual_post_aspect_{backend}",
                "capture_step_count": len(CAPTURE_STEPS),
                "observation_count": len(observations),
            }
            if trace_mode == "trace_on"
            else {
                "status": "not_observed",
                "capture_step_count": 0,
                "observation_count": 0,
            }
        ),
        "shadow_update_applied": False,
        "capture_steps": list(CAPTURE_STEPS) if trace_mode == "trace_on" else [],
        "steps": steps,
        "final_state": {**final_hashes, "rng_state_sha256": final_rng},
        "state_hash_schedule": sorted(STATE_CHECKPOINT_STEPS),
        "raw_trace": (
            {
                "path": str(raw_trace_output.resolve()),
                "sha256": sha256_file(raw_trace_output.resolve()),
                "observation_count": len(observations),
            }
            if raw_trace_output is not None
            else None
        ),
        "shape_inventory": {
            "parameter_count": len(intended_numel),
            "total_numel": sum(intended_numel.values()),
            "parameters": intended_numel,
            "stored_shapes": {
                str(record["name"]): record["shape"]
                for record in binding_audit["parameter_inventory"]
                if record["use_muon"] is True
            },
        },
        "model_optimizer_binding": binding_audit,
        "data": data_validation,
        "source_snapshot": source_snapshot,
        "runtime": _runtime_provenance(backend),
        "elapsed_seconds": elapsed,
        "claim_boundary": (
            "Pinned one-seed Apple-MPS or CUDA real-gradient shadow evidence only; "
            "the shield output was not applied and this is not a neural-loss theorem."
        ),
    }
    write_json_atomic(output, payload)
    return payload


def aggregate_trace(
    *,
    run_manifest: Path,
    output: Path,
) -> dict[str, object]:
    manifest = json.loads(run_manifest.read_text(encoding="utf-8"))
    if manifest.get("trace_mode") != "trace_on":
        raise P22RunError("aggregate requires a trace_on manifest")
    raw_entry = manifest.get("raw_trace")
    if not isinstance(raw_entry, dict):
        raise P22RunError("trace_on manifest has no raw-trace entry")
    raw_path = Path(str(raw_entry["path"]))
    if sha256_file(raw_path) != raw_entry.get("sha256"):
        raise P22RunError("raw trace hash mismatch")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    intended = manifest["shape_inventory"]["parameters"]
    summary = summarize_shadow_trace(
        raw["observations"],
        evidence_kind="real_gradient_shadow_trace",
        provenance={
            "p22_protocol_sha256": protocol_sha256(),
            "run_manifest_path": str(run_manifest.resolve()),
            "run_manifest_sha256": sha256_file(run_manifest),
            "raw_trace_path": str(raw_path.resolve()),
            "raw_trace_sha256": sha256_file(raw_path),
            "source_snapshot": manifest["source_snapshot"],
            "runtime": manifest["runtime"],
            "data_manifest_sha256": manifest["data"]["manifest_sha256"],
            "post_aspect_candidate_captured_from_accelerator": True,
            "aspect_recomputed_by_cpu_observer": False,
        },
        intended_parameter_numel={str(key): int(value) for key, value in intended.items()},
    )
    summary["p22_extension"] = {
        "schema_version": "passive-muon-p22-real-gradient-summary-v1",
        "post_aspect_candidate_captured_from_accelerator": True,
        "shape_768x2304_separately_certified": True,
        "early_middle_late_are_step_windows_not_mature_training_phases": True,
    }
    write_json_atomic(output, summary)
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run")
    run.add_argument("--trace-mode", choices=("trace_off", "trace_on"), required=True)
    run.add_argument("--nanogpt-root", type=Path, required=True)
    run.add_argument("--muon-source", type=Path, required=True)
    run.add_argument("--fineweb-manifest", type=Path, required=True)
    run.add_argument("--instrumentation-patch", type=Path, required=True)
    run.add_argument("--accelerator", choices=("cuda", "mps"), required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--raw-trace-output", type=Path)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--run-manifest", type=Path, required=True)
    aggregate.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.command == "run":
        payload = run_trace(
            trace_mode=args.trace_mode,
            nanogpt_root=args.nanogpt_root.resolve(),
            muon_source=args.muon_source.resolve(),
            data_manifest=args.fineweb_manifest.resolve(),
            instrumentation_patch=args.instrumentation_patch.resolve(),
            backend=args.accelerator,
            output=args.output.resolve(),
            raw_trace_output=(
                args.raw_trace_output.resolve() if args.raw_trace_output is not None else None
            ),
        )
    else:
        payload = aggregate_trace(
            run_manifest=args.run_manifest.resolve(),
            output=args.output.resolve(),
        )
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
