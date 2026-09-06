#!/usr/bin/env python3
"""Execute and independently verify the frozen P23 CUDA shadow trace.

This is a thin orchestration layer.  Training and observation remain in the
hardened P22 runner; P23 adds the CUDA runtime lock, complete execution
identity, late-import guard, ordered acquisition gates, and sanitization.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import secrets
import sys
import tempfile
import time
from collections.abc import Mapping
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

SCRIPT_DIR: Final = Path(__file__).resolve().parent
REPOSITORY_ROOT: Final = SCRIPT_DIR.parents[1]
REPOSITORY_SOURCE_ROOT: Final = REPOSITORY_ROOT / "src"
_PINNED_PYTHON_MAJOR_MINOR: Final = (3, 12)
_PINNED_EXECUTABLE_PATH: Final = (
    "/opt/p23-venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
)
_FROZEN_PREIMPORT_PYTHON_FLAGS: Final = {
    "debug": 0,
    "inspect": 0,
    "interactive": 0,
    "optimize": 0,
    "dont_write_bytecode": 1,
    "no_user_site": 1,
    "no_site": 0,
    "ignore_environment": 0,
    "verbose": 0,
    "bytes_warning": 0,
    "quiet": 0,
    "isolated": 0,
    "hash_randomization": 1,
    "dev_mode": False,
    "utf8_mode": 0,
    "warn_default_encoding": 0,
    "safe_path": False,
    "int_max_str_digits": 4300,
}


def _preimport_python_flags() -> dict[str, object]:
    return {name: getattr(sys.flags, name, None) for name in _FROZEN_PREIMPORT_PYTHON_FLAGS}


def _assert_pre_torch_process_contract() -> None:
    """Reject a non-image interpreter or shadowing environment before Torch import."""

    expected = {
        "PATH": _PINNED_EXECUTABLE_PATH,
        "VIRTUAL_ENV": "/opt/p23-venv",
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONOPTIMIZE": None,
        "PYTHONPATH": None,
        "PYTHONHOME": None,
        "LD_PRELOAD": None,
        "LD_LIBRARY_PATH": None,
        "LD_AUDIT": None,
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "PYTHONHASHSEED": "1337",
        "NVIDIA_TF32_OVERRIDE": "0",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    if sys.executable != "/opt/p23-venv/bin/python":
        raise RuntimeError("P23 must use /opt/p23-venv/bin/python before importing Torch")
    if tuple(sys.version_info[:2]) != _PINNED_PYTHON_MAJOR_MINOR:
        raise RuntimeError(
            "P23 requires Python 3.12 before importing Torch; "
            f"observed {sys.version_info[0]}.{sys.version_info[1]}"
        )
    mismatches = {
        name: {"expected": value, "observed": os.environ.get(name)}
        for name, value in expected.items()
        if os.environ.get(name) != value
    }
    if mismatches:
        raise RuntimeError(f"P23 process-start environment mismatch: {mismatches}")
    observed_flags = _preimport_python_flags()
    if any(
        type(observed_flags[name]) is not type(expected) or observed_flags[name] != expected
        for name, expected in _FROZEN_PREIMPORT_PYTHON_FLAGS.items()
    ):
        raise RuntimeError(f"P23 Python interpreter flags mismatch: {observed_flags}")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    full_gpu_uuid = re.compile(
        r"^GPU-[0-9A-Fa-f]{8}-[0-9A-Fa-f]{4}-[0-9A-Fa-f]{4}-"
        r"[0-9A-Fa-f]{4}-[0-9A-Fa-f]{12}$"
    )
    if visible is None or full_gpu_uuid.fullmatch(visible) is None:
        raise RuntimeError("P23 requires a canonical full GPU UUID before importing Torch")
    if os.environ.get("NVIDIA_VISIBLE_DEVICES") != visible:
        raise RuntimeError("P23 visible-device variables differ before importing Torch")
    if (REPOSITORY_ROOT / ".venv").exists() or (REPOSITORY_ROOT / ".venv").is_symlink():
        raise RuntimeError("P23 forbids a repository-local .venv")
    raw_roots = {
        "repository": os.environ.get("P23_REPO"),
        "nanogpt": os.environ.get("P23_NANOGPT"),
        "muon": os.environ.get("P23_MUON_ROOT"),
    }
    if any(value is None or not Path(value).is_absolute() for value in raw_roots.values()):
        raise RuntimeError("P23 controlled import roots must be absolute process-start paths")
    roots = {name: Path(str(value)).resolve() for name, value in raw_roots.items()}
    if roots["repository"] != REPOSITORY_ROOT.resolve():
        raise RuntimeError("P23_REPO does not identify the executing repository")
    for name, root in roots.items():
        if not root.is_dir():
            raise RuntimeError(f"P23 controlled import root {name} is unavailable: {root}")
        forbidden = sorted(
            str(path.relative_to(root))
            for path in root.rglob("*")
            if path.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}
        )
        if forbidden:
            raise RuntimeError(
                f"P23 controlled import root {name} contains forbidden bytecode: {forbidden[:8]}"
            )


if __name__ == "__main__":
    _assert_pre_torch_process_contract()

for import_root in (REPOSITORY_SOURCE_ROOT, SCRIPT_DIR):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import p22_nanogpt_shadow_trace as P22_CORE  # noqa: E402
import p23_deterministic_cuda_shadow_trace as P23  # noqa: E402
import run_p22_real_gradient_shadow_trace as P22_RUNNER  # noqa: E402

from passive_muon import p21_shadow_trace as P21_TRACE  # noqa: E402
from passive_muon import p22_real_gradient_shadow_trace as P22_OBSERVER  # noqa: E402

P23_STATIC_CONTRACT_SCHEMA: Final = "passive-muon-p23-static-execution-contract-v1"
P23_AGGREGATE_SCHEMA: Final = "passive-muon-p23-cuda-shadow-trace-aggregate-v1"
P23_GATE_REPORT_BINDING_SCHEMA: Final = "passive-muon-p23-gate-report-binding-v1"
P23_PROCESS_INSTANCE_SCHEMA: Final = "passive-muon-p23-process-instance-v1"
P23_P22_ACQUISITION_BINDING_SCHEMA: Final = "passive-muon-p23-p22-acquisition-binding-v1"
RUN_ROLES: Final = ("trace_off_a", "trace_off_b", "trace_on")
_SHA256: Final = re.compile(r"^[0-9a-f]{64}$")
METRIC_RECONSTRUCTION_SCOPE: Final = (
    "P23 validates the exact observation schema, frozen metadata, tensor-hash formats, "
    "and every scalar/gate relation derivable from retained scalar fields. The retained "
    "trace does not contain tensor bytes, so base norms, inner products, and tensor hashes "
    "remain code-produced observations rather than independently tensor-recomputed values."
)
_PROCESS_INSTANCE: Final = {
    "schema_version": P23_PROCESS_INSTANCE_SCHEMA,
    "nonce": secrets.token_hex(32),
    "pid": os.getpid(),
    "created_time_ns": time.time_ns(),
}


class P23AcquisitionError(RuntimeError):
    """Raised when orchestration cannot honor the frozen P23 acquisition."""


def _validate_frozen_fidelity_gate_constants(
    *, addendum: Mapping[str, object] | None = None
) -> None:
    """Bind every live P21/P23 fidelity threshold to the frozen protocol JSON."""

    if addendum is None:
        addendum = P23.load_and_validate_addendum(
            P23.DEFAULT_ADDENDUM_PATH,
            repository_root=REPOSITORY_ROOT,
        )
    if not isinstance(addendum, Mapping):
        raise P23AcquisitionError("P23 addendum has no frozen P21 fidelity binding")
    inherits = addendum.get("inherits")
    if not isinstance(inherits, Mapping):
        raise P23AcquisitionError("P23 addendum has no inherited-artifact inventory")
    p21_binding = inherits.get("p21_fidelity_gates")
    if not isinstance(p21_binding, Mapping):
        raise P23AcquisitionError("P23 addendum does not bind the P21 fidelity protocol")
    expected_relative_path = P21_TRACE.P21_PROTOCOL_RELATIVE_PATH.as_posix()
    if p21_binding.get("path") != expected_relative_path:
        raise P23AcquisitionError("P21 fidelity protocol path differs from the P23 addendum")
    if P21_TRACE.frozen_protocol_path().resolve() != (
        REPOSITORY_ROOT / expected_relative_path
    ).resolve() or P21_TRACE.frozen_protocol_sha256() != p21_binding.get("sha256"):
        raise P23AcquisitionError("P21 fidelity protocol bytes differ from the P23 addendum")

    try:
        protocol = P21_TRACE.load_frozen_protocol()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise P23AcquisitionError(f"cannot validate frozen P21 fidelity gates: {error}") from error
    frozen_gates = protocol.get("frozen_gates")
    expected_gates = {
        "hard": {
            "all_primary_eta_effective_gains_in_1_over_1000_to_1_over_80": True,
            "all_returned_output_to_signal_amplitudes_in_1_over_10_to_1": True,
            "all_supported_events_are_successful_p20_sector_calls": True,
            "capture_schedule_complete": True,
            "intended_muon_matrix_coverage_count_and_numel": (
                "100%; otherwise the trace is explicitly partial"
            ),
            "nonfinite_candidate_count": 0,
            "nonfinite_signal_count": 0,
            "unexpected_shield_failure_count": 0,
            "unrepresentable_all_subnormal_dead_zone_count": 0,
        },
        "mild_intervention": {
            "candidate_output_cosine_minimum": "4/5",
            "candidate_output_cosine_p05_nearest_rank": "19/20",
            "maximum_activation_rate_each_phase_role_cell_with_n_at_least_20": "1/2",
            "maximum_overall_activation_rate": "1/4",
            "output_to_candidate_amplitude_p05_nearest_rank": "3/4",
            "output_to_candidate_amplitude_p95_nearest_rank": "5/4",
            "relative_correction_maximum": "1",
            "relative_correction_p50_nearest_rank": "1/20",
            "relative_correction_p95_nearest_rank": "1/4",
        },
        "p16_p18_reused": {
            "annulus": "3/4 <= ||S||_F <= 25",
            "annulus_output_to_aspect_scaled_upstream_candidate": (
                "1/4 <= ||T||_F/||C||_F <= 8, with at least one observation in every phase"
            ),
            "informative_candidate_best_scalar_departure": ">= 1/100",
            "output_best_scalar_departure": ">= 1/1000 on every informative event",
            "shaping_retention": ">= 1/10 on every informative event",
        },
    }
    if frozen_gates != expected_gates:
        raise P23AcquisitionError("hash-frozen P21 fidelity-gate table changed")
    metric_semantics = protocol.get("metric_semantics")
    if not isinstance(metric_semantics, Mapping) or metric_semantics.get("effective_gain") != (
        "eta*||T||_F/||S||_F, reported for eta=1/120 and eta=1/83."
    ):
        raise P23AcquisitionError("hash-frozen P21 effective-gain operating points changed")

    exact_constants: dict[str, Fraction | int] = {
        "PRIMARY_LEARNING_RATE": Fraction(1, 120),
        "SECONDARY_LEARNING_RATE": Fraction(1, 83),
        "ANNULUS_SIGNAL_NORM_LOWER": Fraction(3, 4),
        "ANNULUS_SIGNAL_NORM_UPPER": Fraction(25),
        "ANNULUS_OUTPUT_TO_CANDIDATE_LOWER": Fraction(1, 4),
        "ANNULUS_OUTPUT_TO_CANDIDATE_UPPER": Fraction(8),
        "OUTPUT_TO_SIGNAL_LOWER": Fraction(1, 10),
        "OUTPUT_TO_SIGNAL_UPPER": Fraction(1),
        "PRIMARY_EFFECTIVE_GAIN_LOWER": Fraction(1, 1_000),
        "PRIMARY_EFFECTIVE_GAIN_UPPER": Fraction(1, 80),
        "MAXIMUM_OVERALL_ACTIVATION_RATE": Fraction(1, 4),
        "MAXIMUM_CELL_ACTIVATION_RATE": Fraction(1, 2),
        "MINIMUM_CELL_COUNT": 20,
        "MAXIMUM_MEDIAN_RELATIVE_CORRECTION": Fraction(1, 20),
        "MAXIMUM_P95_RELATIVE_CORRECTION": Fraction(1, 4),
        "MAXIMUM_RELATIVE_CORRECTION": Fraction(1),
        "MINIMUM_P05_COSINE": Fraction(19, 20),
        "MINIMUM_COSINE": Fraction(4, 5),
        "MINIMUM_P05_OUTPUT_TO_CANDIDATE": Fraction(3, 4),
        "MAXIMUM_P95_OUTPUT_TO_CANDIDATE": Fraction(5, 4),
    }
    float_constants = {
        "INFORMATIVE_CANDIDATE_DEPARTURE": Fraction(1, 100),
        "MINIMUM_OUTPUT_DEPARTURE": Fraction(1, 1_000),
        "MINIMUM_SHAPING_RETENTION": Fraction(1, 10),
    }
    for name, expected in exact_constants.items():
        actual = getattr(P21_TRACE, name, None)
        expected_type = int if isinstance(expected, int) else Fraction
        if type(actual) is not expected_type or actual != expected:
            raise P23AcquisitionError(
                f"live P21 fidelity constant {name} differs from the frozen protocol"
            )
    for name, expected in float_constants.items():
        actual = getattr(P21_TRACE, name, None)
        if type(actual) is not float or actual != float(expected):
            raise P23AcquisitionError(
                f"live P21 fidelity constant {name} differs from the frozen protocol"
            )


def _validate_frozen_p22_acquisition_constants(
    *,
    addendum: Mapping[str, object] | None = None,
    static_contract: Mapping[str, object] | None = None,
) -> None:
    """Bind live P22 trainer/acquisition semantics to its frozen protocol."""

    capture_steps = (
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        124,
        125,
        126,
        127,
        128,
        129,
        130,
        131,
        248,
        249,
        250,
        251,
        252,
        253,
        254,
        255,
    )
    checkpoint_steps = frozenset(capture_steps)
    runner_constants: dict[str, object] = {
        "SEED": 1337,
        "SEQUENCE_LENGTH": 128,
        "MICRO_BATCH_SIZE": 1,
        "GRADIENT_ACCUMULATION_STEPS": 1,
        "MODEL_BLOCK_SIZE": 1024,
        "VOCAB_SIZE": 50_257,
        "MODEL_N_LAYER": 12,
        "MODEL_N_HEAD": 12,
        "MODEL_N_EMBD": 768,
        "MODEL_DROPOUT": 0.0,
        "MODEL_BIAS": False,
        "MUON_LEARNING_RATE": float(Fraction(1, 120)),
        "MUON_MOMENTUM": float(Fraction(19, 20)),
        "MUON_WEIGHT_DECAY": 0.0,
        "AUXILIARY_LEARNING_RATE": float(Fraction(3, 5_000)),
        "AUXILIARY_BETAS": (0.9, 0.95),
        "AUXILIARY_EPSILON": 1.0e-10,
        "AUXILIARY_WEIGHT_DECAY": 0.0,
        "CAPTURE_STEPS": capture_steps,
        "EXPECTED_OPTIMIZER_STEPS": 256,
        "EXPECTED_CANDIDATE_OBSERVATIONS": 1_152,
        "EXPECTED_TRAINING_TOKENS_CONSUMED": 32_768,
        "STATE_CHECKPOINT_STEPS": checkpoint_steps,
        "P22_RUN_MANIFEST_SCHEMA": "passive-muon-p22-run-manifest-v2",
        "ROLE_NAMES": {
            "attn.c_attn": "fused_qkv",
            "attn.c_proj": "attention_output",
            "mlp.c_fc": "mlp_expand",
            "mlp.c_proj": "mlp_contract",
        },
        "EXPECTED_TIED_ALIAS_INVENTORY": (("lm_head.weight", "transformer.wte.weight"),),
    }
    core_constants: dict[str, object] = {
        "P22_PROTOCOL_SCHEMA": "passive-muon-p22-real-gradient-shadow-trace-v1",
        "P22_RUN_MANIFEST_SCHEMA": "passive-muon-p22-run-manifest-v2",
        "NANOGPT_REVISION": "3adf61e154c3fe3fca428ad6bc3818b27a3b8291",
        "NANOGPT_TREE": "ca93bcd9b9c9ff32d3016e1e2556644e68bef86a",
        "NANOGPT_TRAIN_BLOB": "de5785051050d64bbda37a4964c2c079f0739b2b",
        "NANOGPT_MODEL_BLOB": "c698f8b60129d793494de058b0dd4e318c0dcb5e",
        "MUON_REVISION": "f98f1cacc0263b04290753e32be8d498c1efc806",
        "MUON_SOURCE_SHA256": ("2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"),
        "EXPECTED_OPTIMIZER_STEPS": 256,
        "EXPECTED_CAPTURE_STEPS": capture_steps,
        "EXPECTED_CANDIDATE_OBSERVATIONS": 1_152,
        "EXPECTED_MODEL_PARAMETER_COUNT": 75,
        "EXPECTED_MUON_PARAMETER_COUNT": 48,
        "EXPECTED_AUXILIARY_PARAMETER_COUNT": 27,
        "EXPECTED_TIED_PARAMETER_ALIASES": [["lm_head.weight", "transformer.wte.weight"]],
        "EXPECTED_TRAINING_TOKENS_CONSUMED": 32_768,
        "EXPECTED_SEQUENCE_LENGTH": 128,
        "EXPECTED_TRAIN_POOL_TOKENS": 131_072,
        "EXPECTED_VALIDATION_TOKENS": 32_768,
        "EXPECTED_TRAIN_BYTES": 262_144,
        "EXPECTED_VALIDATION_BYTES": 65_536,
        "PROTOCOL_PATH": Path("experiments/training/p22_real_gradient_shadow_trace_protocol.json"),
        "DATA_TEMPLATE_PATH": Path("experiments/training/p22_fineweb_manifest.template.json"),
        "NONINTERFERENCE_STEP_FIELDS": (
            "tokens_seen",
            "data_token_offset",
            "batch_sha256",
            "loss_tensor_sha256",
            "rng_before_step_sha256",
            "rng_after_step_sha256",
        ),
        "NONINTERFERENCE_STATE_FIELDS": (
            "model_state_sha256",
            "optimizer_state_sha256",
        ),
        "STATE_CHECKPOINT_STEPS": checkpoint_steps,
        "NONINTERFERENCE_FINAL_FIELDS": (
            "model_state_sha256",
            "optimizer_state_sha256",
            "rng_state_sha256",
        ),
    }
    for owner_label, owner, constants in (
        ("P22 trainer", P22_RUNNER, runner_constants),
        ("P22 protocol module", P22_CORE, core_constants),
    ):
        for name, expected in constants.items():
            actual = getattr(owner, name, None)
            if type(actual) is not type(expected) or actual != expected:
                raise P23AcquisitionError(
                    f"live {owner_label} constant {name} differs from the frozen P22 contract"
                )
    expected_muon_name = re.compile(
        r"^transformer\.h\.(?P<layer>[0-9]+)\."
        r"(?P<role>attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$"
    )
    actual_muon_name = getattr(P22_RUNNER, "MUON_NAME", None)
    if not isinstance(actual_muon_name, re.Pattern) or (
        actual_muon_name.pattern,
        actual_muon_name.flags,
    ) != (expected_muon_name.pattern, expected_muon_name.flags):
        raise P23AcquisitionError(
            "live P22 trainer constant MUON_NAME differs from the frozen P22 contract"
        )
    if P22_OBSERVER.P22_SHADOW_OBSERVATION_SCHEMA_VERSION != (
        "passive-muon-p22-post-aspect-shadow-observation-v1"
    ):
        raise P23AcquisitionError(
            "live P22 observer constant P22_SHADOW_OBSERVATION_SCHEMA_VERSION differs "
            "from the frozen P22 contract"
        )

    if addendum is None:
        addendum = P23.load_and_validate_addendum(
            P23.DEFAULT_ADDENDUM_PATH,
            repository_root=REPOSITORY_ROOT,
        )
    if not isinstance(addendum, Mapping):
        raise P23AcquisitionError("P23 addendum has no frozen P22 protocol binding")
    inherits = addendum.get("inherits")
    p22_binding = inherits.get("p22_protocol") if isinstance(inherits, Mapping) else None
    expected_relative_path = P22_CORE.PROTOCOL_PATH.as_posix()
    if not isinstance(p22_binding, Mapping) or p22_binding.get("path") != expected_relative_path:
        raise P23AcquisitionError("P23 addendum does not bind the P22 acquisition protocol")
    if P22_CORE.protocol_path().resolve() != (
        REPOSITORY_ROOT / expected_relative_path
    ).resolve() or P22_CORE.protocol_sha256() != p22_binding.get("sha256"):
        raise P23AcquisitionError("P22 protocol bytes differ from the P23 addendum")
    try:
        protocol = P22_CORE.load_protocol()
    except (OSError, ValueError, TypeError, json.JSONDecodeError) as error:
        raise P23AcquisitionError(f"cannot validate frozen P22 semantics: {error}") from error

    expected_protocol_fields: dict[str, dict[str, object]] = {
        "run": {
            "seed": 1337,
            "seed_count": 1,
            "device_count": 1,
            "accelerator_backend": (
                "explicitly selected cuda or mps; never cpu or automatic fallback"
            ),
            "distributed": False,
            "total_optimizer_steps": 256,
            "capture_steps": list(capture_steps),
            "capture_count": 24,
            "micro_batch_size": 1,
            "sequence_length": 128,
            "gradient_accumulation_steps": 1,
            "tokens_per_optimizer_step": 128,
            "total_training_tokens": 32_768,
            "initialization": "from_scratch",
            "evaluation_during_run": False,
            "checkpointing_during_run": False,
            "compile": False,
            "gradient_clip": 0,
            "deterministic_algorithms": True,
            "tf32": False,
            "autocast": (
                "disabled; model forward/backward, parameters, gradients, EMA and Nesterov "
                "remain FP32; only the pinned five Newton-Schulz stages use BF16"
            ),
            "cuda_only_cublas_workspace_config": ":4096:8",
        },
        "model": {
            "family": "nanoGPT GPT-2 small",
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "block_size": 1024,
            "vocab_size": 50_257,
            "dropout": 0.0,
            "bias": False,
            "tie_word_embeddings": True,
            "intended_muon_parameter_count": 48,
        },
        "optimizer": {
            "muon_learning_rate": "1/120",
            "muon_learning_rate_schedule": "constant",
            "muon_momentum": "19/20",
            "muon_nesterov": True,
            "muon_weight_decay": 0,
            "auxiliary_learning_rate": "3/5000",
            "auxiliary_learning_rate_schedule": "constant",
            "auxiliary_betas": ["9/10", "19/20"],
            "auxiliary_epsilon": "1/10000000000",
            "auxiliary_weight_decay": 0,
            "normalization": "X/(||X||_F+1e-7), after one-time wide-matrix transpose",
            "normalization_epsilon": "1e-7",
            "jordan_coefficients": ["6889/2000", "-191/40", "4063/2000"],
            "jordan_stages": 5,
            "aspect_formula": "sqrt(max(1, rows/columns))",
            "candidate_update_scope": (
                "The baseline applies the literal unshielded pinned Muon candidate. "
                "P20 is shadow-only."
            ),
        },
        "data": {
            "dataset_repository": "HuggingFaceFW/fineweb",
            "dataset_config": "sample-10BT",
            "dataset_split": "train",
            "dataset_revision": "9bb295ddab0e05d785b879661af7260fed5140fc",
            "tokenizer_package": "tiktoken==0.14.0",
            "tokenizer_encoding": "gpt2",
            "train_pool_token_count": 131_072,
            "validation_token_count": 32_768,
            "stored_dtype": "little-endian uint16",
            "manifest_template": "experiments/training/p22_fineweb_manifest.template.json",
        },
        "capture": {
            "parameter_selection": (
                "All 48 intended Muon hidden matrices at every one of the 24 capture steps; "
                "no post-hoc parameter, layer, role, norm, or success selection."
            ),
            "timing": (
                "After accumulation and before parameter update, inside the actual pinned "
                "accelerator muon_update path."
            ),
            "post_aspect_candidate": (
                "The actual stored accelerator candidate after the pinned aspect multiply; "
                "this is the fidelity comparator and the candidate supplied to the CPU "
                "proof-reference P20 shadow."
            ),
            "shadow_output": (
                "P20 output computed from detached contiguous CPU copies and never returned "
                "to, or read by, the baseline optimizer."
            ),
        },
        "noninterference": {
            "shadow_update_applied": False,
        },
    }
    for section, expected_fields in expected_protocol_fields.items():
        actual_section = protocol.get(section)
        if not isinstance(actual_section, Mapping):
            raise P23AcquisitionError(f"frozen P22 protocol has no {section} mapping")
        for field, expected in expected_fields.items():
            if (
                type(actual_section.get(field)) is not type(expected)
                or actual_section.get(field) != expected
            ):
                raise P23AcquisitionError(f"frozen P22 protocol field {section}.{field} changed")
    upstream = protocol.get("upstream")
    trainer = upstream.get("trainer") if isinstance(upstream, Mapping) else None
    muon = upstream.get("muon") if isinstance(upstream, Mapping) else None
    expected_trainer = {
        "revision": core_constants["NANOGPT_REVISION"],
        "tree": core_constants["NANOGPT_TREE"],
        "train_py_blob": core_constants["NANOGPT_TRAIN_BLOB"],
        "model_py_blob": core_constants["NANOGPT_MODEL_BLOB"],
        "dirty_checkout_allowed": False,
    }
    expected_muon = {
        "revision": core_constants["MUON_REVISION"],
        "audited_file": "muon.py",
        "audited_file_sha256": core_constants["MUON_SOURCE_SHA256"],
        "optimizer": "SingleDeviceMuonWithAuxAdam",
        "license": "MIT",
    }
    for label, actual, expected in (
        ("trainer", trainer, expected_trainer),
        ("muon", muon, expected_muon),
    ):
        if not isinstance(actual, Mapping) or any(
            type(actual.get(field)) is not type(value) or actual.get(field) != value
            for field, value in expected.items()
        ):
            raise P23AcquisitionError(f"frozen P22 upstream {label} contract changed")

    if static_contract is not None:
        expected_static_fields = {
            "schema_version": P23_STATIC_CONTRACT_SCHEMA,
            "backend": "cuda",
            "p22_protocol_sha256": p22_binding.get("sha256"),
            "p22_run_manifest_schema": "passive-muon-p22-run-manifest-v2",
            "seed": 1337,
            "optimizer_steps": 256,
            "sequence_length": 128,
            "capture_steps": list(capture_steps),
            "muon_parameter_count": 48,
            "candidate_observation_count": 1_152,
            "optimizer": {
                "muon_learning_rate": "1/120",
                "muon_momentum": "19/20",
                "muon_weight_decay": 0,
                "auxiliary_learning_rate": "3/5000",
                "auxiliary_betas": ["9/10", "19/20"],
                "auxiliary_epsilon": "1e-10",
            },
            "model_optimizer_binding": _expected_binding_audit(),
            "shape_inventory": _expected_shape_inventory(),
        }
        for field, expected in expected_static_fields.items():
            if static_contract.get(field) != expected:
                raise P23AcquisitionError(
                    f"P23 static contract field {field} differs from frozen P22 semantics"
                )


def _read_mapping_bytes(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw_bytes = path.resolve().read_bytes()
        value = json.loads(raw_bytes.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise P23AcquisitionError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise P23AcquisitionError(f"{label} must be a JSON mapping")
    return value, raw_bytes


def _read_mapping(path: Path, label: str) -> dict[str, Any]:
    value, _ = _read_mapping_bytes(path, label)
    return value


def _artifact_record(
    path: Path,
    manifest: Mapping[str, object] | None = None,
    *,
    include_run_identity: bool = True,
) -> dict[str, str]:
    resolved = path.resolve()
    loaded, raw_bytes = _read_mapping_bytes(resolved, "artifact being bound")
    if manifest is not None and loaded != manifest:
        raise P23AcquisitionError("artifact bytes differ from the parsed manifest being bound")
    record = {
        "path": str(resolved),
        "sha256": hashlib.sha256(raw_bytes).hexdigest(),
    }
    if manifest is not None and include_run_identity:
        identity = manifest.get("run_identity_sha256")
        if not isinstance(identity, str):
            raise P23AcquisitionError(f"manifest has no run identity: {resolved}")
        record["run_identity_sha256"] = identity
    return record


def _require_ready(result: Mapping[str, object], label: str) -> None:
    if result.get("ready") is not True:
        raise P23AcquisitionError(f"{label} is not ready: {result.get('blockers')}")


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    return True


def _declared_data_paths(manifest_path: Path) -> set[Path]:
    """Collect locally declared data paths without trusting them as validated data."""

    resolved_manifest = manifest_path.resolve()
    paths = {resolved_manifest}
    if not resolved_manifest.is_file():
        return paths
    try:
        payload = json.loads(resolved_manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return paths

    def visit(value: object) -> None:
        if isinstance(value, Mapping):
            raw_path = value.get("path")
            if isinstance(raw_path, str) and raw_path:
                candidate = Path(raw_path)
                if not candidate.is_absolute():
                    candidate = resolved_manifest.parent / candidate
                paths.add(candidate.resolve())
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(payload)
    return paths


def _assert_native_output_paths(
    *,
    outputs: Mapping[str, Path],
    inputs: Mapping[str, Path],
    nanogpt_root: Path,
    data_manifest: Path,
) -> None:
    """Keep native evidence external and prevent any output/input alias."""

    resolved_outputs = {name: path.resolve() for name, path in outputs.items()}
    if len(set(resolved_outputs.values())) != len(resolved_outputs):
        raise P23AcquisitionError("native output paths must be pairwise distinct")
    for name, path in resolved_outputs.items():
        if _is_within(path, REPOSITORY_ROOT):
            raise P23AcquisitionError(f"native {name} output must remain outside the repository")
        if outputs[name].is_symlink() or path.exists():
            raise P23AcquisitionError(f"native {name} output already exists")

    protected_files = {path.resolve() for path in inputs.values()}
    declared_data_paths = _declared_data_paths(data_manifest)
    protected_files.update(declared_data_paths)
    protected_roots = {REPOSITORY_ROOT.resolve(), nanogpt_root.resolve()}
    for label in ("Muon", "FineWeb manifest"):
        source = inputs.get(label)
        if source is not None:
            resolved = source.resolve()
            protected_roots.add(resolved if resolved.is_dir() else resolved.parent)
    for declared in declared_data_paths:
        protected_roots.add(declared if declared.is_dir() else declared.parent)
    for name, path in resolved_outputs.items():
        if path in protected_files:
            raise P23AcquisitionError(f"native {name} output aliases an input or prerequisite")
        if any(_is_within(path, root) for root in protected_roots):
            raise P23AcquisitionError(
                f"native {name} output lies inside a protected source or data tree"
            )


def _assert_sanitize_output_path(
    *, manifest_path: Path, output: Path, path_roots: Mapping[str, Path]
) -> None:
    expected_labels = {
        "repository",
        "preprocessor_alias",
        "nanogpt",
        "muon",
        "data",
        "python_environment",
        "native",
    }
    if set(path_roots) != expected_labels:
        raise P23AcquisitionError(
            "sanitization roots must be exactly repository,preprocessor_alias,"
            "nanogpt,muon,data,python_environment,native"
        )
    resolved_roots = {name: root.resolve() for name, root in path_roots.items()}
    for left_name, left in resolved_roots.items():
        for right_name, right in resolved_roots.items():
            if left_name >= right_name:
                continue
            if _is_within(left, right) or _is_within(right, left):
                raise P23AcquisitionError("sanitization roots must be pairwise non-overlapping")
    native = path_roots.get("native")
    assert native is not None
    resolved_input = manifest_path.resolve()
    resolved_output = output.resolve()
    if manifest_path.is_symlink() or not resolved_input.is_file():
        raise P23AcquisitionError("native sanitization input must be a retained regular file")
    if resolved_input == resolved_output:
        raise P23AcquisitionError("sanitized output must not overwrite its native artifact")
    if output.is_symlink() or resolved_output.exists():
        raise P23AcquisitionError("sanitized output already exists")
    if _is_within(resolved_output, REPOSITORY_ROOT):
        raise P23AcquisitionError("sanitized output must remain outside the repository")
    if not _is_within(resolved_input, native) or not _is_within(resolved_output, native):
        raise P23AcquisitionError(
            "native input and sanitized output must both lie in the declared evidence root"
        )
    protected = {name: root.resolve() for name, root in path_roots.items() if name != "native"}
    if any(_is_within(resolved_output, root) for root in protected.values()):
        raise P23AcquisitionError("sanitized output lies inside a protected source or data tree")


def _expected_shape_inventory() -> dict[str, object]:
    parameters = {
        str(item["name"]): int(item["numel"])
        for item in P22_CORE.expected_gpt2_small_muon_inventory()
    }
    return {
        "parameter_count": P23.EXPECTED_PARAMETER_COUNT,
        "total_numel": sum(parameters.values()),
        "parameters": parameters,
        "stored_shapes": {
            str(item["name"]): list(item["stored_shape"])
            for item in P22_CORE.expected_gpt2_small_muon_inventory()
        },
    }


def _expected_binding_audit() -> dict[str, object]:
    auxiliary_names = ["transformer.wte.weight", "transformer.wpe.weight"]
    for layer in range(12):
        auxiliary_names.extend(
            (
                f"transformer.h.{layer}.ln_1.weight",
                f"transformer.h.{layer}.ln_2.weight",
            )
        )
    auxiliary_names.append("transformer.ln_f.weight")
    muon_names = list(_expected_shape_inventory()["parameters"])
    auxiliary_shapes: dict[str, list[int]] = {
        "transformer.wte.weight": [P22_RUNNER.VOCAB_SIZE, 768],
        "transformer.wpe.weight": [1_024, 768],
        "transformer.ln_f.weight": [768],
    }
    for layer in range(12):
        auxiliary_shapes[f"transformer.h.{layer}.ln_1.weight"] = [768]
        auxiliary_shapes[f"transformer.h.{layer}.ln_2.weight"] = [768]
    muon_shapes = {
        str(item["name"]): list(item["stored_shape"])
        for item in P22_CORE.expected_gpt2_small_muon_inventory()
    }
    parameter_inventory: list[dict[str, object]] = []
    for name in auxiliary_names:
        shape = auxiliary_shapes[name]
        parameter_inventory.append(
            {
                "name": name,
                "aliases": (
                    ["lm_head.weight", "transformer.wte.weight"]
                    if name == "transformer.wte.weight"
                    else [name]
                ),
                "optimizer_group_index": 0,
                "use_muon": False,
                "device": "cuda:0",
                "dtype": "torch.float32",
                "shape": shape,
                "numel": _shape_numel(shape),
            }
        )
    for name in muon_names:
        shape = muon_shapes[name]
        parameter_inventory.append(
            {
                "name": name,
                "aliases": [name],
                "optimizer_group_index": 1,
                "use_muon": True,
                "device": "cuda:0",
                "dtype": "torch.float32",
                "shape": shape,
                "numel": _shape_numel(shape),
            }
        )
    parameter_inventory.sort(key=lambda record: str(record["name"]))
    return {
        "verified": True,
        "verification": "exact in-process Python object identity after accelerator move",
        "selected_device_type": "cuda",
        "stored_parameter_devices": ["cuda:0"],
        "stored_parameter_dtypes": ["torch.float32"],
        "model_unique_parameter_count": 75,
        "optimizer_parameter_count": 75,
        "muon_parameter_count": 48,
        "auxiliary_parameter_count": 27,
        "complete_unique_identity_coverage": True,
        "no_duplicate_optimizer_parameters": True,
        "muon_metadata_exact": True,
        "tied_parameter_aliases": [["lm_head.weight", "transformer.wte.weight"]],
        "model_config": {
            "block_size": 1024,
            "vocab_size": 50_257,
            "n_layer": 12,
            "n_head": 12,
            "n_embd": 768,
            "dropout": 0.0,
            "bias": False,
        },
        "optimizer_groups": [
            {
                "group_index": 0,
                "use_muon": False,
                "hyperparameters": {
                    "learning_rate": float(Fraction(3, 5_000)),
                    "betas": [0.9, 0.95],
                    "epsilon": 1.0e-10,
                    "weight_decay": 0.0,
                },
                "parameter_names": auxiliary_names,
            },
            {
                "group_index": 1,
                "use_muon": True,
                "hyperparameters": {
                    "learning_rate": float(Fraction(1, 120)),
                    "momentum": float(Fraction(19, 20)),
                    "weight_decay": 0.0,
                },
                "parameter_names": muon_names,
            },
        ],
        "parameter_inventory": parameter_inventory,
    }


def _shape_numel(shape: list[int]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
    return result


def _tracked_artifact_logical_path(path: Path, label: str) -> str:
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise P23AcquisitionError(
            f"{label} must be a tracked file inside the OptimizationML repository"
        ) from error
    return relative.as_posix()


def _runtime_lock_logical_path(runtime_lock_path: Path) -> str:
    return _tracked_artifact_logical_path(runtime_lock_path, "runtime lock")


def _host_attestation_logical_path(host_attestation_path: Path) -> str:
    return _tracked_artifact_logical_path(host_attestation_path, "host attestation")


def _validate_tracked_artifact_binding(
    path: Path,
    *,
    expected_sha256: object,
    label: str,
    repository: Mapping[str, object],
) -> str:
    relative = _tracked_artifact_logical_path(path, label)
    tracked = repository.get("tracked_files")
    if not isinstance(tracked, Mapping) or not isinstance(tracked.get(relative), Mapping):
        raise P23AcquisitionError(f"{label} is not tracked in the frozen Git tree")
    record = tracked[relative]
    assert isinstance(record, Mapping)
    if record.get("kind") != "file" or record.get("sha256") != expected_sha256:
        raise P23AcquisitionError(f"{label} bytes differ from the committed Git inventory")
    return relative


def _validate_runtime_lock_repository_binding(
    runtime_lock_path: Path,
    runtime_lock: Mapping[str, object],
    repository: Mapping[str, object],
) -> str:
    return _validate_tracked_artifact_binding(
        runtime_lock_path,
        expected_sha256=runtime_lock.get("runtime_lock_sha256"),
        label="runtime lock",
        repository=repository,
    )


def _validate_host_attestation_repository_binding(
    host_attestation_path: Path,
    host_attestation: Mapping[str, object],
    repository: Mapping[str, object],
) -> str:
    return _validate_tracked_artifact_binding(
        host_attestation_path,
        expected_sha256=host_attestation.get("host_attestation_sha256"),
        label="host attestation",
        repository=repository,
    )


def build_static_contract(
    *,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
    instrumentation_patch: Path,
    runtime_lock_path: Path,
    runtime_lock: Mapping[str, object],
    host_attestation_path: Path,
    host_attestation: Mapping[str, object],
) -> dict[str, object]:
    """Build the role-independent acquisition contract included in every run."""

    nanogpt = P22_CORE.validate_nanogpt_checkout(nanogpt_root)
    muon = P22_CORE.validate_muon_source(muon_source)
    data = P22_CORE.validate_fineweb_manifest(data_manifest)
    instrumentation = P22_CORE.validate_instrumentation_patch(instrumentation_patch)
    for label, result in (
        ("nanoGPT checkout", nanogpt),
        ("Muon source", muon),
        ("FineWeb materialization", data),
        ("observer adapter", instrumentation),
    ):
        _require_ready(result, label)

    details = nanogpt.get("details")
    if not isinstance(details, Mapping):
        raise P23AcquisitionError("nanoGPT validation has no source identity")
    artifacts = data.get("artifacts")
    if not isinstance(artifacts, list):
        raise P23AcquisitionError("FineWeb validation has no artifact inventory")
    data_artifacts: list[dict[str, object]] = []
    for artifact in artifacts:
        if not isinstance(artifact, Mapping):
            raise P23AcquisitionError("FineWeb artifact inventory is malformed")
        data_artifacts.append(
            {
                "sha256": artifact.get("sha256"),
                "byte_count": artifact.get("byte_count"),
            }
        )

    runtime_lock_sha256 = runtime_lock.get("runtime_lock_sha256")
    if not isinstance(runtime_lock_sha256, str):
        raise P23AcquisitionError("runtime lock has no byte digest")
    host_attestation_sha256 = host_attestation.get("host_attestation_sha256")
    if not isinstance(host_attestation_sha256, str):
        raise P23AcquisitionError("host attestation has no byte digest")
    return {
        "schema_version": P23_STATIC_CONTRACT_SCHEMA,
        "backend": "cuda",
        "p22_protocol_sha256": P22_CORE.protocol_sha256(),
        "p22_run_manifest_schema": P22_CORE.P22_RUN_MANIFEST_SCHEMA,
        "seed": P22_RUNNER.SEED,
        "optimizer_steps": P22_CORE.EXPECTED_OPTIMIZER_STEPS,
        "sequence_length": P22_CORE.EXPECTED_SEQUENCE_LENGTH,
        "capture_steps": list(P23.EXPECTED_CAPTURE_STEPS),
        "muon_parameter_count": P23.EXPECTED_PARAMETER_COUNT,
        "candidate_observation_count": P23.EXPECTED_OBSERVATION_COUNT,
        "optimizer": {
            "muon_learning_rate": "1/120",
            "muon_momentum": "19/20",
            "muon_weight_decay": 0,
            "auxiliary_learning_rate": "3/5000",
            "auxiliary_betas": ["9/10", "19/20"],
            "auxiliary_epsilon": "1e-10",
        },
        "nanogpt": {
            key: details.get(key)
            for key in ("revision", "tree", "train_py_blob", "model_py_blob", "origin")
        },
        "muon": {
            "revision": muon.get("revision"),
            "sha256": muon.get("sha256"),
        },
        "observer_adapter": {
            "sha256": instrumentation.get("sha256"),
            "byte_count": instrumentation.get("byte_count"),
        },
        "data": {
            "manifest_sha256": data.get("manifest_sha256"),
            "artifacts": data_artifacts,
        },
        "runtime_lock": {
            "repository_path": _runtime_lock_logical_path(runtime_lock_path),
            "sha256": runtime_lock_sha256,
        },
        "host_attestation": {
            "repository_path": _host_attestation_logical_path(host_attestation_path),
            "sha256": host_attestation_sha256,
            "schema_version": host_attestation.get("schema_version"),
        },
        "model_optimizer_binding": _expected_binding_audit(),
        "shape_inventory": _expected_shape_inventory(),
        "identity_excludes": [
            "acquisition_role",
            "trace_mode",
            "output_path",
            "raw_trace_output_path",
            "elapsed_seconds",
        ],
    }


def _prepare_context(
    *,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
    instrumentation_patch: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    addendum_path: Path,
) -> dict[str, object]:
    """Lock files and flags before the first P22 preflight or CUDA operation."""

    addendum = P23.load_and_validate_addendum(addendum_path, repository_root=REPOSITORY_ROOT)
    _validate_frozen_fidelity_gate_constants(addendum=addendum)
    _validate_frozen_p22_acquisition_constants(addendum=addendum)
    runtime_lock = P23.load_runtime_lock(
        runtime_lock_path,
        addendum_path=addendum_path,
        host_attestation_path=host_attestation_path,
    )
    host_attestation = P23.load_and_validate_host_attestation(
        host_attestation_path,
        runtime_lock=runtime_lock,
    )
    repository = P23.collect_git_provenance(REPOSITORY_ROOT)
    _validate_runtime_lock_repository_binding(runtime_lock_path, runtime_lock, repository)
    _validate_host_attestation_repository_binding(
        host_attestation_path,
        host_attestation,
        repository,
    )
    static_contract = build_static_contract(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        data_manifest=data_manifest,
        instrumentation_patch=instrumentation_patch,
        runtime_lock_path=runtime_lock_path,
        runtime_lock=runtime_lock,
        host_attestation_path=host_attestation_path,
        host_attestation=host_attestation,
    )
    _validate_frozen_p22_acquisition_constants(
        addendum=addendum,
        static_contract=static_contract,
    )
    data_validation = P22_CORE.validate_fineweb_manifest(data_manifest)
    _require_ready(data_validation, "FineWeb materialization")
    # This must precede collect_cuda_runtime and the P22 accelerator preflight.
    P23.configure_cuda_determinism()
    runtime = P23.collect_cuda_runtime(container_identity=runtime_lock)
    P23.validate_runtime_against_lock(runtime, runtime_lock)
    return {
        "addendum_path": addendum_path.resolve(),
        "runtime_lock": runtime_lock,
        "host_attestation": host_attestation,
        "repository": repository,
        "static_contract": static_contract,
        "runtime": runtime,
        "data_validation": data_validation,
        "inputs": {
            "nanogpt_root": nanogpt_root.resolve(),
            "muon_source": muon_source.resolve(),
            "data_manifest": data_manifest.resolve(),
            "instrumentation_patch": instrumentation_patch.resolve(),
            "runtime_lock_path": runtime_lock_path.resolve(),
            "host_attestation_path": host_attestation_path.resolve(),
        },
        "scopes": {
            "repository": REPOSITORY_ROOT,
            "nanogpt": nanogpt_root.resolve(),
            "muon": muon_source.resolve(),
        },
        "loaded_file_roots": {
            "repository": REPOSITORY_ROOT,
            "nanogpt": nanogpt_root.resolve(),
            "muon": muon_source.resolve(),
            "data": data_manifest.resolve().parents[1],
            "preprocessor_alias": Path(
                "/Users/harry/Desktop/temp/OptimizationML/experiments/training/"
                "materialize_p22_fineweb.py"
            ),
            "python_environment": Path(P23.PINNED_PYTHON_ENVIRONMENT),
            "native_evidence": Path("/workspace/evidence/p23"),
            "image_inspection": Path("/mounted-host-evidence/image-inspect.json"),
            "running_container_inspection": Path(
                "/mounted-host-evidence/running-container-inspect.json"
            ),
            "running_mountinfo": Path("/mounted-host-evidence/running-mountinfo.txt"),
            "nvidia_smi_query": Path("/mounted-host-evidence/nvidia-smi.csv"),
            "tmpfs": Path("/tmp"),
            "dev_shm": Path("/dev/shm"),
            "image": Path("/"),
        },
    }


def _collect_source_modules(context: Mapping[str, object]) -> dict[str, dict[str, object]]:
    scopes = context.get("scopes")
    repository = context.get("repository")
    if not isinstance(scopes, Mapping) or not isinstance(repository, Mapping):
        raise P23AcquisitionError("acquisition context has no source scopes or Git inventory")
    tracked = repository.get("tracked_files")
    if not isinstance(tracked, Mapping):
        raise P23AcquisitionError("Git provenance has no tracked-file inventory")
    repository_allowlist = {
        str(path)
        for path, record in tracked.items()
        if isinstance(record, Mapping) and record.get("kind") == "file"
    }
    if not repository_allowlist:
        raise P23AcquisitionError("Git provenance has no tracked source files")
    return P23.collect_runtime_module_hashes(
        {str(name): Path(path) for name, path in scopes.items()},
        scope_file_allowlists={"repository": repository_allowlist},
    )


def _loaded_file_roots(context: Mapping[str, object]) -> dict[str, Path]:
    roots = context.get("loaded_file_roots")
    if not isinstance(roots, Mapping):
        raise P23AcquisitionError("acquisition context has no loaded-file logical roots")
    result = {str(name): Path(path) for name, path in roots.items()}
    expected = {
        "repository",
        "nanogpt",
        "muon",
        "data",
        "preprocessor_alias",
        "python_environment",
        "native_evidence",
        "image_inspection",
        "running_container_inspection",
        "running_mountinfo",
        "nvidia_smi_query",
        "tmpfs",
        "dev_shm",
        "image",
    }
    if set(result) != expected:
        raise P23AcquisitionError("acquisition loaded-file logical roots changed")
    return result


def _collect_loaded_files(context: Mapping[str, object]) -> dict[str, object]:
    runtime = context.get("runtime")
    if not isinstance(runtime, Mapping):
        raise P23AcquisitionError("acquisition context has no live runtime")
    mountinfo_bytes = P23.read_live_mountinfo_bytes()
    if hashlib.sha256(mountinfo_bytes).hexdigest() != runtime.get("live_mountinfo_sha256"):
        raise P23AcquisitionError("live mount namespace changed during acquisition")
    return P23.collect_loaded_file_snapshot(
        _loaded_file_roots(context),
        mountinfo_text=mountinfo_bytes,
    )


def _prime_external_modules(nanogpt_root: Path, muon_source: Path) -> None:
    """Load the same two external modules so a verifier can rebuild the source map."""

    P22_RUNNER._load_module("p22_pinned_nanogpt_model", nanogpt_root / "model.py")
    P22_RUNNER._load_module("p22_pinned_kellerjordan_muon", muon_source)


def _prime_verifier_initialization(context: Mapping[str, object]) -> tuple[object, ...]:
    """Reproduce the frozen pre-training state before recollecting loaded files."""

    inputs = context.get("inputs")
    if not isinstance(inputs, Mapping):
        raise P23AcquisitionError("verifier context has no frozen input paths")
    nanogpt_root = inputs.get("nanogpt_root")
    muon_source = inputs.get("muon_source")
    data_manifest = inputs.get("data_manifest")
    instrumentation_patch = inputs.get("instrumentation_patch")
    if not all(
        isinstance(path, Path)
        for path in (nanogpt_root, muon_source, data_manifest, instrumentation_patch)
    ):
        raise P23AcquisitionError("verifier cannot reproduce the P22 initialization")
    preflight = P22_RUNNER.build_preflight_report(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        fineweb_manifest=data_manifest,
        instrumentation_patch=instrumentation_patch,
        accelerator_backend="cuda",
    )
    if preflight.get("status") != "ready":
        raise P23AcquisitionError("verifier cannot reproduce a ready P22 preflight")
    P22_RUNNER._set_determinism("cuda")
    tokens, data_validation = P22_RUNNER._load_train_tokens(data_manifest)
    model_bundle = P22_RUNNER._model_and_optimizer(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        backend="cuda",
    )
    return tokens, data_validation, model_bundle


def _new_process_instance() -> dict[str, object]:
    """Return this interpreter's immutable process-start attestation."""

    return copy.deepcopy(_PROCESS_INSTANCE)


def _validate_process_instance(manifest: Mapping[str, object], expected_role: str) -> str:
    if manifest.get("acquisition_role") != expected_role:
        raise P23AcquisitionError(
            f"expected acquisition_role={expected_role!r}, got {manifest.get('acquisition_role')!r}"
        )
    process = manifest.get("process_instance")
    if not isinstance(process, Mapping) or process.get("schema_version") != (
        P23_PROCESS_INSTANCE_SCHEMA
    ):
        raise P23AcquisitionError(f"{expected_role} has no process-instance attestation")
    nonce = process.get("nonce")
    pid = process.get("pid")
    created = process.get("created_time_ns")
    if not isinstance(nonce, str) or _SHA256.fullmatch(nonce) is None:
        raise P23AcquisitionError(f"{expected_role} has an invalid process nonce")
    if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
        raise P23AcquisitionError(f"{expected_role} has an invalid process id")
    if not isinstance(created, int) or isinstance(created, bool) or created <= 0:
        raise P23AcquisitionError(f"{expected_role} has an invalid process timestamp")
    return nonce


def _fresh_runtime(context: Mapping[str, object]) -> dict[str, object]:
    runtime_lock = context.get("runtime_lock")
    if not isinstance(runtime_lock, Mapping):
        raise P23AcquisitionError("acquisition context has no runtime lock")
    runtime = P23.collect_cuda_runtime(container_identity=runtime_lock)
    P23.validate_runtime_against_lock(runtime, runtime_lock)
    return runtime


def _assert_repository_unchanged(context: Mapping[str, object]) -> None:
    initialized = context.get("repository")
    if not isinstance(initialized, Mapping):
        raise P23AcquisitionError("acquisition context has no Git provenance")
    completed = P23.collect_git_provenance(REPOSITORY_ROOT)
    if completed != initialized:
        raise P23AcquisitionError("Git provenance changed during acquisition")


def _assert_data_unchanged(context: Mapping[str, object]) -> None:
    initialized = context.get("data_validation")
    inputs = context.get("inputs")
    if not isinstance(initialized, Mapping) or not isinstance(inputs, Mapping):
        raise P23AcquisitionError("acquisition context has no FineWeb provenance")
    manifest_path = inputs.get("data_manifest")
    if not isinstance(manifest_path, Path):
        raise P23AcquisitionError("acquisition context has no FineWeb manifest path")
    completed = P22_CORE.validate_fineweb_manifest(manifest_path)
    _require_ready(completed, "FineWeb materialization after acquisition")
    if completed != initialized:
        raise P23AcquisitionError("FineWeb provenance changed during acquisition")


def _p22_identity_binding(manifest: Mapping[str, object]) -> dict[str, object]:
    source_snapshot = manifest.get("source_snapshot")
    data = manifest.get("data")
    if not isinstance(source_snapshot, Mapping) or not isinstance(data, Mapping):
        raise P23AcquisitionError("P23 manifest cannot reconstruct its underlying P22 identity")
    data_digest = data.get("manifest_sha256")
    if not isinstance(data_digest, str) or _SHA256.fullmatch(data_digest) is None:
        raise P23AcquisitionError("P23 manifest has no valid P22 FineWeb digest")
    normalized_sources = {str(name): value for name, value in source_snapshot.items()}
    if any(
        not isinstance(value, str) or _SHA256.fullmatch(value) is None
        for value in normalized_sources.values()
    ):
        raise P23AcquisitionError("P23 manifest has a malformed P22 source snapshot")
    run_identity = P22_RUNNER._run_identity(normalized_sources, data_digest, "cuda")
    return {
        "schema_version": P23_P22_ACQUISITION_BINDING_SCHEMA,
        "source_manifest_schema_version": P22_CORE.P22_RUN_MANIFEST_SCHEMA,
        "protocol_sha256": P22_CORE.protocol_sha256(),
        "run_identity_sha256": run_identity,
        "run_identity_inputs": {
            "source_snapshot": normalized_sources,
            "data_manifest_sha256": data_digest,
            "accelerator_backend": "cuda",
            "seed": P22_RUNNER.SEED,
        },
    }


def _reconstruct_p22_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    acquisition = manifest.get("p22_acquisition")
    p22_runtime = manifest.get("p22_runtime_provenance")
    if not isinstance(acquisition, Mapping) or not isinstance(p22_runtime, Mapping):
        raise P23AcquisitionError("P23 manifest omits reconstructible P22 provenance")
    reconstructed = copy.deepcopy(dict(manifest))
    for field in (
        "acquisition_role",
        "process_instance",
        "p22_acquisition",
        "p22_runtime_provenance",
        "execution_identity",
        "metric_reconstruction_scope",
    ):
        reconstructed.pop(field, None)
    reconstructed["schema_version"] = P22_CORE.P22_RUN_MANIFEST_SCHEMA
    reconstructed["run_identity_sha256"] = acquisition.get("run_identity_sha256")
    reconstructed["runtime"] = copy.deepcopy(dict(p22_runtime))
    return reconstructed


def _validate_reconstructed_p22_manifest(
    manifest: Mapping[str, object], expected_mode: str
) -> None:
    reconstructed = _reconstruct_p22_manifest(manifest)
    with tempfile.TemporaryDirectory(prefix="p23-p22-reconstruction-") as temporary:
        path = Path(temporary) / "p22-manifest.json"
        P22_CORE.write_json_atomic(path, reconstructed)
        loaded = P22_CORE._load_run_manifest(path, expected_mode)
    if loaded != reconstructed:
        raise P23AcquisitionError("P22 reconstruction changed while it was independently validated")


def _validate_manifest_static_binding(
    manifest: Mapping[str, object],
    *,
    expected_mode: str,
    context: Mapping[str, object],
) -> None:
    identity = manifest.get("execution_identity")
    static_contract = context.get("static_contract")
    inputs = context.get("inputs")
    data_validation = context.get("data_validation")
    runtime = context.get("runtime")
    if (
        not isinstance(identity, Mapping)
        or not isinstance(static_contract, Mapping)
        or not isinstance(inputs, Mapping)
        or not isinstance(data_validation, Mapping)
        or not isinstance(runtime, Mapping)
    ):
        raise P23AcquisitionError("cannot re-bind an incomplete P23 acquisition context")
    if identity.get("static_contract") != static_contract:
        raise P23AcquisitionError("manifest identity does not contain the current static contract")
    exact_top_level = {
        "protocol_sha256": P22_CORE.protocol_sha256(),
        "evidence_kind": "real_gradient_shadow_trace",
        "accelerator_backend": "cuda",
        "model_optimizer_binding": static_contract.get("model_optimizer_binding"),
        "shape_inventory": static_contract.get("shape_inventory"),
        "data": data_validation,
        "runtime": runtime,
        "metric_reconstruction_scope": METRIC_RECONSTRUCTION_SCOPE,
    }
    changed = [
        field for field, expected in exact_top_level.items() if manifest.get(field) != expected
    ]
    if changed:
        raise P23AcquisitionError(
            f"P23 top-level acquisition fields are not identity-bound: {changed}"
        )

    nanogpt_root = inputs.get("nanogpt_root")
    muon_source = inputs.get("muon_source")
    data_manifest = inputs.get("data_manifest")
    if not all(isinstance(path, Path) for path in (nanogpt_root, muon_source, data_manifest)):
        raise P23AcquisitionError("cannot reconstruct the current P22 source snapshot")
    expected_sources = P22_RUNNER._source_snapshot(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        data_manifest=data_manifest,
    )
    if manifest.get("source_snapshot") != expected_sources:
        raise P23AcquisitionError("P23 manifest P22 source snapshot differs from current bytes")
    expected_p22_runtime = P22_RUNNER._runtime_provenance("cuda")
    if manifest.get("p22_runtime_provenance") != expected_p22_runtime:
        raise P23AcquisitionError(
            "P23 manifest P22 runtime provenance differs from the live runtime"
        )
    expected_acquisition = _p22_identity_binding(manifest)
    if manifest.get("p22_acquisition") != expected_acquisition:
        raise P23AcquisitionError("P23 manifest does not reconstruct its exact P22 identity")
    _validate_reconstructed_p22_manifest(manifest, expected_mode)


def _verify_manifest_identity(
    manifest: Mapping[str, object],
    *,
    expected_mode: str,
    expected_role: str,
    context: Mapping[str, object],
    source_modules: Mapping[str, Mapping[str, object]],
    loaded_file_initialized: Mapping[str, object],
) -> dict[str, object]:
    P23.validate_run_manifest(manifest, expected_mode)
    _validate_manifest_static_binding(
        manifest,
        expected_mode=expected_mode,
        context=context,
    )
    P22_CORE._validate_model_optimizer_binding(manifest, expected_mode)
    _validate_process_instance(manifest, expected_role)
    repository = context.get("repository")
    runtime = context.get("runtime")
    static_contract = context.get("static_contract")
    addendum_path = context.get("addendum_path")
    if (
        not isinstance(repository, Mapping)
        or not isinstance(runtime, Mapping)
        or not isinstance(static_contract, Mapping)
        or not isinstance(addendum_path, Path)
    ):
        raise P23AcquisitionError("acquisition context is incomplete")

    def source_collector() -> Mapping[str, Mapping[str, object]]:
        # The richer loaded-file closure below independently rehashes every
        # initialized and completed artifact.  ``source_modules`` remains the
        # compatibility view of the initialized controlled-root membership.
        return copy.deepcopy(dict(source_modules))

    return P23.verify_execution_identity(
        manifest,
        addendum_path=addendum_path,
        repository_collector=lambda: P23.collect_git_provenance(REPOSITORY_ROOT),
        source_collector=source_collector,
        loaded_file_initialized_collector=lambda: copy.deepcopy(dict(loaded_file_initialized)),
        loaded_file_roots=_loaded_file_roots(context),
        runtime_collector=lambda: _fresh_runtime(context),
        expected_static_contract=static_contract,
        repository_root=REPOSITORY_ROOT,
    )


def _repeatability_report(
    *,
    trace_off_a_path: Path,
    trace_off_b_path: Path,
    context: Mapping[str, object],
    source_modules: Mapping[str, Mapping[str, object]],
    loaded_file_initialized: Mapping[str, object],
) -> dict[str, object]:
    if trace_off_a_path.resolve() == trace_off_b_path.resolve():
        raise P23AcquisitionError("trace-off A and B must be distinct artifacts")
    trace_off_a = _read_mapping(trace_off_a_path, "trace-off A manifest")
    trace_off_b = _read_mapping(trace_off_b_path, "trace-off B manifest")
    identity_a = _verify_manifest_identity(
        trace_off_a,
        expected_mode="trace_off",
        expected_role="trace_off_a",
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    identity_b = _verify_manifest_identity(
        trace_off_b,
        expected_mode="trace_off",
        expected_role="trace_off_b",
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    report = P23.compare_repeatability_manifests(trace_off_a, trace_off_b)
    nonce_a = _validate_process_instance(trace_off_a, "trace_off_a")
    nonce_b = _validate_process_instance(trace_off_b, "trace_off_b")
    if nonce_a == nonce_b:
        raise P23AcquisitionError("trace-off A and B reuse one process-instance nonce")
    report.update(
        {
            "binding_schema": P23_GATE_REPORT_BINDING_SCHEMA,
            "inputs": {
                "trace_off_a": _artifact_record(trace_off_a_path, trace_off_a),
                "trace_off_b": _artifact_record(trace_off_b_path, trace_off_b),
            },
            "identity_verification": {
                "trace_off_a": identity_a,
                "trace_off_b": identity_b,
            },
        }
    )
    return report


def verify_repeatability(
    *,
    trace_off_a_path: Path,
    trace_off_b_path: Path,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
    instrumentation_patch: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    addendum_path: Path,
    output: Path,
) -> dict[str, object]:
    _assert_native_output_paths(
        outputs={"repeatability report": output},
        inputs={
            "trace-off A": trace_off_a_path,
            "trace-off B": trace_off_b_path,
            "nanoGPT": nanogpt_root,
            "Muon": muon_source,
            "FineWeb manifest": data_manifest,
            "observer": instrumentation_patch,
            "runtime lock": runtime_lock_path,
            "host attestation": host_attestation_path,
            "addendum": addendum_path,
        },
        nanogpt_root=nanogpt_root,
        data_manifest=data_manifest,
    )
    context = _prepare_context(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        data_manifest=data_manifest,
        instrumentation_patch=instrumentation_patch,
        runtime_lock_path=runtime_lock_path,
        host_attestation_path=host_attestation_path,
        addendum_path=addendum_path,
    )
    primed_initialization = _prime_verifier_initialization(context)
    source_modules = _collect_source_modules(context)
    loaded_file_initialized = _collect_loaded_files(context)
    report = _repeatability_report(
        trace_off_a_path=trace_off_a_path,
        trace_off_b_path=trace_off_b_path,
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    del primed_initialization
    P22_CORE.write_json_atomic(output, report)
    return report


def _require_passing_gate_report(
    *,
    report_path: Path,
    schema: str,
    expected_inputs: Mapping[str, tuple[Path, Mapping[str, object]]],
) -> dict[str, object]:
    report = _read_mapping(report_path, "gate report")
    if report.get("schema_version") != schema or report.get("passes") is not True:
        raise P23AcquisitionError("supplied gate report is not a passing P23 report")
    if report.get("mismatch_count") != 0 or report.get("mismatches") != []:
        raise P23AcquisitionError("supplied passing gate report does not record zero mismatches")
    if report.get("binding_schema") != P23_GATE_REPORT_BINDING_SCHEMA:
        raise P23AcquisitionError("supplied gate report has no exact artifact binding")
    inputs = report.get("inputs")
    if not isinstance(inputs, Mapping):
        raise P23AcquisitionError("supplied gate report has no input inventory")
    for name, (path, manifest) in expected_inputs.items():
        if inputs.get(name) != _artifact_record(path, manifest):
            raise P23AcquisitionError(f"supplied gate report does not bind {name}")
    return report


def _require_exact_gate_replay(
    supplied: Mapping[str, object], replayed: Mapping[str, object], *, gate: str
) -> None:
    if replayed.get("passes") is not True:
        raise P23AcquisitionError(f"{gate} is barred because fresh exact replay does not pass")
    if supplied != replayed:
        raise P23AcquisitionError(
            f"{gate} is barred because the supplied report differs from fresh exact replay"
        )


def _noninterference_report(
    *,
    trace_off_a_path: Path,
    trace_off_b_path: Path,
    trace_on_path: Path,
    context: Mapping[str, object],
    source_modules: Mapping[str, Mapping[str, object]],
    loaded_file_initialized: Mapping[str, object],
) -> dict[str, object]:
    repeatability = _repeatability_report(
        trace_off_a_path=trace_off_a_path,
        trace_off_b_path=trace_off_b_path,
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    trace_on = _read_mapping(trace_on_path, "trace-on manifest")
    if repeatability.get("passes") is not True:
        return {
            "schema_version": P23.P23_NONINTERFERENCE_SCHEMA,
            "passes": False,
            "comparison": "blocked_by_failed_exact_repeatability",
            "mismatch_count": repeatability.get("mismatch_count"),
            "mismatches": repeatability.get("mismatches"),
            "binding_schema": P23_GATE_REPORT_BINDING_SCHEMA,
            "inputs": {
                **dict(repeatability["inputs"]),
                "trace_on": _artifact_record(trace_on_path, trace_on),
            },
            "repeatability_gate": repeatability,
        }
    identity_on = _verify_manifest_identity(
        trace_on,
        expected_mode="trace_on",
        expected_role="trace_on",
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    trace_off_a = _read_mapping(trace_off_a_path, "trace-off A manifest")
    trace_off_b = _read_mapping(trace_off_b_path, "trace-off B manifest")
    process_nonces = {
        _validate_process_instance(trace_off_a, "trace_off_a"),
        _validate_process_instance(trace_off_b, "trace_off_b"),
        _validate_process_instance(trace_on, "trace_on"),
    }
    if len(process_nonces) != 3:
        raise P23AcquisitionError("off-A, off-B, and trace-on must be fresh processes")
    report = P23.compare_noninterference_manifests(trace_off_a, trace_on)
    report.update(
        {
            "binding_schema": P23_GATE_REPORT_BINDING_SCHEMA,
            "inputs": {
                **dict(repeatability["inputs"]),
                "trace_on": _artifact_record(trace_on_path, trace_on),
            },
            "identity_verification": {
                **dict(repeatability["identity_verification"]),
                "trace_on": identity_on,
            },
            "repeatability_gate": repeatability,
        }
    )
    return report


def verify_noninterference(
    *,
    trace_off_a_path: Path,
    trace_off_b_path: Path,
    trace_on_path: Path,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
    instrumentation_patch: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    addendum_path: Path,
    output: Path,
) -> dict[str, object]:
    _assert_native_output_paths(
        outputs={"noninterference report": output},
        inputs={
            "trace-off A": trace_off_a_path,
            "trace-off B": trace_off_b_path,
            "trace-on": trace_on_path,
            "nanoGPT": nanogpt_root,
            "Muon": muon_source,
            "FineWeb manifest": data_manifest,
            "observer": instrumentation_patch,
            "runtime lock": runtime_lock_path,
            "host attestation": host_attestation_path,
            "addendum": addendum_path,
        },
        nanogpt_root=nanogpt_root,
        data_manifest=data_manifest,
    )
    context = _prepare_context(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        data_manifest=data_manifest,
        instrumentation_patch=instrumentation_patch,
        runtime_lock_path=runtime_lock_path,
        host_attestation_path=host_attestation_path,
        addendum_path=addendum_path,
    )
    primed_initialization = _prime_verifier_initialization(context)
    source_modules = _collect_source_modules(context)
    loaded_file_initialized = _collect_loaded_files(context)
    report = _noninterference_report(
        trace_off_a_path=trace_off_a_path,
        trace_off_b_path=trace_off_b_path,
        trace_on_path=trace_on_path,
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    del primed_initialization
    P22_CORE.write_json_atomic(output, report)
    return report


def _enrich_p22_payload(
    payload: Mapping[str, object],
    *,
    role: str,
    execution_identity: Mapping[str, object],
    runtime: Mapping[str, object],
    process_instance: Mapping[str, object],
) -> dict[str, object]:
    """Convert the hardened P22 v2 acquisition into the P23 manifest schema."""

    expected_mode = "trace_on" if role == "trace_on" else "trace_off"
    if payload.get("schema_version") != P22_CORE.P22_RUN_MANIFEST_SCHEMA:
        raise P23AcquisitionError("P22 acquisition did not emit the hardened v2 schema")
    if payload.get("trace_mode") != expected_mode or payload.get("accelerator_backend") != "cuda":
        raise P23AcquisitionError("P22 acquisition role/backend mismatch")
    P22_CORE._validate_model_optimizer_binding(payload, expected_mode)
    static_contract = execution_identity.get("static_contract")
    if not isinstance(static_contract, Mapping):
        raise P23AcquisitionError("execution identity has no static run contract")
    if payload.get("model_optimizer_binding") != static_contract.get("model_optimizer_binding"):
        raise P23AcquisitionError("actual model/optimizer binding differs from execution identity")
    if payload.get("shape_inventory") != static_contract.get("shape_inventory"):
        raise P23AcquisitionError("actual shape inventory differs from execution identity")

    result = copy.deepcopy(dict(payload))
    p22_identity = result.get("run_identity_sha256")
    p22_runtime = result.get("runtime")
    result["schema_version"] = P23.P23_RUN_MANIFEST_SCHEMA
    result["acquisition_role"] = role
    result["process_instance"] = copy.deepcopy(dict(process_instance))
    p22_acquisition = _p22_identity_binding(payload)
    if p22_acquisition["run_identity_sha256"] != p22_identity:
        raise P23AcquisitionError("P22 run identity does not reconstruct from its exact inputs")
    result["p22_acquisition"] = p22_acquisition
    result["p22_runtime_provenance"] = p22_runtime
    result["runtime"] = copy.deepcopy(dict(runtime))
    result["execution_identity"] = copy.deepcopy(dict(execution_identity))
    result["run_identity_sha256"] = P23.execution_identity_sha256(execution_identity)
    result["metric_reconstruction_scope"] = METRIC_RECONSTRUCTION_SCOPE
    result["candidate_execution"] = {
        "backend": "cuda",
        "actual_accelerator_candidate_computed_and_applied": True,
        "shield_shadow_only": True,
    }
    result["candidate_observation"] = copy.deepcopy(
        P23.TRACE_ON_OBSERVATION if expected_mode == "trace_on" else P23.TRACE_OFF_OBSERVATION
    )

    steps = result.get("steps")
    if not isinstance(steps, list):
        raise P23AcquisitionError("P22 acquisition has no step records")
    for record in steps:
        if not isinstance(record, dict):
            raise P23AcquisitionError("P22 acquisition has a malformed step record")
        capture = record.get("capture")
        if capture is None:
            continue
        if not isinstance(capture, dict) or capture.get("actual_post_aspect_candidate") is not True:
            raise P23AcquisitionError("P22 capture is not the actual post-aspect candidate")
        capture["actual_post_aspect_cuda_candidate"] = True
    result["claim_boundary"] = (
        "One pinned deterministic CUDA shadow acquisition only; the P20 shield remained "
        "record-only. This is not a neural-loss theorem, training-quality result, native "
        "CUDA-shield certificate, or global guarantee for unrepaired upstream Muon."
    )
    return result


def acquire_run(
    *,
    role: str,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
    instrumentation_patch: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    addendum_path: Path,
    output: Path,
    raw_trace_output: Path | None = None,
    trace_off_a_path: Path | None = None,
    trace_off_b_path: Path | None = None,
    repeatability_report_path: Path | None = None,
) -> dict[str, object]:
    """Acquire one role, with trace-on gated before its first training step."""

    if role not in RUN_ROLES:
        raise P23AcquisitionError(f"invalid P23 run role: {role!r}")
    trace_on = role == "trace_on"
    prerequisites = (trace_off_a_path, trace_off_b_path, repeatability_report_path)
    if trace_on and (raw_trace_output is None or any(path is None for path in prerequisites)):
        raise P23AcquisitionError(
            "trace_on requires raw output, trace-off A/B, and a passing repeatability report"
        )
    if not trace_on and raw_trace_output is not None:
        raise P23AcquisitionError("trace-off acquisition cannot capture a raw candidate trace")
    if not trace_on and any(path is not None for path in prerequisites):
        raise P23AcquisitionError("trace-off acquisition does not accept trace-on prerequisites")

    native_outputs = {"run manifest": output}
    if raw_trace_output is not None:
        native_outputs["raw trace"] = raw_trace_output
    native_inputs = {
        "nanoGPT": nanogpt_root,
        "Muon": muon_source,
        "FineWeb manifest": data_manifest,
        "observer": instrumentation_patch,
        "runtime lock": runtime_lock_path,
        "host attestation": host_attestation_path,
        "addendum": addendum_path,
    }
    if trace_off_a_path is not None:
        native_inputs["trace-off A"] = trace_off_a_path
    if trace_off_b_path is not None:
        native_inputs["trace-off B"] = trace_off_b_path
    if repeatability_report_path is not None:
        native_inputs["repeatability report"] = repeatability_report_path
    _assert_native_output_paths(
        outputs=native_outputs,
        inputs=native_inputs,
        nanogpt_root=nanogpt_root,
        data_manifest=data_manifest,
    )

    # Reject a missing, stale, malformed, or nonzero-mismatch prerequisite
    # before runtime setup or model allocation.  The initialized lifecycle
    # callback below then repeats the complete check against live source,
    # repository, runtime, and static-contract maps before the first step.
    if trace_on:
        assert trace_off_a_path is not None
        assert trace_off_b_path is not None
        assert repeatability_report_path is not None
        if trace_off_a_path.resolve() == trace_off_b_path.resolve():
            raise P23AcquisitionError("trace-off A and B must be distinct artifacts")
        preliminary_a = _read_mapping(trace_off_a_path, "trace-off A manifest")
        preliminary_b = _read_mapping(trace_off_b_path, "trace-off B manifest")
        P23.validate_run_manifest(preliminary_a, "trace_off")
        P23.validate_run_manifest(preliminary_b, "trace_off")
        P22_CORE._validate_model_optimizer_binding(preliminary_a, "trace_off")
        P22_CORE._validate_model_optimizer_binding(preliminary_b, "trace_off")
        nonce_a = _validate_process_instance(preliminary_a, "trace_off_a")
        nonce_b = _validate_process_instance(preliminary_b, "trace_off_b")
        if nonce_a == nonce_b:
            raise P23AcquisitionError("trace-off A and B reuse one process-instance nonce")
        if str(_PROCESS_INSTANCE["nonce"]) in {nonce_a, nonce_b}:
            raise P23AcquisitionError("trace_on must run in a third fresh process")
        _require_passing_gate_report(
            report_path=repeatability_report_path,
            schema=P23.P23_REPEATABILITY_SCHEMA,
            expected_inputs={
                "trace_off_a": (trace_off_a_path, preliminary_a),
                "trace_off_b": (trace_off_b_path, preliminary_b),
            },
        )

    context = _prepare_context(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        data_manifest=data_manifest,
        instrumentation_patch=instrumentation_patch,
        runtime_lock_path=runtime_lock_path,
        host_attestation_path=host_attestation_path,
        addendum_path=addendum_path,
    )
    initial_sources: dict[str, dict[str, object]] | None = None
    final_sources: dict[str, dict[str, object]] | None = None
    initial_loaded_files: dict[str, object] | None = None
    final_loaded_files: dict[str, object] | None = None
    loaded_file_closure: dict[str, object] | None = None

    def lifecycle(phase: str) -> None:
        nonlocal initial_sources, final_sources
        nonlocal initial_loaded_files, final_loaded_files, loaded_file_closure
        if phase == "initialized":
            if initial_sources is not None or initial_loaded_files is not None:
                raise P23AcquisitionError("duplicate initialized lifecycle event")
            initial_sources = _collect_source_modules(context)
            initial_loaded_files = _collect_loaded_files(context)
            if trace_on:
                assert trace_off_a_path is not None
                assert trace_off_b_path is not None
                assert repeatability_report_path is not None
                trace_off_a = _read_mapping(trace_off_a_path, "trace-off A manifest")
                trace_off_b = _read_mapping(trace_off_b_path, "trace-off B manifest")
                supplied_report = _require_passing_gate_report(
                    report_path=repeatability_report_path,
                    schema=P23.P23_REPEATABILITY_SCHEMA,
                    expected_inputs={
                        "trace_off_a": (trace_off_a_path, trace_off_a),
                        "trace_off_b": (trace_off_b_path, trace_off_b),
                    },
                )
                replay = _repeatability_report(
                    trace_off_a_path=trace_off_a_path,
                    trace_off_b_path=trace_off_b_path,
                    context=context,
                    source_modules=initial_sources,
                    loaded_file_initialized=initial_loaded_files,
                )
                _require_exact_gate_replay(
                    supplied_report,
                    replay,
                    gate="trace_on",
                )
        elif phase == "completed":
            if (
                initial_sources is None
                or initial_loaded_files is None
                or final_sources is not None
                or final_loaded_files is not None
            ):
                raise P23AcquisitionError("out-of-order completed lifecycle event")
            final_sources = _collect_source_modules(context)
            final_loaded_files = _collect_loaded_files(context)
            loaded_file_closure = P23.build_loaded_file_closure(
                initial_loaded_files,
                final_loaded_files,
            )
            _assert_repository_unchanged(context)
            _assert_data_unchanged(context)
        else:
            raise P23AcquisitionError(f"unknown P22 lifecycle phase: {phase!r}")

    trace_mode = "trace_on" if trace_on else "trace_off"
    process_instance = _new_process_instance()
    runtime = context["runtime"]
    repository = context["repository"]
    static_contract = context["static_contract"]
    addendum = context["addendum_path"]
    if (
        not isinstance(runtime, Mapping)
        or not isinstance(repository, Mapping)
        or not isinstance(static_contract, Mapping)
        or not isinstance(addendum, Path)
    ):
        raise P23AcquisitionError("prepared acquisition context is malformed")

    with tempfile.TemporaryDirectory(prefix="p23-acquisition-") as temporary:
        temporary_root = Path(temporary)
        p22_manifest_path = temporary_root / "p22-manifest.json"
        p22_raw_path = temporary_root / "p22-raw-trace.json" if trace_on else None
        p22_payload = P22_RUNNER.run_trace(
            trace_mode=trace_mode,
            nanogpt_root=nanogpt_root.resolve(),
            muon_source=muon_source.resolve(),
            data_manifest=data_manifest.resolve(),
            instrumentation_patch=instrumentation_patch.resolve(),
            backend="cuda",
            output=p22_manifest_path,
            raw_trace_output=p22_raw_path,
            _lifecycle_callback=lifecycle,
        )
        validated_p22_payload = P22_CORE._load_run_manifest(p22_manifest_path, trace_mode)
        if validated_p22_payload != p22_payload:
            raise P23AcquisitionError(
                "P22 acquisition return value differs from its validated stored manifest"
            )
        if (
            initial_sources is None
            or final_sources is None
            or initial_loaded_files is None
            or final_loaded_files is None
            or loaded_file_closure is None
        ):
            raise P23AcquisitionError("P22 runner did not complete both lifecycle events")
        execution_identity = P23.build_execution_identity(
            addendum_path=addendum,
            repository=repository,
            source_modules=initial_sources,
            loaded_file_closure=loaded_file_closure,
            runtime=runtime,
            static_contract=static_contract,
            repository_root=REPOSITORY_ROOT,
        )
        manifest = _enrich_p22_payload(
            p22_payload,
            role=role,
            execution_identity=execution_identity,
            runtime=runtime,
            process_instance=process_instance,
        )
        if trace_on:
            assert p22_raw_path is not None
            _validate_raw_trace_binding(manifest, raw_path_override=p22_raw_path)
        P23.validate_run_manifest(manifest, trace_mode)
        _verify_manifest_identity(
            manifest,
            expected_mode=trace_mode,
            expected_role=role,
            context=context,
            source_modules=initial_sources,
            loaded_file_initialized=initial_loaded_files,
        )
        if trace_on:
            assert p22_raw_path is not None and raw_trace_output is not None
            raw_trace = _read_mapping(p22_raw_path, "P22 raw trace")
            P22_CORE.write_json_atomic(raw_trace_output, raw_trace)
            manifest["raw_trace"] = {
                "path": str(raw_trace_output.resolve()),
                "sha256": P23.sha256_file(raw_trace_output.resolve()),
                "observation_count": P23.EXPECTED_OBSERVATION_COUNT,
            }
            _validate_raw_trace_binding(manifest)
            P23.validate_run_manifest(manifest, trace_mode)
        P22_CORE.write_json_atomic(output, manifest)
    return manifest


def _require_exact_keys(value: object, expected: set[str], label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or set(value) != expected:
        raise P23AcquisitionError(f"{label} field set changed")
    return value


def _finite_nonnegative(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise P23AcquisitionError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise P23AcquisitionError(f"{label} is not finite and nonnegative")
    return result


def _optional_ratio_from_scalars(numerator: float, denominator: float) -> tuple[float | None, str]:
    if denominator != 0.0:
        return numerator / denominator, "finite_denominator"
    return (None, "both_zero" if numerator == 0.0 else "nonzero_over_zero")


def _expected_raw_parameter_inventory() -> dict[str, Mapping[str, object]]:
    return {str(record["name"]): record for record in P22_CORE.expected_gpt2_small_muon_inventory()}


def _validate_raw_observation(
    observation: object,
    *,
    expected_step: int,
    inventory: Mapping[str, Mapping[str, object]],
) -> str:
    """Validate all retained metadata and all scalar relations that remain recomputable."""

    record = _require_exact_keys(
        observation,
        {
            "schema_version",
            "evidence_kind",
            "optimizer_step",
            "tokens_seen",
            "phase",
            "seed",
            "parameter",
            "optimizer",
            "storage",
            "shield",
            "metrics",
            "per_observation_gates",
            "observer_rng_sha256",
        },
        "raw observation",
    )
    if (
        record.get("schema_version") != P22_OBSERVER.P22_SHADOW_OBSERVATION_SCHEMA_VERSION
        or record.get("evidence_kind") != "real_gradient_shadow_observation"
        or record.get("optimizer_step") != expected_step
        or record.get("tokens_seen") != (expected_step + 1) * P23.EXPECTED_SEQUENCE_LENGTH
        or record.get("phase") != P21_TRACE.phase_for_step(expected_step)
        or record.get("seed") != P22_RUNNER.SEED
        or not isinstance(record.get("observer_rng_sha256"), str)
        or _SHA256.fullmatch(str(record.get("observer_rng_sha256"))) is None
    ):
        raise P23AcquisitionError("raw observation identity, phase, or schedule changed")

    parameter = _require_exact_keys(
        record.get("parameter"),
        {
            "name",
            "layer",
            "role",
            "original_shape",
            "shield_shape",
            "transposed_for_shield",
            "entry_count",
        },
        "raw observation parameter",
    )
    name = parameter.get("name")
    if not isinstance(name, str) or name not in inventory:
        raise P23AcquisitionError("raw observation has an undeclared parameter name")
    expected_parameter = inventory[name]
    match = P22_RUNNER.MUON_NAME.fullmatch(name)
    if match is None:
        raise P23AcquisitionError("raw observation parameter name is not a frozen Muon matrix")
    expected_parameter_fields = {
        "name": name,
        "layer": int(match.group("layer")),
        "role": P22_RUNNER.ROLE_NAMES[match.group("role")],
        "original_shape": expected_parameter.get("stored_shape"),
        "shield_shape": expected_parameter.get("shield_shape"),
        "transposed_for_shield": expected_parameter.get("transposed_for_shield"),
        "entry_count": expected_parameter.get("numel"),
    }
    if dict(parameter) != expected_parameter_fields:
        raise P23AcquisitionError(f"raw observation metadata changed for {name}")

    optimizer = _require_exact_keys(
        record.get("optimizer"),
        {
            "beta",
            "learning_rate",
            "weight_decay",
            "upstream_aspect_factor",
            "candidate_captured_after_accelerator_aspect",
            "aspect_applied_by_observer",
        },
        "raw observation optimizer",
    )
    stored_shape = tuple(int(value) for value in expected_parameter["stored_shape"])  # type: ignore[arg-type]
    if dict(optimizer) != {
        "beta": P22_RUNNER.MUON_MOMENTUM,
        "learning_rate": P22_RUNNER.MUON_LEARNING_RATE,
        "weight_decay": 0.0,
        "upstream_aspect_factor": P21_TRACE.upstream_aspect_factor(stored_shape),
        "candidate_captured_after_accelerator_aspect": True,
        "aspect_applied_by_observer": False,
    }:
        raise P23AcquisitionError(f"raw observation optimizer contract changed for {name}")

    storage = _require_exact_keys(
        record.get("storage"),
        {
            "signal_dtype",
            "raw_candidate_dtype",
            "aspect_candidate_dtype",
            "output_dtype",
            "source_device",
            "device",
            "raw_candidate_finite",
            "aspect_candidate_finite",
            "signal_sha256",
            "raw_candidate_sha256",
            "aspect_candidate_sha256",
            "output_sha256",
        },
        "raw observation storage",
    )
    if {
        key: storage.get(key)
        for key in (
            "signal_dtype",
            "raw_candidate_dtype",
            "aspect_candidate_dtype",
            "output_dtype",
            "source_device",
            "device",
            "raw_candidate_finite",
            "raw_candidate_sha256",
        )
    } != {
        "signal_dtype": "torch.float32",
        "raw_candidate_dtype": None,
        "aspect_candidate_dtype": "torch.bfloat16",
        "output_dtype": "torch.float32",
        "source_device": "cuda",
        "device": "cpu",
        "raw_candidate_finite": None,
        "raw_candidate_sha256": None,
    }:
        raise P23AcquisitionError(f"raw observation storage contract changed for {name}")
    if storage.get("aspect_candidate_finite") is not True:
        raise P23AcquisitionError(
            "P23 aborts and discards an acquisition with a nonfinite CUDA candidate"
        )
    for digest_field in ("signal_sha256", "aspect_candidate_sha256", "output_sha256"):
        digest = storage.get(digest_field)
        if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
            raise P23AcquisitionError(f"raw observation has an invalid {digest_field}")

    shield = _require_exact_keys(
        record.get("shield"),
        {
            "action",
            "active",
            "sector_certified_by_successful_p20_call",
            "sector_certificate_scope",
            "signal_guard",
            "reason",
            "dead_zone_zero",
            "dead_zone_update_disturbance_upper",
        },
        "raw observation shield",
    )
    actions = {"pass_through", "radial_clip", "half_fallback", "zero_singleton", "dead_zone_zero"}
    guards = {"zero", "normal_anchored", "subnormal_exact_halving", "subnormal_unrepresentable"}
    if (
        shield.get("action") not in actions
        or shield.get("signal_guard") not in guards
        or not isinstance(shield.get("active"), bool)
        or not isinstance(shield.get("sector_certified_by_successful_p20_call"), bool)
        or not isinstance(shield.get("dead_zone_zero"), bool)
        or shield.get("sector_certificate_scope") != "P20 plus exact P22 768x2304 extension"
    ):
        raise P23AcquisitionError(f"raw observation shield contract changed for {name}")

    metrics = _require_exact_keys(
        record.get("metrics"),
        {
            "signal_norm",
            "raw_candidate_norm",
            "aspect_candidate_norm",
            "output_norm",
            "correction_norm",
            "relative_correction",
            "relative_correction_status",
            "candidate_output_cosine",
            "candidate_output_cosine_status",
            "output_to_signal_amplitude",
            "output_to_signal_status",
            "output_to_candidate_amplitude",
            "output_to_candidate_status",
            "candidate_best_scalar_departure",
            "output_best_scalar_departure",
            "shaping_retention",
            "informative_candidate",
            "in_frozen_operating_annulus",
            "primary_eta_1_over_120_effective_gain",
            "secondary_eta_1_over_83_effective_gain",
            "observed_training_step_norm",
            "primary_theorem_step_norm",
        },
        "raw observation metrics",
    )
    signal_norm = _finite_nonnegative(metrics.get("signal_norm"), "signal norm")
    output_norm = _finite_nonnegative(metrics.get("output_norm"), "output norm")
    candidate_finite = storage["aspect_candidate_finite"] is True
    if metrics.get("raw_candidate_norm") is not None:
        raise P23AcquisitionError("raw pre-aspect norm must remain unobserved")
    if candidate_finite:
        candidate_norm = _finite_nonnegative(
            metrics.get("aspect_candidate_norm"), "aspect candidate norm"
        )
        correction_norm = _finite_nonnegative(
            metrics.get("correction_norm"), "candidate correction norm"
        )
    else:
        candidate_norm = 0.0
        correction_norm = 0.0
        if (
            metrics.get("aspect_candidate_norm") is not None
            or metrics.get("correction_norm") is not None
        ):
            raise P23AcquisitionError("nonfinite candidate has finite-only norm diagnostics")

    ratios = (
        ("output_to_signal_amplitude", "output_to_signal_status", output_norm, signal_norm),
        (
            "output_to_candidate_amplitude",
            "output_to_candidate_status",
            output_norm,
            candidate_norm,
        ),
        (
            "relative_correction",
            "relative_correction_status",
            correction_norm,
            candidate_norm,
        ),
    )
    for value_field, status_field, numerator, denominator in ratios:
        expected_value, expected_status = _optional_ratio_from_scalars(numerator, denominator)
        if (
            metrics.get(value_field) != expected_value
            or metrics.get(status_field) != expected_status
        ):
            raise P23AcquisitionError(f"raw observation {value_field} relation changed")

    cosine = metrics.get("candidate_output_cosine")
    cosine_status = metrics.get("candidate_output_cosine_status")
    if not candidate_finite:
        if cosine is not None or cosine_status != "nonfinite_candidate":
            raise P23AcquisitionError("nonfinite candidate cosine status changed")
    elif candidate_norm == 0.0 or output_norm == 0.0:
        expected_status = "both_zero" if candidate_norm == output_norm == 0.0 else "one_zero"
        if cosine is not None or cosine_status != expected_status:
            raise P23AcquisitionError("zero-norm candidate cosine status changed")
    elif (
        isinstance(cosine, bool)
        or not isinstance(cosine, (int, float))
        or not math.isfinite(float(cosine))
        or not -1.0 <= float(cosine) <= 1.0
        or cosine_status != "finite_denominator"
    ):
        raise P23AcquisitionError("finite candidate cosine is malformed")

    candidate_departure = metrics.get("candidate_best_scalar_departure")
    output_departure = metrics.get("output_best_scalar_departure")
    if not candidate_finite:
        if candidate_departure is not None:
            raise P23AcquisitionError("nonfinite candidate has a scalar-departure diagnostic")
    else:
        candidate_departure = _finite_nonnegative(candidate_departure, "candidate departure")
        if candidate_departure > 1.0:
            raise P23AcquisitionError("candidate departure exceeds one")
    output_departure = _finite_nonnegative(output_departure, "output departure")
    if output_departure > 1.0:
        raise P23AcquisitionError("output departure exceeds one")
    informative = bool(
        candidate_departure is not None
        and candidate_departure >= P21_TRACE.INFORMATIVE_CANDIDATE_DEPARTURE
    )
    if metrics.get("informative_candidate") is not informative:
        raise P23AcquisitionError("raw observation informative-candidate flag changed")
    shaping_retention = output_departure / candidate_departure if informative else None
    if metrics.get("shaping_retention") != shaping_retention:
        raise P23AcquisitionError("raw observation shaping-retention relation changed")
    in_annulus = (
        float(P21_TRACE.ANNULUS_SIGNAL_NORM_LOWER)
        <= signal_norm
        <= float(P21_TRACE.ANNULUS_SIGNAL_NORM_UPPER)
    )
    if metrics.get("in_frozen_operating_annulus") is not in_annulus:
        raise P23AcquisitionError("raw observation annulus classification changed")

    output_to_signal = metrics.get("output_to_signal_amplitude")
    primary_gain = (
        float(P21_TRACE.PRIMARY_LEARNING_RATE) * output_to_signal
        if isinstance(output_to_signal, (int, float)) and not isinstance(output_to_signal, bool)
        else None
    )
    secondary_gain = (
        float(P21_TRACE.SECONDARY_LEARNING_RATE) * output_to_signal
        if isinstance(output_to_signal, (int, float)) and not isinstance(output_to_signal, bool)
        else None
    )
    exact_derived_metrics = {
        "primary_eta_1_over_120_effective_gain": primary_gain,
        "secondary_eta_1_over_83_effective_gain": secondary_gain,
        "observed_training_step_norm": P22_RUNNER.MUON_LEARNING_RATE * output_norm,
        "primary_theorem_step_norm": float(P21_TRACE.PRIMARY_LEARNING_RATE) * output_norm,
    }
    if any(metrics.get(field) != value for field, value in exact_derived_metrics.items()):
        raise P23AcquisitionError("raw observation effective-step relation changed")

    dead_zone = shield["dead_zone_zero"] is True
    if dead_zone:
        if (
            shield.get("action") != "dead_zone_zero"
            or shield.get("active") is not True
            or shield.get("sector_certified_by_successful_p20_call") is not False
            or shield.get("signal_guard") != "subnormal_unrepresentable"
            or not isinstance(shield.get("reason"), str)
            or not shield.get("reason")
            or shield.get("dead_zone_update_disturbance_upper") != 0.5 * signal_norm
            or output_norm != 0.0
        ):
            raise P23AcquisitionError("raw observation dead-zone relation changed")
    elif (
        shield.get("action") == "dead_zone_zero"
        or shield.get("sector_certified_by_successful_p20_call") is not True
        or shield.get("dead_zone_update_disturbance_upper") is not None
        or shield.get("active") is not (shield.get("action") != "pass_through")
    ):
        raise P23AcquisitionError("raw observation successful-shield relation changed")

    gates = _require_exact_keys(
        record.get("per_observation_gates"),
        {
            "successful_p20_sector_call",
            "frozen_fidelity",
            "frozen_annulus_output_to_candidate",
            "frozen_output_to_signal",
            "frozen_primary_effective_gain",
        },
        "raw observation gates",
    )
    output_to_candidate = metrics.get("output_to_candidate_amplitude")
    expected_gates = {
        "successful_p20_sector_call": shield.get("sector_certified_by_successful_p20_call"),
        "frozen_fidelity": (
            output_departure >= P21_TRACE.MINIMUM_OUTPUT_DEPARTURE
            and shaping_retention is not None
            and shaping_retention >= P21_TRACE.MINIMUM_SHAPING_RETENTION
            if informative
            else None
        ),
        "frozen_annulus_output_to_candidate": (
            output_to_candidate is not None
            and float(P21_TRACE.ANNULUS_OUTPUT_TO_CANDIDATE_LOWER)
            <= output_to_candidate
            <= float(P21_TRACE.ANNULUS_OUTPUT_TO_CANDIDATE_UPPER)
            if in_annulus
            else None
        ),
        "frozen_output_to_signal": (
            output_to_signal is not None
            and float(P21_TRACE.OUTPUT_TO_SIGNAL_LOWER)
            <= output_to_signal
            <= float(P21_TRACE.OUTPUT_TO_SIGNAL_UPPER)
        ),
        "frozen_primary_effective_gain": (
            primary_gain is not None
            and float(P21_TRACE.PRIMARY_EFFECTIVE_GAIN_LOWER)
            <= primary_gain
            <= float(P21_TRACE.PRIMARY_EFFECTIVE_GAIN_UPPER)
        ),
    }
    if dict(gates) != expected_gates:
        raise P23AcquisitionError("raw observation frozen gate relations changed")
    return name


def _validate_raw_trace_payload(raw: Mapping[str, object]) -> None:
    if set(raw) != {
        "schema_version",
        "evidence_kind",
        "protocol_sha256",
        "run_identity_sha256",
        "capture_steps",
        "observation_count",
        "observations",
    }:
        raise P23AcquisitionError("raw-trace field set changed")
    if raw.get("schema_version") != "passive-muon-p22-raw-real-gradient-trace-v1":
        raise P23AcquisitionError("raw-trace schema mismatch")
    if raw.get("evidence_kind") != "real_gradient_shadow_trace":
        raise P23AcquisitionError("raw trace is not real-gradient shadow evidence")
    if raw.get("protocol_sha256") != P22_CORE.protocol_sha256():
        raise P23AcquisitionError("raw trace does not use the current frozen P22 protocol")
    run_identity = raw.get("run_identity_sha256")
    if not isinstance(run_identity, str) or _SHA256.fullmatch(run_identity) is None:
        raise P23AcquisitionError("raw trace has an invalid P22 run identity")
    if raw.get("capture_steps") != list(P23.EXPECTED_CAPTURE_STEPS):
        raise P23AcquisitionError("raw-trace capture schedule changed")
    if raw.get("observation_count") != P23.EXPECTED_OBSERVATION_COUNT:
        raise P23AcquisitionError("raw trace does not declare exactly 1,152 observations")
    observations = raw.get("observations")
    if not isinstance(observations, list) or len(observations) != P23.EXPECTED_OBSERVATION_COUNT:
        raise P23AcquisitionError("raw trace does not contain exactly 1,152 observations")
    inventory = _expected_raw_parameter_inventory()
    for capture_index, step in enumerate(P23.EXPECTED_CAPTURE_STEPS):
        start = capture_index * P23.EXPECTED_PARAMETER_COUNT
        stop = start + P23.EXPECTED_PARAMETER_COUNT
        names = [
            _validate_raw_observation(
                observation,
                expected_step=step,
                inventory=inventory,
            )
            for observation in observations[start:stop]
        ]
        if set(names) != set(inventory) or len(names) != len(set(names)):
            raise P23AcquisitionError("raw capture does not cover the exact 48-name inventory")


def _validate_artifact_record(
    record: object, *, require_identity: bool
) -> tuple[Mapping[str, object], dict[str, Any]]:
    expected_fields = {"path", "sha256"}
    if require_identity:
        expected_fields.add("run_identity_sha256")
    artifact = _require_exact_keys(record, expected_fields, "artifact record")
    raw_path = artifact.get("path")
    digest = artifact.get("sha256")
    if not isinstance(raw_path, str) or not Path(raw_path).is_absolute():
        raise P23AcquisitionError("native artifact record path must be absolute")
    if not isinstance(digest, str) or _SHA256.fullmatch(digest) is None:
        raise P23AcquisitionError("native artifact record has an invalid digest")
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise P23AcquisitionError("native artifact record does not match retained bytes")
    linked, raw_bytes = _read_mapping_bytes(path, "linked native artifact")
    if hashlib.sha256(raw_bytes).hexdigest() != digest:
        raise P23AcquisitionError("native artifact record does not match retained bytes")
    if require_identity:
        identity = artifact.get("run_identity_sha256")
        if not isinstance(identity, str) or _SHA256.fullmatch(identity) is None:
            raise P23AcquisitionError("native artifact record has an invalid run identity")
        if linked.get("run_identity_sha256") != identity:
            raise P23AcquisitionError("native artifact record run identity does not match bytes")
    return artifact, linked


def _validate_gate_report_payload(report: Mapping[str, object], *, expected_schema: str) -> None:
    repeatability = expected_schema == P23.P23_REPEATABILITY_SCHEMA
    blocked_noninterference = (
        not repeatability and report.get("comparison") == "blocked_by_failed_exact_repeatability"
    )
    expected_fields = {
        "schema_version",
        "passes",
        "comparison",
        "mismatch_count",
        "mismatches",
        "binding_schema",
        "inputs",
    }
    if not blocked_noninterference:
        expected_fields.add("identity_verification")
    if not repeatability and not blocked_noninterference:
        expected_fields.update({"observation_count", "repeatability_gate"})
    elif blocked_noninterference:
        expected_fields.add("repeatability_gate")
    _require_exact_keys(report, expected_fields, "gate report")
    if (
        report.get("schema_version") != expected_schema
        or report.get("binding_schema") != P23_GATE_REPORT_BINDING_SCHEMA
    ):
        raise P23AcquisitionError("P23 gate-report schema or artifact binding changed")
    if not blocked_noninterference and report.get("comparison") != ("bitwise_exact_no_tolerances"):
        raise P23AcquisitionError("P23 gate-report comparison rule changed")
    input_names = (
        {"trace_off_a", "trace_off_b"}
        if repeatability
        else {"trace_off_a", "trace_off_b", "trace_on"}
    )
    inputs = _require_exact_keys(report.get("inputs"), input_names, "gate-report inputs")
    verification = (
        None
        if blocked_noninterference
        else _require_exact_keys(
            report.get("identity_verification"),
            input_names,
            "gate-report identity verification",
        )
    )
    linked_manifests: dict[str, dict[str, object]] = {}
    for name in input_names:
        _artifact, linked = _validate_artifact_record(inputs[name], require_identity=True)
        linked_manifests[name] = linked
        expected_mode = "trace_on" if name == "trace_on" else "trace_off"
        expected_role = name
        P23.validate_run_manifest(linked, expected_mode)
        _validate_process_instance(linked, expected_role)
        identity = linked.get("execution_identity")
        if not isinstance(identity, Mapping):
            raise P23AcquisitionError("linked run manifest has no complete execution identity")
        source_modules = identity.get("source_modules")
        if not isinstance(source_modules, Mapping):
            raise P23AcquisitionError("linked run manifest has no source-module inventory")
        loaded_file_closure = identity.get("loaded_file_closure")
        if not isinstance(loaded_file_closure, Mapping):
            raise P23AcquisitionError("linked run manifest has no loaded-file closure")
        closure = P23.validate_loaded_file_closure(loaded_file_closure)
        initialized = closure["initialized"]
        completed = closure["completed"]
        assert isinstance(initialized, Mapping) and isinstance(completed, Mapping)
        unique_loaded_paths: set[str] = set()
        for snapshot in (initialized, completed):
            modules = snapshot["modules"]
            mapped_files = snapshot["mapped_files"]
            assert isinstance(modules, Mapping) and isinstance(mapped_files, Mapping)
            unique_loaded_paths.update(str(path) for path in mapped_files)
            for module_record in modules.values():
                assert isinstance(module_record, Mapping)
                artifacts = module_record["artifacts"]
                assert isinstance(artifacts, Mapping)
                unique_loaded_paths.update(
                    str(record["logical_path"])
                    for record in artifacts.values()
                    if isinstance(record, Mapping)
                )
        expected_verification = {
            "passes": True,
            "comparison": "independently_recomputed_exact_identity",
            "run_identity_sha256": linked.get("run_identity_sha256"),
            "source_module_count": len(source_modules),
            "loaded_file_verification": {
                "passes": True,
                "comparison": ("initialized_membership_recollected_all_retained_bytes_rehashed"),
                "initialized_module_count": initialized["module_count"],
                "completed_module_count": completed["module_count"],
                "initialized_mapped_file_count": initialized["mapped_file_count"],
                "completed_mapped_file_count": completed["mapped_file_count"],
                "unique_file_count": len(unique_loaded_paths),
            },
        }
        if verification is not None and verification[name] != expected_verification:
            raise P23AcquisitionError("gate report identity-verification record changed")
    if blocked_noninterference:
        embedded = report.get("repeatability_gate")
        if not isinstance(embedded, Mapping):
            raise P23AcquisitionError("blocked noninterference report has no repeatability gate")
        _validate_gate_report_payload(
            embedded,
            expected_schema=P23.P23_REPEATABILITY_SCHEMA,
        )
        if embedded.get("passes") is not False:
            raise P23AcquisitionError(
                "blocked noninterference report does not contain a failed repeatability gate"
            )
        expected_blocked = {
            "schema_version": P23.P23_NONINTERFERENCE_SCHEMA,
            "passes": False,
            "comparison": "blocked_by_failed_exact_repeatability",
            "mismatch_count": embedded.get("mismatch_count"),
            "mismatches": embedded.get("mismatches"),
            "binding_schema": P23_GATE_REPORT_BINDING_SCHEMA,
            "inputs": dict(inputs),
            "repeatability_gate": dict(embedded),
        }
        if dict(report) != expected_blocked:
            raise P23AcquisitionError(
                "blocked noninterference report differs from exact repeatability evidence"
            )
        if embedded.get("inputs") != {
            "trace_off_a": inputs["trace_off_a"],
            "trace_off_b": inputs["trace_off_b"],
        }:
            raise P23AcquisitionError("nested repeatability gate binds different trace-off runs")
        return
    comparison = (
        P23.compare_repeatability_manifests(
            linked_manifests["trace_off_a"],
            linked_manifests["trace_off_b"],
        )
        if repeatability
        else P23.compare_noninterference_manifests(
            linked_manifests["trace_off_a"],
            linked_manifests["trace_on"],
        )
    )
    if any(report.get(field) != value for field, value in comparison.items()):
        raise P23AcquisitionError("gate report differs from exact linked-manifest replay")
    if not repeatability:
        if report.get("observation_count") != P23.EXPECTED_OBSERVATION_COUNT:
            raise P23AcquisitionError("noninterference report observation count changed")
        embedded = report.get("repeatability_gate")
        if not isinstance(embedded, Mapping):
            raise P23AcquisitionError("noninterference report has no repeatability gate")
        _validate_gate_report_payload(
            embedded,
            expected_schema=P23.P23_REPEATABILITY_SCHEMA,
        )
        if embedded.get("inputs") != {
            "trace_off_a": inputs["trace_off_a"],
            "trace_off_b": inputs["trace_off_b"],
        }:
            raise P23AcquisitionError("nested repeatability gate binds different trace-off runs")


def _validate_aggregate_payload(payload: Mapping[str, object]) -> None:
    p23 = _require_exact_keys(
        payload.get("p23"),
        {
            "schema_version",
            "cuda_only",
            "repeatability_report",
            "noninterference_report",
            "trace_on_manifest",
            "repeatability_report_replayed_exactly",
            "noninterference_report_replayed_exactly",
            "repeatability_passes",
            "noninterference_passes",
            "metric_reconstruction_scope",
        },
        "P23 aggregate extension",
    )
    if p23 != {
        **dict(p23),
        "schema_version": P23_AGGREGATE_SCHEMA,
        "cuda_only": True,
        "repeatability_report_replayed_exactly": True,
        "noninterference_report_replayed_exactly": True,
        "repeatability_passes": True,
        "noninterference_passes": True,
        "metric_reconstruction_scope": METRIC_RECONSTRUCTION_SCOPE,
    }:
        raise P23AcquisitionError("P23 aggregate gate flags or disclosure changed")
    repeatability_record, repeatability = _validate_artifact_record(
        p23["repeatability_report"], require_identity=False
    )
    _noninterference_record, noninterference = _validate_artifact_record(
        p23["noninterference_report"], require_identity=False
    )
    trace_on_record, trace_on = _validate_artifact_record(
        p23["trace_on_manifest"], require_identity=True
    )
    _validate_gate_report_payload(
        repeatability,
        expected_schema=P23.P23_REPEATABILITY_SCHEMA,
    )
    _validate_gate_report_payload(
        noninterference,
        expected_schema=P23.P23_NONINTERFERENCE_SCHEMA,
    )
    P23.validate_run_manifest(trace_on, "trace_on")
    _validate_process_instance(trace_on, "trace_on")
    raw = _validate_raw_trace_binding(trace_on)

    repeatability_inputs = repeatability.get("inputs")
    noninterference_inputs = noninterference.get("inputs")
    if (
        noninterference.get("repeatability_gate") != repeatability
        or not isinstance(repeatability_inputs, Mapping)
        or not isinstance(noninterference_inputs, Mapping)
        or {name: noninterference_inputs.get(name) for name in ("trace_off_a", "trace_off_b")}
        != dict(repeatability_inputs)
        or noninterference_inputs.get("trace_on") != trace_on_record
        or p23.get("repeatability_report") != repeatability_record
    ):
        raise P23AcquisitionError("P23 aggregate splices inconsistent gate/run artifacts")

    observations = payload.get("observations")
    if not isinstance(observations, list):
        raise P23AcquisitionError("P23 aggregate has no retained observation sequence")
    if observations != raw.get("observations"):
        raise P23AcquisitionError("P23 aggregate observations differ from the hash-bound raw trace")
    raw_view = {
        "schema_version": "passive-muon-p22-raw-real-gradient-trace-v1",
        "evidence_kind": "real_gradient_shadow_trace",
        "protocol_sha256": P22_CORE.protocol_sha256(),
        "run_identity_sha256": trace_on["p22_acquisition"]["run_identity_sha256"],  # type: ignore[index]
        "capture_steps": list(P23.EXPECTED_CAPTURE_STEPS),
        "observation_count": len(observations),
        "observations": observations,
    }
    _validate_raw_trace_payload(raw_view)
    intended = {
        str(record["name"]): int(record["numel"])
        for record in P22_CORE.expected_gpt2_small_muon_inventory()
    }
    provenance = payload.get("provenance")
    if not isinstance(provenance, Mapping):
        raise P23AcquisitionError("P23 aggregate has no provenance mapping")
    raw_entry = trace_on.get("raw_trace")
    data = trace_on.get("data")
    if not isinstance(raw_entry, Mapping) or not isinstance(data, Mapping):
        raise P23AcquisitionError("linked trace-on manifest has incomplete raw/data provenance")
    expected_provenance = {
        "p22_protocol_sha256": P22_CORE.protocol_sha256(),
        "run_manifest_path": str(Path(str(trace_on_record["path"])).resolve()),
        "run_manifest_sha256": trace_on_record["sha256"],
        "raw_trace_path": str(Path(str(raw_entry.get("path"))).resolve()),
        "raw_trace_sha256": raw_entry.get("sha256"),
        "source_snapshot": trace_on.get("source_snapshot"),
        "runtime": trace_on.get("runtime"),
        "data_manifest_sha256": data.get("manifest_sha256"),
        "post_aspect_candidate_captured_from_accelerator": True,
        "aspect_recomputed_by_cpu_observer": False,
    }
    if dict(provenance) != expected_provenance:
        raise P23AcquisitionError("P23 aggregate provenance differs from linked trace-on bytes")
    try:
        expected = P21_TRACE.summarize_shadow_trace(
            observations,
            evidence_kind="real_gradient_shadow_trace",
            provenance=provenance,
            intended_parameter_numel=intended,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise P23AcquisitionError(
            f"P23 aggregate observation reconstruction failed: {error}"
        ) from error
    expected["p22_extension"] = {
        "schema_version": "passive-muon-p22-real-gradient-summary-v1",
        "post_aspect_candidate_captured_from_accelerator": True,
        "shape_768x2304_separately_certified": True,
        "early_middle_late_are_step_windows_not_mature_training_phases": True,
    }
    expected["p23"] = copy.deepcopy(dict(p23))
    if dict(payload) != expected:
        raise P23AcquisitionError("P23 aggregate differs from exact scalar reconstruction")


def _validate_sanitizable_artifact(payload: Mapping[str, object]) -> None:
    addendum = P23.load_and_validate_addendum(
        P23.DEFAULT_ADDENDUM_PATH,
        repository_root=REPOSITORY_ROOT,
    )
    _validate_frozen_fidelity_gate_constants(addendum=addendum)
    _validate_frozen_p22_acquisition_constants(addendum=addendum)
    schema = payload.get("schema_version")
    if schema == P23.P23_RUN_MANIFEST_SCHEMA:
        trace_mode = payload.get("trace_mode")
        if trace_mode not in {"trace_off", "trace_on"}:
            raise P23AcquisitionError("P23 run manifest has no valid trace mode")
        P23.validate_run_manifest(payload, str(trace_mode))
        role = payload.get("acquisition_role")
        if not isinstance(role, str) or role not in RUN_ROLES:
            raise P23AcquisitionError("native P23 run manifest has no valid acquisition role")
        _validate_process_instance(payload, role)
        if payload.get("metric_reconstruction_scope") != METRIC_RECONSTRUCTION_SCOPE:
            raise P23AcquisitionError("P23 run manifest omits the metric reconstruction boundary")
        if payload.get("p22_acquisition") != _p22_identity_binding(payload):
            raise P23AcquisitionError("P23 run manifest does not reconstruct its P22 identity")
        _validate_reconstructed_p22_manifest(payload, str(trace_mode))
        if trace_mode == "trace_on":
            _validate_raw_trace_binding(payload)
        return
    if schema == "passive-muon-p22-raw-real-gradient-trace-v1":
        _validate_raw_trace_payload(payload)
        return
    if schema in {P23.P23_REPEATABILITY_SCHEMA, P23.P23_NONINTERFERENCE_SCHEMA}:
        _validate_gate_report_payload(payload, expected_schema=str(schema))
        return
    if schema == P21_TRACE.P21_SHADOW_TRACE_SCHEMA_VERSION and "p23" in payload:
        _validate_aggregate_payload(payload)
        return
    raise P23AcquisitionError("sanitize accepts only one of the seven native P23 artifacts")


def _validate_raw_trace_binding(
    manifest: Mapping[str, object], *, raw_path_override: Path | None = None
) -> dict[str, object]:
    """Bind every deeply validated raw observation chunk to its manifest capture."""

    raw_entry = manifest.get("raw_trace")
    if not isinstance(raw_entry, Mapping):
        raise P23AcquisitionError("trace-on manifest has no raw-trace record")
    raw_path = (
        raw_path_override.resolve()
        if raw_path_override is not None
        else Path(str(raw_entry.get("path"))).resolve()
    )
    if not raw_path.is_file():
        raise P23AcquisitionError("raw-trace bytes do not match the manifest digest")
    raw, raw_bytes = _read_mapping_bytes(raw_path, "raw candidate trace")
    if hashlib.sha256(raw_bytes).hexdigest() != raw_entry.get("sha256"):
        raise P23AcquisitionError("raw-trace bytes do not match the manifest digest")
    _validate_raw_trace_payload(raw)
    if raw.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise P23AcquisitionError("raw trace does not bind the P22 protocol")
    p22_acquisition = manifest.get("p22_acquisition")
    if not isinstance(p22_acquisition, Mapping) or raw.get("run_identity_sha256") != (
        p22_acquisition.get("run_identity_sha256")
    ):
        raise P23AcquisitionError("raw trace does not bind the underlying P22 run identity")
    observations = raw.get("observations")
    assert isinstance(observations, list)
    steps = manifest.get("steps")
    shape_inventory = manifest.get("shape_inventory")
    if not isinstance(steps, list) or not isinstance(shape_inventory, Mapping):
        raise P23AcquisitionError("trace-on manifest has no step/shape inventory")
    intended = shape_inventory.get("parameters")
    if not isinstance(intended, Mapping):
        raise P23AcquisitionError("trace-on manifest has no named shape inventory")
    inventory = _expected_raw_parameter_inventory()
    intended_names = set(inventory)
    if dict(intended) != {name: record["numel"] for name, record in inventory.items()}:
        raise P23AcquisitionError("trace-on manifest does not carry the frozen shape inventory")

    for capture_index, step in enumerate(P23.EXPECTED_CAPTURE_STEPS):
        start = capture_index * P23.EXPECTED_PARAMETER_COUNT
        stop = start + P23.EXPECTED_PARAMETER_COUNT
        chunk = observations[start:stop]
        names: list[str] = []
        for observation in chunk:
            names.append(
                _validate_raw_observation(
                    observation,
                    expected_step=step,
                    inventory=inventory,
                )
            )
        if set(names) != intended_names or len(names) != len(set(names)):
            raise P23AcquisitionError("raw capture does not cover the exact 48-name inventory")
        step_record = steps[step]
        if not isinstance(step_record, Mapping) or not isinstance(
            step_record.get("capture"), Mapping
        ):
            raise P23AcquisitionError("manifest has no matching capture record")
        capture = step_record["capture"]
        assert isinstance(capture, Mapping)
        if capture.get("record_sha256") != P22_CORE.canonical_state_sha256(chunk):
            raise P23AcquisitionError("raw observation chunk does not match its capture digest")
    return raw


def aggregate(
    *,
    trace_off_a_path: Path,
    trace_off_b_path: Path,
    trace_on_path: Path,
    repeatability_report_path: Path,
    noninterference_report_path: Path,
    nanogpt_root: Path,
    muon_source: Path,
    data_manifest: Path,
    instrumentation_patch: Path,
    runtime_lock_path: Path,
    host_attestation_path: Path,
    addendum_path: Path,
    output: Path,
) -> dict[str, object]:
    """Aggregate fidelity only after reports and fresh exact replays both pass."""

    _assert_native_output_paths(
        outputs={"aggregate": output},
        inputs={
            "trace-off A": trace_off_a_path,
            "trace-off B": trace_off_b_path,
            "trace-on": trace_on_path,
            "repeatability report": repeatability_report_path,
            "noninterference report": noninterference_report_path,
            "nanoGPT": nanogpt_root,
            "Muon": muon_source,
            "FineWeb manifest": data_manifest,
            "observer": instrumentation_patch,
            "runtime lock": runtime_lock_path,
            "host attestation": host_attestation_path,
            "addendum": addendum_path,
        },
        nanogpt_root=nanogpt_root,
        data_manifest=data_manifest,
    )
    context = _prepare_context(
        nanogpt_root=nanogpt_root,
        muon_source=muon_source,
        data_manifest=data_manifest,
        instrumentation_patch=instrumentation_patch,
        runtime_lock_path=runtime_lock_path,
        host_attestation_path=host_attestation_path,
        addendum_path=addendum_path,
    )
    trace_off_a = _read_mapping(trace_off_a_path, "trace-off A manifest")
    trace_off_b = _read_mapping(trace_off_b_path, "trace-off B manifest")
    trace_on = _read_mapping(trace_on_path, "trace-on manifest")
    expected_inputs = {
        "trace_off_a": (trace_off_a_path, trace_off_a),
        "trace_off_b": (trace_off_b_path, trace_off_b),
    }
    repeatability_report = _require_passing_gate_report(
        report_path=repeatability_report_path,
        schema=P23.P23_REPEATABILITY_SCHEMA,
        expected_inputs=expected_inputs,
    )
    noninterference_report = _require_passing_gate_report(
        report_path=noninterference_report_path,
        schema=P23.P23_NONINTERFERENCE_SCHEMA,
        expected_inputs={**expected_inputs, "trace_on": (trace_on_path, trace_on)},
    )
    primed_initialization = _prime_verifier_initialization(context)
    source_modules = _collect_source_modules(context)
    loaded_file_initialized = _collect_loaded_files(context)
    replay_repeatability = _repeatability_report(
        trace_off_a_path=trace_off_a_path,
        trace_off_b_path=trace_off_b_path,
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    replay_noninterference = _noninterference_report(
        trace_off_a_path=trace_off_a_path,
        trace_off_b_path=trace_off_b_path,
        trace_on_path=trace_on_path,
        context=context,
        source_modules=source_modules,
        loaded_file_initialized=loaded_file_initialized,
    )
    if replay_repeatability.get("passes") is not True:
        raise P23AcquisitionError("aggregate is barred by failed exact repeatability")
    if replay_noninterference.get("passes") is not True:
        raise P23AcquisitionError("aggregate is barred by failed exact noninterference")
    _validate_raw_trace_binding(trace_on)

    with tempfile.TemporaryDirectory(prefix="p23-aggregate-") as temporary:
        p22_output = Path(temporary) / "p22-aggregate.json"
        summary = P22_RUNNER.aggregate_trace(run_manifest=trace_on_path, output=p22_output)
    result = copy.deepcopy(dict(summary))
    result["p23"] = {
        "schema_version": P23_AGGREGATE_SCHEMA,
        "cuda_only": True,
        "repeatability_report": _artifact_record(
            repeatability_report_path,
            repeatability_report,
            include_run_identity=False,
        ),
        "noninterference_report": _artifact_record(
            noninterference_report_path,
            noninterference_report,
            include_run_identity=False,
        ),
        "trace_on_manifest": _artifact_record(trace_on_path, trace_on),
        "repeatability_report_replayed_exactly": repeatability_report == replay_repeatability,
        "noninterference_report_replayed_exactly": (
            noninterference_report == replay_noninterference
        ),
        "repeatability_passes": True,
        "noninterference_passes": True,
        "metric_reconstruction_scope": METRIC_RECONSTRUCTION_SCOPE,
    }
    del primed_initialization
    if not result["p23"]["repeatability_report_replayed_exactly"]:
        raise P23AcquisitionError("stored repeatability report differs from fresh exact replay")
    if not result["p23"]["noninterference_report_replayed_exactly"]:
        raise P23AcquisitionError("stored noninterference report differs from fresh exact replay")
    P22_CORE.write_json_atomic(output, result)
    return result


def sanitize(
    *, manifest_path: Path, output: Path, path_roots: Mapping[str, Path]
) -> dict[str, object]:
    _assert_sanitize_output_path(
        manifest_path=manifest_path,
        output=output,
        path_roots=path_roots,
    )
    payload, native_bytes = _read_mapping_bytes(manifest_path, "native P23 artifact")
    native_digest = hashlib.sha256(native_bytes).hexdigest()
    native_byte_count = len(native_bytes)
    _validate_sanitizable_artifact(payload)
    if (
        P23.sha256_file(manifest_path.resolve()) != native_digest
        or manifest_path.resolve().stat().st_size != native_byte_count
    ):
        raise P23AcquisitionError("native artifact changed while it was validated")
    result = P23.sanitize_complete_manifest(payload, path_roots=path_roots)
    result["native_artifact_sha256"] = native_digest
    result["native_artifact_byte_count"] = native_byte_count
    P22_CORE.write_json_atomic(output, result)
    return result


def _add_common_environment_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--nanogpt-root", type=Path, required=True)
    parser.add_argument("--muon-source", type=Path, required=True)
    parser.add_argument("--fineweb-manifest", type=Path, required=True)
    parser.add_argument("--instrumentation-patch", type=Path, required=True)
    parser.add_argument("--runtime-lock", type=Path, required=True)
    parser.add_argument("--host-attestation", type=Path, required=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    freeze = commands.add_parser("freeze-runtime")
    freeze.add_argument("--container-image", required=True)
    freeze.add_argument("--container-repository-digest", required=True)
    freeze.add_argument("--host-image-inspection", type=Path, required=True)
    freeze.add_argument("--host-running-container-inspection", type=Path, required=True)
    freeze.add_argument("--host-running-mountinfo", type=Path, required=True)
    freeze.add_argument("--host-nvidia-smi-query", type=Path, required=True)
    freeze.add_argument("--host-attestation-output", type=Path, required=True)
    freeze.add_argument("--runtime-lock-output", type=Path, required=True)

    run = commands.add_parser("run")
    run.add_argument("--role", choices=RUN_ROLES, required=True)
    _add_common_environment_arguments(run)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--raw-trace-output", type=Path)
    run.add_argument("--trace-off-a", type=Path)
    run.add_argument("--trace-off-b", type=Path)
    run.add_argument("--repeatability-report", type=Path)

    repeatability = commands.add_parser("verify-repeatability")
    _add_common_environment_arguments(repeatability)
    repeatability.add_argument("--trace-off-a", type=Path, required=True)
    repeatability.add_argument("--trace-off-b", type=Path, required=True)
    repeatability.add_argument("--output", type=Path, required=True)

    noninterference = commands.add_parser("verify-noninterference")
    _add_common_environment_arguments(noninterference)
    noninterference.add_argument("--trace-off-a", type=Path, required=True)
    noninterference.add_argument("--trace-off-b", type=Path, required=True)
    noninterference.add_argument("--trace-on", type=Path, required=True)
    noninterference.add_argument("--output", type=Path, required=True)

    aggregate_parser = commands.add_parser("aggregate")
    _add_common_environment_arguments(aggregate_parser)
    aggregate_parser.add_argument("--trace-off-a", type=Path, required=True)
    aggregate_parser.add_argument("--trace-off-b", type=Path, required=True)
    aggregate_parser.add_argument("--trace-on", type=Path, required=True)
    aggregate_parser.add_argument("--repeatability-report", type=Path, required=True)
    aggregate_parser.add_argument("--noninterference-report", type=Path, required=True)
    aggregate_parser.add_argument("--output", type=Path, required=True)

    sanitize_parser = commands.add_parser("sanitize")
    sanitize_parser.add_argument("--manifest", type=Path, required=True)
    sanitize_parser.add_argument("--path-root", action="append", required=True)
    sanitize_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _path_roots(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        name, separator, raw_path = value.partition("=")
        if not separator or not name or not raw_path or name in result:
            raise P23AcquisitionError("--path-root must be unique NAME=/absolute/path entries")
        path = Path(raw_path)
        if not path.is_absolute():
            raise P23AcquisitionError("--path-root values must be absolute")
        result[name] = path.resolve()
    expected = {
        "repository",
        "preprocessor_alias",
        "nanogpt",
        "muon",
        "data",
        "python_environment",
        "native",
    }
    if set(result) != expected:
        raise P23AcquisitionError(
            "--path-root names must be exactly repository,preprocessor_alias,"
            "nanogpt,muon,data,python_environment,native"
        )
    return result


def _environment_kwargs(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "nanogpt_root": args.nanogpt_root.resolve(),
        "muon_source": args.muon_source.resolve(),
        "data_manifest": args.fineweb_manifest.resolve(),
        "instrumentation_patch": args.instrumentation_patch.resolve(),
        "runtime_lock_path": args.runtime_lock.resolve(),
        "host_attestation_path": args.host_attestation.resolve(),
        "addendum_path": P23.DEFAULT_ADDENDUM_PATH.resolve(),
    }


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "freeze-runtime":
            payload = P23.freeze_runtime_artifacts(
                container_image=args.container_image,
                container_repository_digest=args.container_repository_digest,
                host_image_inspection=args.host_image_inspection.resolve(),
                host_running_container_inspection=(
                    args.host_running_container_inspection.resolve()
                ),
                host_running_mountinfo=args.host_running_mountinfo.resolve(),
                host_nvidia_smi_query=args.host_nvidia_smi_query.resolve(),
                host_attestation_output=args.host_attestation_output.resolve(),
                runtime_lock_output=args.runtime_lock_output.resolve(),
                addendum_path=P23.DEFAULT_ADDENDUM_PATH.resolve(),
            )
        elif args.command == "run":
            payload = acquire_run(
                role=args.role,
                **_environment_kwargs(args),
                output=args.output.resolve(),
                raw_trace_output=(
                    args.raw_trace_output.resolve() if args.raw_trace_output else None
                ),
                trace_off_a_path=args.trace_off_a.resolve() if args.trace_off_a else None,
                trace_off_b_path=args.trace_off_b.resolve() if args.trace_off_b else None,
                repeatability_report_path=(
                    args.repeatability_report.resolve() if args.repeatability_report else None
                ),
            )
        elif args.command == "verify-repeatability":
            payload = verify_repeatability(
                **_environment_kwargs(args),
                trace_off_a_path=args.trace_off_a.resolve(),
                trace_off_b_path=args.trace_off_b.resolve(),
                output=args.output.resolve(),
            )
        elif args.command == "verify-noninterference":
            payload = verify_noninterference(
                **_environment_kwargs(args),
                trace_off_a_path=args.trace_off_a.resolve(),
                trace_off_b_path=args.trace_off_b.resolve(),
                trace_on_path=args.trace_on.resolve(),
                output=args.output.resolve(),
            )
        elif args.command == "aggregate":
            payload = aggregate(
                **_environment_kwargs(args),
                trace_off_a_path=args.trace_off_a.resolve(),
                trace_off_b_path=args.trace_off_b.resolve(),
                trace_on_path=args.trace_on.resolve(),
                repeatability_report_path=args.repeatability_report.resolve(),
                noninterference_report_path=args.noninterference_report.resolve(),
                output=args.output.resolve(),
            )
        else:
            payload = sanitize(
                manifest_path=args.manifest.resolve(),
                output=args.output.resolve(),
                path_roots=_path_roots(args.path_root),
            )
    except (P23AcquisitionError, P23.P23ProvenanceError, P22_RUNNER.P22RunError) as error:
        print(f"P23 blocked: {error}", file=sys.stderr)
        return 2

    if args.command == "freeze-runtime":
        print(json.dumps({"command": args.command, **payload}, indent=2, sort_keys=True))
        return 0

    output = args.output.resolve()
    print(
        json.dumps(
            {
                "command": args.command,
                "output": str(output),
                "output_sha256": P23.sha256_file(output),
                "passes": payload.get("passes", payload.get("all_predeclared_checks_pass", True)),
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0 if payload.get("passes", payload.get("all_predeclared_checks_pass", True)) else 1


if __name__ == "__main__":
    raise SystemExit(main())
