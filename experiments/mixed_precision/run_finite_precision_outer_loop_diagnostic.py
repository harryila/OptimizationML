#!/usr/bin/env python3
"""Deterministic CPU falsification diagnostic for the P10 arithmetic shell.

This script exercises representative binary32 edge and seeded-random inputs.
Every reported residual-envelope comparison and ``TwoSum`` identity is checked
entrywise with :class:`fractions.Fraction`; nevertheless, the selected inputs
are only falsification cases and do not replace the global P10 certificate.
The raw-FP32 stalling record is a separate exact finite witness.

No file is written unless ``--output PATH`` or ``--write-canonical`` is passed.
The deliberately modest shape cap prevents this diagnostic from accidentally
invoking a large proof-reference P9 kernel.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from torch import Tensor

from passive_muon.finite_precision_outer_loop import (
    BETA_FP32_BITS,
    BETA_FP32_EXACT,
    EMA_GRADIENT_ENVELOPE_COEFFICIENT,
    EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
    EMA_RESIDUAL_CRUMB_PER_ENTRY,
    FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION,
    LEARNING_RATE_FP32_BITS,
    LEARNING_RATE_FP32_EXACT,
    LOCKED_MASTER_HIGH_MAX_ABS_POWER,
    LOCKED_MASTER_LOW_MAX_ABS_POWER,
    LOCKED_MASTER_MIDDLE_MAX_ABS_POWER,
    LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER,
    LOCKED_OUTER_BACKEND,
    MASTER_LOW_WORD_ENVELOPE_COEFFICIENT,
    MASTER_OUTPUT_ENVELOPE_COEFFICIENT,
    MASTER_RESIDUAL_CRUMB_PER_ENTRY,
    ONE_MINUS_BETA_FP32_BITS,
    ONE_MINUS_BETA_FP32_EXACT,
    FinitePrecisionOuterLoopConfig,
    ThreeWordFP32Master,
    compensated_master_weight_update,
    fp32_ema_nesterov,
    fp32_master_stalling_witness,
    locked_outer_backend_self_check,
    outer_residual_envelope,
    two_sum_fp32,
)
from passive_muon.scalable_mixed_precision_certificate import (
    EXACT_A,
    EXACT_B,
    EXACT_C,
    EXACT_RHO,
    LOCKED_STAGES,
)
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_FLOOR,
    LOCKED_LEARNING_RATE,
)

SCHEMA_VERSION: Final = "passive-muon-finite-precision-outer-loop-diagnostic-v1"
SEED: Final = 20_260_903
DEFAULT_SHAPES: Final = ((1, 1), (2, 3), (8, 16))
MAX_DIAGNOSTIC_ENTRIES: Final = 4_096
EMA_CASE_FAMILIES: Final = (
    "binary32_edge_cycle",
    "seeded_dense",
    "seeded_near_cancellation",
)
MASTER_CASE_FAMILIES: Final = ("guard_edge_cycle", "seeded_interior")

ROOT = Path(__file__).resolve().parents[2]
CANONICAL_TARGET = ROOT / "results" / "summaries" / "finite_precision_outer_loop_diagnostic.json"
SOURCE_PATHS = (
    Path("experiments/mixed_precision/run_finite_precision_outer_loop_diagnostic.py"),
    Path("src/passive_muon/finite_precision_outer_loop.py"),
    Path("src/passive_muon/scalable_mixed_precision.py"),
    Path("src/passive_muon/scalable_mixed_precision_certificate.py"),
    Path("src/passive_muon/structure_aware_stability.py"),
    Path("tests/test_finite_precision_outer_loop_diagnostic.py"),
    Path("pyproject.toml"),
    Path("uv.lock"),
)


def _validate_shape(shape: tuple[int, int]) -> None:
    if len(shape) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0 for value in shape
    ):
        raise ValueError("each shape must contain two positive integers")
    if math.prod(shape) > MAX_DIAGNOSTIC_ENTRIES:
        raise ValueError(f"diagnostic shapes may contain at most {MAX_DIAGNOSTIC_ENTRIES} entries")


@dataclass(frozen=True)
class DiagnosticConfig:
    """Complete deterministic P10 diagnostic configuration."""

    seed: int = SEED
    shapes: tuple[tuple[int, int], ...] = DEFAULT_SHAPES
    torch_threads: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.seed, int) or isinstance(self.seed, bool) or self.seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        if not self.shapes:
            raise ValueError("at least one diagnostic shape is required")
        for shape in self.shapes:
            _validate_shape(shape)
        if len(set(self.shapes)) != len(self.shapes):
            raise ValueError("diagnostic shapes must be unique")
        if (
            not isinstance(self.torch_threads, int)
            or isinstance(self.torch_threads, bool)
            or self.torch_threads <= 0
        ):
            raise ValueError("torch_threads must be a positive integer")


def _fp32_from_bits(bits: int) -> Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32, device="cpu").view(torch.float32)[0]


def _fraction(value: Tensor) -> Fraction:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("exact conversion requires one CPU binary32 value")
    runtime = float(value.reshape(1).item())
    if not math.isfinite(runtime):
        raise ValueError("exact conversion requires a finite value")
    return Fraction.from_float(runtime)


def _tensor_sha256(matrix: Tensor) -> str:
    contiguous = matrix.detach().contiguous().numpy()
    return hashlib.sha256(contiguous.tobytes(order="C")).hexdigest()


def _cycle_tensor(values: tuple[float, ...], shape: tuple[int, int], *, offset: int = 0) -> Tensor:
    count = math.prod(shape)
    flat = [values[(index + offset) % len(values)] for index in range(count)]
    return torch.tensor(flat, dtype=torch.float32, device="cpu").reshape(shape).contiguous()


def build_ema_cases(
    shape: tuple[int, int], *, seed: int, shape_index: int
) -> tuple[tuple[str, Tensor, Tensor], ...]:
    """Build binary32 edges, random values, and a cancellation family."""

    _validate_shape(shape)
    edge_values = (
        0.0,
        -0.0,
        2.0**-149,
        -(2.0**-149),
        2.0**-126,
        -(2.0**-126),
        1.0,
        -1.0,
        1.0 + 2.0**-23,
        -(1.0 + 2.0**-23),
        2.0**20,
        -(2.0**20),
    )
    edge_momentum = _cycle_tensor(edge_values, shape, offset=shape_index)
    edge_gradient = _cycle_tensor(edge_values, shape, offset=5 + 2 * shape_index)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 1_000_003 * shape_index)
    dense_momentum = 128.0 * torch.randn(shape, dtype=torch.float32, generator=generator)
    dense_gradient = 128.0 * torch.randn(shape, dtype=torch.float32, generator=generator)
    cancellation_momentum = 16.0 * torch.randn(shape, dtype=torch.float32, generator=generator)
    cancellation_gradient = torch.mul(cancellation_momentum, -19.0)
    return (
        ("binary32_edge_cycle", edge_momentum, edge_gradient),
        ("seeded_dense", dense_momentum.contiguous(), dense_gradient.contiguous()),
        (
            "seeded_near_cancellation",
            cancellation_momentum.contiguous(),
            cancellation_gradient.contiguous(),
        ),
    )


def build_master_cases(
    shape: tuple[int, int], *, seed: int, shape_index: int
) -> tuple[tuple[str, ThreeWordFP32Master, Tensor], ...]:
    """Build a guarded edge family and a seeded interior family."""

    _validate_shape(shape)
    high = _cycle_tensor(
        (2.0**30, -(2.0**30), 2.0**20, -(2.0**20), 1.0, -1.0, 0.0, 2.0**-126),
        shape,
        offset=2 * shape_index,
    )
    middle = _cycle_tensor((-64.0, 64.0, -0.25, 0.25, 0.0, 2.0**-20), shape)
    low = _cycle_tensor(
        (2.0**-17, -(2.0**-17), 2.0**-20, -(2.0**-20), 0.0, 2.0**-149),
        shape,
    )
    output = _cycle_tensor(
        (2.0**15, -(2.0**15), 2.0**14, -(2.0**14), 17.0, -17.0, 0.25, -0.25),
        shape,
        offset=2 * shape_index,
    )
    edge_master = ThreeWordFP32Master(high=high, middle=middle, low=low)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + 9_000_019 + 1_000_003 * shape_index)
    random_high = (2.0**20) * torch.randn(shape, dtype=torch.float32, generator=generator)
    random_middle = 0.5 * torch.randn(shape, dtype=torch.float32, generator=generator)
    random_low = (2.0**-20) * torch.randn(shape, dtype=torch.float32, generator=generator)
    random_output = 2_048.0 * torch.randn(shape, dtype=torch.float32, generator=generator)
    interior_master = ThreeWordFP32Master(
        high=random_high.contiguous(),
        middle=random_middle.contiguous(),
        low=random_low.contiguous(),
    )
    return (
        ("guard_edge_cycle", edge_master, output),
        ("seeded_interior", interior_master, random_output.contiguous()),
    )


def _exact_envelope_audit(
    *,
    residual_at: Any,
    first: Tensor,
    second: Tensor,
    first_coefficient: Fraction,
    second_coefficient: Fraction,
    crumb: Fraction,
) -> dict[str, object]:
    """Check an affine residual envelope at every selected matrix entry."""

    rows, columns = first.shape
    violation_count = 0
    worst_ratio = Fraction(-1)
    worst_record: dict[str, object] | None = None
    minimum_slack: Fraction | None = None
    for row in range(rows):
        for column in range(columns):
            residual = residual_at(row, column)
            bound = (
                first_coefficient * abs(_fraction(first[row, column]))
                + second_coefficient * abs(_fraction(second[row, column]))
                + crumb
            )
            slack = bound - abs(residual)
            ratio = abs(residual) / bound
            if slack < 0:
                violation_count += 1
            if minimum_slack is None or slack < minimum_slack:
                minimum_slack = slack
            if ratio > worst_ratio:
                worst_ratio = ratio
                worst_record = {
                    "coordinate": [row, column],
                    "residual_exact": str(residual),
                    "absolute_residual_exact": str(abs(residual)),
                    "bound_exact": str(bound),
                    "slack_exact": str(slack),
                    "utilization_exact": str(ratio),
                }
    if worst_record is None or minimum_slack is None:  # pragma: no cover - positive shapes
        raise AssertionError("an exact envelope audit checked no entries")
    return {
        "entry_check_count": rows * columns,
        "violation_count": violation_count,
        "all_entries_within_exact_envelope": violation_count == 0,
        "minimum_slack_exact": str(minimum_slack),
        "maximum_utilization_exact": str(worst_ratio),
        "worst_entry": worst_record,
    }


def _ema_case_record(
    label: str,
    momentum: Tensor,
    gradient: Tensor,
    config: FinitePrecisionOuterLoopConfig,
) -> dict[str, object]:
    result = fp32_ema_nesterov(momentum, gradient, config)
    r_m = _exact_envelope_audit(
        residual_at=result.residual_ports.r_m.exact_entry,
        first=momentum,
        second=gradient,
        first_coefficient=EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
        second_coefficient=EMA_GRADIENT_ENVELOPE_COEFFICIENT,
        crumb=EMA_RESIDUAL_CRUMB_PER_ENTRY,
    )
    r_s = _exact_envelope_audit(
        residual_at=result.residual_ports.r_s.exact_entry,
        first=result.momentum,
        second=gradient,
        first_coefficient=EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
        second_coefficient=EMA_GRADIENT_ENVELOPE_COEFFICIENT,
        crumb=EMA_RESIDUAL_CRUMB_PER_ENTRY,
    )
    return {
        "label": label,
        "shape": list(config.matrix_shape),
        "inputs": {
            "momentum_sha256_float32_bytes": _tensor_sha256(momentum),
            "gradient_sha256_float32_bytes": _tensor_sha256(gradient),
            "momentum_max_abs": float(torch.amax(torch.abs(momentum))),
            "gradient_max_abs": float(torch.amax(torch.abs(gradient))),
        },
        "output_hashes": {
            "momentum_next_sha256_float32_bytes": _tensor_sha256(result.momentum),
            "signal_sha256_float32_bytes": _tensor_sha256(result.signal),
        },
        "r_m": r_m,
        "r_s": r_s,
        "all_exact_checks_pass": bool(
            r_m["all_entries_within_exact_envelope"] and r_s["all_entries_within_exact_envelope"]
        ),
    }


def _master_case_record(
    label: str,
    master: ThreeWordFP32Master,
    operator_output: Tensor,
    config: FinitePrecisionOuterLoopConfig,
) -> dict[str, object]:
    update = compensated_master_weight_update(master, operator_output, config)
    audit = _exact_envelope_audit(
        residual_at=update.residual_port.exact_entry,
        first=operator_output,
        second=master.low,
        first_coefficient=MASTER_OUTPUT_ENVELOPE_COEFFICIENT,
        second_coefficient=MASTER_LOW_WORD_ENVELOPE_COEFFICIENT,
        crumb=MASTER_RESIDUAL_CRUMB_PER_ENTRY,
    )
    rows, columns = config.matrix_shape
    middle_two_sum_violations = 0
    high_two_sum_violations = 0
    logical_update_violations = 0
    for row in range(rows):
        for column in range(columns):
            middle_identity = _fraction(update.trace.middle_candidate[row, column]) + _fraction(
                update.trace.middle_two_sum_residual[row, column]
            ) == _fraction(master.middle[row, column]) + _fraction(
                update.trace.pending[row, column]
            )
            high_identity = _fraction(update.master.high[row, column]) + _fraction(
                update.trace.high_two_sum_residual[row, column]
            ) == _fraction(master.high[row, column]) + _fraction(
                update.trace.middle_candidate[row, column]
            )
            logical_identity = update.master.exact_entry(row, column) - master.exact_entry(
                row, column
            ) == _fraction(update.trace.pending[row, column]) - _fraction(master.low[row, column])
            middle_two_sum_violations += not middle_identity
            high_two_sum_violations += not high_identity
            logical_update_violations += not logical_identity
    identity_checks = {
        "middle_two_sum_check_count": rows * columns,
        "middle_two_sum_violation_count": middle_two_sum_violations,
        "high_two_sum_check_count": rows * columns,
        "high_two_sum_violation_count": high_two_sum_violations,
        "logical_update_check_count": rows * columns,
        "logical_update_violation_count": logical_update_violations,
    }
    return {
        "label": label,
        "shape": list(config.matrix_shape),
        "inputs": {
            "high_sha256_float32_bytes": _tensor_sha256(master.high),
            "middle_sha256_float32_bytes": _tensor_sha256(master.middle),
            "low_sha256_float32_bytes": _tensor_sha256(master.low),
            "operator_output_sha256_float32_bytes": _tensor_sha256(operator_output),
            "operator_output_max_abs": float(torch.amax(torch.abs(operator_output))),
        },
        "output_hashes": {
            "high_sha256_float32_bytes": _tensor_sha256(update.master.high),
            "middle_sha256_float32_bytes": _tensor_sha256(update.master.middle),
            "low_sha256_float32_bytes": _tensor_sha256(update.master.low),
        },
        "r_W": audit,
        "exact_identities": identity_checks,
        "all_exact_checks_pass": bool(
            audit["all_entries_within_exact_envelope"]
            and middle_two_sum_violations == 0
            and high_two_sum_violations == 0
            and logical_update_violations == 0
        ),
    }


def _standalone_two_sum_checks() -> dict[str, object]:
    pairs = (
        (2.0**30, -(2.0**4 + 0.375)),
        (1.0, 2.0**-24),
        (2.0**-126, 2.0**-149),
        (-12_345.5, 0.03125),
    )
    records: list[dict[str, object]] = []
    for left_value, right_value in pairs:
        left = torch.tensor([[left_value]], dtype=torch.float32, device="cpu")
        right = torch.tensor([[right_value]], dtype=torch.float32, device="cpu")
        result = two_sum_fp32(left, right, name="diagnostic standalone TwoSum")
        left_exact = _fraction(left)
        right_exact = _fraction(right)
        rounded_exact = _fraction(result.rounded)
        residual_exact = _fraction(result.residual)
        identity = rounded_exact + residual_exact == left_exact + right_exact
        records.append(
            {
                "left_exact": str(left_exact),
                "right_exact": str(right_exact),
                "rounded_exact": str(rounded_exact),
                "residual_exact": str(residual_exact),
                "identity_holds": identity,
            }
        )
    return {
        "case_count": len(records),
        "violation_count": sum(not bool(record["identity_holds"]) for record in records),
        "cases": records,
    }


def _envelope_record(config: FinitePrecisionOuterLoopConfig) -> dict[str, object]:
    envelope = outer_residual_envelope(config)
    return {
        key: str(value) if isinstance(value, Fraction) else value
        for key, value in asdict(envelope).items()
    }


def _git_state() -> dict[str, object]:
    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        return completed.stdout.strip() if completed.returncode == 0 else None

    status = run("status", "--porcelain")
    return {
        "sha": run("rev-parse", "HEAD") or "unavailable",
        "branch": run("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _source_snapshot() -> dict[str, str]:
    return {
        path.as_posix(): hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        for path in SOURCE_PATHS
    }


def _provenance(config: DiagnosticConfig) -> dict[str, object]:
    return {
        "git": _git_state(),
        "seed": config.seed,
        "hardware": {
            "device": "cpu",
            "machine": platform.machine(),
            "processor": platform.processor() or "unreported",
            "platform": platform.platform(),
            "logical_cpu_count": os.cpu_count(),
        },
        "software": {
            "python": sys.version,
            "numpy": str(np.__version__),
            "torch": str(torch.__version__),
            "torch_config": torch.__config__.show(),
        },
        "runtime": {
            "torch_num_threads": torch.get_num_threads(),
            "torch_num_interop_threads": torch.get_num_interop_threads(),
            "deterministic_algorithms_enabled": torch.are_deterministic_algorithms_enabled(),
        },
        "source_snapshot": _source_snapshot(),
    }


def _summary(
    ema_cases: list[dict[str, object]],
    master_cases: list[dict[str, object]],
    two_sum: dict[str, object],
    stalling_witness: dict[str, object],
) -> dict[str, object]:
    ema_checks = sum(
        int(case[port]["entry_check_count"])  # type: ignore[index]
        for case in ema_cases
        for port in ("r_m", "r_s")
    )
    ema_violations = sum(
        int(case[port]["violation_count"])  # type: ignore[index]
        for case in ema_cases
        for port in ("r_m", "r_s")
    )
    master_checks = sum(int(case["r_W"]["entry_check_count"]) for case in master_cases)  # type: ignore[index]
    master_violations = sum(int(case["r_W"]["violation_count"]) for case in master_cases)  # type: ignore[index]
    middle_checks = sum(
        int(case["exact_identities"]["middle_two_sum_check_count"])  # type: ignore[index]
        for case in master_cases
    )
    high_checks = sum(
        int(case["exact_identities"]["high_two_sum_check_count"])  # type: ignore[index]
        for case in master_cases
    )
    logical_checks = sum(
        int(case["exact_identities"]["logical_update_check_count"])  # type: ignore[index]
        for case in master_cases
    )
    identity_violations = sum(
        int(case["exact_identities"][key])  # type: ignore[index]
        for case in master_cases
        for key in (
            "middle_two_sum_violation_count",
            "high_two_sum_violation_count",
            "logical_update_violation_count",
        )
    )
    standalone_violations = int(two_sum["violation_count"])
    all_pass = (
        ema_violations == 0
        and master_violations == 0
        and identity_violations == 0
        and standalone_violations == 0
        and bool(stalling_witness["certified"])
    )
    return {
        "shape_count": len({tuple(case["shape"]) for case in ema_cases}),  # type: ignore[arg-type]
        "ema_case_count": len(ema_cases),
        "master_case_count": len(master_cases),
        "exact_ema_residual_entry_check_count": ema_checks,
        "exact_ema_residual_violation_count": ema_violations,
        "exact_master_residual_entry_check_count": master_checks,
        "exact_master_residual_violation_count": master_violations,
        "exact_two_sum_identity_check_count": middle_checks
        + high_checks
        + int(two_sum["case_count"]),
        "exact_logical_update_identity_check_count": logical_checks,
        "exact_identity_violation_count": identity_violations + standalone_violations,
        "actual_p9_stalling_witness_certified": bool(stalling_witness["certified"]),
        "all_selected_falsification_checks_pass": all_pass,
    }


def run_diagnostic(config: DiagnosticConfig) -> dict[str, object]:
    """Run the complete deterministic modest-shape CPU diagnostic."""

    previous_threads = torch.get_num_threads()
    previous_deterministic = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(config.torch_threads)
    torch.use_deterministic_algorithms(True)
    try:
        ema_records: list[dict[str, object]] = []
        master_records: list[dict[str, object]] = []
        shape_envelopes: dict[str, dict[str, object]] = {}
        for shape_index, shape in enumerate(config.shapes):
            outer_config = FinitePrecisionOuterLoopConfig(shape)
            shape_envelopes[f"{shape[0]}x{shape[1]}"] = _envelope_record(outer_config)
            for label, momentum, gradient in build_ema_cases(
                shape, seed=config.seed, shape_index=shape_index
            ):
                ema_records.append(_ema_case_record(label, momentum, gradient, outer_config))
            for label, master, operator_output in build_master_cases(
                shape, seed=config.seed, shape_index=shape_index
            ):
                master_records.append(
                    _master_case_record(label, master, operator_output, outer_config)
                )

        two_sum = _standalone_two_sum_checks()
        stalling_witness = fp32_master_stalling_witness()
        summary = _summary(ema_records, master_records, two_sum, stalling_witness)
        return {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": {
                "status": (
                    "sampled CPU falsification diagnostic plus one exact finite stalling "
                    "witness; not a global certificate"
                ),
                "exact_check_qualification": (
                    "Fraction arithmetic is authoritative for each selected binary32 entry, "
                    "but selecting finitely many entries cannot prove the global envelopes"
                ),
                "stalling_qualification": (
                    "raw subtraction stalling is an exact executable P9 finite witness; "
                    "compensated progress here is not a convergence theorem"
                ),
                "operator_qualification": (
                    "only the small 2x2 stalling witness invokes the actual slow P9 operator; "
                    "the envelope cases isolate outer-shell arithmetic"
                ),
            },
            "config": {
                **asdict(config),
                "shapes": [list(shape) for shape in config.shapes],
                "maximum_diagnostic_entries": MAX_DIAGNOSTIC_ENTRIES,
                "ema_case_families": list(EMA_CASE_FAMILIES),
                "master_case_families": list(MASTER_CASE_FAMILIES),
                "entries_checked_exhaustively_within_each_selected_case": True,
                "backend": LOCKED_OUTER_BACKEND,
                "outer_loop": {
                    "beta_source_exact": str(LOCKED_BETA),
                    "beta_runtime_exact": str(BETA_FP32_EXACT),
                    "beta_runtime_bits_hex": f"0x{BETA_FP32_BITS:08x}",
                    "one_minus_beta_source_exact": str(1 - LOCKED_BETA),
                    "one_minus_beta_runtime_exact": str(ONE_MINUS_BETA_FP32_EXACT),
                    "one_minus_beta_runtime_bits_hex": f"0x{ONE_MINUS_BETA_FP32_BITS:08x}",
                    "learning_rate_source_exact": str(LOCKED_LEARNING_RATE),
                    "learning_rate_runtime_exact": str(LEARNING_RATE_FP32_EXACT),
                    "learning_rate_runtime_bits_hex": f"0x{LEARNING_RATE_FP32_BITS:08x}",
                    "ema_nesterov_operation_order": (
                        "bg=fl32(a32*g); bm=fl32(beta32*m); "
                        "m_next=fl32(bm+bg); bs=fl32(beta32*m_next); "
                        "s_next=fl32(bs+bg)"
                    ),
                    "master_operation_order": (
                        "step=fl32(eta32*Rhat); pending=fl32(low-step); "
                        "middle_candidate,new_low=TwoSum(middle,pending); "
                        "new_high,new_middle=TwoSum(high,middle_candidate)"
                    ),
                    "master_word_guards_max_abs": {
                        "high": f"2^{LOCKED_MASTER_HIGH_MAX_ABS_POWER}",
                        "middle": f"2^{LOCKED_MASTER_MIDDLE_MAX_ABS_POWER}",
                        "low": f"2^{LOCKED_MASTER_LOW_MAX_ABS_POWER}",
                    },
                },
                "bound_operator": {
                    "domain": "finite matrices of each locked shape",
                    "normalization": f"s/max({LOCKED_FLOOR},||s||_F)",
                    "epsilon": "none",
                    "polynomial": "h(x)=a*x+b*x^3+c*x^5",
                    "coefficients_exact": {
                        "a": str(EXACT_A),
                        "b": str(EXACT_B),
                        "c": str(EXACT_C),
                    },
                    "iteration_count": LOCKED_STAGES,
                    "repair_rho_exact": str(EXACT_RHO),
                    "operator_output_max_abs_guard": (f"2^{LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER}"),
                },
                "entrywise_roundoff_model": (
                    "|fl32(x)-x|<=2^-24*|x|+2^-150; RNE, gradual underflow, "
                    "finite named intermediates, no fusion or reassociation"
                ),
            },
            "backend_self_check": locked_outer_backend_self_check(),
            "shape_envelopes_exact": shape_envelopes,
            "summary": summary,
            "ema_nesterov_cases": ema_records,
            "compensated_master_cases": master_records,
            "standalone_two_sum": two_sum,
            "actual_p9_raw_stall_vs_compensation_witness": stalling_witness,
            "experiment_provenance": _provenance(config),
            "canonical_target": str(CANONICAL_TARGET.relative_to(ROOT)),
            "outer_shell_schema_version": FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION,
        }
    finally:
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_deterministic)


def canonical_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _parse_shape(value: str) -> tuple[int, int]:
    parts = value.lower().split("x")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must use ROWSxCOLUMNS")
    try:
        shape = (int(parts[0]), int(parts[1]))
        _validate_shape(shape)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from error
    return shape


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument(
        "--shape",
        dest="shapes",
        action="append",
        type=_parse_shape,
        help="modest executable ROWSxCOLUMNS shape; repeat to replace defaults",
    )
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument("--output", type=Path, help="explicit JSON output path")
    output_group.add_argument(
        "--write-canonical",
        action="store_true",
        help=f"write the explicit canonical target {CANONICAL_TARGET.relative_to(ROOT)}",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    config = DiagnosticConfig(
        seed=arguments.seed,
        shapes=tuple(arguments.shapes) if arguments.shapes else DEFAULT_SHAPES,
        torch_threads=arguments.torch_threads,
    )
    payload = run_diagnostic(config)
    rendered = canonical_json(payload)
    output = CANONICAL_TARGET if arguments.write_canonical else arguments.output
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
