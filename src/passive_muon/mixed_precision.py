"""Locked 2-by-2 mixed-precision reference kernel for the P8 design.

This module specifies a proposed, certifiable operator. It is not upstream
Muon. Its certified interface is a finite, contiguous, CPU ``torch.float32``
2-by-2 matrix. Max-floor normalization is evaluated by a fixed serial FP32
algorithm. The five Jordan stages store only their boundary states in BF16;
every stage body is evaluated by separate, eager FP32 multiply and add
operations in a fixed row-major order, with no ``torch.matmul`` or fused
expression. The repair multiply and final add are FP32.

The accompanying P8 error theorem treats this operator as an implementation
error inside P7's exact-real outer EMA/Nesterov loop. The FP32 optimizer-step
helper at the end of this file is only an executable prototype: FP32 state and
parameter rounding introduce additional disturbance ports and are not covered
by that theorem.
"""

from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.structure_aware_stability import (
    LOCKED_BETA,
    LOCKED_FLOOR,
    LOCKED_LEARNING_RATE,
    LOCKED_REPAIR_RHO,
)

MIXED_PRECISION_SCHEMA_VERSION: Final = "passive-muon-mixed-precision-design-v2"
LOCKED_JORDAN_STEPS: Final = 5
LOCKED_MATRIX_SHAPE: Final = (2, 2)
LOCKED_BACKEND: Final = "torch-cpu-eager-ieee-rne-serial-fp32-v1"
LOCKED_SAFE_MAX_ABS_POWER: Final = 116
LOCKED_SAFE_MAX_ABS: Final = Fraction(2**LOCKED_SAFE_MAX_ABS_POWER)

# Exact binary32 encodings used by the stage and repair kernels.
COEFFICIENT_A_FP32_BITS: Final = 0x405C72B0
COEFFICIENT_B_FP32_BITS: Final = 0xC098CCCD
COEFFICIENT_C_FP32_BITS: Final = 0x40020419
REPAIR_RHO_FP32_BITS: Final = 0x4449E421


@dataclass(frozen=True)
class MixedPrecisionConfig:
    """The unique shape/backend configuration covered by the P8 kernel."""

    matrix_shape: tuple[int, int] = LOCKED_MATRIX_SHAPE
    backend: str = LOCKED_BACKEND

    def __post_init__(self) -> None:
        if self.matrix_shape != LOCKED_MATRIX_SHAPE:
            raise ValueError(f"certified matrix_shape must be exactly {LOCKED_MATRIX_SHAPE}")
        if self.backend != LOCKED_BACKEND:
            raise ValueError(f"the proposed implementation requires backend={LOCKED_BACKEND!r}")
        _require_locked_rounding_contract()


@dataclass(frozen=True)
class MixedPrecisionStepResult:
    """One record from the non-certified FP32 outer-loop prototype."""

    position: Tensor
    momentum: Tensor
    signal: Tensor
    operator_output: Tensor


def _fp32_from_bits(bits: int) -> Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32, device="cpu").view(torch.float32)[0]


def _fp32_bits(value: Tensor) -> int:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("bit inspection requires one CPU float32 value")
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _require_locked_rounding_contract() -> None:
    """Probe representative requirements of the locked arithmetic contract.

    Passing these finite probes is necessary, not an exhaustive proof of every
    backend operation. The P8 theorem remains conditional on the complete
    arithmetic contract recorded in :func:`mixed_precision_manifest`.
    """

    one = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    fp32_tie = torch.add(one, torch.tensor(2.0**-24, dtype=torch.float32))
    fp32_min_subnormal = torch.mul(
        torch.tensor(2.0**-74, dtype=torch.float32),
        torch.tensor(2.0**-75, dtype=torch.float32),
    )
    fp32_subnormal_input_add = torch.add(
        _fp32_from_bits(0x0080_0000),
        _fp32_from_bits(0x0000_0001),
    )
    bf16_tie = torch.tensor(1.0 + 2.0**-8, dtype=torch.float32).to(torch.bfloat16)
    bf16_min_subnormal = torch.tensor(2.0**-133, dtype=torch.float32).to(torch.bfloat16)
    bf16_subnormal_readback = bf16_min_subnormal.to(torch.float32)
    if _fp32_bits(fp32_tie) != 0x3F80_0000:
        raise RuntimeError("CPU FP32 addition does not satisfy the locked ties-to-even check")
    if _fp32_bits(fp32_min_subnormal) != 0x0000_0001:
        raise RuntimeError("CPU FP32 gradual underflow is required; FTZ/DAZ is excluded")
    if _fp32_bits(fp32_subnormal_input_add) != 0x0080_0001:
        raise RuntimeError("CPU FP32 subnormal inputs must not be treated as zero (DAZ excluded)")
    if int(bf16_tie.view(torch.uint16).item()) != 0x3F80:
        raise RuntimeError("CPU FP32-to-BF16 cast does not satisfy the locked ties-to-even check")
    if int(bf16_min_subnormal.view(torch.uint16).item()) != 0x0001:
        raise RuntimeError("CPU BF16 gradual underflow is required; FTZ/DAZ is excluded")
    if _fp32_bits(bf16_subnormal_readback) != 0x0001_0000:
        raise RuntimeError("CPU BF16 subnormal stage boundaries must survive FP32 readback")


def locked_backend_self_check() -> dict[str, object]:
    """Run and report the executable IEEE round-to-nearest-even checks."""

    _require_locked_rounding_contract()
    return {
        "fp32_add_halfway_bits_hex": "0x3f800000",
        "fp32_min_subnormal_bits_hex": "0x00000001",
        "fp32_subnormal_input_add_bits_hex": "0x00800001",
        "bf16_cast_halfway_bits_hex": "0x3f80",
        "bf16_min_subnormal_bits_hex": "0x0001",
        "bf16_min_subnormal_to_fp32_bits_hex": "0x00010000",
        "rounding": "IEEE-754 roundTiesToEven",
        "gradual_underflow": True,
        "ftz_daz": False,
    }


def _row_major_scalars(matrix: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    return matrix[0, 0], matrix[0, 1], matrix[1, 0], matrix[1, 1]


def _matrix_2x2(values: tuple[Tensor, Tensor, Tensor, Tensor]) -> Tensor:
    return torch.stack((torch.stack(values[:2]), torch.stack(values[2:])))


def _max_abs_row_major(matrix: Tensor) -> Tensor:
    values = _row_major_scalars(torch.abs(matrix))
    maximum = values[0]
    for value in values[1:]:
        maximum = torch.maximum(maximum, value)
    return maximum


def _check_fp32_matrix(name: str, matrix: Tensor, config: MixedPrecisionConfig) -> None:
    if matrix.ndim != 2 or tuple(matrix.shape) != config.matrix_shape:
        raise ValueError(f"{name} must have certified shape {config.matrix_shape}")
    if matrix.device.type != "cpu":
        raise ValueError(f"{name} must be on the locked CPU backend")
    if matrix.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")
    if matrix.layout != torch.strided or not matrix.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")
    safe_maximum = matrix.new_tensor(float(LOCKED_SAFE_MAX_ABS))
    if bool(_max_abs_row_major(matrix) > safe_maximum):
        raise ValueError(
            f"{name} exceeds the certified max-absolute range 2^{LOCKED_SAFE_MAX_ABS_POWER}"
        )


def _check_finite(name: str, value: Tensor) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite value produced at {name}; overflow is excluded")
    return value


def _fp32_mul(left: Tensor, right: Tensor) -> Tensor:
    """One materialized eager binary32 multiplication."""

    return torch.mul(left, right)


def _fp32_add(left: Tensor, right: Tensor) -> Tensor:
    """One materialized eager binary32 addition, separate from multiplication."""

    return torch.add(left, right)


def _fp32_div(left: Tensor, right: Tensor) -> Tensor:
    return torch.div(left, right)


def _serial_mm_2x2(left: Tensor, right: Tensor) -> Tensor:
    """Two-term row-major dots with two multiplies then one add each."""

    output: list[Tensor] = []
    for row in range(2):
        for column in range(2):
            first = _fp32_mul(left[row, 0], right[0, column])
            second = _fp32_mul(left[row, 1], right[1, column])
            output.append(_fp32_add(first, second))
    return _matrix_2x2(tuple(output))  # type: ignore[arg-type]


def fp32_max_floor_normalize(signal: Tensor, config: MixedPrecisionConfig) -> Tensor:
    """Evaluate ``signal/max(1, ||signal||_F)`` by the locked serial algorithm.

    If ``max(abs(signal)) <= 1/2``, the Frobenius norm is at most one for a
    2-by-2 matrix and the input is returned bit-for-bit (as a clone). This
    branch avoids making a false relative-error claim for subnormal maxima.
    Otherwise the routine scales by the maximum, serially accumulates four
    FP32 squares in row-major order, takes one FP32 square root, rescales, and
    divides every entry by ``max(1, norm)`` in row-major order.
    """

    _check_fp32_matrix("signal", signal, config)
    maximum = _max_abs_row_major(signal)
    half = signal.new_tensor(0.5)
    if bool(maximum <= half):
        return signal.clone()

    scaled_values = tuple(_fp32_div(value, maximum) for value in _row_major_scalars(signal))
    squares = tuple(_fp32_mul(value, value) for value in scaled_values)
    sum_squares = squares[0]
    for square in squares[1:]:
        sum_squares = _fp32_add(sum_squares, square)
    scaled_norm = torch.sqrt(sum_squares)
    norm = _fp32_mul(maximum, scaled_norm)
    denominator = torch.maximum(signal.new_tensor(float(LOCKED_FLOOR)), norm)
    normalized = _matrix_2x2(
        tuple(_fp32_div(value, denominator) for value in _row_major_scalars(signal))
    )
    return _check_finite("serial FP32 max-floor normalization", normalized)


def _round_bf16(name: str, value: Tensor) -> Tensor:
    return _check_finite(name, value.to(dtype=torch.bfloat16))


def _fp32_stage_coefficients() -> tuple[Tensor, Tensor, Tensor]:
    return (
        _fp32_from_bits(COEFFICIENT_A_FP32_BITS),
        _fp32_from_bits(COEFFICIENT_B_FP32_BITS),
        _fp32_from_bits(COEFFICIENT_C_FP32_BITS),
    )


def _jordan_stage(stage_input: Tensor, *, stage: int) -> Tensor:
    """Evaluate one complete Horner-form stage in serial FP32, then cast."""

    x = stage_input.to(torch.float32)
    a, b, c = _fp32_stage_coefficients()
    zero = x.new_tensor(0.0)

    gram = _serial_mm_2x2(x, x.mT)
    t_values: list[Tensor] = []
    for row in range(2):
        for column in range(2):
            c_gram = _fp32_mul(c, gram[row, column])
            b_identity = b if row == column else zero
            t_values.append(_fp32_add(b_identity, c_gram))
    t_matrix = _matrix_2x2(tuple(t_values))  # type: ignore[arg-type]

    d_matrix = _serial_mm_2x2(t_matrix, gram)
    e_values: list[Tensor] = []
    for row in range(2):
        for column in range(2):
            a_identity = a if row == column else zero
            e_values.append(_fp32_add(a_identity, d_matrix[row, column]))
    e_matrix = _matrix_2x2(tuple(e_values))  # type: ignore[arg-type]

    stage_output = _serial_mm_2x2(e_matrix, x)
    _check_finite(f"stage {stage} FP32 body", stage_output)
    return _round_bf16(f"stage {stage} BF16 boundary", stage_output)


def bf16_jordan_five_stage(
    normalized_signal: Tensor,
    config: MixedPrecisionConfig,
    *,
    return_stage_outputs: bool = False,
) -> Tensor | tuple[Tensor, tuple[Tensor, ...]]:
    """Run five locked stages, storing only stage boundaries in BF16."""

    _check_fp32_matrix("normalized_signal", normalized_signal, config)
    state = _round_bf16("initial BF16 boundary", normalized_signal)
    stage_outputs: list[Tensor] = []
    for stage in range(1, LOCKED_JORDAN_STEPS + 1):
        state = _jordan_stage(state, stage=stage)
        if return_stage_outputs:
            stage_outputs.append(state.detach().clone())

    output = _check_finite("final BF16 boundary cast to FP32", state.to(torch.float32))
    if return_stage_outputs:
        return output, tuple(stage_outputs)
    return output


def mixed_precision_repaired_operator(signal: Tensor, config: MixedPrecisionConfig) -> Tensor:
    """Evaluate the locked P8 operator with FP32 repair multiply/final add."""

    normalized = fp32_max_floor_normalize(signal, config)
    polynomial = bf16_jordan_five_stage(normalized, config)
    if not isinstance(polynomial, Tensor):  # pragma: no cover - default argument guards this
        raise AssertionError("unexpected stage trace")
    rho = _fp32_from_bits(REPAIR_RHO_FP32_BITS)
    repair = _check_finite("FP32 linear repair", torch.mul(signal, rho))
    return _check_finite("FP32 repaired output", torch.add(polynomial, repair))


def mixed_precision_ema_nesterov_step(
    position: Tensor,
    momentum: Tensor,
    gradient: Tensor,
    config: MixedPrecisionConfig,
) -> MixedPrecisionStepResult:
    """Run one FP32 outer-loop prototype step (not covered by the P8 theorem)."""

    _check_fp32_matrix("position", position, config)
    _check_fp32_matrix("momentum", momentum, config)
    _check_fp32_matrix("gradient", gradient, config)

    beta = position.new_tensor(float(LOCKED_BETA))
    one_minus_beta = position.new_tensor(float(1 - LOCKED_BETA))
    momentum_next = _check_finite(
        "prototype FP32 EMA state",
        torch.add(torch.mul(momentum, beta), torch.mul(gradient, one_minus_beta)),
    )
    signal = _check_finite(
        "prototype FP32 Nesterov signal",
        torch.add(torch.mul(momentum_next, beta), torch.mul(gradient, one_minus_beta)),
    )
    operator_output = mixed_precision_repaired_operator(signal, config)
    learning_rate = position.new_tensor(float(LOCKED_LEARNING_RATE))
    position_next = _check_finite(
        "prototype FP32 parameter update",
        torch.sub(position, torch.mul(operator_output, learning_rate)),
    )
    return MixedPrecisionStepResult(
        position=position_next,
        momentum=momentum_next,
        signal=signal,
        operator_output=operator_output,
    )


def _git_state() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]

    def run(*arguments: str) -> str | None:
        completed = subprocess.run(
            ["git", *arguments],
            cwd=root,
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


def _fp32_scalar_record_from_bits(bits: int) -> dict[str, object]:
    value = _fp32_from_bits(bits)
    runtime = float(value)
    return {
        "runtime_fraction": str(Fraction.from_float(runtime)),
        "fp32_value": runtime,
        "fp32_hex": runtime.hex(),
        "fp32_bits_hex": f"0x{bits:08x}",
    }


def _fp32_scalar_record(exact: Fraction) -> dict[str, object]:
    value = torch.tensor(float(exact), dtype=torch.float32)
    return {
        "source_exact": str(exact),
        **_fp32_scalar_record_from_bits(_fp32_bits(value)),
    }


def _coefficient_record(
    name: str,
    source: Fraction,
    runtime: Fraction,
    bits: int,
) -> dict[str, object]:
    record = _fp32_scalar_record_from_bits(bits)
    if record["runtime_fraction"] != str(runtime):
        raise AssertionError(f"locked {name} coefficient fraction/bit mismatch")
    return {"source_fraction": str(source), **record}


def mixed_precision_manifest(config: MixedPrecisionConfig) -> dict[str, object]:
    """Return the locked implementation contract and local provenance."""

    source_path = Path(__file__).resolve()
    source_coefficients = JORDAN_QUINTIC.fractions()
    return {
        "schema_version": MIXED_PRECISION_SCHEMA_VERSION,
        "claim_scope": (
            "certified 2x2 mixed-precision repaired max-floor operator inside the "
            "exact-real P7 outer loop; not an end-to-end FP32 or upstream Muon theorem"
        ),
        "matrix_domain": {
            "shape": list(LOCKED_MATRIX_SHAPE),
            "layout": "contiguous torch.strided",
            "input_dtype": "torch.float32 (hence a finite binary dyadic)",
            "device": "cpu",
            "finite_inputs_required": True,
            "max_abs_inclusive": f"2^{LOCKED_SAFE_MAX_ABS_POWER}",
        },
        "operator": {
            "formula": "Rhat(s)=BF16BoundaryStages(Nhat_1(s))+rho32*s",
            "normalization_model": "s/max(1,||s||_F)",
            "normalization_runtime": (
                "if maxabs<=1/2 clone s; else max-scale, four FP32 squares, three "
                "serial FP32 adds, sqrt, rescale, max with one, four FP32 divides"
            ),
            "normalization_order": "fixed row-major",
            "floor_exact": str(LOCKED_FLOOR),
            "floor_runtime": _fp32_scalar_record(LOCKED_FLOOR),
            "epsilon": "none",
            "polynomial": JORDAN_QUINTIC.name,
            "steps": LOCKED_JORDAN_STEPS,
            "stage_formula": "A=X X^T; T=b32 I+c32 A; D=T A; E=a32 I+D; Y=E X",
            "coefficients": {
                "a": _coefficient_record(
                    "a",
                    source_coefficients[0],
                    Fraction(902_955, 262_144),
                    COEFFICIENT_A_FP32_BITS,
                ),
                "b": _coefficient_record(
                    "b",
                    source_coefficients[1],
                    Fraction(-10_013_901, 2_097_152),
                    COEFFICIENT_B_FP32_BITS,
                ),
                "c": _coefficient_record(
                    "c",
                    source_coefficients[2],
                    Fraction(8_520_729, 4_194_304),
                    COEFFICIENT_C_FP32_BITS,
                ),
            },
            "coefficient_source": JORDAN_QUINTIC.source,
            "repair_rho_exact": str(LOCKED_REPAIR_RHO),
            "repair_rho_runtime": {
                "source_exact": str(LOCKED_REPAIR_RHO),
                **_fp32_scalar_record_from_bits(REPAIR_RHO_FP32_BITS),
            },
        },
        "precision_contract": {
            "input_and_normalizer": "torch.float32",
            "stored_stage_boundaries": "torch.bfloat16 (initial plus five outputs)",
            "stage_body_and_coefficients": "torch.float32",
            "stage_dot_product": (
                "fixed row-major two-term dot: materialize two FP32 multiplies, then one "
                "FP32 add; torch.matmul and fused multiply-add expressions are forbidden"
            ),
            "linear_repair_and_final_add": "torch.float32",
            "rounding": "IEEE-754 roundTiesToEven at every explicit torch operation/cast",
            "gradual_underflow_required": True,
            "ftz_daz_allowed": False,
            "torch_compile_used": False,
        },
        "operation_order": [
            "validate one finite contiguous 2x2 FP32 CPU matrix with maxabs<=2^116",
            "apply the fixed piecewise scaled serial FP32 max-floor normalizer",
            "cast the normalized state once to BF16",
            "repeat five times: cast state to FP32, evaluate A,T,D,E,Y serially, cast Y",
            "cast the fifth BF16 boundary to FP32",
            "materialize rho32*s in FP32 and then the final FP32 add",
        ],
        "backend": {
            "name": config.backend,
            "self_check": locked_backend_self_check(),
            "native_bf16_matmul_used": False,
            "torch_matmul_used": False,
            "overflow_policy": "reject inputs outside maxabs<=2^116 and any nonfinite result",
        },
        "outer_loop_prototype": {
            "status": "executable only; outside the P7/P8 theorem",
            "state_dtype": "torch.float32",
            "beta_source_exact": str(LOCKED_BETA),
            "beta_runtime": _fp32_scalar_record(LOCKED_BETA),
            "learning_rate_source_exact": str(LOCKED_LEARNING_RATE),
            "learning_rate_runtime": _fp32_scalar_record(LOCKED_LEARNING_RATE),
            "update": (
                "m_next=beta32*m+(1-beta)32*g; s=beta32*m_next+(1-beta)32*g; W-=eta32*Rhat(s)"
            ),
        },
        "provenance": {
            "git": _git_state(),
            "source_file": str(source_path.relative_to(source_path.parents[2])),
            "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "python": sys.version,
            "torch": str(torch.__version__),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "excluded_claims": [
            "literal upstream Muon parity or coverage",
            "end-to-end P7 stability for the FP32 outer-loop prototype",
            "arbitrary matrix shapes or dimension-uniform mixed-precision error",
            "native accelerator or native-BF16 matrix-kernel behavior",
            "BF16 momentum/Nesterov state, current-plus-epsilon normalization, or aspect scaling",
            "overflow, FTZ/DAZ, stochastic rounding, fused, or compiler-reassociated executions",
        ],
    }
