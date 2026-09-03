"""Shape-parameterized mixed-precision reference kernel for P9.

This module specifies a proposed operator, not upstream Muon and not a fast
BLAS implementation.  It accepts a finite, contiguous CPU ``float32`` matrix
whose shape is locked by :class:`ScalableMixedPrecisionConfig` and whose
maximum magnitude is at most ``2**116``.  Tall matrices are transposed once so
that every polynomial stage acts on an oriented ``p x q`` matrix with
``p=min(rows, columns)``.

The max-floor normalizer uses balanced FP32 reductions.  A polynomial stage
keeps FP32 arithmetic but stores its boundary as a two-term BF16 expansion:
``high=RN_b(Y)``, ``residual=fl32(Y-high)``, ``low=RN_b(residual)``, and
``X_next=fl32(high+low)``.  Matrix products use explicitly rounded FP32
products and a deterministic balanced FP32 reduction.  ``torch.matmul``,
FMA contraction, reassociation, and native BF16 matrix kernels are excluded.

The explicit dot-product implementation is intentionally a slow proof
reference.  A production kernel needs a separately audited parity
implementation before inheriting any certificate built for these semantics.
"""

from __future__ import annotations

import hashlib
import math
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
from passive_muon.structure_aware_stability import LOCKED_FLOOR, LOCKED_REPAIR_RHO

SCALABLE_MIXED_PRECISION_SCHEMA_VERSION: Final = "passive-muon-scalable-mixed-precision-design-v1"
LOCKED_JORDAN_STEPS: Final = 5
LOCKED_BACKEND: Final = "torch-cpu-eager-ieee-rne-balanced-fp32-two-term-bf16-v1"
LOCKED_SAFE_MAX_ABS_POWER: Final = 116
LOCKED_SAFE_MAX_ABS: Final = Fraction(2**LOCKED_SAFE_MAX_ABS_POWER)
LOCKED_MAX_ENTRIES: Final = 2**52

# Exact binary32 encodings shared with the P8 arithmetic contract.
COEFFICIENT_A_FP32_BITS: Final = 0x405C72B0
COEFFICIENT_B_FP32_BITS: Final = 0xC098CCCD
COEFFICIENT_C_FP32_BITS: Final = 0x40020419
REPAIR_RHO_FP32_BITS: Final = 0x4449E421

FP32_UNIT_ROUNDOFF: Final = Fraction(1, 2**24)
SERIAL_UNIT_SUM_ABSORPTION_COUNT: Final = 2**24


@dataclass(frozen=True)
class ScalableMixedPrecisionConfig:
    """One shape/backend instance of the P9 reference design."""

    matrix_shape: tuple[int, int]
    backend: str = LOCKED_BACKEND

    def __post_init__(self) -> None:
        if len(self.matrix_shape) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.matrix_shape
        ):
            raise ValueError("matrix_shape must contain two positive integers")
        if math.prod(self.matrix_shape) > LOCKED_MAX_ENTRIES:
            raise ValueError(f"matrix_shape may contain at most 2^52={LOCKED_MAX_ENTRIES} entries")
        if self.backend != LOCKED_BACKEND:
            raise ValueError(f"the proposed implementation requires backend={LOCKED_BACKEND!r}")
        _require_locked_rounding_contract()

    @property
    def oriented_shape(self) -> tuple[int, int]:
        return min(self.matrix_shape), max(self.matrix_shape)


@dataclass(frozen=True)
class TwoTermBF16State:
    """The two BF16 tensors stored at one polynomial-stage boundary."""

    high: Tensor
    low: Tensor

    def reconstruct_fp32(self) -> Tensor:
        """Decode both terms and perform one materialized FP32 addition."""

        if self.high.dtype != torch.bfloat16 or self.low.dtype != torch.bfloat16:
            raise TypeError("both stored boundary terms must use torch.bfloat16")
        if self.high.shape != self.low.shape or self.high.device != self.low.device:
            raise ValueError("two-term BF16 boundary tensors must have matching metadata")
        return _check_finite(
            "two-term BF16 reconstruction",
            _fp32_add(self.high.to(torch.float32), self.low.to(torch.float32)),
        )


def _fp32_from_bits(bits: int) -> Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32, device="cpu").view(torch.float32)[0]


def _fp32_bits(value: Tensor) -> int:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("bit inspection requires one CPU float32 value")
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _require_locked_rounding_contract() -> None:
    """Probe necessary IEEE RNE and gradual-underflow behavior."""

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
        raise RuntimeError("CPU FP32 subnormal inputs must not be treated as zero")
    if int(bf16_tie.view(torch.uint16).item()) != 0x3F80:
        raise RuntimeError("CPU FP32-to-BF16 conversion must use ties-to-even")
    if int(bf16_min_subnormal.view(torch.uint16).item()) != 0x0001:
        raise RuntimeError("CPU BF16 gradual underflow is required; FTZ is excluded")
    if _fp32_bits(bf16_subnormal_readback) != 0x0001_0000:
        raise RuntimeError("CPU BF16 subnormal boundaries must survive FP32 readback")


def locked_backend_self_check() -> dict[str, object]:
    """Run and report the executable arithmetic-contract probes."""

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


def _check_finite(name: str, value: Tensor) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite value produced at {name}; overflow is excluded")
    return value


def _check_fp32_matrix(
    name: str,
    matrix: Tensor,
    config: ScalableMixedPrecisionConfig,
) -> None:
    if matrix.ndim != 2 or tuple(matrix.shape) != config.matrix_shape:
        raise ValueError(f"{name} must have configured shape {config.matrix_shape}")
    if matrix.device.type != "cpu":
        raise ValueError(f"{name} must be on the locked CPU backend")
    if matrix.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")
    if matrix.layout != torch.strided or not matrix.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")
    maximum = torch.amax(torch.abs(matrix))
    if bool(maximum > matrix.new_tensor(float(LOCKED_SAFE_MAX_ABS))):
        raise ValueError(
            f"{name} exceeds the certified max-absolute range 2^{LOCKED_SAFE_MAX_ABS_POWER}"
        )


def _fp32_mul(left: Tensor, right: Tensor) -> Tensor:
    return torch.mul(left, right)


def _fp32_add(left: Tensor, right: Tensor) -> Tensor:
    return torch.add(left, right)


def _fp32_sub(left: Tensor, right: Tensor) -> Tensor:
    return torch.sub(left, right)


def _fp32_div(left: Tensor, right: Tensor) -> Tensor:
    return torch.div(left, right)


def _ceil_log2(value: int) -> int:
    if value <= 0:
        raise ValueError("ceil(log2(value)) requires a positive integer")
    return (value - 1).bit_length()


def _balanced_sum_fp32(values: Tensor) -> Tensor:
    """Reduce a nonempty FP32 vector by fixed adjacent balanced additions."""

    if values.ndim != 1 or values.numel() == 0 or values.dtype != torch.float32:
        raise ValueError("balanced FP32 reduction requires a nonempty FP32 vector")
    level = values.contiguous()
    while level.numel() > 1:
        pair_count = level.numel() // 2
        paired = _fp32_add(level[: 2 * pair_count : 2], level[1 : 2 * pair_count : 2])
        level = torch.cat((paired, level[-1:].clone())) if level.numel() % 2 else paired
    return level[0]


def _balanced_mm_fp32(left: Tensor, right: Tensor) -> Tensor:
    """Reference matrix product with one balanced FP32 tree per dot.

    Products for one dot are materialized simultaneously in inner-index order.
    Adjacent pairs are added at every level; an unpaired final value is carried
    unchanged.  This gives maximum addition depth ``ceil(log2(inner))``.
    """

    if (
        left.ndim != 2
        or right.ndim != 2
        or left.shape[1] != right.shape[0]
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or left.device.type != "cpu"
        or right.device.type != "cpu"
    ):
        raise ValueError("balanced matrix product requires compatible CPU FP32 matrices")
    rows, columns = left.shape[0], right.shape[1]
    output = torch.empty((rows, columns), dtype=torch.float32, device="cpu")
    for row in range(rows):
        for column in range(columns):
            products = _fp32_mul(left[row, :], right[:, column])
            output[row, column] = _balanced_sum_fp32(products)
    return _check_finite("balanced FP32 matrix product", output)


def serial_normalizer_obstruction(matrix_shape: tuple[int, int]) -> dict[str, object]:
    """Report the exact large-shape obstruction to a serial-sum proof.

    The textbook serial reduction factor ``gamma_n=n*u/(1-n*u)`` requires
    ``n*u<1``.  It ceases to exist at ``n=2^24`` in FP32.  Moreover, summing
    unit squares sequentially in RNE FP32 absorbs every further ``+1`` after
    the partial sum reaches ``2^24``.  This executable record distinguishes a
    failure of serial normalization semantics from a failure of the balanced
    normalizer used by P9.
    """

    if len(matrix_shape) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in matrix_shape
    ):
        raise ValueError("matrix_shape must contain two positive integers")
    entry_count = math.prod(matrix_shape)
    n_times_u = Fraction(entry_count, 2**24)
    serial_unit_sum = min(entry_count, SERIAL_UNIT_SUM_ABSORPTION_COUNT)
    returned_singular_value_squared = Fraction(entry_count, serial_unit_sum)
    return {
        "matrix_shape": list(matrix_shape),
        "entry_count": entry_count,
        "fp32_unit_roundoff": str(FP32_UNIT_ROUNDOFF),
        "n_times_u": str(n_times_u),
        "standard_gamma_n_defined": n_times_u < 1,
        "first_absorbed_unit_increment_after": SERIAL_UNIT_SUM_ABSORPTION_COUNT,
        "unit_square_serial_sum": serial_unit_sum,
        "unit_square_exact_sum": entry_count,
        "unit_square_sum_is_exact": serial_unit_sum == entry_count,
        "returned_singular_value_squared": str(returned_singular_value_squared),
        "leaves_five_quarters_spectral_tube": returned_singular_value_squared > Fraction(25, 16),
        "balanced_addition_depth": _ceil_log2(entry_count),
    }


def fp32_balanced_max_floor_normalize(
    signal: Tensor,
    config: ScalableMixedPrecisionConfig,
) -> Tensor:
    """Evaluate the unit max-floor normalizer without a serial long sum.

    Put ``sigma=max(1,maxabs(signal))`` and ``z=fl32(signal/sigma)``.  The
    result is ``fl32(z/max(fl32(1/sigma),fl32(sqrt(sum(z*z)))))``.  Thus no
    large norm is ever materialized.  The computed denominator is at least
    one: if ``sigma>1`` a maximum-magnitude component of ``z`` is exactly one,
    while if ``sigma==1`` the floor term is exactly one.
    """

    _check_fp32_matrix("signal", signal, config)
    one = signal.new_tensor(float(LOCKED_FLOOR))
    maximum = torch.amax(torch.abs(signal))
    sigma = torch.maximum(one, maximum)
    scaled = _check_finite("normalizer maximum scaling", _fp32_div(signal, sigma))
    squares = _fp32_mul(scaled.reshape(-1), scaled.reshape(-1))
    scaled_square_sum = _balanced_sum_fp32(squares)
    scaled_norm = _check_finite("balanced scaled Frobenius norm", torch.sqrt(scaled_square_sum))
    inverse_sigma = _fp32_div(one, sigma)
    denominator = torch.maximum(inverse_sigma, scaled_norm)
    return _check_finite("balanced FP32 max-floor normalization", _fp32_div(scaled, denominator))


def _encode_two_term_bf16(name: str, value: Tensor) -> TwoTermBF16State:
    """Encode one FP32 boundary as a BF16 high/low expansion."""

    high = _check_finite(f"{name} BF16 high", value.to(torch.bfloat16))
    high_fp32 = high.to(torch.float32)
    residual = _check_finite(f"{name} FP32 residual", _fp32_sub(value, high_fp32))
    low = _check_finite(f"{name} BF16 low", residual.to(torch.bfloat16))
    return TwoTermBF16State(high=high, low=low)


def _fp32_stage_coefficients() -> tuple[Tensor, Tensor, Tensor]:
    return (
        _fp32_from_bits(COEFFICIENT_A_FP32_BITS),
        _fp32_from_bits(COEFFICIENT_B_FP32_BITS),
        _fp32_from_bits(COEFFICIENT_C_FP32_BITS),
    )


def _coefficient_identity(size: int, coefficient: Tensor) -> Tensor:
    """Materialize ``coefficient*I`` without an additional multiply."""

    result = torch.zeros((size, size), dtype=torch.float32, device="cpu")
    diagonal = torch.arange(size, device="cpu")
    result[diagonal, diagonal] = coefficient
    return result


def _jordan_stage_horner(stage_input: Tensor, *, stage: int) -> Tensor:
    """Evaluate one P8-order Horner Jordan stage in balanced FP32.

    The locked order is ``G=X X^T``, ``T=c32*G+b32*I``, ``D=T G``,
    ``E=D+a32*I``, and ``Y=E X``.  This is the exact operation graph used by
    the accompanying shape recurrence; algebraically equivalent thin forms
    have different rounding and are deliberately excluded.
    """

    x = stage_input
    a, b, c = _fp32_stage_coefficients()
    size = x.shape[0]
    gram = _balanced_mm_fp32(x, x.mT)
    scaled_gram = _fp32_mul(c, gram)
    shifted_gram = _fp32_add(scaled_gram, _coefficient_identity(size, b))
    product = _balanced_mm_fp32(shifted_gram, gram)
    affine = _fp32_add(product, _coefficient_identity(size, a))
    return _check_finite(f"stage {stage} FP32 body", _balanced_mm_fp32(affine, x))


def two_term_bf16_jordan_five_stage(
    normalized_signal: Tensor,
    *,
    return_stage_states: bool = False,
) -> Tensor | tuple[Tensor, tuple[TwoTermBF16State, ...]]:
    """Run five stages with a two-term BF16 expansion at every boundary."""

    if (
        normalized_signal.ndim != 2
        or normalized_signal.dtype != torch.float32
        or normalized_signal.device.type != "cpu"
    ):
        raise ValueError("normalized_signal must be one CPU FP32 matrix")
    state = _encode_two_term_bf16("initial boundary", normalized_signal)
    states: list[TwoTermBF16State] = []
    for stage in range(1, LOCKED_JORDAN_STEPS + 1):
        stage_input = state.reconstruct_fp32()
        stage_output = _jordan_stage_horner(stage_input, stage=stage)
        state = _encode_two_term_bf16(f"stage {stage} boundary", stage_output)
        if return_stage_states:
            states.append(
                TwoTermBF16State(
                    high=state.high.detach().clone(),
                    low=state.low.detach().clone(),
                )
            )
    output = state.reconstruct_fp32()
    if return_stage_states:
        return output, tuple(states)
    return output


def scalable_mixed_precision_repaired_operator(
    signal: Tensor,
    config: ScalableMixedPrecisionConfig,
) -> Tensor:
    """Evaluate the shape-locked P9 repaired max-floor operator."""

    _check_fp32_matrix("signal", signal, config)
    normalized = fp32_balanced_max_floor_normalize(signal, config)
    transposed = signal.shape[0] > signal.shape[1]
    oriented = normalized.mT.contiguous() if transposed else normalized
    polynomial = two_term_bf16_jordan_five_stage(oriented)
    if not isinstance(polynomial, Tensor):  # pragma: no cover - default argument guards this
        raise AssertionError("unexpected stage trace")
    restored = polynomial.mT.contiguous() if transposed else polynomial
    rho = _fp32_from_bits(REPAIR_RHO_FP32_BITS)
    repair = _check_finite("FP32 linear repair", _fp32_mul(signal, rho))
    return _check_finite("FP32 repaired output", _fp32_add(restored, repair))


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


def scalable_mixed_precision_manifest(
    config: ScalableMixedPrecisionConfig,
) -> dict[str, object]:
    """Return the exact shape-specific implementation contract."""

    source_path = Path(__file__).resolve()
    rows, columns = config.matrix_shape
    p, q = config.oriented_shape
    return {
        "schema_version": SCALABLE_MIXED_PRECISION_SCHEMA_VERSION,
        "claim_scope": (
            "shape-parameterized proof-reference repaired max-floor operator; "
            "not an upstream or production-throughput kernel"
        ),
        "matrix_domain": {
            "shape": [rows, columns],
            "oriented_shape": [p, q],
            "orientation_rule": "transpose once iff rows>columns; restore once at output",
            "layout": "contiguous torch.strided",
            "input_dtype": "torch.float32",
            "device": "cpu",
            "finite_inputs_required": True,
            "max_abs_inclusive": f"2^{LOCKED_SAFE_MAX_ABS_POWER}",
            "max_entries_inclusive": f"2^52 ({LOCKED_MAX_ENTRIES})",
        },
        "operator": {
            "normalization": "s/max(1,||s||_F)",
            "normalization_runtime": (
                "sigma=max(1,maxabs(s)); z=fl32(s/sigma); balanced FP32 sum of "
                "z^2; d=max(fl32(1/sigma),fl32(sqrt(sum))); fl32(z/d)"
            ),
            "floor_exact": str(LOCKED_FLOOR),
            "epsilon": "none",
            "polynomial": JORDAN_QUINTIC.name,
            "steps": LOCKED_JORDAN_STEPS,
            "stage_formula": "G=X X^T; T=c32 G+b32 I; D=T G; E=D+a32 I; Y=E X",
            "repair_rho_exact": str(LOCKED_REPAIR_RHO),
            "repair_rho_fp32_bits_hex": f"0x{REPAIR_RHO_FP32_BITS:08x}",
        },
        "precision_contract": {
            "normalizer_stage_body_repair_and_output": "torch.float32",
            "stored_stage_boundary": ("two BF16 tensors high=RN_b(Y), low=RN_b(fl32(Y-high))"),
            "boundary_reconstruction": "fl32(float32(high)+float32(low))",
            "matrix_product": (
                "each FP32 product materialized; each dot reduced by adjacent balanced "
                "FP32 additions; unpaired final values carried unchanged"
            ),
            "norm_reduction_depth": _ceil_log2(rows * columns),
            "gram_dot_reduction_depth": _ceil_log2(q),
            "square_product_dot_reduction_depth": _ceil_log2(p),
            "rounding": "IEEE-754 roundTiesToEven at every operation and cast",
            "gradual_underflow_required": True,
            "ftz_daz_allowed": False,
            "torch_matmul_used": False,
            "fma_allowed": False,
        },
        "serial_normalizer_obstruction": serial_normalizer_obstruction(config.matrix_shape),
        "backend": {
            "name": config.backend,
            "self_check": locked_backend_self_check(),
            "purpose": "slow executable proof reference",
            "overflow_policy": "reject maxabs>2^116 and every nonfinite intermediate",
        },
        "excluded_claims": [
            "literal upstream Muon parity or coverage",
            "production performance or native BLAS/GPU parity",
            "state compression; two BF16 buffers occupy one FP32 buffer's storage",
            "native BF16 matrix multiplication or tensor-core accumulation",
            "FP32 EMA/Nesterov state or parameter-update stability",
            "aspect-ratio scaling, weight decay, current-plus-epsilon normalization",
            "FTZ/DAZ, stochastic rounding, fused, compiler-reassociated executions",
        ],
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
    }


__all__ = [
    "LOCKED_BACKEND",
    "LOCKED_MAX_ENTRIES",
    "LOCKED_SAFE_MAX_ABS_POWER",
    "SCALABLE_MIXED_PRECISION_SCHEMA_VERSION",
    "ScalableMixedPrecisionConfig",
    "TwoTermBF16State",
    "fp32_balanced_max_floor_normalize",
    "locked_backend_self_check",
    "scalable_mixed_precision_manifest",
    "scalable_mixed_precision_repaired_operator",
    "serial_normalizer_obstruction",
    "two_term_bf16_jordan_five_stage",
]
