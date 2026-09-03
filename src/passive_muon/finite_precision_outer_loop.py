"""Executable FP32 outer shell for the proposed repaired P9 operator.

This module specifies a finite-precision arithmetic graph; it does not prove a
P7/P10 dissipation certificate and it does not claim parity with upstream
``lerp_`` kernels.  The gradient is supplied by the caller as one represented,
finite CPU FP32 sample.  The executable residuals are defined relative to that
sample; a surrounding certificate must separately specify how it relates to a
true gradient.  P10 absorbs one final exact-gradient-to-FP32 boundary cast,
while any earlier gradient-evaluation error remains P7's separate ``xi``
input.  The same rounded gradient product is reused in the EMA and Nesterov
expressions::

    bg = fl32(a32 * g)
    bm = fl32(beta32 * m)
    m_next = fl32(bm + bg)
    bs = fl32(beta32 * m_next)
    s_next = fl32(bs + bg)

where ``beta=19/20`` and ``a=1/20``.  Relative to the exact-real P7 ordering,
the two named residual ports are defined *exactly* by

``r^m = m_next - beta*m - a*g`` and
``r^s = s_next - beta*m_next - a*g``.

An :class:`ExactAffineResidualPort` retains that algebraic definition and can
return any entry as :class:`fractions.Fraction`; it does not confuse an FP64
diagnostic approximation with exact arithmetic.

Parameters use a three-word FP32 master ``high + middle + low``.  After the
rounded FP32 step is formed, the update graph is::

    pending = fl32(low - step)
    middle_candidate, low_next = TwoSum(middle, pending)
    high_next, middle_next = TwoSum(high, middle_candidate)

Subject to round-to-nearest, gradual underflow, and finite intermediates,
``TwoSum`` is error free.  Thus the logical real sum changes by ``pending``
exactly.  The only master-update residual relative to ``-eta*output`` is the
step multiplication plus ``low-step`` rounding; it is independent of the
magnitude of ``high``.  This is a proof-reference error-feedback design, not a
claim that model forward passes consume three-word parameters.

The P9 implementation is reached only through
:class:`P9RepairedOperatorAdapter`, which implements the small
:class:`FP32RepairedOperator` protocol.  No P9 private helper is imported.
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
from typing import Final, Protocol, runtime_checkable

import torch
from torch import Tensor

from passive_muon.scalable_mixed_precision import (
    LOCKED_MAX_ENTRIES,
    LOCKED_SAFE_MAX_ABS_POWER,
    SCALABLE_MIXED_PRECISION_SCHEMA_VERSION,
    ScalableMixedPrecisionConfig,
    scalable_mixed_precision_manifest,
    scalable_mixed_precision_repaired_operator,
)
from passive_muon.structure_aware_stability import LOCKED_BETA, LOCKED_LEARNING_RATE
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)

FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION: Final = "passive-muon-finite-precision-outer-loop-v1"
LOCKED_OUTER_BACKEND: Final = "torch-cpu-eager-ieee-rne-fp32-three-word-master-v1"
LOCKED_ONE_MINUS_BETA: Final = 1 - LOCKED_BETA

FP32_UNIT_ROUNDOFF: Final = Fraction(1, 2**24)
FP32_HALF_MIN_SUBNORMAL: Final = Fraction(1, 2**150)

# The master-word guards make every TwoSum intermediate finite and record the
# scale separation expected of a three-word FP32 expansion.  The output guard
# is deliberately tighter than P9's complete operator range: it is the P10
# outer-shell domain on which the step and word invariants are intended to be
# certified.
LOCKED_MASTER_HIGH_MAX_ABS_POWER: Final = 30
LOCKED_MASTER_MIDDLE_MAX_ABS_POWER: Final = 7
LOCKED_MASTER_LOW_MAX_ABS_POWER: Final = -16
LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER: Final = 16
CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS_POWER: Final = 15
LOCKED_ROUNDED_STEP_MAX_ABS_POWER: Final = 1
LOCKED_MASTER_HIGH_MAX_ABS: Final = Fraction(2**LOCKED_MASTER_HIGH_MAX_ABS_POWER)
LOCKED_MASTER_MIDDLE_MAX_ABS: Final = Fraction(2**LOCKED_MASTER_MIDDLE_MAX_ABS_POWER)
LOCKED_MASTER_LOW_MAX_ABS: Final = Fraction(1, 2 ** (-LOCKED_MASTER_LOW_MAX_ABS_POWER))
LOCKED_OPERATOR_OUTPUT_MAX_ABS: Final = Fraction(2**LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER)
CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS: Final = Fraction(
    2**CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS_POWER
)
LOCKED_ROUNDED_STEP_MAX_ABS: Final = Fraction(2**LOCKED_ROUNDED_STEP_MAX_ABS_POWER)

# Exact binary32 encodings of the separately represented runtime scalars.
BETA_FP32_BITS: Final = 0x3F73_3333
ONE_MINUS_BETA_FP32_BITS: Final = 0x3D4C_CCCD
LEARNING_RATE_FP32_BITS: Final = 0x3803_126F

BETA_FP32_EXACT: Final = Fraction(15_938_355, 16_777_216)
ONE_MINUS_BETA_FP32_EXACT: Final = Fraction(13_421_773, 268_435_456)
LEARNING_RATE_FP32_EXACT: Final = Fraction(8_589_935, 274_877_906_944)

_TWO_OPERATION_RELATIVE_FACTOR: Final = 2 * FP32_UNIT_ROUNDOFF + FP32_UNIT_ROUNDOFF**2
EMA_MOMENTUM_ENVELOPE_COEFFICIENT: Final = (
    abs(BETA_FP32_EXACT - LOCKED_BETA) + _TWO_OPERATION_RELATIVE_FACTOR * BETA_FP32_EXACT
)
EMA_GRADIENT_ENVELOPE_COEFFICIENT: Final = (
    abs(ONE_MINUS_BETA_FP32_EXACT - LOCKED_ONE_MINUS_BETA)
    + _TWO_OPERATION_RELATIVE_FACTOR * ONE_MINUS_BETA_FP32_EXACT
)
EMA_RESIDUAL_CRUMB_PER_ENTRY: Final = (3 + 2 * FP32_UNIT_ROUNDOFF) * FP32_HALF_MIN_SUBNORMAL
MASTER_OUTPUT_ENVELOPE_COEFFICIENT: Final = (
    abs(LEARNING_RATE_FP32_EXACT - LOCKED_LEARNING_RATE)
    + _TWO_OPERATION_RELATIVE_FACTOR * LEARNING_RATE_FP32_EXACT
)
MASTER_LOW_WORD_ENVELOPE_COEFFICIENT: Final = FP32_UNIT_ROUNDOFF
MASTER_RESIDUAL_CRUMB_PER_ENTRY: Final = (2 + FP32_UNIT_ROUNDOFF) * FP32_HALF_MIN_SUBNORMAL


def _fp32_from_bits(bits: int) -> Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32, device="cpu").view(torch.float32)[0]


def _fp32_bits(value: Tensor) -> int:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("bit inspection requires one CPU float32 value")
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _fp32_fraction(value: Tensor) -> Fraction:
    """Return the exact real value of one finite binary32 scalar."""

    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("exact conversion requires one CPU float32 value")
    runtime = float(value.reshape(1).item())
    if not math.isfinite(runtime):
        raise ValueError("exact conversion requires a finite value")
    return Fraction.from_float(runtime)


def _check_finite(name: str, value: Tensor) -> Tensor:
    if not bool(torch.isfinite(value).all()):
        raise FloatingPointError(f"nonfinite value produced at {name}; overflow is excluded")
    return value


def _check_fp32_matrix(
    name: str,
    matrix: Tensor,
    matrix_shape: tuple[int, int],
) -> None:
    if matrix.ndim != 2 or tuple(matrix.shape) != matrix_shape:
        raise ValueError(f"{name} must have configured shape {matrix_shape}")
    if matrix.device.type != "cpu":
        raise ValueError(f"{name} must be on the locked CPU backend")
    if matrix.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32")
    if matrix.layout != torch.strided or not matrix.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if not bool(torch.isfinite(matrix).all()):
        raise ValueError(f"{name} must contain only finite values")


def _check_max_abs(name: str, matrix: Tensor, bound: Fraction, label: str) -> None:
    maximum = torch.amax(torch.abs(matrix))
    if bool(maximum > matrix.new_tensor(float(bound))):
        raise ValueError(f"{name} exceeds the locked max-absolute range {label}")


def _ceil_sqrt(value: int) -> int:
    if value <= 0:
        raise ValueError("ceil(sqrt(value)) requires a positive integer")
    root = math.isqrt(value)
    return root if root * root == value else root + 1


def _fp32_mul(left: Tensor, right: Tensor) -> Tensor:
    return torch.mul(left, right)


def _fp32_add(left: Tensor, right: Tensor) -> Tensor:
    return torch.add(left, right)


def _fp32_sub(left: Tensor, right: Tensor) -> Tensor:
    return torch.sub(left, right)


def _require_outer_rounding_contract() -> None:
    """Probe necessary RNE and gradual-underflow properties of the CPU shell."""

    one = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    tie = _fp32_add(one, torch.tensor(2.0**-24, dtype=torch.float32, device="cpu"))
    minimum_subnormal = _fp32_mul(
        torch.tensor(2.0**-74, dtype=torch.float32, device="cpu"),
        torch.tensor(2.0**-75, dtype=torch.float32, device="cpu"),
    )
    subnormal_input = _fp32_add(
        _fp32_from_bits(0x0080_0000),
        _fp32_from_bits(0x0000_0001),
    )
    if _fp32_bits(tie) != 0x3F80_0000:
        raise RuntimeError("CPU FP32 addition does not satisfy the locked ties-to-even check")
    if _fp32_bits(minimum_subnormal) != 0x0000_0001:
        raise RuntimeError("CPU FP32 gradual underflow is required; FTZ/DAZ is excluded")
    if _fp32_bits(subnormal_input) != 0x0080_0001:
        raise RuntimeError("CPU FP32 subnormal inputs must not be treated as zero")


def locked_outer_backend_self_check() -> dict[str, object]:
    """Run and report the executable FP32 arithmetic probes."""

    _require_outer_rounding_contract()
    return {
        "fp32_add_halfway_bits_hex": "0x3f800000",
        "fp32_min_subnormal_bits_hex": "0x00000001",
        "fp32_subnormal_input_add_bits_hex": "0x00800001",
        "rounding": "IEEE-754 roundTiesToEven",
        "gradual_underflow": True,
        "ftz_daz": False,
    }


@dataclass(frozen=True)
class FinitePrecisionOuterLoopConfig:
    """One matrix-shape instance of the P10 outer arithmetic graph."""

    matrix_shape: tuple[int, int]
    backend: str = LOCKED_OUTER_BACKEND

    def __post_init__(self) -> None:
        if len(self.matrix_shape) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.matrix_shape
        ):
            raise ValueError("matrix_shape must contain two positive integers")
        if math.prod(self.matrix_shape) > LOCKED_MAX_ENTRIES:
            raise ValueError(f"matrix_shape may contain at most 2^52={LOCKED_MAX_ENTRIES} entries")
        if self.backend != LOCKED_OUTER_BACKEND:
            raise ValueError(f"the proposed outer shell requires backend={LOCKED_OUTER_BACKEND!r}")
        _require_outer_rounding_contract()


@dataclass(frozen=True)
class OuterResidualEnvelope:
    """Exact Frobenius-envelope coefficients for the three P10 ports.

    With ``n=rows*columns`` and ``k=ceil(sqrt(n))``, the componentwise FP32
    model gives

    ``||r^m|| <= c_beta||m|| + c_g||g|| + b_ema``,
    ``||r^s|| <= c_beta||m_next|| + c_g||g|| + b_ema``, and
    ``||r^W|| <= c_u||operator_output|| + u||low|| + b_master``.

    The ``k`` factor is a rational upper bound on ``sqrt(n)``.  These are
    arithmetic envelopes, not by themselves a closed-loop certificate.
    """

    entry_count: int
    sqrt_entry_count_upper: int
    ema_momentum_coefficient: Fraction
    ema_gradient_coefficient: Fraction
    ema_absolute_crumb: Fraction
    master_output_coefficient: Fraction
    master_low_word_coefficient: Fraction
    master_absolute_crumb: Fraction
    master_absolute_part_under_low_guard: Fraction
    master_absolute_bound_under_output_and_low_guards: Fraction
    master_absolute_bound_under_certificate_guards: Fraction


def outer_residual_envelope(
    config: FinitePrecisionOuterLoopConfig,
) -> OuterResidualEnvelope:
    """Return exact, shape-specific residual coefficients and guarded crumbs."""

    entry_count = math.prod(config.matrix_shape)
    sqrt_entries = _ceil_sqrt(entry_count)
    ema_crumb = sqrt_entries * EMA_RESIDUAL_CRUMB_PER_ENTRY
    master_crumb = sqrt_entries * MASTER_RESIDUAL_CRUMB_PER_ENTRY
    low_part = (
        sqrt_entries * MASTER_LOW_WORD_ENVELOPE_COEFFICIENT * LOCKED_MASTER_LOW_MAX_ABS
        + master_crumb
    )
    fully_guarded = (
        MASTER_OUTPUT_ENVELOPE_COEFFICIENT * sqrt_entries * LOCKED_OPERATOR_OUTPUT_MAX_ABS
        + low_part
    )
    certificate_guarded = (
        MASTER_OUTPUT_ENVELOPE_COEFFICIENT
        * sqrt_entries
        * CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS
        + low_part
    )
    return OuterResidualEnvelope(
        entry_count=entry_count,
        sqrt_entry_count_upper=sqrt_entries,
        ema_momentum_coefficient=EMA_MOMENTUM_ENVELOPE_COEFFICIENT,
        ema_gradient_coefficient=EMA_GRADIENT_ENVELOPE_COEFFICIENT,
        ema_absolute_crumb=ema_crumb,
        master_output_coefficient=MASTER_OUTPUT_ENVELOPE_COEFFICIENT,
        master_low_word_coefficient=MASTER_LOW_WORD_ENVELOPE_COEFFICIENT,
        master_absolute_crumb=master_crumb,
        master_absolute_part_under_low_guard=low_part,
        master_absolute_bound_under_output_and_low_guards=fully_guarded,
        master_absolute_bound_under_certificate_guards=certificate_guarded,
    )


def master_guard_invariant_audit() -> dict[str, object]:
    """Return exact sufficient checks preserving the middle/low word guards.

    The high-word guard is a separate conditional range premise checked again
    after every update.  It is not inferred from PL storage because the
    minimizer set may be translation invariant or nonunique.
    """

    pending_derived_upper = (1 + FP32_UNIT_ROUNDOFF) * (
        LOCKED_MASTER_LOW_MAX_ABS + LOCKED_ROUNDED_STEP_MAX_ABS
    ) + FP32_HALF_MIN_SUBNORMAL
    pending_strict_upper = Fraction(3)
    middle_candidate_derived_upper = (1 + FP32_UNIT_ROUNDOFF) * (
        LOCKED_MASTER_MIDDLE_MAX_ABS + pending_derived_upper
    ) + FP32_HALF_MIN_SUBNORMAL
    middle_candidate_strict_upper = Fraction(2**8)
    new_low_upper = Fraction(1, 2**17)
    new_middle_upper = Fraction(2**6)
    high_sum_abs_upper = LOCKED_MASTER_HIGH_MAX_ABS + middle_candidate_derived_upper
    certificate_step_upper = (
        (1 + FP32_UNIT_ROUNDOFF)
        * LEARNING_RATE_FP32_EXACT
        * CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS
        + FP32_HALF_MIN_SUBNORMAL
    )
    checks = {
        "pending_below_three": pending_derived_upper < pending_strict_upper,
        "middle_candidate_below_2^8": (
            middle_candidate_derived_upper < middle_candidate_strict_upper
        ),
        "new_low_within_guard": new_low_upper <= LOCKED_MASTER_LOW_MAX_ABS,
        "new_middle_within_guard": new_middle_upper <= LOCKED_MASTER_MIDDLE_MAX_ABS,
        "high_two_sum_cannot_overflow": high_sum_abs_upper < Fraction(2**31),
        "certificate_output_profile_implies_step_guard": (
            certificate_step_upper <= LOCKED_ROUNDED_STEP_MAX_ABS
        ),
    }
    return {
        "premises": {
            "old_high_max_abs": str(LOCKED_MASTER_HIGH_MAX_ABS),
            "old_middle_max_abs": str(LOCKED_MASTER_MIDDLE_MAX_ABS),
            "old_low_max_abs": str(LOCKED_MASTER_LOW_MAX_ABS),
            "rounded_step_max_abs": str(LOCKED_ROUNDED_STEP_MAX_ABS),
        },
        "derived_strict_upper": {
            "pending_derived": str(pending_derived_upper),
            "pending_lock": str(pending_strict_upper),
            "middle_candidate_derived": str(middle_candidate_derived_upper),
            "middle_candidate_lock": str(middle_candidate_strict_upper),
            "certificate_profile_rounded_step": str(certificate_step_upper),
        },
        "two_sum_residual_upper": {
            "new_middle": str(new_middle_upper),
            "new_low": str(new_low_upper),
        },
        "high_guard": (
            "maxabs(high)<=2^30 is checked again after the update; preservation is an "
            "explicit conditional premise and is not implied by PL storage"
        ),
        "checks": checks,
        "certified": all(checks.values()),
    }


@runtime_checkable
class FP32RepairedOperator(Protocol):
    """Clean callable interface consumed by the outer shell."""

    @property
    def matrix_shape(self) -> tuple[int, int]:
        """Return the one accepted matrix shape."""

    @property
    def interface_id(self) -> str:
        """Return a stable human-readable implementation identifier."""

    def __call__(self, signal: Tensor) -> Tensor:
        """Evaluate one repaired operator call."""

    def manifest_reference(self) -> dict[str, object]:
        """Return a compact provenance record for the bound implementation."""


@dataclass(frozen=True)
class P9RepairedOperatorAdapter:
    """Public-interface adapter for the P9 proof-reference operator."""

    config: ScalableMixedPrecisionConfig

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return self.config.matrix_shape

    @property
    def interface_id(self) -> str:
        return SCALABLE_MIXED_PRECISION_SCHEMA_VERSION

    def __call__(self, signal: Tensor) -> Tensor:
        return scalable_mixed_precision_repaired_operator(signal, self.config)

    def manifest_reference(self) -> dict[str, object]:
        manifest = scalable_mixed_precision_manifest(self.config)
        provenance = manifest["provenance"]
        if not isinstance(provenance, dict):  # pragma: no cover - fixed P9 manifest schema
            raise TypeError("P9 provenance must be a mapping")
        return {
            "interface_id": self.interface_id,
            "matrix_shape": list(self.matrix_shape),
            "operator": manifest["operator"],
            "precision_contract": manifest["precision_contract"],
            "source_file": provenance["source_file"],
            "source_sha256": provenance["source_sha256"],
            "claim_scope": manifest["claim_scope"],
        }


@dataclass(frozen=True)
class ExactAffineResidualPort:
    """An exact named residual ``realized - sum_i coefficient_i*input_i``.

    Tensors are references to the immutable-by-contract inputs and outputs of
    one completed step.  :meth:`exact_entry` converts binary32 values to exact
    fractions only on demand, so large-shape execution does not allocate a
    matrix of Python rationals.
    """

    symbol: str
    realized: Tensor
    terms: tuple[tuple[Fraction, Tensor], ...]
    formula: str

    def __post_init__(self) -> None:
        if self.symbol not in {"r^m", "r^s"}:
            raise ValueError("outer affine residual symbol must be r^m or r^s")
        if not self.terms:
            raise ValueError("an affine residual requires at least one reference term")
        shape = tuple(self.realized.shape)
        for coefficient, value in self.terms:
            if not isinstance(coefficient, Fraction):
                raise TypeError("residual coefficients must use fractions.Fraction")
            if tuple(value.shape) != shape:
                raise ValueError("all residual tensors must have one shape")

    def exact_entry(self, row: int, column: int) -> Fraction:
        """Return one residual entry as an authoritative exact fraction."""

        result = _fp32_fraction(self.realized[row, column])
        for coefficient, value in self.terms:
            result -= coefficient * _fp32_fraction(value[row, column])
        return result

    def float64_diagnostic(self) -> Tensor:
        """Return a non-authoritative FP64 visualization of the residual."""

        result = self.realized.to(torch.float64)
        for coefficient, value in self.terms:
            result = result - float(coefficient) * value.to(torch.float64)
        return result


@dataclass(frozen=True)
class OuterResidualPorts:
    """The separately placed EMA and Nesterov arithmetic residuals."""

    r_m: ExactAffineResidualPort
    r_s: ExactAffineResidualPort

    def __post_init__(self) -> None:
        if self.r_m.symbol != "r^m" or self.r_s.symbol != "r^s":
            raise ValueError("residual ports must retain the names r^m and r^s")


@dataclass(frozen=True)
class FP32EmaNesterovTrace:
    """Every named binary32 intermediate in the EMA/Nesterov graph."""

    beta_momentum: Tensor
    weighted_gradient: Tensor
    momentum_next: Tensor
    beta_momentum_next: Tensor
    signal: Tensor


@dataclass(frozen=True)
class FP32EmaNesterovResult:
    """Runtime state/signal plus exact algebraic residual-port definitions."""

    momentum: Tensor
    signal: Tensor
    residual_ports: OuterResidualPorts
    trace: FP32EmaNesterovTrace


def fp32_ema_nesterov(
    momentum: Tensor,
    gradient: Tensor,
    config: FinitePrecisionOuterLoopConfig,
) -> FP32EmaNesterovResult:
    """Evaluate the locked non-fused FP32 EMA/Nesterov arithmetic graph."""

    _check_fp32_matrix("momentum", momentum, config.matrix_shape)
    _check_fp32_matrix("gradient", gradient, config.matrix_shape)
    beta = _fp32_from_bits(BETA_FP32_BITS)
    weight = _fp32_from_bits(ONE_MINUS_BETA_FP32_BITS)

    beta_momentum = _check_finite("FP32 beta*momentum", _fp32_mul(beta, momentum))
    weighted_gradient = _check_finite("FP32 a*gradient", _fp32_mul(weight, gradient))
    momentum_next = _check_finite("FP32 EMA addition", _fp32_add(beta_momentum, weighted_gradient))
    beta_momentum_next = _check_finite("FP32 beta*momentum_next", _fp32_mul(beta, momentum_next))
    signal = _check_finite(
        "FP32 Nesterov addition", _fp32_add(beta_momentum_next, weighted_gradient)
    )

    r_m = ExactAffineResidualPort(
        symbol="r^m",
        realized=momentum_next,
        terms=((LOCKED_BETA, momentum), (LOCKED_ONE_MINUS_BETA, gradient)),
        formula="m_hat_next - (19/20)*m - (1/20)*g",
    )
    r_s = ExactAffineResidualPort(
        symbol="r^s",
        realized=signal,
        terms=((LOCKED_BETA, momentum_next), (LOCKED_ONE_MINUS_BETA, gradient)),
        formula="s_hat_next - (19/20)*m_hat_next - (1/20)*g",
    )
    trace = FP32EmaNesterovTrace(
        beta_momentum=beta_momentum,
        weighted_gradient=weighted_gradient,
        momentum_next=momentum_next,
        beta_momentum_next=beta_momentum_next,
        signal=signal,
    )
    return FP32EmaNesterovResult(
        momentum=momentum_next,
        signal=signal,
        residual_ports=OuterResidualPorts(r_m=r_m, r_s=r_s),
        trace=trace,
    )


@dataclass(frozen=True)
class TwoSumResult:
    """Rounded sum and error-free residual from one FP32 ``TwoSum``."""

    rounded: Tensor
    residual: Tensor


def two_sum_fp32(left: Tensor, right: Tensor, *, name: str = "TwoSum") -> TwoSumResult:
    """Apply Knuth ``TwoSum`` entrywise with separately materialized FP32 ops.

    For finite FP32 inputs, round-to-nearest, gradual underflow, and no
    overflow in the named intermediates, ``rounded + residual`` equals
    ``left + right`` exactly over the reals.  Every intermediate is checked;
    compiler fusion and reassociation are outside this reference contract.
    """

    if (
        left.shape != right.shape
        or left.dtype != torch.float32
        or right.dtype != torch.float32
        or left.device.type != "cpu"
        or right.device.type != "cpu"
    ):
        raise ValueError("TwoSum requires matching CPU float32 tensors")
    _check_finite(f"{name} left input", left)
    _check_finite(f"{name} right input", right)
    rounded = _check_finite(f"{name} rounded sum", _fp32_add(left, right))
    right_virtual = _check_finite(f"{name} right virtual", _fp32_sub(rounded, left))
    left_virtual = _check_finite(f"{name} left virtual", _fp32_sub(rounded, right_virtual))
    right_error = _check_finite(f"{name} right error", _fp32_sub(right, right_virtual))
    left_error = _check_finite(f"{name} left error", _fp32_sub(left, left_virtual))
    residual = _check_finite(f"{name} exact residual", _fp32_add(left_error, right_error))
    return TwoSumResult(rounded=rounded, residual=residual)


@dataclass(frozen=True)
class ThreeWordFP32Master:
    """A logical master parameter represented by three finite FP32 words."""

    high: Tensor
    middle: Tensor
    low: Tensor

    def __post_init__(self) -> None:
        shape = tuple(self.high.shape)
        if self.high.ndim != 2:
            raise ValueError("master words must be matrices")
        words = (("high", self.high), ("middle", self.middle), ("low", self.low))
        for name, word in words:
            _check_fp32_matrix(f"master {name}", word, shape)
        _check_max_abs(
            "master high",
            self.high,
            LOCKED_MASTER_HIGH_MAX_ABS,
            f"2^{LOCKED_MASTER_HIGH_MAX_ABS_POWER}",
        )
        _check_max_abs(
            "master middle",
            self.middle,
            LOCKED_MASTER_MIDDLE_MAX_ABS,
            f"2^{LOCKED_MASTER_MIDDLE_MAX_ABS_POWER}",
        )
        _check_max_abs(
            "master low",
            self.low,
            LOCKED_MASTER_LOW_MAX_ABS,
            f"2^{LOCKED_MASTER_LOW_MAX_ABS_POWER}",
        )

    @classmethod
    def from_primary(cls, primary: Tensor) -> ThreeWordFP32Master:
        """Initialize a master exactly from one finite contiguous FP32 matrix."""

        if primary.ndim != 2:
            raise ValueError("primary master value must be a matrix")
        _check_fp32_matrix("primary master value", primary, tuple(primary.shape))
        return cls(
            high=primary.clone(),
            middle=torch.zeros_like(primary),
            low=torch.zeros_like(primary),
        )

    @property
    def matrix_shape(self) -> tuple[int, int]:
        return tuple(self.high.shape)  # type: ignore[return-value]

    def exact_entry(self, row: int, column: int) -> Fraction:
        """Return one logical ``high+middle+low`` entry exactly."""

        return sum(
            (_fp32_fraction(word[row, column]) for word in (self.high, self.middle, self.low)),
            Fraction(0),
        )

    def float64_diagnostic(self) -> Tensor:
        """Return the non-authoritative FP64 sum of the three words."""

        return self.high.double() + self.middle.double() + self.low.double()


@dataclass(frozen=True)
class ExactMasterResidualPort:
    """Exact ``r^W = W_hat_next-W_hat+eta*operator_output`` definition."""

    previous: ThreeWordFP32Master
    updated: ThreeWordFP32Master
    operator_output: Tensor
    symbol: str = "r^W"

    def exact_entry(self, row: int, column: int) -> Fraction:
        return (
            self.updated.exact_entry(row, column)
            - self.previous.exact_entry(row, column)
            + LOCKED_LEARNING_RATE * _fp32_fraction(self.operator_output[row, column])
        )


@dataclass(frozen=True)
class CompensatedMasterUpdateTrace:
    """Every named intermediate in the three-word master update."""

    rounded_step: Tensor
    pending: Tensor
    middle_candidate: Tensor
    middle_two_sum_residual: Tensor
    high_two_sum_residual: Tensor


@dataclass(frozen=True)
class CompensatedMasterUpdateResult:
    """Updated master, exact residual-port definition, and arithmetic trace."""

    master: ThreeWordFP32Master
    residual_port: ExactMasterResidualPort
    trace: CompensatedMasterUpdateTrace


def compensated_master_weight_update(
    master: ThreeWordFP32Master,
    operator_output: Tensor,
    config: FinitePrecisionOuterLoopConfig,
) -> CompensatedMasterUpdateResult:
    """Apply one locked three-word FP32 error-feedback parameter update."""

    if master.matrix_shape != config.matrix_shape:
        raise ValueError(f"master must have configured shape {config.matrix_shape}")
    _check_fp32_matrix("operator_output", operator_output, config.matrix_shape)
    _check_max_abs(
        "operator_output",
        operator_output,
        LOCKED_OPERATOR_OUTPUT_MAX_ABS,
        f"2^{LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER}",
    )
    eta = _fp32_from_bits(LEARNING_RATE_FP32_BITS)
    rounded_step = _check_finite("FP32 eta*operator_output", _fp32_mul(eta, operator_output))
    _check_max_abs(
        "rounded parameter step",
        rounded_step,
        LOCKED_ROUNDED_STEP_MAX_ABS,
        f"2^{LOCKED_ROUNDED_STEP_MAX_ABS_POWER}",
    )
    pending = _check_finite("FP32 low-step", _fp32_sub(master.low, rounded_step))
    middle_update = two_sum_fp32(master.middle, pending, name="middle+pending TwoSum")
    high_update = two_sum_fp32(master.high, middle_update.rounded, name="high+middle TwoSum")
    updated = ThreeWordFP32Master(
        high=high_update.rounded,
        middle=high_update.residual,
        low=middle_update.residual,
    )
    return CompensatedMasterUpdateResult(
        master=updated,
        residual_port=ExactMasterResidualPort(
            previous=master,
            updated=updated,
            operator_output=operator_output,
        ),
        trace=CompensatedMasterUpdateTrace(
            rounded_step=rounded_step,
            pending=pending,
            middle_candidate=middle_update.rounded,
            middle_two_sum_residual=middle_update.residual,
            high_two_sum_residual=high_update.residual,
        ),
    )


def raw_fp32_parameter_update(
    position: Tensor,
    operator_output: Tensor,
    config: FinitePrecisionOuterLoopConfig,
) -> Tensor:
    """Apply the uncompensated FP32 subtraction used by the stall control."""

    _check_fp32_matrix("position", position, config.matrix_shape)
    _check_fp32_matrix("operator_output", operator_output, config.matrix_shape)
    _check_max_abs(
        "operator_output",
        operator_output,
        LOCKED_OPERATOR_OUTPUT_MAX_ABS,
        f"2^{LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER}",
    )
    eta = _fp32_from_bits(LEARNING_RATE_FP32_BITS)
    rounded_step = _check_finite("raw FP32 eta*operator_output", _fp32_mul(eta, operator_output))
    _check_max_abs(
        "raw rounded parameter step",
        rounded_step,
        LOCKED_ROUNDED_STEP_MAX_ABS,
        f"2^{LOCKED_ROUNDED_STEP_MAX_ABS_POWER}",
    )
    return _check_finite("raw FP32 parameter subtraction", _fp32_sub(position, rounded_step))


@dataclass(frozen=True)
class FinitePrecisionOuterStepResult:
    """One complete proposed outer-shell step around an operator interface."""

    master: ThreeWordFP32Master
    momentum: Tensor
    signal: Tensor
    operator_output: Tensor
    residual_ports: OuterResidualPorts
    master_residual_port: ExactMasterResidualPort
    ema_nesterov_trace: FP32EmaNesterovTrace
    master_update_trace: CompensatedMasterUpdateTrace


def finite_precision_outer_step(
    master: ThreeWordFP32Master,
    momentum: Tensor,
    gradient: Tensor,
    operator: FP32RepairedOperator,
    config: FinitePrecisionOuterLoopConfig,
) -> FinitePrecisionOuterStepResult:
    """Run one FP32 EMA/Nesterov, operator, and compensated-master step."""

    if master.matrix_shape != config.matrix_shape:
        raise ValueError(f"master must have configured shape {config.matrix_shape}")
    if not isinstance(operator, FP32RepairedOperator):
        raise TypeError("operator must implement the FP32RepairedOperator interface")
    if operator.matrix_shape != config.matrix_shape:
        raise ValueError(f"operator must have configured shape {config.matrix_shape}")

    ema = fp32_ema_nesterov(momentum, gradient, config)
    signal_maximum = torch.amax(torch.abs(ema.signal))
    if bool(signal_maximum > ema.signal.new_tensor(2.0**LOCKED_SAFE_MAX_ABS_POWER)):
        raise ValueError(
            f"FP32 Nesterov signal exceeds the P9 interface range 2^{LOCKED_SAFE_MAX_ABS_POWER}"
        )
    operator_output = operator(ema.signal)
    _check_fp32_matrix("operator return", operator_output, config.matrix_shape)
    update = compensated_master_weight_update(master, operator_output, config)
    return FinitePrecisionOuterStepResult(
        master=update.master,
        momentum=ema.momentum,
        signal=ema.signal,
        operator_output=operator_output,
        residual_ports=ema.residual_ports,
        master_residual_port=update.residual_port,
        ema_nesterov_trace=ema.trace,
        master_update_trace=update.trace,
    )


def _synthetic_two_step_stalling_control() -> dict[str, object]:
    """Return a minimal arithmetic-only control with an exactly accumulated step."""

    config = FinitePrecisionOuterLoopConfig((1, 1))
    initial = torch.tensor([[2.0**25]], dtype=torch.float32, device="cpu")
    operator_output = torch.tensor([[2.0**14]], dtype=torch.float32, device="cpu")
    raw_first = raw_fp32_parameter_update(initial, operator_output, config)
    raw_second = raw_fp32_parameter_update(raw_first, operator_output, config)
    master_initial = ThreeWordFP32Master.from_primary(initial)
    first = compensated_master_weight_update(master_initial, operator_output, config)
    second = compensated_master_weight_update(first.master, operator_output, config)
    initial_exact = master_initial.exact_entry(0, 0)
    rounded_step_exact = _fp32_fraction(first.trace.rounded_step[0, 0])
    checks = {
        "raw_stalls_twice": (
            _fp32_bits(raw_first[0, 0]) == _fp32_bits(raw_second[0, 0]) == _fp32_bits(initial[0, 0])
        ),
        "first_logical_step_accumulates_exactly": (
            first.master.exact_entry(0, 0) == initial_exact - rounded_step_exact
        ),
        "second_logical_step_accumulates_exactly": (
            second.master.exact_entry(0, 0) == initial_exact - 2 * rounded_step_exact
        ),
        "compensated_high_moves_by_second_step": (
            _fp32_bits(second.master.high[0, 0]) != _fp32_bits(initial[0, 0])
        ),
    }
    return {
        "scope": "synthetic arithmetic control; not an operator witness",
        "operator_output_exact": str(_fp32_fraction(operator_output[0, 0])),
        "rounded_step_exact": str(rounded_step_exact),
        "checks": checks,
        "certified": all(checks.values()),
    }


def fp32_master_stalling_witness() -> dict[str, object]:
    """Replay raw stalling and compensated progress using the actual P9 operator."""

    repetitions = 64
    shape = (2, 2)
    config = FinitePrecisionOuterLoopConfig(shape)
    operator = P9RepairedOperatorAdapter(ScalableMixedPrecisionConfig(shape))
    represented_gradient = torch.tensor(
        [[64.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device="cpu"
    )
    momentum = represented_gradient.clone()
    ema = fp32_ema_nesterov(momentum, represented_gradient, config)
    operator_output = operator(ema.signal)
    _check_max_abs(
        "stalling-witness operator output",
        operator_output,
        LOCKED_OPERATOR_OUTPUT_MAX_ABS,
        f"2^{LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER}",
    )
    initial = torch.tensor([[2.0**30, 0.0], [0.0, 0.0]], dtype=torch.float32, device="cpu")
    raw = initial.clone()
    master = ThreeWordFP32Master.from_primary(initial)
    raw_stalled_every_step = True
    first_high_move: int | None = None
    first_update: CompensatedMasterUpdateResult | None = None
    accumulated_master_residual = Fraction(0)
    for iteration in range(1, repetitions + 1):
        raw = raw_fp32_parameter_update(raw, operator_output, config)
        raw_stalled_every_step &= _fp32_bits(raw[0, 0]) == _fp32_bits(initial[0, 0])
        update = compensated_master_weight_update(master, operator_output, config)
        if first_update is None:
            first_update = update
        accumulated_master_residual += update.residual_port.exact_entry(0, 0)
        master = update.master
        if first_high_move is None and _fp32_bits(master.high[0, 0]) != _fp32_bits(initial[0, 0]):
            first_high_move = iteration
    if first_update is None:  # pragma: no cover - repetitions is a positive lock
        raise AssertionError("stalling witness ran no updates")

    initial_exact = _fp32_fraction(initial[0, 0])
    final_logical_exact = master.exact_entry(0, 0)
    operator_output_exact = _fp32_fraction(operator_output[0, 0])
    rounded_step_exact = _fp32_fraction(first_update.trace.rounded_step[0, 0])
    source_step_exact = LOCKED_LEARNING_RATE * operator_output_exact
    checks = {
        "ema_signal_is_exact_control": torch.equal(ema.signal, represented_gradient),
        "r_m_is_exactly_zero": ema.residual_ports.r_m.exact_entry(0, 0) == 0,
        "r_s_is_exactly_zero": ema.residual_ports.r_s.exact_entry(0, 0) == 0,
        "uses_actual_p9_operator": operator.interface_id == SCALABLE_MIXED_PRECISION_SCHEMA_VERSION,
        "operator_output_is_nonzero": operator_output_exact > 0,
        "rounded_step_is_below_downward_half_ulp": 0 < rounded_step_exact < 32,
        "raw_fp32_stalls_for_every_repetition": raw_stalled_every_step,
        "compensated_middle_records_first_lost_step": (
            _fp32_fraction(first_update.master.middle[0, 0]) == -rounded_step_exact
        ),
        "compensated_high_eventually_moves": first_high_move is not None,
        "compensated_logical_value_decreases": final_logical_exact < initial_exact,
        "exact_telescoping_identity": (
            final_logical_exact
            == initial_exact - repetitions * source_step_exact + accumulated_master_residual
        ),
    }
    return {
        "scope": "actual P9 repaired-operator parameter-stalling witness",
        "matrix_shape": list(shape),
        "represented_gradient_00_exact": str(_fp32_fraction(represented_gradient[0, 0])),
        "momentum_00_exact": str(_fp32_fraction(momentum[0, 0])),
        "signal_00_exact": str(_fp32_fraction(ema.signal[0, 0])),
        "r_m_00_exact": str(ema.residual_ports.r_m.exact_entry(0, 0)),
        "r_s_00_exact": str(ema.residual_ports.r_s.exact_entry(0, 0)),
        "operator_interface_id": operator.interface_id,
        "operator_output_00_exact": str(operator_output_exact),
        "operator_output_00_bits_hex": f"0x{_fp32_bits(operator_output[0, 0]):08x}",
        "learning_rate_source_exact": str(LOCKED_LEARNING_RATE),
        "learning_rate_runtime_exact": str(
            _fp32_fraction(_fp32_from_bits(LEARNING_RATE_FP32_BITS))
        ),
        "source_step_exact": str(source_step_exact),
        "rounded_step_exact": str(rounded_step_exact),
        "step_rounding_residual_exact": str(rounded_step_exact - source_step_exact),
        "initial_exact": str(initial_exact),
        "initial_fp32_bits_hex": f"0x{_fp32_bits(initial[0, 0]):08x}",
        "repetitions": repetitions,
        "raw_final_bits_hex": f"0x{_fp32_bits(raw[0, 0]):08x}",
        "compensated_first_high_bits_hex": (f"0x{_fp32_bits(first_update.master.high[0, 0]):08x}"),
        "compensated_first_middle_exact": str(_fp32_fraction(first_update.master.middle[0, 0])),
        "compensated_first_high_move_iteration": first_high_move,
        "compensated_final": {
            "high_exact": str(_fp32_fraction(master.high[0, 0])),
            "middle_exact": str(_fp32_fraction(master.middle[0, 0])),
            "low_exact": str(_fp32_fraction(master.low[0, 0])),
            "logical_exact": str(final_logical_exact),
        },
        "accumulated_master_residual_exact": str(accumulated_master_residual),
        "synthetic_arithmetic_control": _synthetic_two_step_stalling_control(),
        "checks": checks,
        "certified": all(checks.values()),
    }


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


def _scalar_record(source: Fraction, bits: int) -> dict[str, object]:
    runtime = _fp32_fraction(_fp32_from_bits(bits))
    return {
        "source_exact": str(source),
        "runtime_exact": str(runtime),
        "runtime_fp32_bits_hex": f"0x{bits:08x}",
        "representation_residual_exact": str(runtime - source),
    }


def finite_precision_outer_loop_manifest(
    config: FinitePrecisionOuterLoopConfig,
    operator: FP32RepairedOperator,
) -> dict[str, object]:
    """Return the locked executable contract and local source provenance."""

    if not isinstance(operator, FP32RepairedOperator):
        raise TypeError("operator must implement the FP32RepairedOperator interface")
    if operator.matrix_shape != config.matrix_shape:
        raise ValueError(f"operator must have configured shape {config.matrix_shape}")
    source_path = Path(__file__).resolve()
    envelope = outer_residual_envelope(config)
    return {
        "schema_version": FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION,
        "claim_scope": (
            "executable FP32 EMA/Nesterov and three-word compensated-master shell "
            "around one bound repaired-operator interface; no stability theorem"
        ),
        "matrix_domain": {
            "shape": list(config.matrix_shape),
            "layout": "contiguous torch.strided",
            "dtype": "torch.float32",
            "device": "cpu",
            "finite_inputs_and_intermediates_required": True,
            "max_entries_inclusive": f"2^52 ({LOCKED_MAX_ENTRIES})",
            "operator_signal_max_abs_inclusive": f"2^{LOCKED_SAFE_MAX_ABS_POWER}",
            "master_word_max_abs_inclusive": {
                "high": f"2^{LOCKED_MASTER_HIGH_MAX_ABS_POWER}",
                "middle": f"2^{LOCKED_MASTER_MIDDLE_MAX_ABS_POWER}",
                "low": f"2^{LOCKED_MASTER_LOW_MAX_ABS_POWER}",
            },
            "operator_output_max_abs_inclusive": (f"2^{LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER}"),
            "certified_storage_profile_operator_output_max_abs_inclusive": (
                f"2^{CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS_POWER}"
            ),
            "rounded_step_max_abs_inclusive": f"2^{LOCKED_ROUNDED_STEP_MAX_ABS_POWER}",
        },
        "runtime_scalars": {
            "beta": _scalar_record(LOCKED_BETA, BETA_FP32_BITS),
            "one_minus_beta": _scalar_record(
                LOCKED_ONE_MINUS_BETA,
                ONE_MINUS_BETA_FP32_BITS,
            ),
            "learning_rate": _scalar_record(
                LOCKED_LEARNING_RATE,
                LEARNING_RATE_FP32_BITS,
            ),
        },
        "ema_nesterov_graph": [
            "g is the represented FP32 gradient sample; evaluation error remains P7 xi",
            "bg=fl32(a32*g)",
            "bm=fl32(beta32*m)",
            "m_next=fl32(bm+bg)",
            "bs=fl32(beta32*m_next)",
            "s_next=fl32(bs+bg); reuse the same bg tensor",
        ],
        "ordering_context": {
            "pinned_upstream_revision": PINNED_KELLER_JORDAN_MUON_REVISION,
            "pinned_muon_py_sha256": PINNED_MUON_PY_SHA256,
            "scope": (
                "same EMA/Nesterov algebraic ordering only; the separately rounded graph "
                "does not claim bit parity with upstream torch.lerp_"
            ),
        },
        "exact_residual_ports": {
            "r^m": "m_next-(19/20)*m-(1/20)*g",
            "r^s": "s_next-(19/20)*m_next-(1/20)*g",
            "placement": ("m_next=(19/20)m+(1/20)g+r^m; s_next=(19/20)m_next+(1/20)g+r^s"),
            "frobenius_envelope": {
                "r^m": (
                    f"{envelope.ema_momentum_coefficient}*||m||_F + "
                    f"{envelope.ema_gradient_coefficient}*||g||_F + "
                    f"{envelope.ema_absolute_crumb}"
                ),
                "r^s": (
                    f"{envelope.ema_momentum_coefficient}*||m_next||_F + "
                    f"{envelope.ema_gradient_coefficient}*||g||_F + "
                    f"{envelope.ema_absolute_crumb}"
                ),
                "r^W": (
                    f"{envelope.master_output_coefficient}*||operator_output||_F + "
                    f"{envelope.master_low_word_coefficient}*||low||_F + "
                    f"{envelope.master_absolute_crumb}"
                ),
                "roundoff_model": (
                    "|fl32(x)-x|<=2^-24|x|+2^-150 entrywise; "
                    "ceil(sqrt(rows*columns)) converts crumbs to Frobenius bounds"
                ),
            },
        },
        "master_update": {
            "logical_value": "high+middle+low over the reals",
            "graph": [
                "step=fl32(eta32*operator_output)",
                "pending=fl32(low-step)",
                "middle_candidate,new_low=TwoSum(middle,pending)",
                "new_high,new_middle=TwoSum(high,middle_candidate)",
            ],
            "two_sum_contract": (
                "error free under RNE, gradual underflow, and finite named intermediates"
            ),
            "exact_residual_port": (
                "r^W=(new_high+new_middle+new_low)-(high+middle+low)+(1/32000)*operator_output"
            ),
            "guarded_r^W_absolute_bound": str(
                envelope.master_absolute_bound_under_output_and_low_guards
            ),
            "certified_storage_profile_r^W_absolute_bound": str(
                envelope.master_absolute_bound_under_certificate_guards
            ),
            "word_guard_audit": master_guard_invariant_audit(),
            "stalling_witness": fp32_master_stalling_witness(),
        },
        "operator_interface": operator.manifest_reference(),
        "precision_contract": {
            "backend": config.backend,
            "rounding": "IEEE-754 roundTiesToEven at every named operation",
            "gradual_underflow_required": True,
            "ftz_daz_allowed": False,
            "fma_allowed": False,
            "torch_compile_used": False,
            "input_mutation": False,
            "exact_port_tensor_lifetime": (
                "input tensors referenced by ExactAffineResidualPort must not be mutated "
                "before exact_entry is queried"
            ),
        },
        "backend_self_check": locked_outer_backend_self_check(),
        "excluded_claims": [
            "a P7/P10 stability or convergence certificate",
            "literal upstream Muon or upstream torch.lerp_ parity",
            "gradient-evaluation rounding or stochastic-gradient semantics",
            "model-forward consumption of the three-word logical parameter",
            "full-state or unique-parameter convergence on a nonunique PL minimizer set",
            "native BLAS, GPU, tensor-core, fused, or compiler-reassociated execution",
            "weight decay, aspect-ratio scaling, current-plus-epsilon normalization",
            "overflow, FTZ/DAZ, stochastic rounding, or nonfinite states",
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
    "BETA_FP32_BITS",
    "BETA_FP32_EXACT",
    "CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS",
    "CERTIFIED_STORAGE_PROFILE_OPERATOR_OUTPUT_MAX_ABS_POWER",
    "EMA_GRADIENT_ENVELOPE_COEFFICIENT",
    "EMA_MOMENTUM_ENVELOPE_COEFFICIENT",
    "EMA_RESIDUAL_CRUMB_PER_ENTRY",
    "FINITE_PRECISION_OUTER_LOOP_SCHEMA_VERSION",
    "FP32_HALF_MIN_SUBNORMAL",
    "FP32_UNIT_ROUNDOFF",
    "LEARNING_RATE_FP32_BITS",
    "LEARNING_RATE_FP32_EXACT",
    "LOCKED_MASTER_HIGH_MAX_ABS",
    "LOCKED_MASTER_HIGH_MAX_ABS_POWER",
    "LOCKED_MASTER_LOW_MAX_ABS_POWER",
    "LOCKED_MASTER_MIDDLE_MAX_ABS",
    "LOCKED_MASTER_MIDDLE_MAX_ABS_POWER",
    "LOCKED_OPERATOR_OUTPUT_MAX_ABS",
    "LOCKED_OPERATOR_OUTPUT_MAX_ABS_POWER",
    "LOCKED_OUTER_BACKEND",
    "LOCKED_ROUNDED_STEP_MAX_ABS",
    "LOCKED_ROUNDED_STEP_MAX_ABS_POWER",
    "MASTER_LOW_WORD_ENVELOPE_COEFFICIENT",
    "MASTER_OUTPUT_ENVELOPE_COEFFICIENT",
    "MASTER_RESIDUAL_CRUMB_PER_ENTRY",
    "ONE_MINUS_BETA_FP32_BITS",
    "ONE_MINUS_BETA_FP32_EXACT",
    "CompensatedMasterUpdateResult",
    "CompensatedMasterUpdateTrace",
    "ExactAffineResidualPort",
    "ExactMasterResidualPort",
    "FP32EmaNesterovResult",
    "FP32EmaNesterovTrace",
    "FP32RepairedOperator",
    "FinitePrecisionOuterLoopConfig",
    "FinitePrecisionOuterStepResult",
    "OuterResidualEnvelope",
    "OuterResidualPorts",
    "P9RepairedOperatorAdapter",
    "ThreeWordFP32Master",
    "TwoSumResult",
    "compensated_master_weight_update",
    "finite_precision_outer_loop_manifest",
    "finite_precision_outer_step",
    "fp32_ema_nesterov",
    "fp32_master_stalling_witness",
    "locked_outer_backend_self_check",
    "master_guard_invariant_audit",
    "outer_residual_envelope",
    "raw_fp32_parameter_update",
    "two_sum_fp32",
]
