"""Scalable mixed-precision proof-reference implementation of the P19 shield.

P19 checks every binary64 result with exact dyadic arithmetic.  This module
instead locks a binary32 operation graph whose *static* P20 certificate pays
for all rounding through an inward disk reserve.  Given the P18/P19 disk

``||U - gamma*S||_F <= r*||S||_F``,

the runtime screens a finite candidate with coefficient ``k=r-1/1024``.  The
static shape audit proves that any accepted stored candidate lies at a
shape-dependent positive distance inside the original disk.  A rejected
finite candidate is radially clipped using the stricter coefficient
``k_clip=890/2048`` and directed-down FP64/FP32 scale construction.  The
fallback ``fl32(S/2)`` is used only if clipping cannot be computed.  It is
well inside the disk under the certified normal-anchor guard.  An
all-subnormal nonzero signal is accepted only when halving every stored entry
is exact; otherwise the call fails closed.

This is a CPU proof-reference graph, not a claim about an arbitrary BLAS,
GPU, compiler, or tensor-core implementation.  Inputs may be FP32 or BF16;
both are widened once to FP32 before any arithmetic and the output is FP32.
Norm squares use materialized FP32 products and fixed adjacent balanced FP32
additions.  FP32 norm parts widen into one exact FP64 scalar product; the
acceptance threshold is the only other FP64 operation.  FMA contraction,
reassociation, FTZ/DAZ, and stochastic rounding are excluded.  An exact
generator supplies a frozen shape table, and no runtime ``Fraction`` or
big-integer calculation is used.
"""

from __future__ import annotations

import hashlib
import math
import platform
import subprocess
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

import torch
from torch import Tensor

SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION: Final = "passive-muon-scalable-sector-shield-v1"
LOCKED_SCALABLE_SHIELD_BACKEND: Final = (
    "torch-cpu-eager-ieee-rne-balanced-fp32-screen-radial-clip-half-fallback-v1"
)

# Exact binary32 dyadics.  The exact P19 radius is 893/2048 and the locked
# acceptance radius is two 1/2048 ticks inward.
SHIELD_CENTER_FP32_BITS: Final = 0x3F0E_E000  # 1143/2048
SHIELD_RADIUS_FP32_BITS: Final = 0x3EDF_4000  # 893/2048
ACCEPTANCE_RADIUS_FP32_BITS: Final = 0x3EDE_C000  # 891/2048
CLIP_COEFFICIENT_FP32_BITS: Final = 0x3EDE_8000  # 890/2048 = 445/1024
FALLBACK_GAIN_FP32_BITS: Final = 0x3F00_0000  # 1/2

LOCKED_SHIELD_CENTER: Final = 1143.0 / 2048.0
LOCKED_SHIELD_RADIUS: Final = 893.0 / 2048.0
LOCKED_ACCEPTANCE_RADIUS: Final = 891.0 / 2048.0
LOCKED_CLIP_COEFFICIENT: Final = 890.0 / 2048.0
LOCKED_RESERVED_INWARD_MARGIN: Final = 1.0 / 1024.0
LOCKED_FALLBACK_GAIN: Final = 0.5

FP32_MIN_NORMAL: Final = 2.0**-126
FP32_MIN_SUBNORMAL: Final = 2.0**-149
LOCKED_REDUCTION_BLOCK_SIZE: Final = 2**20
LOCKED_MAX_ENTRIES: Final = 2**52

LOCKED_REPRESENTATIVE_SHAPES: Final = (
    (768, 768),
    (768, 3_072),
    (768, 50_257),
    (3_072, 12_288),
    (4_096, 4_096),
    (4_096, 11_008),
    (4_096, 14_336),
)
LOCKED_DIAGNOSTIC_SHAPES: Final = ((1, 1), (1, 2), (2, 2))
LOCKED_CERTIFIED_SHAPES: Final = LOCKED_DIAGNOSTIC_SHAPES + LOCKED_REPRESENTATIVE_SHAPES

# Generated from the exact P20 shape audits.  Hexadecimal binary64 literals
# avoid runtime rational arithmetic and make the embedded values replayable.
LOCKED_SHAPE_CERTIFIED_RADIUS_HEX: Final = {
    (1, 1): "0x1.bd8014ac80326p-2",
    (1, 2): "0x1.bd801e6a004efp-2",
    (2, 2): "0x1.bd8020278064bp-2",
    (768, 768): "0x1.bd982f7aa9510p-2",
    (768, 3_072): "0x1.bdb032f5ce482p-2",
    (768, 50_257): "0x1.be4261ecabb10p-2",
    (3_072, 12_288): "0x1.be4039eca8750p-2",
    (4_096, 4_096): "0x1.be00367147542p-2",
    (4_096, 11_008): "0x1.be5211ecc3390p-2",
    (4_096, 14_336): "0x1.be6fb1ecefa90p-2",
}
LOCKED_SHAPE_INWARD_MARGIN_HEX: Final = {
    (1, 1): "0x1.ffd6a6ff9b330p-11",
    (1, 2): "0x1.ffc32bff62228p-11",
    (2, 2): "0x1.ffbfb0ff36a10p-11",
    (768, 768): "0x1.cfa10aad5e098p-11",
    (768, 3_072): "0x1.9f9a14636fcf0p-11",
    (768, 50_257): "0x1.ecf09aa278300p-13",
    (3_072, 12_288): "0x1.fe309abc58300p-13",
    (4_096, 4_096): "0x1.ff263ae2af9f0p-12",
    (4_096, 11_008): "0x1.6f7099e638300p-13",
    (4_096, 14_336): "0x1.04e13105705c0p-14",
}


class SignalGuardClass(StrEnum):
    """Exhaustive signal classes relevant to the P20 representability guard."""

    ZERO = "zero"
    NORMAL_ANCHORED = "normal_anchored"
    SUBNORMAL_EXACT_HALVING = "subnormal_exact_halving"


class ShieldAction(StrEnum):
    """The output path taken by one successful shield call."""

    PASS_THROUGH = "pass_through"
    RADIAL_CLIP = "radial_clip"
    HALF_FALLBACK = "half_fallback"
    ZERO_SINGLETON = "zero_singleton"


class ScalableSectorShieldFailure(FloatingPointError):
    """Raised when the locked graph cannot emit a sector-certified output."""


class NearZeroUnrepresentable(ScalableSectorShieldFailure):
    """An all-subnormal signal whose dyadic half is not exactly representable."""

    def __init__(
        self,
        *,
        matrix_shape: tuple[int, int],
        signal_max_abs: float,
        signal_frobenius_strict_upper: float,
    ) -> None:
        self.matrix_shape = matrix_shape
        self.signal_max_abs = signal_max_abs
        self.signal_frobenius_strict_upper = signal_frobenius_strict_upper
        super().__init__(
            "all nonzero signal entries are FP32-subnormal and S/2 is not exactly "
            "representable; no sector-certified output is emitted; the excluded "
            "near-zero region has ||S||_F < "
            f"{signal_frobenius_strict_upper:.17g} for shape {matrix_shape}"
        )


@dataclass(frozen=True)
class ScalableSectorShieldConfig:
    """One statically audited P20 matrix-shape/backend instance."""

    matrix_shape: tuple[int, int]
    backend: str = LOCKED_SCALABLE_SHIELD_BACKEND

    def __post_init__(self) -> None:
        if len(self.matrix_shape) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.matrix_shape
        ):
            raise ValueError("matrix_shape must contain two positive integers")
        if math.prod(self.matrix_shape) > LOCKED_MAX_ENTRIES:
            raise ValueError(f"matrix_shape may contain at most 2^52={LOCKED_MAX_ENTRIES} entries")
        if self.matrix_shape not in LOCKED_CERTIFIED_SHAPES:
            raise ValueError("matrix_shape is not in the generated P20 certified shape table")
        if self.backend != LOCKED_SCALABLE_SHIELD_BACKEND:
            raise ValueError(
                f"the scalable shield requires backend={LOCKED_SCALABLE_SHIELD_BACKEND!r}"
            )
        _require_locked_rounding_contract()

    @property
    def entry_count(self) -> int:
        return math.prod(self.matrix_shape)

    @property
    def balanced_reduction_depth(self) -> int:
        return (self.entry_count - 1).bit_length()

    @property
    def near_zero_frobenius_upper(self) -> float:
        """Strict upper bound for an all-subnormal signal's Frobenius norm."""

        return math.sqrt(self.entry_count) * FP32_MIN_NORMAL

    @property
    def certified_inward_margin(self) -> float:
        """Shape-specific exact-audit margin, exposed as a diagnostic float."""

        return float.fromhex(LOCKED_SHAPE_INWARD_MARGIN_HEX[self.matrix_shape])

    @property
    def certified_inward_radius(self) -> float:
        """Shape-specific outward-rounded radius, exposed as a diagnostic float."""

        return float.fromhex(LOCKED_SHAPE_CERTIFIED_RADIUS_HEX[self.matrix_shape])


@dataclass(frozen=True)
class ScalableNormDiagnostics:
    """Trace of one locked scale-free balanced FP32 norm computation."""

    maximum_absolute_value: float
    scaled_square_sum: float
    scaled_norm: float
    returned_norm: float | None
    final_binary64_product_exact: bool
    reduction_depth: int
    reduction_block_size: int


@dataclass(frozen=True)
class ScalableSectorShieldDiagnostics:
    """Runtime trace for one successful P20 shield call."""

    signal_input_dtype: str
    candidate_input_dtype: str
    widened_dtype: str
    output_dtype: str
    signal_guard_class: SignalGuardClass
    normal_anchor: bool
    exact_halving_guard: bool
    candidate_finite: bool
    candidate_accepted: bool
    candidate_clipped: bool
    active: bool
    used_fallback: bool
    fail_closed: bool
    action: ShieldAction
    reason: str | None
    signal_norm: ScalableNormDiagnostics | None
    displacement_norm: ScalableNormDiagnostics | None
    acceptance_threshold: float | None
    clip_ratio_fp64_downward: float | None
    clip_scale_fp64_downward: float | None
    clip_alpha_fp32: float | None
    clip_alpha_fp32_bits: int | None
    original_radius: float
    acceptance_radius: float
    clip_coefficient: float
    reserved_inward_margin: float
    certified_inward_margin: float
    certified_inward_radius: float
    backend: str
    reduced_spectrum_diagnostic: bool

    @property
    def certified_by_static_contract(self) -> bool:
        """Successful results are certified by the P20 static contract."""

        return True


@dataclass(frozen=True)
class ScalableSectorShieldResult:
    """Stored FP32 output and its proof-reference execution trace."""

    output: Tensor
    diagnostics: ScalableSectorShieldDiagnostics


def _fp32_from_bits(bits: int) -> Tensor:
    signed = bits if bits < 2**31 else bits - 2**32
    return torch.tensor([signed], dtype=torch.int32, device="cpu").view(torch.float32)[0]


def _fp32_bits(value: Tensor) -> int:
    if value.numel() != 1 or value.dtype != torch.float32 or value.device.type != "cpu":
        raise TypeError("bit inspection requires one CPU float32 scalar")
    return int(value.reshape(1).view(torch.int32).item()) & 0xFFFF_FFFF


def _require_locked_rounding_contract() -> None:
    """Probe RNE, gradual underflow, subnormal inputs, and BF16 widening."""

    one = torch.tensor(1.0, dtype=torch.float32, device="cpu")
    fp32_tie = torch.add(one, torch.tensor(2.0**-24, dtype=torch.float32, device="cpu"))
    fp32_min_subnormal = torch.mul(
        torch.tensor(2.0**-74, dtype=torch.float32, device="cpu"),
        torch.tensor(2.0**-75, dtype=torch.float32, device="cpu"),
    )
    fp32_subnormal_input_add = torch.add(
        _fp32_from_bits(0x0080_0000),
        _fp32_from_bits(0x0000_0001),
    )
    bf16_tie = torch.tensor(1.0 + 2.0**-8, dtype=torch.float32).to(torch.bfloat16)
    bf16_min_subnormal = torch.tensor(2.0**-133, dtype=torch.float32).to(torch.bfloat16)
    bf16_readback = bf16_min_subnormal.to(torch.float32)
    fp64_to_fp32_tie = torch.tensor(
        1.0 + 2.0**-24,
        dtype=torch.float64,
        device="cpu",
    ).to(torch.float32)
    fp32_next_down = torch.nextafter(one, torch.zeros((), dtype=torch.float32))
    if _fp32_bits(fp32_tie) != 0x3F80_0000:
        raise RuntimeError("CPU FP32 addition must use round-to-nearest, ties-to-even")
    if _fp32_bits(fp32_min_subnormal) != 0x0000_0001:
        raise RuntimeError("CPU FP32 gradual underflow is required; FTZ/DAZ is excluded")
    if _fp32_bits(fp32_subnormal_input_add) != 0x0080_0001:
        raise RuntimeError("CPU FP32 subnormal inputs must not be treated as zero")
    if int(bf16_tie.view(torch.uint16).item()) != 0x3F80:
        raise RuntimeError("CPU FP32-to-BF16 conversion must use ties-to-even")
    if int(bf16_min_subnormal.view(torch.uint16).item()) != 0x0001:
        raise RuntimeError("CPU BF16 gradual underflow is required; FTZ is excluded")
    if _fp32_bits(bf16_readback) != 0x0001_0000:
        raise RuntimeError("CPU BF16-to-FP32 widening must preserve subnormal values exactly")
    if _fp32_bits(fp64_to_fp32_tie) != 0x3F80_0000:
        raise RuntimeError("CPU FP64-to-FP32 conversion must use ties-to-even")
    if _fp32_bits(fp32_next_down) != 0x3F7F_FFFF:
        raise RuntimeError("CPU FP32 nextafter toward zero does not match the locked graph")


def locked_scalable_shield_backend_self_check() -> dict[str, object]:
    """Run and report the executable locked-backend probes."""

    _require_locked_rounding_contract()
    return {
        "fp32_add_halfway_bits_hex": "0x3f800000",
        "fp32_min_subnormal_bits_hex": "0x00000001",
        "fp32_subnormal_input_add_bits_hex": "0x00800001",
        "bf16_cast_halfway_bits_hex": "0x3f80",
        "bf16_min_subnormal_bits_hex": "0x0001",
        "bf16_min_subnormal_to_fp32_bits_hex": "0x00010000",
        "fp64_to_fp32_halfway_bits_hex": "0x3f800000",
        "fp32_one_nextafter_zero_bits_hex": "0x3f7fffff",
        "rounding": "IEEE-754 roundTiesToEven",
        "gradual_underflow": True,
        "ftz_daz": False,
    }


def _check_input_tensor(name: str, value: Tensor, expected_shape: tuple[int, ...]) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu":
        raise ValueError(f"{name} must be on the locked CPU backend")
    if value.dtype not in (torch.float32, torch.bfloat16):
        raise TypeError(f"{name} must use torch.float32 or torch.bfloat16")
    if value.layout != torch.strided or not value.is_contiguous():
        raise ValueError(f"{name} must be a contiguous strided tensor")
    if tuple(value.shape) != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}")


def _widen_fp32(value: Tensor) -> Tensor:
    """Apply the contract's first and only input cast."""

    if value.dtype == torch.float32:
        return value
    return value.to(torch.float32)


def _balanced_sum_one_block(values: Tensor) -> Tensor:
    """Reduce one nonempty FP32 vector by fixed adjacent additions."""

    if values.ndim != 1 or values.numel() == 0 or values.dtype != torch.float32:
        raise ValueError("balanced reduction requires a nonempty FP32 vector")
    level = values.contiguous()
    while level.numel() > 1:
        pair_count = level.numel() // 2
        paired = torch.add(level[: 2 * pair_count : 2], level[1 : 2 * pair_count : 2])
        level = torch.cat((paired, level[-1:].clone())) if level.numel() % 2 else paired
    return level[0]


def _balanced_sum_fp32(values: Tensor) -> Tensor:
    """Memory-bounded realization of the locked global adjacent tree.

    The block size is a power of two.  Reducing aligned blocks and then their
    roots has the same value as padding the global adjacent tree with zeros;
    missing zero additions introduce no rounding.  Peak temporary storage is
    therefore bounded by one block instead of the whole flattened matrix.
    """

    if values.ndim != 1 or values.numel() == 0 or values.dtype != torch.float32:
        raise ValueError("balanced reduction requires a nonempty FP32 vector")
    roots: list[Tensor] = []
    for start in range(0, values.numel(), LOCKED_REDUCTION_BLOCK_SIZE):
        stop = min(start + LOCKED_REDUCTION_BLOCK_SIZE, values.numel())
        roots.append(_balanced_sum_one_block(values[start:stop]))
    return _balanced_sum_one_block(torch.stack(roots))


def _scale_free_balanced_norm_fp32(values: Tensor) -> ScalableNormDiagnostics:
    """Evaluate the P20 norm graph and its exact final binary64 product."""

    flat = values.reshape(-1)
    maximum_tensor = torch.amax(torch.abs(flat))
    maximum = float(maximum_tensor)
    depth = (flat.numel() - 1).bit_length()
    if maximum == 0.0:
        return ScalableNormDiagnostics(
            maximum_absolute_value=0.0,
            scaled_square_sum=0.0,
            scaled_norm=0.0,
            returned_norm=0.0,
            final_binary64_product_exact=True,
            reduction_depth=depth,
            reduction_block_size=LOCKED_REDUCTION_BLOCK_SIZE,
        )

    roots: list[Tensor] = []
    for start in range(0, flat.numel(), LOCKED_REDUCTION_BLOCK_SIZE):
        stop = min(start + LOCKED_REDUCTION_BLOCK_SIZE, flat.numel())
        scaled = torch.div(flat[start:stop], maximum_tensor)
        squares = torch.mul(scaled, scaled)
        roots.append(_balanced_sum_one_block(squares))
    scaled_square_sum_tensor = _balanced_sum_one_block(torch.stack(roots))
    scaled_norm_tensor = torch.sqrt(scaled_square_sum_tensor)
    # Both operands widen exactly from FP32.  Their product has at most 48
    # significant bits, so the binary64 product is exact.  Its exponent also
    # remains in range for every configured shape (n<=2^52).
    returned_norm = maximum * float(scaled_norm_tensor)
    returned_finite = math.isfinite(returned_norm)
    return ScalableNormDiagnostics(
        maximum_absolute_value=maximum,
        scaled_square_sum=float(scaled_square_sum_tensor),
        scaled_norm=float(scaled_norm_tensor),
        returned_norm=returned_norm if returned_finite else None,
        final_binary64_product_exact=returned_finite,
        reduction_depth=depth,
        reduction_block_size=LOCKED_REDUCTION_BLOCK_SIZE,
    )


def _directed_clip_alpha_fp32(
    signal_norm: float,
    displacement_norm: float,
) -> tuple[Tensor, float, float]:
    """Build the locked downward radial scale in its exact operation order.

    ``signal_norm`` and ``displacement_norm`` are exact binary64 products of
    FP32 norm parts.  Division and multiplication round normally in binary64;
    one ``nextafter`` toward zero follows each.  The RNE FP64-to-FP32 cast is
    followed unconditionally by one FP32 ``nextafter`` toward zero whenever
    the cast is nonzero.  Hence the returned FP32 scale cannot exceed the
    directed binary64 target.
    """

    if (
        not math.isfinite(signal_norm)
        or not math.isfinite(displacement_norm)
        or signal_norm <= 0.0
        or displacement_norm <= 0.0
    ):
        raise ScalableSectorShieldFailure("clip norms must be positive and finite")
    ratio = signal_norm / displacement_norm
    if not math.isfinite(ratio) or ratio < 0.0:
        raise ScalableSectorShieldFailure("clip norm ratio is not finite and nonnegative")
    ratio_down = math.nextafter(ratio, 0.0) if ratio != 0.0 else 0.0
    scaled = LOCKED_CLIP_COEFFICIENT * ratio_down
    if not math.isfinite(scaled) or scaled < 0.0:
        raise ScalableSectorShieldFailure("clip scale is not finite and nonnegative")
    scaled_down = math.nextafter(scaled, 0.0) if scaled != 0.0 else 0.0
    scaled_down = min(1.0, scaled_down)
    alpha_rne = torch.tensor(scaled_down, dtype=torch.float64, device="cpu").to(torch.float32)
    if bool(alpha_rne != 0.0):
        alpha = torch.nextafter(alpha_rne, torch.zeros((), dtype=torch.float32))
    else:
        alpha = alpha_rne
    if not bool(torch.isfinite(alpha)) or bool(alpha < 0.0) or bool(alpha > 1.0):
        raise ScalableSectorShieldFailure("directed FP32 clip scale left [0,1]")
    return alpha, ratio_down, scaled_down


def _all_entries_halved_exactly_fp32(signal: Tensor) -> bool:
    """Bit-level exactness test for ``fl32(signal/2)==signal/2`` entrywise."""

    bits = signal.reshape(-1).view(torch.int32).to(torch.int64)
    magnitudes = torch.bitwise_and(bits, 0x7FFF_FFFF)
    exponents = torch.bitwise_and(torch.bitwise_right_shift(magnitudes, 23), 0xFF)
    fractions = torch.bitwise_and(magnitudes, 0x007F_FFFF)
    # Exponents >=2 halve by decrementing the exponent.  At exponent 0 or 1,
    # the represented integer significand must be even.  For exponent 1 the
    # hidden leading bit is even, so the stored fraction LSB is decisive too.
    small = exponents <= 1
    small_even = torch.bitwise_and(fractions, 1) == 0
    return bool(torch.logical_or(~small, small_even).all())


def _signal_guard_class(
    signal: Tensor,
    config: ScalableSectorShieldConfig,
) -> tuple[SignalGuardClass, bool, bool]:
    maximum = float(torch.amax(torch.abs(signal)))
    exact_halving = _all_entries_halved_exactly_fp32(signal)
    if maximum == 0.0:
        return SignalGuardClass.ZERO, False, exact_halving
    if maximum >= FP32_MIN_NORMAL:
        return SignalGuardClass.NORMAL_ANCHORED, True, exact_halving
    if exact_halving:
        return SignalGuardClass.SUBNORMAL_EXACT_HALVING, False, True
    raise NearZeroUnrepresentable(
        matrix_shape=config.matrix_shape,
        signal_max_abs=maximum,
        signal_frobenius_strict_upper=config.near_zero_frobenius_upper,
    )


def _fallback_half(signal: Tensor) -> Tensor:
    return torch.mul(_fp32_from_bits(FALLBACK_GAIN_FP32_BITS), signal)


def _shield_widened_fp32(
    signal: Tensor,
    candidate: Tensor,
    config: ScalableSectorShieldConfig,
    *,
    signal_input_dtype: torch.dtype,
    candidate_input_dtype: torch.dtype,
    reduced_spectrum_diagnostic: bool,
) -> ScalableSectorShieldResult:
    """Run the shield after the explicit exact widening boundary."""

    signal_finite = bool(torch.isfinite(signal).all())
    if not signal_finite:
        raise ScalableSectorShieldFailure("nonfinite signal rejected without an update")
    guard_class, normal_anchor, exact_halving = _signal_guard_class(signal, config)
    candidate_finite = bool(torch.isfinite(candidate).all())

    signal_norm: ScalableNormDiagnostics | None = None
    displacement_norm: ScalableNormDiagnostics | None = None
    threshold: float | None = None
    clip_ratio: float | None = None
    clip_scale: float | None = None
    clip_alpha: Tensor | None = None
    accepted = False
    clipped = False
    used_fallback = False
    reason: str | None = None
    fail_closed = False
    action = ShieldAction.PASS_THROUGH

    if guard_class is SignalGuardClass.ZERO:
        accepted = candidate_finite and bool(torch.eq(candidate, 0.0).all())
        output = candidate.clone() if accepted else torch.zeros_like(signal)
        reason = None if accepted else "zero_signal_singleton_fallback"
        fail_closed = not candidate_finite
        action = ShieldAction.PASS_THROUGH if accepted else ShieldAction.ZERO_SINGLETON
    elif not normal_anchor:
        # Candidate screening is intentionally disabled without the normal
        # anchor used by the shape-specific rounding proof.  Exact halving is
        # the only admitted all-subnormal nonzero return.
        output = _fallback_half(signal)
        reason = "subnormal_signal_exact_half_fallback"
        fail_closed = not candidate_finite
        used_fallback = True
        action = ShieldAction.HALF_FALLBACK
    elif not candidate_finite:
        output = _fallback_half(signal)
        reason = "nonfinite_candidate_half_fallback"
        fail_closed = True
        used_fallback = True
        action = ShieldAction.HALF_FALLBACK
    else:
        center = _fp32_from_bits(SHIELD_CENTER_FP32_BITS)
        centered_signal = torch.mul(center, signal)
        displacement = torch.sub(candidate, centered_signal)
        if not bool(torch.isfinite(displacement).all()):
            output = _fallback_half(signal)
            reason = "displacement_overflow_half_fallback"
            fail_closed = True
            used_fallback = True
            action = ShieldAction.HALF_FALLBACK
        else:
            signal_norm = _scale_free_balanced_norm_fp32(signal)
            displacement_norm = _scale_free_balanced_norm_fp32(displacement)
            if signal_norm.returned_norm is None or displacement_norm.returned_norm is None:
                output = _fallback_half(signal)
                reason = "norm_overflow_half_fallback"
                fail_closed = True
                used_fallback = True
                action = ShieldAction.HALF_FALLBACK
            else:
                # k is exactly representable in binary64, and an FP32 norm
                # widens exactly.  One nextafter makes the scalar comparison
                # conservatively inward under the locked host operation.
                threshold = math.nextafter(
                    LOCKED_ACCEPTANCE_RADIUS * signal_norm.returned_norm,
                    -math.inf,
                )
                accepted = displacement_norm.returned_norm <= threshold
                if accepted:
                    output = candidate.clone()
                else:
                    try:
                        clip_alpha, clip_ratio, clip_scale = _directed_clip_alpha_fp32(
                            signal_norm.returned_norm,
                            displacement_norm.returned_norm,
                        )
                        clip_offset = torch.mul(clip_alpha, displacement)
                        clipped_output = torch.add(centered_signal, clip_offset)
                        if not bool(torch.isfinite(clipped_output).all()):
                            raise ScalableSectorShieldFailure(
                                "radial clip produced a nonfinite output"
                            )
                        output = clipped_output
                        clipped = True
                        action = ShieldAction.RADIAL_CLIP
                        reason = "candidate_outside_inward_screen_radial_clip"
                    except ScalableSectorShieldFailure:
                        output = _fallback_half(signal)
                        used_fallback = True
                        fail_closed = True
                        action = ShieldAction.HALF_FALLBACK
                        reason = "radial_clip_failed_half_fallback"

    if not bool(torch.isfinite(output).all()):
        raise ScalableSectorShieldFailure("shield output became nonfinite")
    output = output.contiguous()
    diagnostics = ScalableSectorShieldDiagnostics(
        signal_input_dtype=str(signal_input_dtype),
        candidate_input_dtype=str(candidate_input_dtype),
        widened_dtype=str(torch.float32),
        output_dtype=str(output.dtype),
        signal_guard_class=guard_class,
        normal_anchor=normal_anchor,
        exact_halving_guard=exact_halving,
        candidate_finite=candidate_finite,
        candidate_accepted=accepted,
        candidate_clipped=clipped,
        active=not accepted,
        used_fallback=used_fallback,
        fail_closed=fail_closed,
        action=action,
        reason=reason,
        signal_norm=signal_norm,
        displacement_norm=displacement_norm,
        acceptance_threshold=threshold,
        clip_ratio_fp64_downward=clip_ratio,
        clip_scale_fp64_downward=clip_scale,
        clip_alpha_fp32=float(clip_alpha) if clip_alpha is not None else None,
        clip_alpha_fp32_bits=(_fp32_bits(clip_alpha) if clip_alpha is not None else None),
        original_radius=LOCKED_SHIELD_RADIUS,
        acceptance_radius=LOCKED_ACCEPTANCE_RADIUS,
        clip_coefficient=LOCKED_CLIP_COEFFICIENT,
        reserved_inward_margin=LOCKED_RESERVED_INWARD_MARGIN,
        certified_inward_margin=config.certified_inward_margin,
        certified_inward_radius=config.certified_inward_radius,
        backend=config.backend,
        reduced_spectrum_diagnostic=reduced_spectrum_diagnostic,
    )
    return ScalableSectorShieldResult(output=output, diagnostics=diagnostics)


def shield_sector_candidate_mixed_precision(
    signal: Tensor,
    candidate: Tensor,
    config: ScalableSectorShieldConfig,
) -> ScalableSectorShieldResult:
    """Shield one full configured matrix under the locked P20 graph."""

    if not isinstance(config, ScalableSectorShieldConfig):
        raise TypeError("config must be a ScalableSectorShieldConfig")
    _check_input_tensor("signal", signal, config.matrix_shape)
    _check_input_tensor("candidate", candidate, config.matrix_shape)
    signal_dtype = signal.dtype
    candidate_dtype = candidate.dtype
    signal_fp32 = _widen_fp32(signal).contiguous()
    candidate_fp32 = _widen_fp32(candidate).contiguous()
    return _shield_widened_fp32(
        signal_fp32,
        candidate_fp32,
        config,
        signal_input_dtype=signal_dtype,
        candidate_input_dtype=candidate_dtype,
        reduced_spectrum_diagnostic=False,
    )


def shield_reduced_spectrum_mixed_precision(
    signal_values: Tensor,
    candidate_values: Tensor,
    config: ScalableSectorShieldConfig,
) -> ScalableSectorShieldResult:
    """Run a packed-coordinate diagnostic under one declared shape margin.

    This helper supports large-shape spectrum studies without allocating a
    dense matrix.  Its balanced reduction contains only the supplied packed
    coordinates, so it is *not* bit parity for the declared full flattened
    tree.  The static shape margin is still conservative for this shorter
    tree.  No claim about a matrix backend follows from this diagnostic.
    """

    if not isinstance(config, ScalableSectorShieldConfig):
        raise TypeError("config must be a ScalableSectorShieldConfig")
    if signal_values.ndim != 1 or signal_values.numel() == 0:
        raise ValueError("signal_values must be a nonempty vector")
    if candidate_values.ndim != 1 or candidate_values.numel() == 0:
        raise ValueError("candidate_values must be a nonempty vector")
    if signal_values.shape != candidate_values.shape:
        raise ValueError("packed signal and candidate vectors must have equal shapes")
    if signal_values.numel() > config.entry_count:
        raise ValueError("packed diagnostic may not exceed the configured matrix entry count")
    packed_shape = tuple(signal_values.shape)
    _check_input_tensor("signal_values", signal_values, packed_shape)
    _check_input_tensor("candidate_values", candidate_values, packed_shape)
    signal_dtype = signal_values.dtype
    candidate_dtype = candidate_values.dtype
    return _shield_widened_fp32(
        _widen_fp32(signal_values).contiguous(),
        _widen_fp32(candidate_values).contiguous(),
        config,
        signal_input_dtype=signal_dtype,
        candidate_input_dtype=candidate_dtype,
        reduced_spectrum_diagnostic=True,
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


def scalable_sector_shield_manifest(config: ScalableSectorShieldConfig) -> dict[str, object]:
    """Return the complete shape-specific P20 runtime contract."""

    source_path = Path(__file__).resolve()
    return {
        "schema_version": SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION,
        "claim_scope": (
            "statically certified CPU FP32/BF16-input proof-reference sector shield; "
            "not arbitrary backend or production-throughput parity"
        ),
        "matrix_domain": {
            "shape": list(config.matrix_shape),
            "entry_count": config.entry_count,
            "representative_shapes": [list(shape) for shape in LOCKED_REPRESENTATIVE_SHAPES],
            "diagnostic_shapes": [list(shape) for shape in LOCKED_DIAGNOSTIC_SHAPES],
            "is_frozen_representative_shape": (config.matrix_shape in LOCKED_REPRESENTATIVE_SHAPES),
            "is_diagnostic_shape": config.matrix_shape in LOCKED_DIAGNOSTIC_SHAPES,
            "input_dtypes": ["torch.float32", "torch.bfloat16"],
            "input_cast": "BF16->FP32 exact widening once; FP32 unchanged",
            "output_dtype": "torch.float32",
            "device": "cpu",
            "layout": "contiguous torch.strided",
        },
        "disk": {
            "center": "1143/2048",
            "original_radius": "893/2048",
            "reserved_inward_margin": "1/1024",
            "acceptance_radius": "891/2048",
            "clip_coefficient": "890/2048 = 445/1024",
            "center_fp32_bits_hex": f"0x{SHIELD_CENTER_FP32_BITS:08x}",
            "radius_fp32_bits_hex": f"0x{SHIELD_RADIUS_FP32_BITS:08x}",
            "acceptance_radius_fp32_bits_hex": f"0x{ACCEPTANCE_RADIUS_FP32_BITS:08x}",
            "clip_coefficient_fp32_bits_hex": f"0x{CLIP_COEFFICIENT_FP32_BITS:08x}",
            "shape_certified_inward_radius": config.certified_inward_radius,
            "shape_certified_inward_margin": config.certified_inward_margin,
        },
        "operation_graph": {
            "displacement": "d=fl32(C-fl32(gamma32*S))",
            "norm": (
                "a=maxabs(x); z=fl32(x/a); squares=fl32(z*z); fixed adjacent "
                "balanced FP32 sum; fl32(sqrt(sum)); exact float64(a)*float64(sqrt(sum))"
            ),
            "acceptance": ("Nhat(d)<=nextafter(float64(891/2048)*float64(Nhat(S)),-inf)"),
            "accepted_output": "stored widened FP32 candidate, unchanged",
            "radial_clip": (
                "ratio64=nextafter(Nhat(S)/Nhat(d),0); "
                "scale64=nextafter((890/2048)*ratio64,0); "
                "alpha32=nextafter(RNE32(scale64),0) unless zero; "
                "q=fl32(alpha32*d); U=fl32(fl32(gamma32*S)+q)"
            ),
            "fallback": "fl32((1/2)_32*S), only if radial clipping cannot be computed",
            "reduction_depth": config.balanced_reduction_depth,
            "reduction_block_size": LOCKED_REDUCTION_BLOCK_SIZE,
            "fma_allowed": False,
            "reassociation_allowed": False,
            "fp64_to_fp32_conversion": "round-to-nearest, ties-to-even",
        },
        "underflow_contract": {
            "rounding": "IEEE-754 roundTiesToEven",
            "gradual_underflow_required": True,
            "ftz_daz_allowed": False,
            "normal_anchor": "maxabs(S)>=2^-126",
            "all_subnormal_policy": (
                "return S/2 only when bit-level guard proves exact halving; otherwise fail closed"
            ),
            "all_subnormal_frobenius_strict_upper": config.near_zero_frobenius_upper,
            "zero_signal": "disk singleton; return zero",
        },
        "backend": {
            "name": config.backend,
            "self_check": locked_scalable_shield_backend_self_check(),
            "purpose": "memory-bounded executable proof reference",
            "per_output_fraction_or_big_integer_postcheck": False,
            "runtime_uses_generated_shape_table": True,
        },
        "excluded_claims": [
            "literal arbitrary GPU, BLAS, tensor-core, fused, or compiler-reassociated parity",
            "FTZ/DAZ or stochastic-rounding coverage",
            "throughput competitiveness",
            "full optimizer, aspect scaling, weight decay, or distributed semantics",
            "unmodified upstream Muon stability",
            "global fidelity of an arbitrary candidate",
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
    "ACCEPTANCE_RADIUS_FP32_BITS",
    "CLIP_COEFFICIENT_FP32_BITS",
    "FALLBACK_GAIN_FP32_BITS",
    "FP32_MIN_NORMAL",
    "FP32_MIN_SUBNORMAL",
    "LOCKED_ACCEPTANCE_RADIUS",
    "LOCKED_CERTIFIED_SHAPES",
    "LOCKED_CLIP_COEFFICIENT",
    "LOCKED_DIAGNOSTIC_SHAPES",
    "LOCKED_FALLBACK_GAIN",
    "LOCKED_REDUCTION_BLOCK_SIZE",
    "LOCKED_REPRESENTATIVE_SHAPES",
    "LOCKED_RESERVED_INWARD_MARGIN",
    "LOCKED_SCALABLE_SHIELD_BACKEND",
    "LOCKED_SHAPE_CERTIFIED_RADIUS_HEX",
    "LOCKED_SHAPE_INWARD_MARGIN_HEX",
    "LOCKED_SHIELD_CENTER",
    "LOCKED_SHIELD_RADIUS",
    "SCALABLE_SECTOR_SHIELD_SCHEMA_VERSION",
    "SHIELD_CENTER_FP32_BITS",
    "SHIELD_RADIUS_FP32_BITS",
    "NearZeroUnrepresentable",
    "ScalableNormDiagnostics",
    "ScalableSectorShieldConfig",
    "ScalableSectorShieldDiagnostics",
    "ScalableSectorShieldFailure",
    "ScalableSectorShieldResult",
    "ShieldAction",
    "SignalGuardClass",
    "locked_scalable_shield_backend_self_check",
    "scalable_sector_shield_manifest",
    "shield_reduced_spectrum_mixed_precision",
    "shield_sector_candidate_mixed_precision",
]
