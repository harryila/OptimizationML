"""P22 runtime extension of the frozen P20 scalable sector shield.

P20 deliberately froze seven Transformer shapes.  The P22 real-gradient
shadow protocol needs the GPT-2 fused-QKV orientation ``(768, 2304)``.  This
module adds that single shape through the same P20 stored-FP32 operation graph
without changing the P20 table, artifact, or checkpoint.  The reversed
``(2304, 768)`` parameter orientation is handled by an exact transpose before
and after shielding.

The arithmetic and backend contract is otherwise exactly P20's.  In
particular this remains a CPU proof-reference graph with gradual underflow,
fixed balanced FP32 reductions, and no FMA contraction or reassociation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction
from typing import Final

from torch import Tensor

from passive_muon.scalable_sector_shield import (
    LOCKED_CERTIFIED_SHAPES,
    LOCKED_SCALABLE_SHIELD_BACKEND,
    ScalableSectorShieldConfig,
    ScalableSectorShieldResult,
    shield_reduced_spectrum_mixed_precision,
    shield_sector_candidate_mixed_precision,
)

P22_SCALABLE_SHIELD_EXTENSION_SCHEMA_VERSION: Final = (
    "passive-muon-p22-scalable-sector-shield-extension-v1"
)
P22_CANONICAL_SHIELD_SHAPE: Final = (768, 2_304)
P22_CANONICAL_PARAMETER_SHAPE: Final = (2_304, 768)

# Exact values obtained by replaying the unchanged P20 recurrence on
# 768*2304 stored coordinates.  Denominators are the locked 2^-60 outward
# radius grid.
P22_CANONICAL_INWARD_MARGIN: Final = Fraction(942_123_070_212_169, 2**60)
P22_CANONICAL_CERTIFIED_RADIUS: Final = Fraction(501_772_185_335_019_447, 2**60)
P22_CANONICAL_INWARD_MARGIN_HEX: Final = "0x1.ac6d8f77a4248p-11"
P22_CANONICAL_CERTIFIED_RADIUS_HEX: Final = "0x1.bda9c938442dfp-2"

P22_CERTIFIED_SHIELD_SHAPES: Final = (
    *LOCKED_CERTIFIED_SHAPES,
    P22_CANONICAL_SHIELD_SHAPE,
)


@dataclass(frozen=True)
class P22ScalableSectorShieldConfig(ScalableSectorShieldConfig):
    """P20 configuration plus the separately certified P22 QKV shape."""

    def __post_init__(self) -> None:
        if self.matrix_shape in LOCKED_CERTIFIED_SHAPES:
            super().__post_init__()
            return
        if len(self.matrix_shape) != 2 or any(
            not isinstance(value, int) or isinstance(value, bool) or value <= 0
            for value in self.matrix_shape
        ):
            raise ValueError("matrix_shape must contain two positive integers")
        if math.prod(self.matrix_shape) > 2**52:
            raise ValueError("matrix_shape may contain at most 2^52 entries")
        if self.matrix_shape != P22_CANONICAL_SHIELD_SHAPE:
            raise ValueError("matrix_shape is not in the frozen P20 table or P22 extension")
        if self.backend != LOCKED_SCALABLE_SHIELD_BACKEND:
            raise ValueError(
                f"the scalable shield requires backend={LOCKED_SCALABLE_SHIELD_BACKEND!r}"
            )
        # Constructing a frozen diagnostic-shape configuration executes P20's
        # public backend/rounding self-check without importing private helpers.
        ScalableSectorShieldConfig((1, 1), backend=self.backend)

    @property
    def certified_inward_margin(self) -> float:
        if self.matrix_shape == P22_CANONICAL_SHIELD_SHAPE:
            return float.fromhex(P22_CANONICAL_INWARD_MARGIN_HEX)
        return super().certified_inward_margin

    @property
    def certified_inward_radius(self) -> float:
        if self.matrix_shape == P22_CANONICAL_SHIELD_SHAPE:
            return float.fromhex(P22_CANONICAL_CERTIFIED_RADIUS_HEX)
        return super().certified_inward_radius


def p22_canonical_shield_orientation(
    matrix_shape: tuple[int, int],
) -> tuple[tuple[int, int], bool]:
    """Return a certified shield orientation and whether to transpose.

    Existing P20 shapes retain their historical orientation.  The P22 QKV
    parameter orientation ``(2304, 768)`` maps isometrically to the newly
    certified ``(768, 2304)`` shield shape.
    """

    if len(matrix_shape) != 2 or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in matrix_shape
    ):
        raise ValueError("matrix_shape must contain two positive integers")
    if matrix_shape in P22_CERTIFIED_SHIELD_SHAPES:
        return matrix_shape, False
    reversed_shape = (matrix_shape[1], matrix_shape[0])
    if reversed_shape in P22_CERTIFIED_SHIELD_SHAPES:
        return reversed_shape, True
    raise ValueError(
        f"matrix shape {matrix_shape} has neither orientation in the P20/P22 certified table"
    )


def p22_shield_sector_candidate_mixed_precision(
    signal: Tensor,
    candidate: Tensor,
    config: P22ScalableSectorShieldConfig,
) -> ScalableSectorShieldResult:
    """Run the unchanged P20 graph under a P22-extended shape config."""

    if not isinstance(config, P22ScalableSectorShieldConfig):
        raise TypeError("config must be a P22ScalableSectorShieldConfig")
    return shield_sector_candidate_mixed_precision(signal, candidate, config)


def p22_shield_reduced_spectrum_mixed_precision(
    signal_values: Tensor,
    candidate_values: Tensor,
    config: P22ScalableSectorShieldConfig,
) -> ScalableSectorShieldResult:
    """Run P20's packed diagnostic under the P22 extension margin."""

    if not isinstance(config, P22ScalableSectorShieldConfig):
        raise TypeError("config must be a P22ScalableSectorShieldConfig")
    return shield_reduced_spectrum_mixed_precision(signal_values, candidate_values, config)


__all__ = [
    "P22_CANONICAL_CERTIFIED_RADIUS",
    "P22_CANONICAL_CERTIFIED_RADIUS_HEX",
    "P22_CANONICAL_INWARD_MARGIN",
    "P22_CANONICAL_INWARD_MARGIN_HEX",
    "P22_CANONICAL_PARAMETER_SHAPE",
    "P22_CANONICAL_SHIELD_SHAPE",
    "P22_CERTIFIED_SHIELD_SHAPES",
    "P22_SCALABLE_SHIELD_EXTENSION_SCHEMA_VERSION",
    "P22ScalableSectorShieldConfig",
    "p22_canonical_shield_orientation",
    "p22_shield_reduced_spectrum_mixed_precision",
    "p22_shield_sector_candidate_mixed_precision",
]
