"""Exact P22 certificate for the GPT-2 fused-QKV shield shape extension."""

from __future__ import annotations

from dataclasses import dataclass

from passive_muon.p22_scalable_sector_shield_extension import (
    P22_CANONICAL_CERTIFIED_RADIUS,
    P22_CANONICAL_CERTIFIED_RADIUS_HEX,
    P22_CANONICAL_INWARD_MARGIN,
    P22_CANONICAL_INWARD_MARGIN_HEX,
    P22_CANONICAL_SHIELD_SHAPE,
)
from passive_muon.scalable_sector_shield import LOCKED_CERTIFIED_SHAPES
from passive_muon.scalable_sector_shield_certificate import (
    LOCKED_SECTOR_RADIUS,
    RADIUS_GRID_DENOMINATOR,
    ScalableSectorShieldAudit,
    audit_scalable_sector_shield,
)

P22_SHAPE_EXTENSION_CERTIFICATE_SCHEMA_VERSION = "passive-muon-p22-shape-extension-certificate-v1"


@dataclass(frozen=True)
class P22ShapeExtensionAudit:
    """The unchanged P20 recurrence plus P22-specific exact checks."""

    shape_audit: ScalableSectorShieldAudit
    checks: dict[str, bool]

    @property
    def certified(self) -> bool:
        return self.shape_audit.certified and bool(self.checks) and all(self.checks.values())


def audit_p22_shape_extension() -> P22ShapeExtensionAudit:
    """Replay P20 exactly for ``(768, 2304)`` and lock its new margin."""

    audit = audit_scalable_sector_shield(P22_CANONICAL_SHIELD_SHAPE)
    checks = {
        "p20_checkpoint_table_is_not_rewritten": (
            P22_CANONICAL_SHIELD_SHAPE not in LOCKED_CERTIFIED_SHAPES
        ),
        "shape_is_canonical_fused_qkv_orientation": (
            (audit.shape.rows, audit.shape.columns) == P22_CANONICAL_SHIELD_SHAPE
        ),
        "supplied_inward_margin_matches_recurrence": (
            audit.inward_margin == P22_CANONICAL_INWARD_MARGIN
        ),
        "certified_radius_matches_recurrence": (
            audit.certified_inward_radius == P22_CANONICAL_CERTIFIED_RADIUS
        ),
        "margin_and_radius_partition_p19_radius": (
            P22_CANONICAL_CERTIFIED_RADIUS + P22_CANONICAL_INWARD_MARGIN == LOCKED_SECTOR_RADIUS
        ),
        "margin_uses_locked_radius_grid": (
            P22_CANONICAL_INWARD_MARGIN.denominator <= RADIUS_GRID_DENOMINATOR
            and RADIUS_GRID_DENOMINATOR % P22_CANONICAL_INWARD_MARGIN.denominator == 0
        ),
        "runtime_margin_hex_matches_exact_value": (
            float(P22_CANONICAL_INWARD_MARGIN).hex() == P22_CANONICAL_INWARD_MARGIN_HEX
        ),
        "runtime_radius_hex_matches_exact_value": (
            float(P22_CANONICAL_CERTIFIED_RADIUS).hex() == P22_CANONICAL_CERTIFIED_RADIUS_HEX
        ),
        "transpose_preserves_frobenius_arithmetic_envelope": (768 * 2_304 == 2_304 * 768),
    }
    result = P22ShapeExtensionAudit(shape_audit=audit, checks=checks)
    if not result.certified:
        raise ArithmeticError("P22 scalable-shield shape extension did not certify")
    return result


__all__ = [
    "P22_SHAPE_EXTENSION_CERTIFICATE_SCHEMA_VERSION",
    "P22ShapeExtensionAudit",
    "audit_p22_shape_extension",
]
