from __future__ import annotations

from fractions import Fraction

import pytest
import torch

from passive_muon.p22_scalable_sector_shield_extension import (
    P22_CANONICAL_CERTIFIED_RADIUS,
    P22_CANONICAL_CERTIFIED_RADIUS_HEX,
    P22_CANONICAL_INWARD_MARGIN,
    P22_CANONICAL_INWARD_MARGIN_HEX,
    P22_CANONICAL_PARAMETER_SHAPE,
    P22_CANONICAL_SHIELD_SHAPE,
    P22ScalableSectorShieldConfig,
    p22_canonical_shield_orientation,
    p22_shield_reduced_spectrum_mixed_precision,
)
from passive_muon.p22_scalable_sector_shield_extension_certificate import (
    audit_p22_shape_extension,
)
from passive_muon.scalable_sector_shield import (
    LOCKED_CERTIFIED_SHAPES,
    LOCKED_SHIELD_RADIUS,
    ScalableSectorShieldConfig,
    ShieldAction,
)


def test_exact_p20_recurrence_gives_the_supplied_p22_margin() -> None:
    result = audit_p22_shape_extension()
    audit = result.shape_audit

    assert result.certified
    assert P22_CANONICAL_SHIELD_SHAPE == (768, 2_304)
    assert (
        audit.inward_margin == P22_CANONICAL_INWARD_MARGIN == Fraction(942_123_070_212_169, 2**60)
    )
    assert (
        audit.certified_inward_radius
        == P22_CANONICAL_CERTIFIED_RADIUS
        == Fraction(501_772_185_335_019_447, 2**60)
    )
    assert audit.inward_margin + audit.certified_inward_radius == Fraction(893, 2_048)
    assert audit.norm.root_entries_upper == 1_331
    assert audit.norm.pairwise_path_length == 22
    assert audit.norm.reduction_block_count == 2
    assert all(result.checks.values())


def test_runtime_hex_values_match_exact_certificate() -> None:
    config = P22ScalableSectorShieldConfig(P22_CANONICAL_SHIELD_SHAPE)

    assert config.certified_inward_margin.hex() == P22_CANONICAL_INWARD_MARGIN_HEX
    assert config.certified_inward_radius.hex() == P22_CANONICAL_CERTIFIED_RADIUS_HEX
    assert config.certified_inward_margin == float(P22_CANONICAL_INWARD_MARGIN)
    assert config.certified_inward_radius == float(P22_CANONICAL_CERTIFIED_RADIUS)
    assert config.certified_inward_margin > 0.0
    assert config.certified_inward_radius < LOCKED_SHIELD_RADIUS


def test_p22_is_an_extension_not_a_rewrite_of_p20_table() -> None:
    assert P22_CANONICAL_SHIELD_SHAPE not in LOCKED_CERTIFIED_SHAPES
    with pytest.raises(ValueError, match="P20 certified shape table"):
        ScalableSectorShieldConfig(P22_CANONICAL_SHIELD_SHAPE)

    # Historical P20 shapes remain available through the P22 config.
    inherited = P22ScalableSectorShieldConfig((768, 768))
    assert inherited.certified_inward_margin > 0.0


def test_p22_orientation_supports_qkv_weight_and_shield_views() -> None:
    assert P22_CANONICAL_PARAMETER_SHAPE == (2_304, 768)
    assert p22_canonical_shield_orientation(P22_CANONICAL_SHIELD_SHAPE) == (
        P22_CANONICAL_SHIELD_SHAPE,
        False,
    )
    assert p22_canonical_shield_orientation(P22_CANONICAL_PARAMETER_SHAPE) == (
        P22_CANONICAL_SHIELD_SHAPE,
        True,
    )
    with pytest.raises(ValueError, match="neither orientation"):
        p22_canonical_shield_orientation((769, 2_304))


def test_p22_runtime_path_executes_unchanged_p20_graph() -> None:
    config = P22ScalableSectorShieldConfig(P22_CANONICAL_SHIELD_SHAPE)
    signal = torch.tensor([3.0, 4.0, -2.0, 1.0], dtype=torch.float32)
    candidate = torch.mul(signal, torch.tensor(0.5, dtype=torch.float32))

    result = p22_shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.action is ShieldAction.PASS_THROUGH
    assert result.diagnostics.candidate_accepted
    assert not result.diagnostics.active
    assert result.diagnostics.reduced_spectrum_diagnostic
    assert result.diagnostics.certified_inward_margin == config.certified_inward_margin
    assert result.diagnostics.certified_inward_radius == config.certified_inward_radius
    assert torch.equal(result.output, candidate)


def test_p22_runtime_path_shields_a_corrupted_candidate() -> None:
    config = P22ScalableSectorShieldConfig(P22_CANONICAL_SHIELD_SHAPE)
    signal = torch.tensor([3.0, 4.0], dtype=torch.float32)
    candidate = torch.tensor([100.0, -200.0], dtype=torch.float32)

    result = p22_shield_reduced_spectrum_mixed_precision(signal, candidate, config)

    assert result.diagnostics.active
    assert result.diagnostics.candidate_clipped
    assert result.diagnostics.action is ShieldAction.RADIAL_CLIP
    assert bool(torch.isfinite(result.output).all())
