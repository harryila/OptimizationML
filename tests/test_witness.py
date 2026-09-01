from __future__ import annotations

from decimal import Decimal

from passive_muon.witness import canonical_jordan_witness


def test_canonical_witness_is_exact_and_causally_clean() -> None:
    payload = canonical_jordan_witness()
    exact = payload["local_exact_certificate"]
    exact_pair = payload["finite_pair_exact_certificate"]
    pair = payload["finite_pair_high_precision"]

    assert exact["certified_indefinite"] is True
    assert exact["derivatives_positive"] is True
    assert exact["derivatives_unequal"] is True
    assert exact["determinant_sign"] == "negative"
    assert exact_pair["certified_normalized_gap_negative"] is True
    assert exact_pair["certified_fixed_scale_gap_positive"] is True
    assert Decimal(pair["normalized_gap"]) < 0
    assert Decimal(pair["epsilon_regularized_gap"]) < 0
    assert Decimal(pair["fixed_scale_gap"]) > 0


def test_witness_payload_is_deterministic() -> None:
    assert canonical_jordan_witness() == canonical_jordan_witness()
