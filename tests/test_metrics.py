from __future__ import annotations

import pytest

from passive_muon.metrics import linear_repair_matmul_count, polynomial_matmul_count
from passive_muon.specs import CANS_5X4, CLASSICAL_CUBIC, JORDAN_QUINTIC, POLAR_EXPRESS_5


def test_static_matmul_accounting() -> None:
    assert polynomial_matmul_count(JORDAN_QUINTIC, steps=5) == 15
    assert polynomial_matmul_count(CLASSICAL_CUBIC, steps=5) == 10
    assert polynomial_matmul_count(POLAR_EXPRESS_5, steps=5) == 15
    assert polynomial_matmul_count(CANS_5X4, steps=4) == 12
    assert linear_repair_matmul_count() == 0


def test_staged_cost_accounting_rejects_unfrozen_stages() -> None:
    with pytest.raises(ValueError, match="has 4 frozen stages; requested 5"):
        polynomial_matmul_count(CANS_5X4, steps=5)
