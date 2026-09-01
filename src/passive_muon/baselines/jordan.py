"""Jordan quintic baseline.

Equation and decimal coefficients are reimplemented from KellerJordan/Muon,
commit f98f1cacc0263b04290753e32be8d498c1efc806 (MIT). The upstream copyright
and license text are recorded in ``third_party/LICENSES.md``.
"""

from torch import Tensor

from passive_muon.orthogonalizers import jordan_ns

__all__ = ["jordan_ns"]


def apply(matrix: Tensor, *, steps: int = 5) -> Tensor:
    return jordan_ns(matrix, steps=steps)
