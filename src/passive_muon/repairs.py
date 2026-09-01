"""Constant linear (parallel-resistor) repairs."""

from __future__ import annotations

from collections.abc import Callable

from torch import Tensor

MatrixMap = Callable[[Tensor], Tensor]


def add_linear_repair(operator: MatrixMap, rho: float) -> MatrixMap:
    """Return ``M -> T(M) + rho*M`` for a fixed nonnegative ``rho``.

    The returned map has symmetric Jacobian ``Sym(JT) + rho*I``. If ``rho`` is
    instead recomputed as a function of the input, extra derivative terms
    appear and this certificate no longer applies.
    """

    if rho < 0:
        raise ValueError("rho must be nonnegative")

    def repaired(matrix: Tensor) -> Tensor:
        return operator(matrix) + matrix.new_tensor(rho) * matrix

    return repaired
