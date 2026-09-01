"""Classical cubic Newton--Schulz baseline."""

from torch import Tensor

from passive_muon.orthogonalizers import classical_ns

__all__ = ["classical_ns"]


def apply(matrix: Tensor, *, steps: int = 5) -> Tensor:
    return classical_ns(matrix, steps=steps)
