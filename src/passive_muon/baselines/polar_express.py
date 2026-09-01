"""Pinned Polar Express five-stage baseline.

The coefficient generator and recurrence were audited at
NoahAmsel/PolarExpress commit
71cc37943d99cae780024c1d198977f2f8795407 (MIT).  The upstream notice is
recorded in ``third_party/LICENSES.md``.  Normalization and dtype conversion
are deliberately outside this baseline.
"""

from torch import Tensor

from passive_muon.orthogonalizers import polar_express

__all__ = ["polar_express"]


def apply(matrix: Tensor, *, steps: int = 5) -> Tensor:
    return polar_express(matrix, steps=steps)
