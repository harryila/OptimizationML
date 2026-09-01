"""Clean-room CANS degree-5, four-stage, ``delta=0.3`` baseline.

Only equations and numerical coefficients published in Appendix J of
arXiv:2506.10935v2 are used.  The authors' repository has no software license,
so no source code from it is copied; see ``third_party/LICENSES.md``.
Normalization and dtype conversion are deliberately outside this baseline.
"""

from torch import Tensor

from passive_muon.orthogonalizers import cans_5x4

__all__ = ["cans_5x4"]


def apply(matrix: Tensor, *, steps: int = 4) -> Tensor:
    return cans_5x4(matrix, steps=steps)
