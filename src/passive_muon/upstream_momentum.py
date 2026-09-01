"""Real-arithmetic shadow of the pinned Muon EMA/Nesterov signal update.

Only the momentum state and signal ordering are represented here.  The
orthogonalizer, aspect-ratio scaling, weight decay, and parameter update from
the upstream optimizer are intentionally outside this small parity helper.
"""

from __future__ import annotations

from torch import Tensor

PINNED_KELLER_JORDAN_MUON_REVISION = "f98f1cacc0263b04290753e32be8d498c1efc806"
PINNED_MUON_PY_SHA256 = "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
PINNED_DEFAULT_BETA = 0.95


def ema_nesterov_state_and_signal(
    gradient: Tensor,
    momentum: Tensor,
    *,
    beta: float = PINNED_DEFAULT_BETA,
) -> tuple[Tensor, Tensor]:
    """Return the pinned EMA state and Nesterov signal without mutating inputs.

    The returned tensors implement

    ``m_next = beta * momentum + (1 - beta) * gradient`` and
    ``signal = beta * m_next + (1 - beta) * gradient``.

    This is the algebraic real-arithmetic model used by the stability
    experiment.  Use :func:`pinned_ema_nesterov_lerp_` when literal PyTorch
    operation ordering and upstream input mutation are part of the check.
    """

    weight = 1.0 - beta
    momentum_next = beta * momentum + weight * gradient
    signal = beta * momentum_next + weight * gradient
    return momentum_next, signal


def pinned_ema_nesterov_lerp_(
    gradient: Tensor,
    momentum: Tensor,
    *,
    beta: float = PINNED_DEFAULT_BETA,
) -> Tensor:
    """Execute the pinned upstream two-``lerp_`` sequence literally.

    Both inputs are mutated, matching ``muon_update`` at the pinned revision:
    ``momentum`` becomes the EMA state and ``gradient`` becomes the Nesterov
    signal returned by this function.
    """

    momentum.lerp_(gradient, 1.0 - beta)
    return gradient.lerp_(momentum, beta)
