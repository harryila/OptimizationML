"""Matched rank-one probes for the pinned Muon EMA/Nesterov ordering."""

from __future__ import annotations

import math
from dataclasses import dataclass

from passive_muon.momentum_experiment import rank_one_repaired_update


@dataclass(frozen=True)
class EmaNesterovTrajectory:
    """Compact summary of one invariant scalar EMA/Nesterov trajectory."""

    requested_iterations: int
    executed_iterations: int
    initial_position: float
    final_position: float
    final_momentum: float
    final_signal: float
    maximum_absolute_position: float
    maximum_absolute_momentum: float
    maximum_absolute_signal: float
    first_hundredfold_amplification_iteration: int | None
    divergence_threshold: float
    exceeded_divergence_threshold: bool
    tail: tuple[tuple[float, float, float], ...]


def run_rank_one_ema_nesterov_trajectory(
    *,
    learning_rate: float,
    beta: float,
    curvature: float,
    floor: float,
    repair_rho: float,
    initial_position: float,
    initial_momentum: float = 0.0,
    iterations: int,
    divergence_threshold: float = 1e100,
    tail_length: int = 8,
) -> EmaNesterovTrajectory:
    """Run the exact EMA then Nesterov signal order on a rank-one mode.

    The recurrence is

    ``g=curvature*position``;
    ``momentum=beta*momentum+(1-beta)*g``;
    ``signal=beta*momentum+(1-beta)*g``;
    ``position-=learning_rate*R(signal)``.

    The full floored matrix operator reduces exactly to the repository's
    scalar Jordan response on this invariant rank-one subspace.
    """

    if learning_rate <= 0 or curvature <= 0 or floor <= 0:
        raise ValueError("learning rate, curvature, and floor must be positive")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0, 1)")
    if repair_rho < 0:
        raise ValueError("repair conductance must be nonnegative")
    if iterations <= 0:
        raise ValueError("iterations must be positive")
    if divergence_threshold <= 0:
        raise ValueError("divergence threshold must be positive")
    if tail_length <= 0:
        raise ValueError("tail length must be positive")

    position = float(initial_position)
    momentum = float(initial_momentum)
    signal = 0.0
    maximum_position = abs(position)
    maximum_momentum = abs(momentum)
    maximum_signal = 0.0
    amplification_iteration: int | None = None
    tail: list[tuple[float, float, float]] = []
    exceeded = False
    executed = 0
    one_minus_beta = 1.0 - beta

    for iteration in range(1, iterations + 1):
        gradient = curvature * position
        momentum = beta * momentum + one_minus_beta * gradient
        signal = beta * momentum + one_minus_beta * gradient
        update = rank_one_repaired_update(
            momentum=signal,
            floor=floor,
            repair_rho=repair_rho,
        )
        position -= learning_rate * update
        executed = iteration
        maximum_position = max(maximum_position, abs(position))
        maximum_momentum = max(maximum_momentum, abs(momentum))
        maximum_signal = max(maximum_signal, abs(signal))
        if (
            amplification_iteration is None
            and initial_position != 0
            and abs(position) >= 100 * abs(initial_position)
        ):
            amplification_iteration = iteration
        if not math.isfinite(position) or abs(position) > divergence_threshold:
            exceeded = True
            break
        tail.append((position, momentum, signal))
        if len(tail) > tail_length:
            tail.pop(0)

    return EmaNesterovTrajectory(
        requested_iterations=iterations,
        executed_iterations=executed,
        initial_position=float(initial_position),
        final_position=position,
        final_momentum=momentum,
        final_signal=signal,
        maximum_absolute_position=maximum_position,
        maximum_absolute_momentum=maximum_momentum,
        maximum_absolute_signal=maximum_signal,
        first_hundredfold_amplification_iteration=amplification_iteration,
        divergence_threshold=divergence_threshold,
        exceeded_divergence_threshold=exceeded,
        tail=tuple(tail),
    )


def ema_nesterov_period_residual(trajectory: EmaNesterovTrajectory, *, period: int) -> float:
    """Return the largest state mismatch across one retained period."""

    if period <= 0:
        raise ValueError("period must be positive")
    if len(trajectory.tail) <= period:
        raise ValueError("trajectory tail is too short for the requested period")
    return max(
        max(abs(left - right) for left, right in zip(current, previous, strict=True))
        for current, previous in zip(
            trajectory.tail[period:], trajectory.tail[:-period], strict=True
        )
    )
