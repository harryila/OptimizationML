"""Deterministic rank-one momentum probes for the repaired floored map."""

from __future__ import annotations

import math
from dataclasses import dataclass

from passive_muon.specs import JORDAN_QUINTIC


@dataclass(frozen=True)
class MomentumTrajectory:
    """Compact summary of one scalar invariant-mode trajectory."""

    requested_iterations: int
    executed_iterations: int
    initial_position: float
    final_position: float
    final_momentum: float
    maximum_absolute_position: float
    maximum_absolute_momentum: float
    first_hundredfold_amplification_iteration: int | None
    divergence_threshold: float
    exceeded_divergence_threshold: bool
    tail: tuple[tuple[float, float], ...]


def _jordan_scalar(value: float, *, steps: int = 5) -> float:
    a, b, c = (float(coefficient) for coefficient in JORDAN_QUINTIC.fractions())
    output = value
    for _ in range(steps):
        square = output * output
        correction = b * square + c * (square * square)
        output = a * output + correction * output
    return output


def rank_one_repaired_update(*, momentum: float, floor: float, repair_rho: float) -> float:
    """Evaluate the actual floored Jordan repair on an invariant rank-one mode."""

    if floor <= 0:
        raise ValueError("floor must be positive")
    if repair_rho < 0:
        raise ValueError("repair conductance must be nonnegative")
    normalized = momentum / max(floor, abs(momentum))
    return _jordan_scalar(normalized) + repair_rho * momentum


def run_rank_one_momentum_trajectory(
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
) -> MomentumTrajectory:
    """Run the exact requested update order on an invariant rank-one mode.

    This is the curvature-``curvature`` eigenspace of a matrix quadratic.  The
    full matrix operator reduces exactly to the scalar Jordan singular-value
    response on this subspace; no scalar surrogate is substituted.
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
    maximum_position = abs(position)
    maximum_momentum = abs(momentum)
    amplification_iteration: int | None = None
    tail: list[tuple[float, float]] = []
    exceeded = False
    executed = 0

    for iteration in range(1, iterations + 1):
        momentum = beta * momentum + curvature * position
        update = rank_one_repaired_update(
            momentum=momentum,
            floor=floor,
            repair_rho=repair_rho,
        )
        position -= learning_rate * update
        executed = iteration
        maximum_position = max(maximum_position, abs(position))
        maximum_momentum = max(maximum_momentum, abs(momentum))
        if (
            amplification_iteration is None
            and initial_position != 0
            and abs(position) >= 100 * abs(initial_position)
        ):
            amplification_iteration = iteration
        if not math.isfinite(position) or abs(position) > divergence_threshold:
            exceeded = True
            break
        tail.append((position, momentum))
        if len(tail) > tail_length:
            tail.pop(0)

    return MomentumTrajectory(
        requested_iterations=iterations,
        executed_iterations=executed,
        initial_position=float(initial_position),
        final_position=position,
        final_momentum=momentum,
        maximum_absolute_position=maximum_position,
        maximum_absolute_momentum=maximum_momentum,
        first_hundredfold_amplification_iteration=amplification_iteration,
        divergence_threshold=divergence_threshold,
        exceeded_divergence_threshold=exceeded,
        tail=tuple(tail),
    )


def period_residual(trajectory: MomentumTrajectory, *, period: int) -> float:
    """Maximum state mismatch across one period in the retained tail."""

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
