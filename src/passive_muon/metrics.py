"""Static implementation-cost accounting."""

from __future__ import annotations

from passive_muon.specs import QuinticCoefficients, StagedQuinticCoefficients


def _quintic_stage_matmul_count(coefficients: QuinticCoefficients) -> int:
    return 2 if coefficients.fractions()[2] == 0 else 3


def polynomial_matmul_count(
    coefficients: QuinticCoefficients | StagedQuinticCoefficients, *, steps: int
) -> int:
    """Count dense matrix multiplications in the implemented iteration."""

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if isinstance(coefficients, StagedQuinticCoefficients):
        return sum(_quintic_stage_matmul_count(stage) for stage in coefficients.prefix(steps=steps))
    return _quintic_stage_matmul_count(coefficients) * steps


def linear_repair_matmul_count() -> int:
    """The elementwise ``rho*M`` correction adds no matrix multiplication."""

    return 0
