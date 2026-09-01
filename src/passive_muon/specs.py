"""Pure-Python polynomial specifications shared by numerics and certificates."""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class QuinticCoefficients:
    """Coefficients for ``q(s) = a*s + b*s**3 + c*s**5``."""

    name: str
    a: str
    b: str
    c: str
    source: str

    def fractions(self) -> tuple[Fraction, Fraction, Fraction]:
        return Fraction(self.a), Fraction(self.b), Fraction(self.c)


@dataclass(frozen=True)
class StagedQuinticCoefficients:
    """An ordered, finite sequence of odd quintic update rules.

    Unlike a repeated Newton--Schulz rule, adaptive methods such as Polar
    Express and CANS use different coefficients at each stage.  ``steps``
    therefore selects a prefix and may never silently exceed the frozen
    sequence.
    """

    name: str
    stages: tuple[QuinticCoefficients, ...]
    source: str

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("a staged quintic specification must contain at least one stage")

    @property
    def default_steps(self) -> int:
        return len(self.stages)

    def prefix(self, *, steps: int | None = None) -> tuple[QuinticCoefficients, ...]:
        """Return exactly ``steps`` leading stages, or every stage by default."""

        count = self.default_steps if steps is None else steps
        if count < 0:
            raise ValueError("steps must be nonnegative")
        if count > len(self.stages):
            raise ValueError(f"{self.name} has {len(self.stages)} frozen stages; requested {count}")
        return self.stages[:count]


JORDAN_QUINTIC = QuinticCoefficients(
    name="jordan_quintic",
    a="3.4445",
    b="-4.7750",
    c="2.0315",
    source="KellerJordan/Muon@f98f1cacc0263b04290753e32be8d498c1efc806",
)

CLASSICAL_CUBIC = QuinticCoefficients(
    name="classical_cubic",
    a="1.5",
    b="-0.5",
    c="0",
    source="classical Newton--Schulz polar iteration",
)

TAYLOR_QUINTIC = QuinticCoefficients(
    name="taylor_quintic",
    a="1.875",
    b="-1.25",
    c="0.375",
    source="second-order Taylor Newton--Schulz iteration",
)


_POLAR_EXPRESS_SOURCE = (
    "NoahAmsel/PolarExpress@71cc37943d99cae780024c1d198977f2f8795407; "
    "first five stages generated from the pinned repository defaults"
)

POLAR_EXPRESS_5 = StagedQuinticCoefficients(
    name="polar_express_repo_71cc_steps5",
    source=_POLAR_EXPRESS_SOURCE,
    stages=(
        QuinticCoefficients(
            name="polar_express_repo_71cc_stage_1",
            a="8.237312490495558",
            b="-23.15774741455821",
            c="16.68056841144592",
            source=_POLAR_EXPRESS_SOURCE,
        ),
        QuinticCoefficients(
            name="polar_express_repo_71cc_stage_2",
            a="4.082441999064829",
            b="-2.8930477353325825",
            c="0.525284925697564",
            source=_POLAR_EXPRESS_SOURCE,
        ),
        QuinticCoefficients(
            name="polar_express_repo_71cc_stage_3",
            a="3.926347992254658",
            b="-2.854746803476532",
            c="0.5318022422894996",
            source=_POLAR_EXPRESS_SOURCE,
        ),
        QuinticCoefficients(
            name="polar_express_repo_71cc_stage_4",
            a="3.2982187133085143",
            b="-2.424541981026707",
            c="0.48632008358844103",
            source=_POLAR_EXPRESS_SOURCE,
        ),
        QuinticCoefficients(
            name="polar_express_repo_71cc_stage_5",
            a="2.297036943455258",
            b="-1.636625581259032",
            c="0.4002628455953631",
            source=_POLAR_EXPRESS_SOURCE,
        ),
    ),
)


_CANS_SOURCE = (
    "Grishina--Smirnov--Rakhuba, arXiv:2506.10935v2, Appendix J; "
    "CANS degree=5, iterations=4, delta=0.3"
)

CANS_5X4 = StagedQuinticCoefficients(
    name="cans_degree5_iter4_delta0.3",
    source=_CANS_SOURCE,
    stages=(
        QuinticCoefficients(
            name="cans_degree5_iter4_stage_1",
            a="8.420293602126344",
            b="-24.910491192120688",
            c="18.472094206318726",
            source=_CANS_SOURCE,
        ),
        QuinticCoefficients(
            name="cans_degree5_iter4_stage_2",
            a="4.101228661246281",
            b="-3.0518555467946813",
            c="0.5741241025302702",
            source=_CANS_SOURCE,
        ),
        QuinticCoefficients(
            name="cans_degree5_iter4_stage_3",
            a="3.6809819251109155",
            b="-2.75396502307162",
            c="0.5401902781108926",
            source=_CANS_SOURCE,
        ),
        QuinticCoefficients(
            name="cans_degree5_iter4_stage_4",
            a="2.7280916801566666",
            b="-2.0315492757300913",
            c="0.45866431681858805",
            source=_CANS_SOURCE,
        ),
    ),
)


PolynomialCoefficients = QuinticCoefficients | StagedQuinticCoefficients


def coefficient_stages(
    coefficients: PolynomialCoefficients, *, steps: int
) -> tuple[QuinticCoefficients, ...]:
    """Return the exact ordered coefficient rules used by a composition."""

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    if isinstance(coefficients, StagedQuinticCoefficients):
        return coefficients.prefix(steps=steps)
    return (coefficients,) * steps


def zero_slope_gain_exact(coefficients: PolynomialCoefficients, *, steps: int) -> Fraction:
    """Return the derivative at zero of the composed odd polynomial."""

    gain = Fraction(1)
    for stage in coefficient_stages(coefficients, steps=steps):
        gain *= stage.fractions()[0]
    return gain


def scalar_response_and_derivative_exact(
    value: Fraction,
    coefficients: QuinticCoefficients,
    *,
    steps: int,
) -> tuple[Fraction, Fraction]:
    """Evaluate a composed response and derivative in rational arithmetic."""

    if steps < 0:
        raise ValueError("steps must be nonnegative")
    a, b, c = coefficients.fractions()
    out = value
    derivative = Fraction(1)
    for _ in range(steps):
        derivative *= a + 3 * b * out**2 + 5 * c * out**4
        out = a * out + b * out**3 + c * out**5
    return out, derivative


def staged_scalar_response_and_derivative_exact(
    value: Fraction,
    coefficients: StagedQuinticCoefficients,
    *,
    steps: int | None = None,
) -> tuple[Fraction, Fraction]:
    """Evaluate a finite staged response and derivative exactly."""

    out = value
    derivative = Fraction(1)
    for stage in coefficients.prefix(steps=steps):
        a, b, c = stage.fractions()
        derivative *= a + 3 * b * out**2 + 5 * c * out**4
        out = a * out + b * out**3 + c * out**5
    return out, derivative
