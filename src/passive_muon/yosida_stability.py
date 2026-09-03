"""Exact Yosida regularization and smooth-PL stability certificate.

Let ``A`` be a continuous, everywhere-defined monotone map on a finite
Frobenius/Hilbert space and let ``B=A+mu*I``.  For ``lambda>0`` define

``J=(I+lambda*B)^(-1)`` and ``Y=(I-J)/lambda``.

Continuity and full domain make ``B`` maximal monotone, so Minty's theorem
gives a globally defined, single-valued resolvent.  If ``u=Delta J``,
``v=Delta Y`` and ``x=Delta input``, then ``x=u+lambda*v`` and

``<v,u> >= mu*||u||^2``.

Equivalently, ``Y`` obeys the exact incremental sector ``[m,M]`` with
``m=mu/(1+lambda*mu)`` and ``M=1/lambda``.  In particular, the sometimes
missed input-space inequality

``<Delta Y,Delta x> >= m*||Delta x||^2``

is valid; the graph-space strong-monotonicity inequality is also retained by
:func:`yosida_graph_iqc_matrix`.  The locked parameters give sector
``[500,1000]`` and hence the centered decomposition
``Y=750*I+E``, ``Lip(E)<=250``.  The latter is replayed through the exact
four-by-four smooth-PL LMI used by :mod:`passive_muon.pl_convergence`.

All authoritative arithmetic in this module uses :class:`fractions.Fraction`.
The result is for the exact-real P13 operator and its mathematical resolvent;
it does not assert that an implicit solve has been implemented in BF16.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

from passive_muon.additive_epsilon_deficit import DEPLOYED_EPSILON
from passive_muon.momentum_iqc import RationalMatrix
from passive_muon.pl_convergence import (
    PlConvergenceAudit,
    PlConvergenceCertificate,
    audit_pl_convergence,
    pl_convergence_lmi_matrix,
)
from passive_muon.radial_passivation_tradeoff import (
    ExplicitStepControl,
    corrected_origin_slope,
    ema_nesterov_explicit_step_control,
)

YOSIDA_STABILITY_SCHEMA_VERSION = "passive-muon-yosida-stability-certificate-v1"

LOCKED_RESOLVENT_PARAMETER = Fraction(1, 1_000)
INSUFFICIENT_RESOLVENT_PARAMETER = Fraction(1, 100_000)
LOCKED_BASE_STRONG_MONOTONICITY = Fraction(1_000)
LOCKED_YOSIDA_STRONG_MONOTONICITY = Fraction(500)
LOCKED_YOSIDA_LIPSCHITZ = Fraction(1_000)
LOCKED_YOSIDA_CENTER_GAIN = Fraction(750)
LOCKED_YOSIDA_RESIDUAL_LIPSCHITZ = Fraction(250)

LOCKED_PL_CONSTANT = Fraction(1)
LOCKED_SMOOTHNESS = Fraction(10)
LOCKED_BETA = Fraction(19, 20)
LOCKED_LEARNING_RATE = Fraction(1, 32_000)
LOCKED_TAU = Fraction(499, 500)
LOCKED_RATE_SQUARED = Fraction(249_001, 250_000)


def _require_fraction(value: Fraction, name: str) -> None:
    if not isinstance(value, Fraction):
        raise TypeError(f"{name} must be a fractions.Fraction")


def _require_positive_fraction(value: Fraction, name: str) -> None:
    _require_fraction(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class YosidaSector:
    """Sharp dimension-independent bounds over the admissible operator class."""

    resolvent_parameter: Fraction
    base_strong_monotonicity: Fraction

    def __post_init__(self) -> None:
        _require_positive_fraction(self.resolvent_parameter, "resolvent_parameter")
        _require_positive_fraction(
            self.base_strong_monotonicity,
            "base_strong_monotonicity",
        )

    @property
    def denominator(self) -> Fraction:
        """Return ``1+lambda*mu``."""

        return 1 + self.resolvent_parameter * self.base_strong_monotonicity

    @property
    def resolvent_lipschitz(self) -> Fraction:
        """Sharp universal Lipschitz bound for ``J`` over the class."""

        return 1 / self.denominator

    @property
    def strong_monotonicity(self) -> Fraction:
        """Sharp universal strong-monotonicity bound for ``Y``."""

        return self.base_strong_monotonicity / self.denominator

    @property
    def lipschitz(self) -> Fraction:
        """Sharp universal Lipschitz bound for ``Y`` over the class."""

        return 1 / self.resolvent_parameter

    @property
    def cocoercivity(self) -> Fraction:
        """Sharp universal cocoercivity bound for ``Y`` over the class."""

        return self.resolvent_parameter

    @property
    def center_gain(self) -> Fraction:
        """Midpoint of the exact incremental sector."""

        return (self.strong_monotonicity + self.lipschitz) / 2

    @property
    def residual_lipschitz(self) -> Fraction:
        """Radius of the sector, hence ``Lip(Y-center_gain*I)``."""

        return (self.lipschitz - self.strong_monotonicity) / 2


@dataclass(frozen=True)
class ScalarJuryControl:
    """Exact scalar EMA/Nesterov Jury diagnostic for a Yosida slope."""

    resolvent_parameter: Fraction
    base_slope: Fraction
    yosida_slope: Fraction
    curvature: Fraction
    beta: Fraction
    learning_rate: Fraction
    dimensionless_gain: Fraction
    schur_gain_threshold: Fraction
    p_at_minus_one: Fraction

    @property
    def locally_schur_stable(self) -> bool:
        return self.dimensionless_gain > 0 and self.p_at_minus_one > 0


@dataclass(frozen=True)
class SkewBoundaryControl:
    """Exact two-dimensional sharpness witness on the Yosida sector circle."""

    skew_gain: Fraction
    yosida_real: Fraction
    yosida_imaginary: Fraction
    centered_circle_residual: Fraction
    sector_residual: Fraction


@dataclass(frozen=True)
class YosidaStabilityAudit:
    """Exact sector identities and four-by-four smooth-PL LMI replay."""

    sector: YosidaSector
    pl_certificate: PlConvergenceCertificate
    pl_audit: PlConvergenceAudit
    checks: tuple[tuple[str, bool], ...]

    @property
    def all_checks_pass(self) -> bool:
        return all(value for _name, value in self.checks)

    @property
    def certified(self) -> bool:
        return self.pl_audit.certified and self.all_checks_pass


def locked_yosida_sector() -> YosidaSector:
    """Return the selected ``lambda=1/1000, mu=1000`` sector."""

    return YosidaSector(
        resolvent_parameter=LOCKED_RESOLVENT_PARAMETER,
        base_strong_monotonicity=LOCKED_BASE_STRONG_MONOTONICITY,
    )


def yosida_graph_iqc_matrix(sector: YosidaSector | None = None) -> RationalMatrix:
    """Return the graph IQC in coordinates ``(Delta x, Delta Y)``.

    Its quadratic form is

    ``<y,x-lambda*y> - mu*||x-lambda*y||^2 >= 0``.

    This is exactly the strong-monotonicity inequality for ``B`` pulled back
    through its resolvent, rather than a consequence inferred only from a
    scalar slope interval.
    """

    selected = locked_yosida_sector() if sector is None else sector
    lam = selected.resolvent_parameter
    mu = selected.base_strong_monotonicity
    return (
        (-mu, Fraction(1 + 2 * lam * mu, 2)),
        (Fraction(1 + 2 * lam * mu, 2), -lam * (1 + lam * mu)),
    )


def yosida_sector_iqc_matrix(sector: YosidaSector | None = None) -> RationalMatrix:
    """Return ``<y-m*x,M*x-y> >= 0`` in coordinates ``(x,y)``."""

    selected = locked_yosida_sector() if sector is None else sector
    lower = selected.strong_monotonicity
    upper = selected.lipschitz
    return (
        (-lower * upper, (lower + upper) / 2),
        ((lower + upper) / 2, Fraction(-1)),
    )


def centered_residual_iqc_matrix(sector: YosidaSector | None = None) -> RationalMatrix:
    """Return ``r^2||x||^2-||y-gamma*x||^2 >= 0``.

    Algebraically this is the exact sector IQC, now exposed in the centered
    form consumed by the PL certificate.
    """

    selected = locked_yosida_sector() if sector is None else sector
    center = selected.center_gain
    radius = selected.residual_lipschitz
    return (
        (radius**2 - center**2, center),
        (center, Fraction(-1)),
    )


def yosida_strong_iqc_matrix(sector: YosidaSector | None = None) -> RationalMatrix:
    """Return ``<y,x>-m||x||^2 >= 0`` in coordinates ``(x,y)``."""

    selected = locked_yosida_sector() if sector is None else sector
    return (
        (-selected.strong_monotonicity, Fraction(1, 2)),
        (Fraction(1, 2), Fraction(0)),
    )


def yosida_cocoercive_iqc_matrix(sector: YosidaSector | None = None) -> RationalMatrix:
    """Return ``<y,x>-lambda||y||^2 >= 0`` in coordinates ``(x,y)``."""

    selected = locked_yosida_sector() if sector is None else sector
    return (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), -selected.resolvent_parameter),
    )


def resolvent_strong_firm_iqc_matrix(
    sector: YosidaSector | None = None,
) -> RationalMatrix:
    """Return ``<j,x>-(1+lambda*mu)||j||^2 >= 0`` in ``(x,j)``."""

    selected = locked_yosida_sector() if sector is None else sector
    return (
        (Fraction(0), Fraction(1, 2)),
        (Fraction(1, 2), -selected.denominator),
    )


def locked_yosida_pl_certificate() -> PlConvergenceCertificate:
    """Return the exact full-step smooth nonconvex-PL certificate."""

    denominator = 1_000_000
    return PlConvergenceCertificate(
        pl_constant=LOCKED_PL_CONSTANT,
        smoothness=LOCKED_SMOOTHNESS,
        beta=LOCKED_BETA,
        center_gain=LOCKED_YOSIDA_CENTER_GAIN,
        residual_lipschitz=LOCKED_YOSIDA_RESIDUAL_LIPSCHITZ,
        learning_rate=LOCKED_LEARNING_RATE,
        tau=LOCKED_TAU,
        storage=(
            (Fraction(674_389, denominator), Fraction(-73_827, denominator)),
            (Fraction(-73_827, denominator), Fraction(12_368, denominator)),
        ),
        function_storage=Fraction(313_243, denominator),
        interpolation_reverse_weight=Fraction(622_414, denominator),
        lambda_residual_norm=Fraction(27_530, denominator),
    )


def yosida_pl_lmi_matrix() -> RationalMatrix:
    """Return the selected exact ``4 x 4`` smooth-PL dissipation LMI."""

    return pl_convergence_lmi_matrix(locked_yosida_pl_certificate())


def p13_base_origin_slope() -> Fraction:
    """Return the deployed-epsilon origin slope of P13's direct map ``A``."""

    return corrected_origin_slope(DEPLOYED_EPSILON)


def regularized_base_origin_slope() -> Fraction:
    """Return the origin slope of ``B=A+1000*I``."""

    return p13_base_origin_slope() + LOCKED_BASE_STRONG_MONOTONICITY


def ema_nesterov_jury_control(
    *,
    resolvent_parameter: Fraction,
    base_slope: Fraction | None = None,
    curvature: Fraction = LOCKED_SMOOTHNESS,
    beta: Fraction = LOCKED_BETA,
    learning_rate: Fraction = LOCKED_LEARNING_RATE,
) -> ScalarJuryControl:
    """Return the exact local scalar Jury diagnostic.

    The linearized Yosida slope is ``b/(1+lambda*b)``.  ``lambda=0`` is
    accepted only for the explicit P13 stiffness control; theorem sectors
    themselves require a strictly positive parameter.
    """

    _require_fraction(resolvent_parameter, "resolvent_parameter")
    if resolvent_parameter < 0:
        raise ValueError("resolvent_parameter must be nonnegative")
    selected_slope = regularized_base_origin_slope() if base_slope is None else base_slope
    _require_positive_fraction(selected_slope, "base_slope")
    _require_positive_fraction(curvature, "curvature")
    _require_fraction(beta, "beta")
    if not 0 <= beta < 1:
        raise ValueError("beta must lie in [0,1)")
    _require_positive_fraction(learning_rate, "learning_rate")

    yosida_slope = selected_slope / (1 + resolvent_parameter * selected_slope)
    dimensionless_gain = learning_rate * curvature * yosida_slope
    threshold = 2 * (1 + beta) / ((1 - beta) * (1 + 2 * beta))
    p_at_minus_one = 2 * (1 + beta) - dimensionless_gain * (1 - beta) * (1 + 2 * beta)
    return ScalarJuryControl(
        resolvent_parameter=resolvent_parameter,
        base_slope=selected_slope,
        yosida_slope=yosida_slope,
        curvature=curvature,
        beta=beta,
        learning_rate=learning_rate,
        dimensionless_gain=dimensionless_gain,
        schur_gain_threshold=threshold,
        p_at_minus_one=p_at_minus_one,
    )


def selected_origin_jury_control() -> ScalarJuryControl:
    """Return the passing actual-origin control at ``lambda=1/1000``."""

    return ema_nesterov_jury_control(resolvent_parameter=LOCKED_RESOLVENT_PARAMETER)


def insufficient_regularization_control() -> ScalarJuryControl:
    """Return the failing actual-origin control at ``lambda=1/100000``."""

    return ema_nesterov_jury_control(resolvent_parameter=INSUFFICIENT_RESOLVENT_PARAMETER)


def direct_p13_stiff_control() -> ExplicitStepControl:
    """Return P13's locked direct explicit-step failure at ``lambda=0``."""

    return ema_nesterov_explicit_step_control()


def skew_boundary_control(
    skew_gain: Fraction = LOCKED_BASE_STRONG_MONOTONICITY,
    sector: YosidaSector | None = None,
) -> SkewBoundaryControl:
    """Return a complex-skew slope that lies exactly on the sector boundary.

    The admissible two-dimensional linear map is ``B=mu*I+k*S``, where
    ``S^T=-S`` and ``S^T S=I``.  Its Yosida eigenvalue is represented by the
    exact real and imaginary fields below.
    """

    _require_positive_fraction(skew_gain, "skew_gain")
    selected = locked_yosida_sector() if sector is None else sector
    lam = selected.resolvent_parameter
    mu = selected.base_strong_monotonicity
    denominator = (1 + lam * mu) ** 2 + (lam * skew_gain) ** 2
    real = (mu * (1 + lam * mu) + lam * skew_gain**2) / denominator
    imaginary = skew_gain / denominator
    center = selected.center_gain
    radius = selected.residual_lipschitz
    centered_residual = (real - center) ** 2 + imaginary**2 - radius**2
    sector_residual = (real - selected.strong_monotonicity) * (
        selected.lipschitz - real
    ) - imaginary**2
    return SkewBoundaryControl(
        skew_gain=skew_gain,
        yosida_real=real,
        yosida_imaginary=imaginary,
        centered_circle_residual=centered_residual,
        sector_residual=sector_residual,
    )


def upper_lipschitz_gap(base_slope: Fraction, sector: YosidaSector | None = None) -> Fraction:
    """Return ``M-b/(1+lambda*b)>0`` for an admissible scalar slope.

    Scalar maps ``B=b*I`` in the locked theorem class require ``b>=mu``.
    The returned gap tends to zero as ``b`` grows, proving sharpness of the
    universal upper Lipschitz endpoint.
    """

    _require_positive_fraction(base_slope, "base_slope")
    selected = locked_yosida_sector() if sector is None else sector
    if base_slope < selected.base_strong_monotonicity:
        raise ValueError("base_slope must be at least the base strong-monotonicity constant")
    slope = base_slope / (1 + selected.resolvent_parameter * base_slope)
    return selected.lipschitz - slope


def exact_certificate_checks() -> dict[str, bool]:
    """Replay exact theorem identities, controls, and the PL certificate."""

    sector = locked_yosida_sector()
    certificate = locked_yosida_pl_certificate()
    audit = audit_pl_convergence(certificate)
    selected_control = selected_origin_jury_control()
    insufficient_control = insufficient_regularization_control()
    direct_control = direct_p13_stiff_control()
    skew = skew_boundary_control()
    graph = yosida_graph_iqc_matrix(sector)
    graph_to_sector_scale = 1 / (sector.resolvent_parameter * sector.denominator)
    scaled_graph = tuple(tuple(graph_to_sector_scale * entry for entry in row) for row in graph)
    return {
        "locked_sector_constants": (
            sector.denominator == 2
            and sector.resolvent_lipschitz == Fraction(1, 2)
            and sector.strong_monotonicity == LOCKED_YOSIDA_STRONG_MONOTONICITY
            and sector.lipschitz == LOCKED_YOSIDA_LIPSCHITZ
            and sector.center_gain == LOCKED_YOSIDA_CENTER_GAIN
            and sector.residual_lipschitz == LOCKED_YOSIDA_RESIDUAL_LIPSCHITZ
        ),
        "graph_and_sector_iqcs_are_equivalent": scaled_graph == yosida_sector_iqc_matrix(sector),
        "centered_and_sector_iqcs_are_identical": centered_residual_iqc_matrix(sector)
        == yosida_sector_iqc_matrix(sector),
        "rate_is_tau_squared": certificate.tau**2 == LOCKED_RATE_SQUARED,
        "pl_lmi_is_exactly_certified": audit.certified,
        "selected_actual_origin_is_stable": selected_control.locally_schur_stable,
        "insufficient_regularization_fails": not insufficient_control.locally_schur_stable,
        "direct_p13_map_fails": not direct_control.locally_schur_stable,
        "skew_boundary_is_exact": (
            skew.centered_circle_residual == 0 and skew.sector_residual == 0
        ),
        "lower_endpoint_is_attained": (
            sector.strong_monotonicity
            == LOCKED_BASE_STRONG_MONOTONICITY
            / (1 + LOCKED_RESOLVENT_PARAMETER * LOCKED_BASE_STRONG_MONOTONICITY)
        ),
        "upper_endpoint_is_a_strict_supremum": (
            upper_lipschitz_gap(LOCKED_BASE_STRONG_MONOTONICITY)
            > upper_lipschitz_gap(Fraction(1_000_000))
            > 0
        ),
    }


def audit_yosida_stability() -> YosidaStabilityAudit:
    """Return the authoritative exact P14 audit."""

    sector = locked_yosida_sector()
    certificate = locked_yosida_pl_certificate()
    return YosidaStabilityAudit(
        sector=sector,
        pl_certificate=certificate,
        pl_audit=audit_pl_convergence(certificate),
        checks=tuple(exact_certificate_checks().items()),
    )
