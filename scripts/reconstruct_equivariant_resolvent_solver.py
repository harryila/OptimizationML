#!/usr/bin/env python3
"""Independently reconstruct the exact P16 reduction and shaping witness.

Only the Python standard library is imported.  The exact modal noncollapse
claim is rebuilt with :class:`fractions.Fraction` arithmetic before an
optional canonical artifact is read.  An outward-directed, high-precision
Decimal interval calculation independently certifies the canonical graph
residual, exact-output enclosure, and locked fidelity decisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import dataclass
from decimal import ROUND_CEILING, ROUND_FLOOR, Decimal, localcontext
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/equivariant_resolvent_solver_certificate.json"
SCHEMA_VERSION = "passive-muon-equivariant-resolvent-solver-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p16-independent-reconstruction-v1"

EPSILON = Fraction(1, 10_000_000)
LAMBDA = Fraction(1, 1_000)
MU = Fraction(1_000)
KAPPA = Fraction(1, 250)
COEFFICIENTS = (Fraction(6_889, 2_000), Fraction(-191, 40), Fraction(4_063, 2_000))
STEPS = 5
U_MODES = (Fraction(3, 5), Fraction(4, 5))
CANONICAL_INPUT = (Fraction(3), Fraction(4))
CANONICAL_ROOT_HEX = ("0x1.705228c08d605p-1", "0x1.eb135495cac74p-1")

UNIT_UPPER = Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)
FINAL_BAND_LOWER = Fraction(-199_437, 1_250)
FINAL_BAND_PROJECTION = Fraction(-41_528_474_059_081, 260_261_360_000)
Z0 = Fraction(63, 9_937)

ABSOLUTE_GATE = Fraction(1, 1_000)
RETENTION_GATE = Fraction(1, 10)
FORWARD_DEPARTURE_GUARD = (
    Fraction(5_704_8, 10_000_000_000),
    Fraction(5_704_9, 10_000_000_000),
)
FORWARD_UPSTREAM_GUARD = (
    Fraction(69_988_4, 10_000_000),
    Fraction(69_988_5, 10_000_000),
)
FORWARD_RETENTION_GUARD = (
    Fraction(8_151, 100_000_000),
    Fraction(8_152, 100_000_000),
)
DEPARTURE_GUARD = (Fraction(58_791, 10_000_000_000), Fraction(58_793, 10_000_000_000))
UPSTREAM_GUARD = (Fraction(699_674, 10_000_000), Fraction(699_676, 10_000_000))
RETENTION_GUARD = (Fraction(84_027, 1_000_000_000), Fraction(84_029, 1_000_000_000))

# The canonical enclosure below is rebuilt without Arb or any project module.
# All algebraic operations use directed Decimal rounding.  Decimal's ``sqrt``
# and ``ln`` are correctly rounded; one additional representable number is
# included on each side so their enclosures do not depend on a tie case.
DIRECTED_DECIMAL_PRECISION = 120

P15_ARTIFACT_PATH = "results/summaries/inexact_yosida_robustness_certificate.json"
P15_ARTIFACT_SHA256 = "6b0c1da2c30172cd37988f0488e400fe6ecd2872300926191619c27fea647255"
P15_SOURCE_COMMIT = "287079ba0a22b4b79aef3a5f9d0ede27dfe4269a"
P15_ARTIFACT_COMMIT = "8ae64e8e11fdd0b16bd19523a2baf78400dd743c"
P15_CHECKPOINT_TAG = "p15-inexact-yosida-robustness-checkpoint"
P15_CHECKPOINT_TAG_OBJECT = "24263cb3a20cd6008895c6da66ad25d5a673b0fb"

EXPECTED_SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p16-equivariant-resolvent-solver.yml",
    "experiments/resolvent/run_p16_solver_study.py",
    "scripts/certify_equivariant_resolvent_solver.py",
    "scripts/reconstruct_equivariant_resolvent_solver.py",
    "src/passive_muon/equivariant_resolvent_certificate.py",
    "src/passive_muon/equivariant_resolvent_solver.py",
    "src/passive_muon/inexact_yosida_robustness.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "src/passive_muon/yosida_stability.py",
    "tests/test_equivariant_resolvent_certificate.py",
    "tests/test_equivariant_resolvent_solver.py",
    "tests/test_p16_solver_study.py",
    "tests/test_equivariant_resolvent_solver_cli.py",
    "tests/test_equivariant_resolvent_solver_reconstruction.py",
    "tests/test_equivariant_resolvent_solver_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/equivariant_resolvent_solver.md",
    "theory/audits/P16_EQUIVARIANT_RESOLVENT_SOLVER_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P16_EQUIVARIANT_RESOLVENT_SOLVER_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _h_fraction(value: Fraction) -> Fraction:
    a, b, c = COEFFICIENTS
    result = value
    for _ in range(STEPS):
        result = a * result + b * result**3 + c * result**5
    return result


def _p_terms(z: Fraction) -> tuple[Fraction, Fraction, Fraction]:
    if z <= Z0:
        return UNIT_UPPER * z, Fraction(0), Fraction(1)
    exact_part = UNIT_UPPER * Z0 + (FINAL_BAND_LOWER - FINAL_BAND_PROJECTION) * (
        1 / (1 + z) - 1 / (1 + Z0)
    )
    return exact_part, -FINAL_BAND_PROJECTION, (1 + z) / (1 + Z0)


def _decimal(value: Fraction) -> Decimal:
    return Decimal(value.numerator) / Decimal(value.denominator)


def _h_decimal(value: Decimal) -> Decimal:
    a, b, c = (_decimal(item) for item in COEFFICIENTS)
    result = value
    for _ in range(STEPS):
        result = a * result + b * result**3 + c * result**5
    return result


def _rounded_binary(
    left: Decimal,
    right: Decimal,
    operation: str,
    *,
    upward: bool,
) -> Decimal:
    rounding = ROUND_CEILING if upward else ROUND_FLOOR
    with localcontext() as context:
        context.prec = DIRECTED_DECIMAL_PRECISION
        context.rounding = rounding
        if operation == "add":
            return left + right
        if operation == "subtract":
            return left - right
        if operation == "multiply":
            return left * right
        if operation == "divide":
            return left / right
    raise ValueError(f"unknown directed operation: {operation}")


def _rounded_fraction(value: Fraction, *, upward: bool) -> Decimal:
    rounding = ROUND_CEILING if upward else ROUND_FLOOR
    with localcontext() as context:
        context.prec = DIRECTED_DECIMAL_PRECISION
        context.rounding = rounding
        return Decimal(value.numerator) / Decimal(value.denominator)


@dataclass(frozen=True)
class _DirectedInterval:
    """Closed Decimal interval with outward-rounded elementary operations."""

    lower: Decimal
    upper: Decimal

    def __post_init__(self) -> None:
        if self.lower > self.upper:
            raise ValueError("directed interval endpoints are reversed")

    @classmethod
    def fraction(cls, value: Fraction) -> _DirectedInterval:
        return cls(
            _rounded_fraction(value, upward=False),
            _rounded_fraction(value, upward=True),
        )

    def __neg__(self) -> _DirectedInterval:
        return _DirectedInterval(-self.upper, -self.lower)

    def __add__(self, other: _DirectedInterval) -> _DirectedInterval:
        return _DirectedInterval(
            _rounded_binary(self.lower, other.lower, "add", upward=False),
            _rounded_binary(self.upper, other.upper, "add", upward=True),
        )

    def __sub__(self, other: _DirectedInterval) -> _DirectedInterval:
        return _DirectedInterval(
            _rounded_binary(self.lower, other.upper, "subtract", upward=False),
            _rounded_binary(self.upper, other.lower, "subtract", upward=True),
        )

    def __mul__(self, other: _DirectedInterval) -> _DirectedInterval:
        lower_candidates = (
            _rounded_binary(left, right, "multiply", upward=False)
            for left in (self.lower, self.upper)
            for right in (other.lower, other.upper)
        )
        upper_candidates = (
            _rounded_binary(left, right, "multiply", upward=True)
            for left in (self.lower, self.upper)
            for right in (other.lower, other.upper)
        )
        return _DirectedInterval(min(lower_candidates), max(upper_candidates))

    def reciprocal(self) -> _DirectedInterval:
        if self.lower <= 0 <= self.upper:
            raise ZeroDivisionError("directed interval contains zero")
        one = Decimal(1)
        return _DirectedInterval(
            _rounded_binary(one, self.upper, "divide", upward=False),
            _rounded_binary(one, self.lower, "divide", upward=True),
        )

    def __truediv__(self, other: _DirectedInterval) -> _DirectedInterval:
        return self * other.reciprocal()

    def square(self) -> _DirectedInterval:
        zero = Decimal(0)
        if self.lower >= zero:
            lower = _rounded_binary(self.lower, self.lower, "multiply", upward=False)
            upper = _rounded_binary(self.upper, self.upper, "multiply", upward=True)
        elif self.upper <= zero:
            lower = _rounded_binary(self.upper, self.upper, "multiply", upward=False)
            upper = _rounded_binary(self.lower, self.lower, "multiply", upward=True)
        else:
            lower = zero
            upper = max(
                _rounded_binary(self.lower, self.lower, "multiply", upward=True),
                _rounded_binary(self.upper, self.upper, "multiply", upward=True),
            )
        return _DirectedInterval(lower, upper)

    def sqrt(self) -> _DirectedInterval:
        if self.lower < 0:
            raise ValueError("cannot take the square root of a negative interval")
        with localcontext() as context:
            context.prec = DIRECTED_DECIMAL_PRECISION
            lower_nearest = self.lower.sqrt(context=context)
            upper_nearest = self.upper.sqrt(context=context)
            lower = max(Decimal(0), context.next_minus(lower_nearest))
            upper = context.next_plus(upper_nearest)
        return _DirectedInterval(lower, upper)

    def ln(self) -> _DirectedInterval:
        if self.lower <= 0:
            raise ValueError("logarithm interval must be strictly positive")
        with localcontext() as context:
            context.prec = DIRECTED_DECIMAL_PRECISION
            lower_nearest = self.lower.ln(context=context)
            upper_nearest = self.upper.ln(context=context)
            lower = context.next_minus(lower_nearest)
            upper = context.next_plus(upper_nearest)
        return _DirectedInterval(lower, upper)

    def absolute(self) -> _DirectedInterval:
        zero = Decimal(0)
        if self.lower >= zero:
            return self
        if self.upper <= zero:
            return -self
        return _DirectedInterval(zero, max(-self.lower, self.upper))

    def symmetric_inflation(self) -> _DirectedInterval:
        radius = max(abs(self.lower), abs(self.upper))
        return _DirectedInterval(-radius, radius)

    def record(self) -> dict[str, str]:
        return {"lower": str(self.lower), "upper": str(self.upper)}


def _interval_h(value: _DirectedInterval) -> _DirectedInterval:
    coefficient_a, coefficient_b, coefficient_c = (
        _DirectedInterval.fraction(item) for item in COEFFICIENTS
    )
    result = value
    for _ in range(STEPS):
        square = result.square()
        result = result * (coefficient_a + coefficient_b * square + coefficient_c * square.square())
    return result


def _interval_norm(values: tuple[_DirectedInterval, ...]) -> _DirectedInterval:
    total = _DirectedInterval.fraction(Fraction(0))
    for value in values:
        total = total + value.square()
    return total.sqrt()


def _interval_chi(
    output: tuple[_DirectedInterval, _DirectedInterval],
    source: tuple[_DirectedInterval, _DirectedInterval],
) -> _DirectedInterval:
    determinant = (output[0] * source[1] - output[1] * source[0]).absolute()
    return determinant / (_interval_norm(source) * _interval_norm(output))


def _canonical_directed_interval_reconstruction() -> dict[str, object]:
    """Independently enclose the canonical P16 output and fidelity metrics."""

    zero = _DirectedInterval.fraction(Fraction(0))
    one = _DirectedInterval.fraction(Fraction(1))
    epsilon = _DirectedInterval.fraction(EPSILON)
    lam = _DirectedInterval.fraction(LAMBDA)
    shunt = _DirectedInterval.fraction(MU)
    unit_upper = _DirectedInterval.fraction(UNIT_UPPER)
    z0 = _DirectedInterval.fraction(Z0)
    derivative = _DirectedInterval.fraction(FINAL_BAND_LOWER)
    projection = _DirectedInterval.fraction(FINAL_BAND_PROJECTION)
    root_exact = tuple(Fraction.from_float(float.fromhex(value)) for value in CANONICAL_ROOT_HEX)
    root = tuple(_DirectedInterval.fraction(value) for value in root_exact)
    source = tuple(_DirectedInterval.fraction(value) for value in CANONICAL_INPUT)

    radius = _interval_norm(root)
    raw_ratio = radius / epsilon
    normalized = tuple(value / (radius + epsilon) for value in root)
    jordan = tuple(_interval_h(value) for value in normalized)
    logarithm_argument = (one + raw_ratio) / (one + z0)
    primitive = (
        unit_upper * z0
        - projection * logarithm_argument.ln()
        + (derivative - projection) * ((one / (one + raw_ratio)) - (one / (one + z0)))
    )
    graph_output = tuple(
        response + (primitive / radius + shunt) * value
        for response, value in zip(jordan, root, strict=True)
    )
    graph_residual = tuple(
        sigma - value - lam * output
        for sigma, value, output in zip(source, root, graph_output, strict=True)
    )
    graph_residual_norm = _interval_norm(graph_residual)
    output_error_norm = _DirectedInterval.fraction(Fraction(500)) * graph_residual_norm
    approximate_output = tuple(
        (sigma - value) / lam for sigma, value in zip(source, root, strict=True)
    )
    error_box = output_error_norm.symmetric_inflation()
    exact_output = tuple(value + error_box for value in approximate_output)
    modal_gains = tuple(value / sigma for value, sigma in zip(exact_output, source, strict=True))
    modal_gain_gap = modal_gains[1] - modal_gains[0]
    departure = _interval_chi(exact_output, source)

    source_norm = _interval_norm(source)
    upstream_normalized = tuple(value / (source_norm + epsilon) for value in source)
    upstream_output = tuple(_interval_h(value) for value in upstream_normalized)
    upstream_departure = _interval_chi(upstream_output, source)
    retention = departure / upstream_departure

    def strictly_inside(value: _DirectedInterval, bounds: tuple[Fraction, Fraction]) -> bool:
        lower = _rounded_fraction(bounds[0], upward=True)
        upper = _rounded_fraction(bounds[1], upward=False)
        return value.lower > lower and value.upper < upper

    absolute_gate_upper = _rounded_fraction(ABSOLUTE_GATE, upward=False)
    retention_gate_upper = _rounded_fraction(RETENTION_GATE, upward=False)
    checks = {
        "graph_residual_norm_is_strictly_positive": graph_residual_norm.lower > zero.upper,
        "p15_output_error_enclosure_is_small": (
            output_error_norm.upper < _rounded_fraction(Fraction(1, 10**10), upward=False)
        ),
        "modal_gain_intervals_are_disjoint": modal_gain_gap.lower > 0,
        "departure_inside_locked_guard": strictly_inside(departure, DEPARTURE_GUARD),
        "upstream_inside_locked_guard": strictly_inside(upstream_departure, UPSTREAM_GUARD),
        "retention_inside_locked_guard": strictly_inside(retention, RETENTION_GUARD),
        "absolute_gate_rigorously_fails": departure.upper < absolute_gate_upper,
        "retention_gate_rigorously_fails": retention.upper < retention_gate_upper,
    }
    return {
        "arithmetic": (
            "Python standard-library Decimal intervals with directed algebraic rounding; "
            "correctly-rounded sqrt/ln padded outward by one representable number"
        ),
        "precision_decimal_digits": DIRECTED_DECIMAL_PRECISION,
        "approximate_root_exact_binary": [str(value) for value in root_exact],
        "radius": radius.record(),
        "graph_output": [value.record() for value in graph_output],
        "graph_residual": [value.record() for value in graph_residual],
        "graph_residual_norm": graph_residual_norm.record(),
        "exact_output_error_norm_upper": output_error_norm.record(),
        "exact_output_singular_values": [value.record() for value in exact_output],
        "modal_gains": [value.record() for value in modal_gains],
        "modal_gain_gap": modal_gain_gap.record(),
        "best_scalar_departure": departure.record(),
        "upstream_best_scalar_departure": upstream_departure.record(),
        "upstream_shaping_retention": retention.record(),
        "checks": checks,
        "all_checks_pass": all(checks.values()),
    }


def _chi(output: tuple[Decimal, Decimal], source: tuple[Decimal, Decimal]) -> Decimal:
    determinant = abs(output[0] * source[1] - output[1] * source[0])
    source_norm = (source[0] ** 2 + source[1] ** 2).sqrt()
    output_norm = (output[0] ** 2 + output[1] ** 2).sqrt()
    return determinant / (source_norm * output_norm)


def _forward_decimal_cross_check(
    primitive: tuple[Fraction, Fraction, Fraction],
    jordan: tuple[Fraction, Fraction],
) -> dict[str, str | bool]:
    with localcontext() as context:
        context.prec = 100
        rational, coefficient, argument = primitive
        p_value = _decimal(rational) + _decimal(coefficient) * _decimal(argument).ln()
        outputs = tuple(
            _decimal(response) + (_decimal(MU) + p_value) * _decimal(mode)
            for mode, response in zip(U_MODES, jordan, strict=True)
        )
        inputs = tuple(
            _decimal(mode) + _decimal(LAMBDA) * output
            for mode, output in zip(U_MODES, outputs, strict=True)
        )
        gains = tuple(output / source for output, source in zip(outputs, inputs, strict=True))
        departure = _chi(outputs, inputs)
        input_norm = (inputs[0] ** 2 + inputs[1] ** 2).sqrt()
        normalized = tuple(value / (input_norm + _decimal(EPSILON)) for value in inputs)
        upstream = tuple(_h_decimal(value) for value in normalized)
        upstream_departure = _chi(upstream, inputs)
        retention = departure / upstream_departure

        def inside(value: Decimal, bounds: tuple[Fraction, Fraction]) -> bool:
            return _decimal(bounds[0]) < value < _decimal(bounds[1])

        return {
            "modal_gain_1": str(gains[0]),
            "modal_gain_2": str(gains[1]),
            "modal_gain_gap": str(gains[1] - gains[0]),
            "best_scalar_departure": str(departure),
            "upstream_best_scalar_departure": str(upstream_departure),
            "upstream_shaping_retention": str(retention),
            "departure_inside_locked_guard": inside(departure, FORWARD_DEPARTURE_GUARD),
            "upstream_inside_locked_guard": inside(upstream_departure, FORWARD_UPSTREAM_GUARD),
            "retention_inside_locked_guard": inside(retention, FORWARD_RETENTION_GUARD),
            "absolute_gate_fails": departure < _decimal(ABSOLUTE_GATE),
            "retention_gate_fails": retention < _decimal(RETENTION_GATE),
        }


def _canonical_decimal_cross_check() -> dict[str, str | bool]:
    with localcontext() as context:
        context.prec = 100
        root = tuple(
            _decimal(Fraction.from_float(float.fromhex(value))) for value in CANONICAL_ROOT_HEX
        )
        source = tuple(_decimal(value) for value in CANONICAL_INPUT)
        output = tuple(
            (sigma - value) / _decimal(LAMBDA) for sigma, value in zip(source, root, strict=True)
        )
        gains = tuple(value / sigma for value, sigma in zip(output, source, strict=True))
        departure = _chi(output, source)
        upstream_normalized = tuple(
            value / (_decimal(Fraction(5)) + _decimal(EPSILON)) for value in source
        )
        upstream = tuple(_h_decimal(value) for value in upstream_normalized)
        upstream_departure = _chi(upstream, source)
        retention = departure / upstream_departure

        def inside(value: Decimal, bounds: tuple[Fraction, Fraction]) -> bool:
            return _decimal(bounds[0]) < value < _decimal(bounds[1])

        return {
            "modal_gain_1": str(gains[0]),
            "modal_gain_2": str(gains[1]),
            "modal_gain_gap": str(gains[1] - gains[0]),
            "best_scalar_departure": str(departure),
            "upstream_best_scalar_departure": str(upstream_departure),
            "upstream_shaping_retention": str(retention),
            "departure_inside_locked_guard": inside(departure, DEPARTURE_GUARD),
            "upstream_inside_locked_guard": inside(upstream_departure, UPSTREAM_GUARD),
            "retention_inside_locked_guard": inside(retention, RETENTION_GUARD),
            "absolute_gate_fails": departure < _decimal(ABSOLUTE_GATE),
            "retention_gate_fails": retention < _decimal(RETENTION_GATE),
        }


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=ROOT, check=False, capture_output=True
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read frozen source {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _source_snapshot_for(canonical: dict[str, object]) -> dict[str, str]:
    expected = canonical["proof_replay_provenance"]["source_snapshot"]
    if set(expected) != set(EXPECTED_SOURCE_PATHS):
        raise RuntimeError("source snapshot path set does not match the P16 lock")
    git = canonical["git"]
    if git["dirty"] is False:
        return {path: _git_blob_sha256(git["sha"], path) for path in expected}
    return {path: _sha256(path) for path in expected}


def _reconstruct() -> dict[str, object]:
    normalized = tuple(value / (1 + EPSILON) for value in U_MODES)
    jordan = tuple(_h_fraction(value) for value in normalized)
    modal_difference = jordan[1] / U_MODES[1] - jordan[0] / U_MODES[0]
    primitive = _p_terms(1 / EPSILON)
    margins = (
        UNIT_UPPER - Fraction(504, 5),
        UNIT_UPPER - Fraction(77_877, 500) * Fraction(199, 200),
        UNIT_UPPER - Fraction(6_379, 40) * Fraction(497, 500),
    )
    forward_decimal_check = _forward_decimal_cross_check(primitive, jordan)
    decimal_check = _canonical_decimal_cross_check()
    directed_interval = _canonical_directed_interval_reconstruction()
    a, b, c = COEFFICIENTS
    checks = {
        "witness_radius_is_one": sum(value**2 for value in U_MODES) == 1,
        "quintic_discriminant_is_locked": b**2 - 4 * a * c == Fraction(-2_594_691, 500_000),
        "modal_difference_is_exactly_positive": modal_difference > 0,
        "all_diagonal_margins_are_positive": all(value > 0 for value in margins),
        "smallest_diagonal_margin_is_locked": min(margins)
        == Fraction(1_098_544_686_059_863, 41_641_817_600_000_000),
        "forward_map_is_two_strongly_monotone": 1 + LAMBDA * MU == 2,
        "p15_relative_tolerance_is_locked": Fraction(1, 250) == KAPPA,
        "p15_artifact_hash_matches": _sha256(P15_ARTIFACT_PATH) == P15_ARTIFACT_SHA256,
        "decimal_departure_cross_check": bool(decimal_check["departure_inside_locked_guard"]),
        "decimal_upstream_cross_check": bool(decimal_check["upstream_inside_locked_guard"]),
        "decimal_retention_cross_check": bool(decimal_check["retention_inside_locked_guard"]),
        "decimal_absolute_gate_fails": bool(decimal_check["absolute_gate_fails"]),
        "decimal_retention_gate_fails": bool(decimal_check["retention_gate_fails"]),
        "forward_witness_decimal_cross_check": all(
            bool(forward_decimal_check[key])
            for key in (
                "departure_inside_locked_guard",
                "upstream_inside_locked_guard",
                "retention_inside_locked_guard",
                "absolute_gate_fails",
                "retention_gate_fails",
            )
        ),
        "canonical_directed_interval_reconstruction_closes": bool(
            directed_interval["all_checks_pass"]
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent P16 reconstruction failed: {failed}")
    return {
        "checks": checks,
        "parameters": {
            "epsilon": str(EPSILON),
            "lambda": str(LAMBDA),
            "mu": str(MU),
            "kappa": str(KAPPA),
            "coefficients": [str(value) for value in COEFFICIENTS],
            "steps": STEPS,
        },
        "witness": {
            "u": [str(value) for value in U_MODES],
            "normalized_u": [str(value) for value in normalized],
            "radial_primitive": {
                "rational_part": str(primitive[0]),
                "logarithm_coefficient": str(primitive[1]),
                "logarithm_argument": str(primitive[2]),
            },
            "pre_resolvent_modal_gain_difference_exact": str(modal_difference),
        },
        "jacobian_diagonal_margins": [str(value) for value in margins],
        "fidelity": {
            "canonical_input": [str(value) for value in CANONICAL_INPUT],
            "approximate_root_hex": list(CANONICAL_ROOT_HEX),
            "absolute_gate": str(ABSOLUTE_GATE),
            "retention_gate": str(RETENTION_GATE),
            "departure_guard": [str(value) for value in DEPARTURE_GUARD],
            "upstream_guard": [str(value) for value in UPSTREAM_GUARD],
            "retention_guard": [str(value) for value in RETENTION_GUARD],
            "high_precision_decimal_cross_check": decimal_check,
            "forward_witness_decimal_cross_check": forward_decimal_check,
            "canonical_directed_interval_reconstruction": directed_interval,
            "decimal_is_authoritative_interval_proof": False,
            "directed_decimal_is_authoritative_independent_interval_proof": True,
        },
        "prior_p15": {
            "artifact_path": P15_ARTIFACT_PATH,
            "artifact_sha256": P15_ARTIFACT_SHA256,
            "source_commit": P15_SOURCE_COMMIT,
            "artifact_commit": P15_ARTIFACT_COMMIT,
            "checkpoint_tag": P15_CHECKPOINT_TAG,
            "checkpoint_tag_object": P15_CHECKPOINT_TAG_OBJECT,
        },
    }


def _compare_canonical(
    canonical_path: Path, reconstruction: dict[str, object]
) -> dict[str, object]:
    if not canonical_path.exists():
        return {"status": "not_found", "comparisons": {}, "all_exact_fields_match": False}
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    witness = canonical["exact_forward_shaping_witness"]
    fidelity = canonical["meaningful_fidelity_decision"]
    directed_interval = reconstruction["fidelity"]["canonical_directed_interval_reconstruction"]
    comparisons = {
        "schema": canonical["schema_version"] == SCHEMA_VERSION,
        "parameters": (
            canonical["operator"]["lambda"]["exact"] == reconstruction["parameters"]["lambda"]
            and canonical["operator"]["mu"]["exact"] == reconstruction["parameters"]["mu"]
            and canonical["claim_scope"]["epsilon"]["exact"]
            == reconstruction["parameters"]["epsilon"]
            and canonical["claim_scope"]["coefficients_exact"]
            == reconstruction["parameters"]["coefficients"]
        ),
        "witness_modes": witness["resolvent_singular_values"] == reconstruction["witness"]["u"],
        "witness_normalized_modes": witness["normalized_singular_values"]
        == reconstruction["witness"]["normalized_u"],
        "witness_radial_primitive": {
            key: row["exact"] for key, row in witness["radial_primitive"].items()
        }
        == reconstruction["witness"]["radial_primitive"],
        "exact_modal_difference": witness["pre_resolvent_modal_gain_difference_exact"]
        == reconstruction["witness"]["pre_resolvent_modal_gain_difference_exact"],
        "diagonal_margins": [
            row["exact"] for row in canonical["equivariance_and_reduction"]["band_diagonal_margins"]
        ]
        == reconstruction["jacobian_diagonal_margins"],
        "fidelity_thresholds": (
            fidelity["absolute_gate"]["exact"] == reconstruction["fidelity"]["absolute_gate"]
            and fidelity["retention_gate"]["exact"] == reconstruction["fidelity"]["retention_gate"]
        ),
        "fidelity_canonical_input": (
            fidelity["canonical_input"] == reconstruction["fidelity"]["canonical_input"]
            and fidelity["approximate_root_hex"]
            == reconstruction["fidelity"]["approximate_root_hex"]
        ),
        "fidelity_guards": (
            [
                fidelity["locked_departure_guard"]["strict_lower"],
                fidelity["locked_departure_guard"]["strict_upper"],
            ]
            == reconstruction["fidelity"]["departure_guard"]
            and [
                fidelity["upstream_departure_guard"]["strict_lower"],
                fidelity["upstream_departure_guard"]["strict_upper"],
            ]
            == reconstruction["fidelity"]["upstream_guard"]
            and [
                fidelity["retention_guard"]["strict_lower"],
                fidelity["retention_guard"]["strict_upper"],
            ]
            == reconstruction["fidelity"]["retention_guard"]
        ),
        "fidelity_classification": (
            fidelity["algebraic_noncollapse_passes"] is True
            and fidelity["meaningful_fidelity_passes"] is False
        ),
        "independent_canonical_interval_enclosure": (
            directed_interval["all_checks_pass"] is True
            and all(
                replay["approximate_root_exact_binary"]
                == directed_interval["approximate_root_exact_binary"]
                for replay in fidelity["arb_cross_precision"]
            )
        ),
        "prior_p15": canonical["prior_p15"] == reconstruction["prior_p15"],
        "source_snapshot": canonical["proof_replay_provenance"]["source_snapshot"]
        == _source_snapshot_for(canonical),
        "all_primary_checks": all(canonical["audit"]["checks"].values()),
    }
    return {
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def main() -> int:
    args = parse_args()
    reconstruction = _reconstruct()
    comparison = _compare_canonical(args.canonical, reconstruction)
    if args.require_canonical and comparison["status"] != "matched":
        raise SystemExit("canonical P16 certificate is missing or mismatched")
    payload = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "canonical_read_order": "only after complete independent reconstruction",
            "decimal_role": (
                "the legacy central-value calculation is non-authoritative; a separate "
                "outward-directed Decimal interval implementation independently proves "
                "the canonical residual, output enclosure, and fidelity guards"
            ),
        },
        "reconstruction": reconstruction,
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": all(reconstruction["checks"].values()),
        "all_exact_checks_passed": (
            all(reconstruction["checks"].values()) and comparison["status"] == "matched"
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
