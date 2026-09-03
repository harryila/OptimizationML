#!/usr/bin/env python3
"""Independently reconstruct the additive-epsilon deficit certificate.

This script imports only the Python standard library.  It redoes the scalar
interval proof with directed-rounded :mod:`decimal` endpoint arithmetic,
rebuilds the exact rational full-matrix envelope, and evaluates the exact
finite pair.  The canonical artifact is read only after all internal checks.
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

if hasattr(sys, "set_int_max_str_digits"):
    sys.set_int_max_str_digits(0)

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/additive_epsilon_deficit_certificate.json"
SCHEMA_VERSION = "passive-muon-additive-epsilon-deficit-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p12-additive-epsilon-independent-reconstruction-v1"

DECIMAL_PRECISION = 90
STEPS = 5
A = Fraction(6_889, 2_000)
B = Fraction(-191, 40)
C = Fraction(4_063, 2_000)
GLOBAL_LOWER = Fraction(-199_437, 1_250)
GLOBAL_UPPER = Fraction(4_848_763, 10_000)
BANDS = (
    (Fraction(0), Fraction(1, 200), Fraction(-504, 5)),
    (Fraction(1, 200), Fraction(3, 500), Fraction(-77_877, 500)),
    (Fraction(3, 500), Fraction(63, 10_000), Fraction(-6_379, 40)),
    (Fraction(63, 10_000), Fraction(1), GLOBAL_LOWER),
)
PAIR_T = Fraction(8_974_467, 1_000_000_000)
PAIR_U_LOW = Fraction(803_760, 1_136_689)
PAIR_U_HIGH = Fraction(803_761, 1_136_689)
PAIR_LOWER = Fraction(98_823_281, 625_000)
DEPLOYED_EPSILON = Fraction(1, 10_000_000)
EXPECTED_UPPER = Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)
FLOORED_C1_UPPER = Fraction(41_528_474_059_081, 260_261_360_000)
EXPECTED_PAIR_SHA256 = "de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336"
EXPECTED_UPSTREAM_REVISION = "f98f1cacc0263b04290753e32be8d498c1efc806"
EXPECTED_UPSTREAM_FILE_SHA256 = "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
EXPECTED_PRIMARY_TRACES = {
    "global": "701b9047e67c96d1f727642c693fb545f13c4981765310162a9a51b730f34d5e",
    "1/200": "8bd62eb57bed686f163694b0a076b04c891a773482e794f05c29967bd90be3c7",
    "3/500": "32a09cd669930a67d0b1be0079f359f0b56ceb16f785b60ac3a57a14af403a23",
    "63/10000": "bda19826bba8413a2a8b8b0e12911da78ca38e6cd089545f561e772206860b1a",
}
EXPECTED_PRIMARY_PRECISIONS = (160, 224)
EXPECTED_PRIMARY_GLOBAL_METADATA = {
    "initial_dyadic_power": 12,
    "configured_maximum_dyadic_power": 48,
    "leaf_count": 25_370,
    "maximum_dyadic_power": 32,
    "ordered_leaf_trace_sha256": EXPECTED_PRIMARY_TRACES["global"],
}
EXPECTED_PRIMARY_PREFIX_METADATA = (
    {
        "endpoint_exact": "1/200",
        "derivative_strict_lower_exact": "-504/5",
        "initial_dyadic_power": 8,
        "configured_maximum_dyadic_power": 48,
        "leaf_count": 276,
        "maximum_dyadic_power": 19,
        "ordered_leaf_trace_sha256": EXPECTED_PRIMARY_TRACES["1/200"],
    },
    {
        "endpoint_exact": "3/500",
        "derivative_strict_lower_exact": "-77877/500",
        "initial_dyadic_power": 8,
        "configured_maximum_dyadic_power": 48,
        "leaf_count": 367,
        "maximum_dyadic_power": 23,
        "ordered_leaf_trace_sha256": EXPECTED_PRIMARY_TRACES["3/500"],
    },
    {
        "endpoint_exact": "63/10000",
        "derivative_strict_lower_exact": "-6379/40",
        "initial_dyadic_power": 8,
        "configured_maximum_dyadic_power": 48,
        "leaf_count": 909,
        "maximum_dyadic_power": 23,
        "ordered_leaf_trace_sha256": EXPECTED_PRIMARY_TRACES["63/10000"],
    },
)
EXPECTED_DECIMAL_GLOBAL_TRACE = "5224e1af1fed5fb677f47de03a8f0f11701aef2a90618f438266e664915dfce1"
EXPECTED_SOURCE_PATHS = (
    ".github/workflows/p12-additive-epsilon-deficit.yml",
    "README.md",
    "TASKS.md",
    "experiments/README.md",
    "results/README.md",
    "results/summaries/RESULTS.md",
    "scripts/certify_additive_epsilon_deficit.py",
    "scripts/reconstruct_additive_epsilon_deficit.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_additive_epsilon_deficit.py",
    "tests/test_additive_epsilon_deficit_cli.py",
    "tests/test_additive_epsilon_deficit_reconstruction.py",
    "tests/test_additive_epsilon_deficit_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/additive_epsilon_deficit_certificate.md",
    "theory/audits/P12_ADDITIVE_EPSILON_HUMAN_PROOF_AUDIT.md",
    "theory/claims.md",
    "results/summaries/P12_ADDITIVE_EPSILON_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)
EXPECTED_PRIOR_HASHES = {
    "results/summaries/floored_repair_certificate.json": (
        "244e9abd35f86451d8cd5d7f48b59abe961313ca5b633fc4a6365d396997ade9"
    ),
    "results/summaries/deficit_audit.json": (
        "cc5ab80928a66fb12f9a68448ef0438a393959f88fd455a66db81381d077018a"
    ),
    "results/summaries/bf16_witness.json": (
        "b5b1c0cba2326b3c07b06c3dc85dceb4e10a31ab57318d5b348934c4aef0694c"
    ),
    "results/summaries/implementation_margin_certificate.json": (
        "ad74050d66f711d2bb8e3e37f80fc29f4d6162419a2f5224931e1ca7fd2e536f"
    ),
}


@dataclass(frozen=True)
class DecimalInterval:
    lower: Decimal
    upper: Decimal

    @classmethod
    def from_fraction(cls, value: Fraction) -> DecimalInterval:
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_FLOOR
            lower = Decimal(value.numerator) / Decimal(value.denominator)
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_CEILING
            upper = Decimal(value.numerator) / Decimal(value.denominator)
        return cls(lower, upper)

    def __neg__(self) -> DecimalInterval:
        # ``Decimal.__neg__`` applies the ambient context and can silently
        # round; copy_negate changes only the sign and is exact.
        return DecimalInterval(self.upper.copy_negate(), self.lower.copy_negate())

    def __add__(self, other: DecimalInterval | int) -> DecimalInterval:
        if isinstance(other, int):
            other = DecimalInterval.from_fraction(Fraction(other))
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_FLOOR
            lower = self.lower + other.lower
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_CEILING
            upper = self.upper + other.upper
        return DecimalInterval(lower, upper)

    __radd__ = __add__

    def __mul__(self, other: DecimalInterval | int) -> DecimalInterval:
        if isinstance(other, int):
            other = DecimalInterval.from_fraction(Fraction(other))
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_FLOOR
            lower_products = tuple(
                left * right
                for left in (self.lower, self.upper)
                for right in (other.lower, other.upper)
            )
        with localcontext() as context:
            context.prec = DECIMAL_PRECISION
            context.rounding = ROUND_CEILING
            upper_products = tuple(
                left * right
                for left in (self.lower, self.upper)
                for right in (other.lower, other.upper)
            )
        return DecimalInterval(min(lower_products), max(upper_products))

    __rmul__ = __mul__


@dataclass(frozen=True, order=True)
class Cell:
    left: int
    power: int


def _interval_from_endpoints(lower: Fraction, upper: Fraction) -> DecimalInterval:
    return DecimalInterval(
        DecimalInterval.from_fraction(lower).lower,
        DecimalInterval.from_fraction(upper).upper,
    )


IA = DecimalInterval.from_fraction(A)
IB = DecimalInterval.from_fraction(B)
IC = DecimalInterval.from_fraction(C)


def _derivative_interval(lower: Fraction, upper: Fraction) -> DecimalInterval:
    value = _interval_from_endpoints(lower, upper)
    derivative = DecimalInterval.from_fraction(Fraction(1))
    for _ in range(5):
        square = value * value
        derivative = derivative * (IA + square * (3 * IB + 5 * IC * square))
        value = value * (IA + square * (IB + IC * square))
    return derivative


def _cover(
    *,
    endpoint: Fraction,
    lower: Fraction,
    upper: Fraction | None,
    initial_power: int,
    maximum_power: int = 48,
) -> dict[str, object]:
    pending = [Cell(index, initial_power) for index in range(1 << initial_power)]
    leaves: list[Cell] = []
    lower_guard = DecimalInterval.from_fraction(lower).upper
    upper_guard = None if upper is None else DecimalInterval.from_fraction(upper).lower
    while pending:
        cell = pending.pop()
        denominator = 1 << cell.power
        interval = _derivative_interval(
            endpoint * Fraction(cell.left, denominator),
            endpoint * Fraction(cell.left + 1, denominator),
        )
        accepted = interval.lower > lower_guard and (
            upper_guard is None or interval.upper < upper_guard
        )
        if accepted:
            leaves.append(cell)
            continue
        if cell.power >= maximum_power:
            raise RuntimeError(f"independent interval cover failed at {cell.left}/2^{cell.power}")
        pending.append(Cell(2 * cell.left + 1, cell.power + 1))
        pending.append(Cell(2 * cell.left, cell.power + 1))

    maximum = max(cell.power for cell in leaves)
    leaves.sort(key=lambda cell: cell.left << (maximum - cell.power))
    expected = 0
    for cell in leaves:
        scaled_left = cell.left << (maximum - cell.power)
        scaled_width = 1 << (maximum - cell.power)
        if scaled_left != expected:
            raise AssertionError("independent cover is not contiguous")
        expected += scaled_width
    if expected != 1 << maximum:
        raise AssertionError("independent cover is incomplete")
    cell_text = "".join(f"{cell.left}/{cell.power}\n" for cell in leaves)
    if upper is None:
        header = (
            f"{endpoint.numerator}/{endpoint.denominator};{lower.numerator}/{lower.denominator}\n"
        )
    else:
        # The inherited global proof-cover digest predates scaled prefixes and
        # intentionally hashes only its ordered cells.
        header = ""
    canonical = (header + cell_text).encode()
    return {
        "endpoint": str(endpoint),
        "strict_lower": str(lower),
        "strict_upper": None if upper is None else str(upper),
        "leaf_count": len(leaves),
        "maximum_power": maximum,
        "trace_sha256": hashlib.sha256(canonical).hexdigest(),
    }


def _projection_lower(lower: Fraction, upper: Fraction) -> Fraction:
    if lower + upper > 0 and upper + 3 * lower >= 0:
        return -((upper - lower) ** 2) / (8 * (upper + lower))
    return min(lower, Fraction(0))


def _band_bound(left: Fraction, right: Fraction, lower: Fraction) -> dict[str, Fraction]:
    gamma = _projection_lower(lower, GLOBAL_UPPER)

    def deficit(t: Fraction) -> Fraction:
        return -(1 - t) * ((1 - t) * lower + t * gamma)

    candidates = [(deficit(left), left), (deficit(right), right)]
    if lower != gamma:
        stationary = (2 * lower - gamma) / (2 * (lower - gamma))
        if left <= stationary <= right:
            candidates.append((deficit(stationary), stationary))
    maximum, argmax = max(candidates)
    return {
        "left": left,
        "right": right,
        "derivative_lower": lower,
        "projection_lower": gamma,
        "maximizing_radius": argmax,
        "deficit_upper": max(Fraction(0), maximum),
    }


def _jordan_response(value: Fraction) -> Fraction:
    for _ in range(STEPS):
        square = value * value
        value = value * (A + square * (B + C * square))
    return value


def _pair_deficit() -> Fraction:
    if PAIR_U_LOW**2 + PAIR_U_HIGH**2 != 1:
        raise AssertionError("Pythagorean direction failed")
    low = _jordan_response(PAIR_T * PAIR_U_LOW)
    high = _jordan_response(PAIR_T * PAIR_U_HIGH)
    raw_radius = PAIR_T / (1 - PAIR_T)
    return -(high - low) / (raw_radius * (PAIR_U_HIGH - PAIR_U_LOW))


def _fraction_sha256(value: Fraction) -> str:
    return hashlib.sha256(f"{value.numerator}/{value.denominator}".encode()).hexdigest()


def _file_sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, path: str) -> str:
    """Hash ``path`` exactly as recorded at ``commit``.

    Frozen certificates bind the source tree that generated them, not a later
    branch's mutable overview files.  CI therefore checks out full history and
    replays these hashes from the manifest's recorded clean source commit.
    """

    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read frozen source {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _source_snapshot_for(canonical: dict[str, object]) -> dict[str, str]:
    """Rebuild a canonical snapshot from its recorded repository state."""

    provenance = canonical["proof_replay_provenance"]
    expected = provenance["source_snapshot"]
    if set(expected) != set(EXPECTED_SOURCE_PATHS):
        raise RuntimeError("frozen source snapshot path set does not match the P12 lock")
    git = canonical["git"]
    if git["dirty"] is False:
        return {path: _git_blob_sha256(git["sha"], path) for path in expected}
    # A freshly generated temporary artifact may intentionally record a dirty
    # development tree.  In that case its snapshot refers to the worktree.
    return {path: _file_sha256(path) for path in expected}


def _reconstruct() -> dict[str, object]:
    global_cover = _cover(
        endpoint=Fraction(1),
        lower=GLOBAL_LOWER,
        upper=GLOBAL_UPPER,
        initial_power=12,
    )
    prefix_covers = [
        _cover(endpoint=right, lower=lower, upper=None, initial_power=8)
        for _left, right, lower in BANDS[:-1]
    ]
    bands = [_band_bound(*band) for band in BANDS]
    upper = max(item["deficit_upper"] for item in bands)
    pair = _pair_deficit()
    pair_sha = _fraction_sha256(pair)
    raw_radius = PAIR_T / (1 - PAIR_T)
    absolute_width = upper - PAIR_LOWER
    relative_width = absolute_width / PAIR_LOWER
    prior_hashes = {path: _file_sha256(path) for path in EXPECTED_PRIOR_HASHES}
    checks = {
        "independent_global_interval_cover_closed": bool(global_cover["leaf_count"]),
        "independent_prefix_interval_covers_closed": all(
            bool(item["leaf_count"]) for item in prefix_covers
        ),
        "independent_global_interval_trace_matches_lock": (
            global_cover["trace_sha256"] == EXPECTED_DECIMAL_GLOBAL_TRACE
            and global_cover["leaf_count"] == 24_338
            and global_cover["maximum_power"] == 32
        ),
        "independent_prefix_interval_traces_match_arb": all(
            item["trace_sha256"] == EXPECTED_PRIMARY_TRACES[item["endpoint"]]
            for item in prefix_covers
        ),
        "radius_bands_are_contiguous": all(
            BANDS[index][1] == BANDS[index + 1][0] for index in range(len(BANDS) - 1)
        )
        and BANDS[0][0] == 0
        and BANDS[-1][1] == 1,
        "exact_upper_matches_lock": upper == EXPECTED_UPPER,
        "pair_exceeds_strict_lower": pair > PAIR_LOWER,
        "pair_below_global_upper": pair < upper,
        "pair_hash_matches_lock": pair_sha == EXPECTED_PAIR_SHA256,
        "prior_artifact_hashes_match": prior_hashes == EXPECTED_PRIOR_HASHES,
        "deployed_repair_is_finite": upper / DEPLOYED_EPSILON > 0,
        "zero_input_derivative_is_positive": A**5 > 0,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent exact checks failed: {failed}")
    return {
        "checks": checks,
        "decimal_interval_replay": {
            "arithmetic": (
                "stdlib Decimal endpoint intervals with explicit ROUND_FLOOR/ROUND_CEILING"
            ),
            "precision_decimal_digits": DECIMAL_PRECISION,
            "global": global_cover,
            "prefixes": prefix_covers,
            "independent_cover_note": (
                "this proof cover is independently generated and need not equal Arb's cover"
            ),
        },
        "primary_arb_trace_locks": EXPECTED_PRIMARY_TRACES,
        "bands": [{key: str(value) for key, value in item.items()} for item in bands],
        "unit_epsilon_upper": str(upper),
        "pair": {
            "normalized_radius": str(PAIR_T),
            "raw_radius_at_eps_1": str(raw_radius),
            "direction_low": str(PAIR_U_LOW),
            "direction_high": str(PAIR_U_HIGH),
            "unit_direction_check": str(PAIR_U_LOW**2 + PAIR_U_HIGH**2),
            "strict_lower": str(PAIR_LOWER),
            "sha256": pair_sha,
            "numerator_digits": len(str(pair.numerator)),
            "denominator_digits": len(str(pair.denominator)),
        },
        "operator": {
            "steps": STEPS,
            "coefficients_exact": {
                "a": str(A),
                "b": str(B),
                "c": str(C),
            },
        },
        "bracket": {
            "strict_lower": str(PAIR_LOWER),
            "upper": str(upper),
            "absolute_width": str(absolute_width),
            "relative_width": str(relative_width),
            "relative_width_percent": str(100 * relative_width),
        },
        "deployed_epsilon": str(DEPLOYED_EPSILON),
        "deployed_strict_lower": str(PAIR_LOWER / DEPLOYED_EPSILON),
        "deployed_upper": str(upper / DEPLOYED_EPSILON),
        "zero_input_gain_at_epsilon_1": str(A**5),
        "repair": {
            "epsilon_1": str(upper),
            "max_floor_c_1_sufficient": str(FLOORED_C1_UPPER),
            "deployed_sufficient_over_floor_c_1": str(
                (upper / DEPLOYED_EPSILON) / FLOORED_C1_UPPER
            ),
        },
        "epsilon_evaluations": [
            {
                "epsilon": str(epsilon),
                "strict_lower": str(PAIR_LOWER / epsilon),
                "upper_and_sufficient_repair": str(upper / epsilon),
                "zero_input_derivative_gain": str(A**5 / epsilon),
            }
            for epsilon in (
                Fraction(1),
                Fraction(1, 10),
                Fraction(1, 1_000),
                DEPLOYED_EPSILON,
            )
        ],
        "prior_artifact_hashes": prior_hashes,
        "upstream": {
            "repository": "https://github.com/KellerJordan/Muon",
            "revision": EXPECTED_UPSTREAM_REVISION,
            "audited_file_sha256": EXPECTED_UPSTREAM_FILE_SHA256,
            "formula": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
            "epsilon_placement": "added to the BF16 Frobenius norm before division",
            "operation_order": (
                "cast input to BF16; orient once; norm; add Python-float epsilon; "
                "divide; run five BF16 Jordan stages; orient back"
            ),
            "coefficient_order": "A=X@X.T; B=b*A+(c*A)@A; X=a*X+B@X",
        },
    }


def _compare_canonical(
    canonical_path: Path, reconstruction: dict[str, object]
) -> dict[str, object]:
    if not canonical_path.exists():
        return {"status": "not_found", "comparisons": {}, "all_exact_fields_match": False}
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    arb_passes = canonical["arb_interval_certificate"]["passes"]
    normalized_arb_passes = [
        {
            "precision_bits": item["precision_bits"],
            "global_derivative_cover": item["global_derivative_cover"],
            "prefix_derivative_covers": item["prefix_derivative_covers"],
        }
        for item in arb_passes
    ]
    expected_arb_passes = [
        {
            "precision_bits": precision,
            "global_derivative_cover": EXPECTED_PRIMARY_GLOBAL_METADATA,
            "prefix_derivative_covers": list(EXPECTED_PRIMARY_PREFIX_METADATA),
        }
        for precision in EXPECTED_PRIMARY_PRECISIONS
    ]
    canonical_bracket = canonical["certified_bracket_at_epsilon_1"]
    canonical_pair = canonical["exact_finite_pair_lower_witness"]
    canonical_repair = canonical["repair"]
    comparisons = {
        "schema": canonical["schema_version"] == SCHEMA_VERSION,
        "claim_scope": {
            key: canonical["claim_scope"][key]
            for key in (
                "matrix_domain",
                "dimension_uniform_upper",
                "lower_witness_shape",
                "arithmetic_model",
            )
        }
        == {
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "dimension_uniform_upper": True,
            "lower_witness_shape": [2, 2],
            "arithmetic_model": "exact real arithmetic",
        },
        "operator_scope": {
            key: canonical["operator"][key]
            for key in (
                "formula",
                "normalization_rule",
                "epsilon_domain",
                "orthogonalizer",
                "scalar_composition",
                "steps",
            )
        }
        == {
            "formula": "E_h,eps(M)=H_h(M/(||M||_F+eps))",
            "normalization_rule": "additive Frobenius epsilon",
            "epsilon_domain": "every real eps>0",
            "orthogonalizer": "jordan_quintic",
            "scalar_composition": "h=q composed with itself exactly 5 times",
            "steps": STEPS,
        },
        "deficit_scope": {
            key: canonical["deficit_definition"][key]
            for key in (
                "formula",
                "authority",
                "exact_scaling",
                "scaling_identity",
                "restricted_domain_caveat",
            )
        }
        == {
            "formula": "sup_(A!=B)[-<E(A)-E(B),A-B>_F/||A-B||_F^2]_+",
            "authority": "global pairwise deficit",
            "exact_scaling": "delta(E_h,eps)=delta(E_h,1)/eps",
            "scaling_identity": "E_h,eps(eps*X)=E_h,1(X)",
            "restricted_domain_caveat": "a restricted domain must scale with eps",
        },
        "operator_steps_and_coefficients": (
            canonical["operator"]["steps"] == reconstruction["operator"]["steps"]
            and canonical["operator"]["coefficients_exact"]
            == reconstruction["operator"]["coefficients_exact"]
        ),
        "bracket": {
            key: canonical_bracket[key]["exact"]
            for key in (
                "strict_lower",
                "upper",
                "absolute_width",
                "relative_width",
                "relative_width_percent",
            )
        }
        == reconstruction["bracket"],
        "pair_parameters": {
            "normalized_radius": canonical_pair["normalized_radius"]["exact"],
            "raw_radius_at_eps_1": canonical_pair["raw_radius_at_eps_1"]["exact"],
            "direction_low": canonical_pair["unit_direction_low"]["exact"],
            "direction_high": canonical_pair["unit_direction_high"]["exact"],
            "unit_direction_check": canonical_pair["unit_direction_check"],
            "strict_lower": canonical_pair["strict_lower"]["exact"],
        }
        == {
            key: reconstruction["pair"][key]
            for key in (
                "normalized_radius",
                "raw_radius_at_eps_1",
                "direction_low",
                "direction_high",
                "unit_direction_check",
                "strict_lower",
            )
        },
        "pair_hash": canonical_pair["exact_deficit_sha256"] == reconstruction["pair"]["sha256"],
        "pair_digits": (
            canonical_pair["exact_numerator_decimal_digits"]
            == reconstruction["pair"]["numerator_digits"]
            and canonical_pair["exact_denominator_decimal_digits"]
            == reconstruction["pair"]["denominator_digits"]
        ),
        "deployed_epsilon": canonical_repair["deployed_epsilon"]["exact"]
        == reconstruction["deployed_epsilon"],
        "deployed_lower": canonical_repair["deployed_necessary_strict_lower"]["exact"]
        == reconstruction["deployed_strict_lower"],
        "deployed_upper": canonical_repair["deployed_sufficient"]["exact"]
        == reconstruction["deployed_upper"],
        "repair_comparison": {
            "epsilon_1": canonical_repair["epsilon_1"]["exact"],
            "max_floor_c_1_sufficient": canonical_repair["max_floor_c_1_sufficient"]["exact"],
            "deployed_sufficient_over_floor_c_1": canonical_repair[
                "deployed_sufficient_over_floor_c_1"
            ]["exact"],
        }
        == reconstruction["repair"],
        "epsilon_evaluations": [
            {
                "epsilon": row["epsilon"]["exact"],
                "strict_lower": row["strict_lower"]["exact"],
                "upper_and_sufficient_repair": row["upper_and_sufficient_repair"]["exact"],
                "zero_input_derivative_gain": row["zero_input_derivative_gain"]["exact"],
            }
            for row in canonical["epsilon_evaluations"]
        ]
        == reconstruction["epsilon_evaluations"],
        "origin_gain": canonical["negative_controls_and_boundaries"]["origin_control"]["exact"]
        == reconstruction["zero_input_gain_at_epsilon_1"],
        "bands": [
            {
                "left": row["left"]["exact"],
                "right": row["right"]["exact"],
                "derivative_lower": row["prefix_derivative_strict_lower"]["exact"],
                "projection_lower": row["projection_lower"]["exact"],
                "maximizing_radius": row["maximizing_radius"]["exact"],
                "deficit_upper": row["deficit_upper"]["exact"],
            }
            for row in canonical["radius_band_envelope"]
        ]
        == reconstruction["bands"],
        "active_band": [row["active"] for row in canonical["radius_band_envelope"]]
        == [False, False, False, True],
        "global_derivative_upper": canonical["arb_interval_certificate"][
            "global_derivative_strict_upper"
        ]["exact"]
        == str(GLOBAL_UPPER),
        "all_primary_arb_passes": normalized_arb_passes == expected_arb_passes
        and canonical["arb_interval_certificate"]["cross_precision_cover_match"] is True,
        "prior_hashes": {
            path: record["sha256"] for path, record in canonical["prior_artifacts"].items()
        }
        == reconstruction["prior_artifact_hashes"],
        "source_snapshot": canonical["proof_replay_provenance"]["source_snapshot"]
        == _source_snapshot_for(canonical),
        "upstream_provenance": {
            key: canonical["upstream_formula_provenance"][key]
            for key in (
                "repository",
                "revision",
                "audited_file_sha256",
                "formula",
                "epsilon_placement",
                "operation_order",
                "coefficient_order",
            )
        }
        == reconstruction["upstream"],
        "scaling_statement": canonical["deficit_definition"]["exact_scaling"]
        == "delta(E_h,eps)=delta(E_h,1)/eps",
        "real_bf16_separation": canonical["real_arithmetic_vs_bf16"]["theorem"]
        == "continuous exact-real surrogate only",
    }
    return {
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    reconstruction = _reconstruct()
    comparison = _compare_canonical(args.canonical, reconstruction)
    if args.require_canonical and comparison["status"] != "matched":
        raise SystemExit("canonical additive-epsilon certificate is missing or mismatched")
    payload = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "implementation_scope": {
            "project_package_imported": False,
            "third_party_numerical_library_imported": False,
            "canonical_read_order": "only after complete independent reconstruction",
            "qualification": "independent machine replay, not a human proof audit",
        },
        "reconstruction": reconstruction,
        "canonical_comparison": comparison,
        "all_internal_checks_passed": all(reconstruction["checks"].values()),
        "all_checks_passed": all(reconstruction["checks"].values())
        and comparison["status"] == "matched",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
