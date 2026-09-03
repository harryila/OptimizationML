#!/usr/bin/env python3
"""Independently reconstruct the P13 radial-passivation certificate.

Only the Python standard library is imported.  Logarithms are enclosed with
exact :class:`fractions.Fraction` arithmetic after binary range reduction and
an atanh series with an explicit geometric tail.  The canonical artifact is
read only after every internal exact and interval check has passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/radial_passivation_tradeoff_certificate.json"
SCHEMA_VERSION = "passive-muon-radial-passivation-tradeoff-certificate-v1"
RECONSTRUCTION_SCHEMA_VERSION = "passive-muon-p13-independent-reconstruction-v1"

A = Fraction(6_889, 2_000)
STEPS = 5
EPSILON = Fraction(1, 10_000_000)
UNIT_UPPER = Fraction(6_602_082_433_275_499_863, 41_641_817_600_000_000)
FINAL_BAND_LOWER = Fraction(-199_437, 1_250)
FINAL_BAND_PROJECTION = Fraction(-41_528_474_059_081, 260_261_360_000)
PAIR_STRICT_LOWER = Fraction(98_823_281, 625_000)
Z0 = Fraction(63, 9_937)
T0 = Fraction(63, 10_000)
P11_SIGNAL_BOUND = Fraction(13_872_266_672_489, 549_755_813_888)
BETA = Fraction(19, 20)
ETA = Fraction(1, 32_000)

MAGNITUDE_GUARDS = {
    Fraction(1): (
        Fraction(2_571_857_826_470_212_145, 10**15),
        Fraction(2_571_857_826_470_212_146, 10**15),
    ),
    P11_SIGNAL_BOUND: (
        Fraction(3_086_959_580_254_301_425, 10**15),
        Fraction(3_086_959_580_254_301_426, 10**15),
    ),
}

P12_ARTIFACT_SHA256 = "e764a60723543fa0342d0c178886fe57a91b5f4511739cd69d8f2e07dc21df2c"
P11_ARTIFACT_SHA256 = "ad74050d66f711d2bb8e3e37f80fc29f4d6162419a2f5224931e1ca7fd2e536f"
P12_PAIR_EXACT_DEFICIT_SHA256 = "de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336"

EXPECTED_SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p12-additive-epsilon-deficit.yml",
    ".github/workflows/p13-radial-passivation-tradeoff.yml",
    "scripts/certify_radial_passivation_tradeoff.py",
    "scripts/reconstruct_radial_passivation_tradeoff.py",
    "scripts/reconstruct_additive_epsilon_deficit.py",
    "scripts/reconstruct_outer_loop_roundoff.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_radial_passivation_tradeoff.py",
    "tests/test_radial_passivation_tradeoff_cli.py",
    "tests/test_radial_passivation_tradeoff_reconstruction.py",
    "tests/test_radial_passivation_tradeoff_result_manifest.py",
    "tests/test_additive_epsilon_deficit_result_manifest.py",
    "tests/test_current_research_index.py",
    "tests/test_outer_loop_roundoff_cli.py",
    "tests/test_result_manifests.py",
    "theory/radial_passivation_tradeoff.md",
    "theory/audits/P13_RADIAL_PASSIVATION_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P13_RADIAL_PASSIVATION_RESULTS.md",
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
    parser.add_argument("--series-terms", type=int, default=96)
    return parser.parse_args()


def _log_atanh_bounds(value: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    """Enclose ``log(value)`` for ``1 <= value <= 2`` exactly."""

    if not 1 <= value <= 2:
        raise ValueError("atanh log input must lie in [1,2]")
    if terms < 1:
        raise ValueError("series terms must be positive")
    y = (value - 1) / (value + 1)
    lower = 2 * sum((y ** (2 * index + 1) / (2 * index + 1) for index in range(terms)), Fraction())
    first_omitted_power = 2 * terms + 1
    tail = 2 * y**first_omitted_power / (first_omitted_power * (1 - y**2))
    return lower, lower + tail


def _floor_log2(value: Fraction) -> int:
    if value < 1:
        raise ValueError("binary range reduction expects value >= 1")
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    while value < 2**exponent:
        exponent -= 1
    while value >= 2 ** (exponent + 1):
        exponent += 1
    return exponent


def _log_bounds(value: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    """Enclose ``log(value)`` with exact range reduction and rational tails."""

    exponent = _floor_log2(value)
    reduced = value / 2**exponent
    reduced_lower, reduced_upper = _log_atanh_bounds(reduced, terms)
    two_lower, two_upper = _log_atanh_bounds(Fraction(2), terms)
    return (
        exponent * two_lower + reduced_lower,
        exponent * two_upper + reduced_upper,
    )


def _tail_majorant(z: Fraction) -> Fraction:
    return -(FINAL_BAND_LOWER + FINAL_BAND_PROJECTION * z) / (1 + z) ** 2


def _p_bounds(z: Fraction, terms: int) -> tuple[Fraction, Fraction]:
    if z < 0:
        raise ValueError("normalized radius must be nonnegative")
    if z <= Z0:
        value = UNIT_UPPER * z
        return value, value
    log_argument = (1 + z) / (1 + Z0)
    log_lower, log_upper = _log_bounds(log_argument, terms)
    exact_part = UNIT_UPPER * Z0 + (FINAL_BAND_LOWER - FINAL_BAND_PROJECTION) * (
        1 / (1 + z) - 1 / (1 + Z0)
    )
    positive_coefficient = -FINAL_BAND_PROJECTION
    return (
        exact_part + positive_coefficient * log_lower,
        exact_part + positive_coefficient * log_upper,
    )


def _sha256(path: str) -> str:
    return hashlib.sha256((ROOT / path).read_bytes()).hexdigest()


def _git_blob_sha256(commit: str, path: str) -> str:
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
    expected = canonical["proof_replay_provenance"]["source_snapshot"]
    if set(expected) != set(EXPECTED_SOURCE_PATHS):
        raise RuntimeError("source snapshot path set does not match the P13 lock")
    git = canonical["git"]
    if git["dirty"] is False:
        return {path: _git_blob_sha256(git["sha"], path) for path in expected}
    return {path: _sha256(path) for path in expected}


def _reconstruct(terms: int) -> dict[str, object]:
    join = _tail_majorant(Z0)
    tail_derivative_constant = 2 * FINAL_BAND_LOWER - FINAL_BAND_PROJECTION
    origin_polynomial_gain = A**STEPS
    combined_origin_gain = origin_polynomial_gain + UNIT_UPPER
    critical_theta = 2 * (1 + BETA) / ((1 - BETA) * (1 + 2 * BETA))
    full_theta = ETA * combined_origin_gain / EPSILON
    repair_theta = ETA * UNIT_UPPER / EPSILON
    full_jury_margin = 2 * (1 + BETA) - full_theta * (1 - BETA) * (1 + 2 * BETA)
    repair_jury_margin = 2 * (1 + BETA) - repair_theta * (1 - BETA) * (1 + 2 * BETA)
    full_eta_critical = critical_theta * EPSILON / combined_origin_gain
    repair_eta_critical = critical_theta * EPSILON / UNIT_UPPER

    magnitudes: list[dict[str, str]] = []
    magnitude_checks: dict[str, bool] = {}
    for raw_radius, (guard_lower, guard_upper) in MAGNITUDE_GUARDS.items():
        z = raw_radius / EPSILON
        lower, upper = _p_bounds(z, terms)
        key = str(raw_radius)
        magnitude_checks[f"magnitude_{key}_inside_locked_guard"] = (
            guard_lower < lower <= upper < guard_upper
        )
        magnitude_checks[f"magnitude_{key}_fraction_interval_is_sharp"] = upper - lower < Fraction(
            1, 10**30
        )
        magnitudes.append(
            {
                "raw_radius": key,
                "z": str(z),
                "guard_lower": str(guard_lower),
                "guard_upper": str(guard_upper),
                "reconstruction_width_strict_upper": "1/1000000000000000000000000000000",
                "constant_repair_output": str(UNIT_UPPER * raw_radius / EPSILON),
            }
        )

    w0 = Fraction(1, 389_025_000)
    s1 = (1 - BETA**2) * w0
    one_step_ratio_lower = ETA * UNIT_UPPER / (399 * w0)
    one_step_objective_lower = (one_step_ratio_lower - 1) ** 2
    checks = {
        "z0_maps_to_t0": Z0 / (1 + Z0) == T0,
        "majorant_join_is_exact": join == UNIT_UPPER,
        "tail_derivative_constant_is_negative": tail_derivative_constant < 0,
        "tail_slope_is_negative": FINAL_BAND_PROJECTION < 0,
        "tail_is_positive_at_join": join > 0,
        "p_log_coefficient_is_positive": -FINAL_BAND_PROJECTION > 0,
        "universal_lower_is_positive": PAIR_STRICT_LOWER > 0,
        "constructed_lipschitz_exceeds_witness_lower": UNIT_UPPER > PAIR_STRICT_LOWER,
        "critical_theta_is_780_over_29": critical_theta == Fraction(780, 29),
        "repair_only_jury_control_fails": repair_jury_margin < 0,
        "full_repaired_jury_control_fails": full_jury_margin < 0,
        "prior_p12_artifact_matches": _sha256(
            "results/summaries/additive_epsilon_deficit_certificate.json"
        )
        == P12_ARTIFACT_SHA256,
        "prior_p11_artifact_matches": _sha256(
            "results/summaries/implementation_margin_certificate.json"
        )
        == P11_ARTIFACT_SHA256,
        "one_step_signal_is_epsilon_over_399": s1 == EPSILON / 399,
        "one_step_amplification_is_large": one_step_ratio_lower > 4_800,
        "one_step_objective_expansion_is_large": one_step_objective_lower > 23_325_554,
        **magnitude_checks,
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"independent P13 reconstruction failed: {failed}")

    return {
        "checks": checks,
        "constants": {
            "unit_upper": str(UNIT_UPPER),
            "final_band_lower": str(FINAL_BAND_LOWER),
            "final_band_projection": str(FINAL_BAND_PROJECTION),
            "pair_strict_lower": str(PAIR_STRICT_LOWER),
            "z0": str(Z0),
            "t0": str(T0),
        },
        "closed_form": {
            "tail_majorant": "-(a+Gamma*z)/(1+z)^2",
            "tail_integral": ("U*z0-Gamma*log((1+z)/(1+z0))+(a-Gamma)*(1/(1+z)-1/(1+z0))"),
            "join_value": str(join),
            "tail_derivative_constant": str(tail_derivative_constant),
        },
        "stiffness": {
            "constructed_lipschitz_unit": str(UNIT_UPPER),
            "universal_strict_lower_unit": str(PAIR_STRICT_LOWER),
            "relative_gap": str((UNIT_UPPER - PAIR_STRICT_LOWER) / PAIR_STRICT_LOWER),
            "relative_gap_percent": str(100 * (UNIT_UPPER - PAIR_STRICT_LOWER) / PAIR_STRICT_LOWER),
            "exact_pair_deficit_sha256": P12_PAIR_EXACT_DEFICIT_SHA256,
        },
        "magnitudes": magnitudes,
        "log_reconstruction": {
            "method": "exact Fraction atanh series after binary range reduction",
            "series_terms": terms,
            "explicit_positive_geometric_tail": True,
        },
        "explicit_step": {
            "beta": str(BETA),
            "eta": str(ETA),
            "epsilon": str(EPSILON),
            "origin_polynomial_gain_unit": str(origin_polynomial_gain),
            "combined_origin_gain_unit": str(combined_origin_gain),
            "critical_theta": str(critical_theta),
            "repair_only_eta_critical": str(repair_eta_critical),
            "full_repaired_eta_critical": str(full_eta_critical),
            "repair_only_jury_margin": str(repair_jury_margin),
            "full_repaired_jury_margin": str(full_jury_margin),
            "one_step_w0": str(w0),
            "one_step_signal": str(s1),
            "one_step_normalized_signal": str(s1 / (s1 + EPSILON)),
            "one_step_repair_ratio_lower": str(one_step_ratio_lower),
            "one_step_objective_growth_strict_lower": str(one_step_objective_lower),
        },
        "prior_artifact_hashes": {
            "results/summaries/additive_epsilon_deficit_certificate.json": P12_ARTIFACT_SHA256,
            "results/summaries/implementation_margin_certificate.json": P11_ARTIFACT_SHA256,
        },
        "p11_signal_bound": str(P11_SIGNAL_BOUND),
    }


def _compare_canonical(
    canonical_path: Path, reconstruction: dict[str, object]
) -> dict[str, object]:
    if not canonical_path.exists():
        return {"status": "not_found", "comparisons": {}, "all_exact_fields_match": False}
    canonical = json.loads(canonical_path.read_text(encoding="utf-8"))
    comparisons = {
        "schema": canonical["schema_version"] == SCHEMA_VERSION,
        "constants": {
            key: canonical["radial_majorant"][key]["exact"]
            for key in (
                "unit_deficit_upper",
                "final_band_derivative_lower",
                "final_band_projection_lower",
                "pair_strict_lower",
                "z0",
                "t0",
            )
        }
        == {
            "unit_deficit_upper": reconstruction["constants"]["unit_upper"],
            "final_band_derivative_lower": reconstruction["constants"]["final_band_lower"],
            "final_band_projection_lower": reconstruction["constants"]["final_band_projection"],
            "pair_strict_lower": reconstruction["constants"]["pair_strict_lower"],
            "z0": reconstruction["constants"]["z0"],
            "t0": reconstruction["constants"]["t0"],
        },
        "closed_form": canonical["radial_majorant"]["closed_form"] == reconstruction["closed_form"],
        "stiffness": {
            "constructed_lipschitz_unit": canonical["stiffness_tradeoff"]["constructed_lipschitz"][
                "unit_constant"
            ]["exact"],
            "universal_strict_lower_unit": canonical["stiffness_tradeoff"]["universal_lower_bound"][
                "unit_strict_lower"
            ]["exact"],
            "relative_gap": canonical["stiffness_tradeoff"]["relative_gap"]["exact"],
            "relative_gap_percent": canonical["stiffness_tradeoff"]["relative_gap_percent"][
                "exact"
            ],
            "exact_pair_deficit_sha256": canonical["stiffness_tradeoff"]["universal_lower_bound"][
                "exact_pair_deficit_sha256"
            ],
        }
        == reconstruction["stiffness"],
        "magnitudes": [
            {
                "raw_radius": row["raw_radius"]["exact"],
                "z": row["z"]["exact"],
                "guard_lower": row["radial_output_guard"]["lower_exact"],
                "guard_upper": row["radial_output_guard"]["upper_exact"],
                "reconstruction_width_strict_upper": row["independent_fraction_enclosure"][
                    "width_strict_upper_exact"
                ],
                "constant_repair_output": row["constant_repair_output"]["exact"],
            }
            for row in canonical["magnitude_evaluations"]
        ]
        == reconstruction["magnitudes"],
        "explicit_step": {
            key: canonical["explicit_step_negative_control"][key]["exact"]
            for key in reconstruction["explicit_step"]
        }
        == reconstruction["explicit_step"],
        "p11_signal": canonical["p11_signal_benchmark"]["exact"]
        == reconstruction["p11_signal_bound"],
        "prior_artifacts": {
            path: row["sha256"] for path, row in canonical["prior_artifacts"].items()
        }
        == reconstruction["prior_artifact_hashes"],
        "source_snapshot": canonical["proof_replay_provenance"]["source_snapshot"]
        == _source_snapshot_for(canonical),
        "all_primary_checks": all(canonical["audit"]["checks"].values()),
        "scope": (
            canonical["claim_scope"]["arithmetic_model"] == "exact real arithmetic"
            and canonical["claim_scope"]["matrix_domain"]
            == "R^(m x n) for every fixed finite positive m,n"
            and canonical["claim_scope"]["lower_tradeoff_shape_condition"] == "min(m,n)>=2"
        ),
    }
    return {
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def main() -> int:
    args = parse_args()
    reconstruction = _reconstruct(args.series_terms)
    comparison = _compare_canonical(args.canonical, reconstruction)
    if args.require_canonical and comparison["status"] != "matched":
        raise SystemExit("canonical P13 certificate is missing or mismatched")
    payload = {
        "schema_version": RECONSTRUCTION_SCHEMA_VERSION,
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "canonical_read_order": "only after complete independent reconstruction",
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
