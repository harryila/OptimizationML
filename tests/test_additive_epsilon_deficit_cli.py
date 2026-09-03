from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _generate(root: Path, output: Path) -> dict[str, object]:
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_additive_epsilon_deficit.py"),
            "--output",
            str(output),
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(completed.stdout) == json.loads(output.read_text(encoding="utf-8"))
    return json.loads(completed.stdout)


def test_additive_epsilon_generator_records_the_complete_certificate(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _generate(root, tmp_path / "certificate.json")

    assert payload["schema_version"] == ("passive-muon-additive-epsilon-deficit-certificate-v1")
    assert payload["claim_scope"]["dimension_uniform_upper"] is True
    assert payload["claim_scope"]["arithmetic_model"] == "exact real arithmetic"
    assert payload["operator"]["steps"] == 5
    assert payload["operator"]["coefficients_exact"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert payload["deficit_definition"]["exact_scaling"] == ("delta(E_h,eps)=delta(E_h,1)/eps")

    bracket = payload["certified_bracket_at_epsilon_1"]
    assert Fraction(bracket["strict_lower"]["exact"]) == Fraction(98_823_281, 625_000)
    assert Fraction(bracket["upper"]["exact"]) == Fraction(
        6_602_082_433_275_499_863, 41_641_817_600_000_000
    )
    assert Fraction(bracket["relative_width_percent"]["exact"]) < Fraction(28, 100)

    witness = payload["exact_finite_pair_lower_witness"]
    assert witness["unit_direction_check"] == "1"
    assert witness["exact_deficit_sha256"] == (
        "de076057ce58ad174a52a8e0e492f7efffe0488bbca19c03cd1ce0fee75a7336"
    )
    assert witness["exact_numerator_decimal_digits"] == 49_624
    assert witness["exact_denominator_decimal_digits"] == 49_621

    repair = payload["repair"]
    assert Fraction(repair["deployed_epsilon"]["exact"]) == Fraction(1, 10_000_000)
    assert Fraction(repair["deployed_necessary_strict_lower"]["exact"]) == 1_581_172_496
    assert Fraction(repair["deployed_sufficient"]["exact"]) > 1_585_000_000
    assert payload["usefulness_at_deployed_epsilon"]["classification"].startswith(
        "catastrophically large"
    )

    interval = payload["arb_interval_certificate"]
    assert interval["cross_precision_cover_match"] is True
    assert [item["precision_bits"] for item in interval["passes"]] == [160, 224]
    global_cover = interval["passes"][0]["global_derivative_cover"]
    assert global_cover["initial_dyadic_power"] == 12
    assert global_cover["configured_maximum_dyadic_power"] == 48
    assert global_cover["leaf_count"] == 25_370
    assert all(
        row["initial_dyadic_power"] == 8 and row["configured_maximum_dyadic_power"] == 48
        for row in interval["passes"][0]["prefix_derivative_covers"]
    )
    assert payload["upstream_formula_provenance"]["revision"] == (
        "f98f1cacc0263b04290753e32be8d498c1efc806"
    )
    assert payload["real_arithmetic_vs_bf16"]["theorem"] == ("continuous exact-real surrogate only")
    assert all(item["matched"] for item in payload["prior_artifacts"].values())
    assert payload["audit"]["all_exact_checks_passed"] is True


def test_generator_requires_two_distinct_interval_precisions(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    completed = subprocess.run(
        [
            sys.executable,
            str(root / "scripts/certify_additive_epsilon_deficit.py"),
            "--precision-bits",
            "160",
            "--output",
            str(tmp_path / "invalid.json"),
        ],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "at least two distinct Arb precisions" in completed.stderr
