from __future__ import annotations

import hashlib
import json
from fractions import Fraction
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/summaries/additive_epsilon_deficit_certificate.json"


def _load() -> dict[str, object]:
    if not RESULT.exists():
        pytest.skip("committed additive-epsilon certificate has not landed yet")
    return json.loads(RESULT.read_text(encoding="utf-8"))


def test_committed_manifest_records_clean_additive_epsilon_source_commit() -> None:
    payload = _load()
    assert payload["schema_version"] == ("passive-muon-additive-epsilon-deficit-certificate-v1")
    assert payload["git"]["branch"] == "p12-additive-epsilon-deficit"
    assert payload["git"]["dirty"] is False
    assert len(payload["git"]["sha"]) == 40
    int(payload["git"]["sha"], 16)
    assert payload["claim_scope"]["dimension_uniform_upper"] is True
    assert payload["claim_scope"]["lower_witness_shape"] == [2, 2]
    assert "BF16" in " ".join(payload["claim_scope"]["not_claimed"])


def test_committed_manifest_has_a_nonzero_finite_bracket_and_catastrophic_repair() -> None:
    payload = _load()
    bracket = payload["certified_bracket_at_epsilon_1"]
    lower = Fraction(bracket["strict_lower"]["exact"])
    upper = Fraction(bracket["upper"]["exact"])
    assert 0 < lower < upper < 159
    assert Fraction(bracket["relative_width_percent"]["exact"]) < Fraction(28, 100)

    repair = payload["repair"]
    assert Fraction(repair["deployed_epsilon"]["exact"]) == Fraction(1, 10_000_000)
    assert Fraction(repair["deployed_necessary_strict_lower"]["exact"]) > 1_500_000_000
    assert Fraction(repair["deployed_sufficient"]["exact"]) < 1_600_000_000
    usefulness = payload["usefulness_at_deployed_epsilon"]
    assert Fraction(usefulness["necessary_repair_over_base_upper"]["exact"]) > 3_000_000
    assert usefulness["classification"].startswith("catastrophically large")


def test_committed_manifest_source_snapshot_and_prior_artifacts_match() -> None:
    payload = _load()
    snapshot = payload["proof_replay_provenance"]["source_snapshot"]
    assert snapshot
    for path, expected in snapshot.items():
        source = ROOT / path
        assert source.is_file(), path
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected
    for path, record in payload["prior_artifacts"].items():
        assert record["matched"] is True
        assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == record["sha256"]


def test_committed_manifest_records_complete_replay_environment_and_configuration() -> None:
    payload = _load()
    provenance = payload["proof_replay_provenance"]
    assert provenance["seed"] is None
    assert provenance["randomness"].startswith("none;")
    assert provenance["command"] == (
        "uv run --locked python scripts/certify_additive_epsilon_deficit.py "
        "--output results/summaries/additive_epsilon_deficit_certificate.json"
    )

    software = payload["software"]
    for key in ("python", "python_implementation", "python_flint", "flint"):
        assert software[key]
    hardware = payload["hardware"]
    for key in ("platform", "operating_system", "operating_system_release", "machine_architecture"):
        assert hardware[key]
    assert hardware["logical_cpu_count"] > 0
    assert hardware["byte_order"] in {"little", "big"}

    interval = payload["arb_interval_certificate"]
    assert interval["cross_precision_cover_match"] is True
    assert [row["precision_bits"] for row in interval["passes"]] == [160, 224]
    for replay in interval["passes"]:
        global_cover = replay["global_derivative_cover"]
        assert global_cover["initial_dyadic_power"] == 12
        assert global_cover["configured_maximum_dyadic_power"] == 48
        for prefix in replay["prefix_derivative_covers"]:
            assert prefix["initial_dyadic_power"] == 8
            assert prefix["configured_maximum_dyadic_power"] == 48


def test_committed_manifest_keeps_real_and_bf16_claims_separate() -> None:
    payload = _load()
    provenance = payload["upstream_formula_provenance"]
    assert provenance["revision"] == "f98f1cacc0263b04290753e32be8d498c1efc806"
    assert provenance["audited_file_sha256"] == (
        "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
    )
    distinction = payload["real_arithmetic_vs_bf16"]
    assert distinction["theorem"] == "continuous exact-real surrogate only"
    assert "discontinuous" in distinction["deployed_map"]
    assert payload["audit"]["human_proof_audit"].startswith("pending")
