from __future__ import annotations

import json
import subprocess
import sys
from fractions import Fraction
from pathlib import Path


def _run_generator(root: Path, *arguments: str) -> dict[str, object]:
    completed = subprocess.run(
        [sys.executable, str(root / "scripts/certify_outer_loop_composition.py"), *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def test_p21_generator_records_complete_exact_contract() -> None:
    root = Path(__file__).resolve().parents[1]
    payload = _run_generator(root)

    assert payload["schema_version"] == (
        "passive-muon-certified-outer-loop-composition-artifact-v1"
    )
    assert payload["all_exact_checks_passed"]
    assert all(payload["checks"].values())
    assert payload["prior_p20"]["artifact_sha256"] == (
        "6bca86676eb7164a617a48f34228c40355dbc682b86b4e64ddf3cd54b8d15951"
    )

    fields = payload["reconstruction_fields"]
    assert len(fields["variable_order"]) == 7
    assert fields["stored_graph"]["no_incremental_comparison"] is True
    operator = fields["operator_contract"]
    assert operator["epsilon"] == "1/10000000"
    assert operator["newton_schulz_stages"] == 5
    assert operator["polynomial_coefficients"] == {
        "a": "6889/2000",
        "b": "-191/40",
        "c": "4063/2000",
    }
    assert operator["aspect_scaling_placement"].endswith("before P20 shield")
    assert operator["shield_sector"] == {
        "lower": "125/1024",
        "upper": "509/512",
        "center": "1143/2048",
        "radius": "893/2048",
    }

    arithmetic = fields["arithmetic_contract"]
    assert "reused bg" in arithmetic["chain"]
    assert arithmetic["shield_output_dtype"] == "binary32"
    assert arithmetic["beta_fp32_bits"] == "0x3f733333"
    assert arithmetic["primary_eta_fp32_bits"] == "0x3c088889"
    assert arithmetic["maximum_eta_fp32_bits"] == "0x3c4565c8"
    assert arithmetic["operator_output_max_abs"] == "64"
    assert arithmetic["total_step_max_abs"] == "1"

    cores = fields["core_certificates"]
    assert len(cores) == 2
    assert [core["learning_rate"] for core in cores] == ["1/120", "1/83"]
    assert [core["rate"] for core in cores] == [
        "624350169/625000000",
        "999598040401/1000000000000",
    ]
    for core in cores:
        assert len(core["raw_lmi"]) == 7
        assert all(len(row) == 7 for row in core["raw_lmi"])
        assert len(core["negative_lmi_leading_minors"]) == 7
        assert all(Fraction(value) > 0 for value in core["negative_lmi_leading_minors"])
        assert core["zero_port_lmi"] == core["frozen_p18_lmi"]
        assert core["certified"]

    profiles = fields["shape_profiles"]
    assert len(profiles) == 42
    assert {tuple(item["shape"]) for item in profiles} >= {
        (4_096, 11_008),
        (4_096, 14_336),
    }
    assert all(item["certified"] for item in profiles)
    assert all(Fraction(item["reported"]["rate_upper"]) < 1 for item in profiles)
    assert all(item["shield_shape"] == item["shape"] for item in profiles)
    assert not any(item["transposed_for_shield"] for item in profiles)
    robust = fields["roundoff_constants"]
    assert robust["external_gradient_slope"] == "1/8192"
    assert robust["model_reconstruction_slope"] == "1/81920"
    assert robust["robust_gradient_slope"] == "1/4096"
    assert "global L=10" in robust["gradient_source_composition"]
    assert fields["weight_decay_scope"]["main_pl_theorem_weight_decay"] == "zero"
    assert fields["weight_decay_scope"]["bounded_port"] == {
        "name": "conditional_1_over_131072",
        "logical_displacement": "1/131072",
        "rounded_step": "1/131072",
        "status": "conditional runtime premise; not inferred from wd or master range",
    }
    assert fields["weight_decay_scope"]["ordinary_decay_nonzero_minimizer_control"][
        "original_minimizer_is_not_an_equilibrium"
    ]


def test_p21_generator_output_file_matches_stdout(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    output = tmp_path / "p21.json"
    payload = _run_generator(root, "--output", str(output))
    assert json.loads(output.read_text(encoding="utf-8")) == payload
