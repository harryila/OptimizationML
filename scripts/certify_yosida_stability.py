#!/usr/bin/env python3
"""Generate the exact P14 Yosida nonlinear-PL stability certificate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
from decimal import Decimal, localcontext
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import Any

from passive_muon.pl_convergence import (
    PlConvergenceCertificate,
    audit_pl_convergence,
    pl_convergence_matrices,
)
from passive_muon.specs import JORDAN_QUINTIC
from passive_muon.upstream_momentum import (
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
)
from passive_muon.yosida_stability import (
    YOSIDA_STABILITY_SCHEMA_VERSION,
    audit_yosida_stability,
    direct_p13_stiff_control,
    ema_nesterov_jury_control,
    exact_certificate_checks,
    insufficient_regularization_control,
    locked_yosida_pl_certificate,
    locked_yosida_sector,
    selected_origin_jury_control,
    skew_boundary_control,
    upper_lipschitz_gap,
)

ROOT = Path(__file__).resolve().parents[1]
RESULT_PATH = "results/summaries/yosida_stability_certificate.json"
SCHEMA_VERSION = YOSIDA_STABILITY_SCHEMA_VERSION

EPSILON = Fraction(1, 10_000_000)
LAMBDA = Fraction(1, 1_000)
MU = Fraction(1_000)
BETA = Fraction(19, 20)
ETA = Fraction(1, 32_000)
SMOOTHNESS = Fraction(10)
PL_CONSTANT = Fraction(1)
TAU = Fraction(499, 500)
D14 = Fraction(0)

P13_UNIT_ORIGIN_SLOPE = Fraction(
    66_983_030_848_166_374_889_967_883,
    104_104_544_000_000_000_000_000,
)
P13_DIRECT_JURY_MARGIN = Fraction(
    -1_942_248_049_655_000_871_809_068_607,
    66_626_908_160_000_000_000_000,
)
BAD_LAMBDA = Fraction(1, 100_000)

STORAGE = (
    (Fraction(674_389, 1_000_000), Fraction(-73_827, 1_000_000)),
    (Fraction(-73_827, 1_000_000), Fraction(12_368, 1_000_000)),
)
FUNCTION_STORAGE = Fraction(313_243, 1_000_000)
INTERPOLATION_REVERSE = Fraction(622_414, 1_000_000)
RESIDUAL_MULTIPLIER = Fraction(27_530, 1_000_000)

P13_ARTIFACT_PATH = "results/summaries/radial_passivation_tradeoff_certificate.json"
P13_ARTIFACT_SHA256 = "3541bf2478178bc7487d23dbad36b6b7cb05bfb00c82e64ca7c5386b054caed9"
P13_SOURCE_COMMIT = "ef566f7d04a5c2ef1a69def69724e84c67702008"
P13_ARTIFACT_COMMIT = "31710a9a47ecd134fecfc6fea93deba27b5f40b8"
P13_CHECKPOINT_TAG = "p13-radial-passivation-checkpoint"
P13_CHECKPOINT_TAG_OBJECT = "9d42f5d18bef47de26dded7809214dfdc685c5dd"

SOURCE_PATHS = (
    ".github/workflows/ci.yml",
    ".github/workflows/p14-yosida-stability.yml",
    "scripts/certify_yosida_stability.py",
    "scripts/reconstruct_yosida_stability.py",
    "src/passive_muon/yosida_stability.py",
    "src/passive_muon/additive_epsilon_deficit.py",
    "src/passive_muon/momentum_iqc.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/radial_passivation_tradeoff.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_yosida_stability.py",
    "tests/test_yosida_stability_cli.py",
    "tests/test_yosida_stability_reconstruction.py",
    "tests/test_yosida_stability_result_manifest.py",
    "tests/test_current_research_index.py",
    "theory/yosida_stability_certificate.md",
    "theory/audits/P14_YOSIDA_STABILITY_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P14_YOSIDA_STABILITY_RESULTS.md",
    "third_party/UPSTREAM_COMMITS.md",
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def _decimal(value: Fraction, *, digits: int = 60) -> str:
    with localcontext() as context:
        context.prec = digits
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def _fraction(value: Fraction) -> dict[str, str]:
    return {"exact": str(value), "decimal": _decimal(value)}


def _matrix(matrix: tuple[tuple[Fraction, ...], ...]) -> list[list[str]]:
    return [[str(value) for value in row] for row in matrix]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_output(*arguments: str) -> str | None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _git_blob_sha256(commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"could not read frozen source {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def _git_state() -> dict[str, object]:
    status = _git_output("status", "--porcelain")
    return {
        "sha": _git_output("rev-parse", "HEAD") or "unavailable",
        "branch": _git_output("branch", "--show-current") or "unavailable",
        "dirty": bool(status) if status is not None else None,
    }


def _source_snapshot() -> dict[str, str]:
    return {path: _sha256(ROOT / path) for path in SOURCE_PATHS}


def _prior_p13() -> tuple[dict[str, str], dict[str, bool]]:
    path = ROOT / P13_ARTIFACT_PATH
    observed = _sha256(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    artifact_at_commit = _git_blob_sha256(P13_ARTIFACT_COMMIT, P13_ARTIFACT_PATH)
    record = {
        "artifact_path": P13_ARTIFACT_PATH,
        "artifact_sha256": observed,
        "source_commit": payload["git"]["sha"],
        "artifact_commit": P13_ARTIFACT_COMMIT,
        "checkpoint_tag": P13_CHECKPOINT_TAG,
        "checkpoint_tag_object": P13_CHECKPOINT_TAG_OBJECT,
    }
    checks = {
        "p13_artifact_hash_matches": observed == P13_ARTIFACT_SHA256,
        "p13_historical_artifact_blob_matches": artifact_at_commit == P13_ARTIFACT_SHA256,
        "p13_source_commit_matches": payload["git"]["sha"] == P13_SOURCE_COMMIT,
        "p13_source_was_clean": payload["git"]["dirty"] is False,
        "p13_base_was_globally_monotone": payload["full_matrix_monotonicity"]["dimension_uniform"]
        is True,
        "p13_checkpoint_tag_matches": (
            _git_output("rev-parse", P13_CHECKPOINT_TAG) == P13_CHECKPOINT_TAG_OBJECT
            and _git_output("rev-parse", f"{P13_CHECKPOINT_TAG}^{{}}") == P13_ARTIFACT_COMMIT
        ),
    }
    return record, checks


def _selected_certificate() -> PlConvergenceCertificate:
    return PlConvergenceCertificate(
        pl_constant=PL_CONSTANT,
        smoothness=SMOOTHNESS,
        beta=BETA,
        center_gain=Fraction(750),
        residual_lipschitz=Fraction(250),
        learning_rate=ETA,
        tau=TAU,
        storage=STORAGE,
        function_storage=FUNCTION_STORAGE,
        interpolation_reverse_weight=INTERPOLATION_REVERSE,
        lambda_residual_norm=RESIDUAL_MULTIPLIER,
    )


def _jury_margin(operator_slope: Fraction, curvature: Fraction) -> Fraction:
    theta = ETA * curvature * operator_slope
    return 2 * (1 + BETA) - theta * (1 - BETA) * (1 + 2 * BETA)


def _boundary_controls() -> dict[str, Any]:
    base_origin_slope = P13_UNIT_ORIGIN_SLOPE / EPSILON + MU

    def yosida_slope(parameter: Fraction) -> Fraction:
        return base_origin_slope / (1 + parameter * base_origin_slope)

    critical_theta = 2 * (1 + BETA) / ((1 - BETA) * (1 + 2 * BETA))
    critical_lambda = ETA * SMOOTHNESS / critical_theta - 1 / base_origin_slope
    selected_slope = yosida_slope(LAMBDA)
    bad_slope = yosida_slope(BAD_LAMBDA)
    exact = {
        "base_origin_slope": base_origin_slope,
        "critical_theta": critical_theta,
        "critical_lambda_curvature_10": critical_lambda,
        "selected_lambda": LAMBDA,
        "selected_actual_origin_slope": selected_slope,
        "selected_actual_jury_margin_curvature_10": _jury_margin(selected_slope, SMOOTHNESS),
        "selected_sector_endpoint_jury_margin_curvature_10": _jury_margin(1 / LAMBDA, SMOOTHNESS),
        "bad_lambda": BAD_LAMBDA,
        "bad_actual_origin_slope": bad_slope,
        "bad_actual_jury_margin_curvature_10": _jury_margin(bad_slope, SMOOTHNESS),
        "bad_sector_endpoint_jury_margin_curvature_10": _jury_margin(1 / BAD_LAMBDA, SMOOTHNESS),
        "shifted_lambda_zero_actual_origin_slope": base_origin_slope,
        "shifted_lambda_zero_actual_jury_margin_curvature_10": _jury_margin(
            base_origin_slope, SMOOTHNESS
        ),
        "p13_direct_jury_margin_curvature_1": P13_DIRECT_JURY_MARGIN,
    }
    return {
        "reconstruction_fields": {key: str(value) for key, value in exact.items()},
        "selected_point": {
            "operator_origin_slope": _fraction(selected_slope),
            "actual_curvature_10_jury_margin": _fraction(
                exact["selected_actual_jury_margin_curvature_10"]
            ),
            "sector_endpoint_curvature_10_jury_margin": _fraction(
                exact["selected_sector_endpoint_jury_margin_curvature_10"]
            ),
            "classification": "strict local pass; the global LMI is authoritative",
        },
        "finite_insufficient_regularization": {
            "lambda": _fraction(BAD_LAMBDA),
            "operator_origin_slope": _fraction(bad_slope),
            "actual_curvature_10_jury_margin": _fraction(
                exact["bad_actual_jury_margin_curvature_10"]
            ),
            "sector_endpoint_curvature_10_jury_margin": _fraction(
                exact["bad_sector_endpoint_jury_margin_curvature_10"]
            ),
            "classification": "real eigenvalue below -1; locally unstable",
        },
        "exact_lambda_boundary": _fraction(critical_lambda),
        "shifted_lambda_zero": {
            "operator_origin_slope": _fraction(base_origin_slope),
            "actual_curvature_10_jury_margin": _fraction(
                exact["shifted_lambda_zero_actual_jury_margin_curvature_10"]
            ),
            "classification": "real eigenvalue below -1; locally unstable",
        },
        "p13_direct_lambda_zero_baseline": {
            "jury_margin_curvature_1": _fraction(P13_DIRECT_JURY_MARGIN),
            "qualification": (
                "direct P13 A_epsilon baseline without the P14 strong shift; inherited "
                "negative control, not the definition of Y at lambda=0"
            ),
        },
    }


def _abstract_sector_sharpness() -> dict[str, Any]:
    lower_yosida_slope = MU / (1 + LAMBDA * MU)
    skew = skew_boundary_control()
    base_slopes = (Fraction(1_000), Fraction(1_000_000), Fraction(1_000_000_000))
    upper = 1 / LAMBDA
    approach = []
    for base_slope in base_slopes:
        yosida_slope = base_slope / (1 + LAMBDA * base_slope)
        approach.append(
            {
                "base_A_slope": _fraction(base_slope - MU),
                "shifted_B_slope": _fraction(base_slope),
                "yosida_slope": _fraction(yosida_slope),
                "gap_to_upper": _fraction(upper - yosida_slope),
            }
        )
    return {
        "scope": (
            "sharpness over the abstract full-domain monotone A class; these examples are "
            "not asserted to be values of the locked P13 map"
        ),
        "lower_endpoint": {
            "base_operator": "A=0",
            "shifted_operator": "B=mu*I",
            "yosida": "Y=500*I",
            "base_A_slope": _fraction(Fraction(0)),
            "yosida_slope": _fraction(lower_yosida_slope),
            "attains_lower_endpoint": lower_yosida_slope == 500,
        },
        "skew_boundary": {
            "base_operator": "A=1000*S on R^2, S^T=-S and S^T*S=I",
            "yosida": "Y=600*I+200*S",
            "real_part": _fraction(skew.yosida_real),
            "imaginary_part": _fraction(skew.yosida_imaginary),
            "centered_circle_residual": _fraction(skew.centered_circle_residual),
            "sector_residual": _fraction(skew.sector_residual),
        },
        "upper_endpoint_approach": {
            "upper_endpoint": _fraction(upper),
            "sequence": approach,
            "gaps_strictly_decrease_to_zero": all(
                Fraction(left["gap_to_upper"]["exact"])
                > Fraction(right["gap_to_upper"]["exact"])
                > 0
                for left, right in pairwise(approach)
            ),
            "endpoint_status": "strict supremum over finite scalar slopes",
        },
    }


def build_payload() -> dict[str, Any]:
    certificate = _selected_certificate()
    exact_audit = audit_pl_convergence(certificate)
    lifted = pl_convergence_matrices(certificate)
    core_certificate = locked_yosida_pl_certificate()
    core_audit = audit_yosida_stability()
    core_sector = locked_yosida_sector()
    core_checks = exact_certificate_checks()
    if certificate != core_certificate:
        raise AssertionError("literal P14 certificate differs from the core exact certificate")
    if not core_audit.certified or not all(core_checks.values()):
        raise AssertionError("core P14 exact audit failed")
    lower = MU / (1 + LAMBDA * MU)
    upper = 1 / LAMBDA
    center = (lower + upper) / 2
    residual_radius = (upper - lower) / 2
    surplus_factor = 1 / (LAMBDA * (1 + LAMBDA * MU))
    prior_p13, prior_checks = _prior_p13()
    controls = _boundary_controls()
    sharpness = _abstract_sector_sharpness()
    control_fields = {
        key: Fraction(value) for key, value in controls["reconstruction_fields"].items()
    }

    checks = {
        **prior_checks,
        "base_shift_is_strongly_monotone": MU > 0,
        "resolvent_map_is_two_strong": 1 + LAMBDA * MU == 2,
        "joint_sector_is_500_to_1000": lower == 500 and upper == 1_000,
        "joint_sector_surplus_factor_is_500": surplus_factor == 500,
        "centered_residual_is_exact": (
            center == 750
            and residual_radius == 250
            and center**2 - residual_radius**2 == lower * upper
        ),
        "dimensionless_step_is_15_over_64": certificate.dimensionless_step == Fraction(15, 64),
        "residual_ratio_is_one_third": certificate.residual_ratio == Fraction(1, 3),
        "storage_is_positive_definite": exact_audit.storage_positive_definite,
        "lmi_is_strictly_negative_definite": exact_audit.lmi_negative_definite,
        "function_value_flow_cancels": exact_audit.function_values_cancel,
        "rate_is_strict": TAU**2 == Fraction(249_001, 250_000) < 1,
        "deterministic_additive_term_is_zero": D14 == 0,
        "selected_actual_point_passes": control_fields["selected_actual_jury_margin_curvature_10"]
        > 0,
        "selected_sector_endpoint_passes": control_fields[
            "selected_sector_endpoint_jury_margin_curvature_10"
        ]
        > 0,
        "finite_bad_lambda_fails": control_fields["bad_actual_jury_margin_curvature_10"] < 0,
        "finite_bad_endpoint_fails": control_fields["bad_sector_endpoint_jury_margin_curvature_10"]
        < 0,
        "bad_and_selected_straddle_boundary": (
            BAD_LAMBDA < control_fields["critical_lambda_curvature_10"] < LAMBDA
        ),
        "p13_direct_control_fails": P13_DIRECT_JURY_MARGIN < 0,
        "shifted_lambda_zero_control_fails": control_fields[
            "shifted_lambda_zero_actual_jury_margin_curvature_10"
        ]
        < 0,
        "core_certificate_matches_literal_replay": certificate == core_certificate,
        "core_sector_matches_literal_replay": (
            core_sector.strong_monotonicity == lower
            and core_sector.lipschitz == upper
            and core_sector.center_gain == center
            and core_sector.residual_lipschitz == residual_radius
        ),
        "core_exact_checks_pass": all(core_checks.values()),
        "abstract_sector_sharpness_replays": (
            sharpness["lower_endpoint"]["attains_lower_endpoint"]
            and Fraction(sharpness["skew_boundary"]["centered_circle_residual"]["exact"]) == 0
            and Fraction(sharpness["skew_boundary"]["sector_residual"]["exact"]) == 0
            and sharpness["upper_endpoint_approach"]["gaps_strictly_decrease_to_zero"]
            and all(
                upper_lipschitz_gap(Fraction(row["shifted_B_slope"]["exact"]))
                == Fraction(row["gap_to_upper"]["exact"])
                for row in sharpness["upper_endpoint_approach"]["sequence"]
            )
        ),
    }
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        raise AssertionError(f"P14 exact checks failed: {failed}")

    certificate_fields = {
        "storage": _matrix(STORAGE),
        "transition": _matrix(lifted.transition),
        "selection": _matrix(lifted.selection),
        "signal_selector": [str(value) for value in lifted.signal_selector],
        "step_selector": [str(value) for value in lifted.step_selector],
        "interpolation_12": _matrix(lifted.interpolation_12),
        "interpolation_21": _matrix(lifted.interpolation_21),
        "pl_next": _matrix(lifted.pl_next),
        "residual_norm": _matrix(lifted.residual_norm),
        "lmi_matrix": _matrix(exact_audit.lmi),
        "function_storage": str(FUNCTION_STORAGE),
        "interpolation_reverse": str(INTERPOLATION_REVERSE),
        "lambda_12": str(certificate.lambda_interpolation_12),
        "lambda_21": str(certificate.lambda_interpolation_21),
        "lambda_pl_next": str(certificate.lambda_pl_next),
        "lambda_residual": str(certificate.lambda_residual_norm),
        "storage_leading_minors": [str(value) for value in exact_audit.storage_leading_minors],
        "negative_lmi_leading_minors": [
            str(value) for value in exact_audit.negative_lmi_leading_minors
        ],
        "function_coefficients": [str(value) for value in exact_audit.supply_function_coefficients],
        "expected_function_coefficients": [
            str(value) for value in exact_audit.expected_function_coefficients
        ],
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "canonical_result_path": RESULT_PATH,
        "claim_scope": {
            "evidence_kind": (
                "exact full-matrix Yosida sector and dimension-independent nonlinear PL LMI"
            ),
            "matrix_domain": "R^(m x n) for every fixed finite positive m,n",
            "objective_class": (
                "differentiable globally 10-smooth with finite infimum, global PL constant 1"
            ),
            "arithmetic_model": "exact real arithmetic",
            "guarantee": (
                "global function-value, gradient, and momentum convergence; each trajectory "
                "converges to some global minimizer"
            ),
            "not_claimed": [
                "a finite-iteration or finite-tolerance resolvent algorithm",
                "BF16, FP32, stochastic, weight-decayed, or neural-network convergence",
                "a unique minimizer or arbitrary-pair trajectory contraction",
                "literal upstream Muon parity",
            ],
        },
        "parameters": {
            "epsilon": _fraction(EPSILON),
            "lambda": _fraction(LAMBDA),
            "mu": _fraction(MU),
            "beta": _fraction(BETA),
            "eta": _fraction(ETA),
            "smoothness": _fraction(SMOOTHNESS),
            "pl_constant": _fraction(PL_CONSTANT),
        },
        "operator": {
            "base": "A_eps=E_h,eps+G_eps from P13",
            "domain": "R^(m x n) for every fixed finite positive m,n",
            "epsilon_domain": "every real epsilon>0; 1e-7 is pinned only for controls",
            "additive_normalization": "M/(||M||_F+epsilon); no max floor",
            "orthogonalizer": JORDAN_QUINTIC.name,
            "coefficients_exact": {
                name: str(value)
                for name, value in zip(("a", "b", "c"), JORDAN_QUINTIC.fractions(), strict=True)
            },
            "newton_schulz_stages": 5,
            "shift": "B=A_eps+mu*I",
            "resolvent": "J_lambdaB=(I+lambda*B)^(-1)",
            "yosida": "Y_lambdaB=(I-J_lambdaB)/lambda=B composed with J_lambdaB",
            "zero_preserving": True,
            "well_posedness": (
                "I+lambda*B is continuous, coercive, and (1+lambda*mu)-strongly monotone"
            ),
            "upstream_formula_provenance": {
                "repository": "https://github.com/KellerJordan/Muon",
                "revision": PINNED_KELLER_JORDAN_MUON_REVISION,
                "audited_file": "muon.py",
                "audited_file_sha256": PINNED_MUON_PY_SHA256,
                "formula": "X/(X.norm(dim=(-2,-1),keepdim=True)+1e-7)",
                "epsilon_placement": "added to the BF16 Frobenius norm before division",
                "qualification": (
                    "formula provenance only; the P14 theorem is the exact-real surrogate "
                    "with arbitrary positive epsilon"
                ),
            },
        },
        "yosida_sector": {
            "lower": _fraction(lower),
            "upper": _fraction(upper),
            "center": _fraction(center),
            "residual_radius": _fraction(residual_radius),
            "surplus_factor": _fraction(surplus_factor),
            "joint_incremental_iqc": (
                "<dy-m*ds,M*ds-dy>=(<dJ,dy>-mu*||dJ||^2)/(lambda*(1+lambda*mu))>=0"
            ),
            "centered_equivalent": "||dy-center*ds||^2<=residual_radius^2*||ds||^2",
            "qualification": (
                "joint sector inequality for a possibly nonsymmetric map; not a derivative "
                "Loewner-order assertion"
            ),
        },
        "normalized_dynamics": {
            "signal": "p=beta^2*z+(1-beta^2)*u",
            "momentum": "z_next=beta*z+(1-beta)*u",
            "step": "W_next-W=-alpha*(p+r*v)",
            "dimensionless_step": _fraction(certificate.dimensionless_step),
            "residual_ratio": _fraction(certificate.residual_ratio),
            "residual_supply": "||p||_F^2-||v||_F^2>=0",
        },
        "exact_pl_certificate": {
            "tau": _fraction(TAU),
            "q": _fraction(TAU**2),
            "D14": _fraction(D14),
            "storage": _matrix(STORAGE),
            "function_storage": _fraction(FUNCTION_STORAGE),
            "multipliers": {
                "interpolation_12": _fraction(certificate.lambda_interpolation_12),
                "interpolation_21": _fraction(certificate.lambda_interpolation_21),
                "pl_next": _fraction(certificate.lambda_pl_next),
                "residual": _fraction(certificate.lambda_residual_norm),
            },
            "storage_leading_minors": [
                _fraction(value) for value in exact_audit.storage_leading_minors
            ],
            "negative_lmi_leading_minors": [
                _fraction(value) for value in exact_audit.negative_lmi_leading_minors
            ],
            "function_coefficients": [
                _fraction(value) for value in exact_audit.supply_function_coefficients
            ],
            "expected_function_coefficients": [
                _fraction(value) for value in exact_audit.expected_function_coefficients
            ],
            "reconstruction_fields": certificate_fields,
            "inequality": "V_next<=(249001/250000)*V+D14 with D14=0",
            "certified": exact_audit.certified,
        },
        "consequences": {
            "function_value": "f(W_t)-f_star <= (L/c_F)*V_0*q^t",
            "gradient_and_momentum": "converge geometrically to zero",
            "iterates": "absolute summability gives convergence to some global minimizer",
            "minimizer_uniqueness": "not claimed",
        },
        "boundary_controls": controls,
        "abstract_sector_sharpness": sharpness,
        "core_boundary_cross_checks": {
            "selected_actual_jury_margin": str(selected_origin_jury_control().p_at_minus_one),
            "finite_bad_actual_jury_margin": str(
                insufficient_regularization_control().p_at_minus_one
            ),
            "direct_p13_jury_margin": str(direct_p13_stiff_control().second_jury_margin),
            "shifted_lambda_zero_jury_margin": str(
                ema_nesterov_jury_control(resolvent_parameter=Fraction(0)).p_at_minus_one
            ),
            "skew_sector_boundary_residual": str(skew_boundary_control().sector_residual),
            "all_match_literal_controls": (
                selected_origin_jury_control().p_at_minus_one
                == control_fields["selected_actual_jury_margin_curvature_10"]
                and insufficient_regularization_control().p_at_minus_one
                == control_fields["bad_actual_jury_margin_curvature_10"]
                and direct_p13_stiff_control().second_jury_margin == P13_DIRECT_JURY_MARGIN
                and ema_nesterov_jury_control(resolvent_parameter=Fraction(0)).p_at_minus_one
                == control_fields["shifted_lambda_zero_actual_jury_margin_curvature_10"]
                and skew_boundary_control().sector_residual == 0
            ),
        },
        "prior_p13": prior_p13,
        "implementation_boundary": {
            "resolvent_evaluation": "mathematical exact oracle",
            "solve_error": "none; D14=0; a solve-error port is deferred",
            "finite_precision": "not modeled",
            "bf16": "not certified",
            "upstream": "not literal upstream Muon",
        },
        "software": {
            "python": platform.python_version(),
            "python_implementation": platform.python_implementation(),
        },
        "hardware": {
            "platform": platform.platform(),
            "operating_system": platform.system(),
            "operating_system_release": platform.release(),
            "machine_architecture": platform.machine(),
            "processor": platform.processor() or None,
            "logical_cpu_count": os.cpu_count(),
            "byte_order": sys.byteorder,
        },
        "git": _git_state(),
        "proof_replay_provenance": {
            "seed": None,
            "randomness": "none; exact rational arithmetic",
            "command": (
                "uv run --locked python scripts/certify_yosida_stability.py "
                "--output results/summaries/yosida_stability_certificate.json"
            ),
            "independent_command": (
                "uv run --locked python scripts/reconstruct_yosida_stability.py --require-canonical"
            ),
            "source_snapshot": _source_snapshot(),
        },
        "audit": {
            "checks": checks,
            "all_exact_checks_passed": all(checks.values()),
            "human_proof_audit": (
                "pending and unsigned: theory/audits/P14_YOSIDA_STABILITY_HUMAN_PROOF_AUDIT.md"
            ),
        },
    }


def main() -> int:
    args = parse_args()
    payload = build_payload()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
