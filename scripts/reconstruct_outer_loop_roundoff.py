#!/usr/bin/env python3
"""Independently reconstruct the exact P10 outer-loop certificate.

This script imports neither :mod:`passive_muon` nor a numerical library.  It
starts from the stated rational P7, P9, and executable-shell constants,
rebuilds every storage-affine port envelope and guard, and only then reads an
optional canonical artifact for comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import string
import subprocess
from fractions import Fraction
from math import isqrt
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/outer_loop_roundoff_certificate.json"
GRID = 2**40
SOURCE_PATHS = (
    "scripts/certify_outer_loop_roundoff.py",
    "scripts/reconstruct_outer_loop_roundoff.py",
    "src/passive_muon/outer_loop_roundoff_certificate.py",
    "src/passive_muon/finite_precision_outer_loop.py",
    "src/passive_muon/scalable_mixed_precision.py",
    "src/passive_muon/scalable_mixed_precision_certificate.py",
    "src/passive_muon/mixed_precision_certificate.py",
    "src/passive_muon/floored_certificate.py",
    "src/passive_muon/robust_dissipativity.py",
    "src/passive_muon/pl_convergence.py",
    "src/passive_muon/specs.py",
    "src/passive_muon/structure_aware_stability.py",
    "src/passive_muon/upstream_momentum.py",
    "tests/test_outer_loop_roundoff_certificate.py",
    "tests/test_outer_loop_roundoff_cli.py",
    "tests/test_outer_loop_roundoff_reconstruction.py",
    "tests/test_finite_precision_outer_loop.py",
    "tests/test_finite_precision_outer_loop_diagnostic.py",
    "tests/test_result_manifests.py",
    "theory/finite_precision_outer_loop_certificate.md",
    "theory/audits/P10_HUMAN_PROOF_AUDIT.md",
    "results/summaries/P10_RESULTS.md",
    ".github/workflows/p10-finite-precision.yml",
    "pyproject.toml",
    "uv.lock",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    return parser.parse_args()


def _upper(value: Fraction) -> Fraction:
    numerator, remainder = divmod(value.numerator * GRID, value.denominator)
    return Fraction(numerator + bool(remainder), GRID)


def _lower(value: Fraction) -> Fraction:
    return Fraction(value.numerator * GRID // value.denominator, GRID)


def _sqrt_upper(value: Fraction) -> Fraction:
    scaled_numerator = value.numerator * GRID**2
    candidate = isqrt(scaled_numerator // value.denominator)
    if candidate**2 * value.denominator < scaled_numerator:
        candidate += 1
    return Fraction(candidate, GRID)


def _storage_slope(momentum: Fraction, gradient: Fraction) -> Fraction:
    # P=[[14487/20000,-637/20000],[-637/20000,499/100000]],
    # z=m/10 and u=g/10.  This is exactly L^2*c^T*P^{-1}*c.
    square = Fraction(
        124_750_000 * momentum**2
        + 1_592_500_000 * momentum * gradient
        + 18_108_750_000 * gradient**2,
        650_021,
    )
    return _sqrt_upper(square)


def _envelope(slope: Fraction, intercept: Fraction) -> dict[str, Fraction]:
    return {"slope": slope, "intercept": intercept}


def _source_snapshot() -> dict[str, str]:
    return {path: hashlib.sha256((ROOT / path).read_bytes()).hexdigest() for path in SOURCE_PATHS}


def _source_snapshot_for(git: dict[str, object]) -> dict[str, str]:
    """Rebuild a clean canonical snapshot at its recorded source commit."""

    if git.get("dirty") is not False:
        return _source_snapshot()
    commit = git.get("sha")
    if not isinstance(commit, str):
        raise RuntimeError("canonical P10 source commit is missing")
    snapshot: dict[str, str] = {}
    for path in SOURCE_PATHS:
        completed = subprocess.run(
            ["git", "show", f"{commit}:{path}"],
            cwd=ROOT,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"could not read frozen P10 source {commit}:{path}")
        snapshot[path] = hashlib.sha256(completed.stdout).hexdigest()
    return snapshot


def reconstruct() -> dict[str, Any]:
    """Rebuild P10 solely from its public exact rational inputs."""

    rows, columns = 4_096, 11_008
    entries = rows * columns
    root_n_floor = isqrt(entries)
    root_n = root_n_floor if root_n_floor**2 == entries else root_n_floor + 1
    u = Fraction(1, 2**24)
    tau = Fraction(1, 2**150)
    beta = Fraction(19, 20)
    one_minus_beta = Fraction(1, 20)
    eta = Fraction(1, 32_000)
    beta32 = Fraction(15_938_355, 16_777_216)
    one_minus_beta32 = Fraction(13_421_773, 268_435_456)
    eta32 = Fraction(8_589_935, 274_877_906_944)

    two_operation_factor = 2 * u + u**2
    c_beta = abs(beta32 - beta) + two_operation_factor * beta32
    c_gradient = abs(one_minus_beta32 - one_minus_beta) + two_operation_factor * one_minus_beta32
    ema_crumb = (3 + 2 * u) * root_n * tau
    c_eta = abs(eta32 - eta) + two_operation_factor * eta32
    master_crumb = (2 + u) * root_n * tau
    cast_crumb = root_n * tau

    represented_momentum = (1 + u) ** 2 * beta32
    represented_gradient = (1 + u) ** 2 * one_minus_beta32

    rm_m = c_beta
    rm_g = c_gradient * (1 + u) + one_minus_beta * u
    rm_b = ema_crumb + (c_gradient + one_minus_beta) * cast_crumb
    rm = _envelope(_storage_slope(rm_m, rm_g), _upper(rm_b))

    mnext_m = represented_momentum
    mnext_g = represented_gradient * (1 + u)
    mnext_b = ema_crumb + represented_gradient * cast_crumb
    rs_m = c_beta * mnext_m
    rs_g = c_beta * mnext_g + c_gradient * (1 + u) + one_minus_beta * u
    rs_b = c_beta * mnext_b + ema_crumb + (c_gradient + one_minus_beta) * cast_crumb
    rs = _envelope(_storage_slope(rs_m, rs_g), _upper(rs_b))

    signal_m = represented_momentum**2
    signal_g_hat = represented_momentum * represented_gradient + represented_gradient
    signal_b_hat = (1 + represented_momentum) * ema_crumb
    signal_g = signal_g_hat * (1 + u)
    signal_b = signal_b_hat + signal_g_hat * cast_crumb
    signal = _envelope(_storage_slope(signal_m, signal_g), _upper(signal_b))

    p9_slope = Fraction(60_114_853, 549_755_813_888)
    p9_intercept = Fraction(1_089_541_606_515, 549_755_813_888)
    repaired_lipschitz = Fraction(336_372_400_608_849, 260_261_360_000)
    output = _envelope(
        _upper((repaired_lipschitz + p9_slope) * signal["slope"]),
        _upper((repaired_lipschitz + p9_slope) * signal["intercept"] + p9_intercept),
    )

    low_guard = Fraction(1, 2**16)
    low_frobenius = root_n * low_guard
    runtime_output_max = Fraction(2**16)
    certificate_output_max = Fraction(2**15)
    master_absolute_part_under_low_guard = u * low_frobenius + master_crumb
    master_absolute_bound_under_output_and_low_guards = (
        c_eta * root_n * runtime_output_max + master_absolute_part_under_low_guard
    )
    master_absolute_bound_under_certificate_guards = (
        c_eta * root_n * certificate_output_max + master_absolute_part_under_low_guard
    )
    r_w = _envelope(
        _upper(c_eta * output["slope"]),
        _upper(c_eta * output["intercept"] + u * low_frobenius + master_crumb),
    )
    xi = _envelope(
        _upper(rm["slope"] / one_minus_beta),
        _upper(rm["intercept"] / one_minus_beta),
    )
    effective_error = _envelope(
        _upper(
            p9_slope * signal["slope"]
            + repaired_lipschitz * (rm["slope"] + rs["slope"])
            + r_w["slope"] / eta
        ),
        _upper(
            p9_intercept
            + repaired_lipschitz * (rm["intercept"] + rs["intercept"])
            + r_w["intercept"] / eta
        ),
    )

    p7_rate = Fraction(399_960_001, 400_000_000)
    gamma_g = Fraction(1, 2)
    gamma_r = Fraction(1, 2_000_000)
    theta_g = Fraction(1)
    theta_e = Fraction(837)
    rate = _upper(
        p7_rate
        + gamma_g * (1 + theta_g) * xi["slope"] ** 2
        + gamma_r * (1 + theta_e) * effective_error["slope"] ** 2
    )
    forcing = _upper(
        gamma_g * (1 + 1 / theta_g) * xi["intercept"] ** 2
        + gamma_r * (1 + 1 / theta_e) * effective_error["intercept"] ** 2
    )
    function_gap_ultimate = _upper(
        Fraction(10, 1) / Fraction(13_533, 50_000) * forcing / (1 - rate)
    )

    signal_at_one = _upper(signal["slope"] + signal["intercept"])
    output_at_one = _upper(output["slope"] + output["intercept"])
    rounded_step_at_one = _upper((1 + u) * eta32 * output_at_one + tau)
    signal_max = Fraction(2**116)
    rounded_step_max = Fraction(2)
    signal_capacity = _lower(((signal_max - signal["intercept"]) / signal["slope"]) ** 2)
    output_capacity = _lower(
        ((certificate_output_max - output["intercept"]) / output["slope"]) ** 2
    )
    forcing_capacity = _lower(1 - rate)

    momentum_at_one = _storage_slope(Fraction(1), Fraction(0))
    gradient_at_one = _storage_slope(Fraction(0), Fraction(1))
    represented_gradient_at_one = _upper((1 + u) * gradient_at_one + cast_crumb)
    beta_momentum_at_one = _upper((1 + u) * beta32 * momentum_at_one + cast_crumb)
    weighted_gradient_at_one = _upper(
        (1 + u) * one_minus_beta32 * represented_gradient_at_one + cast_crumb
    )
    momentum_next_at_one = _upper(
        (1 + u) * (beta_momentum_at_one + weighted_gradient_at_one) + cast_crumb
    )
    beta_momentum_next_at_one = _upper((1 + u) * beta32 * momentum_next_at_one + cast_crumb)
    ema_intermediate_bounds = {
        "true_gradient": _upper(gradient_at_one),
        "represented_gradient": represented_gradient_at_one,
        "beta_momentum": beta_momentum_at_one,
        "weighted_gradient": weighted_gradient_at_one,
        "momentum_next": momentum_next_at_one,
        "beta_momentum_next": beta_momentum_next_at_one,
        "signal": signal_at_one,
    }
    fp32_max_finite = Fraction((2**24 - 1) * 2**104)

    pending_bound = (1 + u) * (low_guard + rounded_step_max) + tau
    middle_guard = Fraction(2**7)
    middle_candidate_bound = (1 + u) * (middle_guard + pending_bound) + tau
    high_guard = Fraction(2**30)
    master_checks = {
        "pending_below_three": pending_bound < 3,
        "middle_candidate_below_2^8": middle_candidate_bound < 2**8,
        "new_low_within_guard": Fraction(1, 2**17) <= low_guard,
        "new_middle_within_guard": Fraction(2**6) <= middle_guard,
        "high_two_sum_cannot_overflow": high_guard + middle_candidate_bound < 2**31,
        "certificate_output_profile_implies_step_guard": (
            (1 + u) * eta32 * certificate_output_max + tau <= rounded_step_max
        ),
    }

    envelopes = {
        "momentum_port_r_m": rm,
        "signal_port_r_s": rs,
        "actual_fp32_signal": signal,
        "p9_operator_output": output,
        "master_update_port_r_W": r_w,
        "effective_p7_gradient_error": xi,
        "effective_p7_operator_error": effective_error,
    }
    checks = {
        "entry_root_is_6715": root_n == 6_715,
        "rate_matches_lock": rate == Fraction(549_700_907_325, 549_755_813_888),
        "forcing_matches_lock": forcing == Fraction(2_162_331, 1_099_511_627_776),
        "objective_neighborhood_matches_lock": function_gap_ultimate
        == Fraction(399_957_341_889, 549_755_813_888),
        "rate_is_strict": rate < 1,
        "unit_storage_is_forward_invariant": forcing <= forcing_capacity,
        "p9_signal_guard_closes": signal_at_one <= signal_max,
        "operator_output_below_2^15": output_at_one <= certificate_output_max,
        "certificate_output_is_in_runtime_domain": certificate_output_max <= runtime_output_max,
        "rounded_step_below_two": rounded_step_at_one <= rounded_step_max,
        "middle_low_guards_are_preserved": all(master_checks.values()),
        "objective_neighborhood_is_below_one": function_gap_ultimate < 1,
    }
    return {
        "shape": [rows, columns],
        "sqrt_entries_upper": root_n,
        "raw": {
            "ema_momentum_coefficient": str(c_beta),
            "ema_gradient_coefficient": str(c_gradient),
            "ema_crumb": str(ema_crumb),
            "master_output_coefficient": str(c_eta),
            "master_low_word_coefficient": str(u),
            "master_crumb": str(master_crumb),
            "master_absolute_part_under_low_guard": str(master_absolute_part_under_low_guard),
            "master_absolute_bound_under_output_and_low_guards": str(
                master_absolute_bound_under_output_and_low_guards
            ),
            "master_absolute_bound_under_certificate_guards": str(
                master_absolute_bound_under_certificate_guards
            ),
            "represented_momentum_coefficient": str(represented_momentum),
            "represented_gradient_coefficient": str(represented_gradient),
            "p9_binary32_slope": str(p9_slope),
            "p9_binary32_intercept": str(p9_intercept),
        },
        "envelopes": {
            name: {key: str(value) for key, value in envelope.items()}
            for name, envelope in envelopes.items()
        },
        "certificate": {
            "p7_rate": str(p7_rate),
            "p7_gradient_gain": str(gamma_g),
            "p7_operator_gain": str(gamma_r),
            "gradient_young": int(theta_g),
            "operator_young": int(theta_e),
            "rate_q10": str(rate),
            "constant_forcing_D10": str(forcing),
            "one_minus_q10": str(1 - rate),
            "function_gap_ultimate": str(function_gap_ultimate),
        },
        "guards": {
            "storage_radius": "1",
            "signal_at_radius": str(signal_at_one),
            "p9_signal_max_abs": str(signal_max),
            "operator_output_at_radius": str(output_at_one),
            "certificate_operator_output_max_abs": str(certificate_output_max),
            "runtime_operator_output_max_abs": str(runtime_output_max),
            "rounded_step_at_radius": str(rounded_step_at_one),
            "rounded_step_quantity": "entrywise maxabs",
            "rounded_step_max_abs": str(rounded_step_max),
            "master_word_max_abs": {
                "high": str(high_guard),
                "middle": str(middle_guard),
                "low": str(low_guard),
            },
            "signal_capacity_radius": str(signal_capacity),
            "operator_capacity_radius": str(output_capacity),
            "forcing_capacity": str(forcing_capacity),
            "ema_intermediate_frobenius_bounds": {
                name: str(value) for name, value in ema_intermediate_bounds.items()
            },
            "fp32_max_finite": str(fp32_max_finite),
            "ema_intermediates_are_finite": all(
                value < fp32_max_finite / 2 for value in ema_intermediate_bounds.values()
            ),
            "master_checks": master_checks,
            "high_word_guard": (
                "conditional and rechecked after every update; not implied by PL storage"
            ),
        },
        "checks": checks,
        "all_internal_exact_checks_passed": all(checks.values()),
    }


def _canonical_comparison(path: Path, rebuilt: dict[str, Any]) -> dict[str, Any]:
    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
            "status": "not_found",
            "comparisons": {},
            "all_exact_fields_match": None,
        }
    canonical = json.loads(path.read_text(encoding="utf-8"))

    claim_scope = {
        "evidence_kind": (
            "exact rational compositional P7/P9/P10 certificate with upward rounding"
        ),
        "guarantee": (
            "at shape 4096x11008, V_(t+1)<=q10*V_t+D10; if V_0<=1 and the "
            "conditional high-word guard is rechecked, V_t<=1 and the objective-gap "
            "limsup is bounded by the displayed value below one"
        ),
        "objective_domain": (
            "differentiable globally 10-smooth objectives satisfying the global PL "
            "inequality with constant 1; nonconvex and nonunique minimizers are included"
        ),
        "gradient_boundary": (
            "the exact objective gradient at logical W=high+middle+low is rounded once "
            "entrywise to FP32 before the locked EMA/Nesterov graph"
        ),
        "matrix_domain": (
            "one fixed 4096x11008 CPU FP32 shell and P9 operator instance under every "
            "displayed finite-range guard"
        ),
        "normalization": "exact target s/max(1,||s||_F), no additive epsilon",
        "not_claimed": [
            "literal upstream Muon, current-plus-epsilon normalization, or an unrepaired map",
            "native BLAS, GPU, tensor-core, fused, or compiler-reassociated arithmetic",
            "rounding incurred while computing the real objective gradient before its final cast",
            "model-forward parity when a model consumes only the high master word",
            "an unconditional invariant for the high master word",
            "full-parameter ISS or iterate convergence on nonunique PL minimizer sets",
            "a unique minimizer, arbitrary-pair contraction, or tightness of the gains",
        ],
    }
    outer_update = {
        "gradient_boundary": "g_hat=C32(grad f(W)); W=high+middle+low over the reals",
        "raw_shell_ports": "r_tilde_m and r_tilde_s are defined relative to represented g_hat",
        "total_theorem_ports": (
            "r_m=r_tilde_m+(1-beta)*(g_hat-grad f(W)); r_s=r_tilde_s+(1-beta)*(g_hat-grad f(W))"
        ),
        "momentum": "m_next=beta*m+(1-beta)*grad f(W)+r_m",
        "nesterov_signal": "s_next=beta*m_next+(1-beta)*grad f(W)+r_s",
        "master_update": "W_next=W-eta*Rhat(s_next)+r_W",
        "effective_gradient_error": "xi_eff=r_m/(1-beta)",
        "nominal_p7_signal": "s0=beta^2*m+(1-beta^2)*grad f(W)+(1+beta)*r_m",
        "signal_difference": "s_next-s0=r_s-r_m",
        "effective_operator_error": ("e_eff=R(s_next)-R(s0)+(Rhat(s_next)-R(s_next))-r_W/eta"),
    }
    operator = canonical.get("operator", {})
    operator_config = {
        "formula": operator.get("formula"),
        "domain": operator.get("domain"),
        "normalization_rule": operator.get("normalization_rule"),
        "floor_c": operator.get("floor_c", {}).get("exact"),
        "epsilon": operator.get("epsilon"),
        "orthogonalizer": operator.get("orthogonalizer"),
        "polynomial_formula": operator.get("polynomial_formula"),
        "polynomial_coefficients_exact": operator.get("polynomial_coefficients_exact"),
        "newton_schulz_iteration_count": operator.get("newton_schulz_iteration_count"),
        "constant_repair_rho": operator.get("constant_repair_rho", {}).get("exact"),
        "upstream_revision": operator.get("upstream_revision"),
        "upstream_muon_sha256": operator.get("upstream_muon_sha256"),
    }
    expected_operator_config = {
        "formula": "R(M)=H_(q composed 5 times)(M/max(1,||M||_F))+rho*M",
        "domain": "fixed finite real 4096x11008 matrices under the P9/P10 guards",
        "normalization_rule": "M/max(c,||M||_F), exact fixed Frobenius max floor",
        "floor_c": "1",
        "epsilon": "none; this is a max-floor normalizer, not additive epsilon",
        "orthogonalizer": "jordan_quintic",
        "polynomial_formula": "q(x)=a*x+b*x^3+c*x^5",
        "polynomial_coefficients_exact": {
            "a": "6889/2000",
            "b": "-191/40",
            "c": "4063/2000",
        },
        "newton_schulz_iteration_count": 5,
        "constant_repair_rho": "210177835339081/260261360000",
        "upstream_revision": "f98f1cacc0263b04290753e32be8d498c1efc806",
        "upstream_muon_sha256": (
            "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
        ),
    }
    expected_arithmetic_contract = {
        "outer_schema": "passive-muon-finite-precision-outer-loop-v1",
        "outer_backend": "torch-cpu-eager-ieee-rne-fp32-three-word-master-v1",
        "rounding": "IEEE-754 roundTiesToEven at every named FP32 operation",
        "gradual_underflow_required": True,
        "ftz_daz_allowed": False,
        "fma_allowed": False,
        "reassociation_allowed": False,
        "seed": "none; deterministic exact certificate and executable witness",
        "raw_port_formulas": {
            "r_tilde_m": "||r_tilde_m||_F<=C_beta||m||_F+C_g||g_hat||_F+b_ema",
            "r_tilde_s": "||r_tilde_s||_F<=C_beta||m_next||_F+C_g||g_hat||_F+b_ema",
            "gradient_boundary": ("||g_hat-grad f(W)||_F<=u||grad f(W)||_F+ceil(sqrt(n))*tau"),
            "r_W": "||r_W||_F<=C_eta||operator_output||_F+u||low||_F+b_master",
        },
    }
    port = canonical.get("port_reduction", {})
    raw_fields = {
        "raw_ema_momentum_coefficient": "ema_momentum_coefficient",
        "raw_ema_gradient_coefficient": "ema_gradient_coefficient",
        "raw_ema_crumb": "ema_crumb",
        "raw_master_output_coefficient": "master_output_coefficient",
        "raw_master_low_word_coefficient": "master_low_word_coefficient",
        "raw_master_crumb": "master_crumb",
        "raw_master_absolute_part_under_low_guard": ("master_absolute_part_under_low_guard"),
        "raw_master_absolute_bound_under_output_and_low_guards": (
            "master_absolute_bound_under_output_and_low_guards"
        ),
        "raw_master_absolute_bound_under_certificate_guards": (
            "master_absolute_bound_under_certificate_guards"
        ),
        "represented_momentum_coefficient": "represented_momentum_coefficient",
        "represented_gradient_coefficient": "represented_gradient_coefficient",
    }
    raw_comparisons = {
        canonical_name: port.get(canonical_name, {}).get("exact") == rebuilt["raw"][rebuilt_name]
        for canonical_name, rebuilt_name in raw_fields.items()
    }
    raw_comparisons.update(
        {
            "p9_binary32_slope": port.get("p9_binary32_operator_error", {})
            .get("slope", {})
            .get("exact")
            == rebuilt["raw"]["p9_binary32_slope"],
            "p9_binary32_intercept": port.get("p9_binary32_operator_error", {})
            .get("intercept", {})
            .get("exact")
            == rebuilt["raw"]["p9_binary32_intercept"],
            "ideal_repaired_lipschitz": port.get("ideal_repaired_lipschitz", {}).get("exact")
            == "336372400608849/260261360000",
        }
    )
    locked = canonical.get("locked_certificate", {})
    expected_locked = rebuilt["certificate"]
    locked_comparisons = {
        name: locked.get(name, {}).get("exact") == value
        for name, value in expected_locked.items()
        if name not in {"gradient_young", "operator_young"}
    }
    locked_comparisons.update(
        {
            "gradient_young": locked.get("gradient_young") == expected_locked["gradient_young"],
            "operator_young": locked.get("operator_young") == expected_locked["operator_young"],
            "storage_inequality": locked.get("storage_inequality") == "V_next<=q10*V+D10",
        }
    )
    guard = canonical.get("guard_closure", {})
    expected_guard = rebuilt["guards"]
    exact_guard_fields = (
        "storage_radius",
        "signal_at_radius",
        "p9_signal_max_abs",
        "operator_output_at_radius",
        "certificate_operator_output_max_abs",
        "runtime_operator_output_max_abs",
        "rounded_step_at_radius",
        "rounded_step_max_abs",
        "signal_capacity_radius",
        "operator_capacity_radius",
        "fp32_max_finite",
    )
    guard_comparisons = {
        name: guard.get(name, {}).get("exact") == expected_guard[name]
        for name in exact_guard_fields
    }
    guard_comparisons["forcing_capacity_at_locked_radius"] = (
        guard.get("forcing_capacity_at_locked_radius", {}).get("exact")
        == expected_guard["forcing_capacity"]
    )
    guard_comparisons["master_word_max_abs"] = {
        name: value.get("exact") for name, value in guard.get("master_word_max_abs", {}).items()
    } == expected_guard["master_word_max_abs"]
    guard_comparisons["ema_intermediate_frobenius_bounds"] = {
        name: value.get("exact")
        for name, value in guard.get("ema_intermediate_frobenius_bounds", {}).items()
    } == expected_guard["ema_intermediate_frobenius_bounds"]
    guard_comparisons.update(
        {
            "rounded_step_quantity": guard.get("rounded_step_quantity")
            == expected_guard["rounded_step_quantity"],
            "ema_intermediates_are_finite": guard.get("ema_intermediates_are_finite")
            is expected_guard["ema_intermediates_are_finite"],
            "middle_low_invariant_checks": guard.get("middle_low_invariant_checks")
            == expected_guard["master_checks"],
            "high_word_guard": guard.get("high_word_guard") == expected_guard["high_word_guard"],
        }
    )
    runtime_scalars = canonical.get("locked_runtime_scalars", {})
    expected_runtime_scalars = {
        "beta_source": "19/20",
        "beta_fp32": "15938355/16777216",
        "beta_fp32_bits_hex": "0x3f733333",
        "one_minus_beta_source": "1/20",
        "one_minus_beta_fp32": "13421773/268435456",
        "one_minus_beta_fp32_bits_hex": "0x3d4ccccd",
        "learning_rate_source": "1/32000",
        "learning_rate_fp32": "8589935/274877906944",
        "learning_rate_fp32_bits_hex": "0x3803126f",
    }
    actual_runtime_scalars = {
        name: (value.get("exact") if isinstance(value, dict) and "exact" in value else value)
        for name, value in runtime_scalars.items()
    }
    expected_audit_checks = {
        "bounded_update_port_does_not_imply_full_state_iss": True,
        "certificate_output_is_in_runtime_domain": True,
        "complete_certificate_accepts": True,
        "effective_gradient_scale_uses_one_minus_beta": True,
        "gradient_boundary_cast_is_absorbed": True,
        "gradient_cast_and_ema_intermediates_are_finite": True,
        "high_guard_is_explicitly_conditional": True,
        "locked_shape_is_4096x11008": True,
        "middle_low_guards_are_preserved": True,
        "objective_neighborhood_is_below_one": True,
        "operator_output_is_below_2^15": True,
        "p7_exact_lmi_replays": True,
        "p9_shape_certificate_replays": True,
        "p9_signal_guard_closes": True,
        "rate_is_strict": True,
        "rounded_step_is_below_two": True,
        "square_summable_update_port_does_not_imply_iterate_convergence": True,
        "unit_storage_is_forward_invariant": True,
    }
    expected_audit = {
        "checks": expected_audit_checks,
        "all_exact_checks_passed": True,
        "p7_exact_lmi_replayed": True,
        "p9_shape_certificate_replayed": True,
    }
    obstruction = canonical.get("flat_direction_obstruction", {})
    expected_obstruction = {
        "objective": "f(x,y)=x^2/2",
        "smoothness_upper": "10",
        "pl_constant": "1",
        "witness": (
            "at zero state, any sufficiently small fixed r_W,t=(0,epsilon) gives "
            "linear flat-direction drift; r_W,t=(0,epsilon/(t+1)) is square "
            "summable but gives harmonic drift for every epsilon>0"
        ),
        "objective_gap": "0",
        "gradient_norm": "0",
        "momentum_norm": "0",
        "bounded_error_causes_unbounded_parameter": True,
        "square_summable_error_causes_parameter_drift": True,
    }
    stalling = canonical.get("actual_p9_fp32_stalling_witness", {})
    stalling_checks = {
        "certified": stalling.get("certified") is True,
        "all_checks": bool(stalling.get("checks")) and all(stalling.get("checks", {}).values()),
        "scope": stalling.get("scope") == "actual P9 repaired-operator parameter-stalling witness",
        "operator_interface": stalling.get("operator_interface_id")
        == "passive-muon-scalable-mixed-precision-design-v1",
        "shape": stalling.get("matrix_shape") == [2, 2],
        "exact_zero_shell_ports": stalling.get("r_m_00_exact") == "0"
        and stalling.get("r_s_00_exact") == "0",
        "raw_stalls": stalling.get("raw_final_bits_hex") == stalling.get("initial_fp32_bits_hex"),
        "compensation_moves": stalling.get("compensated_first_high_move_iteration") == 20,
        "synthetic_control_qualified": stalling.get("synthetic_arithmetic_control", {}).get("scope")
        == "synthetic arithmetic control; not an operator witness",
    }
    provenance = canonical.get("proof_replay_provenance", {})
    snapshot = provenance.get("source_snapshot", {})
    git = canonical.get("git", {})
    current_snapshot = _source_snapshot_for(git)
    sha = git.get("sha")
    provenance_checks = {
        "arithmetic": provenance.get("arithmetic")
        == (
            "fractions.Fraction exact arithmetic, integer square-root ceilings, and "
            "explicit upward/downward 2^-40 rationalization"
        ),
        "numeric_solver_role": provenance.get("numeric_solver_role") == "none",
        "software_complete": bool(provenance.get("software", {}).get("python"))
        and bool(provenance.get("software", {}).get("torch")),
        "hardware_complete": set(provenance.get("hardware", {}))
        == {"platform", "machine", "processor"},
        "source_snapshot_keys": set(snapshot) == set(SOURCE_PATHS),
        "source_snapshot_hashes_current": snapshot == current_snapshot,
        "git_sha": isinstance(sha, str)
        and len(sha) == 40
        and all(character in string.hexdigits for character in sha),
        "git_branch": git.get("branch") == "p10-finite-precision-outer-loop",
        "git_clean": git.get("dirty") is False,
    }
    comparisons: dict[str, bool] = {
        "schema_version": canonical.get("schema_version")
        == "passive-muon-outer-loop-roundoff-certificate-v1",
        "claim_scope": canonical.get("claim_scope") == claim_scope,
        "outer_update": canonical.get("outer_update") == outer_update,
        "operator_config": operator_config == expected_operator_config,
        "arithmetic_contract": canonical.get("arithmetic_contract") == expected_arithmetic_contract,
        "locked_runtime_scalars": actual_runtime_scalars == expected_runtime_scalars,
        "port_shape": port.get("shape") == rebuilt["shape"],
        "port_sqrt_entries_upper": port.get("sqrt_entries_upper") == rebuilt["sqrt_entries_upper"],
        "port_form": port.get("form") == "every envelope is norm<=slope*sqrt(V)+intercept",
        "port_upper_grid_denominator": port.get("upper_grid_denominator") == str(GRID),
        "port_raw_exact_fields": all(raw_comparisons.values()),
        "locked_certificate": all(locked_comparisons.values()),
        "guard_closure": all(guard_comparisons.values()),
        "audit": canonical.get("audit") == expected_audit,
        "flat_direction_obstruction": obstruction == expected_obstruction,
        "actual_p9_fp32_stalling_witness": all(stalling_checks.values()),
        "proof_replay_provenance": all(provenance_checks.values()),
    }
    canonical_envelopes = port.get("affine_storage_envelopes", {})
    for name, envelope in rebuilt["envelopes"].items():
        comparisons[f"{name}_slope"] = (
            canonical_envelopes.get(name, {}).get("slope", {}).get("exact") == envelope["slope"]
        )
        comparisons[f"{name}_intercept"] = (
            canonical_envelopes.get(name, {}).get("intercept", {}).get("exact")
            == envelope["intercept"]
        )
    return {
        "path": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def main() -> None:
    args = parse_args()
    rebuilt = reconstruct()
    if not rebuilt["all_internal_exact_checks_passed"]:
        raise AssertionError("independent P10 internal reconstruction failed")
    comparison = _canonical_comparison(args.canonical, rebuilt)
    if args.require_canonical and comparison["status"] != "matched":
        raise AssertionError(f"canonical P10 comparison failed: {comparison}")
    payload = {
        "schema_version": "passive-muon-p10-independent-reconstruction-v1",
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "canonical_read_order": "only after complete independent exact reconstruction",
            "qualification": (
                "independent standard-library algebraic replay; not a human proof audit and "
                "not an independent reconstruction of the inherited P7/P9 certificates"
            ),
        },
        "reconstruction": rebuilt,
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": rebuilt["all_internal_exact_checks_passed"],
        "all_exact_checks_passed": rebuilt["all_internal_exact_checks_passed"]
        and comparison["status"] == "matched",
    }
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
