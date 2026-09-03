#!/usr/bin/env python3
"""Independently reconstruct the exact P8 mixed-precision certificate.

This script imports neither :mod:`passive_muon` nor numerical libraries.  It
rebuilds the Sturm chain, FP32/BF16 forward-error recurrence, affine operator
bound, and P7 closure from locally stated rational data.  A canonical JSON is
opened only after every internal exact check has passed.
"""

from __future__ import annotations

import argparse
import json
from fractions import Fraction
from itertools import pairwise
from pathlib import Path
from typing import TypeAlias

Polynomial: TypeAlias = tuple[Fraction, ...]

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANONICAL = ROOT / "results/summaries/mixed_precision_certificate.json"


def _trim(polynomial: Polynomial) -> Polynomial:
    values = list(polynomial)
    while len(values) > 1 and values[-1] == 0:
        values.pop()
    return tuple(values)


def _derivative(polynomial: Polynomial) -> Polynomial:
    return tuple(Fraction(index) * value for index, value in enumerate(polynomial[1:], 1))


def _divide(dividend: Polynomial, divisor: Polynomial) -> tuple[Polynomial, Polynomial]:
    numerator = list(_trim(dividend))
    denominator = _trim(divisor)
    quotient = [Fraction(0)] * max(1, len(numerator) - len(denominator) + 1)
    while len(numerator) >= len(denominator) and any(numerator):
        offset = len(numerator) - len(denominator)
        coefficient = numerator[-1] / denominator[-1]
        quotient[offset] = coefficient
        for index, value in enumerate(denominator):
            numerator[offset + index] -= coefficient * value
        numerator = list(_trim(tuple(numerator)))
    return _trim(tuple(quotient)), _trim(tuple(numerator))


def _sturm(polynomial: Polynomial) -> tuple[Polynomial, ...]:
    chain = [polynomial, _derivative(polynomial)]
    while len(chain[-1]) > 1:
        _, remainder = _divide(chain[-2], chain[-1])
        if remainder == (0,):
            break
        chain.append(tuple(-value for value in remainder))
    return tuple(chain)


def _evaluate(polynomial: Polynomial, point: Fraction) -> Fraction:
    result = Fraction(0)
    for coefficient in reversed(polynomial):
        result = result * point + coefficient
    return result


def _signs(chain: tuple[Polynomial, ...], point: Fraction) -> tuple[int, ...]:
    return tuple(
        (value > 0) - (value < 0)
        for value in (_evaluate(polynomial, point) for polynomial in chain)
    )


def _variations(signs: tuple[int, ...]) -> int:
    nonzero = tuple(sign for sign in signs if sign)
    return sum(left != right for left, right in pairwise(nonzero))


def _fraction_map(values: dict[str, Fraction]) -> dict[str, str]:
    return {name: str(value) for name, value in values.items()}


def _chain_payload(chain: tuple[Polynomial, ...]) -> list[list[str]]:
    return [[str(value) for value in polynomial] for polynomial in chain]


def _normalization_reconstruction(
    *, u: Fraction, tau: Fraction, ub: Fraction, taub: Fraction
) -> dict[str, Fraction]:
    gamma4 = 4 * u / (1 - 4 * u)
    ratio_error = u + 2 * tau
    exact_square_error = (2 + ratio_error) * ratio_error
    square_rounding = u * (1 + ratio_error) ** 2 + tau
    pre_sum = 4 * (exact_square_error + square_rounding)
    sum_rounding = gamma4 * (4 + pre_sum) + 4 * tau / (1 - 4 * u)
    squared_norm_error = pre_sum + sum_rounding
    sqrt_sensitivity = squared_norm_error / (2 - squared_norm_error)
    norm_relative_error = (1 + sqrt_sensitivity) * (1 + u) ** 2 - 1
    locked_relative_error = Fraction(1, 2**19)
    output_error = (locked_relative_error + u) / (1 - locked_relative_error) + 2 * tau
    cast_error = ub * (1 + output_error) + 2 * taub
    return {
        "gamma4": gamma4,
        "ratio_component_error": ratio_error,
        "square_exact_error_per_entry": exact_square_error,
        "square_rounding_error_per_entry": square_rounding,
        "pre_sum_error": pre_sum,
        "sum_rounding_error": sum_rounding,
        "squared_norm_relative_error": squared_norm_error,
        "sqrt_sensitivity": sqrt_sensitivity,
        "derived_norm_relative_error": norm_relative_error,
        "locked_norm_relative_error": locked_relative_error,
        "normalized_output_error": output_error,
        "initial_bf16_cast_error": cast_error,
        "initial_total_error": output_error + cast_error,
        "initial_frobenius_bound": 1 + output_error + cast_error,
        "initial_spectral_bound": 1 + output_error + cast_error,
    }


def _stage_reconstruction(
    *,
    a: Fraction,
    b: Fraction,
    c: Fraction,
    a32: Fraction,
    b32: Fraction,
    c32: Fraction,
    u: Fraction,
    tau: Fraction,
    ub: Fraction,
    taub: Fraction,
) -> dict[str, Fraction]:
    root_two = Fraction(99, 70)
    frobenius = Fraction(7, 4)
    spectral = Fraction(5, 4)
    q_upper = Fraction(121, 100)
    local_upper = Fraction(1, 20_000)
    gamma4 = 4 * u / (1 - 4 * u)
    zmm = 8 * tau / (1 - 4 * u)
    zelt = 2 * tau

    a0f = spectral * frobenius
    a0s = spectral**2
    eps_a = gamma4 * frobenius**2 + zmm
    af = a0f + eps_a
    a_spectral = a0s + eps_a

    c0f = abs(c) * a0f
    cpre = abs(c32) * af
    eps_c = abs(c32) * eps_a + abs(c32 - c) * a0f + u * cpre + zelt
    cf = c0f + eps_c

    t0s = abs(b)
    t0f = root_two * t0s
    tpre = cf + abs(b32) * root_two
    eps_t = eps_c + abs(b32 - b) * root_two + u * tpre + zelt
    tf = t0f + eps_t

    d0s = b**2 / (4 * c)
    d0f = root_two * d0s
    eps_d = eps_t * a_spectral + t0s * eps_a + gamma4 * tf * af + zmm
    df = d0f + eps_d

    e0s = a
    e0f = root_two * e0s
    epre = df + abs(a32) * root_two
    eps_e = eps_d + abs(a32 - a) * root_two + u * epre + zelt
    ef = e0f + eps_e
    eps_y = eps_e * spectral + gamma4 * ef * frobenius + zmm

    pre_f = q_upper * root_two + local_upper
    pre_s = q_upper + local_upper
    boundary = ub * pre_f + 2 * taub
    return {
        "gamma4": gamma4,
        "zmm": zmm,
        "zelt": zelt,
        "gram_exact_frobenius": a0f,
        "gram_exact_spectral": a0s,
        "epsilon_gram": eps_a,
        "gram_computed_frobenius": af,
        "gram_computed_spectral": a_spectral,
        "c_exact_frobenius": c0f,
        "c_pre_round_frobenius": cpre,
        "epsilon_c": eps_c,
        "c_computed_frobenius": cf,
        "t_exact_spectral": t0s,
        "t_exact_frobenius": t0f,
        "t_pre_round_frobenius": tpre,
        "epsilon_t": eps_t,
        "t_computed_frobenius": tf,
        "d_exact_spectral": d0s,
        "d_exact_frobenius": d0f,
        "epsilon_d": eps_d,
        "d_computed_frobenius": df,
        "e_exact_spectral": e0s,
        "e_exact_frobenius": e0f,
        "e_pre_round_frobenius": epre,
        "epsilon_e": eps_e,
        "e_computed_frobenius": ef,
        "epsilon_y": eps_y,
        "pre_bf16_frobenius": pre_f,
        "pre_bf16_spectral": pre_s,
        "boundary_rounding_error": boundary,
        "next_frobenius": pre_f + boundary,
        "next_spectral": pre_s + boundary,
    }


def reconstruct() -> dict[str, object]:
    a = Fraction(6_889, 2_000)
    b = Fraction(-191, 40)
    c = Fraction(4_063, 2_000)
    a32 = Fraction(902_955, 262_144)
    b32 = Fraction(-10_013_901, 2_097_152)
    c32 = Fraction(8_520_729, 4_194_304)
    rho = Fraction(210_177_835_339_081, 260_261_360_000)
    rho32 = Fraction(13_231_137, 16_384)
    safe_max_abs = Fraction(2**116)
    u = Fraction(1, 2**24)
    tau = Fraction(1, 2**150)
    ub = Fraction(1, 2**8)
    taub = Fraction(1, 2**134)
    root_two = Fraction(99, 70)
    q_upper = Fraction(121, 100)

    upper_polynomial = (
        q_upper,
        -a,
        Fraction(0),
        -b,
        Fraction(0),
        -c,
    )
    chain = _sturm(upper_polynomial)
    left_signs = _signs(chain, Fraction(0))
    right_signs = _signs(chain, Fraction(5, 4))
    variations = (_variations(left_signs), _variations(right_signs))
    endpoint_values = (
        tuple(_evaluate(polynomial, Fraction(0)) for polynomial in chain),
        tuple(_evaluate(polynomial, Fraction(5, 4)) for polynomial in chain),
    )
    discriminant = b**2 - 4 * a * c

    normalization = _normalization_reconstruction(u=u, tau=tau, ub=ub, taub=taub)
    recurrence = _stage_reconstruction(
        a=a,
        b=b,
        c=c,
        a32=a32,
        b32=b32,
        c32=c32,
        u=u,
        tau=tau,
        ub=ub,
        taub=taub,
    )

    polynomial_error = Fraction(7, 4) + q_upper * root_two
    first_slope = abs(rho32 - rho) + u * rho32
    complete_slope = first_slope + u * (rho + first_slope)
    complete_intercept = polynomial_error + u * Fraction(7, 4) + (4 + 2 * u) * tau
    slope_upper = Fraction(11, 100_000)
    intercept_upper = Fraction(347, 100)
    polynomial_lipschitz = Fraction(4_848_763, 10_000)
    repaired_lipschitz = rho + polynomial_lipschitz
    adapter_slope = complete_slope * (1 + u) + repaired_lipschitz * u
    adapter_intercept = complete_intercept + 2 * (complete_slope + repaired_lipschitz) * tau
    adapter_slope_upper = Fraction(1, 5_000)
    adapter_intercept_upper = intercept_upper

    base_rate = Fraction(399_960_001, 400_000_000)
    error_gain = Fraction(1, 2_000_000)
    p00 = Fraction(72_435, 100_000)
    p01 = Fraction(-3_185, 100_000)
    p11 = Fraction(499, 100_000)
    v0 = Fraction(361, 400)
    v1 = Fraction(39, 400)
    p_determinant = p00 * p11 - p01**2
    derived_kappa = (p11 * v0**2 - 2 * p01 * v0 * v1 + p00 * v1**2) / p_determinant
    kappa = Fraction(66_221_761, 10_400_336)
    signal_gain = Fraction(1_655_544_025, 2_600_084)
    theta = Fraction(5_124)
    p8_rate = base_rate + error_gain * (1 + theta) * adapter_slope_upper**2 * signal_gain
    forcing = error_gain * (1 + 1 / theta) * adapter_intercept_upper**2
    function_lower = Fraction(13_533, 50_000)
    smoothness = Fraction(10)
    ultimate = smoothness / function_lower * forcing / (1 - p8_rate)
    safe_storage_radius = safe_max_abs**2 / signal_gain
    safe_forcing_capacity = (1 - p8_rate) * safe_storage_radius

    checks = {
        "coefficient_a": a32 - a == Fraction(-1, 32_768_000),
        "coefficient_b": b32 - b == Fraction(-1, 10_485_760),
        "coefficient_c": c32 - c == Fraction(53, 524_288_000),
        "sqrt_two": root_two**2 > 2,
        "normalizer_range": 2 * safe_max_abs < 2**118,
        "repair_range": (1 + u) * rho32 * safe_max_abs + tau + Fraction(7, 4) < 2**127,
        "positive_polynomial_factor": discriminant == Fraction(-2_594_691, 500_000),
        "polynomial_upper_closes_scalar_iteration": q_upper < Fraction(5, 4),
        "t_envelope_sign_on_invariant": b + c * Fraction(5, 4) ** 2 < 0,
        "sturm_length": len(chain) == 6,
        "sturm_left_signs": left_signs == (1, -1, -1, 1, -1, -1),
        "sturm_right_signs": right_signs == (1, -1, -1, -1, 1, -1),
        "sturm_variations": variations == (3, 3),
        "upper_left_endpoint": endpoint_values[0][0] == Fraction(121, 100),
        "upper_right_endpoint": endpoint_values[1][0] == Fraction(12_657, 409_600),
        "normalization_relative": normalization["derived_norm_relative_error"] < Fraction(1, 2**19),
        "normalization_output": normalization["normalized_output_error"] < Fraction(1, 500_000),
        "initial_boundary": normalization["initial_total_error"] < Fraction(1, 250),
        "initial_enters_frobenius_invariant": normalization["initial_frobenius_bound"]
        < Fraction(7, 4),
        "initial_enters_spectral_invariant": normalization["initial_spectral_bound"]
        < Fraction(5, 4),
        "gram_frobenius_identity": recurrence["gram_exact_frobenius"] == Fraction(35, 16),
        "gram_spectral_identity": recurrence["gram_exact_spectral"] == Fraction(25, 16),
        "t_spectral_identity": recurrence["t_exact_spectral"] == Fraction(191, 40),
        "t_frobenius_identity": recurrence["t_exact_frobenius"] == root_two * Fraction(191, 40),
        "d_spectral_identity": recurrence["d_exact_spectral"] == Fraction(182_405, 65_008),
        "d_frobenius_identity": recurrence["d_exact_frobenius"]
        == root_two * Fraction(182_405, 65_008),
        "e_spectral_identity": recurrence["e_exact_spectral"] == Fraction(6_889, 2_000),
        "e_frobenius_identity": recurrence["e_exact_frobenius"]
        == root_two * Fraction(6_889, 2_000),
        "all_local_errors": all(
            recurrence[name] < Fraction(1, 20_000)
            for name in (
                "epsilon_gram",
                "epsilon_c",
                "epsilon_t",
                "epsilon_d",
                "epsilon_e",
                "epsilon_y",
            )
        ),
        "pre_frobenius": recurrence["pre_bf16_frobenius"] == Fraction(239_587, 140_000),
        "pre_spectral": recurrence["pre_bf16_spectral"] == Fraction(24_201, 20_000),
        "frobenius_closes": recurrence["next_frobenius"] < Fraction(7, 4),
        "spectral_closes": recurrence["next_spectral"] < Fraction(5, 4),
        "polynomial_error": polynomial_error == Fraction(24_229, 7_000),
        "slope": complete_slope < slope_upper,
        "intercept": complete_intercept < intercept_upper,
        "repaired_lipschitz": repaired_lipschitz == Fraction(336_372_400_608_849, 260_261_360_000),
        "adapter_slope": adapter_slope < adapter_slope_upper,
        "adapter_intercept": adapter_intercept < adapter_intercept_upper,
        "kappa_reconstruction": derived_kappa == kappa,
        "signal_gain": signal_gain == 100 * kappa,
        "rate": p8_rate == Fraction(41_597_186_684_695_561, 41_601_344_000_000_000),
        "rate_strict": p8_rate < 1,
        "forcing": forcing == Fraction(4_936_769, 819_840_000_000),
        "ultimate": ultimate == Fraction(462_392_438_350_000_000, 207_695_315_294_468_001),
        "safe_storage_radius": safe_storage_radius == safe_max_abs**2 / signal_gain,
        "safe_forcing_capacity": safe_forcing_capacity == (1 - p8_rate) * safe_storage_radius,
        "safe_range_invariant": forcing <= safe_forcing_capacity,
    }
    all_passed = all(checks.values())
    if not all_passed:
        raise AssertionError([name for name, passed in checks.items() if not passed])

    repair = {
        "coefficient_difference": abs(rho32 - rho),
        "first_rounding_slope": first_slope,
        "complete_slope": complete_slope,
        "complete_intercept": complete_intercept,
        "slope_upper": slope_upper,
        "intercept_upper": intercept_upper,
        "ideal_polynomial_lipschitz": polynomial_lipschitz,
        "ideal_repaired_lipschitz": repaired_lipschitz,
        "real_adapter_cast_slope": u,
        "real_adapter_cast_intercept": 2 * tau,
        "real_adapter_slope": adapter_slope,
        "real_adapter_intercept": adapter_intercept,
        "real_adapter_slope_upper": adapter_slope_upper,
        "real_adapter_intercept_upper": adapter_intercept_upper,
    }
    closure = {
        "base_rate": base_rate,
        "implementation_error_gain": error_gain,
        "signal_kappa": kappa,
        "derived_signal_kappa": derived_kappa,
        "signal_storage_gain": signal_gain,
        "young_theta": theta,
        "rate": p8_rate,
        "forcing": forcing,
        "one_minus_rate": 1 - p8_rate,
        "function_storage_lower": function_lower,
        "smoothness": smoothness,
        "function_gap_ultimate": ultimate,
        "safe_storage_radius": safe_storage_radius,
        "safe_forcing_capacity": safe_forcing_capacity,
    }
    return {
        "checks": checks,
        "all_internal_exact_checks_passed": all_passed,
        "sturm_chain": _chain_payload(chain),
        "sturm_endpoint_values": [[str(value) for value in values] for values in endpoint_values],
        "sturm_endpoint_signs": [list(left_signs), list(right_signs)],
        "sturm_variations": list(variations),
        "normalization": _fraction_map(normalization),
        "recurrence": _fraction_map(recurrence),
        "repair": _fraction_map(repair),
        "p7_closure": _fraction_map(closure),
    }


def _compare_canonical(reconstruction: dict[str, object], path: Path) -> dict[str, object]:
    if not path.exists():
        return {
            "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
            "status": "not_found",
            "comparisons": {},
            "all_exact_fields_match": None,
        }

    canonical = json.loads(path.read_text(encoding="utf-8"))
    recurrence = canonical["stage_invariant_certificate"]["one_stage_recurrence"]
    normalization = canonical["normalization_certificate"]["values"]
    repair = canonical["operator_error_certificate"]["repair_rounding"]
    closure = canonical["p7_closure"]["values"]
    comparisons = {
        "sturm_chain": canonical["polynomial_range_certificate"][
            "sturm_chain_coefficients_ascending"
        ]
        == reconstruction["sturm_chain"],
        "sturm_signs": canonical["polynomial_range_certificate"]["endpoint_signs"]
        == reconstruction["sturm_endpoint_signs"],
        "normalization": {name: value["exact"] for name, value in normalization.items()}
        == reconstruction["normalization"],
        "stage_recurrence": {name: value["exact"] for name, value in recurrence.items()}
        == reconstruction["recurrence"],
        "repair": {name: value["exact"] for name, value in repair.items()}
        == reconstruction["repair"],
        "p7_closure": {name: value["exact"] for name, value in closure.items()}
        == reconstruction["p7_closure"],
        "audit": canonical["audit"]["all_exact_checks_passed"] is True,
    }
    return {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "status": "matched" if all(comparisons.values()) else "mismatch",
        "comparisons": comparisons,
        "all_exact_fields_match": all(comparisons.values()),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--canonical", type=Path, default=DEFAULT_CANONICAL)
    parser.add_argument("--require-canonical", action="store_true")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reconstruction = reconstruct()
    comparison = _compare_canonical(reconstruction, args.canonical)
    if args.require_canonical and comparison["status"] != "matched":
        raise SystemExit(f"canonical comparison failed: {comparison['status']}")
    payload = {
        "schema_version": "passive-muon-p8-independent-reconstruction-v1",
        "implementation_scope": {
            "project_package_imported": False,
            "numerical_library_imported": False,
            "arithmetic": "stdlib fractions.Fraction",
            "canonical_read_order": "only after complete independent reconstruction passes",
            "qualification": "exact second code path, not a human proof audit",
        },
        "reconstruction": reconstruction,
        "canonical_comparison": comparison,
        "all_internal_exact_checks_passed": reconstruction["all_internal_exact_checks_passed"],
        "all_exact_checks_passed": bool(
            reconstruction["all_internal_exact_checks_passed"]
            and comparison["status"] in {"matched", "not_found"}
        ),
    }
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
