"""Backend-specific executable witness for the deployed Keller--Jordan order.

This module intentionally does not provide a Jacobian or a real-arithmetic
certificate.  BF16 casting makes the evaluated operator discontinuous.  The
only claim supported here is the concrete pairwise inequality returned by the
recorded backend, software stack, and operation order.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any

import torch
from torch import Tensor

from passive_muon.deployed import keller_jordan_map
from passive_muon.specs import JORDAN_QUINTIC

BF16_WITNESS_SCHEMA_VERSION = "passive-muon-bf16-witness-v1"
KELLER_JORDAN_REVISION = "f98f1cacc0263b04290753e32be8d498c1efc806"
KELLER_JORDAN_EPSILON = 1e-7
KELLER_JORDAN_STEPS = 5


def _tensor_record(tensor: Tensor) -> dict[str, Any]:
    """Return JSON-safe values and, for BF16, the observable storage bits."""

    detached = tensor.detach()
    host = detached.cpu().contiguous()
    record: dict[str, Any] = {
        "shape": list(detached.shape),
        "dtype": str(detached.dtype),
        "device": str(detached.device),
        "flat_values": [float(value) for value in host.reshape(-1)],
    }
    if detached.dtype == torch.bfloat16:
        words = host.view(torch.uint16).reshape(-1).tolist()
        record["raw_uint16_hex"] = [f"0x{int(word):04x}" for word in words]
    return record


def _python_coefficients() -> tuple[float, float, float]:
    """Return the three Python floats written as literals in pinned ``muon.py``."""

    return (float(JORDAN_QUINTIC.a), float(JORDAN_QUINTIC.b), float(JORDAN_QUINTIC.c))


def _pinned_upstream_map(matrix: Tensor, *, eps: float, steps: int) -> Tensor:
    """Literal clean-room evaluation of the audited pinned-upstream operations."""

    out = matrix.bfloat16()
    transposed = matrix.shape[-2] > matrix.shape[-1]
    if transposed:
        out = out.mT
    out = out / (out.norm(dim=(-2, -1), keepdim=True) + eps)
    a, b, c = _python_coefficients()
    for _ in range(steps):
        gram = out @ out.mT
        # This mirrors upstream syntax intentionally. Python parses the second
        # term as ``(c * gram) @ gram``; that BF16 order is not interchangeable
        # with ``c * (gram @ gram)``.
        correction = b * gram + c * gram @ gram
        out = a * out + correction @ out
    if transposed:
        out = out.mT
    return out


def _trace_deployed_map(matrix: Tensor, *, eps: float, steps: int) -> dict[str, Any]:
    """Evaluate and trace the literal operations audited in pinned ``muon.py``."""

    low_precision = matrix.bfloat16()
    transposed = matrix.shape[-2] > matrix.shape[-1]
    oriented = low_precision.mT if transposed else low_precision
    norm = oriented.norm(dim=(-2, -1), keepdim=True)
    epsilon_bf16_reference = norm.new_tensor(eps)
    denominator = norm + eps
    denominator_increment = denominator - norm
    out = oriented / denominator
    a, b, c = _python_coefficients()

    stages: list[dict[str, Any]] = []
    for step in range(1, steps + 1):
        stage_input = out
        gram = stage_input @ stage_input.mT
        scaled_gram_b = b * gram
        scaled_gram_c = c * gram
        scaled_gram_c_matmul_gram = scaled_gram_c @ gram
        correction = scaled_gram_b + scaled_gram_c_matmul_gram
        correction_product = correction @ stage_input
        linear_product = a * stage_input
        oriented_output = linear_product + correction_product
        out = oriented_output

        stages.append(
            {
                "step": step,
                "orientation_already_fixed_before_loop": True,
                "input": _tensor_record(stage_input),
                "gram_matmul_result": _tensor_record(gram),
                "b_times_gram": _tensor_record(scaled_gram_b),
                "c_times_gram": _tensor_record(scaled_gram_c),
                "c_times_gram_then_matmul_gram": _tensor_record(scaled_gram_c_matmul_gram),
                "correction": _tensor_record(correction),
                "correction_matmul_result": _tensor_record(correction_product),
                "linear_product": _tensor_record(linear_product),
                "output": _tensor_record(out),
            }
        )

    if transposed:
        out = out.mT
    public_output = keller_jordan_map(matrix, steps=steps, eps=eps)
    independently_recomputed = _pinned_upstream_map(matrix, steps=steps, eps=eps)
    if not torch.equal(out, public_output) or not torch.equal(out, independently_recomputed):
        raise AssertionError("trace differs from public or independent pinned-order evaluation")

    return {
        "input_before_cast": _tensor_record(matrix),
        "input_after_bf16_cast": _tensor_record(low_precision),
        "input_after_one_time_orientation": _tensor_record(oriented),
        "one_time_transpose_applied": transposed,
        "frobenius_norm": _tensor_record(norm),
        "python_float_epsilon": {
            "decimal": repr(eps),
            "binary64_hex": eps.hex(),
        },
        "epsilon_bf16_cast_reference": _tensor_record(epsilon_bf16_reference),
        "normalization_denominator": _tensor_record(denominator),
        "returned_denominator_increment": _tensor_record(denominator_increment),
        "epsilon_absorbed_at_this_denominator": bool(torch.equal(norm, denominator)),
        "normalized_input": _tensor_record(oriented / denominator),
        "quintic_stages": stages,
        "output": _tensor_record(out),
        "matches_public_and_independent_literal_pinned_order": True,
    }


def _accumulation_probe(device: torch.device) -> dict[str, Any]:
    """Characterize observable matmul rounding without guessing kernel internals."""

    # The cast of the literal 0.3 is intentionally nontrivial.  On the current
    # CPU backend this dot product distinguishes a single final BF16 rounding
    # from multiplication and addition rounded to BF16 after every public op.
    left = torch.tensor([[-0.25, -8.0, 0.3, -16.0]], dtype=torch.bfloat16, device=device)
    right = torch.tensor([[3.0], [4.0], [7.0], [-2.0]], dtype=torch.bfloat16, device=device)
    returned = left @ right

    exact_sum = sum(
        Fraction.from_float(float(x)) * Fraction.from_float(float(y))
        for x, y in zip(
            left.detach().cpu().reshape(-1),
            right.detach().cpu().reshape(-1),
            strict=True,
        )
    )
    exact_then_bf16 = torch.tensor(float(exact_sum), dtype=torch.bfloat16, device=device)

    sequential = torch.zeros((), dtype=torch.bfloat16, device=device)
    sequential_terms: list[dict[str, Any]] = []
    for x, y in zip(left.reshape(-1), right.reshape(-1), strict=True):
        product = x * y
        sequential = sequential + product
        sequential_terms.append(
            {
                "rounded_product": _tensor_record(product),
                "rounded_running_sum": _tensor_record(sequential),
            }
        )

    returned_scalar = returned.reshape(())
    return {
        "purpose": (
            "Empirically compare the public BF16 matmul result with two observable "
            "rounding models on this backend."
        ),
        "fixed_source_literals": {
            "left": ["-0.25", "-8.0", "0.3", "-16.0"],
            "right": ["3.0", "4.0", "7.0", "-2.0"],
        },
        "left_after_bf16_cast": _tensor_record(left),
        "right_after_bf16_cast": _tensor_record(right),
        "public_matmul_result": _tensor_record(returned),
        "exact_real_dot_of_cast_values": {
            "fraction": str(exact_sum),
            "decimal": float(exact_sum),
        },
        "exact_sum_then_bf16_reference": _tensor_record(exact_then_bf16),
        "sequential_public_bf16_multiply_add_reference": {
            "terms": sequential_terms,
            "result": _tensor_record(sequential),
        },
        "public_matmul_matches_exact_sum_then_bf16": bool(
            torch.equal(returned_scalar, exact_then_bf16)
        ),
        "public_matmul_matches_sequential_public_bf16_ops": bool(
            torch.equal(returned_scalar, sequential)
        ),
        "interpretation_limit": (
            "These comparisons characterize returned values only. They do not reveal or "
            "certify the kernel's undocumented internal accumulator precision, instruction "
            "selection, or behavior on another backend."
        ),
    }


def _fraction_from_recorded_pair(
    left_output: Tensor,
    right_output: Tensor,
    left_input: Tensor,
    right_input: Tensor,
) -> tuple[Fraction, Fraction, Fraction]:
    left_output_values = left_output.detach().cpu().reshape(-1)
    right_output_values = right_output.detach().cpu().reshape(-1)
    left_input_values = left_input.detach().cpu().reshape(-1)
    right_input_values = right_input.detach().cpu().reshape(-1)
    output_deltas: list[Fraction] = []
    input_deltas: list[Fraction] = []
    for left_y, right_y, left_x, right_x in zip(
        left_output_values,
        right_output_values,
        left_input_values,
        right_input_values,
        strict=True,
    ):
        output_deltas.append(
            Fraction.from_float(float(left_y)) - Fraction.from_float(float(right_y))
        )
        input_deltas.append(
            Fraction.from_float(float(left_x)) - Fraction.from_float(float(right_x))
        )
    gap = sum(dy * dx for dy, dx in zip(output_deltas, input_deltas, strict=True))
    distance_squared = sum(dx**2 for dx in input_deltas)
    return gap, distance_squared, gap / distance_squared


def canonical_bf16_witness(
    *,
    device: str | torch.device = "cpu",
    eps: float = KELLER_JORDAN_EPSILON,
    steps: int = KELLER_JORDAN_STEPS,
) -> dict[str, Any]:
    """Evaluate the canonical pair in the pinned deployed BF16 operation order.

    The returned payload intentionally excludes host software, hardware, Git,
    and source hashes; the executable script adds those run-specific fields.
    """

    eps = float(eps)
    if eps <= 0:
        raise ValueError("eps must be strictly positive")
    if steps != KELLER_JORDAN_STEPS:
        raise ValueError("the canonical deployed witness is fixed at five steps")

    selected_device = torch.device(device)
    left = torch.diag(torch.tensor((3.0, 4.0), dtype=torch.float64, device=selected_device))
    right = torch.diag(torch.tensor((2.5, 3.0), dtype=torch.float64, device=selected_device))

    left_trace = _trace_deployed_map(left, eps=eps, steps=steps)
    right_trace = _trace_deployed_map(right, eps=eps, steps=steps)
    left_output = keller_jordan_map(left, eps=eps, steps=steps)
    right_output = keller_jordan_map(right, eps=eps, steps=steps)

    output_delta = left_output - right_output
    input_delta = left - right
    products = output_delta * input_delta
    gap = torch.sum(products)
    distance_squared = torch.sum(input_delta * input_delta)
    ratio = gap / distance_squared
    rho_required = torch.clamp(-ratio, min=0)
    exact_gap, exact_distance_squared, exact_ratio = _fraction_from_recorded_pair(
        left_output, right_output, left, right
    )
    runtime_gap_fraction = Fraction.from_float(float(gap))

    runtime_coefficients = _python_coefficients()
    coefficient_names = ("a", "b", "c")
    violates = bool(gap < 0)

    return {
        "schema_version": BF16_WITNESS_SCHEMA_VERSION,
        "claim_scope": {
            "evidence_kind": "backend_specific_executable_pairwise_witness",
            "supported_interpretation": (
                "A negative recorded gap is a concrete incremental-monotonicity violation for "
                "only the recorded backend, PyTorch build, dtype, and operation order."
            ),
            "observed_result_kind": (
                "negative_pair_witness" if violates else "nonnegative_pair_evaluation"
            ),
            "not_claimed": [
                "a real-arithmetic Jacobian theorem for the quantized operator",
                "the same returned values on every BF16 backend",
                "a universal claim about undocumented BF16 accumulator precision",
                "a global upper bound or a repair certificate",
            ],
        },
        "configuration": {
            "operator_domain_for_this_run": "two distinct 2x2 torch.float64 matrices",
            "pair_restriction": "positive_diagonal",
            "device": str(selected_device),
            "normalization": {
                "name": "current_frobenius_plus_epsilon_after_bf16_cast",
                "formula": ("X_bf16 / (X_bf16.norm(dim=(-2,-1), keepdim=True) + python_float_eps)"),
                "epsilon_configured": repr(eps),
                "epsilon": eps,
                "matches_pinned_upstream_epsilon": eps == KELLER_JORDAN_EPSILON,
                "norm_scope": "all matrix entries (Frobenius norm), dimensions retained",
            },
            "orthogonalizer": {
                "name": JORDAN_QUINTIC.name,
                "steps": steps,
                "recurrence": "A=X@X.T; B=b*A+(c*A)@A; X=a*X+B@X",
                "shape_rule": (
                    "transpose once before normalization iff rows > columns; transpose back once "
                    "after all stages"
                ),
                "coefficient_representation": (
                    "Pinned decimal literals are Python binary64 scalars. PyTorch applies wrapped "
                    "scalar promotion when each scalar operates on a BF16 tensor; BF16 cast "
                    "references and returned operation dtypes are recorded, without claiming "
                    "undocumented kernel internals."
                ),
                "coefficients": {
                    name: {
                        "source_decimal": source,
                        "source_fraction": str(fraction),
                        "python_binary64": {
                            "decimal": repr(runtime),
                            "hex": runtime.hex(),
                        },
                        "bf16_cast_reference": _tensor_record(left_output.new_tensor(runtime)),
                        "observed_scalar_times_bf16_dtype": str((runtime * left_output).dtype),
                    }
                    for name, source, fraction, runtime in zip(
                        coefficient_names,
                        (JORDAN_QUINTIC.a, JORDAN_QUINTIC.b, JORDAN_QUINTIC.c),
                        JORDAN_QUINTIC.fractions(),
                        runtime_coefficients,
                        strict=True,
                    )
                },
            },
            "operation_order": [
                "construct this artifact's canonical pair as torch.float64 matrices",
                "cast each complete input matrix to torch.bfloat16",
                "transpose once iff rows > columns",
                "compute X.norm(dim=(-2,-1), keepdim=True) on the oriented BF16 matrix",
                "add the Python binary64 epsilon scalar to that returned norm",
                "divide the oriented BF16 matrix by the returned denominator",
                "for each of five stages compute A = X @ X.T",
                "compute B = b*A + c*A @ A, parsed left-associatively as b*A + (c*A)@A",
                "compute X = a*X + B@X",
                "transpose back once after all stages iff the input was tall",
                "subtract the two BF16 outputs",
                "subtract the original torch.float64 inputs",
                "multiply the deltas using PyTorch dtype promotion and sum in returned dtype",
            ],
            "upstream": {
                "repository": "https://github.com/KellerJordan/Muon",
                "revision": KELLER_JORDAN_REVISION,
                "audited_file": "muon.py",
                "license": "MIT",
                "local_status": (
                    "Dedicated clean-room reproduction of the audited expressions and operation "
                    "order; no upstream training or distributed optimizer code is copied. It is "
                    "kept separate from generic polynomial helpers because BF16 grouping changes "
                    "returned values."
                ),
            },
        },
        "canonical_pair": {
            "left_input": _tensor_record(left),
            "right_input": _tensor_record(right),
            "left_execution_trace": left_trace,
            "right_execution_trace": right_trace,
            "left_output": _tensor_record(left_output),
            "right_output": _tensor_record(right_output),
            "measurement": {
                "definition": "<T(left)-T(right), left-right>_F",
                "output_difference": _tensor_record(output_delta),
                "input_difference": _tensor_record(input_delta),
                "elementwise_products": _tensor_record(products),
                "gap": _tensor_record(gap),
                "gap_decimal": float(gap),
                "gap_exact_fraction_from_individually_recorded_values": str(exact_gap),
                "runtime_gap_matches_exact_recorded_value_arithmetic": (
                    runtime_gap_fraction == exact_gap
                ),
                "distance_squared": _tensor_record(distance_squared),
                "distance_squared_exact_fraction": str(exact_distance_squared),
                "ratio": _tensor_record(ratio),
                "ratio_exact_fraction_from_individually_recorded_values": str(exact_ratio),
                "rho_required_for_this_pair": _tensor_record(rho_required),
                "rho_required_scope": "this one returned backend-specific pair only",
                "violates_incremental_monotonicity": violates,
            },
        },
        "observable_bf16_behavior": {
            "input_casts_and_normalization": {
                "left": {
                    "before": left_trace["input_before_cast"],
                    "after": left_trace["input_after_bf16_cast"],
                    "norm": left_trace["frobenius_norm"],
                    "python_float_epsilon": left_trace["python_float_epsilon"],
                    "epsilon_bf16_cast_reference": left_trace["epsilon_bf16_cast_reference"],
                    "denominator": left_trace["normalization_denominator"],
                    "returned_denominator_increment": left_trace["returned_denominator_increment"],
                    "epsilon_absorbed": left_trace["epsilon_absorbed_at_this_denominator"],
                },
                "right": {
                    "before": right_trace["input_before_cast"],
                    "after": right_trace["input_after_bf16_cast"],
                    "norm": right_trace["frobenius_norm"],
                    "python_float_epsilon": right_trace["python_float_epsilon"],
                    "epsilon_bf16_cast_reference": right_trace["epsilon_bf16_cast_reference"],
                    "denominator": right_trace["normalization_denominator"],
                    "returned_denominator_increment": right_trace["returned_denominator_increment"],
                    "epsilon_absorbed": right_trace["epsilon_absorbed_at_this_denominator"],
                },
            },
            "matmul_accumulation_probe": _accumulation_probe(selected_device),
        },
    }


__all__ = [
    "BF16_WITNESS_SCHEMA_VERSION",
    "KELLER_JORDAN_EPSILON",
    "KELLER_JORDAN_REVISION",
    "KELLER_JORDAN_STEPS",
    "canonical_bf16_witness",
]
