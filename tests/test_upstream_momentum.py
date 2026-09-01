from __future__ import annotations

import torch

from passive_muon.upstream_momentum import (
    PINNED_DEFAULT_BETA,
    PINNED_KELLER_JORDAN_MUON_REVISION,
    PINNED_MUON_PY_SHA256,
    ema_nesterov_state_and_signal,
    pinned_ema_nesterov_lerp_,
)


def _assert_float64_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, rtol=1e-15, atol=1e-15)


def test_pinned_revision_and_source_digest_are_locked() -> None:
    assert PINNED_KELLER_JORDAN_MUON_REVISION == ("f98f1cacc0263b04290753e32be8d498c1efc806")
    assert PINNED_MUON_PY_SHA256 == (
        "2479665a90124f62e4df557816665851ca317e42fcfda2af1da02c1f44ab5f3d"
    )
    assert PINNED_DEFAULT_BETA == 0.95


def test_literal_two_lerp_matches_algebra_for_distinct_nonzero_inputs() -> None:
    gradient = torch.tensor([[2.25, -1.5], [0.75, 3.125]], dtype=torch.float64)
    momentum = torch.tensor([[-0.5, 4.0], [1.25, -2.75]], dtype=torch.float64)
    original_gradient = gradient.clone()
    original_momentum = momentum.clone()

    expected_momentum, expected_signal = ema_nesterov_state_and_signal(
        original_gradient, original_momentum
    )
    returned_signal = pinned_ema_nesterov_lerp_(gradient, momentum)

    _assert_float64_close(momentum, expected_momentum)
    _assert_float64_close(gradient, expected_signal)
    assert returned_signal.data_ptr() == gradient.data_ptr()


def test_zero_buffer_default_beta_diagnostic() -> None:
    gradient = torch.tensor([[3.0, -2.0], [1.5, 4.25]], dtype=torch.float64)
    original_gradient = gradient.clone()
    momentum = torch.zeros_like(gradient)

    returned_signal = pinned_ema_nesterov_lerp_(gradient, momentum)

    _assert_float64_close(momentum, 0.05 * original_gradient)
    _assert_float64_close(returned_signal, 0.0975 * original_gradient)


def test_two_consecutive_updates_preserve_ema_state() -> None:
    momentum = torch.tensor([0.75, -1.25, 2.0], dtype=torch.float64)
    expected_momentum = momentum.clone()
    gradients = (
        torch.tensor([2.5, 1.0, -0.5], dtype=torch.float64),
        torch.tensor([-3.0, 0.25, 4.5], dtype=torch.float64),
    )

    for source_gradient in gradients:
        gradient = source_gradient.clone()
        expected_momentum, expected_signal = ema_nesterov_state_and_signal(
            source_gradient, expected_momentum
        )
        returned_signal = pinned_ema_nesterov_lerp_(gradient, momentum)
        _assert_float64_close(momentum, expected_momentum)
        _assert_float64_close(returned_signal, expected_signal)


def test_literal_shadow_mutates_gradient_to_nesterov_signal() -> None:
    gradient = torch.tensor([1.25, -3.5], dtype=torch.float64)
    momentum = torch.tensor([2.0, 0.75], dtype=torch.float64)
    original_gradient = gradient.clone()
    original_momentum = momentum.clone()
    _, expected_signal = ema_nesterov_state_and_signal(original_gradient, original_momentum)

    returned_signal = pinned_ema_nesterov_lerp_(gradient, momentum)

    assert not torch.equal(gradient, original_gradient)
    _assert_float64_close(gradient, expected_signal)
    assert returned_signal is gradient
