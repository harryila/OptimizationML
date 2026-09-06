from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch

from passive_muon.deployed import keller_jordan_map
from passive_muon.p22_real_gradient_shadow_trace import observe_post_aspect_candidate

ROOT = Path(__file__).resolve().parents[1]
ADAPTER_PATH = ROOT / "experiments" / "training" / "p22_observed_muon.py"
SPEC = importlib.util.spec_from_file_location("p22_observed_muon_test", ADAPTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("failed to load the P22 observation adapter")
ADAPTER = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = ADAPTER
SPEC.loader.exec_module(ADAPTER)


def test_post_aspect_observer_uses_supplied_stored_value_without_recomputing() -> None:
    signal = torch.tensor([[0.6, 0.0], [0.0, 0.8]], dtype=torch.float32)
    candidate = keller_jordan_map(signal).contiguous()
    signal_before = signal.clone()
    candidate_before = candidate.clone()

    record = observe_post_aspect_candidate(
        signal,
        candidate,
        optimizer_step=0,
        tokens_seen=128,
        seed=1337,
        parameter_name="transformer.h.0.attn.c_proj.weight",
        layer=0,
        role="attention_output",
        learning_rate=1.0 / 120.0,
        beta=0.95,
        weight_decay=0.0,
        accelerator_backend="mps",
    )

    assert torch.equal(signal, signal_before)
    assert torch.equal(candidate, candidate_before)
    assert record["optimizer"]["candidate_captured_after_accelerator_aspect"] is True
    assert record["optimizer"]["aspect_applied_by_observer"] is False
    assert record["storage"]["source_device"] == "mps"
    assert record["storage"]["aspect_candidate_dtype"] == "torch.bfloat16"
    assert record["shield"]["sector_certificate_scope"].startswith("P20 plus exact P22")


class _FakeOptimizer:
    def __init__(self, parameter: torch.nn.Parameter, module: ModuleType) -> None:
        self.param_groups = [{"params": [parameter], "use_muon": True}]
        self.parameter = parameter
        self.module = module

    def step(self, closure: object = None) -> None:
        del closure
        assert self.parameter.grad is not None
        momentum = torch.zeros_like(self.parameter)
        update = self.module.muon_update(self.parameter.grad, momentum)
        self.parameter.data.add_(update.float(), alpha=-0.01)


def test_adapter_returns_exact_candidate_and_gives_only_clones_to_hook() -> None:
    module = ModuleType("fake_pinned_muon")

    def muon_update(
        grad: torch.Tensor,
        momentum: torch.Tensor,
        beta: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> torch.Tensor:
        assert beta == 0.95 and ns_steps == 5 and nesterov
        momentum.lerp_(grad, 1.0 - beta)
        grad.lerp_(momentum, beta)
        return (2.0 * grad).to(torch.bfloat16)

    module.muon_update = muon_update
    parameter = torch.nn.Parameter(torch.ones((2, 2), dtype=torch.float32))
    parameter.grad = torch.full_like(parameter, 0.25)
    optimizer = _FakeOptimizer(parameter, module)
    metadata = ADAPTER.MuonParameterMetadata("matrix", 0, "projection")
    observed = ADAPTER.ObservedMuonStep(
        muon_module=module,
        optimizer=optimizer,
        parameter_metadata={id(parameter): metadata},
    )
    original_function = module.muon_update
    live_before = parameter.detach().clone()
    hook_calls: list[tuple[torch.Tensor, torch.Tensor]] = []

    def hook(
        received_metadata: object,
        signal: torch.Tensor,
        candidate: torch.Tensor,
    ) -> None:
        assert received_metadata == metadata
        hook_calls.append((signal.clone(), candidate.clone()))
        signal.zero_()
        candidate.zero_()

    observed.step(hook=hook)

    assert module.muon_update is original_function
    assert len(hook_calls) == 1
    expected_signal = torch.full((2, 2), 0.024375, dtype=torch.float32)
    assert torch.allclose(hook_calls[0][0], expected_signal)
    expected_parameter = live_before - 0.01 * (2.0 * expected_signal).to(torch.bfloat16).float()
    assert torch.equal(parameter.detach(), expected_parameter)


def test_adapter_restores_exact_function_after_hook_failure() -> None:
    module = ModuleType("fake_pinned_muon_failure")

    def muon_update(
        grad: torch.Tensor,
        momentum: torch.Tensor,
        beta: float = 0.95,
        ns_steps: int = 5,
        nesterov: bool = True,
    ) -> torch.Tensor:
        del momentum, beta, ns_steps, nesterov
        return grad.to(torch.bfloat16)

    module.muon_update = muon_update
    parameter = torch.nn.Parameter(torch.ones((1, 1), dtype=torch.float32))
    parameter.grad = torch.ones_like(parameter)
    optimizer = _FakeOptimizer(parameter, module)
    observed = ADAPTER.ObservedMuonStep(
        muon_module=module,
        optimizer=optimizer,
        parameter_metadata={
            id(parameter): ADAPTER.MuonParameterMetadata("matrix", 0, "projection")
        },
    )
    original_function = module.muon_update

    def fail(*_args: object) -> None:
        raise RuntimeError("observer failure")

    with pytest.raises(RuntimeError, match="observer failure"):
        observed.step(hook=fail)
    assert module.muon_update is original_function
