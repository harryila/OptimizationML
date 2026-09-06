"""Isolated observation adapter for the pinned KellerJordan/Muon optimizer.

The adapter does not reimplement ``muon_update`` and does not alter the value
returned to the optimizer.  It temporarily wraps the exact module-global
function called by ``SingleDeviceMuonWithAuxAdam.step`` and gives detached
clones of the stored post-Nesterov signal and returned post-aspect candidate
to a record-only hook.

This file is the P22 instrumentation patch.  Its byte hash is included in
every run manifest.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import ModuleType
from typing import Any

import torch
from torch import Tensor


@dataclass(frozen=True)
class MuonParameterMetadata:
    """Frozen identity for one observed hidden matrix."""

    name: str
    layer: int
    role: str


MuonCaptureHook = Callable[[MuonParameterMetadata, Tensor, Tensor], None]


class ObservedMuonStep:
    """Call an exact pinned optimizer step with an optional record-only hook."""

    def __init__(
        self,
        *,
        muon_module: ModuleType,
        optimizer: torch.optim.Optimizer,
        parameter_metadata: Mapping[int, MuonParameterMetadata],
    ) -> None:
        self._module = muon_module
        self._optimizer = optimizer
        self._metadata = dict(parameter_metadata)
        ordered: list[Tensor] = []
        for group in optimizer.param_groups:
            if bool(group.get("use_muon")):
                ordered.extend(group["params"])
        if not ordered:
            raise ValueError("the pinned optimizer has no Muon parameter group")
        if set(map(id, ordered)) != set(self._metadata):
            raise ValueError("Muon metadata must cover the optimizer group exactly")
        self._ordered_parameters = tuple(ordered)

    @property
    def parameter_count(self) -> int:
        return len(self._ordered_parameters)

    def step(
        self,
        *,
        hook: MuonCaptureHook | None,
        closure: Callable[[], Tensor] | None = None,
    ) -> Tensor | None:
        """Run one optimizer step and verify exact call/instrumentation coverage."""

        original = self._module.muon_update
        cursor = 0

        def observed_muon_update(
            grad: Tensor,
            momentum: Tensor,
            beta: float = 0.95,
            ns_steps: int = 5,
            nesterov: bool = True,
        ) -> Tensor:
            nonlocal cursor
            if cursor >= len(self._ordered_parameters):
                raise RuntimeError("pinned optimizer made an unexpected extra Muon call")
            parameter = self._ordered_parameters[cursor]
            if grad is not parameter.grad:
                raise RuntimeError("Muon call order no longer matches the frozen parameter group")
            update = original(
                grad,
                momentum,
                beta=beta,
                ns_steps=ns_steps,
                nesterov=nesterov,
            )
            if grad.dtype != torch.float32 or momentum.dtype != torch.float32:
                raise RuntimeError("P22 requires stored gradient and momentum in FP32")
            if update.dtype != torch.bfloat16 or update.device != grad.device:
                raise RuntimeError(
                    "pinned Muon must return a BF16 candidate on the selected accelerator"
                )
            if ns_steps != 5 or not nesterov:
                raise RuntimeError("P22 requires five Jordan stages and Nesterov enabled")
            if hook is not None:
                signal_copy = grad.detach().clone(memory_format=torch.contiguous_format)
                candidate_copy = update.detach().clone(memory_format=torch.contiguous_format)
                signal_version = grad._version
                candidate_version = update._version
                hook(self._metadata[id(parameter)], signal_copy, candidate_copy)
                if grad._version != signal_version or update._version != candidate_version:
                    raise RuntimeError("P22 hook changed a live optimizer tensor")
            cursor += 1
            return update

        self._module.muon_update = observed_muon_update
        try:
            loss: Any = self._optimizer.step(closure=closure)
        finally:
            self._module.muon_update = original
        if cursor != len(self._ordered_parameters):
            raise RuntimeError(
                f"observed {cursor} Muon calls, expected {len(self._ordered_parameters)}"
            )
        return loss


__all__ = ["MuonParameterMetadata", "ObservedMuonStep"]
