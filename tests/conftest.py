"""Branch-scope provenance tests for frozen research checkpoints."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
P10_BRANCH = "p10-finite-precision-outer-loop"
P10_FRESH_PROVENANCE_TEST = (
    "tests/test_outer_loop_roundoff_reconstruction.py::"
    "test_p10_reconstruction_matches_fresh_generator"
)


def _git_branch() -> str:
    completed = subprocess.run(
        ["git", "branch", "--show-current"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Run a frozen branch's fresh-provenance comparison only on that branch.

    Downstream branches still run P10's independent reconstruction and its
    comparison against the committed canonical artifact.  Only regenerating a
    *fresh* artifact with P10 branch provenance is inapplicable elsewhere.
    """

    if _git_branch() == P10_BRANCH:
        return

    marker = pytest.mark.skip(
        reason=(
            "P10 fresh provenance is branch-locked; downstream branches still "
            "replay the committed P10 canonical artifact exactly"
        )
    )
    for item in items:
        if item.nodeid == P10_FRESH_PROVENANCE_TEST:
            item.add_marker(marker)
