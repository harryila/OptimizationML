#!/usr/bin/env python3
"""One-shot finalizer for the P25 execution-source hash bindings.

Run this only after every listed P25 execution source is review-complete and
before the preregistration commit.  A finalized contract is immutable: this
tool verifies it but will not silently refresh a changed source hash.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "experiments/training/p25_cuda_diagnostic_contract.json"
RECONSTRUCTOR = ROOT / "scripts/reconstruct_p25_cuda_diagnostic_contract.py"
PENDING = "PENDING_SHA256"
PENDING_CONTRACT = "PENDING_CONTRACT_SHA256"
ALLOWED = {
    "corrected_executable_origin_diagnostic",
    "corrected_executable_origin_sanitizer",
    "historical_p24_native_ingester",
    "host_attempt_orchestrator",
}


class FinalizationError(RuntimeError):
    """Raised when finalization would hide drift or mutate the wrong record."""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    return parser.parse_args()


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _load(path: Path) -> dict[str, object]:
    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise FinalizationError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    try:
        payload = json.loads(path.read_bytes(), object_pairs_hook=reject_duplicates)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise FinalizationError(f"cannot read contract: {error}") from error
    if not isinstance(payload, dict):
        raise FinalizationError("contract root is not an object")
    return payload


def _atomic_replace(path: Path, rendered: bytes) -> None:
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as handle:
            handle.write(rendered)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def finalize(contract_path: Path) -> dict[str, object]:
    """Fill only declared pending hashes, or verify an already-final contract."""

    contract = _load(contract_path)
    finalization = contract.get("source_hash_finalization")
    sources = contract.get("execution_sources")
    if not isinstance(finalization, Mapping) or not isinstance(sources, Mapping):
        raise FinalizationError("contract omits source-finalization records")
    if set(finalization.get("allowed_records", [])) != ALLOWED:
        raise FinalizationError("allowed source-finalization inventory changed")

    observed: dict[str, str] = {}
    pending: list[str] = []
    for name, raw_record in sources.items():
        if not isinstance(name, str) or not isinstance(raw_record, Mapping):
            raise FinalizationError("execution-source record is malformed")
        path_text = raw_record.get("path")
        expected = raw_record.get("sha256")
        if not isinstance(path_text, str) or not isinstance(expected, str):
            raise FinalizationError(f"execution-source record is malformed: {name}")
        path = ROOT / path_text
        try:
            digest = _sha256(path.read_bytes())
        except OSError as error:
            raise FinalizationError(f"cannot hash {path_text}: {error}") from error
        observed[name] = digest
        if expected == PENDING:
            if name not in ALLOWED:
                raise FinalizationError(f"pending hash is forbidden for {name}")
            pending.append(name)
        elif expected != digest:
            raise FinalizationError(f"finalized execution source drifted: {name}")

    unknown_pending = {
        name
        for name, raw_record in sources.items()
        if isinstance(raw_record, Mapping) and raw_record.get("sha256") == PENDING
    } - ALLOWED
    if unknown_pending:
        raise FinalizationError(f"unknown pending records: {sorted(unknown_pending)}")

    changed = bool(pending)
    if changed:
        for name in pending:
            record = sources[name]
            assert isinstance(record, dict)
            record["sha256"] = observed[name]
        rendered = (json.dumps(contract, indent=2, ensure_ascii=True) + "\n").encode()
        _atomic_replace(contract_path, rendered)
    else:
        rendered = contract_path.read_bytes()

    contract_sha256 = _sha256(rendered)
    reconstructor_updated = False
    if contract_path.resolve() == DEFAULT_CONTRACT.resolve():
        reconstructor_text = RECONSTRUCTOR.read_text(encoding="utf-8")
        pending_literal = f'EXPECTED_CANONICAL_SHA256 = "{PENDING_CONTRACT}"'
        final_literal = f'EXPECTED_CANONICAL_SHA256 = "{contract_sha256}"'
        if pending_literal in reconstructor_text:
            updated = reconstructor_text.replace(pending_literal, final_literal, 1)
            if pending_literal in updated:
                raise FinalizationError("reconstructor contains multiple pending hash literals")
            _atomic_replace(RECONSTRUCTOR, updated.encode())
            reconstructor_updated = True
        elif final_literal not in reconstructor_text:
            raise FinalizationError("reconstructor pins a different contract SHA-256")

    return {
        "schema_version": "passive-muon-p25-contract-finalization-v1",
        "contract_path": str(contract_path),
        "contract_sha256": contract_sha256,
        "source_hashes": observed,
        "filled_records": sorted(pending),
        "changed": changed,
        "reconstructor_updated": reconstructor_updated,
        "ready_for_reviewed_static_commit": all(
            isinstance(record, Mapping) and record.get("sha256") != PENDING
            for record in sources.values()
        ),
    }


def main() -> None:
    result = finalize(_parse_args().contract.resolve())
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
