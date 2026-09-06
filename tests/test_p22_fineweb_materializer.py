from __future__ import annotations

import hashlib
import importlib.util
import json
import struct
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "experiments/training/materialize_p22_fineweb.py"
TRACE_SCRIPT = ROOT / "experiments/training/p22_nanogpt_shadow_trace.py"


def _load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {path}")
    module = importlib.util.module_from_spec(spec)
    # Dataclass annotation resolution expects the module to be registered.
    import sys

    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


MATERIALIZER = _load_module("p22_fineweb_materializer", SCRIPT)


class FakeGpt2Tokenizer:
    n_vocab = 50_257
    eot_token = 50_256

    def encode_ordinary(self, text: str) -> list[int]:
        if text == "document zero":
            return [11] * 100_000
        if text == "document one":
            return [22] * 100_000
        return [33] * len(text.encode("utf-8"))


def _response(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text(json.dumps({"rows": rows}, sort_keys=True), encoding="utf-8")
    return path


def _row(row_id: int, text: str, *, truncated: list[str] | None = None) -> dict[str, object]:
    return {
        "row_idx": row_id,
        "row": {"text": text, "id": f"row-{row_id}"},
        "truncated_cells": [] if truncated is None else truncated,
    }


def test_complete_rows_are_consumed_in_contiguous_increasing_order(tmp_path: Path) -> None:
    first = _response(tmp_path / "response-000.json", [_row(0, "a"), _row(1, "bb")])
    second = _response(tmp_path / "response-001.json", [_row(2, "ccc")])

    rows = list(MATERIALIZER.iter_complete_fineweb_rows((first, second)))

    assert [(row.row_id, row.text, row.response_index) for row in rows] == [
        (0, "a", 0),
        (1, "bb", 0),
        (2, "ccc", 1),
    ]


def test_truncated_or_noncontiguous_rows_fail_closed(tmp_path: Path) -> None:
    truncated = _response(
        tmp_path / "truncated.json",
        [_row(0, "not complete", truncated=["text"])],
    )
    with pytest.raises(MATERIALIZER.FineWebMaterializationError, match="truncated_cells"):
        list(MATERIALIZER.iter_complete_fineweb_rows((truncated,)))

    skipped = _response(tmp_path / "skipped.json", [_row(1, "starts too late")])
    with pytest.raises(MATERIALIZER.FineWebMaterializationError, match="expected 0"):
        list(MATERIALIZER.iter_complete_fineweb_rows((skipped,)))


def test_token_stream_has_exact_counts_eot_and_boundary_offsets() -> None:
    rows = (
        MATERIALIZER.FineWebRow(0, "document zero", 0),
        MATERIALIZER.FineWebRow(1, "document one", 1),
    )

    result = MATERIALIZER.materialize_token_stream(rows, FakeGpt2Tokenizer())

    assert len(result.train_tokens) == 131_072
    assert len(result.validation_tokens) == 32_768
    assert result.train_tokens[99_999] == 11
    assert result.train_tokens[100_000] == 50_256
    assert result.train_tokens[100_001] == 22
    assert result.train_boundary == MATERIALIZER.TokenBoundary(1, 31_071)
    assert result.validation_boundary == MATERIALIZER.TokenBoundary(1, 63_839)
    assert result.first_row_id == 0
    assert result.last_row_id_inclusive == 1
    assert result.row_count == 2
    assert result.consumed_response_indices == (0, 1)

    digest = hashlib.sha256(MATERIALIZER.CANONICAL_ROW_HASH_DOMAIN)
    for row_id, text in ((0, "document zero"), (1, "document one")):
        encoded = text.encode("utf-8")
        digest.update(struct.pack("<Q", row_id))
        digest.update(struct.pack("<Q", len(encoded)))
        digest.update(encoded)
    assert result.canonical_consumed_rows_sha256 == digest.hexdigest()


def test_insufficient_complete_rows_do_not_emit_a_short_pool() -> None:
    rows = (MATERIALIZER.FineWebRow(0, "tiny", 0),)
    with pytest.raises(MATERIALIZER.FineWebMaterializationError, match="are required"):
        MATERIALIZER.materialize_token_stream(rows, FakeGpt2Tokenizer())


def test_uint16_serialization_is_explicitly_little_endian() -> None:
    packed = MATERIALIZER._packed_uint16((1, 256, 50_256, 65_535))
    assert packed == b"\x01\x00\x00\x01P\xc4\xff\xff"
    assert struct.unpack("<4H", packed) == (1, 256, 50_256, 65_535)


def test_frozen_tokenizer_asset_verifier_rejects_wrong_bytes(tmp_path: Path) -> None:
    vocab = tmp_path / "vocab.bpe"
    encoder = tmp_path / "encoder.json"
    vocab.write_bytes(b"not the frozen vocabulary")
    encoder.write_bytes(b"{}")
    with pytest.raises(MATERIALIZER.FineWebMaterializationError, match="SHA-256 mismatch"):
        MATERIALIZER.verify_tokenizer_assets(vocab, encoder)


def test_mocked_end_to_end_manifest_is_accepted_by_frozen_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _response(
        tmp_path / "response.json",
        [_row(0, "document zero"), _row(1, "document one")],
    )
    vocab = tmp_path / "vocab.bpe"
    encoder = tmp_path / "encoder.json"
    vocab.write_bytes(b"mocked offline vocab fixture")
    encoder.write_bytes(b"mocked offline encoder fixture")
    output = tmp_path / "materialized"

    frozen_hashes = MATERIALIZER.TOKENIZER_ASSET_SHA256

    def mocked_assets(*_args: object, **_kwargs: object) -> dict[str, dict[str, object]]:
        return {
            "vocab.bpe": {
                "path": str(vocab.resolve()),
                "sha256": frozen_hashes["vocab.bpe"],
                "byte_count": vocab.stat().st_size,
            },
            "encoder.json": {
                "path": str(encoder.resolve()),
                "sha256": frozen_hashes["encoder.json"],
                "byte_count": encoder.stat().st_size,
            },
        }

    monkeypatch.setattr(MATERIALIZER, "verify_tokenizer_assets", mocked_assets)
    manifest = MATERIALIZER.materialize_fineweb(
        response_paths=(response,),
        vocab_bpe=vocab,
        encoder_json=encoder,
        output_directory=output,
        tokenizer=FakeGpt2Tokenizer(),
    )
    manifest_path = output / "p22_fineweb_manifest.json"

    assert manifest["status"] == "materialized"
    assert manifest["dataset"]["revision"] == MATERIALIZER.DATASET_REVISION
    assert (
        manifest["dataset"]["source_shards"][0]["sha256"]
        == hashlib.sha256(response.read_bytes()).hexdigest()
    )
    assert manifest["dataset"]["selection"]["train_boundary"] == {
        "row_id": 1,
        "token_offset_exclusive": 31_071,
    }
    assert manifest["dataset"]["selection"]["validation_boundary"] == {
        "row_id": 1,
        "token_offset_exclusive": 63_839,
    }
    assert manifest["tokenizer"]["files"]["vocab.bpe"]["sha256"] == frozen_hashes["vocab.bpe"]
    assert len(manifest["preprocessor"]["sha256"]) == 64
    assert (output / "train.bin").stat().st_size == 262_144
    assert (output / "val.bin").stat().st_size == 65_536
    assert (
        manifest["outputs"]["train.bin"]["sha256"]
        == hashlib.sha256((output / "train.bin").read_bytes()).hexdigest()
    )
    assert (
        manifest["outputs"]["val.bin"]["sha256"]
        == hashlib.sha256((output / "val.bin").read_bytes()).hexdigest()
    )

    trace = _load_module("p22_trace_manifest_validator", TRACE_SCRIPT)
    real_sha256_file = trace.sha256_file

    def mocked_asset_hash(path: Path) -> str:
        resolved = Path(path).resolve()
        if resolved == vocab.resolve():
            return frozen_hashes["vocab.bpe"]
        if resolved == encoder.resolve():
            return frozen_hashes["encoder.json"]
        return real_sha256_file(resolved)

    monkeypatch.setattr(trace, "sha256_file", mocked_asset_hash)
    report = trace.validate_fineweb_manifest(manifest_path)
    assert report["ready"], report["blockers"]
    assert report["blockers"] == []


def test_existing_outputs_are_not_overwritten_without_explicit_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    response = _response(
        tmp_path / "response.json",
        [_row(0, "document zero"), _row(1, "document one")],
    )
    vocab = tmp_path / "vocab.bpe"
    encoder = tmp_path / "encoder.json"
    vocab.write_bytes(b"v")
    encoder.write_bytes(b"e")
    output = tmp_path / "materialized"
    output.mkdir()
    (output / "train.bin").write_bytes(b"preserve me")

    monkeypatch.setattr(
        MATERIALIZER,
        "verify_tokenizer_assets",
        lambda *_args, **_kwargs: {
            "vocab.bpe": MATERIALIZER._file_record(vocab),
            "encoder.json": MATERIALIZER._file_record(encoder),
        },
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        MATERIALIZER.materialize_fineweb(
            response_paths=(response,),
            vocab_bpe=vocab,
            encoder_json=encoder,
            output_directory=output,
            tokenizer=FakeGpt2Tokenizer(),
        )
    assert (output / "train.bin").read_bytes() == b"preserve me"
