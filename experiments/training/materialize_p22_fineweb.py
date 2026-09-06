#!/usr/bin/env python3
"""Deterministically materialize the frozen P22 FineWeb token pools.

The input is one or more locally captured Hugging Face datasets-server JSON
responses.  Each response must contain complete, untruncated rows.  This tool
does not silently fetch mutable network state: acquisition is a separate step
and every response byte string is SHA-256 recorded in the output manifest.

Rows are consumed from row zero in contiguous increasing order.  Each full
``text`` cell is tokenized with the exact GPT-2 assets under
``tiktoken==0.14.0`` and one EOT token is appended.  The first 131072 tokens
become ``train.bin`` and the next 32768 become ``val.bin`` as little-endian
uint16 values.  A boundary may fall inside a document, but the source text
cell is always acquired and tokenized in full and its in-row exclusive offset
is recorded.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import struct
import tempfile
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

SCHEMA_VERSION: Final = "passive-muon-p22-fineweb-materialization-v1"
DATASET_REPOSITORY: Final = "HuggingFaceFW/fineweb"
DATASET_CONFIG: Final = "sample-10BT"
DATASET_SPLIT: Final = "train"
DATASET_REVISION: Final = "9bb295ddab0e05d785b879661af7260fed5140fc"
DATASET_SERVER_ENDPOINT: Final = "https://datasets-server.huggingface.co/rows"
ROW_ORDERING: Final = "Increasing row index at the pinned revision; complete untruncated text only"

TIKTOKEN_VERSION: Final = "0.14.0"
TOKENIZER_ENCODING: Final = "gpt2"
TOKENIZER_VOCAB_SIZE: Final = 50_257
EOT_TOKEN_ID: Final = 50_256
TOKENIZER_ASSET_SHA256: Final = {
    "vocab.bpe": "1ce1664773c50f3e0cc8842619a93edc4624525b728b188a9e0be33b7726adc5",
    "encoder.json": "196139668be63f3b5d6574427317ae82f612a97c5d1cdaf36ed2256dbf636783",
}

TRAIN_TOKEN_COUNT: Final = 131_072
VALIDATION_TOKEN_COUNT: Final = 32_768
TOTAL_TOKEN_COUNT: Final = TRAIN_TOKEN_COUNT + VALIDATION_TOKEN_COUNT
UINT16_MAX: Final = 2**16 - 1
CANONICAL_ROW_HASH_DOMAIN: Final = b"passive-muon-p22-fineweb-consumed-rows-v1\0"


class FineWebMaterializationError(ValueError):
    """Raised when source rows or pinned assets violate the frozen contract."""


class Tokenizer(Protocol):
    """Minimal interface used by the deterministic token-stream builder."""

    @property
    def n_vocab(self) -> int: ...

    @property
    def eot_token(self) -> int: ...

    def encode_ordinary(self, text: str) -> list[int]: ...


@dataclass(frozen=True)
class FineWebRow:
    """One verified complete row and its captured-response provenance."""

    row_id: int
    text: str
    response_index: int


@dataclass(frozen=True)
class TokenBoundary:
    """Exclusive stream boundary located within one fully tokenized row."""

    row_id: int
    token_offset_exclusive: int


@dataclass(frozen=True)
class TokenStreamMaterialization:
    """Exact output token pools and row-selection provenance."""

    train_tokens: tuple[int, ...]
    validation_tokens: tuple[int, ...]
    first_row_id: int
    last_row_id_inclusive: int
    row_count: int
    train_boundary: TokenBoundary
    validation_boundary: TokenBoundary
    canonical_consumed_rows_sha256: str
    consumed_response_indices: tuple[int, ...]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(2**20):
            digest.update(block)
    return digest.hexdigest()


def _file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    resolved = path.resolve()
    rendered = str(resolved)
    if relative_to is not None:
        with contextlib.suppress(ValueError):
            rendered = str(resolved.relative_to(relative_to.resolve()))
    return {
        "path": rendered,
        "sha256": _sha256(resolved),
        "byte_count": resolved.stat().st_size,
    }


def verify_tokenizer_assets(vocab_bpe: Path, encoder_json: Path) -> dict[str, dict[str, object]]:
    """Verify and describe the two frozen GPT-2 tokenizer assets."""

    paths = {"vocab.bpe": vocab_bpe.resolve(), "encoder.json": encoder_json.resolve()}
    records: dict[str, dict[str, object]] = {}
    for name, path in paths.items():
        if not path.is_file():
            raise FineWebMaterializationError(f"tokenizer asset is missing: {path}")
        record = _file_record(path)
        expected = TOKENIZER_ASSET_SHA256[name]
        if record["sha256"] != expected:
            raise FineWebMaterializationError(
                f"tokenizer asset {name} SHA-256 mismatch: {record['sha256']} != {expected}"
            )
        records[name] = record
    return records


def load_frozen_gpt2_tokenizer(vocab_bpe: Path, encoder_json: Path) -> Tokenizer:
    """Build GPT-2 directly from the verified local assets, without fetching."""

    verify_tokenizer_assets(vocab_bpe, encoder_json)
    installed = importlib.metadata.version("tiktoken")
    if installed != TIKTOKEN_VERSION:
        raise FineWebMaterializationError(
            f"tiktoken version is {installed}, expected exactly {TIKTOKEN_VERSION}"
        )

    import tiktoken
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks
    from tiktoken_ext.openai_public import ENDOFTEXT, r50k_pat_str

    mergeable_ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=str(vocab_bpe.resolve()),
        encoder_json_file=str(encoder_json.resolve()),
        vocab_bpe_hash=TOKENIZER_ASSET_SHA256["vocab.bpe"],
        encoder_json_hash=TOKENIZER_ASSET_SHA256["encoder.json"],
    )
    tokenizer = tiktoken.Encoding(
        name="p22-frozen-gpt2",
        explicit_n_vocab=TOKENIZER_VOCAB_SIZE,
        pat_str=r50k_pat_str,
        mergeable_ranks=mergeable_ranks,
        special_tokens={ENDOFTEXT: EOT_TOKEN_ID},
    )
    if tokenizer.n_vocab != TOKENIZER_VOCAB_SIZE or tokenizer.eot_token != EOT_TOKEN_ID:
        raise FineWebMaterializationError("constructed tokenizer does not match frozen GPT-2")
    return tokenizer


def _load_response(path: Path, response_index: int) -> list[FineWebRow]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise FineWebMaterializationError(f"cannot read source response {path}: {error}") from error
    rows = payload.get("rows") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise FineWebMaterializationError(f"source response {path} has no nonempty rows list")

    parsed: list[FineWebRow] = []
    for position, entry in enumerate(rows):
        if not isinstance(entry, dict):
            raise FineWebMaterializationError(
                f"source response {path} row {position} is not a mapping"
            )
        truncated = entry.get("truncated_cells")
        if truncated != []:
            raise FineWebMaterializationError(
                f"source response {path} row {position} has truncated_cells={truncated!r}"
            )
        row_id = entry.get("row_idx")
        row = entry.get("row")
        if not isinstance(row_id, int) or isinstance(row_id, bool) or row_id < 0:
            raise FineWebMaterializationError(
                f"source response {path} row {position} has invalid row_idx"
            )
        if not isinstance(row, dict) or not isinstance(row.get("text"), str):
            raise FineWebMaterializationError(
                f"source response {path} row {position} lacks a complete text cell"
            )
        parsed.append(FineWebRow(row_id=row_id, text=row["text"], response_index=response_index))
    return parsed


def iter_complete_fineweb_rows(response_paths: Sequence[Path]) -> Iterator[FineWebRow]:
    """Yield contiguous complete rows from captured responses in supplied order."""

    if not response_paths:
        raise FineWebMaterializationError("at least one source response is required")
    expected_row_id = 0
    for response_index, raw_path in enumerate(response_paths):
        path = raw_path.resolve()
        if not path.is_file():
            raise FineWebMaterializationError(f"source response is missing: {path}")
        for row in _load_response(path, response_index):
            if row.row_id != expected_row_id:
                raise FineWebMaterializationError(
                    "FineWeb rows must be contiguous from zero in increasing order: "
                    f"expected {expected_row_id}, found {row.row_id}"
                )
            yield row
            expected_row_id += 1


def _update_canonical_row_hash(digest: Any, row: FineWebRow) -> None:
    text = row.text.encode("utf-8")
    digest.update(struct.pack("<Q", row.row_id))
    digest.update(struct.pack("<Q", len(text)))
    digest.update(text)


def materialize_token_stream(
    rows: Iterable[FineWebRow],
    tokenizer: Tokenizer,
) -> TokenStreamMaterialization:
    """Tokenize complete rows and split the exact frozen train/validation pools."""

    if tokenizer.n_vocab != TOKENIZER_VOCAB_SIZE or tokenizer.eot_token != EOT_TOKEN_ID:
        raise FineWebMaterializationError("tokenizer does not expose the frozen GPT-2 vocabulary")

    digest = hashlib.sha256(CANONICAL_ROW_HASH_DOMAIN)
    stream: list[int] = []
    first_row: int | None = None
    last_row: int | None = None
    row_count = 0
    train_boundary: TokenBoundary | None = None
    validation_boundary: TokenBoundary | None = None
    response_indices: list[int] = []

    for expected_row_id, row in enumerate(rows):
        if row.row_id != expected_row_id:
            raise FineWebMaterializationError(
                f"row stream is not contiguous: expected {expected_row_id}, found {row.row_id}"
            )
        if not isinstance(row.text, str):
            raise FineWebMaterializationError(f"row {row.row_id} text is not a string")

        token_ids = list(tokenizer.encode_ordinary(row.text))
        token_ids.append(EOT_TOKEN_ID)
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0 or token > UINT16_MAX
            for token in token_ids
        ):
            raise FineWebMaterializationError(f"row {row.row_id} emitted a non-uint16 token")

        before = len(stream)
        after = before + len(token_ids)
        _update_canonical_row_hash(digest, row)
        stream.extend(token_ids)
        first_row = row.row_id if first_row is None else first_row
        last_row = row.row_id
        row_count += 1
        if not response_indices or response_indices[-1] != row.response_index:
            response_indices.append(row.response_index)

        if train_boundary is None and after >= TRAIN_TOKEN_COUNT:
            train_boundary = TokenBoundary(row.row_id, TRAIN_TOKEN_COUNT - before)
        if after >= TOTAL_TOKEN_COUNT:
            validation_boundary = TokenBoundary(row.row_id, TOTAL_TOKEN_COUNT - before)
            break

    if len(stream) < TOTAL_TOKEN_COUNT or train_boundary is None or validation_boundary is None:
        raise FineWebMaterializationError(
            f"source ended after {len(stream)} tokens; {TOTAL_TOKEN_COUNT} are required"
        )
    if first_row is None or last_row is None:
        raise FineWebMaterializationError("no FineWeb rows were consumed")

    return TokenStreamMaterialization(
        train_tokens=tuple(stream[:TRAIN_TOKEN_COUNT]),
        validation_tokens=tuple(stream[TRAIN_TOKEN_COUNT:TOTAL_TOKEN_COUNT]),
        first_row_id=first_row,
        last_row_id_inclusive=last_row,
        row_count=row_count,
        train_boundary=train_boundary,
        validation_boundary=validation_boundary,
        canonical_consumed_rows_sha256=digest.hexdigest(),
        consumed_response_indices=tuple(response_indices),
    )


def _packed_uint16(tokens: Sequence[int]) -> bytes:
    if any(token < 0 or token > UINT16_MAX for token in tokens):
        raise FineWebMaterializationError("token output exceeds uint16")
    return struct.pack(f"<{len(tokens)}H", *tokens)


def _atomic_write(path: Path, payload: bytes, *, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite existing materialization output: {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def build_manifest(
    *,
    response_paths: Sequence[Path],
    tokenizer_assets: Mapping[str, Mapping[str, object]],
    materialization: TokenStreamMaterialization,
    manifest_root: Path,
    train_path: Path,
    validation_path: Path,
) -> dict[str, object]:
    """Build the validator-compatible frozen materialization manifest."""

    consumed_responses = [
        response_paths[index].resolve() for index in materialization.consumed_response_indices
    ]
    source_records = []
    for index, path in zip(
        materialization.consumed_response_indices, consumed_responses, strict=True
    ):
        record = _file_record(path, relative_to=manifest_root)
        record["response_index"] = index
        record["source_kind"] = "captured datasets-server response"
        record["truncated_cells"] = []
        rows = _load_response(path, index)
        record["row_start"] = rows[0].row_id
        record["row_end_inclusive"] = rows[-1].row_id
        record["row_count"] = len(rows)
        record["request_url"] = (
            f"{DATASET_SERVER_ENDPOINT}?dataset=HuggingFaceFW%2Ffineweb"
            f"&config={DATASET_CONFIG}&split={DATASET_SPLIT}"
            f"&offset={rows[0].row_id}&length={len(rows)}&revision={DATASET_REVISION}"
        )
        source_records.append(record)

    preprocessor_path = Path(__file__).resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "materialized",
        "dataset": {
            "repository": DATASET_REPOSITORY,
            "config": DATASET_CONFIG,
            "split": DATASET_SPLIT,
            "revision": DATASET_REVISION,
            "source_shards": source_records,
            "selection": {
                "ordering": ROW_ORDERING,
                "first_row_id": materialization.first_row_id,
                "last_row_id_inclusive": materialization.last_row_id_inclusive,
                "row_count": materialization.row_count,
                "canonical_consumed_rows_sha256": (materialization.canonical_consumed_rows_sha256),
                "canonical_framing": (
                    "sha256(domain || repeated le64(row_id) || le64(utf8_bytes) || utf8(text))"
                ),
                "train_boundary": {
                    "row_id": materialization.train_boundary.row_id,
                    "token_offset_exclusive": (
                        materialization.train_boundary.token_offset_exclusive
                    ),
                },
                "validation_boundary": {
                    "row_id": materialization.validation_boundary.row_id,
                    "token_offset_exclusive": (
                        materialization.validation_boundary.token_offset_exclusive
                    ),
                },
            },
        },
        "tokenizer": {
            "package": "tiktoken",
            "version": TIKTOKEN_VERSION,
            "encoding": TOKENIZER_ENCODING,
            "vocab_size": TOKENIZER_VOCAB_SIZE,
            "end_of_text_token_id": EOT_TOKEN_ID,
            "append_end_of_text_after_each_document": True,
            "files": {name: dict(record) for name, record in tokenizer_assets.items()},
        },
        "preprocessor": {
            **_file_record(preprocessor_path),
            "normalization": "none",
            "document_field": "text",
        },
        "outputs": {
            "train.bin": {
                **_file_record(train_path, relative_to=manifest_root),
                "token_count": TRAIN_TOKEN_COUNT,
                "dtype": "little-endian uint16",
            },
            "val.bin": {
                **_file_record(validation_path, relative_to=manifest_root),
                "token_count": VALIDATION_TOKEN_COUNT,
                "dtype": "little-endian uint16",
            },
        },
    }


def materialize_fineweb(
    *,
    response_paths: Sequence[Path],
    vocab_bpe: Path,
    encoder_json: Path,
    output_directory: Path,
    manifest_path: Path | None = None,
    overwrite: bool = False,
    tokenizer: Tokenizer | None = None,
) -> dict[str, object]:
    """Materialize exact token files and return their complete manifest."""

    response_paths = tuple(path.resolve() for path in response_paths)
    tokenizer_assets = verify_tokenizer_assets(vocab_bpe, encoder_json)
    selected_tokenizer = tokenizer or load_frozen_gpt2_tokenizer(vocab_bpe, encoder_json)
    materialization = materialize_token_stream(
        iter_complete_fineweb_rows(response_paths), selected_tokenizer
    )

    output_directory = output_directory.resolve()
    selected_manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else output_directory / "p22_fineweb_manifest.json"
    )
    train_path = output_directory / "train.bin"
    validation_path = output_directory / "val.bin"
    if len({train_path, validation_path, selected_manifest_path}) != 3:
        raise FineWebMaterializationError(
            "train, validation, and manifest outputs must use three distinct paths"
        )
    protected_inputs = {
        *response_paths,
        vocab_bpe.resolve(),
        encoder_json.resolve(),
        Path(__file__).resolve(),
    }
    overlapping = protected_inputs.intersection(
        {train_path, validation_path, selected_manifest_path}
    )
    if overlapping:
        raise FineWebMaterializationError(
            f"materialization outputs may not overwrite provenance inputs: {sorted(overlapping)}"
        )
    for path in (train_path, validation_path, selected_manifest_path):
        if path.exists() and not overwrite:
            raise FileExistsError(f"refusing to overwrite existing materialization output: {path}")

    _atomic_write(train_path, _packed_uint16(materialization.train_tokens), overwrite=overwrite)
    _atomic_write(
        validation_path,
        _packed_uint16(materialization.validation_tokens),
        overwrite=overwrite,
    )
    manifest = build_manifest(
        response_paths=response_paths,
        tokenizer_assets=tokenizer_assets,
        materialization=materialization,
        manifest_root=selected_manifest_path.parent,
        train_path=train_path,
        validation_path=validation_path,
    )
    rendered = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _atomic_write(selected_manifest_path, rendered, overwrite=overwrite)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-response", action="append", required=True, type=Path)
    parser.add_argument("--vocab-bpe", required=True, type=Path)
    parser.add_argument("--encoder-json", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = materialize_fineweb(
        response_paths=args.source_response,
        vocab_bpe=args.vocab_bpe,
        encoder_json=args.encoder_json,
        output_directory=args.output_directory,
        manifest_path=args.manifest,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
