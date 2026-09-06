from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "scripts/verify_p30_local_bundle.py"
RUNBOOK = ROOT / "experiments/training/P30_UMASK_BOUND_CONTROL_SEAL_RUNBOOK.md"
SPEC = importlib.util.spec_from_file_location("p30_local_bundle_verifier", SOURCE)
assert SPEC is not None and SPEC.loader is not None
verifier = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = verifier
SPEC.loader.exec_module(verifier)

GIT = Path("/usr/bin/git")
P29_REF = "refs/heads/p29-permission-safe-bundle-localization-bridge"


def _git(repository: Path, *arguments: str) -> str:
    completed = subprocess.run(
        [str(GIT), "-C", str(repository), *arguments],
        check=True,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _commit(repository: Path, label: str) -> str:
    marker = repository / "marker.txt"
    marker.write_text(label + "\n", encoding="utf-8")
    _git(repository, "add", "marker.txt")
    _git(repository, "commit", "--quiet", "-m", label)
    return _git(repository, "rev-parse", "HEAD")


@pytest.fixture
def synthetic_repository(tmp_path: Path) -> dict[str, object]:
    repository = tmp_path / "source"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.name", "P30 Test")
    _git(repository, "config", "user.email", "p30@example.invalid")

    authority = _commit(repository, "authority")
    p26 = _commit(repository, "p26")
    p27 = _commit(repository, "p27")
    p28 = _commit(repository, "p28")
    p29 = _commit(repository, "p29")
    source = _commit(repository, "p30-source")
    runtime = _commit(repository, "p30-runtime")

    branch_targets = {
        "refs/heads/p29-permission-safe-bundle-localization-bridge": p29,
        "refs/heads/p28-bundle-complete-localization-bridge": p28,
        "refs/heads/p27-cuda-deleted-mapping-localization": p27,
    }
    for refname, object_id in branch_targets.items():
        _git(repository, "update-ref", refname, object_id)

    tag_targets = {
        "refs/tags/p29-orchestrator-source-mode-diagnostic": p29,
        "refs/tags/p28-control-parent-permission-diagnostic": p28,
        "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic": p27,
        "refs/tags/p26-permission-safe-acquisition-checkpoint": p26,
        "refs/tags/p26-attempt02-acquisition-source": authority,
    }
    for refname, object_id in tag_targets.items():
        short_name = refname.removeprefix("refs/tags/")
        _git(repository, "tag", "-a", "-m", short_name, short_name, object_id)

    fixed: dict[str, object] = {}
    for refname, commit in branch_targets.items():
        fixed[refname] = verifier.RefSpec(commit, "commit", commit)
    for refname, commit in tag_targets.items():
        fixed[refname] = verifier.RefSpec(_git(repository, "rev-parse", refname), "tag", commit)
    return {
        "repository": repository,
        "fixed": fixed,
        "p29": p29,
        "source": source,
        "source_tree": _git(repository, "rev-parse", f"{source}^{{tree}}"),
        "runtime": runtime,
        "runtime_tree": _git(repository, "rev-parse", f"{runtime}^{{tree}}"),
    }


def _make_bundle(
    fixture: dict[str, object],
    tmp_path: Path,
    *,
    p30_commit: str,
    extra_ref: str | None = None,
) -> Path:
    repository = fixture["repository"]
    assert isinstance(repository, Path)
    _git(repository, "update-ref", verifier.P30_REF, p30_commit)
    bundle = tmp_path / f"{p30_commit[:8]}.bundle"
    refs = [verifier.P30_REF, *fixture["fixed"].keys()]
    if extra_ref is not None:
        refs.append(extra_ref)
    _git(repository, "bundle", "create", str(bundle), *refs)
    return bundle


def _arguments(fixture: dict[str, object], bundle: Path, *, phase: str = "source") -> list[str]:
    if phase == "source":
        commit = fixture["source"]
        tree = fixture["source_tree"]
        parent = fixture["p29"]
    else:
        commit = fixture["runtime"]
        tree = fixture["runtime_tree"]
        parent = fixture["source"]
    return [
        "--phase",
        phase,
        "--bundle",
        str(bundle),
        "--git-executable",
        str(GIT),
        "--expected-p30-commit",
        str(commit),
        "--expected-p30-tree",
        str(tree),
        "--expected-p30-parent",
        str(parent),
        "--expected-verifier-source-sha256",
        hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    ]


def test_source_bundle_emits_one_canonical_binding(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["source"])
    )
    assert (
        verifier.main(
            _arguments(synthetic_repository, bundle), fixed_refs=synthetic_repository["fixed"]
        )
        == 0
    )
    captured = capsys.readouterr()
    expected = {
        "bundle_byte_count": bundle.stat().st_size,
        "bundle_sha256": hashlib.sha256(bundle.read_bytes()).hexdigest(),
        "p30_commit": synthetic_repository["source"],
        "p30_tree": synthetic_repository["source_tree"],
        "phase": "source",
        "schema": verifier.SCHEMA,
        "verified_ref_count": 9,
    }
    assert captured.err == ""
    assert captured.out == json.dumps(expected, sort_keys=True, separators=(",", ":")) + "\n"


def test_frozen_production_ref_map_is_exact() -> None:
    assert {
        "refs/heads/p29-permission-safe-bundle-localization-bridge": verifier.RefSpec(
            "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf",
            "commit",
            "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf",
        ),
        "refs/tags/p29-orchestrator-source-mode-diagnostic": verifier.RefSpec(
            "cf3c60b1d5c004b9e601a6afbf630358f80f1d79",
            "tag",
            "8a263722c7f8ce1d99b8796a3ccc0425dd1cefcf",
        ),
        "refs/heads/p28-bundle-complete-localization-bridge": verifier.RefSpec(
            "7274367f2b05fb3c9ed9f876a59be647703bbdc2",
            "commit",
            "7274367f2b05fb3c9ed9f876a59be647703bbdc2",
        ),
        "refs/tags/p28-control-parent-permission-diagnostic": verifier.RefSpec(
            "f3c81a2f14f853f369015a4f81b6695b52eec74e",
            "tag",
            "7274367f2b05fb3c9ed9f876a59be647703bbdc2",
        ),
        "refs/heads/p27-cuda-deleted-mapping-localization": verifier.RefSpec(
            "ec63550331925ded158e3f389e294e4d1f12db3a",
            "commit",
            "ec63550331925ded158e3f389e294e4d1f12db3a",
        ),
        "refs/tags/p27-cuda-deleted-mapping-localization-diagnostic": verifier.RefSpec(
            "c88b98eae9be2458abde45b05d3dccfe09c0c7ed",
            "tag",
            "ec63550331925ded158e3f389e294e4d1f12db3a",
        ),
        "refs/tags/p26-permission-safe-acquisition-checkpoint": verifier.RefSpec(
            "bacad707d3779bfa10957e18cb4c69b1a7f0cbce",
            "tag",
            "5429da23ff18888daa2312c530a4587780484d8b",
        ),
        "refs/tags/p26-attempt02-acquisition-source": verifier.RefSpec(
            "f35a7dca8f6bc39e9748e79b6712fab4a203396b",
            "tag",
            "185e444afc0b44ca0a09b1bde49a6b6fa3973355",
        ),
    } == verifier.FIXED_REFS


def test_runtime_bundle_locks_runtime_head_and_source_parent(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["runtime"])
    )
    assert (
        verifier.main(
            _arguments(synthetic_repository, bundle, phase="runtime-review"),
            fixed_refs=synthetic_repository["fixed"],
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["phase"] == "runtime-review"
    assert payload["p30_commit"] == synthetic_repository["runtime"]


@pytest.mark.parametrize("link_kind", ["symbolic", "hard"])
def test_linked_bundle_is_rejected(
    synthetic_repository: dict[str, object],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    link_kind: str,
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["source"])
    )
    actual = tmp_path / "actual.bundle"
    bundle.rename(actual)
    if link_kind == "symbolic":
        bundle.symlink_to(actual.name)
    else:
        os.link(actual, bundle)
    assert (
        verifier.main(
            _arguments(synthetic_repository, bundle), fixed_refs=synthetic_repository["fixed"]
        )
        == 1
    )
    assert "failed:" in capsys.readouterr().err


def test_symlinked_parent_component_is_rejected(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    bundle = _make_bundle(
        synthetic_repository,
        real_parent,
        p30_commit=str(synthetic_repository["source"]),
    )
    linked_parent = tmp_path / "linked-parent"
    linked_parent.symlink_to(real_parent.name)
    linked_bundle = linked_parent / bundle.name
    assert (
        verifier.main(
            _arguments(synthetic_repository, linked_bundle),
            fixed_refs=synthetic_repository["fixed"],
        )
        == 1
    )
    assert "cannot no-follow open" in capsys.readouterr().err


def test_prerequisite_is_rejected_before_git_consumption(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    fixed = synthetic_repository["fixed"]
    expected = {
        verifier.P30_REF: str(synthetic_repository["source"]),
        **{name: spec.object_id for name, spec in fixed.items()},
    }
    bundle = tmp_path / "prerequisite.bundle"
    lines = [b"# v2 git bundle\n"]
    lines.append((f"-{synthetic_repository['p29']} prerequisite\n").encode("ascii"))
    lines.extend(
        f"{object_id} {refname}\n".encode("ascii") for refname, object_id in expected.items()
    )
    lines.append(b"\nPACK")
    bundle.write_bytes(b"".join(lines))
    assert verifier.main(_arguments(synthetic_repository, bundle), fixed_refs=fixed) == 1
    assert "zero prerequisites" in capsys.readouterr().err


def test_extra_advertised_ref_is_rejected(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    repository = synthetic_repository["repository"]
    assert isinstance(repository, Path)
    _git(repository, "update-ref", "refs/heads/extra", str(synthetic_repository["p29"]))
    bundle = _make_bundle(
        synthetic_repository,
        tmp_path,
        p30_commit=str(synthetic_repository["source"]),
        extra_ref="refs/heads/extra",
    )
    assert (
        verifier.main(
            _arguments(synthetic_repository, bundle), fixed_refs=synthetic_repository["fixed"]
        )
        == 1
    )
    assert "exact nine-ref object map" in capsys.readouterr().err


def test_wrong_locked_object_is_rejected(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["source"])
    )
    altered = dict(synthetic_repository["fixed"])
    changed_ref = "refs/heads/p28-bundle-complete-localization-bridge"
    spec = altered[changed_ref]
    altered[changed_ref] = verifier.RefSpec(
        str(synthetic_repository["source"]), spec.object_type, spec.peeled_commit
    )
    assert verifier.main(_arguments(synthetic_repository, bundle), fixed_refs=altered) == 1
    assert "exact nine-ref object map" in capsys.readouterr().err


def test_annotated_tag_peel_is_checked_independently(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["source"])
    )
    altered = dict(synthetic_repository["fixed"])
    tag_ref = "refs/tags/p29-orchestrator-source-mode-diagnostic"
    spec = altered[tag_ref]
    altered[tag_ref] = verifier.RefSpec(spec.object_id, "tag", str(synthetic_repository["source"]))
    assert verifier.main(_arguments(synthetic_repository, bundle), fixed_refs=altered) == 1
    assert "peeled commit differs" in capsys.readouterr().err


def test_object_type_is_checked_independently(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["source"])
    )
    altered = dict(synthetic_repository["fixed"])
    tag_ref = "refs/tags/p29-orchestrator-source-mode-diagnostic"
    spec = altered[tag_ref]
    altered[tag_ref] = verifier.RefSpec(spec.object_id, "commit", spec.peeled_commit)
    assert verifier.main(_arguments(synthetic_repository, bundle), fixed_refs=altered) == 1
    assert "object type differs" in capsys.readouterr().err


def test_wrong_p30_tree_and_parent_are_rejected(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["runtime"])
    )
    arguments = _arguments(synthetic_repository, bundle, phase="runtime-review")
    arguments[arguments.index("--expected-p30-tree") + 1] = str(synthetic_repository["source_tree"])
    assert verifier.main(arguments, fixed_refs=synthetic_repository["fixed"]) == 1
    assert "tree differs" in capsys.readouterr().err

    arguments = _arguments(synthetic_repository, bundle, phase="runtime-review")
    arguments[arguments.index("--expected-p30-parent") + 1] = str(synthetic_repository["p29"])
    assert verifier.main(arguments, fixed_refs=synthetic_repository["fixed"]) == 1
    assert "exact direct child" in capsys.readouterr().err


def test_corrupt_pack_fails_full_git_bundle_verify(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["source"])
    )
    raw = bytearray(bundle.read_bytes())
    raw[-1] ^= 1
    bundle.write_bytes(raw)
    assert (
        verifier.main(
            _arguments(synthetic_repository, bundle), fixed_refs=synthetic_repository["fixed"]
        )
        == 1
    )
    assert "Git verification failed" in capsys.readouterr().err


def test_source_self_digest_is_mandatory(
    synthetic_repository: dict[str, object], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    bundle = _make_bundle(
        synthetic_repository, tmp_path, p30_commit=str(synthetic_repository["source"])
    )
    arguments = _arguments(synthetic_repository, bundle)
    arguments[-1] = "0" * 64
    assert verifier.main(arguments, fixed_refs=synthetic_repository["fixed"]) == 1
    assert "source digest differs" in capsys.readouterr().err


def test_runbook_verifies_each_bundle_before_its_upload_and_exports_binding() -> None:
    text = RUNBOOK.read_text(encoding="utf-8")
    source_create = text.index('p30_local_git bundle create "$P30_LOCAL_SOURCE_BUNDLE"')
    source_verify = text.index(
        '"$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_LOCAL_BUNDLE_VERIFIER"',
        source_create,
    )
    source_upload = text.index("02-source-bundle-upload", source_verify)
    assert source_create < source_verify < source_upload
    assert "--phase source \\\n" in text[source_verify:source_upload]
    assert "P30_EXPECTED_SOURCE_BUNDLE_SHA256" in text[source_verify:source_upload]
    assert "P30_EXPECTED_SOURCE_BUNDLE_BYTE_COUNT" in text[source_verify:source_upload]

    runtime_create = text.index('p30_local_git bundle create "$P30_LOCAL_RUNTIME_BUNDLE"')
    runtime_verify = text.index(
        '"$P30_CLIENT_PYTHON" -I -S "$P30_FROZEN_LOCAL_BUNDLE_VERIFIER"',
        runtime_create,
    )
    runtime_upload = text.index("13-runtime-review-bundle-upload", runtime_verify)
    assert runtime_create < runtime_verify < runtime_upload
    assert "--phase runtime-review \\\n" in text[runtime_verify:runtime_upload]
    assert "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_SHA256" in text[runtime_verify:runtime_upload]
    assert "P30_EXPECTED_RUNTIME_REVIEW_BUNDLE_BYTE_COUNT" in text[runtime_verify:runtime_upload]

    upload_helper = text.split("p30_o_excl_upload() {", 1)[1].split("\n}", 1)[0]
    assert '<"$source_path"' not in upload_helper
    assert '"$P30_FROZEN_AUTHENTICATED_TRANSFER"' in upload_helper
    assert "--expected-transfer-source-sha256" in upload_helper
    assert '"$P30_EXPECTED_AUTHENTICATED_TRANSFER_SOURCE_SHA256"' in upload_helper
    assert "--direction upload" in upload_helper
    assert '--source "$source_path"' in upload_helper
    assert '--staging-root "$P30_CLIENT_TRANSFER_ROOT"' in upload_helper
    assert '--expected-source-sha256 "$expected_sha256"' in upload_helper
    assert '--expected-source-byte-count "$expected_byte_count"' in upload_helper
    assert '-- "${p30_strict_ssh[@]}"' in upload_helper
