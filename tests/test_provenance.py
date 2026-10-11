"""Independent hash fixtures and identities that must distinguish paired runs."""

import hashlib
from dataclasses import replace

import pytest

from shiftlab.contracts import MethodName, ModelName
from shiftlab.experiments.provenance import (
    RunKey,
    RunStatus,
    canonical_json,
    content_hash,
    file_sha256,
    transition,
)


def key() -> RunKey:
    # Explicit synthetic digests: not claimed to identify an actual data archive.
    return RunKey(
        dataset_hash="a" * 64,
        task_id="synthetic_transfer",
        split_hash="b" * 64,
        model=ModelName.LR,
        selected_source_parameter_hash="c" * 64,
        preprocessing_hash="d" * 64,
        method=MethodName.SOURCE_ONLY,
        adaptation_parameter_hash="e" * 64,
        seed=17,
        protocol_version="1.0",
        code_version="f" * 40,
        dependency_lock_hash="1" * 64,
        config_hash="2" * 64,
    )


def test_hash_matches_independent_exact_json_and_file_fixture(tmp_path):
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    assert content_hash({"b": 2, "a": 1}) == hashlib.sha256(b'{"a":1,"b":2}').hexdigest()
    path = tmp_path / "fixture.txt"
    path.write_bytes(b"abc")
    assert file_sha256(path) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    with pytest.raises(ValueError):
        canonical_json({"score": float("nan")})


@pytest.mark.parametrize(
    "field,value",
    [
        ("dataset_hash", "3" * 64),
        ("split_hash", "3" * 64),
        ("model", ModelName.RF),
        ("selected_source_parameter_hash", "3" * 64),
        ("preprocessing_hash", "3" * 64),
        ("method", MethodName.CORAL),
        ("adaptation_parameter_hash", "3" * 64),
        ("seed", 29),
        ("protocol_version", "1.1"),
        ("code_version", "3" * 40),
        ("code_snapshot_hash", "3" * 64),
        ("dependency_lock_hash", "3" * 64),
        ("config_hash", "3" * 64),
        ("task_id", "different_transfer"),
    ],
)
def test_every_scientific_identity_component_changes_key(field, value):
    original = key()
    assert replace(original, **{field: value}).run_id != original.run_id
    assert replace(original).run_id == original.run_id


def test_invalid_provenance_cannot_be_presented_as_a_run():
    for field, value in (
        ("dataset_hash", "unknown"),
        ("code_version", "uncommitted"),
        ("code_snapshot_hash", "invalid"),
        ("seed", True),
    ):
        with pytest.raises(ValueError):
            replace(key(), **{field: value})


def test_state_cannot_jump_to_results_or_overwrite_terminal_records():
    with pytest.raises(ValueError):
        transition(RunStatus.PENDING, RunStatus.EVALUATED)
    assert transition(RunStatus.RUNNING, RunStatus.FITTED) == RunStatus.FITTED
    assert transition(RunStatus.FITTED, RunStatus.EVALUATED) == RunStatus.EVALUATED
    for state in (RunStatus.FAILED, RunStatus.CANCELLED, RunStatus.EVALUATED):
        with pytest.raises(ValueError):
            transition(state, RunStatus.RUNNING)
