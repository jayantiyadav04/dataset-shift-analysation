"""Role separation, immutable freezes and deliberately injected leakage."""

import copy
import zipfile
from pathlib import Path

import numpy as np
import pytest

from shiftlab.data import parsers
from shiftlab.data.acquisition import fetch
from shiftlab.data.prepare import audit_dataset, prepare
from shiftlab.data.splits import (
    assign_roles,
    audit_splits,
    freeze_splits,
    validate_split,
)
from shiftlab.data.storage import DataError, read_json, seal
from shiftlab.data.training import load_training
from shiftlab.experiments.config import load_config
from shiftlab.features.preprocessing import preprocess_training


@pytest.fixture
def frozen_gas(tmp_path, monkeypatch):
    """Explicitly tiny offline archive; reference override is test-local only."""
    archive = tmp_path / "tiny-gas.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for batch in range(1, 11):
            rows = []
            for row in range(12):
                values = " ".join(f"{j}:{batch * 100 + row + j / 1000}" for j in range(1, 129))
                rows.append(f"{row % 6 + 1} {values}")
            z.writestr(f"Dataset/batch{batch}.dat", "\n".join(rows) + "\n")
    monkeypatch.setattr(parsers, "GAS_REFERENCE_ROWS", 120)
    root = tmp_path / "data"
    fetch(root, "gas", archive=archive)
    prepare(root, "gas")
    config = load_config(Path(__file__).resolve().parents[1] / "configs/smoke.yaml")
    result = freeze_splits(root, config)
    return root, config, root / "splits/1.0/gas_b01_b02", result


def test_freeze_has_explicit_hashed_ids_and_reuses_immutable_files(frozen_gas):
    root, config, directory, result = frozen_gas
    assert result["completed_tasks"] == 1
    saved = read_json(directory / "manifest.json")
    assert saved["sample_counts"] == {"source": 12, "target_adapt": 6, "target_eval": 6}
    assert len(saved["source_ids"]) == 12
    validate_split(saved)
    before = {p: p.stat().st_mtime_ns for p in directory.rglob("*") if p.is_file()}
    assert freeze_splits(root, config) == result
    assert before == {p: p.stat().st_mtime_ns for p in before}
    assert audit_splits(root, config)["status"] == "split_audit_passed"


def test_audit_missing_files_schema_and_injected_sample_leak(frozen_gas):
    root, config, directory, _ = frozen_gas
    saved = read_json(directory / "manifest.json")
    leak = copy.deepcopy(saved)
    leak["target_adapt_ids"][0] = leak["target_eval_ids"][0]
    leak = seal({key: value for key, value in leak.items() if key != "manifest_hash"})
    with pytest.raises(DataError, match="Leakage"):
        validate_split(leak)
    schema = copy.deepcopy(saved)
    schema["schema_version"] = 99
    schema = seal({key: value for key, value in schema.items() if key != "manifest_hash"})
    with pytest.raises(DataError, match="schema"):
        validate_split(schema)
    (directory / "training/target_adapt.npz").unlink()
    with pytest.raises(DataError, match="Missing or checksum"):
        audit_splits(root, config)
    (root / "processed/gas/features.npz").write_bytes(b"corrupted")
    with pytest.raises(DataError, match="checksum"):
        audit_dataset(root, "gas")


def metadata(sid, domain, pool, group, duplicate=None):
    return {
        "sample_id": sid,
        "domain": domain,
        "pool": pool,
        "group_id": group,
        "duplicate_id": duplicate or sid,
        "product_id": None,
    }


def test_har_whole_participants_and_injected_window_leak():
    rows = [metadata(f"source{i}", "train", "train", f"participant:{i:02d}") for i in range(1, 22)]
    for participant in (2, 4, 9, 10, 12, 13, 18, 20, 24):
        # Use a disjoint fixture source identity from the official target people.
        rows.extend(
            metadata(
                f"test{participant}-{window}",
                "test",
                "test",
                f"target-participant:{participant:02d}",
            )
            for window in range(3)
        )
    roles, _, source_domain, target_domain = assign_roles(
        "har", "har_official_group_transfer", rows
    )
    adapt_groups = {r["group_id"] for r in roles["target_adapt"]}
    eval_groups = {r["group_id"] for r in roles["target_eval"]}
    assert adapt_groups == {f"target-participant:{i:02d}" for i in (2, 4, 9, 10)}
    assert len(eval_groups) == 5 and not adapt_groups & eval_groups
    payload = {
        "schema_version": 1,
        "split_seed": 2026,
        "source_domain": source_domain,
        "target_domain": target_domain,
    }
    for role, records in roles.items():
        payload[f"{role}_ids"] = [r["sample_id"] for r in records]
        payload[f"{role}_group_ids"] = [r["group_id"] for r in records]
    validate_split(seal(payload))
    payload["target_eval_group_ids"][0] = payload["target_adapt_group_ids"][0]
    with pytest.raises(DataError, match="prohibited group"):
        validate_split(seal(payload))


def test_amazon_target_duplicates_removed_in_all_ordered_transfers():
    domains = ("books", "dvd", "electronics", "kitchen")
    rows = []
    for domain in domains:
        for pool in ("labelled", "unlabelled"):
            rows.extend(
                metadata(f"{domain}-{pool}-{i}", domain, pool, f"{domain}-{pool}-{i}")
                for i in range(4)
            )
    # A target duplicate of source body must be removed before partitioning.
    rows.append(
        metadata(
            "dvd-source-duplicate", "dvd", "unlabelled", "books-labelled-0", "books-labelled-0"
        )
    )
    for source in domains:
        for target in domains:
            if source == target:
                continue
            roles, excluded, _, _ = assign_roles("amazon", f"amazon_{source}_{target}", rows)
            assert all(len(records) >= 2 for records in roles.values())
            bodies = [
                {r["duplicate_id"] for r in roles[role]}
                for role in ("source", "target_adapt", "target_eval")
            ]
            assert (
                not bodies[0] & bodies[1]
                and not bodies[0] & bodies[2]
                and not bodies[1] & bodies[2]
            )
            if source == "books" and target == "dvd":
                assert excluded[0]["sample_id"] == "dvd-source-duplicate"


def test_training_never_opens_evaluation_and_ignores_its_changed_features_labels(
    frozen_gas, monkeypatch
):
    _, _, directory, _ = frozen_gas
    before = load_training(directory)
    pre_before, dense_before = preprocess_training(before)
    # The test changes synthetic held-out artifacts, never real study inputs.
    (directory / "evaluation/target_eval_labels.npy").write_bytes(b"unreadable labels")
    (directory / "evaluation/target_eval.npz").write_bytes(b"unreadable evaluation features")
    original_open = Path.open

    def guarded_open(path, *args, **kwargs):
        if "evaluation" in path.parts or "private" in path.parts:
            pytest.fail("training opened held-out data or the global label store")
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded_open)
    after = load_training(directory)
    pre_after, dense_after = preprocess_training(after)
    np.testing.assert_array_equal(dense_before.X_source, dense_after.X_source)
    np.testing.assert_array_equal(dense_before.X_target_adapt, dense_after.X_target_adapt)
    np.testing.assert_array_equal(pre_before.scaler_.mean_, pre_after.scaler_.mean_)
    assert pre_before.preprocessing_hash_ == pre_after.preprocessing_hash_
    assert not hasattr(after, "y_target") and not hasattr(after, "target_eval")


def test_training_rejects_a_role_path_redirected_to_evaluation(frozen_gas):
    _, _, directory, _ = frozen_gas
    saved = read_json(directory / "manifest.json")
    saved["artifacts"]["source"] = saved["artifacts"]["target_eval"]
    changed = seal({key: value for key, value in saved.items() if key != "manifest_hash"})
    import json

    (directory / "manifest.json").write_text(json.dumps(changed))
    with pytest.raises(DataError, match="permitted role"):
        load_training(directory)
