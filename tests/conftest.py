"""Paths and raw config fixtures; no network or public study data."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from shiftlab.data.storage import seal, write_json
from shiftlab.experiments.provenance import file_sha256

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def primary_dict():
    return yaml.safe_load((ROOT / "configs" / "study.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def smoke_dict():
    return yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def baseline_split(tmp_path):
    directory = tmp_path / "data/splits/1.0/gas_b01_b02"
    rng = np.random.default_rng(17)
    manifest = {
        "schema_version": 1,
        "split_seed": 2026,
        "protocol_version": "1.0",
        "task_id": "gas_b01_b02",
        "source_domain": "batch01",
        "target_domain": "batch02",
        "dataset_id": "gas",
        "dataset_hash": "4" * 64,
        "source_product_ids": [None] * 60,
    }
    artifacts, counts = {}, {}
    for role, count in (("source", 60), ("target_adapt", 2), ("target_eval", 4)):
        ids = [f"{role}:{i}" for i in range(count)]
        values = rng.normal(size=(count, 128))
        values[:, 0] = np.arange(count) % 2 * 5
        section = "evaluation" if role == "target_eval" else "training"
        name = f"{section}/{role}.npz"
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, X=values, ids=np.array(ids))
        artifacts[role] = name
        manifest[f"{role}_ids"] = ids
        manifest[f"{role}_group_ids"] = ids
        counts[role] = count
    artifacts["source_labels"] = "training/source_labels.npy"
    artifacts["target_eval_labels"] = "evaluation/target_eval_labels.npy"
    np.save(directory / artifacts["source_labels"], np.arange(60) % 2, allow_pickle=False)
    np.save(directory / artifacts["target_eval_labels"], np.arange(4) % 2, allow_pickle=False)
    manifest.update(
        {
            "artifacts": artifacts,
            "sample_counts": counts,
            "checksums": {name: file_sha256(directory / name) for name in artifacts.values()},
        }
    )
    write_json(directory / "manifest.json", seal(manifest))
    return directory
