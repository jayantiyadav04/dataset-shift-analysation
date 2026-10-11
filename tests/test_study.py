"""Matrix, interruption, provenance and dependence-aware uncertainty checks."""

import csv
import os
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from shiftlab.data.storage import DataError, read_json, seal, write_json
from shiftlab.data.training import load_training
from shiftlab.evaluation.analysis import load_predictions, validate_pair
from shiftlab.evaluation.metrics import classification_metrics
from shiftlab.evaluation.statistics import (
    confusion_by_group,
    evaluation_units,
    paired_group_bootstrap,
    scores_from_confusion,
)
from shiftlab.experiments.checkpoints import (
    Journal,
    atomic_json,
    job_owner,
    recover_owner,
    recover_records,
)
from shiftlab.experiments.config import load_config
from shiftlab.experiments.selection import select_source
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.experiments.study import evaluate_study, initialize, open_study, schedule

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {
    "code_version": "1" * 40,
    "code_snapshot_hash": "2" * 64,
    "dependency_lock_hash": "3" * 64,
}


def test_frozen_matrix_counts_and_sensitivity_reuse():
    rows = schedule(load_config(ROOT / "configs/study.yaml"))
    counts = Counter(r["setting"] for r in rows)
    assert counts == {
        "primary": 1100,
        "importance_weighting:5": 45,
        "importance_weighting:20": 45,
        "coral:0.0001": 45,
        "coral:0.01": 45,
    }
    assert len({r["row_id"] for r in rows}) == 1280
    primary = [r for r in rows if r["setting"] == "primary"]
    assert sum(r["model"] == "majority" for r in primary) == 110
    assert Counter(r["dataset"] for r in primary) == {"gas": 450, "har": 50, "amazon": 600}


def test_group_resampling_keeps_multiplicity_and_fixed_class_list():
    y = np.array([0, 0, 1, 1, 1])
    pred = np.array([[0, 0, 0, 1, 1]])
    groups = ("a", "a", "b", "b", "b")
    cm, names = confusion_by_group(y, pred, groups, (0, 1, 2))
    assert names == ("a", "b")
    # Draw a twice, b once: all windows, including duplicates of the group.
    repeated = np.array([0, 1, 0, 1, 2, 3, 4])
    counts = np.einsum("g,gscd->scd", np.array([2, 1]), cm)
    result = scores_from_confusion(counts, np.array([0, 1]))
    expected = classification_metrics(
        y[repeated], pred[0, repeated], classes=(0, 1), known_classes=(0, 1, 2)
    )
    for metric in ("accuracy", "macro_f1", "macro_precision", "macro_recall"):
        assert result[metric][0] == pytest.approx(expected[metric])
    # A degenerate resample with no class 1 keeps the full-test macro list.
    only_a = scores_from_confusion(cm[0], np.array([0, 1]))
    assert only_a["macro_f1"][0] == 0.5


def test_bootstrap_matches_independent_explicit_group_draws():
    y = np.array([0, 0, 1, 1, 1])
    baseline = np.tile(np.array([0, 0, 0, 1, 1]), (5, 1))
    adapted = np.tile(np.array([1, 0, 1, 1, 1]), (5, 1))
    groups = ("a", "a", "b", "b", "b")
    result = paired_group_bootstrap(
        y, baseline, adapted, groups, known_classes=(0, 1), replicates=1000, seed=2026
    )
    draws = np.random.default_rng(2026).multinomial(2, [0.5, 0.5], size=1000)
    deltas = []
    for draw in draws:
        indices = np.concatenate(
            [
                np.tile(np.flatnonzero(np.array(groups) == group), count)
                for group, count in zip(("a", "b"), draw, strict=True)
            ]
        )
        b = classification_metrics(y[indices], baseline[0, indices], classes=(0, 1))
        a = classification_metrics(y[indices], adapted[0, indices], classes=(0, 1))
        deltas.append(a["macro_f1"] - b["macro_f1"])
    assert [result["macro_f1_lower"], result["macro_f1_upper"]] == pytest.approx(
        np.percentile(deltas, [2.5, 97.5])
    )
    zero = paired_group_bootstrap(y, baseline, baseline, groups, known_classes=(0, 1))
    assert zero["macro_f1_lower"] == zero["macro_f1_upper"] == 0
    insufficient = paired_group_bootstrap(y, baseline, adapted, ("one",) * 5, known_classes=(0, 1))
    assert insufficient["usable_replicates"] == 0
    assert insufficient["macro_f1_lower"] is None
    with pytest.raises(ValueError, match="five"):
        paired_group_bootstrap(y, baseline[:4], adapted[:4], groups, known_classes=(0, 1))


def test_amazon_uses_product_groups_only_when_complete():
    split = {
        "dataset_id": "amazon",
        "target_eval_ids": ["1", "2"],
        "target_eval_group_ids": ["body1", "body2"],
        "target_eval_product_ids": ["product", "product"],
    }
    assert evaluation_units(split)[0] == ("product", "product")
    split["target_eval_product_ids"][0] = None
    assert evaluation_units(split)[0] == ("body1", "body2")


def test_pair_rejects_preprocessing_params_seed_and_split_changes():
    base = {
        "canonical_key": {
            f: "fixed"
            for f in (
                "dataset_hash",
                "task_id",
                "split_hash",
                "model",
                "selected_source_parameter_hash",
                "preprocessing_hash",
                "seed",
                "protocol_version",
                "config_hash",
                "code_version",
                "code_snapshot_hash",
                "dependency_lock_hash",
            )
        }
    }
    for field in ("preprocessing_hash", "selected_source_parameter_hash", "seed", "split_hash"):
        other = {"canonical_key": {**base["canonical_key"], field: "different"}}
        with pytest.raises(DataError, match=field):
            validate_pair(base, other)


def test_predictions_reject_missing_duplicate_or_reordered_ids(tmp_path):
    path = tmp_path / "predictions.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=["sample_id", "y_true", "y_pred", "evaluation_group_id"]
        )
        writer.writeheader()
        writer.writerows(
            [
                {"sample_id": s, "y_true": 0, "y_pred": 0, "evaluation_group_id": s}
                for s in ("a", "b")
            ]
        )
    assert len(load_predictions(path, ["a", "b"])["true"]) == 2
    for ids in (["b", "a"], ["a"], ["a", "a"]):
        with pytest.raises(DataError):
            load_predictions(path, ids)


def test_journal_detects_mutation_and_owner_refuses_live_recovery(tmp_path):
    journal = Journal(tmp_path)
    journal.append({"event": "started"})
    journal.append({"event": "fitted", "run": "fixture"})
    assert len(Journal(tmp_path).events) == 2
    with job_owner(tmp_path):
        with pytest.raises(DataError, match="confirmed dead"):
            recover_owner(tmp_path)
    journal.path.write_text(journal.path.read_text().replace('"fixture"', '"changed"'))
    with pytest.raises(DataError, match="Invalid"):
        Journal(tmp_path)


def test_source_cv_resumes_completed_candidates_after_interruption(
    baseline_split, tmp_path, monkeypatch
):
    from shiftlab.experiments import selection

    config = load_config(ROOT / "configs/smoke.yaml")
    view = load_training(baseline_split)
    store = LocalArtifactStore(tmp_path / "checkpoints")
    original = selection.fit_estimator
    fitted = 0

    def interrupt(*args, **kwargs):
        nonlocal fitted
        fitted += 1
        if fitted == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, "fit_estimator", interrupt)
    with pytest.raises(KeyboardInterrupt):
        select_source(store, view, config, 17, IDENTITY)
    assert list((store.root / "failures").rglob("source_cv.csv"))
    assert not list((store.root / "locks").glob("*.lock"))
    calls = 0

    def count(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(selection, "fit_estimator", count)
    record, resumed = select_source(store, view, config, 17, IDENTITY)
    assert not resumed
    assert record["source_search_candidates_resumed"] == 2
    assert record["source_cv_rows"] == 15 and calls == 13


def test_dead_record_recovery_preserves_attempt_and_refuses_live_pid(tmp_path, monkeypatch):
    store = LocalArtifactStore(tmp_path)
    key = {"fixture": True}
    from shiftlab.experiments.provenance import content_hash

    identity = content_hash(key)
    destination = store.directory("runs", identity)
    stage = destination.with_name(f".{identity}.pending-fixture")
    stage.mkdir(parents=True)
    write_json(
        stage / "attempt.json", {"canonical_key": key, "identity": identity, "status": "running"}
    )
    (stage / "partial.txt").write_text("original")
    atomic_json(
        tmp_path / "locks" / f"runs-{identity}.lock",
        {"pid": os.getpid(), "kind": "runs", "identity": identity},
    )
    with pytest.raises(DataError, match="alive"):
        recover_records(store)
    monkeypatch.setattr("shiftlab.experiments.checkpoints.pid_alive", lambda pid: False)
    assert len(recover_records(store)) == 1
    assert len(list((tmp_path / "failures").rglob("partial.txt"))) == 1
    assert not stage.exists()


@pytest.fixture
def frozen_fixture(tmp_path, baseline_split, monkeypatch):
    config = load_config(ROOT / "configs/study.yaml")
    manifest = read_json(baseline_split / "manifest.json")
    for task in config.selected_tasks():
        if task == manifest["task_id"]:
            continue
        write_json(
            tmp_path / "data/splits/1.0" / task / "manifest.json",
            seal(
                {
                    **{k: v for k, v in manifest.items() if k != "manifest_hash"},
                    "task_id": task,
                    "dataset_id": task.split("_")[0],
                }
            ),
        )
    for dataset in config.datasets:
        write_json(
            tmp_path / "data/processed" / dataset / "manifest.json",
            {"raw_sha256": {"fixture": "4" * 64}},
        )
    monkeypatch.setattr("shiftlab.experiments.study.runtime_identity", lambda root: IDENTITY)
    monkeypatch.setattr(
        "shiftlab.experiments.study.code_snapshot", lambda root: {"fixture": "2" * 64}
    )
    monkeypatch.setattr("shiftlab.experiments.study.hardware_report", lambda: {"fixture": True})
    return config, tmp_path, tmp_path / "data"


def test_freeze_never_opens_evaluation_and_rejects_changed_matrix(frozen_fixture, monkeypatch):
    config, root, data = frozen_fixture
    original = Path.open

    def guarded(path, *args, **kwargs):
        if "evaluation" in path.parts:
            pytest.fail("Freeze attempted to open held-out data")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", guarded)
    directory, lock = initialize(config, root, data)
    assert lock["status"] == "frozen_planned"
    assert open_study(root, lock["study_id"])[1] == lock
    with pytest.raises(DataError, match="fit_completion_lock"):
        evaluate_study(root, lock["study_id"])
    path = directory / "expected_runs.csv"
    path.write_text(path.read_text().replace("gas_b01_b02", "gas_b01_b03", 1))
    with pytest.raises(DataError, match="matrix"):
        open_study(root, lock["study_id"])


def test_complete_saved_prediction_analysis_and_immutable_export(frozen_fixture, monkeypatch):
    """Offline saved-prediction fixture; no dataset fits or claimed research findings."""
    import pandas as pd

    from shiftlab.data.parsers import CLASS_MAPS
    from shiftlab.evaluation.analysis import summarize_study
    from shiftlab.experiments.provenance import content_hash, file_sha256

    config, root, data = frozen_fixture
    directory, lock = initialize(config, root, data)
    store = LocalArtifactStore(root / "artifacts")
    journal = Journal(directory)
    records = {}
    for row in schedule(config):
        run_id = content_hash({"synthetic_analysis_run": row["row_id"]})
        evaluation_id = content_hash({"synthetic_analysis_evaluation": row["row_id"]})
        fitted = {**row, "run_id": run_id, "source_cv_id": "a" * 64, "status": "fitted"}
        journal.append({"event": "fit_saved", "row": fitted})
        journal.append(
            {"event": "evaluation_saved", "row_id": row["row_id"], "evaluation_id": evaluation_id}
        )
        split = read_json(data / "splits/1.0" / row["task_id"] / "manifest.json")
        key = {
            "dataset_hash": split["dataset_hash"],
            "task_id": row["task_id"],
            "split_hash": split["manifest_hash"],
            "model": row["model"],
            "selected_source_parameter_hash": "b" * 64,
            "preprocessing_hash": "c" * 64,
            "seed": row["seed"],
            "protocol_version": "1.0",
            "config_hash": lock["config_hash"],
            **IDENTITY,
        }
        records[("runs", run_id)] = {"canonical_key": key, "warnings": []}
        run_dir = store.directory("runs", run_id)
        write_json(run_dir / "diagnostics.json", {"importance_weighting": None, "coral": None})
        write_json(run_dir / "manifest.json", {"fixture": True})
        y = np.array([0, 0, 0, 1])
        pred = {
            "source_only": np.array([0, 0, 0, 0]),
            "importance_weighting": np.array([0, 1, 1, 1]),
            "coral": np.array([1, 1, 1, 1]),
        }[row["method"]]
        ep = store.directory("evaluations", evaluation_id)
        ep.mkdir(parents=True)
        with (ep / "predictions.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=["sample_id", "y_true", "y_pred", "evaluation_group_id"]
            )
            writer.writeheader()
            writer.writerows(
                {"sample_id": s, "y_true": int(t), "y_pred": int(p), "evaluation_group_id": g}
                for s, t, p, g in zip(
                    split["target_eval_ids"], y, pred, split["target_eval_group_ids"], strict=True
                )
            )
        write_json(
            ep / "metrics.json",
            classification_metrics(y, pred, known_classes=tuple(CLASS_MAPS[row["dataset"]])),
        )
        records[("evaluations", evaluation_id)] = {
            "timing": {
                "preprocessing_fit_seconds": 1.0,
                "source_transform_seconds": 0.5,
                "shared_source_search_seconds": 3.0,
                "adaptation_fit_seconds": 0.25,
                "classifier_fit_seconds": 2.0,
                "target_prediction_seconds": 0.5,
                "pipeline_fit_predict_seconds": 4.75,
            }
        }
    write_json(directory / "fit_completion_lock.json", {"fixture": True})
    monkeypatch.setattr(
        LocalArtifactStore, "verify", lambda self, kind, identity: records[(kind, identity)]
    )
    monkeypatch.setattr("shiftlab.evaluation.analysis._diagnostic_figures", lambda *args: None)
    result = summarize_study(root, lock["study_id"])
    assert result["status"] == "complete"
    assert result["findings"]["independent_metric_recomputations"] == 1280
    analysis = directory / "analysis"
    assert len(pd.read_csv(analysis / "paired_deltas.csv")) == 840
    assert len(pd.read_csv(analysis / "bootstrap_intervals.csv")) == 168
    assert len(pd.read_csv(analysis / "ablations.csv")) == 270
    assert len(pd.read_csv(analysis / "per_task_summary.csv")) == 256
    table = pd.read_csv(analysis / "dataset_summary.csv")
    assert table[table.method == "source_only"].macro_f1_mean.tolist() == pytest.approx(
        [3 / 7] * 12
    )
    hashes = {p: file_sha256(p) for p in analysis.rglob("*") if p.is_file()}
    assert summarize_study(root, lock["study_id"])["status"] == "analysis_verified"
    assert all(file_sha256(p) == value for p, value in hashes.items())
