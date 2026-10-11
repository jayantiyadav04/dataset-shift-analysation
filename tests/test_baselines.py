"""Offline algorithm, fold, run storage and evaluation boundary acceptance tests."""

import builtins
import csv
from dataclasses import replace
from pathlib import Path

import joblib
import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from shiftlab.data.storage import DataError, read_json, seal, write_json
from shiftlab.data.training import load_training
from shiftlab.evaluation.evaluator import evaluate_frozen_run
from shiftlab.evaluation.metrics import classification_metrics
from shiftlab.experiments.config import load_config
from shiftlab.experiments.provenance import file_sha256
from shiftlab.experiments.runner import train_baselines
from shiftlab.experiments.selection import candidate_grid, selection_key
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.models.estimators import MajorityClassifier, estimator_factory, fit_estimator

ROOT = Path(__file__).resolve().parents[1]
IDENTITY = {
    "code_version": "1" * 40,
    "code_snapshot_hash": "2" * 64,
    "dependency_lock_hash": "3" * 64,
}


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


@pytest.fixture
def fitted_fixture(baseline_split, monkeypatch, tmp_path):
    # A synthetic temporary root; never a fabricated study in the repository.
    monkeypatch.setattr("shiftlab.experiments.runner.runtime_identity", lambda root: IDENTITY)
    monkeypatch.setattr(
        "shiftlab.experiments.runner.code_snapshot", lambda root: {"fixture": "2" * 64}
    )
    monkeypatch.setattr(
        "shiftlab.experiments.runner.environment_report", lambda root: {"fixture": True}
    )
    config = load_config(ROOT / "configs/smoke.yaml")
    result = train_baselines(config, root=tmp_path, data_root=tmp_path / "data")
    return config, result, LocalArtifactStore(tmp_path / "artifacts"), baseline_split


def test_hand_checked_accuracy_macro_f1_and_missing_classes():
    scores = classification_metrics(
        np.array([0, 0, 0, 1]), np.array([0, 0, 0, 0]), known_classes=(0, 1, 2)
    )
    assert scores["accuracy"] == 0.75
    assert scores["macro_f1"] == pytest.approx(3 / 7)
    assert scores["macro_precision"] == 0.375
    assert scores["macro_recall"] == 0.5
    assert scores["absent_target_classes"] == [2]
    assert scores["confusion_matrix"] == [[3, 0, 0], [1, 0, 0], [0, 0, 0]]
    # Bootstrap replicates retain the full held-out class rule, even when absent.
    subset = classification_metrics(np.array([0, 0]), np.array([0, 0]), classes=(0, 1))
    assert subset["macro_f1"] == 0.5


@pytest.mark.parametrize("true,pred", [([0], [0, 1]), ([], []), ([0.5], [0]), ([0], [2])])
def test_metric_contract_rejects_bad_labels(true, pred):
    with pytest.raises(ValueError):
        classification_metrics(np.asarray(true), np.asarray(pred), known_classes=(0, 1))


def test_majority_uses_source_counts_and_lowest_class_tie():
    X = np.zeros((4, 2))
    model = MajorityClassifier().fit(X, np.array([3, 1, 3, 1]))
    np.testing.assert_array_equal(model.predict(np.ones((5, 2))), [1] * 5)
    assert MajorityClassifier().fit(X, np.array([3, 3, 3, 1])).majority_class_ == 3


@pytest.mark.parametrize("family", ["lr", "linear_svm", "rf"])
def test_factories_receive_weights_with_frozen_random_states(family):
    model = estimator_factory(family, candidate_grid(family)[0], 29)
    assert model.random_state == 29 and model.class_weight is None
    if family == "rf":
        assert model.n_jobs == 1 and model.n_estimators == 200
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
    y, weights = np.array([0, 0, 1, 1]), np.array([0.2, 0.5, 1.0, 2.0])
    original_fit = model.fit
    received = []

    def spy(features, labels, *, sample_weight):
        received.append(sample_weight.copy())
        return original_fit(features, labels, sample_weight=sample_weight)

    model.fit = spy
    assert fit_estimator(model, X, y, sample_weight=weights) == []
    np.testing.assert_array_equal(received[0], weights)


def test_convergence_warning_is_a_failure():
    import warnings

    class NotConverged:
        def fit(self, X, y):
            warnings.warn("fixture solver stopped", ConvergenceWarning, stacklevel=2)

    with pytest.raises(RuntimeError, match="Unresolved source convergence"):
        fit_estimator(NotConverged(), np.zeros((2, 2)), np.array([0, 1]))


def test_source_selection_identity_excludes_target_and_includes_source(baseline_split):
    view = load_training(baseline_split)
    config = load_config(ROOT / "configs/smoke.yaml")
    first = selection_key(view, config, 17, IDENTITY)
    changed = replace(
        view,
        split_hash="9" * 64,
        target_adapt=replace(view.target_adapt, values=view.target_adapt.values + 999),
    )
    assert first == selection_key(changed, config, 17, IDENTITY)
    changed = replace(view, y_source=1 - view.y_source)
    assert first != selection_key(changed, config, 17, IDENTITY)
    assert first != selection_key(view, config, 29, IDENTITY)
    assert candidate_grid("rf")[0]["max_depth"] == 20
    assert candidate_grid("rf")[0]["min_samples_leaf"] == 3


def test_cv_fit_provenance_contains_only_fold_training_rows(fitted_fixture):
    _, result, store, directory = fitted_fixture
    source = load_training(directory).source
    selection = read_json(
        store.directory("source_cache", result["runs"][0]["source_cv_id"]) / "selection.json"
    )
    positions = {sid: index for index, sid in enumerate(source.ids)}
    for fold in selection["folds"]:
        assert fold["train_ids"] == fold["preprocessing"]["fit_ids"]
        assert not set(fold["train_ids"]) & set(fold["validation_ids"])
        assert not set(fold["train_groups"]) & set(fold["validation_groups"])
        expected = source.values[[positions[sid] for sid in fold["train_ids"]]].mean(axis=0)
        np.testing.assert_allclose(fold["preprocessing"]["scaler_mean"], expected)


def test_resume_is_byte_identical_and_never_opens_target_evaluation(
    fitted_fixture, tmp_path, monkeypatch
):
    config, result, store, directory = fitted_fixture
    run = store.directory("runs", result["runs"][0]["run_id"])
    before = {p: (file_sha256(p), p.stat().st_mtime_ns) for p in run.rglob("*") if p.is_file()}
    (directory / "evaluation/target_eval_labels.npy").write_bytes(
        b"intentionally corrupted fixture"
    )
    original = builtins.open

    def guard(path, *args, **kwargs):
        if isinstance(path, (str, Path)) and "evaluation" in Path(path).parts:
            pytest.fail("Trainer accessed final target evaluation")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", guard)
    monkeypatch.setattr(
        "shiftlab.models.estimators.LogisticRegression.fit",
        lambda *args, **kwargs: pytest.fail("A resumed run retrained"),
    )
    resumed = train_baselines(config, root=tmp_path, data_root=tmp_path / "data")
    assert resumed["new_fits"] == 0 and resumed["resumed_fits"] == 1
    assert before == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in before}
    assert not (run / "predictions.csv").exists() and not (run / "metrics.json").exists()
    timings = read_json(run / "timings.json")
    assert timings["target_prediction_seconds"] is None
    assert timings["pipeline_fit_seconds"] == pytest.approx(
        sum(
            timings[field]
            for field in (
                "preprocessing_fit_seconds",
                "source_transform_seconds",
                "adaptation_fit_seconds",
                "classifier_fit_seconds",
            )
        )
    )


@pytest.mark.parametrize(
    "exception,status", [(RuntimeError, "failed"), (KeyboardInterrupt, "cancelled")]
)
def test_failure_and_cancellation_preserve_attempts_and_allow_new_retry(
    tmp_path, exception, status
):
    store = LocalArtifactStore(tmp_path)
    key = {"fixture": True}

    def fail(stage):
        (stage / "partial.txt").write_text("original attempt")
        raise exception("fixture interruption")

    with pytest.raises(exception):
        store.create("runs", key, fail)
    failures = list((tmp_path / "failures").rglob("manifest.json"))
    assert len(failures) == 1 and read_json(failures[0])["status"] == status
    record, hit = store.create("runs", key, lambda stage: {"fixture": True})
    assert not hit and record["status"] == "fitted"
    assert failures[0].is_file() and not list((tmp_path / "locks").iterdir())


def test_concurrent_writer_cannot_claim_same_key(tmp_path):
    store = LocalArtifactStore(tmp_path)
    key = {"fixture": True}

    def outer(stage):
        with pytest.raises(DataError, match="locked"):
            store.create("runs", key, lambda stage: {})
        return {}

    store.create("runs", key, outer)
    with pytest.raises(DataError, match="already exists"):
        store.create("runs", key, lambda stage: pytest.fail("overwrote"), resume=False)


def test_untrusted_or_tampered_pickle_is_rejected_before_loading(tmp_path, monkeypatch):
    store = LocalArtifactStore(tmp_path)
    key = {"fixture": True}

    def produce(stage):
        joblib.dump({"safe_fixture": True}, stage / "fitted_model.joblib")
        return {}

    record, _ = store.create("runs", key, produce)
    identity = record["identity"]
    assert store.load_joblib("runs", identity, "fitted_model.joblib") == {"safe_fixture": True}
    monkeypatch.setattr(joblib, "load", lambda *args: pytest.fail("Deserialized untrusted bytes"))
    (store.directory("runs", identity) / "fitted_model.joblib").write_bytes(b"tampered")
    with pytest.raises(DataError, match="checksum"):
        store.load_joblib("runs", identity, "fitted_model.joblib")
    with pytest.raises(DataError):
        store.load_joblib("runs", "0" * 64, "fitted_model.joblib")
    with pytest.raises(DataError, match="known"):
        store.load_joblib("runs", identity, "../../upload.joblib")


def evaluation_lock(store, run_id, directory, path, *, status="frozen_fits_complete"):
    record = store.verify("runs", run_id)
    key = record["canonical_key"]
    split = read_json(directory / "manifest.json")
    lock = {
        field: key[field]
        for field in (
            "config_hash",
            "code_version",
            "code_snapshot_hash",
            "dependency_lock_hash",
            "protocol_version",
        )
    }
    lock.update(
        {
            "status": status,
            "run_ids": [run_id],
            "split_hashes": {key["task_id"]: key["split_hash"]},
            "evaluation_ids": {key["task_id"]: split["target_eval_ids"]},
        }
    )
    write_json(path, seal(lock))


def test_evaluator_requires_frozen_fit_before_opening_labels(fitted_fixture, tmp_path, monkeypatch):
    _, result, store, directory = fitted_fixture
    run_id = result["runs"][0]["run_id"]
    lock_path = tmp_path / "lock.json"
    evaluation_lock(store, run_id, directory, lock_path, status="unfrozen")
    monkeypatch.setattr(np, "load", lambda *args, **kwargs: pytest.fail("Opened held-out arrays"))
    with pytest.raises(DataError, match="frozen study"):
        evaluate_frozen_run(store, run_id, directory, lock_path)


def test_synthetic_evaluator_writes_ordered_predictions_and_recomputed_metrics(
    fitted_fixture, tmp_path
):
    _, result, store, directory = fitted_fixture
    run_id = result["runs"][0]["run_id"]
    lock_path = tmp_path / "lock.json"
    evaluation_lock(store, run_id, directory, lock_path)
    record, hit = evaluate_frozen_run(store, run_id, directory, lock_path)
    assert not hit and record["status"] == "evaluated"
    evaluated = store.directory("evaluations", record["identity"])
    with (evaluated / "predictions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    assert [row["sample_id"] for row in rows] == read_json(directory / "manifest.json")[
        "target_eval_ids"
    ]
    metrics = classification_metrics(
        np.array([int(r["y_true"]) for r in rows]),
        np.array([int(r["y_pred"]) for r in rows]),
        known_classes=tuple(range(6)),
    )
    assert metrics == read_json(evaluated / "metrics.json")
    assert evaluate_frozen_run(store, run_id, directory, lock_path)[1]
    assert store.verify("runs", run_id)["status"] == "fitted"  # original immutable fit retained
