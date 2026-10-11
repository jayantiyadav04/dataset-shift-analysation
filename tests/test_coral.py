"""Independent CORAL arithmetic, shared fit inputs and target coordinate boundary."""

import builtins
import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shiftlab.adaptation.coral import (
    CoralState,
    fit_coral,
    sample_covariance,
    symmetric_matrix_power,
)
from shiftlab.contracts import TrainingView
from shiftlab.data.storage import DataError, read_json, seal, write_json
from shiftlab.evaluation.evaluator import evaluate_frozen_run
from shiftlab.experiments.alignment import load_coral_state
from shiftlab.experiments.config import CoralConfig, validate_config
from shiftlab.experiments.provenance import content_hash, file_sha256
from shiftlab.experiments.runner import train_methods
from shiftlab.experiments.storage import LocalArtifactStore

PARAMETERS = CoralConfig(ridge=0.001, eigenvalue_floor=1e-8, centre_and_restore_target_mean=True)


def matrix_fixture():
    scale = np.sqrt(1.5)
    source = scale * np.array([[2, 0], [-2, 0], [0, 1], [0, -1]]) + [3, -2]
    target = scale / np.sqrt(2) * np.array([[3, 3], [-3, -3], [-2, 2], [2, -2]]) + [11, 5]
    return TrainingView(
        source,
        np.array([0, 0, 1, 1]),
        target,
        tuple(f"s:{i}" for i in range(4)),
        tuple(f"t:{i}" for i in range(4)),
    )


def test_unbiased_covariance_hand_calculation_and_single_feature():
    mean, covariance = sample_covariance(np.array([[1, 2], [3, 2], [5, 8]]))
    np.testing.assert_allclose(mean, [3, 4])
    np.testing.assert_allclose(covariance, [[4, 6], [6, 12]])
    mean, covariance = sample_covariance(np.array([[1], [3], [5]]))
    np.testing.assert_allclose(mean, [3])
    np.testing.assert_allclose(covariance, [[4]])


@pytest.mark.parametrize("ridge", [0.0001, 0.001, 0.01])
def test_independent_noncommuting_roots_row_orientation_and_ridge_tolerance(ridge):
    view = matrix_fixture()
    state = fit_coral(view, PARAMETERS.model_copy(update={"ridge": ridge}))
    # Analytic eigensystem: source diagonal; target axes rotated 45 degrees.
    inverse_source = np.diag(1 / np.sqrt([4 + ridge, 1 + ridge]))
    high, low = np.sqrt([9 + ridge, 4 + ridge])
    root_target = np.array([[high + low, high - low], [high - low, high + low]]) / 2
    expected = inverse_source @ root_target
    np.testing.assert_allclose(state.alignment, expected, atol=1e-12, rtol=1e-12)
    assert not np.allclose(expected, root_target @ inverse_source)
    assert not np.allclose(expected, expected.T), "The CORAL product is generally not symmetric"
    aligned = state.transform_source(view.X_source)
    np.testing.assert_allclose(aligned, (view.X_source - [3, -2]) @ expected + [11, 5])
    np.testing.assert_allclose(aligned.mean(axis=0), [11, 5], atol=1e-12)
    Ct = np.array([[6.5, 2.5], [2.5, 6.5]])
    aligned_covariance = np.cov(aligned, rowvar=False, ddof=1)
    # Without eigenvalue flooring, ridge implies this exact covariance residual.
    expected_residual = ridge * (np.eye(2) - expected.T @ expected)
    np.testing.assert_allclose(aligned_covariance - Ct, expected_residual, atol=2e-12)
    ridge_tolerance = np.linalg.norm(expected_residual, "fro") / np.linalg.norm(Ct, "fro")
    assert state.diagnostics["covariance_mismatch_after"] <= ridge_tolerance + 1e-12
    assert (
        state.diagnostics["covariance_mismatch_after"]
        < state.diagnostics["covariance_mismatch_before"]
    )
    assert state.diagnostics["regularised_covariance_residual"] < 1e-12


@pytest.mark.parametrize("constant", [False, True])
def test_same_domain_is_identity_including_singular_features(constant):
    view = matrix_fixture()
    X = np.full((4, 2), 9.0) if constant else view.X_source
    state = fit_coral(replace(view, X_source=X, X_target_adapt=X), PARAMETERS)
    np.testing.assert_allclose(state.alignment, np.eye(2), atol=1e-12)
    np.testing.assert_allclose(state.transform_source(X), X, atol=1e-12)
    assert state.diagnostics["covariance_mismatch_before"] == 0


def test_redundant_and_constant_features_remain_finite_with_zero_target_covariance():
    view = matrix_fixture()
    source = np.column_stack((view.X_source[:, 0], view.X_source[:, 0], np.ones(4)))
    target = np.full((4, 3), 7.0)
    state = fit_coral(replace(view, X_source=source, X_target_adapt=target), PARAMETERS)
    assert state.diagnostics["source_covariance_rank"] == 1
    assert state.diagnostics["target_covariance_rank"] == 0
    assert np.isfinite(state.transform_source(source)).all()
    assert np.isfinite(state.diagnostics["covariance_mismatch_after"])


def test_eigenvalue_floor_is_applied_to_singular_symmetric_roots():
    matrix = np.diag([0.0, 4.0])
    root, diagnostics = symmetric_matrix_power(matrix, 0.5, 1e-8)
    inverse, _ = symmetric_matrix_power(matrix, -0.5, 1e-8)
    np.testing.assert_allclose(root, np.diag([1e-4, 2.0]))
    np.testing.assert_allclose(inverse, np.diag([1e4, 0.5]))
    np.testing.assert_allclose(root @ inverse, np.eye(2))
    assert diagnostics["floored_eigenvalues"] == 1


@pytest.mark.parametrize(
    "matrix", [np.ones((2, 3)), [[1, 2], [0, 1]], [[1, 0], [0, -1]], [[np.nan]], [[1 + 2j]]]
)
def test_bad_covariance_matrices_fail_explicitly(matrix):
    with pytest.raises(ValueError):
        symmetric_matrix_power(np.asarray(matrix), 0.5, 1e-8)


@pytest.mark.parametrize(
    "features", [np.ones((1, 2)), np.ones(3), [[np.inf], [1]], np.ones((3, 0)), [["a"], ["b"]]]
)
def test_bad_feature_contracts_fail_explicitly(features):
    with pytest.raises(ValueError):
        sample_covariance(np.asarray(features))


def test_source_labels_are_unused_and_no_target_label_channel_exists():
    view = matrix_fixture()
    first = fit_coral(view, PARAMETERS)
    second = fit_coral(replace(view, y_source=1 - view.y_source), PARAMETERS)
    np.testing.assert_array_equal(first.alignment, second.alignment)
    assert first.diagnostics == second.diagnostics
    assert not {"y_target", "target_labels", "target_eval"} & set(
        inspect.signature(fit_coral).parameters
    )
    with pytest.raises(TypeError):
        fit_coral(view, PARAMETERS, target_labels=np.ones(4))
    with pytest.raises(ValueError):
        first.transform_source(np.ones((2, 3)))
    with pytest.raises(ValueError):
        first.alignment[0, 0] = 0


def test_shared_coral_inputs_label_isolation_resume_and_target_prediction_coordinates(
    baseline_split,
    tmp_path,
    monkeypatch,
    smoke_dict,
):
    smoke_dict.update(
        models=["lr", "linear_svm", "rf"],
        methods=["source_only", "importance_weighting", "coral"],
        include_majority_baseline=True,
    )
    config = validate_config(smoke_dict)
    identity = {
        "code_version": "1" * 40,
        "code_snapshot_hash": "2" * 64,
        "dependency_lock_hash": "3" * 64,
    }
    monkeypatch.setattr("shiftlab.experiments.runner.runtime_identity", lambda root: identity)
    monkeypatch.setattr(
        "shiftlab.experiments.runner.environment_report", lambda root: {"fixture": True}
    )
    monkeypatch.setattr(
        "shiftlab.experiments.runner.code_snapshot", lambda root: {"fixture": "2" * 64}
    )
    from shiftlab.models.estimators import fit_estimator

    received = []

    def fitting_spy(estimator, X, y, *, sample_weight=None):
        received.append(
            (type(estimator).__name__, content_hash(X.tolist()), sample_weight is not None)
        )
        return fit_estimator(estimator, X, y, sample_weight=sample_weight)

    monkeypatch.setattr("shiftlab.experiments.runner.fit_estimator", fitting_spy)
    heldout = {p: p.read_bytes() for p in (baseline_split / "evaluation").glob("*")}
    for p in heldout:
        p.write_bytes(b"corrupted synthetic held-out data before fit")
    path_open, builtin_open = Path.open, builtins.open

    def guard(path):
        if isinstance(path, (str, Path)) and {"evaluation", "private"} & set(Path(path).parts):
            pytest.fail("CORAL fitting opened forbidden target data")

    def open_path(path, *args, **kwargs):
        guard(path)
        return path_open(path, *args, **kwargs)

    def open_builtin(path, *args, **kwargs):
        guard(path)
        return builtin_open(path, *args, **kwargs)

    store = LocalArtifactStore(tmp_path / "artifacts")
    with monkeypatch.context() as boundary:
        boundary.setattr(Path, "open", open_path)
        boundary.setattr(builtins, "open", open_builtin)
        result = train_methods(config, root=tmp_path, data_root=tmp_path / "data")
        assert result["fitted_records"] == result["expected_fits"] == 10
        coral_rows = [row for row in result["runs"] if row["method"] == "coral"]
        assert len(coral_rows) == 3
        cache_ids = {row["coral_cache_id"] for row in coral_rows}
        assert len(cache_ids) == 1
        cache_id = cache_ids.pop()
        cache = store.verify("coral_cache", cache_id)
        aligned_hash = cache["aligned_source_hash"]
        assert {
            name for name, digest, weighted in received if digest == aligned_hash and not weighted
        } == {"LogisticRegression", "LinearSVC", "RandomForestClassifier"}
        assert sum(digest == aligned_hash for _, digest, _ in received) == 3
        assert len({row["source_cv_id"] for row in result["runs"]}) == 1
        for row in coral_rows:
            baseline = next(
                r
                for r in result["runs"]
                if r["model"] == row["model"] and r["method"] == "source_only"
            )
            key = store.verify("runs", row["run_id"])["canonical_key"]
            baseline_key = store.verify("runs", baseline["run_id"])["canonical_key"]
            assert key["preprocessing_hash"] == baseline_key["preprocessing_hash"]
            assert (
                key["selected_source_parameter_hash"]
                == baseline_key["selected_source_parameter_hash"]
            )
            directory = store.directory("runs", row["run_id"])
            assert (
                read_json(directory / "diagnostics.json")["source_fit_features_hash"]
                == aligned_hash
            )
            assert not (directory / "predictions.csv").exists()
        for p in heldout:
            with path_open(p, "wb") as stream:
                stream.write(b"different synthetic held-out features and labels")
        files = {
            p: (file_sha256(p), p.stat().st_mtime_ns) for p in store.root.rglob("*") if p.is_file()
        }
        again = train_methods(config, root=tmp_path, data_root=tmp_path / "data")
        assert again["new_fits"] == 0 and again["resumed_fits"] == 10
        assert files == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in files}
        assert len(received) == 10, "Resuming retrained a task model"

    # Restore only temporary synthetic evaluation inputs, then test the evaluator.
    for p, original in heldout.items():
        p.write_bytes(original)
    row = coral_rows[0]
    run_id = row["run_id"]
    run = store.directory("runs", run_id)
    split = read_json(baseline_split / "manifest.json")
    key = store.verify("runs", run_id)["canonical_key"]
    state = load_coral_state(
        run / "adaptation_state.npz",
        preprocessing_hash=key["preprocessing_hash"],
        source_ids=tuple(split["source_ids"]),
        target_adapt_ids=tuple(split["target_adapt_ids"]),
    )
    with pytest.raises(DataError, match="preprocessing"):
        load_coral_state(
            run / "adaptation_state.npz",
            preprocessing_hash="9" * 64,
            source_ids=tuple(split["source_ids"]),
            target_adapt_ids=tuple(split["target_adapt_ids"]),
        )
    assert np.isfinite(state.alignment).all()
    monkeypatch.setattr(
        CoralState, "transform_source", lambda *a: pytest.fail("Aligned target features")
    )
    original_loader = store.load_joblib
    predicted_features = []

    def loader(kind, identity, name):
        obj = original_loader(kind, identity, name)
        if name == "fitted_model.joblib":
            original_predict = obj.predict

            def predict(X):
                predicted_features.append(X.copy())
                return original_predict(X)

            obj.predict = predict
        return obj

    monkeypatch.setattr(store, "load_joblib", loader)
    lock = tmp_path / "synthetic_lock.json"
    authorization = {
        field: key[field]
        for field in (
            "config_hash",
            "code_version",
            "code_snapshot_hash",
            "dependency_lock_hash",
            "protocol_version",
        )
    }
    authorization.update(
        status="frozen_fits_complete",
        run_ids=[run_id],
        split_hashes={key["task_id"]: split["manifest_hash"]},
        evaluation_ids={key["task_id"]: split["target_eval_ids"]},
    )
    write_json(lock, seal(authorization))
    evaluation, _ = evaluate_frozen_run(store, run_id, baseline_split, lock)
    assert evaluation["sample_count"] == 4
    pre = original_loader("runs", run_id, "preprocessing.joblib")
    with np.load(baseline_split / "evaluation/target_eval.npz", allow_pickle=False) as arrays:
        expected = (arrays["X"] - pre.scaler_.mean_) / pre.scaler_.scale_
    np.testing.assert_allclose(predicted_features[0], expected, atol=1e-12)
    timings = store.verify("runs", run_id)["timing"]
    assert timings["adaptation_fit_seconds"] == cache["adaptation_fit_seconds"]
    assert timings["pipeline_fit_seconds"] == pytest.approx(
        sum(
            timings[k]
            for k in (
                "preprocessing_fit_seconds",
                "source_transform_seconds",
                "adaptation_fit_seconds",
                "classifier_fit_seconds",
            )
        )
    )
