"""Independent arithmetic, grouped OOF, provenance and weight delivery tests."""

import builtins
import inspect
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

from shiftlab.adaptation.importance import (
    density_ratios,
    fit_importance_weights,
    validate_crossfit_folds,
    weight_diagnostics,
)
from shiftlab.contracts import TrainingView
from shiftlab.data.storage import read_json, write_json
from shiftlab.experiments.config import load_config, validate_config
from shiftlab.experiments.provenance import content_hash, file_sha256
from shiftlab.experiments.runner import train_methods
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.features.preprocessing import SourceFolds

ROOT = Path(__file__).resolve().parents[1]
PARAMETERS = load_config(ROOT / "configs/smoke.yaml").importance_weighting


def grouped_fixture(count=30, target_count=11):
    source = np.repeat(np.arange(count // 2), 2)
    X = np.column_stack((source, source / 10)).astype(float)
    ids = tuple(f"s{i}" for i in range(count))
    groups = tuple(f"person:{i}" for i in source)
    target = np.column_stack((np.arange(target_count), np.arange(target_count) / 10)).astype(float)
    view = TrainingView(
        X,
        np.arange(count) % 2,
        target,
        ids,
        tuple(f"t{i}" for i in range(target_count)),
        groups,
        tuple(f"target-person:{i}" for i in range(target_count)),
    )
    positions = np.arange(count)
    splits = tuple(
        (
            positions[~np.isin(source, np.arange(count // 2)[number::5])],
            positions[np.isin(source, np.arange(count // 2)[number::5])],
        )
        for number in range(5)
    )
    return view, SourceFolds(splits, groups, "grouped_fixture", "whole fixture people")


def test_unequal_prior_and_varying_fold_counts_return_unit_ratios():
    counts, nt = np.array([12, 15, 18]), 30
    q = nt / (counts + nt)
    _, raw, clipped, weights = density_ratios(q, counts, nt, PARAMETERS)
    np.testing.assert_allclose(raw, 1, rtol=1e-14)
    np.testing.assert_allclose(clipped, 1, rtol=1e-14)
    np.testing.assert_allclose(weights, 1, rtol=1e-14)
    assert not np.allclose(q / (1 - q), raw), "Omitting the actual fold prior must fail"


def test_probability_bounds_clipping_then_global_normalization_hand_calculation():
    q, counts, nt = np.array([0, 1, 0.5]), np.array([10, 10, 20]), 10
    bounded, raw, clipped, weights = density_ratios(q, counts, nt, PARAMETERS)
    np.testing.assert_allclose(bounded, [0.001, 0.999, 0.5])
    np.testing.assert_allclose(raw, [1 / 999, 999, 2])
    np.testing.assert_allclose(clipped, [0.05, 10, 2])
    np.testing.assert_allclose(weights, np.array([0.05, 10, 2]) / (12.05 / 3))
    diagnostics = weight_diagnostics(q, bounded, raw, clipped, weights, PARAMETERS)
    assert diagnostics["weight_clipping_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["probability_clipping_fraction"] == pytest.approx(2 / 3)
    assert diagnostics["ess"] == pytest.approx(12.05**2 / (0.05**2 + 100 + 4))


@pytest.mark.parametrize("upper", [5.0, 10.0, 20.0])
def test_predeclared_clipping_bounds_do_not_bound_normalized_weights(upper):
    parameters = PARAMETERS.model_copy(update={"weight_clip_upper": upper})
    q = np.array([1.0] + [0.0] * 100)
    _, _, clipped, weights = density_ratios(q, np.ones(101, dtype=int), 1, parameters)
    assert clipped.max() == upper
    assert weights.max() > upper and weights.mean() == pytest.approx(1.0)
    assert np.isfinite(weights).all() and np.all(weights > 0)


def test_collapsed_ess_warns_and_keeps_positive_weights():
    q = np.array([1.0] + [0.0] * 199)
    bounded, raw, clipped, weights = density_ratios(q, np.ones(200, dtype=int), 1, PARAMETERS)
    diagnostics = weight_diagnostics(q, bounded, raw, clipped, weights, PARAMETERS)
    independent = float(sum(weights) ** 2 / sum(float(w) ** 2 for w in weights))
    assert diagnostics["ess"] == pytest.approx(independent)
    assert 0 < diagnostics["ess_fraction"] < 0.1
    assert diagnostics["warnings"] and np.all(weights > 0)


@pytest.mark.parametrize(
    "q,counts,nt",
    [
        ([np.nan], [3], 2),
        ([-0.1], [3], 2),
        ([1.1], [3], 2),
        ([0.5], [0], 2),
        ([0.5], [3.0], 2),
        ([0.5], [3], 0),
        ([0.5], [3], True),
        ([0.5, 0.5], [3], 2),
    ],
)
def test_bad_probabilities_or_domain_counts_fail(q, counts, nt):
    with pytest.raises(ValueError):
        density_ratios(np.array(q), np.array(counts), nt, PARAMETERS)


def test_oof_source_exclusion_actual_counts_domain_labels_and_row_order(monkeypatch):
    view, folds = grouped_fixture()
    fit_calls = []

    class ConstantPrior:
        def __init__(self, **kwargs):
            assert kwargs["class_weight"] is None and kwargs["C"] == 1
            assert kwargs["random_state"] == 17

        def fit(self, X, labels):
            self.classes_ = np.array([0, 1])
            self.coef_, self.intercept_, self.n_iter_ = np.zeros((1, 2)), np.zeros(1), np.array([1])
            ns, nt = int(np.sum(labels == 0)), int(np.sum(labels == 1))
            self.q = nt / (ns + nt)
            self.source_rows = X[:ns].copy()
            np.testing.assert_array_equal(X[ns:], view.X_target_adapt)
            np.testing.assert_array_equal(labels, [0] * ns + [1] * nt)
            fit_calls.append((ns, nt))
            return self

        def predict_proba(self, held):
            assert not set(held[:, 0]) & set(self.source_rows[:, 0])
            return np.column_stack((np.full(len(held), 1 - self.q), np.full(len(held), self.q)))

    monkeypatch.setattr("shiftlab.adaptation.importance.LogisticRegression", ConstantPrior)
    state = fit_importance_weights(view, folds, PARAMETERS, 17)
    assert len(fit_calls) == 5
    np.testing.assert_allclose(state.raw_weights, 1.0)
    np.testing.assert_allclose(state.weights, 1.0)
    for number, (train, held) in enumerate(folds.indices):
        np.testing.assert_array_equal(state.source_fit_counts[held], len(train))
        np.testing.assert_array_equal(state.fold_assignment[held], number)
    assert not state.weights.flags.writeable


def test_near_identical_domains_have_unit_ratios_without_score_improvement_claim():
    view, folds = grouped_fixture()
    # Identical constant feature distributions with unequal domain sizes.
    view = replace(
        view,
        X_source=np.zeros_like(view.X_source),
        X_target_adapt=np.zeros_like(view.X_target_adapt),
    )
    state = fit_importance_weights(view, folds, PARAMETERS, 17)
    np.testing.assert_allclose(state.raw_weights, 1.0, atol=0.01)
    assert state.diagnostics["ess_fraction"] > 0.99


@pytest.mark.parametrize("problem", ["repeat", "missing", "group_leak", "not_complement"])
def test_invalid_crossfit_partitions_fail(problem):
    view, folds = grouped_fixture()
    splits = list(folds.indices)
    if problem == "repeat":
        splits[0] = splits[1]
    elif problem == "missing":
        splits = splits[:-1]
    elif problem == "not_complement":
        train, valid = splits[0]
        splits[0] = (train[:-1], valid)
    else:
        view = replace(view, source_group_ids=("same-person",) * len(view.source_ids))
    with pytest.raises(ValueError):
        validate_crossfit_folds(view, replace(folds, indices=tuple(splits)))


def test_api_has_no_target_label_or_evaluation_channel():
    assert not {"y_target", "target_labels", "target_eval"} & set(
        inspect.signature(fit_importance_weights).parameters
    )
    assert not {"y_target", "target_labels", "target_eval"} & {
        field.name for field in fields(TrainingView)
    }
    view, folds = grouped_fixture()
    with pytest.raises(TypeError):
        fit_importance_weights(view, folds, PARAMETERS, 17, y_target=np.ones(11))


def test_failed_discriminator_keeps_completed_fold_evidence(tmp_path, monkeypatch):
    view, folds = grouped_fixture()
    from shiftlab.models.estimators import fit_estimator

    calls = 0

    def fail_third(estimator, X, labels):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("synthetic domain solver failure")
        return fit_estimator(estimator, X, labels)

    monkeypatch.setattr("shiftlab.adaptation.importance.fit_estimator", fail_third)
    store = LocalArtifactStore(tmp_path)

    def produce(stage):
        fit_importance_weights(
            view,
            folds,
            PARAMETERS,
            17,
            on_fold=lambda reference: write_json(
                stage / f"domain_fold_{reference['fold']}.json", reference
            ),
        )
        pytest.fail("A failed discriminator reached successful publication")

    with pytest.raises(RuntimeError, match="fold 2 failed"):
        store.create("importance_cache", {"synthetic_fixture": True}, produce)
    failures = list((tmp_path / "failures").rglob("manifest.json"))
    assert len(failures) == 1 and read_json(failures[0])["status"] == "failed"
    directory = failures[0].parent
    assert (directory / "domain_fold_0.json").is_file()
    assert (directory / "domain_fold_1.json").is_file()
    assert not (directory / "domain_fold_2.json").exists()
    assert not list((tmp_path / "locks").iterdir())


def test_weighted_job_delivers_exact_saved_weights_and_preserves_baselines(
    baseline_split,
    tmp_path,
    monkeypatch,
    smoke_dict,
):
    smoke_dict.update(
        models=["lr", "linear_svm", "rf"],
        methods=["source_only", "importance_weighting"],
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

    delivered = []

    def spy(estimator, X, labels, *, sample_weight=None):
        if sample_weight is not None:
            delivered.append((type(estimator).__name__, sample_weight.copy()))
        return fit_estimator(estimator, X, labels, sample_weight=sample_weight)

    monkeypatch.setattr("shiftlab.experiments.runner.fit_estimator", spy)
    # Corrupt only synthetic held-out data; training must neither open nor checksum it.
    (baseline_split / "evaluation/target_eval.npz").write_bytes(
        b"unreadable synthetic target features"
    )
    (baseline_split / "evaluation/target_eval_labels.npy").write_bytes(
        b"unreadable synthetic labels"
    )
    builtin_open, path_open = builtins.open, Path.open

    def guard(path):
        if isinstance(path, (str, Path)) and {"evaluation", "private"} & set(Path(path).parts):
            pytest.fail("Fitting opened forbidden held-out data")

    def open_builtin(path, *args, **kwargs):
        guard(path)
        return builtin_open(path, *args, **kwargs)

    def open_path(path, *args, **kwargs):
        guard(path)
        return path_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", open_builtin)
    monkeypatch.setattr(Path, "open", open_path)
    result = train_methods(config, root=tmp_path, data_root=tmp_path / "data")
    assert result["fitted_records"] == result["expected_fits"] == 7
    assert {name for name, _ in delivered} == {
        "LogisticRegression",
        "LinearSVC",
        "RandomForestClassifier",
    }
    assert len(delivered) == 3
    store = LocalArtifactStore(tmp_path / "artifacts")
    source_ids = set()
    importance_ids = set()
    for row in result["runs"]:
        source_ids.add(row["source_cv_id"])
        run = store.directory("runs", row["run_id"])
        diagnostic = read_json(run / "diagnostics.json")
        if row["method"] == "importance_weighting":
            importance_ids.add(row["importance_cache_id"])
            with np.load(run / "adaptation_state.npz", allow_pickle=False) as arrays:
                weights = arrays["weights"]
            np.testing.assert_array_equal(weights, delivered[0][1])
            assert diagnostic["sample_weight_hash"] == content_hash(weights.tolist())
            assert diagnostic["sample_weight_passed"]
        else:
            assert not diagnostic["sample_weight_passed"]
            assert not (run / "adaptation_state.npz").exists()
        assert not (run / "metrics.json").exists()
    assert len(source_ids) == len(importance_ids) == 1
    # A second alteration of forbidden synthetic inputs cannot affect saved weights/models.
    for name in ("target_eval.npz", "target_eval_labels.npy"):
        with path_open(baseline_split / "evaluation" / name, "wb") as stream:
            stream.write(b"second intentional synthetic corruption")
    files = {
        p: (file_sha256(p), p.stat().st_mtime_ns)
        for p in (tmp_path / "artifacts").rglob("*")
        if p.is_file()
    }
    again = train_methods(config, root=tmp_path, data_root=tmp_path / "data")
    assert again["new_fits"] == 0 and again["resumed_fits"] == 7
    assert files == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in files}
    assert len(delivered) == 3, "Resume refitted weighted task models"
