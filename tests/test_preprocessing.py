"""Source-fitting permission, fold isolation, finite shared representations."""

import inspect

import numpy as np
import pytest

from shiftlab.data.storage import DataError
from shiftlab.data.training import FeatureBatch, RawTrainingView
from shiftlab.features.preprocessing import SourcePreprocessor, fit_source_fold, source_folds


def batch(dataset="gas", role="source", count=8, offset=0):
    ids = tuple(f"{role}:{i}" for i in range(count))
    if dataset == "amazon":
        texts = (
            "good quality item",
            "good quality excellent",
            "bad quality item",
            "bad quality poor",
        ) * 2
        values = texts[:count]
    else:
        features = 128 if dataset == "gas" else 561
        values = np.asarray([[offset + i + j / 100 for j in range(features)] for i in range(count)])
    return FeatureBatch(dataset, role, ids, ids, values)


@pytest.mark.parametrize("role", ["target_adapt", "target_eval", "source_validation"])
def test_fit_rejects_non_source_roles(role):
    with pytest.raises(DataError, match="source training"):
        SourcePreprocessor().fit(batch(role=role))


def test_scaling_fits_only_source_fold_and_transform_does_not_refit():
    source = batch()
    train, valid = np.array([0, 1, 2, 3]), np.array([4, 5, 6, 7])
    pre, transformed_train, transformed_valid = fit_source_fold(source, train, valid)
    np.testing.assert_allclose(pre.scaler_.mean_, source.values[train].mean(axis=0))
    np.testing.assert_allclose(transformed_train.mean(axis=0), 0, atol=1e-12)
    mean = pre.scaler_.mean_.copy()
    adapt = pre.transform(batch(role="target_adapt", offset=1000))
    assert np.isfinite(adapt).all() and np.isfinite(transformed_valid).all()
    np.testing.assert_array_equal(mean, pre.scaler_.mean_)
    assert pre.fit_ids_ == source.ids[:4]
    with pytest.raises(DataError, match="fresh preprocessor"):
        pre.fit(source)


def test_validation_and_adaptation_tokens_never_enter_tfidf_vocabulary():
    texts = (
        "good quality item",
        "good quality excellent",
        "bad quality item",
        "bad quality poor",
        "good quality item",
        "bad quality item",
        "validationonly secret good",
        "validationonly secret bad",
    )
    source = FeatureBatch(
        "amazon",
        "source",
        tuple(f"s{i}" for i in range(8)),
        tuple(f"g{i}" for i in range(8)),
        texts,
    )
    pre, X_train, X_valid = fit_source_fold(source, np.arange(6), np.array([6, 7]))
    assert "validationonly" not in pre.vectorizer_.vocabulary_
    assert "secret" not in pre.vectorizer_.vocabulary_
    target = FeatureBatch(
        "amazon",
        "target_adapt",
        ("a", "b"),
        ("a", "b"),
        ("adaptationonly words", "adaptationonly other"),
    )
    X_adapt = pre.transform(target)
    assert "adaptationonly" not in pre.vectorizer_.vocabulary_
    assert X_train.shape[1] == X_valid.shape[1] == X_adapt.shape[1] == pre.actual_components_
    assert pre.actual_components_ < len(source.ids[:6])
    assert np.isfinite(X_adapt).all()
    np.testing.assert_allclose(X_train.mean(axis=0), 0, atol=1e-12)


def test_source_fold_overlap_and_nonfinite_input_fail():
    source = batch()
    with pytest.raises(DataError, match="leakage"):
        fit_source_fold(source, np.array([0, 1]), np.array([1, 2]))
    values = source.values.copy()
    values[0, 0] = np.nan
    with pytest.raises(DataError, match="finite"):
        FeatureBatch("gas", "source", source.ids, source.groups, values)


def test_source_folds_keep_har_people_whole_and_amazon_products_when_feasible():
    for dataset in ("har", "amazon"):
        if dataset == "har":
            people = tuple(f"person:{i // 6}" for i in range(60))
            ids = tuple(f"s{i}" for i in range(60))
            source = FeatureBatch("har", "source", ids, people, np.ones((60, 561)))
            labels = np.tile(np.arange(6), 10)
            products = (None,) * 60
        else:
            ids = tuple(f"s{i}" for i in range(60))
            source = FeatureBatch("amazon", "source", ids, ids, ("good text", "bad text") * 30)
            labels = np.tile(np.arange(2), 30)
            products = tuple(f"B{i // 2:09d}" for i in range(60))
        target = batch(dataset, "target_adapt", count=2)
        view = RawTrainingView(source, labels, target, products, "a" * 64)
        folds = source_folds(view)
        assert len(folds.indices) == 5
        for train, valid in folds.indices:
            assert not set(np.asarray(folds.groups)[train]) & set(np.asarray(folds.groups)[valid])
            assert set(labels[train]) == set(labels[valid])
        if dataset == "amazon":
            assert all(group.startswith("product:") for group in folds.groups)


def test_no_target_labels_in_preprocessing_api():
    assert list(inspect.signature(SourcePreprocessor.fit).parameters) == ["self", "source"]
    assert "y_target" not in RawTrainingView.__dataclass_fields__
    assert "target_eval" not in RawTrainingView.__dataclass_fields__


def test_preprocessing_identity_tracks_source_content_not_just_ids():
    source = batch()
    changed = FeatureBatch(
        source.dataset, source.role, source.ids, source.groups, source.values + 100
    )
    original_pre = SourcePreprocessor().fit(source)
    changed_pre = SourcePreprocessor().fit(changed)
    assert original_pre.preprocessing_hash_ != changed_pre.preprocessing_hash_
