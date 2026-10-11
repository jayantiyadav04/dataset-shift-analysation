"""Fixed classical factories and a source-frequency sanity baseline."""

import warnings
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.utils.validation import check_is_fitted
from threadpoolctl import threadpool_limits


class MajorityClassifier(ClassifierMixin, BaseEstimator):
    """Lowest canonical integer class wins an exact source-count tie."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MajorityClassifier":
        labels = np.asarray(y)
        if labels.dtype.kind not in "iu" or labels.shape != (len(X),) or not len(labels):
            raise ValueError("Majority needs aligned nonempty integer source labels")
        self.classes_, counts = np.unique(labels, return_counts=True)
        self.majority_class_ = self.classes_[int(np.argmax(counts))]
        self.n_features_in_ = X.shape[1]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self)
        if X.ndim != 2 or X.shape[1] != self.n_features_in_ or not np.isfinite(X).all():
            raise ValueError("Majority prediction feature schema differs")
        return np.full(len(X), self.majority_class_, dtype=np.int64)


def estimator_factory(model: str, parameters: dict[str, Any], seed: int) -> Any:
    """No target data, class balancing, hidden search or parallel forest fits."""
    if model == "majority" and not parameters:
        return MajorityClassifier()
    if model in ("lr", "linear_svm") and set(parameters) == {"C"}:
        if parameters["C"] not in (0.1, 1.0, 10.0):
            raise ValueError("C is outside the frozen source grid")
        if model == "lr":
            return LogisticRegression(**parameters, max_iter=3000, random_state=seed)
        return LinearSVC(**parameters, dual="auto", max_iter=10000, random_state=seed)
    if model == "rf" and set(parameters) == {"n_estimators", "max_depth", "min_samples_leaf"}:
        if (
            parameters["n_estimators"] != 200
            or parameters["max_depth"] not in (None, 20)
            or parameters["min_samples_leaf"] not in (1, 3)
        ):
            raise ValueError("Forest parameters are outside the frozen source grid")
        return RandomForestClassifier(**parameters, random_state=seed, n_jobs=1)
    raise ValueError(f"Unsupported estimator or parameter fields: {model}")


def fit_estimator(
    estimator: Any,
    X_source: np.ndarray,
    y_source: np.ndarray,
    *,
    sample_weight: np.ndarray | None = None,
) -> list[str]:
    """Capture solver messages; a convergence failure cannot become a fitted run."""
    if X_source.ndim != 2 or not np.isfinite(X_source).all():
        raise ValueError("Task features must be a finite dense matrix")
    if np.asarray(y_source).shape != (len(X_source),):
        raise ValueError("Source labels must align with task features")
    kwargs = {}
    if sample_weight is not None:
        weights = np.asarray(sample_weight, dtype=np.float64)
        if (
            weights.shape != (len(X_source),)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError("Weights must be finite, positive and aligned with source rows")
        kwargs["sample_weight"] = weights
    with threadpool_limits(limits=1), warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(X_source, y_source, **kwargs)
    messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        raise RuntimeError("Unresolved source convergence: " + "; ".join(messages))
    return messages
