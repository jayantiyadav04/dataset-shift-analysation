"""Source-out-of-fold density ratios; target task labels have no input channel."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from shiftlab.contracts import TrainingView
from shiftlab.experiments.config import ImportanceConfig
from shiftlab.features.preprocessing import SourceFolds
from shiftlab.models.estimators import fit_estimator

DOMAIN_MAX_ITER = 10000


@dataclass(frozen=True)
class ImportanceState:
    probabilities: np.ndarray
    bounded_probabilities: np.ndarray
    source_fit_counts: np.ndarray
    raw_weights: np.ndarray
    clipped_weights: np.ndarray
    weights: np.ndarray
    fold_assignment: np.ndarray
    diagnostics: dict
    folds: tuple[dict, ...]


def density_ratios(
    probabilities: np.ndarray,
    source_fit_counts: np.ndarray,
    target_count: int,
    parameters: ImportanceConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Prior correction is per row's fitting fold, then clip, then normalize globally."""
    q = np.asarray(probabilities, dtype=np.float64)
    counts = np.asarray(source_fit_counts)
    if (
        q.ndim != 1
        or not q.size
        or not np.isfinite(q).all()
        or np.any((q < 0) | (q > 1))
        or counts.shape != q.shape
        or counts.dtype.kind not in "iu"
        or np.any(counts <= 0)
        or type(target_count) is not int
        or target_count < 1
    ):
        raise ValueError("Probabilities and actual positive domain counts must align")
    bounded = np.clip(q, parameters.probability_epsilon, 1 - parameters.probability_epsilon)
    raw = bounded / (1 - bounded) * (counts.astype(np.float64) / target_count)
    clipped = np.clip(raw, parameters.weight_clip_lower, parameters.weight_clip_upper)
    weights = clipped / clipped.mean()
    if not np.isfinite(raw).all() or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Density-ratio computation produced invalid weights")
    return bounded, raw, clipped, weights


def weight_diagnostics(
    probabilities: np.ndarray,
    bounded: np.ndarray,
    raw: np.ndarray,
    clipped: np.ndarray,
    weights: np.ndarray,
    parameters: ImportanceConfig,
) -> dict:
    ess = float(weights.sum() ** 2 / np.square(weights).sum())
    fraction = ess / len(weights)
    warnings = []
    if fraction < 0.1:
        warnings.append("Effective sample size below 10% of source count; retain this run")
    return {
        "source_count": len(weights),
        "min": float(weights.min()),
        "max": float(weights.max()),
        "mean": float(weights.mean()),
        "median": float(np.median(weights)),
        "quantiles": {
            str(q): float(np.quantile(weights, q)) for q in (0.01, 0.05, 0.25, 0.75, 0.95, 0.99)
        },
        "raw_weight_min": float(raw.min()),
        "raw_weight_max": float(raw.max()),
        "clipped_weight_mean_before_normalization": float(clipped.mean()),
        "probability_clipping_fraction": float(np.mean(probabilities != bounded)),
        "weight_clipping_fraction": float(np.mean(raw != clipped)),
        "lower_clipping_fraction": float(np.mean(raw < parameters.weight_clip_lower)),
        "upper_clipping_fraction": float(np.mean(raw > parameters.weight_clip_upper)),
        "ess": ess,
        "ess_fraction": fraction,
        "warnings": warnings,
        "normalization": "Global source mean one after raw clipping; normalized bounds can differ",
        "parameters": parameters.model_dump(mode="json"),
        "domain_labels": {"source": 0, "target_adapt": 1},
        "domain_max_iter": DOMAIN_MAX_ITER,
        "domain_class_weight": None,
        "domain_resampling": False,
        "target_task_labels_used": False,
        "target_evaluation_opened": False,
    }


def validate_crossfit_folds(view: TrainingView, folds: SourceFolds) -> None:
    """Every source row is held out once; every fit uses precisely its complement."""
    size = len(view.source_ids)
    if len(folds.indices) != 5 or len(folds.groups) != size:
        raise ValueError("Importance weighting requires five aligned source folds")
    assignments = np.zeros(size, dtype=int)
    all_rows = set(range(size))
    group_sets = [folds.groups]
    if view.source_group_ids is not None:
        group_sets.append(view.source_group_ids)
    for train, valid in folds.indices:
        for indices in (train, valid):
            if (
                not isinstance(indices, np.ndarray)
                or indices.ndim != 1
                or indices.dtype.kind not in "iu"
                or not len(indices)
                or np.any(indices < 0)
                or np.any(indices >= size)
                or len(set(indices.tolist())) != len(indices)
            ):
                raise ValueError("Crossfit indices must be unique nonempty source positions")
        if set(train.tolist()) != all_rows - set(valid.tolist()):
            raise ValueError("Crossfit train must be exactly the held-out fold complement")
        for groups in group_sets:
            if {groups[i] for i in train} & {groups[i] for i in valid}:
                raise ValueError("Crossfit source groups overlap")
        assignments[valid] += 1
    if not np.all(assignments == 1):
        raise ValueError("Every source row must receive one out-of-fold probability")


def fit_importance_weights(
    view: TrainingView,
    folds: SourceFolds,
    parameters: ImportanceConfig,
    seed: int,
    *,
    on_fold: Callable[[dict], None] | None = None,
) -> ImportanceState:
    """Fit discriminator folds in the common full-source-fitted representation.

    Source task labels are allowed in TrainingView and source fold construction;
    only domain-membership labels are passed to the discriminator. No task labels
    from either target pool or evaluation features are accepted.
    """
    if not isinstance(view, TrainingView):
        raise ValueError("Importance fitting requires the label-isolated training contract")
    validate_crossfit_folds(view, folds)
    nt, size = len(view.target_adapt_ids), len(view.source_ids)
    probabilities = np.empty(size, dtype=np.float64)
    counts = np.empty(size, dtype=np.int64)
    assignment = np.empty(size, dtype=np.int64)
    references, messages = [], []
    for number, (train, valid) in enumerate(folds.indices):
        ns = len(train)
        X = np.concatenate((view.X_source[train], view.X_target_adapt))
        membership = np.concatenate((np.zeros(ns, dtype=np.int64), np.ones(nt, dtype=np.int64)))
        discriminator = LogisticRegression(
            C=parameters.domain_C,
            class_weight=None,
            max_iter=DOMAIN_MAX_ITER,
            random_state=seed,
        )
        try:
            warnings = fit_estimator(discriminator, X, membership)
        except RuntimeError as error:
            raise RuntimeError(f"Domain-classifier fold {number} failed: {error}") from error
        messages.extend(warnings)
        with threadpool_limits(limits=1):
            q = discriminator.predict_proba(view.X_source[valid])[:, 1]
        probabilities[valid], counts[valid], assignment[valid] = q, ns, number
        reference = {
            "fold": number,
            "source_fit_count": ns,
            "target_adapt_fit_count": nt,
            "prior_factor": ns / nt,
            "source_fit_ids": [view.source_ids[i] for i in train],
            "source_heldout_ids": [view.source_ids[i] for i in valid],
            "source_fit_groups": [folds.groups[i] for i in train],
            "source_heldout_groups": [folds.groups[i] for i in valid],
            "target_adapt_ids": view.target_adapt_ids,
            "domain_classes": discriminator.classes_.tolist(),
            "coefficient": discriminator.coef_.tolist(),
            "intercept": discriminator.intercept_.tolist(),
            "n_iter": discriminator.n_iter_.tolist(),
            "warnings": warnings,
        }
        references.append(reference)
        if on_fold is not None:
            on_fold(reference)
    bounded, raw, clipped, weights = density_ratios(probabilities, counts, nt, parameters)
    diagnostics = weight_diagnostics(probabilities, bounded, raw, clipped, weights, parameters)
    diagnostics["warnings"] += messages
    diagnostics.update(
        {
            "crossfit_strategy": folds.strategy,
            "crossfit_group_note": folds.note,
            "target_adapt_count": nt,
            "seed": seed,
        }
    )
    for values in (probabilities, bounded, counts, raw, clipped, weights, assignment):
        values.setflags(write=False)
    return ImportanceState(
        probabilities,
        bounded,
        counts,
        raw,
        clipped,
        weights,
        assignment,
        diagnostics,
        tuple(references),
    )
