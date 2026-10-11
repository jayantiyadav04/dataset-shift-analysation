"""Paired test-unit uncertainty, conditional on the five fitted seeds and domains."""

import numpy as np


def scores_from_confusion(matrix: np.ndarray, macro_positions: np.ndarray) -> dict[str, np.ndarray]:
    cm = np.asarray(matrix, dtype=np.float64)
    tp = np.diagonal(cm, axis1=-2, axis2=-1)
    actual, predicted = cm.sum(axis=-1), cm.sum(axis=-2)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    f1 = np.divide(
        2 * tp, actual + predicted, out=np.zeros_like(tp), where=(actual + predicted) > 0
    )
    return {
        "accuracy": tp.sum(axis=-1) / cm.sum(axis=(-2, -1)),
        "macro_f1": f1[..., macro_positions].mean(axis=-1),
        "macro_precision": precision[..., macro_positions].mean(axis=-1),
        "macro_recall": recall[..., macro_positions].mean(axis=-1),
    }


def confusion_by_group(
    true: np.ndarray,
    predictions: np.ndarray,
    groups: tuple[str, ...],
    known_classes: tuple[int, ...],
) -> tuple[np.ndarray, tuple[str, ...]]:
    y, pred = np.asarray(true), np.asarray(predictions)
    if (
        y.ndim != 1
        or not len(y)
        or pred.ndim != 2
        or pred.shape[1] != len(y)
        or len(groups) != len(y)
    ):
        raise ValueError("Group bootstrap requires aligned labels and seed-by-sample predictions")
    if (
        y.dtype.kind not in "iu"
        or pred.dtype.kind not in "iu"
        or not set(y) | set(pred.ravel()) <= set(known_classes)
    ):
        raise ValueError("Bootstrap labels must use the canonical integer mapping")
    if any(not isinstance(g, str) or not g for g in groups):
        raise ValueError("Bootstrap group IDs must be nonempty")
    units = tuple(sorted(set(groups)))
    positions = {group: i for i, group in enumerate(units)}
    labels = {label: i for i, label in enumerate(known_classes)}
    g = np.array([positions[value] for value in groups])
    t = np.array([labels[int(value)] for value in y])
    c = len(known_classes)
    counts = np.zeros((len(units), len(pred), c, c), dtype=np.int64)
    for seed, values in enumerate(pred):
        p = np.array([labels[int(value)] for value in values])
        counts[:, seed] = np.bincount(g * c * c + t * c + p, minlength=len(units) * c * c).reshape(
            len(units), c, c
        )
    return counts, units


def paired_group_bootstrap(
    true: np.ndarray,
    baseline: np.ndarray,
    adapted: np.ndarray,
    groups: tuple[str, ...],
    *,
    known_classes: tuple[int, ...],
    replicates: int = 1000,
    seed: int = 2026,
) -> dict:
    if baseline.shape != adapted.shape or baseline.shape[0] != 5:
        raise ValueError("Uncertainty requires all five aligned fitted seeds")
    base, units = confusion_by_group(true, baseline, groups, known_classes)
    other, other_units = confusion_by_group(true, adapted, groups, known_classes)
    if units != other_units:
        raise ValueError("Paired evaluation groups differ")
    if len(units) < 2:
        return {
            "status": "insufficient_independent_units",
            "bootstrap_replicates": replicates,
            "bootstrap_seed": seed,
            "usable_replicates": 0,
            "group_count": len(units),
            "sample_count": len(true),
            "class_set": np.unique(true).tolist(),
            "conditional_on": "A single evaluation group cannot provide group uncertainty",
            "accuracy_lower": None,
            "accuracy_upper": None,
            "macro_f1_lower": None,
            "macro_f1_upper": None,
        }
    macro_positions = np.array([known_classes.index(int(v)) for v in np.unique(true)])
    rng = np.random.default_rng(seed)
    values = {metric: [] for metric in ("accuracy", "macro_f1")}
    # Multinomial counts represent sampling whole groups WITH multiplicity.
    # Chunking limits memory; identical draws are applied to both methods/seeds.
    for start in range(0, replicates, 100):
        counts = rng.multinomial(
            len(units), np.full(len(units), 1 / len(units)), size=min(100, replicates - start)
        )
        b = scores_from_confusion(np.einsum("rg,gscd->rscd", counts, base), macro_positions)
        a = scores_from_confusion(np.einsum("rg,gscd->rscd", counts, other), macro_positions)
        for metric in values:
            values[metric].extend((a[metric] - b[metric]).mean(axis=1).tolist())
    result = {
        "status": "descriptive_interval",
        "bootstrap_replicates": replicates,
        "bootstrap_seed": seed,
        "usable_replicates": replicates,
        "group_count": len(units),
        "sample_count": len(true),
        "class_set": np.unique(true).tolist(),
        "conditional_on": "Fitted models and observed domains; "
        "seeds share the same test units; intervals unadjusted",
    }
    for metric, deltas in values.items():
        lower, upper = np.percentile(deltas, [2.5, 97.5])
        result[f"{metric}_lower"] = float(lower)
        result[f"{metric}_upper"] = float(upper)
    return result


def evaluation_units(split: dict) -> tuple[tuple[str, ...], str]:
    if split["dataset_id"] == "amazon":
        products = split.get("target_eval_product_ids", [])
        if len(products) == len(split["target_eval_ids"]) and all(
            isinstance(p, str) and p for p in products
        ):
            return tuple(products), "Amazon product groups; canonical review bodies deduplicated"
        return tuple(
            split["target_eval_group_ids"]
        ), "Amazon canonical review groups; reliable product IDs unavailable, independence assumed"
    if split["dataset_id"] == "har":
        return tuple(
            split["target_eval_group_ids"]
        ), "HAR whole participants (five); all windows and resampled multiplicity retained"
    return tuple(
        split["target_eval_group_ids"]
    ), "Gas duplicate-safe measurement groups; within-batch temporal dependence may remain"
