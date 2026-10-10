"""Explicit array permissions and names shared by CLI, engine and interface."""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
from numpy.typing import NDArray

FeatureArray = NDArray[np.float64]
LabelArray = NDArray[np.int64]


class DatasetName(StrEnum):
    GAS = "gas"
    HAR = "har"
    AMAZON = "amazon"


class ModelName(StrEnum):
    LR = "lr"
    LINEAR_SVM = "linear_svm"
    RF = "rf"


class MethodName(StrEnum):
    SOURCE_ONLY = "source_only"
    IMPORTANCE_WEIGHTING = "importance_weighting"
    CORAL = "coral"


def _features(value: FeatureArray, name: str) -> FeatureArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must be a two-dimensional matrix with at least two rows")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} contains non-finite values")
    array.setflags(write=False)
    return array


def _ids(values: tuple[str, ...], size: int, name: str) -> tuple[str, ...]:
    values = tuple(values)
    if len(values) != size or any(not isinstance(v, str) or not v.strip() for v in values):
        raise ValueError(f"{name} must contain one nonempty string per row")
    if len(set(values)) != size:
        raise ValueError(f"{name} contains duplicate sample IDs")
    return values


@dataclass(frozen=True)
class TrainingView:
    """Dense fitting input. There are no target task labels or evaluation arrays.

    Feature columns share one source-fitted schema. Canonical source labels are
    one-dimensional integers; sample identities follow row order. The contract
    copies arrays to detach them from caller buffers and marks them read-only.
    """

    X_source: FeatureArray
    y_source: LabelArray
    X_target_adapt: FeatureArray
    source_ids: tuple[str, ...]
    target_adapt_ids: tuple[str, ...]
    source_group_ids: tuple[str, ...] | None = None
    target_adapt_group_ids: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        source = _features(self.X_source, "X_source")
        target = _features(self.X_target_adapt, "X_target_adapt")
        if source.shape[1] != target.shape[1]:
            raise ValueError("Source and adaptation features must have the same column count")
        labels = np.asarray(self.y_source)
        if labels.ndim != 1 or labels.size != source.shape[0] or labels.dtype.kind not in "iu":
            raise ValueError("y_source must be a one-dimensional integer label per source row")
        if np.unique(labels).size < 2:
            raise ValueError("Source classification requires at least two classes")
        if labels.min() < 0 or labels.max() > np.iinfo(np.int64).max:
            raise ValueError("Canonical source labels must be nonnegative int64 values")
        labels = np.array(labels, dtype=np.int64, copy=True)
        labels.setflags(write=False)
        source_ids = _ids(self.source_ids, len(source), "source_ids")
        target_ids = _ids(self.target_adapt_ids, len(target), "target_adapt_ids")
        if set(source_ids) & set(target_ids):
            raise ValueError("Source and target adaptation sample IDs overlap")
        groups = (self.source_group_ids, self.target_adapt_group_ids)
        for values, size in zip(groups, (len(source), len(target)), strict=True):
            if values is not None and (
                len(values) != size or any(not isinstance(v, str) or not v.strip() for v in values)
            ):
                raise ValueError("Group IDs must contain one nonempty string per row")
        if all(g is not None for g in groups) and set(groups[0]) & set(groups[1]):
            raise ValueError("Source and target adaptation groups overlap")
        for name, value in (
            ("X_source", source),
            ("y_source", labels),
            ("X_target_adapt", target),
            ("source_ids", source_ids),
            ("target_adapt_ids", target_ids),
        ):
            object.__setattr__(self, name, value)
        for name in ("source_group_ids", "target_adapt_group_ids"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, tuple(value))
