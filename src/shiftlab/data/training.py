"""Role-isolated training access: never opens evaluation files or global labels."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np

from shiftlab.data.parsers import FEATURE_COUNTS
from shiftlab.data.splits import read_role_features, validate_split
from shiftlab.data.storage import DataError, read_json, verify_files

Role = Literal["source", "source_validation", "target_adapt", "target_eval"]


@dataclass(frozen=True)
class FeatureBatch:
    dataset: str
    role: Role
    ids: tuple[str, ...]
    groups: tuple[str, ...]
    values: np.ndarray | tuple[str, ...]

    def __post_init__(self) -> None:
        if self.dataset not in FEATURE_COUNTS or self.role not in {
            "source",
            "source_validation",
            "target_adapt",
            "target_eval",
        }:
            raise DataError("Unknown dataset or feature role")
        ids, groups = tuple(self.ids), tuple(self.groups)
        if (
            not ids
            or len(ids) != len(set(ids))
            or len(ids) != len(groups)
            or len(ids) != len(self.values)
        ):
            raise DataError("Feature rows, IDs and groups must align uniquely")
        if any(not isinstance(value, str) or not value for value in ids + groups):
            raise DataError("Empty sample/group identity")
        if self.dataset == "amazon":
            if not isinstance(self.values, (tuple, list)) or any(
                not isinstance(v, str) or not v.strip() for v in self.values
            ):
                raise DataError("Text features must be a sequence of nonempty bodies")
            values = tuple(self.values)
        else:
            values = np.array(self.values, dtype=np.float64, copy=True)
            if (
                values.shape != (len(ids), FEATURE_COUNTS[self.dataset])
                or not np.isfinite(values).all()
            ):
                raise DataError("Numerical feature shape/finite contract failed")
            values.setflags(write=False)
        object.__setattr__(self, "ids", ids)
        object.__setattr__(self, "groups", groups)
        object.__setattr__(self, "values", values)

    def subset(self, positions: np.ndarray, *, role: Role) -> "FeatureBatch":
        if self.role != "source" or role not in ("source", "source_validation"):
            raise DataError("Only source data may be partitioned into validation folds")
        indices = np.asarray(positions)
        if (
            indices.ndim != 1
            or indices.dtype.kind not in "iu"
            or np.any(indices < 0)
            or np.any(indices >= len(self.ids))
        ):
            raise DataError("Invalid source fold positions")
        values = (
            self.values[indices]
            if isinstance(self.values, np.ndarray)
            else tuple(self.values[i] for i in indices)
        )
        return FeatureBatch(
            self.dataset,
            role,
            tuple(self.ids[i] for i in indices),
            tuple(self.groups[i] for i in indices),
            values,
        )


@dataclass(frozen=True)
class RawTrainingView:
    source: FeatureBatch
    y_source: np.ndarray
    target_adapt: FeatureBatch
    source_product_ids: tuple[str | None, ...]
    split_hash: str

    def __post_init__(self) -> None:
        if (
            self.source.role != "source"
            or self.target_adapt.role != "target_adapt"
            or self.source.dataset != self.target_adapt.dataset
        ):
            raise DataError("Training roles/datasets differ")
        if set(self.source.ids) & set(self.target_adapt.ids) or set(self.source.groups) & set(
            self.target_adapt.groups
        ):
            raise DataError("Training roles overlap")
        labels = np.asarray(self.y_source)
        if (
            labels.dtype.kind not in "iu"
            or labels.shape != (len(self.source.ids),)
            or len(np.unique(labels)) < 2
        ):
            raise DataError("Invalid source labels")
        if labels.min() < 0 or labels.max() >= (2 if self.source.dataset == "amazon" else 6):
            raise DataError("Source label mapping differs")
        if len(self.source_product_ids) != len(labels):
            raise DataError("Product grouping must align with source rows")
        labels = labels.astype(np.int64, copy=True)
        labels.setflags(write=False)
        object.__setattr__(self, "y_source", labels)
        object.__setattr__(self, "source_product_ids", tuple(self.source_product_ids))


def load_training(directory: Path) -> RawTrainingView:
    manifest = read_json(directory / "manifest.json")
    validate_split(manifest)
    dataset = manifest["dataset_id"]
    extension = ".jsonl" if dataset == "amazon" else ".npz"
    allowed = {
        "source": f"training/source{extension}",
        "target_adapt": f"training/target_adapt{extension}",
        "source_labels": "training/source_labels.npy",
    }
    for role, name in allowed.items():
        if manifest["artifacts"][role] != name:
            raise DataError("Training artifact path is not in its permitted role")
    # Do not checksum/open evaluation artifacts here: only the evaluator/auditor may do so.
    verify_files(directory, {name: manifest["checksums"][name] for name in allowed.values()})
    batches = {}
    for role in ("source", "target_adapt"):
        ids, values = read_role_features(directory, allowed[role])
        if list(ids) != manifest[f"{role}_ids"]:
            raise DataError("Training artifact sample order differs")
        batches[role] = FeatureBatch(
            dataset, role, ids, tuple(manifest[f"{role}_group_ids"]), values
        )
    labels = np.load(directory / allowed["source_labels"], allow_pickle=False)
    return RawTrainingView(
        batches["source"],
        labels,
        batches["target_adapt"],
        tuple(manifest["source_product_ids"]),
        manifest["manifest_hash"],
    )
