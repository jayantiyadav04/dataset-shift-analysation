"""Source-only feature fitting with explicit role and fold permissions."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from shiftlab.contracts import TrainingView
from shiftlab.data.storage import DataError
from shiftlab.data.training import FeatureBatch, RawTrainingView
from shiftlab.experiments.provenance import content_hash


class SourcePreprocessor:
    """One fit per instance. Every CV fold must create a fresh instance."""

    def __init__(self, seed: int = 17):
        self.seed = seed

    def fit(self, source: FeatureBatch) -> "SourcePreprocessor":
        if not isinstance(source, FeatureBatch) or source.role != "source":
            raise DataError("Preprocessing fit permits current source training rows only")
        if hasattr(self, "fit_ids_"):
            raise DataError("Create a fresh preprocessor for each source fold/refit")
        if len(source.ids) < 2:
            raise DataError("Preprocessing needs at least two source rows")
        self.dataset_ = source.dataset
        self.scaler_ = StandardScaler()
        if source.dataset == "amazon":
            self.vectorizer_ = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_features=20000,
                sublinear_tf=True,
                stop_words=None,
                dtype=np.float64,
            )
            sparse = self.vectorizer_.fit_transform(source.values)
            probe = min(129, *sparse.shape)
            if probe < 2:
                raise DataError("Source text has insufficient rank for dense SVD")
            self.svd_ = TruncatedSVD(n_components=probe, random_state=self.seed)
            with threadpool_limits(limits=1):
                dense = self.svd_.fit_transform(sparse)
            singular = self.svd_.singular_values_
            tolerance = np.finfo(np.float64).eps * max(sparse.shape) * singular[0]
            observed_rank = int(np.count_nonzero(singular > tolerance))
            self.actual_components_ = min(128, observed_rank - 1)
            if self.actual_components_ < 1:
                raise DataError(
                    "Source text rank must exceed one for the declared SVD representation"
                )
            self.rank_note_ = (
                f"{probe} singular values probed; dimension is below observed numerical rank"
            )
            dense = dense[:, : self.actual_components_]
        else:
            dense = source.values
            self.actual_components_ = dense.shape[1]
            self.rank_note_ = "Supplied numerical columns preserved"
        self.scaler_.fit(dense)
        self.fit_ids_ = source.ids
        self.provenance_ = {
            "fit_scope": "source_only",
            "fit_ids": source.ids,
            "dataset": source.dataset,
            "seed": self.seed,
            "actual_dimension": self.actual_components_,
            "rank_note": self.rank_note_,
            "recipe": "standard"
            if source.dataset != "amazon"
            else "tfidf-20000-min_df2-ngram1_2-sublinear-no_stopwords-svd128-standard",
            "source_features_sha256": content_hash(
                source.values.tolist() if isinstance(source.values, np.ndarray) else source.values
            ),
            "scaler_mean": self.scaler_.mean_.tolist(),
            "scaler_scale": self.scaler_.scale_.tolist(),
        }
        if source.dataset == "amazon":
            self.provenance_["vocabulary_sha256"] = content_hash(
                {token: int(index) for token, index in self.vectorizer_.vocabulary_.items()}
            )
            self.provenance_["idf_sha256"] = content_hash(self.vectorizer_.idf_.tolist())
            self.provenance_["svd_sha256"] = content_hash(
                self.svd_.components_[: self.actual_components_].tolist()
            )
        self.preprocessing_hash_ = content_hash(self.provenance_)
        return self

    def transform(self, batch: FeatureBatch) -> np.ndarray:
        if not hasattr(self, "fit_ids_"):
            raise DataError("Preprocessor has not been fitted")
        if batch.dataset != self.dataset_:
            raise DataError("Preprocessor dataset mismatch")
        if self.dataset_ == "amazon":
            with threadpool_limits(limits=1):
                dense = self.svd_.transform(self.vectorizer_.transform(batch.values))[
                    :, : self.actual_components_
                ]
        else:
            dense = batch.values
        result = self.scaler_.transform(dense)
        if (
            result.shape != (len(batch.ids), self.actual_components_)
            or not np.isfinite(result).all()
        ):
            raise DataError("Preprocessed feature shape/finite contract failed")
        return result


@dataclass(frozen=True)
class SourceFolds:
    indices: tuple[tuple[np.ndarray, np.ndarray], ...]
    groups: tuple[str, ...]
    strategy: str
    note: str


def source_folds(view: RawTrainingView, seed: int = 17) -> SourceFolds:
    """Five-fold source-only split; participant/body/product groups remain intact."""
    y = view.y_source
    groups = view.source.groups
    note = "Duplicate-safe groups"
    if view.source.dataset == "amazon" and all(view.source_product_ids):
        candidate = tuple(f"product:{pid}" for pid in view.source_product_ids)
        if all(
            len({candidate[i] for i in np.flatnonzero(y == label)}) >= 5 for label in np.unique(y)
        ):
            groups, note = (
                candidate,
                "All source ASINs valid and at least five product groups per class",
            )
        else:
            note = "Product grouping infeasible; canonical bodies used"
    elif view.source.dataset == "amazon":
        note = "Incomplete product IDs; canonical bodies used"
    if len(set(groups)) < 5:
        raise DataError("At least five source groups are required for source CV")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    indices = tuple(splitter.split(np.zeros(len(y)), y, groups))
    classes = set(y.tolist())
    feasible = all(
        set(y[train]) == classes and set(y[valid]) == classes for train, valid in indices
    )
    strategy = "StratifiedGroupKFold"
    if not feasible:
        if view.source.dataset != "har":
            raise DataError("Five-fold source stratification cannot retain every class")
        indices = tuple(GroupKFold(n_splits=5).split(np.zeros(len(y)), y, groups))
        if any(set(y[train]) != classes for train, _ in indices):
            raise DataError("No feasible five-fold HAR group alternative")
        strategy, note = (
            "GroupKFold",
            "Stratified group folds lacked class coverage; "
            "deterministic whole-participant alternative",
        )
    for train, valid in indices:
        if set(train) & set(valid) or set(np.asarray(groups)[train]) & set(
            np.asarray(groups)[valid]
        ):
            raise DataError("Source CV group leakage")
    return SourceFolds(indices, groups, strategy, note)


def fit_source_fold(
    source: FeatureBatch, train: np.ndarray, valid: np.ndarray, seed: int = 17
) -> tuple[SourcePreprocessor, np.ndarray, np.ndarray]:
    train_batch = source.subset(train, role="source")
    valid_batch = source.subset(valid, role="source_validation")
    if set(train_batch.ids) & set(valid_batch.ids) or set(train_batch.groups) & set(
        valid_batch.groups
    ):
        raise DataError("Source fold sample/group leakage")
    preprocessor = SourcePreprocessor(seed).fit(train_batch)
    return preprocessor, preprocessor.transform(train_batch), preprocessor.transform(valid_batch)


def preprocess_training(
    view: RawTrainingView, seed: int = 17
) -> tuple[SourcePreprocessor, TrainingView]:
    preprocessor = SourcePreprocessor(seed).fit(view.source)
    dense = TrainingView(
        preprocessor.transform(view.source),
        view.y_source,
        preprocessor.transform(view.target_adapt),
        view.source.ids,
        view.target_adapt.ids,
        view.source.groups,
        view.target_adapt.groups,
    )
    return preprocessor, dense
