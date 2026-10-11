"""Immutable discriminator/weight cache shared by paired task classifiers."""

from pathlib import Path
from time import perf_counter

import numpy as np

from shiftlab.adaptation.importance import fit_importance_weights
from shiftlab.contracts import TrainingView
from shiftlab.data.storage import DataError, write_json
from shiftlab.data.training import RawTrainingView
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.features.preprocessing import source_folds


def prepare_importance(
    store: LocalArtifactStore,
    view: RawTrainingView,
    config: StudyConfig,
    seed: int,
    source: dict,
    identity: dict[str, str],
) -> tuple[dict, bool]:
    source_id = source["identity"]
    key = {
        "split_hash": view.split_hash,
        "source_cv_id": source_id,
        "preprocessing_hash": source["preprocessing_hash"],
        "seed": seed,
        "parameters": config.importance_weighting.model_dump(mode="json"),
        "protocol_version": config.protocol_version,
        **identity,
    }

    def produce(stage: Path) -> dict:
        pre = store.load_joblib("source_cache", source_id, "preprocessing.joblib")
        if pre.preprocessing_hash_ != source["preprocessing_hash"]:
            raise DataError("Importance preprocessing differs from the paired source cache")
        X_source = np.load(
            store.directory("source_cache", source_id) / "source_features.npy", allow_pickle=False
        )
        started = perf_counter()
        X_adapt = pre.transform(view.target_adapt)
        transform_seconds = perf_counter() - started
        dense = TrainingView(
            X_source,
            view.y_source,
            X_adapt,
            view.source.ids,
            view.target_adapt.ids,
            view.source.groups,
            view.target_adapt.groups,
        )
        folds = source_folds(view, seed)
        started = perf_counter()
        state = fit_importance_weights(
            dense,
            folds,
            config.importance_weighting,
            seed,
            on_fold=lambda reference: write_json(
                stage / f"domain_fold_{reference['fold']}.json", reference
            ),
        )
        fit_seconds = perf_counter() - started
        np.savez_compressed(
            stage / "adaptation_state.npz",
            source_ids=np.asarray(view.source.ids),
            target_adapt_ids=np.asarray(view.target_adapt.ids),
            probabilities=state.probabilities,
            bounded_probabilities=state.bounded_probabilities,
            source_fit_counts=state.source_fit_counts,
            raw_weights=state.raw_weights,
            clipped_weights=state.clipped_weights,
            weights=state.weights,
            fold_assignment=state.fold_assignment,
        )
        write_json(stage / "diagnostics.json", state.diagnostics)
        return {
            "source_cv_id": source_id,
            "preprocessing_hash": source["preprocessing_hash"],
            "adaptation_fit_seconds": transform_seconds + fit_seconds,
            "target_adapt_transform_seconds": transform_seconds,
            "domain_classifier_and_weights_fit_seconds": fit_seconds,
            "warnings": state.diagnostics["warnings"],
            "fold_count": 5,
            "source_count": len(view.source.ids),
            "target_adapt_count": len(view.target_adapt.ids),
        }

    return store.create("importance_cache", key, produce, resume=config.execution.resume)
