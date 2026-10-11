"""Immutable shared CORAL transform and source feature cache."""

from pathlib import Path
from time import perf_counter

import numpy as np

from shiftlab.adaptation.coral import CoralState, fit_coral
from shiftlab.contracts import TrainingView
from shiftlab.data.storage import DataError, read_json, write_json
from shiftlab.data.training import RawTrainingView
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.provenance import content_hash
from shiftlab.experiments.storage import LocalArtifactStore


def load_coral_state(
    path: Path,
    *,
    preprocessing_hash: str,
    source_ids: tuple[str, ...],
    target_adapt_ids: tuple[str, ...],
) -> CoralState:
    """Read numeric state only, after the caller verifies its artifact manifest."""
    with np.load(path, allow_pickle=False) as arrays:
        if (
            arrays["preprocessing_hash"].item() != preprocessing_hash
            or tuple(arrays["source_ids"].tolist()) != source_ids
            or tuple(arrays["target_adapt_ids"].tolist()) != target_adapt_ids
            or arrays["ridge"].item() <= 0
            or arrays["eigenvalue_floor"].item() != 1e-8
        ):
            raise DataError("Saved CORAL state differs from preprocessing or ordered training IDs")
        diagnostics = read_json(path.parent / "diagnostics.json")
        # Per-run diagnostics nest adaptation diagnostics; shared caches do not.
        diagnostics = diagnostics.get("coral", diagnostics)
        if (
            diagnostics["parameters"]["ridge"] != arrays["ridge"].item()
            or diagnostics["parameters"]["eigenvalue_floor"] != arrays["eigenvalue_floor"].item()
        ):
            raise DataError("CORAL numerical settings differ from saved diagnostics")
        return CoralState(
            arrays["A"], arrays["ms"], arrays["mt"], arrays["Cs0"], arrays["Ct0"], diagnostics
        )


def prepare_coral(
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
        "parameters": config.coral.model_dump(mode="json"),
        "protocol_version": config.protocol_version,
        **identity,
    }

    def produce(stage: Path) -> dict:
        pre = store.load_joblib("source_cache", source_id, "preprocessing.joblib")
        if pre.preprocessing_hash_ != source["preprocessing_hash"]:
            raise DataError("CORAL preprocessing differs from the paired source cache")
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
        started = perf_counter()
        state = fit_coral(dense, config.coral)
        aligned = state.transform_source(X_source)
        fit_seconds = perf_counter() - started
        np.savez_compressed(
            stage / "adaptation_state.npz",
            A=state.alignment,
            ms=state.source_mean,
            mt=state.target_mean,
            Cs0=state.source_covariance,
            Ct0=state.target_covariance,
            ridge=np.asarray(config.coral.ridge),
            eigenvalue_floor=np.asarray(config.coral.eigenvalue_floor),
            preprocessing_hash=np.asarray(source["preprocessing_hash"]),
            source_ids=np.asarray(view.source.ids),
            target_adapt_ids=np.asarray(view.target_adapt.ids),
        )
        np.save(stage / "aligned_source_features.npy", aligned, allow_pickle=False)
        write_json(stage / "diagnostics.json", state.diagnostics)
        return {
            "source_cv_id": source_id,
            "preprocessing_hash": source["preprocessing_hash"],
            "adaptation_fit_seconds": transform_seconds + fit_seconds,
            "target_adapt_transform_seconds": transform_seconds,
            "coral_and_source_alignment_fit_seconds": fit_seconds,
            "aligned_source_hash": content_hash(aligned.tolist()),
            "warnings": state.diagnostics["warnings"],
            "source_count": len(view.source.ids),
            "target_adapt_count": len(view.target_adapt.ids),
        }

    return store.create("coral_cache", key, produce, resume=config.execution.resume)
