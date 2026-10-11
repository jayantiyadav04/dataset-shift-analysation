"""Explicit fit-only jobs sharing source selection; no held-out reader here."""

import shutil
from collections.abc import Callable
from dataclasses import asdict
from functools import partial
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np

from shiftlab.data.parsers import CLASS_MAPS
from shiftlab.data.storage import DataError, read_json, write_json
from shiftlab.data.training import RawTrainingView, load_training
from shiftlab.environment import environment_report
from shiftlab.experiments.alignment import load_coral_state, prepare_coral
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.provenance import (
    RunKey,
    code_snapshot,
    content_hash,
    file_sha256,
    source_revision,
)
from shiftlab.experiments.selection import select_source
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.experiments.weighting import prepare_importance
from shiftlab.models.estimators import estimator_factory, fit_estimator


def runtime_identity(root: Path) -> dict[str, str]:
    """A lock digest represents this actual runtime only after checking its pins."""
    for line in (root / "requirements-lock.txt").read_text(encoding="utf-8").splitlines():
        if "==" in line and not line.startswith("#"):
            name, expected = line.strip().split("==")
            if version(name) != expected:
                raise DataError(f"Installed {name} differs from dependency lock")
    revision = source_revision(root)
    if revision.commit is None:
        raise DataError("A real Git source commit is required for fit provenance")
    return {
        "code_version": revision.commit,
        "code_snapshot_hash": content_hash(code_snapshot(root)),
        "dependency_lock_hash": file_sha256(root / "requirements-lock.txt"),
    }


def _save_fit(
    stage: Path,
    *,
    store: LocalArtifactStore,
    source: dict,
    view: RawTrainingView,
    split: dict,
    model: str,
    parameters: dict,
    seed: int,
    configuration: dict,
    environment: dict,
    study_id: str,
    importance: dict | None,
    coral: dict | None,
) -> dict[str, Any]:
    source_id = source["identity"]
    source_directory = store.directory("source_cache", source_id)
    X_source = np.load(source_directory / "source_features.npy", allow_pickle=False)
    weights, weight_diagnostics = None, None
    adaptation_seconds = 0.0
    coral_diagnostics, aligned_hash = None, None
    if importance is not None and coral is not None:
        raise DataError("Primary methods cannot combine weighting and CORAL")
    if importance is not None:
        directory = store.directory("importance_cache", importance["identity"])
        with np.load(directory / "adaptation_state.npz", allow_pickle=False) as arrays:
            if (
                tuple(arrays["source_ids"].tolist()) != view.source.ids
                or tuple(arrays["target_adapt_ids"].tolist()) != view.target_adapt.ids
            ):
                raise DataError("Cached importance weights differ from ordered training IDs")
            weights = arrays["weights"]
        if (
            weights.shape != (len(view.source.ids),)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
            or not np.isclose(weights.mean(), 1.0, atol=1e-12, rtol=0)
        ):
            raise DataError("Cached importance weights violate their source contract")
        weight_diagnostics = read_json(directory / "diagnostics.json")
        adaptation_seconds = importance["adaptation_fit_seconds"]
        shutil.copyfile(directory / "adaptation_state.npz", stage / "adaptation_state.npz")
        write_json(
            stage / "importance_reference.json",
            {
                "identity": importance["identity"],
                "manifest_hash": importance["manifest_hash"],
                "preprocessing_hash": importance["preprocessing_hash"],
                "shared_across_models": True,
            },
        )
    if coral is not None:
        directory = store.directory("coral_cache", coral["identity"])
        state = load_coral_state(
            directory / "adaptation_state.npz",
            preprocessing_hash=source["preprocessing_hash"],
            source_ids=view.source.ids,
            target_adapt_ids=view.target_adapt.ids,
        )
        aligned = np.load(directory / "aligned_source_features.npy", allow_pickle=False)
        aligned_hash = content_hash(aligned.tolist())
        if (
            aligned.shape != X_source.shape
            or not np.isfinite(aligned).all()
            or aligned_hash != coral["aligned_source_hash"]
            or len(state.source_mean) != X_source.shape[1]
        ):
            raise DataError("Cached CORAL features violate their source/schema contract")
        X_source = aligned
        coral_diagnostics = state.diagnostics
        adaptation_seconds = coral["adaptation_fit_seconds"]
        shutil.copyfile(directory / "adaptation_state.npz", stage / "adaptation_state.npz")
        write_json(
            stage / "coral_reference.json",
            {
                "identity": coral["identity"],
                "manifest_hash": coral["manifest_hash"],
                "preprocessing_hash": coral["preprocessing_hash"],
                "aligned_source_hash": aligned_hash,
                "shared_across_models": True,
                "target_prediction_coordinates": "source preprocessing only; no source alignment",
            },
        )
    estimator = estimator_factory(model, parameters, seed)
    started = perf_counter()
    messages = fit_estimator(estimator, X_source, view.y_source, sample_weight=weights)
    classifier_seconds = perf_counter() - started
    if importance is not None:
        messages = importance["warnings"] + messages
    if coral is not None:
        messages = coral["warnings"] + messages
    joblib.dump(estimator, stage / "fitted_model.joblib")
    shutil.copyfile(source_directory / "preprocessing.joblib", stage / "preprocessing.joblib")
    timings = {
        "preprocessing_fit_seconds": source["preprocessing_fit_seconds"],
        "source_transform_seconds": source["source_transform_seconds"],
        "adaptation_fit_seconds": adaptation_seconds,
        "classifier_fit_seconds": classifier_seconds,
        "target_prediction_seconds": None,
        "pipeline_fit_seconds": source["preprocessing_fit_seconds"]
        + source["source_transform_seconds"]
        + adaptation_seconds
        + classifier_seconds,
        "pipeline_fit_predict_seconds": None,
        "shared_source_search_seconds": source["source_search_seconds"],
        "definition": (
            "fit = preprocessing fit + source transform + adaptation fit + classifier fit; "
            "adaptation includes target adaptation transform; prediction includes target "
            "evaluation transform; source search excluded"
        ),
        "source_cache_id": source_id,
        "cached_component_timings": "Original computation duration retained",
    }
    if importance is not None:
        timings.update(
            {
                "importance_cache_id": importance["identity"],
                "target_adapt_transform_seconds": importance["target_adapt_transform_seconds"],
                "domain_classifier_and_weights_fit_seconds": importance[
                    "domain_classifier_and_weights_fit_seconds"
                ],
            }
        )
    if coral is not None:
        timings.update(
            {
                "coral_cache_id": coral["identity"],
                "target_adapt_transform_seconds": coral["target_adapt_transform_seconds"],
                "coral_and_source_alignment_fit_seconds": coral[
                    "coral_and_source_alignment_fit_seconds"
                ],
            }
        )
    write_json(stage / "config.json", configuration)
    write_json(stage / "environment.json", environment)
    write_json(
        stage / "source_cv_reference.json",
        {
            "identity": source_id,
            "manifest_hash": source["manifest_hash"],
            "selected_parameters": parameters,
            "preprocessing_hash": source["preprocessing_hash"],
            "source_search_shared": True,
            "majority_tuned": False,
        },
    )
    write_json(stage / "timings.json", timings)
    write_json(
        stage / "diagnostics.json",
        {
            "source_classes": np.unique(view.y_source).tolist(),
            "source_count": len(view.y_source),
            "feature_dimension": X_source.shape[1],
            "preprocessing_fit_ids": view.source.ids,
            "target_evaluation_opened": False,
            "estimator_parameters": estimator.get_params(),
            "sample_weight_passed": weights is not None,
            "sample_weight_hash": content_hash(weights.tolist()) if weights is not None else None,
            "importance_weighting": weight_diagnostics,
            "coral": coral_diagnostics,
            "source_fit_features_hash": aligned_hash,
            "warnings": messages,
        },
    )
    return {
        "study_id": study_id,
        "sample_counts": split["sample_counts"],
        "class_mapping": CLASS_MAPS[view.source.dataset],
        "selected_params": parameters,
        "seed": seed,
        "source_cv_id": source_id,
        "environment": "environment.json",
        "timing": timings,
        "warnings": messages,
        "evaluation_performed": False,
    }


def train_methods(
    config: StudyConfig,
    *,
    root: Path,
    data_root: Path,
    progress: Callable[[dict], None] | None = None,
    baseline_only: bool = False,
) -> dict[str, Any]:
    """Fit implemented methods; unsupported settings fail before creating artifacts."""
    methods = ("source_only",) if baseline_only else config.methods
    if baseline_only and "source_only" not in config.methods:
        raise DataError("Baseline training needs source_only in the declared configuration")
    if "source_only" not in methods and config.include_majority_baseline:
        raise DataError("Majority requires source_only in a method job configuration")
    identity = runtime_identity(root)
    environment = environment_report(root)
    environment.update(
        {
            "evidence_kind": "real_fit_only",
            "threads_per_job": 1,
            "code_snapshot": code_snapshot(root),
        }
    )
    store = LocalArtifactStore(root / config.execution.output_root)
    configuration = config.model_dump(mode="json")
    config_hash = content_hash(configuration)
    study_id = content_hash({"config_hash": config_hash, **identity})
    results = []
    for task in config.selected_tasks():
        directory = data_root / "splits" / config.protocol_version / task
        split = read_json(directory / "manifest.json")
        if split["task_id"] != task or split["protocol_version"] != config.protocol_version:
            raise DataError("Requested task or protocol differs from frozen split")
        view = load_training(directory)
        for seed in config.seeds:
            if progress:
                progress({"event": "source_selection", "task": task, "seed": seed})
            source, cache_hit = select_source(store, view, config, seed, identity)
            for method in methods:
                importance, coral = None, None
                if method == "importance_weighting":
                    if progress:
                        progress({"event": "importance_crossfit", "task": task, "seed": seed})
                    importance, _ = prepare_importance(store, view, config, seed, source, identity)
                if method == "coral":
                    if progress:
                        progress({"event": "coral_alignment", "task": task, "seed": seed})
                    coral, _ = prepare_coral(store, view, config, seed, source, identity)
                families = list(config.models)
                if method == "source_only" and config.include_majority_baseline:
                    families.append("majority")
                for model in families:
                    parameters = {} if model == "majority" else source["selected_parameters"][model]
                    key = RunKey(
                        dataset_hash=split["dataset_hash"],
                        task_id=task,
                        split_hash=view.split_hash,
                        model=model,
                        selected_source_parameter_hash=content_hash(parameters),
                        preprocessing_hash=source["preprocessing_hash"],
                        method=method,
                        adaptation_parameter_hash=content_hash(
                            config.importance_weighting.model_dump(mode="json")
                            if importance
                            else config.coral.model_dump(mode="json")
                            if coral
                            else {}
                        ),
                        seed=seed,
                        protocol_version=config.protocol_version,
                        config_hash=config_hash,
                        **identity,
                    )
                    produce = partial(
                        _save_fit,
                        store=store,
                        source=source,
                        view=view,
                        split=split,
                        model=model,
                        parameters=parameters,
                        seed=seed,
                        configuration=configuration,
                        environment=environment,
                        study_id=study_id,
                        importance=importance,
                        coral=coral,
                    )
                    record, resumed = store.create(
                        "runs", asdict(key), produce, resume=config.execution.resume
                    )
                    results.append(
                        {
                            "run_id": record["run_id"],
                            "task_id": task,
                            "model": str(model),
                            "method": str(method),
                            "seed": seed,
                            "status": record["status"],
                            "resumed": resumed,
                            "source_cache_reused": cache_hit,
                            "source_cv_id": source["identity"],
                            "importance_cache_id": importance["identity"] if importance else None,
                            "coral_cache_id": coral["identity"] if coral else None,
                        }
                    )
                    if progress:
                        progress({"event": "fit_saved", **results[-1]})
    expected = (
        len(config.selected_tasks())
        * len(config.seeds)
        * (len(config.models) * len(methods) + int(config.include_majority_baseline))
    )
    return {
        "status": "baselines_fitted" if baseline_only else "methods_fitted",
        "study_id": study_id,
        "expected_fits": expected,
        "expected_baseline_fits": expected if baseline_only else None,
        "fitted_records": len(results),
        "new_fits": sum(not r["resumed"] for r in results),
        "resumed_fits": sum(r["resumed"] for r in results),
        "evaluated_records": 0,
        "runs": results,
        "notice": "Fit-only development evidence. Final target data remain closed.",
    }


def train_baselines(
    config: StudyConfig,
    *,
    root: Path,
    data_root: Path,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Preserve the baseline-only CLI without changing the declared config hash."""
    return train_methods(
        config, root=root, data_root=data_root, progress=progress, baseline_only=True
    )
