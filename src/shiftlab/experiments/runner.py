"""Explicit fit-only baseline jobs. No evaluator or target label access here."""

import shutil
from collections.abc import Callable
from dataclasses import asdict
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


def train_baselines(
    config: StudyConfig,
    *,
    root: Path,
    data_root: Path,
    progress: Callable[[dict], None] | None = None,
) -> dict[str, Any]:
    """Train declared source families; adaptation methods remain separately implemented."""
    if "source_only" not in config.methods:
        raise DataError("Baseline training needs source_only in the declared configuration")
    identity = runtime_identity(root)
    environment = environment_report(root)
    environment.update(
        {
            "evidence_kind": "real_source_fit_only",
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
            source_id = source["identity"]
            source_directory = store.directory("source_cache", source_id)
            # allow_pickle=False; no target features/labels are opened or transformed.
            X_source = np.load(source_directory / "source_features.npy", allow_pickle=False)
            families = list(config.models) + (
                ["majority"] if config.include_majority_baseline else []
            )
            for model in families:
                parameters = {} if model == "majority" else source["selected_parameters"][model]
                key = RunKey(
                    dataset_hash=split["dataset_hash"],
                    task_id=task,
                    split_hash=view.split_hash,
                    model=model,
                    selected_source_parameter_hash=content_hash(parameters),
                    preprocessing_hash=source["preprocessing_hash"],
                    method="source_only",
                    adaptation_parameter_hash=content_hash({}),
                    seed=seed,
                    protocol_version=config.protocol_version,
                    config_hash=config_hash,
                    **identity,
                )

                def produce(
                    stage: Path,
                    *,
                    model: str = model,
                    parameters: dict = parameters,
                    seed: int = seed,
                    X_source: np.ndarray = X_source,
                    view: RawTrainingView = view,
                    source_directory: Path = source_directory,
                    source: dict = source,
                    source_id: str = source_id,
                    split: dict = split,
                ) -> dict[str, Any]:
                    estimator = estimator_factory(model, parameters, seed)
                    started = perf_counter()
                    messages = fit_estimator(estimator, X_source, view.y_source)
                    classifier_seconds = perf_counter() - started
                    joblib.dump(estimator, stage / "fitted_model.joblib")
                    shutil.copyfile(
                        source_directory / "preprocessing.joblib", stage / "preprocessing.joblib"
                    )
                    timings = {
                        "preprocessing_fit_seconds": source["preprocessing_fit_seconds"],
                        "source_transform_seconds": source["source_transform_seconds"],
                        "adaptation_fit_seconds": 0.0,
                        "classifier_fit_seconds": classifier_seconds,
                        "target_prediction_seconds": None,
                        "pipeline_fit_seconds": source["preprocessing_fit_seconds"]
                        + source["source_transform_seconds"]
                        + classifier_seconds,
                        "pipeline_fit_predict_seconds": None,
                        "shared_source_search_seconds": source["source_search_seconds"],
                        "definition": (
                            "fit = preprocessing fit + source transform + adaptation fit + "
                            "classifier fit; prediction includes target transform; search excluded"
                        ),
                        "source_cache_id": source_id,
                        "cached_component_timings": "Original computation duration retained",
                    }
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

                record, resumed = store.create(
                    "runs", asdict(key), produce, resume=config.execution.resume
                )
                results.append(
                    {
                        "run_id": record["run_id"],
                        "task_id": task,
                        "model": str(model),
                        "seed": seed,
                        "status": record["status"],
                        "resumed": resumed,
                        "source_cache_reused": cache_hit,
                        "source_cv_id": source_id,
                    }
                )
                if progress:
                    progress({"event": "baseline_saved", **results[-1]})
    return {
        "status": "baselines_fitted",
        "study_id": study_id,
        "expected_baseline_fits": len(config.selected_tasks())
        * len(config.seeds)
        * (len(config.models) + int(config.include_majority_baseline)),
        "fitted_records": len(results),
        "new_fits": sum(not row["resumed"] for row in results),
        "resumed_fits": sum(row["resumed"] for row in results),
        "evaluated_records": 0,
        "runs": results,
        "notice": "Fit-only development evidence. Final target data remain closed.",
    }
