"""Source-domain selection shared across target transfers and adaptation methods."""

import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
from sklearn.metrics import f1_score
from threadpoolctl import threadpool_limits

from shiftlab.data.storage import DataError, read_json, seal, verify_seal, write_json
from shiftlab.data.training import RawTrainingView
from shiftlab.experiments.checkpoints import atomic_json
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.provenance import canonical_json, content_hash
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.features.preprocessing import SourcePreprocessor, fit_source_fold, source_folds
from shiftlab.models.estimators import estimator_factory, fit_estimator


def candidate_grid(model: str) -> list[dict[str, Any]]:
    """Order is the deterministic simplicity tie breaker (exact equal means)."""
    if model in {"lr", "linear_svm"}:
        return [{"C": value} for value in (0.1, 1.0, 10.0)]
    if model == "rf":
        return [
            {"n_estimators": 200, "max_depth": depth, "min_samples_leaf": leaf}
            for depth in (20, None)
            for leaf in (3, 1)
        ]
    raise ValueError(f"No source tuning grid for {model}")


def selection_key(
    view: RawTrainingView,
    config: StudyConfig,
    seed: int,
    identity: dict[str, str],
) -> dict[str, Any]:
    """Exclude target rows/domain/split/method; include actual source bytes and labels."""
    values = view.source.values
    return {
        "source_dataset": view.source.dataset,
        "source_ids": view.source.ids,
        "source_groups": view.source.groups,
        "product_groups": view.source_product_ids,
        "source_features_hash": content_hash(
            values.tolist() if isinstance(values, np.ndarray) else values
        ),
        "source_labels_hash": content_hash(view.y_source.tolist()),
        "preprocessing_recipe": config.preprocessing.model_dump(mode="json"),
        "source_tuning": config.tuning.model_dump(mode="json"),
        "models": sorted(config.models),
        "seed": seed,
        "protocol_version": config.protocol_version,
        **identity,
    }


def select_source(
    store: LocalArtifactStore,
    view: RawTrainingView,
    config: StudyConfig,
    seed: int,
    identity: dict[str, str],
) -> tuple[dict[str, Any], bool]:
    key = selection_key(view, config, seed, identity)
    checkpoint = store.root / "cv_checkpoints" / content_hash(key)

    def produce(stage: Path) -> dict[str, Any]:
        started = perf_counter()
        folds = source_folds(view, seed)
        grids = {model: candidate_grid(model) for model in config.models}
        scores = {model: [[] for _ in grid] for model, grid in grids.items()}
        rows, references, messages = [], [], []
        original_seconds, reused_candidates = 0.0, 0
        for number, (train, valid) in enumerate(folds.indices):
            fold_started = perf_counter()
            pre, X_train, X_valid = fit_source_fold(view.source, train, valid, seed)
            fold_seconds = perf_counter() - fold_started
            references.append(
                {
                    "fold": number,
                    "train_ids": pre.fit_ids_,
                    "validation_ids": tuple(view.source.ids[i] for i in valid),
                    "train_groups": tuple(folds.groups[i] for i in train),
                    "validation_groups": tuple(folds.groups[i] for i in valid),
                    "preprocessing": pre.provenance_,
                    "preprocessing_hash": pre.preprocessing_hash_,
                }
            )
            fold_file = checkpoint / f"fold_{number}.json"
            if fold_file.exists():
                prior = read_json(fold_file)
                verify_seal(prior)
                if prior["reference"] != json.loads(canonical_json(references[-1])):
                    raise DataError("Source CV checkpoint differs from fold preprocessing")
                fold_seconds = prior["seconds"]
            else:
                atomic_json(fold_file, seal({"reference": references[-1], "seconds": fold_seconds}))
            original_seconds += fold_seconds
            for model, grid in grids.items():
                for index, parameters in enumerate(grid):
                    candidate_file = checkpoint / f"{number}-{model}-{index}.json"
                    candidate_key = {
                        "selection_hash": content_hash(key),
                        "fold": number,
                        "preprocessing_hash": pre.preprocessing_hash_,
                        "model": str(model),
                        "parameters": parameters,
                    }
                    if candidate_file.exists():
                        prior = read_json(candidate_file)
                        verify_seal(prior)
                        if prior["key"] != candidate_key:
                            raise DataError("Source CV candidate checkpoint identity differs")
                        score, seconds, warnings = (
                            prior["score"],
                            prior["seconds"],
                            prior["warnings"],
                        )
                        reused_candidates += 1
                    else:
                        candidate_started = perf_counter()
                        estimator = estimator_factory(model, parameters, seed)
                        warnings = fit_estimator(estimator, X_train, view.y_source[train])
                        with threadpool_limits(limits=1):
                            prediction = estimator.predict(X_valid)
                        score = float(
                            f1_score(
                                view.y_source[valid],
                                prediction,
                                average="macro",
                                labels=np.unique(view.y_source),
                                zero_division=0,
                            )
                        )
                        seconds = perf_counter() - candidate_started
                        atomic_json(
                            candidate_file,
                            seal(
                                {
                                    "key": candidate_key,
                                    "score": score,
                                    "seconds": seconds,
                                    "warnings": warnings,
                                }
                            ),
                        )
                    original_seconds += seconds
                    messages.extend(warnings)
                    scores[model][index].append(score)
                    row = {
                        "model": str(model),
                        "candidate": index,
                        "fold": number,
                        "parameters": canonical_json(parameters),
                        "macro_f1": score,
                    }
                    rows.append(row)
                    # Persist partial source evidence even if a later candidate fails.
                    with (stage / "source_cv.csv").open(
                        "a", encoding="utf-8", newline=""
                    ) as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(row))
                        if stream.tell() == 0:
                            writer.writeheader()
                        writer.writerow(row)
            write_json(stage / f"fold_{number}.json", references[-1])
        session_search_seconds = perf_counter() - started
        search_seconds = original_seconds if reused_candidates else session_search_seconds
        selected, summary = {}, []
        for model, grid in grids.items():
            means = [float(np.mean(values)) for values in scores[model]]
            winner = int(np.argmax(means))
            selected[model] = grid[winner]
            summary.extend(
                {
                    "model": str(model),
                    "candidate": index,
                    "parameters": params,
                    "mean_source_macro_f1": means[index],
                    "std_source_macro_f1": float(np.std(scores[model][index], ddof=1)),
                    "selected": index == winner,
                }
                for index, params in enumerate(grid)
            )
        started = perf_counter()
        pre = SourcePreprocessor(seed).fit(view.source)
        preprocessing_fit_seconds = perf_counter() - started
        started = perf_counter()
        X_source = pre.transform(view.source)
        source_transform_seconds = perf_counter() - started
        joblib.dump(pre, stage / "preprocessing.joblib")
        np.save(stage / "source_features.npy", X_source, allow_pickle=False)
        write_json(stage / "preprocessing.json", pre.provenance_)
        write_json(
            stage / "selection.json",
            {
                "selected_parameters": selected,
                "candidates": summary,
                "folds": references,
                "strategy": folds.strategy,
                "group_note": folds.note,
                "tie_breaker": "Exact mean ties: lower C; depth 20 then None; leaf 3 then 1",
                "search_seconds": search_seconds,
            },
        )
        return {
            "selected_parameters": selected,
            "preprocessing_hash": pre.preprocessing_hash_,
            "preprocessing_fit_seconds": preprocessing_fit_seconds,
            "source_transform_seconds": source_transform_seconds,
            "source_search_seconds": search_seconds,
            "source_search_original_component_seconds": original_seconds,
            "source_search_candidates_resumed": reused_candidates,
            "source_search_current_session_seconds": session_search_seconds,
            "warnings": messages,
            "source_cv_rows": len(rows),
            "source_count": len(view.source.ids),
            "feature_dimension": pre.actual_components_,
            "seed": seed,
        }

    return store.create("source_cache", key, produce, resume=config.execution.resume)
