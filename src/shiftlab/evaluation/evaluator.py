"""The held-out reader requires frozen fit/config/split authorization.

Not exposed through the development CLI. Real final evaluation is reserved for
the frozen study stage; current acceptance tests use synthetic split directories.
"""

import csv
from pathlib import Path
from time import perf_counter

import numpy as np
from threadpoolctl import threadpool_limits

from shiftlab.data.parsers import CLASS_MAPS
from shiftlab.data.splits import read_role_features, validate_split
from shiftlab.data.storage import DataError, read_json, verify_files, verify_seal, write_json
from shiftlab.data.training import FeatureBatch
from shiftlab.evaluation.metrics import classification_metrics
from shiftlab.experiments.provenance import content_hash
from shiftlab.experiments.storage import LocalArtifactStore


def evaluate_frozen_run(
    store: LocalArtifactStore,
    run_id: str,
    split_directory: Path,
    study_lock_file: Path,
) -> tuple[dict, bool]:
    """Check the study lock and fitted hashes before opening any evaluation file."""
    record = store.verify("runs", run_id)
    key = record["canonical_key"]
    lock = read_json(study_lock_file)
    verify_seal(lock)
    if lock.get("status") != "frozen_fits_complete" or run_id not in lock.get("run_ids", []):
        raise DataError("Evaluation requires a frozen study with completed permitted fits")
    for field in (
        "config_hash",
        "code_version",
        "code_snapshot_hash",
        "dependency_lock_hash",
        "protocol_version",
    ):
        if lock.get(field) != key[field]:
            raise DataError(f"Study lock differs from fitted {field}")
    split = read_json(split_directory / "manifest.json")
    validate_split(split)
    task = key["task_id"]
    if (
        split["task_id"] != task
        or split["manifest_hash"] != key["split_hash"]
        or split["dataset_hash"] != key["dataset_hash"]
        or lock.get("split_hashes", {}).get(task) != split["manifest_hash"]
        or lock.get("evaluation_ids", {}).get(task) != split["target_eval_ids"]
    ):
        raise DataError("Evaluation split/ordered IDs differ from the frozen study")
    extension = ".jsonl" if split["dataset_id"] == "amazon" else ".npz"
    features_path = f"evaluation/target_eval{extension}"
    labels_path = "evaluation/target_eval_labels.npy"
    if (
        split["artifacts"]["target_eval"] != features_path
        or split["artifacts"]["target_eval_labels"] != labels_path
    ):
        raise DataError("Evaluation artifact path is outside its permitted role")
    verify_files(
        split_directory, {name: split["checksums"][name] for name in (features_path, labels_path)}
    )
    ids, values = read_role_features(split_directory, features_path)
    if list(ids) != split["target_eval_ids"]:
        raise DataError("Held-out feature order differs from frozen IDs")
    batch = FeatureBatch(
        split["dataset_id"], "target_eval", ids, tuple(split["target_eval_group_ids"]), values
    )
    labels = np.load(split_directory / labels_path, allow_pickle=False)
    known = tuple(CLASS_MAPS[batch.dataset])
    if (
        labels.dtype.kind not in "iu"
        or labels.shape != (len(ids),)
        or not set(labels.tolist()) <= set(known)
    ):
        raise DataError("Held-out label shape or canonical mapping differs")
    evaluation_key = {
        "run_id": run_id,
        "study_lock_hash": lock["manifest_hash"],
        "split_hash": split["manifest_hash"],
        "evaluation_ids_hash": content_hash(ids),
        "class_policy": "present_in_full_target_test",
        "zero_division": 0,
    }

    def produce(stage: Path) -> dict:
        preprocessor = store.load_joblib("runs", run_id, "preprocessing.joblib")
        estimator = store.load_joblib("runs", run_id, "fitted_model.joblib")
        if preprocessor.preprocessing_hash_ != key["preprocessing_hash"]:
            raise DataError("Saved preprocessing identity differs from fitted run")
        started = perf_counter()
        with threadpool_limits(limits=1):
            prediction = estimator.predict(preprocessor.transform(batch))
        seconds = perf_counter() - started
        columns = [
            "run_id",
            "sample_id",
            "task_id",
            "model",
            "method",
            "seed",
            "y_true",
            "y_pred",
            "evaluation_group_id",
        ]
        with (stage / "predictions.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            for sample, group, true, pred in zip(
                ids, batch.groups, labels, prediction, strict=True
            ):
                writer.writerow(
                    dict(
                        zip(
                            columns,
                            [
                                run_id,
                                sample,
                                task,
                                key["model"],
                                key["method"],
                                key["seed"],
                                int(true),
                                int(pred),
                                group,
                            ],
                            strict=True,
                        )
                    )
                )
        # Metrics are derived from the persisted prediction rows, never a hand-edited table.
        with (stage / "predictions.csv").open(encoding="utf-8", newline="") as stream:
            rows = list(csv.DictReader(stream))
        true = np.array([int(row["y_true"]) for row in rows])
        pred = np.array([int(row["y_pred"]) for row in rows])
        metrics = classification_metrics(true, pred, known_classes=known)
        timings = {
            **record["timing"],
            "target_prediction_seconds": seconds,
            "pipeline_fit_predict_seconds": record["timing"]["pipeline_fit_seconds"] + seconds,
        }
        write_json(stage / "metrics.json", metrics)
        write_json(stage / "timings.json", timings)
        write_json(
            stage / "evaluation_manifest.json",
            {
                "run_id": run_id,
                "study_lock_hash": lock["manifest_hash"],
                "evaluation_ids": ids,
                "evaluation_groups": batch.groups,
                "macro_classes": metrics["macro_classes"],
                "class_policy": metrics["class_policy"],
            },
        )
        return {
            "run_id": run_id,
            "task_id": task,
            "evaluation_performed": True,
            "sample_count": len(ids),
            "timing": timings,
        }

    return store.create("evaluations", evaluation_key, produce)
