"""Recheck observed study evidence without fitting or reopening raw target data."""

import argparse
import csv
from collections import Counter
from pathlib import Path

import numpy as np

from shiftlab.data.storage import DataError, read_json, seal, verify_files, verify_seal, write_json
from shiftlab.evaluation.analysis import load_predictions, validate_pair
from shiftlab.evaluation.metrics import classification_metrics
from shiftlab.evaluation.statistics import evaluation_units
from shiftlab.experiments.checkpoints import Journal
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.provenance import code_snapshot, content_hash, file_sha256
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.experiments.study import fitted_rows, open_study, schedule
from shiftlab.scope import FINAL_OBJECTIVE


def audit(root: Path, study_id: str) -> dict:
    directory, lock = open_study(root, study_id)
    config = StudyConfig.model_validate(lock["configuration"])
    rows = schedule(config)
    journal = Journal(directory)
    fits = fitted_rows(journal)
    done = {
        e["row_id"]: e["evaluation_id"]
        for e in journal.events
        if e.get("event") == "evaluation_saved"
    }
    if set(fits) != {r["row_id"] for r in rows} or set(done) != set(fits):
        raise DataError("Audit requires complete declared coverage")
    completion = read_json(directory / "fit_completion_lock.json")
    verify_seal(completion)
    if completion["fit_rows_hash"] != content_hash([fits[r["row_id"]] for r in rows]):
        raise DataError("Fit authorization changed")
    store = LocalArtifactStore(root / config.execution.output_root)
    index = {(r["task_id"], r["model"], r["method"], r["setting"], r["seed"]): r for r in rows}
    records, predictions, source_ids, adaptation_ids = {}, {}, set(), set()
    actual_inputs = {}
    primary, sensitivity, warnings = 0, 0, []
    manifests = {}
    for row in rows:
        fit = fits[row["row_id"]]
        run = store.verify("runs", fit["run_id"])
        evaluation = store.verify("evaluations", done[row["row_id"]])
        if run["study_id"] != study_id or evaluation["run_id"] != run["run_id"]:
            raise DataError("Run/evaluation study association differs")
        key = run["canonical_key"]
        for field in (
            "config_hash",
            "code_version",
            "code_snapshot_hash",
            "dependency_lock_hash",
            "protocol_version",
        ):
            if key[field] != lock[field]:
                raise DataError("Run identity differs from frozen lock")
        if (
            key["task_id"] != row["task_id"]
            or key["model"] != row["model"]
            or key["method"] != row["method"]
            or key["seed"] != row["seed"]
        ):
            raise DataError("Run differs from expected row")
        split = read_json(
            root
            / lock["data_root"]
            / "splits"
            / config.protocol_version
            / row["task_id"]
            / "manifest.json"
        )
        if split["manifest_hash"] != lock["split_hashes"][row["task_id"]]:
            raise DataError("Split changed")
        manifests[row["task_id"]] = split
        pre = read_json(store.directory("source_cache", fit["source_cv_id"]) / "preprocessing.json")
        if pre["fit_ids"] != split["source_ids"]:
            raise DataError("Source preprocessing fit IDs differ")
        source_ids.add(fit["source_cv_id"])
        target_dir = store.directory("evaluations", done[row["row_id"]])
        prediction = load_predictions(target_dir / "predictions.csv", split["target_eval_ids"])
        if prediction["groups"] != tuple(split["target_eval_group_ids"]):
            raise DataError("Saved prediction groups differ from frozen split")
        actual_inputs[run["run_id"]] = {
            "fit_manifest_sha256": file_sha256(
                store.directory("runs", fit["run_id"]) / "manifest.json"
            ),
            "evaluation_id": done[row["row_id"]],
            "prediction_path": (target_dir / "predictions.csv").relative_to(root).as_posix(),
            "predictions_sha256": file_sha256(target_dir / "predictions.csv"),
            "metrics_sha256": file_sha256(target_dir / "metrics.json"),
        }
        saved = read_json(target_dir / "metrics.json")
        recomputed = classification_metrics(
            prediction["true"],
            prediction["pred"],
            known_classes=tuple(int(c) for c in run["class_mapping"]),
        )
        if saved != recomputed:
            raise DataError("Saved metrics fail exact recomputation")
        records[row["row_id"]], predictions[row["row_id"]] = run, prediction
        diag = read_json(store.directory("runs", fit["run_id"]) / "diagnostics.json")
        if diag["target_evaluation_opened"] or run["evaluation_performed"]:
            raise DataError("Fit artifact contains evaluation evidence")
        run_path = store.directory("runs", fit["run_id"])
        if row["method"] == "importance_weighting":
            reference = read_json(run_path / "importance_reference.json")
            store.verify("importance_cache", reference["identity"])
            adaptation_ids.add(reference["identity"])
            with np.load(run_path / "adaptation_state.npz", allow_pickle=False) as arrays:
                weights = arrays["weights"]
                upper = 10.0 if row["setting"] == "primary" else float(row["setting"].split(":")[1])
                clipped = np.clip(arrays["raw_weights"], 0.05, upper)
                if not np.array_equal(clipped, arrays["clipped_weights"]) or not np.allclose(
                    weights, clipped / clipped.mean(), rtol=0, atol=1e-12
                ):
                    raise DataError("Stored clipping or global normalization differs")
                if (
                    arrays["source_ids"].tolist() != split["source_ids"]
                    or arrays["target_adapt_ids"].tolist() != split["target_adapt_ids"]
                ):
                    raise DataError("Adaptation IDs differ")
            if (
                not np.isfinite(weights).all()
                or np.any(weights <= 0)
                or not np.isclose(weights.mean(), 1.0, atol=1e-12, rtol=0)
            ):
                raise DataError("Saved weights violate their contract")
            if not diag["sample_weight_passed"]:
                raise DataError("Weighted classifier did not receive sample weights")
            ess = float(weights.sum() ** 2 / np.square(weights).sum())
            if not np.isclose(ess, diag["importance_weighting"]["ess"], rtol=0, atol=1e-10):
                raise DataError("Saved ESS fails independent recomputation")
        if row["method"] == "coral":
            reference = read_json(run_path / "coral_reference.json")
            store.verify("coral_cache", reference["identity"])
            adaptation_ids.add(reference["identity"])
            if diag["sample_weight_passed"]:
                raise DataError("CORAL must not receive weights")
        warnings.extend(
            {"run_id": fit["run_id"], "warning": message} for message in run["warnings"]
        )
        primary += row["setting"] == "primary"
        sensitivity += row["setting"] != "primary"
    pairs = 0
    for row in rows:
        if row["method"] == "source_only":
            continue
        base = index[(row["task_id"], row["model"], "source_only", "primary", row["seed"])]
        validate_pair(records[base["row_id"]], records[row["row_id"]])
        b, a = predictions[base["row_id"]], predictions[row["row_id"]]
        if not np.array_equal(b["true"], a["true"]) or b["groups"] != a["groups"]:
            raise DataError("Pair prediction labels/groups differ")
        pairs += 1
    analysis = directory / "analysis"
    export = read_json(analysis / "export_manifest.json")
    verify_seal(export)
    verify_files(analysis, export["output_hashes"])
    if export["input_hashes"] != actual_inputs:
        raise DataError("Export input identities or actual file hashes differ")
    traces = read_json(analysis / "figure_traces.json")
    for name, run_ids in traces.items():
        if (
            not run_ids
            or set(run_ids) - set(export["input_hashes"])
            or not (analysis / "figures" / name).is_file()
        ):
            raise DataError("Scientific figure trace differs")
    with (analysis / "ablations.csv").open(encoding="utf-8", newline="") as stream:
        ablations = list(csv.DictReader(stream))
    if len(ablations) != 270:
        raise DataError("Every declared sensitivity value including reused primary must appear")
    if (primary, sensitivity, pairs, len(source_ids), len(adaptation_ids)) != (
        1100,
        180,
        840,
        30,
        280,
    ):
        raise DataError("Observed runs/source/adaptation cache coverage differs from declared plan")
    cv_scores = 0
    for identity in source_ids:
        store.verify("source_cache", identity)
        with (store.directory("source_cache", identity) / "source_cv.csv").open(
            encoding="utf-8", newline=""
        ) as stream:
            scores = list(csv.DictReader(stream))
        if len(scores) != 50 or Counter(s["model"] for s in scores) != {
            "lr": 15,
            "linear_svm": 15,
            "rf": 20,
        }:
            raise DataError("Saved source CV differs from the complete candidate/fold grid")
        cv_scores += len(scores)
    boundary = read_json(directory / "fit_boundary_receipt.json")
    verify_seal(boundary)
    if boundary["forbidden_open_attempts"]:
        raise DataError("Fit boundary failed")
    for task in config.ablations.tasks:
        boundary = read_json(directory / f"benchmark_boundary_{task}.json")
        verify_seal(boundary)
        if boundary["forbidden_open_attempts"]:
            raise DataError("Benchmark fit boundary failed")
    return {
        "status": "passed",
        "study_id": study_id,
        "objective_verbatim": lock["objective"] == FINAL_OBJECTIVE,
        "code_snapshot_matches_actual": lock["code_snapshot"] == code_snapshot(root),
        "primary_evaluations": primary,
        "sensitivity_evaluations": sensitivity,
        "prediction_metric_recomputations": len(done),
        "paired_seed_comparisons": pairs,
        "source_cache_records": len(source_ids),
        "source_cv_candidate_fold_scores": cv_scores,
        "adaptation_cache_records": len(adaptation_ids),
        "figures": len(traces),
        "all_22_split_hashes_match": True,
        "group_units": {
            task: {
                "policy": evaluation_units(split)[1],
                "count": len(set(evaluation_units(split)[0])),
                "sample_count": len(split["target_eval_ids"]),
            }
            for task, split in manifests.items()
        },
        "warning_records": warnings,
        "failed_attempts": [e for e in journal.events if e["event"] == "job_failed"],
        "export_manifest_sha256": file_sha256(analysis / "export_manifest.json"),
        "source_commit": lock["code_version"],
        "engine_snapshot_hash": lock["code_snapshot_hash"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(Path.cwd(), args.study_id)
    write_json(args.out, seal(result))
    print(
        f"PASS: {result['primary_evaluations']} primary, "
        f"{result['sensitivity_evaluations']} sensitivity; "
        f"{result['prediction_metric_recomputations']} metric recomputations"
    )
