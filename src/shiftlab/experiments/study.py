"""Frozen primary/sensitivity schedule with resumable, auditable fit/evaluate stages."""

import csv
import re
from pathlib import Path
from time import perf_counter

from shiftlab.data.splits import validate_split
from shiftlab.data.storage import DataError, read_json, seal, verify_seal, write_json
from shiftlab.evaluation.evaluator import evaluate_frozen_run
from shiftlab.experiments.checkpoints import Journal, atomic_json, job_owner, write_csv
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.provenance import code_snapshot, content_hash
from shiftlab.experiments.resources import hardware_report, resource_measurement
from shiftlab.experiments.runner import runtime_identity, train_methods
from shiftlab.experiments.storage import LocalArtifactStore, now
from shiftlab.scope import FINAL_OBJECTIVE


def schedule(config: StudyConfig) -> list[dict]:
    rows = []
    for task in config.selected_tasks():
        for seed in config.seeds:
            settings = [("primary", method) for method in config.methods]
            if task in config.ablations.tasks:
                settings += [
                    (f"importance_weighting:{v:g}", "importance_weighting") for v in (5.0, 20.0)
                ]
                settings += [(f"coral:{v:g}", "coral") for v in (0.0001, 0.01)]
            for setting, method in settings:
                models = list(config.models)
                if (
                    setting == "primary"
                    and method == "source_only"
                    and config.include_majority_baseline
                ):
                    models.append("majority")
                for model in models:
                    row = {
                        "task_id": task,
                        "dataset": task.split("_", 1)[0],
                        "seed": seed,
                        "setting": setting,
                        "method": str(method),
                        "model": str(model),
                    }
                    rows.append({"row_id": content_hash(row), **row})
    return rows


def initialize(config: StudyConfig, root: Path, data_root: Path) -> tuple[Path, dict]:
    if config.mode != "primary":
        raise DataError("Frozen study execution requires the complete primary configuration")
    identity = runtime_identity(root)
    configuration = config.model_dump(mode="json")
    splits = {}
    for task in config.selected_tasks():
        split = read_json(data_root / "splits" / config.protocol_version / task / "manifest.json")
        validate_split(split)
        if split["task_id"] != task or split["protocol_version"] != config.protocol_version:
            raise DataError("Study split differs from configured task/protocol")
        splits[task] = split
    # Identity is source bytes, not the publication commit: publishing the same
    # source cannot make a resumable study vanish. The original commit is retained.
    study_id = content_hash(
        {
            "config_hash": content_hash(configuration),
            "code_snapshot_hash": identity["code_snapshot_hash"],
            "dependency_lock_hash": identity["dependency_lock_hash"],
            "splits": {t: s["manifest_hash"] for t, s in splits.items()},
        }
    )
    directory = root / config.execution.output_root / "studies" / study_id
    path = directory / "study_lock.json"
    if path.exists():
        lock = read_json(path)
        verify_seal(lock)
        if lock["study_id"] != study_id or lock["configuration"] != configuration:
            raise DataError("Existing study lock differs")
        return directory, lock
    rows = schedule(config)
    lock = seal(
        {
            "schema_version": 1,
            "study_id": study_id,
            "status": "frozen_planned",
            "frozen_at": now(),
            "objective": FINAL_OBJECTIVE,
            "objective_hash": content_hash(FINAL_OBJECTIVE),
            "configuration": configuration,
            "config_hash": content_hash(configuration),
            "protocol_version": config.protocol_version,
            **identity,
            "code_snapshot": code_snapshot(root),
            "split_hashes": {t: s["manifest_hash"] for t, s in splits.items()},
            "dataset_hashes": {t: s["dataset_hash"] for t, s in splits.items()},
            "raw_archive_hashes": {
                s["dataset_id"]: read_json(
                    data_root / "processed" / s["dataset_id"] / "manifest.json"
                )["raw_sha256"]
                for s in splits.values()
            },
            "evaluation_ids": {t: s["target_eval_ids"] for t, s in splits.items()},
            "expected_rows_hash": content_hash(rows),
            "expected_primary": 1100,
            "expected_sensitivity": 180,
            "data_root": data_root.resolve().relative_to(root.resolve()).as_posix(),
            "hardware": hardware_report(),
        }
    )
    with job_owner(directory):
        if path.exists():
            raise DataError("Another writer froze this study; resume it")
        write_csv(directory / "expected_runs.csv", rows)
        write_json(path, lock)
    return directory, lock


def open_study(root: Path, study_id: str, output_root: str = "artifacts") -> tuple[Path, dict]:
    if not re.fullmatch(r"[0-9a-f]{64}", study_id):
        raise DataError("Study ID must be a canonical SHA-256 digest")
    directory = root / output_root / "studies" / study_id
    lock = read_json(directory / "study_lock.json")
    verify_seal(lock)
    identity = runtime_identity(root)
    for field in ("code_snapshot_hash", "dependency_lock_hash"):
        if lock[field] != identity[field]:
            raise DataError(
                "Study source/dependency bytes differ; preserve this study and version corrections"
            )
    with (directory / "expected_runs.csv").open(encoding="utf-8", newline="") as stream:
        actual = list(csv.DictReader(stream))
    for row in actual:
        row["seed"] = int(row["seed"])
    if content_hash(actual) != lock["expected_rows_hash"] or lock["objective"] != FINAL_OBJECTIVE:
        raise DataError("Frozen study matrix or objective differs")
    return directory, lock


def fitted_rows(journal: Journal) -> dict[str, dict]:
    return {e["row"]["row_id"]: e["row"] for e in journal.events if e.get("event") == "fit_saved"}


def checkpoint(directory: Path, lock: dict, journal: Journal, status: str) -> dict:
    fitted = fitted_rows(journal)
    evaluated = {e["row_id"]: e for e in journal.events if e.get("event") == "evaluation_saved"}
    rows = schedule(StudyConfig.model_validate(lock["configuration"]))
    missing = [
        {
            **r,
            "status": "fitted_pending_evaluation"
            if r["row_id"] in fitted
            else "unresolved_missing_fit",
        }
        for r in rows
        if r["row_id"] not in evaluated
    ]
    complete = [
        {**fitted[r], "evaluation_id": e["evaluation_id"], "status": "evaluated"}
        for r, e in evaluated.items()
    ]
    columns = list(rows[0]) + ["status"]
    write_csv(directory / "missing_runs.csv", missing, columns)
    write_csv(
        directory / "completed_runs.csv",
        complete,
        list(rows[0]) + ["run_id", "source_cv_id", "status", "evaluation_id"],
    )
    report = {
        "study_id": lock["study_id"],
        "status": status,
        "expected_primary": 1100,
        "expected_sensitivity": 180,
        "fitted_primary": sum(r["setting"] == "primary" for r in fitted.values()),
        "fitted_sensitivity": sum(r["setting"] != "primary" for r in fitted.values()),
        "evaluated_records": len(evaluated),
        "missing_evaluations": len(missing),
        "failed_attempts": sum(e.get("event") == "job_failed" for e in journal.events),
        "latest_event_hash": journal.previous,
        "updated_at": now(),
    }
    atomic_json(directory / "checkpoint.json", seal(report))
    return report


def fit_study(config: StudyConfig, root: Path, data_root: Path, progress=None) -> dict:
    directory, lock = initialize(config, root, data_root)
    expected = schedule(config)
    identity = {
        field: lock[field]
        for field in ("code_version", "code_snapshot_hash", "dependency_lock_hash")
    }
    lookup = {(r["task_id"], r["seed"], r["setting"], r["method"], r["model"]): r for r in expected}
    with job_owner(directory):
        journal = Journal(directory)
        known = fitted_rows(journal)
        store = LocalArtifactStore(root / config.execution.output_root)
        for row in known.values():
            record = store.verify("runs", row["run_id"])
            if any(
                record["canonical_key"][f] != lock[f]
                for f in (
                    "config_hash",
                    "code_snapshot_hash",
                    "dependency_lock_hash",
                    "protocol_version",
                )
            ):
                raise DataError("Resumed fit differs from the frozen study")

        def observe(event):
            if event["event"] == "fit_saved":
                reference = lookup[
                    (
                        event["task_id"],
                        event["seed"],
                        event["setting"],
                        event["method"],
                        event["model"],
                    )
                ]
                row = {
                    **reference,
                    "run_id": event["run_id"],
                    "source_cv_id": event["source_cv_id"],
                    "status": "fitted",
                }
                if row["row_id"] not in known:
                    journal.append({"event": "fit_saved", "row": row})
                    known[row["row_id"]] = row
                    checkpoint(directory, lock, journal, "fitting")
            elif event["event"] in {"source_selection", "importance_crossfit", "coral_alignment"}:
                atomic_json(directory / "active_job.json", {**event, "observed_at": now()})
            if progress:
                progress(event)

        try:
            for task in config.selected_tasks():
                for seed in config.seeds:
                    settings = [None]
                    if task in config.ablations.tasks:
                        settings += [
                            ("importance_weighting", 5.0),
                            ("importance_weighting", 20.0),
                            ("coral", 0.0001),
                            ("coral", 0.01),
                        ]
                    for sensitivity in settings:
                        setting = (
                            "primary"
                            if sensitivity is None
                            else f"{sensitivity[0]}:{sensitivity[1]:g}"
                        )
                        required = [
                            r
                            for r in expected
                            if r["task_id"] == task
                            and r["seed"] == seed
                            and r["setting"] == setting
                        ]
                        if all(r["row_id"] in known for r in required):
                            continue
                        journal.append(
                            {
                                "event": "job_started",
                                "task_id": task,
                                "seed": seed,
                                "setting": setting,
                            }
                        )
                        try:
                            train_methods(
                                config,
                                root=root,
                                data_root=data_root,
                                progress=observe,
                                tasks=(task,),
                                seeds=(seed,),
                                sensitivity=sensitivity,
                                frozen_identity=identity,
                                frozen_study_id=lock["study_id"],
                            )
                        except (OSError, ValueError, RuntimeError) as error:
                            journal.append(
                                {
                                    "event": "job_failed",
                                    "task_id": task,
                                    "seed": seed,
                                    "setting": setting,
                                    "error": {"type": type(error).__name__, "message": str(error)},
                                }
                            )
                            checkpoint(directory, lock, journal, "fitting_with_failures")
                            if progress:
                                progress(
                                    {
                                        "event": "job_failed",
                                        "task": task,
                                        "seed": seed,
                                        "setting": setting,
                                        "error": str(error),
                                    }
                                )
            status = "frozen_fits_complete" if len(known) == len(expected) else "incomplete_fits"
            if status == "frozen_fits_complete":
                # Recheck bytes before authorising the first final evaluation.
                open_study(root, lock["study_id"], config.execution.output_root)
                authorization = directory / "fit_completion_lock.json"
                if not authorization.exists():
                    write_json(
                        authorization,
                        seal(
                            {
                                **{k: v for k, v in lock.items() if k != "manifest_hash"},
                                "status": status,
                                "study_lock_hash": lock["manifest_hash"],
                                "completed_at": now(),
                                "run_ids": [known[r["row_id"]]["run_id"] for r in expected],
                                "fit_rows_hash": content_hash(
                                    [known[r["row_id"]] for r in expected]
                                ),
                            }
                        ),
                    )
            return checkpoint(directory, lock, journal, status)
        except KeyboardInterrupt:
            journal.append({"event": "cancelled"})
            checkpoint(directory, lock, journal, "cancelled_resumable")
            raise


def evaluate_study(root: Path, study_id: str, progress=None, output_root="artifacts") -> dict:
    directory, lock = open_study(root, study_id, output_root)
    authorization = read_json(directory / "fit_completion_lock.json")
    verify_seal(authorization)
    if (
        authorization.get("study_lock_hash") != lock["manifest_hash"]
        or authorization.get("status") != "frozen_fits_complete"
    ):
        raise DataError("Completed-fit authorisation does not match the frozen plan")
    config = StudyConfig.model_validate(lock["configuration"])
    store = LocalArtifactStore(root / config.execution.output_root)
    with job_owner(directory):
        journal = Journal(directory)
        fitted = fitted_rows(journal)
        expected = schedule(config)
        if set(fitted) != {r["row_id"] for r in expected} or authorization[
            "fit_rows_hash"
        ] != content_hash([fitted[r["row_id"]] for r in expected]):
            raise DataError(
                "Every declared primary and sensitivity fit must be saved before evaluation"
            )
        done = {e["row_id"]: e for e in journal.events if e.get("event") == "evaluation_saved"}
        try:
            for row in expected:
                if row["row_id"] in done:
                    store.verify("evaluations", done[row["row_id"]]["evaluation_id"])
                    continue
                fit = fitted[row["row_id"]]
                record, resumed = evaluate_frozen_run(
                    store,
                    fit["run_id"],
                    root / lock["data_root"] / "splits" / config.protocol_version / row["task_id"],
                    directory / "fit_completion_lock.json",
                )
                event = journal.append(
                    {
                        "event": "evaluation_saved",
                        "row_id": row["row_id"],
                        "run_id": fit["run_id"],
                        "evaluation_id": record["identity"],
                        "resumed": resumed,
                    }
                )
                done[row["row_id"]] = event
                checkpoint(directory, lock, journal, "evaluating")
                if progress:
                    progress(event)
            return checkpoint(directory, lock, journal, "evaluated_analysis_pending")
        except KeyboardInterrupt:
            journal.append({"event": "cancelled"})
            checkpoint(directory, lock, journal, "cancelled_resumable")
            raise


def benchmark(config: StudyConfig, root: Path, data_root: Path, task: str, progress=None) -> dict:
    if task not in config.ablations.tasks:
        raise DataError("Benchmark must use one of the three predeclared representative tasks")
    directory, lock = initialize(config, root, data_root)
    destination = directory / "benchmarks" / task
    with job_owner(directory):
        if (destination / "benchmark.json").exists():
            raise DataError(
                "Benchmark already exists; use its measured evidence rather than timing a cache hit"
            )
        store = LocalArtifactStore(root / config.execution.output_root)
        identity = {
            f: lock[f] for f in ("code_version", "code_snapshot_hash", "dependency_lock_hash")
        }
        started = perf_counter()
        result = train_methods(
            config,
            root=root,
            data_root=data_root,
            tasks=(task,),
            seeds=(17,),
            frozen_identity=identity,
            frozen_study_id=lock["study_id"],
            progress=progress,
        )
        wall = perf_counter() - started
        if result["new_fits"] != 10 or any(r["source_cache_reused"] for r in result["runs"]):
            raise DataError("Fresh benchmark requires uncached source selection and ten fresh fits")
        records = [store.verify("runs", r["run_id"]) for r in result["runs"]]
        source = store.verify("source_cache", records[0]["source_cv_id"])
        adaptation = {
            r["canonical_key"]["method"]: r["timing"]["adaptation_fit_seconds"]
            for r in records
            if r["canonical_key"]["method"] != "source_only"
        }
        result = {
            "task_id": task,
            "seed": 17,
            "evidence_kind": "real_fit_only_benchmark",
            "target_evaluation_opened": False,
            "observed_at": now(),
            "study_id": lock["study_id"],
            "hardware": hardware_report(),
            **resource_measurement(),
            "wall_seconds": wall,
            "preprocessing_seconds": source["preprocessing_fit_seconds"]
            + source["source_transform_seconds"],
            "tuning_seconds": source["source_search_seconds"],
            "training_seconds": sum(r["timing"]["classifier_fit_seconds"] for r in records),
            "adaptation_seconds_by_method": adaptation,
            "classifier_seconds_by_method": {
                m: sum(
                    r["timing"]["classifier_fit_seconds"]
                    for r in records
                    if r["canonical_key"]["method"] == m
                )
                for m in ("source_only", "importance_weighting", "coral")
            },
            "run_ids": [r["run_id"] for r in records],
            "source_cv_id": source["identity"],
            "source_samples": source["source_count"],
            "new_fits": 10,
            "source_search_cached": False,
            "scope": "Full frozen source CV and preprocessing; all three classifiers "
            "and methods plus majority; persistence included in wall time",
        }
        write_json(destination / "benchmark.json", seal(result))
        return result


def estimate_runtime(directory: Path) -> dict:
    tasks = ("gas_b01_b10", "har_official_group_transfer", "amazon_books_electronics")
    rows = [read_json(directory / "benchmarks" / task / "benchmark.json") for task in tasks]
    for row in rows:
        verify_seal(row)
    components = []
    for row, transfers, sources in zip(rows, (9, 1, 12), (1, 1, 4), strict=True):
        tuning = 5 * sources * row["tuning_seconds"]
        preprocessing = 5 * sources * row["preprocessing_seconds"]
        training = 5 * transfers * row["training_seconds"]
        adaptation = 5 * transfers * sum(row["adaptation_seconds_by_method"].values())
        sensitivity = 5 * 2 * sum(row["adaptation_seconds_by_method"].values()) + 5 * 2 * sum(
            row["classifier_seconds_by_method"][m] for m in ("importance_weighting", "coral")
        )
        residual = max(
            0,
            row["wall_seconds"]
            - row["tuning_seconds"]
            - row["preprocessing_seconds"]
            - row["training_seconds"]
            - sum(row["adaptation_seconds_by_method"].values()),
        )
        overhead = 5 * transfers * residual + 20 * residual
        components.append(
            {
                "dataset": row["task_id"].split("_", 1)[0],
                "source_searches": 5 * sources,
                "task_seed_jobs": 5 * transfers,
                "estimated_tuning_seconds": tuning,
                "estimated_preprocessing_seconds": preprocessing,
                "estimated_primary_training_seconds": training,
                "estimated_primary_adaptation_seconds": adaptation,
                "estimated_sensitivity_seconds": sensitivity,
                "estimated_io_and_validation_seconds": overhead,
                "estimated_fit_total_seconds": tuning
                + preprocessing
                + training
                + adaptation
                + sensitivity
                + overhead,
            }
        )
    total = sum(r["estimated_fit_total_seconds"] for r in components)
    result = {
        "evidence_kind": "measurement_based_projection_not_observed_completion",
        "observed_at": now(),
        "components": components,
        "estimated_all_fitting_seconds": total,
        "estimated_complete_study_seconds_range": [total * 1.2, total * 1.6],
        "range_definition": "20–60% allowance for unmeasured final prediction, "
        "1000-group-bootstrap analysis, exports and variation; "
        "planning range, not a statistical interval",
        "peak_benchmark_process_ram_bytes": max(r["peak_process_ram_bytes"] for r in rows),
        "expected_primary_evaluations": 1100,
        "expected_sensitivity_evaluations": 180,
        "source_searches": 30,
        "limitations": "One transfer/seed per dataset. Gas batch sizes, four Amazon source "
        "vocabularies and target pools, thermal throttling, other applications and bootstrap "
        "cost vary. CPU single-thread policy unchanged. OS process RAM excludes other "
        "applications. Whole-source CV is shared across targets; primary fits from benchmark "
        "are reusable. Estimate is from scratch and includes the benchmark work.",
    }
    write_json(directory / "runtime_estimate.json", seal(result))
    write_csv(directory / "runtime_estimate.csv", components)
    return result
