"""Prepare small reviewable study evidence; raw datasets, predictions and models stay local."""

import argparse
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from shiftlab.data.storage import (
    DataError,
    publish_directory,
    read_json,
    seal,
    stage_directory,
    verify_files,
    verify_seal,
    write_json,
)
from shiftlab.experiments.checkpoints import write_csv
from shiftlab.experiments.provenance import file_sha256
from shiftlab.experiments.study import open_study


def prepare(
    root: Path, study_id: str, destination: Path, figure_review: Path | None = None
) -> dict:
    directory, lock = open_study(root, study_id)
    progress = read_json(directory / "checkpoint.json")
    verify_seal(progress)
    if progress["status"] != "complete" or progress["evaluated_records"] != 1280:
        raise DataError("Evidence publication requires the actual completed study")
    analysis = directory / "analysis"
    export = read_json(analysis / "export_manifest.json")
    verify_seal(export)
    verify_files(analysis, export["output_hashes"])
    reviewed = None
    if figure_review is not None:
        reviewed = read_json(figure_review / "visual_review_manifest.json")
        verify_seal(reviewed)
        verify_files(figure_review, reviewed["output_hashes"])
        if reviewed["study_id"] != study_id or not reviewed["numerical_inputs_unchanged"]:
            raise DataError("Publication figure review must match the unchanged study")
        verify_files(analysis, reviewed["input_hashes"])
    stage = stage_directory(destination)
    tables = (
        "per_run_metrics.csv",
        "paired_deltas.csv",
        "paired_summary.csv",
        "per_task_summary.csv",
        "dataset_summary.csv",
        "descriptive_overall.csv",
        "bootstrap_intervals.csv",
        "ablations.csv",
        "findings.json",
    )
    for name in tables:
        shutil.copyfile(analysis / name, stage / name)
    # Diagnostic aggregates have no sample-level predictions or review bodies.
    diagnostic_rows = read_json(analysis / "diagnostics.json")["runs"]
    diagnostics = {r["run_id"]: r for r in diagnostic_rows}
    pairs = pd.read_csv(analysis / "paired_summary.csv")
    intervals = pd.read_csv(analysis / "bootstrap_intervals.csv")
    conditions = []
    for row in pairs[pairs.setting == "primary"].to_dict("records"):
        split = read_json(
            root
            / lock["data_root"]
            / "splits"
            / lock["protocol_version"]
            / row["task_id"]
            / "manifest.json"
        )
        values = [diagnostics[identity] for identity in row["adapted_run_ids"].split("|")]
        ci = intervals[
            (intervals.task_id == row["task_id"])
            & (intervals.model == row["model"])
            & (intervals.method == row["method"])
            & (intervals.setting == "primary")
        ].iloc[0]
        conditions.append(
            {
                **row,
                "source_samples": len(split["source_ids"]),
                "target_adaptation_samples": len(split["target_adapt_ids"]),
                "evaluation_samples": int(ci.sample_count),
                "evaluation_groups": int(ci.group_count),
                "bootstrap_unit": ci.unit_policy,
                "interval_includes_zero": row["macro_f1_lower"] <= 0 <= row["macro_f1_upper"],
                "observed_negative_transfer": row["macro_f1_delta_mean"] < 0,
                "mean_ess_fraction": float(
                    np.mean([d["importance"]["ess_fraction"] for d in values])
                )
                if row["method"] == "importance_weighting"
                else None,
                "mean_covariance_mismatch_before": float(
                    np.mean([d["coral"]["covariance_mismatch_before"] for d in values])
                )
                if row["method"] == "coral"
                else None,
                "mean_covariance_mismatch_after": float(
                    np.mean([d["coral"]["covariance_mismatch_after"] for d in values])
                )
                if row["method"] == "coral"
                else None,
            }
        )
    write_csv(stage / "conditions.csv", conditions)
    benchmarks = []
    benchmark_metadata = {}
    for task in lock["configuration"]["ablations"]["tasks"]:
        path = directory / "benchmarks" / task / "benchmark.json"
        measurement = read_json(path)
        verify_seal(measurement)
        benchmarks.append(
            {
                "task_id": task,
                "seed": measurement["seed"],
                "source_samples": measurement["source_samples"],
                "preprocessing_seconds": measurement["preprocessing_seconds"],
                "source_tuning_seconds_including_fold_preprocessing": measurement["tuning_seconds"],
                "classifier_training_seconds_all_ten_fits": measurement["training_seconds"],
                "importance_adaptation_seconds": measurement["adaptation_seconds_by_method"][
                    "importance_weighting"
                ],
                "coral_adaptation_seconds": measurement["adaptation_seconds_by_method"]["coral"],
                "wall_seconds": measurement["wall_seconds"],
                "peak_process_working_set_bytes": measurement["peak_process_ram_bytes"],
                "peak_process_commit_bytes": measurement["peak_process_commit_bytes"],
                "new_fits": measurement["new_fits"],
                "source_cv_id": measurement["source_cv_id"],
                "run_ids": "|".join(measurement["run_ids"]),
            }
        )
        benchmark_metadata[task] = {
            "benchmark_sha256": file_sha256(path),
            "boundary_sha256": file_sha256(directory / f"benchmark_boundary_{task}.json"),
            "observed_at": measurement["observed_at"],
            "cpu": measurement["hardware"]["cpu"],
            "physical_ram_bytes": measurement["hardware"]["physical_ram_bytes"],
            "memory_definition": measurement["definition"],
        }
    write_csv(stage / "benchmarks.csv", benchmarks)
    shutil.copyfile(directory / "runtime_estimate.json", stage / "runtime_estimate.json")
    shutil.copyfile(directory / "runtime_estimate.csv", stage / "runtime_estimate.csv")
    if (directory / "execution_receipt.json").exists():
        receipt = read_json(directory / "execution_receipt.json")
        write_json(
            stage / "execution_time.json",
            {
                "elapsed_seconds_this_invocation": receipt["elapsed_seconds_this_invocation"],
                "definition": (
                    "Successful invocation with cache verification and its remaining fitting, "
                    "evaluation and analysis work. Excludes earlier unsuccessful invocations, "
                    "benchmarks, startup and idle time; not total elapsed study runtime."
                ),
                "initial_fitting_boundary_seconds": read_json(
                    directory / "fit_boundary_receipt.json"
                )["elapsed_seconds"],
                "fitting_boundary_definition": (
                    "First completed fitting-stage boundary, including exact benchmark reuse. "
                    "Can overlap a successful invocation; do not sum these fields blindly."
                ),
            },
        )
    condition_frame = pd.DataFrame(conditions)
    weighted = condition_frame[condition_frame.method == "importance_weighting"]
    low = weighted[weighted.mean_ess_fraction < 0.1]
    coral = condition_frame[condition_frame.method == "coral"]
    reduced = coral[coral.mean_covariance_mismatch_after < coral.mean_covariance_mismatch_before]
    findings = read_json(analysis / "findings.json")
    operations, incidents = [], []
    for path in sorted(
        (root / lock["configuration"]["execution"]["output_root"] / "operations").glob("*.json")
    ):
        record = read_json(path)
        verify_seal(record)
        if record.get("study_id") == study_id:
            operations.append(
                {"receipt_name": path.name, "sha256": file_sha256(path), "record": record}
            )
            incidents.extend(record.get("incidents", []))
    write_json(stage / "operational_events.json", {"receipts": operations})
    text = [
        "# Saved study evidence",
        "",
        f"Study `{study_id}` has **1,100 primary and 180 sensitivity evaluations**, "
        "30 shared source searches and complete declared coverage. All 1,280 metric records "
        "independently recompute from saved predictions. There are 132 primary and 36 "
        "sensitivity paired comparisons, each using five fitted seeds and 1,000 grouped "
        "bootstrap replicates.",
        "",
        "`benchmarks.csv` records three fresh CPU tasks with separate final preprocessing, "
        "source tuning (including fold preprocessing), classifier training, adaptation and "
        "process peak RAM. `runtime_estimate.json` is a projection from these measurements, "
        "not an observed completion. `execution_time.json` records the observed invocation "
        "duration when available, separate from the first completed fitting boundary. "
        "Interrupted attempts and idle time are excluded from that invocation measurement. "
        "CPU policy remains one job/thread; the RTX GPU is unused.",
        "",
        f"Observed operational progress-file interruptions: {len(incidents)}. Saved fits and "
        "evaluations were preserved and resumed under the same frozen settings. "
        "`operational_events.json` records incident/log hashes and bounded progress-only "
        "replacement retries. These are metadata I/O interruptions, not classifier failures.",
        "",
        "## Observed adaptation conditions",
        "",
        f"Importance weighting shows {int((weighted.macro_f1_delta_mean > 0).sum())} "
        f"positive and {int((weighted.macro_f1_delta_mean < 0).sum())} negative mean macro F1 "
        f"changes across 66 model/task comparisons. {len(low)} comparisons have mean ESS "
        f"below 10% of source size; {int((low.macro_f1_delta_mean < 0).sum())} of those show "
        "observed negative transfer. These model/task counts are dependent descriptive "
        "evidence, not independent statistical trials.",
        "",
        f"CORAL reduces mean adaptation-pool covariance mismatch in {len(reduced)} of 66 "
        f"model/task comparisons. {int((reduced.macro_f1_delta_mean < 0).sum())} of those "
        "nevertheless show a negative mean macro F1 change. Covariance agreement measures "
        "marginal feature geometry and does not guarantee that source labels transfer correctly. "
        "Models share the transform within each transfer/seed; these rows are dependent.",
        "",
        "`conditions.csv` ties every primary comparison to the actual transfer, model, "
        "sample/group counts, ESS or covariance diagnostics, paired delta, conditional "
        "interval and exact run IDs. These are observed associations. The study does not "
        "establish causal explanations, label-conditional invariance, optimal target settings "
        "or a universal winning method.",
        "",
        "## Examples for the viva",
        "",
    ]
    for name, example in findings["examples"].items():
        if example:
            text.append(
                f"- {name.capitalize()}: `{example['task_id']}` / `{example['model']}` / "
                f"`{example['method']}`: mean macro F1 change "
                f"{100 * example['macro_f1_delta_mean']:.3f} percentage points, "
                f"unadjusted interval [{100 * example['macro_f1_lower']:.3f}, "
                f"{100 * example['macro_f1_upper']:.3f}]. Exact paired IDs are in findings.json."
            )
        else:
            text.append(f"- {name.capitalize()}: no example was observed; none is invented.")
    text.extend(
        [
            "",
            "## Limits and interpretation",
            "",
            "Five seeds share the same target rows; they are not five independent datasets. "
            "Resampling holds the fitted models fixed and includes entire participants/products/"
            "duplicate-safe measurement groups with multiplicity. HAR has only five evaluation "
            "participants. Gas temporal dependence can remain, and overlapping Amazon transfers "
            "are dependent. Accuracy and macro F1 intervals are unadjusted: 168 paired comparison "
            "rows contain 168 intervals per metric (336 total). No naive seed-level significance "
            "test is reported.",
            "",
            "Mean negative deltas are observed negative transfer. Intervals including zero "
            "communicate uncertainty; they do not prove equivalence. Macro class sets are those "
            "present in each full held-out target test. Task sizes and class composition vary. "
            "Dataset summaries weight tasks equally within each dataset; descriptive overall "
            "summaries weight gas, HAR and Amazon equally.",
            "",
            "Every predeclared clipping/ridge value appears in ablations.csv; primary clipping "
            "10 and ridge 0.001 are reused. Target scores were not used to select a new primary "
            "setting. Timing tables retain originating cached component costs and exclude shared "
            "tuning from per-pipeline runtime; the complete-study estimate counts shared searches/"
            "caches once.",
            "",
            "This folder contains aggregate scientific evidence and hashes. Raw reviews/features, "
            "sample-level true/predicted labels, serialized models, full environment paths and "
            "local working logs remain in ignored storage. The full local export manifest links "
            "prediction hashes and run IDs. The original objective and proposal abstract remain "
            "unchanged; no grade, novelty or publication guarantee is implied.",
            "",
            "Study design and method/dataset references: "
            "[research design](../research_design.md), "
            "[literature matrix](../literature_review_matrix.md) and "
            "[dataset cards](../data_preparation/).",
            "",
        ]
    )
    (stage / "README.md").write_text("\n".join(text), encoding="utf-8")
    selected_figures = (
        "gas_paired_gains.png",
        "har_paired_gains.png",
        "amazon_paired_gains.png",
        "gas_temporal_macro_f1.png",
        "coral_covariance_mismatch.png",
    )
    (stage / "figures").mkdir()
    for name in selected_figures:
        source = analysis / "figures" / name
        if reviewed is not None and name in reviewed["output_hashes"]:
            source = figure_review / name
        shutil.copyfile(source, stage / "figures" / name)
    traces = read_json(analysis / "figure_traces.json")
    if reviewed is not None:
        shutil.copyfile(
            figure_review / "visual_review_manifest.json", stage / "visual_review_manifest.json"
        )
        for name, run_ids in reviewed["figure_traces"].items():
            traces[name] = run_ids
    write_json(stage / "figure_traces.json", {name: traces[name] for name in selected_figures})
    output_hashes = {
        p.relative_to(stage).as_posix(): file_sha256(p) for p in stage.rglob("*") if p.is_file()
    }
    manifest = seal(
        {
            "study_id": study_id,
            "objective": lock["objective"],
            "status": "complete",
            "primary_evaluations": 1100,
            "sensitivity_evaluations": 180,
            "source_base_commit": lock["code_version"],
            "frozen_engine_snapshot_hash": lock["code_snapshot_hash"],
            "engine_file_hashes": lock["code_snapshot"],
            "dependency_lock_hash": lock["dependency_lock_hash"],
            "config_hash": lock["config_hash"],
            "raw_archive_hashes": lock["raw_archive_hashes"],
            "split_hashes": lock["split_hashes"],
            "local_export_manifest_sha256": file_sha256(analysis / "export_manifest.json"),
            "benchmark_evidence": benchmark_metadata,
            "publication_visual_revision": reviewed,
            "output_hashes": output_hashes,
            "publication_scope": (
                "Aggregate metrics/diagnostics, figures and provenance; "
                "no raw datasets, predictions or models"
            ),
        }
    )
    write_json(stage / "evidence_manifest.json", manifest)
    publish_directory(stage, destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/study"))
    parser.add_argument("--figure-review", type=Path)
    args = parser.parse_args()
    result = prepare(Path.cwd(), args.study_id, args.out, args.figure_review)
    print(f"Prepared aggregate evidence for {result['primary_evaluations']} primary evaluations")
