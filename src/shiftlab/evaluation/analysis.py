"""Saved-prediction analysis with strict pairing, independent scores and traceable exports."""

import csv
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shiftlab.data.parsers import CLASS_MAPS
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
from shiftlab.evaluation.statistics import (
    confusion_by_group,
    evaluation_units,
    paired_group_bootstrap,
    scores_from_confusion,
)
from shiftlab.experiments.checkpoints import Journal, job_owner, write_csv
from shiftlab.experiments.provenance import file_sha256
from shiftlab.experiments.storage import LocalArtifactStore, now
from shiftlab.experiments.study import checkpoint, fitted_rows, open_study

METRICS = ("accuracy", "macro_f1", "macro_precision", "macro_recall")


def validate_pair(base: dict, other: dict) -> None:
    for field in (
        "dataset_hash",
        "task_id",
        "split_hash",
        "model",
        "selected_source_parameter_hash",
        "preprocessing_hash",
        "seed",
        "protocol_version",
        "config_hash",
        "code_version",
        "code_snapshot_hash",
        "dependency_lock_hash",
    ):
        if base["canonical_key"][field] != other["canonical_key"][field]:
            raise DataError(f"Incompatible paired comparison: {field}")


def load_predictions(path: Path, expected_ids: list[str]) -> dict:
    with path.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if [r["sample_id"] for r in rows] != expected_ids:
        raise DataError("Saved predictions must contain every ordered evaluation ID exactly once")
    return {
        "true": np.array([int(r["y_true"]) for r in rows]),
        "pred": np.array([int(r["y_pred"]) for r in rows]),
        "groups": tuple(r["evaluation_group_id"] for r in rows),
    }


def _save_figure(path: Path, figure, trace: list[str], traces: dict):
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    traces[path.name] = sorted(set(trace))


def summarize_study(root: Path, study_id: str, progress=None, output_root="artifacts") -> dict:
    directory, lock = open_study(root, study_id, output_root)
    destination = directory / "analysis"
    with job_owner(directory):
        journal = Journal(directory)
        fitted = fitted_rows(journal)
        evaluations = {
            e["row_id"]: e["evaluation_id"]
            for e in journal.events
            if e.get("event") == "evaluation_saved"
        }
        if len(evaluations) != 1280 or set(evaluations) != set(fitted):
            raise DataError(
                "Analysis requires complete coverage; "
                "missing runs remain explicit in missing_runs.csv"
            )
        store = LocalArtifactStore(root / lock["configuration"]["execution"]["output_root"])
        if destination.exists():
            manifest = read_json(destination / "export_manifest.json")
            verify_seal(manifest)
            if manifest["study_lock_hash"] != lock["manifest_hash"]:
                raise DataError("Analysis lock differs")
            verify_files(destination, manifest["output_hashes"])
            return {
                "status": "analysis_verified",
                "study_id": study_id,
                "path": destination.relative_to(root).as_posix(),
            }
        stage = stage_directory(destination)
        records, predictions, metric_rows, inputs, diagnostics = {}, {}, [], {}, []
        splits = {
            task: read_json(
                root
                / lock["data_root"]
                / "splits"
                / lock["protocol_version"]
                / task
                / "manifest.json"
            )
            for task in lock["split_hashes"]
        }
        for number, (row_id, row) in enumerate(fitted.items()):
            record = store.verify("runs", row["run_id"])
            evaluation = store.verify("evaluations", evaluations[row_id])
            epath = store.directory("evaluations", evaluations[row_id])
            scores = read_json(epath / "metrics.json")
            pred = load_predictions(
                epath / "predictions.csv", lock["evaluation_ids"][row["task_id"]]
            )
            split = splits[row["task_id"]]
            if pred["groups"] != tuple(split["target_eval_group_ids"]):
                raise DataError("Prediction grouping differs from frozen split")
            known = tuple(CLASS_MAPS[row["dataset"]])
            cm, _ = confusion_by_group(pred["true"], pred["pred"][None, :], pred["groups"], known)
            macro = np.array([known.index(int(c)) for c in np.unique(pred["true"])])
            independent = scores_from_confusion(cm.sum(axis=0), macro)
            for metric in METRICS:
                if not np.isclose(scores[metric], independent[metric][0], rtol=0, atol=1e-12):
                    raise DataError(f"Independent saved-prediction recomputation differs: {metric}")
            if (
                scores["confusion_matrix"] != cm.sum(axis=(0, 1)).tolist()
                or scores["macro_classes"] != np.unique(pred["true"]).tolist()
            ):
                raise DataError("Saved confusion matrix or class policy differs")
            records[row_id], predictions[row_id] = record, pred
            timings = evaluation["timing"]
            metric_rows.append(
                {
                    **row,
                    "evaluation_id": evaluations[row_id],
                    **{m: scores[m] for m in METRICS},
                    "macro_classes": ",".join(map(str, scores["macro_classes"])),
                    "sample_count": scores["sample_count"],
                    "preprocessing_seconds": timings["preprocessing_fit_seconds"]
                    + timings["source_transform_seconds"],
                    "source_tuning_seconds": timings["shared_source_search_seconds"],
                    "adaptation_seconds": timings["adaptation_fit_seconds"],
                    "training_seconds": timings["classifier_fit_seconds"],
                    "prediction_seconds": timings["target_prediction_seconds"],
                    "pipeline_seconds": timings["pipeline_fit_predict_seconds"],
                }
            )
            inputs[row["run_id"]] = {
                "fit_manifest_sha256": file_sha256(
                    store.directory("runs", row["run_id"]) / "manifest.json"
                ),
                "evaluation_id": evaluations[row_id],
                "prediction_path": (epath / "predictions.csv").relative_to(root).as_posix(),
                "predictions_sha256": file_sha256(epath / "predictions.csv"),
                "metrics_sha256": file_sha256(epath / "metrics.json"),
            }
            diag = read_json(store.directory("runs", row["run_id"]) / "diagnostics.json")
            diagnostics.append(
                {
                    **row,
                    "warnings": " | ".join(record["warnings"]),
                    "importance": diag["importance_weighting"],
                    "coral": diag["coral"],
                }
            )
            if progress and number % 100 == 0:
                progress({"event": "metrics_verified", "count": number + 1})
        write_csv(stage / "per_run_metrics.csv", metric_rows)
        frame = pd.DataFrame(metric_rows)
        index = {
            (r["task_id"], r["model"], r["seed"], r["setting"], r["method"]): r
            for r in fitted.values()
        }
        deltas, intervals, pair_summary = [], [], []
        for number, ((task, model, setting, method), _selected) in enumerate(
            frame[frame.method != "source_only"].groupby(
                ["task_id", "model", "setting", "method"], sort=True
            )
        ):
            source_predictions, adapted_predictions, pairs = [], [], []
            true = None
            for seed in lock["configuration"]["seeds"]:
                base = index[(task, model, seed, "primary", "source_only")]
                other = index[(task, model, seed, setting, method)]
                validate_pair(records[base["row_id"]], records[other["row_id"]])
                b, a = predictions[base["row_id"]], predictions[other["row_id"]]
                if not np.array_equal(b["true"], a["true"]) or b["groups"] != a["groups"]:
                    raise DataError("Paired target truth or groups differ")
                if true is not None and not np.array_equal(true, b["true"]):
                    raise DataError("Five seeds do not share evaluation truth")
                true = b["true"]
                source_predictions.append(b["pred"])
                adapted_predictions.append(a["pred"])
                bs = frame[frame.run_id == base["run_id"]].iloc[0]
                ars = frame[frame.run_id == other["run_id"]].iloc[0]
                paired = {
                    "dataset": base["dataset"],
                    "task_id": task,
                    "model": model,
                    "method": method,
                    "setting": setting,
                    "seed": seed,
                    "source_run_id": base["run_id"],
                    "adapted_run_id": other["run_id"],
                    **{f"{m}_delta": float(ars[m] - bs[m]) for m in METRICS},
                    "pipeline_seconds_delta": float(ars.pipeline_seconds - bs.pipeline_seconds),
                }
                deltas.append(paired)
                pairs.append(paired)
            units, unit_policy = evaluation_units(splits[task])
            ci = paired_group_bootstrap(
                true,
                np.array(source_predictions),
                np.array(adapted_predictions),
                units,
                known_classes=tuple(CLASS_MAPS[pairs[0]["dataset"]]),
            )
            reference = {
                "dataset": pairs[0]["dataset"],
                "task_id": task,
                "model": model,
                "method": method,
                "setting": setting,
                "source_run_ids": "|".join(p["source_run_id"] for p in pairs),
                "adapted_run_ids": "|".join(p["adapted_run_id"] for p in pairs),
                "paired_seeds": 5,
            }
            intervals.append({**reference, "unit_policy": unit_policy, **ci})
            pair_summary.append(
                {
                    **reference,
                    **{
                        f"{m}_delta_mean": float(np.mean([p[f"{m}_delta"] for p in pairs]))
                        for m in METRICS
                    },
                    **{
                        f"{m}_delta_std": float(np.std([p[f"{m}_delta"] for p in pairs], ddof=1))
                        for m in METRICS
                    },
                    "macro_f1_lower": ci["macro_f1_lower"],
                    "macro_f1_upper": ci["macro_f1_upper"],
                }
            )
            if progress and number % 10 == 0:
                progress({"event": "bootstrap_complete", "comparisons": number + 1})
        write_csv(stage / "paired_deltas.csv", deltas)
        write_csv(stage / "bootstrap_intervals.csv", intervals)
        write_csv(stage / "paired_summary.csv", pair_summary)
        task_rows = []
        pair_lookup = {
            (r["task_id"], r["model"], r["setting"], r["method"]): r for r in pair_summary
        }
        for (dataset, task, model, setting, method), group in frame.groupby(
            ["dataset", "task_id", "model", "setting", "method"], sort=True
        ):
            row = {
                "dataset": dataset,
                "task_id": task,
                "model": model,
                "method": method,
                "setting": setting,
                "seed_count": len(group),
                "run_ids": "|".join(group.run_id),
                "sample_count": int(group.sample_count.iloc[0]),
                **{f"{m}_mean": float(group[m].mean()) for m in (*METRICS, "pipeline_seconds")},
                **{f"{m}_std": float(group[m].std(ddof=1)) for m in (*METRICS, "pipeline_seconds")},
            }
            pair = pair_lookup.get((task, model, setting, method))
            row.update(
                {
                    "macro_f1_delta_mean": pair["macro_f1_delta_mean"] if pair else None,
                    "macro_f1_delta_std": pair["macro_f1_delta_std"] if pair else None,
                    "macro_f1_lower": pair["macro_f1_lower"] if pair else None,
                    "macro_f1_upper": pair["macro_f1_upper"] if pair else None,
                    "source_run_ids": pair["source_run_ids"] if pair else None,
                }
            )
            task_rows.append(row)
        write_csv(stage / "per_task_summary.csv", task_rows)
        summaries = []
        task_frame = pd.DataFrame(task_rows)
        columns = [f"{m}_mean" for m in (*METRICS, "pipeline_seconds")] + ["macro_f1_delta_mean"]
        for (dataset, model, method), group in task_frame[task_frame.setting == "primary"].groupby(
            ["dataset", "model", "method"], sort=True
        ):
            summaries.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "method": method,
                    "task_count": len(group),
                    "weighting": "Equal task weights within dataset",
                    "run_ids": "|".join(group.run_ids),
                    **{
                        c: None if group[c].isna().all() else float(group[c].mean())
                        for c in columns
                    },
                }
            )
        write_csv(stage / "dataset_summary.csv", summaries)
        overall = []
        for (model, method), group in pd.DataFrame(summaries).groupby(
            ["model", "method"], sort=True
        ):
            overall.append(
                {
                    "model": model,
                    "method": method,
                    "datasets": len(group),
                    "weighting": "Descriptive equal weights across gas, HAR, Amazon dataset means",
                    "run_ids": "|".join(group.run_ids),
                    **{
                        c: None if group[c].isna().all() else float(group[c].mean())
                        for c in columns
                    },
                }
            )
        write_csv(stage / "descriptive_overall.csv", overall)
        ablations = frame[
            (frame.task_id.isin(lock["configuration"]["ablations"]["tasks"]))
            & (frame.method != "source_only")
        ].copy()
        ablations["sensitivity_value"] = ablations.apply(
            lambda r: (
                (10.0 if r.method == "importance_weighting" else 0.001)
                if r.setting == "primary"
                else float(r.setting.split(":")[1])
            ),
            axis=1,
        )
        write_csv(stage / "ablations.csv", ablations.to_dict("records"))
        write_json(stage / "diagnostics.json", {"runs": diagnostics})
        traces = {}
        figures = stage / "figures"
        figures.mkdir()
        colors = {"source_only": "#475569", "importance_weighting": "#0369a1", "coral": "#b45309"}
        pairs_frame = pd.DataFrame(pair_summary)
        for dataset, selected in pairs_frame[pairs_frame.setting == "primary"].groupby("dataset"):
            selected = selected.sort_values(["task_id", "model", "method"])
            fig, axis = plt.subplots(figsize=(12, max(4, len(selected) * 0.17)))
            labels = []
            for i, r in enumerate(selected.itertuples()):
                axis.plot(
                    [100 * r.macro_f1_lower, 100 * r.macro_f1_upper], [i, i], color=colors[r.method]
                )
                axis.scatter(100 * r.macro_f1_delta_mean, i, color=colors[r.method], s=18)
                labels.append(f"{r.task_id} / {r.model} / {r.method}")
            axis.axvline(0, color="#64748b", linewidth=1)
            axis.set(
                yticks=range(len(labels)),
                yticklabels=labels,
                xlabel="Paired macro F1 change (percentage points); unadjusted 95% group bootstrap",
                title=f"{dataset.upper()}: mean over five fitted seeds; negative transfer retained",
            )
            axis.tick_params(axis="y", labelsize=7)
            _save_figure(
                figures / f"{dataset}_paired_gains.png",
                fig,
                "|".join(
                    selected.source_run_ids.tolist() + selected.adapted_run_ids.tolist()
                ).split("|"),
                traces,
            )
        gas = task_frame[
            (task_frame.dataset == "gas")
            & (task_frame.setting == "primary")
            & (task_frame.model != "majority")
        ]
        fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
        for axis, model in zip(axes, ("lr", "linear_svm", "rf"), strict=True):
            for method, selected in gas[gas.model == model].groupby("method"):
                selected = selected.sort_values("task_id")
                axis.errorbar(
                    [int(t[-2:]) for t in selected.task_id],
                    selected.macro_f1_mean,
                    yerr=selected.macro_f1_std,
                    label=method,
                    color=colors[method],
                    marker="o",
                    markersize=3,
                )
            axis.set(title=model, xlabel="Target gas batch", ylim=(0, 1))
        axes[0].set_ylabel("Macro F1 (fraction); error bars: sample SD across five seeds")
        axes[-1].legend(fontsize=7)
        _save_figure(
            figures / "gas_temporal_macro_f1.png", fig, "|".join(gas.run_ids).split("|"), traces
        )
        for task in lock["configuration"]["ablations"]["tasks"]:
            for method in ("importance_weighting", "coral"):
                selected = ablations[(ablations.task_id == task) & (ablations.method == method)]
                fig, axis = plt.subplots(figsize=(6, 4))
                for model, group in selected.groupby("model"):
                    stat = group.groupby("sensitivity_value").macro_f1.agg(["mean", "std"])
                    axis.errorbar(
                        stat.index, stat["mean"], yerr=stat["std"], marker="o", label=model
                    )
                if method == "coral":
                    axis.set_xscale("log")
                axis.set(
                    xlabel="Raw upper weight clipping bound"
                    if method == "importance_weighting"
                    else "CORAL covariance ridge",
                    ylabel="Macro F1; mean and sample SD across five seeds",
                    title=f"{task}: predeclared {method} sensitivity",
                )
                axis.legend()
                _save_figure(
                    figures / f"{task}_{method}_sensitivity.png",
                    fig,
                    selected.run_id.tolist(),
                    traces,
                )
            for method in ("source_only", "importance_weighting", "coral"):
                for model in ("lr", "linear_svm", "rf"):
                    row = index[(task, model, 17, "primary", method)]
                    scores = read_json(
                        store.directory("evaluations", evaluations[row["row_id"]]) / "metrics.json"
                    )
                    fig, axis = plt.subplots(figsize=(5, 4))
                    image = axis.imshow(scores["confusion_matrix"], cmap="Blues")
                    cls = scores["confusion_matrix_classes"]
                    axis.set(
                        xticks=range(len(cls)),
                        yticks=range(len(cls)),
                        xticklabels=cls,
                        yticklabels=cls,
                        xlabel="Predicted canonical class",
                        ylabel="Actual canonical class",
                        title=f"{task}\n{model} / {method} / seed 17",
                    )
                    fig.colorbar(image, ax=axis, label="Evaluation samples")
                    _save_figure(
                        figures / f"{task}_{model}_{method}_confusion.png",
                        fig,
                        [row["run_id"]],
                        traces,
                    )
        _diagnostic_figures(store, fitted, records, figures, traces)
        primary_pairs = pairs_frame[pairs_frame.setting == "primary"]
        positive = primary_pairs[primary_pairs.macro_f1_delta_mean > 0]
        negative = primary_pairs[primary_pairs.macro_f1_delta_mean < 0]
        uncertain = primary_pairs[
            (primary_pairs.macro_f1_lower <= 0) & (primary_pairs.macro_f1_upper >= 0)
        ]
        examples = {}
        for name, selected in (
            ("positive", positive.sort_values("macro_f1_delta_mean", ascending=False)),
            ("negative", negative.sort_values("macro_f1_delta_mean")),
            ("uncertain", uncertain.sort_values("macro_f1_delta_mean", key=abs)),
        ):
            examples[name] = selected.iloc[0].to_dict() if len(selected) else None
        findings = {
            "observed_at": now(),
            "study_id": study_id,
            "primary_comparisons": len(primary_pairs),
            "sensitivity_comparisons": len(pairs_frame) - len(primary_pairs),
            "observed_positive_macro_f1_changes": len(positive),
            "observed_negative_transfer": len(negative),
            "zero_changes": int((primary_pairs.macro_f1_delta_mean == 0).sum()),
            "intervals_including_zero": len(uncertain),
            "examples": examples,
            "independent_metric_recomputations": len(frame),
            "limitations": [
                "Five seeds share target rows and are not independent datasets.",
                "Bootstrap intervals are conditional on fitted models and observed domains; "
                "168 intervals are unadjusted and establish no universal winner.",
                "HAR has only five evaluation participants; "
                "population inference and interval stability are limited.",
                "Amazon transfers overlap categories; use product groups when available. "
                "Duplicate-safe gas measurements may retain temporal dependence.",
                "Class sets are those observed in each full target test; "
                "cross-task scores may average different classes.",
                "Only predeclared sensitivity values are reported; "
                "primary settings are unchanged after evaluation.",
                "Reduced covariance mismatch does not imply improved prediction; "
                "weighting may fail when support/conditional-label assumptions fail.",
                "Single-thread CPU timings include cached original component costs "
                "for fair per-pipeline comparisons; sum unique caches for actual study compute, "
                "rather than summing shared costs across every row.",
            ],
        }
        write_json(stage / "findings.json", findings)
        prose = [
            "# Observed study findings",
            "",
            f"Study `{study_id}` completed 1,100 primary and 180 sensitivity evaluations. "
            "All 1,280 score records independently recompute from saved predictions.",
            "",
            f"Across 132 primary paired comparisons, {len(positive)} mean macro F1 changes "
            f"are positive, {len(negative)} show observed negative transfer and "
            f"{findings['zero_changes']} are zero. {len(uncertain)} unadjusted intervals "
            "include zero. These counts describe this study; they do not establish "
            "statistical significance or a universal winner.",
            "",
            "## Examples for interpretation",
            "",
        ]
        for name, example in examples.items():
            prose.append(
                f"{name.capitalize()}: "
                + (
                    f"{example['task_id']} / {example['model']} / {example['method']}: "
                    f"mean macro F1 change {100 * example['macro_f1_delta_mean']:.3f} "
                    f"percentage points; interval [{100 * example['macro_f1_lower']:.3f}, "
                    f"{100 * example['macro_f1_upper']:.3f}]. "
                    "Run IDs are in findings.json and paired_summary.csv."
                    if example
                    else "No such example was observed; none is invented."
                )
            )
            prose.append("")
        prose.extend(
            ["## Interpretation and limitations", "", *[f"- {v}" for v in findings["limitations"]]]
        )
        (stage / "findings.md").write_text("\n".join(prose) + "\n", encoding="utf-8")
        write_json(stage / "figure_traces.json", traces)
        output_hashes = {
            p.relative_to(stage).as_posix(): file_sha256(p)
            for p in sorted(stage.rglob("*"))
            if p.is_file()
        }
        write_json(
            stage / "export_manifest.json",
            seal(
                {
                    "schema_version": 1,
                    "study_id": study_id,
                    "study_lock_hash": lock["manifest_hash"],
                    "fit_completion_lock_sha256": file_sha256(
                        directory / "fit_completion_lock.json"
                    ),
                    "code_version": lock["code_version"],
                    "code_snapshot_hash": lock["code_snapshot_hash"],
                    "dependency_lock_hash": lock["dependency_lock_hash"],
                    "command": f"shiftlab study summarize --study-id {study_id}",
                    "input_hashes": inputs,
                    "output_hashes": output_hashes,
                    "figure_run_ids": traces,
                    "created_at": now(),
                }
            ),
        )
        publish_directory(stage, destination)
        journal.append(
            {
                "event": "analysis_completed",
                "export_manifest_sha256": file_sha256(destination / "export_manifest.json"),
            }
        )
        checkpoint(directory, lock, journal, "complete")
        return {
            "status": "complete",
            "study_id": study_id,
            "primary_evaluations": 1100,
            "sensitivity_evaluations": 180,
            "path": destination.relative_to(root).as_posix(),
            "findings": findings,
        }


def _diagnostic_figures(store, fitted, records, figures, traces):
    weighted, aligned = [], []
    for _row_id, row in fitted.items():
        if row["setting"] != "primary" or row["seed"] != 17 or row["model"] != "lr":
            continue
        run_dir = store.directory("runs", row["run_id"])
        if row["method"] == "importance_weighting":
            diag = read_json(run_dir / "diagnostics.json")["importance_weighting"]
            weighted.append((row, diag, run_dir))
        elif row["method"] == "coral":
            aligned.append((row, read_json(run_dir / "diagnostics.json")["coral"]))
    # Read only numerical saved adaptation state; never deserialize external models.
    for row, _, run_dir in weighted:
        with np.load(run_dir / "adaptation_state.npz", allow_pickle=False) as arrays:
            weights = arrays["weights"]
        fig, axis = plt.subplots(figsize=(6, 4))
        axis.hist(weights, bins=40, color="#0369a1")
        ess = weights.sum() ** 2 / np.sum(weights**2)
        axis.set(
            xlabel="Mean-one source importance weight",
            ylabel="Source samples",
            title=f"{row['task_id']} / seed 17\nESS {ess:.2f} / {len(weights)} source samples",
        )
        _save_figure(
            figures / f"{row['task_id']}_weights.png",
            fig,
            [
                r["run_id"]
                for r in fitted.values()
                if r["task_id"] == row["task_id"]
                and r["seed"] == 17
                and r["method"] == "importance_weighting"
                and r["setting"] == "primary"
            ],
            traces,
        )
    if aligned:
        fig, axis = plt.subplots(figsize=(12, 5))
        before = [r[1]["covariance_mismatch_before"] for r in aligned]
        after = [r[1]["covariance_mismatch_after"] for r in aligned]
        x = np.arange(len(aligned))
        axis.bar(x - 0.18, before, width=0.36, label="Before CORAL", color="#475569")
        axis.bar(x + 0.18, after, width=0.36, label="After CORAL", color="#b45309")
        axis.set(
            xticks=x,
            xticklabels=[r[0]["task_id"] for r in aligned],
            ylabel="Relative Frobenius covariance mismatch",
            title="Allowed adaptation features, seed 17; "
            "mismatch does not imply predictive benefit",
        )
        axis.tick_params(axis="x", rotation=80, labelsize=7)
        axis.legend()
        _save_figure(
            figures / "coral_covariance_mismatch.png",
            fig,
            [r[0]["run_id"] for r in aligned],
            traces,
        )
