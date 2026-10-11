"""Render reviewed publication figures from saved evidence; leave frozen exports intact."""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
from shiftlab.experiments.provenance import file_sha256
from shiftlab.experiments.study import open_study


def render(root: Path, study_id: str, destination: Path) -> dict:
    study, _lock = open_study(root, study_id)
    analysis = study / "analysis"
    export = read_json(analysis / "export_manifest.json")
    verify_seal(export)
    verify_files(analysis, export["output_hashes"])
    stage = stage_directory(destination)
    frame = pd.read_csv(analysis / "per_task_summary.csv")
    gas = frame[(frame.dataset == "gas") & (frame.setting == "primary")]
    colors = {"source_only": "#475569", "importance_weighting": "#0369a1", "coral": "#b45309"}
    names = {
        "source_only": "Source only",
        "importance_weighting": "Importance weighting",
        "coral": "CORAL",
    }
    titles = {"lr": "Logistic Regression", "linear_svm": "Linear SVM", "rf": "Random Forest"}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), sharey=True, layout="constrained")
    traces, plotted = {}, []
    for axis, model in zip(axes, titles, strict=True):
        for method, color in colors.items():
            rows = gas[(gas.model == model) & (gas.method == method)].sort_values("task_id")
            axis.errorbar(
                rows.task_id.str.rsplit("b", n=1).str[-1].astype(int),
                rows.macro_f1_mean,
                yerr=rows.macro_f1_std,
                color=color,
                marker="o",
                markersize=3,
                label=names[method],
            )
            plotted.extend("|".join(rows.run_ids).split("|"))
        axis.set(title=titles[model], xlabel="Target batch (source: batch 1)", ylim=(0, 1))
        axis.set_xticks(range(2, 11))
    axes[0].set_ylabel("Macro F1 (fraction)")
    axes[-1].legend(fontsize=8)
    fig.suptitle(
        "Gas temporal transfers: five-seed means; error bars show sample seed SD", fontsize=11
    )
    filename = "gas_temporal_macro_f1.png"
    fig.savefig(stage / filename, dpi=160)
    plt.close(fig)
    traces[filename] = sorted(set(plotted))
    diagnostics = read_json(analysis / "diagnostics.json")["runs"]
    selected = sorted(
        (
            r
            for r in diagnostics
            if r["method"] == "coral"
            and r["model"] == "lr"
            and r["seed"] == 17
            and r["setting"] == "primary"
        ),
        key=lambda r: r["task_id"],
    )
    before = np.array([r["coral"]["covariance_mismatch_before"] for r in selected])
    after = np.array([r["coral"]["covariance_mismatch_after"] for r in selected])
    if not np.isfinite([before, after]).all() or np.any(before <= 0) or np.any(after <= 0):
        raise DataError("Log rendering requires actual finite positive saved mismatch values")
    fig, axis = plt.subplots(figsize=(14, 5.5), layout="constrained")
    x = np.arange(len(selected))
    axis.vlines(x, after, before, color="#94a3b8", linewidth=1)
    axis.scatter(x, before, color="#475569", label="Before CORAL", s=24)
    axis.scatter(x, after, color="#b45309", label="After CORAL", s=24)
    axis.set(
        yscale="log",
        ylabel="Relative covariance mismatch (log scale)",
        xticks=x,
        xticklabels=[r["task_id"] for r in selected],
        title=(
            "Allowed adaptation features, seed 17; covariance agreement is not predictive benefit"
        ),
    )
    axis.tick_params(axis="x", labelrotation=70, labelsize=7)
    axis.legend()
    filename = "coral_covariance_mismatch.png"
    fig.savefig(stage / filename, dpi=160)
    plt.close(fig)
    traces[filename] = [r["run_id"] for r in selected]
    manifest = seal(
        {
            "study_id": study_id,
            "revision": "Publication layout only; shortened label, readable positive log residuals",
            "numerical_inputs_unchanged": True,
            "input_hashes": {
                name: file_sha256(analysis / name)
                for name in ("per_task_summary.csv", "diagnostics.json", "export_manifest.json")
            },
            "figure_traces": traces,
            "renderer_sha256": file_sha256(Path(__file__)),
            "output_hashes": {p.name: file_sha256(p) for p in stage.iterdir() if p.is_file()},
        }
    )
    write_json(stage / "visual_review_manifest.json", manifest)
    publish_directory(stage, destination)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = render(Path.cwd(), args.study_id, args.out)
    print(f"Rendered {len(result['figure_traces'])} reviewed scientific figures")
