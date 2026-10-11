"""Presentation of saved summary values; never estimate new scientific results."""

from io import BytesIO

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

COLORS = {"source_only": "#596574", "importance_weighting": "#2457A7", "coral": "#986515"}


def summary_figure(frame: pd.DataFrame, metric: str = "macro_f1"):
    rows = frame.reset_index(drop=True)
    figure, axis = plt.subplots(figsize=(7, max(3, len(rows) * 0.38)), layout="constrained")
    models = {"lr": "LR", "linear_svm": "Linear SVM", "rf": "RF", "majority": "Majority"}
    methods = {
        "source_only": "Source only",
        "importance_weighting": "Importance weights",
        "coral": "CORAL",
    }
    labels = [
        (f"{r.task_id} · " if rows.task_id.nunique() > 1 else "")
        + f"{models[r.model]} · {methods[r.method]}"
        + (f" · {r.setting}" if r.setting != "primary" else "")
        for r in rows.itertuples()
    ]
    positions = np.arange(len(rows))
    for position, row in enumerate(rows.to_dict("records")):
        axis.errorbar(
            row[f"{metric}_mean"],
            position,
            xerr=row[f"{metric}_std"],
            fmt="o",
            color=COLORS.get(row["method"], "#596574"),
            capsize=3,
        )
    axis.set(
        yticks=positions,
        yticklabels=labels,
        xlabel=(
            "Pipeline time (seconds)"
            if metric == "pipeline_seconds"
            else f"Target {metric.replace('_', ' ')} (fraction)"
        ),
    )
    axis.set_title("Saved seed means\nError bars: sample SD, not confidence intervals", fontsize=11)
    axis.tick_params(axis="y", labelsize=10)
    axis.invert_yaxis()
    if metric != "pipeline_seconds":
        axis.set_xlim(0, 1)
    return figure


def delta_figure(row: dict, metric: str = "macro_f1"):
    figure, axis = plt.subplots(figsize=(7, 2.5), layout="constrained")
    value = 100 * row[f"{metric}_delta_mean"]
    lower, upper = 100 * row[f"{metric}_lower"], 100 * row[f"{metric}_upper"]
    axis.hlines(0, lower, upper, color="#2457A7", linewidth=2)
    axis.scatter([value], [0], color="#2457A7", marker="o")
    axis.axvline(0, color="#596574", linestyle="--", linewidth=1)
    axis.set(
        yticks=[],
        xlabel="Change from matched source-only (percentage points)",
        title="Mean paired change; unadjusted conditional 95% group-bootstrap interval",
    )
    return figure


def png_bytes(figure) -> bytes:
    buffer = BytesIO()
    figure.savefig(buffer, format="png", dpi=160)
    plt.close(figure)
    return buffer.getvalue()
