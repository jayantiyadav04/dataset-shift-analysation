"""Six-page research dashboard; rerenders only read saved evidence."""

import hashlib
import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.exports import build_bundle, json_bytes, run_ids, selected  # noqa: E402
from app.jobs import ACTIVE, JobManager, small_config  # noqa: E402
from app.plots import delta_figure, png_bytes, summary_figure  # noqa: E402
from app.repository import Context, EvidenceRepository  # noqa: E402
from shiftlab.data.storage import DataError  # noqa: E402
from shiftlab.experiments.config import ConfigError, load_config  # noqa: E402
from shiftlab.experiments.provenance import file_sha256  # noqa: E402
from shiftlab.scope import FINAL_OBJECTIVE  # noqa: E402

PAGES = ("Overview", "Datasets", "Experiment Runner", "Results", "Diagnostics", "Report Exports")
DATASETS = {
    "gas": "Gas Sensor Array Drift",
    "har": "Human Activity Recognition",
    "amazon": "Amazon Multi-Domain Sentiment",
}
MODELS = {
    "lr": "Logistic Regression",
    "linear_svm": "Linear SVM",
    "rf": "Random Forest",
    "majority": "Source majority",
}
METHODS = {
    "source_only": "Source only",
    "importance_weighting": "Importance weighting",
    "coral": "CORAL",
}
EMPTY = "No completed experiments yet. Prepare a dataset, then run the source-only baseline."


def filters(context: Context, frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    key = hashlib.sha256(json_bytes(context.__dict__)).hexdigest()[:12]
    cols = st.columns(3)
    models = cols[0].multiselect(
        "Models",
        sorted(frame.model.unique()),
        default=sorted(frame.model.unique()),
        format_func=MODELS.get,
        key=f"{prefix}-models-{key}",
    )
    methods = cols[1].multiselect(
        "Methods",
        sorted(frame.method.unique()),
        default=sorted(frame.method.unique()),
        format_func=METHODS.get,
        key=f"{prefix}-methods-{key}",
    )
    settings = cols[2].multiselect(
        "Settings",
        sorted(frame.setting.unique()),
        default=["primary"] if "primary" in set(frame.setting) else [],
        key=f"{prefix}-settings-{key}",
    )
    return selected(frame, models, methods, settings)


def show_table(frame: pd.DataFrame, columns: list[str] | None = None) -> None:
    visible = frame if columns is None else frame[[c for c in columns if c in frame]]
    config = {
        name: st.column_config.NumberColumn()
        for name in visible
        if pd.api.types.is_float_dtype(visible[name])
    }
    explanations = {
        "macro_f1_mean": (
            "Macro F1 mean",
            "Equal weight to full-target-present classes; "
            "mean across saved fitted seeds; zero_division=0.",
        ),
        "accuracy_mean": (
            "Accuracy mean",
            "Correct predictions / held-out target samples; mean across saved fitted seeds.",
        ),
        "macro_f1_delta_mean": (
            "Change in macro F1",
            "Adapted minus matched source-only, "
            "fraction units. Negative means observed negative transfer.",
        ),
        "pipeline_seconds_mean": (
            "Pipeline time (s)",
            "Saved preprocessing + adaptation + "
            "classifier fit + prediction. Shared source CV excluded.",
        ),
    }
    for name, (label, explanation) in explanations.items():
        if name in visible:
            config[name] = st.column_config.NumberColumn(label, help=explanation)
    presentation = visible.copy()
    # The grid's numeric null renderer overrides Styler's na_rep with "None".
    # Keep scientific frames numeric; explicitly render absence in this view only.
    for name in presentation:
        if presentation[name].isna().any():
            is_number = pd.api.types.is_numeric_dtype(presentation[name])
            presentation[name] = presentation[name].map(
                lambda value, numeric=is_number: (
                    "—" if pd.isna(value) else f"{value:.6f}" if numeric else str(value)
                )
            )
            column = config.get(name, {})
            config[name] = st.column_config.TextColumn(
                column.get("label", name), help=column.get("help")
            )
    st.dataframe(
        presentation.style.format(precision=6, na_rep="—"),
        hide_index=True,
        width="stretch",
        column_config=config,
    )


def figure_display(figure) -> None:
    st.pyplot(figure, width="stretch")
    plt.close(figure)


def empty_analysis(repo: EvidenceRepository, study_id: str) -> bool:
    if repo.export(study_id) is not None:
        return False
    progress = repo.progress(study_id)
    st.info(
        "No completed analysis for this study. "
        f"Saved evaluations: {progress['evaluated_records']}. "
        "Fit and evaluate the frozen study explicitly; existing artifacts are preserved."
    )
    st.caption("Incomplete pairs have no final mean, interval or substituted zero score.")
    return True


def overview(repo: EvidenceRepository, context: Context | None) -> None:
    st.subheader("Frozen research scope")
    st.write("Evaluating Domain Adaptation Techniques for Classical Machine Learning Models")
    with st.expander("Exact assessment objective"):
        st.write(FINAL_OBJECTIVE)
    st.write(
        "Three independent datasets. Logistic Regression, Linear SVM and Random Forest; "
        "source-only, importance weighting and CORAL, plus a source-majority baseline."
    )
    if context is None:
        st.info(EMPTY)
        return
    lock = repo.lock(context.study_id)
    progress = repo.progress(context.study_id)
    total = lock["expected_primary"] + lock["expected_sensitivity"]
    st.subheader("Saved study coverage")
    show_table(
        pd.DataFrame(
            [
                {
                    "Evidence": "Primary fits",
                    "Completed": progress["fitted_primary"],
                    "Expected": lock["expected_primary"],
                },
                {
                    "Evidence": "Sensitivity fits",
                    "Completed": progress["fitted_sensitivity"],
                    "Expected": lock["expected_sensitivity"],
                },
                {
                    "Evidence": "Evaluations",
                    "Completed": progress["evaluated_records"],
                    "Expected": total,
                },
            ]
        )
    )
    st.write(
        f"Status: **{progress['status']}** · recorded failed fitting attempts: "
        f"{progress['failed_attempts']}"
    )
    if progress["evaluated_records"] < total:
        st.warning("This study is incomplete. Expected and completed counts remain distinct.")
    else:
        st.success("Declared evaluation coverage is complete.")
    coverage = [
        {
            "Dataset": DATASETS[dataset],
            "Frozen transfers": sum(
                task.startswith(dataset + "_") for task in lock["split_hashes"]
            ),
        }
        for dataset in DATASETS
    ]
    show_table(pd.DataFrame(coverage))
    if not empty_analysis(repo, context.study_id):
        value = json.loads(repo.saved_bytes(context.study_id, "findings.json"))
        st.subheader("Observed findings")
        show_table(
            pd.DataFrame(
                [
                    {
                        "Primary comparisons": value["primary_comparisons"],
                        "Positive mean macro F1 changes": value[
                            "observed_positive_macro_f1_changes"
                        ],
                        "Negative mean changes": value["observed_negative_transfer"],
                        "Intervals including zero": value["intervals_including_zero"],
                    }
                ]
            )
        )
        with st.expander("Saved limitations and interpretation"):
            for limitation in value["limitations"]:
                st.write(limitation)
        st.caption(
            "Comparisons share test rows and transforms. Counts describe observed changes; "
            "they are not independent statistical trials."
        )
    with st.expander("Study identity and reproducibility"):
        st.code(context.study_id, language=None)
        st.json(
            {
                key: lock[key]
                for key in (
                    "protocol_version",
                    "config_hash",
                    "code_version",
                    "code_snapshot_hash",
                    "dependency_lock_hash",
                )
            }
        )
    st.caption(
        "Five seeds share each held-out set. Group-bootstrap intervals are conditional "
        "on fitted models and unadjusted. HAR has five evaluation participants."
    )


def datasets(repo: EvidenceRepository, context: Context | None) -> None:
    if context is None:
        st.info("No frozen study selected. Prepare the official datasets and freeze their splits.")
        st.code(
            "shiftlab data fetch --dataset gas\nshiftlab data prepare --all\n"
            "shiftlab splits freeze --config configs/study.yaml",
            language=None,
        )
        return
    manifest, archives = repo.dataset(context.study_id, context.dataset)
    st.subheader(DATASETS[context.dataset])
    show_table(
        pd.DataFrame(
            [
                {
                    "Retained samples": manifest["retained_sample_count"],
                    "Excluded samples": manifest["excluded_sample_count"],
                    "Features": manifest["feature_count"],
                    "Feature kind": manifest["feature_kind"],
                }
            ]
        )
    )
    st.write("Group policy: " + manifest["group_policy"])
    st.caption("Target label profiles are not opened here; counts come from saved manifests.")
    st.subheader("Archive provenance")
    show_table(pd.DataFrame(archives), ["asset", "filename", "bytes", "accessed_at", "present"])
    for archive in archives:
        with st.expander(f"{archive['asset']} archive · source, checksum and local path"):
            st.write(archive["source_url"])
            st.code(f"{archive['path']}\nSHA-256: {archive['sha256']}", language=None)
        if not archive["present"]:
            st.warning("Dataset archive is missing. Add the official archive to " + archive["path"])
    if st.button("Verify present archive checksums"):
        for archive in archives:
            if not archive["present"]:
                st.error(f"Missing archive: {archive['path']}")
            elif file_sha256(ROOT / archive["path"]) == archive["sha256"]:
                st.success(f"Checksum verified: {archive['filename']}")
            else:
                st.error(f"Archive checksum mismatch: {archive['filename']}")
    with st.expander("Exclusions, source classes and parser notes"):
        st.json(
            {
                key: manifest[key]
                for key in (
                    "exclusion_counts",
                    "class_mapping",
                    "source_label_counts",
                    "parser_notes",
                    "excluded_records_path",
                    "terms",
                )
            }
        )
    lock = repo.lock(context.study_id)
    tasks = (
        [context.task_id]
        if context.task_id
        else [task for task in lock["split_hashes"] if task.startswith(context.dataset + "_")]
    )
    rows = []
    for task in tasks:
        path, split = repo.split(context.study_id, task)
        rows.append(
            {
                "Task": task,
                "Source": split["source_domain"],
                "Target": split["target_domain"],
                "Source samples": len(split["source_ids"]),
                "Adaptation samples": len(split["target_adapt_ids"]),
                "Evaluation samples": len(split["target_eval_ids"]),
                "Source groups": len(set(split["source_group_ids"])),
                "Adaptation groups": len(set(split["target_adapt_group_ids"])),
                "Evaluation groups": len(set(split["target_eval_group_ids"])),
            }
        )
        if context.task_id:
            st.subheader("Frozen data roles")
            st.write(
                "**Source features + source labels** → source preprocessing, "
                "source validation and classifier fitting."
            )
            st.write("**Target adaptation features** → importance weights or covariance only.")
            st.write("**Held-out features + labels** → evaluator after all fitting is frozen.")
            st.write("Split algorithm: " + split["split_algorithm"])
            st.write("Duplicate policy: " + split["deduplication_policy"])
            st.code(f"{path.relative_to(ROOT).as_posix()}\n{split['manifest_hash']}", language=None)
            st.download_button(
                "Download exact split IDs and groups",
                json_bytes(split),
                file_name=f"{task}_split.json",
                mime="application/json",
            )
    show_table(pd.DataFrame(rows))


@st.fragment(run_every=2.0)
def job_status(manager: JobManager) -> None:
    current = manager.current()
    if current is None:
        st.info("No job has been launched by this workbench.")
        return
    st.write(f"Job `{current['job_id']}` · **{current['status']}** · {current['action']}")
    if current.get("error"):
        st.warning(current["error"])
    if current["status"] in ACTIVE and current["owned"]:
        if st.button("Cancel owned job", key="cancel-job"):
            manager.cancel(current["job_id"])
            st.info(
                "Cancellation requested. The worker stops at an operation boundary; "
                "saved checkpoints remain available."
            )
    st.code(manager.log(current["job_id"]), language=None, height=280)
    if current.get("result"):
        with st.expander("Saved job result"):
            st.json(current["result"])


def runner(repo: EvidenceRepository, context: Context | None) -> None:
    manager = JobManager(ROOT)
    primary = load_config(ROOT / "configs/study.yaml")
    st.write(
        "Default: one fit-only development job. It uses the declared source-validation "
        "grid and adaptation settings and does not evaluate target labels."
    )
    tasks = primary.available_tasks()
    with st.form("small-job"):
        columns = st.columns(2)
        task = columns[0].selectbox(
            "Job transfer",
            tasks,
            index=tasks.index(context.task_id) if context and context.task_id in tasks else 0,
        )
        model = columns[1].selectbox(
            "Job model", ("lr", "linear_svm", "rf"), format_func=MODELS.get
        )
        method = columns[0].selectbox("Job method", tuple(METHODS), format_func=METHODS.get)
        seed = columns[1].selectbox("Job seed", primary.seeds)
        submitted = st.form_submit_button("Start one fit-only job", type="primary")
    config = small_config(ROOT, task, model, method, seed)
    with st.expander("Validated job configuration"):
        st.json(config.model_dump(mode="json"))
    if repo.artifacts != ROOT / "artifacts":
        st.warning("This alternate storage view is read-only. Select artifacts to launch jobs.")
    elif submitted:
        try:
            split_path = ROOT / "data/splits" / primary.protocol_version / task / "manifest.json"
            if not split_path.exists():
                raise DataError(
                    f"Split missing: {split_path.relative_to(ROOT)}. Prepare data first."
                )
            manager.start(action="fit", task=task, model=model, method=method, seed=seed)
            st.success("One worker launched. Rerenders read its status and do not launch again.")
        except (DataError, OSError, ValueError) as error:
            st.error(f"Job cannot start: {error}")
    with st.expander("Explicit full frozen study / resume"):
        st.write(
            "1,100 primary fits and 180 sensitivity fits. Evaluation follows the complete "
            "fitting gate. Existing exact-key artifacts are reused; settings remain frozen."
        )
        confirmed = st.checkbox("I intend to run or resume all 1,280 declared records")
        if confirmed and repo.artifacts == ROOT / "artifacts":
            if st.button("Run frozen study"):
                try:
                    manager.start(action="study")
                    st.success("Frozen study worker launched explicitly.")
                except (DataError, OSError, ValueError) as error:
                    st.error(f"Study cannot start: {error}")
        st.code("python tools/run_frozen_study.py execute", language=None)
    st.subheader("Worker status and log")
    job_status(manager)
    history = manager.history()
    if history:
        with st.expander("Job history, including failures and cancellations"):
            show_table(pd.DataFrame(history))
    st.caption(
        "Closing the browser does not cancel a worker. CLI recovery preserves abandoned "
        "attempts after a hard interruption; this app never cancels an unowned PID."
    )


def results(repo: EvidenceRepository, context: Context | None) -> None:
    if context is None:
        st.info(EMPTY)
        return
    if empty_analysis(repo, context.study_id):
        return
    frame = filters(context, repo.table(context, "per_task_summary"), "results")
    if frame.empty:
        st.info("No completed runs match these filters.")
        return
    st.caption(
        "Sortable saved comparison table. Scores and paired changes here are fractions. "
        "Detailed change plots use percentage points. Missing intervals are not zero."
    )
    show_table(
        frame,
        [
            "model",
            "method",
            "macro_f1_mean",
            "macro_f1_delta_mean",
            "accuracy_mean",
            "pipeline_seconds_mean",
            "seed_count",
            "sample_count",
            "task_id",
            "setting",
            "macro_f1_std",
            "accuracy_std",
            "macro_f1_lower",
            "macro_f1_upper",
        ],
    )
    with st.expander("Metric definitions, denominators and uncertainty"):
        st.write(
            "Accuracy is correct predictions divided by held-out samples. Macro F1 gives "
            "equal weight to classes present in the full target test, using zero_division=0. "
            "Means and sample SD use five fitted seeds. Seeds share test rows. "
            "Conditional unadjusted 95% intervals use 1,000 whole-group bootstrap draws "
            "and average each draw's paired changes over the fitted seeds."
        )
        st.write(
            "Pipeline time includes preprocessing, adaptation, classifier fit and target "
            "prediction. Shared source tuning is separate; cache hits retain originating "
            "component costs. Timings are not additive study wall times."
        )
    metric = st.selectbox(
        "Plot metric",
        ("macro_f1", "accuracy", "pipeline_seconds"),
        format_func=lambda x: {
            "macro_f1": "Macro F1",
            "accuracy": "Accuracy",
            "pipeline_seconds": "Pipeline time",
        }[x],
    )
    if len(frame) <= 30:
        figure_display(summary_figure(frame, metric))
    else:
        st.info("Select one transfer or fewer settings to see a readable plot.")
    choices = [f"{r.task_id} / {r.model} / {r.method} / {r.setting}" for r in frame.itertuples()]
    key = hashlib.sha256("|".join(choices).encode()).hexdigest()[:12]
    choice = st.selectbox("Inspect comparison", choices, key=f"comparison-{key}")
    row = frame.iloc[choices.index(choice)].to_dict()
    ids = run_ids(pd.DataFrame([row]))
    if row["seed_count"] != 5:
        st.warning(f"{int(row['seed_count'])} of 5 seeds completed. This comparison is incomplete.")
    elif row["method"] != "source_only" and row["model"] != "majority":
        pairs = repo.table(context, "paired_summary")
        pair = pairs[
            (pairs.task_id == row["task_id"])
            & (pairs.model == row["model"])
            & (pairs.method == row["method"])
            & (pairs.setting == row["setting"])
        ]
        if pair.empty or pair.iloc[0].paired_seeds != 5:
            st.warning("Matched source-only comparison incomplete; no interval is shown.")
        else:
            paired = pair.iloc[0].to_dict()
            change = 100 * paired["macro_f1_delta_mean"]
            st.write(f"Change from source-only: **{change:+.6f} percentage points** in macro F1.")
            figure_display(delta_figure(paired))
            st.caption(
                "The dashed line is zero change. Crossing zero expresses uncertainty, "
                "not proven equivalence. Negative changes remain visible."
            )
    with st.expander("Exact baseline and adapted run IDs, saved prediction files"):
        traces = repo.prediction_trace(context.study_id, ids)
        st.caption("Detailed trace table: scroll horizontally for full IDs, paths and checksums.")
        show_table(
            pd.DataFrame(traces),
            ["run_id", "evaluation_id", "prediction_path", "predictions_sha256"],
        )
        _, split = repo.split(context.study_id, row["task_id"])
        st.code(
            f"data/splits/{split['protocol_version']}/{row['task_id']}/manifest.json\n"
            f"Split SHA-256 identity: {split['manifest_hash']}",
            language=None,
        )
        clean = {k: None if pd.isna(v) else v for k, v in row.items()}
        st.download_button(
            "Download prediction trace JSON",
            json_bytes(traces),
            file_name="prediction_trace.json",
            mime="application/json",
        )
        st.download_button(
            "Download selected comparison values",
            json_bytes(clean),
            file_name="selected_comparison.json",
            mime="application/json",
        )


def diagnostics(repo: EvidenceRepository, context: Context | None) -> None:
    if context is None:
        st.info("No saved diagnostics yet. Fit a declared method explicitly.")
        return
    if empty_analysis(repo, context.study_id):
        return
    rows = repo.diagnostics(context)
    if not rows:
        st.info("No diagnostic records match this context.")
        return
    keys = [
        f"{r['task_id']} / {r['model']} / {r['method']} / {r['setting']} / seed {r['seed']}"
        for r in rows
    ]
    selection_key = hashlib.sha256("|".join(keys).encode()).hexdigest()[:12]
    choice = st.selectbox("Diagnostic run", keys, key=f"diagnostic-{selection_key}")
    row = rows[keys.index(choice)]
    fit, evaluation, metrics = repo.run(context.study_id, row["run_id"])
    st.code(row["run_id"], language=None)
    st.caption("Saved evaluated run; the immutable fit manifest retains its fitted-stage status.")
    for warning in fit["warnings"]:
        st.warning(warning)
    if row["method"] == "importance_weighting":
        value = row["importance"]
        st.subheader("Source importance weights")
        show_table(
            pd.DataFrame(
                [
                    {
                        k: value[k]
                        for k in (
                            "source_count",
                            "target_adapt_count",
                            "ess",
                            "ess_fraction",
                            "min",
                            "max",
                            "weight_clipping_fraction",
                        )
                    }
                ]
            )
        )
        weights = repo.weights(context.study_id, row["run_id"])
        figure, axis = plt.subplots(figsize=(7, 3), layout="constrained")
        axis.hist(weights, bins=30, color="#2457A7", edgecolor="white")
        axis.set(xlabel="Mean-one normalized source weight", ylabel="Source samples")
        figure_display(figure)
        st.write(
            "ESS measures weight concentration, not prediction quality. Raw clipping "
            "precedes global mean-one normalization, so normalized bounds can differ."
        )
    elif row["method"] == "coral":
        st.subheader("Saved covariance alignment")
        st.json(row["coral"])
        st.write(
            "Source features are aligned; target features use source preprocessing "
            "without applying the source alignment matrix."
        )
        pairs = repo.table(context, "paired_summary")
        matching = pairs[
            (pairs.task_id == row["task_id"])
            & (pairs.model == row["model"])
            & (pairs.method == "coral")
            & (pairs.setting == row["setting"])
        ]
        if (
            row["coral"]["covariance_mismatch_after"] < row["coral"]["covariance_mismatch_before"]
            and not matching.empty
            and matching.iloc[0].macro_f1_delta_mean < 0
        ):
            st.warning(
                "CORAL reduced covariance mismatch, but mean target macro F1 did not improve."
            )
    else:
        st.info("Weight and covariance diagnostics are not applicable to source-only training.")
    st.subheader("Held-out class errors from saved predictions")
    labels = [fit["class_mapping"][str(c)] for c in metrics["confusion_matrix_classes"]]
    matrix = np.asarray(metrics["confusion_matrix"])
    figure, axis = plt.subplots(figsize=(6, 4), layout="constrained")
    visual = axis.imshow(matrix, cmap="Blues")
    axis.set(
        xticks=range(len(labels)),
        yticks=range(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        xlabel="Predicted class",
        ylabel="True class",
    )
    axis.tick_params(axis="x", labelrotation=45)
    for (i, j), count in np.ndenumerate(matrix):
        axis.text(
            j,
            i,
            str(count),
            ha="center",
            va="center",
            color="white" if count > matrix.max() / 2 else "#17212B",
        )
    figure.colorbar(visual, ax=axis, label="Held-out samples")
    figure_display(figure)
    per_class = metrics["per_class"]
    show_table(
        pd.DataFrame.from_dict(per_class, orient="index").reset_index()
        if isinstance(per_class, dict)
        else pd.DataFrame(per_class)
    )
    st.caption(
        f"Denominator: {evaluation['sample_count']} target samples; one fitted seed. "
        "Macro classes are fixed to those present in the full held-out task."
    )
    st.subheader("Predeclared sensitivity")
    ablations = repo.table(context, "ablations")
    ablations = ablations[
        (ablations.task_id == row["task_id"])
        & (ablations.model == row["model"])
        & (ablations.method == row["method"])
    ]
    if ablations.empty:
        st.info(
            "Sensitivity was not predeclared for this transfer/method. No new setting is tuned."
        )
    else:
        show_table(
            ablations,
            [
                "setting",
                "sensitivity_value",
                "seed",
                "macro_f1",
                "accuracy",
                "pipeline_seconds",
                "run_id",
            ],
        )
        means = ablations.groupby("sensitivity_value").macro_f1.agg(["mean", "std"])
        figure, axis = plt.subplots(figsize=(6, 3), layout="constrained")
        axis.errorbar(
            means.index.astype(str),
            means["mean"],
            yerr=means["std"],
            fmt="o-",
            color="#2457A7",
            capsize=3,
        )
        axis.set(
            xlabel="Raw clipping upper bound"
            if row["method"] == "importance_weighting"
            else "CORAL ridge",
            ylabel="Target macro F1 (fraction)",
            ylim=(0, 1),
        )
        figure_display(figure)
        st.caption(
            "Five-seed means and sample SD. Primary settings remain clipping 10 / ridge 0.001."
        )


def exports(repo: EvidenceRepository, context: Context | None) -> None:
    if context is None:
        st.info("No frozen study available for export.")
        return
    if empty_analysis(repo, context.study_id):
        return
    frame = filters(context, repo.table(context, "per_task_summary"), "export")
    if frame.empty:
        st.info("No completed runs match these export filters.")
        return
    st.write(
        "Exports contain filtered CSV/JSON values, exact split IDs, methodology, "
        "prediction checksums and figures. Building an export reads saved evidence only."
    )
    include = st.checkbox("Include scientific figures", value=True)
    selection = {
        "context": context.__dict__,
        "models": sorted(frame.model.unique()),
        "methods": sorted(frame.method.unique()),
        "settings": sorted(frame.setting.unique()),
        "figures": include,
        "export_hash": repo.export(context.study_id)["manifest_hash"],
    }
    key = hashlib.sha256(json_bytes(selection)).hexdigest()
    if st.button("Build verified bundle", type="primary"):
        with st.spinner("Verifying selected evidence and preparing downloads"):
            try:
                bundle = build_bundle(
                    repo,
                    context,
                    selection["models"],
                    selection["methods"],
                    selection["settings"],
                    include,
                )
                st.session_state["report_bundle"] = (key, bundle)
                st.success("Export ready. Input and output hashes are included.")
            except (DataError, OSError, ValueError) as error:
                st.session_state.pop("report_bundle", None)
                st.error(f"Export failed: {error}")
    bundle = st.session_state.get("report_bundle")
    if bundle and bundle[0] == key:
        st.download_button(
            "Download verified report bundle",
            bundle[1],
            file_name=f"shiftlab_{context.study_id[:12]}_{context.dataset}.zip",
            mime="application/zip",
        )
    elif bundle:
        st.info("Selection changed. Build a new bundle for the current filters.")
    show_table(frame, ["task_id", "model", "method", "setting", "seed_count", "macro_f1_mean"])
    st.download_button(
        "Download current summary CSV",
        frame.to_csv(index=False).encode("utf-8"),
        file_name="selected_summary.csv",
        mime="text/csv",
    )
    st.download_button(
        "Download current summary JSON",
        json_bytes(frame.astype(object).where(pd.notna(frame), None).to_dict("records")),
        file_name="selected_summary.json",
        mime="application/json",
    )
    if len(frame) <= 30 and include:
        st.download_button(
            "Download current macro F1 figure",
            png_bytes(summary_figure(frame)),
            file_name="selected_macro_f1.png",
            mime="image/png",
        )


def main() -> None:
    st.set_page_config(page_title="Domain Shift Workbench", page_icon="🔬", layout="wide")
    st.markdown(
        """<style>
        .stApp {font-family: system-ui, sans-serif;}
        h1 {font-size: 1.75rem !important;} h2 {font-size: 1.3rem !important;}
        h3 {font-size: 1.1rem !important;}
        .block-container {padding-top: 2rem; padding-bottom: 2rem;}
        button:focus-visible, input:focus-visible {outline: 3px solid #2457A7 !important;}
        </style>""",
        unsafe_allow_html=True,
    )
    st.sidebar.title("Domain Shift Workbench")
    page = st.sidebar.radio("Page", PAGES)
    storage = st.sidebar.text_input(
        "Evidence storage (relative path)",
        value=os.environ.get("SHIFTLAB_ARTIFACT_ROOT", "artifacts"),
    )
    try:
        repo = EvidenceRepository(ROOT, storage)
        studies, errors = repo.studies()
        for error in errors:
            st.sidebar.error("Invalid study: " + error)
        context = None
        if studies:
            by_id = {value["study_id"]: value for value in studies}
            study_id = st.sidebar.selectbox(
                "Frozen study", list(by_id), format_func=lambda x: x[:12]
            )
            lock = by_id[study_id]
            dataset = st.sidebar.selectbox("Dataset", tuple(DATASETS), format_func=DATASETS.get)
            tasks = sorted(task for task in lock["split_hashes"] if task.startswith(dataset + "_"))
            task = st.sidebar.selectbox(
                "Transfer",
                ["All transfers", *tasks],
                index=1,
                key=f"context-task-{study_id}-{dataset}",
            )
            context = Context(study_id, dataset, None if task == "All transfers" else task)
            st.sidebar.caption(f"Protocol {lock['protocol_version']} · five seeds")
        st.title(page)
        if context:
            task_label = context.task_id or "All transfers"
            st.caption(
                f"{DATASETS[context.dataset]} · {task_label} · "
                f"protocol {lock['protocol_version']} · study {context.study_id[:12]}"
            )
            if context.task_id:
                st.caption("Frozen split: " + lock["split_hashes"][context.task_id])
        else:
            st.caption("No frozen study in the selected evidence storage.")
        if st.sidebar.button("Refresh saved evidence"):
            st.rerun()
        handlers = dict(
            zip(PAGES, (overview, datasets, runner, results, diagnostics, exports), strict=True)
        )
        handlers[page](repo, context)
    except (DataError, ConfigError, OSError, KeyError, ValueError) as error:
        st.error(f"Saved evidence or configuration cannot be validated: {error}")
        st.info(
            "Inspect the named file, restore its verified version or prepare the missing "
            "dataset. Existing studies are preserved; no replacement training is automatic."
        )


main()
