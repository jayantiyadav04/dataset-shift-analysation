"""Explicit exports from verified saved values; no fitting or evaluation imports."""

import hashlib
import json
from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

from app.plots import png_bytes, summary_figure
from app.repository import Context, EvidenceRepository
from shiftlab.data.storage import DataError, seal
from shiftlab.experiments.provenance import file_sha256


def selected(
    frame: pd.DataFrame, models: list[str], methods: list[str], settings: list[str]
) -> pd.DataFrame:
    for field, values in (("model", models), ("method", methods), ("setting", settings)):
        if field in frame:
            frame = frame[frame[field].isin(values)]
    return frame.reset_index(drop=True)


def run_ids(frame: pd.DataFrame) -> list[str]:
    identities = set()
    for name in ("run_id", "run_ids", "source_run_ids", "adapted_run_ids"):
        if name in frame:
            for value in frame[name].dropna():
                identities.update(str(value).split("|"))
    return sorted(identities)


def json_bytes(value) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")


def build_bundle(
    repo: EvidenceRepository,
    context: Context,
    models: list[str],
    methods: list[str],
    settings: list[str],
    figures: bool = True,
) -> bytes:
    export = repo.export(context.study_id)
    if export is None or repo.progress(context.study_id)["status"] != "complete":
        raise DataError("Complete verified analysis is required for a report bundle")
    summaries = selected(repo.table(context, "per_task_summary"), models, methods, settings)
    if summaries.empty:
        raise DataError("No completed runs match these export filters")
    if (summaries.seed_count != 5).any():
        raise DataError("Comparison is incomplete; five saved seeds are required")
    payloads, tables = {}, {}
    for name in (
        "per_task_summary",
        "per_run_metrics",
        "paired_summary",
        "paired_deltas",
        "bootstrap_intervals",
        "ablations",
    ):
        frame = selected(repo.table(context, name), models, methods, settings)
        tables[name] = frame
        payloads[f"{name}.csv"] = frame.to_csv(index=False, lineterminator="\n").encode("utf-8")
        records = frame.astype(object).where(pd.notna(frame), None).to_dict("records")
        payloads[f"{name}.json"] = json_bytes(records)
    identities = sorted(set().union(*(set(run_ids(frame)) for frame in tables.values())))
    traces = repo.prediction_trace(context.study_id, identities)
    lock = repo.lock(context.study_id)
    tasks = sorted(set(summaries.task_id))
    for task in tasks:
        _, split = repo.split(context.study_id, task)
        payloads[f"splits/{task}.json"] = json_bytes(split)
    payloads["methodology.json"] = json_bytes(
        {
            "objective": lock["objective"],
            "configuration": lock["configuration"],
            "config_hash": lock["config_hash"],
            "protocol_version": lock["protocol_version"],
            "raw_archive_hashes": lock["raw_archive_hashes"],
            "engine_snapshot_hash": lock["code_snapshot_hash"],
            "source_base_commit": lock["code_version"],
            "dependency_lock_hash": lock["dependency_lock_hash"],
            "uncertainty": "1000 whole-group replicates, conditional on five fitted models; "
            "unadjusted intervals; seeds share target rows",
            "timing": "Pipeline time excludes shared source tuning; "
            "originating cached costs retained",
            "scope": "Filtered saved evidence; no raw features, review bodies or serialized models",
        }
    )
    figure_traces = {}
    if figures:
        payloads["figures/selected_summary.png"] = png_bytes(summary_figure(summaries))
        figure_traces["figures/selected_summary.png"] = run_ids(summaries)
        for name, ids in export["figure_run_ids"].items():
            if ids and set(ids).issubset(identities):
                path = f"figures/{name}"
                payloads[path] = repo.saved_bytes(context.study_id, path)
                figure_traces[path] = ids
    manifest = seal(
        {
            "schema_version": 1,
            "study_id": context.study_id,
            "selection": {
                "dataset": context.dataset,
                "task_id": context.task_id,
                "models": models,
                "methods": methods,
                "settings": settings,
            },
            "study_lock_hash": lock["manifest_hash"],
            "original_export_manifest_sha256": file_sha256(
                repo.directory(context.study_id) / "analysis/export_manifest.json"
            ),
            "command": "Report Exports / Build verified bundle (saved evidence only)",
            "run_ids": identities,
            "prediction_inputs": traces,
            "figure_run_ids": figure_traces,
            "input_table_hashes": {name: export["output_hashes"][f"{name}.csv"] for name in tables},
            "output_hashes": {
                name: hashlib.sha256(value).hexdigest() for name, value in payloads.items()
            },
        }
    )
    payloads["export_manifest.json"] = json_bytes(manifest)
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for name, value in payloads.items():
            archive.writestr(name, value)
    return buffer.getvalue()
