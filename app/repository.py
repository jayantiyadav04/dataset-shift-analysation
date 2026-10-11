"""Hash-validated saved evidence without estimator deserialization or training reads."""

import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from shiftlab.data.splits import validate_split
from shiftlab.data.storage import DataError, read_json, safe_child, verify_files, verify_seal
from shiftlab.experiments.provenance import file_sha256

TABLES = {
    "per_task_summary",
    "paired_summary",
    "per_run_metrics",
    "paired_deltas",
    "bootstrap_intervals",
    "ablations",
    "dataset_summary",
    "descriptive_overall",
}


@dataclass(frozen=True)
class Context:
    study_id: str
    dataset: str
    task_id: str | None


class EvidenceRepository:
    """One read boundary shared by all six pages and exports.

    Verify each selected file before reading; no cached DataFrame survives a
    filter or file change. Incomplete studies never borrow another study's data.
    """

    def __init__(self, root: Path, artifact_path: str = "artifacts") -> None:
        self.root = root.resolve()
        self.artifacts = safe_child(self.root, artifact_path)

    def studies(self) -> tuple[list[dict], list[str]]:
        found, errors = [], []
        for path in sorted((self.artifacts / "studies").glob("*/study_lock.json")):
            try:
                lock = read_json(path)
                verify_seal(lock)
                if lock["study_id"] != path.parent.name:
                    raise DataError("Study directory and identity differ")
                found.append(lock)
            except (DataError, KeyError) as error:
                errors.append(f"{path.parent.name}: {error}")
        return found, errors

    def directory(self, study_id: str) -> Path:
        if not re.fullmatch(r"[0-9a-f]{64}", study_id):
            raise DataError("Invalid study identifier")
        return self.artifacts / "studies" / study_id

    def lock(self, study_id: str) -> dict:
        path = self.directory(study_id) / "study_lock.json"
        record = read_json(path)
        verify_seal(record)
        if record["study_id"] != study_id:
            raise DataError("Study identity mismatch")
        return record

    def progress(self, study_id: str) -> dict:
        path = self.directory(study_id) / "checkpoint.json"
        if not path.exists():
            return {
                "status": "frozen_planned",
                "evaluated_records": 0,
                "fitted_primary": 0,
                "fitted_sensitivity": 0,
                "failed_attempts": 0,
            }
        record = read_json(path)
        verify_seal(record)
        if record["study_id"] != study_id:
            raise DataError("Progress belongs to another study")
        return record

    def export(self, study_id: str) -> dict | None:
        path = self.directory(study_id) / "analysis/export_manifest.json"
        if not path.exists():
            return None
        record = read_json(path)
        verify_seal(record)
        lock = self.lock(study_id)
        if record["study_id"] != study_id or record["study_lock_hash"] != lock["manifest_hash"]:
            raise DataError("Analysis does not match the selected study lock")
        return record

    def saved_bytes(self, study_id: str, name: str) -> bytes:
        manifest = self.export(study_id)
        if manifest is None or name not in manifest["output_hashes"]:
            raise DataError("No verified analysis for this study or selected file")
        analysis = self.directory(study_id) / "analysis"
        verify_files(analysis, {name: manifest["output_hashes"][name]})
        return safe_child(analysis, name).read_bytes()

    def table(self, context: Context, name: str) -> pd.DataFrame:
        if name not in TABLES:
            raise DataError("Unsupported evidence table")
        frame = pd.read_csv(BytesIO(self.saved_bytes(context.study_id, f"{name}.csv")))
        if "dataset" in frame:
            frame = frame[frame.dataset == context.dataset]
        if context.task_id is not None and "task_id" in frame:
            frame = frame[frame.task_id == context.task_id]
        return frame.reset_index(drop=True)

    def split(self, study_id: str, task_id: str) -> tuple[Path, dict]:
        lock = self.lock(study_id)
        if task_id not in lock["split_hashes"]:
            raise DataError("Task is absent from this study")
        data = safe_child(self.root, lock["data_root"])
        path = data / "splits" / lock["protocol_version"] / task_id / "manifest.json"
        value = read_json(path)
        validate_split(value)
        if (
            value["manifest_hash"] != lock["split_hashes"][task_id]
            or value["dataset_hash"] != lock["dataset_hashes"][task_id]
        ):
            raise DataError("Split version/hash differs from the frozen study")
        return path, value

    def dataset(self, study_id: str, dataset: str) -> tuple[dict, list[dict]]:
        lock = self.lock(study_id)
        if dataset not in ("gas", "har", "amazon"):
            raise DataError("Unknown dataset")
        data = safe_child(self.root, lock["data_root"])
        manifest = read_json(data / "processed" / dataset / "manifest.json")
        verify_seal(manifest)
        tasks = [t for t in lock["dataset_hashes"] if t.startswith(dataset + "_")]
        if not tasks or any(lock["dataset_hashes"][t] != manifest["manifest_hash"] for t in tasks):
            raise DataError("Dataset manifest version differs from the study")
        archives = []
        for receipt in manifest["archives"]:
            path = data / "raw" / dataset / receipt["asset"] / receipt["filename"]
            archives.append(
                {
                    **receipt,
                    "path": path.relative_to(self.root).as_posix(),
                    "present": path.is_file(),
                }
            )
        return manifest, archives

    def diagnostics(self, context: Context) -> list[dict]:
        import json

        rows = json.loads(self.saved_bytes(context.study_id, "diagnostics.json"))["runs"]
        return [
            r
            for r in rows
            if r["dataset"] == context.dataset
            and (context.task_id is None or r["task_id"] == context.task_id)
        ]

    def run(self, study_id: str, run_id: str) -> tuple[dict, dict, dict]:
        export = self.export(study_id)
        if export is None or run_id not in export["input_hashes"]:
            raise DataError("Run is not an input to the selected analysis")
        trace = export["input_hashes"][run_id]
        fit_path = self.artifacts / "runs" / run_id / "manifest.json"
        if file_sha256(fit_path) != trace["fit_manifest_sha256"]:
            raise DataError("Fitted run manifest changed")
        fit = read_json(fit_path)
        verify_seal(fit)
        evaluation = self.artifacts / "evaluations" / trace["evaluation_id"]
        record = read_json(evaluation / "manifest.json")
        verify_seal(record)
        if record["run_id"] != run_id or record["status"] != "evaluated":
            raise DataError("Run has no matched completed evaluation")
        verify_files(evaluation, {"metrics.json": trace["metrics_sha256"]})
        return fit, record, read_json(evaluation / "metrics.json")

    def weights(self, study_id: str, run_id: str) -> np.ndarray:
        self.run(study_id, run_id)
        directory = self.artifacts / "runs" / run_id
        fit = read_json(directory / "manifest.json")
        if fit["canonical_key"]["method"] != "importance_weighting":
            raise DataError("Weights are not applicable to this method")
        verify_files(directory, {"adaptation_state.npz": fit["checksums"]["adaptation_state.npz"]})
        with np.load(directory / "adaptation_state.npz", allow_pickle=False) as state:
            return state["weights"].copy()

    def prediction_trace(self, study_id: str, run_ids: list[str]) -> list[dict[str, Any]]:
        export = self.export(study_id)
        if export is None:
            raise DataError("Predictions are unavailable for this study")
        traces = []
        for run_id in sorted(set(run_ids)):
            if run_id not in export["input_hashes"]:
                raise DataError("Selected run is outside this study")
            trace = export["input_hashes"][run_id]
            self.run(study_id, run_id)
            # Paths are relative to the original project, even in alternate UI storage.
            prediction = self.artifacts / "evaluations" / trace["evaluation_id"] / "predictions.csv"
            if file_sha256(prediction) != trace["predictions_sha256"]:
                raise DataError("Saved prediction checksum mismatch")
            traces.append({"run_id": run_id, **trace})
        return traces
