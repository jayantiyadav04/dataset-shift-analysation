"""Real source baseline gate. Guard final target/global labels throughout fitting."""

import argparse
import builtins
import csv
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

from shiftlab.data.storage import read_json, write_json
from shiftlab.experiments.config import load_config
from shiftlab.experiments.provenance import canonical_json, code_snapshot, content_hash, file_sha256
from shiftlab.experiments.runner import train_baselines
from shiftlab.experiments.storage import LocalArtifactStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/baselines.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new evidence filename; previous records are immutable")
    root, data = Path.cwd(), Path("data").resolve()
    config = load_config(args.config)
    manifests = {
        p.parent.name: read_json(p)["manifest_hash"]
        for p in sorted((data / "splits/1.0").glob("*/manifest.json"))
    }
    prior = read_json(root / "docs/phase_02_final_verification.json")
    assert manifests == {
        row["task_id"]: row["manifest_hash"] for row in prior["split_audit"]["tasks"]
    }
    builtin_open, path_open = builtins.open, Path.open

    def guard(path) -> None:
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if resolved.is_relative_to(data) and {"evaluation", "private"} & set(resolved.parts):
                raise AssertionError(f"Fit accessed held-out/global label data: {resolved}")

    def guarded_builtin(path, *arguments, **kwargs):
        guard(path)
        return builtin_open(path, *arguments, **kwargs)

    def guarded_path(path, *arguments, **kwargs):
        guard(path)
        return path_open(path, *arguments, **kwargs)

    started = perf_counter()
    with patch("builtins.open", guarded_builtin), patch.object(Path, "open", guarded_path):
        result = train_baselines(
            config,
            root=root,
            data_root=data,
            progress=lambda event: print(canonical_json(event), flush=True),
        )
        store = LocalArtifactStore(root / config.execution.output_root)
        before = {
            p: (file_sha256(p), p.stat().st_mtime_ns)
            for row in result["runs"]
            for p in store.directory("runs", row["run_id"]).rglob("*")
            if p.is_file()
        }
        resumed = train_baselines(config, root=root, data_root=data)
        assert resumed["new_fits"] == 0 and resumed["resumed_fits"] == result["fitted_records"]
        assert before == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in before}
        counts = Counter((row["task_id"].split("_")[0], row["model"]) for row in result["runs"])
        assert all(
            counts[(dataset, model)] >= 1
            for dataset in ("gas", "har", "amazon")
            for model in ("lr", "linear_svm", "rf", "majority")
        )
        gas_cache = {
            row["source_cv_id"] for row in result["runs"] if row["task_id"].startswith("gas_")
        }
        assert len(gas_cache) == 1, "A second gas transfer retuned the same source"
        artifacts, tables = [], []
        for row in result["runs"]:
            record = store.verify("runs", row["run_id"])
            directory = store.directory("runs", row["run_id"])
            assert record["status"] == "fitted" and not record["evaluation_performed"]
            assert (
                "predictions.csv" not in record["checksums"]
                and "metrics.json" not in record["checksums"]
            )
            pre = store.load_joblib("runs", row["run_id"], "preprocessing.joblib")
            model = store.load_joblib("runs", row["run_id"], "fitted_model.joblib")
            assert pre.preprocessing_hash_ == record["canonical_key"]["preprocessing_hash"]
            assert hasattr(model, "classes_")
            reference = read_json(directory / "source_cv_reference.json")
            assert reference["identity"] == row["source_cv_id"]
            artifacts.append(
                {
                    **row,
                    "manifest_hash": record["manifest_hash"],
                    "selected_parameters": record["selected_params"],
                    "sample_counts": record["sample_counts"],
                    "timing": record["timing"],
                    "warnings": record["warnings"],
                    "path": directory.relative_to(root).as_posix(),
                }
            )
        for source_id in sorted({row["source_cv_id"] for row in result["runs"]}):
            record = store.verify("source_cache", source_id)
            directory = store.directory("source_cache", source_id)
            selection = read_json(directory / "selection.json")
            for fold in selection["folds"]:
                assert fold["train_ids"] == fold["preprocessing"]["fit_ids"]
                assert not set(fold["train_ids"]) & set(fold["validation_ids"])
                assert not set(fold["train_groups"]) & set(fold["validation_groups"])
            with (directory / "source_cv.csv").open(encoding="utf-8", newline="") as stream:
                tables.extend(
                    {
                        "source_cv_id": source_id,
                        "dataset": record["canonical_key"]["source_dataset"],
                        **row,
                    }
                    for row in csv.DictReader(stream)
                )
        report = root / "reports/source_baselines"
        report.mkdir(parents=True, exist_ok=True)
        with (report / "source_cv.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(tables[0]))
            writer.writeheader()
            writer.writerows(tables)
        evidence = {
            "evidence_kind": "real_source_fit_only",
            "observed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "command_config": args.config.as_posix(),
            "result": result,
            "resume": resumed,
            "artifacts": artifacts,
            "source_cv_rows": len(tables),
            "target_evaluation_opened": False,
            "all_prior_split_hashes_unchanged": True,
            "code_snapshot": code_snapshot(root),
            "code_snapshot_hash": content_hash(code_snapshot(root)),
        }
        write_json(report / "verification.json", evidence)
        write_json(args.out, evidence)
    print(
        canonical_json(
            {
                "status": "baseline_gate_passed",
                "fitted": result["fitted_records"],
                "evaluated": 0,
                "source_cv_rows": len(tables),
                "elapsed_seconds": evidence["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
