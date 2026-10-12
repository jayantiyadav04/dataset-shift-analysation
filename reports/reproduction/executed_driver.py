"""Fresh-fit one fixed real transfer in an isolated checkout; compare saved predictions."""

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from run_frozen_study import guard_heldout
from threadpoolctl import threadpool_limits

from shiftlab.data.storage import read_json, seal, verify_files, verify_seal, write_json
from shiftlab.evaluation.evaluator import evaluate_frozen_run
from shiftlab.experiments.config import load_config
from shiftlab.experiments.provenance import code_snapshot, file_sha256
from shiftlab.experiments.resources import resource_measurement
from shiftlab.experiments.runner import runtime_identity, train_methods
from shiftlab.experiments.storage import LocalArtifactStore, now

STUDY = "a05bea2f175b9d81091a413637276d174c378efaf31d1c539430e98055df77d0"
TASK = "gas_b01_b02"


def rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def reproduce(checkout: Path, reference: Path) -> dict:
    checkout, reference = checkout.resolve(), reference.resolve()
    if checkout == reference or not checkout.is_relative_to(reference / "artifacts"):
        raise ValueError("Use a separate checkout inside the local artifacts directory")
    if sys.prefix == sys.base_prefix or Path(sys.prefix).resolve() == reference / ".venv":
        raise ValueError("Run in a newly installed isolated virtual environment")
    output = checkout / "artifacts" / "reproduction"
    output.mkdir(parents=True, exist_ok=False)
    config = load_config(checkout / "configs/study.yaml")
    original = read_json(reference / "artifacts/studies" / STUDY / "study_lock.json")
    verify_seal(original)
    if code_snapshot(checkout) != original["code_snapshot"]:
        raise ValueError("Scientific source differs from the frozen study")
    if config.model_dump(mode="json") != original["configuration"]:
        raise ValueError("Published configuration differs from the original study")
    identity = runtime_identity(checkout)
    data = reference / "data"
    split_directory = data / "splits" / config.protocol_version / TASK
    split = read_json(split_directory / "manifest.json")
    verify_seal(split)
    assert split["manifest_hash"] == original["split_hashes"][TASK]
    acquisition = read_json(data / "raw/gas/main/receipt.json")
    verify_files(data / "raw/gas/main", {acquisition["filename"]: acquisition["sha256"]})
    assert acquisition["sha256"] == original["raw_archive_hashes"]["gas"]["main"]
    command = ["python", "tools/reproduce_task.py", "--checkout", checkout.name]
    request = seal(
        {
            "purpose": "fresh_environment_reproduction_only",
            "created_at": now(),
            "configuration": config.model_dump(mode="json"),
            "task_id": TASK,
            "seeds": list(config.seeds),
            "expected_fits": 50,
            "prior_study_id": STUDY,
            "split_hash": split["manifest_hash"],
            "config_hash": original["config_hash"],
            **identity,
            "python": platform.python_version(),
            "venv_isolated": True,
            "driver_sha256": file_sha256(Path(__file__)),
            "fit_guard_sha256": file_sha256(Path(__file__).with_name("run_frozen_study.py")),
            "command": command,
            "data_policy": "Existing archive and frozen split reused after hash validation",
        }
    )
    write_json(output / "request.json", request)
    started = perf_counter()

    def progress(event):
        print(json.dumps(event, separators=(",", ":")), flush=True)

    try:
        with guard_heldout(data) as forbidden, threadpool_limits(limits=1):
            fits = train_methods(
                config, root=checkout, data_root=data, tasks=(TASK,), progress=progress
            )
        assert fits["new_fits"] == 50 and fits["resumed_fits"] == 0
        assert not forbidden
        fit_seconds = perf_counter() - started
        write_json(output / "fit_boundary.json", seal({"forbidden_reads": forbidden, **fits}))
        store = LocalArtifactStore(checkout / config.execution.output_root)
        for run in fits["runs"]:
            store.verify("runs", run["run_id"])
        authorization = seal(
            {
                "status": "frozen_fits_complete",
                "purpose": "reproduction_only_50_fits_not_a_new_full_study",
                "request_hash": request["manifest_hash"],
                "run_ids": [r["run_id"] for r in fits["runs"]],
                "config_hash": original["config_hash"],
                **identity,
                "protocol_version": config.protocol_version,
                "split_hashes": {TASK: split["manifest_hash"]},
                "evaluation_ids": {TASK: split["target_eval_ids"]},
            }
        )
        write_json(output / "fit_completion_lock.json", authorization)
        references = {
            (r["model"], r["method"], int(r["seed"])): r
            for r in rows(reference / "reports/study/per_run_metrics.csv")
            if r["task_id"] == TASK and r["setting"] == "primary"
        }
        ref_store = LocalArtifactStore(reference / "artifacts")
        compared = []
        for run in fits["runs"]:
            evaluation, reused = evaluate_frozen_run(
                store, run["run_id"], split_directory, output / "fit_completion_lock.json"
            )
            assert not reused
            saved = references[(run["model"], run["method"], run["seed"])]
            ref_store.verify("evaluations", saved["evaluation_id"])
            current_dir = store.directory("evaluations", evaluation["identity"])
            previous_dir = ref_store.directory("evaluations", saved["evaluation_id"])
            fresh, prior = (
                rows(current_dir / "predictions.csv"),
                rows(previous_dir / "predictions.csv"),
            )
            columns = ("sample_id", "evaluation_group_id", "y_true", "y_pred")
            same = [[r[k] for k in columns] for r in fresh] == [
                [r[k] for k in columns] for r in prior
            ]
            metrics = read_json(current_dir / "metrics.json")
            old_metrics = read_json(previous_dir / "metrics.json")
            same_metrics = metrics == old_metrics
            current_fit = store.verify("runs", run["run_id"])
            old_fit = ref_store.verify("runs", saved["run_id"])
            same_parameters = current_fit["selected_params"] == old_fit["selected_params"]
            compared.append(
                {
                    "model": run["model"],
                    "method": run["method"],
                    "seed": run["seed"],
                    "fresh_run_id": run["run_id"],
                    "reference_run_id": saved["run_id"],
                    "fresh_evaluation_id": evaluation["identity"],
                    "reference_evaluation_id": saved["evaluation_id"],
                    "fresh_predictions_sha256": file_sha256(current_dir / "predictions.csv"),
                    "reference_predictions_sha256": file_sha256(previous_dir / "predictions.csv"),
                    "ordered_predictions_equal": same,
                    "all_metrics_equal": same_metrics,
                    "selected_parameters_equal": same_parameters,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "pipeline_seconds": evaluation["timing"]["pipeline_fit_predict_seconds"],
                }
            )
        matched = all(
            r[k]
            for r in compared
            for k in ("ordered_predictions_equal", "all_metrics_equal", "selected_parameters_equal")
        )
        versions = subprocess.run(
            [sys.executable, "-m", "pip", "freeze", "--all", "--exclude-editable"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        (output / "installed_dependencies.txt").write_text(versions, encoding="utf-8")
        result = seal(
            {
                "status": "passed" if matched else "mismatch_preserved",
                "request_hash": request["manifest_hash"],
                "source_commit": identity["code_version"],
                "source_snapshot_hash": identity["code_snapshot_hash"],
                "dependency_lock_hash": identity["dependency_lock_hash"],
                "task_id": TASK,
                "seeds": list(config.seeds),
                "fresh_fits": fits["new_fits"],
                "evaluations": len(compared),
                "evaluation_rows_per_run": len(split["target_eval_ids"]),
                "fit_seconds": fit_seconds,
                "total_seconds": perf_counter() - started,
                "resources": resource_measurement(),
                "comparisons": compared,
                "all_predictions_and_metrics_equal": matched,
                "forbidden_training_reads": forbidden,
                "limits": "Same hardware/OS and verified existing data; not cross-platform or a new independent dataset. Timings are remeasured, not expected identical.",
            }
        )
        write_json(output / "verification.json", result)
        print(json.dumps({k: v for k, v in result.items() if k != "comparisons"}), flush=True)
        if not matched:
            raise ValueError("Reproduction differs; preserved comparisons require review")
        return result
    except BaseException as error:
        write_json(
            output / "failure.json", seal({"type": type(error).__name__, "error": str(error)})
        )
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    reproduce(args.checkout, args.reference_root)
