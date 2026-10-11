"""Explicit benchmark/study jobs with observed fit-boundary and progress receipts."""

import argparse
import builtins
import io
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter, sleep
from unittest.mock import patch

from shiftlab.data.storage import read_json, seal, write_json
from shiftlab.evaluation.analysis import summarize_study
from shiftlab.experiments.checkpoints import Journal, job_owner
from shiftlab.experiments.config import load_config
from shiftlab.experiments.provenance import canonical_json, file_sha256
from shiftlab.experiments.study import benchmark, checkpoint, evaluate_study, fit_study, initialize


@contextmanager
def retry_progress_io(directory: Path):
    """Retry the same atomic replacement only for mutable study progress files."""
    original = os.replace
    directory = directory.resolve()
    mutable = {"checkpoint.json", "completed_runs.csv", "missing_runs.csv", "active_job.json"}
    retries = []

    def replace(source, target, *args, **kwargs):
        path = Path(target).resolve() if isinstance(target, (str, Path)) else None
        for attempt in range(40):
            try:
                return original(source, target, *args, **kwargs)
            except PermissionError as error:
                if (
                    path is None
                    or path.parent != directory
                    or path.name not in mutable
                    or getattr(error, "winerror", None) not in (5, 32, 33)
                    or attempt == 39
                ):
                    raise
                retries.append(
                    {"file": path.name, "attempt": attempt + 1, "winerror": error.winerror}
                )
                sleep(0.2)

    try:
        with patch.object(os, "replace", replace):
            yield retries
    finally:
        # Operational receipts are separate from immutable study/analysis evidence.
        write_json(
            directory.parents[1] / "operations" / f"progress-io-{uuid.uuid4().hex}.json",
            seal(
                {
                    "study_id": directory.name,
                    "driver_sha256": file_sha256(Path(__file__)),
                    "policy": "Same atomic replacement; four mutable progress files only; "
                    "Windows errors 5/32/33; at most 40 attempts, 0.2s between attempts",
                    "retries": retries,
                }
            ),
        )


@contextmanager
def guard_heldout(data_root):
    """Block all ordinary Python file readers to held-out/private stores during fits."""
    original = io.open
    data_root = data_root.resolve()
    attempts = []

    def guarded(file, *args, **kwargs):
        if isinstance(file, (str, Path)):
            path = Path(file).resolve()
            if path.is_relative_to(data_root) and {"evaluation", "private"} & set(path.parts):
                attempts.append(path.as_posix())
                raise AssertionError("Training attempted to open forbidden held-out/private data")
        return original(file, *args, **kwargs)

    with patch.object(io, "open", guarded), patch.object(builtins, "open", guarded):
        yield attempts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("benchmark", "execute"))
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--task")
    args = parser.parse_args()
    root, data_root = Path.cwd(), Path("data")
    config = load_config(args.config)
    directory, lock = initialize(config, root, data_root)
    with retry_progress_io(directory):
        return run_job(root, data_root, config, directory, lock, args)


def run_job(root, data_root, config, directory, lock, args):
    started = perf_counter()

    def progress(event):
        print(canonical_json(event), flush=True)

    with guard_heldout(data_root) as attempts:
        if args.action == "benchmark":
            result = benchmark(config, root, data_root, args.task, progress)
        else:
            result = fit_study(config, root, data_root, progress)
    receipt = {
        "observed_at_stage": "fitting_complete",
        "forbidden_open_attempts": attempts,
        "elapsed_seconds": perf_counter() - started,
        "result": result,
    }
    destination = directory / (
        f"benchmark_boundary_{args.task}.json"
        if args.action == "benchmark"
        else "fit_boundary_receipt.json"
    )
    if not destination.exists():
        write_json(destination, seal(receipt))
    if args.action == "execute":
        if result["status"] != "frozen_fits_complete":
            print(json.dumps(result, indent=2), flush=True)
            return 3
        result = evaluate_study(root, lock["study_id"], progress)
        result = summarize_study(root, lock["study_id"], progress)
        if result["status"] == "analysis_verified":
            # Earlier stages refresh progress on resume. Cached summaries verify
            # immutable exports but do not otherwise refresh the progress state.
            with job_owner(directory):
                checkpoint(directory, lock, Journal(directory), "complete")
        completion = directory / "execution_receipt.json"
        if not completion.exists():
            write_json(
                completion,
                seal(
                    {
                        "elapsed_seconds_this_invocation": perf_counter() - started,
                        "result": result,
                        "checkpoint": read_json(directory / "checkpoint.json"),
                    }
                ),
            )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
