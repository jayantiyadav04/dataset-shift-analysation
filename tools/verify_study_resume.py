"""Run the completed study again and verify every immutable file is unchanged."""

import argparse
import subprocess
import sys
from pathlib import Path

from shiftlab.data.storage import DataError, read_json, seal, verify_seal, write_json
from shiftlab.experiments.provenance import content_hash, file_sha256
from shiftlab.experiments.study import open_study


def inventory(root: Path, study: Path, output: Path) -> dict:
    immutable_roots = [
        output / kind
        for kind in (
            "runs",
            "source_cache",
            "importance_cache",
            "coral_cache",
            "evaluations",
            "cv_checkpoints",
        )
    ]
    paths = [p for directory in immutable_roots for p in directory.rglob("*") if p.is_file()]
    mutable = {"checkpoint.json", "completed_runs.csv", "missing_runs.csv", "active_job.json"}
    paths.extend(p for p in study.rglob("*") if p.is_file() and p.name not in mutable)
    return {
        p.relative_to(root).as_posix(): {
            "sha256": file_sha256(p),
            "size": p.stat().st_size,
            "mtime_ns": p.stat().st_mtime_ns,
        }
        for p in paths
    }


def verify(root: Path, study_id: str, receipt_path: Path) -> dict:
    directory, lock = open_study(root, study_id)
    progress = read_json(directory / "checkpoint.json")
    verify_seal(progress)
    if progress["status"] != "complete" or progress["evaluated_records"] != 1280:
        raise DataError("The real resume audit requires a fully completed study")
    if (directory / "writer.lock").exists():
        raise DataError("Refusing to audit resume while a study writer exists")
    output = root / lock["configuration"]["execution"]["output_root"]
    print("Hashing completed immutable evidence before resume", flush=True)
    before = inventory(root, directory, output)
    log = output / "frozen_study_resume.log"
    print("Executing completed-study resume; models and predictions must be reused", flush=True)
    with log.open("x", encoding="utf-8") as stream:
        result = subprocess.run(
            [sys.executable, str(root / "tools" / "run_frozen_study.py"), "execute"],
            cwd=root,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    if result.returncode:
        raise DataError(f"Resume exited {result.returncode}; preserved log: {log}")
    print("Hashing completed immutable evidence after resume", flush=True)
    after = inventory(root, directory, output)
    if before != after:
        changed = sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k))
        raise DataError(f"Resume changed immutable hashes, sizes or timestamps: {changed[:5]}")
    progress = read_json(directory / "checkpoint.json")
    verify_seal(progress)
    if progress["status"] != "complete" or progress["evaluated_records"] != 1280:
        raise DataError("Verified cached analysis did not restore complete progress")
    receipt = seal(
        {
            "status": "passed",
            "study_id": study_id,
            "immutable_files_checked": len(before),
            "inventory_hash_before": content_hash(before),
            "inventory_hash_after": content_hash(after),
            "hashes_sizes_timestamps_unchanged": True,
            "resume_exit_code": result.returncode,
            "progress_after": progress,
            "resume_log_sha256": file_sha256(log),
        }
    )
    write_json(receipt_path, receipt)
    return receipt


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify(Path.cwd(), args.study_id, args.out)
    print(f"PASS: {result['immutable_files_checked']} immutable files unchanged", flush=True)
