"""Exercise real owned-worker completion, duplicate refusal and cancellation."""

import argparse
import sys
from pathlib import Path
from time import monotonic, sleep

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.jobs import ACTIVE, JobManager  # noqa: E402
from shiftlab.data.storage import DataError, seal, write_json  # noqa: E402


def wait(manager: JobManager, timeout: float = 90) -> dict:
    deadline = monotonic() + timeout
    while monotonic() < deadline:
        record = manager.current()
        if record["status"] not in ACTIVE and not record["process_running"]:
            return record
        sleep(0.2)
    raise RuntimeError("Owned worker did not finish; inspect its preserved log")


def verify(destination: Path) -> dict:
    manager = JobManager(ROOT)
    started = manager.start(action="fit", task="gas_b01_b02", model="lr", method="source_only")
    duplicate_refused = False
    try:
        manager.start(action="fit", task="gas_b01_b02")
    except DataError:
        duplicate_refused = True
    if not duplicate_refused:
        raise RuntimeError("Duplicate worker was incorrectly allowed")
    completed = wait(manager)
    if completed["status"] != "completed":
        raise RuntimeError(manager.log(started["job_id"]))
    if completed["result"]["evaluated_records"] != 0:
        raise RuntimeError("Development UI job unexpectedly evaluated target data")
    cancelled_start = manager.start(action="fit", task="har_official_group_transfer", model="rf")
    deadline = monotonic() + 30
    while "source_selection" not in manager.log(cancelled_start["job_id"]):
        if monotonic() > deadline:
            raise RuntimeError("Cancellation worker did not reach the fitting boundary")
        sleep(0.2)
    manager.cancel(cancelled_start["job_id"])
    cancelled = wait(manager)
    if cancelled["status"] != "cancelled":
        raise RuntimeError(manager.log(cancelled_start["job_id"]))
    result = seal(
        {
            "status": "passed",
            "duplicate_launch_refused": duplicate_refused,
            "completed": completed,
            "cancelled": cancelled,
            "development_target_evaluations": 0,
        }
    )
    write_json(destination, result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("docs/phase_07_jobs_verification.json"))
    result = verify(parser.parse_args().out)
    print(
        "PASS: real one-fit worker, duplicate refusal, zero target evaluation, owned cancellation"
    )
