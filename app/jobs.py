"""Persistent single-worker UI jobs with cancellation of owned processes only."""

import os
import subprocess
import sys
import uuid
from pathlib import Path
from time import sleep

from shiftlab.data.storage import DataError, read_json, seal, verify_seal, write_json
from shiftlab.experiments.checkpoints import atomic_json, job_owner, pid_alive
from shiftlab.experiments.config import StudyConfig, load_config
from shiftlab.experiments.storage import now

OWNER = uuid.uuid4().hex
HANDLES: dict[str, subprocess.Popen] = {}
ACTIVE = {"starting", "running", "cancelling", "detached"}


def update_status(path: Path, value: dict) -> None:
    """Bound transient Windows reader contention on mutable UI status only."""
    for attempt in range(40):
        try:
            atomic_json(path, seal(value))
            return
        except OSError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 39:
                raise
            sleep(0.2)


def small_config(root: Path, task: str, model: str, method: str, seed: int) -> StudyConfig:
    primary = load_config(root / "configs/study.yaml")
    if task not in primary.available_tasks() or seed not in primary.seeds:
        raise DataError("Choose a declared transfer and seed")
    payload = primary.model_dump(mode="json")
    payload.update(
        mode="smoke",
        study_name="workbench_development",
        datasets=[task.split("_")[0]],
        task_filter=[task],
        models=[model],
        methods=[method],
        seeds=[seed],
        include_majority_baseline=False,
    )
    payload["ablations"]["tasks"] = []
    return StudyConfig.model_validate(payload)


class JobManager:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()
        self.directory = self.root / "artifacts/ui_jobs"

    def current(self) -> dict | None:
        active = self.directory / "active.json"
        if not active.exists():
            return None
        launch = read_json(active)
        verify_seal(launch)
        job = self.directory / launch["job_id"]
        status_path = job / "status.json"
        status = read_json(status_path) if status_path.exists() else launch
        verify_seal(status)
        handle = HANDLES.get(launch["job_id"])
        if status["status"] in ACTIVE:
            code = handle.poll() if handle else None
            alive = handle is not None and code is None
            if handle is None:
                alive = pid_alive(launch["pid"])
            if not alive:
                status = {
                    **status,
                    "status": "failed",
                    "ended_at": now(),
                    "error": f"Worker exited without a terminal receipt (exit {code}). "
                    "Saved artifacts/logs are preserved; inspect CLI recovery.",
                }
                status.pop("manifest_hash", None)
                update_status(status_path, status)
            elif handle is None:
                status = {
                    **status,
                    "status": "detached",
                    "error": "Worker outlived this dashboard process. "
                    "Use its log and CLI recovery; "
                    "this session cannot cancel a process it does not own.",
                }
        return {
            **status,
            "job_id": launch["job_id"],
            "owned": handle is not None,
            "process_running": handle is not None and handle.poll() is None,
            "directory": job.relative_to(self.root).as_posix(),
        }

    def start(
        self,
        *,
        action: str,
        task: str | None = None,
        model: str = "lr",
        method: str = "source_only",
        seed: int = 17,
    ) -> dict:
        if action not in ("fit", "study"):
            raise DataError("Unsupported UI job action")
        configuration = (
            small_config(self.root, task, model, method, seed)
            if action == "fit"
            else (load_config(self.root / "configs/study.yaml"))
        )
        # A single cross-session/process launch guard; no launch on a rerender.
        with job_owner(self.directory):
            existing = self.current()
            if existing and (existing["status"] in ACTIVE or existing["process_running"]):
                raise DataError("One worker is already active; finish or cancel it first")
            job_id = uuid.uuid4().hex
            job = self.directory / job_id
            job.mkdir()
            request = seal(
                {
                    "job_id": job_id,
                    "owner": OWNER,
                    "action": action,
                    "configuration": configuration.model_dump(mode="json"),
                    "created_at": now(),
                }
            )
            write_json(job / "request.json", request)
            log = (job / "worker.log").open("xb")
            arguments = [
                sys.executable,
                "-u",
                str(self.root / "tools/run_workbench_job.py"),
                "--job-id",
                job_id,
            ]
            try:
                handle = subprocess.Popen(
                    arguments,
                    cwd=self.root,
                    stdout=log,
                    stderr=log,
                    shell=False,
                    creationflags=(subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0),
                )
            finally:
                log.close()
            HANDLES[job_id] = handle
            launch = {
                "job_id": job_id,
                "owner": OWNER,
                "pid": handle.pid,
                "status": "starting",
                "started_at": now(),
                "action": action,
            }
            try:
                update_status(self.directory / "active.json", launch)
            except BaseException:
                # Do not leave an unregistered child that a later rerender could duplicate.
                if handle.poll() is None:
                    handle.terminate()
                raise
            return launch

    def cancel(self, job_id: str) -> None:
        current = self.current()
        if (
            current is None
            or current["job_id"] != job_id
            or current["status"] not in ACTIVE
            or job_id not in HANDLES
        ):
            raise DataError("Cancellation is limited to this app's active owned worker")
        path = self.directory / job_id / "cancel.request"
        if not path.exists():
            path.write_text("Cancellation requested by owning dashboard\n", encoding="utf-8")

    def log(self, job_id: str, limit: int = 24000) -> str:
        if len(job_id) != 32 or any(c not in "0123456789abcdef" for c in job_id):
            raise DataError("Invalid job identifier")
        path = self.directory / job_id / "worker.log"
        if not path.exists():
            return "Waiting for worker log."
        with path.open("rb") as stream:
            stream.seek(max(0, path.stat().st_size - limit))
            return stream.read().decode("utf-8", errors="replace")

    def history(self) -> list[dict]:
        rows = []
        for path in sorted(self.directory.glob("*/status.json")):
            value = read_json(path)
            verify_seal(value)
            rows.append(
                {
                    k: value.get(k)
                    for k in ("job_id", "action", "status", "started_at", "ended_at", "error")
                }
            )
        return rows
