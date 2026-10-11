"""Owned UI worker: fixed requests call the existing CLI scientific engine."""

import _thread
import argparse
import importlib.util
import json
import sys
import threading
from pathlib import Path

import yaml
from threadpoolctl import threadpool_limits

from shiftlab.data.storage import DataError, read_json, verify_seal, write_json
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.runner import train_methods
from shiftlab.experiments.storage import now

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.jobs import update_status  # noqa: E402


def configuration_snapshot(config: StudyConfig, directory: Path) -> Path:
    """The full driver reads the launched request, never a subsequently edited config."""
    path = directory / "configuration.yaml"
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        yaml.safe_dump(config.model_dump(mode="json"), stream, sort_keys=True)
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    args = parser.parse_args()
    if len(args.job_id) != 32 or any(c not in "0123456789abcdef" for c in args.job_id):
        raise DataError("Invalid owned job identifier")
    directory = ROOT / "artifacts/ui_jobs" / args.job_id
    request = read_json(directory / "request.json")
    verify_seal(request)
    if request["job_id"] != args.job_id:
        raise DataError("Worker request identity differs")
    config = StudyConfig.model_validate(request["configuration"])
    status = {
        "job_id": args.job_id,
        "action": request["action"],
        "status": "running",
        "started_at": now(),
        "error": None,
    }
    update_status(directory / "status.json", status)
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(0.2):
            if (directory / "cancel.request").exists():
                print(
                    "Cancellation requested; interrupting at a Python operation boundary",
                    flush=True,
                )
                _thread.interrupt_main()
                return

    threading.Thread(target=monitor, daemon=True).start()
    try:
        if (directory / "cancel.request").exists():
            raise KeyboardInterrupt
        print(
            json.dumps(
                {
                    "job_id": args.job_id,
                    "action": request["action"],
                    "configuration": config.model_dump(mode="json"),
                }
            ),
            flush=True,
        )
        with threadpool_limits(limits=1):
            if request["action"] == "fit":
                result = train_methods(
                    config,
                    root=ROOT,
                    data_root=ROOT / "data",
                    progress=lambda event: print(json.dumps(event), flush=True),
                )
            elif request["action"] == "study":
                if config.mode != "primary":
                    raise DataError("Full study requires the locked primary configuration")
                script = ROOT / "tools/run_frozen_study.py"
                spec = importlib.util.spec_from_file_location("frozen_study_driver", script)
                driver = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(driver)
                snapshot = configuration_snapshot(config, directory)
                sys.argv = [str(script), "execute", "--config", str(snapshot)]
                code = driver.main()
                if code:
                    raise DataError(f"Study execution exited {code}; see preserved log")
                result = {"status": "study_complete", "notice": "See frozen study artifacts"}
            else:
                raise DataError("Unknown worker action")
        write_json(directory / "result.json", result)
        status.update(status="completed", result=result)
        return_code = 0
    except KeyboardInterrupt:
        status.update(status="cancelled", error="Cancelled; checkpoints and attempts preserved")
        print(status["error"], flush=True)
        return_code = 130
    except Exception as error:
        status.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(status["error"], flush=True)
        return_code = 1
    finally:
        stop.set()
        status["ended_at"] = now()
        update_status(directory / "status.json", status)
    print(json.dumps(status), flush=True)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
