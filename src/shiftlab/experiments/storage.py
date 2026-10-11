"""Immutable local records, exclusive writers and checksum-gated serialization.

The trust directory is a local creation receipt, not a signature against an attacker
who can rewrite the entire workspace. Never import receipts or pickle files from
uploads or external artifact bundles.
"""

import re
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib

from shiftlab.data.storage import (
    DataError,
    publish_directory,
    read_json,
    safe_child,
    seal,
    stage_directory,
    verify_files,
    verify_seal,
    write_json,
)
from shiftlab.experiments.provenance import content_hash, file_sha256


def now() -> str:
    return datetime.now(UTC).isoformat()


class LocalArtifactStore:
    """Only this store creates loadable models; failed attempts are retained."""

    def __init__(self, root: Path):
        self.root = root.resolve()

    def directory(self, kind: str, identity: str) -> Path:
        if kind not in {"runs", "source_cache", "evaluations"} or not re.fullmatch(
            r"[0-9a-f]{64}", identity
        ):
            raise DataError("Unknown record namespace or invalid content identity")
        return safe_child(self.root, f"{kind}/{identity}")

    def verify(self, kind: str, identity: str) -> dict[str, Any]:
        directory = self.directory(kind, identity)
        receipt = read_json(self.root / "trust" / kind / f"{identity}.json")
        if receipt.get("manifest_sha256") != file_sha256(directory / "manifest.json"):
            raise DataError("Local creation receipt does not match manifest")
        manifest = read_json(directory / "manifest.json")
        verify_seal(manifest)
        if (
            manifest.get("identity") != identity
            or content_hash(manifest["canonical_key"]) != identity
        ):
            raise DataError("Record key does not match its directory")
        if manifest["status"] not in {"fitted", "evaluated"}:
            raise DataError("Only successfully frozen records may be reused")
        verify_files(directory, manifest["checksums"])
        return manifest

    def load_joblib(self, kind: str, identity: str, name: str) -> Any:
        if kind not in {"runs", "source_cache"} or name not in {
            "preprocessing.joblib",
            "fitted_model.joblib",
        }:
            raise DataError("Only known, locally created model files may be loaded")
        manifest = self.verify(kind, identity)
        if name not in manifest["checksums"]:
            raise DataError("Serialized file is not in the frozen checksum inventory")
        return joblib.load(self.directory(kind, identity) / name)

    def create(
        self,
        kind: str,
        key: dict[str, Any],
        produce: Callable[[Path], dict[str, Any]],
        *,
        resume: bool = True,
    ) -> tuple[dict[str, Any], bool]:
        identity = content_hash(key)
        destination = self.directory(kind, identity)
        # Exclusive file creation also guards a cache hit against a competing writer.
        locks = self.root / "locks"
        locks.mkdir(parents=True, exist_ok=True)
        lock = locks / f"{kind}-{identity}.lock"
        try:
            stream = lock.open("x", encoding="utf-8")
        except FileExistsError as error:
            raise DataError(
                f"Record is locked: {lock}. Inspect an interrupted writer before recovery."
            ) from error
        stage = None
        started = now()
        try:
            with stream:
                stream.write(started)
            if destination.exists():
                if not resume:
                    raise DataError("Immutable record already exists; enable resume to reuse it")
                return self.verify(kind, identity), True
            stage = stage_directory(destination)
            with (stage / "run.log").open("x", encoding="utf-8") as log:
                log.write(f"{started} pending\n{now()} running\n")
            write_json(
                stage / "attempt.json",
                {
                    "status": "running",
                    "identity": identity,
                    "canonical_key": key,
                    "started_at": started,
                },
            )
            metadata = produce(stage)
            status = "evaluated" if kind == "evaluations" else "fitted"
            with (stage / "run.log").open("a", encoding="utf-8") as log:
                log.write(f"{now()} {status}\n")
            checksums = {
                p.relative_to(stage).as_posix(): file_sha256(p)
                for p in sorted(stage.rglob("*"))
                if p.is_file()
            }
            manifest = seal(
                {
                    **metadata,
                    "schema_version": 1,
                    "identity": identity,
                    "run_id": identity if kind == "runs" else metadata.get("run_id"),
                    "canonical_key": key,
                    "status": status,
                    "started_at": started,
                    "ended_at": now(),
                    "checksums": checksums,
                    "paths": {name: name for name in checksums},
                    "error": None,
                }
            )
            write_json(stage / "manifest.json", manifest)
            publish_directory(stage, destination)
            stage = None
            write_json(
                self.root / "trust" / kind / f"{identity}.json",
                {
                    "manifest_sha256": file_sha256(destination / "manifest.json"),
                    "created_locally_at": now(),
                },
            )
            return self.verify(kind, identity), False
        except BaseException as error:
            if stage is not None and stage.exists():
                status = "cancelled" if isinstance(error, KeyboardInterrupt) else "failed"
                with (stage / "run.log").open("a", encoding="utf-8") as log:
                    log.write(f"{now()} {status}: {type(error).__name__}: {error}\n")
                if (stage / "manifest.json").exists():
                    (stage / "manifest.json").rename(stage / "unpublished_manifest.json")
                write_json(
                    stage / "manifest.json",
                    seal(
                        {
                            "schema_version": 1,
                            "identity": identity,
                            "canonical_key": key,
                            "status": status,
                            "started_at": started,
                            "ended_at": now(),
                            "error": {"type": type(error).__name__, "message": str(error)},
                            "checksums": {
                                p.relative_to(stage).as_posix(): file_sha256(p)
                                for p in stage.rglob("*")
                                if p.is_file()
                            },
                        }
                    ),
                )
                failure = self.root / "failures" / identity / uuid.uuid4().hex
                failure.parent.mkdir(parents=True, exist_ok=True)
                publish_directory(stage, failure)
            raise
        finally:
            lock.unlink()
