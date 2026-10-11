"""Atomic progress and exclusive job ownership; immutable scientific records stay separate."""

import csv
import ctypes
import json
import os
import uuid
from contextlib import contextmanager
from pathlib import Path

from shiftlab.data.storage import DataError, read_json, seal, verify_seal
from shiftlab.experiments.provenance import canonical_json
from shiftlab.experiments.storage import now


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    with temporary.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_csv(path: Path, rows: list[dict], columns: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}")
    fields = columns or list(rows[0])
    with temporary.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def pid_alive(pid: int) -> bool:
    if os.name == "nt":
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = ctypes.c_void_p
        handle = kernel.OpenProcess(0x1000, False, pid)
        if not handle:
            if ctypes.get_last_error() == 87:
                return False
            raise DataError("Cannot establish whether the lock owner is alive")
        kernel.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel.CloseHandle(handle)
        return True
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


@contextmanager
def job_owner(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "writer.lock"
    try:
        with path.open("x", encoding="utf-8") as stream:
            stream.write(canonical_json({"pid": os.getpid(), "created_at": now()}))
    except FileExistsError as error:
        raise DataError(
            "Study has a writer lock; use explicit recover after inspecting its owner"
        ) from error
    try:
        yield
    finally:
        path.unlink()


class Journal:
    """Hash-chained events are append-only; a partial final line blocks silent recovery."""

    def __init__(self, directory: Path):
        self.path = directory / "events.jsonl"
        self.events = []
        previous = None
        if self.path.exists():
            for line in self.path.read_text(encoding="utf-8").splitlines():
                event = json.loads(line)
                verify_seal(event)
                if event["previous"] != previous:
                    raise DataError("Study event chain is broken")
                self.events.append(event)
                previous = event["manifest_hash"]
        self.previous = previous

    def append(self, value: dict) -> dict:
        event = seal({**value, "observed_at": now(), "previous": self.previous})
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(canonical_json(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        self.events.append(event)
        self.previous = event["manifest_hash"]
        return event


def recover_owner(directory: Path) -> dict:
    path = directory / "writer.lock"
    owner = read_json(path)
    if type(owner.get("pid")) is not int or pid_alive(owner["pid"]):
        raise DataError("Recovery requires a confirmed dead writer")
    destination = directory / f"recovered-writer-{uuid.uuid4().hex}.json"
    path.rename(destination)
    return {"status": "dead_writer_lock_preserved", "receipt": destination.name}


def recover_records(store) -> list[dict]:
    """Explicitly preserve abandoned attempts, only after proving their owner dead."""
    from shiftlab.data.storage import publish_directory
    from shiftlab.experiments.provenance import file_sha256

    recovered = []
    for lock in sorted((store.root / "locks").glob("*.lock")):
        owner = read_json(lock)
        if type(owner.get("pid")) is not int or pid_alive(owner["pid"]):
            raise DataError("A record writer is alive or cannot be inspected; recovery stopped")
        kind, identity = owner["kind"], owner["identity"]
        destination = store.directory(kind, identity)
        if lock.name != f"{kind}-{identity}.lock":
            raise DataError("Abandoned lock identity differs")
        for stage in destination.parent.glob(f".{identity}.pending-*"):
            if (stage / "manifest.json").exists():
                (stage / "manifest.json").rename(stage / "abandoned_manifest.json")
            attempt = read_json(stage / "attempt.json")
            atomic_json(
                stage / "manifest.json",
                seal(
                    {
                        **attempt,
                        "status": "cancelled",
                        "recovered_at": now(),
                        "error": {
                            "type": "InterruptedProcess",
                            "message": "Owner confirmed dead; original attempt preserved",
                        },
                        "checksums": {
                            p.relative_to(stage).as_posix(): file_sha256(p)
                            for p in stage.rglob("*")
                            if p.is_file()
                        },
                    }
                ),
            )
            failure = store.root / "failures" / identity / uuid.uuid4().hex
            failure.parent.mkdir(parents=True, exist_ok=True)
            publish_directory(stage, failure)
        receipt = store.root / "recovery_receipts" / f"{lock.stem}-{uuid.uuid4().hex}.json"
        receipt.parent.mkdir(parents=True, exist_ok=True)
        lock.rename(receipt)
        recovered.append({"kind": kind, "identity": identity, "dead_pid": owner["pid"]})
    return recovered
