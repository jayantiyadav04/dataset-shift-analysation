"""Complete content identities, observed source revisions and run states."""

import hashlib
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from shiftlab.contracts import MethodName, ModelName


def canonical_json(value: Any) -> str:
    """Stable UTF-8 JSON; non-finite numbers and unsupported objects fail."""
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    )


def content_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class RunStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    FITTED = "fitted"
    EVALUATED = "evaluated"
    FAILED = "failed"
    CANCELLED = "cancelled"


_TRANSITIONS = {
    RunStatus.PENDING: {RunStatus.RUNNING, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.RUNNING: {RunStatus.FITTED, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.FITTED: {RunStatus.EVALUATED, RunStatus.FAILED, RunStatus.CANCELLED},
    RunStatus.EVALUATED: set(),
    RunStatus.FAILED: set(),
    RunStatus.CANCELLED: set(),
}


def transition(current: RunStatus, new: RunStatus) -> RunStatus:
    """Terminal records cannot be overwritten by a later attempt."""
    current, new = RunStatus(current), RunStatus(new)
    if new not in _TRANSITIONS[current]:
        raise ValueError(f"Invalid run transition: {current} -> {new}")
    return new


@dataclass(frozen=True)
class RunKey:
    dataset_hash: str
    task_id: str
    split_hash: str
    model: ModelName | str
    selected_source_parameter_hash: str
    preprocessing_hash: str
    method: MethodName
    adaptation_parameter_hash: str
    seed: int
    protocol_version: str
    code_version: str
    dependency_lock_hash: str
    config_hash: str
    code_snapshot_hash: str | None = None

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if name == "code_snapshot_hash" and value is None:
                continue
            if name.endswith("_hash") and (
                not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None
            ):
                raise ValueError(f"{name} must be a computed SHA-256 hex digest")
        if re.fullmatch(r"[0-9a-f]{40}", self.code_version) is None:
            raise ValueError("code_version must be a full Git commit ID")
        if not self.task_id or re.fullmatch(r"[1-9]\d*\.\d+", self.protocol_version) is None:
            raise ValueError("A task ID and versioned protocol are required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**32:
            raise ValueError("seed must be an unsigned 32-bit integer")
        if self.model != "majority":
            object.__setattr__(self, "model", ModelName(self.model))
        object.__setattr__(self, "method", MethodName(self.method))

    @property
    def run_id(self) -> str:
        return content_hash(asdict(self))


@dataclass(frozen=True)
class SourceRevision:
    commit: str | None
    dirty: bool | None


def source_revision(root: Path) -> SourceRevision:
    """Read Git using fixed argument lists; no shell or mutation."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        status = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=normal"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return SourceRevision(None, None)
    revision = commit.stdout.strip() if commit.returncode == 0 else None
    return SourceRevision(revision, bool(status.stdout.strip()) if status.returncode == 0 else None)


def code_snapshot(root: Path) -> dict[str, str]:
    """Include actual engine bytes even when development is not committed yet."""
    paths = sorted((root / "src" / "shiftlab").rglob("*.py"))
    paths += [root / "pyproject.toml"]
    return {path.relative_to(root).as_posix(): file_sha256(path) for path in paths}
