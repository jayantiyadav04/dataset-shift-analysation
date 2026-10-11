"""Hash-checked local artifacts; existing evidence is never replaced."""

import json
import os
import uuid
from pathlib import Path
from typing import Any

from shiftlab.experiments.provenance import canonical_json, content_hash, file_sha256


class DataError(ValueError):
    """Missing, inconsistent or unsafe research input."""


def read_json(path: Path) -> Any:
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DataError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def reject_constant(value):
        raise DataError(f"Non-finite JSON number: {value}")

    try:
        return json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise DataError(f"Cannot read {path}: {error}") from error


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(canonical_json(value) + "\n")


def stage_directory(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    stage = destination.with_name(f".{destination.name}.pending-{uuid.uuid4().hex}")
    stage.mkdir()
    return stage


def publish_directory(stage: Path, destination: Path) -> None:
    """One atomic directory rename; a competing writer or prior freeze fails."""
    if destination.exists():
        raise DataError(f"Refusing to replace immutable data: {destination}")
    os.rename(stage, destination)


def safe_child(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative:
        raise DataError("Artifact path must be relative without drive or backslash")
    path = root / relative
    if path.is_absolute() and Path(relative).is_absolute():
        raise DataError("Absolute artifact path is forbidden")
    if ".." in Path(relative).parts or not path.resolve().is_relative_to(root.resolve()):
        raise DataError("Artifact path escapes its data root")
    return path


def verify_files(root: Path, checksums: dict[str, str]) -> None:
    for name, expected in checksums.items():
        path = safe_child(root, name)
        if not path.is_file() or file_sha256(path) != expected:
            raise DataError(f"Missing or checksum-mismatched artifact: {path}")


def seal(value: dict[str, Any], key: str = "manifest_hash") -> dict[str, Any]:
    return {**value, key: content_hash(value)}


def verify_seal(value: dict[str, Any], key: str = "manifest_hash") -> None:
    if not isinstance(value, dict):
        raise DataError("Manifest must be a JSON object")
    payload = {k: v for k, v in value.items() if k != key}
    if value.get(key) != content_hash(payload):
        raise DataError(f"Invalid {key}")
