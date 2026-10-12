"""Package permitted source and aggregate evidence with an exact byte inventory."""

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile

ROOT_FILES = {
    "README.md",
    "pyproject.toml",
    "requirements-lock.txt",
    ".gitignore",
    ".gitattributes",
}
DIRECTORIES = {"src", "app", "tests", "tools", "configs", "reports", "examples"}
SUFFIXES = {
    ".py",
    ".typed",
    ".md",
    ".txt",
    ".yaml",
    ".toml",
    ".csv",
    ".json",
    ".png",
    ".pdf",
    ".review",
}
INVENTORY = "_PACKAGE_MANIFEST.json"


def permitted(name: str) -> bool:
    """Allow project files, never arbitrary directories, data, models or credentials."""
    path = PurePosixPath(name)
    if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        return False
    if name in ROOT_FILES or name == ".streamlit/config.toml":
        return True
    return (
        len(path.parts) > 1
        and path.parts[0] in DIRECTORIES
        and not any(part.startswith(".") or part == "__pycache__" for part in path.parts)
        and path.suffix in SUFFIXES
        and not any(word in path.name.lower() for word in ("credential", "secret", "private_key"))
    )


def payloads(root: Path, names: list[str]) -> dict[str, bytes]:
    root = root.resolve()
    result = {}
    for name in sorted(set(names)):
        if not permitted(name):
            raise ValueError(f"Not a permitted package path: {name}")
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f"Path escapes the project: {name}")
        if any(p.is_symlink() for p in (path, *path.parents) if p != root.parent):
            raise ValueError(f"Symlink not permitted: {name}")
        if not path.is_file():
            raise ValueError(f"Missing regular file: {name}")
        result[name] = path.read_bytes()
    return result


def inspect_archive(path: Path) -> dict:
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or INVENTORY not in names:
            raise ValueError("Duplicate entries or missing inventory")
        manifest = json.loads(archive.read(INVENTORY))
        files = manifest["files"]
        if set(names) != set(files) | {INVENTORY}:
            raise ValueError("Archive members differ from the inventory")
        for name, expected in files.items():
            if not permitted(name):
                raise ValueError(f"Unsafe archive entry: {name}")
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Archive hash mismatch: {name}")
    return manifest


def write_archive(path: Path, files: dict[str, bytes], revision: str, preview: bool) -> dict:
    if path.exists():
        raise FileExistsError(f"Preserve the existing archive: {path}")
    if not files or not all(permitted(name) for name in files):
        raise ValueError("Archive requires permitted project files")
    manifest = {
        "schema_version": 1,
        "source_revision": revision,
        "uncommitted_preview": preview,
        "scope": "Source/configuration and aggregate evidence; no raw data, models or split IDs",
        "files": {name: hashlib.sha256(data).hexdigest() for name, data in sorted(files.items())},
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with ZipFile(temporary, "w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
            for name, data in sorted(files.items()):
                archive.writestr(name, data)
            archive.writestr(INVENTORY, json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        inspect_archive(temporary)
        # Hard-link creation is atomic and refuses to replace an existing destination.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest


def package(root: Path, destination: Path, preview: bool = False) -> dict:
    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(root), *args], text=True)

    if not preview and git("status", "--porcelain"):
        raise ValueError("Commit reviewed changes before creating the final source archive")
    arguments = ["ls-files", "-z", "--cached"]
    if preview:
        arguments.extend(["--others", "--exclude-standard"])
    names = [name for name in git(*arguments).split("\0") if name]
    contents = payloads(root, names)
    revision = git("rev-parse", "HEAD").strip()
    if not preview:
        # A clean checkout can still have normalized working-tree line endings.
        # Final packages use the exact committed bytes, including sealed evidence.
        contents = {
            name: subprocess.check_output(["git", "-C", str(root), "show", f"{revision}:{name}"])
            for name in contents
        }
    destination.mkdir(parents=True, exist_ok=True)
    archives = {
        "source_and_evidence.zip": contents,
        "result_exports.zip": {
            name: data
            for name, data in contents.items()
            if name.startswith(("reports/study/", "reports/final/"))
        },
    }
    output = {"source_revision": revision, "uncommitted_preview": preview, "archives": {}}
    for filename, files in archives.items():
        path = destination / filename
        manifest = write_archive(path, files, revision, preview)
        output["archives"][filename] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file_count": len(manifest["files"]),
            "bytes": path.stat().st_size,
        }
    with (destination / "package_verification.json").open("x", encoding="utf-8") as stream:
        json.dump(output, stream, indent=2)
    print(json.dumps(output, indent=2))
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--preview", action="store_true")
    args = parser.parse_args()
    package(args.root.resolve(), args.out.resolve(), args.preview)
