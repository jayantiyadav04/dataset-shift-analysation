"""Official downloads, manual imports, and bounded non-executable extraction."""

import gzip
import stat
import tarfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath

from shiftlab.contracts import DatasetName
from shiftlab.data.storage import (
    DataError,
    publish_directory,
    read_json,
    stage_directory,
    verify_files,
    write_json,
)
from shiftlab.experiments.provenance import file_sha256

SOURCES = {
    "gas": {
        "main": "https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip",
    },
    "har": {
        "main": "https://archive.ics.uci.edu/static/public/240/"
        "human+activity+recognition+using+smartphones.zip",
    },
    "amazon": {
        "main": "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz",
        "books_unlabeled": "https://www.cs.jhu.edu/~mdredze/datasets/sentiment/book.unlabeled.gz",
    },
}
MAX_DOWNLOAD = 256 * 1024**2
MAX_EXTRACTED = 2 * 1024**3


def _copy_bounded(source, target, limit: int) -> int:
    count = 0
    while block := source.read(1024 * 1024):
        count += len(block)
        if count > limit:
            raise DataError("Archive exceeds the declared size limit")
        target.write(block)
    return count


def fetch(root: Path, dataset: str, *, asset: str = "main", archive: Path | None = None) -> dict:
    dataset = DatasetName(dataset).value
    if asset not in SOURCES[dataset]:
        raise DataError(f"Unknown {dataset} archive asset: {asset}")
    url = SOURCES[dataset][asset]
    destination = root / "raw" / dataset / asset
    if destination.exists():
        receipt = read_json(destination / "receipt.json")
        verify_files(destination, {receipt["filename"]: receipt["sha256"]})
        if archive is not None and file_sha256(archive) != receipt["sha256"]:
            raise DataError("Different archive already imported; use a separate data root")
        return receipt
    if archive is not None and not archive.is_file():
        raise DataError(f"Manual archive is missing: {archive}")
    stage = stage_directory(destination)
    filename = url.rsplit("/", 1)[1]
    output = stage / filename
    if archive is None:
        request = urllib.request.Request(url, headers={"User-Agent": "DomainShiftWorkbench/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=60) as response, output.open("xb") as dst:
                if not response.url.startswith("https://"):
                    raise DataError("Download redirected to a non-HTTPS URL")
                _copy_bounded(response, dst, MAX_DOWNLOAD)
                resolved_url = response.url
        except OSError as error:
            raise DataError(
                f"Download failed for {url}: {error}. Use --archive with the official archive."
            ) from error
    else:
        with archive.open("rb") as src, output.open("xb") as dst:
            _copy_bounded(src, dst, MAX_DOWNLOAD)
        resolved_url = None
    receipt = {
        "schema_version": 1,
        "dataset_id": dataset,
        "asset": asset,
        "source_url": url,
        "resolved_url": resolved_url,
        "accessed_at": datetime.now(UTC).isoformat(),
        "acquisition": "official_download" if archive is None else "manual_import",
        "filename": filename,
        "sha256": file_sha256(output),
        "bytes": output.stat().st_size,
    }
    write_json(stage / "receipt.json", receipt)
    publish_directory(stage, destination)
    return receipt


def _member_path(destination: Path, name: str, seen: set[str]) -> Path:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or ":" in name
        or "\x00" in name
        or path.is_absolute()
        or ".." in path.parts
        or any(part.rstrip(" .") != part for part in path.parts)
        or any(
            part.split(".")[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *(f"COM{i}" for i in range(10)),
                *(f"LPT{i}" for i in range(10)),
            }
            for part in path.parts
        )
    ):
        raise DataError(f"Unsafe archive member: {name!r}")
    normalized = path.as_posix().casefold()
    if normalized in seen:
        raise DataError(f"Duplicate archive member: {name!r}")
    seen.add(normalized)
    target = destination.joinpath(*path.parts)
    if not target.resolve().is_relative_to(destination.resolve()):
        raise DataError("Archive member escapes extraction directory")
    return target


def safe_extract(archive: Path, destination: Path, *, limit: int = MAX_EXTRACTED) -> None:
    """Reject links, special files, ambiguous Windows paths and archive bombs."""
    destination.mkdir(parents=True, exist_ok=False)
    seen: set[str] = set()
    remaining = limit
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as container:
            for member in container.infolist():
                path = _member_path(destination, member.filename, seen)
                mode = member.external_attr >> 16
                if stat.S_ISLNK(mode) or (stat.S_IFMT(mode) not in (0, stat.S_IFREG, stat.S_IFDIR)):
                    raise DataError("Archive links and special files are forbidden")
                if member.is_dir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                if member.file_size > remaining:
                    raise DataError("Archive exceeds extraction limit")
                path.parent.mkdir(parents=True, exist_ok=True)
                with container.open(member) as src, path.open("xb") as dst:
                    remaining -= _copy_bounded(src, dst, remaining)
    elif tarfile.is_tarfile(archive):
        with tarfile.open(archive, "r:*") as container:
            for member in container:
                path = _member_path(destination, member.name, seen)
                if member.isdir():
                    path.mkdir(parents=True, exist_ok=True)
                    continue
                if not member.isfile():
                    raise DataError("Archive links and special files are forbidden")
                if member.size > remaining:
                    raise DataError("Archive exceeds extraction limit")
                path.parent.mkdir(parents=True, exist_ok=True)
                with container.extractfile(member) as src, path.open("xb") as dst:
                    remaining -= _copy_bounded(src, dst, remaining)
    elif archive.name.endswith(".gz"):
        with gzip.open(archive, "rb") as src, (destination / "book.unlabeled").open("xb") as dst:
            _copy_bounded(src, dst, remaining)
    else:
        raise DataError(f"Unsupported or corrupt archive: {archive.name}")


def extracted_inputs(root: Path, dataset: str) -> tuple[Path, list[dict], dict[str, str]]:
    receipts = []
    for asset in SOURCES[dataset]:
        base = root / "raw" / dataset / asset
        receipt = read_json(base / "receipt.json")
        verify_files(base, {receipt["filename"]: receipt["sha256"]})
        receipts.append(receipt)
    destination = root / "raw" / dataset / "extracted"
    if destination.exists():
        index = read_json(destination / "extraction.json")
        if index["raw_sha256"] != {r["asset"]: r["sha256"] for r in receipts}:
            raise DataError("Extracted archive identity differs")
        verify_files(destination, index["files"])
        return destination, receipts, index["files"]
    stage = stage_directory(destination)
    for receipt in receipts:
        archive = root / "raw" / dataset / receipt["asset"] / receipt["filename"]
        safe_extract(archive, stage / receipt["asset"])
    # UCI's current download wraps the original HAR archive in an outer ZIP.
    nested = list((stage / "main").rglob("UCI HAR Dataset.zip")) if dataset == "har" else []
    if len(nested) > 1:
        raise DataError("Ambiguous nested HAR archive")
    if nested:
        safe_extract(nested[0], stage / "har_original")
    files = {
        p.relative_to(stage).as_posix(): file_sha256(p)
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    write_json(
        stage / "extraction.json",
        {
            "raw_sha256": {r["asset"]: r["sha256"] for r in receipts},
            "files": files,
        },
    )
    publish_directory(stage, destination)
    return destination, receipts, files
