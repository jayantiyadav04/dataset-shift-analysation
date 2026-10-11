"""Immutable ingestion artifacts. This module is not a training data loader."""

import json
from collections import Counter
from pathlib import Path

import numpy as np

from shiftlab.data.acquisition import SOURCES, extracted_inputs
from shiftlab.data.parsers import (
    CLASS_MAPS,
    FEATURE_COUNTS,
    PARSER_VERSION,
    ParsedDataset,
    clean_duplicates,
    parse_amazon,
    parse_gas,
    parse_har,
)
from shiftlab.data.storage import (
    DataError,
    publish_directory,
    read_json,
    seal,
    stage_directory,
    verify_files,
    verify_seal,
    write_json,
)
from shiftlab.experiments.provenance import content_hash, file_sha256

PARSERS = {"gas": parse_gas, "har": parse_har, "amazon": parse_amazon}
GROUP_POLICIES = {
    "gas": "Exact float64 feature groups, no rounded deduplication; source precedes target",
    "har": "Whole official participants; identical features across people quarantined",
    "amazon": "Canonical body for role separation; product groups for source CV when feasible",
}
TERMS = {
    "gas": (
        "UCI displays CC BY 4.0 and older research-only wording. "
        "Retain locally; distribute acquisition instructions."
    ),
    "har": (
        "UCI displays CC BY 4.0; cite the dataset and original paper. Raw archives remain local."
    ),
    "amazon": (
        "Author requests citation of Blitzer et al. (ACL 2007); "
        "no explicit redistribution licence on the linked page. Raw reviews remain local."
    ),
}


def read_jsonl(path: Path) -> list:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def write_jsonl(path: Path, rows) -> None:
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(
                json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
            )


def _source_counts(parsed: ParsedDataset) -> dict:
    allowed = {
        "gas": {"b01"},
        "har": {"train"},
        "amazon": {"books", "dvd", "electronics", "kitchen"},
    }
    result = {}
    for domain in sorted(allowed[parsed.dataset]):
        result[domain] = dict(
            sorted(
                Counter(
                    str(r.label)
                    for r in parsed.records
                    if r.domain == domain and r.label is not None
                ).items()
            )
        )
    return result


def _card(manifest: dict) -> str:
    lines = [
        f"# {manifest['dataset_id']} retained-count data card",
        "",
        "Generated from archived inputs. No model performance has been measured.",
        "",
        f"Parser: `{manifest['parser_version']}`. Schema: {manifest['schema_version']}.",
        f"Raw rows: **{manifest['original_sample_count']}**. "
        f"Retained: **{manifest['retained_sample_count']}**.",
        f"Excluded: **{manifest['excluded_sample_count']}**. "
        f"Feature fields: **{manifest['feature_count']}**.",
        f"Processed SHA-256: `{manifest['processed_sha256']}`.",
        "",
        "## Acquisition",
        "",
    ]
    for receipt in manifest["archives"]:
        lines.extend(
            [
                f"- {receipt['asset']}: [{receipt['filename']}]({receipt['source_url']}); "
                f"accessed {receipt['accessed_at']}; SHA-256 `{receipt['sha256']}`.",
            ]
        )
    lines.extend(
        [
            "",
            manifest["terms"],
            "",
            "## Counts and permissions",
            "",
            "Original file/domain counts: "
            f"`{json.dumps(manifest['original_counts'], sort_keys=True)}`.",
            "Retained domain/pool counts: "
            f"`{json.dumps(manifest['domain_counts'], sort_keys=True)}`.",
            f"Exclusions by reason: `{json.dumps(manifest['exclusion_counts'], sort_keys=True)}`.",
            f"Class mapping: `{json.dumps(manifest['class_mapping'], sort_keys=True)}`.",
            "Allowed source label counts: "
            f"`{json.dumps(manifest['source_label_counts'], sort_keys=True)}`.",
            "Target label profiles are withheld until evaluation. "
            "Amazon categories each serve as source in separate tasks.",
            "Missing numerical data: validation fails; no imputation or rounded deduplication.",
            "Amazon model inputs contain review bodies only; "
            "rating and reviewer fields are excluded.",
            "",
            "## Grouping and preprocessing",
            "",
            manifest["group_policy"],
            "Cross-domain exact duplicates are removed from target pools per task; "
            "see split exclusions.",
            "All scalers, TF-IDF vocabulary/IDF and SVD fit only "
            "on the current source training fold.",
            "Text uses 20,000 maximum unigram/bigram features, min_df=2, sublinear TF, "
            "no stopword list, and at most 128 SVD components followed by scaling.",
            "Numerical features retain the supplied 128 (gas) or 561 (HAR) columns.",
            "HAR windows are dependent within participants; "
            "gas may retain within-batch temporal dependence.",
            "Amazon product identifiers are preserved separately; "
            "body groups determine duplicate separation.",
            "",
            f"Parser notes: `{json.dumps(manifest['parser_notes'], sort_keys=True)}`.",
            "",
            "Exact IDs, groups and role file hashes are in "
            "data/splits/<protocol>/<task>/manifest.json.",
            "Run `shiftlab data audit --all` and "
            "`shiftlab splits audit --config configs/study.yaml` for validation.",
        ]
    )
    return "\n".join(lines) + "\n"


def prepare(root: Path, dataset: str) -> dict:
    if dataset not in PARSERS:
        raise DataError(f"Unknown dataset: {dataset}")
    destination = root / "processed" / dataset
    if destination.exists():
        return audit_dataset(root, dataset)
    extracted, receipts, extracted_hashes = extracted_inputs(root, dataset)
    parsed = clean_duplicates(PARSERS[dataset](extracted))
    stage = stage_directory(destination)
    (stage / "private").mkdir()
    records = parsed.records
    write_jsonl(stage / "metadata.jsonl", (r.metadata() for r in records))
    write_jsonl(stage / "exclusions.jsonl", parsed.exclusions)
    labels = np.asarray([r.label if r.label is not None else -1 for r in records], dtype=np.int64)
    np.save(stage / "private" / "labels.npy", labels, allow_pickle=False)
    if dataset == "amazon":
        write_jsonl(stage / "bodies.jsonl", (r.values for r in records))
    else:
        np.savez_compressed(
            stage / "features.npz", X=np.asarray([r.values for r in records], dtype=np.float64)
        )
    checksums = {
        p.relative_to(stage).as_posix(): file_sha256(p)
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    original = sum(parsed.original_counts.values())
    if original != len(records) + len(parsed.exclusions):
        raise DataError("Retained and excluded counts do not reconcile")
    manifest = seal(
        {
            "schema_version": 1,
            "dataset_id": dataset,
            "source_url": SOURCES[dataset],
            "accessed_at": {r["asset"]: r["accessed_at"] for r in receipts},
            "raw_sha256": {r["asset"]: r["sha256"] for r in receipts},
            "archives": receipts,
            "extracted_sha256": extracted_hashes,
            "parser_version": PARSER_VERSION,
            "original_sample_count": original,
            "original_counts": parsed.original_counts,
            "retained_sample_count": len(records),
            "excluded_sample_count": len(parsed.exclusions),
            "feature_count": FEATURE_COUNTS[dataset],
            "feature_kind": "text_body" if dataset == "amazon" else "numeric",
            "class_mapping": CLASS_MAPS[dataset],
            "domain_counts": dict(sorted(Counter(f"{r.domain}/{r.pool}" for r in records).items())),
            "source_label_counts": _source_counts(parsed),
            "group_policy": GROUP_POLICIES[dataset],
            "excluded_records_path": "exclusions.jsonl",
            "exclusion_counts": dict(Counter(e["reason"] for e in parsed.exclusions)),
            "parser_notes": parsed.notes,
            "terms": TERMS[dataset],
            "checksums": checksums,
            "processed_sha256": content_hash(checksums),
        }
    )
    write_json(stage / "manifest.json", manifest)
    (stage / "DATA_CARD.md").write_text(_card(manifest), encoding="utf-8", newline="\n")
    publish_directory(stage, destination)
    return audit_dataset(root, dataset)


def load_prepared(root: Path, dataset: str) -> tuple[dict, list[dict], np.ndarray | list[str]]:
    """Preparation/audit use only; training must use the role-isolated loader."""
    directory = root / "processed" / dataset
    manifest = read_json(directory / "manifest.json")
    verify_seal(manifest)
    verify_files(directory, manifest["checksums"])
    if manifest["parser_version"] != PARSER_VERSION or manifest["processed_sha256"] != content_hash(
        manifest["checksums"]
    ):
        raise DataError("Processed schema or content identity differs")
    rows = read_jsonl(directory / "metadata.jsonl")
    if dataset == "amazon":
        values = read_jsonl(directory / "bodies.jsonl")
    else:
        with np.load(directory / "features.npz", allow_pickle=False) as data:
            values = data["X"]
    return manifest, rows, values


def audit_dataset(root: Path, dataset: str) -> dict:
    manifest, rows, values = load_prepared(root, dataset)
    directory = root / "processed" / dataset
    n = manifest["retained_sample_count"]
    if (
        manifest["schema_version"] != 1
        or manifest["dataset_id"] != dataset
        or manifest["feature_count"] != FEATURE_COUNTS[dataset]
    ):
        raise DataError("Data manifest schema mismatch")
    if len(rows) != n or len(values) != n or len({r["sample_id"] for r in rows}) != n:
        raise DataError("Data IDs and feature rows are not aligned")
    if dataset == "amazon":
        if any(not isinstance(body, str) or not body.strip() for body in values):
            raise DataError("Empty or invalid review body")
    elif values.shape != (n, FEATURE_COUNTS[dataset]) or not np.isfinite(values).all():
        raise DataError("Numerical feature schema mismatch or non-finite value")
    labels = np.load(directory / "private" / "labels.npy", allow_pickle=False)
    if labels.shape != (n,) or labels.dtype != np.int64:
        raise DataError("Task label store schema mismatch")
    for row, label in zip(rows, labels, strict=True):
        if row["pool"] == "unlabelled":
            if label != -1:
                raise DataError("Unlabelled review retained a task label")
        elif int(label) not in CLASS_MAPS[dataset]:
            raise DataError("Inconsistent canonical label mapping")
    for receipt in manifest["archives"]:
        verify_files(
            root / "raw" / dataset / receipt["asset"], {receipt["filename"]: receipt["sha256"]}
        )
    verify_files(root / "raw" / dataset / "extracted", manifest["extracted_sha256"])
    if (directory / "DATA_CARD.md").read_text(encoding="utf-8") != _card(manifest):
        raise DataError("Data card differs from the saved evidence")
    return manifest
