"""Deterministic role assignment and immutable, physically separated role files."""

from pathlib import Path

import numpy as np

from shiftlab.data.parsers import DOMAINS, FEATURE_COUNTS, normalized_body, numeric_signature
from shiftlab.data.prepare import audit_dataset, load_prepared, read_jsonl, write_jsonl
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
from shiftlab.experiments.config import StudyConfig
from shiftlab.experiments.provenance import file_sha256

ROLES = ("source", "target_adapt", "target_eval")
DEDUPLICATION_POLICY = (
    "Global conflicting duplicate labels quarantined; exact float64 / NFKC-whitespace body "
    "duplicates retained once per domain; labelled precedes unlabelled; source precedes target. "
    "HAR participant overlap forbidden; Amazon product overlap is permitted "
    "but body overlap is not."
)


def assign_roles(
    dataset: str, task: str, rows: list[dict], seed: int = 2026
) -> tuple[dict, list[dict], str, str]:
    if seed != 2026:
        raise DataError("Split seed is frozen at 2026")
    exclusions = []
    if dataset == "gas":
        valid = {f"gas_b01_b{batch:02d}" for batch in range(2, 11)}
        if task not in valid:
            raise DataError("Unknown gas transfer")
        source_domain, target_domain = "b01", task[-3:]
    elif dataset == "har":
        if task != "har_official_group_transfer":
            raise DataError("Unknown HAR transfer")
        source_domain, target_domain = "train", "test"
    else:
        pairs = {f"amazon_{s}_{t}": (s, t) for s in DOMAINS for t in DOMAINS if s != t}
        if task not in pairs:
            raise DataError("Unknown Amazon transfer")
        source_domain, target_domain = pairs[task]
    source = [r for r in rows if r["domain"] == source_domain and r["pool"] != "unlabelled"]
    source_bodies = {r["duplicate_id"] for r in source}
    target = []
    for row in rows:
        if row["domain"] != target_domain:
            continue
        if row["duplicate_id"] in source_bodies:
            exclusions.append(
                {
                    "sample_id": row["sample_id"],
                    "reason": "source_target_duplicate",
                    "duplicate_id": row["duplicate_id"],
                }
            )
        else:
            target.append(row)
    if dataset == "gas":
        groups = sorted({r["duplicate_id"] for r in target})
        shuffled = np.random.default_rng(seed).permutation(len(groups))
        adapt_groups = {groups[i] for i in shuffled[: len(groups) // 2]}
        adapt = [r for r in target if r["duplicate_id"] in adapt_groups]
        evaluate = [r for r in target if r["duplicate_id"] not in adapt_groups]
    elif dataset == "har":
        groups = sorted({r["group_id"] for r in target})
        if len(groups) != 9:
            raise DataError("HAR requires nine complete target participants after cleaning")
        adapt_groups = set(groups[:4])
        adapt = [r for r in target if r["group_id"] in adapt_groups]
        evaluate = [r for r in target if r["group_id"] not in adapt_groups]
    else:
        adapt = [r for r in target if r["pool"] == "unlabelled"]
        evaluate = [r for r in target if r["pool"] == "labelled"]
    roles = dict(zip(ROLES, (source, adapt, evaluate), strict=True))
    if any(len(records) < 2 for records in roles.values()):
        raise DataError(
            "Source, adaptation and evaluation pools must each contain at least two rows"
        )
    return roles, exclusions, source_domain, target_domain


def validate_split(manifest: dict) -> None:
    verify_seal(manifest)
    if (
        type(manifest.get("schema_version")) is not int
        or manifest["schema_version"] != 1
        or type(manifest.get("split_seed")) is not int
        or manifest["split_seed"] != 2026
    ):
        raise DataError("Split schema or seed mismatch")
    for field in ("source_domain", "target_domain"):
        if not isinstance(manifest.get(field), str) or not manifest[field]:
            raise DataError("Missing split domain identity")
    if manifest["source_domain"] == manifest["target_domain"]:
        raise DataError("Self transfer is forbidden")
    seen_ids, seen_groups = set(), set()
    for role in ROLES:
        ids, groups = manifest.get(f"{role}_ids"), manifest.get(f"{role}_group_ids")
        if not isinstance(ids, list) or not isinstance(groups, list):
            raise DataError("Split IDs/groups must be explicit lists")
        if any(not isinstance(value, str) or not value for value in ids + groups):
            raise DataError("Invalid sample/group identity")
        if len(ids) < 2 or len(ids) != len(set(ids)) or len(groups) != len(ids):
            raise DataError(f"Empty, repeated or unaligned {role} IDs/groups")
        if seen_ids.intersection(ids) or seen_groups.intersection(groups):
            raise DataError("Leakage: sample or prohibited group overlaps across roles")
        seen_ids.update(ids)
        seen_groups.update(groups)


def _write_features(path: Path, values, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(values, np.ndarray):
        np.savez_compressed(path, X=values, ids=np.asarray(ids))
    else:
        write_jsonl(
            path, ({"sample_id": sid, "body": body} for sid, body in zip(ids, values, strict=True))
        )


def read_role_features(
    directory: Path, artifact: str
) -> tuple[tuple[str, ...], np.ndarray | tuple[str, ...]]:
    from shiftlab.data.storage import safe_child

    path = safe_child(directory, artifact)
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"X", "ids"}:
                raise DataError("Unexpected fields in numerical feature artifact")
            return tuple(arrays["ids"].tolist()), arrays["X"]
    rows = read_jsonl(path)
    if any(set(row) != {"sample_id", "body"} for row in rows):
        raise DataError("Only review bodies may enter text features")
    return tuple(r["sample_id"] for r in rows), tuple(r["body"] for r in rows)


def _freeze_task(
    root: Path, config: StudyConfig, task: str, manifest: dict, rows: list[dict], values
) -> dict:
    dataset = manifest["dataset_id"]
    destination = root / "splits" / config.protocol_version / task
    roles, exclusions, source_domain, target_domain = assign_roles(
        dataset, task, rows, config.split_seed
    )
    if destination.exists():
        saved = audit_split(root, destination, (manifest, rows, values))
        if saved["dataset_hash"] != manifest["manifest_hash"]:
            raise DataError(
                "Existing split uses different data; do not overwrite a frozen protocol"
            )
        return saved
    index = {row["sample_id"]: i for i, row in enumerate(rows)}
    # Preparation is allowed to parse/store labels; none are passed to split assignment.
    labels = np.load(root / "processed" / dataset / "private" / "labels.npy", allow_pickle=False)
    stage = stage_directory(destination)
    artifacts, payload = {}, {}
    for role, records in roles.items():
        ids = [r["sample_id"] for r in records]
        positions = [index[sid] for sid in ids]
        payload[f"{role}_ids"] = ids
        payload[f"{role}_group_ids"] = [r["group_id"] for r in records]
        payload[f"{role}_product_ids"] = [r["product_id"] for r in records]
        extension = ".npz" if dataset != "amazon" else ".jsonl"
        area = "evaluation" if role == "target_eval" else "training"
        name = f"{area}/{role}{extension}"
        selected = (
            values[positions] if isinstance(values, np.ndarray) else [values[i] for i in positions]
        )
        _write_features(stage / name, selected, ids)
        artifacts[role] = name
        if role != "target_adapt":
            label_name = f"{area}/{role}_labels.npy"
            np.save(stage / label_name, labels[positions], allow_pickle=False)
            artifacts[f"{role}_labels"] = label_name
    write_jsonl(stage / "exclusions.jsonl", exclusions)
    checksums = {
        p.relative_to(stage).as_posix(): file_sha256(p)
        for p in sorted(stage.rglob("*"))
        if p.is_file()
    }
    saved = seal(
        {
            "schema_version": 1,
            "protocol_version": config.protocol_version,
            "dataset_id": dataset,
            "dataset_hash": manifest["manifest_hash"],
            "task_id": task,
            "source_domain": source_domain,
            "target_domain": target_domain,
            "split_seed": config.split_seed,
            "deduplication_policy": DEDUPLICATION_POLICY,
            "split_algorithm": "numpy-default_rng-permuted-sorted-groups-v1",
            "feature_count": manifest["feature_count"],
            "class_mapping": manifest["class_mapping"],
            "sample_counts": {role: len(records) for role, records in roles.items()},
            "target_adaptation_fraction": len(roles["target_adapt"])
            / (len(roles["target_adapt"]) + len(roles["target_eval"])),
            "excluded_target_count": len(exclusions),
            "exclusions_path": "exclusions.jsonl",
            "artifacts": artifacts,
            "checksums": checksums,
            **payload,
        }
    )
    validate_split(saved)
    write_json(stage / "manifest.json", saved)
    publish_directory(stage, destination)
    return saved


def freeze_splits(root: Path, config: StudyConfig) -> dict:
    outputs = []
    # Complete the acquisition/preparation audit before committing any split.
    for dataset in config.datasets:
        audit_dataset(root, dataset)
    for dataset in config.datasets:
        manifest, rows, values = load_prepared(root, dataset)
        for task in config.selected_tasks():
            if task.startswith(f"{dataset}_"):
                saved = _freeze_task(root, config, task, manifest, rows, values)
                outputs.append(
                    {
                        "task_id": task,
                        "manifest_hash": saved["manifest_hash"],
                        "sample_counts": saved["sample_counts"],
                    }
                )
    return {
        "status": "splits_frozen",
        "completed_tasks": len(outputs),
        "expected_tasks": len(config.selected_tasks()),
        "tasks": outputs,
    }


def audit_split(root: Path, directory: Path, prepared: tuple | None = None) -> dict:
    saved = read_json(directory / "manifest.json")
    validate_split(saved)
    dataset = saved["dataset_id"]
    manifest, rows, values = prepared or load_prepared(root, dataset)
    if saved["dataset_hash"] != manifest["manifest_hash"]:
        raise DataError("Split dataset hash differs")
    expected, exclusions, source_domain, target_domain = assign_roles(
        dataset, saved["task_id"], rows, saved["split_seed"]
    )
    if saved["source_domain"] != source_domain or saved["target_domain"] != target_domain:
        raise DataError("Split domain mapping differs")
    verify_files(directory, saved["checksums"])
    index = {row["sample_id"]: i for i, row in enumerate(rows)}
    seen_duplicates = set()
    import hashlib

    for role in ROLES:
        role_rows = expected[role]
        ids, features = read_role_features(directory, saved["artifacts"][role])
        if list(ids) != saved[f"{role}_ids"] or list(ids) != [r["sample_id"] for r in role_rows]:
            raise DataError("Split IDs differ from frozen assignment or feature rows")
        if saved[f"{role}_group_ids"] != [r["group_id"] for r in role_rows]:
            raise DataError("Split group identities differ from data provenance")
        duplicates = set()
        for sid, value in zip(ids, features, strict=True):
            original = values[index[sid]]
            if dataset == "amazon":
                if (
                    not isinstance(value, str)
                    or normalized_body(value) != value
                    or value != original
                ):
                    raise DataError("Review body schema/content mismatch")
                duplicates.add(hashlib.sha256(value.encode("utf-8")).hexdigest())
            else:
                if (
                    value.shape != (FEATURE_COUNTS[dataset],)
                    or not np.isfinite(value).all()
                    or not np.array_equal(value, original)
                ):
                    raise DataError("Numeric role schema/content mismatch")
                duplicates.add(numeric_signature(tuple(value)))
        if seen_duplicates & duplicates:
            raise DataError("Leakage: identical feature/body across roles")
        seen_duplicates.update(duplicates)
        if saved["sample_counts"][role] != len(ids):
            raise DataError("Saved sample count mismatch")
    if read_jsonl(directory / "exclusions.jsonl") != exclusions:
        raise DataError("Split exclusions differ from declared policy")
    return saved


def audit_splits(root: Path, config: StudyConfig) -> dict:
    results = []
    for dataset in config.datasets:
        prepared = load_prepared(root, dataset)
        for task in config.selected_tasks():
            if task.startswith(f"{dataset}_"):
                manifest = audit_split(
                    root, root / "splits" / config.protocol_version / task, prepared
                )
                results.append(
                    {
                        "task_id": task,
                        "manifest_hash": manifest["manifest_hash"],
                        "sample_counts": manifest["sample_counts"],
                    }
                )
    return {"status": "split_audit_passed", "completed_tasks": len(results), "tasks": results}
