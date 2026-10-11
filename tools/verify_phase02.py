"""Real-data Phase 2 gate evidence, without training or target evaluation."""

import argparse
import builtins
import json
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from shiftlab.data.prepare import audit_dataset
from shiftlab.data.splits import audit_splits
from shiftlab.data.storage import read_json, write_json
from shiftlab.data.training import load_training
from shiftlab.environment import environment_report
from shiftlab.experiments.config import load_config
from shiftlab.experiments.provenance import content_hash, file_sha256
from shiftlab.features.preprocessing import fit_source_fold, preprocess_training, source_folds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    code_paths = [Path("pyproject.toml"), Path("requirements-lock.txt")]
    for directory in ("src", "app", "tests", "tools", "configs"):
        code_paths.extend(
            path
            for path in Path(directory).rglob("*")
            if path.is_file() and path.suffix in {".py", ".yaml", ".toml", ".review", ".typed"}
        )
    code_snapshot = {path.as_posix(): file_sha256(path) for path in sorted(code_paths)}
    started = perf_counter()
    manifests = [audit_dataset(args.data_root, dataset) for dataset in config.datasets]
    splits = audit_splits(args.data_root, config)
    representatives, seen = [], set()
    for task in config.selected_tasks():
        directory = args.data_root / "splits" / config.protocol_version / task
        manifest = read_json(directory / "manifest.json")
        key = (manifest["dataset_id"], manifest["source_domain"])
        if key not in seen:
            representatives.append((directory, manifest))
            seen.add(key)
    data_root = args.data_root.resolve()
    path_open, builtin_open = Path.open, builtins.open

    def check_path(path) -> None:
        if not isinstance(path, (str, Path)):
            return
        resolved = Path(path).resolve()
        if resolved.is_relative_to(data_root) and ({"evaluation", "private"} & set(resolved.parts)):
            raise AssertionError(f"Held-out/global-label access attempted: {resolved}")

    def guarded_path_open(path, *arguments, **kwargs):
        check_path(path)
        return path_open(path, *arguments, **kwargs)

    def guarded_builtin_open(path, *arguments, **kwargs):
        check_path(path)
        return builtin_open(path, *arguments, **kwargs)

    checks = []
    with (
        patch.object(Path, "open", guarded_path_open),
        patch("builtins.open", guarded_builtin_open),
    ):
        for directory, manifest in representatives:
            view = load_training(directory)
            cv = []
            for seed in config.seeds:
                folds = source_folds(view, seed)
                cv.append(
                    {
                        "seed": seed,
                        "strategy": folds.strategy,
                        "group_policy": folds.note,
                        "fold_sizes": [
                            {"train": len(t), "validation": len(v)} for t, v in folds.indices
                        ],
                        "fold_hash": content_hash(
                            [
                                {
                                    "train_ids": [view.source.ids[i] for i in t],
                                    "validation_ids": [view.source.ids[i] for i in v],
                                }
                                for t, v in folds.indices
                            ]
                        ),
                    }
                )
            component_start = perf_counter()
            preprocessor, dense = preprocess_training(view, seed=config.seeds[0])
            folds = source_folds(view, config.seeds[0])
            train, valid = folds.indices[0]
            fold_pre, X_train, X_valid = fit_source_fold(view.source, train, valid, config.seeds[0])
            assert set(fold_pre.fit_ids_).isdisjoint(view.source.ids[i] for i in valid)
            assert np.isfinite(X_train).all() and np.isfinite(X_valid).all()
            checks.append(
                {
                    "dataset": manifest["dataset_id"],
                    "source_domain": manifest["source_domain"],
                    "task_id": manifest["task_id"],
                    "split_hash": view.split_hash,
                    "source_shape": list(dense.X_source.shape),
                    "adaptation_shape": list(dense.X_target_adapt.shape),
                    "fold_train_shape": list(X_train.shape),
                    "fold_validation_shape": list(X_valid.shape),
                    "preprocessing_sha256": preprocessor.preprocessing_hash_,
                    "fold_preprocessing_sha256": fold_pre.preprocessing_hash_,
                    "rank_note": preprocessor.rank_note_,
                    "source_cv": cv,
                    "preprocessing_check_seconds": perf_counter() - component_start,
                    "held_out_files_opened_by_training": False,
                }
            )
            print(
                f"Verified {manifest['dataset_id']}/{manifest['source_domain']}: "
                f"{dense.X_source.shape[1]} shared finite features",
                flush=True,
            )
    evidence = {
        "phase": 2,
        "created_at": datetime.now(UTC).isoformat(),
        "config_sha256": content_hash(config.model_dump(mode="json")),
        "dependency_lock_sha256": file_sha256(Path("requirements-lock.txt")),
        "code_snapshot_sha256": content_hash(code_snapshot),
        "code_snapshot": code_snapshot,
        "environment": environment_report(Path.cwd()),
        "dataset_audits": [
            {
                key: m[key]
                for key in (
                    "dataset_id",
                    "manifest_hash",
                    "raw_sha256",
                    "original_sample_count",
                    "retained_sample_count",
                    "excluded_sample_count",
                    "domain_counts",
                )
            }
            for m in manifests
        ],
        "split_audit": splits,
        "real_preprocessing_checks": checks,
        "elapsed_seconds": perf_counter() - started,
        "notice": (
            "Data preparation and source feature checks only; "
            "no target metrics or experimental findings."
        ),
    }
    if any(file_sha256(Path(name)) != digest for name, digest in code_snapshot.items()):
        raise RuntimeError("Code changed during verification; preserve this attempt and rerun")
    write_json(args.out, evidence)
    print(
        json.dumps(
            {
                "status": "phase02_real_gate_passed",
                "tasks": len(splits["tasks"]),
                "source_representations_checked": len(checks),
                "evidence": str(args.out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
