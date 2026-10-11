"""Guarded real importance-weighting gate; no target task evaluation."""

import argparse
import builtins
import csv
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.special import expit

from shiftlab.data.storage import read_json, write_json
from shiftlab.experiments.config import load_config
from shiftlab.experiments.provenance import canonical_json, code_snapshot, content_hash, file_sha256
from shiftlab.experiments.runner import train_methods
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.models.estimators import fit_estimator


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/importance.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Choose a new evidence filename; never replace a completed gate")
    root, data = Path.cwd(), Path("data").resolve()
    config = load_config(args.config)
    store = LocalArtifactStore(root / config.execution.output_root)
    prior = read_json(root / "reports/source_baselines/verification.json")
    original_files = {
        p: (file_sha256(p), p.stat().st_mtime_ns)
        for row in prior["result"]["runs"]
        for p in store.directory("runs", row["run_id"]).rglob("*")
        if p.is_file()
    }
    for row in prior["result"]["runs"]:
        store.verify("runs", row["run_id"])
    prior_splits = read_json(root / "docs/phase_02_final_verification.json")["split_audit"]["tasks"]
    assert {
        p.parent.name: read_json(p)["manifest_hash"]
        for p in (data / "splits/1.0").glob("*/manifest.json")
    } == {row["task_id"]: row["manifest_hash"] for row in prior_splits}
    path_open, builtin_open = Path.open, builtins.open
    delivered = []

    def guard(path):
        if isinstance(path, (str, Path)):
            resolved = Path(path).resolve()
            if resolved.is_relative_to(data) and {"evaluation", "private"} & set(resolved.parts):
                raise AssertionError(f"Fitting opened held-out/global label data: {resolved}")

    def guarded_path(path, *arguments, **kwargs):
        guard(path)
        return path_open(path, *arguments, **kwargs)

    def guarded_builtin(path, *arguments, **kwargs):
        guard(path)
        return builtin_open(path, *arguments, **kwargs)

    def fitting_spy(estimator, X, y, *, sample_weight=None):
        if sample_weight is not None:
            delivered.append(
                {
                    "model_class": type(estimator).__name__,
                    "weight_hash": content_hash(sample_weight.tolist()),
                    "source_count": len(sample_weight),
                }
            )
        return fit_estimator(estimator, X, y, sample_weight=sample_weight)

    started = perf_counter()
    with (
        patch.object(Path, "open", guarded_path),
        patch("builtins.open", guarded_builtin),
        patch("shiftlab.experiments.runner.fit_estimator", fitting_spy),
    ):
        result = train_methods(
            config,
            root=root,
            data_root=data,
            progress=lambda event: print(canonical_json(event), flush=True),
        )
        assert result["fitted_records"] == result["expected_fits"] == 21
        weighted = [row for row in result["runs"] if row["method"] == "importance_weighting"]
        assert len(weighted) == 9
        if result["new_fits"]:
            assert len(delivered) == 9
            assert {row["model_class"] for row in delivered} == {
                "LogisticRegression",
                "LinearSVC",
                "RandomForestClassifier",
            }
        checks, artifact_summary = [], []
        for cache_id in sorted({row["importance_cache_id"] for row in weighted}):
            manifest = store.verify("importance_cache", cache_id)
            directory = store.directory("importance_cache", cache_id)
            diagnostics = read_json(directory / "diagnostics.json")
            with np.load(directory / "adaptation_state.npz", allow_pickle=False) as arrays:
                values = {name: arrays[name] for name in arrays.files}
            ids = tuple(values["source_ids"].tolist())
            positions = {sid: i for i, sid in enumerate(ids)}
            counts, nt = values["source_fit_counts"], manifest["target_adapt_count"]
            parameters = manifest["canonical_key"]["parameters"]
            q = np.clip(
                values["probabilities"],
                parameters["probability_epsilon"],
                1 - parameters["probability_epsilon"],
            )
            ratios = q / (1 - q) * counts / nt
            clipped = np.clip(
                ratios, parameters["weight_clip_lower"], parameters["weight_clip_upper"]
            )
            weights = clipped / clipped.mean()
            ess = float(sum(weights) ** 2 / sum(float(w) ** 2 for w in weights))
            np.testing.assert_allclose(values["raw_weights"], ratios, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(values["clipped_weights"], clipped, atol=1e-12, rtol=1e-12)
            np.testing.assert_allclose(values["weights"], weights, atol=1e-12, rtol=1e-12)
            assert np.isfinite(weights).all() and np.all(weights > 0)
            assert abs(weights.mean() - 1) < 1e-12
            assert abs(diagnostics["ess"] - ess) < 1e-9
            source_directory = store.directory("source_cache", manifest["source_cv_id"])
            X = np.load(source_directory / "source_features.npy", allow_pickle=False)
            seen = []
            for number in range(5):
                fold = read_json(directory / f"domain_fold_{number}.json")
                held = np.array([positions[sid] for sid in fold["source_heldout_ids"]])
                assert not set(fold["source_fit_groups"]) & set(fold["source_heldout_groups"])
                assert not set(fold["source_fit_ids"]) & set(fold["source_heldout_ids"])
                assert len(fold["source_fit_ids"]) + len(held) == len(ids)
                assert fold["target_adapt_ids"] == values["target_adapt_ids"].tolist()
                assert fold["target_adapt_fit_count"] == nt
                assert np.all(counts[held] == len(fold["source_fit_ids"]))
                np.testing.assert_array_equal(values["fold_assignment"][held], number)
                reconstructed = expit(
                    X[held] @ np.asarray(fold["coefficient"])[0] + fold["intercept"][0]
                )
                np.testing.assert_allclose(
                    values["probabilities"][held], reconstructed, atol=1e-10, rtol=1e-10
                )
                seen.extend(held.tolist())
            assert sorted(seen) == list(range(len(ids)))
            checks.append(
                {
                    "importance_cache_id": cache_id,
                    "split_hash": manifest["canonical_key"]["split_hash"],
                    "source_count": len(ids),
                    "target_adapt_count": nt,
                    "weight_hash": content_hash(values["weights"].tolist()),
                    "ess": ess,
                    "ess_fraction": ess / len(ids),
                    "min_weight": float(weights.min()),
                    "max_weight": float(weights.max()),
                    "weight_clipping_fraction": diagnostics["weight_clipping_fraction"],
                    "warnings": diagnostics["warnings"],
                    "five_folds_independently_verified": True,
                }
            )
        by_key = {
            (row["task_id"], row["model"], row["seed"], row["method"]): row
            for row in result["runs"]
        }
        for row in result["runs"]:
            manifest = store.verify("runs", row["run_id"])
            assert manifest["status"] == "fitted" and not manifest["evaluation_performed"]
            directory = store.directory("runs", row["run_id"])
            assert (
                not (directory / "predictions.csv").exists()
                and not (directory / "metrics.json").exists()
            )
            diagnostic = read_json(directory / "diagnostics.json")
            if row["method"] == "importance_weighting":
                baseline = by_key[(row["task_id"], row["model"], row["seed"], "source_only")]
                base_key = store.verify("runs", baseline["run_id"])["canonical_key"]
                assert diagnostic["sample_weight_passed"]
                assert row["source_cv_id"] == baseline["source_cv_id"]
                for field in (
                    "dataset_hash",
                    "task_id",
                    "split_hash",
                    "selected_source_parameter_hash",
                    "preprocessing_hash",
                    "seed",
                    "config_hash",
                    "code_snapshot_hash",
                ):
                    assert base_key[field] == manifest["canonical_key"][field]
                if result["new_fits"]:
                    assert any(
                        receipt["weight_hash"] == diagnostic["sample_weight_hash"]
                        for receipt in delivered
                    )
            artifact_summary.append(
                {
                    **row,
                    "manifest_hash": manifest["manifest_hash"],
                    "timing": manifest["timing"],
                    "warnings": manifest["warnings"],
                    "selected_parameters": manifest["selected_params"],
                    "path": directory.relative_to(root).as_posix(),
                }
            )
        files = {
            p: (file_sha256(p), p.stat().st_mtime_ns)
            for kind in ("runs", "source_cache", "importance_cache")
            for p in (store.root / kind).rglob("*")
            if p.is_file()
        }
        resumed = train_methods(config, root=root, data_root=data)
        assert resumed["new_fits"] == 0 and resumed["resumed_fits"] == 21
        assert files == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in files}
        assert original_files == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in original_files}
        evidence = {
            "evidence_kind": "real_importance_fit_only",
            "observed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "config": args.config.as_posix(),
            "result": result,
            "resume": resumed,
            "weight_checks": checks,
            "weight_delivery_receipts": delivered,
            "artifacts": artifact_summary,
            "target_evaluation_opened": False,
            "prior_baseline_files_unchanged": len(original_files),
            "prior_22_split_hashes_match": True,
            "code_snapshot": code_snapshot(root),
            "code_snapshot_hash": content_hash(code_snapshot(root)),
        }
        report = root / "reports/importance_weighting"
        report.mkdir(parents=True, exist_ok=True)
        with (report / "weight_diagnostics.csv").open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)
        write_json(report / "verification.json", evidence)
        write_json(args.out, evidence)
    print(
        canonical_json(
            {
                "status": "importance_gate_passed",
                "fitted": 21,
                "weighted_fits": 9,
                "target_evaluations": 0,
                "elapsed_seconds": evidence["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
