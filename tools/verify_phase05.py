"""Guarded CORAL real-data fitting and independent Schur-root verification."""

import argparse
import builtins
import csv
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np
from scipy.linalg import sqrtm
from threadpoolctl import threadpool_limits

from shiftlab.data.storage import read_json, write_json
from shiftlab.data.training import load_training
from shiftlab.experiments.alignment import load_coral_state
from shiftlab.experiments.config import load_config
from shiftlab.experiments.provenance import canonical_json, code_snapshot, content_hash, file_sha256
from shiftlab.experiments.runner import train_methods
from shiftlab.experiments.storage import LocalArtifactStore
from shiftlab.models.estimators import fit_estimator


def file_snapshot(root: Path) -> dict:
    return {
        p: (file_sha256(p), p.stat().st_mtime_ns)
        for kind in ("runs", "source_cache", "importance_cache", "coral_cache", "trust")
        for p in (root / kind).rglob("*")
        if p.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/coral.yaml"))
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root, data = Path.cwd(), Path("data").resolve()
    report = root / "reports/coral_alignment"
    if args.out.exists() or report.exists():
        raise ValueError("Choose new evidence paths; never replace a completed gate")
    config = load_config(args.config)
    store = LocalArtifactStore(root / config.execution.output_root)
    prior = read_json(root / "reports/importance_weighting/verification.json")
    for row in prior["result"]["runs"]:
        store.verify("runs", row["run_id"])
    original_files = file_snapshot(store.root)
    prior_splits = read_json(root / "docs/phase_02_final_verification.json")["split_audit"]["tasks"]
    assert {
        p.parent.name: read_json(p)["manifest_hash"]
        for p in (data / "splits/1.0").glob("*/manifest.json")
    } == {row["task_id"]: row["manifest_hash"] for row in prior_splits}
    path_open, builtin_open = Path.open, builtins.open
    deliveries = []

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
        deliveries.append(
            {
                "model_class": type(estimator).__name__,
                "source_features_hash": content_hash(X.tolist()),
                "sample_weight_passed": sample_weight is not None,
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
        assert result["fitted_records"] == result["expected_fits"] == 30
        assert len(deliveries) == result["new_fits"]
        coral_rows = [row for row in result["runs"] if row["method"] == "coral"]
        assert len(coral_rows) == 9
        checks = []
        for task in config.selected_tasks():
            rows = [row for row in coral_rows if row["task_id"] == task]
            cache_ids = {row["coral_cache_id"] for row in rows}
            assert len(cache_ids) == 1
            cache_id = cache_ids.pop()
            manifest = store.verify("coral_cache", cache_id)
            directory = store.directory("coral_cache", cache_id)
            view = load_training(data / "splits" / config.protocol_version / task)
            source_id = manifest["source_cv_id"]
            pre = store.load_joblib("source_cache", source_id, "preprocessing.joblib")
            X_source = np.load(
                store.directory("source_cache", source_id) / "source_features.npy",
                allow_pickle=False,
            )
            X_adapt = pre.transform(view.target_adapt)
            state = load_coral_state(
                directory / "adaptation_state.npz",
                preprocessing_hash=pre.preprocessing_hash_,
                source_ids=view.source.ids,
                target_adapt_ids=view.target_adapt.ids,
            )
            with threadpool_limits(limits=1):
                Cs, Ct = np.cov(X_source, rowvar=False), np.cov(X_adapt, rowvar=False)
                np.testing.assert_allclose(state.source_covariance, Cs, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(state.target_covariance, Ct, atol=1e-10, rtol=1e-10)
                np.testing.assert_allclose(state.source_mean, X_source.mean(axis=0), atol=1e-12)
                np.testing.assert_allclose(state.target_mean, X_adapt.mean(axis=0), atol=1e-12)
                ridge = config.coral.ridge * np.eye(X_source.shape[1])
                # Independent Schur-based roots and solve, rather than the engine's eigh.
                expected = np.linalg.solve(sqrtm(Cs + ridge), sqrtm(Ct + ridge))
                np.testing.assert_allclose(state.alignment, expected, atol=1e-8, rtol=1e-8)
                aligned = np.load(directory / "aligned_source_features.npy", allow_pickle=False)
                np.testing.assert_allclose(
                    aligned,
                    (X_source - X_source.mean(axis=0)) @ expected + X_adapt.mean(axis=0),
                    atol=1e-8,
                    rtol=1e-8,
                )
                np.testing.assert_allclose(aligned.mean(axis=0), state.target_mean, atol=1e-10)
                after_covariance = np.cov(aligned, rowvar=False)
                denominator = max(np.linalg.norm(Ct, "fro"), 1e-12)
                before = float(np.linalg.norm(Cs - Ct, "fro") / denominator)
                after = float(np.linalg.norm(after_covariance - Ct, "fro") / denominator)
                regularised_residual = float(
                    np.linalg.norm(
                        state.alignment.T @ (Cs + ridge) @ state.alignment - (Ct + ridge), "fro"
                    )
                    / max(np.linalg.norm(Ct + ridge, "fro"), 1e-12)
                )
                np.testing.assert_allclose(
                    after_covariance - Ct,
                    config.coral.ridge
                    * (np.eye(Cs.shape[0]) - state.alignment.T @ state.alignment),
                    atol=1e-8,
                    rtol=1e-8,
                )
            diagnostics = state.diagnostics
            assert abs(before - diagnostics["covariance_mismatch_before"]) < 1e-9
            assert abs(after - diagnostics["covariance_mismatch_after"]) < 1e-9
            assert regularised_residual < 1e-8
            assert np.isfinite(aligned).all()
            assert manifest["aligned_source_hash"] == content_hash(aligned.tolist())
            receipts = [
                delivery
                for delivery in deliveries
                if delivery["source_features_hash"] == manifest["aligned_source_hash"]
            ]
            if any(not row["resumed"] for row in rows):
                assert {r["model_class"] for r in receipts} == {
                    "LogisticRegression",
                    "LinearSVC",
                    "RandomForestClassifier",
                }
                assert len(receipts) == 3 and all(not r["sample_weight_passed"] for r in receipts)
            checks.append(
                {
                    "task_id": task,
                    "coral_cache_id": cache_id,
                    "source_cv_id": source_id,
                    "preprocessing_hash": pre.preprocessing_hash_,
                    "source_count": len(X_source),
                    "target_adapt_count": len(X_adapt),
                    "feature_dimension": X_source.shape[1],
                    "ridge": config.coral.ridge,
                    "eigenvalue_floor": config.coral.eigenvalue_floor,
                    "source_covariance_rank": diagnostics["source_covariance_rank"],
                    "target_covariance_rank": diagnostics["target_covariance_rank"],
                    "covariance_mismatch_before": before,
                    "covariance_mismatch_after": after,
                    "regularised_covariance_residual": regularised_residual,
                    "independent_alignment_relative_error": float(
                        np.linalg.norm(state.alignment - expected, "fro")
                        / max(np.linalg.norm(expected, "fro"), 1e-12)
                    ),
                    "aligned_source_hash": manifest["aligned_source_hash"],
                    "state_sha256": file_sha256(directory / "adaptation_state.npz"),
                    "adaptation_fit_seconds": manifest["adaptation_fit_seconds"],
                    "warnings": diagnostics["warnings"],
                }
            )
        by_key = {(row["task_id"], row["model"], row["method"]): row for row in result["runs"]}
        artifacts = []
        for row in result["runs"]:
            record = store.verify("runs", row["run_id"])
            directory = store.directory("runs", row["run_id"])
            assert record["status"] == "fitted" and not record["evaluation_performed"]
            assert (
                not (directory / "predictions.csv").exists()
                and not (directory / "metrics.json").exists()
            )
            if row["method"] != "source_only":
                baseline = by_key[row["task_id"], row["model"], "source_only"]
                base_key = store.verify("runs", baseline["run_id"])["canonical_key"]
                assert row["source_cv_id"] == baseline["source_cv_id"]
                for field in (
                    "dataset_hash",
                    "split_hash",
                    "task_id",
                    "seed",
                    "selected_source_parameter_hash",
                    "preprocessing_hash",
                    "config_hash",
                    "code_snapshot_hash",
                ):
                    assert base_key[field] == record["canonical_key"][field]
            artifacts.append(
                {
                    **row,
                    "manifest_hash": record["manifest_hash"],
                    "timing": record["timing"],
                    "warnings": record["warnings"],
                    "selected_parameters": record["selected_params"],
                    "path": directory.relative_to(root).as_posix(),
                }
            )
        files = file_snapshot(store.root)
        resumed = train_methods(config, root=root, data_root=data)
        assert resumed["new_fits"] == 0 and resumed["resumed_fits"] == 30
        assert files == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in files}
        assert original_files == {p: (file_sha256(p), p.stat().st_mtime_ns) for p in original_files}
        evidence = {
            "evidence_kind": "real_coral_fit_only",
            "observed_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": perf_counter() - started,
            "config": args.config.as_posix(),
            "result": result,
            "resume": resumed,
            "coral_checks": checks,
            "source_feature_delivery_receipts": deliveries,
            "artifacts": artifacts,
            "target_evaluation_opened": False,
            "prior_artifact_files_unchanged": len(original_files),
            "prior_22_split_hashes_match": True,
            "code_snapshot": code_snapshot(root),
            "code_snapshot_hash": content_hash(code_snapshot(root)),
        }
        report.mkdir(parents=True)
        with (report / "covariance_diagnostics.csv").open(
            "x", encoding="utf-8", newline=""
        ) as stream:
            writer = csv.DictWriter(stream, fieldnames=list(checks[0]))
            writer.writeheader()
            writer.writerows(checks)
        write_json(report / "verification.json", evidence)
        write_json(args.out, evidence)
    print(
        canonical_json(
            {
                "status": "coral_gate_passed",
                "fitted": 30,
                "coral_fits": 9,
                "target_evaluations": 0,
                "elapsed_seconds": evidence["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
