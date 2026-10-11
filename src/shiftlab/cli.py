"""Shared-engine configuration and data commands; explicit errors for future work."""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from shiftlab import __version__
from shiftlab.contracts import DatasetName
from shiftlab.experiments.config import ConfigError, load_config
from shiftlab.experiments.provenance import content_hash
from shiftlab.scope import FINAL_OBJECTIVE

NOT_IMPLEMENTED = 4


def _validate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return {
        "status": "valid_configuration",
        "study_name": config.study_name,
        "mode": config.mode,
        "default_stage": config.execution.default_stage,
        "config_sha256": content_hash(config.model_dump(mode="json")),
        "planned_counts": config.planned_counts(),
        "notice": "Configuration validation only. No data prepared or experiments executed.",
    }


def _config_argument(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, required=True, help="UTF-8 study YAML")


def _data_arguments(parser: argparse.ArgumentParser, *, config: bool = True) -> None:
    parser.add_argument(
        "--data-root", type=Path, default=Path("data"), help="Local immutable data root"
    )
    if config:
        parser.add_argument("--config", type=Path, default=Path("configs/study.yaml"))


def _data_command(args: argparse.Namespace) -> dict[str, Any]:
    from shiftlab.data.acquisition import SOURCES, fetch
    from shiftlab.data.prepare import audit_dataset, prepare

    config = load_config(args.config)
    datasets = list(config.datasets) if getattr(args, "all", False) else [args.dataset]
    if any(dataset not in config.datasets for dataset in datasets):
        raise ValueError("Selected dataset is excluded by the configuration")
    if args.action == "fetch":
        if args.books_unlabeled is not None and args.dataset != "amazon":
            raise ValueError("--books-unlabeled is only valid for Amazon")
        return {
            "status": "archives_available",
            "archives": [
                fetch(
                    args.data_root,
                    args.dataset,
                    asset=asset,
                    archive=args.archive if asset == "main" else args.books_unlabeled,
                )
                for asset in SOURCES[args.dataset]
            ],
        }
    operation = prepare if args.action == "prepare" else audit_dataset
    manifests = [operation(args.data_root, dataset) for dataset in datasets]
    return {
        "status": "data_prepared" if args.action == "prepare" else "data_audit_passed",
        "datasets": [
            {
                key: manifest[key]
                for key in (
                    "dataset_id",
                    "retained_sample_count",
                    "excluded_sample_count",
                    "manifest_hash",
                )
            }
            for manifest in manifests
        ],
    }


def _planned(parser: argparse.ArgumentParser, phase: int) -> None:
    parser.set_defaults(phase=phase)
    parser.description = (
        f"Planned command. Implemented in Phase {phase}; currently exits with code 4."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Domain Shift Workbench research engine")
    parser.add_argument("--version", action="version", version=f"shiftlab {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("scope", help="Print the exact frozen objective")
    commands.add_parser("doctor", help="Check actual versions and offline estimator compatibility")

    config = commands.add_parser(
        "config", help="Validate a configuration without executing a study"
    )
    config_commands = config.add_subparsers(dest="action", required=True)
    _config_argument(config_commands.add_parser("validate"))

    data = commands.add_parser("data", help="Dataset preparation (Phase 2)")
    data_commands = data.add_subparsers(dest="action", required=True)
    for action in ("fetch", "prepare"):
        child = data_commands.add_parser(action)
        child.add_argument("--dataset", choices=list(DatasetName), required=True)
        _data_arguments(child)
        if action == "fetch":
            child.add_argument(
                "--archive", type=Path, help="Import an official archive already downloaded"
            )
            child.add_argument(
                "--books-unlabeled",
                type=Path,
                help="Import the separate Amazon books unlabelled gzip",
            )
    audit = data_commands.add_parser("audit")
    audit_selection = audit.add_mutually_exclusive_group(required=True)
    audit_selection.add_argument("--all", action="store_true")
    audit_selection.add_argument("--dataset", choices=list(DatasetName))
    _data_arguments(audit)

    splits = commands.add_parser("splits", help="Freeze split identities (Phase 2)")
    split_commands = splits.add_subparsers(dest="action", required=True)
    for action in ("freeze", "audit"):
        child = split_commands.add_parser(action)
        _config_argument(child)
        _data_arguments(child, config=False)

    for name in ("train-source", "run"):
        child = commands.add_parser(name, help="Explicit fit-only job for implemented methods")
        _config_argument(child)
        _data_arguments(child, config=False)
    study = commands.add_parser("study", help="Frozen study execution and evaluation (Phase 6)")
    study_commands = study.add_subparsers(dest="action", required=True)
    run = study_commands.add_parser("run")
    _config_argument(run)
    _data_arguments(run, config=False)
    run.add_argument("--resume", action="store_true")
    bench = study_commands.add_parser("benchmark")
    _config_argument(bench)
    _data_arguments(bench, config=False)
    bench.add_argument("--task", required=True)
    for action in ("evaluate", "summarize", "estimate", "recover", "status"):
        child = study_commands.add_parser(action)
        child.add_argument("--study-id", required=True)
        child.add_argument("--output-root", default="artifacts")
    export = commands.add_parser("export", help="Export saved evidence (Phase 7)")
    export.add_argument("--study-id", required=True)
    export.add_argument("--out", type=Path, required=True)
    _planned(export, 7)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "scope":
            print(FINAL_OBJECTIVE)
            return 0
        if args.command == "config":
            result = _validate(args)
        elif args.command == "doctor":
            from shiftlab.environment import environment_report

            result = environment_report(Path.cwd())
        elif args.command == "data":
            result = _data_command(args)
        elif args.command == "splits":
            from shiftlab.data.splits import audit_splits, freeze_splits

            config = load_config(args.config)
            operation = freeze_splits if args.action == "freeze" else audit_splits
            result = operation(args.data_root, config)
        elif args.command in {"train-source", "run"}:
            from shiftlab.experiments.runner import train_baselines, train_methods

            config = load_config(args.config)
            operation = train_baselines if args.command == "train-source" else train_methods
            result = operation(config, root=Path.cwd(), data_root=args.data_root)
        elif args.command == "study":
            from shiftlab.experiments.provenance import canonical_json
            from shiftlab.experiments.study import (
                benchmark,
                estimate_runtime,
                evaluate_study,
                fit_study,
                open_study,
            )

            def progress(event):
                print(canonical_json(event), file=sys.stderr, flush=True)

            if args.action in {"run", "benchmark"}:
                config = load_config(args.config)
                if args.action == "benchmark":
                    result = benchmark(config, Path.cwd(), args.data_root, args.task, progress)
                else:
                    result = fit_study(config, Path.cwd(), args.data_root, progress)
                    if result["status"] != "frozen_fits_complete":
                        print(json.dumps(result, indent=2), flush=True)
                        return 3
            elif args.action == "evaluate":
                result = evaluate_study(Path.cwd(), args.study_id, progress, args.output_root)
            elif args.action == "summarize":
                from shiftlab.evaluation.analysis import summarize_study

                result = summarize_study(Path.cwd(), args.study_id, progress, args.output_root)
            else:
                from shiftlab.data.storage import read_json
                from shiftlab.experiments.checkpoints import recover_owner, recover_records
                from shiftlab.experiments.storage import LocalArtifactStore

                directory, lock = open_study(Path.cwd(), args.study_id, args.output_root)
                if args.action == "estimate":
                    result = estimate_runtime(directory)
                elif args.action == "status":
                    result = (
                        read_json(directory / "checkpoint.json")
                        if (directory / "checkpoint.json").exists()
                        else {
                            "status": "frozen_planned",
                            "study_id": args.study_id,
                            "expected_primary": 1100,
                            "expected_sensitivity": 180,
                            "evaluated_records": 0,
                        }
                    )
                else:
                    result = {
                        "study_writer": recover_owner(directory)
                        if (directory / "writer.lock").exists()
                        else "no_writer_lock",
                        "records": recover_records(
                            LocalArtifactStore(
                                Path.cwd() / lock["configuration"]["execution"]["output_root"]
                            )
                        ),
                    }
        else:
            if hasattr(args, "config"):
                load_config(args.config)
            print(
                f"Not implemented: {args.command} {getattr(args, 'action', '')}. "
                f"This command is scheduled for Phase {args.phase}. "
                "No experiment or dataset operation was executed.",
                file=sys.stderr,
            )
            return NOT_IMPLEMENTED
    except (ConfigError, OSError, RuntimeError, ImportError, ValueError) as error:
        print(f"shiftlab: {error}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("shiftlab: cancelled; any started fit attempt has been preserved", file=sys.stderr)
        return 130
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
