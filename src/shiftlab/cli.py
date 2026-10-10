"""Shared-engine commands with explicit nonzero errors for future-phase work."""

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
        _planned(child, 2)
    audit = data_commands.add_parser("audit")
    audit_selection = audit.add_mutually_exclusive_group(required=True)
    audit_selection.add_argument("--all", action="store_true")
    audit_selection.add_argument("--dataset", choices=list(DatasetName))
    _planned(audit, 2)

    splits = commands.add_parser("splits", help="Freeze split identities (Phase 2)")
    freeze = splits.add_subparsers(dest="action", required=True).add_parser("freeze")
    _config_argument(freeze)
    _planned(freeze, 2)

    for name in ("train-source", "run"):
        child = commands.add_parser(name, help="Fitting engine (Phase 3 and later)")
        _config_argument(child)
        _planned(child, 3)
    study = commands.add_parser("study", help="Frozen study execution and evaluation (Phase 6)")
    study_commands = study.add_subparsers(dest="action", required=True)
    run = study_commands.add_parser("run")
    _config_argument(run)
    run.add_argument("--resume", action="store_true")
    _planned(run, 6)
    for action in ("evaluate", "summarize"):
        child = study_commands.add_parser(action)
        child.add_argument("--study-id", required=True)
        _planned(child, 6)
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
    print(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False))
    return 0
