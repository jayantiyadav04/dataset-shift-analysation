"""Headless export using the dashboard's validated saved-evidence path."""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.exports import build_bundle  # noqa: E402
from app.repository import Context, EvidenceRepository  # noqa: E402
from shiftlab.data.storage import safe_child  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--dataset", choices=("gas", "har", "amazon"), required=True)
    parser.add_argument("--task")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("lr", "linear_svm", "rf", "majority"),
        default=["lr", "linear_svm", "rf", "majority"],
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("source_only", "importance_weighting", "coral"),
        default=["source_only", "importance_weighting", "coral"],
    )
    parser.add_argument("--settings", nargs="+", default=["primary"])
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    repo = EvidenceRepository(ROOT, args.artifact_root)
    bundle = build_bundle(
        repo,
        Context(args.study_id, args.dataset, args.task),
        args.models,
        args.methods,
        args.settings,
        not args.no_figures,
    )
    destination = safe_child(ROOT, args.out)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(bundle)
    print(f"Exported {len(bundle)} bytes from saved evidence to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
