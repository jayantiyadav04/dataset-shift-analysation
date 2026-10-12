"""Audit report facts and rehearse the saved-evidence demo path without training."""

import argparse
import inspect
import json
import re
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import unquote
from zipfile import ZipFile

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from app.exports import build_bundle  # noqa: E402
from app.repository import Context, EvidenceRepository  # noqa: E402
from shiftlab.adaptation.coral import fit_coral  # noqa: E402
from shiftlab.adaptation.importance import density_ratios  # noqa: E402
from shiftlab.data.storage import (  # noqa: E402
    read_json,
    seal,
    verify_files,
    verify_seal,
    write_json,
)
from shiftlab.experiments.config import load_config  # noqa: E402
from shiftlab.experiments.provenance import file_sha256  # noqa: E402
from shiftlab.scope import FINAL_OBJECTIVE, PROPOSAL_ABSTRACT  # noqa: E402


def verify() -> dict:
    final = ROOT / "reports/final"
    values = read_json(final / "report_values.json")
    verify_seal(values)
    verify_files(ROOT, values["input_hashes"])
    verify_files(final, values["generated_table_hashes"])
    for directory, filename in [
        ("study", "evidence_manifest.json"),
        ("workbench", "interface_review.json"),
    ]:
        folder = ROOT / "reports" / directory
        manifest = read_json(folder / filename)
        verify_seal(manifest)
        verify_files(folder, manifest["output_hashes"])
    study = values["study_id"]
    lock = read_json(ROOT / "artifacts/studies" / study / "study_lock.json")
    verify_seal(lock)
    verify_files(ROOT, lock["code_snapshot"])
    assert len(lock["code_snapshot"]) == 38
    assert file_sha256(ROOT / "requirements-lock.txt") == lock["dependency_lock_hash"]
    configuration = load_config(ROOT / "configs/study.yaml")
    assert configuration.model_dump(mode="json") == lock["configuration"]
    assert (ROOT / "configs/study.yaml").read_bytes() == (
        final / "study_configuration.yaml"
    ).read_bytes()
    assert lock["objective"] == FINAL_OBJECTIVE
    assert file_sha256(ROOT / "reports/proposal.md") == (
        "62b39726fa5f7df98432e7a063cf34676846018b91921e489791422056e92f3a"
    )
    assert file_sha256(ROOT / "docs/00_OBJECTIVE_LOCK.md") == (
        "450e1dcf97dd518c8b6a3cdaa2920de98f3cb426d6da6526d51ec21190e6ffad"
    )
    abstract = (final / "paper_abstract.txt").read_text(encoding="utf-8").strip()
    assert len(abstract.split()) == 150
    for name in ("final_report.md", "paper_draft.md", "slide_outline.md"):
        text = (final / name).read_text(encoding="utf-8")
        assert FINAL_OBJECTIVE in text
        if name != "slide_outline.md":
            assert abstract in text
    assert PROPOSAL_ABSTRACT in (final / "final_report.md").read_text(encoding="utf-8")
    links = []
    for path in [*final.rglob("*.md"), ROOT / "reports/reproduction/README.md"]:
        text = path.read_text(encoding="utf-8")
        assert "{{" not in text and "TODO" not in text and "TBD" not in text
        for target in re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text):
            if target.startswith(("https://", "http://", "#", "mailto:")):
                continue
            resolved = (path.parent / unquote(target.split("#")[0])).resolve()
            assert resolved.is_relative_to(ROOT) and resolved.exists(), (path, target)
            links.append(resolved.relative_to(ROOT).as_posix())
    reproduction = read_json(ROOT / "reports/reproduction/verification.json")
    request = read_json(ROOT / "reports/reproduction/request.json")
    verify_seal(reproduction)
    verify_seal(request)
    assert reproduction["request_hash"] == request["manifest_hash"]
    assert file_sha256(ROOT / "reports/reproduction/executed_driver.py") == request["driver_sha256"]
    assert reproduction["manifest_hash"] == values["reproduction_receipt_hash"]
    assert reproduction["fresh_fits"] == reproduction["evaluations"] == 50
    assert reproduction["all_predictions_and_metrics_equal"]
    assert not reproduction["forbidden_training_reads"]
    for row in reproduction["comparisons"]:
        assert all(
            row[key]
            for key in (
                "ordered_predictions_equal",
                "all_metrics_equal",
                "selected_parameters_equal",
            )
        )
    for trace in values["example_prediction_traces"]:
        assert file_sha256(ROOT / trace["prediction_path"]) == trace["prediction_sha256"]
    repo = EvidenceRepository(ROOT)
    progress = repo.progress(study)
    assert progress["status"] == "complete"
    examples = pd.read_csv(final / "tables/interpretation_examples.csv")
    verified_examples = []
    for example in examples.to_dict("records"):
        context = Context(study, "gas", example["task_id"])
        pairs = repo.table(context, "paired_summary")
        pair = pairs[
            (pairs.model == example["model"])
            & (pairs.method == example["method"])
            & (pairs.setting == "primary")
        ].iloc[0]
        for field in ("macro_f1_delta_mean", "macro_f1_lower", "macro_f1_upper"):
            assert np.isclose(pair[field], example[field], rtol=0, atol=1e-14)
        verified_examples.append(example["task_id"] + ":" + example["model"])
    # Rehearse the actual explicit export path; the private split-ID ZIP stays local.
    bundle = build_bundle(
        repo,
        Context(study, "gas", "gas_b01_b02"),
        ["rf"],
        ["source_only", "importance_weighting", "coral"],
        ["primary"],
        False,
    )
    with ZipFile(BytesIO(bundle)) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        verify_seal(manifest)
        for name, expected in manifest["output_hashes"].items():
            import hashlib

            assert hashlib.sha256(archive.read(name)).hexdigest() == expected
    _, raw, _, weights = density_ratios(
        np.array([1 / 3]), np.array([80]), 40, configuration.importance_weighting
    )
    assert np.allclose(raw, [1]) and np.allclose(weights, [1])
    assert list(inspect.signature(fit_coral).parameters) == ["view", "parameters"]
    return seal(
        {
            "status": "passed",
            "scope": "Local evidence audit and automated demo/code checks; no training",
            "study_id": study,
            "frozen_source_files_unchanged": 38,
            "objective_and_proposal_unchanged": True,
            "abstract_words": 150,
            "reproduction_fresh_fits": 50,
            "reproduction_exact_matches": 50,
            "verified_example_predictions": len(values["example_prediction_traces"]),
            "verified_demo_examples": verified_examples,
            "demo_export_bytes": len(bundle),
            "demo_export_manifest_hash": manifest["manifest_hash"],
            "conceptual_count_correction_example": "q=1/3, ns=80, nt=40: raw ratio=1",
            "checked_local_links": sorted(set(links)),
            "student_rehearsal": "pending; cannot be certified by this script",
            "full_regression_status": (
                "See separate test receipt; this audit is not a test-suite substitute"
            ),
        }
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = verify()
    write_json(args.out, result)
    print(
        json.dumps(
            {key: value for key, value in result.items() if key != "checked_local_links"}, indent=2
        )
    )
