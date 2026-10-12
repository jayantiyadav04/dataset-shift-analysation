"""Publication must preserve exact evidence and exclude local private artifacts."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from tools.package_submission import INVENTORY, inspect_archive, payloads, permitted, write_archive


@pytest.mark.parametrize(
    "name",
    [
        "../reports/scores.csv",
        "C:/data.csv",
        "reports/../../data.csv",
        "reports\\scores.csv",
        "data/raw/reviews.txt",
        "artifacts/runs/model.joblib",
        ".git/config",
        ".env",
        "reports/.env.json",
        "reports/credentials.json",
        "reports/model.joblib",
    ],
)
def test_publication_rejects_private_or_unsafe_paths(name):
    assert not permitted(name)


def test_exact_bytes_and_existing_archive_preserved(tmp_path):
    original = b"model,score\r\nlr,0.5123456789012345\r\n"
    path = tmp_path / "output.zip"
    write_archive(path, {"reports/study/scores.csv": original}, "reviewed-commit", False)
    assert inspect_archive(path)["source_revision"] == "reviewed-commit"
    with ZipFile(path) as archive:
        assert archive.read("reports/study/scores.csv") == original
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_archive(path, {"README.md": b"changed"}, "other", True)
    assert path.read_bytes() == before


def test_tampered_evidence_rejected(tmp_path):
    path = tmp_path / "tampered.zip"
    write_archive(path, {"reports/score.csv": b"0.5"}, "revision", False)
    with ZipFile(path) as archive:
        manifest = archive.read(INVENTORY)
    with ZipFile(path, "w") as archive:
        archive.writestr("reports/score.csv", b"0.9")
        archive.writestr(INVENTORY, manifest)
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_archive(path)


def test_extra_unlisted_member_rejected(tmp_path):
    path = tmp_path / "extra.zip"
    write_archive(path, {"README.md": b"report"}, "revision", False)
    with ZipFile(path, "a") as archive:
        archive.writestr("data/raw/labels.csv", "private")
    with pytest.raises(ValueError, match="differ from the inventory"):
        inspect_archive(path)


def test_payload_missing_or_symlink_escape_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="Missing regular file"):
        payloads(tmp_path, ["README.md"])
    (tmp_path / "README.md").write_text("safe")
    real_resolve = Path.resolve

    def outside(path, *args, **kwargs):
        if path.name == "README.md":
            return tmp_path.parent / "external.md"
        return real_resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", outside)
    with pytest.raises(ValueError, match="escapes the project"):
        payloads(tmp_path, ["README.md"])


def test_report_objective_abstract_and_claim_traces():
    from shiftlab.data.storage import read_json, verify_seal
    from shiftlab.experiments.provenance import file_sha256
    from shiftlab.scope import FINAL_OBJECTIVE, PROPOSAL_ABSTRACT

    root = Path(__file__).resolve().parents[1]
    folder = root / "reports/final"
    if not folder.exists():
        pytest.skip("Final report artifacts are not present in this checkout")
    abstract = (folder / "paper_abstract.txt").read_text(encoding="utf-8").strip()
    assert len(abstract.split()) == 150
    for name in ("final_report.md", "paper_draft.md"):
        text = (folder / name).read_text(encoding="utf-8")
        assert FINAL_OBJECTIVE in text and abstract in text and "{{" not in text
    assert PROPOSAL_ABSTRACT in (folder / "final_report.md").read_text(encoding="utf-8")
    values = read_json(folder / "report_values.json")
    verify_seal(values)
    for name, expected in values["generated_table_hashes"].items():
        assert file_sha256(folder / name) == expected
    traces = values["example_prediction_traces"]
    assert len(traces) == len({r["run_id"] for r in traces}) == 30
    # Public evidence carries hashes, not redistributed private predictions.
    assert all(len(r["prediction_sha256"]) == 64 for r in traces)
    assert json.loads((root / "reports/reproduction/verification.json").read_text())


def test_final_archive_uses_committed_bytes_despite_checkout_normalization(tmp_path, monkeypatch):
    from tools.package_submission import package

    (tmp_path / "README.md").write_bytes(b"reviewed\r\n")
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports/final").mkdir()
    (tmp_path / "reports/final/result.md").write_bytes(b"result\r\n")
    committed = {"README.md": b"reviewed\n", "reports/final/result.md": b"result\n"}

    def git_output(command, **kwargs):
        args = command[3:]
        if args[0] == "status":
            return ""
        if args[0] == "ls-files":
            return "\0".join(committed) + "\0"
        if args[0] == "rev-parse":
            return "reviewed-commit\n"
        assert args[0] == "show" and not kwargs
        return committed[args[1].split(":", 1)[1]]

    monkeypatch.setattr("tools.package_submission.subprocess.check_output", git_output)
    destination = tmp_path / "package"
    package(tmp_path, destination)
    with ZipFile(destination / "source_and_evidence.zip") as archive:
        for name, expected in committed.items():
            assert archive.read(name) == expected
