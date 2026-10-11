"""Installed command boundaries and comparison with preserved assessment text."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from shiftlab.cli import NOT_IMPLEMENTED, main
from shiftlab.scope import FINAL_OBJECTIVE, PROPOSAL_ABSTRACT

ROOT = Path(__file__).resolve().parents[1]


def test_assessment_text_is_preserved():
    proposal = (ROOT / "reports" / "proposal.md").read_text(encoding="utf-8")
    assert FINAL_OBJECTIVE in proposal
    assert PROPOSAL_ABSTRACT in proposal
    assert len(PROPOSAL_ABSTRACT.split()) == 150
    local_lock = ROOT / "docs" / "00_OBJECTIVE_LOCK.md"
    if local_lock.is_file():
        objective = local_lock.read_text(encoding="utf-8").split("## Final objective\n\n")[1]
        assert objective.split("\n\n##")[0].strip() == FINAL_OBJECTIVE


def test_cli_validates_without_running_a_study(capsys):
    assert main(["config", "validate", "--config", str(ROOT / "configs/study.yaml")]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "valid_configuration"
    assert result["planned_counts"]["main_fits"] == 990
    assert "completed" not in result["planned_counts"]


@pytest.mark.parametrize(
    "arguments",
    [
        ["study", "run", "--config", "configs/study.yaml", "--resume"],
        ["study", "evaluate", "--study-id", "no-study"],
        ["study", "summarize", "--study-id", "no-study"],
        ["export", "--study-id", "no-study", "--out", "reports/generated"],
    ],
)
def test_planned_commands_fail_explicitly_without_writing(arguments, tmp_path, monkeypatch, capsys):
    arguments = [str(ROOT / a) if a.endswith(".yaml") else a for a in arguments]
    monkeypatch.chdir(tmp_path)
    assert main(arguments) == NOT_IMPLEMENTED
    assert "Not implemented" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_scientific_engine_imports_without_interface():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import shiftlab.cli; "
            "assert 'streamlit' not in sys.modules; assert 'shiftlab.cli' in sys.modules",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_fitting_rejects_missing_prerequisites_without_outputs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--config", str(ROOT / "configs/smoke.yaml")]) == 2
    assert "shiftlab:" in capsys.readouterr().err
    assert not list(tmp_path.iterdir())


def test_all_method_job_requires_a_real_runtime_before_any_write(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["run", "--config", str(ROOT / "configs/study.yaml")]) == 2
    assert not list(tmp_path.iterdir())


def test_actual_estimator_apis_accept_weights():
    from shiftlab.environment import compatibility_check

    assert compatibility_check() == {"lr": True, "linear_svm": True, "rf": True}
