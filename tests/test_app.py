"""UI isolation, artifact integrity, explicit jobs and actual saved-evidence parity."""

import hashlib
import json
import shutil
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import pytest
from streamlit.testing.v1 import AppTest

from app.exports import build_bundle
from app.jobs import JobManager, small_config
from app.repository import Context, EvidenceRepository
from shiftlab.data.storage import DataError, seal, verify_seal, write_json
from shiftlab.experiments.config import load_config
from tools.run_workbench_job import configuration_snapshot

ROOT = Path(__file__).resolve().parents[1]
PAGES = ("Overview", "Datasets", "Experiment Runner", "Results", "Diagnostics", "Report Exports")
STUDY = "a05bea2f175b9d81091a413637276d174c378efaf31d1c539430e98055df77d0"
REAL = ROOT / "artifacts/studies" / STUDY / "analysis/export_manifest.json"


@pytest.mark.parametrize("page", PAGES)
def test_empty_pages_do_not_launch_or_borrow_completed_evidence(page, monkeypatch):
    monkeypatch.setenv("SHIFTLAB_ARTIFACT_ROOT", "artifacts/nonexistent_ui_test_storage")
    monkeypatch.setattr(JobManager, "start", lambda *a, **k: pytest.fail("Implicit training"))
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception and not app.error
    assert len(app.dataframe) <= 1  # Runner may retain actual prior UI job history.
    assert "No frozen study" in " ".join(c.value for c in app.caption)


@pytest.mark.skipif(not REAL.exists(), reason="Real study artifacts are deliberately local")
@pytest.mark.parametrize("page", PAGES)
def test_completed_pages_read_real_evidence_without_launch(page, monkeypatch):
    monkeypatch.setattr(JobManager, "start", lambda *a, **k: pytest.fail("Implicit training"))
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value(page).run()
    assert not app.exception and not app.error


def test_repository_detects_changed_file_and_different_study(tmp_path):
    identity = "a" * 64
    directory = tmp_path / "artifacts/studies" / identity
    analysis = directory / "analysis"
    analysis.mkdir(parents=True)
    lock = seal({"study_id": identity})
    write_json(directory / "study_lock.json", lock)
    table = analysis / "per_task_summary.csv"
    table.write_text("dataset,task_id,macro_f1_mean\ngas,gas_b01_b02,0.25\n", encoding="utf-8")
    export = {
        "study_id": identity,
        "study_lock_hash": lock["manifest_hash"],
        "output_hashes": {table.name: hashlib.sha256(table.read_bytes()).hexdigest()},
    }
    write_json(analysis / "export_manifest.json", seal(export))
    repo = EvidenceRepository(tmp_path)
    assert (
        repo.table(Context(identity, "gas", "gas_b01_b02"), "per_task_summary").iloc[0][
            "macro_f1_mean"
        ]
        == 0.25
    )
    assert repo.table(Context(identity, "har", None), "per_task_summary").empty
    table.write_text(table.read_text().replace("0.25", "0.99"))
    with pytest.raises(DataError, match="checksum"):
        repo.table(Context(identity, "gas", None), "per_task_summary")
    with pytest.raises(DataError):
        EvidenceRepository(tmp_path, "../outside")


def test_development_config_keeps_locked_scientific_settings():
    primary = load_config(ROOT / "configs/study.yaml")
    small = small_config(ROOT, "amazon_books_electronics", "rf", "coral", 29)
    for field in ("data", "preprocessing", "tuning", "importance_weighting", "coral", "evaluation"):
        assert getattr(small, field) == getattr(primary, field)
    assert small.planned_counts()["main_fits"] == 1
    with pytest.raises(ValueError):
        small_config(ROOT, "gas_b01_b02", "neural", "coral", 29)


def test_full_worker_freezes_launched_configuration(tmp_path):
    original = load_config(ROOT / "configs/study.yaml")
    path = configuration_snapshot(original, tmp_path)
    assert load_config(path) == original
    with pytest.raises(FileExistsError):
        configuration_snapshot(original, tmp_path)


def test_single_worker_across_managers_and_owned_cancellation(tmp_path, monkeypatch):
    (tmp_path / "configs").mkdir()
    shutil.copyfile(ROOT / "configs/study.yaml", tmp_path / "configs/study.yaml")
    calls = []

    class Worker:
        pid = 123456

        def __init__(self, args, **kwargs):
            calls.append((args, kwargs))

        def poll(self):
            return None

    monkeypatch.setattr("app.jobs.HANDLES", {})
    monkeypatch.setattr("app.jobs.subprocess.Popen", Worker)
    first, second = JobManager(tmp_path), JobManager(tmp_path)
    job = first.start(action="fit", task="gas_b01_b02")
    with pytest.raises(DataError, match="already active"):
        second.start(action="fit", task="gas_b01_b02")
    assert len(calls) == 1 and calls[0][1]["shell"] is False
    assert "--job-id" in calls[0][0]
    with pytest.raises(DataError, match="Cancellation"):
        second.cancel("b" * 32)
    first.cancel(job["job_id"])
    assert (tmp_path / "artifacts/ui_jobs" / job["job_id"] / "cancel.request").exists()
    monkeypatch.setattr("app.jobs.HANDLES", {})
    monkeypatch.setattr("app.jobs.pid_alive", lambda pid: True)
    assert second.current()["status"] == "detached"
    with pytest.raises(DataError):
        second.cancel(job["job_id"])


@pytest.mark.skipif(not REAL.exists(), reason="Real study artifacts are deliberately local")
def test_real_filters_export_values_hashes_and_prediction_trace_agree():
    repo = EvidenceRepository(ROOT)
    context = Context(STUDY, "gas", "gas_b01_b02")
    bundle = build_bundle(repo, context, ["lr"], ["source_only", "coral"], ["primary"])
    with ZipFile(BytesIO(bundle)) as archive:
        manifest = json.loads(archive.read("export_manifest.json"))
        verify_seal(manifest)
        for name, digest in manifest["output_hashes"].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == digest
        exported = pd.read_csv(BytesIO(archive.read("per_task_summary.csv")))
        saved = repo.table(context, "per_task_summary")
        saved = saved[(saved.model == "lr") & saved.method.isin(["source_only", "coral"])]
        pd.testing.assert_frame_equal(exported.reset_index(drop=True), saved.reset_index(drop=True))
        assert len(manifest["prediction_inputs"]) == 10
        assert all(
            "joblib" not in name and "predictions.csv" not in name for name in archive.namelist()
        )


@pytest.mark.skipif(not REAL.exists(), reason="Real study artifacts are deliberately local")
def test_dataset_change_cannot_keep_previous_results(monkeypatch):
    monkeypatch.setattr(JobManager, "start", lambda *a, **k: pytest.fail("Implicit launch"))
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Results").run()
    assert set(app.dataframe[0].value.task_id) == {"gas_b01_b02"}
    table = app.dataframe[0].value
    assert set(table[table.method == "source_only"].macro_f1_delta_mean) == {"—"}
    next(w for w in app.sidebar.selectbox if w.label == "Dataset").set_value("har").run()
    assert not app.exception and not app.error
    assert set(app.dataframe[0].value.task_id) == {"har_official_group_transfer"}


@pytest.mark.skipif(not REAL.exists(), reason="Real study artifacts are deliberately local")
def test_incomplete_study_does_not_show_another_studys_results(tmp_path, monkeypatch):
    directory = tmp_path / "studies" / STUDY
    directory.mkdir(parents=True)
    shutil.copyfile(REAL.parents[1] / "study_lock.json", directory / "study_lock.json")
    monkeypatch.setenv("SHIFTLAB_ARTIFACT_ROOT", tmp_path.relative_to(ROOT).as_posix())
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Results").run()
    assert not app.exception and not app.error
    assert not app.dataframe
    assert "No completed analysis" in app.info[0].value


@pytest.mark.skipif(not REAL.exists(), reason="Real study artifacts are deliberately local")
def test_export_invalidates_after_filter_change():
    app = AppTest.from_file(str(ROOT / "app/streamlit_app.py"), default_timeout=30).run()
    app.sidebar.radio[0].set_value("Report Exports").run()
    next(w for w in app.button if w.label == "Build verified bundle").click().run()
    assert not app.exception and not app.error
    assert app.session_state["report_bundle"]
    next(w for w in app.multiselect if w.label == "Models").set_value(["lr"]).run()
    assert "Selection changed" in " ".join(w.value for w in app.info)
