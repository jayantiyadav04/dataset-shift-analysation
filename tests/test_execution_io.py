"""Windows progress retries must preserve atomic bytes and immutable evidence."""

import importlib.util
from pathlib import Path

import pytest

from shiftlab.data.storage import read_json, verify_seal

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("study_driver", ROOT / "tools/run_frozen_study.py")
DRIVER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(DRIVER)


def denied(winerror):
    error = PermissionError("simulated file sharing conflict")
    error.winerror = winerror
    return error


def test_transient_progress_replacement_retries_same_bytes(tmp_path, monkeypatch):
    study = tmp_path / "artifacts/studies/frozen-id"
    study.mkdir(parents=True)
    target, source = study / "checkpoint.json", study / ".pending"
    target.write_bytes(b"previous complete snapshot")
    source.write_bytes(b"next complete snapshot")
    original = DRIVER.os.replace
    calls = []

    def transient(first, second):
        calls.append((first, second))
        if len(calls) < 3:
            assert target.read_bytes() == b"previous complete snapshot"
            assert source.read_bytes() == b"next complete snapshot"
            raise denied(5)
        return original(first, second)

    monkeypatch.setattr(DRIVER.os, "replace", transient)
    monkeypatch.setattr(DRIVER, "sleep", lambda _seconds: None)
    with DRIVER.retry_progress_io(study) as retries:
        DRIVER.os.replace(source, target)
    assert len(calls) == 3 and all(pair == (source, target) for pair in calls)
    assert target.read_bytes() == b"next complete snapshot" and not source.exists()
    assert len(retries) == 2
    receipts = list((tmp_path / "artifacts/operations").glob("*.json"))
    assert len(receipts) == 1
    receipt = read_json(receipts[0])
    verify_seal(receipt)
    assert receipt["retries"] == retries


@pytest.mark.parametrize(
    ("relative", "winerror", "expected_calls"),
    [
        ("runs/model.joblib", 5, 1),
        ("checkpoint.json", None, 1),
        ("checkpoint.json", 5, 40),
    ],
)
def test_nonprogress_or_persistent_denial_preserves_evidence(
    tmp_path, monkeypatch, relative, winerror, expected_calls
):
    study = tmp_path / "artifacts/studies/frozen-id"
    target, source = study / relative, study / ".pending"
    target.parent.mkdir(parents=True)
    target.write_bytes(b"saved evidence")
    source.write_bytes(b"new snapshot")
    calls = []

    def fail(first, second):
        calls.append((first, second))
        raise denied(winerror)

    monkeypatch.setattr(DRIVER.os, "replace", fail)
    monkeypatch.setattr(DRIVER, "sleep", lambda _seconds: None)
    with pytest.raises(PermissionError), DRIVER.retry_progress_io(study):
        DRIVER.os.replace(source, target)
    assert len(calls) == expected_calls
    assert target.read_bytes() == b"saved evidence"
    assert source.read_bytes() == b"new snapshot"
