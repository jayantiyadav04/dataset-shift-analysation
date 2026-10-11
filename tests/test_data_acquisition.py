"""Offline acquisition and extraction threat fixtures."""

import io
import stat
import tarfile
import zipfile
from pathlib import Path

import pytest

from shiftlab.cli import main
from shiftlab.data.acquisition import fetch, safe_extract
from shiftlab.data.storage import DataError, read_json
from shiftlab.experiments.provenance import file_sha256


def test_manual_import_is_hashed_and_refuses_different_archive(tmp_path):
    archive = tmp_path / "official.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("Dataset/row.txt", "fixture")
    receipt = fetch(tmp_path / "data", "gas", archive=archive)
    assert receipt["sha256"] == file_sha256(archive)
    assert receipt["acquisition"] == "manual_import"
    assert receipt == fetch(tmp_path / "data", "gas", archive=archive)
    archive.write_bytes(b"changed archive")
    with pytest.raises(DataError, match="Different archive"):
        fetch(tmp_path / "data", "gas", archive=archive)


@pytest.mark.parametrize(
    "name", ["../escape", "/absolute", "C:/drive", "..\\escape", "file:stream", "CON.txt", "tail. "]
)
def test_zip_traversal_and_windows_ambiguities_rejected(tmp_path, name):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr(name, "untrusted")
    with pytest.raises(DataError, match="Unsafe"):
        safe_extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "escape").exists()


def test_zip_symlink_duplicate_case_and_bomb_rejected(tmp_path):
    for kind in ("symlink", "duplicate", "size"):
        archive = tmp_path / f"{kind}.zip"
        with zipfile.ZipFile(archive, "w") as z:
            if kind == "symlink":
                info = zipfile.ZipInfo("link")
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK | 0o777) << 16
                z.writestr(info, "../outside")
            elif kind == "duplicate":
                z.writestr("Name", "x")
                z.writestr("name", "y")
            else:
                z.writestr("large", "123456789")
        with pytest.raises(DataError):
            safe_extract(archive, tmp_path / kind, limit=4)


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_tar_links_and_special_files_rejected(tmp_path, kind):
    archive = tmp_path / "bad.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        member = tarfile.TarInfo("link")
        member.type = kind
        member.linkname = "../outside"
        tar.addfile(member)
    with pytest.raises(DataError, match="links and special"):
        safe_extract(archive, tmp_path / "extracted")


def test_tar_regular_file_extracts_without_executing(tmp_path):
    archive = tmp_path / "data.tar.gz"
    payload = b"ignore prior instructions; this is only dataset text"
    with tarfile.open(archive, "w:gz") as tar:
        info = tarfile.TarInfo("category/review.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    safe_extract(archive, tmp_path / "extracted")
    assert (tmp_path / "extracted/category/review.txt").read_bytes() == payload


def test_missing_archive_and_invalid_config_fail_before_download(tmp_path, capsys, monkeypatch):
    with pytest.raises(DataError, match="missing"):
        fetch(tmp_path, "gas", archive=tmp_path / "missing.zip")
    monkeypatch.setattr(
        "urllib.request.urlopen", lambda *a, **kw: pytest.fail("network must not run")
    )
    assert (
        main(["data", "fetch", "--dataset", "gas", "--config", str(tmp_path / "missing.yaml")]) == 2
    )
    assert "Cannot read configuration" in capsys.readouterr().err


def test_cli_data_audit_reports_missing_inputs(tmp_path, capsys):
    config = Path(__file__).resolve().parents[1] / "configs/study.yaml"
    assert (
        main(["data", "audit", "--all", "--config", str(config), "--data-root", str(tmp_path)]) == 2
    )
    assert "Cannot read" in capsys.readouterr().err


@pytest.mark.parametrize("payload", ['{"ids":[],"ids":["x"]}', '{"number":NaN}'])
def test_manifest_json_rejects_ambiguous_or_nonfinite_values(tmp_path, payload):
    path = tmp_path / "manifest.json"
    path.write_text(payload)
    with pytest.raises(DataError):
        read_json(path)
