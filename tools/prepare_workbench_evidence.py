"""Publish aggregate interface review; full split-ID export bundles stay local."""

import argparse
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from zipfile import ZipFile

from shiftlab.data.storage import (
    publish_directory,
    read_json,
    seal,
    stage_directory,
    verify_files,
    verify_seal,
    write_json,
)
from shiftlab.experiments.provenance import file_sha256


def prepare(
    root: Path, browser_directory: Path, bundle: Path, jobs_path: Path, destination: Path
) -> dict:
    browser = read_json(browser_directory / "browser_review.json")
    verify_seal(browser)
    verify_files(browser_directory, browser["screenshot_hashes"])
    verify_files(root, browser["app_file_hashes"])
    jobs = read_json(jobs_path)
    verify_seal(jobs)
    if jobs["status"] != "passed" or not jobs["duplicate_launch_refused"]:
        raise ValueError("Real worker verification did not pass")
    tests = []
    for name in ("phase_07_release_tests.xml", "phase_07_missing_display_tests.xml"):
        path = root / "docs" / name
        suites = ET.parse(path).getroot().findall("testsuite")
        counts = {
            key: sum(int(s.attrib[key]) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        if counts["failures"] or counts["errors"] or counts["skipped"]:
            raise ValueError("Tests must actually pass without skipped real-evidence checks")
        tests.append({"receipt": name, "sha256": file_sha256(path), **counts})
    with ZipFile(bundle) as archive:
        manifest = read_json_from_bytes(archive.read("export_manifest.json"))
        verify_seal(manifest)
        import hashlib

        for name, expected in manifest["output_hashes"].items():
            if hashlib.sha256(archive.read(name)).hexdigest() != expected:
                raise ValueError(f"Bundle checksum mismatch: {name}")
    stage = stage_directory(destination)
    (stage / "screenshots").mkdir()
    for name in browser["screenshot_hashes"]:
        shutil.copyfile(browser_directory / name, stage / "screenshots" / name)
    write_json(stage / "browser_review.json", browser)
    write_json(stage / "demo_export_manifest.json", manifest)
    readme = """# Verified research interface

The six-page local workbench reads the completed frozen study and launches only
explicit jobs. CSV/JSON and figures share saved numerical evidence. The original
scientific engine, objective, proposal and 1,280 evaluations remain unchanged.

The real browser review contains all six pages at 1280 × 800 and 820 × 1000,
plus paired trace, weighting, CORAL and export-ready screens: 16 PNGs. Chrome's
actual headless browser rendered them; no screenshots were generated from a
template. Browser source hashes, screenshot hashes, viewport observations and
keyboard results are in browser_review.json. Keyboard navigation used radio
arrows, Tab, select search/Enter and Enter on Build verified bundle. Navigation
had no horizontal body overflow; labelled detailed grids can scroll horizontally.

Full regression suite: 212 passed. Final presentation/integrity checks: 20 passed.
Real one-fit development worker completed, duplicate launch was refused and an
owned worker was cancelled. Development jobs performed zero target evaluations.
Full jobs use a saved request configuration, even if repository YAML later changes.
The complete study was not retrained for screenshots or exports. Empty storage,
incomplete analysis, changed filters, checksum failures and detached worker states
are handled explicitly. Mixed columns with missing values use text to show an
em dash; filter to adapted methods for numeric delta sorting. Means remain numeric.

The actual sample ZIP is local at artifacts/ui_review/verified_report_bundle.zip;
demo_export_manifest.json publishes its trace and hashes. The ZIP contains exact
split IDs and stays local, alongside raw data, predictions, models and working logs.
Generate a new bundle with tools/export_workbench.py or the Report Exports page.
Exports never fit a classifier or open a fresh held-out dataset.

Use the app with:

```powershell
.\\.venv\\Scripts\\python.exe -m streamlit run app/streamlit_app.py
```

Open Results, select one transfer and comparison, then expand the exact-run trace
to inspect matched prediction paths/checksums and the frozen split. On Datasets,
download exact role/group IDs. If only three of five seeds are saved, the UI marks
the comparison incomplete and does not present a final paired interval.

Review limitations: automated keyboard checks are recorded, not a screen-reader
certification. Full-study worker argument/configuration wiring uses the unchanged
existing engine; the full matrix was already completed and was not recomputed in
this interface phase. Browser-control initialization was unavailable; installed
Chrome and locked websockets supplied the isolated local QA fallback. Earlier
capture attempts and server logs remain preserved in artifacts/ui_review/.
"""
    (stage / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    review = seal(
        {
            "status": "passed",
            "study_id": manifest["study_id"],
            "tests": tests,
            "browser_review_hash": browser["manifest_hash"],
            "real_jobs": jobs,
            "local_demo_zip_sha256": file_sha256(bundle),
            "demo_export_manifest_hash": manifest["manifest_hash"],
            "original_engine_unchanged": True,
            "output_hashes": {
                p.relative_to(stage).as_posix(): file_sha256(p)
                for p in stage.rglob("*")
                if p.is_file()
            },
        }
    )
    write_json(stage / "interface_review.json", review)
    publish_directory(stage, destination)
    return review


def read_json_from_bytes(value: bytes) -> dict:
    import json

    return json.loads(value)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--browser-directory", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--jobs", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("reports/workbench"))
    args = parser.parse_args()
    result = prepare(Path.cwd(), args.browser_directory, args.bundle, args.jobs, args.out)
    print(f"Prepared verified interface evidence for {result['study_id']}")
