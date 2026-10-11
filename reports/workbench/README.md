# Verified research interface

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
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
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
