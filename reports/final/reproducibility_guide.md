# Reproducibility guide

Use CPython 3.12.10 on the tested Windows environment. The scientific dependency
snapshot records 64 actual versions. No model or trust receipt from an external
source should be imported. Use an empty artifact directory for a fresh study;
use the same directory and unchanged configuration to resume your own run.

## Install and check

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[dev]"
.\.venv\Scripts\python.exe tools/lock_environment.py --check
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\shiftlab.exe doctor
.\.venv\Scripts\shiftlab.exe config validate --config configs/study.yaml
```

The editable installation is required by the dependency-lock checker so the
project's local path is not mistaken for a registry dependency. A new platform
may need an explicitly documented compatible environment; do not silently
regenerate the original lock and claim that it is the same experiment.

## Acquire and freeze data

```powershell
.\.venv\Scripts\shiftlab.exe data fetch --dataset gas
.\.venv\Scripts\shiftlab.exe data fetch --dataset har
.\.venv\Scripts\shiftlab.exe data fetch --dataset amazon
.\.venv\Scripts\shiftlab.exe data prepare --dataset gas
.\.venv\Scripts\shiftlab.exe data prepare --dataset har
.\.venv\Scripts\shiftlab.exe data prepare --dataset amazon
.\.venv\Scripts\shiftlab.exe data audit --all
.\.venv\Scripts\shiftlab.exe splits freeze --config configs/study.yaml
.\.venv\Scripts\shiftlab.exe splits audit --config configs/study.yaml
```

Compare actual archive SHA-256 values, cleaned counts and all split hashes to
the completed data cards and split table. If a source is unavailable, import the
official archive with `data fetch --dataset NAME --archive PATH`. Amazon also
requires `--books-unlabeled PATH`. Do not replace the released corpus with newly
scraped reviews or fabricated data. Changed archive bytes require investigation,
not an assumed match. All raw inputs and split role files stay local.

## Execute or resume the frozen matrix

```powershell
.\.venv\Scripts\python.exe tools/run_frozen_study.py benchmark --task gas_b01_b10
.\.venv\Scripts\python.exe tools/run_frozen_study.py benchmark --task har_official_group_transfer
.\.venv\Scripts\python.exe tools/run_frozen_study.py benchmark --task amazon_books_electronics
.\.venv\Scripts\python.exe tools/run_frozen_study.py execute
```

Benchmarks cover full source tuning and ten primary fits each, with measured
component times and peak working set. The same execution command resumes verified
checkpoints. It freezes the plan and requires all 1,280 fits before full-study
target evaluation. Use Ctrl+C for graceful cancellation. After a hard interruption,
inspect the saved log and `writer.lock`; `shiftlab study recover --study-id ID`
requires a confirmed dead owner. Do not delete successful artifacts to clean up
a failure. The saved source snapshot and lock must still match on resume.

Final analyses recompute metrics from saved predictions and pair exact evaluation
IDs and seeds. There are 1,100 primary and 180 sensitivity records, 132 primary
and 36 sensitivity comparisons, and 336 accuracy/macro-F1 intervals. Expected and
completed counts remain separate. Full-study run IDs refer to the original source
commit in the study lock, even when a later publication commit contains the same
scientific source bytes.

## Reproduce one completed real task

The [clean reproduction record](../reproduction/README.md) contains the actual
commands and results of the gas Batch 1 to Batch 2 check. A newly created
environment and clean checkout produced 50 fresh fits with exact prediction,
metric and parameter agreement. It reused verified local data and requires the
original completed study as a comparison reference. It does not replace the full
study acquisition/execution sequence above on a machine without those artifacts.

## Inspect and export

```powershell
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
.\.venv\Scripts\python.exe tools/export_workbench.py --study-id a05bea2f175b9d81091a413637276d174c378efaf31d1c539430e98055df77d0 --dataset gas --task gas_b01_b02 --models lr --out artifacts/exports/gas_lr.zip
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\ruff.exe check src app tests tools
```

On a fresh checkout without local artifacts, the app correctly shows an empty
state. Published tables and screenshots in reports/ are static evidence and are
not loaded as pretend live runs. The working export wrapper reads the same
validated evidence as the UI; the older frozen `shiftlab export` placeholder is
not the export command. Exports refuse existing output filenames.

The source and aggregate evidence archive has an internal per-file SHA-256
inventory. The final package manifest records its archive hash and document hashes.
The original root MANIFEST.sha256 is a historical documentation-pack manifest;
do not use it as a checksum list for newly generated code or reports.
