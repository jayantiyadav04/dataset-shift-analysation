# Domain Shift Workbench

A local research project evaluating classical classifiers under distribution shift.
The package includes validated configuration, official data acquisition and parsing,
immutable split manifests and source-only numerical/text preprocessing. Training,
adaptation, evaluation and the six-page dashboard are later phases. No study results
exist yet.

## Frozen objective

To evaluate whether importance weighting and feature alignment using CORAL improve the performance of Logistic Regression, Support Vector Machine, and Random Forest when source and target data follow different distributions. The study will compare these techniques against models trained without adaptation across three public datasets, using accuracy, macro F1 score, and computational time. It will identify the conditions under which adaptation improves performance, provides little benefit, or reduces prediction quality.

## Setup

The tested environment is Windows with CPython 3.12.10. `requirements-lock.txt`
records all 64 actual installed dependencies, including packaging tools. Use these
pins before installing the package; the lock is an environment snapshot without
artifact hashes, not a guarantee of compatibility on every operating system.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements-lock.txt
.\.venv\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e ".[dev]"
.\.venv\Scripts\shiftlab.exe --help
.\.venv\Scripts\shiftlab.exe config validate --config configs/study.yaml
.\.venv\Scripts\shiftlab.exe doctor
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe -m ruff check src app tests tools
.\.venv\Scripts\python.exe tools/lock_environment.py --check
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.address 127.0.0.1
```

The package works independently of Streamlit. Configuration, environment, data
and split commands are implemented. Training, study and export commands return
a nonzero exit code and explain the implementing phase.

## Prepare the declared datasets

Run from the repository root. The configuration is validated before acquisition
or preparation. All data stay in the ignored local `data/` directory.

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
.\.venv\Scripts\python.exe tools/verify_phase02.py --out docs/phase_02_verification.json
```

The verification command refuses to replace an existing evidence file; choose a
new output filename for a justified rerun. It audits all splits and tests real
source preprocessing with training access to evaluation files forbidden.

If a host is unavailable, download its official archive and import it with
`data fetch --dataset gas --archive path/to/archive.zip`. For Amazon supply both
`--archive path/to/domain_sentiment_data.tar.gz` and
`--books-unlabeled path/to/book.unlabeled.gz`. Use `--data-root` consistently to
prepare a separate archive version; existing archives and frozen splits cannot be
overwritten. Partial operations leave staging directories and do not appear complete.

Official acquisition sources:

- [UCI Gas Sensor Array Drift, dataset 224](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset)
- [UCI HAR Using Smartphones, dataset 240](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)
- [Original four-domain Amazon sentiment corpus and separate books file](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)

Each prepared dataset has `manifest.json`, `DATA_CARD.md`, explicit metadata and
`exclusions.jsonl` in `data/processed/<dataset>/`. Archives and extracted files have
actual SHA-256 hashes. Review features contain bodies only; ratings and reviewer
metadata are excluded. Conflicting duplicates are quarantined. Duplicate bodies
stay in the labelled pool before the unlabelled pool, and each transfer removes
source duplicates from its target pools.

The frozen role IDs and hashes are in `data/splits/1.0/<task>/manifest.json`.
Training files contain source features/labels and target adaptation features;
evaluation files are in a separate subdirectory. `load_training` opens only the
permitted training files. This is an application boundary on a local filesystem,
not an operating-system access-control system.

Source CV keeps participants and duplicate bodies together and uses valid product
groups when feasible. Each fold creates a new scaler or TF-IDF/SVD/scaler fit using
that fold's source training rows. The text representation uses at most 128 dense
components, capped below observed source rank; actual dimensions are recorded.
All three future methods will share this representation.

Raw datasets remain local. The gas page has both current CC BY 4.0 and older
research-only wording; the [author's Amazon page](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
does not give an explicit redistribution licence. Dataset cards preserve the
source terms; acquisition instructions are supplied instead of raw archives.

## Study boundaries

Gas Sensor Array Drift, HAR Using Smartphones and the original four-domain Amazon
sentiment corpus are three independent datasets. LR, LinearSVC and RF are compared
under source-only, importance weighting and CORAL, with a source-majority baseline.
The full configuration declares 22 transfers and five seeds (17, 29, 43, 71, 101).
All run counts are planned until supported by saved evidence.

Source labels and target adaptation features are permitted for fitting. Target
evaluation features and labels remain outside the training contract. Source
validation must fit preprocessing within each fold; source-selected parameters
are shared across paired methods. No method is assumed to improve performance.

## Layout and local materials

`src/shiftlab/` holds the shared engine; `app/` holds the interface; `tests/` holds
offline acceptance checks; `configs/` holds study contracts; `reports/` holds the
preserved proposal and reading plan. The synthetic fixture is development data and
is never written to `artifacts/` as a real study.

At the user's request, `docs/`, `prompts/`, `templates/`, `submission/`,
`README_START_HERE.md` and `AGENTS.md` stay local and are ignored by Git. Keep a
separate local backup of those materials and logs. The source distribution keeps
the exact objective and proposal abstract in the package and proposal artifact.
Raw data, local environments, credentials and generated study artifacts are also
excluded. The supplied `MANIFEST.sha256` describes the original documentation
pack, not the newly implemented source or a completed experiment.

The installation resolved NumPy 2.5.3, SciPy 1.18.1, pandas 2.3.3,
scikit-learn 1.9.1 and Streamlit 1.65.0. All other exact versions are in the lock.
`pip freeze --all --exclude-editable` supplies the lock through
`tools/lock_environment.py`; no personal editable-install path is retained.
The checks run offline after installation. The supplied pack validator checks
only project Markdown, so it ignores installed third-party documentation.

Pushes and public deployment require a separate user instruction.
