# Domain Shift Workbench

A local research project evaluating classical classifiers under distribution shift.
Phase 1 builds the package, configuration contract and development checks. Real
data preparation, training, adaptation, evaluation and the six-page dashboard are
later phases. No study results exist yet.

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

The package works independently of Streamlit. `config validate` and `doctor` are
implemented; planned data, training, study and export commands return a nonzero
exit code and explain the implementing phase. They do not create study artifacts.

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

No repository push or public deployment is part of Phase 1.
