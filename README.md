# Domain Shift Workbench

A local research project evaluating classical classifiers under distribution shift.
The package includes validated configuration, official data acquisition and parsing,
immutable split manifests, source-only numerical/text preprocessing and classical
baseline fitting with fold-safe source selection and domain-classifier importance
weighting and centred regularised CORAL alignment. The frozen study runner supports
resource benchmarks, checkpoints, held-out evaluation and paired grouped analysis.
The six-page dashboard displays verified completed evidence and launches explicit
local workers. All shown counts and findings come from saved study evidence.

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
and split commands are implemented. `train-source` fits source baselines; `run`
accepts source-only, importance-weighting and CORAL configurations. Study commands
benchmark, fit, evaluate, summarize, inspect progress and recover interrupted work.
Use `tools/export_workbench.py` for headless report exports through the same saved-evidence
reader as the dashboard. The frozen engine's earlier `shiftlab export` placeholder is
preserved; the working export command below is independent of Streamlit.

## Research dashboard

Open the local app with the Streamlit command above. Overview, Datasets, Experiment
Runner, Results, Diagnostics and Report Exports share the selected study, dataset,
transfer, protocol and split hash. Missing or invalid artifacts are named explicitly;
an empty storage view does not borrow another study's results. Inspect real interface
evidence in [reports/workbench](reports/workbench/README.md).

Results show saved five-seed means, sample SD, paired changes and conditional group
intervals. The comparison detail supplies exact baseline/adapted run IDs, prediction
paths/checksums and split identity. Diagnostics read saved mean-one weights, ESS,
covariance diagnostics, class errors and predeclared sensitivity settings; no fitted
estimator is deserialized by the interface. Large detailed tables scroll horizontally.

The runner defaults to one development fit with no target evaluation. A persistent
job ID and launch guard prevent duplicate workers across rerenders/sessions. Logs,
failures and cancellations remain visible. Cancellation is limited to an owned
process and interrupts at a Python operation boundary; a long numerical call may
finish first. Closing the browser does not cancel a worker. A detached worker blocks
new launches and requires its preserved log/CLI recovery. Full frozen execution or
resume is an explicit 1,280-record control; inspection and exports never train.

Build a report bundle explicitly on Report Exports. Changing filters invalidates a
prepared bundle. CSV/JSON, figures, exact split IDs, methodology and prediction traces
have a sealed input/output hash manifest. Raw features, review bodies and serialized
models are excluded. Bundles stay local by default.

```powershell
.\.venv\Scripts\python.exe tools/export_workbench.py --study-id a05bea2f175b9d81091a413637276d174c378efaf31d1c539430e98055df77d0 --dataset gas --task gas_b01_b02 --models lr --out artifacts/exports/gas_lr.zip
```

Exports refuse to overwrite existing files. UI code lives in `app/`; fixed-request
workers call the unchanged `src/shiftlab` engine. The completed scientific snapshot
remains byte-identical. Verification: 212 full-suite tests passed, followed by 20 final
dashboard checks after the missing-value presentation change; 16 actual desktop/tablet
Chrome renders, keyboard checks and real
worker completion/cancellation are recorded locally and in the interface review.
Mixed columns containing missing values display em dashes as text; filter to adapted
methods for numeric delta sorting. Scientific exports retain their numeric values.

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
All three primary methods use this declared representation.

Raw datasets remain local. The gas page has both current CC BY 4.0 and older
research-only wording; the [author's Amazon page](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html)
does not give an explicit redistribution licence. Dataset cards preserve the
source terms; acquisition instructions are supplied instead of raw archives.

## Fit source baselines

```powershell
.\.venv\Scripts\shiftlab.exe train-source --config configs/baselines.yaml
.\.venv\Scripts\shiftlab.exe run --config configs/smoke.yaml
```

`baselines.yaml` is a development check: two gas transfers, the HAR transfer,
books to electronics, all three classifiers and a majority baseline, seed 17.
It is separate from the primary five-seed study. `train-source` can also prepare
baselines from `study.yaml`; it executes only source-only families.

Each source domain/seed uses five source-only CV folds and the fixed parameter
grid. Exact mean macro-F1 ties prefer lower C, depth 20 before unlimited depth,
then leaf size 3 before 1. Target changes cannot affect source selection.

Artifacts in `artifacts/runs/<run_id>/` contain configuration, environment,
source CV reference, preprocessing/model files, diagnostics, timings and a sealed
checksum manifest. Shared CV and preprocessing live in `artifacts/source_cache/`.
Resume verifies the complete identity and checksums and preserves original timings.
Fit-only records have status `fitted`; predictions and target metrics are absent.
Lightweight source-validation evidence is in `reports/source_baselines/`.

Failed attempts stay in `artifacts/failures/`; Ctrl+C records cancellation.
An exclusive lock prevents concurrent writers. After a hard process termination,
inspect the lock and pending record before manual recovery; the engine refuses
to guess that a writer is dead. It never silently overwrites an earlier attempt.

The evaluator requires a sealed study lock with completed fit IDs and exact
configuration, code, dependency and split hashes before reading target data.
It has been tested on synthetic fixtures only; final target evaluation remains
closed. Evaluation artifacts are separate immutable records referring to fitted
run IDs. Only locally generated, receipt-backed, checksum-verified joblib files
may be loaded. Do not import models or trust receipts from outside sources.

## Fit importance-weighted models

```powershell
.\.venv\Scripts\shiftlab.exe run --config configs/importance.yaml
```

This explicit development job fits matched source-only and importance-weighted
LR, LinearSVC and RF plus one source-majority baseline on gas Batch 1 to Batch 10,
HAR, and Amazon books to electronics, seed 17. It creates 21 fitted records;
it does not run the five-seed study or open target evaluation data.

The discriminator uses five source folds with intact participant/duplicate/product
groups. Each fold fits domain labels (source=0, adaptation=1) on the other source
rows and all adaptation features, using logistic regression C=1, no balancing,
no resampling, max_iter=10000 and the recorded seed. It predicts held-out source
probabilities in the shared full-source-fitted representation. For each source
row the density ratio is probability odds times that fold's actual source count
divided by the adaptation count. Bound probabilities to [0.001,0.999], clip raw
ratios to [0.05,10], then normalize the complete source vector to mean one.
The final normalized weights can exceed the raw clipping bounds.

All three task classifiers receive exactly that same saved weight vector and
reuse their source-selected parameters. `artifacts/importance_cache/` holds
domain fold IDs, coefficients, probabilities, counts, raw/clipped/final weights,
diagnostics and originating timings. Each weighted run contains
`adaptation_state.npz` and an `importance_reference.json`. Adaptation timing
includes transformation of the permitted target adaptation features and domain
cross-fitting. Shared cache timings retain their original values.

Effective sample size (ESS) measures concentration of the weights, not prediction
quality. ESS below 10% of source count warns but retains the fit. No target task
labels, pseudo-labels or target score tuning are used. Lightweight real fitting
diagnostics are in `reports/importance_weighting/`; raw weights/models stay local.

## Fit CORAL models

```powershell
.\.venv\Scripts\shiftlab.exe run --config configs/coral.yaml
```

This development job compares all three implemented methods and a majority
baseline on the three diagnostic transfers at seed 17: 30 fitted records.
It keeps target evaluation features and labels closed.

In the common source-preprocessed feature space, CORAL estimates source and
target adaptation means and unbiased covariances. It adds ridge 0.001 to each
covariance, uses symmetric eigenvalue roots with floor 1e-8, and computes
`A = Cs^(-1/2) @ Ct^(1/2)`. For row-vector observations, source fitting uses
`(X_source - source_mean) @ A + target_mean`. This is a centred CORAL variant
with target mean restoration. The product A is generally not symmetric.
Target prediction uses the saved source preprocessing and the fitted classifier;
the source alignment matrix is never applied to target rows.

`artifacts/coral_cache/` preserves A, means, original covariances, ridge, floor,
preprocessing identifier, ordered source/adaptation IDs, aligned source features,
covariance mismatch and numerical diagnostics. All three classifiers share the
same transform and source-selected parameters. Each CORAL run copies its numeric
`adaptation_state.npz` and records a `coral_reference.json`. Adaptation timing
includes target adaptation preprocessing, transform estimation and source alignment;
cache hits retain the originating component timings.

Mismatch uses the Frobenius difference from the original target adaptation
covariance, divided by its norm with a 1e-12 denominator floor. Regularisation
can leave residual mismatch; reduced mismatch does not establish better
prediction quality or identical distributions. The lightweight real-fit audit
is in `reports/coral_alignment/`; transforms and models remain local.

## Benchmark and execute the frozen study

The CPU runner keeps the declared preprocessing, grids, five seeds and one-thread
policy. Benchmark each diagnostic transfer in a fresh Python process before
starting the complete study:

```powershell
.\.venv\Scripts\python.exe tools/run_frozen_study.py benchmark --task gas_b01_b10
.\.venv\Scripts\python.exe tools/run_frozen_study.py benchmark --task har_official_group_transfer
.\.venv\Scripts\python.exe tools/run_frozen_study.py benchmark --task amazon_books_electronics
.\.venv\Scripts\shiftlab.exe study estimate --study-id <saved-study-id>
.\.venv\Scripts\python.exe tools/run_frozen_study.py execute
```

Each benchmark performs the complete source CV search, all three classifiers and
methods, and majority at seed 17. It saves measured preprocessing, tuning,
adaptation and training times, wall time, and the OS process-lifetime peak working
set. The GPU is unused. The runtime projection counts 30 shared source searches,
1,100 primary runs and 180 additional sensitivity runs. Its allowance for final
prediction, bootstrap analysis and variation is explicitly an estimate.

`artifacts/studies/<study-id>/study_lock.json` freezes configuration, source bytes,
dependency lock, archives, splits and the expected matrix before fitting. Final
evaluation requires all 1,280 saved fits and a separate completed-fit lock. The
execution tool blocks reads of held-out/private data during fitting, then invokes
the evaluator and saved-prediction analysis explicitly. Scientific results do not
depend on Streamlit.

Completed fits, source-CV candidates and evaluations are checkpoints. Repeat the
execution command to resume; verified artifacts retain their original timings
and remain immutable. `checkpoint.json`, `events.jsonl`, `completed_runs.csv` and
`missing_runs.csv` distinguish expected, fitted and evaluated counts and preserve
failed attempts. Use Ctrl+C for graceful cancellation. After a hard interruption,
inspect `writer.lock`, then use `shiftlab study recover --study-id <id>`: recovery
requires a confirmed dead owner and preserves abandoned attempts. It refuses a
live or uninspectable process. Source/dependency changes require a new audited
study rather than silently resuming an incompatible one.

Separate commands are available for `study run --config configs/study.yaml --resume`,
`study evaluate --study-id <id>`, `study summarize --study-id <id>` and
`study status --study-id <id>`. Summaries independently recompute scores from
predictions, check pairing, average bootstrap deltas over the five fitted seeds,
and resample whole evaluation groups 1,000 times with seed 2026. The primary
settings remain fixed. Tables, figures, negative changes and limitations are in
the immutable `analysis/` directory; its export manifest links exact run IDs and
input/output hashes. Seeds sharing one test set are not independent datasets.

## Research boundaries

The completed frozen study is saved in [reports/study](reports/study/README.md):
1,100 primary and 180 sensitivity evaluations across all 22 transfers and five
seeds, with 30 shared source searches. All 1,280 metric records independently
recompute from saved predictions. The full regression suite passed 193 tests;
the resume audit preserved hashes, sizes and timestamps of 28,729 immutable files.

Across 132 primary adaptation comparisons, mean macro F1 increased in 40 and
decreased in 92; 27 unadjusted group-bootstrap intervals included zero. These
dependent comparisons describe observed transfer, not independent statistical
trials. Saved tables retain negative changes, low-ESS warnings and all declared
sensitivity settings, with exact run IDs and checksums.

Fresh gas, HAR and Amazon benchmarks measured peak process working sets of
284, 687 and 517 MiB. The benchmark-based complete-study planning range was
2.98–3.97 hours from scratch, including a stated allowance for unmeasured costs.
It is a projection, not observed elapsed completion time. Two Windows progress-file
I/O interruptions were preserved and resumed; bounded retries apply only to four
mutable progress files. Classifier fitting failures: zero. The study engine,
objective, splits and primary settings stayed frozen. Publication figure layout
revisions trace to unchanged saved numerical inputs and preserve original exports.

Gas Sensor Array Drift, HAR Using Smartphones and the original four-domain Amazon
sentiment corpus are three independent datasets. LR, LinearSVC and RF are compared
under source-only, importance weighting and CORAL, with a source-majority baseline.
The full configuration declares 22 transfers and five seeds (17, 29, 43, 71, 101).
Other configuration subsets remain planned until supported by their own saved evidence.

Source labels and target adaptation features are permitted for fitting. Target
evaluation features and labels remain outside the training contract. Source
validation must fit preprocessing within each fold; source-selected parameters
are shared across paired methods. No method is assumed to improve performance.

## Layout and local materials

`src/shiftlab/` holds the shared engine; `app/` holds the interface; `tests/` holds
offline acceptance checks; `configs/` holds study contracts; `reports/` holds the
preserved proposal, reading plan and aggregate completed-study evidence. The synthetic fixture is development data and
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
