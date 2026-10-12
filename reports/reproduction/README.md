# Clean environment reproduction

The real gas Batch 1 to Batch 2 transfer was rebuilt in a fresh virtual environment
and clean checkout of commit `9478f4f667b7628e348332bd35cbdf22f07f5231`.
All 50 primary fits were new: LR, LinearSVC and RF under three methods plus majority,
at seeds 17, 29, 43, 71 and 101. No original fitted model or feature cache was reused.

All 50 runs reproduced the original ordered sample IDs, group IDs, labels and
predicted classes exactly. All metric dictionaries and selected model parameters
also matched exactly. Each evaluation has 622 rows. The checksum-verified original
archive and frozen split were reused; this is not a fresh network-download or
cross-platform test. No held-out file was opened during the instrumented fitting
stage. Only the evaluator opened it after all 50 fits were frozen.

Measured fresh fitting: 204.295694 seconds. Total fitting, evaluation and comparison:
245.421423 seconds. Peak process working set: 226,557,952 bytes (216.0625 MiB).
Times are new measurements and are not expected to equal the original study times.
The two CSV file hashes differ when run IDs differ; the actual ordered prediction
content was compared after excluding only provenance columns that identify the
new run. The verification receipt records both prediction hashes and run IDs.

## Observed installation and commands

```powershell
.\.venv\Scripts\python.exe -m venv .venv-reproduction
.\.venv-reproduction\Scripts\python.exe -m pip install -r requirements-lock.txt --disable-pip-version-check --progress-bar off
git clone --no-hardlinks . artifacts/reproduction_checkout
.\.venv-reproduction\Scripts\python.exe -m pip install --no-deps --no-build-isolation -e artifacts/reproduction_checkout --disable-pip-version-check --progress-bar off
.\.venv-reproduction\Scripts\python.exe tools/lock_environment.py --check
.\.venv-reproduction\Scripts\python.exe -m pip check
.\.venv-reproduction\Scripts\python.exe tools/reproduce_task.py --checkout artifacts/reproduction_checkout
```

All 64 locked installed pins matched, and pip reported no broken requirements.
Import inspection confirmed shiftlab came from the clean checkout. An earlier
noneditable local installation caused the lock utility to reject its local URL;
the utility requires an editable project so only registry dependency pins are
compared. The editable clean-checkout installation resolved this before fitting.
That earlier install log remains local. The original environment was unchanged.

The actual executed tool is preserved as [executed_driver.py](executed_driver.py),
whose hash matches [request.json](request.json). The reusable tool subsequently
received only line wrapping and a relative-command-path correction: the request's
short command field omitted `artifacts/`, while the actual command above is correct.
No fitting, evaluation or comparison logic changed. Preserve this discrepancy
instead of rewriting the original receipt. The measured results and full
comparison records are in [verification.json](verification.json); exact installed
pins are in [installed_dependencies.txt](installed_dependencies.txt).

Original local logs: artifacts/reproduction_install.log,
artifacts/reproduction_package_install.log, artifacts/reproduction_editable_install.log
and artifacts/reproduction_execution.log. Fresh models, source-CV records, all
predictions and the completed-fit lock remain inside the ignored clean checkout's
artifacts directory. Only aggregate verification and the executed source snapshot
are published. Rerunning requires a new empty checkout; it never overwrites this
evidence or the original complete study.
