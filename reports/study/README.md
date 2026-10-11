# Saved study evidence

Study `a05bea2f175b9d81091a413637276d174c378efaf31d1c539430e98055df77d0` has **1,100 primary and 180 sensitivity evaluations**, 30 shared source searches and complete declared coverage. All 1,280 metric records independently recompute from saved predictions. There are 132 primary and 36 sensitivity paired comparisons, each using five fitted seeds and 1,000 grouped bootstrap replicates.

`benchmarks.csv` records three fresh CPU tasks with separate final preprocessing, source tuning (including fold preprocessing), classifier training, adaptation and process peak RAM. `runtime_estimate.json` is a projection from these measurements, not an observed completion. `execution_time.json` records the observed invocation duration when available, separate from the first completed fitting boundary. Interrupted attempts and idle time are excluded from that invocation measurement. CPU policy remains one job/thread; the RTX GPU is unused.

Observed operational progress-file interruptions: 2. Saved fits and evaluations were preserved and resumed under the same frozen settings. `operational_events.json` records incident/log hashes and bounded progress-only replacement retries. These are metadata I/O interruptions, not classifier failures.

## Observed adaptation conditions

Importance weighting shows 13 positive and 53 negative mean macro F1 changes across 66 model/task comparisons. 27 comparisons have mean ESS below 10% of source size; 17 of those show observed negative transfer. These model/task counts are dependent descriptive evidence, not independent statistical trials.

CORAL reduces mean adaptation-pool covariance mismatch in 66 of 66 model/task comparisons. 39 of those nevertheless show a negative mean macro F1 change. Covariance agreement measures marginal feature geometry and does not guarantee that source labels transfer correctly. Models share the transform within each transfer/seed; these rows are dependent.

`conditions.csv` ties every primary comparison to the actual transfer, model, sample/group counts, ESS or covariance diagnostics, paired delta, conditional interval and exact run IDs. These are observed associations. The study does not establish causal explanations, label-conditional invariance, optimal target settings or a universal winning method.

## Examples for the viva

- Negative: `gas_b01_b02` / `rf` / `coral`: mean macro F1 change -34.076 percentage points, unadjusted interval [-36.619, -18.709]. Exact paired IDs are in findings.json.
- Positive: `gas_b01_b07` / `lr` / `coral`: mean macro F1 change 41.284 percentage points, unadjusted interval [38.974, 43.483]. Exact paired IDs are in findings.json.
- Uncertain: `gas_b01_b08` / `linear_svm` / `importance_weighting`: mean macro F1 change -0.149 percentage points, unadjusted interval [-1.267, 0.828]. Exact paired IDs are in findings.json.

## Limits and interpretation

Five seeds share the same target rows; they are not five independent datasets. Resampling holds the fitted models fixed and includes entire participants/products/duplicate-safe measurement groups with multiplicity. HAR has only five evaluation participants. Gas temporal dependence can remain, and overlapping Amazon transfers are dependent. Accuracy and macro F1 intervals are unadjusted: 168 paired comparison rows contain 168 intervals per metric (336 total). No naive seed-level significance test is reported.

Mean negative deltas are observed negative transfer. Intervals including zero communicate uncertainty; they do not prove equivalence. Macro class sets are those present in each full held-out target test. Task sizes and class composition vary. Dataset summaries weight tasks equally within each dataset; descriptive overall summaries weight gas, HAR and Amazon equally.

Every predeclared clipping/ridge value appears in ablations.csv; primary clipping 10 and ridge 0.001 are reused. Target scores were not used to select a new primary setting. Timing tables retain originating cached component costs and exclude shared tuning from per-pipeline runtime; the complete-study estimate counts shared searches/caches once.

This folder contains aggregate scientific evidence and hashes. Raw reviews/features, sample-level true/predicted labels, serialized models, full environment paths and local working logs remain in ignored storage. The full local export manifest links prediction hashes and run IDs. The original objective and proposal abstract remain unchanged; no grade, novelty or publication guarantee is implied.

Study design and method/dataset references: [research design](../research_design.md), [literature matrix](../literature_review_matrix.md) and [dataset cards](../data_preparation/).
