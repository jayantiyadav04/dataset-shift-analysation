# Five-minute demonstration and viva notes

This is an evidence-backed script and code walkthrough. Automated evidence checks
do not establish that the student rehearsed or understands the answers. Practise
the sequence locally and record personal corrections before the viva.

## Demonstration preparation

Start `python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1`
from the project environment. Select the saved completed study ending `f77d0`.
Keep the final PDFs and genuine screenshots in reports/workbench/screenshots
available offline. Confirm the local data/artifacts remain present; a public fresh
checkout intentionally has an honest empty state until acquisition and execution.
No full-study job is required during this demonstration.

| Time | Action | Explanation supported by saved evidence |
| --- | --- | --- |
| 0:00-0:40 | Overview, objective and counts | Evaluate when weighting/CORAL help classical models. 1,280 expected and completed evaluations; three independent datasets. |
| 0:40-1:20 | Datasets, gas Batch 1 to 2; then HAR participant roles | 445/622/622 source/adaptation/evaluation gas rows. HAR evaluation participants 12,13,18,20,24 never enter fitting. Show archive and split hashes. |
| 1:20-2:10 | Results, gas Batch 1 to 7, LR, primary methods | CORAL paired macro-F1 mean gain +41.284465 percentage points, 95% descriptive interval [+38.973957,+43.482547]. Show all five seed records and run IDs. |
| 2:10-3:00 | Diagnostics, weighting then CORAL | Explain ESS and clipping, then the covariance mismatch before/after. Small mismatch alone does not prove correct class decisions. |
| 3:00-3:50 | Results, gas Batch 1 to 2, RF | CORAL loses 34.075518 macro-F1 points. Preserve this finding. Then show gas Batch 8 SVM weighting, whose interval includes zero. |
| 3:50-4:30 | Report Exports, choose filters and explicitly build a bundle | Open CSV and manifest; trace one run to its local prediction CSV, configuration, scientific source hash and dependency lock. ZIP includes local split IDs, so the public package is a separate aggregate-only archive. |
| 4:30-5:00 | Summarise and disclose | Conditional gains, frequent negative transfer, limited independent domains. Explain the clean 50-fit reproduction and your actual contribution, including AI assistance. |

All numeric examples are in [interpretation_examples.csv](tables/interpretation_examples.csv).
For a failed live action, read the actual error and show an existing completed
artifact. Do not claim a job completed or a screenshot was live when it did not.

## Code walkthrough to practise

1. Open [split construction](../../src/shiftlab/data/splits.py) and
   [training view](../../src/shiftlab/data/training.py). Explain how frozen role IDs
   select source labels and unlabelled adaptation features. The trainer has no
   target-task-label input. HAR roles contain disjoint participants.
2. Open [density_ratios](../../src/shiftlab/adaptation/importance.py). Explain
   bounded probabilities, per-fold count correction, clipping and normalization
   in that order. For q=1/3, ns=80 and nt=40, raw r=1. Multiplying every weight by
   a common constant changes neither its relative influence nor ESS; normalized
   weights need not remain within the raw clipping interval.
3. Open [fit_coral](../../src/shiftlab/adaptation/coral.py). Point to centring,
   unbiased covariance, ridge, symmetric powers and source transformation.
   Explain why target inputs stay in target coordinates after common preprocessing.
4. Open [importance tests](../../tests/test_importance.py),
   [CORAL tests](../../tests/test_coral.py) and
   [study tests](../../tests/test_study.py). Choose the unequal-count calculation,
   independent covariance check and held-out-read refusal. Explain the failure
   each catches; neither method is required to improve a synthetic score.
5. Open [evaluator](../../src/shiftlab/evaluation/evaluator.py) and
   [analysis](../../src/shiftlab/evaluation/analysis.py). The completed-fit boundary
   permits evaluation only after fitting/configuration freeze. Scores are recomputed
   from saved predictions, with the target-present class rule fixed before resampling.
6. Open [build_bundle](../../app/exports.py). Trace filters to saved tables and
   run IDs, prediction hashes, split manifests and ZIP checksums. No estimator is
   fitted by an export or UI rerender. The public
   [package tool](../../tools/package_submission.py) excludes raw data and models.

## Conceptual answers tied to this implementation

| Question | Answer anchor |
| --- | --- |
| What is a domain? | A batch, participant population or product category within a dataset. The four Amazon categories are four domains within one dataset. |
| Why unsupervised adaptation? | Source task labels are permitted; adaptation data have no task labels available to the fitting contract. Source/target membership is known. |
| Why hold out evaluation features too? | Fitting a scaler, vocabulary, domain classifier or covariance on them changes the evaluated protocol, even without labels. |
| Why source-only tuning? | It prevents selecting parameters using target evaluation outcomes and keeps paired base models comparable. It may miss target-optimal settings. |
| Why out-of-fold weights? | Each scored source row is excluded from the discriminator fit. Actual fold source counts correct the membership sampling prior. |
| Why clipping and ESS? | Extreme weights can let a few rows dominate. ESS summarizes concentration; a warning is retained, not grounds for silently dropping a run. |
| What can weighting fail to fix? | Absent source support and changed conditional labels. A calibrated-looking domain classifier does not establish the covariate-shift assumption. |
| What does CORAL change? | Source mean/covariance geometry after common preprocessing. It does not align labels or guarantee matching class-conditional distributions. |
| Why ridge? | Covariance can be singular or ill-conditioned, especially with limited samples. The declared ridge/floor stabilize matrix powers. |
| What is macro F1 here? | Equal-weight average of per-class F1 over classes present in the full held-out task, with zero_division=0. Absent target classes are reported. |
| Why participant grouping? | Overlapping HAR windows from one person are dependent; mixing that person between roles or validation folds creates leakage. |
| Why not mix random gas batches? | It weakens the intended temporal source-to-later-batch shift and asks a different question. Within-batch adaptation/evaluation is still a holdout, not forecasting. |
| Are five seeds five datasets? | No. They reuse observations. Means/SD describe seed variability; they cannot establish independent-domain significance. |
| What does the interval cover? | Paired whole-group test resampling conditional on the five fitted models and observed domains. It does not include retraining or population-wide domain uncertainty. |
| Does an interval including zero prove equivalence? | No. For gas Batch 8 SVM weighting it indicates uncertainty about a small mean change. No equivalence test was declared. |
| Give an actual failure case. | Gas Batch 2 RF CORAL has -34.075518 macro-F1 points. The code remains valid if covariance alignment hurts prediction. Two separate Windows progress-file interruptions were also recorded and resumed. |
| What are the sensitivity controls? | Clipping 5/10/20 and ridge 0.0001/0.001/0.01 on three predeclared tasks. The primary setting was not changed after viewing target scores. |
| How is time reported? | Per-pipeline time includes originating cached component costs and excludes shared source tuning. Sum unique work for actual compute; do not sum reused costs and call it wall time. |
| Was the GPU used? | No. Classical scikit-learn computation used one CPU job and one numerical thread. The RTX 4060 was not part of the measured training path. |
| What was reproduced? | One real gas transfer, all five seeds and 50 new fits, exact ordered prediction/metric/parameter agreement, verified existing local data and fresh locked environment. |
| What is new? | This controlled cross-application comparison, auditable evidence and local tool are project contributions. CORAL and importance weighting were not invented here. |
| What did you personally do? | Review the factual disclosure. State only decisions, coding, testing, reading and interpretation you actually performed. Codex supplied substantial implementation and drafting support. |
| What remains limited? | Few independent domains, five HAR evaluation participants, shared Amazon categories, possible remaining dependence and unadjusted intervals. More independent domains are a future study. |

## Student rehearsal record

- [ ] Run the five-minute sequence and time it personally.
- [ ] Explain the count-correction calculation without reading the answer.
- [ ] Explain a covariance calculation and the leakage boundary in code.
- [ ] Open one prediction artifact from a reported run ID and recompute its metric.
- [ ] Explain the actual negative and uncertain examples without claiming significance.
- [ ] Review each contribution statement and add only work actually completed.
- [ ] Add verified identity/institution details and adapt to any faculty template.

These boxes intentionally remain unchecked. A passing automated audit cannot
certify student understanding or faculty approval.
