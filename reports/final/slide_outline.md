# Ten-slide presentation content

These are ready-to-edit slide contents and speaker notes. No faculty format was
supplied. Add verified student/institution details only after personal review.
Use the linked saved figures and actual screenshots; do not substitute generated
screenshots. Sources R01-R11 and D01-D03 refer to [verified references](references.md).

## 1. Evaluating Domain Adaptation Techniques for Classical Machine Learning Models

**Exact objective (display verbatim):**

To evaluate whether importance weighting and feature alignment using CORAL improve the performance of Logistic Regression, Support Vector Machine, and Random Forest when source and target data follow different distributions. The study will compare these techniques against models trained without adaptation across three public datasets, using accuracy, macro F1 score, and computational time. It will identify the conditions under which adaptation improves performance, provides little benefit, or reduces prediction quality.

Speaker note: The question is when adaptation helps. The contribution is a
controlled comparison and reproducible workbench; the methods are established.

## 2. Domain shift and the permitted information

Show this flow: source features and labels -> source-selected preprocessing/model;
target adaptation features -> weighting or CORAL; frozen fitted model -> held-out
target evaluation -> saved predictions and paired analysis.

Explain with gas Batch 1 to Batch 2: 445 source rows, 622 adaptation rows and
622 separate evaluation rows. The domain classifier predicts source/target
membership, not the gas class. Evaluation features and labels cannot enter fitting.

Evidence: [split counts](tables/split_counts.csv), report sections 3-5.

## 3. Three independent datasets, 22 transfers

| Dataset | Retained observations | Domain design |
| --- | ---: | --- |
| UCI gas drift | 13,910 | Batch 1 to each of Batches 2-10: 9 transfers |
| UCI smartphone HAR | 10,299 | 21 source, 4 adaptation and 5 evaluation participants: 1 transfer |
| Original Amazon sentiment | 70,509 | Books, DVDs, electronics, kitchen: 12 ordered transfers |

Amazon excludes 8,170 duplicate bodies globally and one additional adaptation
duplicate for books-to-DVD. Three independent datasets are never mixed into a
single classifier. Cite D01-D03 and the three [data cards](data_cards/gas.md).

## 4. Fair comparison and coverage

LR, LinearSVC and RF share source-selected parameters and the same permitted
representation across source-only, weighting and CORAL. Retain source-majority.
Fit preprocessing inside source validation folds. Text uses source TF-IDF and SVD.

Five seeds: 17, 29, 43, 71, 101. Completed coverage: 1,100 primary evaluations
(990 classifier and 110 majority) plus 180 predeclared sensitivity evaluations.
Source selection produced 30 searches and 1,500 candidate-fold scores.

Speaker note: Five seeds share test observations. They are not five independent
datasets. All 1,280 fits were required before the full-study evaluator opened data.

## 5. Importance weighting

Let q be the out-of-fold probability of target membership. Correct the odds using
the actual source count ns in that discriminator fold and target count nt:
`r = q / (1-q) * ns / nt`. Bound probabilities, clip raw ratios to [0.05, 10], then
divide by the mean clipped source weight. Pass the weights to the task estimator.

Worked explanation: q = 1/3, ns = 80, nt = 40 gives r = 1; sample-count imbalance
alone must not imply a different target density. ESS = sum(w)^2 / sum(w^2).
Weight concentration and incorrect shift assumptions can hurt prediction.
Density-ratio motivation: R02, R03, R11. The project's choices are explicit.

## 6. Centred, regularised CORAL

Using unbiased source and adaptation covariance estimates, add 0.001 times the
identity. Compute `A = Cs^(-1/2) Ct^(1/2)` by symmetric eigendecomposition with
eigenvalue floor 1e-8. Train on `(Xs-ms) A + mt` and source labels. Target data
receive the common source preprocessing, not the source alignment matrix.

Show the saved [covariance diagnostic](../study/figures/coral_covariance_mismatch.png).
Reduced covariance mismatch is a geometric check and does not imply better
classification. Mean restoration is the declared centred variant. Cite R01.

## 7. What happened

Show [gas temporal results](../study/figures/gas_temporal_macro_f1.png) and the
[direction counts](tables/direction_counts.csv). Error bars on the temporal plot
are sample SD across five seeds, not bootstrap confidence intervals.

Across 132 primary adaptation comparisons, 40 mean macro-F1 changes are positive
and 92 negative; 27 paired descriptive intervals include zero. Gas CORAL improves
LR and linear SVM dataset means by 22.576452 and 11.347881 percentage points.
All six HAR adaptation comparisons decline. Most Amazon comparisons also decline;
RF with CORAL has a small positive dataset mean despite mixed task effects.

Speaker note: Read the entire task distribution alongside dataset averages.
Do not claim a universal winner or statistical significance from these counts.

## 8. Negative transfer, sensitivity and cost

Show [paired examples](tables/interpretation_examples.csv): gas Batch 2, RF with
CORAL changes macro F1 by -34.075518 percentage points; its descriptive interval
is [-36.618668, -18.709011]. On gas Batch 8, SVM weighting changes F1 by -0.148662
points with an interval spanning zero, so equivalence is not established.

Primary clipping 10 and ridge 0.001 were frozen. Sensitivity uses clipping
5/10/20 and ridge 0.0001/0.001/0.01 on three predeclared tasks. Preserve all results;
do not promote the best target result into a new primary setting.

Benchmarked peak process RAM: gas 284, HAR 687, Amazon 517 MiB (rounded).
Planning projection: 2.98-3.97 hours from scratch, not observed full-study wall time.
Two Windows progress-file interruptions were recorded and resumed; no classifier
fit failed. See [runtime evidence](../study/execution_time.json) and report section 7.

## 9. Working application and reproduction

Use actual [Overview](../workbench/screenshots/1280_overview.png) and
[result trace](../workbench/screenshots/1280_result_trace.png) screenshots.
The six pages read saved evidence. Jobs and exports require explicit actions;
rerendering does not train. Run IDs connect tables to predictions and settings.

A fresh locked environment and clean checkout performed 50 new gas Batch 1 to 2
fits. All ordered predictions, metric dictionaries and selected parameters matched
the original study exactly. This is one-task reproduction, not a second full study.
The [receipt](../reproduction/verification.json) records measured time and RAM.

## 10. Conclusion, limitations and disclosure

Adaptation is conditional: CORAL helps several gas transfers, while both methods
often reduce HAR and Amazon performance. Support mismatch, class-conditional
changes and representation limitations are plausible explanations, not separately
established causal findings here. Only five HAR evaluation participants and
overlapping Amazon tasks limit generalization. Intervals are unadjusted and
conditional on fitted models and observed domains.

Name your personally completed work accurately after reviewing the
[assistance disclosure](assistance_and_contributions.md). Codex supplied substantial
implementation, testing and writing assistance. Do not claim personal independent
coding or reading without having done it. End with R01, R02, R05, R09, R11 and
D01-D03; the complete report contains 20 references. Student rehearsal is pending.
