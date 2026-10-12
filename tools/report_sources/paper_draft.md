# Evaluating Domain Adaptation Techniques for Classical Machine Learning Models

Research paper draft for faculty review

## Abstract

{{ABSTRACT}}

## 1 Introduction

Classical classifiers are useful when labeled data and computing resources are
limited, but a change between training and deployment inputs can undermine their
reliability. Adaptation may exploit unlabeled target observations, although
improvement depends on assumptions about support and label relationships. This
study asks whether importance weighting and covariance alignment help LR, linear
SVM and RF under three distinct applications. It reports negative transfer and
computational cost rather than selecting a method only from successful examples.

The frozen objective is:

{{OBJECTIVE}}

The contribution is a controlled, reproducible comparison of established methods
and an application that exposes the exact evidence behind each displayed result.
No new adaptation algorithm or publication acceptance is claimed.

## 2 Related work

Discriminative estimation links domain-membership probabilities to density ratios
[R11]. Importance-weighted validation [R02] and kernel mean matching [R03] are
related approaches, but neither is implemented here. CORAL aligns covariance
statistics [R01], whereas subspace alignment uses eigenvector mappings [R06].
Adaptation theory includes labeling compatibility as well as distribution
divergence [R04]; reduced marginal discrepancy alone cannot guarantee target
accuracy. Transfer-learning terminology and negative transfer are reviewed in
R07. Dataset context comes from the original Amazon sentiment [R05], gas drift
[R08] and smartphone HAR [R09] papers. Their training protocols differ from ours,
so published scores are not directly compared with this study.

## 3 Datasets and experimental protocol

Gas Sensor Array Drift uses 13,910 retained measurements with 128 features and
nine transfers from Batch 1 to Batches 2-10. Each target batch is divided into
separate adaptation and evaluation groups using split seed 2026. HAR uses 10,299
retained windows and 561 features. Its source contains 21 official training
participants; four official test participants form adaptation and the other five
form evaluation. Amazon uses the original four categories, including the separate
books unlabeled file, with all twelve ordered transfers. Deduplication retains
70,509 of 78,679 records; only review bodies enter the text model.

Training uses source task labels and target adaptation features. Target evaluation
features and labels are unavailable to preprocessing, model selection, weighting
and covariance estimation. A completed-fit lock authorizes the final evaluator.
Source scalers and, for Amazon, TF-IDF followed by at most 128 SVD components are
refitted inside each source CV fold. All methods share the permitted representation
and source-selected base hyperparameters. HAR folds preserve whole participants.

LR and LinearSVC search C in {0.1, 1, 10}. RF uses 200 trees and four combinations
of depth unlimited or 20 and minimum leaf size 1 or 3. Five source folds optimize
macro F1. The study uses seeds 17, 29, 43, 71 and 101, a source-majority baseline,
and a single CPU job with one numerical thread. Scikit-learn 1.9.1 [R10, S01-S06]
is installed within the actual 64-package lock.

## 4 Adaptation and analysis

Importance weighting fits a logistic domain classifier with C 1 and five source
cross-fitting folds. For target-membership probability q, source fold-training
count ns and adaptation count nt, the ratio is `q/(1-q) * ns/nt`. Probabilities
are bounded to [0.001, 0.999], ratios to [0.05, 10], then weights are normalized
to source mean one. All task classifiers receive sample weights. ESS measures
weight concentration and warnings are retained.

The centred CORAL implementation adds ridge 0.001 to unbiased source and target
adaptation covariances, computes `A = Cs^(-1/2) Ct^(1/2)` by symmetric
eigendecomposition with floor 1e-8, and trains on `(Xs-ms)A+mt`. Only the source
features receive this alignment transform. Mean restoration and the chosen ridge
are explicitly disclosed implementation choices.

Metrics are calculated from saved predictions. Macro F1 uses the full target
task's present classes with zero_division 0. Paired deltas share evaluation rows,
base parameters and seeds. Descriptive 95% percentile intervals use 1,000 paired
group resamples, averaging deltas over five fitted seeds without refitting.
Groups are HAR participants, Amazon products where available and duplicate-safe
gas measurements. These unadjusted intervals describe conditional test-sample
uncertainty, not independent replications across seeds.

## 5 Results

All 1,100 primary and 180 additional sensitivity evaluations completed. Thirty
source searches generated 1,500 candidate-fold scores. Among 132 primary
adaptation comparisons, 40 mean macro-F1 changes are positive and 92 negative;
27 corresponding intervals include zero. There were zero classifier fitting
failures. Two operational progress-file errors were retained and resumed.

{{DIRECTION_TABLE}}

Table 1 counts task-model comparisons within each dataset and adaptation method.
Tasks share data and these counts are descriptive. CORAL's mean gas improvements
for LR and linear SVM are 22.576452 and 11.347881 percentage points across nine
transfers. All six primary HAR adaptations reduce mean macro F1, and most Amazon
comparisons also decline. Amazon RF with CORAL has a small positive dataset mean,
illustrating why task effects must accompany an overall ranking.

{{EXAMPLE_TABLE}}

Table 2 uses saved five-seed paired macro-F1 means and group-bootstrap intervals
in percentage points. The examples represent a gain, an uncertain small loss and
a large loss. Their exact source/adapted run IDs appear in the accompanying
claim-trace file and complete paired table.

![Gas temporal results](../study/figures/gas_temporal_macro_f1.png)

Figure 1. Gas Batch 1 to Batches 2-10, macro F1 in fractions. Points show primary
five-seed means; error bars show sample seed SD. Tasks share the source batch.
The saved figure-trace manifest identifies the exact runs behind every series.

Computational time includes preprocessing, adaptation, classifier fit and target
prediction, with source search reported separately. Representative gas, HAR and
Amazon benchmarks measured peak process working sets of approximately 284, 687
and 517 MiB. The benchmark-derived full-study planning estimate was 2.98-3.97 hours,
not a measured total elapsed runtime. Cached component costs are retained and
must not be summed repeatedly as unique study computation.

Sensitivity retains clipping 5, 10 and 20 and ridge 0.0001, 0.001 and 0.01 on the
three declared diagnostic tasks. Gas linear SVM macro F1 increases from 0.319973
to 0.354481 between the extreme clipping settings, while HAR RF decreases from
0.934667 to 0.923518. Primary settings remain fixed after seeing these results.
Lower covariance mismatch and larger ESS are diagnostics rather than automatic
certificates of better classification.

## 6 Reproducibility and limitations

{{REPRODUCTION}}

The source package provides immutable split identities, configurations, exact
versions, checksummed traces, complete result tables and real dashboard screenshots.
Raw archives and serialized models are excluded from distribution. Acquisition
instructions explain the documented source terms. Fresh-environment reproduction
uses verified local data and the same hardware/OS, so it does not establish
portability across all platforms.

Source-only tuning may differ from target-optimal selection. Density-ratio support
and stable conditional-label assumptions may fail; CORAL does not align all
class structure. Amazon text uses a deliberately bounded common representation.
There are only three applications and five HAR evaluation participants. Shared
test data, overlapping Amazon tasks, possible within-batch gas dependence,
varying target-present class sets and unadjusted intervals limit inference.

## 7 Conclusion

Adaptation produced substantial gains in some gas transfers but was often
ineffective or harmful elsewhere. The evidence favors evaluating a matched
source-only baseline, preserving negative transfer and inspecting assumptions
and computational cost. It does not support a universal adaptation winner.
Future independent-domain or target-budget studies require a new declared
protocol, rather than post hoc changes to these primary settings.

## Assistance and submission status

Codex supplied substantial implementation, experiment execution, checks, writing
and document-review assistance, as detailed in the companion contribution record.
The student's independent review and viva rehearsal are not recorded as completed.
This is a neutral academic draft for faculty review; no target venue, author
approval, institutional endorsement or publication status has been invented.

{{REFERENCES}}
