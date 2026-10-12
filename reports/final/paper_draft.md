# Evaluating Domain Adaptation Techniques for Classical Machine Learning Models

Research paper draft for faculty review

## Abstract

Distribution shift can reduce the reliability of classical classifiers. This study evaluates domain-classifier importance weighting and centred, regularised CORAL against source-only Logistic Regression, linear Support Vector Machine and Random Forest models. Experiments use UCI Gas Sensor Array Drift, smartphone Human Activity Recognition and the original four-domain Amazon sentiment corpus, covering 22 transfers and five seeds. Source-only validation selects shared hyperparameters; separate target adaptation features support adaptation, while held-out target observations remain excluded until evaluation. The completed study contains 1,100 primary and 180 sensitivity evaluations. Across 132 primary adaptation comparisons, mean macro F1 improves in 40 and declines in 92; 27 descriptive group-bootstrap intervals include zero. CORAL benefits several gas transfers but adaptation generally harms HAR and Amazon performance. Accuracy, runtime, weight concentration and covariance diagnostics contextualise these outcomes. Shared test observations and few independent domains limit inference. The contribution is reproducible comparative evidence and an independently auditable local research workbench.

## 1 Introduction

Classical classifiers are useful when labeled data and computing resources are
limited, but a change between training and deployment inputs can undermine their
reliability. Adaptation may exploit unlabeled target observations, although
improvement depends on assumptions about support and label relationships. This
study asks whether importance weighting and covariance alignment help LR, linear
SVM and RF under three distinct applications. It reports negative transfer and
computational cost rather than selecting a method only from successful examples.

The frozen objective is:

To evaluate whether importance weighting and feature alignment using CORAL improve the performance of Logistic Regression, Support Vector Machine, and Random Forest when source and target data follow different distributions. The study will compare these techniques against models trained without adaptation across three public datasets, using accuracy, macro F1 score, and computational time. It will identify the conditions under which adaptation improves performance, provides little benefit, or reduces prediction quality.

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

| Dataset | Method | Pairs | Positive | Negative | CI includes 0 |
| --- | --- | --- | --- | --- | --- |
| amazon | CORAL | 36 | 8 | 28 | 7 |
| amazon | Weighting | 36 | 3 | 33 | 4 |
| gas | CORAL | 27 | 19 | 8 | 5 |
| gas | Weighting | 27 | 10 | 17 | 9 |
| har | CORAL | 3 | 0 | 3 | 1 |
| har | Weighting | 3 | 0 | 3 | 1 |

Table 1 counts task-model comparisons within each dataset and adaptation method.
Tasks share data and these counts are descriptive. CORAL's mean gas improvements
for LR and linear SVM are 22.576452 and 11.347881 percentage points across nine
transfers. All six primary HAR adaptations reduce mean macro F1, and most Amazon
comparisons also decline. Amazon RF with CORAL has a small positive dataset mean,
illustrating why task effects must accompany an overall ranking.

| Example | Transfer | Model | Method | Delta pp | 95% interval pp |
| --- | --- | --- | --- | --- | --- |
| positive | Gas 1→07 | LR | CORAL | +41.284465 | [+38.973957, +43.482547] |
| uncertain | Gas 1→08 | Linear SVM | Weighting | -0.148662 | [-1.266615, +0.828300] |
| negative | Gas 1→02 | RF | CORAL | -34.075518 | [-36.618668, -18.709011] |

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

A clean checkout and fresh virtual environment reproduced gas Batch 1 to Batch 2 using all five seeds and 50 newly fitted models/baselines. All 50 ordered prediction vectors, metric dictionaries and selected parameters exactly matched the original study on 622 evaluation rows per run. Fitting took 204.295694 seconds; fitting, evaluation and comparison took 245.421423 seconds. Peak process working set was 216.0625 MiB. No forbidden held-out training reads occurred. The original archive/split were verified and reused; new caches/models/predictions were written separately. The detailed reproduction record preserves the exact executed driver and logs.

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

## References

Checked against the primary publisher, proceedings, author or official software
source on 11 October 2026. Method and dataset claims below are attributed to their
original sources. Project scores come from the saved local study, not these papers.

R01. Sun, B., Feng, J. and Saenko, K. (2016). Return of Frustratingly Easy Domain
Adaptation. Proceedings of AAAI, 30(1). DOI: 10.1609/aaai.v30i1.10306.
[Publisher and paper](https://ojs.aaai.org/index.php/AAAI/article/view/10306).

R02. Sugiyama, M., Krauledat, M. and Müller, K.-R. (2007). Covariate Shift
Adaptation by Importance Weighted Cross Validation. JMLR, 8, 985-1005.
[Paper](https://www.jmlr.org/papers/v8/sugiyama07a.html).

R03. Huang, J., Smola, A. J., Gretton, A., Borgwardt, K. M. and Schölkopf, B.
(2007). Correcting Sample Selection Bias by Unlabeled Data. Advances in Neural
Information Processing Systems 19, proceedings of NIPS 2006.
[Paper](https://proceedings.neurips.cc/paper_files/paper/2006/file/a2186aa7c086b46ad4e8bf81e2a3a19b-Paper.pdf).
Author order follows the paper PDF; the online proceedings metadata orders authors differently.

R04. Ben-David, S., Blitzer, J., Crammer, K., Kulesza, A., Pereira, F. and
Vaughan, J. W. (2010). A theory of learning from different domains. Machine
Learning, 79, 151-175. DOI: 10.1007/s10994-009-5152-4.
[Article](https://link.springer.com/article/10.1007/s10994-009-5152-4).
Online publication was in 2009; the cited journal volume is 2010.

R05. Blitzer, J., Dredze, M. and Pereira, F. (2007). Biographies, Bollywood,
Boom-boxes and Blenders: Domain Adaptation for Sentiment Classification.
Proceedings of ACL, 440-447.
[Paper](https://aclanthology.org/P07-1056/).

R06. Fernando, B., Habrard, A., Sebban, M. and Tuytelaars, T. (2013).
Unsupervised Visual Domain Adaptation Using Subspace Alignment. ICCV, 2960-2967.
[Proceedings](https://openaccess.thecvf.com/content_iccv_2013/html/Fernando_Unsupervised_Visual_Domain_2013_ICCV_paper.html).

R07. Pan, S. J. and Yang, Q. (2010). A Survey on Transfer Learning. IEEE
Transactions on Knowledge and Data Engineering, 22(10), 1345-1359.
DOI: 10.1109/TKDE.2009.191.
[Publisher](https://ieeexplore.ieee.org/document/5288526);
[author manuscript](https://www.cse.ust.hk/~qyang/Docs/2009/tkde_transfer_learning.pdf).

R08. Vergara, A., Vembu, S., Ayhan, T., Ryan, M. A., Homer, M. L. and
Huerta, R. (2012). Chemical gas sensor drift compensation using classifier
ensembles. Sensors and Actuators B: Chemical, 166-167, 320-329.
DOI: 10.1016/j.snb.2012.01.074.
[Publisher](https://www.sciencedirect.com/science/article/pii/S0925400512002018).

R09. Anguita, D., Ghio, A., Oneto, L., Parra, X. and Reyes-Ortiz, J. L. (2013).
A Public Domain Dataset for Human Activity Recognition Using Smartphones.
ESANN, 437-442.
[Proceedings paper](https://www.esann.org/sites/default/files/proceedings/legacy/es2013-84.pdf).

R10. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python.
JMLR, 12, 2825-2830.
[Paper](https://www.jmlr.org/papers/v12/pedregosa11a.html).

R11. Bickel, S., Brückner, M. and Scheffer, T. (2009). Discriminative Learning
Under Covariate Shift. JMLR, 10, 2137-2155.
[Paper](https://www.jmlr.org/papers/v10/bickel09a.html).

D01. Vergara, A. (2012). Gas Sensor Array Drift Dataset. UCI Machine Learning
Repository, dataset 224. DOI: 10.24432/C5RP6W.
[Dataset](https://archive.ics.uci.edu/dataset/224/gas+sensor+array+drift+dataset).

D02. Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L. and Parra, X. (2013).
Human Activity Recognition Using Smartphones. UCI Machine Learning Repository,
dataset 240. DOI: 10.24432/C54S4K.
[Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones).

D03. Dredze, M. Multi-Domain Sentiment Dataset, original four-domain release.
Author-hosted distribution, page last updated 2 August 2007; cite R05.
[Dataset and separate books file](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/index2.html).

S01. Scikit-learn developers. LogisticRegression, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html).

S02. Scikit-learn developers. LinearSVC, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.svm.LinearSVC.html).

S03. Scikit-learn developers. RandomForestClassifier, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html).

S04. Scikit-learn developers. StratifiedGroupKFold, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html).

S05. Scikit-learn developers. TfidfVectorizer, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html).

S06. Scikit-learn developers. TruncatedSVD, version 1.9.1 documentation.
[API](https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).

## Verification scope

R01-R05, R07, R09-R11 were checked using primary full-paper text, including the
method or dataset sections used here. R06 was checked using the indexed CVF
proceedings abstract and metadata; direct HTML/PDF access returned HTTP 403.
R08 was checked using the indexed publisher abstract, introduction and method
preview; direct publisher access also returned HTTP 403. No unseen experimental
detail from either restricted paper is claimed. IEEE's page required JavaScript;
the author-hosted R07 manuscript supplied the text. The DOI, volume and page
metadata identify the final publication separately from that manuscript.

D01-D03 and S01-S06 were opened directly. The online scikit-learn documentation
identified itself as 1.9.1, matching the installed scientific environment.
The reading matrix records limitations and relevance; it does not claim that
the student independently read or verified every source.

