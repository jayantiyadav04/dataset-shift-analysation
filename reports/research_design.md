# Phase 1 research design

This is a pre-experiment protocol summary. No dataset has been downloaded and no
real target performance has been evaluated. The objective and 150-word proposal
abstract are preserved in `proposal.md` and `shiftlab.scope`.

## Research questions

1. Does importance weighting or CORAL change target macro F1 relative to the paired
   source-only classifier, and what happens to accuracy and computational time?
2. How do those changes differ across temporal gas batches, disjoint HAR participant
   populations and Amazon product categories?
3. How are observed changes associated with weight concentration, covariance
   mismatch and computational cost?

## Hypotheses and interpretation

H1: Benefits depend on the domain pair; there may be positive, negligible or
negative transfer. H2: Weak source-target overlap can concentrate importance weights
and produce unstable estimates. H3: Reducing covariance mismatch need not improve
classification. These are hypotheses, not measured findings or causal conclusions.

Importance weighting has a covariate-shift motivation: source support must cover
the relevant target region and the conditional label relationship is assumed
unchanged. Real datasets can violate those assumptions. This study uses a domain
classifier with source cross-fitting and prior-corrected odds; it does not implement
IWCV or kernel mean matching. See the [IWCV paper](https://www.jmlr.org/papers/v8/sugiyama07a.html)
and the [KMM paper](https://proceedings.neurips.cc/paper/2006/hash/a2186aa7c086b46ad4e8bf81e2a3a19b-Abstract.html)
as related work.

CORAL aligns second-order feature statistics. The planned implementation centres
source features, applies the source-to-target transformation and restores the
adaptation mean; target features remain in the source-preprocessed target coordinate
system. The original method is attributed to [Sun, Feng and Saenko](https://ojs.aaai.org/index.php/AAAI/article/view/10306).
This reproduced algorithm is not claimed as a novel contribution.

## Frozen comparisons

The configuration contains nine gas Batch 1 to later-batch transfers, one HAR
official participant-group transfer and twelve ordered Amazon category transfers.
Each task has LR, LinearSVC and RF under source-only, importance weighting and CORAL,
with seeds 17, 29, 43, 71 and 101. This declares 990 main fits, 110 majority
evaluations and 180 additional predeclared sensitivity fits. These counts are
configuration arithmetic and do not report completed work.

Gas follows the temporal direction; HAR keeps complete participants together;
Amazon uses review bodies with label-bearing fields removed and duplicate pools
separated. Split seed 2026 is independent of model seeds. Preprocessing is fitted
within source CV folds, and selected source parameters are reused across methods.
No target evaluation feature or label enters fitting or tuning. Target adaptation
features carry no target task labels. Phase 2 must implement these data rules;
Phase 1 validates names and configuration boundaries only.

Macro F1 is the primary endpoint; accuracy and component timings accompany it.
Pairing requires identical evaluation rows, preprocessing, base parameters and
seed. Planned uncertainty uses shared group bootstrap resamples across predictions
and averages paired deltas across fitted seeds. Repeated seeds on the same test set
are not independent datasets. Sensitivity settings remain distinct from primary
settings; final test results cannot choose a new primary setting.

## Engineering decisions and viva anchors

CPython 3.12.10 is used because 3.11 is unavailable on this machine. Pydantic adds
strict typed, frozen nested configuration schemas; PyYAML handles loading with
unsafe objects, aliases and duplicate keys rejected. The engine has no Streamlit
dependency at import time. Run states and deterministic content identities are
defined now, while atomic run storage and actual training remain later work.

For the viva, explain a domain as a data-producing setting. Source examples have
task labels; target adaptation examples supply features; held-out target evaluation
examples are reserved to assess the fitted model. A dashboard presents evidence,
but cannot establish valid adaptation without fair splits, matching baselines and
recorded predictions. Read and explain the immutable `TrainingView`, strict config
validator and content-based `RunKey`. Student understanding has not been assessed
by these automated checks.
