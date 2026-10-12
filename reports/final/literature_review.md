# Literature review and critical reading matrix

The eleven papers below connect the implemented comparison to established work.
Citation details and access limits are in [references](references.md). Findings
in this matrix describe the cited work; the final report presents this project's
own measured results. This is a focused review, not a systematic search of all
domain adaptation literature.

| Paper | Method and setting | Reported contribution | Limitation and relevance to this project |
| --- | --- | --- | --- |
| R01 Sun et al. | Unsupervised covariance alignment on visual domain benchmarks | Aligns second-order statistics without target labels using a compact transform | Visual results do not guarantee gains for gas, HAR or dense sentiment features. This project explicitly adds centring, target mean restoration and ridge 0.001. |
| R02 Sugiyama et al. | Importance-weighted validation under covariate shift; simulations and brain-computer interfaces | Develops a validation-risk estimator suited to changed input distributions | The unchanged conditional-label assumption and estimator variance matter. This study deliberately uses shared source-only CV, not IWCV, so target-optimal model selection is not claimed. |
| R03 Huang et al. | Kernel mean matching for sample-selection bias | Estimates weights by matching feature-space means without separate density estimation | Support and weight bounds matter; finite-sample matching is not a general accuracy guarantee. KMM is a conceptual alternative, not an implemented method here. |
| R04 Ben-David et al. | Theoretical bounds for source/target learning | Relates target error to source error, domain divergence and shared prediction feasibility | Divergence alone omits the joint labeling difficulty. Bounds motivate caution; this project neither estimates the bound nor uses it as a performance certificate. |
| R05 Blitzer et al. | Structural correspondence learning on four Amazon categories | Uses pivot-feature relations and studies domain similarity for sentiment transfer | The author's released files differ from the paper's sampled protocol. This study uses the explicit released pools, body deduplication and source-only TF-IDF/SVD; scores are not directly comparable. |
| R06 Fernando et al. | Eigenvector subspace alignment for visual adaptation | Reports a closed-form mapping between source and target subspaces | The reviewed abstract does not establish results for these three datasets. Subspace alignment and covariance matching optimize different objects; only CORAL is implemented. |
| R07 Pan and Yang | Survey of transfer learning across classification, regression and clustering | Organizes settings by domains, tasks and available labels; discusses negative transfer | A taxonomy does not select a safe method for a new target. The present task keeps labels and feature schemas within each dataset and evaluates losses as well as gains. |
| R08 Vergara et al. | Supervised SVM ensembles across gas-sensor acquisition periods | Reports sensor drift and compensation using classifiers from different times | Its access to successive labeled batches differs from this unlabeled target-adaptation protocol. It supplies dataset motivation, not a directly matched baseline score. |
| R09 Anguita et al. | Smartphone HAR dataset and SVM recognition | Documents participants, overlapping windows and 561 engineered features | Windows from a person are dependent. The current four-person adaptation/five-person evaluation protocol differs from evaluating the entire official test partition. |
| R10 Pedregosa et al. | General machine-learning software library | Describes scikit-learn's reusable estimator ecosystem | The 2011 paper is a software citation, not a guarantee about current APIs. Actual 1.9.1 versions, arguments, sample weights and numerical behavior were checked locally. |
| R11 Bickel et al. | Discriminative covariate-shift learning; spam, text and landmine tasks | Connects source-versus-target discrimination to density ratios, including a two-stage approach | Calibration, support and a stable conditional-label relationship remain assumptions. The implementation uses explicit fold-count prior correction and clipping rather than reproducing its integrated optimizer. |

Weighting changes the influence of source observations, whereas alignment changes
their coordinates. These interventions can fail for different reasons. A domain
classifier can separate domains confidently while providing unstable ratios in
regions with little source support. Covariance alignment can reduce a geometric
distance while disturbing a classifier's relationship to class labels. The
theoretical literature therefore supports evaluation conditional on the chosen
data and assumptions, rather than a promise of improvement.

The empirical contribution is a controlled comparison of these established
methods over three applications with consistent source model selection, data-role
permissions, paired predictions and diagnostics. It is not algorithmic novelty.
Negative transfer, dependency between transfers, and the limited number of
independent HAR evaluation participants restrict any general recommendation.
