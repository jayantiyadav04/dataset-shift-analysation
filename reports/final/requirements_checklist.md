# Requirements and evidence checklist

This maps the supplied guide to actual evidence. It is not an official grading
rubric and assigns no invented marks. Technical verification cannot certify
student understanding, institutional identity, faculty approval or publication.

| Requirement | Evidence and status |
| --- | --- |
| Frozen objective and original 150-word proposal | Exact objective in report, paper and slides; unchanged proposal file; report Appendix A preserves its abstract |
| Separate paper abstract | 150 words in paper_abstract.txt, generated report and paper; requested word limit followed |
| Three datasets and documented cleaning | Completed gas/HAR/Amazon cards, archive hashes, retained counts, exclusions and split table |
| Within-dataset domain adaptation | 9 gas, 1 participant-group HAR and 12 ordered Amazon transfers; no mixed-dataset classifier |
| Leakage control and validation | TrainingView excludes target task labels; source-fold preprocessing; disjoint evaluation roles and held-out-read tests |
| Models and simple baseline | LR, LinearSVC, RF under three methods plus source-majority; all 1,100 primary records retained |
| Correct numerical methods | Independently checked count correction, covariance operations and acceptance tests; source snapshot remains unchanged |
| Paired comparison fairness | Same source-selected parameters, representations, held-out rows and seeds; paired run IDs preserved |
| Five repetitions | Seeds 17,29,43,71,101; full tables include means and sample SD, including zero variation |
| Metrics beyond accuracy | Macro F1, precision/recall, class support and local confusion matrices; target-present class convention explained |
| Uncertainty | 1,000 whole-group resamples, paired deltas, 336 metric intervals; seed dependence and unadjusted intervals disclosed |
| Sensitivity | All 180 predeclared evaluations; 54 setting groups in sensitivity_summary.csv; no target-driven primary changes |
| Negative transfer and failed operations | 92 negative primary mean-F1 changes; low-ESS warnings; two preserved/resumed progress-file errors; no classifier-fit failures |
| Runtime and hardware | Actual three-dataset benchmark components and peak RAM; projection distinguished from observed invocation/fit-boundary clocks |
| Clean reproduction | Fresh environment/checkout, 50 new fits, all predictions/metrics/parameters equal; reused data and platform limits explicit |
| Usable dashboard and exports | Six pages, 16 real screenshots, explicit job/cancel/export checks; actual automated saved-evidence demo checks |
| Numerical consistency and provenance | Generated report tables, saved study exports, exact run IDs, prediction hashes and source/dependency checks |
| Literature and citations | 11 related papers and 20 total references; primary sources checked; two restricted full-text accesses disclosed |
| Document inspection | Report and paper PDFs rendered with page images and visually inspected; layout and labels reviewed |
| Permitted packaging | Source and aggregate evidence ZIP inventories; raw data, model objects, credentials and local split IDs excluded |
| Assistance disclosure | Codex's implementation, execution and drafting assistance stated factually; no unsupported independent student contribution |
| Final regression and publication | 228 tests passed in 139.96 seconds, then 17 targeted archive checks passed after the committed-byte improvement; lint/format and locked environment passed. Exact receipts are in delivery_verification.json; Git history and the committed archive identify publication. |

## Required student actions before submission

- [ ] Verify and supply student/registration/institution/supervisor details.
- [ ] Check the faculty's required report and slide format; no template was supplied.
- [ ] Personally review the analysis, claims, citations and assistance disclosure.
- [ ] Rehearse the five-minute demonstration and code walkthrough.
- [ ] Explain weighting, CORAL, leakage controls, uncertainty and one real failure.
- [ ] Record personal contributions accurately, without treating Git attribution as proof.
- [ ] Confirm final technical checks and publication/package receipts before submitting.

No automated tool has checked these personal actions as complete.
