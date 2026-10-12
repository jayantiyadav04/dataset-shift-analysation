# Research report and submission evidence

The final documents report the completed frozen study. They retain negative
transfer and uncertainty, and preserve the exact objective and original proposal
abstract. The separate results abstract contains 150 words, as requested.

| Material | File |
| --- | --- |
| Final report | [PDF](final_report.pdf), [editable Markdown](final_report.md) |
| Paper draft | [PDF](paper_draft.pdf), [editable Markdown](paper_draft.md) |
| Results abstract | [150-word text](paper_abstract.txt) |
| Literature and sources | [11-paper matrix](literature_review.md), [20 references](references.md) |
| Completed data cards | [Gas](data_cards/gas.md), [HAR](data_cards/har.md), [Amazon](data_cards/amazon.md) |
| Presentation | [Ten-slide content](slide_outline.md), [five-minute demo and viva](demo_and_viva.md) |
| Reproduction | [Installation/acquisition/execution guide](reproducibility_guide.md), [actual 50-fit check](../reproduction/README.md) |
| Integrity and contributions | [Assistance disclosure](assistance_and_contributions.md), [requirements checklist](requirements_checklist.md) |
| Numerical inputs | [All study tables/figures](../study/README.md), [report claim traces](report_values.json) |
| Genuine interface images | [Saved screenshot evidence](../workbench/README.md) |

The report is 13 pages; the paper is a six-page neutral academic draft. Neither
invented student details nor an unprovided faculty template is included. The
paper is not a publication-acceptance claim. Its abstract is separate from the
unchanged proposal abstract in reports/proposal.md.

## Evidence and package boundaries

The study contains 1,100 primary and 180 sensitivity evaluations. The 50 fresh
reproduction fits are separately identified and do not inflate those study counts.
Report tables derive from verified saved inputs; claim traces connect examples
to exact run IDs and prediction SHA-256 values. Original full predictions, sample
IDs, acquisition archives and models remain in the local ignored data/artifacts
folders. Acquisition and execution instructions explain how to obtain them.

Use `python tools/package_submission.py --out submission/domain_shift_workbench`
after committing the reviewed delivery. It requires a clean working tree and
creates source_and_evidence.zip, result_exports.zip and package_verification.json.
Each ZIP has an internal per-file hash inventory. Existing archives are preserved.
The preview option explicitly labels an uncommitted review package; it is not a
claim that the current files have been published. The local submission folder
remains ignored at the user's request.

The bundle includes permitted source/configuration, aggregate evidence and real
screenshots. It excludes raw archives, prediction rows, split-ID files, serialized
models, credentials, environments and caches. Historical development fixtures
remain under tests/ and are not experimental findings. An application export can
include local split IDs; that is a different artifact from the public package.

## Review status

The fresh environment reproduction passed. Both PDFs have been rendered and
visually checked. The final regression suite passed all 228 tests in 139.96 seconds;
Ruff lint/format, all 64 installed scientific pins and the documentation validator
also passed. The [evidence audit](evidence_verification.json) verified the three
demonstration examples and a real export without fitting. The
[document review](document_review.json) records all 19 reviewed pages. Exact test
receipt hashes and checks are in [delivery verification](delivery_verification.json).
A final archive-only improvement reads exact committed bytes despite Windows
checkout line endings; all 17 targeted publication checks then passed in 0.22 seconds.

Student identity details, faculty formatting, personal contribution confirmation
and timed viva rehearsal remain human review tasks. The unchecked items in the
requirements checklist are intentional and prevent automated checks from being
mistaken for student understanding or institutional approval.

## Rebuilding document outputs

The scientific environment remains locked and unchanged. A separate reporting
environment used Markdown 3.11, ReportLab 5.0.1, PyMuPDF 1.28.2 and Pillow 12.3.0.
The PDF generator uses Windows Times fonts, so font layout is platform-specific.
Run tools/assemble_report.py in the scientific environment with the verified local
study present; it refuses to replace the existing sealed report_values.json.
Prepare a separate report workspace when regenerating. Then run
tools/render_research_reports.py with source, destination and --qa paths in the
reporting environment. Review every rendered page after any change.

Poppler 26.09.0 was also used for independent page rendering. The downloaded
Windows package matched its published SHA-256
`7a6f256a0ddf7536182246a5733331bf4677cbcc34f4663774947ad34556c8d0`.
This reporting utility is separate from the 64-package scientific environment.
