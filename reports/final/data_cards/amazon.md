# AMAZON completed data card


Parser 2.0.0; schema 1. Actual parsed records: 78,679; retained: 70,509; excluded: 8,170.

This card describes the evaluated frozen study. Counts and label profiles below are saved evidence after final evaluation, not training permissions.

## Source and exact archive identity


[domain_sentiment_data.tar.gz](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz)

Accessed 2026-10-10T19:14:00.766564+00:00; SHA-256 `e65d4dae3f223316a5db426b2c3f7f3667e9b165f03c024cb4875e11af4cd941`.

[book.unlabeled.gz](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/book.unlabeled.gz)

Accessed 2026-10-10T19:14:03.715192+00:00; SHA-256 `94ece1bd3edb164e23ec679d817547dc2386303332b16722a6652d277a048a38`.

Processed dataset hash: `dcd441a8dcbe0e1ac15f623572a28c14f20817b3ac47ff76372f01fce91f55fe`.

Source terms: Author requests citation of Blitzer et al. (ACL 2007); no explicit redistribution licence on the linked page. Raw reviews remain local.

## Schema and cleaning


Feature kind: text_body; stored feature fields: 1. The one text field is transformed into source-fitted TF-IDF and at most 128 dense SVD features.

Class map: {"0": "negative", "1": "positive"}.

Exclusions by reason: {"duplicate_body_or_exact_features": 8170}.

Retained domain/pool counts: {"books/labelled": 1956, "books/unlabelled": 5473, "dvd/labelled": 1954, "dvd/unlabelled": 29281, "electronics/labelled": 1965, "electronics/unlabelled": 12405, "kitchen/labelled": 1977, "kitchen/unlabelled": 15498}.

Source class counts: {"books": {"0": 963, "1": 993}, "dvd": {"0": 959, "1": 995}, "electronics": {"0": 978, "1": 987}, "kitchen": {"0": 981, "1": 996}}.

Group policy: Canonical body for role separation; product groups for source CV when feasible.

Exact feature/body groups prevent role duplicates; task-specific target exclusions are additional to global parser exclusions.

## Frozen task counts


| Task | Source | Adaptation | Evaluation | Extra exclusions |
| --- | --- | --- | --- | --- |
| amazon_books_dvd | 1956 | 29280 | 1954 | 1 |
| amazon_books_electronics | 1956 | 12405 | 1965 | 0 |
| amazon_books_kitchen | 1956 | 15498 | 1977 | 0 |
| amazon_dvd_books | 1954 | 5473 | 1956 | 0 |
| amazon_dvd_electronics | 1954 | 12405 | 1965 | 0 |
| amazon_dvd_kitchen | 1954 | 15498 | 1977 | 0 |
| amazon_electronics_books | 1965 | 5473 | 1956 | 0 |
| amazon_electronics_dvd | 1965 | 29281 | 1954 | 0 |
| amazon_electronics_kitchen | 1965 | 15498 | 1977 | 0 |
| amazon_kitchen_books | 1977 | 5473 | 1956 | 0 |
| amazon_kitchen_dvd | 1977 | 29281 | 1954 | 0 |
| amazon_kitchen_electronics | 1977 | 12405 | 1965 | 0 |

Split seed 2026. The complete SHA-256 and relative manifest paths are in [split counts](../tables/split_counts.csv). Exact IDs are local in each frozen data/splits/1.0/TASK/manifest.json and downloadable from the application.

Evaluation class support for amazon_books_dvd: {"0": 959, "1": 995}. Absent target classes: [].

Evaluation class support for amazon_books_electronics: {"0": 978, "1": 987}. Absent target classes: [].

Evaluation class support for amazon_books_kitchen: {"0": 981, "1": 996}. Absent target classes: [].

Evaluation class support for amazon_dvd_books: {"0": 963, "1": 993}. Absent target classes: [].

Evaluation class support for amazon_dvd_electronics: {"0": 978, "1": 987}. Absent target classes: [].

Evaluation class support for amazon_dvd_kitchen: {"0": 981, "1": 996}. Absent target classes: [].

Evaluation class support for amazon_electronics_books: {"0": 963, "1": 993}. Absent target classes: [].

Evaluation class support for amazon_electronics_dvd: {"0": 959, "1": 995}. Absent target classes: [].

Evaluation class support for amazon_electronics_kitchen: {"0": 981, "1": 996}. Absent target classes: [].

Evaluation class support for amazon_kitchen_books: {"0": 963, "1": 993}. Absent target classes: [].

Evaluation class support for amazon_kitchen_dvd: {"0": 959, "1": 995}. Absent target classes: [].

Evaluation class support for amazon_kitchen_electronics: {"0": 978, "1": 987}. Absent target classes: [].

## Limits and permissions


Only review bodies enter features. Ratings, label-bearing filenames, reviewer fields and titles are excluded. Product IDs are separate grouping metadata; canonical bodies determine duplicate exclusion. Global product-ID coverage was 78,679/78,679 parsed rows. In books-to-DVD, one extra duplicate is removed from the adaptation pool. Source-only vocabulary/SVD can lose target-specific sentiment information. The original release is distinct from later Amazon corpora.

Fitting permits source labels and adaptation features only. Evaluation labels are used only by the authorized evaluator. Raw records, IDs and private labels are not included in the public package. See [references](../references.md) for the dataset and method citations.
