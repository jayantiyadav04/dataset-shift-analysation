# amazon retained-count data card

Generated from archived inputs. No model performance has been measured.

Parser: `2.0.0`. Schema: 1.
Raw rows: **78679**. Retained: **70509**.
Excluded: **8170**. Feature fields: **1**.
Processed SHA-256: `dcd441a8dcbe0e1ac15f623572a28c14f20817b3ac47ff76372f01fce91f55fe`.

## Acquisition

- main: [domain_sentiment_data.tar.gz](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/domain_sentiment_data.tar.gz); accessed 2026-10-10T19:14:00.766564+00:00; SHA-256 `e65d4dae3f223316a5db426b2c3f7f3667e9b165f03c024cb4875e11af4cd941`.
- books_unlabeled: [book.unlabeled.gz](https://www.cs.jhu.edu/~mdredze/datasets/sentiment/book.unlabeled.gz); accessed 2026-10-10T19:14:03.715192+00:00; SHA-256 `94ece1bd3edb164e23ec679d817547dc2386303332b16722a6652d277a048a38`.

Author requests citation of Blitzer et al. (ACL 2007); no explicit redistribution licence on the linked page. Raw reviews remain local.

## Counts and permissions

Original file/domain counts: `{"books/negative.review": 1000, "books/positive.review": 1000, "books/unlabeled.review": 6000, "dvd/negative.review": 1000, "dvd/positive.review": 1000, "dvd/unlabeled.review": 34741, "electronics/negative.review": 1000, "electronics/positive.review": 1000, "electronics/unlabeled.review": 13153, "kitchen/negative.review": 1000, "kitchen/positive.review": 1000, "kitchen/unlabeled.review": 16785}`.
Retained domain/pool counts: `{"books/labelled": 1956, "books/unlabelled": 5473, "dvd/labelled": 1954, "dvd/unlabelled": 29281, "electronics/labelled": 1965, "electronics/unlabelled": 12405, "kitchen/labelled": 1977, "kitchen/unlabelled": 15498}`.
Exclusions by reason: `{"duplicate_body_or_exact_features": 8170}`.
Class mapping: `{"0": "negative", "1": "positive"}`.
Allowed source label counts: `{"books": {"0": 963, "1": 993}, "dvd": {"0": 959, "1": 995}, "electronics": {"0": 978, "1": 987}, "kitchen": {"0": 981, "1": 996}}`.
Target label profiles are withheld until evaluation. Amazon categories each serve as source in separate tasks.
Missing numerical data: validation fails; no imputation or rounded deduplication.
Amazon model inputs contain review bodies only; rating and reviewer fields are excluded.

## Grouping and preprocessing

Canonical body for role separation; product groups for source CV when feasible
Cross-domain exact duplicates are removed from target pools per task; see split exclusions.
All scalers, TF-IDF vocabulary/IDF and SVD fit only on the current source training fold.
Text uses 20,000 maximum unigram/bigram features, min_df=2, sublinear TF, no stopword list, and at most 128 SVD components followed by scaling.
Numerical features retain the supplied 128 (gas) or 561 (HAR) columns.
HAR windows are dependent within participants; gas may retain within-batch temporal dependence.
Amazon product identifiers are preserved separately; body groups determine duplicate separation.

Parser notes: `{"canonical_body_policy": "HTML entity decoding, Unicode NFKC, collapse whitespace; preserve case and negation", "file_encodings": {"books_unlabeled/book.unlabeled": "utf-8", "main/sorted_data_acl/books/negative.review": "utf-8", "main/sorted_data_acl/books/positive.review": "utf-8", "main/sorted_data_acl/dvd/negative.review": "utf-8", "main/sorted_data_acl/dvd/positive.review": "utf-8", "main/sorted_data_acl/dvd/unlabeled.review": "utf-8", "main/sorted_data_acl/electronics/negative.review": "utf-8", "main/sorted_data_acl/electronics/positive.review": "utf-8", "main/sorted_data_acl/electronics/unlabeled.review": "utf-8", "main/sorted_data_acl/kitchen_&_housewares/negative.review": "utf-8", "main/sorted_data_acl/kitchen_&_housewares/positive.review": "utf-8", "main/sorted_data_acl/kitchen_&_housewares/unlabeled.review": "utf-8"}, "product_id_coverage": {"usable": 78679}}`.

Exact IDs, groups and role file hashes are in data/splits/<protocol>/<task>/manifest.json.
Run `shiftlab data audit --all` and `shiftlab splits audit --config configs/study.yaml` for validation.
