# har retained-count data card

Generated from archived inputs. No model performance has been measured.

Parser: `2.0.0`. Schema: 1.
Raw rows: **10299**. Retained: **10299**.
Excluded: **0**. Feature fields: **561**.
Processed SHA-256: `29288ea07ccf1b9f3039f9ac12a5fef7e517b5d854dd673394f68e4fe5c85269`.

## Acquisition

- main: [human+activity+recognition+using+smartphones.zip](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip); accessed 2026-10-10T19:14:02.068556+00:00; SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.

UCI displays CC BY 4.0; cite the dataset and original paper. Raw archives remain local.

## Counts and permissions

Original file/domain counts: `{"test": 2947, "train": 7352}`.
Retained domain/pool counts: `{"test/test": 2947, "train/train": 7352}`.
Exclusions by reason: `{}`.
Class mapping: `{"0": "WALKING", "1": "WALKING_UPSTAIRS", "2": "WALKING_DOWNSTAIRS", "3": "SITTING", "4": "STANDING", "5": "LAYING"}`.
Allowed source label counts: `{"train": {"0": 1226, "1": 1073, "2": 986, "3": 1286, "4": 1374, "5": 1407}}`.
Target label profiles are withheld until evaluation. Amazon categories each serve as source in separate tasks.
Missing numerical data: validation fails; no imputation or rounded deduplication.
Amazon model inputs contain review bodies only; rating and reviewer fields are excluded.

## Grouping and preprocessing

Whole official participants; identical features across people quarantined
Cross-domain exact duplicates are removed from target pools per task; see split exclusions.
All scalers, TF-IDF vocabulary/IDF and SVD fit only on the current source training fold.
Text uses 20,000 maximum unigram/bigram features, min_df=2, sublinear TF, no stopword list, and at most 128 SVD components followed by scaling.
Numerical features retain the supplied 128 (gas) or 561 (HAR) columns.
HAR windows are dependent within participants; gas may retain within-batch temporal dependence.
Amazon product identifiers are preserved separately; body groups determine duplicate separation.

Parser notes: `{"official_participants": {"test": [2, 4, 9, 10, 12, 13, 18, 20, 24], "train": [1, 3, 5, 6, 7, 8, 11, 14, 15, 16, 17, 19, 21, 22, 23, 25, 26, 27, 28, 29, 30]}}`.

Exact IDs, groups and role file hashes are in data/splits/<protocol>/<task>/manifest.json.
Run `shiftlab data audit --all` and `shiftlab splits audit --config configs/study.yaml` for validation.
