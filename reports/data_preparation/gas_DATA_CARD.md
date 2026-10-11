# gas retained-count data card

Generated from archived inputs. No model performance has been measured.

Parser: `2.0.0`. Schema: 1.
Raw rows: **13910**. Retained: **13910**.
Excluded: **0**. Feature fields: **128**.
Processed SHA-256: `28c4543b20838b612e5fc4102d45bd2bf215360767ba9bbbfd51b024d71995c5`.

## Acquisition

- main: [gas+sensor+array+drift+dataset.zip](https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip); accessed 2026-10-10T19:14:36.810078+00:00; SHA-256 `91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4`.

UCI displays CC BY 4.0 and older research-only wording. Retain locally; distribute acquisition instructions.

## Counts and permissions

Original file/domain counts: `{"b01": 445, "b02": 1244, "b03": 1586, "b04": 161, "b05": 197, "b06": 2300, "b07": 3613, "b08": 294, "b09": 470, "b10": 3600}`.
Retained domain/pool counts: `{"b01/labelled": 445, "b02/labelled": 1244, "b03/labelled": 1586, "b04/labelled": 161, "b05/labelled": 197, "b06/labelled": 2300, "b07/labelled": 3613, "b08/labelled": 294, "b09/labelled": 470, "b10/labelled": 3600}`.
Exclusions by reason: `{}`.
Class mapping: `{"0": "Ethanol", "1": "Ethylene", "2": "Ammonia", "3": "Acetaldehyde", "4": "Acetone", "5": "Toluene"}`.
Allowed source label counts: `{"b01": {"0": 90, "1": 98, "2": 83, "3": 30, "4": 70, "5": 74}}`.
Target label profiles are withheld until evaluation. Amazon categories each serve as source in separate tasks.
Missing numerical data: validation fails; no imputation or rounded deduplication.
Amazon model inputs contain review bodies only; rating and reviewer fields are excluded.

## Grouping and preprocessing

Exact float64 feature groups, no rounded deduplication; source precedes target
Cross-domain exact duplicates are removed from target pools per task; see split exclusions.
All scalers, TF-IDF vocabulary/IDF and SVD fit only on the current source training fold.
Text uses 20,000 maximum unigram/bigram features, min_df=2, sublinear TF, no stopword list, and at most 128 SVD components followed by scaling.
Numerical features retain the supplied 128 (gas) or 561 (HAR) columns.
HAR windows are dependent within participants; gas may retain within-batch temporal dependence.
Amazon product identifiers are preserved separately; body groups determine duplicate separation.

Parser notes: `{}`.

Exact IDs, groups and role file hashes are in data/splits/<protocol>/<task>/manifest.json.
Run `shiftlab data audit --all` and `shiftlab splits audit --config configs/study.yaml` for validation.
