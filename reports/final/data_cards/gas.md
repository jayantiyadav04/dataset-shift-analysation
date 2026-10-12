# GAS completed data card


Parser 2.0.0; schema 1. Actual parsed records: 13,910; retained: 13,910; excluded: 0.

This card describes the evaluated frozen study. Counts and label profiles below are saved evidence after final evaluation, not training permissions.

## Source and exact archive identity


[gas+sensor+array+drift+dataset.zip](https://archive.ics.uci.edu/static/public/224/gas+sensor+array+drift+dataset.zip)

Accessed 2026-10-10T19:14:36.810078+00:00; SHA-256 `91e8f466f202e7a093d657673ce47311c3e90416f7df3057966058961c351fe4`.

Processed dataset hash: `28c4543b20838b612e5fc4102d45bd2bf215360767ba9bbbfd51b024d71995c5`.

Source terms: UCI displays CC BY 4.0 and older research-only wording. Retain locally; distribute acquisition instructions.

## Schema and cleaning


Feature kind: numeric; stored feature fields: 128. Numerical values must be finite; unexpected missing values fail validation.

Class map: {"0": "Ethanol", "1": "Ethylene", "2": "Ammonia", "3": "Acetaldehyde", "4": "Acetone", "5": "Toluene"}.

Exclusions by reason: {}.

Retained domain/pool counts: {"b01/labelled": 445, "b02/labelled": 1244, "b03/labelled": 1586, "b04/labelled": 161, "b05/labelled": 197, "b06/labelled": 2300, "b07/labelled": 3613, "b08/labelled": 294, "b09/labelled": 470, "b10/labelled": 3600}.

Source class counts: {"b01": {"0": 90, "1": 98, "2": 83, "3": 30, "4": 70, "5": 74}}.

Group policy: Exact float64 feature groups, no rounded deduplication; source precedes target.

Exact feature/body groups prevent role duplicates; task-specific target exclusions are additional to global parser exclusions.

## Frozen task counts


| Task | Source | Adaptation | Evaluation | Extra exclusions |
| --- | --- | --- | --- | --- |
| gas_b01_b02 | 445 | 622 | 622 | 0 |
| gas_b01_b03 | 445 | 793 | 793 | 0 |
| gas_b01_b04 | 445 | 80 | 81 | 0 |
| gas_b01_b05 | 445 | 98 | 99 | 0 |
| gas_b01_b06 | 445 | 1150 | 1150 | 0 |
| gas_b01_b07 | 445 | 1806 | 1807 | 0 |
| gas_b01_b08 | 445 | 147 | 147 | 0 |
| gas_b01_b09 | 445 | 235 | 235 | 0 |
| gas_b01_b10 | 445 | 1800 | 1800 | 0 |

Split seed 2026. The complete SHA-256 and relative manifest paths are in [split counts](../tables/split_counts.csv). Exact IDs are local in each frozen data/splits/1.0/TASK/manifest.json and downloadable from the application.

Evaluation class support for gas_b01_b02: {"0": 82, "1": 153, "2": 59, "3": 60, "4": 266, "5": 2}. Absent target classes: [].

Evaluation class support for gas_b01_b03: {"0": 188, "1": 235, "2": 104, "3": 116, "4": 150}. Absent target classes: [5].

Evaluation class support for gas_b01_b04: {"0": 32, "1": 18, "2": 5, "3": 20, "4": 6}. Absent target classes: [5].

Evaluation class support for gas_b01_b05: {"0": 16, "1": 17, "2": 14, "3": 21, "4": 31}. Absent target classes: [5].

Evaluation class support for gas_b01_b06: {"0": 239, "1": 316, "2": 48, "3": 17, "4": 296, "5": 234}. Absent target classes: [].

Evaluation class support for gas_b01_b07: {"0": 344, "1": 331, "2": 176, "3": 376, "4": 296, "5": 284}. Absent target classes: [].

Evaluation class support for gas_b01_b08: {"0": 14, "1": 14, "2": 20, "3": 13, "4": 75, "5": 11}. Absent target classes: [].

Evaluation class support for gas_b01_b09: {"0": 31, "1": 32, "2": 51, "3": 36, "4": 35, "5": 50}. Absent target classes: [].

Evaluation class support for gas_b01_b10: {"0": 284, "1": 311, "2": 298, "3": 295, "4": 309, "5": 303}. Absent target classes: [].

## Limits and permissions


All source rows are Batch 1. Target batches follow time direction but their internal adaptation/evaluation holdout is not chronological forecasting. Finer within-batch dependence remains possible. Class composition varies by batch.

Fitting permits source labels and adaptation features only. Evaluation labels are used only by the authorized evaluator. Raw records, IDs and private labels are not included in the public package. See [references](../references.md) for the dataset and method citations.
