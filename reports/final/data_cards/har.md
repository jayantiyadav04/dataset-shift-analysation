# HAR completed data card


Parser 2.0.0; schema 1. Actual parsed records: 10,299; retained: 10,299; excluded: 0.

This card describes the evaluated frozen study. Counts and label profiles below are saved evidence after final evaluation, not training permissions.

## Source and exact archive identity


[human+activity+recognition+using+smartphones.zip](https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip)

Accessed 2026-10-10T19:14:02.068556+00:00; SHA-256 `c00b803081a5c797cd5e4b83700a9810b38d53d9d84e01917e090e1fdbc81031`.

Processed dataset hash: `29288ea07ccf1b9f3039f9ac12a5fef7e517b5d854dd673394f68e4fe5c85269`.

Source terms: UCI displays CC BY 4.0; cite the dataset and original paper. Raw archives remain local.

## Schema and cleaning


Feature kind: numeric; stored feature fields: 561. Numerical values must be finite; unexpected missing values fail validation.

Class map: {"0": "WALKING", "1": "WALKING_UPSTAIRS", "2": "WALKING_DOWNSTAIRS", "3": "SITTING", "4": "STANDING", "5": "LAYING"}.

Exclusions by reason: {}.

Retained domain/pool counts: {"test/test": 2947, "train/train": 7352}.

Source class counts: {"train": {"0": 1226, "1": 1073, "2": 986, "3": 1286, "4": 1374, "5": 1407}}.

Group policy: Whole official participants; identical features across people quarantined.

Exact feature/body groups prevent role duplicates; task-specific target exclusions are additional to global parser exclusions.

## Frozen task counts


| Task | Source | Adaptation | Evaluation | Extra exclusions |
| --- | --- | --- | --- | --- |
| har_official_group_transfer | 7352 | 1201 | 1746 | 0 |

Split seed 2026. The complete SHA-256 and relative manifest paths are in [split counts](../tables/split_counts.csv). Exact IDs are local in each frozen data/splits/1.0/TASK/manifest.json and downloadable from the application.

Evaluation class support for har_official_group_transfer: {"0": 272, "1": 275, "2": 248, "3": 291, "4": 333, "5": 327}. Absent target classes: [].

## Limits and permissions


Source participants are 1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30. Adaptation participants are 2,4,9,10; evaluation participants are 12,13,18,20,24. Source has 21 groups, adaptation four and evaluation five. Overlapping windows from one participant remain grouped. Population inference is limited.

Fitting permits source labels and adaptation features only. Evaluation labels are used only by the authorized evaluator. Raw records, IDs and private labels are not included in the public package. See [references](../references.md) for the dataset and method citations.
