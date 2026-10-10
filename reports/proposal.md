# Project proposal

## Project title

Evaluating Domain Adaptation Techniques for Classical Machine Learning Models

## Final objective

To evaluate whether importance weighting and feature alignment using CORAL improve the performance of Logistic Regression, Support Vector Machine, and Random Forest when source and target data follow different distributions. The study will compare these techniques against models trained without adaptation across three public datasets, using accuracy, macro F1 score, and computational time. It will identify the conditions under which adaptation improves performance, provides little benefit, or reduces prediction quality.

## Research question

When a classical classifier moves to a different data distribution, which of the two adaptation approaches helps, how much does it cost, and when can it make predictions worse?

## Datasets and intended use

| Dataset | Published scale | Intended source and target |
| --- | --- | --- |
| UCI Gas Sensor Array Drift | 13,910 records, 128 features, six gas classes, ten batches | Batch 1 as source; batches 2 to 10 evaluated separately as target domains |
| UCI Human Activity Recognition Using Smartphones | 10,299 observations, 561 features, six activities, 30 participants | Official training participants as source; official test participants partitioned into an adaptation group and a held-out evaluation group |
| Amazon Multi-Domain Sentiment by Blitzer and colleagues | Original benchmark has 2,000 labelled reviews per domain across four domains, plus unlabelled reviews | All twelve ordered transfers between books, DVDs, electronics and kitchen products |

Published counts are reference values. The implemented data cards must report actual counts after parsing and deduplication. The dataset and method sources are listed in docs/12_REFERENCES.md.

## Abstract

Machine learning models can perform well on training data but struggle when applied to data collected under different conditions. This project will investigate whether domain adaptation can reduce this performance gap for Logistic Regression, Support Vector Machine, and Random Forest classifiers. It will compare importance weighting and correlation alignment (CORAL) with models trained without adaptation. Experiments will use the Gas Sensor Array Drift dataset, Human Activity Recognition Using Smartphones dataset, and Amazon Multi-Domain Sentiment dataset. These datasets represent changes over time, between users, and across product categories. Adaptation will use labelled source data and unlabelled target data, while target labels will be reserved for final evaluation. Accuracy, macro F1 score, and computational time will be measured across repeated experiments. The study aims to identify when adaptation improves predictions, when it offers limited benefit, and when it reduces performance, helping explain which approaches are useful under different types of distribution shift.

Word count: 150, counting Multi-Domain as one word. This is the proposal abstract and describes planned work.

## Planned contribution

The contribution is a controlled empirical comparison across three applications, with honest analysis of improvements, negative transfer and runtime. An accompanying research dashboard will make the experiment evidence inspectable. No claim is made that importance weighting or CORAL is a new algorithm, or that this comparison guarantees publication.
