"""Assessment wording frozen from the original documentation pack."""

TITLE = "Evaluating Domain Adaptation Techniques for Classical Machine Learning Models"

FINAL_OBJECTIVE = (
    "To evaluate whether importance weighting and feature alignment using CORAL improve the "
    "performance of Logistic Regression, Support Vector Machine, and Random Forest when source "
    "and target data follow different distributions. The study will compare these techniques "
    "against models trained without adaptation across three public datasets, using accuracy, "
    "macro F1 score, and computational time. It will identify the conditions under which "
    "adaptation improves performance, provides little benefit, or reduces prediction quality."
)

PROPOSAL_ABSTRACT = (
    "Machine learning models can perform well on training data but struggle when applied to "
    "data collected under different conditions. This project will investigate whether domain "
    "adaptation can reduce this performance gap for Logistic Regression, Support Vector Machine, "
    "and Random Forest classifiers. It will compare importance weighting and correlation "
    "alignment (CORAL) with models trained without adaptation. Experiments will use the Gas "
    "Sensor Array Drift dataset, Human Activity Recognition Using Smartphones dataset, and "
    "Amazon Multi-Domain Sentiment dataset. These datasets represent changes over time, between "
    "users, and across product categories. Adaptation will use labelled source data and "
    "unlabelled target data, while target labels will be reserved for final evaluation. "
    "Accuracy, macro F1 score, and computational time will be measured across repeated "
    "experiments. The study aims to identify when adaptation improves predictions, when it "
    "offers limited benefit, and when it reduces performance, helping explain which approaches "
    "are useful under different types of distribution shift."
)

PRIMARY_SEEDS = (17, 29, 43, 71, 101)
