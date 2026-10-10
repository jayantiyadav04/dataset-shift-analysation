"""Fitting interface; importance weighting and CORAL arrive in Phases 4 and 5."""

from typing import Protocol, Self

from shiftlab.contracts import FeatureArray, LabelArray


class Adaptation(Protocol):
    """A target task label cannot be passed to this fitting contract."""

    def fit(
        self,
        X_source: FeatureArray,
        y_source: LabelArray,
        X_target_adapt: FeatureArray,
    ) -> Self: ...

    def predict(self, X_target: FeatureArray) -> LabelArray: ...
