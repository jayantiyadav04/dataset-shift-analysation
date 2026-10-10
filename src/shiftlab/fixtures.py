"""Tiny offline DEVELOPMENT fixture. Not one of the study's public datasets."""

import numpy as np

from shiftlab.contracts import TrainingView


def synthetic_training_view(seed: int = 17) -> TrainingView:
    """Return source labels and shifted adaptation features, without target labels."""
    generator = np.random.default_rng(seed)
    labels = np.tile(np.array([0, 1], dtype=np.int64), 20)
    source = generator.normal(size=(40, 3)) + labels[:, None] * np.array([2.0, -1.0, 1.0])
    adaptation = generator.normal(size=(16, 3)) + np.array([0.5, 0.0, 1.0])
    return TrainingView(
        X_source=source,
        y_source=labels,
        X_target_adapt=adaptation,
        source_ids=tuple(f"synthetic_source_{i:03d}" for i in range(len(source))),
        target_adapt_ids=tuple(f"synthetic_adapt_{i:03d}" for i in range(len(adaptation))),
    )
