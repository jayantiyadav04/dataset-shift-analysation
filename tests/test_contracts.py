"""Boundary checks on permitted fitting arrays and identities."""

import inspect
from dataclasses import asdict

import numpy as np
import pytest

from shiftlab.adaptation import Adaptation
from shiftlab.contracts import TrainingView
from shiftlab.fixtures import synthetic_training_view


def test_fitting_contract_has_no_target_label_or_evaluation_parameter():
    assert tuple(inspect.signature(Adaptation.fit).parameters) == (
        "self",
        "X_source",
        "y_source",
        "X_target_adapt",
    )
    values = asdict(synthetic_training_view())
    for forbidden in ("y_target", "y_target_adapt", "X_target_eval", "y_target_eval"):
        with pytest.raises(TypeError):
            TrainingView(**values, **{forbidden: np.array([0, 1])})


@pytest.mark.parametrize(
    "field,value",
    [
        ("X_source", np.zeros((2, 2))),
        ("X_source", np.array([[np.nan] for _ in range(40)])),
        ("X_target_adapt", np.zeros((0, 3))),
        ("X_target_adapt", np.zeros((16, 2))),
        ("y_source", np.zeros(40, dtype=int)),
        ("y_source", np.zeros(40, dtype=float)),
        ("y_source", np.arange(40).reshape(40, 1)),
        ("source_ids", ("same",) * 40),
    ],
)
def test_invalid_training_arrays_or_ids_fail(field, value):
    values = asdict(synthetic_training_view())
    values[field] = value
    with pytest.raises(ValueError):
        TrainingView(**values)


def test_source_and_adaptation_identity_and_participant_overlap_fail():
    values = asdict(synthetic_training_view())
    values["target_adapt_ids"] = values["source_ids"][:16]
    with pytest.raises(ValueError, match="sample IDs overlap"):
        TrainingView(**values)
    values = asdict(synthetic_training_view())
    values["source_group_ids"] = ("participant_1",) * 40
    values["target_adapt_group_ids"] = ("participant_1",) * 16
    with pytest.raises(ValueError, match="groups overlap"):
        TrainingView(**values)


def test_fixture_is_deterministic_detached_and_read_only():
    first, second = synthetic_training_view(), synthetic_training_view()
    np.testing.assert_array_equal(first.X_source, second.X_source)
    np.testing.assert_array_equal(first.X_target_adapt, second.X_target_adapt)
    values = asdict(first)
    third = TrainingView(**values)
    values["X_source"][:] = 1000
    np.testing.assert_array_equal(third.X_source, first.X_source)
    with pytest.raises(ValueError):
        third.X_target_adapt[0, 0] = 1000
