"""Centred, ridge-regularised CORAL for row-vector observations."""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from threadpoolctl import threadpool_limits

from shiftlab.contracts import TrainingView
from shiftlab.experiments.config import CoralConfig


def feature_matrix(values: np.ndarray, *, minimum_rows: int = 2) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype.kind not in "fiu" or array.ndim != 2:
        raise ValueError("CORAL requires a real, dense feature matrix")
    array = np.asarray(array, dtype=np.float64)
    if array.shape[0] < minimum_rows or array.shape[1] < 1 or not np.isfinite(array).all():
        raise ValueError("CORAL features must be nonempty and finite with sufficient rows")
    return array


def sample_covariance(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unbiased covariance (n-1 denominator), including the one-column case."""
    X = feature_matrix(values)
    mean = X.mean(axis=0)
    centred = X - mean
    covariance = centred.T @ centred / (len(X) - 1)
    if not np.isfinite(mean).all() or not np.isfinite(covariance).all():
        raise ValueError("CORAL covariance computation overflowed")
    return mean, (covariance + covariance.T) / 2


def symmetric_matrix_power(
    matrix: np.ndarray,
    power: Literal[-0.5, 0.5],
    eigenvalue_floor: float,
) -> tuple[np.ndarray, dict]:
    """Symmetric eigendecomposition; never symmetrise the subsequent CORAL product."""
    array = np.asarray(matrix)
    if (
        array.dtype.kind not in "fiu"
        or array.ndim != 2
        or array.shape[0] != array.shape[1]
        or array.shape[0] < 1
        or not np.isfinite(array).all()
        or not np.allclose(array, array.T, atol=1e-12, rtol=1e-12)
        or power not in (-0.5, 0.5)
        or not np.isfinite(eigenvalue_floor)
        or eigenvalue_floor <= 0
    ):
        raise ValueError("Symmetric powers require a finite symmetric matrix and positive floor")
    array = np.asarray(array, dtype=np.float64)
    with threadpool_limits(limits=1):
        eigenvalues, vectors = np.linalg.eigh((array + array.T) / 2)
        tolerance = np.finfo(float).eps * len(array) * max(abs(eigenvalues).max(), 1.0) * 10
        if eigenvalues.min() < -tolerance:
            raise ValueError("Covariance matrix is not positive semidefinite")
        bounded = np.maximum(eigenvalues, eigenvalue_floor)
        result = (vectors * np.power(bounded, power)) @ vectors.T
    result = (result + result.T) / 2
    if not np.isfinite(result).all():
        raise ValueError("Symmetric matrix power produced non-finite values")
    return result, {
        "minimum_eigenvalue": float(eigenvalues.min()),
        "maximum_eigenvalue": float(eigenvalues.max()),
        "floored_eigenvalues": int(np.count_nonzero(eigenvalues < eigenvalue_floor)),
        "condition_number_after_floor": float(bounded.max() / bounded.min()),
    }


def covariance_mismatch(covariance: np.ndarray, target: np.ndarray) -> float:
    value = float(np.linalg.norm(covariance - target, ord="fro")) / max(
        float(np.linalg.norm(target, ord="fro")), 1e-12
    )
    if not np.isfinite(value):
        raise ValueError("CORAL covariance mismatch is non-finite")
    return value


@dataclass(frozen=True)
class CoralState:
    alignment: np.ndarray
    source_mean: np.ndarray
    target_mean: np.ndarray
    source_covariance: np.ndarray
    target_covariance: np.ndarray
    diagnostics: dict

    def __post_init__(self) -> None:
        dimension = len(self.source_mean)
        for name in (
            "alignment",
            "source_mean",
            "target_mean",
            "source_covariance",
            "target_covariance",
        ):
            array = np.array(getattr(self, name), dtype=np.float64, copy=True)
            expected = (dimension,) if name.endswith("mean") else (dimension, dimension)
            if dimension < 1 or array.shape != expected or not np.isfinite(array).all():
                raise ValueError("Saved CORAL state has invalid dimensions or non-finite values")
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def transform_source(self, values: np.ndarray) -> np.ndarray:
        """For source fitting only; target features stay in their existing coordinates."""
        X = feature_matrix(values, minimum_rows=1)
        if X.shape[1] != len(self.source_mean):
            raise ValueError("Source feature schema differs from the fitted CORAL state")
        with threadpool_limits(limits=1):
            aligned = (X - self.source_mean) @ self.alignment + self.target_mean
        if not np.isfinite(aligned).all():
            raise ValueError("CORAL alignment produced non-finite source features")
        return aligned


def fit_coral(view: TrainingView, parameters: CoralConfig) -> CoralState:
    """Uses only allowed source and adaptation features; source labels are unused."""
    if not isinstance(view, TrainingView):
        raise ValueError("CORAL fitting requires the label-isolated training contract")
    with threadpool_limits(limits=1):
        ms, Cs0 = sample_covariance(view.X_source)
        mt, Ct0 = sample_covariance(view.X_target_adapt)
        ridge = parameters.ridge * np.eye(Cs0.shape[0])
        inverse_source, source_spectrum = symmetric_matrix_power(
            Cs0 + ridge, -0.5, parameters.eigenvalue_floor
        )
        root_target, target_spectrum = symmetric_matrix_power(
            Ct0 + ridge, 0.5, parameters.eigenvalue_floor
        )
        alignment = inverse_source @ root_target
        aligned_covariance = alignment.T @ Cs0 @ alignment
        regularised_covariance = alignment.T @ (Cs0 + ridge) @ alignment
        diagnostics = {
            "variant": "centred CORAL with target mean restoration",
            "orientation": "row vectors: (X_source - source_mean) @ A + target_mean",
            "target_prediction_coordinates": "source preprocessing only; no source alignment",
            "source_count": len(view.source_ids),
            "target_adapt_count": len(view.target_adapt_ids),
            "feature_dimension": len(ms),
            "parameters": parameters.model_dump(mode="json"),
            "source_regularised_spectrum": source_spectrum,
            "target_regularised_spectrum": target_spectrum,
            "source_covariance_rank": int(np.linalg.matrix_rank(Cs0)),
            "target_covariance_rank": int(np.linalg.matrix_rank(Ct0)),
            "covariance_mismatch_before": covariance_mismatch(Cs0, Ct0),
            "covariance_mismatch_after": covariance_mismatch(aligned_covariance, Ct0),
            "regularised_covariance_residual": covariance_mismatch(
                regularised_covariance, Ct0 + ridge
            ),
            "mean_distance_before": float(np.linalg.norm(ms - mt)),
            "mismatch_definition": "Frobenius(Cs-Ct0)/max(Frobenius(Ct0),1e-12)",
            "target_task_labels_used": False,
            "target_evaluation_opened": False,
            "warnings": [],
        }
    if diagnostics["covariance_mismatch_after"] > diagnostics["covariance_mismatch_before"]:
        diagnostics["warnings"].append("Unregularised covariance mismatch increased; retain run")
    return CoralState(alignment, ms, mt, Cs0, Ct0, diagnostics)
