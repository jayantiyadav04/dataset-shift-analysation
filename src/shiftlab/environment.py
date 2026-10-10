"""Actual package metadata and an offline estimator API compatibility check."""

import importlib
import os
import platform
import warnings
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits

from shiftlab.experiments.provenance import file_sha256, source_revision
from shiftlab.fixtures import synthetic_training_view

PACKAGES = (
    "domain-shift-workbench",
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "PyYAML",
    "pydantic",
    "joblib",
    "matplotlib",
    "seaborn",
    "streamlit",
    "pytest",
    "ruff",
    "pip",
    "setuptools",
    "wheel",
)


def compatibility_check() -> dict[str, bool]:
    """Check fit/predict with sample_weight. Do not calculate research scores."""
    for name in ("numpy", "scipy", "pandas", "yaml", "pydantic", "joblib", "matplotlib", "seaborn"):
        importlib.import_module(name)
    view = synthetic_training_view()
    features = StandardScaler().fit_transform(view.X_source)
    weights = np.linspace(0.5, 1.5, len(features))
    estimators = {
        "lr": LogisticRegression(max_iter=3000, random_state=17),
        "linear_svm": LinearSVC(max_iter=10000, dual="auto", random_state=17),
        "rf": RandomForestClassifier(n_estimators=10, random_state=17, n_jobs=1),
    }
    with threadpool_limits(limits=1), warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        for name, estimator in estimators.items():
            estimator.fit(features, view.y_source, sample_weight=weights)
            predictions = estimator.predict(features)
            if predictions.shape != view.y_source.shape:
                raise RuntimeError(f"{name} produced an invalid prediction shape")
    return dict.fromkeys(estimators, True)


def environment_report(root: Path) -> dict[str, Any]:
    lock = root / "requirements-lock.txt"
    return {
        "evidence_kind": "offline_synthetic_compatibility",
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "os": platform.platform(),
        "architecture": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpus": os.cpu_count(),
        "packages": {name: version(name) for name in PACKAGES},
        "dependency_lock_sha256": file_sha256(lock) if lock.is_file() else None,
        "source_revision": asdict(source_revision(root)),
        "compatibility_sample_weight_fit_predict": compatibility_check(),
        "study_evaluation_performed": False,
    }
