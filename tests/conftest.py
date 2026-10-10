"""Paths and raw config fixtures; no network or public study data."""

from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def primary_dict():
    return yaml.safe_load((ROOT / "configs" / "study.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def smoke_dict():
    return yaml.safe_load((ROOT / "configs" / "smoke.yaml").read_text(encoding="utf-8"))
