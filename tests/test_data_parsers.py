"""Tiny deterministic schema fixtures; no public dataset substituted in studies."""

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from shiftlab.data import parsers
from shiftlab.data.parsers import (
    ParsedDataset,
    Record,
    canonical_body,
    clean_duplicates,
    normalized_body,
    numeric_signature,
    parse_amazon,
    parse_gas,
    parse_har,
)
from shiftlab.data.storage import DataError


def test_gas_schema_ids_and_reference_count(tmp_path, monkeypatch):
    for batch in range(1, 11):
        row = "1 " + " ".join(f"{j}:{batch + j / 1000}" for j in range(1, 129))
        (tmp_path / f"batch{batch}.dat").write_text(row + "\n")
    with pytest.raises(DataError, match="version mismatch"):
        parse_gas(tmp_path)
    monkeypatch.setattr(parsers, "GAS_REFERENCE_ROWS", 10)
    parsed = parse_gas(tmp_path)
    assert len(parsed.records) == 10
    assert len(parsed.records[0].values) == 128
    assert parsed.records[0].label == 0
    assert parse_gas(tmp_path).records == parsed.records
    (tmp_path / "batch2.dat").write_text("1 1:nan\n")
    with pytest.raises(DataError, match="schema mismatch"):
        parse_gas(tmp_path)
    (tmp_path / "batch2.dat").unlink()
    with pytest.raises(DataError, match="exactly one"):
        parse_gas(tmp_path)


def har_fixture(root):
    (root / "features.txt").write_text("\n".join(f"{i} feature{i}" for i in range(1, 562)))
    (root / "activity_labels.txt").write_text(
        "\n".join(f"{i + 1} {name}" for i, name in parsers.CLASS_MAPS["har"].items())
    )
    for partition, subjects in (("train", range(1, 22)), ("test", range(22, 31))):
        folder = root / partition
        folder.mkdir()
        np.savetxt(
            folder / f"X_{partition}.txt",
            np.asarray([[subject + j / 1000 for j in range(561)] for subject in subjects]),
        )
        np.savetxt(folder / f"subject_{partition}.txt", list(subjects), fmt="%d")
        np.savetxt(folder / f"y_{partition}.txt", [s % 6 + 1 for s in subjects], fmt="%d")


def test_har_schema_participants_and_row_alignment(tmp_path, monkeypatch):
    har_fixture(tmp_path)
    monkeypatch.setattr(parsers, "HAR_REFERENCE_COUNTS", {"train": 21, "test": 9})
    parsed = parse_har(tmp_path)
    assert len(parsed.records) == 30
    assert len(parsed.records[0].values) == 561
    assert len(parsed.notes["official_participants"]["test"]) == 9
    (tmp_path / "test/subject_test.txt").write_text("1\n" * 9)
    with pytest.raises(DataError, match="participant leakage"):
        parse_har(tmp_path)
    (tmp_path / "test/subject_test.txt").write_text("1\n")
    with pytest.raises(DataError, match="row counts"):
        parse_har(tmp_path)


def amazon_fixture(root):
    text = (Path(__file__).parent / "fixtures/reviews.review").read_text()
    for domain in ("books", "dvd", "electronics", "kitchen_&_housewares"):
        (root / domain).mkdir()
        for name in ("positive.review", "negative.review", "unlabeled.review"):
            path = (
                root / "book.unlabeled"
                if domain == "books" and name == "unlabeled.review"
                else root / domain / name
            )
            path.write_text(text, encoding="utf-8")


def test_amazon_uses_rating_not_filename_and_discards_unlabelled_ratings(tmp_path):
    amazon_fixture(tmp_path)
    parsed = parse_amazon(tmp_path)
    labelled = [r for r in parsed.records if r.pool == "labelled"]
    assert {r.label for r in labelled if r.source_file.endswith("positive.review")} == {0, 1}
    assert all(r.label is None for r in parsed.records if r.pool == "unlabelled")
    assert all(
        "MUST_NOT_ENTER_FEATURES" not in r.values and "<rating>" not in r.values
        for r in parsed.records
    )
    assert any("Not bad." in r.values and "&" in r.values for r in parsed.records)
    assert all(r.product_id for r in parsed.records)
    path = tmp_path / "book.unlabeled"
    path.write_text(path.read_text().replace("5.0", "invalid rating"))
    assert len(parse_amazon(tmp_path).records) == len(parsed.records)
    path = tmp_path / "books/positive.review"
    path.write_text(path.read_text().replace("5.0", "3.0").replace("1.0", "NaN"))
    invalid = parse_amazon(tmp_path)
    assert {e["reason"] for e in invalid.exclusions} == {
        "neutral_labelled_rating",
        "invalid_labelled_rating",
    }


def test_duplicate_conflicts_quarantined_and_labelled_precedes_unlabelled():
    first = Record("a", "books", "labelled", "body:a", "a", "first", 1, 1, "Not bad")
    repeated = replace(first, sample_id="b", pool="unlabelled", label=None)
    parsed = clean_duplicates(ParsedDataset("amazon", [repeated, first], {"all": 2}))
    assert parsed.records == [first]
    assert len(parsed.exclusions) == 1
    conflict = replace(first, sample_id="c", domain="dvd", label=0)
    other = replace(first, sample_id="d", duplicate_id="d", group_id="body:d", values="Other")
    parsed = clean_duplicates(ParsedDataset("amazon", [first, conflict, other], {"all": 3}))
    assert parsed.records == [other]
    assert [e["reason"] for e in parsed.exclusions] == ["conflicting_labels"] * 2


def test_canonicalisation_keeps_negation_and_numeric_precision():
    assert canonical_body("  not\u00a0good\n&amp; bad ") == "not good & bad"
    assert canonical_body("Ａ product") == "A product"
    escaped = canonical_body("A literal &amp;quot; entity remains in the body")
    assert "&quot;" in escaped and normalized_body(escaped) == escaped
    assert numeric_signature((1.0,)) != numeric_signature((1.0 + 1e-12,))
    assert numeric_signature((0.0,)) == numeric_signature((-0.0,))
