"""Ingestion only: parse labels separately from model-visible feature values."""

import hashlib
import html
import re
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from shiftlab.data.storage import DataError

PARSER_VERSION = "2.0.0"
CLASS_MAPS = {
    "gas": {
        0: "Ethanol",
        1: "Ethylene",
        2: "Ammonia",
        3: "Acetaldehyde",
        4: "Acetone",
        5: "Toluene",
    },
    "har": {
        0: "WALKING",
        1: "WALKING_UPSTAIRS",
        2: "WALKING_DOWNSTAIRS",
        3: "SITTING",
        4: "STANDING",
        5: "LAYING",
    },
    "amazon": {0: "negative", 1: "positive"},
}
DOMAINS = ("books", "dvd", "electronics", "kitchen")
FEATURE_COUNTS = {"gas": 128, "har": 561, "amazon": 1}
GAS_REFERENCE_ROWS = 13910
HAR_REFERENCE_COUNTS = {"train": 7352, "test": 2947}


@dataclass(frozen=True)
class Record:
    sample_id: str
    domain: str
    pool: str
    group_id: str
    duplicate_id: str
    source_file: str
    row: int
    label: int | None
    values: tuple[float, ...] | str
    product_id: str | None = None

    def metadata(self) -> dict:
        return {
            name: getattr(self, name)
            for name in (
                "sample_id",
                "domain",
                "pool",
                "group_id",
                "duplicate_id",
                "source_file",
                "row",
                "product_id",
            )
        }


@dataclass
class ParsedDataset:
    dataset: str
    records: list[Record]
    original_counts: dict[str, int]
    exclusions: list[dict] = field(default_factory=list)
    notes: dict = field(default_factory=dict)


def _id(dataset: str, file: str, row: int) -> str:
    return dataset + "_" + hashlib.sha256(f"{file}:{row}".encode()).hexdigest()


def numeric_signature(values: tuple[float, ...]) -> str:
    array = np.asarray(values, dtype="<f8").copy()
    array[array == 0] = 0  # Treat +0 and -0 as identical, without rounding any feature.
    return hashlib.sha256(array.tobytes()).hexdigest()


def normalized_body(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).split())


def canonical_body(value: str) -> str:
    return normalized_body(html.unescape(value))


def _find(root: Path, suffix: str) -> Path:
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.as_posix().endswith(suffix))
    if len(paths) != 1:
        raise DataError(f"Expected exactly one {suffix}; found {len(paths)}")
    return paths[0]


def parse_gas(root: Path) -> ParsedDataset:
    records, counts = [], {}
    for batch in range(1, 11):
        path = _find(root, f"/batch{batch}.dat")
        relative = path.relative_to(root).as_posix()
        count = 0
        with path.open(encoding="ascii") as stream:
            for row, line in enumerate(stream, 1):
                fields = line.split()
                if len(fields) != 129 or fields[0] not in {"1", "2", "3", "4", "5", "6"}:
                    raise DataError(f"Gas schema mismatch: {relative}:{row}")
                values = []
                for index, token in enumerate(fields[1:], 1):
                    pair = token.split(":")
                    if len(pair) != 2 or pair[0] != str(index):
                        raise DataError(f"Gas feature indices must be 1..128: {relative}:{row}")
                    values.append(float(pair[1]))
                if not np.isfinite(values).all():
                    raise DataError(f"Non-finite gas feature: {relative}:{row}")
                values = tuple(values)
                duplicate = numeric_signature(values)
                records.append(
                    Record(
                        _id("gas", relative, row),
                        f"b{batch:02d}",
                        "labelled",
                        f"gas:{duplicate}",
                        duplicate,
                        relative,
                        row,
                        int(fields[0]) - 1,
                        values,
                    )
                )
                count += 1
        counts[f"b{batch:02d}"] = count
    if sum(counts.values()) != GAS_REFERENCE_ROWS:
        raise DataError(
            f"Gas archive version mismatch: expected 13910 rows; got {sum(counts.values())}"
        )
    return ParsedDataset("gas", records, counts)


def parse_har(root: Path) -> ParsedDataset:
    feature_file = _find(root, "/features.txt")
    features = feature_file.read_text(encoding="ascii").splitlines()
    if len(features) != 561 or [line.split()[0] for line in features] != [
        str(i) for i in range(1, 562)
    ]:
        raise DataError("HAR feature schema must contain 561 ordered indices")
    label_file = _find(root, "/activity_labels.txt")
    expected = {str(key + 1): name for key, name in CLASS_MAPS["har"].items()}
    if dict(line.split() for line in label_file.read_text().splitlines()) != expected:
        raise DataError("HAR activity label mapping differs from the official six activities")
    records, counts, groups = [], {}, {}
    for partition in ("train", "test"):
        path = _find(root, f"/{partition}/X_{partition}.txt")
        X = np.loadtxt(path, dtype=np.float64, ndmin=2)
        y = np.loadtxt(_find(root, f"/{partition}/y_{partition}.txt"), ndmin=1)
        subjects = np.loadtxt(_find(root, f"/{partition}/subject_{partition}.txt"), ndmin=1)
        if X.shape[1] != 561 or X.shape[0] != y.size or y.shape != subjects.shape:
            raise DataError("HAR X, y and subject row counts/schema disagree")
        if not np.isfinite(X).all() or not np.isin(y, range(1, 7)).all():
            raise DataError("HAR contains non-finite features or invalid labels")
        if not np.isin(subjects, range(1, 31)).all():
            raise DataError("HAR contains invalid participant IDs")
        groups[partition] = set(subjects.astype(int).tolist())
        relative = path.relative_to(root).as_posix()
        for row, (values, label, subject) in enumerate(zip(X, y, subjects, strict=True), 1):
            values = tuple(values.tolist())
            records.append(
                Record(
                    _id("har", relative, row),
                    partition,
                    partition,
                    f"participant:{int(subject):02d}",
                    numeric_signature(values),
                    relative,
                    row,
                    int(label) - 1,
                    values,
                )
            )
        counts[partition] = len(X)
    if counts != HAR_REFERENCE_COUNTS or groups["train"] & groups["test"]:
        raise DataError("HAR official partition version mismatch or participant leakage")
    if (
        len(groups["train"]) != 21
        or len(groups["test"]) != 9
        or len(groups["train"] | groups["test"]) != 30
    ):
        raise DataError("HAR requires 21 source and nine target participants")
    return ParsedDataset(
        "har",
        records,
        counts,
        notes={
            "official_participants": {key: sorted(value) for key, value in groups.items()},
        },
    )


def review_blocks(path: Path) -> tuple[list[str], str]:
    """Pseudo-XML is data: no entity resolver, executable XML or filename labels."""
    raw = path.read_bytes()
    try:
        text, encoding = raw.decode("utf-8"), "utf-8"
    except UnicodeDecodeError:
        text, encoding = raw.decode("latin-1"), "latin-1"
    pattern = re.compile(r"<review>\s*(.*?)\s*</review>", re.DOTALL)
    blocks = pattern.findall(text)
    if not blocks or pattern.sub("", text).strip() or text.count("<review>") != len(blocks):
        raise DataError(f"Malformed review container in {path.name}")
    return blocks, encoding


def _field(block: str, name: str) -> str | None:
    # Anchor to field lines: review prose containing inline tags is still prose.
    matches = re.findall(rf"^<{name}>\s*\n?(.*?)\n?</{name}>\s*$", block, re.DOTALL | re.MULTILINE)
    return matches[0].strip() if len(matches) == 1 else None


def parse_amazon(root: Path) -> ParsedDataset:
    records, counts, exclusions, encodings = [], {}, [], {}
    products = Counter()
    for domain in DOMAINS:
        directory = "kitchen_&_housewares" if domain == "kitchen" else domain
        for pool_file in ("positive.review", "negative.review", "unlabeled.review"):
            suffix = (
                "/book.unlabeled"
                if domain == "books" and pool_file == "unlabeled.review"
                else f"/{directory}/{pool_file}"
            )
            path = _find(root, suffix)
            relative = path.relative_to(root).as_posix()
            blocks, encoding = review_blocks(path)
            encodings[relative] = encoding
            counts[f"{domain}/{pool_file}"] = len(blocks)
            labelled = pool_file != "unlabeled.review"
            for row, block in enumerate(blocks, 1):
                sample_id = _id("amazon", relative, row)
                body = _field(block, "review_text")
                reason, label = None, None
                if body is None or not canonical_body(body):
                    reason = "missing_or_empty_review_body"
                if labelled:
                    rating = _field(block, "rating")
                    try:
                        number = float(rating) if rating is not None else float("nan")
                    except ValueError:
                        number = float("nan")
                    if number not in (1, 2, 3, 4, 5):
                        reason = reason or "invalid_labelled_rating"
                    elif number == 3:
                        reason = reason or "neutral_labelled_rating"
                    else:
                        label = int(number > 3)
                # Unlabelled ratings are intentionally never parsed.
                if reason:
                    exclusions.append(
                        {
                            "sample_id": sample_id,
                            "source_file": relative,
                            "row": row,
                            "reason": reason,
                        }
                    )
                    continue
                body = canonical_body(body)
                duplicate = hashlib.sha256(body.encode("utf-8")).hexdigest()
                product = _field(block, "asin")
                product = product.upper() if product else None
                if product is not None and re.fullmatch(r"[A-Z0-9]{10}", product) is None:
                    product = None
                products["usable" if product else "missing_or_invalid"] += 1
                records.append(
                    Record(
                        sample_id,
                        domain,
                        "labelled" if labelled else "unlabelled",
                        f"amazon:{duplicate}",
                        duplicate,
                        relative,
                        row,
                        label,
                        body,
                        product,
                    )
                )
    return ParsedDataset(
        "amazon",
        records,
        counts,
        exclusions,
        {
            "file_encodings": encodings,
            "product_id_coverage": dict(products),
            "canonical_body_policy": (
                "HTML entity decoding, Unicode NFKC, collapse whitespace; "
                "preserve case and negation"
            ),
        },
    )


def clean_duplicates(parsed: ParsedDataset) -> ParsedDataset:
    """Quarantine conflicts; deterministically retain one record per permitted pool."""
    by_body = defaultdict(list)
    for record in parsed.records:
        by_body[record.duplicate_id].append(record)
    retained = []
    for _duplicate, records in sorted(by_body.items()):
        labels = {r.label for r in records if r.label is not None}
        conflict = len(labels) > 1
        har_group_conflict = parsed.dataset == "har" and len({r.group_id for r in records}) > 1
        if conflict or har_group_conflict:
            for record in records:
                parsed.exclusions.append(
                    {
                        **record.metadata(),
                        "reason": "conflicting_labels"
                        if conflict
                        else "identical_features_across_participants",
                    }
                )
            continue
        seen = {}
        # Labelled has precedence over unlabelled within a category.
        for record in sorted(
            records, key=lambda r: (r.domain, r.pool == "unlabelled", r.source_file, r.row)
        ):
            key = (record.domain, record.group_id if parsed.dataset == "har" else "")
            if key in seen:
                parsed.exclusions.append(
                    {
                        **record.metadata(),
                        "reason": "duplicate_body_or_exact_features",
                        "retained_sample_id": seen[key].sample_id,
                    }
                )
            else:
                seen[key] = record
                retained.append(record)
    parsed.records = sorted(retained, key=lambda r: (r.domain, r.pool, r.source_file, r.row))
    if not parsed.records:
        raise DataError("No records remain after quarantine")
    return parsed
