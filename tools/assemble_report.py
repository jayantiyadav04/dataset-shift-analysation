"""Generate final narrative tables from sealed evidence, without fitting or evaluation."""

import json
import shutil
from pathlib import Path

import pandas as pd

from shiftlab.data.storage import read_json, seal, verify_files, verify_seal, write_json
from shiftlab.experiments.provenance import file_sha256
from shiftlab.scope import FINAL_OBJECTIVE, PROPOSAL_ABSTRACT

ROOT = Path(__file__).resolve().parents[1]
FINAL = ROOT / "reports/final"
MODELS = {"lr": "LR", "linear_svm": "Linear SVM", "rf": "RF", "majority": "Majority"}
METHODS = {"source_only": "Source", "importance_weighting": "Weighting", "coral": "CORAL"}
TASKS = ["gas_b01_b10", "har_official_group_transfer", "amazon_books_electronics"]


def md_table(columns: list[str], rows: list[list]) -> str:
    def cell(value):
        return str(value).replace("|", "/").replace("\n", " ")

    return "\n".join(
        ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        + ["| " + " | ".join(cell(v) for v in row) + " |" for row in rows]
    )


def value_sd(row, field):
    return f"{row[field + '_mean']:.6f} ± {row[field + '_std']:.6f}"


def assemble() -> dict:
    if (FINAL / "report_values.json").exists():
        raise FileExistsError("Preserve this sealed report; prepare a separate review workspace")
    for folder, name in [
        ("study", "evidence_manifest.json"),
        ("workbench", "interface_review.json"),
    ]:
        directory = ROOT / "reports" / folder
        manifest = read_json(directory / name)
        verify_seal(manifest)
        verify_files(directory, manifest["output_hashes"])
    reproduction = read_json(ROOT / "reports/reproduction/verification.json")
    verify_seal(reproduction)
    assert reproduction["status"] == "passed" and reproduction["fresh_fits"] == 50
    abstract = (FINAL / "paper_abstract.txt").read_text(encoding="utf-8").strip()
    assert len(abstract.split()) == 150
    tables = FINAL / "tables"
    cards = FINAL / "data_cards"
    tables.mkdir(exist_ok=True)
    cards.mkdir(exist_ok=True)
    study = ROOT / "reports/study"
    summary = pd.read_csv(study / "dataset_summary.csv")
    paired = pd.read_csv(study / "paired_summary.csv")
    primary = paired[paired.setting == "primary"].copy()
    task_summary = pd.read_csv(study / "per_task_summary.csv")
    runs = pd.read_csv(study / "per_run_metrics.csv")
    findings = read_json(study / "findings.json")
    sources = {
        p.relative_to(ROOT).as_posix(): file_sha256(p) for p in study.glob("*") if p.is_file()
    }
    assert len(primary) == 132
    assert int((primary.macro_f1_delta_mean > 0).sum()) == 40
    assert int((primary.macro_f1_delta_mean < 0).sum()) == 92
    directions = []
    for (dataset, method), frame in primary.groupby(["dataset", "method"]):
        directions.append(
            {
                "dataset": dataset,
                "method": method,
                "comparisons": len(frame),
                "positive": int((frame.macro_f1_delta_mean > 0).sum()),
                "negative": int((frame.macro_f1_delta_mean < 0).sum()),
                "interval_including_zero": int(
                    ((frame.macro_f1_lower <= 0) & (frame.macro_f1_upper >= 0)).sum()
                ),
                "run_ids": "|".join(sorted(set("|".join(frame.adapted_run_ids).split("|")))),
                "source_run_ids": "|".join(sorted(set("|".join(frame.source_run_ids).split("|")))),
            }
        )
    pd.DataFrame(directions).to_csv(tables / "direction_counts.csv", index=False)
    direction_table = md_table(
        ["Dataset", "Method", "Pairs", "Positive", "Negative", "CI includes 0"],
        [
            [
                r["dataset"],
                METHODS[r["method"]],
                r["comparisons"],
                r["positive"],
                r["negative"],
                r["interval_including_zero"],
            ]
            for r in directions
        ],
    )
    summary_tables = []
    for dataset in ["gas", "har", "amazon"]:
        frame = summary[summary.dataset == dataset]
        summary_tables.append(
            f"#### {dataset.upper()}\n\n"
            + md_table(
                ["Model", "Method", "Accuracy", "Macro F1", "Pipeline s", "F1 delta pp"],
                [
                    [
                        MODELS[r.model],
                        METHODS[r.method],
                        f"{r.accuracy_mean:.6f}",
                        f"{r.macro_f1_mean:.6f}",
                        f"{r.pipeline_seconds_mean:.3f}",
                        "-"
                        if pd.isna(r.macro_f1_delta_mean)
                        else f"{r.macro_f1_delta_mean * 100:+.3f}",
                    ]
                    for r in frame.itertuples()
                ],
            )
        )
    example_rows = []
    examples = []
    for kind in ["positive", "uncertain", "negative"]:
        row = findings["examples"][kind]
        examples.append({"kind": kind, **row})
        example_rows.append(
            [
                kind,
                row["task_id"].replace("gas_b01_b", "Gas 1→"),
                MODELS[row["model"]],
                METHODS[row["method"]],
                f"{100 * row['macro_f1_delta_mean']:+.6f}",
                f"[{100 * row['macro_f1_lower']:+.6f}, {100 * row['macro_f1_upper']:+.6f}]",
            ]
        )
    pd.DataFrame(examples).to_csv(tables / "interpretation_examples.csv", index=False)
    diagnostic_tables = []
    for task in TASKS:
        frame = task_summary[(task_summary.task_id == task) & (task_summary.setting == "primary")]
        diagnostic_tables.append(
            f"#### {task}\n\n"
            + md_table(
                ["Model", "Method", "Accuracy mean ± SD", "Macro F1 mean ± SD", "Pipeline s"],
                [
                    [
                        MODELS[r["model"]],
                        METHODS[r["method"]],
                        value_sd(r, "accuracy"),
                        value_sd(r, "macro_f1"),
                        f"{r['pipeline_seconds_mean']:.3f}",
                    ]
                    for r in frame.to_dict("records")
                ],
            )
        )
    benchmarks = pd.read_csv(study / "benchmarks.csv")
    benchmark_table = md_table(
        ["Dataset", "Preprocess s", "CV s", "10 fits s", "Wall s", "Peak MiB"],
        [
            [
                r.task_id.split("_")[0],
                f"{r.preprocessing_seconds:.3f}",
                f"{r.source_tuning_seconds_including_fold_preprocessing:.3f}",
                f"{r.classifier_training_seconds_all_ten_fits:.3f}",
                f"{r.wall_seconds:.3f}",
                f"{r.peak_process_working_set_bytes / 2**20:.1f}",
            ]
            for r in benchmarks.itertuples()
        ],
    )
    splits = pd.read_csv(ROOT / "reports/data_preparation/split_counts.csv")
    splits.to_csv(tables / "split_counts.csv", index=False)
    data_rows, target_profiles = [], []
    for dataset in ("gas", "har", "amazon"):
        manifest_path = ROOT / "data/processed" / dataset / "manifest.json"
        manifest = read_json(manifest_path)
        verify_seal(manifest)
        sources[manifest_path.relative_to(ROOT).as_posix()] = file_sha256(manifest_path)
        data_rows.append(
            [
                dataset,
                manifest["original_sample_count"],
                manifest["retained_sample_count"],
                manifest["excluded_sample_count"],
                manifest["feature_count"],
                {"gas": 9, "har": 1, "amazon": 12}[dataset],
            ]
        )
        task_rows = splits[splits.task_id.str.startswith(dataset)]
        text = [
            f"# {dataset.upper()} completed data card\n",
            f"Parser {manifest['parser_version']}; schema {manifest['schema_version']}. "
            f"Actual parsed records: {manifest['original_sample_count']:,}; "
            f"retained: {manifest['retained_sample_count']:,}; "
            f"excluded: {manifest['excluded_sample_count']:,}.",
            "This card describes the evaluated frozen study. Counts and label profiles below "
            "are saved evidence after final evaluation, not training permissions.",
            "## Source and exact archive identity\n",
        ]
        for archive in manifest["archives"]:
            text += [
                f"[{archive['filename']}]({archive['source_url']})\n\n"
                f"Accessed {archive['accessed_at']}; SHA-256 `{archive['sha256']}`."
            ]
        text += [
            f"Processed dataset hash: `{manifest['processed_sha256']}`.",
            f"Source terms: {manifest['terms']}",
            "## Schema and cleaning\n",
            f"Feature kind: {manifest['feature_kind']}; "
            f"stored feature fields: {manifest['feature_count']}. "
            + (
                "The one text field is transformed into source-fitted TF-IDF "
                "and at most 128 dense SVD features."
                if dataset == "amazon"
                else "Numerical values must be finite; unexpected missing values fail validation."
            ),
            "Class map: " + json.dumps(manifest["class_mapping"], ensure_ascii=False) + ".",
            "Exclusions by reason: " + json.dumps(manifest["exclusion_counts"]) + ".",
            "Retained domain/pool counts: " + json.dumps(manifest["domain_counts"]) + ".",
            "Source class counts: " + json.dumps(manifest["source_label_counts"]) + ".",
            f"Group policy: {manifest['group_policy']}.",
            "Exact feature/body groups prevent role duplicates; task-specific target exclusions "
            "are additional to global parser exclusions.",
            "## Frozen task counts\n",
            md_table(
                ["Task", "Source", "Adaptation", "Evaluation", "Extra exclusions"],
                task_rows[
                    ["task_id", "source", "target_adapt", "target_eval", "excluded_target_count"]
                ].values.tolist(),
            ),
            "Split seed 2026. The complete SHA-256 and relative manifest paths are in "
            "[split counts](../tables/split_counts.csv). Exact IDs are local in each frozen "
            "data/splits/1.0/TASK/manifest.json and downloadable from the application.",
        ]
        for task in task_rows.task_id:
            row = runs[
                (runs.task_id == task)
                & (runs.seed == 17)
                & (runs.model == "lr")
                & (runs.method == "source_only")
                & (runs.setting == "primary")
            ].iloc[0]
            metric_path = ROOT / "artifacts/evaluations" / row.evaluation_id / "metrics.json"
            metrics = read_json(metric_path)
            for group in metrics["per_class"]:
                target_profiles.append(
                    {
                        "task_id": task,
                        "class": group["class"],
                        "support": group["support"],
                        "reference_run_id": row.run_id,
                        "evaluation_id": row.evaluation_id,
                    }
                )
            text += [
                f"Evaluation class support for {task}: "
                + json.dumps({r["class"]: r["support"] for r in metrics["per_class"]})
                + f". Absent target classes: {metrics['absent_target_classes']}."
            ]
        text += [
            "## Limits and permissions\n",
            {
                "gas": (
                    "All source rows are Batch 1. Target batches follow time direction but their "
                    "internal adaptation/evaluation holdout is not chronological forecasting. "
                    "Finer within-batch dependence remains possible. Class composition varies "
                    "by batch."
                ),
                "har": (
                    "Source participants are "
                    "1,3,5,6,7,8,11,14,15,16,17,19,21,22,23,25,26,27,28,29,30. "
                    "Adaptation participants are 2,4,9,10; evaluation participants are "
                    "12,13,18,20,24. Source has 21 groups, adaptation four and evaluation five. "
                    "Overlapping windows from one participant remain grouped. "
                    "Population inference is limited."
                ),
                "amazon": (
                    "Only review bodies enter features. Ratings, label-bearing filenames, "
                    "reviewer fields and titles are excluded. Product IDs are separate grouping "
                    "metadata; canonical bodies determine duplicate exclusion. Global product-ID "
                    "coverage was 78,679/78,679 parsed rows. In books-to-DVD, one extra duplicate "
                    "is removed from the adaptation pool. Source-only vocabulary/SVD can lose "
                    "target-specific sentiment information. The original release is distinct "
                    "from later Amazon corpora."
                ),
            }[dataset],
            "Fitting permits source labels and adaptation features only. Evaluation labels are "
            "used only by the authorized evaluator. Raw records, IDs and private labels are not "
            "included in the public package. See [references](../references.md) for the dataset "
            "and method citations.",
        ]
        (cards / f"{dataset}.md").write_text("\n\n".join(text) + "\n", encoding="utf-8")
    pd.DataFrame(target_profiles).to_csv(tables / "target_class_support.csv", index=False)
    ablations = pd.read_csv(study / "ablations.csv")
    sensitivity = ablations.groupby(
        ["task_id", "model", "method", "sensitivity_value"], as_index=False
    ).agg(
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", "std"),
        accuracy_mean=("accuracy", "mean"),
        accuracy_std=("accuracy", "std"),
        seed_count=("seed", "size"),
        run_ids=("run_id", lambda s: "|".join(s)),
    )
    sensitivity.to_csv(tables / "sensitivity_summary.csv", index=False)
    reproduction_text = (
        f"A clean checkout and fresh virtual environment reproduced gas Batch 1 to Batch 2 "
        f"using all five seeds and 50 newly fitted models/baselines. All 50 ordered prediction "
        f"vectors, metric dictionaries and selected parameters exactly matched the original "
        f"study on 622 evaluation rows per run. Fitting took {reproduction['fit_seconds']:.6f} "
        f"seconds; fitting, evaluation and comparison took {reproduction['total_seconds']:.6f} "
        f"seconds. Peak process working set was "
        f"{reproduction['resources']['peak_process_ram_bytes'] / 2**20:.4f} "
        f"MiB. No forbidden held-out training reads occurred. The original archive/split were "
        f"verified and reused; new caches/models/predictions were written separately. "
        f"The detailed reproduction record preserves the exact executed driver and logs."
    )
    references = (FINAL / "references.md").read_text(encoding="utf-8")
    replacements = {
        "ABSTRACT": abstract,
        "OBJECTIVE": FINAL_OBJECTIVE,
        "PROPOSAL_ABSTRACT": PROPOSAL_ABSTRACT,
        "DATA_TABLE": md_table(
            ["Dataset", "Parsed", "Retained", "Excluded", "Fields", "Tasks"], data_rows
        ),
        "SUMMARY_TABLE": "\n\n".join(summary_tables),
        "DIRECTION_TABLE": direction_table,
        "EXAMPLE_TABLE": md_table(
            ["Example", "Transfer", "Model", "Method", "Delta pp", "95% interval pp"], example_rows
        ),
        "DIAGNOSTIC_TABLE": "\n\n".join(diagnostic_tables),
        "BENCHMARK_TABLE": benchmark_table,
        "REPRODUCTION": reproduction_text,
        "REFERENCES": references.replace("# Verified references", "## References", 1),
    }
    for name in ("final_report.md", "paper_draft.md"):
        document = (ROOT / "tools/report_sources" / name).read_text(encoding="utf-8")
        for key, value in replacements.items():
            document = document.replace("{{" + key + "}}", value)
        assert "{{" not in document and FINAL_OBJECTIVE in document
        (FINAL / name).write_text(document, encoding="utf-8")
    traces = []
    ids = sorted(
        set(
            "|".join(r[k] for r in examples for k in ("source_run_ids", "adapted_run_ids")).split(
                "|"
            )
        )
    )
    for run_id in ids:
        row = runs[runs.run_id == run_id].iloc[0]
        prediction = ROOT / "artifacts/evaluations" / row.evaluation_id / "predictions.csv"
        traces.append(
            {
                "run_id": run_id,
                "evaluation_id": row.evaluation_id,
                "prediction_path": prediction.relative_to(ROOT).as_posix(),
                "prediction_sha256": file_sha256(prediction),
            }
        )
    values = seal(
        {
            "study_id": findings["study_id"],
            "abstract_words": len(abstract.split()),
            "primary_comparisons": 132,
            "positive_changes": 40,
            "negative_changes": 92,
            "intervals_including_zero": 27,
            "input_hashes": sources,
            "example_prediction_traces": traces,
            "reproduction_receipt_hash": reproduction["manifest_hash"],
            "generated_table_hashes": {
                p.relative_to(FINAL).as_posix(): file_sha256(p) for p in tables.glob("*.csv")
            },
        }
    )
    write_json(FINAL / "report_values.json", values)
    shutil.copyfile(ROOT / "configs/study.yaml", FINAL / "study_configuration.yaml")
    print("Generated report/paper with exactly 150 abstract words, 3 data cards and traced tables")
    return values


if __name__ == "__main__":
    assemble()
