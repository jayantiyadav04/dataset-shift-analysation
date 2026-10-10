"""Configuration failures that could invalidate the declared scientific protocol."""

import copy

import pytest

from shiftlab.experiments.config import ConfigError, load_config, validate_config
from shiftlab.experiments.provenance import content_hash


def test_primary_and_smoke_counts(primary_dict, smoke_dict):
    primary = validate_config(primary_dict)
    assert primary.planned_counts() == {
        "tasks": 22,
        "main_fits": 990,
        "majority_evaluations": 110,
        "sensitivity_fits": 180,
    }
    assert validate_config(smoke_dict).planned_counts() == {
        "tasks": 1,
        "main_fits": 1,
        "majority_evaluations": 0,
        "sensitivity_fits": 0,
    }


@pytest.mark.parametrize("method", ["source_only", "importance_weighting", "coral"])
def test_each_method_name_validates_in_fit_only_smoke(smoke_dict, method):
    smoke_dict["methods"] = [method]
    assert validate_config(smoke_dict).methods == (method,)


@pytest.mark.parametrize(
    "field,value",
    [
        ("models", ["neural_network"]),
        ("methods", ["CORAL"]),
        ("datasets", ["iris"]),
        ("seeds", [17]),
        ("seeds", [17, 29, 43, 71, 71]),
        ("seeds", [True, 29, 43, 71, 101]),
        ("task_filter", ["gas_b01_b01"]),
        ("task_filter", ["amazon_books_books"]),
        ("include_majority_baseline", False),
        ("schema_version", True),
        ("split_seed", 2026.0),
    ],
)
def test_reject_invalid_primary_contract(primary_dict, field, value):
    primary_dict[field] = value
    with pytest.raises(ConfigError):
        validate_config(primary_dict)


@pytest.mark.parametrize(
    "section,field,value",
    [
        ("preprocessing", "fit_scope", "source_and_target_test"),
        ("tuning", "reuse_source_parameters_across_methods", False),
        ("execution", "default_stage", "evaluate"),
        ("execution", "output_root", "../other_study"),
        ("execution", "output_root", "C:/private"),
        ("execution", "output_root", "src/results"),
        ("importance_weighting", "weight_clip_upper", float("nan")),
        ("importance_weighting", "weight_clip_upper", 0.01),
        ("importance_weighting", "weight_clip_upper", 20.0),
        ("coral", "ridge", 0.01),
        ("coral", "centre_and_restore_target_mean", False),
        ("evaluation", "paired_seed_average", False),
        ("evaluation", "bootstrap_unit_policy", "rows_for_har"),
    ],
)
def test_reject_unsafe_or_changed_primary_settings(primary_dict, section, field, value):
    primary_dict[section][field] = value
    with pytest.raises(ConfigError):
        validate_config(primary_dict)


@pytest.mark.parametrize("section", [None, "data", "preprocessing", "execution", "evaluation"])
def test_unknown_fields_are_never_silently_ignored(primary_dict, section):
    container = primary_dict if section is None else primary_dict[section]
    container["target_task_labels"] = [0, 1]
    with pytest.raises(ConfigError, match="Extra inputs are not permitted"):
        validate_config(primary_dict)


def test_smoke_task_must_belong_to_selected_dataset(smoke_dict):
    smoke_dict["task_filter"] = ["har_official_group_transfer"]
    with pytest.raises(ConfigError):
        validate_config(smoke_dict)


def test_yaml_rejects_objects_duplicate_keys_aliases_and_malformed_inputs(tmp_path):
    documents = [
        "!!python/object/apply:os.system ['echo unexpected']",
        "schema_version: 1\nschema_version: 2\n",
        "key: &loop [*loop]\n",
        "[invalid",
        "[]",
        "null",
        "123: value",
    ]
    for document in documents:
        path = tmp_path / "unsafe.yaml"
        path.write_text(document, encoding="utf-8")
        with pytest.raises(ConfigError):
            load_config(path)


def test_config_is_immutable_and_input_changes_cannot_change_hash(primary_dict):
    config = validate_config(primary_dict)
    digest = content_hash(config.model_dump(mode="json"))
    original = copy.deepcopy(primary_dict)
    primary_dict["seeds"][0] = 99
    primary_dict["importance_weighting"]["domain_C"] = 99
    assert content_hash(config.model_dump(mode="json")) == digest
    with pytest.raises(ValueError):
        config.execution.output_root = "changed"
    assert config == validate_config(original)


def test_missing_or_oversized_config_fails(tmp_path):
    with pytest.raises(ConfigError, match="Cannot read"):
        load_config(tmp_path / "missing.yaml")
    path = tmp_path / "large.yaml"
    path.write_text("#" * 100_001, encoding="utf-8")
    with pytest.raises(ConfigError, match="100 KB"):
        load_config(path)
