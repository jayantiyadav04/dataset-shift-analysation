"""Strict, immutable validation of the supplied YAML research contracts."""

import math
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictBool,
    StrictInt,
    StrictStr,
    ValidationError,
    model_validator,
)

from shiftlab.contracts import DatasetName, MethodName, ModelName
from shiftlab.scope import PRIMARY_SEEDS

PositiveInt = Annotated[StrictInt, Field(gt=0)]
Seed = Annotated[StrictInt, Field(ge=0, le=2**32 - 1)]
PositiveFloat = Annotated[float, Field(strict=True, gt=0, allow_inf_nan=False)]


class ConfigError(ValueError):
    """Invalid configuration; never starts a job."""


class Schema(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class DataConfig(Schema):
    gas_uci_id: Literal[224]
    har_uci_id: Literal[240]
    gas_source_batch: Literal[1]
    gas_target_batches: tuple[PositiveInt, ...]
    gas_target_adaptation_fraction: Annotated[float, Field(strict=True, gt=0, lt=1)]
    har_target_adaptation_participants: Literal[4]
    amazon_domains: tuple[StrictStr, ...]
    amazon_all_ordered_pairs: StrictBool

    @model_validator(mode="after")
    def frozen_splits(self) -> Self:
        if self.gas_target_batches != tuple(range(2, 11)):
            raise ValueError("Gas transfers must run from Batch 1 to Batches 2 through 10")
        if self.gas_target_adaptation_fraction != 0.5:
            raise ValueError("The declared gas adaptation fraction is 0.5")
        if self.amazon_domains != ("books", "dvd", "electronics", "kitchen"):
            raise ValueError("Use the original four Amazon domains in the declared order")
        if not self.amazon_all_ordered_pairs:
            raise ValueError("All twelve ordered Amazon transfers are required")
        return self


class PreprocessingConfig(Schema):
    numeric_scaler: Literal["standard"]
    text_max_features: PositiveInt
    text_min_df: PositiveInt
    text_ngram_range: tuple[PositiveInt, PositiveInt]
    text_sublinear_tf: StrictBool
    text_svd_components: PositiveInt
    text_scaler_after_svd: Literal["standard"]
    fit_scope: Literal["source_only"]

    @model_validator(mode="after")
    def shared_representation(self) -> Self:
        if (
            self.text_max_features,
            self.text_min_df,
            self.text_ngram_range,
            self.text_sublinear_tf,
            self.text_svd_components,
        ) != (20000, 2, (1, 2), True, 128):
            raise ValueError("Preserve the declared source-fitted TF-IDF/SVD recipe")
        return self


class TuningConfig(Schema):
    folds: Literal[5]
    metric: Literal["macro_f1"]
    lr_C: tuple[PositiveFloat, ...]
    linear_svm_C: tuple[PositiveFloat, ...]
    rf_n_estimators: Literal[200]
    rf_max_depth: tuple[PositiveInt | None, ...]
    rf_min_samples_leaf: tuple[PositiveInt, ...]
    reuse_source_parameters_across_methods: StrictBool

    @model_validator(mode="after")
    def source_search(self) -> Self:
        if self.lr_C != (0.1, 1.0, 10.0) or self.linear_svm_C != (0.1, 1.0, 10.0):
            raise ValueError("Use the frozen LR and linear SVM source search grids")
        if self.rf_max_depth != (None, 20) or self.rf_min_samples_leaf != (1, 3):
            raise ValueError("Use the frozen RF source search grid")
        if not self.reuse_source_parameters_across_methods:
            raise ValueError("Paired methods must reuse source-selected parameters")
        return self


class ImportanceConfig(Schema):
    source_crossfit_folds: Literal[5]
    domain_C: PositiveFloat
    probability_epsilon: Annotated[float, Field(strict=True, gt=0, lt=0.5)]
    weight_clip_lower: PositiveFloat
    weight_clip_upper: PositiveFloat
    normalise_mean_one: StrictBool

    @model_validator(mode="after")
    def weight_bounds(self) -> Self:
        if self.weight_clip_lower >= self.weight_clip_upper:
            raise ValueError("Weight clipping lower bound must be below upper bound")
        if not self.normalise_mean_one:
            raise ValueError("Importance weights must be normalised to mean one")
        if (self.domain_C, self.probability_epsilon, self.weight_clip_lower) != (1.0, 0.001, 0.05):
            raise ValueError("Preserve the declared domain-classifier and probability settings")
        return self


class CoralConfig(Schema):
    ridge: PositiveFloat
    eigenvalue_floor: PositiveFloat
    centre_and_restore_target_mean: StrictBool

    @model_validator(mode="after")
    def centred_variant(self) -> Self:
        if not self.centre_and_restore_target_mean or self.eigenvalue_floor != 1e-8:
            raise ValueError("Use centred CORAL with target mean restoration and floor 1e-8")
        return self


class EvaluationConfig(Schema):
    primary_metric: Literal["macro_f1"]
    supporting_metrics: tuple[StrictStr, ...]
    macro_class_policy: Literal["present_in_full_target_test"]
    zero_division: Literal[0]
    bootstrap_replicates: Literal[1000]
    bootstrap_seed: Literal[2026]
    bootstrap_unit_policy: Literal["dataset_specific_groups"]
    paired_seed_average: StrictBool

    @model_validator(mode="after")
    def uncertainty_policy(self) -> Self:
        if self.supporting_metrics != ("accuracy", "macro_precision", "macro_recall"):
            raise ValueError("Preserve the declared supporting metric set")
        if not self.paired_seed_average:
            raise ValueError("Bootstrap uses paired deltas averaged over fitted seeds")
        return self


class AblationConfig(Schema):
    tasks: tuple[StrictStr, ...]
    weight_clip_upper: tuple[PositiveFloat, ...]
    coral_ridge: tuple[PositiveFloat, ...]

    @model_validator(mode="after")
    def predeclared_values(self) -> Self:
        if self.weight_clip_upper != (5.0, 10.0, 20.0):
            raise ValueError("Predeclared upper clipping bounds are 5, 10 and 20")
        if self.coral_ridge != (0.0001, 0.001, 0.01):
            raise ValueError("Predeclared CORAL ridges are 0.0001, 0.001 and 0.01")
        return self


class ExecutionConfig(Schema):
    default_stage: Literal["fit_only"]
    max_parallel_jobs: Literal[1]
    threads_per_job: Literal[1]
    resume: StrictBool
    output_root: StrictStr

    @model_validator(mode="after")
    def relative_output(self) -> Self:
        path = PurePosixPath(self.output_root)
        if (
            not self.output_root
            or path.is_absolute()
            or ".." in path.parts
            or "\\" in self.output_root
            or ":" in self.output_root
            or path == PurePosixPath(".")
        ):
            raise ValueError("output_root must be a nonempty relative path without traversal")
        if path.parts[0] in {"src", "app", "configs", "tests", ".git", "docs", "reports"}:
            raise ValueError("output_root must not replace source or specification directories")
        return self


def _unique(values: tuple[Any, ...], name: str) -> None:
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must not contain duplicate entries")


class StudyConfig(Schema):
    schema_version: Literal[1]
    protocol_version: Annotated[StrictStr, Field(pattern=r"^[1-9]\d*\.\d+$")]
    study_name: Annotated[StrictStr, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")]
    mode: Literal["primary", "smoke"]
    split_seed: Literal[2026]
    seeds: tuple[Seed, ...]
    datasets: tuple[DatasetName, ...]
    task_filter: tuple[StrictStr, ...]
    models: tuple[ModelName, ...]
    methods: tuple[MethodName, ...]
    include_majority_baseline: StrictBool
    data: DataConfig
    preprocessing: PreprocessingConfig
    tuning: TuningConfig
    importance_weighting: ImportanceConfig
    coral: CoralConfig
    evaluation: EvaluationConfig
    ablations: AblationConfig
    execution: ExecutionConfig

    @model_validator(mode="after")
    def study_boundaries(self) -> Self:
        for name in ("seeds", "datasets", "models", "methods", "task_filter"):
            values = getattr(self, name)
            _unique(values, name)
            if name != "task_filter" and not values:
                raise ValueError(f"{name} must not be empty")
        available = set(self.available_tasks())
        if set(self.task_filter) - available:
            raise ValueError(
                "task_filter contains an unknown, self-transfer or unselected dataset task"
            )
        _unique(self.ablations.tasks, "ablations.tasks")
        if self.mode == "primary":
            if self.seeds != PRIMARY_SEEDS:
                raise ValueError("Primary seeds must be 17, 29, 43, 71 and 101 in declared order")
            if set(self.datasets) != set(DatasetName) or set(self.models) != set(ModelName):
                raise ValueError("Primary study requires all three datasets and classifiers")
            if set(self.methods) != set(MethodName) or not self.include_majority_baseline:
                raise ValueError(
                    "Primary study requires all three methods and the majority baseline"
                )
            if self.task_filter:
                raise ValueError("Primary study cannot filter the 22 declared tasks")
            if self.ablations.tasks != (
                "gas_b01_b10",
                "har_official_group_transfer",
                "amazon_books_electronics",
            ):
                raise ValueError("Preserve the three predeclared diagnostic tasks")
            if self.importance_weighting.weight_clip_upper != 10.0 or self.coral.ridge != 0.001:
                raise ValueError("Primary clipping is 10 and CORAL ridge is 0.001")
        else:
            if not self.task_filter or self.ablations.tasks:
                raise ValueError("Smoke mode requires explicit tasks and no sensitivity study")
            if not set(self.seeds).issubset(PRIMARY_SEEDS):
                raise ValueError("Smoke seeds must be selected from the declared primary seeds")
            if self.importance_weighting.weight_clip_upper not in self.ablations.weight_clip_upper:
                raise ValueError("Smoke clipping must use a predeclared bound")
            if self.coral.ridge not in self.ablations.coral_ridge:
                raise ValueError("Smoke CORAL must use a predeclared ridge")
        return self

    def available_tasks(self) -> tuple[str, ...]:
        tasks: list[str] = []
        if DatasetName.GAS in self.datasets:
            tasks.extend(f"gas_b01_b{batch:02d}" for batch in self.data.gas_target_batches)
        if DatasetName.HAR in self.datasets:
            tasks.append("har_official_group_transfer")
        if DatasetName.AMAZON in self.datasets:
            tasks.extend(
                f"amazon_{source}_{target}"
                for source in self.data.amazon_domains
                for target in self.data.amazon_domains
                if source != target
            )
        return tuple(tasks)

    def selected_tasks(self) -> tuple[str, ...]:
        return self.task_filter or self.available_tasks()

    def planned_counts(self) -> dict[str, int]:
        """Configuration arithmetic only; never a count of completed runs."""
        task_seeds = len(self.selected_tasks()) * len(self.seeds)
        main = task_seeds * len(self.models) * len(self.methods)
        majority = task_seeds if self.include_majority_baseline else 0
        sensitivity = len(self.ablations.tasks) * len(self.models) * len(self.seeds) * 4
        return {
            "tasks": len(self.selected_tasks()),
            "main_fits": main,
            "majority_evaluations": majority,
            "sensitivity_fits": sensitivity,
        }


class ContractLoader(yaml.SafeLoader):
    """Safe YAML with duplicate keys and aliases rejected rather than silently merged."""

    def compose_node(self, parent: Any, index: Any) -> Any:
        if self.check_event(yaml.AliasEvent):
            raise ConfigError("YAML aliases are not permitted in study configurations")
        return super().compose_node(parent, index)

    def construct_mapping(self, node: Any, deep: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key_node, value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if not isinstance(key, str):
                raise ConfigError("YAML mapping keys must be strings")
            if key in result:
                raise ConfigError(f"Duplicate YAML key: {key}")
            result[key] = self.construct_object(value_node, deep=deep)
        return result


def _check_scalars(value: Any) -> None:
    """Literal integer fields must not accept bools or floats that compare equal."""
    if isinstance(value, dict):
        for key, child in value.items():
            if (
                key
                in {
                    "schema_version",
                    "split_seed",
                    "gas_uci_id",
                    "har_uci_id",
                    "gas_source_batch",
                    "har_target_adaptation_participants",
                    "folds",
                    "rf_n_estimators",
                    "source_crossfit_folds",
                    "zero_division",
                    "bootstrap_replicates",
                    "bootstrap_seed",
                    "max_parallel_jobs",
                    "threads_per_job",
                }
                and type(child) is not int
            ):
                raise ConfigError(f"{key} must be an integer, not a boolean, string or float")
            _check_scalars(child)
    elif isinstance(value, list):
        for child in value:
            _check_scalars(child)
    elif isinstance(value, float) and not math.isfinite(value):
        raise ConfigError("Configuration numbers must be finite")


def validate_config(value: Any) -> StudyConfig:
    if not isinstance(value, dict):
        raise ConfigError("Configuration root must be a mapping")
    _check_scalars(value)
    try:
        return StudyConfig.model_validate(value)
    except (ValidationError, ValueError) as error:
        raise ConfigError(str(error)) from error


def load_config(path: Path | str) -> StudyConfig:
    """Load UTF-8 YAML without executing objects, accepting merges or guessing fields."""
    path = Path(path)
    try:
        if path.stat().st_size > 100_000:
            raise ConfigError("Configuration exceeds the 100 KB contract limit")
        with path.open(encoding="utf-8") as stream:
            value = yaml.load(stream, Loader=ContractLoader)
    except (OSError, UnicodeError, yaml.YAMLError, RecursionError) as error:
        raise ConfigError(f"Cannot read configuration {path.name}: {error}") from error
    return validate_config(value)
