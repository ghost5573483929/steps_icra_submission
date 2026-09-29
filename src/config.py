from __future__ import annotations

import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from dataset_paths import TASK_CONFIG_PATH

# Every strategy the forecast index supports. The analysis index supports only a
# subset, spelled separately next to the guard that enforces it.
SPLIT_STRATEGIES = ("grouped_scene_sequence", "leave_one_scene_out", "temporal", "held_out_recording")


@dataclass(frozen=True)
class SeedConfig:
    data_seed: int = 42
    model_seed: int = 42


@dataclass(frozen=True)
class CategoryConfig:
    static_threshold: float = 1.0
    linear_threshold: float = 0.5
    inter_dist_thresh: float = 5.0
    inter_pos_range: float = 15.0
    grp_dist_thresh: float = 0.8
    grp_std_thresh: float = 0.2
    acceptance: tuple[float, float, float, float] = (0.1, 1.0, 1.0, 1.0)


@dataclass(frozen=True)
class SizingConfig:
    w_target: float = 2.0
    n_target: int | None = 100_000


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    path: Path
    fraction: float = 1.0
    weight: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", Path(self.path))
        if not 0.0 < self.fraction <= 1.0:
            raise ValueError("DatasetSpec.fraction must be in (0, 1]")
        if self.weight <= 0:
            raise ValueError("DatasetSpec.weight must be positive")


@dataclass(frozen=True)
class TrainConfig:
    iterations: int = 0
    patience: int = 0


@dataclass(frozen=True)
class Config:
    seeds: SeedConfig = field(default_factory=SeedConfig)
    category: CategoryConfig = field(default_factory=CategoryConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    batch_size: int = 64
    num_workers: int = 0
    # Legacy flat fields, translated into ``forecasting``; new code should set
    # ForecastingConfig directly.
    train_rotation: bool | None = None
    eval_stride: int | None = 20
    analysis_preprocessing: str = "opentraj_kalman_smooth"
    forecast_input_preprocessing: str | None = None
    forecast_target_preprocessing: str | None = None
    model_normalization: str | None = None
    forecasting: Any = None
    canonical_release: str = "v3_2p5hz"
    split_strategy: str = "grouped_scene_sequence"
    held_out_scene: str | None = None
    task_path: Path = TASK_CONFIG_PATH

    def __post_init__(self) -> None:
        object.__setattr__(self, "task_path", Path(self.task_path))
        if self.split_strategy not in SPLIT_STRATEGIES:
            names = ", ".join(SPLIT_STRATEGIES[:-1]) + f", or {SPLIT_STRATEGIES[-1]}"
            raise ValueError(f"split_strategy must be {names}")
        if self.split_strategy == "leave_one_scene_out" and not self.held_out_scene:
            raise ValueError("held_out_scene is required for leave_one_scene_out")
        from forecasting_config import ForecastingConfig, resolve_forecasting_config

        if self.forecasting is not None and not isinstance(self.forecasting, ForecastingConfig):
            raise TypeError("forecasting must be a resolved ForecastingConfig")
        if self.forecasting is None:
            task = json.loads(self.task_path.read_text(encoding="utf-8"))
            task_forecasting = dict(task.get("forecasting", {}))
            preset = task_forecasting.pop("preset", "steps_default")
            translated: dict[str, Any] = {}
            if "input_preprocessing" in task_forecasting:
                translated.setdefault("signal", {})["input_mode"] = task_forecasting.pop("input_preprocessing")
            if "target_preprocessing" in task_forecasting:
                translated.setdefault("signal", {})["target_mode"] = task_forecasting.pop("target_preprocessing")
            if "normalization" in task_forecasting:
                translated.setdefault("coordinates", {})["frame"] = task_forecasting.pop("normalization")
            if "dense_stride" in task_forecasting:
                raise ValueError(
                    "forecasting.dense_stride is retired and is no longer read; "
                    "remove it from configs/task.json's forecasting block."
                )
            for key, value in task_forecasting.items():
                translated[key] = value
            legacy: dict[str, Any] = {}
            if self.forecast_input_preprocessing is not None:
                legacy.setdefault("signal", {})["input_mode"] = self.forecast_input_preprocessing
            if self.forecast_target_preprocessing is not None:
                legacy.setdefault("signal", {})["target_mode"] = self.forecast_target_preprocessing
            if self.model_normalization is not None:
                legacy.setdefault("coordinates", {})["frame"] = self.model_normalization
            if self.train_rotation is not None:
                legacy.setdefault("augmentation", {}).setdefault("random_rotation", {})["enabled"] = self.train_rotation
            if legacy:
                warnings.warn(
                    "Flat forecasting settings are deprecated; use ForecastingConfig/presets",
                    DeprecationWarning,
                    stacklevel=2,
                )
                # Legacy constructor arguments have explicit-config precedence.
                def merge(target: dict, source: dict) -> None:
                    for key, value in source.items():
                        if isinstance(value, dict):
                            merge(target.setdefault(key, {}), value)
                        else:
                            target[key] = value
                merge(translated, legacy)
            resolved = resolve_forecasting_config(preset=preset, config_overrides=translated)
        else:
            resolved = self.forecasting
            # A flat value passed alongside an explicit forecasting config would be
            # silently discarded, so refuse any that disagrees with the resolved config.
            conflicts = [
                f"{name}={supplied!r} conflicts with forecasting {source}={effective!r}"
                for name, supplied, source, effective in (
                    ("train_rotation", self.train_rotation, "augmentation.random_rotation.enabled",
                     resolved.augmentation.random_rotation.enabled),
                    ("forecast_input_preprocessing", self.forecast_input_preprocessing,
                     "signal.input_mode", resolved.signal.input_mode),
                    ("forecast_target_preprocessing", self.forecast_target_preprocessing,
                     "signal.target_mode", resolved.signal.target_mode),
                    ("model_normalization", self.model_normalization,
                     "coordinates.frame", resolved.coordinates.frame),
                )
                if supplied is not None and supplied != effective
            ]
            if conflicts:
                raise ValueError(
                    "Legacy flat forecasting fields are ignored when a resolved forecasting "
                    "config is supplied: " + "; ".join(conflicts)
                    + ". Express the setting in the ForecastingConfig/preset instead."
                )
        object.__setattr__(self, "forecasting", resolved)
        # Compatibility projections for older run schemas; runtime code reads only
        # ``forecasting``.
        object.__setattr__(self, "forecast_input_preprocessing", resolved.signal.input_mode)
        object.__setattr__(self, "forecast_target_preprocessing", resolved.signal.target_mode)
        object.__setattr__(self, "model_normalization", resolved.coordinates.frame)
        object.__setattr__(self, "train_rotation", resolved.augmentation.random_rotation.enabled)


def stable_config_hash(value: Any) -> str:
    if hasattr(value, "__dataclass_fields__"):
        value = asdict(value)

    def normalize(item: Any) -> Any:
        if hasattr(item, "__dataclass_fields__"):
            return normalize(asdict(item))
        if isinstance(item, Path):
            return str(item)
        if isinstance(item, dict):
            return {key: normalize(val) for key, val in sorted(item.items())}
        if isinstance(item, (list, tuple)):
            return [normalize(val) for val in item]
        return item

    payload = json.dumps(normalize(value), sort_keys=True, separators=(",", ":"))
    return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


def scoped_task_config(task: dict[str, Any], dataset: str) -> dict[str, Any]:
    """Return shared task fields plus only this dataset's override."""
    scoped = {key: value for key, value in task.items() if key != "datasets"}
    scoped["dataset"] = {dataset: task.get("datasets", {}).get(dataset, {})}
    return scoped


def seed_everything(seeds: SeedConfig) -> tuple[np.random.Generator, np.random.Generator]:
    """Seed training libraries and return independent data/model generators."""
    import random
    import torch

    random.seed(seeds.model_seed)
    np.random.seed(seeds.model_seed)
    torch.manual_seed(seeds.model_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seeds.model_seed)
    return np.random.default_rng(seeds.data_seed), np.random.default_rng(seeds.model_seed)
