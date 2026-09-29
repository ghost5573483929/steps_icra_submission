"""Single source of truth for forecasting representations and literature presets."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import warnings
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

# `preprocessing_views.py` redeclares the input/target modes as `Literal` types and cannot
# derive them from these tuples, so the two must be kept in sync by hand.
INPUT_MODES: tuple[str, ...] = ("raw", "causal_kalman", "observation_only_smooth", "opentraj_offline_smooth_compat")
TARGET_MODES: tuple[str, ...] = ("raw", "opentraj_offline_smooth_compat")
COORDINATE_FRAMES: tuple[str, ...] = (
    "world", "target_translation_last_observation", "target_centric_se2_observations_only", "scene_mean_center",
)
FEATURES: tuple[str, ...] = ("absolute_xy", "absolute_xy_and_relative_displacement", "position_velocity_acceleration")
# Channel count each representation resolves to in `forecasting_representation.build_features`.
FEATURE_CHANNEL_COUNTS: dict[str, int] = {
    "absolute_xy": 2,
    "absolute_xy_and_relative_displacement": 4,
    "position_velocity_acceleration": 6,
}
DERIVATIVES: tuple[str, ...] = ("none", "per_step_difference", "physical_derivative_dt")
STANDARDIZATION: tuple[str, ...] = ("none", "trajectron_pp")
SCENE_MEMBERSHIPS: tuple[str, ...] = (
    "target_complete", "individual_track", "social_gan_complete_scene", "scholler_masked_track",
)
# Frames that centre/rotate on a single designated target agent, which
# `forecasting_representation.apply_coordinate_transform` takes to be agent index 0.
TARGET_RELATIVE_FRAMES: frozenset[str] = frozenset({
    "target_translation_last_observation", "target_centric_se2_observations_only",
})
# Memberships for which `enumerate_forecast_windows` sets `target_agent_id=None`.
MEMBERSHIPS_WITH_NO_TARGET: frozenset[str] = frozenset({"social_gan_complete_scene"})
# Memberships whose window enumeration actually reads `min_complete_agents`; elsewhere the
# field would be hashed and published without changing which windows exist.
MEMBERSHIPS_APPLYING_MIN_COMPLETE_AGENTS: frozenset[str] = frozenset({"social_gan_complete_scene"})


def _check_min_complete_agents_applies(scene_membership: str, min_complete_agents: int) -> None:
    """Reject `min_complete_agents > 1` for memberships whose window enumeration never reads it."""
    if min_complete_agents > 1 and scene_membership not in MEMBERSHIPS_APPLYING_MIN_COMPLETE_AGENTS:
        raise ValueError(
            f"min_complete_agents={min_complete_agents!r} was requested, but "
            f"scene_membership={scene_membership!r} does not apply it: window enumeration for "
            f"this membership never reads the field (forecasting_windows.py only checks it "
            f"inside the social_gan_complete_scene branch), so it would be recorded in "
            f"provenance and hashed into the run identity while changing nothing about which "
            f"windows exist. Only "
            f"{sorted(MEMBERSHIPS_APPLYING_MIN_COMPLETE_AGENTS)!r} honour this field today."
        )


@dataclass(frozen=True)
class ForecastWindowConfig:
    observation_steps: int = 8
    prediction_steps: int = 12
    stride: int = 1


@dataclass(frozen=True)
class SignalPreprocessingConfig:
    input_mode: str = "raw"
    target_mode: str = "raw"


@dataclass(frozen=True)
class CoordinateConfig:
    frame: str = "world"


@dataclass(frozen=True)
class FeatureConfig:
    representation: str = "absolute_xy"
    derivative_mode: str = "none"


@dataclass(frozen=True)
class StandardizationConfig:
    mode: str = "none"


@dataclass(frozen=True)
class RandomRotationConfig:
    enabled: bool = False
    mode: str = "continuous"
    step_degrees: float = 15.0


@dataclass(frozen=True)
class RandomFlipConfig:
    enabled: bool = False
    axes: str = "independent_xy"


@dataclass(frozen=True)
class RandomScaleConfig:
    enabled: bool = False
    distribution: str = "normal"
    mean: float = 1.0
    std: float = 0.05


@dataclass(frozen=True)
class AugmentationConfig:
    random_rotation: RandomRotationConfig = field(default_factory=RandomRotationConfig)
    random_flip: RandomFlipConfig = field(default_factory=RandomFlipConfig)
    random_scale: RandomScaleConfig = field(default_factory=RandomScaleConfig)


@dataclass(frozen=True)
class PresetMetadata:
    name: str
    paper: str
    venue: str
    upstream_repo: str
    upstream_commit: str


@dataclass(frozen=True)
class ForecastingConfig:
    requested_preset: str
    preset_modified: bool
    literature_exact: bool
    window: ForecastWindowConfig
    signal: SignalPreprocessingConfig
    coordinates: CoordinateConfig
    features: FeatureConfig
    standardization: StandardizationConfig
    augmentation: AugmentationConfig
    scene_membership: str
    min_complete_agents: int
    forecast_preprocessing_hash: str
    upstream_repo: str
    upstream_commit: str
    explicit_overrides: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # `_validate` runs only inside `resolve_forecasting_config`, so a directly constructed
        # instance would otherwise bypass this guard.
        _check_min_complete_agents_applies(self.scene_membership, self.min_complete_agents)

    def flat_provenance(self) -> dict[str, Any]:
        return {
            "forecast_preset": self.requested_preset,
            "forecast_preset_modified": self.preset_modified,
            "forecast_literature_exact": self.literature_exact,
            "observation_steps": self.window.observation_steps,
            "prediction_steps": self.window.prediction_steps,
            "forecast_stride": self.window.stride,
            "forecast_input_preprocessing": self.signal.input_mode,
            "forecast_target_preprocessing": self.signal.target_mode,
            "coordinate_frame": self.coordinates.frame,
            "feature_representation": self.features.representation,
            "derivative_mode": self.features.derivative_mode,
            "standardization_mode": self.standardization.mode,
            "augmentation_rotation": self.augmentation.random_rotation.enabled,
            "augmentation_flip": self.augmentation.random_flip.enabled,
            "augmentation_scale": self.augmentation.random_scale.enabled,
            "scene_membership": self.scene_membership,
            "min_complete_agents": self.min_complete_agents,
            "forecast_preprocessing_hash": self.forecast_preprocessing_hash,
            "upstream_repository": self.upstream_repo,
            "upstream_pinned_commit": self.upstream_commit,
        }


PRESET_METADATA = {
    "social_gan": PresetMetadata(
        "social_gan", "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks",
        "CVPR 2018", "https://github.com/agrimgupta92/sgan",
        "691231e1adb6a344c7bcea9ebf2534518b226ead",
    ),
    "scholler": PresetMetadata(
        "scholler", "What the Constant Velocity Model Can Teach Us About Pedestrian Motion Prediction",
        "IEEE RA-L 2020", "https://github.com/cschoeller/constant_velocity_pedestrian_motion",
        "7fe07167659bb93fb400cb752c135ebee59221f5",
    ),
    "trajectron_pp": PresetMetadata(
        "trajectron_pp", "Trajectron++: Dynamically-Feasible Trajectory Forecasting With Heterogeneous Data",
        "ECCV 2020", "https://github.com/StanfordASL/Trajectron-plus-plus",
        "1031c7bd1a444273af378c1ec1dcca907ba59830",
    ),
    "social_vae": PresetMetadata(
        "social_vae", "SocialVAE: Human Trajectory Prediction using Timewise Latents",
        "ECCV 2022", "https://github.com/xupei0610/SocialVAE",
        "eb23cbae562bac2c2dcf0d81529e6ab2ac280962",
    ),
    "steps_default": PresetMetadata(
        "steps_default", "STEPS native forecasting representation", "STEPS",
        "local", "local",
    ),
}


_COMMON = {
    "window": {"observation_steps": 8, "prediction_steps": 12, "stride": 1},
    "signal": {"input_mode": "raw", "target_mode": "raw"},
    "coordinates": {"frame": "world"},
    "features": {"representation": "absolute_xy", "derivative_mode": "none"},
    "standardization": {"mode": "none"},
    "augmentation": {
        "random_rotation": {"enabled": False, "mode": "continuous", "step_degrees": 15.0},
        "random_flip": {"enabled": False, "axes": "independent_xy"},
        "random_scale": {"enabled": False, "distribution": "normal", "mean": 1.0, "std": 0.05},
    },
    "scene_membership": "target_complete",
    "min_complete_agents": 1,
}


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any], prefix: str = "") -> list[str]:
    changed: list[str] = []
    for key, value in source.items():
        if key in {"preset", "requested_preset"}:
            continue
        if key not in target:
            raise ValueError(f"Unknown forecasting config field: {prefix}{key}")
        path = f"{prefix}{key}"
        if isinstance(value, Mapping):
            if not isinstance(target[key], dict):
                raise ValueError(f"{path} does not accept nested fields")
            changed.extend(_deep_merge(target[key], value, path + "."))
        else:
            target[key] = value
            changed.append(path)
    return changed


def _recipe(**changes: Any) -> dict[str, Any]:
    recipe = copy.deepcopy(_COMMON)
    _deep_merge(recipe, changes)
    return recipe


FORECASTING_PRESETS: dict[str, dict[str, Any]] = {
    "social_gan": _recipe(
        features={"representation": "absolute_xy_and_relative_displacement", "derivative_mode": "per_step_difference"},
        scene_membership="social_gan_complete_scene", min_complete_agents=2,
    ),
    "scholler": _recipe(
        features={"representation": "absolute_xy", "derivative_mode": "per_step_difference"},
        scene_membership="individual_track",
    ),
    "trajectron_pp": _recipe(
        coordinates={"frame": "scene_mean_center"},
        features={"representation": "position_velocity_acceleration", "derivative_mode": "physical_derivative_dt"},
        standardization={"mode": "trajectron_pp"},
        augmentation={"random_rotation": {"enabled": True, "mode": "discrete", "step_degrees": 15.0}},
    ),
    "social_vae": _recipe(
        features={"representation": "position_velocity_acceleration", "derivative_mode": "per_step_difference"},
        augmentation={
            "random_rotation": {"enabled": True, "mode": "continuous"},
            "random_flip": {"enabled": True},
            "random_scale": {"enabled": True, "distribution": "normal", "mean": 1.0, "std": 0.05},
        },
    ),
    "steps_default": _recipe(
        coordinates={"frame": "target_centric_se2_observations_only"},
        # Heading alignment already removes global orientation.
        augmentation={"random_rotation": {"enabled": False}},
    ),
}


ALIASES = {
    "socialgan": "social_gan", "social_gan": "social_gan",
    "scholler": "scholler", "schöller": "scholler",
    "trajectronplusplus": "trajectron_pp", "trajectron++": "trajectron_pp",
    "trajectronpp": "trajectron_pp", "trajectron_pp": "trajectron_pp",
    "socialvae": "social_vae", "social_vae": "social_vae",
    "steps": "steps_default", "steps_default": "steps_default",
}


def normalize_preset_name(value: str) -> str:
    normalized = value.strip().lower().replace("-", "_").replace(" ", "")
    normalized = ALIASES.get(normalized, normalized)
    if normalized not in FORECASTING_PRESETS:
        raise ValueError(f"Unknown forecasting preset {value!r}; choices: {sorted(FORECASTING_PRESETS)}")
    return normalized


def _dot_overrides(overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    nested: dict[str, Any] = {}
    for path, value in (overrides or {}).items():
        if value is None:
            continue
        current = nested
        parts = path.split(".")
        for part in parts[:-1]:
            current = current.setdefault(part, {})
        current[parts[-1]] = value
    return nested


def _validate(recipe: dict[str, Any]) -> None:
    window = recipe["window"]
    if window["observation_steps"] < 2 or window["prediction_steps"] < 1 or window["stride"] < 1:
        raise ValueError("Forecast window sizes and stride must be positive (observation >=2)")
    if recipe["signal"]["input_mode"] not in INPUT_MODES:
        raise ValueError("Unsupported forecast input preprocessing")
    if recipe["signal"]["target_mode"] not in TARGET_MODES:
        raise ValueError("Unsupported forecast target preprocessing")
    if recipe["coordinates"]["frame"] not in COORDINATE_FRAMES:
        raise ValueError("Unsupported coordinate frame")
    if recipe["features"]["representation"] not in FEATURES:
        raise ValueError("Unsupported feature representation")
    if recipe["features"]["derivative_mode"] not in DERIVATIVES:
        raise ValueError("Unsupported derivative mode")
    if recipe["standardization"]["mode"] not in STANDARDIZATION:
        raise ValueError("Unsupported standardization mode")
    if recipe["scene_membership"] not in SCENE_MEMBERSHIPS:
        raise ValueError("Unsupported scene membership")
    frame = recipe["coordinates"]["frame"]
    scene_membership = recipe["scene_membership"]
    if frame in TARGET_RELATIVE_FRAMES and scene_membership in MEMBERSHIPS_WITH_NO_TARGET:
        raise ValueError(
            f"coordinates.frame={frame!r} is target-relative, but "
            f"scene_membership={scene_membership!r} designates no target agent "
            f"(target_agent_id is None); combining them would silently centre/rotate "
            f"every window on an arbitrary agent (the lowest track id). Use a "
            f"membership that designates a target (e.g. 'individual_track', "
            f"'scholler_masked_track', 'target_complete') or a non-target-relative "
            f"frame ('world', 'scene_mean_center')."
        )
    if recipe["min_complete_agents"] < 1:
        raise ValueError("min_complete_agents must be >=1")
    _check_min_complete_agents_applies(scene_membership, recipe["min_complete_agents"])
    rotation = recipe["augmentation"]["random_rotation"]
    if rotation["mode"] not in {"continuous", "discrete"}:
        raise ValueError("Random rotation mode must be continuous or discrete")
    if rotation["step_degrees"] <= 0:
        raise ValueError("Random rotation step_degrees must be positive")
    flip = recipe["augmentation"]["random_flip"]
    if flip["axes"] != "independent_xy":
        raise ValueError("Unsupported random flip axes")
    scale = recipe["augmentation"]["random_scale"]
    if scale["distribution"] != "normal" or scale["mean"] <= 0 or scale["std"] < 0:
        raise ValueError("Random scale requires a positive-mean normal distribution and non-negative std")
    representation = recipe["features"]["representation"]
    derivative_mode = recipe["features"]["derivative_mode"]
    if representation == "position_velocity_acceleration" and derivative_mode == "none":
        raise ValueError(
            f"features.representation={representation!r} requires velocity/acceleration derivatives, but "
            f"features.derivative_mode={derivative_mode!r} cannot construct them"
        )
    standardization_mode = recipe["standardization"]["mode"]
    if representation not in FEATURE_CHANNEL_COUNTS:
        raise ValueError(
            f"features.representation={representation!r} is listed in FEATURES but has no entry in "
            f"FEATURE_CHANNEL_COUNTS; add its resolved channel count to both FEATURES and "
            f"FEATURE_CHANNEL_COUNTS in forecasting_config.py"
        )
    channel_count = FEATURE_CHANNEL_COUNTS[representation]
    if standardization_mode == "trajectron_pp" and channel_count != 6:
        raise ValueError(
            f"standardization.mode={standardization_mode!r} requires 6 channels (x,y,vx,vy,ax,ay), but "
            f"features.representation={representation!r} resolves to {channel_count} channel(s)"
        )


def resolve_forecasting_config(
    *,
    preset: str | None = None,
    config_overrides: Mapping[str, Any] | None = None,
    cli_overrides: Mapping[str, Any] | None = None,
    default_preset: str = "steps_default",
) -> ForecastingConfig:
    config_overrides = dict(config_overrides or {})
    requested = normalize_preset_name(str(preset or config_overrides.pop("preset", default_preset)))
    base = copy.deepcopy(FORECASTING_PRESETS[requested])
    config_paths = _deep_merge(base, config_overrides)
    cli_nested = _dot_overrides(cli_overrides)
    cli_paths = _deep_merge(base, cli_nested)
    _validate(base)
    exact = base == FORECASTING_PRESETS[requested]
    metadata = PRESET_METADATA[requested]
    recipe_payload = json.dumps({"preset": requested, **base}, sort_keys=True, separators=(",", ":"))
    recipe_hash = "sha256:" + hashlib.sha256(recipe_payload.encode()).hexdigest()
    return ForecastingConfig(
        requested_preset=requested,
        preset_modified=not exact,
        literature_exact=exact,
        window=ForecastWindowConfig(**base["window"]),
        signal=SignalPreprocessingConfig(**base["signal"]),
        coordinates=CoordinateConfig(**base["coordinates"]),
        features=FeatureConfig(**base["features"]),
        standardization=StandardizationConfig(mode=base["standardization"]["mode"]),
        augmentation=AugmentationConfig(
            RandomRotationConfig(**base["augmentation"]["random_rotation"]),
            RandomFlipConfig(**base["augmentation"]["random_flip"]),
            RandomScaleConfig(**base["augmentation"]["random_scale"]),
        ),
        scene_membership=base["scene_membership"],
        min_complete_agents=int(base["min_complete_agents"]),
        forecast_preprocessing_hash=recipe_hash,
        upstream_repo=metadata.upstream_repo,
        upstream_commit=metadata.upstream_commit,
        explicit_overrides=tuple(config_paths + cli_paths),
    )


def config_as_dict(config: ForecastingConfig) -> dict[str, Any]:
    return asdict(config)


def format_forecasting_summary(config: ForecastingConfig) -> str:
    augment = [name for name, enabled in (
        ("rotation", config.augmentation.random_rotation.enabled),
        ("flip", config.augmentation.random_flip.enabled),
        ("scale", config.augmentation.random_scale.enabled),
    ) if enabled]
    lines = [
        "Forecast preprocessing",
        f"  preset:             {config.requested_preset}",
        f"  literature exact:   {'yes' if config.literature_exact else 'NO'}",
        f"  window:             {config.window.observation_steps} obs + {config.window.prediction_steps} pred, stride {config.window.stride}",
        f"  signal input:       {config.signal.input_mode}",
        f"  signal target:      {config.signal.target_mode}",
        f"  coordinates:        {config.coordinates.frame}",
        f"  features:           {config.features.representation} / {config.features.derivative_mode}",
        f"  standardization:    {config.standardization.mode}",
        f"  augmentation:       {', '.join(augment) if augment else 'none'}",
        f"  membership:         {config.scene_membership}, >={config.min_complete_agents} agent(s)",
    ]
    if config.explicit_overrides:
        lines.append("  overrides:          " + ", ".join(config.explicit_overrides))
    return "\n".join(lines)


def list_forecast_presets() -> str:
    fields =("window", "frame", "features", "standardization", "augmentation", "membership")
    label_width = max(len(f) for f in fields)
    lines = []
    for name, recipe in FORECASTING_PRESETS.items():
        window = recipe["window"]
        features = recipe["features"]
        augmentation = recipe["augmentation"]
        parts = []
        if augmentation["random_rotation"]["enabled"]:
            parts.append(f"rotate({augmentation['random_rotation']['mode']})")
        if augmentation["random_flip"]["enabled"]:
            parts.append("flip")
        if augmentation["random_scale"]["enabled"]:
            parts.append("scale")
        values = {
            "window": f"{window['observation_steps']}+{window['prediction_steps']}",
            "frame": recipe["coordinates"]["frame"],
            "features": f"{features['representation']} / {features['derivative_mode']}",
            "standardization": recipe["standardization"]["mode"],
            "augmentation": "+".join(parts) if parts else "none",
            "membership": f"{recipe['scene_membership']} (>={recipe['min_complete_agents']})",
        }
        lines.append(name)
        lines.extend(f"    {field:<{label_width}}  {values[field]}" for field in fields)
        lines.append("")
    return "\n".join(lines).rstrip()


def main() -> None:
    # Imported here to avoid a module-level cycle: the CLI helper imports this module.
    from baselines.preprocessing_options import (
        add_preprocessing_arguments,
        resolved_forecasting_from_args,
    )
    parser = argparse.ArgumentParser(description="Inspect resolved forecasting preprocessing")
    add_preprocessing_arguments(parser)
    args = parser.parse_args()
    resolved_forecasting_from_args(args)


if __name__ == "__main__":
    main()
