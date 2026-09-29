from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from forecasting_config import (
    COORDINATE_FRAMES,
    DERIVATIVES,
    FEATURES,
    FORECASTING_PRESETS,
    INPUT_MODES,
    SCENE_MEMBERSHIPS,
    STANDARDIZATION,
    TARGET_MODES,
    config_as_dict,
    format_forecasting_summary,
    list_forecast_presets,
    resolve_forecasting_config,
)


def _boolean_pair(parser: argparse.ArgumentParser, name: str, destination: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=destination, action="store_true")
    group.add_argument(f"--no-{name}", dest=destination, action="store_false")
    parser.set_defaults(**{destination: None})


def add_preprocessing_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional overrides; None defaults preserve config/preset precedence."""
    parser.add_argument("--forecast-config", type=Path, help="JSON/YAML file containing forecasting: ...")
    parser.add_argument("--forecast-preset", help=f"Named recipe: {', '.join(FORECASTING_PRESETS)}")
    parser.add_argument("--list-forecast-presets", action="store_true")
    parser.add_argument("--show-resolved-forecast-config", action="store_true")
    parser.add_argument("--obs-steps", type=int)
    parser.add_argument("--pred-steps", type=int)
    parser.add_argument("--forecast-stride", type=int)
    parser.add_argument("--forecast-input-preprocessing", choices=INPUT_MODES)
    parser.add_argument("--forecast-target-preprocessing", choices=TARGET_MODES)
    parser.add_argument("--coordinate-frame", choices=COORDINATE_FRAMES)
    parser.add_argument("--feature-representation", choices=FEATURES)
    parser.add_argument("--derivative-mode", choices=DERIVATIVES)
    parser.add_argument("--standardization", choices=STANDARDIZATION)
    parser.add_argument("--scene-membership", choices=SCENE_MEMBERSHIPS)
    parser.add_argument("--min-complete-agents", type=int)
    _boolean_pair(parser, "random-rotation", "random_rotation")
    _boolean_pair(parser, "random-flip", "random_flip")
    _boolean_pair(parser, "random-scale", "random_scale")
    parser.add_argument("--rotation-mode", choices=("continuous", "discrete"))
    parser.add_argument("--rotation-step-degrees", type=float)
    parser.add_argument("--flip-axes", choices=("independent_xy",))
    parser.add_argument("--scale-distribution", choices=("normal",))
    parser.add_argument("--scale-mean", type=float)
    parser.add_argument("--scale-std", type=float)


def handle_preprocessing_introspection(
    *, argv: list[str] | None = None, default_preset: str = "steps_default"
) -> None:
    """Handle recipe-only CLI requests before a trainer enforces required arguments,
    so that ``--list-forecast-presets`` works without a run directory or model."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not ({"--list-forecast-presets", "--show-resolved-forecast-config"} & set(arguments)):
        return
    parser = argparse.ArgumentParser(add_help=False)
    add_preprocessing_arguments(parser)
    args, _unknown = parser.parse_known_args(arguments)
    resolved_forecasting_from_args(args, default_preset=default_preset)
    raise AssertionError("forecasting introspection must exit")


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("PyYAML is required for YAML --forecast-config files") from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Forecast config must be an object")
    return dict(payload.get("forecasting", payload))


def resolved_forecasting_from_args(args, *, default_preset: str = "steps_default"):
    if getattr(args, "list_forecast_presets", False):
        print(list_forecast_presets())
        raise SystemExit(0)
    file_config = _load_config(getattr(args, "forecast_config", None))
    preset = getattr(args, "forecast_preset", None) or file_config.pop("preset", default_preset)
    cli = {
        "window.observation_steps": getattr(args, "obs_steps", None),
        "window.prediction_steps": getattr(args, "pred_steps", None),
        "window.stride": getattr(args, "forecast_stride", None),
        "signal.input_mode": getattr(args, "forecast_input_preprocessing", None),
        "signal.target_mode": getattr(args, "forecast_target_preprocessing", None),
        "coordinates.frame": getattr(args, "coordinate_frame", None),
        "features.representation": getattr(args, "feature_representation", None),
        "features.derivative_mode": getattr(args, "derivative_mode", None),
        "standardization.mode": getattr(args, "standardization", None),
        "scene_membership": getattr(args, "scene_membership", None),
        "min_complete_agents": getattr(args, "min_complete_agents", None),
        "augmentation.random_rotation.enabled": getattr(args, "random_rotation", None),
        "augmentation.random_rotation.mode": getattr(args, "rotation_mode", None),
        "augmentation.random_rotation.step_degrees": getattr(args, "rotation_step_degrees", None),
        "augmentation.random_flip.enabled": getattr(args, "random_flip", None),
        "augmentation.random_flip.axes": getattr(args, "flip_axes", None),
        "augmentation.random_scale.enabled": getattr(args, "random_scale", None),
        "augmentation.random_scale.distribution": getattr(args, "scale_distribution", None),
        "augmentation.random_scale.mean": getattr(args, "scale_mean", None),
        "augmentation.random_scale.std": getattr(args, "scale_std", None),
    }
    resolved = resolve_forecasting_config(
        preset=preset, config_overrides=file_config, cli_overrides=cli,
        default_preset=default_preset,
    )
    print(format_forecasting_summary(resolved))
    if getattr(args, "show_resolved_forecast_config", False):
        print(json.dumps(config_as_dict(resolved), indent=2))
        raise SystemExit(0)
    return resolved


def config_preprocessing(args, default_preset: str = "steps_default") -> dict:
    return {"forecasting": resolved_forecasting_from_args(args, default_preset=default_preset)}
