"""SocialVAE-compatible state adapter over the shared representation pipeline."""
from __future__ import annotations

import numpy as np

from forecasting_config import ForecastingConfig, resolve_forecasting_config
from forecasting_representation import prepare_forecast_representation


def social_vae_state(
    xy: np.ndarray,
    valid: np.ndarray,
    *,
    config: ForecastingConfig | None = None,
    training: bool = False,
    rng: np.random.Generator | None = None,
    source_scene_center: np.ndarray | None = None,
) -> dict[str, object]:
    config = config or resolve_forecasting_config(preset="social_vae")
    representation = prepare_forecast_representation(
        xy, valid, config, training=training, rng=rng or np.random.default_rng(0),
        source_scene_center=source_scene_center,
    )
    if representation.feature_names != ("x", "y", "vx", "vy", "ax", "ay"):
        raise ValueError("SocialVAE adapter requires position_velocity_acceleration features")
    obs = config.window.observation_steps
    history = np.array(representation.features[:obs, 0], copy=True)
    future = np.array(representation.xy[obs:, 0], copy=True)
    neighbors = np.array(representation.features[:, 1:], copy=True)
    neighbor_valid = representation.valid[:, 1:]
    # Reference parity: the pinned SocialVAE loader drops agents seen at a single frame.
    if neighbors.shape[1]:
        keep = np.any(neighbor_valid[1:] & neighbor_valid[:-1], axis=0)
        neighbors = neighbors[:, keep]
        neighbor_valid = neighbor_valid[:, keep]
    if neighbors.shape[1] == 0:
        # Reference parity: one distant dummy neighbour when the target is alone.
        neighbors = np.full((len(representation.xy), 1, 6), 1e9, dtype=np.float32)
    else:
        neighbors[~neighbor_valid] = 1e9
    return {
        "history": history,
        "future": future,
        "neighbors": neighbors,
        "valid": representation.valid,
        "transform": representation.transform,
        "target_world_xy": np.array(representation.target_world_xy[:, 0], copy=True),
        "model_xy": np.array(representation.xy, copy=True),
        "augmentation": dict(representation.augmentation),
    }
