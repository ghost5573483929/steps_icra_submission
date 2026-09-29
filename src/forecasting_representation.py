"""Composable forecasting representation pipeline over canonical world trajectories."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from forecasting_config import ForecastingConfig, resolve_forecasting_config
from normalization import inverse_scene, observed_target_transform, theta_rotation
from preprocessing_views import preprocess_forecast_scene


@dataclass(frozen=True)
class CoordinateTransform:
    frame: str
    origin: np.ndarray
    rotation: float
    heading_source: str

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        """Undo `apply_coordinate_transform` only.

        This does NOT undo `apply_training_augmentation`, which is applied after the
        coordinate transform, so on a `training=True` representation the result is
        not world coordinates.
        """
        values = np.asarray(xy)
        matrix_shape = values.shape
        flat = values.reshape(values.shape[0], -1, 2)
        restored = inverse_scene(flat, self.rotation, self.origin)
        return restored.reshape(matrix_shape)


@dataclass(frozen=True)
class ForecastRepresentation:
    xy: np.ndarray
    input_xy: np.ndarray
    target_xy: np.ndarray
    target_world_xy: np.ndarray
    features: np.ndarray
    relative_xy: np.ndarray | None
    valid: np.ndarray
    transform: CoordinateTransform
    feature_names: tuple[str, ...]
    input_uses_future_information: bool
    augmentation: dict[str, float | bool]


def apply_signal_preprocessing(
    xy: np.ndarray, valid: np.ndarray, config: ForecastingConfig, dt: float = 0.4
):
    return preprocess_forecast_scene(
        xy, valid, config.window.observation_steps, dt,
        config.signal.input_mode, config.signal.target_mode,
    )


def apply_coordinate_transform(
    xy: np.ndarray,
    config: ForecastingConfig,
    *,
    source_scene_center: np.ndarray | None = None,
) -> tuple[np.ndarray, CoordinateTransform]:
    values = np.asarray(xy, dtype=np.float32)
    obs = config.window.observation_steps
    frame = config.coordinates.frame
    if frame == "world":
        transform = CoordinateTransform(frame, np.zeros(2, dtype=np.float32), 0.0, "identity")
    elif frame == "target_translation_last_observation":
        transform = CoordinateTransform(frame, np.array(values[obs - 1, 0], copy=True), 0.0, "translation_only")
    elif frame == "target_centric_se2_observations_only":
        target = observed_target_transform(values, obs)
        transform = CoordinateTransform(frame, target.center, target.rotation, target.heading_source)
    elif frame == "scene_mean_center":
        if source_scene_center is None:
            raise ValueError("scene_mean_center requires a source/file center computed outside the sample window")
        center = np.asarray(source_scene_center, dtype=np.float32)
        if center.shape != (2,) or not np.isfinite(center).all():
            raise ValueError("source_scene_center must be a finite (2,) vector")
        transform = CoordinateTransform(frame, center, 0.0, "source_scene_mean")
    else:
        raise ValueError(f"Unknown coordinate frame: {frame}")
    centered = values - transform.origin
    return theta_rotation(centered, transform.rotation), transform


def relative_displacement(xy: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    values = np.asarray(xy)
    relative = np.zeros_like(values)
    relative[1:] = values[1:] - values[:-1]
    if valid is not None:
        mask = np.asarray(valid, dtype=bool)
        relative[0, ~mask[0]] = np.nan
        relative[1:][~(mask[1:] & mask[:-1])] = np.nan
    return relative


def _derivatives(xy: np.ndarray, valid: np.ndarray, mode: str, dt: float) -> tuple[np.ndarray, np.ndarray]:
    if mode not in {"per_step_difference", "physical_derivative_dt"}:
        raise ValueError(f"Derivative mode {mode!r} cannot construct velocity/acceleration")
    # Both pinned SocialVAE and Trajectron++ initialize the first derivative
    # in each contiguous segment with the first forward difference, then use
    # backward differences. SocialVAE omits dt; Trajectron++ divides by dt.
    divisor = 1.0 if mode == "per_step_difference" else dt
    def derivative(values: np.ndarray) -> np.ndarray:
        output = np.full_like(values, np.nan)
        for agent in range(values.shape[1]):
            indices = np.flatnonzero(valid[:, agent])
            for segment in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1):
                if len(segment) < 2:
                    if len(segment): output[segment, agent] = 0.0
                    continue
                output[segment[0], agent] = (values[segment[1], agent] - values[segment[0], agent]) / divisor
                output[segment[1:], agent] = (values[segment[1:], agent] - values[segment[:-1], agent]) / divisor
        return output
    velocity = derivative(np.asarray(xy))
    acceleration = derivative(velocity)
    return velocity, acceleration


def build_features(
    xy: np.ndarray, valid: np.ndarray, config: ForecastingConfig, dt: float = 0.4
) -> tuple[np.ndarray, np.ndarray | None, tuple[str, ...]]:
    representation = config.features.representation
    if representation == "absolute_xy":
        return np.asarray(xy).copy(), None, ("x", "y")
    if representation == "absolute_xy_and_relative_displacement":
        relative = relative_displacement(xy, valid)
        return np.concatenate([xy, relative], axis=-1), relative, ("x", "y", "dx", "dy")
    if representation == "position_velocity_acceleration":
        velocity, acceleration = _derivatives(xy, valid, config.features.derivative_mode, dt)
        return np.concatenate([xy, velocity, acceleration], axis=-1), None, ("x", "y", "vx", "vy", "ax", "ay")
    raise ValueError(f"Unknown feature representation: {representation}")


def apply_standardization(features: np.ndarray, config: ForecastingConfig) -> np.ndarray:
    values = np.asarray(features, dtype=np.float32)
    mode = config.standardization.mode
    if mode == "none":
        return values.copy()
    if mode == "trajectron_pp":
        if values.shape[-1] != 6:
            raise ValueError("Trajectron++ standardization requires x,y,vx,vy,ax,ay features")
        # Trajectron++ std-only standardization (all means = 0), channel order
        # x, y, vx, vy, ax, ay. baselines/trajectron_data.py::STANDARDIZATION holds a
        # structured copy of the same table and must stay numerically identical.
        return values / np.asarray([1.0, 1.0, 2.0, 2.0, 1.0, 1.0], dtype=np.float32)
    raise ValueError(f"Unknown standardization mode: {mode}")


def _transform_pairs(values: np.ndarray, theta: float, flip_x: bool, flip_y: bool, scale: float) -> np.ndarray:
    result = np.array(values, copy=True)
    if result.shape[-1] % 2:
        raise ValueError("Geometric augmentation requires paired x/y-like feature channels")
    for start in range(0, result.shape[-1], 2):
        pair = result[..., start : start + 2]
        shape = pair.shape
        pair = theta_rotation(pair.reshape(pair.shape[0], -1, 2), theta).reshape(shape)
        if flip_x:
            pair[..., 0] *= -1
        if flip_y:
            pair[..., 1] *= -1
        result[..., start : start + 2] = pair * scale
    return result


def apply_training_augmentation(
    features: np.ndarray,
    xy: np.ndarray,
    config: ForecastingConfig,
    rng: np.random.Generator,
    *,
    training: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, float | bool]]:
    if not training:
        return np.array(features, copy=True), np.array(xy, copy=True), {
            "applied": False, "rotation": 0.0, "flip_x": False, "flip_y": False, "scale": 1.0,
        }
    augmentation = config.augmentation
    theta = 0.0
    if augmentation.random_rotation.enabled:
        if augmentation.random_rotation.mode == "continuous":
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
        elif augmentation.random_rotation.mode == "discrete":
            step = augmentation.random_rotation.step_degrees
            if step <= 0 or not np.isclose(360.0 / step, round(360.0 / step)):
                raise ValueError("Discrete rotation step must divide 360 degrees")
            theta = math.radians(step * int(rng.integers(round(360.0 / step))))
        else:
            raise ValueError("Rotation mode must be continuous or discrete")
    flip_x = flip_y = False
    if augmentation.random_flip.enabled:
        flip_x, flip_y = (bool(value) for value in (rng.random(2) < 0.5))
    scale = 1.0
    if augmentation.random_scale.enabled:
        if augmentation.random_scale.distribution != "normal":
            raise ValueError("Only normal random scale is supported")
        scale = float(rng.normal(augmentation.random_scale.mean, augmentation.random_scale.std))
        if scale <= 0:
            # A non-positive draw collapses the whole window to ~0; clamping would
            # make that corruption silent.
            raise ValueError(
                f"random_scale draw was non-positive ({scale}); check random_scale mean/std"
            )
    transformed_features = _transform_pairs(features, theta, flip_x, flip_y, scale)
    transformed_xy = _transform_pairs(xy, theta, flip_x, flip_y, scale)
    return transformed_features, transformed_xy, {
        "applied": bool(theta or flip_x or flip_y or not np.isclose(scale, 1.0)),
        "rotation": theta, "flip_x": flip_x, "flip_y": flip_y, "scale": scale,
    }


def prepare_forecast_representation(
    xy: np.ndarray,
    valid: np.ndarray,
    config: ForecastingConfig | None = None,
    *,
    source_scene_center: np.ndarray | None = None,
    training: bool = False,
    rng: np.random.Generator | None = None,
    dt: float = 0.4,
) -> ForecastRepresentation:
    config = config or resolve_forecasting_config()
    signal = apply_signal_preprocessing(xy, valid, config, dt)
    world = np.array(signal.input_xy, copy=True)
    world[config.window.observation_steps :] = signal.target_xy
    coordinate_xy, transform = apply_coordinate_transform(world, config, source_scene_center=source_scene_center)
    features, relative, names = build_features(coordinate_xy, valid, config, dt)
    standardized = apply_standardization(features, config)
    augmented_features, augmented_xy, augmentation = apply_training_augmentation(
        standardized, coordinate_xy, config, rng or np.random.default_rng(0), training=training,
    )
    obs = config.window.observation_steps
    if relative is not None:
        relative = augmented_features[..., 2:4]
    return ForecastRepresentation(
        xy=augmented_xy,
        input_xy=augmented_xy[:obs],
        target_xy=augmented_xy[obs:],
        target_world_xy=np.array(signal.target_xy, copy=True),
        features=augmented_features,
        relative_xy=relative,
        valid=np.array(valid, copy=True),
        transform=transform,
        feature_names=names,
        input_uses_future_information=signal.uses_future_information_for_input,
        augmentation=augmentation,
    )
