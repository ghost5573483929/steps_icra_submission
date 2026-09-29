from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class NormalizationStats:
    stationary_boundaries: int = 0


@dataclass(frozen=True)
class SceneTransform:
    rotation: float
    center: np.ndarray
    heading_source: str


def theta_rotation(xy: np.ndarray, theta: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    # Integer dtypes would truncate cos/sin to zero; any floating dtype is left
    # alone so float32 scenes are not upcast.
    dtype = xy.dtype if np.issubdtype(xy.dtype, np.floating) else np.float64
    rotation_matrix = np.array([[ct, st], [-st, ct]], dtype=dtype)
    return np.einsum("tnc,ci->tni", xy, rotation_matrix)


def center_scene(
    xy: np.ndarray,
    obs_length: int = 8,
    ped_id: int = 0,
    stats: NormalizationStats | None = None,
) -> tuple[np.ndarray, float, np.ndarray]:
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("xy must have shape (time, agents, 2)")
    center = np.array(xy[obs_length - 1, ped_id], copy=True)
    centered = xy - center[np.newaxis, np.newaxis, :]
    transform = observed_target_transform(xy, obs_length, ped_id, stats=stats)
    return theta_rotation(centered, transform.rotation), transform.rotation, center


def observed_target_transform(
    xy: np.ndarray,
    obs_length: int = 8,
    ped_id: int = 0,
    *,
    stats: NormalizationStats | None = None,
    # Displacement floor, in metres, below which arctan2 carries no stable direction.
    # Deliberately distinct from heading.HEADING_SPEED_EPSILON: that one scopes the
    # canonical `heading` column, this one sets the SE(2) model input frame.
    threshold: float = 1e-6,
) -> SceneTransform:
    """Compute target-centric SE(2) parameters using observations only."""
    xy = np.asarray(xy)
    if xy.ndim != 3 or xy.shape[-1] != 2 or obs_length < 2:
        raise ValueError("xy must have shape (time, agents, 2) and at least two observations")
    center = np.array(xy[obs_length - 1, ped_id], copy=True)
    # Heading tiers, most to least local; an all-stationary target falls through to
    # the identity rotation.
    vector = np.zeros(2, dtype=float)
    source = "identity"
    candidate = xy[obs_length - 1, ped_id] - xy[obs_length - 2, ped_id]
    if np.isfinite(candidate).all() and np.linalg.norm(candidate) >= threshold:
        vector = candidate
        source = "final_observed_displacement"
    if np.linalg.norm(vector) < threshold:
        candidate = xy[obs_length - 1, ped_id] - xy[0, ped_id]
        if np.isfinite(candidate).all() and np.linalg.norm(candidate) >= threshold:
            vector = candidate
            source = "multi_observation_displacement"
    if np.linalg.norm(vector) < threshold:
        rotation = 0.0
        if stats is not None:
            stats.stationary_boundaries += 1
    else:
        rotation = -float(np.arctan2(vector[1], vector[0])) + np.pi / 2
    return SceneTransform(rotation, center, source)


def normalize_scene(
    xy: np.ndarray,
    obs_length: int = 8,
    ped_id: int = 0,
    **kwargs,
) -> tuple[np.ndarray, SceneTransform]:
    transform = observed_target_transform(xy, obs_length, ped_id, **kwargs)
    centered = np.asarray(xy) - transform.center[np.newaxis, np.newaxis, :]
    return theta_rotation(centered, transform.rotation), transform


def inverse_scene(xy: np.ndarray, rotation: float, center: np.ndarray) -> np.ndarray:
    return theta_rotation(xy, -rotation) + center[np.newaxis, np.newaxis, :]


def random_rotation(xy: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    theta = float(rng.random() * 2.0 * np.pi)
    return theta_rotation(xy, theta), theta


def rotate_heading(heading: np.ndarray, rotation: float) -> np.ndarray:
    return (heading + rotation + np.pi) % (2 * np.pi) - np.pi
