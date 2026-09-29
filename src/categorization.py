"""Port of upstream trajnetplusplustools interaction/characterization logic, pinned at
vita-epfl/trajnetplusplustools@39d56e0ee64ad85b71584e48819042aecd3ca434 (src/baselines/upstream.json):
  _interaction_geometry, _angle_mask,
  check_interaction, check_group,
  get_interaction_type  <- trajnetplusplustools/interactions.py
  kalman_fde             <- trajnetplusplustools/kalman.py

Known intentional deviations from upstream:
  - `obs_len` defaults to 8 here (:77) vs upstream's 9.
  - Explicit empty-mask guards for a single-agent scene (`:79-80`, `:99-100`); upstream has
    none and instead indexes the empty neighbour axis.
  - `get_interaction_type` (:134) forwards `group_dist_thresh`/`group_std_thresh` into
    `check_group`, whereas upstream's `group(rows, obs_len=obs_len)` always hardcodes 0.8/0.2.
    We match upstream only while `CategoryConfig.grp_dist_thresh`/`grp_std_thresh`
    (src/config.py) stay at their 0.8/0.2 defaults; changing either diverges from TrajNet++.
  - `_angle_mask` does not port upstream's `theta_rel < 500` sentinel and has no finite-value
    filter. The mask is unchanged in practice: a NaN `theta` already fails the strict angle
    comparisons, and `distance` is an `np.linalg.norm` output so it is never `-inf`.
  - Kalman seeding derives a `RandomState` from a caller-supplied `Generator` (see comment
    at kalman_fde) instead of upstream's global legacy `np.random.seed`.
  - `check_interaction`'s "bothpos" and "bothvel" choices are bit-identical here: upstream's
    two branches differ only via its unimplemented `output='all'` mode.

Parity against the vendored checkout is exercised in
tests/test_trajnet_categorization_parity.py.
"""

from __future__ import annotations

import hashlib

import numpy as np

from config import CategoryConfig


def _interaction_geometry(xy: np.ndarray, obs_len: int, stride: int = 3):
    primary = xy[:, 0]
    neighbors = xy[:, 1:]
    primary_velocity = primary[obs_len:] - primary[obs_len - stride : -stride]
    primary_theta = np.arctan2(primary_velocity[:, 1], primary_velocity[:, 0])
    relative = neighbors[obs_len:] - primary[obs_len:, np.newaxis, :]
    neighbor_velocity = neighbors[obs_len:] - neighbors[obs_len - stride : -stride]

    position_theta = np.arctan2(relative[..., 1], relative[..., 0])
    theta_position = np.degrees(position_theta - primary_theta[:, np.newaxis]) % 360
    neighbor_theta = np.arctan2(neighbor_velocity[..., 1], neighbor_velocity[..., 0])
    theta_velocity = np.degrees(neighbor_theta - primary_theta[:, np.newaxis]) % 360
    distance = np.linalg.norm(relative, axis=2)
    return theta_position, theta_velocity, distance


def _angle_mask(theta: np.ndarray, distance: np.ndarray, angle: float, threshold: float, half_range: float):
    adjusted = theta.copy()
    if angle - half_range < 0:
        adjusted[adjusted > 180] -= 360
    if angle + half_range > 360:
        raise ValueError("angular window exceeds 360 degrees")
    return (
        (angle - half_range < adjusted)
        & (adjusted <= angle + half_range)
        & (distance < threshold)
    )


def check_interaction(
    xy: np.ndarray,
    pos_range: float = 15,
    dist_thresh: float = 5,
    choice: str = "pos",
    pos_angle: float = 0,
    vel_angle: float = 0,
    vel_range: float = 15,
    obs_len: int = 8,
) -> np.ndarray:
    if xy.shape[1] <= 1:
        return np.zeros((xy.shape[0] - obs_len, 0), dtype=bool)
    position, velocity, distance = _interaction_geometry(xy, obs_len)
    position_mask = _angle_mask(position, distance, pos_angle, dist_thresh, pos_range)
    if choice == "pos":
        return position_mask
    velocity_mask = _angle_mask(velocity, distance, vel_angle, dist_thresh, vel_range)
    if choice == "vel":
        return velocity_mask
    if choice in {"bothpos", "bothvel"}:
        return position_mask & velocity_mask
    raise ValueError(f"Unknown interaction choice: {choice}")


def check_group(
    xy: np.ndarray,
    dist_thresh: float = 0.8,
    std_thresh: float = 0.2,
    obs_len: int = 8,
) -> np.ndarray:
    if xy.shape[1] <= 1:
        return np.zeros(0, dtype=bool)
    side = np.any(check_interaction(xy, pos_angle=90, pos_range=45, obs_len=obs_len), axis=0)
    side |= np.any(check_interaction(xy, pos_angle=270, pos_range=45, obs_len=obs_len), axis=0)
    distance = np.linalg.norm(xy[:, 1:] - xy[:, 0, np.newaxis, :], axis=2)
    with np.errstate(invalid="ignore"):
        mean = np.mean(distance, axis=0)
        std = np.std(distance, axis=0)
    return (mean < dist_thresh) & (std < std_thresh) & side


def get_interaction_type(
    xy: np.ndarray,
    pos_range: float = 15,
    dist_thresh: float = 5,
    obs_len: int = 8,
    group_dist_thresh: float = 0.8,
    group_std_thresh: float = 0.2,
) -> list[int]:
    tags = []
    leader = check_interaction(
        xy, pos_range, dist_thresh, choice="bothpos", obs_len=obs_len
    )
    if np.any(np.sum(leader, axis=0) >= 5):
        tags.append(1)
    avoidance = check_interaction(
        xy,
        pos_range,
        dist_thresh,
        choice="bothpos",
        vel_angle=180,
        obs_len=obs_len,
    )
    if np.any(np.sum(avoidance, axis=0) >= 1):
        tags.append(2)
    if np.any(check_group(xy, group_dist_thresh, group_std_thresh, obs_len)):
        tags.append(3)
    return tags or [4]


def kalman_fde(
    xy: np.ndarray,
    obs_len: int,
    pred_len: int,
    rng: np.random.Generator,
) -> float:
    try:
        import pykalman
    except ModuleNotFoundError as error:
        raise ModuleNotFoundError("Categorization requires pykalman") from error

    primary = np.asarray(xy[:, 0], dtype=np.float64)
    # pykalman requires a RandomState; derive it from, but never replace, the
    # independently managed data Generator.
    kalman_rng = np.random.RandomState(int(rng.integers(0, 2**32 - 1)))
    transition = np.array(
        [[1, 1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    observation = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], dtype=np.float64)
    kf = pykalman.KalmanFilter(
        transition_matrices=transition,
        observation_matrices=observation,
        transition_covariance=1e-5 * np.eye(4),
        observation_covariance=0.05**2 * np.eye(2),
        initial_state_mean=[primary[0, 0], 0, primary[0, 1], 0],
        random_state=kalman_rng,
    )
    kf.em(primary[:obs_len])
    states, _ = kf.smooth(primary[:obs_len])
    predictions = []
    for _ in range(5):
        _, sampled = kf.sample(pred_len + 1, initial_state=states[-1], random_state=kalman_rng)
        predictions.append(sampled)
    prediction = np.mean(predictions, axis=0)[1:]
    return float(np.linalg.norm(prediction[-1] - primary[-1]))


def scene_rng(data_seed: int, scene_key: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{data_seed}:{scene_key}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def get_type(
    xy: np.ndarray,
    obs_len: int = 8,
    pred_len: int = 12,
    cfg: CategoryConfig | None = None,
    rng: np.random.Generator | None = None,
) -> tuple[int, list[int]]:
    cfg = cfg or CategoryConfig()
    rng = rng or np.random.default_rng(0)
    if xy.shape[0] != obs_len + pred_len or xy.shape[1] < 1:
        raise ValueError("xy must have shape (obs_len + pred_len, agents, 2)")
    if not np.isfinite(xy[:, 0]).all():
        raise ValueError("primary trajectory must be complete")

    primary = xy[:, 0]
    if np.linalg.norm(primary[-1] - primary[0]) < cfg.static_threshold:
        return 1, []
    if kalman_fde(xy, obs_len, pred_len, rng) < cfg.linear_threshold:
        return 2, []
    interacting = np.any(
        check_interaction(xy, cfg.inter_pos_range, cfg.inter_dist_thresh, obs_len=obs_len)
    ) or np.any(check_group(xy, cfg.grp_dist_thresh, cfg.grp_std_thresh, obs_len))
    if interacting:
        return 3, get_interaction_type(
            xy,
            cfg.inter_pos_range,
            cfg.inter_dist_thresh,
            obs_len,
            cfg.grp_dist_thresh,
            cfg.grp_std_thresh,
        )
    return 4, []
