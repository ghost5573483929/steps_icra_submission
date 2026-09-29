"""Adapter from a canonical common scene window to official Social GAN tensors."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from forecasting_representation import relative_displacement


@dataclass(frozen=True)
class SocialGANWindow:
    obs_traj: np.ndarray
    pred_traj: np.ndarray
    obs_traj_rel: np.ndarray
    pred_traj_rel: np.ndarray
    loss_mask: np.ndarray
    seq_start_end: np.ndarray
    agent_indices: np.ndarray


def social_gan_window(
    xy: np.ndarray,
    valid: np.ndarray,
    *,
    observation_steps: int = 8,
    prediction_steps: int = 12,
    min_complete_agents: int = 2,
) -> SocialGANWindow | None:
    """Match upstream ``min_ped=1``: require >1 complete pedestrians."""
    xy = np.asarray(xy, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    total = observation_steps + prediction_steps
    if xy.shape[0] != total or valid.shape != xy.shape[:2]:
        raise ValueError("Social GAN input must be one common 8+12 scene window")
    complete = np.flatnonzero(valid.all(axis=0))
    if len(complete) < min_complete_agents:
        return None
    positions = xy[:, complete]
    relative = relative_displacement(positions, np.ones(positions.shape[:2], dtype=bool))
    # Official loader convention is (agents, 2, time).
    transpose = lambda values: np.transpose(values, (1, 2, 0))
    n = len(complete)
    return SocialGANWindow(
        obs_traj=transpose(positions[:observation_steps]),
        pred_traj=transpose(positions[observation_steps:]),
        obs_traj_rel=transpose(relative[:observation_steps]),
        pred_traj_rel=transpose(relative[observation_steps:]),
        loss_mask=np.ones((n, total), dtype=np.float32),
        seq_start_end=np.asarray([[0, n]], dtype=np.int64),
        agent_indices=complete,
    )
