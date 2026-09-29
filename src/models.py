from __future__ import annotations

from typing import Protocol

import numpy as np

from config import TrainConfig


def constant_velocity_prediction(observed: np.ndarray, pred_len: int = 12) -> np.ndarray:
    """Extrapolate the displacement over the final observed timestep.

    Same Schöller et al. (2020) constant-velocity definition as
    baselines/scholler.py::predict; this copy feeds the published cv_fde_mean statistic.
    """
    observed = np.asarray(observed)
    if observed.ndim != 2 or observed.shape[1] != 2 or len(observed) < 2:
        raise ValueError("observed must have shape (T, 2) with at least two timesteps")
    velocity = observed[-1] - observed[-2]
    return observed[-1] + np.arange(1, pred_len + 1)[:, None] * velocity


class TrajectoryModel(Protocol):
    def fit(self, train, val, cfg: TrainConfig) -> None: ...

    def predict(self, scene, k: int = 1) -> np.ndarray:
        """Return (k, pred_len, 2) in the same centered frame as scene.xy."""


class ConstantVelocityModel(TrajectoryModel):
    def __init__(self, obs_len: int = 8, pred_len: int = 12) -> None:
        self.obs_len = obs_len
        self.pred_len = pred_len

    def fit(self, train, val, cfg: TrainConfig) -> None:
        return None

    def predict(self, scene, k: int = 1) -> np.ndarray:
        if k != 1:
            raise ValueError("constant velocity is deterministic and only supports k=1")
        observed = scene.xy[: self.obs_len, 0]
        prediction = constant_velocity_prediction(observed, self.pred_len)
        return prediction[None]
