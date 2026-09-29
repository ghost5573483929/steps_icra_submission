from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from pykalman import KalmanFilter


ForecastInputPreprocessing = Literal[
    "raw",
    "causal_kalman",
    "observation_only_smooth",
    "opentraj_offline_smooth_compat",
]
ForecastTargetPreprocessing = Literal["raw", "opentraj_offline_smooth_compat"]


@dataclass(frozen=True)
class ForecastPreprocessingResult:
    input_xy: np.ndarray
    target_xy: np.ndarray
    input_mode: str
    target_mode: str
    uses_future_information_for_input: bool


class OpenTrajKalmanModel:
    """Modern port of OpenTraj's constant-acceleration Kalman smoother
    (crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a,
    toolkit/utils/kalman_smoother.py; references/opentraj.json pins the commit).

    Deliberate deviations from that reference:

    - `n_iter` defaults to 7, matching upstream's own production call site rather
      than its unused class default of 4.
    - The transition matrix's top-right entry uses `dt**2`, not the textbook
      `dt**2/2`, to stay byte-identical to upstream; "correcting" it breaks parity.
    - `_filter` builds a fresh `pykalman.KalmanFilter` per call, reproducing the
      per-trajectory-group freshness upstream gets from a fresh model per group.
    """

    def __init__(self, dt: float, n_dim: int = 2, n_iter: int = 7) -> None:
        self.n_iter = n_iter
        self.n_dim = n_dim
        self.transition = np.array(
            [[1, dt, dt**2], [0, 1, dt], [0, 0, 1]], dtype=float
        )
        self.observation = np.array([[1, 0, 0]], dtype=float)
        self.transition_covariance = 0.5 * np.array(
            [
                [dt**5 / 20, dt**4 / 8, dt**3 / 6],
                [dt**4 / 8, dt**3 / 3, dt**2 / 2],
                [dt**3 / 6, dt**2 / 2, dt],
            ],
            dtype=float,
        )
        self.observation_covariance = np.array([[1.0]])

    def _filter(self, measurement: np.ndarray, *, smooth: bool) -> tuple[np.ndarray, np.ndarray]:
        measurement = np.asarray(measurement, dtype=float)
        if measurement.ndim != 2 or measurement.shape[1] != self.n_dim:
            raise ValueError(f"measurement must have shape (T, {self.n_dim})")
        if len(measurement) == 1:
            # Upstream guards only `smooth`; its `filter` would run EM on a single
            # sample. Guarding both is deliberate hardening beyond upstream.
            return measurement.copy(), np.zeros_like(measurement)
        states = []
        for dim in range(self.n_dim):
            kalman = KalmanFilter(
                transition_matrices=self.transition,
                observation_matrices=self.observation,
                transition_covariance=self.transition_covariance,
                observation_covariance=self.observation_covariance,
            ).em(measurement[:, dim], n_iter=self.n_iter)
            means, _ = kalman.smooth(measurement[:, dim]) if smooth else kalman.filter(measurement[:, dim])
            states.append(means)
        stacked = np.stack(states)
        return stacked[:, :, 0].T, stacked[:, :, 1].T

    def smooth(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._filter(measurement, smooth=True)

    def filter(self, measurement: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._filter(measurement, smooth=False)


def _segments(valid: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(valid)
    if not len(indices):
        return []
    return [part for part in np.split(indices, np.flatnonzero(np.diff(indices) != 1) + 1) if len(part)]


def _process_agents(
    xy: np.ndarray,
    valid: np.ndarray,
    dt: float,
    *,
    smooth: bool,
    stop: int | None = None,
) -> np.ndarray:
    output = np.array(xy, dtype=np.float32, copy=True)
    limit = len(output) if stop is None else min(stop, len(output))
    model = OpenTrajKalmanModel(dt=dt, n_dim=2, n_iter=7)
    for agent in range(output.shape[1]):
        mask = np.asarray(valid[:limit, agent], dtype=bool)
        for segment in _segments(mask):
            positions = output[segment, agent].astype(float)
            processed, _ = model.smooth(positions) if smooth else model.filter(positions)
            output[segment, agent] = processed
    return output


def preprocess_forecast_scene(
    xy: np.ndarray,
    valid: np.ndarray,
    obs_len: int = 8,
    dt: float = 0.4,
    input_mode: ForecastInputPreprocessing = "raw",
    target_mode: ForecastTargetPreprocessing = "raw",
) -> ForecastPreprocessingResult:
    """Build explicitly separated inputs and targets without silent future leakage."""
    xy = np.asarray(xy, dtype=np.float32)
    valid = np.asarray(valid, dtype=bool)
    if xy.ndim != 3 or valid.shape != xy.shape[:2]:
        raise ValueError("xy/valid must have shapes (T,N,2)/(T,N)")
    if not 1 < obs_len < len(xy):
        raise ValueError("obs_len must split the observed and future samples")

    input_xy = np.array(xy, copy=True)
    target_xy = np.array(xy[obs_len:], copy=True)
    leaks = False
    # Set only when the input mode ran the offline smoother over the whole window;
    # observation_only_smooth passes stop=obs_len, a different computation.
    offline_smoothed: np.ndarray | None = None
    if input_mode == "causal_kalman":
        processed = _process_agents(xy, valid, dt, smooth=False, stop=obs_len)
        input_xy[:obs_len] = processed[:obs_len]
    elif input_mode == "observation_only_smooth":
        processed = _process_agents(xy, valid, dt, smooth=True, stop=obs_len)
        input_xy[:obs_len] = processed[:obs_len]
    elif input_mode == "opentraj_offline_smooth_compat":
        offline_smoothed = _process_agents(xy, valid, dt, smooth=True)
        input_xy = offline_smoothed
        leaks = True
    elif input_mode != "raw":
        raise ValueError(f"Unknown forecast input preprocessing: {input_mode}")

    if target_mode == "opentraj_offline_smooth_compat":
        # _process_agents is pure and pykalman's EM is deterministic, so an identical
        # full-sequence smoothing is bit-identical and the second fit is skipped.
        if offline_smoothed is None:
            offline_smoothed = _process_agents(xy, valid, dt, smooth=True)
        # Copy so target_xy never becomes a view into input_xy when the array is shared.
        target_xy = np.array(offline_smoothed[obs_len:], copy=True)
    elif target_mode != "raw":
        raise ValueError(f"Unknown forecast target preprocessing: {target_mode}")
    return ForecastPreprocessingResult(input_xy, target_xy, input_mode, target_mode, leaks)
