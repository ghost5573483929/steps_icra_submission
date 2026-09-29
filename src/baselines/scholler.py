"""Deterministic last-observed-displacement constant velocity baseline.

Implements the non-sampling "OUR" constant-velocity model from Christoph Schöller,
Vincent Aravantinos, Florian Lay, Alois Knoll, "What the Constant Velocity Model Can
Teach Us About Pedestrian Motion Prediction", IEEE Robotics and Automation Letters
(RA-L) 5(2), 2020, https://arxiv.org/abs/1903.07933. Reference implementation vendored
at references/scholler-cv (see evaluate.py::constant_velocity_model, sample=False).
models.constant_velocity_prediction is an algorithmically identical copy that feeds
the published cv_fde_mean statistic.
"""
from __future__ import annotations

import numpy as np


def predict(observed_xy: np.ndarray, prediction_steps: int = 12) -> np.ndarray:
    observed = np.asarray(observed_xy)
    if observed.ndim != 2 or observed.shape[1] != 2 or len(observed) < 2:
        raise ValueError("observed_xy must have shape (T>=2, 2)")
    delta = observed[-1] - observed[-2]
    return observed[-1] + np.arange(1, prediction_steps + 1)[:, None] * delta


def errors(observed_xy: np.ndarray, future_xy: np.ndarray) -> tuple[float, float]:
    prediction = predict(observed_xy, len(future_xy))
    distances = np.linalg.norm(prediction - np.asarray(future_xy), axis=-1)
    return float(distances.mean()), float(distances[-1])
