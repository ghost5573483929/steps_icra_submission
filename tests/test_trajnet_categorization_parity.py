"""Parity of :mod:`categorization` against the vendored TrajNet++ reference.

Only the interaction primitives and Kalman predictor are diffed against real
upstream code.  ``reference_get_type`` is our own restatement of the four-way
cascade (``trajnetplusplusdataset`` is not vendored), so
``test_get_type_matches_reference_cascade`` cannot catch a design-level error
shared by both restatements; ``test_boundary_case_is_not_static`` pins one such
boundary independently.

Upstream observes 9 frames and this repository observes 8, so every comparison
runs at both lengths.
"""

from __future__ import annotations

import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np

from categorization import (
    check_group,
    check_interaction,
    get_interaction_type,
    get_type,
    kalman_fde,
)
from config import CategoryConfig

REFERENCE_ROOT = Path(__file__).resolve().parents[1] / "references" / "trajnetplusplustools"
REFERENCE_AVAILABLE = (REFERENCE_ROOT / "trajnetplusplustools" / "interactions.py").exists()

if REFERENCE_AVAILABLE:
    if str(REFERENCE_ROOT) not in sys.path:
        sys.path.insert(0, str(REFERENCE_ROOT))
    from trajnetplusplustools import interactions as reference_interactions
    from trajnetplusplustools import kalman as reference_kalman
    from trajnetplusplustools import metrics as reference_metrics
    from trajnetplusplustools.data import TrackRow


PRED_LEN = 12
OBSERVATION_LENGTHS = (8, 9)
DEFAULTS = CategoryConfig()

# (speed low, speed high, per-step heading noise, minimum agents), in metres at the
# TrajNet++ 2.5 Hz cadence, covering all four branches of the cascade.
REGIMES = (
    (0.0, 0.05, 0.15, 1),
    (0.3, 0.60, 0.02, 1),
    (0.3, 0.80, 0.35, 2),
    (0.3, 0.80, 0.35, 1),
)


def _walk(rng, steps, speed_low, speed_high, turn, start):
    heading = rng.uniform(0, 2 * np.pi) + np.cumsum(rng.normal(0.0, turn, size=steps))
    speed = rng.uniform(speed_low, speed_high)
    velocity = speed * np.stack([np.cos(heading), np.sin(heading)], axis=1)
    return start + np.cumsum(velocity, axis=0)


def random_scene(rng: np.random.Generator, obs_len: int, agents: int, regime: int) -> np.ndarray:
    speed_low, speed_high, turn, minimum_agents = REGIMES[regime]
    agents = max(agents, minimum_agents)
    steps = obs_len + PRED_LEN
    xy = np.empty((steps, agents, 2), dtype=np.float64)
    xy[:, 0] = _walk(rng, steps, speed_low, speed_high, turn, np.zeros(2))
    for agent in range(1, agents):
        if regime == 2:
            xy[:, agent] = xy[:, 0] + rng.uniform(-1.5, 1.5, size=2) + rng.normal(0.0, 0.05, size=(steps, 2))
        else:
            xy[:, agent] = _walk(rng, steps, speed_low, speed_high, turn, rng.uniform(-8.0, 8.0, size=2))
    return xy


def scenes(seed: int, count: int = 32):
    for obs_len in OBSERVATION_LENGTHS:
        rng = np.random.default_rng(seed)
        for index in range(count):
            yield obs_len, random_scene(rng, obs_len, 1 + index % 5, index % 4)


def reference_get_type(xy: np.ndarray, obs_len: int, cfg: CategoryConfig, seed: int):
    """``trajnetplusplusdataset.get_type`` restated over reference primitives."""

    primary = xy[:, 0]
    if np.linalg.norm(primary[-1] - primary[0]) < cfg.static_threshold:
        return 1, []

    np.random.seed(seed)
    path = [TrackRow(step, 0, float(x), float(y)) for step, (x, y) in enumerate(primary)]
    prediction = reference_kalman.predict([path], obs_len, PRED_LEN)[0][0]
    if reference_metrics.final_l2(path, prediction) < cfg.linear_threshold:
        return 2, []

    interacting = np.any(
        reference_interactions.check_interaction(
            xy, pos_range=cfg.inter_pos_range, dist_thresh=cfg.inter_dist_thresh, obs_len=obs_len
        )
    ) or np.any(
        reference_interactions.check_group(
            xy, dist_thresh=cfg.grp_dist_thresh, std_thresh=cfg.grp_std_thresh, obs_len=obs_len
        )
    )
    if interacting:
        return 3, reference_interactions.get_interaction_type(
            xy, cfg.inter_pos_range, cfg.inter_dist_thresh, obs_len
        )
    return 4, []


@unittest.skipUnless(REFERENCE_AVAILABLE, "references/trajnetplusplustools submodule not checked out")
class InteractionPrimitiveParityTests(unittest.TestCase):
    def test_check_interaction_matches_reference(self):
        cases = [
            dict(choice="pos"),
            dict(choice="vel"),
            dict(choice="bothpos"),
            dict(choice="bothvel"),
            dict(choice="bothpos", vel_angle=180),
            dict(choice="pos", pos_angle=90, pos_range=45),
            dict(choice="pos", pos_angle=270, pos_range=45),
        ]
        for obs_len, xy in scenes(20260911):
            if xy.shape[1] == 1:
                # Upstream indexes an empty neighbour axis here; our guard returns
                # the empty matrix instead.
                self.assertEqual(check_interaction(xy, obs_len=obs_len).shape, (PRED_LEN, 0))
                continue
            for case in cases:
                with self.subTest(obs_len=obs_len, agents=xy.shape[1], **case):
                    np.testing.assert_array_equal(
                        check_interaction(xy, obs_len=obs_len, **case),
                        reference_interactions.check_interaction(xy, obs_len=obs_len, **case),
                    )

    def test_check_group_matches_reference(self):
        grouped = 0
        for obs_len, xy in scenes(20260911):
            if xy.shape[1] == 1:
                continue
            with self.subTest(obs_len=obs_len, agents=xy.shape[1]):
                ours = check_group(xy, obs_len=obs_len)
                np.testing.assert_array_equal(
                    ours, reference_interactions.check_group(xy, obs_len=obs_len)
                )
            grouped += int(np.any(ours))
        self.assertGreater(grouped, 0, "no scene was ever grouped; agreement would be trivial")

    def test_get_interaction_type_matches_reference(self):
        observed = Counter()
        for obs_len, xy in scenes(20260911):
            if xy.shape[1] == 1:
                continue
            with self.subTest(obs_len=obs_len, agents=xy.shape[1]):
                ours = get_interaction_type(xy, obs_len=obs_len)
                self.assertEqual(ours, reference_interactions.get_interaction_type(xy, obs_len=obs_len))
            observed.update(ours)
        for sub_tag in (1, 2, 3, 4):
            self.assertGreater(observed[sub_tag], 0, f"interaction sub-tag {sub_tag} never exercised")

    def test_group_thresholds_are_the_reference_defaults(self):
        # Upstream's grouping sub-tag ignores its caller's thresholds and always uses
        # these, so our pass-through agrees only while the configuration keeps them.
        self.assertEqual(DEFAULTS.grp_dist_thresh, 0.8)
        self.assertEqual(DEFAULTS.grp_std_thresh, 0.2)


@unittest.skipUnless(REFERENCE_AVAILABLE, "references/trajnetplusplustools submodule not checked out")
class KalmanParityTests(unittest.TestCase):
    def test_kalman_fde_matches_reference(self):
        for index, (obs_len, xy) in enumerate(scenes(11, count=6)):
            ours = kalman_fde(xy, obs_len, PRED_LEN, np.random.default_rng(index))

            np.random.seed(int(np.random.default_rng(index).integers(0, 2**32 - 1)))
            path = [TrackRow(s, 0, float(x), float(y)) for s, (x, y) in enumerate(xy[:, 0])]
            prediction = reference_kalman.predict([path], obs_len, PRED_LEN)[0][0]

            with self.subTest(obs_len=obs_len, index=index):
                self.assertAlmostEqual(ours, float(reference_metrics.final_l2(path, prediction)), places=9)


@unittest.skipUnless(REFERENCE_AVAILABLE, "references/trajnetplusplustools submodule not checked out")
class SceneTypeParityTests(unittest.TestCase):
    def test_get_type_matches_reference_cascade(self):
        observed = Counter()
        for index, (obs_len, xy) in enumerate(scenes(2718)):
            seed = int(np.random.default_rng(index).integers(0, 2**32 - 1))
            tag, sub_tags = get_type(xy, obs_len, PRED_LEN, DEFAULTS, np.random.default_rng(index))
            observed[tag] += 1
            with self.subTest(obs_len=obs_len, index=index, agents=xy.shape[1]):
                self.assertEqual((tag, sub_tags), reference_get_type(xy, obs_len, DEFAULTS, seed))
        for tag in (1, 2, 3, 4):
            self.assertGreater(observed[tag], 0, f"scene type {tag} never exercised")


class StaticThresholdBoundaryTests(unittest.TestCase):
    """``get_type``'s first branch is ``norm(displacement) < static_threshold -> tag 1``,
    so a displacement equal to the threshold must not be tagged static."""

    def test_boundary_case_is_not_static(self):
        cfg = CategoryConfig()
        obs_len, pred_len = 8, PRED_LEN
        xy = np.zeros((obs_len + pred_len, 1, 2), dtype=np.float64)
        # Stationary until the very last frame, then displaced by exactly the
        # static threshold: norm(primary[-1] - primary[0]) == cfg.static_threshold.
        xy[-1, 0] = (cfg.static_threshold, 0.0)
        tag, _ = get_type(xy, obs_len, pred_len, cfg, np.random.default_rng(0))
        self.assertNotEqual(tag, 1, "displacement equal to the threshold must not be tagged static")


if __name__ == "__main__":
    unittest.main()
