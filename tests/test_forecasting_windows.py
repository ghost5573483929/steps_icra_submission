from __future__ import annotations

import json
import os
from dataclasses import replace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from config import Config, DatasetSpec
from data import SceneDataset, prepare_splits
from forecasting_config import resolve_forecasting_config
from forecasting_representation import prepare_forecast_representation
from forecasting_windows import (
    ForecastWindowRecord,
    RecordingWindowCache,
    enumerate_forecast_windows,
    materialize_forecast_window,
)
from conftest import write_canonical_fixture


ROOT = Path(__file__).resolve().parents[1]


def _write_window_fixture(path: Path, tracks, *, recording: str = "recording") -> Path:
    """Canonical fixture whose position is ``x = step`` and ``y = track_id``."""
    return write_canonical_fixture(
        path,
        {
            track_id: [(int(step), float(step), float(track_id), np.nan) for step in steps]
            for track_id, steps in tracks.items()
        },
        dataset="test", scene_id="scene", scene_sequence="sequence", recording=recording,
    )


def test_scholler_masked_window_zero_pads_after_its_actual_samples(tmp_path):
    # On a 12-step track the third 20-step window starts at step 2 and therefore
    # has exactly 10 real samples (steps 2..11) followed by 10 padded ones.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(12)})
    config = resolve_forecasting_config(
        preset="scholler", config_overrides={"scene_membership": "scholler_masked_track"}
    )
    record = enumerate_forecast_windows(path, config)[2]
    assert (record.start_step, record.actual_samples) == (2, 10)
    assert (record.observation_end_step, record.future_end_step) == (9, 21)

    xy, valid = materialize_forecast_window(record)
    assert xy.shape == (20, 1, 2)
    assert valid.shape == (20, 1)
    assert valid[:10, 0].all()
    assert not valid[10:, 0].any()
    np.testing.assert_array_equal(xy[10:], np.zeros((10, 1, 2), dtype=np.float32))
    # Fixture encoding: the sample at window offset k is the track's step 2 + k,
    # written as (x, y) = (step, track_id) = (2 + k, 1).
    expected = np.stack(
        (np.arange(2, 12, dtype=np.float32), np.ones(10, dtype=np.float32)), axis=1
    )
    np.testing.assert_array_equal(xy[:10, 0], expected)


def test_complete_window_materializes_every_step_of_every_agent(tmp_path):
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(20), 2: np.arange(20)})
    config = resolve_forecasting_config(preset="social_gan")
    record = enumerate_forecast_windows(path, config)[0]
    assert record.actual_samples == 20
    assert record.complete_agent_ids == (1, 2)

    xy, valid = materialize_forecast_window(record)
    assert valid.all()
    np.testing.assert_array_equal(xy[:, 0, 0], np.arange(20, dtype=np.float32))
    np.testing.assert_array_equal(xy[:, 1, 1], np.full(20, 2.0, dtype=np.float32))


def test_interleaved_tracks_land_in_their_own_agent_columns(tmp_path):
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {7: np.arange(20), 3: np.arange(20), 40: np.arange(4, 14)})

    on_file = pq.read_table(path, columns=["track_id", "step"])
    file_track_ids = on_file["track_id"].to_numpy()
    file_steps = on_file["step"].to_numpy()
    assert np.all(np.diff(file_steps) >= 0), "canonical fixtures are step-sorted"
    assert not np.all(np.diff(file_track_ids) >= 0), (
        "the fixture must interleave tracks by step; a track-major file would "
        "not exercise the row-to-agent matching this test pins"
    )

    record = ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path, 0, 7, 19, None, (7, 3, 40, 99), 0,
    )
    xy, valid = materialize_forecast_window(record)
    assert xy.shape == (20, 4, 2) and xy.dtype == np.float32
    assert valid.shape == (20, 4) and valid.dtype == np.bool_

    # Fixture encoding (x, y) = (step, track_id), requested agent order (7, 3, 40, 99):
    #   column 0 -> track 7,  steps 0..19  -> valid everywhere, y == 7
    #   column 1 -> track 3,  steps 0..19  -> valid everywhere, y == 3
    #   column 2 -> track 40, steps 4..13  -> valid only at offsets 4..13, y == 40
    #   column 3 -> track 99, absent       -> never valid, all NaN
    expected_valid = np.zeros((20, 4), dtype=bool)
    expected_valid[:, 0] = True
    expected_valid[:, 1] = True
    expected_valid[4:14, 2] = True
    np.testing.assert_array_equal(valid, expected_valid)

    steps = np.arange(20, dtype=np.float32)
    np.testing.assert_array_equal(xy[:, 0], np.stack((steps, np.full(20, 7.0, np.float32)), axis=1))
    np.testing.assert_array_equal(xy[:, 1], np.stack((steps, np.full(20, 3.0, np.float32)), axis=1))
    np.testing.assert_array_equal(
        xy[4:14, 2],
        np.stack((np.arange(4, 14, dtype=np.float32), np.full(10, 40.0, np.float32)), axis=1),
    )
    assert np.isnan(xy[:4, 2]).all() and np.isnan(xy[14:, 2]).all()
    assert np.isnan(xy[:, 3]).all()


def test_materialize_rejects_a_complete_window_the_recording_cannot_supply(tmp_path):
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(20)})
    complete = ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path, 0, 7, 19, 1, (1,), 20,
    )
    xy, valid = materialize_forecast_window(complete)
    assert valid.all()
    assert np.isfinite(xy).all()

    # One agent, 20 declared steps, none of them on file -> 0 of 20.
    absent = replace(complete, target_agent_id=99, complete_agent_ids=(99,))
    with pytest.raises(AssertionError, match="0 of 20"):
        materialize_forecast_window(absent)

    # Steps 0..19 minus step 10 is 19 rows for the one agent -> 19 of 20.
    _write_window_fixture(path, {1: np.delete(np.arange(20), 10)})
    with pytest.raises(AssertionError, match="19 of 20"):
        materialize_forecast_window(complete)


def test_materialize_rejects_a_padded_window_missing_part_of_its_valid_prefix(tmp_path):
    # The declared prefix is steps 0..9 for one agent = 10 slots, of which
    # step 5 is missing on file -> 9 of 10.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.delete(np.arange(12), 5)})
    padded = ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path, 0, 7, 19, 1, (1,), 10,
    )
    with pytest.raises(AssertionError, match="9 of 10"):
        materialize_forecast_window(padded)

    _write_window_fixture(path, {1: np.arange(10)})
    xy, valid = materialize_forecast_window(padded)
    assert valid[:10, 0].all()
    assert not valid[10:, 0].any()
    np.testing.assert_array_equal(xy[10:], np.zeros((10, 1, 2), dtype=np.float32))


def _generic_dataset(root: Path) -> Path:
    converted = root / "converted"
    converted.mkdir(parents=True)
    _write_window_fixture(converted / "recording.traj.parquet", {1: np.arange(20), 2: np.arange(20)})
    (root / "source.json").write_text(json.dumps({"canonical_recordings": ["recording"]}))
    prepare_splits(root, 42, strategy="grouped_scene_sequence")
    return root


def _task_with_stride(path: Path, stride: int) -> Path:
    task = json.loads((ROOT / "configs" / "task.json").read_text(encoding="utf-8"))
    task["canonical_stride"] = stride
    path.write_text(json.dumps(task), encoding="utf-8")
    return path


def test_generic_membership_requires_unit_canonical_stride(tmp_path):
    root = _generic_dataset(tmp_path / "dataset")
    forecasting = resolve_forecasting_config(preset="social_gan")
    unit = Config(forecasting=forecasting, task_path=_task_with_stride(tmp_path / "stride1.json", 1))
    assert len(SceneDataset(DatasetSpec("test", root), "train", unit)) == 1

    strided = Config(forecasting=forecasting, task_path=_task_with_stride(tmp_path / "stride2.json", 2))
    with pytest.raises(AssertionError, match="canonical_stride"):
        SceneDataset(DatasetSpec("test", root), "train", strided)


class _FixedScaleRng:
    def __init__(self, value: float) -> None:
        self._rng = np.random.default_rng(0)
        self._value = value

    def normal(self, *args, **kwargs) -> float:
        return self._value

    def __getattr__(self, name):
        return getattr(self._rng, name)


def _straight_line_window() -> tuple[np.ndarray, np.ndarray]:
    xy = np.zeros((20, 1, 2), dtype=np.float32)
    xy[:, 0, 0] = np.arange(20, dtype=np.float32)
    return xy, np.ones((20, 1), dtype=bool)


@pytest.mark.parametrize("draw", [0.0, -0.5])
def test_non_positive_random_scale_draw_raises_instead_of_collapsing_the_window(draw):
    config = resolve_forecasting_config(preset="social_vae")
    assert config.augmentation.random_scale.enabled
    xy, valid = _straight_line_window()
    with pytest.raises(ValueError, match="random_scale draw was non-positive"):
        prepare_forecast_representation(
            xy, valid, config, training=True, rng=_FixedScaleRng(draw)
        )


def test_default_random_scale_sampling_still_augments_normally():
    config = resolve_forecasting_config(preset="social_vae")
    xy, valid = _straight_line_window()
    representation = prepare_forecast_representation(
        xy, valid, config, training=True, rng=np.random.default_rng(0)
    )
    assert representation.augmentation["scale"] > 0
    assert np.isfinite(representation.xy).all()
    # scale is drawn with mean 1.0, std 0.05, so a few sigma stay well inside 0.25.
    assert abs(representation.augmentation["scale"] - 1.0) < 0.25
    assert np.ptp(representation.xy[:, 0, :], axis=0).max() > 10.0


def _social_gan_record(path: Path, start: int) -> ForecastWindowRecord:
    return ForecastWindowRecord(
        "test", "scene", "sequence", "recording", path,
        start, start + 7, start + 19, None, (1, 2), 20,
    )


def test_cache_serves_repeated_windows_with_the_values_independent_reads_give(tmp_path):
    # Fixture encoding (x, y) = (step, track_id), so for the window starting at
    # step 5: xy[k, agent] = (5 + k, agent + 1).
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(30), 2: np.arange(30)})
    record = _social_gan_record(path, 5)
    expected_xy = np.empty((20, 2, 2), dtype=np.float32)
    expected_xy[:, 0, 0] = expected_xy[:, 1, 0] = np.arange(5, 25, dtype=np.float32)
    expected_xy[:, 0, 1], expected_xy[:, 1, 1] = 1.0, 2.0

    cache = RecordingWindowCache()
    first_xy, first_valid = materialize_forecast_window(record, cache)
    second_xy, second_valid = materialize_forecast_window(record, cache)
    uncached_xy, uncached_valid = materialize_forecast_window(record, RecordingWindowCache())

    for xy, valid in ((first_xy, first_valid), (second_xy, second_valid), (uncached_xy, uncached_valid)):
        assert xy.dtype == np.float32 and xy.shape == (20, 2, 2)
        assert valid.all()
        np.testing.assert_array_equal(xy, expected_xy)
        assert xy.tobytes() == expected_xy.tobytes()
    assert cache.stats()["reads"] == 1
    assert cache.stats()["hits"] == 1

    shifted_xy, shifted_valid = materialize_forecast_window(_social_gan_record(path, 9), cache)
    assert shifted_valid.all()
    np.testing.assert_array_equal(shifted_xy[:, 0, 0], np.arange(9, 29, dtype=np.float32))
    assert cache.stats()["reads"] == 1


def test_cache_rereads_a_recording_that_changed_on_disk(tmp_path):
    # The rewrite keeps the row count (and so the file size) identical, exercising
    # the st_mtime_ns half of the cache key; utime pins a distinct mtime.
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(30), 2: np.arange(30)})
    record = _social_gan_record(path, 5)
    cache = RecordingWindowCache()
    before_xy, _ = materialize_forecast_window(record, cache)
    np.testing.assert_array_equal(before_xy[:, 0, 1], np.ones(20, dtype=np.float32))

    write_canonical_fixture(
        path,
        {
            track_id: [(int(step), float(step), float(track_id) + 100.0, np.nan) for step in np.arange(30)]
            for track_id in (1, 2)
        },
        dataset="test", scene_id="scene", scene_sequence="sequence", recording="recording",
    )
    os.utime(path, ns=(1_000_000_000, 1_500_000_000))

    after_xy, after_valid = materialize_forecast_window(record, cache)
    assert after_valid.all()
    assert cache.stats()["reads"] == 2
    np.testing.assert_array_equal(after_xy[:, 0, 1], np.full(20, 101.0, dtype=np.float32))
    np.testing.assert_array_equal(after_xy[:, 1, 1], np.full(20, 102.0, dtype=np.float32))
    np.testing.assert_array_equal(after_xy[:, 0, 0], np.arange(5, 25, dtype=np.float32))


def test_cache_stays_within_its_byte_budget_and_still_serves_correct_windows(tmp_path):
    first, second = tmp_path / "a.traj.parquet", tmp_path / "b.traj.parquet"
    _write_window_fixture(first, {1: np.arange(30), 2: np.arange(30)}, recording="a")
    _write_window_fixture(second, {1: np.arange(30), 2: np.arange(30)}, recording="b")
    # 30 steps x 2 tracks x 4 columns x 4 bytes = 960 bytes per recording.
    cache = RecordingWindowCache(budget_bytes=960)
    for _ in range(3):
        for path in (first, second):
            xy, valid = materialize_forecast_window(_social_gan_record(path, 5), cache)
            assert valid.all()
            np.testing.assert_array_equal(xy[:, 1, 0], np.arange(5, 25, dtype=np.float32))
            assert cache.stats()["entries"] == 1
            assert cache.stats()["cached_bytes"] == 960 <= cache.stats()["budget_bytes"]
    assert cache.stats() == {"reads": 6, "hits": 0, "entries": 1, "cached_bytes": 960, "budget_bytes": 960}

    tiny = RecordingWindowCache(budget_bytes=100)
    xy, valid = materialize_forecast_window(_social_gan_record(first, 5), tiny)
    assert valid.all()
    np.testing.assert_array_equal(xy[:, 0, 0], np.arange(5, 25, dtype=np.float32))
    assert tiny.stats()["entries"] == 0 and tiny.stats()["cached_bytes"] == 0


def test_cached_recording_columns_are_read_only(tmp_path):
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(30), 2: np.arange(30)})
    cache = RecordingWindowCache()
    track_ids, steps, xs, ys = cache.window(path, 5, 24)
    assert len(track_ids) == len(steps) == len(xs) == len(ys) == 40
    for column in (track_ids, steps, xs, ys):
        with pytest.raises(ValueError):
            column[0] = 0
