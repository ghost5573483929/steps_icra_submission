from __future__ import annotations

import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

import preprocess
from config import Config, DatasetSpec, SeedConfig, SizingConfig
from conftest import write_canonical_fixture
from preprocess import INDEX_SCHEMA, _track_segments, auto_chunk_stride, preprocess_dataset

# A window is HISTORY + FUTURE canonical steps; consecutive candidate windows of
# one track start CHUNK_STRIDE canonical steps apart.
HISTORY_STEPS = 4
FUTURE_STEPS = 4
CHUNK_SIZE = HISTORY_STEPS + FUTURE_STEPS
CANONICAL_STRIDE = 1
CHUNK_STRIDE = 3
EVAL_STRIDE = 5
TRACK_STEPS = 20
TRACK_IDS = (1, 2, 3)
RECORDINGS = ("r0", "r1", "r2", "r3", "r4")

# floor((N - (obs + pred)) / chunk_stride) + 1 = floor(12 / 3) + 1
EXPECTED_WINDOWS_PER_TRACK = (TRACK_STEPS - CHUNK_SIZE) // CHUNK_STRIDE + 1
EXPECTED_START_STEPS = [0, 3, 6, 9, 12]


def write_task(path: Path, **overrides) -> Path:
    task = {
        "sample": "scene",
        "sample_hz": 2.5,
        "canonical_stride": CANONICAL_STRIDE,
        "history_steps": HISTORY_STEPS,
        "future_steps": FUTURE_STEPS,
        "min_length": 0.0,
        "w_target": 2.0,
        "eval_stride": EVAL_STRIDE,
        "datasets": {"synthetic": {"chunk_stride": CHUNK_STRIDE}},
        "forecasting": {"preset": "steps_default"},
    }
    task.update(overrides)
    path.write_text(json.dumps(task, indent=2) + "\n")
    return path


def build_dataset(root: Path, *, recordings=RECORDINGS, steps: int = TRACK_STEPS) -> Path:
    """All tracks present at every step and moving, so no window is dropped by the
    neighbour-density or min-length filters; each recording is its own scene."""
    converted = root / "converted"
    converted.mkdir(parents=True, exist_ok=True)
    for recording in recordings:
        write_canonical_fixture(
            converted / f"{recording}.traj.parquet",
            {
                track_id: [
                    (step, float(step * (track_id + 1)), float(track_id), None)
                    for step in range(steps)
                ]
                for track_id in TRACK_IDS
            },
            dataset="synthetic",
            scene_id=f"scene-{recording}",
            scene_sequence=recording,
            recording=recording,
        )
    (root / "source.json").write_text(
        json.dumps({"canonical_recordings": list(recordings)}) + "\n"
    )
    return root


def make_config(task_path: Path, *, data_seed: int = 7, n_target: int | None = None) -> Config:
    return Config(
        seeds=SeedConfig(data_seed=data_seed, model_seed=42),
        sizing=SizingConfig(n_target=n_target),
        eval_stride=None,
        task_path=task_path,
    )


@pytest.fixture(scope="module")
def preprocessed(tmp_path_factory) -> dict:
    root = build_dataset(tmp_path_factory.mktemp("synthetic"))
    task_path = write_task(root / "task.json")
    cfg = make_config(task_path)
    index_path, stats = preprocess_dataset(DatasetSpec("synthetic", root), cfg, workers=1)
    table = pq.read_table(index_path)
    manifest_path = root / "split_manifests" / f"grouped-seed{cfg.seeds.data_seed}.json"
    return {
        "root": root,
        "index_path": index_path,
        "rows": table.to_pylist(),
        "schema": table.schema,
        "stats": stats,
        "manifest": json.loads(manifest_path.read_text()),
    }


def test_track_segments_split_on_gaps_in_the_canonical_grid() -> None:
    # Track 1 is missing step 4, so it contributes two runs; track 2 is
    # uninterrupted.  Input rows are deliberately unsorted.
    track_ids = np.array([1, 1, 2, 1, 1, 2, 1, 1, 2])
    steps = np.array([1, 0, 2, 3, 2, 0, 5, 6, 1])
    segments = [
        (track_id, segment.tolist())
        for track_id, segment in _track_segments(track_ids, steps, 1)
    ]
    assert segments == [(1, [0, 1, 2, 3]), (1, [5, 6]), (2, [0, 1, 2])]

    # With canonical_stride 2 only even steps lie on the canonical grid, and
    # step 6 is two grid points after step 2, so it starts a new segment.
    strided = [
        (track_id, segment.tolist())
        for track_id, segment in _track_segments(
            np.full(6, 1), np.array([0, 1, 2, 3, 6, 7]), 2
        )
    ]
    assert strided == [(1, [0, 2]), (1, [6])]


@pytest.mark.parametrize(
    "length_median,chunk_size,w_target",
    [(20, 8, 2.0), (20, 8, 4.0), (100, 20, 2.0), (100, 20, 5.0), (10, 8, 10.0), (8, 8, 2.0)],
)
def test_auto_chunk_stride_realizes_at_most_the_target_window_count(
    length_median: int, chunk_size: int, w_target: float
) -> None:
    stride, realized = auto_chunk_stride(length_median, chunk_size, w_target)
    assert realized == (length_median - chunk_size) // stride + 1
    assert stride >= 2, "consecutive windows must be at least two steps apart"
    assert 1 <= realized <= max(1, int(w_target))


def test_auto_chunk_stride_hits_the_target_when_it_is_achievable() -> None:
    # L_med=20, chunk=8 leaves 12 steps of slack: two windows need a stride of
    # 12, four windows a stride of 12/3 = 4.
    assert auto_chunk_stride(20, 8, 2.0) == (12, 2)
    assert auto_chunk_stride(20, 8, 4.0) == (4, 4)
    # w_target <= 1 asks for a single window per median-length track.
    assert auto_chunk_stride(20, 8, 1.0) == (13, 1)


def test_index_matches_the_declared_schema(preprocessed) -> None:
    assert preprocessed["schema"].remove_metadata() == INDEX_SCHEMA
    assert preprocessed["rows"], "the pipeline produced an empty index"


def test_window_enumeration_follows_the_configured_chunk_stride(preprocessed) -> None:
    rows = preprocessed["rows"]
    # Hand count: each of the 3 tracks of each of the 5 recordings spans 20
    # canonical steps, giving floor((20 - 8) / 3) + 1 = 5 candidate windows
    # starting at 0, 3, 6, 9 and 12.
    assert EXPECTED_WINDOWS_PER_TRACK == 5
    assert len(rows) == len(RECORDINGS) * len(TRACK_IDS) * EXPECTED_WINDOWS_PER_TRACK

    for recording in RECORDINGS:
        for track_id in TRACK_IDS:
            starts = sorted(
                row["start_step"] for row in rows
                if row["recording"] == recording and row["primary_track_id"] == track_id
            )
            assert starts == EXPECTED_START_STEPS, (recording, track_id)

    for row in rows:
        assert row["observation_end_step"] == row["start_step"] + (HISTORY_STEPS - 1) * CANONICAL_STRIDE
        assert row["future_end_step"] == row["start_step"] + (CHUNK_SIZE - 1) * CANONICAL_STRIDE
        assert row["target_agent_id"] == row["primary_track_id"]
        assert row["dataset"] == "synthetic"


def test_every_indexed_window_agrees_with_the_written_split_manifest(preprocessed) -> None:
    manifest_split = {
        record["recording"]: record["split"] for record in preprocessed["manifest"]["records"]
    }
    assert set(manifest_split) == set(RECORDINGS)
    assert set(manifest_split.values()) == {"train", "val", "test"}
    for row in preprocessed["rows"]:
        assert row["split"] == manifest_split[row["recording"]], row["recording"]


def test_a_recording_never_appears_in_two_splits(preprocessed) -> None:
    splits_by_recording: dict[str, set[str]] = {}
    for row in preprocessed["rows"]:
        splits_by_recording.setdefault(row["recording"], set()).add(row["split"])
    leaked = {key: value for key, value in splits_by_recording.items() if len(value) > 1}
    assert not leaked, f"recording appears in multiple splits: {leaked}"


def test_acceptance_and_track_subsampling_apply_to_training_only(preprocessed) -> None:
    evaluation_rows = [row for row in preprocessed["rows"] if row["split"] != "train"]
    assert evaluation_rows
    for row in evaluation_rows:
        assert row["accepted"] is True, row
        assert row["track_kept"] is True, row


def test_evaluation_windows_are_thinned_to_the_evaluation_stride(preprocessed) -> None:
    minimum_separation = EVAL_STRIDE * CANONICAL_STRIDE
    groups: dict[tuple[str, int, str], list[dict]] = {}
    for row in preprocessed["rows"]:
        if row["split"] == "train":
            continue
        groups.setdefault((row["recording"], row["primary_track_id"], row["split"]), []).append(row)
    assert groups

    dropped = 0
    for key, rows in groups.items():
        kept = sorted(row["start_step"] for row in rows if row["eval_kept"])
        assert kept, key
        assert kept[0] == min(row["start_step"] for row in rows), key
        for earlier, later in zip(kept, kept[1:]):
            assert later - earlier >= minimum_separation, (key, kept)
        dropped += len(rows) - len(kept)
    # Candidates are 3 canonical steps apart but must end up >= 5 apart, so
    # thinning has to discard windows here.
    assert dropped > 0


def test_train_rows_are_never_dropped_by_the_evaluation_stride(preprocessed) -> None:
    assert all(row["eval_kept"] for row in preprocessed["rows"] if row["split"] == "train")


def test_sizing_stats_describe_the_synthetic_dataset(preprocessed) -> None:
    stats = preprocessed["stats"]
    assert stats["dataset"] == "synthetic"
    assert stats["data_seed"] == 7
    assert stats["s_d"] == CHUNK_STRIDE
    assert stats["L_med"] == TRACK_STEPS
    assert stats["L_mean"] == pytest.approx(float(TRACK_STEPS))
    assert stats["T_d_raw"] == len(RECORDINGS) * len(TRACK_IDS)
    assert stats["T_d_usable"] == len(RECORDINGS) * len(TRACK_IDS)
    assert stats["dense_forecast_windows"] == len(preprocessed["rows"])
    assert stats["split_candidate_scenes"] == stats["neighbor_filter_pass_scenes"]
    assert stats["scene_filter_survival"] == pytest.approx(1.0)
    assert stats["config_hash"].startswith("sha256:")
    assert (preprocessed["root"] / "stats" / "synthetic_sizing.json").exists()


def test_sizing_stats_carry_run_provenance_fields(preprocessed) -> None:
    stats = preprocessed["stats"]

    for field in ("git_commit", "git_dirty", "created_utc"):
        assert field in stats, f"stats is missing run-provenance field {field!r}"

    git_commit = stats["git_commit"]
    assert isinstance(git_commit, str)
    assert git_commit == "unknown" or re.fullmatch(r"[0-9a-f]{40}", git_commit), (
        f"git_commit {git_commit!r} is neither 'unknown' nor a 40-char hex sha"
    )
    assert git_commit != "unknown"
    assert isinstance(stats["git_dirty"], bool)

    created_utc = stats["created_utc"]
    assert isinstance(created_utc, str) and created_utc.endswith("Z")
    parsed = datetime.fromisoformat(created_utc.replace("Z", "+00:00"))
    assert parsed.tzinfo is not None
    assert parsed.utcoffset() == timezone.utc.utcoffset(None)


def test_manifest_forecast_window_counts_match_the_index(preprocessed) -> None:
    counted: dict[str, int] = {}
    for row in preprocessed["rows"]:
        counted[row["scene_sequence"]] = counted.get(row["scene_sequence"], 0) + 1
    for record in preprocessed["manifest"]["records"]:
        assert record["number_of_forecast_windows"] == counted.get(record["scene_sequence"], 0)


def test_repeated_runs_at_one_seed_reproduce_the_index(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "dataset")
    task_path = write_task(root / "task.json")
    cfg = make_config(task_path, data_seed=11)
    spec = DatasetSpec("synthetic", root)

    first_path, first_stats = preprocess_dataset(spec, cfg, workers=1)
    first_rows = pq.read_table(first_path).to_pylist()
    second_path, second_stats = preprocess_dataset(spec, cfg, workers=1)
    second_rows = pq.read_table(second_path).to_pylist()

    assert second_path == first_path
    assert second_rows == first_rows
    assert second_stats["config_hash"] == first_stats["config_hash"]


def test_the_data_seed_controls_the_retained_training_tracks(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "dataset")
    task_path = write_task(root / "task.json")
    spec = DatasetSpec("synthetic", root)

    def retained(data_seed: int) -> tuple[set[str], set[str]]:
        cfg = make_config(task_path, data_seed=data_seed, n_target=10)
        path, _ = preprocess_dataset(spec, cfg, workers=1)
        rows = pq.read_table(path).to_pylist()
        train = {
            f"{row['recording']}:{row['primary_track_id']}"
            for row in rows if row["split"] == "train" and row["accepted"]
        }
        kept = {
            f"{row['recording']}:{row['primary_track_id']}"
            for row in rows if row["split"] == "train" and row["accepted"] and row["track_kept"]
        }
        assert kept < train, "a budget below the accepted population must drop training tracks"
        return train, kept

    _, kept_one = retained(3)
    _, kept_two = retained(4)
    assert kept_one != kept_two


def test_a_dataset_entry_without_chunk_stride_is_rejected(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "dataset")
    task_path = write_task(root / "task.json", datasets={"synthetic": {}})
    with pytest.raises(KeyError, match="chunk_stride"):
        preprocess_dataset(DatasetSpec("synthetic", root), make_config(task_path), workers=1)


def test_a_dataset_without_a_full_length_track_is_rejected(tmp_path: Path) -> None:
    root = build_dataset(tmp_path / "dataset", recordings=("r0", "r1", "r2"), steps=CHUNK_SIZE - 1)
    task_path = write_task(root / "task.json")
    with pytest.raises(ValueError, match=f"no usable {CHUNK_SIZE}-step tracks"):
        preprocess_dataset(DatasetSpec("synthetic", root), make_config(task_path), workers=1)


def test_missing_recordings_are_reported(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    (empty / "converted").mkdir(parents=True)
    (empty / "source.json").write_text(json.dumps({"canonical_recordings": []}) + "\n")
    task_path = write_task(tmp_path / "task.json")
    with pytest.raises(FileNotFoundError, match="No converted recordings"):
        preprocess_dataset(DatasetSpec("synthetic", empty), make_config(task_path), workers=1)


def _pool_spy(created: list) -> type:
    class SpyExecutor(ProcessPoolExecutor):
        def __init__(self, *args, **kwargs) -> None:
            super().__init__(*args, **kwargs)
            created.append(self)

    return SpyExecutor


def _is_shut_down(executor: ProcessPoolExecutor) -> bool:
    try:
        executor.submit(int).result()
    except RuntimeError:
        return True
    return False


def test_parallel_categorization_reuses_one_pool_and_reproduces_the_serial_index(
    tmp_path: Path, monkeypatch
) -> None:
    recordings = ("r0", "r1", "r2")
    serial_root = build_dataset(tmp_path / "serial", recordings=recordings)
    serial_path, _ = preprocess_dataset(
        DatasetSpec("synthetic", serial_root),
        make_config(write_task(serial_root / "task.json")),
        workers=1,
    )

    created: list = []
    monkeypatch.setattr(preprocess, "ProcessPoolExecutor", _pool_spy(created))
    parallel_root = build_dataset(tmp_path / "parallel", recordings=recordings)
    parallel_path, _ = preprocess_dataset(
        DatasetSpec("synthetic", parallel_root),
        make_config(write_task(parallel_root / "task.json")),
        workers=2,
    )

    assert len(created) == 1, "one worker pool must serve every recording"
    assert _is_shut_down(created[0]), "the pool must be closed when the call returns"
    assert parallel_path.read_bytes() == serial_path.read_bytes()


def test_the_worker_pool_is_closed_when_categorization_raises(
    tmp_path: Path, monkeypatch
) -> None:
    root = build_dataset(tmp_path / "dataset", recordings=("r0", "r1"))
    task_path = write_task(root / "task.json")
    created: list = []
    monkeypatch.setattr(preprocess, "ProcessPoolExecutor", _pool_spy(created))

    real_assign_split = preprocess.assign_split
    calls = {"n": 0}

    def failing_assign_split(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 4:
            raise RuntimeError("induced categorization failure")
        return real_assign_split(*args, **kwargs)

    monkeypatch.setattr(preprocess, "assign_split", failing_assign_split)

    try:
        with pytest.raises(RuntimeError, match="induced categorization failure"):
            preprocess_dataset(DatasetSpec("synthetic", root), make_config(task_path), workers=2)
        assert len(created) == 1
        assert _is_shut_down(created[0]), "the pool must be closed when the call raises"
    finally:
        for executor in created:
            executor.shutdown(wait=True)
