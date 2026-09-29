"""Lightweight forecasting-window membership over canonical trajectory files."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from canonical import recording_identity, validate_canonical_file
from forecasting_config import ForecastingConfig

WINDOW_COLUMNS = ("track_id", "step", "x", "y")
# CANONICAL_SCHEMA pins these four columns to 4-byte types, so a cached
# recording costs 16 bytes per row and 64 MiB holds ~4.2M rows.
RECORDING_CACHE_BUDGET_BYTES = 64 * 1024 * 1024


@dataclass(frozen=True)
class ForecastWindowRecord:
    dataset: str
    scene_id: str
    scene_sequence: str
    recording: str
    path: Path
    start_step: int
    observation_end_step: int
    future_end_step: int
    target_agent_id: int | None
    complete_agent_ids: tuple[int, ...]
    actual_samples: int

    @property
    def agent_ids(self) -> tuple[int, ...]:
        assert self.complete_agent_ids, self
        return self.complete_agent_ids


def _tracks(path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    table = pq.read_table(path, columns=list(WINDOW_COLUMNS)).to_pandas()
    return {
        int(track_id): (
            track.step.to_numpy(dtype=np.int64),
            track[["x", "y"]].to_numpy(dtype=np.float32),
        )
        for track_id, track in table.sort_values(["track_id", "step"]).groupby("track_id", sort=True)
    }


def _segments(steps: np.ndarray):
    return np.split(np.arange(len(steps)), np.flatnonzero(np.diff(steps) != 1) + 1)


def enumerate_forecast_windows(path: Path, config: ForecastingConfig) -> list[ForecastWindowRecord]:
    membership = config.scene_membership
    if membership == "target_complete":
        raise ValueError(
            "scene_membership='target_complete' is not implemented by "
            "enumerate_forecast_windows: it treats it as byte-identical to "
            "'individual_track' and silently drops neighbor information. "
            "'target_complete' is only meaningful via the precomputed-index, "
            "scene-aware path in data.SceneDataset (see SceneDataset.__init__, "
            "which reads a precomputed index + _scene_arrays instead of calling "
            "this function). Use SceneDataset for scene_membership='target_complete', "
            "or choose 'individual_track'/'scholler_masked_track' to call this "
            "function directly."
        )
    identity = validate_canonical_file(path)
    tracks = _tracks(path)
    obs, pred = config.window.observation_steps, config.window.prediction_steps
    total, stride = obs + pred, config.window.stride
    records: list[ForecastWindowRecord] = []
    if membership == "social_gan_complete_scene":
        candidates: set[int] = set()
        step_sets = {track_id: set(map(int, values[0])) for track_id, values in tracks.items()}
        for steps, _ in tracks.values():
            for segment in _segments(steps):
                candidates.update(int(steps[segment[offset]]) for offset in range(0, max(0, len(segment)-total+1), stride))
        for start in sorted(candidates):
            expected = set(range(start, start + total))
            complete = tuple(sorted(track_id for track_id, steps in step_sets.items() if expected.issubset(steps)))
            if len(complete) < config.min_complete_agents:
                continue
            records.append(ForecastWindowRecord(identity.dataset, identity.scene_id, identity.scene_sequence, identity.recording, path, start, start+obs-1, start+total-1, None, complete, total))
        return records

    if membership not in {"individual_track", "scholler_masked_track"}:
        raise ValueError(f"Unsupported membership policy: {membership}")
    minimum = 10 if membership == "scholler_masked_track" else total
    for track_id, (steps, _) in tracks.items():
        for segment in _segments(steps):
            # Schöller et al. (2020) slice every suffix with >=10 actual samples,
            # then pad to 20 and mask the future labels.
            if membership == "scholler_masked_track":
                starts = range(0, max(0, len(segment)-minimum+1), stride)
            else:
                starts = range(0, max(0, len(segment)-total+1), stride)
            for offset in starts:
                actual = min(total, len(segment)-offset)
                start = int(steps[segment[offset]])
                records.append(ForecastWindowRecord(identity.dataset, identity.scene_id, identity.scene_sequence, identity.recording, path, start, start+obs-1, start+total-1, track_id, (track_id,), actual))
    return records


@dataclass(frozen=True)
class _CachedRecording:
    revision: tuple[int, int]
    columns: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    nbytes: int


class RecordingWindowCache:
    """Serves step-windows out of a byte-budgeted LRU cache of whole canonical recordings.

    Canonical recordings are step-sorted (``canonical.validate_canonical_file``),
    so a step range is one contiguous run of rows and ``searchsorted`` yields it
    in file order. Slices are views and the cached columns are read-only, so a
    downstream write raises instead of corrupting later samples.

    A recording larger than the whole budget is read and served but never
    retained, which keeps the bound hard.

    Entries are keyed on the path plus ``(st_size, st_mtime_ns)`` so a mid-run
    regeneration of a recording invalidates rather than silently persists.
    """

    def __init__(self, budget_bytes: int = RECORDING_CACHE_BUDGET_BYTES) -> None:
        if budget_bytes < 0:
            raise ValueError(f"budget_bytes must be non-negative, got {budget_bytes}")
        self.budget_bytes = int(budget_bytes)
        self._entries: OrderedDict[Path, _CachedRecording] = OrderedDict()
        self._cached_bytes = 0
        self._reads = 0
        self._hits = 0

    def stats(self) -> dict[str, int]:
        return {
            "reads": self._reads,
            "hits": self._hits,
            "entries": len(self._entries),
            "cached_bytes": self._cached_bytes,
            "budget_bytes": self.budget_bytes,
        }

    def window(
        self, path: Path, first_step: int, last_step: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """``WINDOW_COLUMNS`` for ``first_step <= step <= last_step``, in file order."""
        track_ids, steps, xs, ys = self._columns(Path(path))
        start = int(np.searchsorted(steps, first_step, side="left"))
        stop = int(np.searchsorted(steps, last_step, side="right"))
        return track_ids[start:stop], steps[start:stop], xs[start:stop], ys[start:stop]

    def _columns(self, path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        status = path.stat()
        revision = (int(status.st_size), int(status.st_mtime_ns))
        cached = self._entries.get(path)
        if cached is not None:
            if cached.revision == revision:
                self._entries.move_to_end(path)
                self._hits += 1
                return cached.columns
            self._evict(path)
        table = pq.read_table(path, columns=list(WINDOW_COLUMNS))
        self._reads += 1
        columns = tuple(table[column].combine_chunks().to_numpy() for column in WINDOW_COLUMNS)
        steps = columns[1]
        assert steps.size < 2 or bool(np.all(np.diff(steps) >= 0)), (
            f"{path} is not step-sorted, so a window cannot be taken as a contiguous "
            f"slice; canonical recordings are required to be step-sorted (see "
            f"canonical.validate_canonical_file)"
        )
        for column in columns:
            column.flags.writeable = False
        nbytes = sum(int(column.nbytes) for column in columns)
        if nbytes <= self.budget_bytes:
            while self._cached_bytes + nbytes > self.budget_bytes:
                self._evict(next(iter(self._entries)))
            self._entries[path] = _CachedRecording(revision, columns, nbytes)
            self._cached_bytes += nbytes
        return columns

    def _evict(self, path: Path) -> None:
        self._cached_bytes -= self._entries.pop(path).nbytes


def materialize_forecast_window(
    record: ForecastWindowRecord, cache: RecordingWindowCache | None = None
) -> tuple[np.ndarray, np.ndarray]:
    # Rows stay in file order so duplicate (step, agent) slots resolve
    # last-wins; do not sort.
    cache = cache if cache is not None else RecordingWindowCache()
    track_ids, steps, xs, ys = cache.window(
        record.path, record.start_step, record.future_end_step
    )
    agent_ids = record.agent_ids
    total = record.future_end_step - record.start_step + 1
    xy = np.full((total, len(agent_ids), 2), np.nan, dtype=np.float32)
    valid = np.zeros((total, len(agent_ids)), dtype=bool)
    # `searchsorted` runs over `agent_ids` (sorted here) and deliberately not
    # over `track_ids`: recordings are written in (step, track_id) order, so
    # `track_ids` is step-major and a track's rows are not contiguous.
    agents = np.array([int(track_id) for track_id in agent_ids], dtype=np.int64)
    order = np.argsort(agents, kind="stable")
    ordered = agents[order]
    first = np.searchsorted(ordered, track_ids, side="left")
    matches = np.searchsorted(ordered, track_ids, side="right") - first
    rows = np.repeat(np.arange(len(track_ids)), matches)
    within = np.arange(int(matches.sum())) - np.repeat(np.cumsum(matches) - matches, matches)
    columns = order[np.repeat(first, matches) + within]
    offsets = steps[rows].astype(np.int64) - record.start_step
    xy[offsets, columns] = np.stack((xs[rows], ys[rows]), axis=1).astype(np.float32, copy=False)
    valid[offsets, columns] = True
    # Only the padded tail beyond `actual_samples` is allowed to be invalid.
    prefix = valid[: record.actual_samples]
    assert prefix.all(), (
        f"{record.path} supplied only {int(prefix.sum())} of {prefix.size} expected "
        f"(step, agent) samples for the {record.actual_samples} enumerated sample(s) "
        f"of the window over steps {record.start_step}-{record.future_end_step} "
        f"(total={total}, agents={agent_ids})"
    )
    if record.actual_samples < total and record.target_agent_id is not None:
        # Schöller pads with zeros; `valid` carries the label mask so padded
        # samples never enter metrics.
        xy[~valid] = 0.0
    return xy, valid
