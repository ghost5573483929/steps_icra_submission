from __future__ import annotations

import argparse
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any, Iterable

import orjson
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from canonical import CANONICAL_HZ, CANONICAL_SCHEMA as OUTPUT_SCHEMA, canonical_metadata, schema_with_metadata
from artifact_provenance import git_provenance


SOURCE_HZ = 20
OUTPUT_HZ = CANONICAL_HZ
SOURCE_FRAMES_PER_OUTPUT = int(round(SOURCE_HZ / OUTPUT_HZ))
ROW_GROUP_SIZE = 50_000
REPO_ROOT = Path(__file__).resolve().parents[3]

# Every source frame needs ``frame_index`` and ``timestamp_ns``, but only every
# SOURCE_FRAMES_PER_OUTPUT-th frame needs the rest of the ~600 kB record, so
# ``_decimation_keys`` byte-scans those two integers and falls back to a full parse
# for anything it cannot prove.
_FRAME_INDEX_KEY = b'"frame_index"'
_TIMESTAMP_NS_KEY = b'"timestamp_ns"'
_MAX_KEY_CANDIDATES = 4
_INT64_MIN = -(2 ** 63)
_INT64_MAX = 2 ** 63 - 1
# JSON's integer grammar as a complete object member. orjson decodes a literal wider than
# 64 bits as a float, so the scan is bounded to int64 where it matches orjson exactly.
_INTEGER_MEMBER = re.compile(rb"[ \t\n\r]*:[ \t\n\r]*(-?(?:0|[1-9][0-9]*))[ \t\n\r]*(?=[,}])")


QUALITY_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.int32(), nullable=False),
        pa.field("step", pa.int32(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("included_in_trajectory", pa.bool_(), nullable=False),
        pa.field("confirmed", pa.bool_(), nullable=False),
        pa.field("observed_this_frame", pa.bool_(), nullable=False),
        pa.field("stale", pa.bool_(), nullable=False),
        pa.field("state_source", pa.string(), nullable=False),
        pa.field("missed_seconds", pa.float32(), nullable=True),
        pa.field("velocity_x_mps", pa.float32(), nullable=True),
        pa.field("velocity_y_mps", pa.float32(), nullable=True),
        pa.field("covariance_xx", pa.float32(), nullable=True),
        pa.field("covariance_xy", pa.float32(), nullable=True),
        pa.field("covariance_yy", pa.float32(), nullable=True),
        pa.field("spread_m", pa.float32(), nullable=True),
        pa.field("yaw_concentration", pa.float32(), nullable=True),
        pa.field("camera_indices", pa.list_(pa.int16()), nullable=False),
    ]
)

FRAME_SCHEMA = pa.schema(
    [
        pa.field("step", pa.int32(), nullable=False),
        pa.field("source_frame_index", pa.int32(), nullable=False),
        pa.field("timestamp_ns", pa.int64(), nullable=False),
        pa.field("timestamp_s", pa.float64(), nullable=False),
        pa.field("sync_skew_ns", pa.int64(), nullable=True),
        pa.field("included_in_trajectory", pa.bool_(), nullable=False),
        pa.field("finalization_mode", pa.string(), nullable=True),
        pa.field("finalization_reason", pa.string(), nullable=True),
        pa.field("actual_lag_seconds", pa.float32(), nullable=True),
        pa.field("configured_delay_seconds", pa.float32(), nullable=True),
        pa.field("tracked_count", pa.int32(), nullable=False),
        pa.field("fused_count", pa.int32(), nullable=False),
        pa.field("present_cameras", pa.list_(pa.int16()), nullable=False),
        pa.field("fresh_cameras", pa.list_(pa.int16()), nullable=False),
        pa.field("missing_cameras", pa.list_(pa.int16()), nullable=False),
        pa.field("stale_cameras", pa.list_(pa.int16()), nullable=False),
    ]
)


RELEASE_SUFFIX = "_trajectory_data.jsonl"


def source_files(dataset_dir: Path) -> list[Path]:
    files = sorted((dataset_dir / "raw").rglob(f"*{RELEASE_SUFFIX}"))
    if not files:
        raise FileNotFoundError(
            f"No *{RELEASE_SUFFIX} files found under {dataset_dir / 'raw'}; "
            "point raw/ at the release tree (Atrium_Data/MM/DD/STEPS_MM_DD_HHMMSS"
            f"{RELEASE_SUFFIX})"
        )
    return files


def recording_name(path: Path) -> str:
    return path.name.removesuffix(RELEASE_SUFFIX)


def _finite_float(value: Any) -> float | None:
    if value is None:
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _pair(value: Any) -> tuple[float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None
    return _finite_float(value[0]), _finite_float(value[1])


def _covariance(value: Any) -> tuple[float | None, float | None, float | None]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None, None, None
    try:
        return _finite_float(value[0][0]), _finite_float(value[0][1]), _finite_float(value[1][1])
    except (IndexError, TypeError):
        return None, None, None


def _camera_indices(person: dict[str, Any]) -> list[int]:
    values = person.get("last_observed_camera_indices") or []
    return [int(value) for value in values]


def _camera_list(frame: dict[str, Any], key: str) -> list[int]:
    return [int(value) for value in (frame.get(key) or [])]


def _included_person(person: dict[str, Any], frame_included: bool) -> bool:
    return bool(
        frame_included
        and person.get("confirmed", False)
        and person.get("observed_this_frame", False)
        and person.get("state_source") == "global_kalman_posterior"
    )


def _bracket_delta_outside_strings(chunk: bytes) -> int | None:
    """Net bracket balance of *chunk* (``{``/``[`` minus ``}``/``]``) ignoring string contents.

    *chunk* must contain no backslash and must begin either at the start of the document or on
    an opening quote, so splitting on ``"`` yields segments that alternate outside-string
    (even indices) and inside-string (odd indices). Returns ``None`` when the quote count is
    odd, which means the caller's offset was inside a string rather than on a boundary.
    """
    parts = chunk.split(b'"')
    if len(parts) % 2 == 0:
        return None
    outside = b"".join(parts[::2])
    return outside.count(b"{") + outside.count(b"[") - outside.count(b"}") - outside.count(b"]")


def _top_level_integer(line: bytes, key: bytes) -> int | None:
    """Value of the top-level object member *key* (an escaped-quote-free ``b'"name"'`` token).

    *line* must already be known to contain no backslash, so every ``"`` byte delimits a
    string and an occurrence of *key* is a complete string token. Candidates are examined
    right to left and the first one at depth 1 is the member orjson would keep (duplicate
    keys resolve last-wins). Returns ``None`` whenever the scan cannot prove the answer, so
    the caller falls back to a full parse; returning ``None`` is always safe, returning a
    different number than ``orjson`` would is not.
    """
    end = len(line)
    limit = end
    for _ in range(_MAX_KEY_CANDIDATES):
        start = line.rfind(key, 0, limit)
        if start < 0:
            return None
        limit = start
        if 2 * start <= end:
            at_top_level = _bracket_delta_outside_strings(line[:start]) == 1
        else:
            at_top_level = _bracket_delta_outside_strings(line[start:]) == -1
        if not at_top_level:
            continue
        member = _INTEGER_MEMBER.match(line, start + len(key))
        if member is None:
            return None
        value = int(member.group(1))
        return value if _INT64_MIN <= value <= _INT64_MAX else None
    return None


def _decimation_keys(line: bytes) -> tuple[int, int] | None:
    """``(frame_index, timestamp_ns)`` of one frame line without materialising the frame.

    Returns ``None`` unless the scan provably reproduces ``int(orjson.loads(line)[key])``,
    so every uncertain line is handed back to ``orjson``. A decimated line is therefore not
    validated as complete JSON, only as having both keys at top level.
    """
    if line.find(b"\\") != -1:
        return None
    frame_index = _top_level_integer(line, _FRAME_INDEX_KEY)
    if frame_index is None:
        return None
    timestamp_ns = _top_level_integer(line, _TIMESTAMP_NS_KEY)
    if timestamp_ns is None:
        return None
    return frame_index, timestamp_ns


def _write_rows(writer: pq.ParquetWriter, rows: list[dict[str, Any]], schema: pa.Schema) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=schema), row_group_size=ROW_GROUP_SIZE)
        rows.clear()


def _metadata(
    source: Path,
    recording: str,
    *,
    dataset_name: str = "steps",
    scene_id: str | None = None,
) -> dict[str, str]:
    return canonical_metadata(dataset=dataset_name,scene_id=scene_id or dataset_name,scene_sequence=recording,recording=recording,extra={
        "source_file": str(source),
        "source_to_local_3x3": json.dumps(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
        ),
        "heading_transform": (
            "yaw_ground_rad is copied from the global tracker. Any future planar "
            "rotation must also rotate heading and wrap it to [-pi, pi]."
        ),
        "coordinate_source": "tracked_people.position_ground_xy (metres)",
        "source_hz": str(SOURCE_HZ),
        "output_hz": str(OUTPUT_HZ),
        "temporal_sampling": "every eighth 20 Hz source frame, phase anchored to the first frame",
        "trajectory_state_filter": (
            "confirmed and observed_this_frame and state_source=global_kalman_posterior"
        ),
        "end_of_stream_policy": (
            "frames with output_finalization.reason=end_of_stream_flush are excluded"
        ),
        "quality_sidecar": f"quality/{recording}.quality.parquet",
        "frame_sidecar": f"quality/{recording}.frames.parquet",
    })


def convert_file(
    source: Path,
    dataset_dir: Path,
    *,
    name: str | None = None,
    dataset_name: str = "steps",
    scene_id: str | None = None,
    provenance: dict[str, Any] | None = None,
) -> Path:
    """Stream one STEPS fixed-lag ``frames.jsonl`` into canonical and audit Parquets."""
    recording = name or recording_name(source)
    output = dataset_dir / "converted" / f"{recording}.traj.parquet"
    quality_output = dataset_dir / "quality" / f"{recording}.quality.parquet"
    frame_output = dataset_dir / "quality" / f"{recording}.frames.parquet"
    summary_output = dataset_dir / "quality" / f"{recording}.summary.json"
    for path in (output, quality_output, frame_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    temporary = [path.with_suffix(f".tmp{path.suffix}") for path in (output, quality_output, frame_output)]
    main_schema = schema_with_metadata(
        _metadata(source, recording, dataset_name=dataset_name, scene_id=scene_id)
    )
    main_writer = pq.ParquetWriter(temporary[0], main_schema, compression="zstd")
    quality_writer = pq.ParquetWriter(temporary[1], QUALITY_SCHEMA, compression="zstd")
    frame_writer = pq.ParquetWriter(temporary[2], FRAME_SCHEMA, compression="zstd")

    main_rows: list[dict[str, Any]] = []
    quality_rows: list[dict[str, Any]] = []
    frame_rows: list[dict[str, Any]] = []
    counts = {
        "source_frames": 0,
        "sampled_2p5hz_frames": 0,
        "end_of_stream_frames_excluded": 0,
        "trajectory_rows": 0,
        "quality_rows": 0,
        "tracker_predictions_excluded": 0,
        "unconfirmed_states_excluded": 0,
        "invalid_positions_excluded": 0,
    }
    source_deltas_s: list[float] = []
    trajectory_track_ids: set[int] = set()
    quality_track_ids: set[int] = set()
    first_frame_index: int | None = None
    previous_frame_index: int | None = None
    previous_timestamp_ns: int | None = None

    try:
        total_bytes = source.stat().st_size
        with source.open("rb") as stream, tqdm(
            total=total_bytes,
            unit="B",
            unit_scale=True,
            desc=f"converting {recording}",
        ) as progress:
            for line_number, line in enumerate(stream, 1):
                progress.update(len(line))
                if not line or line.isspace():
                    continue
                frame: dict[str, Any] | None = None
                keys = _decimation_keys(line)
                if keys is None:
                    frame = orjson.loads(line)
                    frame_index = int(frame["frame_index"])
                    timestamp_ns = int(frame["timestamp_ns"])
                else:
                    frame_index, timestamp_ns = keys
                counts["source_frames"] += 1
                if first_frame_index is None:
                    first_frame_index = frame_index
                if previous_frame_index is not None and frame_index <= previous_frame_index:
                    raise ValueError(
                        f"{source}:{line_number}: frame_index is not strictly increasing "
                        f"({previous_frame_index} -> {frame_index})"
                    )
                if previous_timestamp_ns is not None and timestamp_ns > previous_timestamp_ns:
                    source_deltas_s.append((timestamp_ns - previous_timestamp_ns) / 1e9)
                previous_frame_index = frame_index
                previous_timestamp_ns = timestamp_ns

                if (frame_index - first_frame_index) % SOURCE_FRAMES_PER_OUTPUT:
                    continue
                if frame is None:
                    frame = orjson.loads(line)
                counts["sampled_2p5hz_frames"] += 1
                step = (frame_index - first_frame_index) // SOURCE_FRAMES_PER_OUTPUT
                finalization = frame.get("output_finalization") or {}
                finalization_reason = finalization.get("reason")
                frame_included = finalization_reason != "end_of_stream_flush"
                if not frame_included:
                    counts["end_of_stream_frames_excluded"] += 1

                tracked_people = sorted(
                    frame.get("tracked_people") or [], key=lambda person: int(person["track_id"])
                )
                seen_track_ids: set[int] = set()
                for person in tracked_people:
                    track_id = int(person["track_id"])
                    if track_id in seen_track_ids:
                        raise ValueError(
                            f"{source}:{line_number}: duplicate tracked person {track_id}"
                        )
                    seen_track_ids.add(track_id)
                    quality_track_ids.add(track_id)
                    x, y = _pair(person.get("position_ground_xy"))
                    included = _included_person(person, frame_included) and x is not None and y is not None
                    if not frame_included:
                        pass
                    elif not person.get("confirmed", False):
                        counts["unconfirmed_states_excluded"] += 1
                    elif not person.get("observed_this_frame", False) or person.get("state_source") != "global_kalman_posterior":
                        counts["tracker_predictions_excluded"] += 1
                    elif x is None or y is None:
                        counts["invalid_positions_excluded"] += 1

                    velocity_x, velocity_y = _pair(person.get("velocity_ground_xy_mps"))
                    covariance_xx, covariance_xy, covariance_yy = _covariance(
                        person.get("position_covariance")
                    )
                    quality_rows.append(
                        {
                            "track_id": track_id,
                            "step": step,
                            "timestamp_ns": timestamp_ns,
                            "included_in_trajectory": included,
                            "confirmed": bool(person.get("confirmed", False)),
                            "observed_this_frame": bool(person.get("observed_this_frame", False)),
                            "stale": bool(person.get("stale", False)),
                            "state_source": str(person.get("state_source") or "unknown"),
                            "missed_seconds": _finite_float(person.get("missed_seconds")),
                            "velocity_x_mps": velocity_x,
                            "velocity_y_mps": velocity_y,
                            "covariance_xx": covariance_xx,
                            "covariance_xy": covariance_xy,
                            "covariance_yy": covariance_yy,
                            "spread_m": _finite_float(person.get("spread_m")),
                            "yaw_concentration": _finite_float(person.get("yaw_concentration")),
                            "camera_indices": _camera_indices(person),
                        }
                    )
                    counts["quality_rows"] += 1
                    if included:
                        heading = _finite_float(person.get("yaw_ground_rad"))
                        main_rows.append(
                            {
                                "track_id": track_id,
                                "step": step,
                                "x": x,
                                "y": y,
                                "heading": heading,
                            }
                        )
                        trajectory_track_ids.add(track_id)
                        counts["trajectory_rows"] += 1

                frame_rows.append(
                    {
                        "step": step,
                        "source_frame_index": frame_index,
                        "timestamp_ns": timestamp_ns,
                        "timestamp_s": float(frame.get("timestamp_s", timestamp_ns / 1e9)),
                        "sync_skew_ns": int(frame["sync_skew_ns"]) if frame.get("sync_skew_ns") is not None else None,
                        "included_in_trajectory": frame_included,
                        "finalization_mode": finalization.get("mode"),
                        "finalization_reason": finalization_reason,
                        "actual_lag_seconds": _finite_float(finalization.get("actual_lag_seconds")),
                        "configured_delay_seconds": _finite_float(finalization.get("configured_delay_seconds")),
                        "tracked_count": len(tracked_people),
                        "fused_count": len(frame.get("fused_people") or []),
                        "present_cameras": _camera_list(frame, "present_cameras"),
                        "fresh_cameras": _camera_list(frame, "fresh_cameras"),
                        "missing_cameras": _camera_list(frame, "missing_cameras"),
                        "stale_cameras": _camera_list(frame, "stale_cameras"),
                    }
                )

                if len(main_rows) >= ROW_GROUP_SIZE:
                    _write_rows(main_writer, main_rows, main_schema)
                if len(quality_rows) >= ROW_GROUP_SIZE:
                    _write_rows(quality_writer, quality_rows, QUALITY_SCHEMA)
                if len(frame_rows) >= ROW_GROUP_SIZE:
                    _write_rows(frame_writer, frame_rows, FRAME_SCHEMA)

        if not counts["trajectory_rows"]:
            raise ValueError(f"{source}: no confirmed observed posterior trajectory rows were found")
        median_source_dt = statistics.median(source_deltas_s) if source_deltas_s else None
        measured_source_hz = 1.0 / median_source_dt if median_source_dt else None
        if measured_source_hz is not None and not 18.0 <= measured_source_hz <= 22.0:
            raise ValueError(
                f"{source}: measured source rate {measured_source_hz:.3f} Hz is incompatible "
                f"with the expected {SOURCE_HZ} Hz frame-decimation rule"
            )
        _write_rows(main_writer, main_rows, main_schema)
        _write_rows(quality_writer, quality_rows, QUALITY_SCHEMA)
        _write_rows(frame_writer, frame_rows, FRAME_SCHEMA)
    except BaseException:
        main_writer.close()
        quality_writer.close()
        frame_writer.close()
        for path in temporary:
            path.unlink(missing_ok=True)
        raise
    else:
        main_writer.close()
        quality_writer.close()
        frame_writer.close()
        for source_path, destination in zip(temporary, (output, quality_output, frame_output)):
            source_path.replace(destination)

    summary = {
        "dataset": dataset_name,
        "recording": recording,
        "source_file": str(source),
        "source_hz_expected": SOURCE_HZ,
        "source_hz_measured_from_median_dt": measured_source_hz,
        "output_hz": OUTPUT_HZ,
        "trajectory_tracks": len(trajectory_track_ids),
        "tracker_tracks": len(quality_track_ids),
        **counts,
        **(provenance or git_provenance(REPO_ROOT)),
    }
    summary_output.write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"[OK] {recording}: {counts['trajectory_rows']:,} rows, "
        f"{len(trajectory_track_ids):,} tracks; excluded "
        f"{counts['tracker_predictions_excluded']:,} predictions and "
        f"{counts['end_of_stream_frames_excluded']:,} final sampled frames",
        flush=True,
    )
    return output


def convert_dataset(
    dataset_dir: Path,
    sources: Iterable[Path] | None = None,
    *,
    dataset_name: str = "steps",
    scene_id: str | None = None,
) -> list[Path]:
    files = list(sources) if sources is not None else source_files(dataset_dir)
    if not files:
        raise FileNotFoundError("No STEPS indoor frames.jsonl files were provided")
    # Resolved once so every sidecar in a run records the same commit/dirty snapshot.
    provenance = git_provenance(REPO_ROOT)
    names: set[str] = set()
    outputs = []
    for number, source in enumerate(files, 1):
        name = recording_name(source)
        if name in names:
            raise ValueError(f"Duplicate recording name {name!r}; source paths must have unique parent names")
        names.add(name)
        print(f"[{number}/{len(files)}] {source}", flush=True)
        outputs.append(
            convert_file(
                source,
                dataset_dir,
                name=name,
                dataset_name=dataset_name,
                scene_id=scene_id,
                provenance=provenance,
            )
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert STEPS indoor fixed-lag tracker JSONL to unified 2.5 Hz Parquet"
    )
    parser.add_argument(
        "dataset_dir",
        type=Path,
        nargs="?",
        default=Path(__file__).resolve().parents[3] / "datasets/steps",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=Path,
        help="Explicit frames.jsonl path; repeat for multiple recordings (default: raw/**/frames.jsonl)",
    )
    args = parser.parse_args()
    convert_dataset(args.dataset_dir, args.source)


if __name__ == "__main__":
    main()
