from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
existing = os.environ.get("PYTHONPATH")
os.environ["PYTHONPATH"] = (
    f"{SOURCE_ROOT}{os.pathsep}{existing}" if existing else str(SOURCE_ROOT)
)

# pytest.ini's ``pythonpath = src`` already put ``src`` on sys.path.
from canonical import canonical_metadata, schema_with_metadata  # noqa: E402


def write_canonical_fixture(
    path: Path,
    tracks: Mapping[int, Sequence[tuple[int, float, float, float | None]]],
    *,
    dataset: str,
    scene_id: str,
    scene_sequence: str,
    recording: str,
) -> Path:
    """Write a canonical-schema parquet fixture from ``{track_id: [(step, x, y, heading)]}``,
    sorted by ``(step, track_id)``.

    ``heading`` is used exactly as supplied: ``None`` gives an Arrow validity-null,
    ``float("nan")`` gives a valid-but-NaN value. These are not equivalent.
    """
    rows = [
        (track_id, step, x, y, heading)
        for track_id, track_rows in tracks.items()
        for step, x, y, heading in track_rows
    ]
    rows.sort(key=lambda row: (row[1], row[0]))
    schema = schema_with_metadata(canonical_metadata(
        dataset=dataset, scene_id=scene_id, scene_sequence=scene_sequence, recording=recording,
    ))
    table = pa.Table.from_arrays([
        pa.array([row[0] for row in rows], pa.int32()),
        pa.array([row[1] for row in rows], pa.int32()),
        pa.array([row[2] for row in rows], pa.float32()),
        pa.array([row[3] for row in rows], pa.float32()),
        pa.array([row[4] for row in rows], pa.float32()),
    ], schema=schema)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path)
    return path


def write_run_result(directory: Path, *, model: str = "moflow_teacher", **evaluation_fields) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({"model": model}))
    (directory / "evaluation.json").write_text(json.dumps(evaluation_fields))
    return directory
