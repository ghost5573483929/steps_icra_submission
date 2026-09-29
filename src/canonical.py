from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


CANONICAL_SCHEMA_VERSION = 3
CANONICAL_HZ = 2.5
CANONICAL_DT = 1.0 / CANONICAL_HZ

CANONICAL_SCHEMA = pa.schema(
    [
        pa.field("track_id", pa.int32(), nullable=False),
        pa.field("step", pa.int32(), nullable=False),
        pa.field("x", pa.float32(), nullable=False),
        pa.field("y", pa.float32(), nullable=False),
        pa.field("heading", pa.float32(), nullable=True),
    ]
)

REQUIRED_METADATA = (
    "dataset",
    "scene_id",
    "scene_sequence",
    "recording",
    "canonical_schema_version",
    "canonical_hz",
    "coordinate_frame",
)


@dataclass(frozen=True)
class RecordingIdentity:
    dataset: str
    scene_id: str
    scene_sequence: str
    recording: str


def canonical_metadata(
    *,
    dataset: str,
    scene_id: str,
    scene_sequence: str,
    recording: str,
    extra: Mapping[str, object] | None = None,
) -> dict[str, str]:
    """Build the common metadata stored on every canonical recording."""
    result = {
        "dataset": str(dataset),
        "scene_id": str(scene_id),
        "scene_sequence": str(scene_sequence),
        "recording": str(recording),
        "canonical_schema_version": str(CANONICAL_SCHEMA_VERSION),
        "canonical_hz": str(CANONICAL_HZ),
        "coordinate_frame": "metric_world_xy_m",
        "position_representation": "original_converted_coordinates",
    }
    if extra:
        protected = set(REQUIRED_METADATA) | {"position_representation"}
        overlap = protected & {str(key) for key in extra}
        if overlap:
            raise ValueError(f"Extra metadata cannot override canonical fields: {sorted(overlap)}")
        result.update({str(key): str(value) for key, value in extra.items()})
    return result


def schema_with_metadata(metadata: Mapping[str, object]) -> pa.Schema:
    missing = set(REQUIRED_METADATA) - set(metadata)
    if missing:
        raise ValueError(f"Canonical metadata is missing {sorted(missing)}")
    return CANONICAL_SCHEMA.with_metadata(
        {str(key).encode(): str(value).encode() for key, value in metadata.items()}
    )


def decode_metadata(schema: pa.Schema) -> dict[str, str]:
    return {
        key.decode(): value.decode()
        for key, value in (schema.metadata or {}).items()
    }


def recording_identity(path: str | Path) -> RecordingIdentity:
    path = Path(path)
    metadata = decode_metadata(pq.ParquetFile(path).schema_arrow)
    recording = metadata.get("recording", path.name.removesuffix(".traj.parquet"))
    # Legacy v2 files are readable only for migration/auditing; v3 always carries
    # both explicit identities.
    scene_id = metadata.get("scene_id", metadata.get("scene", recording))
    sequence = metadata.get("scene_sequence", recording)
    return RecordingIdentity(
        metadata.get("dataset", path.parent.parent.name),
        scene_id,
        sequence,
        recording,
    )


def validate_canonical_file(path: str | Path) -> RecordingIdentity:
    path = Path(path)
    parquet = pq.ParquetFile(path)
    if parquet.schema_arrow.remove_metadata() != CANONICAL_SCHEMA:
        raise ValueError(f"{path} does not match the canonical column schema")
    metadata = decode_metadata(parquet.schema_arrow)
    missing = set(REQUIRED_METADATA) - set(metadata)
    if missing:
        raise ValueError(f"{path} is legacy/incomplete; missing metadata {sorted(missing)}")
    if int(metadata["canonical_schema_version"]) != CANONICAL_SCHEMA_VERSION:
        raise ValueError(f"{path} is not canonical schema v{CANONICAL_SCHEMA_VERSION}")
    if not np.isclose(float(metadata["canonical_hz"]), CANONICAL_HZ):
        raise ValueError(f"{path} is not stored at {CANONICAL_HZ} Hz")
    steps = parquet.read(columns=["step"]).column("step").to_numpy()
    if len(steps) > 1 and not np.all(np.diff(steps) >= 0):
        raise ValueError(f"{path} stores steps out of order; canonical recordings must be step-sorted")
    return recording_identity(path)
