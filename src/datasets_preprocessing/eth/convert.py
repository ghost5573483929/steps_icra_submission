from __future__ import annotations

import argparse
import io
import json
import math
import sys
import tarfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pyarrow as pa
import pyarrow.parquet as pq
from canonical import CANONICAL_HZ, CANONICAL_SCHEMA as OUTPUT_SCHEMA, canonical_metadata, schema_with_metadata
from artifact_provenance import git_provenance


ANNOTATION_HZ = 2.5
CANONICAL_STEPS_PER_ANNOTATION = int(CANONICAL_HZ / ANNOTATION_HZ)
CONVERTER_VERSION = 1
REPO_ROOT = Path(__file__).resolve().parents[3]


def _write_provenance_sidecar(output: Path, provenance: dict[str, object]) -> None:
    payload = {
        "converter_version": CONVERTER_VERSION,
        "invocation": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **provenance,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")


def raw_archives(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "raw").glob("*.tgz")) + sorted(
        (dataset_dir / "raw").glob("*.tar.gz")
    )


def _recording_members(archive: tarfile.TarFile) -> list[tarfile.TarInfo]:
    return sorted(
        (
            member
            for member in archive.getmembers()
            if member.isfile() and PurePosixPath(member.name).name == "obsmat.txt"
        ),
        key=lambda member: member.name,
    )


def _read_obsmat(archive: tarfile.TarFile, member: tarfile.TarInfo) -> list[tuple[int, int, float, float, float, float]]:
    extracted = archive.extractfile(member)
    if extracted is None:
        raise ValueError(f"Could not read {member.name}")

    rows = []
    with io.TextIOWrapper(extracted, encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 8:
                raise ValueError(f"{member.name}:{line_number}: expected 8 fields, found {len(fields)}")
            frame, track_id, x, _z, y, velocity_x, _velocity_z, velocity_y = map(float, fields)
            if not frame.is_integer() or not track_id.is_integer():
                raise ValueError(f"{member.name}:{line_number}: frame and pedestrian ID must be integers")
            values = (x, y, velocity_x, velocity_y)
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"{member.name}:{line_number}: coordinates and velocities must be finite")
            rows.append((int(frame), int(track_id), x, y, velocity_x, velocity_y))
    if not rows:
        raise ValueError(f"ETH observation file contains no rows: {member.name}")
    return rows


def _native_frame_stride(rows: list[tuple[int, int, float, float, float, float]]) -> int:
    frames = sorted({row[0] for row in rows})
    differences = [right - left for left, right in zip(frames, frames[1:])]
    if not differences:
        raise ValueError("A recording must contain at least two annotated frames")
    stride, count = Counter(differences).most_common(1)[0]
    if stride <= 0 or count <= len(differences) / 2:
        raise ValueError("Could not infer a dominant positive annotation-frame interval")
    return stride


def _canonical_rows(rows: list[tuple[int, int, float, float, float, float]]) -> tuple[list[tuple[int, int, float, float, float | None]], int, int]:
    native_stride = _native_frame_stride(rows)
    frames = sorted({row[0] for row in rows})
    first_frame = frames[0]
    frame_to_annotation_index = {first_frame: 0}
    annotation_index = 0
    for previous, frame in zip(frames, frames[1:]):
        # Empty stretches can restart on a different video-frame phase, so elapsed
        # annotation time is accumulated incrementally: a phase shift must not collapse
        # two adjacent annotations onto the same canonical timestep.
        annotation_index += max(1, round((frame - previous) / native_stride))
        frame_to_annotation_index[frame] = annotation_index
    seen: set[tuple[int, int]] = set()
    converted = []
    for frame, track_id, x, y, velocity_x, velocity_y in rows:
        step = frame_to_annotation_index[frame] * CANONICAL_STEPS_PER_ANNOTATION
        key = (step, track_id)
        if key in seen:
            raise ValueError(f"Duplicate pedestrian {track_id} at canonical step {step}")
        seen.add(key)
        speed = math.hypot(velocity_x, velocity_y)
        heading = math.atan2(velocity_y, velocity_x) if speed > 1e-8 else None
        converted.append((step, track_id, x, y, heading))
    converted.sort(key=lambda row: (row[0], row[1]))
    return converted, native_stride, first_frame


def _write_recording(rows, output: Path, metadata: dict[str, str]) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    schema = schema_with_metadata(metadata)
    table = pa.Table.from_arrays(
        [
            pa.array([row[1] for row in rows], type=pa.int32()),
            pa.array([row[0] for row in rows], type=pa.int32()),
            pa.array([row[2] for row in rows], type=pa.float32()),
            pa.array([row[3] for row in rows], type=pa.float32()),
            pa.array([row[4] for row in rows], type=pa.float32()),
        ],
        schema=schema,
    )
    temporary = output.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)


def convert_archive(path: Path, dataset_dir: Path, provenance: dict[str, object] | None = None) -> list[Path]:
    outputs = []
    with tarfile.open(path, mode="r:gz") as archive:
        members = _recording_members(archive)
        if not members:
            raise ValueError(f"No */obsmat.txt recordings found in {path}")
        for member_number, member in enumerate(members, 1):
            recording = PurePosixPath(member.name).parent.name
            print(f"[{member_number}/{len(members)}] {path.name}: {recording}", flush=True)
            rows, native_stride, first_frame = _canonical_rows(_read_obsmat(archive, member))
            output = dataset_dir / "converted" / f"{recording}.traj.parquet"
            scene_id=recording.removeprefix("seq_")
            _write_recording(
                rows,
                output,
                canonical_metadata(dataset="eth",scene_id=scene_id,scene_sequence=recording,recording=recording,extra={
                    "source_to_local_3x3": json.dumps(
                        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
                    ),
                    "heading_transform": "Add the planar rotation from source_to_local_3x3 and wrap to [-pi, pi]; this ETH transform is identity.",
                    "source_archive": path.name,
                    "source_member": member.name,
                    "source_first_frame": str(first_frame),
                    "source_frame_stride": str(native_stride),
                    "annotation_hz": str(ANNOTATION_HZ),
                }),
            )
            _write_provenance_sidecar(output, provenance if provenance is not None else git_provenance(REPO_ROOT))
            outputs.append(output)
    return outputs


def convert_dataset(dataset_dir: Path) -> list[Path]:
    archives = raw_archives(dataset_dir)
    if not archives:
        raise FileNotFoundError(f"No .tgz or .tar.gz archives found in {dataset_dir / 'raw'}")
    outputs = []
    recordings: set[str] = set()
    # Resolved once so every sidecar in a run records the same commit/dirty snapshot.
    provenance = git_provenance(REPO_ROOT)
    for archive_number, archive in enumerate(archives, 1):
        print(f"[archive {archive_number}/{len(archives)}] {archive.name}", flush=True)
        for output in convert_archive(archive, dataset_dir, provenance):
            if output.name in recordings:
                raise ValueError(f"Recording appears in more than one archive: {output.name}")
            recordings.add(output.name)
            outputs.append(output)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert ETH/BIWI pedestrian data to canonical parquet")
    parser.add_argument("dataset_dir", type=Path, default=Path(__file__).resolve().parents[3]/"datasets/eth", nargs="?")
    args = parser.parse_args()
    for output in convert_dataset(args.dataset_dir):
        print(output)


if __name__ == "__main__":
    main()
