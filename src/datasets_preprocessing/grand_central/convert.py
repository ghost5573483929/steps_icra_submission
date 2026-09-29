"""Convert Grand Central Station annotations following pinned OpenTraj loader_gcs.py."""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import interp1d
from tqdm.auto import tqdm

SOURCE_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SOURCE_ROOT.parent
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from canonical import canonical_metadata, schema_with_metadata
from heading import central_difference_heading
from artifact_provenance import git_provenance

CONVERTER_VERSION = 1
REPO_ROOT = PROJECT_ROOT


def _write_provenance_sidecar(output: Path, provenance: dict[str, object]) -> None:
    payload = {
        "converter_version": CONVERTER_VERSION,
        "invocation": " ".join(sys.argv),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        **provenance,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(payload, indent=2) + "\n")


SOURCE_FPS = 25.0
INTERPOLATION_FRAME_STEP = 10
OPENTRAJ_COMMIT = "97ec1d5e1b579f26febeaa9083705952deac338a"
HOMOGRAPHY = np.asarray(
    [
        [4.97412897e-02, -4.24730883e-02, 7.25543911e01],
        [1.45017874e-01, -3.35678711e-03, 7.97920970e00],
        [1.36068797e-03, -4.98339188e-05, 1.00000000e00],
    ],
    dtype=float,
)
WORLD_SCALE = 0.8


def image_to_world(xy: np.ndarray) -> np.ndarray:
    homogeneous = np.c_[np.asarray(xy, dtype=float), np.ones(len(xy))]
    projected = (HOMOGRAPHY @ homogeneous.T).T
    return WORLD_SCALE * projected[:, :2] / projected[:, 2, None]


def read_annotation_directory(path: Path) -> list[tuple[int, np.ndarray, np.ndarray]]:
    """Return OpenTraj-compatible gap-split tracks as (id, frames, [x,y])."""
    files = sorted(item for item in path.iterdir() if item.is_file() and item.suffix.lower() == ".txt")
    if not files:
        raise FileNotFoundError(f"No Grand Central annotation .txt files in {path}")
    tracks: list[tuple[int, np.ndarray, np.ndarray]] = []
    next_id = 0
    for source in tqdm(files, desc="reading Grand Central annotations", unit="file"):
        tokens = source.read_text(encoding="utf-8").split()
        if len(tokens) % 3:
            raise ValueError(f"{source}: expected py px frame triples")
        values = np.asarray(tokens, dtype=float).reshape(-1, 3)
        py, px, frames_float = values.T
        if not (np.isfinite(px).all() and np.isfinite(py).all()):
            bad_row = int(np.flatnonzero(~(np.isfinite(px) & np.isfinite(py)))[0])
            raise ValueError(f"{source}: non-finite pixel coordinate (px/py) at annotation row {bad_row}")
        frames = frames_float.astype(np.int64)
        if not np.array_equal(frames_float, frames) or len(frames) < 1:
            raise ValueError(f"{source}: frame IDs must be integral")
        if np.any(np.diff(frames) <= 0):
            raise ValueError(f"{source}: frame IDs must be strictly increasing")
        # OpenTraj starts a new identity whenever the annotation gap exceeds 20
        # source frames.
        for segment in np.split(np.arange(len(frames)), np.flatnonzero(np.diff(frames) > 20) + 1):
            if len(segment) < 2:
                continue
            next_id += 1
            tracks.append((next_id, frames[segment], np.c_[px[segment], py[segment]]))
    return tracks


def interpolate_track(frames: np.ndarray, pixel_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # Upstream interval: include first, exclude final, step ten frames.
    sampled = np.arange(int(frames[0]), int(frames[-1]), INTERPOLATION_FRAME_STEP, dtype=np.int64)
    if not len(sampled):
        return sampled, np.empty((0, 2))
    x = interp1d(frames, pixel_xy[:, 0], kind="linear")(sampled)
    y = interp1d(frames, pixel_xy[:, 1], kind="linear")(sampled)
    return sampled, image_to_world(np.c_[x, y])


def convert_dataset(dataset_dir: Path) -> Path:
    candidates = [
        dataset_dir / "raw" / "Annotation",
        dataset_dir / "raw" / "extracted" / "Annotation",
        dataset_dir / "raw",
    ]
    annotation_dir = next((path for path in candidates if path.exists() and any(path.glob("*.txt"))), None)
    if annotation_dir is None:
        raise FileNotFoundError(f"Place Grand Central Annotation/*.txt under {dataset_dir / 'raw'}")
    rows = []
    phases = set()
    tracks = read_annotation_directory(annotation_dir)
    for track_id, frames, pixels in tqdm(tracks, desc="interpolating Grand Central tracks", unit="track"):
        sampled, world = interpolate_track(frames, pixels)
        if not len(sampled):
            continue
        phases.add(int(sampled[0] % INTERPOLATION_FRAME_STEP))
        # A mixed interpolation phase cannot be represented on one exact 2.5 Hz
        # lattice, so it is rejected rather than resampled.
        for frame, point in zip(sampled, world):
            rows.append((track_id, int(frame), float(point[0]), float(point[1])))
    if len(phases) != 1:
        raise ValueError(f"Grand Central tracks have incompatible 2.5 Hz phases: {sorted(phases)}")
    phase = next(iter(phases))
    converted = []
    for track_id, frame, x, y in rows:
        converted.append((track_id, (frame - phase) // 10, x, y))
    converted.sort(key=lambda row: (row[1], row[0]))
    by_track: dict[int, list[int]] = {}
    for index, row in enumerate(converted):
        by_track.setdefault(row[0], []).append(index)
    heading = np.full(len(converted), np.nan, dtype=np.float32)
    # Tracks are contiguous on the 2.5 Hz lattice, so the shared central-difference
    # heading convention applies directly.
    for indices in by_track.values():
        xy = np.asarray([[converted[index][2], converted[index][3]] for index in indices])
        heading[indices] = central_difference_heading(xy)
    metadata = canonical_metadata(
        dataset="grand_central", scene_id="grand_central_station",
        scene_sequence="grand_central", recording="grand_central",
        extra={
            "source_file": str(annotation_dir), "source_video_hz": SOURCE_FPS,
            "interpolation": "OpenTraj track-local linear interpolation every 10 source frames; final frame excluded",
            "gap_split_threshold_source_frames": 20,
            "source_to_world_homography": json.dumps(HOMOGRAPHY.tolist()),
            "world_scale": WORLD_SCALE,
            "opentraj_reference": f"crowdbotp/OpenTraj@{OPENTRAJ_COMMIT}:toolkit/loaders/loader_gcs.py",
        },
    )
    schema = schema_with_metadata(metadata)
    table = pa.Table.from_arrays(
        [
            pa.array([row[0] for row in converted], pa.int32()),
            pa.array([row[1] for row in converted], pa.int32()),
            pa.array([row[2] for row in converted], pa.float32()),
            pa.array([row[3] for row in converted], pa.float32()),
            pa.array(heading, pa.float32()),
        ],
        schema=schema,
    )
    output = dataset_dir / "converted" / "grand_central.traj.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)
    _write_provenance_sidecar(output, git_provenance(REPO_ROOT))
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert Grand Central Station annotations")
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=PROJECT_ROOT / "datasets" / "grand_central")
    args = parser.parse_args()
    print(convert_dataset(args.dataset_dir))


if __name__ == "__main__":
    main()
