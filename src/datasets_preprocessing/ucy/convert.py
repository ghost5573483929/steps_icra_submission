from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.interpolate import interp1d
from canonical import CANONICAL_HZ, CANONICAL_SCHEMA as OUTPUT_SCHEMA, canonical_metadata, schema_with_metadata
from heading import central_difference_heading
from artifact_provenance import git_provenance

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


VIDEO_HZ = 25.0
ANNOTATION_FRAME_STRIDE = 10
HOMOGRAPHIES = json.loads((Path(__file__).with_name("homographies.json")).read_text())
FORECASTING_BENCHMARK_SEQUENCES = (
    "crowds_zara01",
    "crowds_zara02",
    "students001",
    "students003",
)
# Forecasting benchmark membership only; publication characterization uses the
# larger PUBLICATION_SEQUENCES below.
BENCHMARK_SEQUENCES = FORECASTING_BENCHMARK_SEQUENCES
PUBLICATION_SEQUENCES = (
    "arxiepiskopi1",
    "crowds_zara01",
    "crowds_zara02",
    "crowds_zara03",
    "students001",
    "students003",
    "uni_examples",
)
SCENE_IDS = {
    "arxiepiskopi1": "arxiepiskopi",
    "crowds_zara01": "zara1",
    "crowds_zara02": "zara2",
    "crowds_zara03": "zara3",
    "students001": "univ",
    "students003": "univ",
    "uni_examples": "univ",
}


@dataclass(frozen=True)
class ControlPoint:
    x: float
    y: float
    frame: int
    gaze_deg: float


def _archive_paths(dataset_dir: Path) -> list[Path]:
    return sorted((dataset_dir / "raw").rglob("data_*.rar"))


def _extract_archives(dataset_dir: Path) -> Path:
    output = dataset_dir / "raw" / "extracted"
    output.mkdir(parents=True, exist_ok=True)
    archives = _archive_paths(dataset_dir)
    if not archives and not list(output.glob("*.vsp")):
        raise FileNotFoundError(f"No UCY .vsp files or data_*.rar archives under {dataset_dir / 'raw'}")
    extractor = shutil.which("bsdtar")
    for archive in archives:
        # Unconditional re-extraction makes a partial earlier extraction self-healing.
        if extractor is None:
            raise RuntimeError("bsdtar is required to extract the supplied UCY RAR archives")
        subprocess.run([extractor, "-xf", str(archive), "-C", str(output)], check=True)
    return output


def parse_vsp(path: Path) -> list[list[ControlPoint]]:
    splines: list[list[ControlPoint]] = []
    current: list[ControlPoint] = []
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        if "- Num of control points" in line or "- the number of splines" in line:
            if current:
                splines.append(current)
                current = []
            continue
        # Annotation comments are introduced by a spaced dash. Negative numbers
        # therefore remain intact.
        content = line.split(" - ", 1)[0]
        fields = content.split()
        if len(fields) != 4:
            continue
        try:
            x, y, frame_value, gaze = map(float, fields)
        except ValueError as error:
            raise ValueError(f"{path}:{line_number}: malformed control point") from error
        if not frame_value.is_integer() or not all(math.isfinite(v) for v in (x, y, frame_value, gaze)):
            raise ValueError(f"{path}:{line_number}: control point values must be finite and frame integral")
        current.append(ControlPoint(x, y, int(frame_value), gaze))
    if current:
        splines.append(current)
    if not splines:
        raise ValueError(f"No splines found in {path}")
    for index, spline in enumerate(splines):
        frames = [point.frame for point in spline]
        if len(spline) < 2 or any(right <= left for left, right in zip(frames, frames[1:])):
            raise ValueError(f"{path}: spline {index} needs at least two strictly increasing frames")
    return splines


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


def interpolate_spline(spline: list[ControlPoint], track_id: int,
                       homography: np.ndarray | None = None) -> list[tuple[int, int, float, float, float]]:
    frames = np.asarray([point.frame for point in spline], dtype=float)
    # TrajNet++ reader convention: exclude the endpoints, sample video-frame multiples of ten.
    sampled_frames = np.arange(int(frames.min()) // 10 * 10 + 10, int(frames.max()), 10)
    if not len(sampled_frames):
        return []
    # Linear, per the UCY toolkit contract: a global cubic overshoots across long
    # annotation gaps (students001 track 165 has a 1343-frame gap).
    source_xy=np.asarray([[p.x,p.y,1.] for p in spline],dtype=float)
    transform=np.eye(3) if homography is None else np.asarray(homography,dtype=float)
    world=(transform@source_xy.T).T; world=world[:,:2]/world[:,2,None]
    x = interp1d(frames, world[:,0], kind="linear")(sampled_frames)
    y = interp1d(frames, world[:,1], kind="linear")(sampled_frames)
    points=np.c_[x,y]
    # Samples are contiguous at every tenth video frame, so the shared
    # central-difference heading convention applies directly.
    heading = central_difference_heading(points)
    steps = np.rint(sampled_frames / VIDEO_HZ * CANONICAL_HZ).astype(np.int32)
    return [(int(track_id), int(step), float(px), float(py), float(ph)) for step, px, py, ph in zip(steps, x, y, heading)]


def convert_vsp(path: Path, output: Path, provenance: dict[str, object] | None = None) -> Path:
    splines = parse_vsp(path)
    # Arxiepiskopi control points are already metric-scale (roughly -20..20 m);
    # Zara/student files are centered image pixels in the hundreds.
    homography=np.asarray(HOMOGRAPHIES[path.stem],dtype=float)
    rows = [row for track_id, spline in enumerate(splines) for row in interpolate_spline(spline, track_id,homography)]
    rows.sort(key=lambda row: (row[1], row[0]))
    if not rows:
        raise ValueError(f"Interpolation emitted no rows for {path}")
    keys = [(row[0], row[1]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError(f"Duplicate (track_id, step) after interpolation in {path}")
    scene_id = SCENE_IDS.get(path.stem, path.stem)
    metadata = canonical_metadata(dataset="ucy",scene_id=scene_id,scene_sequence=path.stem,recording=path.stem,extra={
        "source_to_local_3x3": json.dumps(homography.tolist()),
        "heading_transform": "Heading is derived from transformed metric-world motion. Any future planar source rotation must also rotate heading.",
        "source_file": path.name, "source_video_hz": str(VIDEO_HZ),
        "annotation_hz": str(VIDEO_HZ / ANNOTATION_FRAME_STRIDE),
        "interpolation": "linear at every tenth video frame; gaze linear after angle unwrapping",
        "coordinate_scale": "OpenTraj homography to metric world coordinates",
        "opentraj_homography_reference":"crowdbotp/OpenTraj@97ec1d5e1b579f26febeaa9083705952deac338a",
    })
    schema = schema_with_metadata(metadata)
    table = pa.Table.from_arrays([
        pa.array([r[0] for r in rows], type=pa.int32()), pa.array([r[1] for r in rows], type=pa.int32()),
        pa.array([r[2] for r in rows], type=pa.float32()), pa.array([r[3] for r in rows], type=pa.float32()),
        pa.array([r[4] for r in rows], type=pa.float32()),
    ], schema=schema)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(".tmp.parquet")
    pq.write_table(table, temporary, compression="zstd")
    temporary.replace(output)
    _write_provenance_sidecar(output, provenance if provenance is not None else git_provenance(REPO_ROOT))
    return output


def convert_dataset(dataset_dir: Path) -> list[Path]:
    extracted = _extract_archives(dataset_dir)
    available = {path.stem: path for path in extracted.glob("*.vsp")}
    missing = sorted(set(PUBLICATION_SEQUENCES) - available.keys())
    if missing:
        raise FileNotFoundError(
            f"Missing required UCY publication sequences in {extracted}: {missing}"
        )
    sources = [available[name] for name in PUBLICATION_SEQUENCES]
    outputs = []
    # Resolved once so every sidecar in a run records the same commit/dirty snapshot.
    provenance = git_provenance(REPO_ROOT)
    for number, path in enumerate(sources, 1):
        outputs.append(convert_vsp(path, dataset_dir / "converted" / f"{path.stem}.traj.parquet", provenance))
        print(f"[{number}/{len(sources)}] converted {path.name}", flush=True)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert UCY Crowds-by-Example splines to canonical Parquet")
    parser.add_argument("dataset_dir", type=Path, nargs="?", default=Path(__file__).resolve().parents[3]/"datasets/ucy")
    args = parser.parse_args()
    for output in convert_dataset(args.dataset_dir):
        print(output)


if __name__ == "__main__":
    main()
