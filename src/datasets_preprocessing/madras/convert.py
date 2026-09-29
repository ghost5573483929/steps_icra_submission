from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from canonical import CANONICAL_HZ, CANONICAL_SCHEMA as OUTPUT_SCHEMA, canonical_metadata, schema_with_metadata
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



def trajectory_files(dataset_dir: Path) -> list[Path]:
    root = dataset_dir / "raw" / "Data_Madras"
    top = sorted((root / "TopView_trajectories").glob("TopView_*.txt"))
    large = sorted(path for path in (root / "LargeView_trajectories").glob("LargeView_*.txt") if "density" not in path.stem.lower())
    files = top + large
    if not files: raise FileNotFoundError(f"No MADRAS trajectory text files under {root}")
    return files


def _read_header(path: Path) -> list[str]:
    header=[]
    with path.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.startswith("#"): break
            header.append(line.rstrip())
    return header


def _source_rate(path: Path, header: list[str]) -> int:
    if path.stem.startswith("TopView_"): return 30
    if any("final output" in line and "10 fps" in line for line in header): return 10
    raise ValueError(f"Cannot establish trajectory sample rate from {path}")


def read_rows(path: Path) -> tuple[list[tuple[int,int,float,float]], dict[str,str]]:
    header = _read_header(path); source_hz = _source_rate(path, header); raw=[]
    with path.open(encoding="utf-8-sig") as stream:
        for line_number,line in enumerate(stream,1):
            if not line.strip() or line.startswith("#"): continue
            fields=line.split(); expected=9 if source_hz==30 else 8
            if len(fields)!=expected: raise ValueError(f"{path}:{line_number}: expected {expected} fields, found {len(fields)}")
            try:
                local_id=int(fields[0]); frame=int(fields[1]); x=float(fields[2]); y=float(fields[3])
                track_id=int(fields[5]) if source_hz==30 else local_id
            except ValueError as error: raise ValueError(f"{path}:{line_number}: malformed trajectory row") from error
            if not math.isfinite(x) or not math.isfinite(y): raise ValueError(f"{path}:{line_number}: non-finite position")
            raw.append((track_id,frame,x,y))
    if not raw: raise ValueError(f"No trajectory rows in {path}")
    first_frame=min(row[1] for row in raw); factor=int(round(source_hz/CANONICAL_HZ))
    if not np.isclose(source_hz/factor,CANONICAL_HZ): raise ValueError(f"{source_hz} Hz is not divisible by canonical {CANONICAL_HZ} Hz")
    sampled=[]
    for track_id,frame,x,y in raw:
        relative=frame-first_frame
        if relative%factor: continue
        sampled.append((track_id,relative//factor,x,y))
    grouped=defaultdict(list)
    for track_id,step,x,y in sampled: grouped[(track_id,step)].append((x,y))
    rows=[(track_id,step,float(np.mean([p[0] for p in points])),float(np.mean([p[1] for p in points]))) for (track_id,step),points in grouped.items()]
    duplicate_rows=len(sampled)-len(rows)
    return rows,{"source_hz":str(source_hz),"source_first_frame":str(first_frame),"source_rows":str(len(raw)),"retained_rows":str(len(rows)),"duplicate_time_rows_averaged":str(duplicate_rows)}


def _headings(rows: list[tuple[int,int,float,float]]) -> dict[tuple[int,int],float|None]:
    tracks=defaultdict(list)
    for track_id,step,x,y in rows: tracks[track_id].append((step,x,y))
    result={}
    for track_id,values in tracks.items():
        values.sort(); coordinates=np.asarray([(x,y) for _,x,y in values]); steps=np.asarray([step for step,_,_ in values])
        for index,(step,_x,_y) in enumerate(values):
            left=max(0,index-1); right=min(len(values)-1,index+1)
            contiguous=right!=left and steps[right]-steps[left] <= 2*(right-left)
            difference=coordinates[right]-coordinates[left] if contiguous else np.zeros(2)
            result[(track_id,step)]=float(math.atan2(difference[1],difference[0])) if np.linalg.norm(difference)>1e-8 else None
    return result


def convert_file(path: Path, output: Path, provenance: dict[str, object] | None = None) -> Path:
    rows,details=read_rows(path); rows.sort(key=lambda row:(row[1],row[0])); keys=[(r[0],r[1]) for r in rows]
    if len(keys)!=len(set(keys)): raise ValueError(f"Duplicate (track_id, step) in {path}")
    heading=_headings(rows); scene_id="place_des_terreaux"
    metadata=canonical_metadata(dataset="madras",scene_id=scene_id,scene_sequence=path.stem,recording=path.stem,extra={
        "source_file":path.name,
        "source_to_local_3x3":json.dumps([[1.,0.,0.],[0.,1.,0.],[0.,0.,1.]]),
        "heading_transform":"Heading is derived from local metric x/y motion. Any future planar source rotation must also rotate heading.",
        "coordinate_system":"published camera-local x/m,y/m; metres","identity":"TopView uses id_global; LargeView uses id",
        "storage_hz":str(CANONICAL_HZ),"populated_step_stride":"1",**details,
    })
    schema=schema_with_metadata(metadata)
    table=pa.Table.from_arrays([
        pa.array([r[0] for r in rows],pa.int32()),pa.array([r[1] for r in rows],pa.int32()),
        pa.array([r[2] for r in rows],pa.float32()),pa.array([r[3] for r in rows],pa.float32()),
        pa.array([heading[(r[0],r[1])] for r in rows],pa.float32()),
    ],schema=schema)
    output.parent.mkdir(parents=True,exist_ok=True); temporary=output.with_suffix(".tmp.parquet")
    pq.write_table(table,temporary,compression="zstd"); temporary.replace(output)
    _write_provenance_sidecar(output, provenance if provenance is not None else git_provenance(REPO_ROOT))
    return output


def convert_dataset(dataset_dir: Path) -> list[Path]:
    files=trajectory_files(dataset_dir); outputs=[]
    # Resolved once so every sidecar in a run records the same commit/dirty snapshot.
    provenance=git_provenance(REPO_ROOT)
    for number,path in enumerate(files,1):
        outputs.append(convert_file(path,dataset_dir/"converted"/f"{path.stem}.traj.parquet",provenance))
        print(f"[{number}/{len(files)}] converted {path.name}",flush=True)
    return outputs


def main()->None:
    parser=argparse.ArgumentParser(description="Convert MADRAS metric trajectories to canonical Parquet")
    parser.add_argument("dataset_dir",type=Path,nargs="?",default=Path(__file__).resolve().parents[3]/"datasets/madras"); args=parser.parse_args()
    for output in convert_dataset(args.dataset_dir): print(output)


if __name__=="__main__": main()
