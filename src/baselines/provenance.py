from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from torch.utils.data import ConcatDataset,Subset

ROOT=Path(__file__).resolve().parents[2]


def _rows(dataset):
    if hasattr(dataset,"source"):
        rows=list(_rows(dataset.source));limit=getattr(dataset,"limit",len(rows));yield from rows[:limit];return
    if isinstance(dataset,ConcatDataset):
        for child in dataset.datasets:yield from _rows(child)
        return
    if isinstance(dataset,Subset):
        rows=list(_rows(dataset.dataset))
        for index in dataset.indices:yield rows[int(index)]
        return
    rows=getattr(dataset,"rows",None);spec=getattr(dataset,"spec",None)
    if rows is None:return
    for row in rows:
        yield {"dataset":getattr(spec,"name",None),"split":getattr(dataset,"split",None),
               "scene_sequence":row.get("scene_sequence"),
               "recording":row.get("recording"),"start_step":int(row.get("start_step",-1)),
               "primary_track_id":str(row.get("primary_track_id","")),"scene_tag":int(row.get("scene_tag",0)),
               "accepted":bool(row.get("accepted",True)),"track_kept":bool(row.get("track_kept",True)),
               "eval_kept":bool(row.get("eval_kept",True))}


def snapshot_dataset(run_dir:Path,role:str,dataset)->Path:
    records=list(_rows(dataset));directory=Path(run_dir)/"datasets";directory.mkdir(parents=True,exist_ok=True)
    path=directory/f"{role}.parquet"
    pq.write_table(pa.Table.from_pylist(records),path,compression="zstd")
    return path


def snapshot_environment(run_dir:Path,role:str,env)->Path:
    records=[]
    for scene in env.scenes:
        records.append({"scene":scene.name,"timesteps":int(scene.timesteps),"n_nodes":len(scene.nodes),
                        "first_node":scene.nodes[0].id if scene.nodes else None})
    directory=Path(run_dir)/"datasets";directory.mkdir(parents=True,exist_ok=True);path=directory/f"{role}.parquet"
    pq.write_table(pa.Table.from_pylist(records),path,compression="zstd");return path




def snapshot_split_inputs(run_dir:Path)->None:
    directory=Path(run_dir)/"datasets";directory.mkdir(parents=True,exist_ok=True)
    for source in (ROOT/"configs/task.json",ROOT/"configs/protocols/eth_ucy_five_fold.json"):
        if source.exists():shutil.copy2(source,directory/source.name)
    hashes={}
    for path in sorted((ROOT/"datasets").glob("*/indices/scenes-seed*.parquet")):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path.relative_to(ROOT))]=f"sha256:{digest}"
    for path in sorted((ROOT/"datasets").glob("*/split_manifests/*.json")):
        digest=hashlib.sha256(path.read_bytes()).hexdigest();hashes[str(path.relative_to(ROOT))]=f"sha256:{digest}"
    (directory/"index_hashes.json").write_text(json.dumps(hashes,indent=2,sort_keys=True)+"\n")
