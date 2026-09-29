from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pyarrow.parquet as pq
from torch.utils.data import ConcatDataset,DataLoader

from config import Config,DatasetSpec
from data import SceneDataset,scene_collate,seed_worker

ROOT=Path(__file__).resolve().parents[2]


def protocol()->dict:
    return json.loads((ROOT/"configs/protocols/eth_ucy_five_fold.json").read_text())


def _spec_for(recording:str)->DatasetSpec:
    name="eth" if recording.startswith("seq_") else "ucy"
    return DatasetSpec(name,ROOT/"datasets"/name)


def _rows(spec:DatasetSpec,data_seed:int=42)->list[dict]:
    return pq.read_table(spec.path/"indices"/f"scenes-seed{data_seed}.parquet").to_pylist()


def available_recordings(data_seed:int=42)->set[str]:
    available=set()
    for dataset_name in ("eth","ucy"):
        spec=DatasetSpec(dataset_name,ROOT/"datasets"/dataset_name)
        available.update(row["recording"] for row in _rows(spec,data_seed))
    return available


def _scene_id_by_recording(data_seed:int=42)->dict[str,str]:
    """Map each recording to its physical ``scene_id`` location."""
    mapping:dict[str,str]={}
    for dataset_name in ("eth","ucy"):
        spec=DatasetSpec(dataset_name,ROOT/"datasets"/dataset_name)
        for row in _rows(spec,data_seed):
            mapping[row["recording"]]=row["scene_id"]
    return mapping


def fold_recording_partitions(fold:str,data_seed:int=42,validation_mode:str="none")->dict[str,set[str]]:
    """Return recording-disjoint fold membership.

    ``none`` uses every non-test location for fitting. ``grouped`` reserves one
    complete non-test scene sequence for validation, deterministically.
    """
    definition=protocol();folds=definition["folds"]
    if fold not in folds:raise ValueError(f"Unknown ETH/UCY fold {fold!r}")
    heldout=set(folds[fold])
    fine_tuning_pool=set(definition["auxiliary_train_recordings"])
    for name,recordings in folds.items():
        if name!=fold:fine_tuning_pool.update(recordings)
    available=available_recordings(data_seed)
    missing=(heldout|fine_tuning_pool)-available
    if missing:raise RuntimeError(f"Fold {fold} is missing canonical recordings: {sorted(missing)}")
    if validation_mode not in {"none","grouped"}:raise ValueError(f"Unknown validation mode {validation_mode!r}")
    ordered=sorted(fine_tuning_pool,key=lambda value:hashlib.sha256(f"{data_seed}:{fold}:{value}".encode()).digest())
    if validation_mode=="none":
        validation=set();training=set(ordered)
    else:
        if len(ordered)<2:raise RuntimeError(f"Fold {fold} has too few eligible training sequences for grouped validation")
        # Reserve the whole physical location (scene_id), since two recordings of one
        # location must not land on opposite sides of train/val.
        scene_id_by_recording=_scene_id_by_recording(data_seed)
        selected_scene_id=scene_id_by_recording[ordered[0]]
        validation={recording for recording in fine_tuning_pool if scene_id_by_recording[recording]==selected_scene_id}
        training=fine_tuning_pool-validation
        if not training:raise RuntimeError(f"Fold {fold} has no eligible training sequences left after reserving location {selected_scene_id!r} for grouped validation")
    return {"train":training,"val":validation,"test":heldout,"fine_tuning_pool":fine_tuning_pool}


def _selected_dataset(spec:DatasetSpec,role:str,recordings:set[str],cfg:Config)->SceneDataset:
    dataset=SceneDataset(spec,"train" if role=="train" else "val",cfg,include_rejected=True)
    rows=[row for row in _rows(spec,cfg.seeds.data_seed) if row["recording"] in recordings]
    if role=="train":
        rows=[row for row in rows if row.get("accepted_if_train",row.get("accepted",True)) and row.get("track_kept",True)]
    else:
        separation=int(cfg.eval_stride or 20)*dataset.canonical_stride
        kept=[];last={}
        for row in sorted(rows,key=lambda value:(value["recording"],value["primary_track_id"],value["start_step"])):
            key=(row["recording"],row["primary_track_id"])
            if key not in last or row["start_step"]-last[key]>=separation:
                kept.append(row);last[key]=row["start_step"]
        rows=kept
    dataset.rows=rows;dataset.split=role
    return dataset


def fold_datasets(fold:str,cfg:Config,validation_mode:str="none"):
    partitions=fold_recording_partitions(fold,cfg.seeds.data_seed,validation_mode)
    training,validation,heldout=partitions["train"],partitions["val"],partitions["test"]
    train_parts=[];val_parts=[];test_parts=[]
    for dataset_name in ("eth","ucy"):
        spec=DatasetSpec(dataset_name,ROOT/"datasets"/dataset_name)
        available={row["recording"] for row in _rows(spec,cfg.seeds.data_seed)}
        train_here=training&available;val_here=validation&available;test_here=heldout&available
        if train_here:
            train_parts.append(_selected_dataset(spec,"train",train_here,cfg))
        if val_here:val_parts.append(_selected_dataset(spec,"val",val_here,cfg))
        if test_here:test_parts.append(_selected_dataset(spec,"test",test_here,cfg))
    if not train_parts or not test_parts or (validation_mode=="grouped" and not val_parts):raise RuntimeError(f"Incomplete fold {fold}")
    validation_dataset=ConcatDataset(val_parts) if val_parts else None
    return ConcatDataset(train_parts),validation_dataset,ConcatDataset(test_parts)


def fold_loaders(fold:str,cfg:Config,validation_mode:str="none"):
    import torch
    train,val,test=fold_datasets(fold,cfg,validation_mode)
    generator=torch.Generator().manual_seed(cfg.seeds.model_seed)
    common={"batch_size":cfg.batch_size,"num_workers":cfg.num_workers,"collate_fn":scene_collate,"worker_init_fn":seed_worker}
    return (
        DataLoader(train,shuffle=True,generator=generator,**common),
        DataLoader(val,shuffle=False,**common) if val is not None else None,
        DataLoader(test,shuffle=False,**common),
    )
