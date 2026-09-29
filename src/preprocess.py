from __future__ import annotations

import argparse
import json
import math
import multiprocessing
import os
import sqlite3
import time
from collections import Counter
from concurrent.futures import Future, ProcessPoolExecutor
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from categorization import get_type, scene_rng
from config import SPLIT_STRATEGIES, Config, DatasetSpec, SeedConfig, scoped_task_config, stable_config_hash
from data import ROOT, _analysis_index_path, _index_path, _read_json, _recordings, _scene_arrays, create_splits, prepare_splits
from manifests import steps_subset, manifest_name, write_subset_manifest
from splits import assign_split
from canonical import CANONICAL_HZ, recording_identity, validate_canonical_file
from dataset_paths import MANIFEST_ROOT, PAPER_STATS_ROOT
from split_manifests import update_sample_counts, write_manifest
from artifact_provenance import git_provenance


INDEX_SCHEMA = pa.schema(
    [
        ("dataset", pa.string()),
        ("scene_id", pa.string()),
        ("scene_sequence", pa.string()),
        ("recording", pa.string()),
        ("start_step", pa.int32()),
        ("observation_end_step", pa.int32()),
        ("future_end_step", pa.int32()),
        ("target_agent_id", pa.int64()),
        ("primary_track_id", pa.int64()),
        ("split", pa.string()),
        ("scene_tag", pa.int8()),
        ("sub_tags", pa.list_(pa.int8())),
        ("accepted_if_train", pa.bool_()),
        ("accepted", pa.bool_()),
        ("track_kept", pa.bool_()),
        ("eval_kept", pa.bool_()),
    ]
)


class CategoryCache:
    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path, timeout=60.0)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA busy_timeout=60000")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS categories (scene_key TEXT PRIMARY KEY, tag INTEGER NOT NULL, sub_tags TEXT NOT NULL)"
        )
        self.pending = 0
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> tuple[int, list[int]] | None:
        row = self.connection.execute(
            "SELECT tag, sub_tags FROM categories WHERE scene_key = ?", (key,)
        ).fetchone()
        if row:
            self.hits += 1
            return int(row[0]), json.loads(row[1])
        self.misses += 1
        return None

    def put(self, key: str, tag: int, sub_tags: list[int]) -> None:
        self.connection.execute(
            "INSERT OR REPLACE INTO categories(scene_key, tag, sub_tags) VALUES (?, ?, ?)",
            (key, tag, json.dumps(sub_tags)),
        )
        self.pending += 1
        if self.pending >= 1000:
            self.connection.commit(); self.pending = 0

    def close(self) -> None:
        self.connection.commit(); self.connection.close()


def auto_chunk_stride(length_median: int, chunk_size: int, w_target: float) -> tuple[int, int]:
    maximum = max(2, length_median - chunk_size + 1)
    if w_target <= 1:
        stride = maximum
    else:
        stride = int(np.clip(np.ceil((length_median - chunk_size) / (w_target - 1)), 2, maximum))
    realized = max(1, (length_median - chunk_size) // stride + 1)
    return stride, realized


def _track_segments(track_ids: np.ndarray, steps: np.ndarray, canonical_stride: int):
    order = np.lexsort((steps, track_ids))
    ids, ordered_steps = track_ids[order], steps[order]
    for indices in np.split(np.arange(len(order)), np.flatnonzero(np.diff(ids)) + 1):
        track_id = int(ids[indices[0]])
        sampled = np.unique(ordered_steps[indices])
        sampled = sampled[sampled % canonical_stride == 0]
        gaps = np.flatnonzero(np.diff(sampled) != canonical_stride) + 1
        for segment in np.split(sampled, gaps):
            if len(segment):
                yield track_id, segment


def measure_tracks(files: list[Path], canonical_stride: int, storage_hz: float) -> tuple[list[int], int, float]:
    lengths, tracks, hours = [], set(), 0.0
    for number, path in enumerate(files, 1):
        table = pq.read_table(path, columns=["track_id", "step"])
        track_values=table["track_id"].combine_chunks().to_numpy()
        step_values=table["step"].combine_chunks().to_numpy()
        if len(step_values): hours += float(step_values.max()-step_values.min())/storage_hz/3600.0
        for _track_id, segment in _track_segments(
            track_values,
            step_values,
            canonical_stride,
        ):
            lengths.append(len(segment)); tracks.add((path.name, _track_id))
        print(f"measured tracks {number}/{len(files)}: {path.name}", flush=True)
    return lengths, len(tracks), hours


def cached_track_measurements(dataset_dir: Path, files: list[Path], canonical_stride: int, storage_hz: float) -> tuple[list[int], int, float]:
    revision=stable_config_hash({
        "canonical_stride":canonical_stride,
        "storage_hz":storage_hz,
        "files":[(path.name,path.stat().st_size,path.stat().st_mtime_ns) for path in files],
    })
    cache_path=dataset_dir/"indices"/f"track-measurements-stride{canonical_stride}.npz"
    if cache_path.exists():
        cached=np.load(cache_path,allow_pickle=False)
        if str(cached["revision"].item())==revision:
            print(f"loaded track measurements: {cache_path}",flush=True)
            return cached["lengths"].astype(int).tolist(),int(cached["tracks"].item()),float(cached["hours"].item())
    lengths,tracks,hours=measure_tracks(files,canonical_stride,storage_hz)
    cache_path.parent.mkdir(parents=True,exist_ok=True)
    temporary=cache_path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary,revision=revision,lengths=np.asarray(lengths,dtype=np.int32),tracks=tracks,hours=hours)
    temporary.replace(cache_path)
    return lengths,tracks,hours


def preprocess_dataset(spec: DatasetSpec, cfg: Config, workers: int = 1) -> tuple[Path, dict]:
    task = _read_json(cfg.task_path)
    obs_len, pred_len = int(task["history_steps"]), int(task["future_steps"])
    dataset_task = task.get("datasets",{}).get(spec.name,{})
    chunk_size = obs_len + pred_len
    canonical_stride = int(dataset_task.get("canonical_stride", task["canonical_stride"]))
    files = _recordings(spec.path)
    if not files: raise FileNotFoundError(f"No converted recordings for {spec.name}")
    identities = {path.name.removesuffix(".traj.parquet"): validate_canonical_file(path) for path in files}
    prepare_splits(spec.path, cfg.seeds.data_seed, cfg.split_strategy, cfg.held_out_scene)
    split_config = create_splits(spec.path, cfg.seeds.data_seed, cfg.split_strategy, cfg.held_out_scene)
    storage_hz=CANONICAL_HZ
    lengths, raw_track_segments, hours = cached_track_measurements(spec.path,files,canonical_stride,storage_hz)
    usable_lengths = [length for length in lengths if length >= chunk_size]
    if not usable_lengths: raise ValueError(f"{spec.name} has no usable {chunk_size}-step tracks")
    l_med = int(np.median(usable_lengths))
    if "chunk_stride" not in dataset_task:
        raise KeyError(
            f"task.json datasets.{spec.name} is missing required key 'chunk_stride'"
        )
    configured_stride = dataset_task["chunk_stride"]
    w_target = float(dataset_task.get("w_target", task.get("w_target", cfg.sizing.w_target)))
    if configured_stride == "auto": chunk_stride, w_d = auto_chunk_stride(l_med, chunk_size, w_target)
    else:
        chunk_stride = int(configured_stride); w_d = max(1,(l_med-chunk_size)//chunk_stride+1)
    min_length = float(task.get("min_length",0.0)); window_span=(chunk_size-1)*canonical_stride
    before, after = Counter(), Counter(); rows=[]; usable_track_keys=set();split_candidate_count=0;neighbor_filter_pass_count=0
    legacy_category_hash=stable_config_hash({"category":cfg.category,"obs_len":obs_len,"pred_len":pred_len,"data_seed":cfg.seeds.data_seed})
    category_hash=stable_config_hash({"category":cfg.category,"obs_len":obs_len,"pred_len":pred_len,"data_seed":cfg.seeds.data_seed,"min_length":min_length,"canonical_stride":canonical_stride})
    category_cache=CategoryCache(spec.path/"indices"/f"categories-seed{cfg.seeds.data_seed}.sqlite")

    # One pool for the whole dataset. Reuse is safe because `get_type` carries no
    # process-local state: scene array, category config and seeded Generator are all
    # passed per call.
    executor=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context("spawn")) if workers>1 else None
    with executor if executor is not None else nullcontext():
        for file_number,path in enumerate(files,1):
            recording=path.name.removesuffix(".traj.parquet")
            identity=identities[recording]
            file_started=time.monotonic(); hits_before=category_cache.hits; misses_before=category_cache.misses
            print(f"starting {file_number}/{len(files)}: {path.name}", flush=True)
            file_stat=path.stat()
            file_revision=f"{file_stat.st_size}:{file_stat.st_mtime_ns}"
            table=pq.read_table(path,columns=["track_id","step","x","y"])
            ids=table["track_id"].combine_chunks().to_numpy(); steps=table["step"].combine_chunks().to_numpy()
            xs=table["x"].combine_chunks().to_numpy(); ys=table["y"].combine_chunks().to_numpy()
            unique_steps,counts=np.unique(steps,return_counts=True); count_by_step=dict(zip(unique_steps.tolist(),counts.tolist()))
            candidates=[]
            for primary,segment in _track_segments(ids,steps,canonical_stride):
                windows=max(0,len(segment)-chunk_size+1)
                for offset in range(0,windows,chunk_stride): candidates.append((int(segment[offset]),primary))
            if not candidates: continue
            first,last=min(s for s,_ in candidates),max(s for s,_ in candidates)
            file_rows=[]
            inflight: list[tuple[Future,tuple[int,int,int,str,str,str]]] = []

            def add_result(meta:tuple[int,int,int,str,str,str],result:tuple[int,list[int]]) -> None:
                candidate_number,start,primary,split,key,cache_key=meta
                tag,sub_tags=result
                category_cache.put(cache_key,tag,sub_tags)
                usable_track_keys.add(f"{recording}:{primary}")
                before[(split,tag)]+=1
                accepted_if_train=bool(scene_rng(cfg.seeds.data_seed,key+":accept").uniform() < cfg.category.acceptance[tag-1])
                accepted=accepted_if_train if split=="train" else True
                if accepted: after[(split,tag)]+=1
                file_rows.append((candidate_number,{"dataset":spec.name,"scene_id":identity.scene_id,"scene_sequence":identity.scene_sequence,"recording":recording,"start_step":start,"observation_end_step":start+(obs_len-1)*canonical_stride,"future_end_step":start+window_span,"target_agent_id":primary,"primary_track_id":primary,"split":split,"scene_tag":tag,"sub_tags":sub_tags,"accepted_if_train":accepted_if_train,"accepted":accepted,"track_kept":True,"eval_kept":True}))

            def resolve_oldest() -> None:
                future,meta=inflight.pop(0)
                add_result(meta,future.result())

            for candidate_number,(start,primary) in enumerate(candidates,1):
                split=assign_split(split_config,recording=recording,start=start,first=first,last=last,window_span=window_span)
                if split is None: continue
                split_candidate_count+=1
                expected=start+np.arange(chunk_size)*canonical_stride
                if sum(count_by_step.get(int(step),0) for step in expected) < 2.0*chunk_size: continue
                neighbor_filter_pass_count+=1
                key=f"{recording}:{start}:{primary}"
                cache_key=f"{category_hash}:{file_revision}:{key}"
                cached=category_cache.get(cache_key)
                # Legacy cache keys omitted min_length, so they are only equivalent
                # when the filter is disabled.
                if cached is None and min_length==0.0:
                    cached=category_cache.get(f"{legacy_category_hash}:{file_revision}:{key}")
                if cached is None:
                    lo=np.searchsorted(steps,start,side="left");hi=np.searchsorted(steps,start+window_span,side="right")
                    scene=_scene_arrays(ids[lo:hi],steps[lo:hi],xs[lo:hi],ys[lo:hi],start,obs_len,pred_len,canonical_stride,primary)
                    if np.linalg.norm(scene["xy"][-1,0]-scene["xy"][0,0]) <= min_length: continue
                    meta=(candidate_number,start,primary,split,key,cache_key)
                    if executor is None:
                        add_result(meta,get_type(scene["xy"],obs_len,pred_len,cfg.category,scene_rng(cfg.seeds.data_seed,key)))
                    else:
                        inflight.append((executor.submit(get_type,scene["xy"],obs_len,pred_len,cfg.category,scene_rng(cfg.seeds.data_seed,key)),meta))
                        if len(inflight)>=workers*4: resolve_oldest()
                else:
                    tag,sub_tags=cached
                    usable_track_keys.add(f"{recording}:{primary}")
                    before[(split,tag)]+=1
                    accepted_if_train=bool(scene_rng(cfg.seeds.data_seed,key+":accept").uniform() < cfg.category.acceptance[tag-1])
                    accepted=accepted_if_train if split=="train" else True
                    if accepted: after[(split,tag)]+=1
                    file_rows.append((candidate_number,{"dataset":spec.name,"scene_id":identity.scene_id,"scene_sequence":identity.scene_sequence,"recording":recording,"start_step":start,"observation_end_step":start+(obs_len-1)*canonical_stride,"future_end_step":start+window_span,"target_agent_id":primary,"primary_track_id":primary,"split":split,"scene_tag":tag,"sub_tags":sub_tags,"accepted_if_train":accepted_if_train,"accepted":accepted,"track_kept":True,"eval_kept":True}))
                if candidate_number == 1 or candidate_number % 1000 == 0:
                    elapsed=max(time.monotonic()-file_started,1e-6)
                    print(f"  {path.name}: {candidate_number}/{len(candidates)} windows, "
                          f"{candidate_number/elapsed:.1f}/s, cache hits={category_cache.hits-hits_before} "
                          f"misses={category_cache.misses-misses_before}, completed={len(file_rows)}, "
                          f"inflight={len(inflight)}, workers={workers}", flush=True)
            while inflight: resolve_oldest()
            file_rows.sort(key=lambda item:item[0]); rows.extend(row for _,row in file_rows)
            elapsed=max(time.monotonic()-file_started,1e-6)
            print(f"categorized {file_number}/{len(files)}: {path.name} ({len(rows)} kept total; "
                  f"{len(candidates)/elapsed:.1f} windows/s, cache hits={category_cache.hits-hits_before}, "
                  f"misses={category_cache.misses-misses_before})",flush=True)

    category_cache.close()
    print(f"postprocessing {len(rows):,} categorized windows", flush=True)
    track_keys=sorted({f"{row['recording']}:{row['primary_track_id']}" for row in rows if row["split"]=="train" and row["accepted"]})
    print(f"  found {len(track_keys):,} accepted training tracks", flush=True)
    eval_stride=cfg.eval_stride if cfg.eval_stride is not None else task.get("eval_stride")
    if eval_stride is not None:
        minimum_separation = int(eval_stride) * canonical_stride
        last_by_track: dict[tuple[str, int, str], int] = {}
        # Rows are already chronological within each recording/track, so no sort.
        for row in rows:
            if row["split"] == "train":
                continue
            key = (row["recording"], row["primary_track_id"], row["split"])
            if key not in last_by_track or row["start_step"] - last_by_track[key] >= minimum_separation:
                last_by_track[key] = row["start_step"]
            else:
                row["eval_kept"]=False
        print(f"  applied evaluation stride {eval_stride}", flush=True)
    n_target=cfg.sizing.n_target if cfg.sizing.n_target is not None else task.get("n_target")
    realized_train=sum(row["split"]=="train" and row["accepted"] for row in rows)
    p_d=min(1.0,n_target/realized_train) if n_target and realized_train else 1.0
    data_rng=np.random.default_rng(cfg.seeds.data_seed); keep_mask=data_rng.random(len(track_keys))<p_d
    kept_tracks={key for key,keep in zip(track_keys,keep_mask) if keep}
    for row in rows:
        row["track_kept"] = row["split"] != "train" or f"{row['recording']}:{row['primary_track_id']}" in kept_tracks
    print(
        f"  retained {len(kept_tracks):,}/{len(track_keys):,} training tracks "
        f"(p_d={p_d:.6f})",
        flush=True,
    )

    # A train budget subsets val/test in proportion to the configured split ratios
    # (12.5% of train for 80/10/10) so they do not stay unbounded.
    eval_targets={}
    if n_target:
        ratios=split_config.get("ratios",{"train":.8,"val":.1,"test":.1})
        train_ratio=float(ratios.get("train",.8))
        for split in ("val","test"):
            eval_targets[split]=int(round(float(n_target)*float(ratios.get(split,.1))/train_ratio))
            candidates=[i for i,row in enumerate(rows) if row["split"]==split and row["accepted"] and row["eval_kept"]]
            if len(candidates)>eval_targets[split]:
                split_rng=np.random.default_rng(cfg.seeds.data_seed+(1 if split=="val" else 2))
                selected=set(int(i) for i in split_rng.choice(candidates,eval_targets[split],replace=False))
                for i in candidates:
                    if i not in selected: rows[i]["eval_kept"]=False
            print(
                f"  {split}: retained at most {eval_targets[split]:,} evaluation windows",
                flush=True,
            )

    index_path=_index_path(spec,cfg.seeds.data_seed,cfg.split_strategy,cfg.held_out_scene); index_path.parent.mkdir(parents=True,exist_ok=True)
    print(f"building Arrow table for {len(rows):,} forecast windows", flush=True)
    index_table=pa.Table.from_pylist(rows,schema=INDEX_SCHEMA); temporary=index_path.with_suffix(".tmp.parquet")
    print(f"writing forecast index atomically to {index_path}", flush=True)
    pq.write_table(index_table,temporary,compression="zstd"); temporary.replace(index_path)
    print(f"wrote {index_path} ({index_path.stat().st_size:,} bytes)", flush=True)
    split_manifest_path=prepare_splits(spec.path,cfg.seeds.data_seed,cfg.split_strategy,cfg.held_out_scene)
    split_manifest=json.loads(split_manifest_path.read_text())
    analysis_path=_analysis_index_path(spec.path,cfg.split_strategy,cfg.held_out_scene)
    analysis_rows=[]
    if analysis_path.exists():
        analysis_table=pq.read_table(analysis_path)
        category_by_key={(row["scene_sequence"],int(row["target_agent_id"]),int(row["start_step"])):int(row["scene_tag"]) for row in rows}
        values=[]
        for row in analysis_table.select(["scene_sequence","agent_id","source_start_step"]).to_pylist():
            values.append(category_by_key.get((row["scene_sequence"],int(row["agent_id"]),int(row["source_start_step"]))))
        column_index=analysis_table.schema.get_field_index("trajnet_category")
        analysis_table=analysis_table.set_column(column_index,"trajnet_category",pa.array(values,pa.int8()))
        analysis_tmp=analysis_path.with_suffix(".tmp.parquet")
        pq.write_table(analysis_table,analysis_tmp,compression="zstd");analysis_tmp.replace(analysis_path)
        analysis_rows=analysis_table.select(["scene_sequence"]).to_pylist()
    if cfg.split_strategy == "temporal":
        split_manifest["number_of_forecast_windows_by_split"] = {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        }
        split_manifest["number_of_analysis_trajlets"] = len(analysis_rows)
        temporary_manifest = split_manifest_path.with_suffix(".tmp")
        temporary_manifest.write_text(json.dumps(split_manifest, indent=2) + "\n")
        temporary_manifest.replace(split_manifest_path)
    else:
        if cfg.split_strategy == "held_out_recording":
            # ``record["split"]`` is only a two-value holdout role (train/test), while
            # the per-window split is three-way; report the window-level counts too.
            split_manifest["number_of_forecast_windows_by_split"] = {
                split: sum(row["split"] == split for row in rows)
                for split in ("train", "val", "test")
            }
        split_manifest=update_sample_counts(split_manifest,analysis_rows,rows)
        write_manifest(split_manifest,split_manifest_path)
    histogram_counts,histogram_edges=np.histogram(usable_lengths,bins=min(50,max(1,len(set(usable_lengths)))))
    stats={"dataset":spec.name,"data_seed":cfg.seeds.data_seed,"hours":hours,"L_med":l_med,"L_mean":float(np.mean(usable_lengths)),
           "track_length_histogram":{"edges":histogram_edges.tolist(),"counts":histogram_counts.tolist()},
           "T_d_raw":raw_track_segments,"T_d_usable":len(usable_track_keys),"T_d_post_categorization":len(track_keys),
           "type_counts_before":{f"{split}_{tag}":before[(split,tag)] for split in ("train","val","test") for tag in range(1,5)},
           "type_counts_after":{f"{split}_{tag}":after[(split,tag)] for split in ("train","val","test") for tag in range(1,5)},
           "dense_forecast_windows":len(rows),
           # Retained temporarily for compatibility with existing sizing tables.
           "windows_stride_2":int(sum(max(0,(length-chunk_size)//2+1) for length in usable_lengths)),
           "split_candidate_scenes":split_candidate_count,"neighbor_filter_pass_scenes":neighbor_filter_pass_count,"scene_filter_survival":neighbor_filter_pass_count/split_candidate_count if split_candidate_count else float("nan"),
           "s_d":chunk_stride,"w_d":w_d,"p_d":p_d,"eval_scene_targets":eval_targets,"final_train_scenes":sum(r["split"]=="train" and r["accepted"] and r["track_kept"] for r in rows),
           "final_val_scenes":sum(r["split"]=="val" and r["accepted"] and r["track_kept"] and r["eval_kept"] for r in rows),"final_test_scenes":sum(r["split"]=="test" and r["accepted"] and r["track_kept"] and r["eval_kept"] for r in rows),
           "config_hash":stable_config_hash({"config": cfg, "task": scoped_task_config(task, spec.name)}),
           **git_provenance(ROOT),
           "created_utc":datetime.now(timezone.utc).isoformat().replace("+00:00","Z")}
    canonical_dataset_root=(ROOT/"datasets").resolve()
    resolved_dataset=spec.path.resolve()
    stats_root=PAPER_STATS_ROOT if resolved_dataset.is_relative_to(canonical_dataset_root) else spec.path/"stats"
    stats_path=stats_root/f"{spec.name}_sizing.json"; stats_path.parent.mkdir(parents=True,exist_ok=True)
    stats_path.write_text(json.dumps(stats,indent=2)+"\n")
    pretrain_fractions=dataset_task.get("pretrain_fractions")
    if pretrain_fractions:
        # Training tracks only: val/test identities must not shift the prefix the
        # training subset is drawn from.
        all_tracks=sorted({f"{row['recording']}:{row['primary_track_id']}" for row in rows if row["split"] == "train" and row["accepted"] and row["track_kept"]})
        for fraction in pretrain_fractions:
            selected=steps_subset(all_tracks,fraction,cfg.seeds.data_seed)
            selected_set=set(selected); scene_count=sum(row["split"] == "train" and row["accepted"] and row["track_kept"] and f"{row['recording']}:{row['primary_track_id']}" in selected_set for row in rows)
            write_subset_manifest(MANIFEST_ROOT/manifest_name(spec.name,fraction,cfg.seeds.data_seed),spec.name,fraction,cfg.seeds.data_seed,selected,scene_count,stats["config_hash"])
    return index_path,stats


def main()->None:
    parser=argparse.ArgumentParser(description="Build primary-agent scene indices, categories, and sizing stats")
    parser.add_argument("dataset_dir",type=Path); parser.add_argument("--name"); parser.add_argument("--data-seed",type=int,default=42)
    parser.add_argument("--workers",type=int,default=min(8,os.cpu_count() or 1),help="Parallel categorization workers (default: one per physical core, capped at 8)")
    parser.add_argument("--split-strategy",choices=SPLIT_STRATEGIES,default="grouped_scene_sequence")
    parser.add_argument("--held-out-scene",help="Physical scene_id for leave-one-scene-out")
    args=parser.parse_args(); spec=DatasetSpec(args.name or args.dataset_dir.name,args.dataset_dir)
    path,stats=preprocess_dataset(spec,Config(seeds=SeedConfig(data_seed=args.data_seed,model_seed=42),split_strategy=args.split_strategy,held_out_scene=args.held_out_scene),workers=max(1,args.workers))
    print(path); print(json.dumps(stats,indent=2))


if __name__=="__main__": main()
