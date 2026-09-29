"""Deterministic, nested target-training fractions with persistent manifests."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterator

import numpy as np
from filelock import FileLock
from torch.utils.data import ConcatDataset, Subset

from baselines.status import atomic_json


ROOT = Path(__file__).resolve().parents[2]


def _entries(dataset, offset: int = 0) -> Iterator[tuple[int, str, str]]:
    """Yield top-level index, stable track key, and stable sample key."""
    if isinstance(dataset, ConcatDataset):
        child_offset = offset
        for child in dataset.datasets:
            yield from _entries(child, child_offset)
            child_offset += len(child)
        return
    if isinstance(dataset, Subset):
        source = list(_entries(dataset.dataset))
        for local, source_index in enumerate(dataset.indices):
            _old_index, track_key, sample_key = source[int(source_index)]
            yield offset + local, track_key, sample_key
        return
    rows = getattr(dataset, "rows", None)
    spec = getattr(dataset, "spec", None)
    if rows is None:
        raise TypeError(f"Cannot derive target-fraction identities from {type(dataset).__name__}")
    dataset_name = str(getattr(spec, "name", "unknown"))
    for local, row in enumerate(rows):
        record = row.get("window_record")
        scene_sequence = str(
            row.get("scene_sequence", getattr(record, "scene_sequence", ""))
        )
        recording = str(row.get("recording", getattr(record, "recording", "")))
        agent = row.get("primary_track_id", getattr(record, "target_agent_id", ""))
        start = int(row.get("start_step", getattr(record, "start_step", -1)))
        track_key = f"{dataset_name}:{scene_sequence}:{recording}:{agent}"
        yield offset + local, track_key, f"{track_key}:{start}"


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "_.-" else "_" for character in value)


def fraction_manifest_path(target: str, fold: str | None, fraction: float, data_seed: int) -> Path:
    scope = target if fold is None else f"{target}-{fold}"
    return ROOT / "experiments/manifests/target_fractions" / f"{_safe(scope)}-{fraction:g}-seed{data_seed}.json"


def target_training_fraction(
    dataset,
    *,
    target: str,
    fold: str | None,
    fraction: float,
    data_seed: int,
):
    """Select complete target trajectories, retaining all their train windows.

    One seed defines one shuffled track order and every fraction takes a prefix,
    so 5/10/25/50/100 percent populations are nested.  Validation and test are
    deliberately outside this function and therefore never fractioned.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError("target fraction must be in (0, 1]")
    entries = list(_entries(dataset))
    if not entries:
        raise ValueError("Cannot fraction an empty target training population")
    tracks = sorted({track for _index, track, _sample in entries})
    source_payload = "\n".join(sorted(sample for _index, _track, sample in entries))
    source_revision = "sha256:" + hashlib.sha256(source_payload.encode()).hexdigest()
    path = fraction_manifest_path(target, fold, fraction, data_seed)
    path.parent.mkdir(parents=True, exist_ok=True)

    with FileLock(str(path) + ".lock"):
        manifest = json.loads(path.read_text()) if path.exists() else None
        current = bool(
            manifest
            and manifest.get("source_revision") == source_revision
            and manifest.get("data_seed") == data_seed
            and float(manifest.get("fraction", -1)) == fraction
        )
        if current:
            selected_tracks = set(manifest["track_ids"])
        else:
            shuffled = list(tracks)
            np.random.default_rng(data_seed).shuffle(shuffled)
            count = max(1, int(round(len(shuffled) * fraction)))
            selected_tracks = set(shuffled[:count])
            selected_items = sum(track in selected_tracks for _index, track, _sample in entries)
            manifest = {
                "target_dataset": target,
                "fold": fold,
                "fraction": fraction,
                "data_seed": data_seed,
                "selection_unit": "namespaced_target_trajectory",
                "source_revision": source_revision,
                "n_source_tracks": len(tracks),
                "n_selected_tracks": len(selected_tracks),
                "n_source_windows": len(entries),
                "n_selected_windows": selected_items,
                "track_ids": sorted(selected_tracks),
            }
            atomic_json(path, manifest)
    indices = [index for index, track, _sample in entries if track in selected_tracks]
    if not indices:
        raise RuntimeError(f"Target fraction manifest selected no windows: {path}")
    return Subset(dataset, indices), path
