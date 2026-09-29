from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from artifact_provenance import git_provenance

_ROOT = Path(__file__).resolve().parents[1]


def steps_subset(track_ids: list[str], fraction: float, data_seed: int) -> list[str]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1]")
    rng = np.random.default_rng(data_seed)
    shuffled = list(track_ids)
    rng.shuffle(shuffled)
    return shuffled[: int(round(len(shuffled) * fraction))]


def write_subset_manifest(
    path: Path,
    dataset: str,
    fraction: float,
    data_seed: int,
    track_ids: list[str],
    n_scenes_post_categorization: int,
    config_hash: str,
) -> dict:
    payload = {
        "dataset": dataset,
        "fraction": fraction,
        "data_seed": data_seed,
        "n_tracks": len(track_ids),
        "n_scenes_post_categorization": n_scenes_post_categorization,
        "track_ids": track_ids,
        "config_hash": config_hash,
        **git_provenance(_ROOT),
        "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    return payload


def load_subset_manifest(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "dataset",
        "fraction",
        "data_seed",
        "n_tracks",
        "n_scenes_post_categorization",
        "track_ids",
        "config_hash",
        "created_utc",
    }
    missing = required - payload.keys()
    if missing:
        raise ValueError(f"Manifest is missing fields: {sorted(missing)}")
    if payload["n_tracks"] != len(payload["track_ids"]):
        raise ValueError("Manifest n_tracks does not match track_ids")
    return payload


def manifest_name(dataset: str, fraction: float, data_seed: int) -> str:
    return f"{dataset}_{fraction:g}_{data_seed}.json"
