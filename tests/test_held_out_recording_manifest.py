"""Under ``held_out_recording`` a manifest record's ``split`` is a two-value holdout
role, while the real split is three-way per window: ``assign_split`` divides each
non-held-out recording's own timeline into train/val.
"""
from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

import preprocess
from config import Config, DatasetSpec, SeedConfig, SizingConfig
from conftest import write_canonical_fixture
from preprocess import preprocess_dataset

HISTORY_STEPS = 4
FUTURE_STEPS = 4
CANONICAL_STRIDE = 1
CHUNK_STRIDE = 3
TRACK_STEPS = 200
TRACK_IDS = (1, 2, 3)
RECORDINGS = ("r0", "r1", "r2", "r3", "r4")


def write_task(path: Path) -> Path:
    task = {
        "sample": "scene",
        "sample_hz": 2.5,
        "canonical_stride": CANONICAL_STRIDE,
        "history_steps": HISTORY_STEPS,
        "future_steps": FUTURE_STEPS,
        "min_length": 0.0,
        "w_target": 2.0,
        "eval_stride": 5,
        "datasets": {"synthetic": {"chunk_stride": CHUNK_STRIDE}},
        "forecasting": {"preset": "steps_default"},
    }
    path.write_text(json.dumps(task, indent=2) + "\n")
    return path


def build_held_out_recording_fixture(root: Path) -> Path:
    converted = root / "converted"
    converted.mkdir(parents=True, exist_ok=True)
    for recording in RECORDINGS:
        write_canonical_fixture(
            converted / f"{recording}.traj.parquet",
            {
                track_id: [
                    (step, float(step * (track_id + 1)), float(track_id), None)
                    for step in range(TRACK_STEPS)
                ]
                for track_id in TRACK_IDS
            },
            dataset="synthetic",
            scene_id=f"scene-{recording}",
            scene_sequence=recording,
            recording=recording,
        )
    (root / "source.json").write_text(json.dumps({"canonical_recordings": list(RECORDINGS)}) + "\n")
    (root / "splits.json").write_text(json.dumps({
        "strategy": "held_out_recording",
        "held_out_recording": "r0",
        "ratios": {"train": 0.8, "val": 0.2},
    }) + "\n")
    return root


def run(root: Path, data_seed: int = 7):
    task_path = write_task(root / "task.json")
    cfg = Config(
        seeds=SeedConfig(data_seed=data_seed, model_seed=42),
        sizing=SizingConfig(n_target=None),
        eval_stride=None,
        task_path=task_path,
        split_strategy="held_out_recording",
    )
    index_path, stats = preprocess_dataset(DatasetSpec("synthetic", root), cfg, workers=1)
    table = pq.read_table(index_path)
    manifest_path = root / "split_manifests" / f"held-out-recording-seed{data_seed}.json"
    manifest = json.loads(manifest_path.read_text())
    md_text = manifest_path.with_suffix(".md").read_text()
    return stats, table.to_pylist(), manifest, md_text


def test_held_out_recording_manifest_reconciles_with_window_level_truth(tmp_path):
    root = build_held_out_recording_fixture(tmp_path)
    stats, rows, manifest, md_text = run(root)

    raw = {split: sum(row["split"] == split for row in rows) for split in ("train", "val", "test")}
    print("raw window-level truth:", raw)
    print("final_*_scenes:", stats["final_train_scenes"], stats["final_val_scenes"], stats["final_test_scenes"])
    print(md_text)

    assert raw["val"] > 0, "fixture must exercise a nonzero val split to be meaningful"

    by_split = manifest.get("number_of_forecast_windows_by_split")
    assert by_split == raw, f"manifest by-split counts {by_split} != raw window truth {raw}"

    val_rows = [line for line in md_text.splitlines() if line.startswith("| val")]
    assert val_rows, "expected a val row in the manifest markdown"
    holdout_role_val_is_zero = val_rows[0].split("|")[-2].strip() == "0"
    if holdout_role_val_is_zero and stats["final_val_scenes"] > 0:
        assert str(raw["val"]) in md_text, (
            "val row reads 0 while final_val_scenes>0 and the true val window "
            "count is not reconciled anywhere in the manifest markdown"
        )
