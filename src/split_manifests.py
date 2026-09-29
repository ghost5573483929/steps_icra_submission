from __future__ import annotations

import csv
import hashlib
import json
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq

from artifact_provenance import source_set_revision
from canonical import recording_identity
from splits import SPLITS

GROUPED_DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def canonical_source_revision(files: list[Path]) -> str:
    """Revision of the exact canonical recording population used by a split."""
    if not files:
        raise ValueError("A split manifest requires at least one canonical recording")
    return source_set_revision(files, base=files[0].parent.parent)


def _stable_order(values: Iterable[str], seed: int) -> list[str]:
    return sorted(
        values,
        key=lambda value: hashlib.sha256(f"{seed}:{value}".encode()).digest(),
    )


def _split_counts(total: int, ratios: dict[str, float]) -> dict[str, int]:
    # `total == 0` is unreachable from production, but `total == 1` and `total == 2`
    # are reachable through the deferred pool.
    if total == 0:
        return {split: 0 for split in SPLITS}
    if total == 1:
        warnings.warn("One scene_sequence can populate only train", RuntimeWarning)
        return {"train": 1, "val": 0, "test": 0}
    if total == 2:
        warnings.warn("Two scene_sequences cannot populate train/val/test; validation is empty", RuntimeWarning)
        return {"train": 1, "val": 0, "test": 1}
    val = max(1, round(total * ratios["val"]))
    test = max(1, round(total * ratios["test"]))
    if val + test >= total:
        val, test = 1, 1
    return {"train": total - val - test, "val": val, "test": test}


def assign_grouped_sequences(
    sequence_to_scene: dict[str, str],
    *,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, str]:
    """Assign each indivisible sequence once, approximately stratified by physical scene."""
    if set(ratios) != set(SPLITS) or not np.isclose(sum(ratios.values()), 1.0):
        raise ValueError("Grouped split ratios must contain train/val/test and sum to one")
    by_scene: dict[str, list[str]] = defaultdict(list)
    for sequence, scene in sequence_to_scene.items():
        by_scene[scene].append(sequence)
    assignments: dict[str, str] = {}
    deferred: list[str] = []
    # Three independent sequences are the minimum to fill every split without leaking
    # one; smaller scene groups are allocated globally instead.
    for scene, sequences in sorted(by_scene.items()):
        if len(sequences) < 3:
            deferred.extend(sequences)
            continue
        ordered = _stable_order(sequences, seed)
        counts = _split_counts(len(ordered), ratios)
        cursor = 0
        for split in SPLITS:
            for sequence in ordered[cursor : cursor + counts[split]]:
                assignments[sequence] = split
            cursor += counts[split]
    if deferred:
        ordered = _stable_order(deferred, seed)
        counts = _split_counts(len(ordered), ratios)
        cursor = 0
        for split in SPLITS:
            for sequence in ordered[cursor : cursor + counts[split]]:
                assignments[sequence] = split
            cursor += counts[split]
    return assignments


def assert_no_sequence_leakage(records: list[dict]) -> None:
    sets = {
        split: {record["scene_sequence"] for record in records if record["split"] == split}
        for split in SPLITS
    }
    for index, left in enumerate(SPLITS):
        for right in SPLITS[index + 1 :]:
            overlap = sets[left] & sets[right]
            if overlap:
                raise AssertionError(f"scene_sequence leakage between {left}/{right}: {sorted(overlap)}")
    recording_splits: dict[str, set[str]] = defaultdict(set)
    for record in records:
        recording_splits[record["recording"]].add(record["split"])
    leaked = {key: values for key, values in recording_splits.items() if len(values) > 1}
    if leaked:
        raise AssertionError(f"recording identity appears in multiple splits: {leaked}")


def _base_records(files: list[Path]) -> list[dict]:
    records = []
    seen_sequences: dict[str, Path] = {}
    for path in files:
        identity = recording_identity(path)
        if identity.scene_sequence in seen_sequences:
            raise ValueError(
                f"scene_sequence {identity.scene_sequence!r} is shared by {seen_sequences[identity.scene_sequence]} and {path}"
            )
        seen_sequences[identity.scene_sequence] = path
        records.append(
            {
                "dataset": identity.dataset,
                "scene_id": identity.scene_id,
                "scene_sequence": identity.scene_sequence,
                "recording": identity.recording,
                "source_file": path.name,
                "split": None,
                "number_of_tracks": None,
                "number_of_analysis_trajlets": 0,
                "number_of_forecast_windows": 0,
            }
        )
    return records


def temporal_manifest(
    files: list[Path],
    *,
    seed: int,
    guard_steps: int,
    ratios: dict[str, float],
) -> dict:
    records = _base_records(files)
    for record in records:
        record["split"] = "temporal"
        del record["number_of_tracks"]
        del record["number_of_analysis_trajlets"]
        del record["number_of_forecast_windows"]
    return {
        "version": 1,
        "strategy": "temporal",
        "seed": seed,
        "ratios": ratios,
        "guard_steps": guard_steps,
        "canonical_source_set_revision": canonical_source_revision(files),
        "records": records,
    }


def grouped_manifest(
    files: list[Path],
    *,
    seed: int,
    ratios: dict[str, float] | None = None,
    count_tracks: bool = False,
) -> dict:
    ratios = ratios or dict(GROUPED_DEFAULT_RATIOS)
    records = _base_records(files)
    assignments = assign_grouped_sequences(
        {record["scene_sequence"]: record["scene_id"] for record in records},
        seed=seed,
        ratios=ratios,
    )
    paths = {path.name: path for path in files}
    for record in records:
        record["split"] = assignments[record["scene_sequence"]]
        if count_tracks:
            track_ids = pq.read_table(paths[record["source_file"]], columns=["track_id"])["track_id"]
            record["number_of_tracks"] = len(set(track_ids.to_pylist()))
    assert_no_sequence_leakage(records)
    return {
        "version": 1,
        "strategy": "grouped_scene_sequence",
        "seed": seed,
        "ratios": ratios,
        "canonical_source_set_revision": canonical_source_revision(files),
        "records": records,
    }


def held_out_recording_manifest(
    files: list[Path],
    *,
    held_out_recording: str,
    seed: int,
    ratios: dict[str, float],
    guard_steps: int,
    count_tracks: bool = False,
) -> dict:
    """One named recording becomes the entire test split.

    Every other recording is marked ``train``; ``splits.assign_split`` then divides each
    along its own timeline into train and val using ``ratios`` and ``guard_steps``. Use this
    when the held-out unit is a recording rather than a physical scene, which
    ``loso_manifest`` cannot express.
    """
    records = _base_records(files)
    names = {record["recording"] for record in records}
    if held_out_recording not in names:
        raise ValueError(
            f"Unknown held-out recording {held_out_recording!r}; choose {sorted(names)}"
        )
    if len(names) < 2:
        raise ValueError("held_out_recording needs at least two recordings")
    for record in records:
        record["split"] = "test" if record["recording"] == held_out_recording else "train"
    if count_tracks:
        paths = {path.name: path for path in files}
        for record in records:
            track_ids = pq.read_table(paths[record["source_file"]], columns=["track_id"])["track_id"]
            record["number_of_tracks"] = len(set(track_ids.to_pylist()))
    assert_no_sequence_leakage(records)
    return {
        "version": 1,
        "strategy": "held_out_recording",
        "seed": seed,
        "held_out_recording": held_out_recording,
        "ratios": ratios,
        "guard_steps": guard_steps,
        "canonical_source_set_revision": canonical_source_revision(files),
        "records": records,
    }


def loso_manifest(
    files: list[Path],
    *,
    held_out_scene: str,
    seed: int,
    validation_fraction: float = 0.1,
    count_tracks: bool = False,
) -> dict:
    records = _base_records(files)
    scenes = {record["scene_id"] for record in records}
    if len(scenes) < 2:
        raise ValueError("LOSO is not meaningful with fewer than two physical scenes")
    if held_out_scene not in scenes:
        raise ValueError(f"Unknown held-out scene {held_out_scene!r}; choose {sorted(scenes)}")
    eligible = [record["scene_sequence"] for record in records if record["scene_id"] != held_out_scene]
    ordered = _stable_order(eligible, seed)
    n_val = max(1, round(len(ordered) * validation_fraction)) if len(ordered) > 1 else 0
    validation = set(ordered[:n_val])
    for record in records:
        if record["scene_id"] == held_out_scene:
            record["split"] = "test"
        elif record["scene_sequence"] in validation:
            record["split"] = "val"
        else:
            record["split"] = "train"
    if count_tracks:
        paths = {path.name: path for path in files}
        for record in records:
            track_ids = pq.read_table(paths[record["source_file"]], columns=["track_id"])["track_id"]
            record["number_of_tracks"] = len(set(track_ids.to_pylist()))
    assert_no_sequence_leakage(records)
    return {
        "version": 1,
        "strategy": "leave_one_scene_out",
        "seed": seed,
        "canonical_source_set_revision": canonical_source_revision(files),
        "held_out_scene": held_out_scene,
        "training_scenes": sorted(scenes - {held_out_scene}),
        "validation_scenes": sorted({r["scene_id"] for r in records if r["split"] == "val"}),
        "test_scenes": [held_out_scene],
        "records": records,
    }


def update_sample_counts(manifest: dict, analysis_rows: list[dict], forecast_rows: list[dict]) -> dict:
    analysis = Counter(row["scene_sequence"] for row in analysis_rows)
    forecast = Counter(row["scene_sequence"] for row in forecast_rows)
    for record in manifest["records"]:
        sequence = record["scene_sequence"]
        record["number_of_analysis_trajlets"] = analysis[sequence]
        record["number_of_forecast_windows"] = forecast[sequence]
    assert_no_sequence_leakage(manifest["records"])
    return manifest


def write_manifest(manifest: dict, path: Path) -> tuple[Path, Path, Path]:
    assert_no_sequence_leakage(manifest["records"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(path)
    csv_path = path.with_suffix(".csv")
    columns = [
        "dataset", "scene_id", "scene_sequence", "recording", "split",
        "number_of_tracks", "number_of_analysis_trajlets", "number_of_forecast_windows",
    ]
    with csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: record.get(key) for key in columns} for record in manifest["records"])
    summary_path = path.with_suffix(".md")
    is_held_out_recording = manifest["strategy"] == "held_out_recording"
    # ``held_out_recording`` records carry a two-value holdout role (train/test), not
    # the true three-way per-window split, so the column is labelled accordingly.
    split_header = "Holdout role" if is_held_out_recording else "Split"
    # A ``val`` row would always read 0 here because ``val`` is not a holdout role at
    # all, so only the roles that exist are listed.
    reported_splits = [split for split in SPLITS if split != "val"] if is_held_out_recording else list(SPLITS)
    lines = [f"| {split_header} | Physical scenes | Scene sequences | Tracks | Analysis trajlets | Forecast windows |", "|---|---:|---:|---:|---:|---:|"]
    for split in reported_splits:
        selected = [record for record in manifest["records"] if record["split"] == split]
        def total(key: str) -> str:
            values = [record.get(key) for record in selected]
            return "unknown" if any(value is None for value in values) else str(sum(values))
        lines.append(
            f"| {split} | {len({r['scene_id'] for r in selected})} | {len(selected)} | "
            f"{total('number_of_tracks')} | {total('number_of_analysis_trajlets')} | {total('number_of_forecast_windows')} |"
        )
    if manifest["strategy"] == "leave_one_scene_out":
        lines.extend(["", f"Held-out scene: `{manifest['held_out_scene']}`"])
    if is_held_out_recording:
        by_split = manifest.get("number_of_forecast_windows_by_split")
        lines.extend([
            "",
            "The table above groups records by holdout role (`train` = every non-held-out "
            "recording, `test` = the held-out recording), not by the true per-window split; "
            "`assign_split` further divides each non-held-out recording's own timeline into "
            "train/val. Genuine per-window split totals:",
        ])
        if by_split:
            lines.extend(["", "| Split | Forecast windows |", "|---|---:|"])
            lines.extend(f"| {split} | {by_split.get(split, 'unknown')} |" for split in SPLITS)
    summary_path.write_text("\n".join(lines) + "\n")
    return path, csv_path, summary_path
