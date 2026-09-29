from __future__ import annotations

from pathlib import Path

SPLITS = ("train", "val", "test")


def _recording_key(recording: str) -> str:
    """Canonicalize a recording identity to its stem, with or without the file suffix."""
    return recording.removesuffix(".traj.parquet")


def assign_split(config: dict, *, recording: str, start: int, first: int,
                 last: int, window_span: int) -> str | None:
    """Assign one complete window according to an explicit split strategy."""
    # No production config reaches the "recording" default; it is retained only because
    # data.py's _validate_splits still carries a matching branch.
    strategy = config.get("strategy", "recording")
    if strategy == "temporal":
        ratios = config["ratios"]; guard = int(config["guard_steps"])
        end = last + window_span
        train_boundary = first + (end - first) * ratios["train"]
        val_boundary = train_boundary + (end - first) * ratios["val"]
        if start + window_span <= train_boundary: return "train"
        if start >= train_boundary + guard and start + window_span <= val_boundary: return "val"
        if start >= val_boundary + guard: return "test"
        return None
    if strategy == "held_out_recording":
        # One named recording is the whole test split; every other recording is split
        # train/val along its own timeline, guarded so no window straddles the boundary.
        key = _recording_key(recording)
        role = config.get("split_by_recording", {}).get(key)
        if role is None:
            raise ValueError(
                f"held_out_recording: unrecognized recording {recording!r} is absent from "
                f"split_by_recording (held-out recording: {config.get('held_out_recording')!r}); "
                "refusing to silently default it to train"
            )
        if role == "test":
            return "test"
        ratios = config["ratios"]; guard = int(config["guard_steps"])
        denominator = ratios["train"] + ratios["val"]
        end = last + window_span
        boundary = first + (end - first) * (ratios["train"] / denominator)
        if start + window_span <= boundary: return "train"
        if start >= boundary + guard: return "val"
        return None
    if strategy == "recording":
        name = _recording_key(recording) + ".traj.parquet"
        for split in SPLITS:
            if any(Path(item).name == Path(name).name for item in config.get(split, [])):
                return split
        return None
    if strategy == "grouped_scene_sequence":
        mapping = config.get("split_by_recording", {})
        return mapping.get(_recording_key(recording))
    if strategy == "leave_one_scene_out":
        key = _recording_key(recording)
        mapping = config.get("split_by_recording", {})
        if key in mapping:
            return mapping[key]
        scene_by_recording = config.get("scene_by_recording", {})
        if key not in scene_by_recording:
            raise ValueError(
                f"leave_one_scene_out: unrecognized recording {recording!r} "
                f"is absent from both split_by_recording and scene_by_recording "
                f"(configured held-out scene: {config.get('test_scene')!r}); "
                "refusing to silently default it to train"
            )
        scene = scene_by_recording[key]
        if scene == config.get("test_scene"): return "test"
        if scene in set(config.get("val_scenes", [])): return "val"
        return "train"
    raise ValueError(f"Unknown split strategy: {strategy}")
