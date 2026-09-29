from __future__ import annotations

import json
import math
import random
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm.auto import tqdm

from config import Config, DatasetSpec
from forecasting_representation import prepare_forecast_representation
from forecasting_windows import (
    RecordingWindowCache,
    enumerate_forecast_windows,
    materialize_forecast_window,
)
from canonical import CANONICAL_DT, recording_identity, validate_canonical_file
from manifests import load_subset_manifest, manifest_name
from dataset_paths import MANIFEST_ROOT, TASK_CONFIG_PATH
from splits import SPLITS
from dataset_paths import metadata_path
from split_manifests import (
    held_out_recording_manifest,
    GROUPED_DEFAULT_RATIOS,
    canonical_source_revision,
    grouped_manifest,
    loso_manifest,
    temporal_manifest,
    write_manifest,
)

ROOT = Path(__file__).resolve().parents[1]

# Strategies that assign splits per *window* along a recording's own timeline rather than
# per whole recording, which `split_files` and the generic-membership branch of
# `SceneDataset` cannot express.
PER_WINDOW_SPLIT_STRATEGIES = ("temporal", "held_out_recording")


class PerWindowSplitStrategyUnsupportedError(ValueError):
    """Raised when a per-window split strategy reaches a per-recording split path.

    Only `target_complete` scene membership, which reads the prebuilt per-window index
    filtered by `split`, supports these strategies.
    """
@dataclass
class Scene:
    xy: np.ndarray
    primary_track_id: str
    neighbor_track_ids: list[str]
    dataset: str
    recording: str
    start_step: int
    scene_tag: int
    sub_tags: list[int]
    valid: np.ndarray
    rotation: float
    center: np.ndarray
    scene_id: str = ""
    scene_sequence: str = ""
    observation_end_step: int = 0
    future_end_step: int = 0
    canonical_xy: np.ndarray | None = None
    target_xy_world: np.ndarray | None = None
    normalization_heading_source: str = ""
    forecast_input_preprocessing: str = "raw"
    forecast_target_preprocessing: str = "raw"
    features: np.ndarray | None = None
    relative_xy: np.ndarray | None = None
    feature_names: tuple[str, ...] = ()
    coordinate_frame: str = "world"
    forecast_preprocessing_hash: str = ""
    augmentation: dict | None = None


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _recordings(dataset_dir: Path) -> list[Path]:
    recordings = sorted(
        path for path in (dataset_dir / "converted").rglob("*.traj.parquet")
        if "Test_set" not in path.name and "Test_set" not in path.parts
    )
    try:
        source = _read_json(metadata_path(dataset_dir, "source.json"))
    except FileNotFoundError:
        return recordings
    excluded = excluded_recordings(dataset_dir)
    if excluded:
        recordings = [
            path for path in recordings
            if path.name.removesuffix(".traj.parquet") not in excluded
        ]
    canonical_recordings = source.get("canonical_recordings", source.get("benchmark_sequences"))
    if canonical_recordings is None:
        return recordings
    allowed = set(canonical_recordings)
    return [
        path for path in recordings
        if path.name.removesuffix(".traj.parquet") in allowed
    ]


def excluded_recordings(dataset_dir: Path) -> frozenset[str]:
    """Recordings a dataset's ``source.json`` declares must never enter any split.

    Conversion output stays on disk for provenance, but no split, index or loader
    may draw from it.
    """
    try:
        source = _read_json(metadata_path(dataset_dir, "source.json"))
    except FileNotFoundError:
        return frozenset()
    declared = source.get("excluded_recordings", [])
    if not isinstance(declared, list) or any(not isinstance(name, str) for name in declared):
        raise ValueError(f"{dataset_dir}: excluded_recordings must be a list of recording names")
    return frozenset(declared)


def _validate_splits(splits: dict) -> None:
    strategy=splits.get("strategy","recording")
    if strategy == "temporal":
        ratios = splits.get("ratios", {})
        if any(split not in ratios for split in SPLITS):
            raise ValueError(f"Temporal split ratios must contain {SPLITS}")
        if any(not isinstance(ratios[s], (int, float)) or ratios[s] <= 0 for s in SPLITS):
            raise ValueError("Temporal split ratios must be positive numbers")
        if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-6:
            raise ValueError("Temporal split ratios must sum to 1")
        if splits.get("guard_steps", 0) < 0: raise ValueError("guard_steps cannot be negative")
        return
    if strategy == "grouped_scene_sequence":
        mapping = splits.get("split_by_recording", {})
        if not mapping or any(value not in SPLITS for value in mapping.values()):
            raise ValueError("Grouped split requires a valid split_by_recording mapping")
        return
    if strategy == "leave_one_scene_out":
        if not splits.get("test_scene"): raise ValueError("LOSO manifest requires test_scene")
        return
    if strategy == "held_out_recording":
        if not splits.get("held_out_recording"):
            raise ValueError("held_out_recording manifest requires held_out_recording")
        mapping = splits.get("split_by_recording", {})
        if not mapping or any(value not in SPLITS for value in mapping.values()):
            raise ValueError("held_out_recording requires a valid split_by_recording mapping")
        if splits["held_out_recording"] not in mapping:
            raise ValueError("held_out_recording names a recording absent from split_by_recording")
        ratios = splits.get("ratios", {})
        if not all(key in ratios for key in ("train", "val")):
            raise ValueError("held_out_recording requires train and val ratios")
        return
    if strategy != "recording": raise ValueError(f"Unknown split strategy: {strategy}")
    if any(split not in splits for split in SPLITS): raise ValueError(f"Split manifest must contain {SPLITS}")
    entries = [item for split in SPLITS for item in splits[split]]
    if any(Path(item).is_absolute() or ".." in Path(item).parts for item in entries):
        raise ValueError("Split paths must be relative to converted")
    if len(entries) != len(set(entries)): raise ValueError("A recording appears in multiple splits")


def _split_manifest_path(dataset_dir: Path, seed: int, strategy: str, held_out_scene: str | None) -> Path:
    if strategy == "grouped_scene_sequence":
        name = f"grouped-seed{seed}.json"
    elif strategy == "leave_one_scene_out":
        if not held_out_scene:
            raise ValueError("held_out_scene is required for leave_one_scene_out")
        safe = "".join(character if character.isalnum() or character in "_.-" else "_" for character in held_out_scene)
        name = f"loso-{safe}-seed{seed}.json"
    elif strategy == "temporal":
        name = f"temporal-seed{seed}.json"
    elif strategy == "held_out_recording":
        name = f"held-out-recording-seed{seed}.json"
    else:
        raise ValueError(f"Unsupported split strategy: {strategy}")
    return dataset_dir / "split_manifests" / name


def _split_suffix(strategy: str, held_out_scene: str | None) -> str:
    if strategy == "grouped_scene_sequence":
        return ""
    if strategy == "temporal":
        return "-temporal"
    if strategy == "held_out_recording":
        return "-held-out-recording"
    if strategy != "leave_one_scene_out" or not held_out_scene:
        raise ValueError("A held-out scene is required for leave-one-scene-out")
    safe = "".join(character if character.isalnum() or character in "_.-" else "_" for character in held_out_scene)
    return f"-loso-{safe}"


_TEMPORAL_DEFAULT_RATIOS = {"train": 0.8, "val": 0.1, "test": 0.1}


def _temporal_split_override(dataset_dir: Path) -> dict | None:
    """Read a per-dataset temporal-ratio override, or ``None`` for the shared defaults."""
    try:
        path = metadata_path(dataset_dir, "splits.json")
    except FileNotFoundError:
        return None
    override = _read_json(path)
    allowed = {"strategy", "ratios", "comment"}
    unknown = set(override) - allowed
    if unknown:
        raise ValueError(f"{path}: unknown split override key(s) {sorted(unknown)}")
    if override.get("strategy") != "temporal":
        raise ValueError(f"{path}: unsupported split override strategy {override.get('strategy')!r}")
    ratios = override.get("ratios")
    if not isinstance(ratios, dict) or set(ratios) != set(SPLITS):
        raise ValueError(f"{path}: split override ratios must contain {SPLITS}")
    if any(not isinstance(ratios[s], (int, float)) or ratios[s] <= 0 for s in SPLITS):
        raise ValueError(f"{path}: split override ratios must be positive numbers")
    if abs(sum(ratios[s] for s in SPLITS) - 1.0) > 1e-6:
        raise ValueError(f"{path}: split override ratios must sum to 1")
    return override


def _held_out_recording_override(dataset_dir: Path) -> dict:
    """Read the dataset's required ``splits.json`` declaration for held_out_recording."""
    path = metadata_path(dataset_dir, "splits.json")
    declared = _read_json(path)
    if declared.get("strategy") != "held_out_recording":
        raise ValueError(
            f"{path}: split_strategy is held_out_recording but the file declares "
            f"{declared.get('strategy')!r}"
        )
    name = declared.get("held_out_recording")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{path}: held_out_recording must be a non-empty recording name")
    ratios = declared.get("ratios", {"train": 0.9, "val": 0.1})
    if set(ratios) != {"train", "val"}:
        raise ValueError(f"{path}: held_out_recording ratios must contain exactly train and val")
    if any(not isinstance(ratios[s], (int, float)) or ratios[s] <= 0 for s in ratios):
        raise ValueError(f"{path}: held_out_recording ratios must be positive numbers")
    return {"held_out_recording": name, "ratios": {k: float(v) for k, v in ratios.items()}}


def _build_split_manifest(
    files: list[Path],
    seed: int,
    strategy: str,
    held_out_scene: str | None,
    dataset_dir: Path,
) -> dict:
    if strategy == "grouped_scene_sequence":
        return grouped_manifest(files, seed=seed, count_tracks=True)
    if strategy == "leave_one_scene_out":
        return loso_manifest(
            files,
            held_out_scene=str(held_out_scene),
            seed=seed,
            count_tracks=True,
        )
    if strategy == "temporal":
        window_span = _temporal_guard_steps()
        ratios = _temporal_expected_ratios(dataset_dir)
        return temporal_manifest(files, seed=seed, guard_steps=window_span, ratios=ratios)
    if strategy == "held_out_recording":
        declared = _held_out_recording_override(dataset_dir)
        return held_out_recording_manifest(
            files,
            held_out_recording=declared["held_out_recording"],
            seed=seed,
            ratios=declared["ratios"],
            guard_steps=_temporal_guard_steps(),
            count_tracks=True,
        )
    raise ValueError(f"Unsupported split strategy: {strategy}")


def _temporal_guard_steps() -> int:
    task = _read_json(TASK_CONFIG_PATH)
    missing = [key for key in ("history_steps", "future_steps", "canonical_stride") if key not in task]
    if missing:
        raise KeyError(
            f"{TASK_CONFIG_PATH} is missing required top-level key(s) {missing} "
            "needed to compute the temporal split's guard window"
        )
    return (
        int(task["history_steps"]) + int(task["future_steps"]) - 1
    ) * int(task["canonical_stride"])


def _temporal_expected_ratios(dataset_dir: Path) -> dict:
    override = _temporal_split_override(dataset_dir)
    return override["ratios"] if override is not None else _TEMPORAL_DEFAULT_RATIOS


def _ratios_equal(persisted: object, expected: dict) -> bool:
    """Compare two ratio mappings numerically, so an int 1 and a float 1.0 agree."""
    if not isinstance(persisted, dict) or set(persisted) != set(expected):
        return False
    return all(
        math.isclose(float(persisted[key]), float(expected[key]), rel_tol=0, abs_tol=1e-9)
        for key in expected
    )


def _expected_split_configuration(strategy: str, dataset_dir: Path) -> dict:
    """The ratios/guard_steps a freshly built manifest for `strategy` would carry."""
    if strategy == "temporal":
        return {"ratios": _temporal_expected_ratios(dataset_dir), "guard_steps": _temporal_guard_steps()}
    if strategy == "grouped_scene_sequence":
        return {"ratios": dict(GROUPED_DEFAULT_RATIOS)}
    if strategy == "held_out_recording":
        declared = _held_out_recording_override(dataset_dir)
        return {
            "ratios": declared["ratios"],
            "guard_steps": _temporal_guard_steps(),
            "held_out_recording": declared["held_out_recording"],
        }
    return {}


def _split_manifest_staleness_reasons(
    manifest: dict, files: list[Path], strategy: str, dataset_dir: Path
) -> list[str]:
    """Every reason a persisted split manifest is no longer current; empty means current."""
    reasons = []
    if manifest.get("canonical_source_set_revision") != canonical_source_revision(files):
        reasons.append("canonical source set changed")
    if manifest.get("strategy") != strategy:
        reasons.append(f"strategy changed from {manifest.get('strategy')!r} to {strategy!r}")
    expected = _expected_split_configuration(strategy, dataset_dir)
    if "ratios" in expected and not _ratios_equal(manifest.get("ratios"), expected["ratios"]):
        reasons.append(f"ratios changed from {manifest.get('ratios')!r} to {expected['ratios']!r}")
    if "guard_steps" in expected:
        persisted_guard = manifest.get("guard_steps")
        expected_guard = expected["guard_steps"]
        if not isinstance(persisted_guard, (int, float)) or not math.isclose(
            float(persisted_guard), float(expected_guard), rel_tol=0, abs_tol=1e-9
        ):
            reasons.append(f"guard_steps changed from {persisted_guard!r} to {expected_guard!r}")
    if "held_out_recording" in expected:
        persisted_held_out = manifest.get("held_out_recording")
        expected_held_out = expected["held_out_recording"]
        if persisted_held_out != expected_held_out:
            reasons.append(
                f"held_out_recording changed from {persisted_held_out!r} to {expected_held_out!r}"
            )
    return reasons


def create_splits(
    dataset_dir: str | Path,
    seed: int,
    strategy: str,
    held_out_scene: str | None = None,
) -> dict:
    """Read or propose a scene-sequence-grouped split without mutating data."""
    dataset_dir = Path(dataset_dir)
    files = _recordings(dataset_dir)
    if not files: raise FileNotFoundError(f"No trajectory parquet files in {dataset_dir / 'converted'}")
    path = _split_manifest_path(dataset_dir, seed, strategy, held_out_scene)
    if path.exists():
        persisted = _read_json(path)
        reasons = _split_manifest_staleness_reasons(persisted, files, strategy, dataset_dir)
        if reasons:
            warnings.warn(
                f"Regenerating stale split manifest {path}: {'; '.join(reasons)}",
                RuntimeWarning,
                stacklevel=2,
            )
            manifest = _build_split_manifest(files, seed, strategy, held_out_scene, dataset_dir)
        else:
            manifest = persisted
    else:
        manifest = _build_split_manifest(files, seed, strategy, held_out_scene, dataset_dir)
    if manifest.get("strategy") != strategy:
        raise ValueError(f"{path} contains strategy {manifest.get('strategy')!r}, expected {strategy!r}")
    splits = {
        "source": str(path),
        "strategy": strategy,
        "seed": seed,
        "ratios": manifest.get("ratios", dict(GROUPED_DEFAULT_RATIOS)),
        "guard_steps": int(manifest.get("guard_steps", 0)),
        "split_by_recording": {
            record["recording"]: record["split"] for record in manifest["records"]
            if record["split"] in SPLITS
        },
        "scene_sequence_by_recording": {record["recording"]: record["scene_sequence"] for record in manifest["records"]},
        "scene_id_by_recording": {record["recording"]: record["scene_id"] for record in manifest["records"]},
    }
    if strategy == "held_out_recording":
        splits.update(held_out_recording=manifest["held_out_recording"])
    if strategy == "leave_one_scene_out":
        splits.update(
            test_scene=manifest["held_out_scene"],
            val_scenes=manifest.get("validation_scenes", []),
            scene_by_recording={record["recording"]: record["scene_id"] for record in manifest["records"]},
        )
    _validate_splits(splits); return splits


def prepare_splits(
    dataset_dir: str | Path,
    seed: int,
    strategy: str,
    held_out_scene: str | None = None,
) -> Path:
    from filelock import FileLock
    dataset_dir = Path(dataset_dir)
    path = _split_manifest_path(dataset_dir, seed, strategy, held_out_scene)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path) + ".lock"):
        files = _recordings(dataset_dir)
        if not files:
            raise FileNotFoundError(f"No trajectory parquet files in {dataset_dir / 'converted'}")
        if path.exists():
            manifest = _read_json(path)
            reasons = _split_manifest_staleness_reasons(manifest, files, strategy, dataset_dir)
            if not reasons:
                # Load-bearing despite the discarded return value: `create_splits` ends in
                # `_validate_splits`, the only check that catches a corrupt persisted manifest.
                create_splits(dataset_dir, seed, strategy, held_out_scene)
                return path
            warnings.warn(
                f"Regenerating stale split manifest {path}: {'; '.join(reasons)}",
                RuntimeWarning,
                stacklevel=2,
            )
        manifest = _build_split_manifest(files, seed, strategy, held_out_scene, dataset_dir)
        if strategy == "temporal":
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(manifest, indent=2) + "\n")
            temporary.replace(path)
        else:
            write_manifest(manifest, path)
    return path


def split_files(dataset_dir: str | Path, split: str, seed: int,
                strategy: str, held_out_scene: str | None = None) -> list[Path]:
    if split not in SPLITS: raise ValueError(f"split must be one of {SPLITS}")
    dataset_dir, config = Path(dataset_dir), create_splits(dataset_dir, seed, strategy, held_out_scene)
    if config.get("strategy") in ("grouped_scene_sequence", "leave_one_scene_out"):
        recordings = _recordings(dataset_dir)
        for path in recordings:
            stem = path.name.removesuffix(".traj.parquet")
            metadata_recording = recording_identity(path).recording
            assert metadata_recording == stem, (
                f"{path}: metadata 'recording' key {metadata_recording!r} does not match "
                f"filename stem {stem!r}. split_by_recording is keyed by the metadata value "
                f"while files are matched by filename stem, so a mismatch silently drops this "
                f"recording from every split instead of raising."
            )
        selected = {recording for recording, value in config["split_by_recording"].items() if value == split}
        files = [path for path in recordings if path.name.removesuffix(".traj.parquet") in selected]
    elif config.get("strategy") in PER_WINDOW_SPLIT_STRATEGIES:
        raise PerWindowSplitStrategyUnsupportedError(
            f"split_files cannot express split_strategy={config['strategy']!r}: it assigns "
            f"train/val/test per window within a recording, but split_files only returns whole "
            f"recordings per split. This path (generic scene_membership, i.e. "
            f"scene_membership != 'target_complete') would otherwise silently return every "
            f"recording to every split. Use scene_membership='target_complete', which reads the "
            f"prebuilt per-window index filtered by split, instead."
        )
    else:
        files = [dataset_dir/"converted"/x for x in config[split]]
    if any(not path.exists() for path in files): raise FileNotFoundError("Split references a missing file")
    return files


def _scene_arrays(track_ids, steps, x, y, start_step: int, history_steps: int,
                  future_steps: int, stride: int, primary_track_id: int | None = None) -> dict:
    track_ids,steps,x,y=map(np.asarray,(track_ids,steps,x,y)); total=history_steps+future_steps
    expected=start_step+np.arange(total)*stride; tracks={}; valid_tracks={}
    order=np.lexsort((steps,track_ids)); ordered_ids=track_ids[order]
    groups=np.split(order,np.flatnonzero(np.diff(ordered_ids))+1) if len(order) else []
    for indices in groups:
        track_id=track_ids[indices[0]]
        agent_steps=steps[indices]; positions=np.searchsorted(agent_steps,expected); valid=positions<len(agent_steps)
        vi=np.flatnonzero(valid); valid[vi]=agent_steps[positions[vi]]==expected[vi]
        if not valid.any(): continue
        xy=np.full((total,2),np.nan,np.float32); matched=indices[positions[valid]]
        xy[valid,0],xy[valid,1]=x[matched],y[matched]; tracks[int(track_id)]=xy; valid_tracks[int(track_id)]=valid
    complete=[tid for tid,v in valid_tracks.items() if v.all()]
    if primary_track_id is None:
        if not complete: raise ValueError("Scene has no complete primary")
        primary_track_id=complete[0]
    if primary_track_id not in complete: raise ValueError("Selected primary is incomplete")
    primary_position=tracks[primary_track_id][history_steps-1]
    neighbors=[tid for tid in tracks if tid!=primary_track_id]
    neighbors.sort(key=lambda tid:(float(np.linalg.norm(tracks[tid][history_steps-1]-primary_position)) if valid_tracks[tid][history_steps-1] else float("inf"),tid))
    ordered=[primary_track_id,*neighbors]
    return {"xy":np.stack([tracks[t] for t in ordered],axis=1),
            "valid":np.stack([valid_tracks[t] for t in ordered],axis=1),
            "track_id":np.asarray(ordered,dtype=np.int64)}


def _scene_agent_count(track_ids, steps, start_step: int, history_steps: int,
                        future_steps: int, stride: int) -> int:
    """Number of ordered agent tracks `_scene_arrays` would return for this window.

    Mirrors `_scene_arrays`'s grouping and its `if not valid.any(): continue` filter
    without building the per-agent arrays or resolving a primary track.
    """
    track_ids, steps = np.asarray(track_ids), np.asarray(steps)
    total = history_steps + future_steps
    expected = start_step + np.arange(total) * stride
    order = np.lexsort((steps, track_ids)); ordered_ids = track_ids[order]
    groups = np.split(order, np.flatnonzero(np.diff(ordered_ids)) + 1) if len(order) else []
    count = 0
    for indices in groups:
        agent_steps = steps[indices]; positions = np.searchsorted(agent_steps, expected); valid = positions < len(agent_steps)
        vi = np.flatnonzero(valid); valid[vi] = agent_steps[positions[vi]] == expected[vi]
        if valid.any(): count += 1
    return count


def _index_path(spec: DatasetSpec, data_seed: int, strategy: str = "grouped_scene_sequence", held_out_scene: str | None = None) -> Path:
    suffix = _split_suffix(strategy, held_out_scene)
    return spec.path / "indices" / f"scenes-seed{data_seed}{suffix}.parquet"


def _analysis_index_path(dataset_dir: Path, strategy: str = "grouped_scene_sequence",
                         held_out_scene: str | None = None) -> Path:
    return dataset_dir / "indices" / f"opentraj-analysis-trajlets{_split_suffix(strategy, held_out_scene)}.parquet"


class SceneDataset:
    def __init__(self, spec: DatasetSpec | str | Path, split: str, cfg: Config | None = None, include_rejected: bool = False):
        import pyarrow.parquet as pq
        cfg=cfg or Config()
        if not isinstance(spec,DatasetSpec):
            path=Path(spec); spec=DatasetSpec(path.name,path)
        if split not in SPLITS: raise ValueError(f"split must be one of {SPLITS}")
        self.spec,self.dataset_dir,self.split,self.cfg=spec,spec.path,split,cfg
        self.task=_read_json(cfg.task_path)
        self.history_steps=cfg.forecasting.window.observation_steps
        self.future_steps=cfg.forecasting.window.prediction_steps
        dataset_task=self.task.get("datasets",{}).get(spec.name,{})
        self.canonical_stride=int(dataset_task.get("canonical_stride",self.task["canonical_stride"])); self.total_steps=self.history_steps+self.future_steps
        self.generic_membership = cfg.forecasting.scene_membership != "target_complete"
        if self.generic_membership:
            # enumerate_forecast_windows walks *consecutive* canonical steps, so its
            # windows only span the intended duration at stride 1.
            assert self.canonical_stride == 1, (
                f"scene_membership={cfg.forecasting.scene_membership!r} uses the generic "
                f"enumerate_forecast_windows path, which assumes canonical_stride == 1, but "
                f"{spec.name} declares canonical_stride={self.canonical_stride} in {cfg.task_path}: "
                f"the generic path would read steps start..start+{self.total_steps - 1} instead of "
                f"start..start+{(self.total_steps - 1) * self.canonical_stride}"
            )
            files = split_files(spec.path, split, cfg.seeds.data_seed, cfg.split_strategy, cfg.held_out_scene)
            records = [
                record
                for path in tqdm(files, desc=f"{spec.name}/{split}: enumerating forecast windows", unit="file")
                for record in enumerate_forecast_windows(path, cfg.forecasting)
            ]
            self.rows = [
                {
                    "window_record": record,
                    "recording": record.recording,
                    "scene_id": record.scene_id,
                    "scene_sequence": record.scene_sequence,
                    "start_step": record.start_step,
                    "observation_end_step": record.observation_end_step,
                    "future_end_step": record.future_end_step,
                    "primary_track_id": record.target_agent_id if record.target_agent_id is not None else record.complete_agent_ids[0],
                    "scene_tag": 0,
                    "sub_tags": [],
                }
                for record in records
            ]
        else:
            index_path=_index_path(spec,cfg.seeds.data_seed,cfg.split_strategy,cfg.held_out_scene)
            if (int(self.task["history_steps"]),int(self.task["future_steps"]))!=(self.history_steps,self.future_steps):
                raise ValueError(
                    f"Forecast window mismatch: {cfg.task_path} declares "
                    f"history_steps={self.task['history_steps']}/future_steps={self.task['future_steps']}, "
                    f"but cfg.forecasting.window resolved to history_steps={self.history_steps}/"
                    f"future_steps={self.future_steps}. The index at {index_path} was built from "
                    f"{cfg.task_path}'s window, so a mismatched cfg.forecasting silently slices a "
                    f"population filtered for a different window duration."
                )
            if not index_path.exists(): raise FileNotFoundError(f"Missing {index_path}; run preprocess.py first")
            # Predicates are pushed into Arrow: materializing every candidate window
            # before filtering can consume tens of GiB per training process.
            available=set(pq.ParquetFile(index_path).schema_arrow.names)
            filters=[("split","=",split)]
            if not include_rejected:
                filters.extend(
                    (column,"=",True)
                    for column in ("accepted","track_kept","eval_kept")
                    if column in available
                )
            self.rows=pq.read_table(index_path,filters=filters).to_pylist()
        if spec.fraction < 1.0:
            manifest_path=MANIFEST_ROOT/manifest_name(spec.name,spec.fraction,cfg.seeds.data_seed)
            if not manifest_path.exists():
                raise FileNotFoundError(f"Missing subset manifest {manifest_path}; create it during preprocessing")
            keep=set(load_subset_manifest(manifest_path)["track_ids"])
            self.rows=[row for row in self.rows if f"{row['recording']}:{row['primary_track_id']}" in keep]
        excluded=excluded_recordings(spec.path)
        if excluded:
            # A previously built index still carries rows for a newly excluded
            # recording, so the exclusion holds without forcing an index rebuild.
            self.rows=[row for row in self.rows if row["recording"] not in excluded]
        self.recordings={p.name.removesuffix(".traj.parquet"):p for p in _recordings(spec.path)}
        self._source_centers: dict[str, np.ndarray] = {}
        # Filled lazily, so forked DataLoader workers each own a private cache rather
        # than sharing (and repeatedly copy-on-writing) one.
        self._window_cache = RecordingWindowCache()
        for path in self.recordings.values():
            validate_canonical_file(path)

    def __len__(self): return len(self.rows)

    def agent_count(self, index: int) -> int:
        """Number of agents `self[index].xy.shape[1]` would report, without building a `Scene`.

        Read-only: does not touch `_source_centers` or advance any RNG.
        """
        row = self.rows[index]
        if self.generic_membership:
            record = row["window_record"]
            agent_ids = record.complete_agent_ids or (
                (record.target_agent_id,) if record.target_agent_id is not None else ()
            )
            return len(agent_ids)
        path = self.recordings[row["recording"]]
        final = row["start_step"] + (self.total_steps - 1) * self.canonical_stride
        track_ids, steps, _x, _y = self._window_cache.window(path, row["start_step"], final)
        return _scene_agent_count(
            track_ids, steps, row["start_step"], self.history_steps, self.future_steps, self.canonical_stride
        )

    def _source_center(self, recording: str) -> np.ndarray:
        import pyarrow.parquet as pq
        if recording not in self._source_centers:
            table = pq.read_table(self.recordings[recording], columns=["x", "y"])
            self._source_centers[recording] = np.array(
                [np.mean(table["x"].to_numpy()), np.mean(table["y"].to_numpy())],
                dtype=np.float32,
            )
        return self._source_centers[recording]

    def __getitem__(self,index:int)->Scene:
        row=self.rows[index]; path=self.recordings[row["recording"]]
        final=row["start_step"]+(self.total_steps-1)*self.canonical_stride
        if self.generic_membership:
            window_xy, window_valid = materialize_forecast_window(row["window_record"], self._window_cache)
            record = row["window_record"]
            ids = np.asarray(record.complete_agent_ids, dtype=np.int64)
            arrays = {"xy": window_xy, "valid": window_valid, "track_id": ids}
        else:
            arrays=_scene_arrays(*self._window_cache.window(path,row["start_step"],final),row["start_step"],self.history_steps,self.future_steps,self.canonical_stride,int(row["primary_track_id"]))
        source_center = (
            self._source_center(row["recording"])
            if self.cfg.forecasting.coordinates.frame == "scene_mean_center"
            else None
        )
        representation=prepare_forecast_representation(
            arrays["xy"], arrays["valid"], self.cfg.forecasting, dt=CANONICAL_DT,
            source_scene_center=source_center,
            training=self.split=="train",
            rng=np.random.default_rng(np.random.SeedSequence([self.cfg.seeds.model_seed,index])),
        )
        rotation,center=representation.transform.rotation,representation.transform.origin
        ids=arrays["track_id"]
        return Scene(representation.xy,f"{row['recording']}:{ids[0]}",[f"{row['recording']}:{v}" for v in ids[1:]],self.spec.name,row["recording"],int(row["start_step"]),int(row["scene_tag"]),list(row["sub_tags"]),arrays["valid"],rotation,center,
            str(row.get("scene_id",row["recording"])),str(row.get("scene_sequence",row["recording"])),
            int(row.get("observation_end_step",row["start_step"]+self.history_steps-1)),int(row.get("future_end_step",final)),
            np.array(arrays["xy"],copy=True),np.array(representation.target_world_xy,copy=True),representation.transform.heading_source,
            self.cfg.forecasting.signal.input_mode,self.cfg.forecasting.signal.target_mode,
            representation.features, representation.relative_xy, representation.feature_names,
            representation.transform.frame, self.cfg.forecasting.forecast_preprocessing_hash,
            representation.augmentation)


def scene_collate(scenes:list[Scene])->dict:
    import torch
    if not scenes: raise ValueError("Cannot collate an empty batch")
    batch,time,max_agents=len(scenes),scenes[0].xy.shape[0],max(s.xy.shape[1] for s in scenes)
    xy=torch.full((batch,time,max_agents,2),float("nan")); valid=torch.zeros((batch,time,max_agents),dtype=torch.bool); agent_mask=torch.zeros((batch,max_agents),dtype=torch.bool)
    feature_dim=scenes[0].features.shape[-1] if scenes[0].features is not None else 2
    features=torch.full((batch,time,max_agents,feature_dim),float("nan"))
    relative_xy=torch.full((batch,time,max_agents,2),float("nan")) if scenes[0].relative_xy is not None else None
    for i,scene in enumerate(scenes):
        n=scene.xy.shape[1]; xy[i,:,:n]=torch.from_numpy(scene.xy.copy()); valid[i,:,:n]=torch.from_numpy(scene.valid.copy()); agent_mask[i,:n]=True
        features[i,:,:n]=torch.from_numpy((scene.features if scene.features is not None else scene.xy).copy())
        if relative_xy is not None and scene.relative_xy is not None: relative_xy[i,:,:n]=torch.from_numpy(scene.relative_xy.copy())
    return {"xy":xy,"features":features,"relative_xy":relative_xy,"feature_names":scenes[0].feature_names,"valid":valid,"agent_mask":agent_mask,"primary_track_id":[s.primary_track_id for s in scenes],"neighbor_track_ids":[s.neighbor_track_ids for s in scenes],"dataset":[s.dataset for s in scenes],"scene_id":[s.scene_id for s in scenes],"scene_sequence":[s.scene_sequence for s in scenes],"recording":[s.recording for s in scenes],"start_step":torch.tensor([s.start_step for s in scenes]),"observation_end_step":torch.tensor([s.observation_end_step for s in scenes]),"future_end_step":torch.tensor([s.future_end_step for s in scenes]),"scene_tag":torch.tensor([s.scene_tag for s in scenes]),"sub_tags":[s.sub_tags for s in scenes],"rotation":torch.tensor([s.rotation for s in scenes]),"center":torch.from_numpy(np.stack([s.center for s in scenes])),"normalization_heading_source":[s.normalization_heading_source for s in scenes],"forecast_input_preprocessing":[s.forecast_input_preprocessing for s in scenes],"forecast_target_preprocessing":[s.forecast_target_preprocessing for s in scenes],"coordinate_frame":[s.coordinate_frame for s in scenes],"forecast_preprocessing_hash":[s.forecast_preprocessing_hash for s in scenes],"augmentation":[s.augmentation for s in scenes]}


def seed_worker(worker_id:int)->None:
    import torch
    seed=torch.initial_seed()%2**32; np.random.seed(seed); random.seed(seed)


def make_loaders(train_datasets:list[DatasetSpec],eval_dataset:DatasetSpec,cfg:Config):
    import torch
    from torch.utils.data import ConcatDataset,DataLoader,WeightedRandomSampler
    names={d.name for d in train_datasets}
    steps_names={"steps"}
    if eval_dataset.name.lower() in steps_names and {name.lower() for name in names} & steps_names:
        raise ValueError("STEPS cannot be both training and evaluation data")
    if not train_datasets: raise ValueError("At least one training dataset is required")
    eval_source=_read_json(metadata_path(eval_dataset.path,"source.json"))
    eval_locations=set(eval_source.get("same_location_groups",[]))
    for train_spec in train_datasets:
        train_source=_read_json(metadata_path(train_spec.path,"source.json"))
        overlap=eval_locations&set(train_source.get("same_location_groups",[]))
        if overlap and train_spec.name!=eval_dataset.name:
            raise ValueError(
                "Location leakage between training and evaluation datasets: "
                f"{train_spec.name} and {eval_dataset.name} share {sorted(overlap)}"
            )
    datasets=[SceneDataset(s,"train",cfg) for s in train_datasets]; combined=ConcatDataset(datasets); weights=[]
    for spec,dataset in zip(train_datasets,datasets): weights.extend([spec.weight/max(1,len(dataset))]*len(dataset))
    generator=torch.Generator().manual_seed(cfg.seeds.model_seed); sampler=WeightedRandomSampler(weights,len(weights),replacement=True,generator=generator)
    common=dict(batch_size=cfg.batch_size,num_workers=cfg.num_workers,collate_fn=scene_collate,worker_init_fn=seed_worker)
    return (DataLoader(combined,sampler=sampler,generator=generator,**common),DataLoader(SceneDataset(eval_dataset,"val",cfg),shuffle=False,**common),DataLoader(SceneDataset(eval_dataset,"test",cfg),shuffle=False,**common))

