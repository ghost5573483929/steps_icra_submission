import json
import tempfile
import unittest
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import data
from config import Config, DatasetSpec, SeedConfig
from data import SceneDataset, _recordings, _scene_arrays, create_splits, make_loaders, prepare_splits
from forecasting_config import resolve_forecasting_config
from manifests import manifest_name, write_subset_manifest
from splits import assign_split
from canonical import CANONICAL_SCHEMA, canonical_metadata, schema_with_metadata
from split_manifests import _split_counts, assert_no_sequence_leakage, loso_manifest
from preprocess import INDEX_SCHEMA
from conftest import write_canonical_fixture


class SceneTests(unittest.TestCase):
    def test_scene_arrays_raises_without_a_complete_primary_track(self):
        ids = np.array([1, 1, 1])
        steps = np.array([0, 1, 3])
        x = np.zeros(3, dtype=np.float32)
        y = np.zeros(3, dtype=np.float32)
        with self.assertRaisesRegex(ValueError, "no complete primary"):
            _scene_arrays(ids, steps, x, y, start_step=0, history_steps=2, future_steps=2, stride=1)

    def test_primary_is_zero_and_neighbors_sort_by_observation_distance(self):
        expected=np.arange(20)
        ids=np.concatenate((np.full(20,7),np.full(20,2),np.full(8,9)))
        steps=np.concatenate((expected,expected,expected[:8]))
        x=np.concatenate((np.arange(20),np.arange(20)+2,np.arange(8)+20)).astype(np.float32)
        y=np.zeros_like(x)
        scene=_scene_arrays(ids,steps,x,y,0,8,12,1,primary_track_id=7)
        self.assertEqual(scene["xy"].shape,(20,3,2))
        np.testing.assert_array_equal(scene["track_id"],[7,2,9])
        self.assertTrue(scene["valid"][:,0].all())
        self.assertFalse(scene["valid"][8:,2].any())

    def test_generic_preset_loads_a_complete_scene_end_to_end(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            converted = root / "converted"
            converted.mkdir()
            rows = [
                (track_id, step, float(step + track_id), float(track_id), None)
                for step in range(20) for track_id in (1, 2)
            ]
            schema = schema_with_metadata(canonical_metadata(
                dataset="test", scene_id="scene", scene_sequence="sequence",
                recording="recording",
            ))
            pq.write_table(pa.Table.from_arrays([
                pa.array([row[0] for row in rows], pa.int32()),
                pa.array([row[1] for row in rows], pa.int32()),
                pa.array([row[2] for row in rows], pa.float32()),
                pa.array([row[3] for row in rows], pa.float32()),
                pa.array([row[4] for row in rows], pa.float32()),
            ], schema=schema), converted / "recording.traj.parquet")
            (root / "source.json").write_text(json.dumps({"canonical_recordings": ["recording"]}))
            prepare_splits(root, 42, strategy="grouped_scene_sequence")
            forecasting = resolve_forecasting_config(preset="social_gan")
            dataset = SceneDataset(
                DatasetSpec("test", root), "train", Config(forecasting=forecasting)
            )
            self.assertEqual(len(dataset), 1)
            scene = dataset[0]
            self.assertEqual(scene.xy.shape, (20, 2, 2))
            self.assertEqual(scene.features.shape, (20, 2, 4))
            self.assertEqual(scene.coordinate_frame, "world")
            self.assertEqual(scene.forecast_preprocessing_hash, forecasting.forecast_preprocessing_hash)


class SubsetManifestSceneDatasetTests(unittest.TestCase):
    @staticmethod
    def _build_dataset(root: Path) -> None:
        # With seed=42 and the default 0.8/0.1/0.1 ratios, assign_grouped_sequences
        # assigns train={recording_b, recording_d}, val={recording_c}, test={recording_a}.
        converted = root / "converted"
        converted.mkdir()
        recordings = ["recording_a", "recording_b", "recording_c", "recording_d"]
        for index, recording in enumerate(recordings):
            offset = index * 100
            tracks = {
                1: [(step, float(step) + offset, 0.0, None) for step in range(40)],
                2: [(step, float(step) + 2 + offset, 0.0, None) for step in range(40)],
            }
            write_canonical_fixture(
                converted / f"{recording}.traj.parquet", tracks,
                dataset="probe", scene_id="scene", scene_sequence=recording, recording=recording,
            )
        (root / "source.json").write_text(
            json.dumps({"canonical_recordings": recordings})
        )
        prepare_splits(root, 42, strategy="grouped_scene_sequence")

    def test_fraction_filter_keeps_exactly_the_manifest_track_ids_and_requires_a_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._build_dataset(root)
            forecasting = resolve_forecasting_config(preset="social_gan")
            cfg = Config(
                forecasting=forecasting, seeds=SeedConfig(42, 42),
                split_strategy="grouped_scene_sequence",
            )
            manifest_root = root / "fake_manifests"
            manifest_root.mkdir()
            real_manifest_root = data.MANIFEST_ROOT
            before = (
                sorted(p.name for p in real_manifest_root.iterdir())
                if real_manifest_root.exists() else None
            )

            with patch.object(data, "MANIFEST_ROOT", manifest_root):
                full = SceneDataset(DatasetSpec("probe", root, fraction=1.0), "train", cfg)
                self.assertEqual(len(full), 42)
                self.assertEqual(
                    {row["recording"] for row in full.rows}, {"recording_b", "recording_d"}
                )

                with self.assertRaisesRegex(FileNotFoundError, "subset manifest"):
                    SceneDataset(DatasetSpec("probe", root, fraction=0.5), "train", cfg)

                write_subset_manifest(
                    manifest_root / manifest_name("probe", 0.5, 42),
                    "probe", 0.5, 42, ["recording_b:1"], 0, "sha256:x",
                )
                subset = SceneDataset(DatasetSpec("probe", root, fraction=0.5), "train", cfg)
                self.assertEqual(len(subset), 21)
                self.assertTrue(all(row["recording"] == "recording_b" for row in subset.rows))
                self.assertTrue(all(row["primary_track_id"] == 1 for row in subset.rows))

            after = (
                sorted(p.name for p in real_manifest_root.iterdir())
                if real_manifest_root.exists() else None
            )
            self.assertEqual(before, after)


class IndexBackedSceneDatasetTests(unittest.TestCase):
    OBSERVATION_STEPS, PREDICTION_STEPS = 8, 12
    TOTAL_STEPS = OBSERVATION_STEPS + PREDICTION_STEPS

    @staticmethod
    def index_row(**overrides):
        row = {
            "dataset": "synthetic", "scene_id": "scene", "scene_sequence": "sequence",
            "recording": "rec", "start_step": 0, "observation_end_step": 7,
            "future_end_step": 19, "target_agent_id": 1, "primary_track_id": 1,
            "split": "train", "scene_tag": 2, "sub_tags": [],
            "accepted_if_train": True, "accepted": True, "track_kept": True, "eval_kept": True,
        }
        row.update(overrides)
        return row

    @staticmethod
    def write_index(root: Path, rows: list[dict], seed: int = 42) -> Path:
        path = root / "indices" / f"scenes-seed{seed}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(rows, schema=INDEX_SCHEMA), path)
        return path

    @classmethod
    def write_dataset(cls, root: Path, tracks) -> None:
        converted = root / "converted"
        converted.mkdir(parents=True, exist_ok=True)
        write_canonical_fixture(
            converted / "rec.traj.parquet", tracks,
            dataset="synthetic", scene_id="scene", scene_sequence="sequence", recording="rec",
        )
        (root / "source.json").write_text(json.dumps({"canonical_recordings": ["rec"]}))

    @classmethod
    def two_complete_tracks(cls, steps: int = 40):
        return {
            track_id: [(step, float(step + 10 * track_id), float(track_id), None)
                       for step in range(steps)]
            for track_id in (1, 2)
        }

    def test_default_preset_uses_the_index_backed_membership_branch(self):
        cfg = Config()
        self.assertEqual(cfg.forecasting.scene_membership, "target_complete")

    def test_realized_membership_filter_keeps_only_fully_retained_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_dataset(root, self.two_complete_tracks())
            rows = [
                self.index_row(start_step=0),
                self.index_row(start_step=4, accepted=False, accepted_if_train=False),
                self.index_row(start_step=8, track_kept=False),
                self.index_row(start_step=12, eval_kept=False),
                self.index_row(start_step=0, primary_track_id=2, target_agent_id=2, split="val"),
                self.index_row(start_step=4, primary_track_id=2, target_agent_id=2, split="test"),
            ]
            self.write_index(root, rows)
            spec = DatasetSpec("synthetic", root)

            # Of the four train rows only the first has all three flags true.
            self.assertEqual(len(SceneDataset(spec, "train", Config())), 1)
            self.assertEqual(len(SceneDataset(spec, "train", Config(), include_rejected=True)), 4)
            self.assertEqual(len(SceneDataset(spec, "val", Config())), 1)
            self.assertEqual(len(SceneDataset(spec, "test", Config())), 1)

    def test_index_backed_scene_has_window_shape_frame_and_primary_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_dataset(root, self.two_complete_tracks())
            self.write_index(root, [self.index_row(start_step=3, observation_end_step=10,
                                                   future_end_step=22)])
            cfg = Config()
            dataset = SceneDataset(DatasetSpec("synthetic", root), "train", cfg)
            self.assertEqual(len(dataset), 1)
            scene = dataset[0]

            self.assertEqual(scene.xy.shape, (self.TOTAL_STEPS, 2, 2))
            self.assertEqual(scene.valid.shape, (self.TOTAL_STEPS, 2))
            self.assertTrue(scene.valid.all())
            self.assertEqual(scene.coordinate_frame, "target_centric_se2_observations_only")
            self.assertEqual(scene.coordinate_frame, cfg.forecasting.coordinates.frame)
            self.assertEqual(scene.primary_track_id, "rec:1")
            self.assertEqual(scene.neighbor_track_ids, ["rec:2"])
            self.assertEqual((scene.start_step, scene.observation_end_step, scene.future_end_step),
                             (3, 10, 22))
            self.assertEqual((scene.scene_id, scene.scene_sequence), ("scene", "sequence"))

    def test_scene_mean_center_uses_the_whole_recording_not_the_window(self):
        #   track 1: steps 0-19 at x = 0..19 (sum 190), y = 0
        #   track 2: steps 20-39 at x = 100 (sum 2000), y = 8
        #   file mean = (2190/40, 160/40) = (54.75, 4.0)
        #   window (steps 0-19, track 1 only) mean = (9.5, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_dataset(root, {
                1: [(step, float(step), 0.0, None) for step in range(20)],
                2: [(step, 100.0, 8.0, None) for step in range(20, 40)],
            })
            self.write_index(root, [self.index_row(split="val")])
            cfg = Config(forecasting=resolve_forecasting_config(preset="trajectron_pp"))
            dataset = SceneDataset(DatasetSpec("synthetic", root), "val", cfg)
            scene = dataset[0]

            self.assertEqual(scene.coordinate_frame, "scene_mean_center")
            np.testing.assert_allclose(scene.center, [54.75, 4.0], rtol=0, atol=1e-6)
            window_mean = scene.canonical_xy.reshape(-1, 2).mean(axis=0)
            np.testing.assert_allclose(window_mean, [9.5, 0.0], rtol=0, atol=1e-6)
            self.assertGreater(float(np.abs(scene.center - window_mean).max()), 1.0)
            self.assertEqual(scene.normalization_heading_source, "source_scene_mean")
            self.assertEqual(scene.xy.shape[:2], (self.TOTAL_STEPS, 1))

    def test_missing_index_reports_how_to_build_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_dataset(root, self.two_complete_tracks())
            with self.assertRaisesRegex(FileNotFoundError, "run preprocess.py first"):
                SceneDataset(DatasetSpec("synthetic", root), "train", Config())

    def test_agent_count_matches_getitem_shape_on_the_target_complete_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_dataset(root, self.two_complete_tracks())
            rows = [self.index_row(start_step=start) for start in (0, 4, 8, 12)]
            self.write_index(root, rows)
            cfg = Config()
            dataset = SceneDataset(DatasetSpec("synthetic", root), "train", cfg)
            self.assertFalse(dataset.generic_membership)
            self.assertEqual(len(dataset), len(rows))
            for index in range(len(dataset)):
                self.assertEqual(dataset.agent_count(index), dataset[index].xy.shape[1])

    def test_agent_count_matches_getitem_shape_on_a_generic_membership_branch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.write_dataset(root, self.two_complete_tracks())
            forecasting = resolve_forecasting_config(preset="social_gan")
            self.assertEqual(forecasting.scene_membership, "social_gan_complete_scene")
            cfg = Config(forecasting=forecasting)
            dataset = SceneDataset(DatasetSpec("synthetic", root), "train", cfg)
            self.assertTrue(dataset.generic_membership)
            self.assertGreater(len(dataset), 0)
            for index in range(len(dataset)):
                self.assertEqual(dataset.agent_count(index), dataset[index].xy.shape[1])


class SplitTests(unittest.TestCase):
    @staticmethod
    def write_recording(converted: Path, name: str, scene: str) -> Path:
        return write_canonical_fixture(
            converted/f"{name}.traj.parquet", {1: [(0, 0.0, 0.0, None)]},
            dataset="test", scene_id=scene, scene_sequence=name, recording=name,
        )

    def test_split_counts_partition_the_full_population(self):
        # 10% val / 10% test of 10 rounds to 1 each.
        ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
        self.assertEqual(_split_counts(10, ratios), {"train": 8, "val": 1, "test": 1})
        for total in (0, 1, 2, 3, 25, 97):
            counts = _split_counts(total, ratios)
            self.assertEqual(sum(counts.values()), total)

    def test_canonical_recording_whitelist_ignores_stale_converted_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"included","scene")
            self.write_recording(converted,"stale","scene")
            (root/"source.json").write_text(json.dumps({"canonical_recordings":["included"]}))
            self.assertEqual([path.name for path in _recordings(root)],["included.traj.parquet"])

    def test_proposal_has_no_construction_side_effect(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); converted=root/"converted"; converted.mkdir()
            for i in range(10):
                self.write_recording(converted,f"r{i}",f"s{i%2}")
            first=create_splits(root,7,strategy="grouped_scene_sequence"); second=create_splits(root,7,strategy="grouped_scene_sequence")
            self.assertEqual(first,second); self.assertFalse((root/"splits.json").exists())
            self.assertEqual(first["strategy"],"grouped_scene_sequence")

    def test_persisted_grouped_manifest_has_three_formats_and_no_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            for index in range(9):self.write_recording(converted,f"seq{index}",f"scene{index%3}")
            path=prepare_splits(root,11,strategy="grouped_scene_sequence")
            self.assertTrue(path.exists());self.assertTrue(path.with_suffix(".csv").exists());self.assertTrue(path.with_suffix(".md").exists())
            manifest=json.loads(path.read_text());assert_no_sequence_leakage(manifest["records"])
            self.assertEqual({row["split"] for row in manifest["records"]},{"train","val","test"})

    def test_persisted_manifest_is_regenerated_when_recordings_are_added(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"first","scene")
            path=prepare_splits(root,5,strategy="grouped_scene_sequence")
            original=json.loads(path.read_text())
            self.assertEqual([row["recording"] for row in original["records"]],["first"])

            self.write_recording(converted,"second","scene")
            with self.assertWarnsRegex(RuntimeWarning,"canonical source set changed"):
                prepare_splits(root,5,strategy="grouped_scene_sequence")
            refreshed=json.loads(path.read_text())
            self.assertEqual(
                {row["recording"] for row in refreshed["records"]},
                {"first","second"},
            )
            self.assertNotEqual(
                original["canonical_source_set_revision"],
                refreshed["canonical_source_set_revision"],
            )

    def test_loso_holds_out_every_sequence_from_physical_scene(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            files=[self.write_recording(converted,"a1","a"),self.write_recording(converted,"a2","a"),self.write_recording(converted,"b1","b"),self.write_recording(converted,"b2","b"),self.write_recording(converted,"c1","c")]
            manifest=loso_manifest(files,held_out_scene="a",seed=4,count_tracks=True)
            self.assertTrue(all(row["split"]=="test" for row in manifest["records"] if row["scene_id"]=="a"))
            self.assertFalse(any(row["split"]=="test" for row in manifest["records"] if row["scene_id"]!="a"))
            self.assertTrue(all(row["number_of_tracks"]==1 for row in manifest["records"]))
            assert_no_sequence_leakage(manifest["records"])

    def test_prepare_splits_rejects_a_corrupt_current_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); converted = root / "converted"; converted.mkdir()
            self.write_recording(converted, "only-recording", "only-scene")
            path = prepare_splits(root, 42, strategy="grouped_scene_sequence")
            manifest = json.loads(path.read_text())
            manifest["records"][0]["split"] = "BOGUS_SPLIT_VALUE"
            path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "split_by_recording"):
                prepare_splits(root, 42, strategy="grouped_scene_sequence")

    def test_config_requires_explicit_loso_scene(self):
        with self.assertRaisesRegex(ValueError,"held_out_scene"):
            Config(split_strategy="leave_one_scene_out")

    def test_temporal_manifest_for_single_recording_is_explicit_and_purged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"only-recording","only-scene")
            path=prepare_splits(root,13,strategy="temporal")
            manifest=json.loads(path.read_text())
            self.assertEqual(manifest["strategy"],"temporal")
            self.assertEqual(manifest["ratios"],{"train":.8,"val":.1,"test":.1})
            self.assertEqual(manifest["guard_steps"],19)
            config=create_splits(root,13,strategy="temporal")
            starts=list(range(1000))
            assigned={split:[start for start in starts if assign_split(config,recording="only-recording",start=start,first=0,last=999,window_span=19)==split] for split in ("train","val","test")}
            self.assertLess(assigned["train"][-1]+19,assigned["val"][0])
            self.assertLess(assigned["val"][-1]+19,assigned["test"][0])

    def test_excluded_recordings_are_removed_from_every_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"keep-one","one-scene")
            self.write_recording(converted,"calibration-bag","one-scene")
            (root/"source.json").write_text(json.dumps({"excluded_recordings":["calibration-bag"]}))
            from data import _recordings, excluded_recordings
            self.assertEqual(excluded_recordings(root),frozenset({"calibration-bag"}))
            names={p.name.removesuffix(".traj.parquet") for p in _recordings(root)}
            self.assertEqual(names,{"keep-one"})

    def test_excluded_recordings_defaults_to_empty_and_rejects_bad_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"keep-one","one-scene")
            from data import excluded_recordings
            (root/"source.json").write_text(json.dumps({}))
            self.assertEqual(excluded_recordings(root),frozenset())
            (root/"source.json").write_text(json.dumps({"excluded_recordings":"calibration-bag"}))
            with self.assertRaises(ValueError):
                excluded_recordings(root)

    def test_held_out_recording_splits_one_file_out_and_divides_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"day-one","one-scene")
            self.write_recording(converted,"day-two","one-scene")
            (root/"splits.json").write_text(json.dumps({
                "strategy":"held_out_recording","held_out_recording":"day-two",
                "ratios":{"train":0.9,"val":0.1}}))
            path=prepare_splits(root,13,strategy="held_out_recording")
            manifest=json.loads(path.read_text())
            self.assertEqual(manifest["strategy"],"held_out_recording")
            self.assertEqual(manifest["held_out_recording"],"day-two")
            self.assertEqual(manifest["guard_steps"],19)
            config=create_splits(root,13,strategy="held_out_recording")
            starts=list(range(1000))
            held=[assign_split(config,recording="day-two",start=s,first=0,last=999,window_span=19) for s in starts]
            self.assertEqual(set(held),{"test"})
            assigned={split:[s for s in starts if assign_split(config,recording="day-one",start=s,first=0,last=999,window_span=19)==split] for split in ("train","val","test")}
            self.assertTrue(assigned["train"] and assigned["val"])
            self.assertEqual(assigned["test"],[])
            self.assertLessEqual(assigned["train"][-1]+19+19,assigned["val"][0]+19)
            self.assertLess(assigned["train"][-1],assigned["val"][0])

    def test_held_out_recording_rejects_an_unknown_recording(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"day-one","one-scene")
            self.write_recording(converted,"day-two","one-scene")
            (root/"splits.json").write_text(json.dumps({
                "strategy":"held_out_recording","held_out_recording":"day-two",
                "ratios":{"train":0.9,"val":0.1}}))
            config=create_splits(root,13,strategy="held_out_recording")
            with self.assertRaises(ValueError):
                assign_split(config,recording="day-three",start=0,first=0,last=999,window_span=19)

    def test_held_out_recording_requires_a_matching_declaration(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);converted=root/"converted";converted.mkdir()
            self.write_recording(converted,"day-one","one-scene")
            self.write_recording(converted,"day-two","one-scene")
            (root/"splits.json").write_text(json.dumps({
                "strategy":"held_out_recording","held_out_recording":"missing-day",
                "ratios":{"train":0.9,"val":0.1}}))
            with self.assertRaises(ValueError):
                prepare_splits(root,13,strategy="held_out_recording")

    def test_unchanged_held_out_recording_declaration_reuses_the_persisted_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); converted = root / "converted"; converted.mkdir()
            self.write_recording(converted, "day-one", "one-scene")
            self.write_recording(converted, "day-two", "one-scene")
            (root / "splits.json").write_text(json.dumps({
                "strategy": "held_out_recording", "held_out_recording": "day-two",
                "ratios": {"train": 0.9, "val": 0.1}}))
            path = prepare_splits(root, 13, strategy="held_out_recording")
            persisted = json.loads(path.read_text())
            self.assertEqual(persisted["held_out_recording"], "day-two")

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                prepare_splits(root, 13, strategy="held_out_recording")

            reused = json.loads(path.read_text())
            self.assertEqual(reused["held_out_recording"], "day-two")

    def test_held_out_recording_change_alone_forces_a_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); converted = root / "converted"; converted.mkdir()
            self.write_recording(converted, "day-one", "one-scene")
            self.write_recording(converted, "day-two", "one-scene")
            (root / "splits.json").write_text(json.dumps({
                "strategy": "held_out_recording", "held_out_recording": "day-two",
                "ratios": {"train": 0.9, "val": 0.1}}))
            path = prepare_splits(root, 13, strategy="held_out_recording")
            original = json.loads(path.read_text())
            self.assertEqual(original["held_out_recording"], "day-two")

            (root / "splits.json").write_text(json.dumps({
                "strategy": "held_out_recording", "held_out_recording": "day-one",
                "ratios": {"train": 0.9, "val": 0.1}}))
            with self.assertWarnsRegex(RuntimeWarning, "held_out_recording changed"):
                prepare_splits(root, 13, strategy="held_out_recording")

            rebuilt = json.loads(path.read_text())
            self.assertEqual(rebuilt["held_out_recording"], "day-one")
            self.assertEqual(
                rebuilt["canonical_source_set_revision"],
                original["canonical_source_set_revision"],
            )

            config = create_splits(root, 13, strategy="held_out_recording")
            self.assertEqual(config["split_by_recording"], {"day-one": "test", "day-two": "train"})

    def test_grand_central_temporal_override_is_applied_only_to_grand_central(self):
        # src/datasets_preprocessing/grand_central/splits.json declares
        # {train: .4, val: .1, test: .5}; other datasets use the shared 0.8/0.1/0.1.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "grand_central"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "grand_central", "grand_central_station")
            path = prepare_splits(root, 13, strategy="temporal")
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["ratios"], {"train": .4, "val": .1, "test": .5})
            self.assertEqual(manifest["guard_steps"], 19)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "not_grand_central"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "only-recording", "only-scene")
            path = prepare_splits(root, 13, strategy="temporal")
            manifest = json.loads(path.read_text())
            self.assertEqual(manifest["ratios"], {"train": .8, "val": .1, "test": .1})

    def test_grand_central_temporal_override_respects_the_guard_at_both_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "grand_central"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "grand_central", "grand_central_station")
            config = create_splits(root, 13, strategy="temporal")
            self.assertEqual(config["ratios"], {"train": .4, "val": .1, "test": .5})
            starts = list(range(1000))
            assigned = {
                split: [
                    start for start in starts
                    if assign_split(config, recording="grand_central", start=start, first=0, last=999, window_span=19) == split
                ]
                for split in ("train", "val", "test")
            }
            self.assertLess(assigned["train"][-1] + 19, assigned["val"][0])
            self.assertLess(assigned["val"][-1] + 19, assigned["test"][0])

    def test_malformed_splits_override_raises_and_names_the_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "grand_central"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "grand_central", "grand_central_station")
            override_path = root / "splits.json"
            for payload, pattern in (
                ({"strategy": "temporal", "ratios": {"train": .4, "val": .1, "test": .5}, "bogus": 1}, "unknown split override key"),
                ({"strategy": "temporal", "ratios": {"train": .4, "val": .1, "test": .4}}, "sum to 1"),
                ({"strategy": "temporal", "ratios": {"train": .4, "val": .1, "test": -.5}}, "positive"),
                ({"strategy": "grouped_scene_sequence", "ratios": {"train": .4, "val": .1, "test": .5}}, "unsupported split override strategy"),
            ):
                override_path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, pattern) as cm:
                    create_splits(root, 13, strategy="temporal")
                self.assertIn(str(override_path), str(cm.exception))

    def test_temporal_override_lookup_is_load_bearing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "grand_central"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "grand_central", "grand_central_station")
            with patch("data._temporal_split_override", return_value=None):
                config = create_splits(root, 13, strategy="temporal")
            self.assertEqual(config["ratios"], {"train": .8, "val": .1, "test": .1})
            self.assertNotEqual(config["ratios"], {"train": .4, "val": .1, "test": .5})

    def test_ratio_change_alone_forces_a_rebuild_with_source_data_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "some_dataset"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "only-recording", "only-scene")
            path = prepare_splits(root, 13, strategy="temporal")
            original = json.loads(path.read_text())
            self.assertEqual(original["ratios"], {"train": .8, "val": .1, "test": .1})

            (root / "splits.json").write_text(json.dumps(
                {"strategy": "temporal", "ratios": {"train": .4, "val": .1, "test": .5}}
            ))
            with self.assertWarnsRegex(RuntimeWarning, "ratios changed"):
                prepare_splits(root, 13, strategy="temporal")
            rebuilt = json.loads(path.read_text())
            self.assertEqual(rebuilt["ratios"], {"train": .4, "val": .1, "test": .5})
            self.assertEqual(
                rebuilt["canonical_source_set_revision"],
                original["canonical_source_set_revision"],
            )

    def test_ratio_change_detection_is_load_bearing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "some_dataset"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "only-recording", "only-scene")
            path = prepare_splits(root, 13, strategy="temporal")

            (root / "splits.json").write_text(json.dumps(
                {"strategy": "temporal", "ratios": {"train": .4, "val": .1, "test": .5}}
            ))
            with patch(
                "data._split_manifest_staleness_reasons",
                lambda manifest, files, strategy, dataset_dir: (
                    [] if manifest.get("canonical_source_set_revision")
                    == data.canonical_source_revision(files) else ["canonical source set changed"]
                ),
            ):
                prepare_splits(root, 13, strategy="temporal")
            unchanged = json.loads(path.read_text())
            self.assertEqual(unchanged["ratios"], {"train": .8, "val": .1, "test": .1})
            self.assertNotEqual(unchanged["ratios"], {"train": .4, "val": .1, "test": .5})

    def test_unchanged_temporal_configuration_reuses_the_persisted_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "some_dataset"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "only-recording", "only-scene")
            path = prepare_splits(root, 13, strategy="temporal")
            persisted = json.loads(path.read_text())
            persisted["sentinel"] = "untouched-by-a-rebuild"
            path.write_text(json.dumps(persisted))

            with warnings.catch_warnings():
                warnings.simplefilter("error", RuntimeWarning)
                prepare_splits(root, 13, strategy="temporal")

            reused = json.loads(path.read_text())
            self.assertEqual(reused.get("sentinel"), "untouched-by-a-rebuild")

    def test_guard_steps_or_strategy_change_forces_a_rebuild(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "some_dataset"; converted = root / "converted"; converted.mkdir(parents=True)
            self.write_recording(converted, "only-recording", "only-scene")
            path = prepare_splits(root, 13, strategy="temporal")
            persisted = json.loads(path.read_text())
            self.assertEqual(persisted["guard_steps"], 19)

            corrupted = dict(persisted); corrupted["guard_steps"] = 999
            path.write_text(json.dumps(corrupted))
            with self.assertWarnsRegex(RuntimeWarning, "guard_steps changed"):
                prepare_splits(root, 13, strategy="temporal")
            rebuilt = json.loads(path.read_text())
            self.assertEqual(rebuilt["guard_steps"], 19)

            corrupted = dict(persisted); corrupted["strategy"] = "grouped_scene_sequence"
            path.write_text(json.dumps(corrupted))
            with self.assertWarnsRegex(RuntimeWarning, "strategy changed"):
                prepare_splits(root, 13, strategy="temporal")
            rebuilt = json.loads(path.read_text())
            self.assertEqual(rebuilt["strategy"], "temporal")

    def test_recording_strategy_dispatch(self):
        config={"strategy":"recording","train":["a.traj.parquet"],"val":["b.traj.parquet"],"test":["c.traj.parquet"]}
        self.assertEqual(assign_split(config,recording="b",start=0,first=0,last=100,window_span=76),"val")
        self.assertIsNone(assign_split(config,recording="d",start=0,first=0,last=100,window_span=76))

    def test_grouped_strategy_dispatch(self):
        config={"strategy":"grouped_scene_sequence","split_by_recording":{"a":"train","b":"val","c":"test"}}
        self.assertEqual(assign_split(config,recording="b",start=0,first=0,last=100,window_span=19),"val")
        self.assertEqual(
            assign_split(config,recording="b.traj.parquet",start=0,first=0,last=100,window_span=19),"val",
        )
        self.assertIsNone(assign_split(config,recording="z",start=0,first=0,last=100,window_span=19))

    def test_loso_mapping_is_recording_level_and_independent_of_the_window(self):
        config={"strategy":"leave_one_scene_out","split_by_recording":{"a1":"test","b1":"train","c1":"val"}}
        for start in (0,5_000):
            for recording,expected in (("a1","test"),("b1","train"),("c1","val")):
                self.assertEqual(
                    assign_split(config,recording=recording,start=start,first=0,last=9_999,window_span=19),
                    expected,msg=f"{recording} at start={start}",
                )
        self.assertEqual(
            assign_split(config,recording="a1",start=17,first=17,last=41,window_span=3),"test",
        )

    def test_loso_falls_back_to_scene_membership_for_unmapped_recordings(self):
        # An explicit split_by_recording entry outranks the scene fallback, so b1
        # is val rather than the train its scene would give it.
        config={
            "strategy":"leave_one_scene_out",
            "split_by_recording":{"b1":"val"},
            "scene_by_recording":{"a1":"a","a2":"a","b1":"b","c1":"c","d1":"d"},
            "test_scene":"a",
            "val_scenes":["c"],
        }
        for recording,expected in (("a1","test"),("a2","test"),("c1","val"),("d1","train"),("b1","val")):
            self.assertEqual(
                assign_split(config,recording=recording,start=0,first=0,last=100,window_span=19),
                expected,msg=recording,
            )

    def test_loso_raises_on_an_unrecognized_recording(self):
        config={"strategy":"leave_one_scene_out","split_by_recording":{},
                "scene_by_recording":{"a1":"a"},"test_scene":"a","val_scenes":["c"]}
        with self.assertRaisesRegex(ValueError,"unknown-recording"):
            assign_split(config,recording="unknown-recording",start=0,first=0,last=100,window_span=19)
        with self.assertRaisesRegex(ValueError,"'a'"):
            assign_split(config,recording="unknown-recording",start=0,first=0,last=100,window_span=19)
        self.assertEqual(
            assign_split(config,recording="a1",start=0,first=0,last=100,window_span=19),
            "test",
        )

    def test_loso_and_grouped_agree_on_a_filename_suffixed_recording(self):
        config = {"strategy": "leave_one_scene_out",
                  "split_by_recording": {"a1": "test"},
                  "scene_by_recording": {"a1": "a"},
                  "test_scene": "a"}
        self.assertEqual(
            assign_split(config, recording="a1.traj.parquet", start=0, first=0,
                         last=100, window_span=19),
            "test",
        )
        grouped_config = dict(config, strategy="grouped_scene_sequence")
        self.assertEqual(
            assign_split(grouped_config, recording="a1.traj.parquet", start=0,
                         first=0, last=100, window_span=19),
            "test",
        )

    def test_unknown_split_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError,"Unknown split strategy: chronological"):
            assign_split({"strategy":"chronological"},recording="a1",start=0,first=0,last=100,window_span=19)

    def test_temporal_splits_are_ordered_and_purged(self):
        starts=list(range(0,10_000,4)); window_span=guard_steps=76
        config={"strategy":"temporal","ratios":{"train":.8,"val":.1,"test":.1},"guard_steps":guard_steps}
        parts={
            split: [
                start for start in starts
                if assign_split(config,recording="r",start=start,first=starts[0],last=starts[-1],window_span=window_span)==split
            ]
            for split in ("train","val","test")
        }
        self.assertLess(parts["train"][-1]+window_span,parts["val"][0])
        self.assertLess(parts["val"][-1]+window_span,parts["test"][0])

    def test_loader_rejects_steps_as_train_and_eval(self):
        spec=DatasetSpec("steps",Path("missing"))
        with self.assertRaisesRegex(ValueError,"both training and evaluation"):
            make_loaders([spec],spec,Config())

    def test_loader_rejects_cross_dataset_location_leakage(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            train=root/"train";evaluation=root/"evaluation"
            train.mkdir();evaluation.mkdir()
            (train/"source.json").write_text(json.dumps({"same_location_groups":["Assisi"]}))
            (evaluation/"source.json").write_text(json.dumps({"same_location_groups":["Assisi"]}))
            with self.assertRaisesRegex(ValueError,"Location leakage"):
                make_loaders(
                    [DatasetSpec("steps_comparative",train)],
                    DatasetSpec("steps_seasonal",evaluation),
                    Config(),
                )


class ForecastWindowMismatchGuardTests(unittest.TestCase):
    ETH = Path(__file__).resolve().parents[1] / "datasets" / "eth"

    def _require_eth_index(self):
        index = self.ETH / "indices" / "scenes-seed42.parquet"
        if not index.exists():
            self.skipTest(f"ETH forecast index missing at {index}; regenerate before running this test")

    def test_mismatched_window_raises_naming_both_sources_and_the_index_path(self):
        self._require_eth_index()
        cfg = Config(forecasting=resolve_forecasting_config(
            preset="steps_default", cli_overrides={"window.observation_steps": 6}
        ))
        with self.assertRaises(ValueError) as ctx:
            SceneDataset(self.ETH, "train", cfg)
        message = str(ctx.exception)
        self.assertIn("history_steps=8", message)
        self.assertIn("future_steps=12", message)
        self.assertIn("history_steps=6", message)
        self.assertIn(str(cfg.task_path), message)
        self.assertIn(str(self.ETH / "indices" / "scenes-seed42.parquet"), message)

    def test_default_8_plus_12_window_is_unaffected_by_the_guard(self):
        self._require_eth_index()
        cfg = Config(forecasting=resolve_forecasting_config(preset="steps_default"))
        ds = SceneDataset(self.ETH, "train", cfg)
        self.assertEqual(ds.total_steps, 20)
        self.assertEqual(len(ds), 514)

    def test_generic_membership_presets_still_honour_the_window_override(self):
        self._require_eth_index()
        scholler = Config(forecasting=resolve_forecasting_config(
            preset="scholler", cli_overrides={"window.observation_steps": 6}
        ))
        ds_scholler = SceneDataset(self.ETH, "train", scholler)
        self.assertEqual(len(ds_scholler), 1512)

        social_gan = Config(forecasting=resolve_forecasting_config(
            preset="social_gan", cli_overrides={"window.observation_steps": 6}
        ))
        ds_social_gan = SceneDataset(self.ETH, "train", social_gan)
        self.assertEqual(len(ds_social_gan), 366)


class PerWindowSplitStrategyGuardTests(unittest.TestCase):
    @staticmethod
    def _write_two_recordings(root: Path) -> None:
        converted = root / "converted"
        converted.mkdir()
        for recording, offset in (("recording_a", 0), ("recording_b", 100)):
            tracks = {
                1: [(step, float(step) + offset, 0.0, None) for step in range(40)],
                2: [(step, float(step) + 2 + offset, 0.0, None) for step in range(40)],
            }
            write_canonical_fixture(
                converted / f"{recording}.traj.parquet", tracks,
                dataset="probe", scene_id="scene", scene_sequence=recording, recording=recording,
            )
        (root / "source.json").write_text(
            json.dumps({"canonical_recordings": ["recording_a", "recording_b"]})
        )

    def test_temporal_generic_membership_raises_instead_of_returning_every_window_to_every_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_two_recordings(root)
            prepare_splits(root, 42, strategy="temporal")
            forecasting = resolve_forecasting_config(preset="social_gan")
            cfg = Config(forecasting=forecasting, seeds=SeedConfig(42, 42), split_strategy="temporal")
            for split in ("train", "val", "test"):
                with self.assertRaisesRegex(
                    data.PerWindowSplitStrategyUnsupportedError, "temporal"
                ):
                    SceneDataset(DatasetSpec("probe", root, fraction=1.0), split, cfg)

    def test_held_out_recording_generic_membership_raises_named_error_not_bare_keyerror(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._write_two_recordings(root)
            (root / "splits.json").write_text(json.dumps({
                "strategy": "held_out_recording",
                "held_out_recording": "recording_b",
                "ratios": {"train": 0.8, "val": 0.2},
            }))
            with self.assertRaisesRegex(
                data.PerWindowSplitStrategyUnsupportedError, "held_out_recording"
            ):
                data.split_files(root, "train", 42, "held_out_recording")

    def test_grouped_scene_sequence_generic_membership_is_unaffected(self):
        # With seed=42 and the default 0.8/0.1/0.1 ratios, assign_grouped_sequences
        # assigns train={recording_b, recording_d}, val={recording_c}, test={recording_a}.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            converted = root / "converted"
            converted.mkdir()
            recordings = ["recording_a", "recording_b", "recording_c", "recording_d"]
            for index, recording in enumerate(recordings):
                offset = index * 100
                tracks = {
                    1: [(step, float(step) + offset, 0.0, None) for step in range(40)],
                    2: [(step, float(step) + 2 + offset, 0.0, None) for step in range(40)],
                }
                write_canonical_fixture(
                    converted / f"{recording}.traj.parquet", tracks,
                    dataset="probe", scene_id="scene", scene_sequence=recording, recording=recording,
                )
            (root / "source.json").write_text(json.dumps({"canonical_recordings": recordings}))
            prepare_splits(root, 42, strategy="grouped_scene_sequence")
            forecasting = resolve_forecasting_config(preset="social_gan")
            cfg = Config(
                forecasting=forecasting, seeds=SeedConfig(42, 42),
                split_strategy="grouped_scene_sequence",
            )
            train = SceneDataset(DatasetSpec("probe", root, fraction=1.0), "train", cfg)
            val = SceneDataset(DatasetSpec("probe", root, fraction=1.0), "val", cfg)
            test = SceneDataset(DatasetSpec("probe", root, fraction=1.0), "test", cfg)
            self.assertEqual(len(train), 42)
            self.assertEqual({row["recording"] for row in train.rows}, {"recording_b", "recording_d"})
            self.assertEqual(len(val), 21)
            self.assertEqual({row["recording"] for row in val.rows}, {"recording_c"})
            self.assertEqual(len(test), 21)
            self.assertEqual({row["recording"] for row in test.rows}, {"recording_a"})

    def test_leave_one_scene_out_generic_membership_is_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            converted = root / "converted"
            converted.mkdir()
            for recording, offset in (("recording_a", 0), ("recording_b", 100)):
                tracks = {
                    1: [(step, float(step) + offset, 0.0, None) for step in range(40)],
                    2: [(step, float(step) + 2 + offset, 0.0, None) for step in range(40)],
                }
                write_canonical_fixture(
                    converted / f"{recording}.traj.parquet", tracks,
                    dataset="probe", scene_id=recording, scene_sequence=recording, recording=recording,
                )
            (root / "source.json").write_text(
                json.dumps({"canonical_recordings": ["recording_a", "recording_b"]})
            )
            prepare_splits(root, 42, strategy="leave_one_scene_out", held_out_scene="recording_a")
            forecasting = resolve_forecasting_config(preset="social_gan")
            cfg = Config(
                forecasting=forecasting, seeds=SeedConfig(42, 42),
                split_strategy="leave_one_scene_out", held_out_scene="recording_a",
            )
            train = SceneDataset(DatasetSpec("probe", root, fraction=1.0), "train", cfg)
            test = SceneDataset(DatasetSpec("probe", root, fraction=1.0), "test", cfg)
            self.assertEqual(len(train), 21)
            self.assertEqual({row["recording"] for row in train.rows}, {"recording_b"})
            self.assertEqual(len(test), 21)
            self.assertEqual({row["recording"] for row in test.rows}, {"recording_a"})

    def test_target_complete_membership_never_reaches_split_files_so_is_unaffected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            converted = root / "converted"
            converted.mkdir()
            write_canonical_fixture(
                converted / "rec.traj.parquet",
                {
                    1: [(step, float(step + 10), 0.0, None) for step in range(40)],
                    2: [(step, float(step + 20), 0.0, None) for step in range(40)],
                },
                dataset="synthetic", scene_id="scene", scene_sequence="sequence", recording="rec",
            )
            (root / "source.json").write_text(json.dumps({"canonical_recordings": ["rec"]}))
            row = {
                "dataset": "synthetic", "scene_id": "scene", "scene_sequence": "sequence",
                "recording": "rec", "start_step": 0, "observation_end_step": 7,
                "future_end_step": 19, "target_agent_id": 1, "primary_track_id": 1,
                "split": "train", "scene_tag": 2, "sub_tags": [],
                "accepted_if_train": True, "accepted": True, "track_kept": True, "eval_kept": True,
            }
            index_path = root / "indices" / "scenes-seed42.parquet"
            index_path.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(pa.Table.from_pylist([row], schema=INDEX_SCHEMA), index_path)
            cfg = Config()
            self.assertEqual(cfg.forecasting.scene_membership, "target_complete")
            ds = SceneDataset(DatasetSpec("synthetic", root), "train", cfg)
            self.assertEqual(len(ds), 1)
            self.assertEqual(ds.rows[0]["recording"], "rec")


if __name__=="__main__": unittest.main()
