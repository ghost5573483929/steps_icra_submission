import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import orjson
import pyarrow.parquet as pq

from datasets_preprocessing.steps import convert as steps


def _person(track_id=7, *, observed=True, confirmed=True, source="global_kalman_posterior"):
    return {
        "track_id": track_id,
        "confirmed": confirmed,
        "observed_this_frame": observed,
        "state_source": source,
        "stale": not observed,
        "missed_seconds": 0.0 if observed else 0.05,
        "position_ground_xy": [1.0 + track_id, 2.0],
        "velocity_ground_xy_mps": [0.5, 0.0],
        "position_covariance": [[0.1, 0.01], [0.01, 0.2]],
        "spread_m": 0.2,
        "yaw_concentration": 0.9,
        "yaw_ground_rad": 0.25,
        "last_observed_camera_indices": [0, 2],
    }


def _frame(index, people, reason="age_cutoff"):
    return {
        "frame_index": index,
        "timestamp_ns": 1_000_000_000 + index * 50_000_000,
        "timestamp_s": 1.0 + index * 0.05,
        "sync_skew_ns": 100,
        "tracked_people": people,
        "fused_people": [],
        "present_cameras": [0, 1],
        "fresh_cameras": [0, 1],
        "missing_cameras": [],
        "stale_cameras": [],
        "output_finalization": {
            "mode": "fixed_lag",
            "reason": reason,
            "actual_lag_seconds": 1.0 if reason == "age_cutoff" else 0.0,
            "configured_delay_seconds": 1.0,
        },
    }


class StepsIndoorConversionTests(unittest.TestCase):
    def test_filters_predictions_and_end_flush_but_preserves_audit_rows(self):
        frames = [_frame(index, [_person()]) for index in range(17)]
        frames[8] = _frame(8, [_person(), _person(8, observed=False, source="global_kalman_prediction")])
        frames[16] = _frame(16, [_person()], reason="end_of_stream_flush")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "recording_trajectory_data.jsonl"
            source.write_bytes(b"".join(json.dumps(frame).encode() + b"\n" for frame in frames))

            output = steps.convert_file(source, root / "dataset")
            trajectory = pq.read_table(output)
            quality = pq.read_table(root / "dataset/quality/recording.quality.parquet")
            frame_table = pq.read_table(root / "dataset/quality/recording.frames.parquet")
            summary = json.loads((root / "dataset/quality/recording.summary.json").read_text())

            self.assertEqual(trajectory.schema.remove_metadata(), steps.OUTPUT_SCHEMA)
            self.assertEqual(trajectory["step"].to_pylist(), [0, 1])
            self.assertEqual(quality.num_rows, 4)
            self.assertEqual(quality["included_in_trajectory"].to_pylist(), [True, True, False, False])
            self.assertEqual(frame_table["included_in_trajectory"].to_pylist(), [True, True, False])
            self.assertEqual(summary["tracker_predictions_excluded"], 1)
            self.assertEqual(summary["end_of_stream_frames_excluded"], 1)
            self.assertAlmostEqual(summary["source_hz_measured_from_median_dt"], 20.0)

    def test_duplicate_track_in_one_frame_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "duplicate_trajectory_data.jsonl"
            source.write_text(json.dumps(_frame(0, [_person(), _person()])) + "\n")
            with self.assertRaisesRegex(ValueError, "duplicate tracked person"):
                steps.convert_file(source, root / "dataset")


class DecimationKeyScanTests(unittest.TestCase):
    """``_decimation_keys`` returns ``(frame_index, timestamp_ns)`` exactly as a full parse
    would, or ``None`` to request one -- never a different number."""

    def assert_matches_full_parse(self, line):
        frame = orjson.loads(line)
        expected = (int(frame["frame_index"]), int(frame["timestamp_ns"]))
        self.assertEqual(steps._decimation_keys(line), expected, line[:120])

    def assert_falls_back(self, line):
        orjson.loads(line)
        self.assertIsNone(steps._decimation_keys(line), line[:120])

    def assert_never_wrong(self, line):
        frame = orjson.loads(line)
        expected = (int(frame["frame_index"]), int(frame["timestamp_ns"]))
        result = steps._decimation_keys(line)
        self.assertIn(result, (expected, None), line[:120])

    def test_matches_full_parse_for_real_frame_layouts(self):
        nested = {
            "detections": [{"frame_index": -1, "timestamp_ns": 11}, {"timestamp_ns": 12}],
            "per_camera": {"3": {"frame_index": 98, "timestamp_ns": 99}},
        }
        person = {"track_id": 1, "last_observed_timestamp_ns": 77, "position_ground_xy": [1.0, 2.0]}
        cases = [
            {"association": nested, "frame_index": 5, "timestamp_ns": 1234567890123456789,
             "tracked_people": [person]},
            {"tracked_people": [person], "association": nested, "frame_index": 5,
             "timestamp_ns": 1234567890123456789},
            {"a": nested, "b": nested, "timestamp_ns": 42, "frame_index": 5},
            {"frame_index": 0, "deep": [[{"timestamp_ns": 1}], [{"frame_index": 2}]],
             "timestamp_ns": 3},
            {"note": "{{{[[[", "frame_index": 7, "other": "}}}]]]", "timestamp_ns": 8,
             "tail": "}{"},
            {"frame_index": -9, "timestamp_ns": -(2 ** 63)},
            {"frame_index": 2 ** 31, "timestamp_ns": 2 ** 63 - 1},
            {"frame_index": 5, "timestamp_ns": 8},
        ]
        for frame in cases:
            with self.subTest(keys=sorted(frame)):
                self.assert_matches_full_parse(orjson.dumps(frame))
        duplicated = b'{"frame_index": 1, "timestamp_ns": 2, "frame_index": 3, "timestamp_ns": 4}'
        self.assertEqual(orjson.loads(duplicated)["frame_index"], 3)
        self.assert_matches_full_parse(duplicated)

    def test_never_reports_a_wrong_value_behind_many_nested_decoys(self):
        decoys = {"d": [{"frame_index": 900 + i, "timestamp_ns": 900 + i} for i in range(6)]}
        line = orjson.dumps({"timestamp_ns": 42, "frame_index": 5, "tail": decoys})
        self.assertIsNone(steps._decimation_keys(line))
        self.assert_never_wrong(line)

    def test_matches_full_parse_for_unusual_whitespace_and_spacing(self):
        for line in (
            b'{"frame_index":5,"timestamp_ns":6}',
            b'{ "frame_index" : 5 , "timestamp_ns" : 6 }',
            b'{\n  "frame_index": 5,\n  "timestamp_ns": 6\n}\n',
            b'{"frame_index":\t5,\r\n"timestamp_ns":\t6}\n',
        ):
            with self.subTest(line=line):
                self.assert_matches_full_parse(line)

    def test_requests_a_full_parse_when_it_cannot_prove_the_value(self):
        for line in (
            orjson.dumps({"note": 'a "quoted" word', "frame_index": 1, "timestamp_ns": 2}),
            orjson.dumps({"note": "back\\slash", "frame_index": 1, "timestamp_ns": 2}),
            b'{"frame_index": 1.0, "timestamp_ns": 2}',
            b'{"frame_index": "1", "timestamp_ns": 2}',
            b'{"frame_index": 1, "timestamp_ns": true}',
            b'{"frame_index": 1, "timestamp_ns": 2e3}',
            b'{"inner": {"frame_index": 1, "timestamp_ns": 2}}',
            b'[{"frame_index": 1, "timestamp_ns": 2}]',
        ):
            with self.subTest(line=line):
                self.assert_falls_back(line)

    def test_integers_wider_than_int64_are_left_to_the_parser(self):
        # orjson decodes the over-wide literal as a float, so int(frame["timestamp_ns"])
        # is int(1e20) == 100000000000000000000, not the 99999999999999999999 in the bytes.
        line = b'{"frame_index": 1, "timestamp_ns": 99999999999999999999}'
        parsed = orjson.loads(line)["timestamp_ns"]
        self.assertIsInstance(parsed, float)
        self.assertNotEqual(int(parsed), 99999999999999999999)
        self.assertIsNone(steps._decimation_keys(line))
        self.assert_never_wrong(line)


def _write_frames(source, frames):
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(b"".join(json.dumps(frame).encode() + b"\n" for frame in frames))


def _digests(root, recording):
    return {
        name: hashlib.sha256((root / name.format(recording=recording)).read_bytes()).hexdigest()
        for name in (
            "converted/{recording}.traj.parquet",
            "quality/{recording}.quality.parquet",
            "quality/{recording}.frames.parquet",
        )
    }


class DecimatedFrameHandlingTests(unittest.TestCase):
    def test_out_of_order_frame_index_still_raises_on_a_decimated_frame(self):
        frames = [_frame(index, [_person()]) for index in range(12)]
        frames[5]["frame_index"] = 3
        self.assertTrue((3 - 0) % steps.SOURCE_FRAMES_PER_OUTPUT, "frame must be decimated")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "backwards_trajectory_data.jsonl"
            _write_frames(source, frames)
            with self.assertRaisesRegex(ValueError, r":6: frame_index is not strictly increasing \(4 -> 3\)"):
                steps.convert_file(source, root / "dataset")

    def test_adversarial_key_layout_converts_identically_to_the_plain_layout(self):
        plain = [_frame(index, [_person()]) for index in range(17)]
        decoy = {"per_camera": {"3": {"frame_index": 777, "timestamp_ns": 777}},
                 "detections": [{"frame_index": 888, "timestamp_ns": 888}]}
        adversarial = []
        for frame in plain:
            reordered = {"pre_pose_quality_by_camera": decoy, "note": "}{[]"}
            reordered.update({k: v for k, v in frame.items() if k not in ("frame_index", "timestamp_ns")})
            reordered["projected_detections"] = decoy
            reordered["frame_index"] = frame["frame_index"]
            reordered["timestamp_ns"] = frame["timestamp_ns"]
            adversarial.append(reordered)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "recording_trajectory_data.jsonl"
            digests = {}
            summaries = {}
            for label, frames in (("plain", plain), ("adversarial", adversarial)):
                # One source path for both runs: source_file metadata is part of the
                # canonical Parquet, so a per-label path would differ for that reason alone.
                _write_frames(source, frames)
                dataset = root / label
                steps.convert_file(source, dataset, provenance={"git_commit": "x", "git_dirty": False})
                digests[label] = _digests(dataset, "recording")
                summaries[label] = json.loads((dataset / "quality/recording.summary.json").read_text())
            self.assertEqual(digests["plain"], digests["adversarial"])
            self.assertEqual(summaries["plain"], summaries["adversarial"])
            self.assertEqual(summaries["plain"]["source_frames"], 17)
            self.assertEqual(summaries["plain"]["sampled_2p5hz_frames"], 3)

    def test_only_the_sampled_frames_are_fully_parsed(self):
        frames = [_frame(index, [_person()]) for index in range(17)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "recording_trajectory_data.jsonl"
            _write_frames(source, frames)
            with mock.patch.object(steps.orjson, "loads", wraps=orjson.loads) as loads:
                steps.convert_file(source, root / "dataset")
            self.assertEqual(loads.call_count, 3, "one parse per sampled frame, none per skipped frame")

    def test_blank_lines_between_frames_are_skipped(self):
        frames = [_frame(index, [_person()]) for index in range(9)]
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "blank_trajectory_data.jsonl"
            source.write_bytes(
                b"\n".join(json.dumps(frame).encode() for frame in frames).replace(b"\n", b"\n\n   \n")
                + b"\n"
            )
            steps.convert_file(source, root / "dataset")
            summary = json.loads((root / "dataset/quality/blank.summary.json").read_text())
            self.assertEqual(summary["source_frames"], 9)
            self.assertEqual(summary["sampled_2p5hz_frames"], 2)


if __name__ == "__main__":
    unittest.main()
