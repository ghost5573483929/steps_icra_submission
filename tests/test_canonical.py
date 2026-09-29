import tempfile
import unittest
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from canonical import (
    CANONICAL_SCHEMA,
    canonical_metadata,
    schema_with_metadata,
    validate_canonical_file,
)


class CanonicalContractTests(unittest.TestCase):
    def test_required_metadata_cannot_be_overridden(self):
        with self.assertRaisesRegex(ValueError,"canonical fields"):
            canonical_metadata(dataset="x",scene_id="s",scene_sequence="q",recording="r",extra={"canonical_hz":10})

    def test_v3_schema_is_exact_and_heading_nullable(self):
        metadata=canonical_metadata(dataset="x",scene_id="s",scene_sequence="q",recording="r")
        schema=schema_with_metadata(metadata)
        self.assertEqual(schema.remove_metadata(),CANONICAL_SCHEMA)
        self.assertTrue(schema.field("heading").nullable)

    def _write_canonical_parquet(self, path: Path, steps: list[int]) -> None:
        metadata = canonical_metadata(
            dataset="x", scene_id="s", scene_sequence="q", recording="r"
        )
        schema = schema_with_metadata(metadata)
        n = len(steps)
        table = pa.table(
            {
                "track_id": pa.array([0] * n, type=pa.int32()),
                "step": pa.array(steps, type=pa.int32()),
                "x": pa.array([0.0] * n, type=pa.float32()),
                "y": pa.array([0.0] * n, type=pa.float32()),
                "heading": pa.array([None] * n, type=pa.float32()),
            },
            schema=schema,
        )
        pq.write_table(table, path)

    def test_validate_canonical_file_rejects_out_of_order_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording.traj.parquet"
            self._write_canonical_parquet(path, [0, 1, 2, 20, 7, 7])
            with self.assertRaisesRegex(ValueError, "out of order"):
                validate_canonical_file(path)

    def test_validate_canonical_file_accepts_sorted_steps(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recording.traj.parquet"
            self._write_canonical_parquet(path, [0, 1, 1, 2, 5, 5, 8])
            identity = validate_canonical_file(path)
            self.assertEqual(identity.recording, "r")


if __name__=="__main__":unittest.main()
