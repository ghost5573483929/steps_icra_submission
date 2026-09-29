import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from datasets_preprocessing.ucy import convert as ucy


ROOT = Path(__file__).resolve().parents[1]


class UCYConverterTests(unittest.TestCase):
    def test_benchmark_sequence_whitelist(self):
        self.assertEqual(
            ucy.BENCHMARK_SEQUENCES,
            ("crowds_zara01", "crowds_zara02", "students001", "students003"),
        )
        self.assertTrue(
            {"arxiepiskopi1", "crowds_zara03", "uni_examples"}.isdisjoint(
                ucy.BENCHMARK_SEQUENCES
            )
        )
        self.assertEqual(
            set(ucy.PUBLICATION_SEQUENCES),
            {
                "arxiepiskopi1", "crowds_zara01", "crowds_zara02",
                "crowds_zara03", "students001", "students003", "uni_examples",
            },
        )

    def test_parse_and_interpolate(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tiny.vsp"
            path.write_text("1 - the number of splines\n2 - Num of control points\n0 0 0 0 - point\n100 0 20 90 - point\n")
            splines = ucy.parse_vsp(path)
            rows = ucy.interpolate_spline(splines[0], 7)
            self.assertEqual(rows[0][:2], (7, 1))
            self.assertAlmostEqual(rows[0][2], 50.0)
            self.assertAlmostEqual(rows[0][3], 0.0)
            self.assertTrue(np.isnan(rows[0][4]))

    def test_real_conversion_schema_and_alignment(self):
        source = ROOT / "datasets/ucy/raw/extracted/crowds_zara01.vsp"
        if not source.exists(): self.skipTest("UCY raw archive has not been extracted")
        with tempfile.TemporaryDirectory() as tmp:
            output = ucy.convert_vsp(source, Path(tmp) / "zara.traj.parquet")
            table = pq.read_table(output)
            self.assertEqual(table.schema.remove_metadata(), ucy.OUTPUT_SCHEMA)
            steps = table["step"].to_numpy()
            self.assertTrue(np.all(np.diff(np.unique(steps)) >= 1))
            keys = set(zip(table["track_id"].to_pylist(), table["step"].to_pylist()))
            self.assertEqual(len(keys), table.num_rows)
            self.assertEqual(table.schema.metadata[b"dataset"], b"ucy")

    def test_arxiepiskopi_uses_metric_coordinates(self):
        source=ROOT/"datasets/ucy/raw/extracted/arxiepiskopi1.vsp"
        if not source.exists():self.skipTest("UCY raw archive has not been extracted")
        with tempfile.TemporaryDirectory() as tmp:
            table=pq.read_table(ucy.convert_vsp(source,Path(tmp)/"arx.traj.parquet"))
            self.assertEqual(table.schema.metadata[b"coordinate_scale"],b"OpenTraj homography to metric world coordinates")
            self.assertGreater(float(np.ptp(table["x"].to_numpy())),10.0)

    def test_long_gap_does_not_overshoot(self):
        spline = [
            ucy.ControlPoint(0, 0, 0, 0),
            ucy.ControlPoint(100, 0, 10, 90),
            ucy.ControlPoint(101, 0, 1000, 90),
        ]
        rows = ucy.interpolate_spline(spline, 0)
        self.assertLessEqual(max(row[2] for row in rows), 101)


if __name__ == "__main__": unittest.main()
