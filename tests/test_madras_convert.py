import tempfile
import unittest
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from datasets_preprocessing.madras import convert as madras

ROOT=Path(__file__).resolve().parents[1]


class MadrasConverterTests(unittest.TestCase):
    def test_topview_downsampling_and_global_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"TopView_test.txt"
            path.write_text("# recorded at 30 fps\n# id frame x/m y/m z/m id_global t/s x_RGF y_RGF\n1 0 0 0 1.8 1001 0 0 0\n1 1 1 0 1.8 1001 .033 1 0\n1 12 3 0 1.8 1001 .4 3 0\n")
            rows,metadata=madras.read_rows(path)
            self.assertEqual(rows,[(1001,0,0.,0.),(1001,1,3.,0.)]); self.assertEqual(metadata["source_hz"],"30")

    def test_real_file_schema(self):
        source=ROOT/"datasets/madras/raw/Data_Madras/TopView_trajectories/TopView_2F.txt"
        if not source.exists(): self.skipTest("MADRAS raw files unavailable")
        with tempfile.TemporaryDirectory() as tmp:
            output=madras.convert_file(source,Path(tmp)/"sample.parquet"); table=pq.read_table(output)
            self.assertEqual(table.schema.remove_metadata(),madras.OUTPUT_SCHEMA)
            self.assertTrue(np.all(np.diff(np.sort(np.unique(table["step"].to_numpy())))>=1))
            self.assertEqual(table.schema.metadata[b"identity"],b"TopView uses id_global; LargeView uses id")


if __name__=="__main__": unittest.main()
