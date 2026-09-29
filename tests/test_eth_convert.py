import unittest

import numpy as np

from datasets_preprocessing.eth import convert as eth_convert


class EthConversionTests(unittest.TestCase):
    def test_native_video_frames_map_to_canonical_timeline(self):
        rows = [
            (780, 1, 8.0, 3.0, 1.0, 0.0),
            (786, 1, 8.4, 3.0, 0.0, 1.0),
            (792, 1, 8.4, 3.4, 0.0, 0.0),
        ]

        converted, source_stride, first_frame = eth_convert._canonical_rows(rows)

        self.assertEqual(source_stride, 6)
        self.assertEqual(first_frame, 780)
        self.assertEqual([row[0] for row in converted], [0, 1, 2])
        self.assertAlmostEqual(converted[0][4], 0.0)
        self.assertAlmostEqual(converted[1][4], np.pi / 2)
        self.assertIsNone(converted[2][4])

    def test_duplicate_agent_at_timestep_is_rejected(self):
        rows = [
            (1, 4, 0.0, 0.0, 1.0, 0.0),
            (1, 4, 1.0, 0.0, 1.0, 0.0),
            (11, 4, 2.0, 0.0, 1.0, 0.0),
        ]
        with self.assertRaisesRegex(ValueError, "Duplicate pedestrian"):
            eth_convert._canonical_rows(rows)

    def test_dominant_stride_tolerates_a_new_phase_after_an_empty_interval(self):
        rows = [
            (780, 1, 0.0, 0.0, 1.0, 0.0),
            (786, 1, 0.4, 0.0, 1.0, 0.0),
            (792, 1, 0.8, 0.0, 1.0, 0.0),
            (799, 2, 1.0, 0.0, 1.0, 0.0),
        ]

        converted, source_stride, _ = eth_convert._canonical_rows(rows)

        self.assertEqual(source_stride, 6)
        self.assertEqual([row[0] for row in converted], [0, 1, 2, 3])


if __name__ == "__main__":
    unittest.main()
