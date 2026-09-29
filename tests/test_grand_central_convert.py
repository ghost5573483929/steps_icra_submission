from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from canonical import validate_canonical_file
from datasets_preprocessing.grand_central import convert as MODULE


ROOT = Path(__file__).resolve().parents[1]


def test_gap_split_and_open_traj_interpolation(tmp_path):
    annotation = tmp_path / "raw" / "Annotation"
    annotation.mkdir(parents=True)
    # Source fields are py, px, frame. A >20 frame gap starts a new identity.
    # (px, py) = (20, 30) projects with a third homogeneous coordinate of ~1.0257,
    # so a missing perspective divide or an x/y swap changes the result measurably.
    (annotation / "1.txt").write_text("30 20 0  10 10 10  40 40 40  50 50 50\n")
    tracks = MODULE.read_annotation_directory(annotation)
    assert len(tracks) == 2
    assert [track[0] for track in tracks] == [1, 2]
    sampled, world = MODULE.interpolate_track(tracks[0][1], tracks[0][2])
    np.testing.assert_array_equal(sampled, [0])  # final endpoint is excluded upstream.
    # Sampled frame 0 coincides exactly with the fixture's first pixel (px, py) = (20, 30).
    homogeneous = np.array([20.0, 30.0, 1.0])
    projected = MODULE.HOMOGRAPHY @ homogeneous
    expected = MODULE.WORLD_SCALE * projected[:2] / projected[2]
    np.testing.assert_allclose(world, [expected])


def test_non_finite_pixel_coordinate_is_rejected(tmp_path):
    annotation = tmp_path / "raw" / "Annotation"
    annotation.mkdir(parents=True)
    # Source fields are py, px, frame; the second triple has a NaN px.
    (annotation / "1.txt").write_text("0 0 0  10 nan 10  40 40 40\n")
    with pytest.raises(ValueError, match="non-finite pixel coordinate"):
        MODULE.read_annotation_directory(annotation)


def test_converter_writes_canonical_namespaced_recording(tmp_path):
    annotation = tmp_path / "raw" / "Annotation"
    annotation.mkdir(parents=True)
    (annotation / "1.txt").write_text("0 0 0  10 10 10  20 20 20\n")
    output = MODULE.convert_dataset(tmp_path)
    parquet = pq.ParquetFile(output)
    metadata = {key.decode(): value.decode() for key, value in parquet.schema_arrow.metadata.items()}
    assert metadata["scene_sequence"] == "grand_central"
    assert metadata["recording"] == "grand_central"
    assert metadata["opentraj_reference"].endswith("toolkit/loaders/loader_gcs.py")


def test_homography_is_the_exact_pinned_opentraj_matrix():
    expected = np.asarray(
        [
            [4.97412897e-02, -4.24730883e-02, 7.25543911e01],
            [1.45017874e-01, -3.35678711e-03, 7.97920970e00],
            [1.36068797e-03, -4.98339188e-05, 1.00000000e00],
        ]
    )
    np.testing.assert_array_equal(MODULE.HOMOGRAPHY, expected)
    assert MODULE.WORLD_SCALE == 0.8


def test_real_release_conformance_when_present():
    dataset = ROOT / "datasets" / "grand_central"
    annotation = dataset / "raw" / "extracted" / "Annotation"
    converted = dataset / "converted" / "grand_central.traj.parquet"
    if not annotation.exists() or not converted.exists():
        pytest.skip("Grand Central release is not installed")
    assert len(list(annotation.glob("*.txt"))) == 12_684
    identity = validate_canonical_file(converted)
    assert identity.scene_id == "grand_central_station"
    assert identity.scene_sequence == identity.recording == "grand_central"
    frame = pq.read_table(converted, columns=["track_id", "step", "x", "y"]).to_pandas()
    assert len(frame) == 873_804
    # OpenTraj creates additional identities at annotation gaps >20 frames.
    assert frame.track_id.nunique() == 17_862
    assert not frame.duplicated(["track_id", "step"]).any()
    assert np.isfinite(frame[["x", "y"]].to_numpy()).all()
