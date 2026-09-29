from __future__ import annotations

import copy
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import forecasting_config
from baselines.scholler import errors as scholler_errors, predict as scholler_predict
from baselines.social_gan_data import social_gan_window
from baselines.social_vae_data import social_vae_state
from forecasting_config import (
    FEATURE_CHANNEL_COUNTS,
    FEATURES,
    FeatureConfig,
    FORECASTING_PRESETS,
    resolve_forecasting_config,
)
from forecasting_representation import (
    apply_coordinate_transform,
    build_features,
    prepare_forecast_representation,
)
from forecasting_windows import enumerate_forecast_windows
from conftest import write_canonical_fixture


ROOT = Path(__file__).resolve().parents[1]


def test_every_literature_preset_resolves_exactly():
    expected = {
        "social_gan": ("world", "absolute_xy_and_relative_displacement", "per_step_difference", "none"),
        "scholler": ("world", "absolute_xy", "per_step_difference", "none"),
        "trajectron_pp": ("scene_mean_center", "position_velocity_acceleration", "physical_derivative_dt", "trajectron_pp"),
        "social_vae": ("world", "position_velocity_acceleration", "per_step_difference", "none"),
        "steps_default": ("target_centric_se2_observations_only", "absolute_xy", "none", "none"),
    }
    for name, values in expected.items():
        config = resolve_forecasting_config(preset=name)
        assert (config.window.observation_steps, config.window.prediction_steps, config.window.stride) == (8, 12, 1)
        assert config.signal.input_mode == config.signal.target_mode == "raw"
        assert (config.coordinates.frame, config.features.representation, config.features.derivative_mode, config.standardization.mode) == values
        assert not config.preset_modified
        assert config.literature_exact
        assert len(config.upstream_commit) == 40 or name == "steps_default"


def test_aliases_and_modified_preset_provenance():
    exact = resolve_forecasting_config(preset="Trajectron++")
    assert exact.requested_preset == "trajectron_pp"
    modified = resolve_forecasting_config(
        preset="SocialGAN", config_overrides={"signal": {"input_mode": "causal_kalman"}}
    )
    assert modified.requested_preset == "social_gan"
    assert modified.preset_modified
    assert not modified.literature_exact


def test_override_precedence_and_unspecified_cli_does_not_override():
    config = resolve_forecasting_config(
        preset="social_gan",
        config_overrides={"coordinates": {"frame": "target_centric_se2_observations_only"}},
        cli_overrides={"coordinates.frame": "world", "signal.input_mode": None},
    )
    assert config.coordinates.frame == "world"
    assert config.signal.input_mode == "raw"
    assert config.preset_modified is False


def test_standalone_introspection_uses_config_then_cli_precedence(tmp_path):
    config_path = tmp_path / "forecast.yaml"
    config_path.write_text(
        "forecasting:\n"
        "  preset: social_gan\n"
        "  coordinates:\n"
        "    frame: target_translation_last_observation\n"
    )
    completed = subprocess.run(
        [
            sys.executable, "-m", "forecasting_config",
            "--forecast-config", str(config_path),
            "--coordinate-frame", "world",
            "--show-resolved-forecast-config",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert '"requested_preset": "social_gan"' in completed.stdout
    assert '"frame": "world"' in completed.stdout
    assert '"preset_modified": false' in completed.stdout


def test_target_relative_frame_with_no_target_membership_raises():
    for frame in (
        "target_translation_last_observation",
        "target_centric_se2_observations_only",
    ):
        with pytest.raises(ValueError, match="target-relative"):
            resolve_forecasting_config(
                preset="social_gan",
                config_overrides={"coordinates": {"frame": frame}},
            )


def test_min_complete_agents_rejected_for_membership_that_never_reads_it():
    with pytest.raises(ValueError, match="does not apply it"):
        resolve_forecasting_config(preset="scholler", cli_overrides={"min_complete_agents": 2})
    social_gan = resolve_forecasting_config(preset="social_gan")
    assert social_gan.min_complete_agents == 2
    with pytest.raises(ValueError, match="does not apply it"):
        dataclasses.replace(social_gan, scene_membership="individual_track")


def test_social_gan_relative_displacement_and_membership():
    xy = np.zeros((20, 2, 2), dtype=np.float32)
    xy[:, :, 0] = np.arange(20)[:, None]
    valid = np.ones((20, 2), dtype=bool)
    window = social_gan_window(xy, valid)
    assert window is not None
    np.testing.assert_array_equal(window.obs_traj_rel[:, 0, 0], 0.0)
    np.testing.assert_array_equal(window.obs_traj_rel[:, 0, 1:], 1.0)
    valid[:, 1] = False
    assert social_gan_window(xy, valid) is None
    valid[:, 1] = True
    valid[10, 1] = False
    assert social_gan_window(xy, valid) is None


def test_social_gan_adapter_matches_pinned_official_loader(tmp_path):
    reference = ROOT / "references" / "social-gan"
    if not (reference / "sgan/data/trajectories.py").exists():
        pytest.skip("Pinned Social GAN checkout is unavailable")
    sys.path.insert(0, str(reference))
    try:
        from sgan.data.trajectories import TrajectoryDataset
        fixture = tmp_path / "scene.txt"
        fixture.write_text("".join(
            f"{step}\t{agent}\t{step + agent}\t{agent}\n"
            for step in range(20) for agent in (1, 2)
        ))
        official = TrajectoryDataset(tmp_path, obs_len=8, pred_len=12, skip=1, min_ped=1)[0]
        xy = np.zeros((20, 2, 2), dtype=np.float32)
        for step in range(20):
            for column, agent in enumerate((1, 2)):
                xy[step, column] = (step + agent, agent)
        ours = social_gan_window(xy, np.ones((20, 2), dtype=bool))
        assert ours is not None
        np.testing.assert_array_equal(ours.obs_traj, official[0].numpy())
        np.testing.assert_array_equal(ours.pred_traj, official[1].numpy())
        np.testing.assert_array_equal(ours.obs_traj_rel, official[2].numpy())
        np.testing.assert_array_equal(ours.pred_traj_rel, official[3].numpy())
        np.testing.assert_array_equal(ours.loss_mask, official[5].numpy())
    finally:
        sys.path.remove(str(reference))


def test_scholler_uses_only_last_observed_displacement():
    observed = np.c_[np.arange(8.0), np.zeros(8)]
    future = np.c_[np.arange(8.0, 20.0), np.zeros(12)]
    assert scholler_errors(observed, future) == (0.0, 0.0)
    changed = observed.copy(); changed[:6, 0] = 1000.0
    np.testing.assert_array_equal(scholler_predict(observed), scholler_predict(changed))


def test_scholler_metrics_are_rigid_frame_invariant():
    rng=np.random.default_rng(4)
    xy=np.cumsum(rng.normal(size=(20,2)),axis=0)
    reference=scholler_errors(xy[:8],xy[8:])
    angle=.73;matrix=np.array([[np.cos(angle),-np.sin(angle)],[np.sin(angle),np.cos(angle)]])
    transformed=xy@matrix.T+np.array([120.0,-45.0])
    np.testing.assert_allclose(scholler_errors(transformed[:8],transformed[8:]),reference,atol=1e-12)


def test_feature_channel_counts_matches_every_representation_and_build_features():
    assert set(FEATURES) == set(FEATURE_CHANNEL_COUNTS)

    class _StubConfig:
        def __init__(self, representation: str, derivative_mode: str):
            self.features = FeatureConfig(representation=representation, derivative_mode=derivative_mode)

    xy = np.zeros((4, 1, 2), dtype=np.float32)
    xy[:, 0, 0] = np.arange(4)
    valid = np.ones((4, 1), dtype=bool)
    derivative_mode_by_representation = {
        "absolute_xy": "none",
        "absolute_xy_and_relative_displacement": "none",
        "position_velocity_acceleration": "physical_derivative_dt",
    }
    for representation in FEATURES:
        derivative_mode = derivative_mode_by_representation[representation]
        stub = _StubConfig(representation, derivative_mode)
        features, _, feature_names = build_features(xy, valid, stub, dt=0.4)
        assert features.shape[-1] == len(feature_names) == FEATURE_CHANNEL_COUNTS[representation], (
            f"build_features resolves {representation!r} to {features.shape[-1]} channel(s), but "
            f"FEATURE_CHANNEL_COUNTS[{representation!r}] == {FEATURE_CHANNEL_COUNTS[representation]}"
        )


def test_missing_feature_channel_count_raises_explicit_error_not_key_error(monkeypatch):
    monkeypatch.setattr(
        forecasting_config, "FEATURES", forecasting_config.FEATURES + ("unregistered_new_representation",)
    )
    broken = copy.deepcopy(FORECASTING_PRESETS["steps_default"])
    broken["features"]["representation"] = "unregistered_new_representation"

    with pytest.raises(ValueError, match="unregistered_new_representation"):
        forecasting_config._validate(broken)


def test_derivative_conventions_are_not_interchangeable():
    xy = np.zeros((20, 1, 2), dtype=np.float32); xy[:, 0, 0] = np.arange(20)
    valid = np.ones((20, 1), dtype=bool)
    per_step = resolve_forecasting_config(preset="social_vae")
    physical = resolve_forecasting_config(preset="trajectron_pp")
    per_features, _, _ = build_features(xy, valid, per_step, dt=0.4)
    physical_features, _, _ = build_features(xy, valid, physical, dt=0.4)
    assert per_features[1, 0, 2] == 1.0
    assert per_features[0, 0, 2] == 1.0
    assert per_features[0, 0, 4] == 0.0
    assert physical_features[1, 0, 2] == 2.5
    assert physical_features[0, 0, 2] == 2.5
    assert physical_features[0, 0, 4] == 0.0


def test_social_vae_adapter_matches_state_and_dummy_neighbor_contract():
    xy = np.zeros((20, 1, 2), dtype=np.float32)
    xy[:, 0, 0] = np.arange(20)
    valid = np.ones((20, 1), dtype=bool)
    state = social_vae_state(xy, valid, training=False)
    assert state["history"].shape == (8, 6)
    assert state["future"].shape == (12, 2)
    assert state["neighbors"].shape == (20, 1, 6)
    assert np.all(state["neighbors"] == 1e9)
    np.testing.assert_array_equal(state["history"][:, 2], 1.0)
    np.testing.assert_array_equal(state["history"][:, 4], 0.0)


def test_target_transform_is_observation_only_and_round_trips():
    xy = np.zeros((20, 2, 2), dtype=np.float32)
    xy[:8, 0, 0] = np.arange(8)
    xy[8:, 0, 0] = np.arange(8, 20)
    config = resolve_forecasting_config(preset="steps_default")
    transformed, transform = apply_coordinate_transform(xy, config)
    assert np.allclose(transformed[7, 0], 0.0)
    np.testing.assert_allclose(transform.inverse(transformed), xy, atol=1e-6)
    changed = xy.copy(); changed[8:] += 999
    _, second = apply_coordinate_transform(changed, config)
    np.testing.assert_array_equal(transform.origin, second.origin)
    assert transform.rotation == second.rotation


def test_transform_inverse_round_trips_only_without_training_augmentation():
    # training=True does not round trip because social_vae augments after the
    # transform. If that augmentation ever becomes a no-op this assertion needs
    # revisiting rather than loosening.
    rng = np.random.default_rng(0)
    xy = np.cumsum(rng.normal(size=(20, 2, 2)), axis=0).astype(np.float32)
    valid = np.ones((20, 2), dtype=bool)
    config = resolve_forecasting_config(preset="social_vae")

    eval_repr = prepare_forecast_representation(
        xy, valid, config, training=False, rng=np.random.default_rng(1)
    )
    assert np.allclose(eval_repr.transform.inverse(eval_repr.xy), xy)

    train_repr = prepare_forecast_representation(
        xy, valid, config, training=True, rng=np.random.default_rng(1)
    )
    assert not np.allclose(train_repr.transform.inverse(train_repr.xy), xy)


def test_normal_presets_have_no_future_input_leakage():
    rng = np.random.default_rng(2)
    xy = np.cumsum(rng.normal(size=(20, 2, 2)), axis=0).astype(np.float32)
    valid = np.ones((20, 2), dtype=bool)
    changed = xy.copy(); changed[8:] += 5000
    center = np.array([10.0, 20.0], dtype=np.float32)
    for name in FORECASTING_PRESETS:
        config = resolve_forecasting_config(preset=name)
        kwargs = {"source_scene_center": center} if name == "trajectron_pp" else {}
        first = prepare_forecast_representation(xy, valid, config, training=False, **kwargs)
        second = prepare_forecast_representation(changed, valid, config, training=False, **kwargs)
        np.testing.assert_allclose(first.input_xy, second.input_xy, atol=1e-6)
        np.testing.assert_allclose(first.features[:8], second.features[:8], atol=1e-6)
        assert not first.input_uses_future_information


def test_training_augmentation_is_shared_and_eval_is_deterministic():
    xy = np.zeros((20, 3, 2), dtype=np.float32)
    xy[:, :, 0] = np.arange(20)[:, None]
    valid = np.ones((20, 3), dtype=bool)
    config = resolve_forecasting_config(preset="social_vae")
    train = prepare_forecast_representation(xy, valid, config, training=True, rng=np.random.default_rng(7))
    evaluation = prepare_forecast_representation(xy, valid, config, training=False, rng=np.random.default_rng(7))
    assert train.augmentation["applied"]
    assert not evaluation.augmentation["applied"]
    np.testing.assert_allclose(train.xy[:, 0], train.xy[:, 1])


def test_trajectron_scene_center_derivatives_and_standardization():
    xy=np.zeros((20,1,2),dtype=np.float32);xy[:,0,0]=10+np.arange(20)
    valid=np.ones((20,1),dtype=bool);config=resolve_forecasting_config(preset="trajectron_pp")
    result=prepare_forecast_representation(
        xy,valid,config,source_scene_center=np.array([10.0,0.0]),training=False
    )
    assert result.xy[0,0,0]==0.0
    # Physical velocity is 1/0.4=2.5 m/s and upstream pedestrian std is 2.
    assert result.features[1,0,2]==1.25
    assert result.features[1,0,4]==0.0
    assert not result.augmentation["applied"]


def test_preprocessing_hash_changes_for_every_recipe_change():
    baseline = resolve_forecasting_config(preset="social_gan")
    changed = resolve_forecasting_config(preset="social_gan", cli_overrides={"window.stride": 2})
    assert baseline.forecast_preprocessing_hash != changed.forecast_preprocessing_hash


def _write_window_fixture(path, tracks):
    write_canonical_fixture(
        path,
        {
            track_id: [(int(step), float(step), float(track_id), np.nan) for step in steps]
            for track_id, steps in tracks.items()
        },
        dataset="test", scene_id="scene", scene_sequence="sequence", recording="recording",
    )


def test_social_gan_window_index_requires_two_complete_agents(tmp_path):
    path=tmp_path/"recording.traj.parquet"
    _write_window_fixture(path,{1:np.arange(20),2:np.arange(20),3:np.delete(np.arange(20),10)})
    config=resolve_forecasting_config(preset="social_gan")
    records=enumerate_forecast_windows(path,config)
    assert len(records)==1
    assert records[0].complete_agent_ids==(1,2)
    _write_window_fixture(path,{1:np.arange(20),2:np.delete(np.arange(20),10)})
    assert enumerate_forecast_windows(path,config)==[]


def test_scholler_masked_membership_is_separate_from_strict_publication_population(tmp_path):
    path=tmp_path/"recording.traj.parquet";_write_window_fixture(path,{1:np.arange(12)})
    strict=resolve_forecasting_config(preset="scholler")
    assert enumerate_forecast_windows(path,strict)==[]
    masked=resolve_forecasting_config(
        preset="scholler",config_overrides={"scene_membership":"scholler_masked_track"}
    )
    records=enumerate_forecast_windows(path,masked)
    assert [record.actual_samples for record in records]==[12,11,10]


def test_preset_forecast_preprocessing_hashes_are_pinned():
    # DO NOT update these values to make this test pass. A moved hash means preset
    # resolution changed, which changes the provenance of every run recorded against
    # that preset -- a scientific-settings change that needs a human decision.
    expected = {
        "social_gan": "sha256:9be1a94a1a60cd1f1e526f7e7e4f63d9d270b4147fe791c703da013f214b84ed",
        "scholler": "sha256:87552f1a37536d1bb60e9b677aec93e7487c346d037bd978e4c665ab9cd0a32d",
        "trajectron_pp": "sha256:3420573f250ffa176f346be1dcbbff73aa9db86d73527ea72c3c48287fa9d509",
        "social_vae": "sha256:c9468dd7ef44ecfebafc6d3fb2415bc66aec7b1cc3344c1e3988cd33feac06fc",
        "steps_default": "sha256:9ae74e432c3a970d7ce573f4896e0a7c8344a84a7e0785b85d6717ad9574de17",
    }
    for name, expected_hash in expected.items():
        config = resolve_forecasting_config(preset=name)
        assert config.forecast_preprocessing_hash == expected_hash, (
            f"{name}: forecast_preprocessing_hash moved from {expected_hash} to "
            f"{config.forecast_preprocessing_hash}; see comment above before updating this."
        )


def test_recipe_rejects_unknown_field_like_deep_merge_does():
    with pytest.raises(ValueError, match="scene_memberships"):
        forecasting_config._recipe(scene_memberships="individual_track")


def test_enumerate_forecast_windows_rejects_target_complete_membership(tmp_path):
    path = tmp_path / "recording.traj.parquet"
    _write_window_fixture(path, {1: np.arange(20)})
    config = resolve_forecasting_config(preset="steps_default")
    assert config.scene_membership == "target_complete"
    with np.testing.assert_raises_regex(ValueError, "target_complete"):
        enumerate_forecast_windows(path, config)
