from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import torch

from baselines.social_vae import (
    REFERENCE,
    SocialVAEDataset,
    build,
    collate,
)
from data import Scene
from forecasting_config import resolve_forecasting_config


def _scene(agents: int) -> Scene:
    xy = np.zeros((20, agents, 2), dtype=np.float32)
    time = np.arange(20, dtype=np.float32)
    for agent in range(agents):
        xy[:, agent, 0] = 0.3 * time + agent
        xy[:, agent, 1] = 0.1 * time - agent
    valid = np.ones((20, agents), dtype=bool)
    return Scene(
        xy=xy,
        primary_track_id="recording:1",
        neighbor_track_ids=[f"recording:{index + 2}" for index in range(agents - 1)],
        dataset="synthetic",
        recording="recording",
        start_step=0,
        scene_tag=2,
        sub_tags=[],
        valid=valid,
        rotation=0.0,
        center=np.zeros(2, dtype=np.float32),
        scene_id="scene",
        scene_sequence="sequence",
        observation_end_step=7,
        future_end_step=19,
        canonical_xy=xy.copy(),
    )


def test_adapter_batches_variable_neighbor_sets_with_official_sentinel():
    config = resolve_forecasting_config(preset="social_vae")
    source = [_scene(1), _scene(3)]
    dataset = SocialVAEDataset(source, config, training=False, model_seed=7)
    batch = collate([dataset[0], dataset[1]])
    assert batch["x"].shape == (8, 2, 6)
    assert batch["y"].shape == (12, 2, 2)
    assert batch["neighbor"].shape == (20, 2, 2, 6)
    assert torch.all(batch["neighbor"][:, 0] == 1e9)
    assert torch.all(batch["neighbor"][:, 1] < 1e9)


def test_training_augmentation_is_seeded_by_epoch():
    config = resolve_forecasting_config(preset="social_vae")
    dataset = SocialVAEDataset([_scene(2)], config, training=True, model_seed=11)
    dataset.set_epoch(3)
    first = dataset[0]
    repeated = dataset[0]
    np.testing.assert_array_equal(first["history"], repeated["history"])
    dataset.set_epoch(4)
    changed = dataset[0]
    assert not np.array_equal(first["history"], changed["history"])


def test_pinned_upstream_model_runs_training_and_k_sample_inference():
    config = resolve_forecasting_config(preset="social_vae")
    dataset = SocialVAEDataset([_scene(1), _scene(2)], config, training=False, model_seed=5)
    batch = collate([dataset[0], dataset[1]])
    model = build(torch.device("cpu"), hidden_dim=16)
    assert Path(inspect.getsourcefile(type(model))).resolve() == (REFERENCE / "social_vae.py").resolve()
    model.train()
    error, kl = model(batch["x"], batch["y"], batch["neighbor"])
    losses = model.loss(error, kl)
    assert all(torch.isfinite(value) for value in losses.values())
    losses["loss"].backward()
    model.eval()
    with torch.no_grad():
        predictions = model(batch["x"], batch["neighbor"], n_predictions=3)
    assert predictions.shape == (3, 12, 2, 2)
    assert torch.isfinite(predictions).all()
