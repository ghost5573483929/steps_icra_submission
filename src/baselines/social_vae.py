"""Thin integration around the pinned official SocialVAE implementation.

The model architecture remains in ``references/social-vae/social_vae.py``.  This
module only adapts unified forecasting scenes to the official time-major tensor
contract and converts predictions back to metric world coordinates.
"""
from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from baselines.social_vae_data import social_vae_state
from forecasting_config import ForecastingConfig


ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / "references" / "social-vae"
UPSTREAM_REPOSITORY = "https://github.com/xupei0610/SocialVAE"
UPSTREAM_COMMIT = "eb23cbae562bac2c2dcf0d81529e6ab2ac280962"


@lru_cache(maxsize=1)
def _upstream_module():
    path = REFERENCE / "social_vae.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"Pinned SocialVAE source is missing: {path}. Restore references/social-vae."
        )
    name = "_steps_pinned_social_vae"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load pinned SocialVAE module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def build(
    device: torch.device,
    *,
    prediction_steps: int = 12,
    neighborhood_radius: float = 2.0,
    hidden_dim: int = 256,
) -> torch.nn.Module:
    """Instantiate the unmodified upstream model on the requested device."""
    model = _upstream_module().SocialVAE(
        horizon=prediction_steps,
        ob_radius=neighborhood_radius,
        hidden_dim=hidden_dim,
    )
    return model.to(device)


class SocialVAEDataset(Dataset):
    """Adapt unified ``Scene`` objects to one official SocialVAE target sample."""

    def __init__(
        self,
        source,
        config: ForecastingConfig,
        *,
        training: bool,
        model_seed: int,
        limit: int | None = None,
    ) -> None:
        self.source = source
        self.config = config
        self.training = bool(training)
        self.model_seed = int(model_seed)
        self.limit = min(len(source), limit) if limit is not None else len(source)
        self.epoch = 0

    def __len__(self) -> int:
        return self.limit

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, index: int) -> dict[str, object]:
        scene = self.source[index]
        if scene.canonical_xy is None:
            raise ValueError("SocialVAE requires canonical world coordinates on Scene.canonical_xy")
        canonical_xy = np.asarray(scene.canonical_xy, dtype=np.float32)
        valid = np.asarray(scene.valid, dtype=bool)
        if not valid[:, 0].all():
            raise ValueError("SocialVAE target pedestrian must be complete for all 8+12 samples")
        rng = np.random.default_rng(
            np.random.SeedSequence([self.model_seed, self.epoch, int(index)])
        )
        source_center = (
            np.asarray(scene.center, dtype=np.float32)
            if self.config.coordinates.frame == "scene_mean_center"
            else None
        )
        state = social_vae_state(
            canonical_xy,
            valid,
            config=self.config,
            training=self.training,
            rng=rng,
            source_scene_center=source_center,
        )
        return {
            **state,
            "world_xy": np.array(canonical_xy, copy=True),
            "world_valid": np.array(valid, copy=True),
            "dataset": scene.dataset,
            "scene_sequence": scene.scene_sequence,
            "recording": scene.recording,
            "primary_track_id": scene.primary_track_id,
            "start_step": int(scene.start_step),
        }


def collate(batch: list[dict[str, object]]) -> dict[str, object]:
    """Pad variable neighbor sets and emit upstream time-major tensors."""
    if not batch:
        raise ValueError("Cannot collate an empty SocialVAE batch")
    max_neighbors = max(np.asarray(item["neighbors"]).shape[1] for item in batch)
    padded_neighbors = []
    for item in batch:
        neighbors = np.asarray(item["neighbors"], dtype=np.float32)
        if neighbors.shape[1] < max_neighbors:
            neighbors = np.pad(
                neighbors,
                ((0, 0), (0, max_neighbors - neighbors.shape[1]), (0, 0)),
                mode="constant",
                constant_values=1e9,
            )
        padded_neighbors.append(neighbors)
    # Official shapes: x=(obs,B,6), y=(pred,B,2), neighbor=(20,B,Nn,6).
    return {
        "x": torch.from_numpy(np.stack([np.asarray(item["history"]) for item in batch], axis=1)),
        "y": torch.from_numpy(np.stack([np.asarray(item["future"]) for item in batch], axis=1)),
        "neighbor": torch.from_numpy(np.stack(padded_neighbors, axis=1)),
        "target_world_xy": np.stack(
            [np.asarray(item["target_world_xy"]) for item in batch], axis=0
        ),
        "world_xy": [np.asarray(item["world_xy"]) for item in batch],
        "world_valid": [np.asarray(item["world_valid"]) for item in batch],
        "transform": [item["transform"] for item in batch],
        "augmentation": [item["augmentation"] for item in batch],
        "metadata": [
            {
                key: item[key]
                for key in ("dataset", "scene_sequence", "recording", "primary_track_id", "start_step")
            }
            for item in batch
        ],
    }


def tensors_to_device(batch: dict[str, object], device: torch.device):
    return (
        batch["x"].to(device, non_blocking=True),
        batch["y"].to(device, non_blocking=True),
        batch["neighbor"].to(device, non_blocking=True),
    )


def sample_predictions(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    *,
    k: int = 20,
) -> torch.Tensor:
    x, _y, neighbor = tensors_to_device(batch, device)
    prediction = model(x, neighbor, n_predictions=k)
    if prediction.ndim == 3:
        prediction = prediction.unsqueeze(0)
    # Official output: (K,pred,B,2).
    return prediction


def predictions_to_world(prediction: torch.Tensor, batch: dict[str, object]) -> np.ndarray:
    values = prediction.detach().cpu().numpy()
    world = []
    for index, transform in enumerate(batch["transform"]):
        world.append(transform.inverse(values[:, :, index]))
    return np.stack(world, axis=1)  # (K,B,pred,2)


def best_of_k(
    model: torch.nn.Module,
    batch: dict[str, object],
    device: torch.device,
    *,
    k: int = 20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = sample_predictions(model, batch, device, k=k)
    world = predictions_to_world(prediction, batch)
    target = np.asarray(batch["target_world_xy"])
    distance = np.linalg.norm(world - target[None], axis=-1)  # K,B,T
    return distance.mean(axis=-1).min(axis=0), distance[..., -1].min(axis=0), world
