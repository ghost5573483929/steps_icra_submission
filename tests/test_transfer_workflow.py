from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

import baselines.train_social_vae as train_social_vae
from baselines.aggregate_transfer import aggregate
from baselines.train_social_vae import FixedNumberBatchSampler
from baselines.transfer_data import parse_source_datasets
from baselines.target_fraction import target_training_fraction
from baselines.transfer_config import (
    build_optimizer,
    build_stage_scheduler,
    load_model_weights_for_transfer,
    resolve_stage_optimization,
    validate_finetune_checkpoint,
)
from conftest import write_run_result
from data import Scene


class _Spec:
    name = "target"


class _Rows:
    spec = _Spec()
    split = "train"

    def __init__(self):
        self.rows = [
            {
                "recording": f"recording-{track // 2}",
                "primary_track_id": track,
                "start_step": start,
            }
            for track in range(20)
            for start in (0, 1)
        ]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        return self.rows[index]


def test_native_social_vae_stage_recipes():
    pretrain = resolve_stage_optimization("social_vae", "pretrain")
    fine = resolve_stage_optimization("social_vae", "finetune")
    scratch = resolve_stage_optimization("social_vae", "scratch")
    assert (pretrain.learning_rate, pretrain.batch_size, pretrain.max_epochs) == (5e-4, 128, 600)
    assert pretrain.batches_per_epoch == 100
    assert pretrain.scheduler == "none"
    assert not pretrain.early_stopping
    assert (fine.learning_rate, fine.max_epochs, fine.patience) == (1e-4, 200, 20)
    assert fine.early_stopping
    assert fine == scratch


def test_source_dataset_union_is_ordered_and_deduplicated():
    sources = parse_source_datasets(None, ["steps,eth", "steps"])
    assert sources == ("steps", "eth")


def test_social_vae_fixed_batches_cycle_small_dataset():
    generator = torch.Generator().manual_seed(7)
    sampler = torch.utils.data.RandomSampler(range(3), generator=generator)
    batches = list(FixedNumberBatchSampler(4, sampler, batch_size=2))
    assert len(batches) == 4
    assert all(len(batch) == 2 for batch in batches)
    assert set(index for batch in batches for index in batch) == {0, 1, 2}


def test_native_moflow_stage_recipes():
    pretrain = resolve_stage_optimization("moflow", "pretrain")
    fine = resolve_stage_optimization("moflow", "finetune")
    scratch = resolve_stage_optimization("moflow", "scratch")
    assert (pretrain.learning_rate, pretrain.weight_decay, pretrain.batch_size) == (1e-4, 0.01, 48)
    assert (pretrain.max_epochs, pretrain.scheduler, pretrain.warmup_fraction) == (150, "cosine_warmup", 0.05)
    assert (fine.learning_rate, fine.max_epochs, fine.patience) == (2e-5, 50, 10)
    assert fine.early_stopping
    assert fine == scratch


def test_transfer_config_and_cli_precedence(tmp_path: Path):
    path = tmp_path / "transfer.yaml"
    path.write_text(
        "transfer:\n"
        "  pretrain:\n"
        "    lr: 0.001\n"
        "  finetune:\n"
        "    lr_multiplier: 0.1\n"
        "    patience: 7\n"
    )
    configured = resolve_stage_optimization("social_vae", "finetune", config_path=path)
    assert configured.learning_rate == 1e-4
    assert configured.patience == 7
    cli = resolve_stage_optimization(
        "social_vae",
        "finetune",
        config_path=path,
        cli_overrides={"learning_rate": 2e-4, "patience": 3},
    )
    assert cli.learning_rate == 2e-4
    assert cli.patience == 3


def test_finetune_loads_weights_but_not_optimizer_state(tmp_path: Path):
    source = torch.nn.Linear(2, 1)
    old_optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
    old_optimizer.zero_grad()
    source(torch.ones(1, 2)).sum().backward()
    old_optimizer.step()
    checkpoint = tmp_path / "source.pt"
    torch.save({"model": source.state_dict(), "optimizer": old_optimizer.state_dict()}, checkpoint)

    target = torch.nn.Linear(2, 1)
    for parameter in target.parameters():
        parameter.requires_grad_(False)
    payload = load_model_weights_for_transfer(target, checkpoint, torch.device("cpu"))
    fresh_optimizer = torch.optim.Adam(target.parameters(), lr=2e-4)
    assert payload["optimizer"]["state"]
    assert fresh_optimizer.state_dict()["state"] == {}
    assert all(parameter.requires_grad for parameter in target.parameters())
    for left, right in zip(source.parameters(), target.parameters()):
        torch.testing.assert_close(left, right)

    scheduler_config = resolve_stage_optimization("moflow", "finetune")
    fresh_scheduler = build_stage_scheduler(
        fresh_optimizer, scheduler_config, total_iterations=50
    )
    assert fresh_scheduler.state_dict()["last_epoch"] == 0


def test_finetune_checkpoint_must_be_selected_pretraining_checkpoint(tmp_path: Path):
    path = tmp_path / "best.pt"
    payload = {
        "stage": "pretrain",
        "run_config": {"transfer_optimization": {"learning_rate": 1e-4}},
        "checkpoint_selection": {"metric": "val_minADE", "value": 0.5},
    }
    validate_finetune_checkpoint(path, payload)
    invalid = {**payload, "stage": "finetune"}
    try:
        validate_finetune_checkpoint(path, invalid)
    except ValueError as error:
        assert "pretraining checkpoint" in str(error)
    else:
        raise AssertionError("a fine-tuned checkpoint was accepted as source pretraining")


def test_moflow_warmup_then_cosine_schedule():
    config = resolve_stage_optimization("moflow", "pretrain")
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.AdamW([parameter], lr=config.learning_rate)
    scheduler = build_stage_scheduler(optimizer, config, total_iterations=100)
    rates = [optimizer.param_groups[0]["lr"]]
    for _ in range(10):
        optimizer.step()
        scheduler.step()
        rates.append(optimizer.param_groups[0]["lr"])
    assert rates[0] == config.min_lr
    assert rates[5] == config.learning_rate
    assert rates[10] < rates[5]


def test_optimizer_override_is_realized():
    config = resolve_stage_optimization(
        "social_vae",
        "pretrain",
        cli_overrides={"optimizer": "AdamW", "weight_decay": 0.25},
    )
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = build_optimizer([parameter], config)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["weight_decay"] == 0.25


def test_finetune_cannot_reuse_optimizer_or_freeze_layers(tmp_path: Path):
    path = tmp_path / "invalid.yaml"
    path.write_text(
        "transfer:\n"
        "  finetune:\n"
        "    reset_optimizer: false\n"
        "    train_all_layers: false\n"
    )
    try:
        resolve_stage_optimization("moflow", "finetune", config_path=path)
    except ValueError as error:
        assert "reset optimizer" in str(error)
    else:
        raise AssertionError("invalid fine-tuning transfer state was accepted")


def test_target_fractions_are_nested_reproducible_and_persisted(tmp_path: Path, monkeypatch):
    import baselines.target_fraction as module

    monkeypatch.setattr(module, "ROOT", tmp_path)
    source = _Rows()
    small, small_path = target_training_fraction(
        source, target="eth_ucy", fold="eth", fraction=0.25, data_seed=42
    )
    repeated, _ = target_training_fraction(
        source, target="eth_ucy", fold="eth", fraction=0.25, data_seed=42
    )
    large, _ = target_training_fraction(
        source, target="eth_ucy", fold="eth", fraction=0.5, data_seed=42
    )
    assert list(small.indices) == list(repeated.indices)
    assert set(small.indices) < set(large.indices)
    assert len(small) == 10
    manifest = json.loads(small_path.read_text())
    assert manifest["fraction"] == 0.25
    assert manifest["n_source_tracks"] == 20
    assert manifest["n_selected_tracks"] == 5
    assert manifest["n_selected_windows"] == 10


def _synthetic_scene(offset: float) -> Scene:
    xy = np.zeros((20, 2, 2), dtype=np.float32)
    time = np.arange(20, dtype=np.float32)
    for agent in range(2):
        xy[:, agent, 0] = 0.3 * time + agent + offset
        xy[:, agent, 1] = 0.1 * time - agent
    valid = np.ones((20, 2), dtype=bool)
    return Scene(
        xy=xy,
        primary_track_id="recording:1",
        neighbor_track_ids=["recording:2"],
        dataset="synthetic",
        recording="recording",
        start_step=0,
        scene_tag=2,
        sub_tags=[],
        valid=valid,
        rotation=0.0,
        center=np.zeros(2, dtype=np.float32),
        scene_id=f"scene-{offset}",
        scene_sequence="sequence",
        observation_end_step=7,
        future_end_step=19,
        canonical_xy=xy.copy(),
    )


def test_social_vae_persists_epoch_boundary_resume_state(
    tmp_path: Path,
):
    """With batches_per_epoch=4 and status_every=2 the in-loop save always fires on the
    last batch of an epoch (4 % 2 == 0), so the epoch-boundary save must still follow it.
    """
    run_dir = tmp_path / "run"
    train_scenes = [_synthetic_scene(offset) for offset in range(6)]
    val_scenes = [_synthetic_scene(offset + 100) for offset in range(2)]

    ade_sequence = iter([1.0, 1.0])  # epoch 1 improves over inf; epoch 2 repeats

    def fake_sources(args, cfg):
        return train_scenes, val_scenes, None

    def fake_validate(model, loader, device, *, k, max_batches):
        return next(ade_sequence), 2.0, len(val_scenes)

    latest_save_calls = []
    original_save = train_social_vae._save

    def counting_save(path, model, optimizer, scheduler, args, epoch, iteration, **kwargs):
        if path.name == "latest.pt":
            latest_save_calls.append(
                {
                    "epoch": epoch,
                    "iteration": iteration,
                    "selection_value": kwargs["selection_value"],
                    "best_epoch": kwargs["best_epoch"],
                }
            )
        return original_save(path, model, optimizer, scheduler, args, epoch, iteration, **kwargs)

    argv = [
        "train_social_vae.py",
        "--stage", "pretrain",
        "--pretrain-dataset", "steps",
        "--run-dir", str(run_dir),
        "--device", "cpu",
        "--num-workers", "0",
        "--epochs", "2",
        "--batches-per-epoch", "4",
        "--batch-size", "2",
        "--status-every", "2",
        "--hidden-dim", "8",
        "--k", "1",
        "--max-val-batches", "1",
    ]
    original_argv = sys.argv
    sys.argv = argv
    try:
        with patch.object(train_social_vae, "_sources", side_effect=fake_sources), \
             patch.object(train_social_vae, "_validate", side_effect=fake_validate), \
             patch.object(train_social_vae, "_save", side_effect=counting_save):
            train_social_vae.main()
    finally:
        sys.argv = original_argv

    # Two in-loop status_every saves per epoch (iterations 2 and 4, then 6 and 8),
    # each epoch followed by its end-of-epoch save at the same iteration.
    iterations_written = [call["iteration"] for call in latest_save_calls]
    assert iterations_written == [2, 4, 4, 6, 8, 8], iterations_written

    epoch1_inloop, epoch1_end = latest_save_calls[1], latest_save_calls[2]
    assert epoch1_inloop["selection_value"] is None and epoch1_inloop["best_epoch"] is None
    assert epoch1_end["selection_value"] == 1.0 and epoch1_end["best_epoch"] == 1

    epoch2_inloop, epoch2_end = latest_save_calls[4], latest_save_calls[5]
    assert epoch2_inloop["selection_value"] == 1.0 and epoch2_inloop["best_epoch"] == 1
    assert epoch2_end["selection_value"] == 1.0 and epoch2_end["best_epoch"] == 1

    payload = torch.load(run_dir / "checkpoints/latest.pt", map_location="cpu", weights_only=False)
    assert payload["epoch"] == 2 and payload["iteration"] == 8
    assert payload["epoch_complete"] is True
    assert payload["loader_generator_state"] is not None
    assert payload["checkpoint_selection"]["value"] == 1.0
    assert payload["checkpoint_selection"]["best_epoch"] == 1


def test_transfer_aggregation_uses_seed_level_standard_error(tmp_path: Path):
    runs = []
    for seed, ade, fde in ((0, 1.0, 2.0), (1, 2.0, 3.0), (2, 3.0, 4.0)):
        runs.append(write_run_result(
            tmp_path / f"seed{seed}",
            topk_ade=ade,
            topk_fde=fde,
            target_dataset="madras",
            target_fold=None,
            target_fraction=0.25,
            source_datasets=["steps"],
            fine_tuning_learning_rate=2e-5,
            checkpoint_metric="target_val_minADE",
            transfer_optimization_hash="sha256:optimization",
            forecast_preprocessing_hash="sha256:preprocessing",
            data_seed=42,
            training_stage="finetune",
            k=20,
            finetuning_seed=seed,
        ))
    result = aggregate(runs)
    assert result["metrics"]["topk_ade"]["mean"] == 2.0
    assert np.isclose(result["metrics"]["topk_ade"]["standard_error"], 1 / np.sqrt(3))
    assert result["standard_error_unit"] == "independent model seeds"


def _direct_transfer_run_fields(**overrides):
    fields = dict(
        target_dataset="madras",
        target_fold=None,
        target_fraction=0.25,
        source_datasets=["steps"],
        fine_tuning_learning_rate=None,
        checkpoint_metric="target_val_minADE",
        transfer_optimization_hash="sha256:optimization",
        forecast_preprocessing_hash="sha256:preprocessing",
        data_seed=42,
        training_stage="pretrain",
        k=20,
    )
    fields.update(overrides)
    return fields


def test_transfer_aggregation_resolves_finetuning_seed_none_for_pretrain_stage(tmp_path: Path):
    """``finetuning_seed`` is present-but-None here, so aggregation must fall back to
    ``seed`` via an explicit ``is not None`` check rather than ``dict.get``'s default."""
    runs = []
    for seed, ade, fde in ((0, 1.0, 2.0), (1, 2.0, 3.0), (2, 3.0, 4.0)):
        runs.append(write_run_result(
            tmp_path / f"seed{seed}",
            topk_ade=ade,
            topk_fde=fde,
            seed=seed,
            finetuning_seed=None,
            **_direct_transfer_run_fields(),
        ))
    result = aggregate(runs)
    assert result["seeds"] == [0, 1, 2]
    assert result["metrics"]["topk_ade"]["n_seeds"] == 3


def test_transfer_aggregation_rejects_differing_target_fraction(tmp_path: Path):
    run0 = write_run_result(
        tmp_path / "seed0", topk_ade=1.0, topk_fde=2.0, finetuning_seed=0,
        **_direct_transfer_run_fields(target_fraction=0.25, training_stage="finetune",
                                       fine_tuning_learning_rate=2e-5),
    )
    run1 = write_run_result(
        tmp_path / "seed1", topk_ade=1.0, topk_fde=2.0, finetuning_seed=1,
        **_direct_transfer_run_fields(target_fraction=0.5, training_stage="finetune",
                                       fine_tuning_learning_rate=2e-5),
    )
    with pytest.raises(ValueError, match="target_fraction"):
        aggregate([run0, run1])


def test_transfer_aggregation_rejects_duplicate_seeds(tmp_path: Path):
    fields = _direct_transfer_run_fields(training_stage="finetune", fine_tuning_learning_rate=2e-5)
    run0 = write_run_result(tmp_path / "seed0a", topk_ade=1.0, topk_fde=2.0, finetuning_seed=0, **fields)
    run1 = write_run_result(tmp_path / "seed0b", topk_ade=1.5, topk_fde=2.5, finetuning_seed=0, **fields)
    with pytest.raises(ValueError, match="not unique"):
        aggregate([run0, run1])
