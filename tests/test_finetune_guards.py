from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from baselines import train_moflow, train_social_vae
from baselines.early_stopping import EarlyStopping
from baselines.moflow import PrimaryTrajectoryDataset
from baselines.resume import load_resume_state
from baselines.transfer_config import (
    StageOptimization,
    optimization_as_dict,
    resolve_stage_seed,
    validate_finetune_checkpoint,
)
from config import Config
from forecasting_config import resolve_forecasting_config


PRETRAIN_HASH = "sha256:1111"


def _payload(**overrides):
    payload = {
        "stage": "pretrain",
        "run_config": {
            "seed": 7,
            "transfer_optimization": {"learning_rate": 1e-4},
            "forecast_preprocessing_hash": PRETRAIN_HASH,
        },
        "checkpoint_selection": {"metric": "val_minADE", "value": 0.37},
        "architecture": {"prediction_steps": 12, "neighborhood_radius": 2.0, "hidden_dim": 256},
    }
    payload.update(overrides)
    return payload


SOCIAL_VAE_ARCHITECTURE = {
    "prediction_steps": 12,
    "neighborhood_radius": 2.0,
    "hidden_dim": 256,
}


def test_finetune_without_seed_adopts_the_pretraining_seed():
    seed, source = resolve_stage_seed("finetune", None, _payload())
    assert seed == 7
    assert "checkpoint" in source


def test_explicit_seed_always_wins():
    assert resolve_stage_seed("finetune", 3, _payload())[0] == 3
    assert resolve_stage_seed("pretrain", 3, None)[0] == 3


def test_pretrain_and_scratch_keep_the_previous_default():
    assert resolve_stage_seed("pretrain", None, None) == (42, "stage default (42)")
    assert resolve_stage_seed("scratch", None, None)[0] == 42


def test_finetune_without_a_recorded_seed_fails_loudly():
    payload = _payload(run_config={"transfer_optimization": {"learning_rate": 1e-4}})
    with pytest.raises(ValueError, match="run_config.seed"):
        resolve_stage_seed("finetune", None, payload)


def test_both_trainers_default_seed_to_none_and_use_the_shared_helper():
    for module in (train_moflow, train_social_vae):
        assert module.resolve_stage_seed is resolve_stage_seed
        assert "resolve_stage_seed(" in Path(module.__file__).read_text(encoding="utf-8")
    social_vae_seed = [
        action for action in train_social_vae._parser()._actions if action.dest == "seed"
    ][0]
    assert social_vae_seed.default is None
    assert '"--seed",type=int,default=' not in Path(train_moflow.__file__).read_text(
        encoding="utf-8"
    )


def test_matching_payload_passes():
    validate_finetune_checkpoint(
        Path("best.pt"),
        _payload(),
        expected_preprocessing_hash=PRETRAIN_HASH,
        expected_architecture=SOCIAL_VAE_ARCHITECTURE,
    )


def test_mismatched_preprocessing_hash_names_both_values():
    with pytest.raises(ValueError) as error:
        validate_finetune_checkpoint(
            Path("best.pt"),
            _payload(),
            expected_preprocessing_hash="sha256:2222",
        )
    message = str(error.value)
    assert PRETRAIN_HASH in message and "sha256:2222" in message


def test_missing_checkpoint_hash_is_a_mismatch():
    payload = _payload()
    payload["run_config"] = {
        key: value
        for key, value in payload["run_config"].items()
        if key != "forecast_preprocessing_hash"
    }
    with pytest.raises(ValueError, match="forecast_preprocessing_hash"):
        validate_finetune_checkpoint(
            Path("best.pt"), payload, expected_preprocessing_hash=PRETRAIN_HASH
        )


@pytest.mark.parametrize(
    "key,value",
    [("prediction_steps", 8), ("neighborhood_radius", 4.0), ("hidden_dim", 128)],
)
def test_mismatched_architecture_field_raises(key, value):
    with pytest.raises(ValueError) as error:
        validate_finetune_checkpoint(
            Path("best.pt"),
            _payload(),
            expected_preprocessing_hash=PRETRAIN_HASH,
            expected_architecture={**SOCIAL_VAE_ARCHITECTURE, key: value},
        )
    message = str(error.value)
    assert key in message and repr(value) in message


def test_only_keys_present_in_the_payload_are_compared():
    validate_finetune_checkpoint(
        Path("best.pt"),
        _payload(architecture=None),
        expected_preprocessing_hash=PRETRAIN_HASH,
        expected_architecture=SOCIAL_VAE_ARCHITECTURE,
    )
    validate_finetune_checkpoint(
        Path("best.pt"),
        _payload(architecture={"hidden_dim": 256}),
        expected_preprocessing_hash=PRETRAIN_HASH,
        expected_architecture={**SOCIAL_VAE_ARCHITECTURE, "prediction_steps": 8},
    )


def test_allow_legacy_bypasses_the_guards_with_a_warning():
    with pytest.warns(UserWarning, match="allow-legacy-checkpoint"):
        validate_finetune_checkpoint(
            Path("last.pt"),
            _payload(stage="finetune"),
            allow_legacy=True,
            expected_preprocessing_hash="sha256:2222",
        )


def test_loader_returns_a_three_tuple_without_a_dataset():
    assert train_social_vae._loader(None, argparse.Namespace(), training=False) == (
        None,
        None,
        None,
    )


def test_validation_mode_defaults_to_none_in_the_parser():
    social_vae_mode = [
        action
        for action in train_social_vae._parser()._actions
        if action.dest == "validation_mode"
    ][0]
    assert social_vae_mode.default is None
    moflow_source = Path(train_moflow.__file__).read_text(encoding="utf-8")
    assert '"--validation-mode",choices=["none","grouped"],default=' not in moflow_source


@pytest.mark.parametrize("target", ["madras", "grand_central"])
def test_explicit_validation_mode_is_rejected_for_non_eth_ucy_targets(target):
    parser = argparse.ArgumentParser()
    parser.add_argument("--validation-mode", choices=["none", "grouped"])
    args = parser.parse_args(["--validation-mode", "grouped"])
    args.target_dataset = target
    with pytest.raises(SystemExit):
        _resolve_validation_mode(parser, args)


def _resolve_validation_mode(parser, args):
    """Mirror of the trainers' own resolution."""
    if args.target_dataset == "eth_ucy":
        args.validation_mode = args.validation_mode or "grouped"
    elif args.validation_mode is not None:
        parser.error("--validation-mode applies only to --target-dataset eth_ucy")
    return args.validation_mode


@pytest.mark.parametrize(
    "target,given,expected",
    [
        ("eth_ucy", None, "grouped"),
        ("eth_ucy", "none", "none"),
        ("madras", None, None),
        ("grand_central", None, None),
    ],
)
def test_validation_mode_resolution(target, given, expected):
    parser = argparse.ArgumentParser()
    args = argparse.Namespace(target_dataset=target, validation_mode=given)
    assert _resolve_validation_mode(parser, args) == expected


@pytest.mark.parametrize("module", [train_moflow, train_social_vae])
def test_trainers_reject_explicit_validation_mode_for_other_targets(module):
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "--validation-mode applies only to --target-dataset eth_ucy" in source


class _Source:
    def __init__(self, observation: int, prediction: int):
        self.history_steps = observation
        self.future_steps = prediction

    def __len__(self) -> int:
        return 0


def test_primary_trajectory_dataset_rejects_a_non_8_12_window():
    with pytest.raises(ValueError, match="8 observed / 12 predicted"):
        PrimaryTrajectoryDataset(_Source(10, 10), window=(10, 10))
    with pytest.raises(ValueError, match="8 observed / 12 predicted"):
        PrimaryTrajectoryDataset(_Source(10, 10))


def test_primary_trajectory_dataset_accepts_the_pinned_window():
    dataset = PrimaryTrajectoryDataset(_Source(8, 12))
    assert dataset.window == (8, 12)
    assert len(dataset) == 0


def test_flat_field_conflicting_with_an_explicit_forecasting_config_raises():
    forecasting = resolve_forecasting_config(preset="steps_default")
    assert forecasting.augmentation.random_rotation.enabled is False
    with pytest.raises(ValueError, match="train_rotation"):
        Config(train_rotation=True, forecasting=forecasting)
    with pytest.raises(ValueError, match="model_normalization"):
        Config(model_normalization="world", forecasting=forecasting)


def test_flat_field_without_a_forecasting_config_stays_legal():
    assert Config(train_rotation=False).train_rotation is False


def test_trainers_no_longer_pass_dead_flat_fields():
    for module in ("train_moflow", "train_social_vae"):
        source = (Path(__file__).parents[1] / "src" / "baselines" / f"{module}.py").read_text(
            encoding="utf-8"
        )
        assert "train_rotation" not in source


def _resume_arithmetic(resume_state):
    """Mirrors the ``start_epoch``/``resume_batch`` formula in each trainer's ``main()``."""
    start_epoch = resume_state.epoch + (1 if resume_state.epoch_complete else 0)
    resume_batch = 0 if resume_state.epoch_complete else resume_state.batch_in_epoch
    return start_epoch, resume_batch


def _stage_optimization() -> StageOptimization:
    return StageOptimization(
        optimizer="Adam",
        learning_rate=1e-3,
        lr_multiplier=1.0,
        weight_decay=0.0,
        batch_size=8,
        max_epochs=10,
        batches_per_epoch=None,
        scheduler="none",
        warmup_fraction=0.0,
        min_lr=1e-6,
        grad_clip=None,
        early_stopping=True,
        patience=5,
        min_delta=1e-4,
        checkpoint_metric="val_minADE",
        reset_optimizer=True,
        reset_scheduler=True,
        train_all_layers=True,
    )


def _stopper() -> EarlyStopping:
    stopper = EarlyStopping(5, 1e-4)
    stopper.update(0.5)
    return stopper


@pytest.mark.parametrize("epoch_complete", [False, True])
def test_moflow_save_truncated_epoch_resumes_mid_epoch(tmp_path, epoch_complete):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"model": "moflow", "stage": "pretrain"}))

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimization = _stage_optimization()
    path = run_dir / "checkpoints" / "latest.pt"

    train_moflow.save(
        path, model, optimizer, None, 23, "pretrain", optimization, 2, None, None,
        stopper=_stopper(), batch_in_epoch=3, epoch_complete=epoch_complete,
        loader_generator_epoch_state=None, loader_generator_state=None,
    )

    resume_state = load_resume_state(
        path, stage="pretrain", optimization=optimization_as_dict(optimization),
        expected_run_config={},
    )
    assert resume_state.epoch_complete is epoch_complete
    assert resume_state.batch_in_epoch == 3
    start_epoch, resume_batch = _resume_arithmetic(resume_state)
    if epoch_complete:
        assert (start_epoch, resume_batch) == (3, 0)
    else:
        assert (start_epoch, resume_batch) == (2, 3)


@pytest.mark.parametrize("epoch_complete", [False, True])
def test_social_vae_save_truncated_epoch_resumes_mid_epoch(tmp_path, epoch_complete):
    run_dir = tmp_path / "run"
    (run_dir / "checkpoints").mkdir(parents=True)
    (run_dir / "config.json").write_text(json.dumps({"model": "social_vae", "stage": "pretrain"}))

    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimization = _stage_optimization()
    args = SimpleNamespace(
        stage="pretrain",
        optimization=optimization,
        forecasting=SimpleNamespace(window=SimpleNamespace(prediction_steps=12)),
        neighborhood_radius=2.0,
        hidden_dim=256,
    )
    path = run_dir / "checkpoints" / "latest.pt"

    train_social_vae._save(
        path, model, optimizer, None, args, 2, 23,
        selection_value=None, best_epoch=None, stopper=_stopper(),
        batch_in_epoch=3, epoch_complete=epoch_complete,
        loader_generator_epoch_state=None, loader_generator_state=None,
    )

    resume_state = load_resume_state(
        path, stage="pretrain", optimization=optimization_as_dict(optimization),
        expected_run_config={},
    )
    assert resume_state.epoch_complete is epoch_complete
    assert resume_state.batch_in_epoch == 3
    start_epoch, resume_batch = _resume_arithmetic(resume_state)
    if epoch_complete:
        assert (start_epoch, resume_batch) == (3, 0)
    else:
        assert (start_epoch, resume_batch) == (2, 3)
