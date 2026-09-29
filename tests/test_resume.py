from __future__ import annotations

import argparse
from pathlib import Path

import pytest
import torch

from baselines import resume
from baselines.early_stopping import EarlyStopping
from baselines.resume import (
    add_resume_argument,
    checkpoint_resume_fields,
    load_resume_state,
    resolve_resume_checkpoint,
    restore_training_state,
    resume_guard_config,
)


def _checkpoint(path: Path, *, stage: str = "pretrain") -> tuple[torch.nn.Module, torch.optim.Optimizer]:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss = model(torch.ones(1, 2)).sum()
    loss.backward()
    optimizer.step()
    stopper = EarlyStopping(5, 1e-4)
    stopper.update(0.75)
    run_config = {
        "model": "test",
        "stage": stage,
        "seed": 3,
        "forecast_preprocessing_hash": "sha256:test",
    }
    generator = torch.Generator().manual_seed(7)
    epoch_state = generator.get_state()
    torch.randperm(10, generator=generator)
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "stage": stage,
            "epoch": 4,
            "iteration": 42,
            "optimization": {"optimizer": "Adam", "learning_rate": 1e-3},
            "checkpoint_selection": {"value": 0.75, "best_epoch": 3},
            "run_config": run_config,
            **checkpoint_resume_fields(
                stopper=stopper,
                batch_in_epoch=2,
                epoch_complete=False,
                loader_generator_epoch_state=epoch_state,
                loader_generator_state=generator.get_state(),
            ),
        },
        path,
    )
    return model, optimizer


def test_resume_and_resuming_are_equivalent_cli_aliases(tmp_path: Path):
    parser = argparse.ArgumentParser()
    add_resume_argument(parser)
    assert parser.parse_args(["--resume"]).resume == "auto"
    assert parser.parse_args(["--resuming"]).resume == "auto"
    explicit = tmp_path / "checkpoint.pt"
    assert parser.parse_args(["--resuming", str(explicit)]).resume == str(explicit)


def test_auto_resume_resolves_latest_checkpoint(tmp_path: Path):
    latest = tmp_path / "checkpoints" / "latest.pt"
    latest.parent.mkdir()
    latest.touch()
    assert resolve_resume_checkpoint(tmp_path, "auto") == latest.resolve()


def test_resume_restores_full_training_state(tmp_path: Path):
    path = tmp_path / "latest.pt"
    original, original_optimizer = _checkpoint(path)
    expected_config = {
        "model": "test",
        "stage": "pretrain",
        "seed": 3,
        "forecast_preprocessing_hash": "sha256:test",
    }
    state = load_resume_state(
        path,
        stage="pretrain",
        optimization={"optimizer": "Adam", "learning_rate": 1e-3},
        expected_run_config=expected_config,
    )
    assert (state.epoch, state.iteration, state.batch_in_epoch, state.epoch_complete) == (4, 42, 2, False)
    assert state.best == pytest.approx(0.75)
    assert state.bad_epochs == 0
    assert state.best_epoch == 3

    restored = torch.nn.Linear(2, 1)
    restored_optimizer = torch.optim.Adam(restored.parameters(), lr=9e-2)
    restore_training_state(state, restored, restored_optimizer, None, torch.device("cpu"))
    for expected, actual in zip(original.parameters(), restored.parameters(), strict=True):
        torch.testing.assert_close(expected, actual)
    assert restored_optimizer.state_dict()["state"].keys() == original_optimizer.state_dict()["state"].keys()
    assert restored_optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)


def test_resume_rejects_changed_training_contract(tmp_path: Path):
    path = tmp_path / "latest.pt"
    _checkpoint(path)
    with pytest.raises(ValueError, match="optimization does not match"):
        load_resume_state(
            path,
            stage="pretrain",
            optimization={"optimizer": "Adam", "learning_rate": 2e-3},
            expected_run_config={},
        )
    with pytest.raises(ValueError, match="configuration mismatch"):
        load_resume_state(
            path,
            stage="pretrain",
            optimization={"optimizer": "Adam", "learning_rate": 1e-3},
            expected_run_config={"seed": 99},
        )


RUNSTATUS_ADDED_FIELDS = frozenset(
    {
        "git_commit",
        "git_dirty",
        "python_version",
        "platform",
        "torch_version",
        "cuda_available",
        "cuda_device_name",
        "canonical_release",
        "canonical_schema_version",
        "canonical_hz",
    }
)

TRAINER_RUN_CONFIGS = {
    "moflow": {
        "model": "moflow_teacher",
        "stage": "finetune",
        "seed": 0,
        "data_seed": 42,
        "validation_mode": "grouped",
        "max_scenes": None,
        "max_val_batches": None,
        "forecast_preprocessing_hash": "sha256:abc",
        "num_workers": 4,
        "device": "cuda",
        "device_realized": "cuda",
        "status_every": 100,
        "run_dir": "experiments/pretrained/run",
        "resume": None,
    },
    "social_vae": {
        "model": "social_vae",
        "stage": "finetune",
        "seed": 0,
        "data_seed": 42,
        "validation_mode": "grouped",
        "max_val_batches": None,
        "hidden_dim": 256,
        "neighborhood_radius": 2.0,
        "k": 20,
        "forecast_preprocessing_hash": "sha256:abc",
        "num_workers": 2,
        "device": "cuda",
        "device_realized": "cuda",
        "status_every": 25,
        "run_dir": "experiments/pretrained/run",
        "resume": None,
    },
}

# (trainer, field, recorded value, value requested on the resume)
POPULATION_AFFECTING_CHANGES = [
    ("moflow", "validation_mode", "grouped", "none"),
    ("moflow", "max_scenes", None, 500),
    ("moflow", "max_val_batches", None, 10),
    ("social_vae", "validation_mode", "grouped", "none"),
    ("social_vae", "max_val_batches", None, 10),
]


def _guarded_checkpoint(path: Path, run_config: dict, *, model_key: str = "model") -> None:
    model = torch.nn.Linear(2, 1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.save(
        {
            model_key: model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": None,
            "stage": run_config["stage"],
            "epoch": 7,
            "iteration": 70,
            "optimization": {"optimizer": "Adam", "learning_rate": 1e-3},
            "run_config": run_config,
        },
        path,
    )


def _load(path: Path, run_config: dict, trainer: str, *, model_key: str = "model"):
    return load_resume_state(
        path,
        stage=run_config["stage"],
        optimization={"optimizer": "Adam", "learning_rate": 1e-3},
        expected_run_config=resume_guard_config(run_config, trainer),
        model_key=model_key,
    )


@pytest.mark.parametrize("trainer,field,recorded,requested", POPULATION_AFFECTING_CHANGES)
def test_resume_refuses_a_changed_population_or_metric_field(
    tmp_path: Path, trainer: str, field: str, recorded, requested
):
    recorded_config = {**TRAINER_RUN_CONFIGS[trainer], field: recorded}
    requested_config = {**recorded_config, field: requested}
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)

    assert _load(path, recorded_config, trainer).epoch == 7
    with pytest.raises(ValueError, match=f"configuration mismatch: {field}:"):
        _load(path, requested_config, trainer)


def test_every_population_affecting_field_is_in_the_guarded_set():
    for trainer, field, recorded, _requested in POPULATION_AFFECTING_CHANGES:
        guarded = resume_guard_config({**TRAINER_RUN_CONFIGS[trainer], field: recorded}, trainer)
        assert field in guarded, f"{field} is not guarded for {trainer}"


def test_a_newly_added_argument_is_guarded_without_editing_any_list():
    for trainer, run_config in TRAINER_RUN_CONFIGS.items():
        invented = {**run_config, "max_agents_per_scene_that_nobody_exempted": 4}
        guarded = resume_guard_config(invented, trainer)
        assert "max_agents_per_scene_that_nobody_exempted" in guarded
        assert "max_agents_per_scene_that_nobody_exempted" not in (
            resume.RESUME_EXEMPT_FIELDS | set().union(*resume.RESUME_EXEMPT_FIELDS_BY_TRAINER.values())
        )


@pytest.mark.parametrize("trainer", sorted(TRAINER_RUN_CONFIGS))
def test_resume_accepts_changed_operational_fields(tmp_path: Path, trainer: str):
    recorded_config = TRAINER_RUN_CONFIGS[trainer]
    resumed_config = {
        **recorded_config,
        "num_workers": recorded_config.get("num_workers", 4) + 5,
        "device": "cpu",
        "device_realized": "cpu",
        "status_every": 1,
        "resume": "auto",
        "run_dir": "/somewhere/else/run",
        "iterations": 12,
    }
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    state = _load(path, resumed_config, trainer)
    assert (state.epoch, state.iteration) == (7, 70)
    assert state.unguarded_fields == ()


def test_a_field_the_checkpoint_predates_is_reported_not_refused(tmp_path: Path, caplog):
    recorded_config = dict(TRAINER_RUN_CONFIGS["moflow"])
    requested_config = {**recorded_config, "field_added_after_this_checkpoint": "grouped"}
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    with caplog.at_level("WARNING", logger="baselines.resume"):
        state = _load(path, requested_config, "moflow")
    assert state.unguarded_fields == ("field_added_after_this_checkpoint",)
    assert "field_added_after_this_checkpoint" in caplog.text


def test_a_field_recorded_as_null_is_still_guarded(tmp_path: Path):
    recorded_config = {**TRAINER_RUN_CONFIGS["moflow"], "validation_mode": None}
    requested_config = {**recorded_config, "validation_mode": "grouped"}
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    with pytest.raises(ValueError, match="configuration mismatch: validation_mode:"):
        _load(path, requested_config, "moflow")


def test_guard_compares_recorded_meaning_not_python_representation(tmp_path: Path):
    recorded_config = {
        **TRAINER_RUN_CONFIGS["moflow"],
        "source_datasets": ["steps"],
        "upstream_source": "references/MoFlow",
    }
    path = tmp_path / "latest.pt"
    _guarded_checkpoint(path, recorded_config)
    resumed_config = {**recorded_config, "source_datasets": ("steps",)}
    assert _load(path, resumed_config, "moflow").epoch == 7
    with pytest.raises(ValueError, match="configuration mismatch: source_datasets:"):
        _load(path, {**recorded_config, "source_datasets": ("steps", "eth")}, "moflow")


def test_resume_guard_rejects_an_unknown_trainer():
    with pytest.raises(KeyError, match="Unknown trainer"):
        resume_guard_config({"stage": "pretrain"}, "no_such_trainer")


LIVE_CHECKPOINTS = [
    ("experiments/pretrained/socialvae_transfer_v2_steps_0/checkpoints/best.pt", "social_vae", "model"),
]


@pytest.mark.parametrize("relative,trainer,model_key", LIVE_CHECKPOINTS)
def test_live_checkpoints_stay_resumable_under_the_inverted_guard(relative, trainer, model_key):
    path = Path(__file__).resolve().parents[1] / relative
    if not path.is_file():
        pytest.skip(f"{relative} is not present in this checkout")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    recorded = payload["run_config"]
    live = {key: value for key, value in recorded.items() if key not in RUNSTATUS_ADDED_FIELDS}
    live["resume"] = "auto"
    state = load_resume_state(
        path,
        stage=payload["stage"],
        optimization=payload["optimization"],
        expected_run_config=resume_guard_config(live, trainer),
        model_key=model_key,
    )
    assert state.unguarded_fields == ()
    assert state.epoch >= 1
    guarded = resume_guard_config(live, trainer)
    for field in ("validation_mode", "forecast_preprocessing_hash"):
        assert field in guarded
