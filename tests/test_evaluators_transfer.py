from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from data import Scene


def _synthetic_scene(offset: float = 0.0) -> Scene:
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


def _run_with_argv(argv, fn):
    original_argv = sys.argv
    sys.argv = argv
    try:
        fn()
    finally:
        sys.argv = original_argv


def test_moflow_evaluation_raises_on_empty_population(tmp_path: Path):
    from baselines import evaluate_moflow
    from baselines.moflow import build as moflow_build

    device = torch.device("cpu")
    _, model = moflow_build(device)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    run_dir = tmp_path / "run"
    argv = [
        "evaluate_moflow.py", "--fold", "eth", "--checkpoint", str(checkpoint),
        "--run-dir", str(run_dir), "--device", "cpu", "--visual-scenes", "0",
    ]

    def call():
        with patch("baselines.evaluate_moflow.downstream_test_dataset", return_value=[]):
            with pytest.raises(RuntimeError, match="MoFlow evaluation population is empty"):
                evaluate_moflow.main()

    _run_with_argv(argv, call)


def test_social_vae_evaluation_raises_on_empty_population(tmp_path: Path):
    from baselines import evaluate_social_vae
    from baselines.social_vae import build as social_vae_build

    device = torch.device("cpu")
    model = social_vae_build(device, hidden_dim=8)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model": model.state_dict()}, checkpoint)
    run_dir = tmp_path / "run"
    argv = [
        "evaluate_social_vae.py", "--fold", "eth", "--checkpoint", str(checkpoint),
        "--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0",
        "--visual-scenes", "0", "--hidden-dim", "8",
    ]

    def call():
        with patch("baselines.evaluate_social_vae.downstream_test_dataset", return_value=[]):
            with pytest.raises(RuntimeError, match="SocialVAE evaluation population is empty"):
                evaluate_social_vae.main()

    _run_with_argv(argv, call)


def test_moflow_evaluation_rejects_preprocessing_hash_mismatch(tmp_path: Path):
    from baselines import evaluate_moflow
    from baselines.moflow import build as moflow_build

    device = torch.device("cpu")
    _, model = moflow_build(device)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "run_config": {"forecast_preprocessing_hash": "sha256:not-real"}},
        checkpoint,
    )
    run_dir = tmp_path / "run"
    argv = [
        "evaluate_moflow.py", "--fold", "eth", "--checkpoint", str(checkpoint),
        "--run-dir", str(run_dir), "--device", "cpu", "--visual-scenes", "0",
    ]

    def call():
        with patch("baselines.evaluate_moflow.downstream_test_dataset", return_value=[]):
            with pytest.raises(ValueError) as excinfo:
                evaluate_moflow.main()
            message = str(excinfo.value)
            assert "sha256:not-real" in message
            assert "sha256:" in message.replace("sha256:not-real", "", 1)

    _run_with_argv(argv, call)


def test_moflow_evaluation_records_pretrain_stage_fields(tmp_path: Path):
    from baselines import evaluate_moflow
    from baselines.moflow import build as moflow_build

    device = torch.device("cpu")
    _, model = moflow_build(device)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "run_config": {"stage": "pretrain", "seed": 7, "data_seed": 42},
            "checkpoint_selection": {"metric": "val_minADE", "value": 0.5, "best_epoch": 3},
        },
        checkpoint,
    )
    run_dir = tmp_path / "run"
    argv = [
        "evaluate_moflow.py", "--fold", "eth", "--checkpoint", str(checkpoint),
        "--run-dir", str(run_dir), "--device", "cpu", "--visual-scenes", "0",
        "--data-seed", "42", "--batch-size", "1",
    ]

    def call():
        with patch("baselines.evaluate_moflow.downstream_test_dataset", return_value=[_synthetic_scene()]):
            evaluate_moflow.main()

    _run_with_argv(argv, call)

    import json
    result = json.loads((run_dir / "evaluation.json").read_text())
    assert result["training_stage"] == "pretrain"
    assert result["fine_tuning_learning_rate"] is None
    assert result["fine_tuning_best_epoch"] is None
    assert result["fine_tuning_validation_metric"] is None
    assert result["pretraining_best_epoch"] == 3
    assert result["pretraining_validation_metric"] == 0.5
    assert result["data_seed"] == 42
    assert result["checkpoint_data_seed"] == 42
    assert result["checkpoint_reference"] == str(checkpoint)
    assert result["finetuning_seed"] == 7


def test_moflow_evaluation_warns_on_data_seed_mismatch(tmp_path: Path):
    from baselines import evaluate_moflow
    from baselines.moflow import build as moflow_build

    device = torch.device("cpu")
    _, model = moflow_build(device)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "run_config": {"stage": "pretrain", "data_seed": 1}},
        checkpoint,
    )
    run_dir = tmp_path / "run"
    argv = [
        "evaluate_moflow.py", "--fold", "eth", "--checkpoint", str(checkpoint),
        "--run-dir", str(run_dir), "--device", "cpu", "--visual-scenes", "0",
        "--data-seed", "42", "--batch-size", "1",
    ]

    def call():
        with patch("baselines.evaluate_moflow.downstream_test_dataset", return_value=[_synthetic_scene()]):
            with pytest.warns(UserWarning, match="data_seed"):
                evaluate_moflow.main()

    _run_with_argv(argv, call)

    import json
    result = json.loads((run_dir / "evaluation.json").read_text())
    assert result["data_seed"] == 42
    assert result["checkpoint_data_seed"] == 1


def test_social_vae_evaluation_records_pretrain_stage_fields(tmp_path: Path):
    from baselines import evaluate_social_vae
    from baselines.social_vae import build as social_vae_build

    device = torch.device("cpu")
    model = social_vae_build(device, hidden_dim=8)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "run_config": {"stage": "pretrain", "seed": 7, "data_seed": 42},
            "checkpoint_selection": {"metric": "val_minADE", "value": 0.5, "best_epoch": 3},
        },
        checkpoint,
    )
    run_dir = tmp_path / "run"
    argv = [
        "evaluate_social_vae.py", "--fold", "eth", "--checkpoint", str(checkpoint),
        "--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0",
        "--visual-scenes", "0", "--hidden-dim", "8", "--data-seed", "42",
        "--batch-size", "1",
    ]

    def call():
        with patch("baselines.evaluate_social_vae.downstream_test_dataset", return_value=[_synthetic_scene()]):
            evaluate_social_vae.main()

    _run_with_argv(argv, call)

    import json
    result = json.loads((run_dir / "evaluation.json").read_text())
    assert result["training_stage"] == "pretrain"
    assert result["fine_tuning_learning_rate"] is None
    assert result["fine_tuning_best_epoch"] is None
    assert result["fine_tuning_validation_metric"] is None
    assert result["pretraining_best_epoch"] == 3
    assert result["pretraining_validation_metric"] == 0.5
    assert result["data_seed"] == 42
    assert result["checkpoint_data_seed"] == 42
    assert result["checkpoint_reference"] == str(checkpoint)
    assert result["finetuning_seed"] == 7


def test_social_vae_evaluation_warns_on_data_seed_mismatch(tmp_path: Path):
    from baselines import evaluate_social_vae
    from baselines.social_vae import build as social_vae_build

    device = torch.device("cpu")
    model = social_vae_build(device, hidden_dim=8)
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {"model": model.state_dict(), "run_config": {"stage": "pretrain", "data_seed": 1}},
        checkpoint,
    )
    run_dir = tmp_path / "run"
    argv = [
        "evaluate_social_vae.py", "--fold", "eth", "--checkpoint", str(checkpoint),
        "--run-dir", str(run_dir), "--device", "cpu", "--num-workers", "0",
        "--visual-scenes", "0", "--hidden-dim", "8", "--data-seed", "42",
        "--batch-size", "1",
    ]

    def call():
        with patch("baselines.evaluate_social_vae.downstream_test_dataset", return_value=[_synthetic_scene()]):
            with pytest.warns(UserWarning, match="data_seed"):
                evaluate_social_vae.main()

    _run_with_argv(argv, call)

    import json
    result = json.loads((run_dir / "evaluation.json").read_text())
    assert result["data_seed"] == 42
    assert result["checkpoint_data_seed"] == 1
