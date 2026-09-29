"""``pytest.ini`` sets ``addopts = -m "not slow"``, so a bare ``pytest`` skips the
``@pytest.mark.slow`` tests here and reports green after running only the parser test.
Run them explicitly with::

    python -m pytest -m slow tests/test_train_moflow.py
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from tqdm.auto import tqdm as tqdm_class

from baselines import train_moflow
from baselines.moflow import PrimaryTrajectoryDataset


class _FakeSpec:
    name = "faketest"


class _FakeScene:
    """One agent over 20 timesteps: 8 past + 12 future."""

    def __init__(self, seed: int):
        rng = np.random.default_rng(seed)
        self.xy = rng.normal(size=(20, 1, 2)).astype(np.float32)


class _FakeSource:
    def __init__(self, n: int, split: str):
        self._scenes = [_FakeScene(seed) for seed in range(n)]
        self.spec = _FakeSpec()
        self.split = split
        self.rows = [
            {
                "scene_sequence": f"seq-{i}",
                "recording": f"rec-{i}",
                "start_step": 0,
                "primary_track_id": i,
                "scene_tag": 0,
                "accepted": True,
                "track_kept": True,
                "eval_kept": True,
            }
            for i in range(n)
        ]

    def __len__(self):
        return len(self._scenes)

    def __getitem__(self, index):
        return self._scenes[index]


def _fake_datasets(stage, fold, limit, data_seed=42, model_seed=42, source_datasets=("steps",),
                    preprocessing=None, validation_mode="grouped", target_dataset=None, target_fraction=1.0):
    assert stage == "pretrain"
    train = PrimaryTrajectoryDataset(_FakeSource(6, "train"), limit)
    val = PrimaryTrajectoryDataset(_FakeSource(4, "val"), limit)
    return train, val, None


def _run(monkeypatch, run_dir: Path, extra_args: list[str], status_every: int = 100):
    monkeypatch.setattr(train_moflow, "datasets", _fake_datasets)
    argv = [
        "train_moflow.py", "--stage", "pretrain", "--run-dir", str(run_dir),
        "--device", "cpu",
        "--epochs", "1", "--batch-size", "2", "--no-early-stopping",
        "--status-every", str(status_every), "--pretrain-dataset", "steps",
        *extra_args,
    ]
    monkeypatch.setattr(sys, "argv", argv)
    train_moflow.main()


def _spy_dataloader(monkeypatch) -> list[dict]:
    """Record the kwargs of every ``train_moflow.DataLoader`` construction, in call order."""
    calls: list[dict] = []
    real_loader = train_moflow.DataLoader

    def spy_loader(dataset, *args, **kwargs):
        calls.append(kwargs)
        return real_loader(dataset, *args, **kwargs)

    monkeypatch.setattr(train_moflow, "DataLoader", spy_loader)
    return calls


class _ParserCaptured(Exception):
    pass


def _capture_parser(monkeypatch, main) -> argparse.ArgumentParser:
    """Return the parser ``main()`` builds inline, by intercepting its ``parse_args`` call."""
    captured: list[argparse.ArgumentParser] = []
    original = argparse.ArgumentParser.parse_args

    def capturing_parse_args(self, *args, **kwargs):
        captured.append(self)
        raise _ParserCaptured

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", capturing_parse_args)
    monkeypatch.setattr(sys, "argv", ["trainer.py"])
    with pytest.raises(_ParserCaptured):
        main()
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", original)
    assert len(captured) == 1, f"expected exactly one parse_args call, got {len(captured)}"
    return captured[0]


class TestNumWorkersConfiguration:
    def test_num_workers_flag_exists_and_defaults_to_repository_convention(self, monkeypatch, tmp_path):
        parser = _capture_parser(monkeypatch, train_moflow.main)
        required = ["--stage", "pretrain", "--run-dir", str(tmp_path / "run")]
        assert parser.parse_args(required).num_workers == 4
        assert parser.parse_args([*required, "--num-workers", "3"]).num_workers == 3

    @pytest.mark.slow
    def test_loaders_are_configured_with_workers_pin_memory_and_seed_worker(self, monkeypatch, tmp_path):
        calls = _spy_dataloader(monkeypatch)
        _run(monkeypatch, tmp_path / "run", ["--iterations", "1", "--num-workers", "0"])

        assert len(calls) == 2, "expected exactly one train and one val DataLoader"
        for kwargs in calls:
            assert kwargs["num_workers"] == 0
            assert kwargs["pin_memory"] is True
            assert kwargs["persistent_workers"] is False
            assert kwargs["worker_init_fn"] is train_moflow.seed_worker

    @pytest.mark.slow
    def test_persistent_workers_follows_num_workers_positive(self, monkeypatch, tmp_path):
        calls = _spy_dataloader(monkeypatch)
        _run(monkeypatch, tmp_path / "run", ["--iterations", "1", "--num-workers", "2"])

        assert len(calls) == 2, "expected exactly one train and one val DataLoader"
        for kwargs in calls:
            assert kwargs["num_workers"] == 2
            assert kwargs["persistent_workers"] is True


class TestPostfixThrottling:
    @pytest.mark.slow
    def test_set_postfix_called_only_at_status_every_cadence(self, monkeypatch, tmp_path):
        calls = []
        original = tqdm_class.set_postfix

        def counting_set_postfix(self, *args, **kwargs):
            calls.append(kwargs)
            return original(self, *args, **kwargs)

        monkeypatch.setattr(tqdm_class, "set_postfix", counting_set_postfix)
        # 6 training scenes, batch_size 2 -> 3 iterations/epoch; status_every=2 -> exactly 1 postfix update.
        _run(monkeypatch, tmp_path / "run", ["--num-workers", "0"], status_every=2)
        assert len(calls) == 1
        assert calls[0]["iteration"] == 2

    @pytest.mark.slow
    def test_metrics_jsonl_unaffected_by_postfix_throttle(self, monkeypatch, tmp_path):
        run_dir = tmp_path / "run"
        _run(monkeypatch, run_dir, ["--num-workers", "0"], status_every=2)
        metrics_path = run_dir / "metrics.jsonl"
        assert metrics_path.exists()
        lines = [line for line in metrics_path.read_text().splitlines() if line.strip()]
        assert any('"iteration": 2' in line or '"iteration":2' in line for line in lines)


class TestUnusedBatchKeysNotTransferred:
    @pytest.mark.slow
    def test_fut_traj_original_scale_and_indexes_are_not_sent_to_device(self, monkeypatch, tmp_path):
        seen_batches = []

        import baselines.moflow as moflow_module
        original_build = moflow_module.build

        def spying_build(device, epochs=150):
            cfg, model = original_build(device, epochs)
            real_call = model.forward

            def spy_forward(batch, *args, **kwargs):
                seen_batches.append(batch)
                return real_call(batch, *args, **kwargs)

            model.forward = spy_forward
            return cfg, model

        monkeypatch.setattr(train_moflow, "build", spying_build)
        _run(monkeypatch, tmp_path / "run", ["--iterations", "1", "--num-workers", "0"])

        assert seen_batches, "expected at least one training forward call"
        batch = seen_batches[0]
        assert "fut_traj_original_scale" not in batch
        assert "indexes" not in batch
        assert "fut_traj" in batch and "past_traj_original_scale" in batch
        assert torch.is_tensor(batch["past_traj_original_scale"])
