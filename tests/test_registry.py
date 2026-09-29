from __future__ import annotations

from pathlib import Path

import pytest

from baselines.registry import register, resolve


def _make_checkpoint(path: Path, content: bytes = b"checkpoint-bytes") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_register_then_resolve_by_id(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("TRAJECTORY_MODEL_REGISTRY", str(registry_dir))

    checkpoint = _make_checkpoint(tmp_path / "runs/model_a/checkpoints/best.pt")
    entry_path = register("model_a", checkpoint, model="social_vae", dataset="eth", seed=0)

    assert entry_path == registry_dir / "model_a.json"
    assert entry_path.is_file()

    resolved = resolve("model_a")
    assert resolved == checkpoint.resolve()


def test_reregister_same_id_different_checkpoint_raises(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("TRAJECTORY_MODEL_REGISTRY", str(registry_dir))

    first_checkpoint = _make_checkpoint(tmp_path / "run1/checkpoints/best.pt")
    second_checkpoint = _make_checkpoint(tmp_path / "run2/checkpoints/best.pt")

    register("model_b", first_checkpoint)

    with pytest.raises(FileExistsError):
        register("model_b", second_checkpoint)

    assert resolve("model_b") == first_checkpoint.resolve()


def test_reregister_same_id_same_checkpoint_is_idempotent(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("TRAJECTORY_MODEL_REGISTRY", str(registry_dir))

    checkpoint = _make_checkpoint(tmp_path / "run/checkpoints/best.pt")

    register("model_c", checkpoint)
    register("model_c", checkpoint)

    assert resolve("model_c") == checkpoint.resolve()


def test_resolve_unregistered_id_raises(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("TRAJECTORY_MODEL_REGISTRY", str(registry_dir))

    with pytest.raises(FileNotFoundError):
        resolve("no_such_model_id")


def test_resolve_directory_falls_back_to_checkpoints_best_pt(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("TRAJECTORY_MODEL_REGISTRY", str(registry_dir))

    run_dir = tmp_path / "run_dir"
    checkpoint = _make_checkpoint(run_dir / "checkpoints/best.pt")

    resolved = resolve(run_dir)
    assert resolved == checkpoint.resolve()


def test_resolve_direct_file_path_bypasses_registry(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("TRAJECTORY_MODEL_REGISTRY", str(registry_dir))

    checkpoint = _make_checkpoint(tmp_path / "standalone/best.pt")
    resolved = resolve(checkpoint)
    assert resolved == checkpoint.resolve()


def test_register_missing_checkpoint_raises(tmp_path, monkeypatch):
    registry_dir = tmp_path / "registry"
    monkeypatch.setenv("TRAJECTORY_MODEL_REGISTRY", str(registry_dir))

    with pytest.raises(FileNotFoundError):
        register("model_missing", tmp_path / "does_not_exist.pt")
