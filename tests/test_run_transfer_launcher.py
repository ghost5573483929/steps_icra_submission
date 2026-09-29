"""Tests for the shell logic of scripts/run_transfer_experiment.sh, with a stub
interpreter (passed via ``--python``) in place of the real trainer and evaluator.
"""

from __future__ import annotations

import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts" / "run_transfer_experiment.sh"

STUB_SOURCE = '''#!/usr/bin/env python3
"""Stub interpreter: records argv and fabricates the artifacts the real
trainer/evaluator entry points would produce for --run-dir."""
import json
import os
import sys
from pathlib import Path

argv = sys.argv[1:]
with open(os.environ["STUB_ARGV_LOG"], "a") as fh:
    fh.write(json.dumps(argv) + "\\n")

assert argv[0] == "-m", argv
module = argv[1]
rest = argv[2:]


def get_opt(name):
    if name in rest:
        return rest[rest.index(name) + 1]
    return None


run_dir = get_opt("--run-dir")
if run_dir is not None:
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    if module in ("baselines.train_moflow", "baselines.train_social_vae"):
        (run_dir / "status.json").write_text(json.dumps({"stage": "done"}))
        checkpoints = run_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        (checkpoints / "best.pt").write_bytes(b"stub-checkpoint")
    elif module in ("baselines.evaluate_moflow", "baselines.evaluate_social_vae"):
        (run_dir / "evaluation.json").write_text(json.dumps({"ade": 1.0, "fde": 2.0}))
    else:
        raise AssertionError(f"unexpected module: {module}")
'''


@pytest.fixture()
def stub_python(tmp_path: Path) -> Path:
    stub = tmp_path / "stub_python.py"
    stub.write_text(STUB_SOURCE)
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub


@pytest.fixture()
def registry(tmp_path: Path) -> Path:
    registry_dir = tmp_path / "registry"
    registry_dir.mkdir()
    checkpoint_stand_in = tmp_path / "pretrained_checkpoint.pt"
    checkpoint_stand_in.write_bytes(b"pretrained")
    (registry_dir / "moflow_transfer_v2_steps_0.json").write_text(
        json.dumps({"checkpoint": str(checkpoint_stand_in), "seed": 7})
    )
    return registry_dir


def run_launcher(args, *, tmp_path: Path, stub_python: Path, registry: Path, argv_log: Path):
    env = dict(os.environ)
    env["STUB_ARGV_LOG"] = str(argv_log)
    env["TRAJECTORY_MODEL_REGISTRY"] = str(registry)
    return subprocess.run(
        ["bash", str(LAUNCHER), *args, "--python", str(stub_python)],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )


def read_argv_log(argv_log: Path) -> list[list[str]]:
    if not argv_log.exists():
        return []
    return [json.loads(line) for line in argv_log.read_text().splitlines() if line]


def test_finetune_without_seed_resolves_from_registry(tmp_path, stub_python, registry):
    output = tmp_path / "output"
    argv_log = tmp_path / "argv.log"
    result = run_launcher(
        [
            "--model", "moflow", "--stage", "finetune",
            "--checkpoint", "moflow_transfer_v2_steps_0",
            "--target", "eth_ucy", "--fold", "eth", "--fraction", "0.25",
            "--output", str(output),
        ],
        tmp_path=tmp_path, stub_python=stub_python, registry=registry, argv_log=argv_log,
    )
    assert result.returncode == 0, result.stderr
    run_dir = output / "finetune" / "moflow" / "moflow_transfer_v2_steps_0" / "seed7" / "eth_ucy-eth" / "fraction-0.25"
    assert run_dir.is_dir()
    assert (run_dir / "test" / "evaluation.json").is_file()
    assert "Resolved --seed=7" in result.stdout


def test_explicit_seed_overrides_resolution(tmp_path, stub_python, registry):
    output = tmp_path / "output"
    argv_log = tmp_path / "argv.log"
    result = run_launcher(
        [
            "--model", "moflow", "--stage", "finetune",
            "--checkpoint", "moflow_transfer_v2_steps_0", "--seed", "3",
            "--target", "eth_ucy", "--fold", "eth", "--fraction", "0.25",
            "--output", str(output),
        ],
        tmp_path=tmp_path, stub_python=stub_python, registry=registry, argv_log=argv_log,
    )
    assert result.returncode == 0, result.stderr
    run_dir = output / "finetune" / "moflow" / "moflow_transfer_v2_steps_0" / "seed3" / "eth_ucy-eth" / "fraction-0.25"
    assert run_dir.is_dir()


def test_scratch_without_seed_exits_2(tmp_path, stub_python, registry):
    output = tmp_path / "output"
    argv_log = tmp_path / "argv.log"
    result = run_launcher(
        [
            "--model", "moflow", "--stage", "scratch",
            "--target", "eth_ucy", "--fold", "eth", "--fraction", "0.25",
            "--output", str(output),
        ],
        tmp_path=tmp_path, stub_python=stub_python, registry=registry, argv_log=argv_log,
    )
    assert result.returncode == 2
    assert "--seed is required" in result.stderr


def test_direct_stage_runs_only_the_evaluator(tmp_path, stub_python, registry):
    output = tmp_path / "output"
    argv_log = tmp_path / "argv.log"
    result = run_launcher(
        [
            "--model", "moflow", "--stage", "direct",
            "--checkpoint", "moflow_transfer_v2_steps_0",
            "--target", "eth_ucy", "--fold", "eth",
            "--output", str(output),
        ],
        tmp_path=tmp_path, stub_python=stub_python, registry=registry, argv_log=argv_log,
    )
    assert result.returncode == 0, result.stderr
    run_dir = output / "direct" / "moflow" / "moflow_transfer_v2_steps_0" / "seed7" / "eth_ucy-eth"
    assert run_dir.is_dir()
    assert (run_dir / "test" / "evaluation.json").is_file()
    assert not (run_dir / "train").exists()

    calls = read_argv_log(argv_log)
    assert len(calls) == 1
    module, rest = calls[0][1], calls[0][2:]
    assert module == "baselines.evaluate_moflow"
    assert "--checkpoint" in rest
    assert rest[rest.index("--checkpoint") + 1] == "moflow_transfer_v2_steps_0"


def test_repeated_finetune_run_is_refused_and_params_env_untouched(tmp_path, stub_python, registry):
    output = tmp_path / "output"
    argv_log = tmp_path / "argv.log"
    args = [
        "--model", "moflow", "--stage", "finetune",
        "--checkpoint", "moflow_transfer_v2_steps_0",
        "--target", "eth_ucy", "--fold", "eth", "--fraction", "0.25",
        "--output", str(output),
    ]
    first = run_launcher(args, tmp_path=tmp_path, stub_python=stub_python, registry=registry, argv_log=argv_log)
    assert first.returncode == 0, first.stderr
    params_env = output / "finetune" / "moflow" / "moflow_transfer_v2_steps_0" / "seed7" / "eth_ucy-eth" / "fraction-0.25" / "params.env"
    before = params_env.read_bytes()

    second = run_launcher(args, tmp_path=tmp_path, stub_python=stub_python, registry=registry, argv_log=argv_log)
    assert second.returncode == 3
    assert "Refusing to overwrite" in second.stderr

    after = params_env.read_bytes()
    assert before == after


def test_num_workers_and_validation_mode_forward_only_to_trainer(tmp_path, stub_python, registry):
    output = tmp_path / "output"
    argv_log = tmp_path / "argv.log"
    result = run_launcher(
        [
            "--model", "socialvae", "--stage", "scratch", "--seed", "1",
            "--target", "eth_ucy", "--fold", "eth", "--fraction", "0.25",
            "--num-workers", "2", "--validation-mode", "none",
            "--output", str(output),
        ],
        tmp_path=tmp_path, stub_python=stub_python, registry=registry, argv_log=argv_log,
    )
    assert result.returncode == 0, result.stderr
    calls = read_argv_log(argv_log)
    assert len(calls) == 2
    trainer_module, trainer_args = calls[0][1], calls[0][2:]
    evaluator_module, evaluator_args = calls[1][1], calls[1][2:]
    assert trainer_module == "baselines.train_social_vae"
    assert evaluator_module == "baselines.evaluate_social_vae"

    assert "--num-workers" in trainer_args
    assert trainer_args[trainer_args.index("--num-workers") + 1] == "2"
    assert "--validation-mode" in trainer_args
    assert trainer_args[trainer_args.index("--validation-mode") + 1] == "none"

    assert "--validation-mode" not in evaluator_args


def test_launcher_has_valid_bash_syntax():
    result = subprocess.run(["bash", "-n", str(LAUNCHER)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
