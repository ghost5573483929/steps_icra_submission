"""Crash-safe continuation of an existing training stage.

Resume checkpoints restore the complete optimizer/scheduler/training state, unlike
transfer-initialization checkpoints, which load model weights only.
"""
from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from baselines.status import _clean

logger = logging.getLogger(__name__)

# The guard is inverted: every field a trainer records in its run configuration is compared
# and only the names below are exempt, so a newly added trainer argument is guarded by
# default. Only exempt a field whose mid-run change can alter neither the training or
# validation population nor a reported metric.
RESUME_EXEMPT_FIELDS: frozenset[str] = frozenset(
    {
        "resume",
        "device",
        "device_realized",
        "num_workers",
        "status_every",
        "iterations",
        "list_forecast_presets",
        "show_resolved_forecast_config",
        "run_dir",
        "checkpoint",
        "checkpoint_reference",
        "pretraining_checkpoint",
        "transfer_config",
        "forecast_config",
        "upstream_source",
        "upstream_config",
    }
)

RESUME_EXEMPT_FIELDS_BY_TRAINER: dict[str, frozenset[str]] = {
    "moflow": frozenset(),
    "social_vae": frozenset(),
    "trajectron": frozenset(
        {
            "no_environment_cache",
        }
    ),
    "_shared_transfer": frozenset({"allow_legacy_checkpoint"}),
}


def resume_guard_config(run_config: dict[str, Any], trainer: str) -> dict[str, Any]:
    """Return the run-configuration fields a resume must match, by exempting the rest."""
    if trainer not in RESUME_EXEMPT_FIELDS_BY_TRAINER:
        raise KeyError(
            f"Unknown trainer {trainer!r} for the resume guard; known: "
            f"{sorted(RESUME_EXEMPT_FIELDS_BY_TRAINER)}"
        )
    exempt = (
        RESUME_EXEMPT_FIELDS
        | RESUME_EXEMPT_FIELDS_BY_TRAINER[trainer]
        | RESUME_EXEMPT_FIELDS_BY_TRAINER["_shared_transfer"]
    )
    return {key: value for key, value in run_config.items() if key not in exempt}


def _as_recorded(value: Any) -> Any:
    """Canonicalise a value into the form config.json records it in, so that a comparison
    is of recorded meaning rather than of representation (Path vs str, tuple vs list)."""
    return json.loads(json.dumps(_clean(value), sort_keys=True, allow_nan=False))


@dataclass(frozen=True)
class ResumeState:
    checkpoint: Path
    payload: dict[str, Any]
    epoch: int
    iteration: int
    batch_in_epoch: int
    epoch_complete: bool
    best: float
    bad_epochs: int
    best_epoch: int | None
    unguarded_fields: tuple[str, ...] = field(default_factory=tuple)


def add_resume_argument(parser) -> None:
    parser.add_argument(
        "--resume",
        "--resuming",
        nargs="?",
        const="auto",
        metavar="CHECKPOINT",
        help=(
            "Continue this run with full model/optimizer/scheduler state. Without a path, "
            "load RUN_DIR/checkpoints/latest.pt. --resuming is a compatibility alias."
        ),
    )


def resolve_resume_checkpoint(run_dir: Path, value: str | None) -> Path | None:
    if value is None:
        return None
    path = Path(run_dir) / "checkpoints" / "latest.pt" if value == "auto" else Path(value).expanduser()
    if path.is_dir():
        path = path / "checkpoints" / "latest.pt"
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    return path


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(state: dict[str, Any] | None) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    cuda = state.get("cuda")
    if cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in cuda])


def _optimizer_to_device(optimizer, device: torch.device) -> None:
    for values in optimizer.state.values():
        for key, value in values.items():
            if torch.is_tensor(value):
                values[key] = value.to(device)


def load_resume_state(
    checkpoint: Path,
    *,
    stage: str,
    optimization: dict[str, Any],
    expected_run_config: dict[str, Any],
    model_key: str = "model",
) -> ResumeState:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {model_key, "optimizer", "epoch", "iteration", "stage", "optimization"}
    missing = sorted(required - payload.keys())
    if missing:
        raise ValueError(f"Resume checkpoint {checkpoint} is missing {missing}")
    if payload["stage"] != stage:
        raise ValueError(
            f"Resume stage mismatch: checkpoint={payload['stage']!r}, requested={stage!r}"
        )
    if payload["optimization"] != optimization:
        raise ValueError(
            "Resume optimization does not match the checkpoint. Reuse the original learning "
            "rate, batch size, epoch budget, scheduler, and early-stopping settings."
        )
    saved_config = payload.get("run_config")
    if not isinstance(saved_config, dict) or not saved_config:
        raise ValueError("Resume checkpoint has no embedded run configuration")
    mismatches = []
    unguarded = []
    for key, expected in expected_run_config.items():
        if key not in saved_config:
            # The checkpoint predates the field, so it is warned about rather than refused;
            # a field explicitly recorded as null IS guarded.
            unguarded.append(key)
            continue
        if _as_recorded(saved_config[key]) != _as_recorded(expected):
            mismatches.append(f"{key}: checkpoint={saved_config[key]!r}, requested={expected!r}")
    if mismatches:
        raise ValueError("Resume run configuration mismatch: " + "; ".join(mismatches))
    if unguarded:
        logger.warning(
            "Resume checkpoint %s predates %d guarded run-configuration field(s) and "
            "records no value for them, so they could not be checked: %s",
            checkpoint,
            len(unguarded),
            ", ".join(sorted(unguarded)),
        )
    stopping = payload.get("early_stopping") or {}
    selection = payload.get("checkpoint_selection") or {}
    best = stopping.get("best", selection.get("value"))
    return ResumeState(
        checkpoint=checkpoint,
        payload=payload,
        epoch=int(payload["epoch"]),
        iteration=int(payload["iteration"]),
        batch_in_epoch=int(payload.get("batch_in_epoch", 0)),
        epoch_complete=bool(payload.get("epoch_complete", True)),
        best=float("inf") if best is None else float(best),
        bad_epochs=int(stopping.get("bad_epochs", 0)),
        best_epoch=selection.get("best_epoch"),
        unguarded_fields=tuple(sorted(unguarded)),
    )


def restore_optimizer_scheduler_state(
    state: ResumeState, optimizer, scheduler, device: torch.device
) -> None:
    optimizer.load_state_dict(state.payload["optimizer"])
    _optimizer_to_device(optimizer, device)
    saved_scheduler = state.payload.get("scheduler")
    if (scheduler is None) != (saved_scheduler is None):
        raise ValueError("Resume scheduler presence does not match the current configuration")
    if scheduler is not None:
        scheduler.load_state_dict(saved_scheduler)


def restore_training_state(state: ResumeState, model, optimizer, scheduler, device: torch.device) -> None:
    model.load_state_dict(state.payload["model"])
    restore_optimizer_scheduler_state(state, optimizer, scheduler, device)


def checkpoint_resume_fields(
    *,
    stopper,
    batch_in_epoch: int,
    epoch_complete: bool,
    loader_generator_epoch_state: torch.Tensor | None,
    loader_generator_state: torch.Tensor | None,
) -> dict[str, Any]:
    return {
        "early_stopping": {
            "best": None if stopper.best == float("inf") else stopper.best,
            "bad_epochs": stopper.bad_epochs,
            "patience": stopper.patience,
            "min_delta": stopper.min_delta,
        },
        "batch_in_epoch": int(batch_in_epoch),
        "epoch_complete": bool(epoch_complete),
        "loader_generator_epoch_state": loader_generator_epoch_state,
        "loader_generator_state": loader_generator_state,
        "rng_state": capture_rng_state(),
    }
