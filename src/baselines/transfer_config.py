"""Stage-aware optimization recipes for pretraining and target training.

The model implementation owns its native epoch semantics.  This module only
resolves configuration and records the transfer contract; it never tries to
equalize update counts across different architectures.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import warnings
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class StageOptimization:
    optimizer: str
    learning_rate: float
    lr_multiplier: float
    weight_decay: float
    batch_size: int
    max_epochs: int
    batches_per_epoch: int | None
    scheduler: str
    warmup_fraction: float
    min_lr: float
    grad_clip: float | None
    early_stopping: bool
    patience: int
    min_delta: float
    checkpoint_metric: str
    reset_optimizer: bool
    reset_scheduler: bool
    train_all_layers: bool

    @property
    def optimization_hash(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "sha256:" + hashlib.sha256(payload.encode()).hexdigest()


SOCIAL_VAE_PRETRAIN = StageOptimization(
    optimizer="Adam",
    learning_rate=5e-4,
    lr_multiplier=1.0,
    weight_decay=0.0,
    batch_size=128,
    max_epochs=600,
    batches_per_epoch=100,
    scheduler="none",
    warmup_fraction=0.0,
    min_lr=0.0,
    grad_clip=None,
    early_stopping=False,
    patience=0,
    min_delta=1e-4,
    checkpoint_metric="val_ADE",
    reset_optimizer=True,
    reset_scheduler=True,
    train_all_layers=True,
)

MOFLOW_PRETRAIN = StageOptimization(
    optimizer="AdamW",
    learning_rate=1e-4,
    lr_multiplier=1.0,
    weight_decay=0.01,
    batch_size=48,
    max_epochs=150,
    batches_per_epoch=None,
    scheduler="cosine_warmup",
    warmup_fraction=0.05,
    min_lr=1e-6,
    grad_clip=1.0,
    early_stopping=False,
    patience=0,
    min_delta=1e-4,
    checkpoint_metric="val_minADE",
    reset_optimizer=True,
    reset_scheduler=True,
    train_all_layers=True,
)


_PRETRAIN = {"social_vae": SOCIAL_VAE_PRETRAIN, "moflow": MOFLOW_PRETRAIN}
_FINETUNE = {
    "social_vae": replace(
        SOCIAL_VAE_PRETRAIN,
        learning_rate=1e-4,
        lr_multiplier=0.2,
        max_epochs=200,
        early_stopping=True,
        patience=20,
        checkpoint_metric="target_val_ADE",
    ),
    "moflow": replace(
        MOFLOW_PRETRAIN,
        learning_rate=2e-5,
        lr_multiplier=0.2,
        max_epochs=50,
        early_stopping=True,
        patience=10,
        checkpoint_metric="target_val_minADE",
    ),
}


def _load(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:  # pragma: no cover - pinned dependency
            raise RuntimeError("PyYAML is required for YAML transfer configuration") from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError("Transfer config must be an object")
    transfer = payload.get("transfer", payload)
    if not isinstance(transfer, dict):
        raise ValueError("transfer must be an object")
    return transfer


_ALIASES = {
    "lr": "learning_rate",
    "epochs": "max_epochs",
}


def _normalize(values: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in values.items():
        key = _ALIASES.get(key, key)
        if key not in StageOptimization.__dataclass_fields__:
            raise ValueError(f"Unknown transfer option {key!r}")
        normalized[key] = value
    return normalized


def resolve_stage_optimization(
    model: str,
    stage: str,
    *,
    config_path: Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> StageOptimization:
    """Resolve defaults < transfer file < explicit CLI overrides.

    ``scratch`` intentionally uses the fine-tuning recipe so scratch and
    pretrained initialization receive identical target optimization budgets.
    """
    if model not in _PRETRAIN:
        raise ValueError(f"No transfer recipe for model {model!r}")
    if stage not in {"pretrain", "finetune", "scratch"}:
        raise ValueError(f"Unknown training stage {stage!r}")
    role = "pretrain" if stage == "pretrain" else "finetune"
    base = _PRETRAIN[model] if role == "pretrain" else _FINETUNE[model]
    file_values = _load(config_path)
    section = file_values.get(role, {})
    if not isinstance(section, dict):
        raise ValueError(f"transfer.{role} must be an object")
    changes = _normalize(section)

    # A multiplier without an explicit learning rate is relative to the resolved
    # source-stage learning rate of the same model.
    explicit_lr = "learning_rate" in changes
    if role == "finetune" and "lr_multiplier" in changes and not explicit_lr:
        source_section = file_values.get("pretrain", {})
        source_changes = _normalize(source_section) if isinstance(source_section, dict) else {}
        source_lr = float(source_changes.get("learning_rate", _PRETRAIN[model].learning_rate))
        changes["learning_rate"] = source_lr * float(changes["lr_multiplier"])
    resolved = replace(base, **changes)

    cli = {key: value for key, value in (cli_overrides or {}).items() if value is not None}
    cli = _normalize(cli)
    if role == "finetune" and "lr_multiplier" in cli and "learning_rate" not in cli:
        source_section = file_values.get("pretrain", {})
        source_changes = _normalize(source_section) if isinstance(source_section, dict) else {}
        source_lr = float(source_changes.get("learning_rate", _PRETRAIN[model].learning_rate))
        cli["learning_rate"] = source_lr * float(cli["lr_multiplier"])
    resolved = replace(resolved, **cli)
    _validate(resolved, stage)
    return resolved


def _validate(config: StageOptimization, stage: str) -> None:
    if config.optimizer not in {"Adam", "AdamW"}:
        raise ValueError("optimizer must be Adam or AdamW")
    if config.scheduler not in {"none", "cosine_warmup"}:
        raise ValueError("scheduler must be none or cosine_warmup")
    if config.learning_rate <= 0 or config.batch_size < 1 or config.max_epochs < 1:
        raise ValueError("learning rate, batch size, and epochs must be positive")
    if not 0.0 <= config.warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if config.min_lr < 0 or config.min_lr > config.learning_rate:
        raise ValueError("min_lr must be between zero and learning_rate")
    if config.patience < 0 or config.min_delta < 0:
        raise ValueError("patience and min_delta cannot be negative")
    if config.early_stopping and config.patience < 1:
        raise ValueError("early stopping requires positive patience")
    if stage == "finetune":
        if not config.reset_optimizer or not config.reset_scheduler:
            raise ValueError("fine-tuning must reset optimizer and scheduler state")
        if not config.train_all_layers:
            raise ValueError("fine-tuning must train all model parameters")


def add_transfer_arguments(parser: argparse.ArgumentParser) -> None:
    """Add optional overrides; ``None`` preserves recipe/config precedence."""
    parser.add_argument("--transfer-config", type=Path, help="JSON/YAML with transfer.pretrain/finetune")
    parser.add_argument("--optimizer", choices=("Adam", "AdamW"))
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--lr-multiplier", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batches-per-epoch", type=int)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--scheduler", choices=("none", "cosine_warmup"))
    parser.add_argument("--warmup-fraction", type=float)
    parser.add_argument("--min-lr", type=float)
    parser.add_argument("--grad-clip", type=float)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--min-delta", type=float)
    stopping = parser.add_mutually_exclusive_group()
    stopping.add_argument("--early-stopping", dest="early_stopping", action="store_true")
    stopping.add_argument("--no-early-stopping", dest="early_stopping", action="store_false")
    parser.set_defaults(early_stopping=None)


def stage_optimization_from_args(args, model: str) -> StageOptimization:
    return resolve_stage_optimization(
        model,
        args.stage,
        config_path=getattr(args, "transfer_config", None),
        cli_overrides={
            "optimizer": getattr(args, "optimizer", None),
            "learning_rate": getattr(args, "learning_rate", None),
            "lr_multiplier": getattr(args, "lr_multiplier", None),
            "batch_size": getattr(args, "batch_size", None),
            "max_epochs": getattr(args, "epochs", None),
            "batches_per_epoch": getattr(args, "batches_per_epoch", None),
            "weight_decay": getattr(args, "weight_decay", None),
            "scheduler": getattr(args, "scheduler", None),
            "warmup_fraction": getattr(args, "warmup_fraction", None),
            "min_lr": getattr(args, "min_lr", None),
            "grad_clip": getattr(args, "grad_clip", None),
            "early_stopping": getattr(args, "early_stopping", None),
            "patience": getattr(args, "patience", None),
            "min_delta": getattr(args, "min_delta", None),
        },
    )


def optimization_as_dict(config: StageOptimization) -> dict[str, Any]:
    return asdict(config)


def build_optimizer(parameters, config: StageOptimization):
    """Build a fresh optimizer from the fully resolved stage recipe."""
    import torch

    if config.optimizer == "Adam":
        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    if config.optimizer == "AdamW":
        return torch.optim.AdamW(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer {config.optimizer!r}")


def build_stage_scheduler(optimizer, config: StageOptimization, total_iterations: int):
    """Build a fresh scheduler from the fully resolved stage recipe.

    ``cosine_warmup`` follows MoFlow's native 5%-warmup/cosine structure.
    SocialVAE's literature recipe resolves to ``none``.
    """
    if config.scheduler == "none":
        return None
    import torch

    warmup_iterations = max(1, int(total_iterations * config.warmup_fraction))
    warmup = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: max(config.min_lr / config.learning_rate, step / warmup_iterations),
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_iterations - warmup_iterations),
        eta_min=config.min_lr,
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, [warmup, cosine], milestones=[warmup_iterations]
    )


def format_transfer_summary(config: StageOptimization, stage: str) -> str:
    stopping = f"patience {config.patience}" if config.early_stopping else "disabled"
    batches = "full loader" if config.batches_per_epoch is None else str(config.batches_per_epoch)
    return (
        "Transfer optimization\n"
        f"  stage:              {stage}\n"
        f"  optimizer:          {config.optimizer}\n"
        f"  learning rate:      {config.learning_rate:g} (multiplier {config.lr_multiplier:g})\n"
        f"  epochs:             {config.max_epochs}\n"
        f"  batches per epoch:  {batches}\n"
        f"  batch size:         {config.batch_size}\n"
        f"  scheduler:          {config.scheduler}\n"
        f"  early stopping:     {stopping}\n"
        f"  checkpoint metric:  {config.checkpoint_metric}"
    )


def load_model_weights_for_transfer(model, checkpoint: Path, device):
    """Load model parameters only; optimizer/scheduler state is intentionally ignored."""
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(payload["model"])
    for parameter in model.parameters():
        parameter.requires_grad_(True)
    return payload


def resolve_stage_seed(
    stage: str,
    seed: int | None,
    checkpoint_payload: dict[str, Any] | None = None,
    *,
    default: int = 42,
) -> tuple[int, str]:
    """Resolve the run seed, defaulting fine-tuning to the pretraining seed.

    An omitted ``--seed`` for fine-tuning adopts the seed recorded in the checkpoint's
    ``run_config``, keeping the run on the same seed lineage as its initialization.
    Returns the seed and a short provenance description for logging.
    """
    if seed is not None:
        return int(seed), "explicit --seed"
    if stage == "finetune":
        run_config = (checkpoint_payload or {}).get("run_config") or {}
        pretraining_seed = run_config.get("seed")
        if pretraining_seed is None:
            raise ValueError(
                "fine-tuning without --seed requires the pretrained checkpoint to record "
                "run_config.seed; pass --seed explicitly for this checkpoint"
            )
        return int(pretraining_seed), "pretrained checkpoint run_config.seed"
    return int(default), f"stage default ({default})"


def _values_match(expected: Any, actual: Any) -> bool:
    numbers = (int, float)
    if (
        isinstance(expected, numbers)
        and isinstance(actual, numbers)
        and not isinstance(expected, bool)
        and not isinstance(actual, bool)
    ):
        return math.isclose(float(expected), float(actual), rel_tol=1e-9, abs_tol=0.0)
    return expected == actual


def validate_finetune_checkpoint(
    checkpoint: Path,
    payload: dict[str, Any],
    *,
    allow_legacy: bool = False,
    expected_preprocessing_hash: str | None = None,
    expected_architecture: dict[str, Any] | None = None,
) -> None:
    """Require a selected source-stage checkpoint for normal fine-tuning.

    The checkpoint's representation contract and, where the payload records one, its
    architecture must match this run's resolved configuration: fine-tuning under a
    different input representation or model geometry silently changes what the
    pretrained weights mean.  Only architecture keys the payload records are compared.
    """
    if allow_legacy:
        warnings.warn(
            f"--allow-legacy-checkpoint: skipping the fine-tuning source checks for {checkpoint}; "
            "the pretrained recipe, preprocessing hash and architecture are NOT verified",
            UserWarning,
            stacklevel=2,
        )
        return
    if checkpoint.name != "best.pt":
        raise ValueError("fine-tuning requires the pretrained checkpoints/best.pt")
    if payload.get("stage") != "pretrain":
        raise ValueError(
            f"fine-tuning source must be a pretraining checkpoint, got stage={payload.get('stage')!r}"
        )
    if not payload.get("run_config", {}).get("transfer_optimization"):
        raise ValueError(
            "pretrained checkpoint predates the configurable transfer protocol; "
            "pass --allow-legacy-checkpoint only for an explicit compatibility run"
        )
    selection = payload.get("checkpoint_selection", {})
    if selection.get("value") is None:
        raise ValueError("pretrained best checkpoint has no source validation metric")
    if expected_preprocessing_hash is not None:
        checkpoint_hash = payload.get("run_config", {}).get("forecast_preprocessing_hash")
        if checkpoint_hash != expected_preprocessing_hash:
            raise ValueError(
                "fine-tuning preprocessing does not match the pretrained checkpoint: "
                f"checkpoint forecast_preprocessing_hash={checkpoint_hash!r}, "
                f"requested={expected_preprocessing_hash!r}; reuse the pretraining forecast "
                "configuration or pass --allow-legacy-checkpoint for an explicit compatibility run"
            )
    checkpoint_architecture = payload.get("architecture") or {}
    if expected_architecture and checkpoint_architecture:
        mismatches = [
            f"{key}: checkpoint={checkpoint_architecture[key]!r}, requested={value!r}"
            for key, value in expected_architecture.items()
            if key in checkpoint_architecture
            and not _values_match(value, checkpoint_architecture[key])
        ]
        if mismatches:
            raise ValueError(
                "fine-tuning architecture does not match the pretrained checkpoint: "
                + "; ".join(mismatches)
                + "; reuse the pretraining architecture or pass --allow-legacy-checkpoint "
                "for an explicit compatibility run"
            )
