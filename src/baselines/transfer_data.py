"""Dataset routing shared by transfer-capable baseline trainers."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from torch.utils.data import ConcatDataset

from baselines.dataset_specs import parse_pretrain_names, pretrain_specs
from baselines.eth_ucy import fold_datasets
from baselines.target_fraction import target_training_fraction
from config import Config, DatasetSpec
from data import SceneDataset


ROOT = Path(__file__).resolve().parents[2]
ETH_UCY_FOLDS = ("eth", "hotel", "univ", "zara1", "zara2")
TARGET_DATASETS = ("eth_ucy", "grand_central", "madras")
TARGET_PATHS = {
    "grand_central": ROOT / "datasets" / "grand_central",
    "madras": ROOT / "datasets" / "madras",
}


def parse_source_datasets(single: str | None, multiple: list[str] | None) -> tuple[str, ...]:
    return parse_pretrain_names(single, multiple)


def pretraining_datasets(source_names: tuple[str, ...], cfg: Config, pretrain_fraction: float = 1.0):
    # The subset manifest lists training tracks only, so filtering validation by it
    # would empty the split rather than scale it.
    train = [SceneDataset(spec, "train", cfg) for spec in pretrain_specs(source_names, pretrain_fraction)]
    val = [SceneDataset(spec, "val", cfg) for spec in pretrain_specs(source_names)]
    return (
        train[0] if len(train) == 1 else ConcatDataset(train),
        val[0] if len(val) == 1 else ConcatDataset(val),
    )


def resolve_target_name(target_dataset: str | None, fold: str | None) -> str:
    target = target_dataset or ("eth_ucy" if fold else "")
    if target not in TARGET_DATASETS:
        raise ValueError(f"Target dataset must be one of {TARGET_DATASETS}")
    if target == "eth_ucy" and fold not in ETH_UCY_FOLDS:
        raise ValueError(f"ETH/UCY requires --fold from {ETH_UCY_FOLDS}")
    if target != "eth_ucy" and fold is not None:
        raise ValueError("--fold is only valid with --target-dataset eth_ucy")
    return target


def downstream_datasets(
    target: str,
    fold: str | None,
    cfg: Config,
    *,
    validation_mode: str | None,
    target_fraction: float,
):
    train, val, test = downstream_full_datasets(target, fold, cfg, validation_mode=validation_mode)
    train, manifest = target_training_fraction(
        train,
        target=target,
        fold=fold,
        fraction=target_fraction,
        data_seed=cfg.seeds.data_seed,
    )
    return train, val, test, manifest


def downstream_full_datasets(
    target: str,
    fold: str | None,
    cfg: Config,
    *,
    validation_mode: str | None = "grouped",
):
    if target == "eth_ucy":
        train, val, test = fold_datasets(str(fold), cfg, validation_mode)
    else:
        path = TARGET_PATHS[target]
        if not path.exists():
            raise FileNotFoundError(f"Target dataset is unavailable: {path}")
        target_cfg = replace(cfg, split_strategy="temporal") if target == "grand_central" else cfg
        spec = DatasetSpec(target, path)
        train = SceneDataset(spec, "train", target_cfg)
        val = SceneDataset(spec, "val", target_cfg)
        test = SceneDataset(spec, "test", target_cfg)
    return train, val, test


def downstream_test_dataset(target: str, fold: str | None, cfg: Config):
    _train, _val, test = downstream_full_datasets(
        target, fold, cfg, validation_mode="grouped"
    )
    return test
