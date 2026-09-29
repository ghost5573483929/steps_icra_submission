"""Evaluate a SocialVAE checkpoint on one held-out ETH/UCY fold."""
from __future__ import annotations

import argparse
import math
import traceback
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.device import training_device
from baselines.preprocessing_options import (
    add_preprocessing_arguments,
    config_preprocessing,
    handle_preprocessing_introspection,
)
from baselines.provenance import snapshot_dataset, snapshot_split_inputs
from baselines.qualitative import contact_sheet, render_scene
from baselines.registry import resolve
from baselines.social_vae import (
    REFERENCE,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    SocialVAEDataset,
    best_of_k,
    build,
    collate,
)
from baselines.status import RunStatus, atomic_json
from baselines.transfer_data import downstream_test_dataset, resolve_target_name
from config import Config, SeedConfig
from data import seed_worker
from forecasting_config import config_as_dict


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold", choices=("eth", "hotel", "univ", "zara1", "zara2"))
    parser.add_argument("--target-dataset", choices=("eth_ucy", "grand_central", "madras"))
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--visual-scenes", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int)
    parser.add_argument("--neighborhood-radius", type=float)
    add_preprocessing_arguments(parser)
    return parser


def main() -> None:
    handle_preprocessing_introspection(default_preset="social_vae")
    parser = _parser()
    args = parser.parse_args()
    try:
        args.target_dataset = resolve_target_name(args.target_dataset, args.fold)
    except ValueError as error:
        parser.error(str(error))
    if args.batch_size < 1 or args.k < 1:
        parser.error("--batch-size and --k must be positive")
    args.visual_scenes = max(0, min(30, args.visual_scenes))
    device = training_device(args.device)
    checkpoint = resolve(args.checkpoint)
    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    architecture = payload.get("architecture", {})
    hidden_dim = args.hidden_dim or int(architecture.get("hidden_dim", 256))
    neighborhood_radius = args.neighborhood_radius or float(
        architecture.get("neighborhood_radius", 2.0)
    )
    preprocessing = config_preprocessing(args, default_preset="social_vae")
    forecasting = preprocessing["forecasting"]
    if forecasting.features.representation != "position_velocity_acceleration":
        parser.error("SocialVAE requires --feature-representation position_velocity_acceleration")
    checkpoint_run_config = payload.get("run_config", {})
    checkpoint_hash = checkpoint_run_config.get("forecast_preprocessing_hash")
    if checkpoint_hash and checkpoint_hash != forecasting.forecast_preprocessing_hash:
        raise ValueError(
            "Evaluation preprocessing differs from the checkpoint. Pass the same preset/config "
            f"overrides used for training ({checkpoint_hash} != {forecasting.forecast_preprocessing_hash})."
        )
    checkpoint_data_seed = checkpoint_run_config.get("data_seed")
    if checkpoint_data_seed is not None and checkpoint_data_seed != args.data_seed:
        warnings.warn(
            f"Checkpoint data_seed {checkpoint_data_seed} does not match --data-seed {args.data_seed}"
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    cfg = Config(seeds=SeedConfig(args.data_seed, args.seed), forecasting=forecasting)
    test_source = downstream_test_dataset(args.target_dataset, args.fold, cfg)
    dataset = SocialVAEDataset(
        test_source,
        forecasting,
        training=False,
        model_seed=args.seed,
        limit=args.max_scenes,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        worker_init_fn=seed_worker,
        collate_fn=collate,
        pin_memory=device.type == "cuda",
    )
    model = build(
        device,
        prediction_steps=forecasting.window.prediction_steps,
        neighborhood_radius=neighborhood_radius,
        hidden_dim=hidden_dim,
    )
    model.load_state_dict(payload["model"])
    model.eval()

    run = RunStatus(
        args.run_dir,
        {
            **vars(args),
            "run_dir": str(args.run_dir),
            "checkpoint": str(checkpoint),
            "checkpoint_reference": args.checkpoint,
            "model": "social_vae",
            "device_realized": str(device),
            "hidden_dim": hidden_dim,
            "neighborhood_radius": neighborhood_radius,
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source": str(REFERENCE.relative_to(Path(__file__).resolve().parents[2])),
            **forecasting.flat_provenance(),
            "forecasting_config": config_as_dict(forecasting),
        },
    )
    try:
        snapshot_split_inputs(args.run_dir)
        snapshot_dataset(args.run_dir, "test", dataset)
        run.update(status="running", stage="evaluation", n_predictions=0)
        ades: list[float] = []
        fdes: list[float] = []
        visuals: list[Path] = []
        visual_metrics: list[dict] = []
        with torch.no_grad():
            label = args.fold if args.target_dataset == "eth_ucy" else args.target_dataset
            batches = run.progress(loader, desc=f"SocialVAE evaluate {label}")
            for batch in batches:
                ade, fde, predictions_world = best_of_k(model, batch, device, k=args.k)
                ades.extend(ade.tolist())
                fdes.extend(fde.tolist())
                for batch_index in range(len(ade)):
                    if len(visuals) >= args.visual_scenes:
                        break
                    xy = np.array(batch["world_xy"][batch_index], copy=True)
                    valid = np.array(batch["world_valid"][batch_index], copy=True)
                    obs = forecasting.window.observation_steps
                    xy[obs:, 0] = batch["target_world_xy"][batch_index]
                    number = len(visuals) + 1
                    stem = args.run_dir / "visuals" / f"scene_{number:02d}"
                    metrics = render_scene(
                        xy,
                        valid,
                        predictions_world[:, batch_index],
                        stem,
                        f"{label.upper()} held-out scene {number}",
                        history=obs,
                        future=forecasting.window.prediction_steps,
                    )
                    metadata = batch["metadata"][batch_index]
                    metrics.update(
                        {
                            "dataset": metadata["dataset"],
                            "recording": metadata["recording"],
                            "primary_track_id": metadata["primary_track_id"],
                            "start_step": metadata["start_step"],
                        }
                    )
                    visual_metrics.append(metrics)
                    visuals.append(stem.with_suffix(".png"))
                batches.set_postfix(n=len(ades), ade=f"{float(np.mean(ades)):.3f}")
        if not ades:
            raise RuntimeError("SocialVAE evaluation population is empty")
        contact_sheet(visuals, args.run_dir / "visuals/contact_sheet.png")
        n = len(ades)
        training_config = payload.get("run_config", {})
        selection = payload.get("checkpoint_selection", {})
        stage = training_config.get("stage")
        if stage == "finetune":
            fine_tuning_learning_rate = training_config.get("transfer_optimization", {}).get("learning_rate")
            fine_tuning_best_epoch = selection.get("best_epoch")
            fine_tuning_validation_metric = selection.get("value")
            pretraining_best_epoch = None
            pretraining_validation_metric = None
        else:
            fine_tuning_learning_rate = None
            fine_tuning_best_epoch = None
            fine_tuning_validation_metric = None
            pretraining_best_epoch = selection.get("best_epoch")
            pretraining_validation_metric = selection.get("value")
        result = {
            "k": args.k,
            "fpc": False,
            "n_predictions": n,
            "topk_ade": float(np.mean(ades)),
            "topk_fde": float(np.mean(fdes)),
            "topk_ade_standard_error": float(np.std(ades, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "topk_fde_standard_error": float(np.std(fdes, ddof=1) / math.sqrt(n)) if n > 1 else 0.0,
            "visual_scenes": len(visuals),
            "source_datasets": training_config.get("source_datasets", []),
            "target_dataset": args.target_dataset,
            "target_fold": args.fold,
            "target_fraction": training_config.get("target_fraction"),
            "seed": training_config.get("seed", args.seed),
            "pretraining_seed": training_config.get("pretraining_seed"),
            "finetuning_seed": (
                training_config.get("finetuning_seed")
                if training_config.get("finetuning_seed") is not None
                else training_config.get("seed", args.seed)
            ),
            "pretraining_checkpoint": training_config.get("pretraining_checkpoint"),
            "pretraining_best_validation_metric": training_config.get("pretraining_best_validation_metric"),
            "pretraining_best_epoch": pretraining_best_epoch,
            "pretraining_validation_metric": pretraining_validation_metric,
            "fine_tuning_learning_rate": fine_tuning_learning_rate,
            "fine_tuning_best_epoch": fine_tuning_best_epoch,
            "fine_tuning_validation_metric": fine_tuning_validation_metric,
            "checkpoint_metric": selection.get("metric"),
            "transfer_optimization_hash": training_config.get("transfer_optimization_hash"),
            "forecast_preprocessing_hash": training_config.get("forecast_preprocessing_hash"),
            "data_seed": args.data_seed,
            "checkpoint_data_seed": checkpoint_data_seed,
            "training_stage": stage,
            "checkpoint_reference": args.checkpoint,
        }
        atomic_json(args.run_dir / "visuals/metrics.json", {"scenes": visual_metrics})
        atomic_json(args.run_dir / "evaluation.json", result)
        run.metric({"stage": "test", **result})
        run.update(status="complete", stage="evaluation", **result)
    except Exception as error:
        run.log("run failed", error=repr(error), traceback=traceback.format_exc())
        run.update(status="failed", stage="evaluation", error=repr(error))
        raise


if __name__ == "__main__":
    main()
