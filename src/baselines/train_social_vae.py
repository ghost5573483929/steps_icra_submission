"""Pretrain, train from scratch, or fine-tune the pinned SocialVAE model."""
from __future__ import annotations

import argparse
import os
import shutil
import traceback
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler

from baselines.device import training_device
from baselines.early_stopping import EarlyStopping
from baselines.preprocessing_options import (
    add_preprocessing_arguments,
    config_preprocessing,
    handle_preprocessing_introspection,
)
from baselines.provenance import snapshot_dataset, snapshot_split_inputs
from baselines.registry import resolve
from baselines.resume import (
    add_resume_argument,
    checkpoint_resume_fields,
    load_resume_state,
    resolve_resume_checkpoint,
    restore_rng_state,
    restore_training_state,
    resume_guard_config,
)
from baselines.social_vae import (
    REFERENCE,
    UPSTREAM_COMMIT,
    UPSTREAM_REPOSITORY,
    SocialVAEDataset,
    best_of_k,
    build,
    collate,
    tensors_to_device,
)
from baselines.status import RunStatus, checkpoint_metadata
from baselines.transfer_config import (
    add_transfer_arguments,
    build_optimizer,
    build_stage_scheduler,
    format_transfer_summary,
    load_model_weights_for_transfer,
    optimization_as_dict,
    resolve_stage_seed,
    stage_optimization_from_args,
    validate_finetune_checkpoint,
)
from baselines.transfer_data import (
    downstream_datasets,
    parse_source_datasets,
    pretraining_datasets,
    resolve_target_name,
)
from config import Config, SeedConfig
from data import SceneDataset, seed_worker
from forecasting_config import config_as_dict


def _sources(args, cfg: Config):
    if args.stage == "pretrain":
        train, val = pretraining_datasets(args.source_datasets, cfg, args.pretrain_fraction)
        return train, val, None
    train, val, _test, manifest = downstream_datasets(
        args.target_dataset,
        args.fold,
        cfg,
        validation_mode=args.validation_mode,
        target_fraction=args.target_fraction,
    )
    return train, val, manifest


class FixedNumberBatchSampler(BatchSampler):
    """Faithful SocialVAE sampler: emit exactly N full batches, cycling data."""

    def __init__(self, n_batches: int, sampler, batch_size: int):
        super().__init__(sampler, batch_size, drop_last=False)
        self.n_batches = int(n_batches)

    def __len__(self) -> int:
        return self.n_batches

    def __iter__(self):
        emitted = 0
        batch = []
        sampler_iterator = iter(self.sampler)
        while emitted < self.n_batches:
            try:
                index = next(sampler_iterator)
            except StopIteration:
                sampler_iterator = iter(self.sampler)
                continue
            batch.append(index)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
                emitted += 1


def _loader(dataset, args, *, training: bool):
    if dataset is None:
        return None, None, None
    adapted = SocialVAEDataset(
        dataset,
        args.forecasting,
        training=training,
        model_seed=args.seed,
        limit=args.max_scenes,
    )
    generator = torch.Generator().manual_seed(args.seed)
    common = {
        "num_workers": args.num_workers,
        "worker_init_fn": seed_worker,
        "collate_fn": collate,
        "pin_memory": args.device_realized.type == "cuda",
    }
    if args.num_workers > 0:
        # persistent_workers stays at its default False: SocialVAEDataset.set_epoch()
        # reseeds the augmentation RNG per epoch, which would freeze at the fork-time
        # epoch if workers persisted.
        common["prefetch_factor"] = 4
    if training and args.batches_per_epoch is not None:
        sampler = RandomSampler(adapted, generator=generator)
        loader = DataLoader(
            adapted,
            batch_sampler=FixedNumberBatchSampler(
                args.batches_per_epoch, sampler, args.batch_size
            ),
            **common,
        )
    else:
        loader = DataLoader(
            adapted,
            batch_size=args.batch_size,
            shuffle=training,
            generator=generator,
            **common,
        )
    return adapted, loader, generator


def _save(
    path: Path,
    model,
    optimizer,
    scheduler,
    args,
    epoch: int,
    iteration: int,
    *,
    selection_value: float | None,
    best_epoch: int | None,
    stopper,
    batch_in_epoch: int,
    epoch_complete: bool,
    loader_generator_epoch_state: torch.Tensor | None,
    loader_generator_state: torch.Tensor | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "stage": args.stage,
            "epoch": epoch,
            "iteration": iteration,
            "optimization": optimization_as_dict(args.optimization),
            "checkpoint_selection": {
                "metric": args.optimization.checkpoint_metric,
                "value": selection_value,
                "best_epoch": best_epoch,
            },
            **checkpoint_resume_fields(
                stopper=stopper,
                batch_in_epoch=batch_in_epoch,
                epoch_complete=epoch_complete,
                loader_generator_epoch_state=loader_generator_epoch_state,
                loader_generator_state=loader_generator_state,
            ),
            "architecture": {
                "prediction_steps": args.forecasting.window.prediction_steps,
                "neighborhood_radius": args.neighborhood_radius,
                "hidden_dim": args.hidden_dim,
            },
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            **checkpoint_metadata(path.parents[1]),
        },
        temporary,
    )
    os.replace(temporary, path)


def _validate(model, loader, device, *, k: int, max_batches: int | None):
    model.eval()
    ades: list[float] = []
    fdes: list[float] = []
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            ade, fde, _prediction = best_of_k(model, batch, device, k=k)
            ades.extend(ade.tolist())
            fdes.extend(fde.tolist())
            if max_batches is not None and batch_index + 1 >= max_batches:
                break
    return float(np.mean(ades)), float(np.mean(fdes)), len(ades)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("pretrain", "scratch", "finetune"))
    parser.add_argument("--fold", choices=("eth", "hotel", "univ", "zara1", "zara2"))
    parser.add_argument("--target-dataset", choices=("eth_ucy", "grand_central", "madras"))
    parser.add_argument("--target-fraction", type=float, default=1.0)
    parser.add_argument("--checkpoint")
    parser.add_argument("--allow-legacy-checkpoint", action="store_true", help="Explicitly allow a checkpoint without the frozen transfer recipe metadata")
    parser.add_argument("--run-dir", required=True, type=Path)
    add_resume_argument(parser)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--iterations", type=int)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-scenes", type=int)
    parser.add_argument("--max-val-batches", type=int)
    parser.add_argument("--status-every", type=int, default=25)
    parser.add_argument("--seed", type=int, help="Run seed; fine-tuning without it adopts the pretrained checkpoint's seed, pretrain/scratch default to 42")
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--neighborhood-radius", type=float, default=2.0)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument(
        "--pretrain-dataset",
        choices=("steps",),
    )
    parser.add_argument("--pretrain-datasets", nargs="+", help="One or more source datasets; comma-separated values are also accepted")
    parser.add_argument("--pretrain-fraction", type=float, default=1.0, help="Fraction of the retained pretraining training tracks to use; requires a subset manifest for that dataset/fraction/data-seed")
    parser.add_argument("--validation-mode", choices=("none", "grouped"), help="ETH/UCY target validation policy (ETH/UCY only; defaults to grouped there). MADRAS and Grand Central use their own split manifests")
    add_transfer_arguments(parser)
    add_preprocessing_arguments(parser)
    return parser


def main() -> None:
    handle_preprocessing_introspection(default_preset="social_vae")
    parser = _parser()
    args = parser.parse_args()
    args.source_datasets = parse_source_datasets(args.pretrain_dataset, args.pretrain_datasets)
    if args.stage == "pretrain":
        if args.target_dataset or args.fold:
            parser.error("--target-dataset/--fold are only valid for scratch or fine-tuning")
        args.target_dataset = None
    else:
        try:
            args.target_dataset = resolve_target_name(args.target_dataset, args.fold)
        except ValueError as error:
            parser.error(str(error))
    if args.stage == "finetune" and not args.checkpoint:
        parser.error("--checkpoint is required for finetune")
    if args.stage == "scratch" and args.checkpoint:
        parser.error("scratch training must start from random initialization")
    args.optimization = stage_optimization_from_args(args, "social_vae")
    args.batch_size = args.optimization.batch_size
    args.epochs = args.optimization.max_epochs
    args.batches_per_epoch = args.optimization.batches_per_epoch
    args.learning_rate = args.optimization.learning_rate
    args.patience = args.optimization.patience if args.optimization.early_stopping else 0
    args.min_delta = args.optimization.min_delta
    if args.k < 1:
        parser.error("--k must be positive")
    if (args.batches_per_epoch is not None and args.batches_per_epoch < 1) or (args.max_val_batches is not None and args.max_val_batches < 1):
        parser.error("batch caps cannot be negative")
    if not 0.0 < args.target_fraction <= 1.0:
        parser.error("--target-fraction must be in (0, 1]")
    if not 0.0 < args.pretrain_fraction <= 1.0:
        parser.error("--pretrain-fraction must be in (0, 1]")
    if args.pretrain_fraction != 1.0 and args.stage != "pretrain":
        parser.error("--pretrain-fraction applies only to --stage pretrain")
    if args.target_dataset == "eth_ucy":
        args.validation_mode = args.validation_mode or "grouped"
    elif args.validation_mode is not None:
        parser.error(
            "--validation-mode applies only to --target-dataset eth_ucy; "
            "other targets use their own validation split"
        )
    print(format_transfer_summary(args.optimization, args.stage))

    args.device_realized = training_device(args.device)
    preprocessing = config_preprocessing(args, default_preset="social_vae")
    args.forecasting = preprocessing["forecasting"]
    if args.forecasting.features.representation != "position_velocity_acceleration":
        parser.error("SocialVAE requires --feature-representation position_velocity_acceleration")
    checkpoint = resolve(args.checkpoint) if args.checkpoint else None
    pretraining_selection = None
    checkpoint_payload = None
    if checkpoint:
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        pretraining_selection = checkpoint_payload.get("checkpoint_selection")
        if args.stage == "finetune":
            try:
                validate_finetune_checkpoint(
                    checkpoint,
                    checkpoint_payload,
                    allow_legacy=args.allow_legacy_checkpoint,
                    expected_preprocessing_hash=args.forecasting.forecast_preprocessing_hash,
                    expected_architecture={
                        "prediction_steps": args.forecasting.window.prediction_steps,
                        "neighborhood_radius": args.neighborhood_radius,
                        "hidden_dim": args.hidden_dim,
                    },
                )
            except ValueError as error:
                parser.error(str(error))
    # Every generator below must use the resolved seed, so seeding happens only after this.
    try:
        args.seed, seed_source = resolve_stage_seed(args.stage, args.seed, checkpoint_payload)
    except ValueError as error:
        parser.error(str(error))
    print(f"seed: {args.seed} ({seed_source})")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    source_provenance = (
        list(args.source_datasets)
        if args.stage == "pretrain"
        else list((checkpoint_payload or {}).get("run_config", {}).get("source_datasets", []))
    )
    checkpoint_run_config = (checkpoint_payload or {}).get("run_config", {})
    run_config = {
            **{key: value for key, value in vars(args).items() if key not in {"forecasting", "device_realized"}},
            "run_dir": str(args.run_dir),
            "checkpoint": str(checkpoint) if checkpoint else None,
            "checkpoint_reference": args.checkpoint,
            "model": "social_vae",
            "k": args.k,
            "source_datasets": source_provenance,
            "target_dataset": args.target_dataset,
            "target_fold": args.fold,
            "target_fraction": args.target_fraction if args.stage != "pretrain" else None,
            "pretraining_checkpoint": str(checkpoint) if args.stage == "finetune" else None,
            "pretraining_best_validation_metric": (pretraining_selection or {}).get("value"),
            "pretraining_checkpoint_metric": (pretraining_selection or {}).get("metric"),
            "pretraining_seed": (
                args.seed if args.stage == "pretrain" else checkpoint_run_config.get("seed")
            ),
            "finetuning_seed": args.seed if args.stage != "pretrain" else None,
            "transfer_optimization": optimization_as_dict(args.optimization),
            "transfer_optimization_hash": args.optimization.optimization_hash,
            "device_realized": str(args.device_realized),
            "upstream_repository": UPSTREAM_REPOSITORY,
            "upstream_commit": UPSTREAM_COMMIT,
            "upstream_source": str(REFERENCE.relative_to(Path(__file__).resolve().parents[2])),
            **args.forecasting.flat_provenance(),
            "forecasting_config": config_as_dict(args.forecasting),
        }
    resume_path = resolve_resume_checkpoint(args.run_dir, args.resume)
    resume_state = load_resume_state(
        resume_path,
        stage=args.stage,
        optimization=optimization_as_dict(args.optimization),
        expected_run_config=resume_guard_config(run_config, "social_vae"),
    ) if resume_path is not None else None
    run = RunStatus(
        args.run_dir,
        run_config,
        resume=resume_state is not None,
    )
    try:
        run.log("seed resolved", seed=args.seed, source=seed_source)
        run.update(status="running", stage="loading_datasets", iteration=0)
        cfg = Config(
            seeds=SeedConfig(args.data_seed, args.seed),
            forecasting=args.forecasting,
        )
        train_source, val_source, fraction_manifest = _sources(args, cfg)
        train, train_loader, train_generator = _loader(train_source, args, training=True)
        val, val_loader, _val_generator = _loader(val_source, args, training=False)
        if train is None or train_loader is None or not len(train):
            raise RuntimeError("SocialVAE training population is empty")
        if args.optimization.early_stopping and (val is None or val_loader is None or not len(val)):
            raise RuntimeError("Early stopping requires a non-empty target validation split")
        run.update(
            status="running",
            stage="saving_dataset_manifests",
            n_train=len(train),
            n_val=len(val) if val is not None else 0,
        )
        if not (args.run_dir / "datasets" / "index_hashes.json").exists():
            snapshot_split_inputs(args.run_dir)
        if not (args.run_dir / "datasets" / "train.parquet").exists():
            snapshot_dataset(args.run_dir, "train", train)
        if val is not None and not (args.run_dir / "datasets" / "val.parquet").exists():
            snapshot_dataset(args.run_dir, "val", val)
        if fraction_manifest is not None:
            destination = args.run_dir / "datasets" / "target_fraction_manifest.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(fraction_manifest, destination)

        run.update(status="running", stage="initializing_model")
        model = build(
            args.device_realized,
            prediction_steps=args.forecasting.window.prediction_steps,
            neighborhood_radius=args.neighborhood_radius,
            hidden_dim=args.hidden_dim,
        )
        if checkpoint and resume_state is None:
            payload = load_model_weights_for_transfer(model, checkpoint, args.device_realized)
            run.log(
                "pretrained weights loaded; optimizer and scheduler reset",
                checkpoint=str(checkpoint),
                pretraining_best_validation_metric=payload.get("checkpoint_selection"),
            )
        else:
            for parameter in model.parameters():
                parameter.requires_grad_(True)
        optimizer = build_optimizer(model.parameters(), args.optimization)
        total_iterations = args.epochs * (
            min(len(train_loader), args.batches_per_epoch)
            if args.batches_per_epoch is not None
            else len(train_loader)
        )
        scheduler = build_stage_scheduler(optimizer, args.optimization, max(1, total_iterations))
        if resume_state is not None:
            restore_training_state(
                resume_state, model, optimizer, scheduler, args.device_realized
            )
        run.log(
            "model initialized",
            parameters=sum(parameter.numel() for parameter in model.parameters()),
            device=str(args.device_realized),
            neighborhood_radius=args.neighborhood_radius,
            hidden_dim=args.hidden_dim,
            optimizer_reset=resume_state is None,
            scheduler_reset=resume_state is None,
            train_all_layers=all(parameter.requires_grad for parameter in model.parameters()),
        )

        stopper = EarlyStopping(args.patience, args.min_delta)
        best_epoch: int | None = None
        iteration = 0
        start_epoch = 1
        resume_batch = 0
        stopped = False
        if resume_state is not None:
            iteration = resume_state.iteration
            stopper.best = resume_state.best
            stopper.bad_epochs = resume_state.bad_epochs
            best_epoch = resume_state.best_epoch
            start_epoch = resume_state.epoch + (1 if resume_state.epoch_complete else 0)
            resume_batch = 0 if resume_state.epoch_complete else resume_state.batch_in_epoch
            if start_epoch > args.epochs:
                raise RuntimeError(
                    f"Resume checkpoint already reached epoch {resume_state.epoch} of {args.epochs}"
                )
            run.log(
                "training state resumed",
                checkpoint=str(resume_state.checkpoint),
                epoch=start_epoch,
                iteration=iteration,
                batch_in_epoch=resume_batch,
                best_validation_metric=None if stopper.best == float("inf") else stopper.best,
                unguarded_run_config_fields=list(resume_state.unguarded_fields),
            )
            if resume_state.epoch_complete:
                generator_state = resume_state.payload.get("loader_generator_state")
                if generator_state is None:
                    raise ValueError("Epoch-boundary resume checkpoint lacks loader generator state")
                train_generator.set_state(generator_state)
                restore_rng_state(resume_state.payload.get("rng_state"))
        max_train_batches = args.batches_per_epoch or None
        max_val_batches = args.max_val_batches or None
        run.update(status="running", stage="training", iteration=iteration)
        for epoch in range(start_epoch, args.epochs + 1):
            train.set_epoch(epoch)
            model.train()
            sums = {"loss": 0.0, "rec": 0.0, "kl": 0.0}
            batches_seen = 0
            progress_total = len(train_loader)
            if max_train_batches is not None:
                progress_total = min(progress_total, max_train_batches)
            if resume_state is not None and epoch == start_epoch and resume_batch:
                generator_state = resume_state.payload.get("loader_generator_epoch_state")
                if generator_state is None:
                    raise ValueError("Mid-epoch resume checkpoint lacks loader generator state")
                train_generator.set_state(generator_state)
            loader_generator_epoch_state = train_generator.get_state()
            raw_epoch_loader = iter(train_loader)
            skipped = 0
            for _ in range(resume_batch if epoch == start_epoch else 0):
                try:
                    next(raw_epoch_loader)
                    skipped += 1
                except StopIteration:
                    break
            if resume_state is not None and epoch == start_epoch and resume_batch:
                restore_rng_state(resume_state.payload.get("rng_state"))
            remaining = max(0, progress_total - skipped)
            epoch_loader = islice(raw_epoch_loader, remaining)
            batches = run.progress(
                epoch_loader,
                desc=f"SocialVAE train epoch {epoch}/{args.epochs}",
                total=remaining,
            )
            last_synced_values: dict[str, float] | None = None
            latest_saved_state: tuple[int, float | None, int | None, bool] | None = None
            batch_in_epoch = skipped
            # --iterations stops mid-epoch, so the epoch-end saves below must not claim a
            # complete epoch or a later --resume would skip the remaining batches.
            epoch_truncated = False
            for batch_index, batch in enumerate(batches):
                x, y, neighbor = tensors_to_device(batch, args.device_realized)
                optimizer.zero_grad(set_to_none=True)
                error, kl = model(x, y, neighbor)
                losses = model.loss(error, kl)
                losses["loss"].backward()
                if args.optimization.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), args.optimization.grad_clip
                    )
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                iteration += 1
                batches_seen += 1
                batch_in_epoch += 1
                # Accumulate on device (no host sync) so the epoch average stays exact.
                for key, value in losses.items():
                    sums[key] += value.detach()
                if iteration % args.status_every == 0:
                    values = {key: float(value.detach()) for key, value in losses.items()}
                    last_synced_values = values
                    run.metric({"stage": "train_batch", "epoch": epoch, "iteration": iteration, **values})
                    run.update(epoch=epoch, iteration=iteration, train_loss=values["loss"])
                    loop_selection_value = None if stopper.best == float("inf") else stopper.best
                    _save(
                        args.run_dir / "checkpoints/latest.pt", model, optimizer, scheduler, args, epoch, iteration,
                        selection_value=loop_selection_value,
                        best_epoch=best_epoch,
                        stopper=stopper,
                        batch_in_epoch=batch_in_epoch,
                        epoch_complete=False,
                        loader_generator_epoch_state=loader_generator_epoch_state,
                        loader_generator_state=train_generator.get_state(),
                    )
                    latest_saved_state = (iteration, loop_selection_value, best_epoch, False)
                if last_synced_values is not None:
                    batches.set_postfix(loss=f"{last_synced_values['loss']:.4f}", iteration=iteration)
                else:
                    batches.set_postfix(iteration=iteration)
                if args.iterations is not None and iteration >= args.iterations:
                    # tqdm advances only after a yielded item is resumed, so breaking out
                    # of the loop here would drop the final batch from the count.
                    batches.update(1)
                    epoch_truncated = True
                    break
            if not batches_seen and remaining:
                raise RuntimeError("SocialVAE training loader yielded no batches")
            epoch_values = (
                {key: float(value) / batches_seen for key, value in sums.items()}
                if batches_seen else None
            )
            if epoch_values is not None:
                run.metric(
                    {"stage": "train", "epoch": epoch, "iteration": iteration, **epoch_values}
                )

            if val_loader is not None and len(val):
                ade, fde, n_val = _validate(
                    model,
                    val_loader,
                    args.device_realized,
                    k=args.k,
                    max_batches=max_val_batches,
                )
                run.metric(
                    {
                        "stage": "val",
                        "epoch": epoch,
                        "iteration": iteration,
                        f"top{args.k}_ade": ade,
                        f"top{args.k}_fde": fde,
                        "n_predictions": n_val,
                    }
                )
                improved, stopped = stopper.update(ade)
                if improved:
                    best_epoch = epoch
                    _save(
                        args.run_dir / "checkpoints/best.pt", model, optimizer, scheduler, args, epoch, iteration,
                        selection_value=stopper.best,
                        best_epoch=best_epoch,
                        stopper=stopper,
                        batch_in_epoch=batch_in_epoch,
                        epoch_complete=not epoch_truncated,
                        loader_generator_epoch_state=loader_generator_epoch_state,
                        loader_generator_state=train_generator.get_state(),
                    )
            else:
                best_epoch = epoch
                _save(
                    args.run_dir / "checkpoints/best.pt", model, optimizer, scheduler, args, epoch, iteration,
                    selection_value=None,
                    best_epoch=best_epoch,
                    stopper=stopper,
                    batch_in_epoch=batch_in_epoch,
                    epoch_complete=not epoch_truncated,
                    loader_generator_epoch_state=loader_generator_epoch_state,
                    loader_generator_state=train_generator.get_state(),
                )
            epoch_end_selection_value = None if stopper.best == float("inf") else stopper.best
            epoch_end_state = (iteration, epoch_end_selection_value, best_epoch, not epoch_truncated)
            if latest_saved_state != epoch_end_state:
                # The write is skipped only when an in-loop save already covered this exact
                # state; selection metadata updated by validation must still be persisted.
                _save(
                    args.run_dir / "checkpoints/latest.pt", model, optimizer, scheduler, args, epoch, iteration,
                    selection_value=epoch_end_selection_value,
                    best_epoch=best_epoch,
                    stopper=stopper,
                    batch_in_epoch=batch_in_epoch,
                    epoch_complete=not epoch_truncated,
                    loader_generator_epoch_state=loader_generator_epoch_state,
                    loader_generator_state=train_generator.get_state(),
                )
            run.update(
                status="running",
                epoch=epoch,
                iteration=iteration,
                train_loss=epoch_values["loss"] if epoch_values is not None else None,
                best_val_top20_ade=stopper.best if val_loader is not None else None,
                best_validation_metric=stopper.best if val_loader is not None else None,
                best_checkpoint_epoch=best_epoch,
                checkpoint_metric=args.optimization.checkpoint_metric,
                early_stopping_bad_epochs=stopper.bad_epochs if val_loader is not None else 0,
            )
            if stopped:
                run.log(
                    "early stopping",
                    epoch=epoch,
                    iteration=iteration,
                    best_val_top20_ade=stopper.best,
                    patience=args.patience,
                )
                break
            if args.iterations is not None and iteration >= args.iterations:
                break
            resume_batch = 0
            resume_state = None
        if val_loader is not None and best_epoch is None:
            raise RuntimeError(
                f"No finite {args.optimization.checkpoint_metric} was observed; best.pt was not created"
            )
        run.update(
            status="complete",
            stage="training",
            epoch=epoch,
            iteration=iteration,
            best_val_top20_ade=stopper.best if val_loader is not None else None,
            best_validation_metric=stopper.best if val_loader is not None else None,
            best_checkpoint_epoch=best_epoch,
            checkpoint_metric=args.optimization.checkpoint_metric,
            early_stopped=stopped,
        )
    except Exception as error:
        run.log("run failed", error=repr(error), traceback=traceback.format_exc())
        run.update(status="failed", stage="training", error=repr(error))
        raise


if __name__ == "__main__":
    main()
