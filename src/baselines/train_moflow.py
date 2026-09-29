from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader

from baselines.device import training_device
from baselines.early_stopping import EarlyStopping
from baselines.moflow import PrimaryTrajectoryDataset,UPSTREAM_COMMIT,best_of_k,build
from data import seed_worker
from baselines.status import RunStatus,atomic_json,checkpoint_metadata
from baselines.provenance import snapshot_dataset,snapshot_split_inputs
from baselines.registry import resolve
from baselines.resume import add_resume_argument,checkpoint_resume_fields,load_resume_state,resolve_resume_checkpoint,restore_rng_state,restore_training_state,resume_guard_config
from baselines.preprocessing_options import add_preprocessing_arguments,config_preprocessing,handle_preprocessing_introspection
from baselines.transfer_config import add_transfer_arguments,build_optimizer,build_stage_scheduler,format_transfer_summary,load_model_weights_for_transfer,optimization_as_dict,resolve_stage_seed,stage_optimization_from_args,validate_finetune_checkpoint
from baselines.transfer_data import downstream_datasets,parse_source_datasets,pretraining_datasets,resolve_target_name
from forecasting_config import config_as_dict
from config import Config,SeedConfig

ROOT=Path(__file__).resolve().parents[2]


def datasets(stage,fold,limit,data_seed=42,model_seed=42,source_datasets=("steps",),preprocessing:dict|None=None,validation_mode:str|None="grouped",target_dataset:str|None=None,target_fraction:float=1.0,pretrain_fraction:float=1.0):
    cfg=Config(seeds=SeedConfig(data_seed,model_seed),**(preprocessing or {}))
    window=(cfg.forecasting.window.observation_steps,cfg.forecasting.window.prediction_steps)
    if stage=="pretrain":
        train,val=pretraining_datasets(tuple(source_datasets),cfg,pretrain_fraction)
        return PrimaryTrajectoryDataset(train,limit,window=window),PrimaryTrajectoryDataset(val,limit,window=window),None
    train,val,test,manifest=downstream_datasets(str(target_dataset),fold,cfg,validation_mode=validation_mode,target_fraction=target_fraction)
    return PrimaryTrajectoryDataset(train,limit,window=window),PrimaryTrajectoryDataset(val,limit,window=window) if val is not None else None,PrimaryTrajectoryDataset(test,limit,window=window),manifest


def save(path,model,opt,scheduler,iteration,stage,optimization,epoch,selection_value,best_epoch,*,stopper,batch_in_epoch,epoch_complete,loader_generator_epoch_state,loader_generator_state):
    path.parent.mkdir(parents=True,exist_ok=True);temp=path.with_suffix(".tmp")
    torch.save({"upstream_commit":UPSTREAM_COMMIT,"stage":stage,"iteration":iteration,"epoch":epoch,
                "model":model.state_dict(),"optimizer":opt.state_dict(),"scheduler":scheduler.state_dict() if scheduler is not None else None,
                "optimization":optimization_as_dict(optimization),
                "checkpoint_selection":{"metric":optimization.checkpoint_metric,"value":selection_value,"best_epoch":best_epoch},
                **checkpoint_resume_fields(stopper=stopper,batch_in_epoch=batch_in_epoch,epoch_complete=epoch_complete,loader_generator_epoch_state=loader_generator_epoch_state,loader_generator_state=loader_generator_state),
                **checkpoint_metadata(path.parents[1])},temp);os.replace(temp,path)


def main():
    p=argparse.ArgumentParser();p.add_argument("--stage",required=True,choices=["pretrain","scratch","finetune"])
    p.add_argument("--fold",choices=["eth","hotel","univ","zara1","zara2"]);p.add_argument("--checkpoint")
    p.add_argument("--allow-legacy-checkpoint",action="store_true",help="Explicitly allow a checkpoint without the frozen transfer recipe metadata")
    p.add_argument("--target-dataset",choices=["eth_ucy","grand_central","madras"]);p.add_argument("--target-fraction",type=float,default=1.0)
    p.add_argument("--run-dir",required=True,type=Path);p.add_argument("--device",default="cuda")
    add_resume_argument(p)
    p.add_argument("--iterations",type=int)
    p.add_argument("--max-scenes",type=int);p.add_argument("--max-val-batches",type=int);p.add_argument("--status-every",type=int,default=100);p.add_argument("--seed",type=int,help="Run seed; fine-tuning without it adopts the pretrained checkpoint's seed, pretrain/scratch default to 42")
    p.add_argument("--data-seed",type=int,default=42);p.add_argument("--num-workers",type=int,default=4)
    p.add_argument("--pretrain-dataset",choices=["steps"])
    p.add_argument("--pretrain-datasets",nargs="+",help="One or more source datasets; comma-separated values are also accepted")
    p.add_argument("--pretrain-fraction",type=float,default=1.0,help="Fraction of the retained pretraining training tracks to use; requires a subset manifest for that dataset/fraction/data-seed")
    p.add_argument("--validation-mode",choices=["none","grouped"],help="ETH/UCY target validation policy (ETH/UCY only; defaults to grouped there). MADRAS and Grand Central use their own split manifests")
    add_transfer_arguments(p)
    add_preprocessing_arguments(p)
    handle_preprocessing_introspection()
    a=p.parse_args()
    a.source_datasets=parse_source_datasets(a.pretrain_dataset,a.pretrain_datasets)
    if a.stage=="pretrain":
        if a.target_dataset or a.fold:p.error("--target-dataset/--fold are only valid for scratch or fine-tuning")
        a.target_dataset=None
    else:
        try:a.target_dataset=resolve_target_name(a.target_dataset,a.fold)
        except ValueError as error:p.error(str(error))
    if a.stage=="finetune" and not a.checkpoint:p.error("--checkpoint is required")
    if a.stage=="scratch" and a.checkpoint:p.error("scratch training must start from random initialization")
    if not 0.0<a.target_fraction<=1.0:p.error("--target-fraction must be in (0, 1]")
    if not 0.0<a.pretrain_fraction<=1.0:p.error("--pretrain-fraction must be in (0, 1]")
    if a.pretrain_fraction!=1.0 and a.stage!="pretrain":p.error("--pretrain-fraction applies only to --stage pretrain")
    if a.target_dataset=="eth_ucy":a.validation_mode=a.validation_mode or "grouped"
    elif a.validation_mode is not None:p.error("--validation-mode applies only to --target-dataset eth_ucy; other targets use their own validation split")
    a.optimization=stage_optimization_from_args(a,"moflow")
    a.batch_size=a.optimization.batch_size;a.epochs=a.optimization.max_epochs
    a.patience=a.optimization.patience if a.optimization.early_stopping else 0;a.min_delta=a.optimization.min_delta
    print(format_transfer_summary(a.optimization,a.stage))
    # Resolved before the checkpoint guard, which compares this representation contract
    # against the pretrained one.
    preprocessing=config_preprocessing(a);forecasting=preprocessing["forecasting"]
    checkpoint=resolve(a.checkpoint) if a.checkpoint else None
    pretraining_selection=None;checkpoint_payload=None
    if checkpoint:
        checkpoint_payload=torch.load(checkpoint,map_location="cpu",weights_only=False)
        pretraining_selection=checkpoint_payload.get("checkpoint_selection")
        if a.stage=="finetune":
            try:validate_finetune_checkpoint(checkpoint,checkpoint_payload,allow_legacy=a.allow_legacy_checkpoint,expected_preprocessing_hash=forecasting.forecast_preprocessing_hash)
            except ValueError as error:p.error(str(error))
    try:a.seed,seed_source=resolve_stage_seed(a.stage,a.seed,checkpoint_payload)
    except ValueError as error:p.error(str(error))
    print(f"seed: {a.seed} ({seed_source})")
    source_provenance=list(a.source_datasets) if a.stage=="pretrain" else list((checkpoint_payload or {}).get("run_config",{}).get("source_datasets",[]))
    checkpoint_run_config=(checkpoint_payload or {}).get("run_config",{})
    device=training_device(a.device);torch.manual_seed(a.seed);np.random.seed(a.seed)
    run_config={**vars(a),"run_dir":str(a.run_dir),"checkpoint":str(checkpoint) if checkpoint else None,"checkpoint_reference":a.checkpoint,
                               "model":"moflow_teacher","k":20,"device_realized":str(device),"upstream_commit":UPSTREAM_COMMIT,
                               "source_datasets":source_provenance,"target_dataset":a.target_dataset,
                               "target_fold":a.fold,"target_fraction":a.target_fraction if a.stage!="pretrain" else None,
                               "pretraining_checkpoint":str(checkpoint) if a.stage=="finetune" else None,
                               "pretraining_best_validation_metric":(pretraining_selection or {}).get("value"),
                               "pretraining_checkpoint_metric":(pretraining_selection or {}).get("metric"),
                               "pretraining_seed":a.seed if a.stage=="pretrain" else checkpoint_run_config.get("seed"),
                               "finetuning_seed":a.seed if a.stage!="pretrain" else None,
                               "transfer_optimization":optimization_as_dict(a.optimization),"transfer_optimization_hash":a.optimization.optimization_hash,
                               **forecasting.flat_provenance(),"forecasting_config":config_as_dict(forecasting)}
    resume_path=resolve_resume_checkpoint(a.run_dir,a.resume)
    resume_state=load_resume_state(
        resume_path,stage=a.stage,optimization=optimization_as_dict(a.optimization),
        expected_run_config=resume_guard_config(run_config,"moflow"),
    ) if resume_path is not None else None
    run=RunStatus(a.run_dir,run_config,resume=resume_state is not None)
    run.log("seed resolved",seed=a.seed,source=seed_source)
    run.update(status="running",stage="loading_datasets",iteration=resume_state.iteration if resume_state else 0)
    parts=datasets(a.stage,a.fold,a.max_scenes,a.data_seed,a.seed,a.source_datasets,preprocessing,a.validation_mode,a.target_dataset,a.target_fraction,a.pretrain_fraction);train,val=parts[:2]
    fraction_manifest=parts[3] if a.stage!="pretrain" else None
    loader_common={"num_workers":a.num_workers,"pin_memory":True,"persistent_workers":a.num_workers>0,"worker_init_fn":seed_worker}
    train_generator=torch.Generator().manual_seed(a.seed)
    train_loader=DataLoader(train,batch_size=a.batch_size,shuffle=True,generator=train_generator,**loader_common);val_loader=DataLoader(val,batch_size=a.batch_size,**loader_common) if val is not None else None
    if a.optimization.early_stopping and (val is None or val_loader is None or not len(val)):
        raise RuntimeError("Early stopping requires a non-empty target validation split")
    cfg,model=build(device,a.epochs)
    if checkpoint and resume_state is None:
        payload=load_model_weights_for_transfer(model,checkpoint,device)
        run.log("pretrained weights loaded; optimizer and scheduler reset",checkpoint=str(checkpoint),pretraining_best_validation_metric=payload.get("checkpoint_selection"))
    else:
        for parameter in model.parameters():parameter.requires_grad_(True)
    opt=build_optimizer(model.parameters(),a.optimization)
    total=max(1,a.epochs*len(train_loader));scheduler=build_stage_scheduler(opt,a.optimization,total)
    if resume_state is not None:
        restore_training_state(resume_state,model,opt,scheduler,device)
    run.log("model initialized",parameters=sum(parameter.numel() for parameter in model.parameters()),optimizer_reset=resume_state is None,scheduler_reset=resume_state is None,train_all_layers=all(parameter.requires_grad for parameter in model.parameters()))
    run.update(status="running",stage="saving_dataset_manifests",n_train=len(train),n_val=len(val) if val is not None else 0)
    if not (a.run_dir/"datasets/index_hashes.json").exists():snapshot_split_inputs(a.run_dir)
    if not (a.run_dir/"datasets/train.parquet").exists():snapshot_dataset(a.run_dir,"train",train)
    if val is not None and not (a.run_dir/"datasets/val.parquet").exists():snapshot_dataset(a.run_dir,"val",val)
    if fraction_manifest is not None:
        destination=a.run_dir/"datasets/target_fraction_manifest.json";destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(fraction_manifest,destination)
    stopper=EarlyStopping(a.patience,a.min_delta);best_epoch=None;iteration=0;start_epoch=1;resume_batch=0;stopped=False
    if resume_state is not None:
        iteration=resume_state.iteration;stopper.best=resume_state.best;stopper.bad_epochs=resume_state.bad_epochs;best_epoch=resume_state.best_epoch
        start_epoch=resume_state.epoch+(1 if resume_state.epoch_complete else 0);resume_batch=0 if resume_state.epoch_complete else resume_state.batch_in_epoch
        if start_epoch>a.epochs:raise RuntimeError(f"Resume checkpoint already reached epoch {resume_state.epoch} of {a.epochs}")
        run.log("training state resumed",checkpoint=str(resume_state.checkpoint),epoch=start_epoch,iteration=iteration,batch_in_epoch=resume_batch,best_validation_metric=None if stopper.best==float("inf") else stopper.best,unguarded_run_config_fields=list(resume_state.unguarded_fields))
        if resume_state.epoch_complete:
            generator_state=resume_state.payload.get("loader_generator_state")
            if generator_state is None:raise ValueError("Epoch-boundary resume checkpoint lacks loader generator state")
            train_generator.set_state(generator_state)
            restore_rng_state(resume_state.payload.get("rng_state"))
    run.update(status="running",stage="training",n_train=len(train),n_val=len(val) if val is not None else 0,iteration=iteration)
    for epoch in range(start_epoch,a.epochs+1):
        model.train()
        if resume_state is not None and epoch==start_epoch and resume_batch:
            generator_state=resume_state.payload.get("loader_generator_epoch_state")
            if generator_state is None:raise ValueError("Mid-epoch resume checkpoint lacks loader generator state")
            train_generator.set_state(generator_state)
        loader_generator_epoch_state=train_generator.get_state()
        epoch_iterator=iter(train_loader);batch_in_epoch=0
        for _ in range(resume_batch if epoch==start_epoch else 0):
            try:next(epoch_iterator);batch_in_epoch+=1
            except StopIteration:break
        if resume_state is not None and epoch==start_epoch and resume_batch:restore_rng_state(resume_state.payload.get("rng_state"))
        batches=run.progress(epoch_iterator,desc=f"train epoch {epoch}/{a.epochs}",total=max(0,len(train_loader)-batch_in_epoch))
        # --iterations stops mid-epoch, so the epoch-end saves below must not claim a
        # complete epoch or a later --resume would skip the remaining batches.
        epoch_truncated=False
        for batch in batches:
            # non_blocking is safe only because the loader pins these buffers.
            batch={k:v.to(device,non_blocking=True) for k,v in batch.items() if k not in ("fut_traj_original_scale","indexes")};batch["batch_size"]=torch.tensor(batch["past_traj"].shape[0],device=device)
            opt.zero_grad();loss,reg,classification,velocity=model(batch,{"cur_epoch":epoch-1});loss.backward()
            if a.optimization.grad_clip is not None:clip_grad_norm_(model.parameters(),a.optimization.grad_clip)
            opt.step()
            if scheduler is not None:scheduler.step()
            iteration+=1;batch_in_epoch+=1
            if iteration%a.status_every==0:
                values={"loss":float(loss.detach()),"reg":float(reg.detach()),"classification":float(classification.detach())}
                batches.set_postfix(loss=f"{values['loss']:.4f}",iteration=iteration)
                run.metric({"stage":"train","epoch":epoch,"iteration":iteration,**values});run.update(epoch=epoch,iteration=iteration,**values)
                save(a.run_dir/"checkpoints/latest.pt",model,opt,scheduler,iteration,a.stage,a.optimization,epoch,None if stopper.best==float("inf") else stopper.best,best_epoch,stopper=stopper,batch_in_epoch=batch_in_epoch,epoch_complete=False,loader_generator_epoch_state=loader_generator_epoch_state,loader_generator_state=train_generator.get_state())
            if a.iterations is not None and iteration>=a.iterations:epoch_truncated=True;break
        if val_loader is not None:
            model.eval();ades=[];fdes=[]
            with torch.no_grad():
                for number,batch in enumerate(val_loader):
                    batch={k:v.to(device,non_blocking=True) for k,v in batch.items() if k not in ("fut_traj_original_scale","indexes")};batch["batch_size"]=torch.tensor(batch["past_traj"].shape[0],device=device)
                    ade,fde=best_of_k(model,batch);ades.extend(ade.cpu().tolist());fdes.extend(fde.cpu().tolist())
                    if a.max_val_batches is not None and number+1>=a.max_val_batches:break
            metric=float(np.mean(ades)) if ades else float("nan")
            run.metric({"stage":"val","epoch":epoch,"iteration":iteration,"top20_ade":metric,"top20_fde":float(np.mean(fdes))})
            improved,stopped=stopper.update(metric)
            if improved:
                best_epoch=epoch;save(a.run_dir/"checkpoints/best.pt",model,opt,scheduler,iteration,a.stage,a.optimization,epoch,stopper.best,best_epoch,stopper=stopper,batch_in_epoch=batch_in_epoch,epoch_complete=not epoch_truncated,loader_generator_epoch_state=loader_generator_epoch_state,loader_generator_state=train_generator.get_state())
        else:
            stopped=False;best_epoch=epoch;save(a.run_dir/"checkpoints/best.pt",model,opt,scheduler,iteration,a.stage,a.optimization,epoch,None,best_epoch,stopper=stopper,batch_in_epoch=batch_in_epoch,epoch_complete=not epoch_truncated,loader_generator_epoch_state=loader_generator_epoch_state,loader_generator_state=train_generator.get_state())
        save(a.run_dir/"checkpoints/latest.pt",model,opt,scheduler,iteration,a.stage,a.optimization,epoch,None if stopper.best==float("inf") else stopper.best,best_epoch,stopper=stopper,batch_in_epoch=batch_in_epoch,epoch_complete=not epoch_truncated,loader_generator_epoch_state=loader_generator_epoch_state,loader_generator_state=train_generator.get_state())
        run.update(status="running",epoch=epoch,iteration=iteration,best_val_top20_ade=stopper.best if val_loader is not None else None,best_validation_metric=stopper.best if val_loader is not None else None,best_checkpoint_epoch=best_epoch,checkpoint_metric=a.optimization.checkpoint_metric,early_stopping_bad_epochs=stopper.bad_epochs if val_loader is not None else 0,early_stopping_patience=a.patience if val_loader is not None else 0)
        if stopped:
            run.log("early stopping",epoch=epoch,iteration=iteration,best_val_top20_ade=stopper.best,patience=a.patience);break
        if a.iterations is not None and iteration>=a.iterations:break
        resume_batch=0;resume_state=None
    if val_loader is not None and best_epoch is None:
        raise RuntimeError(f"No finite {a.optimization.checkpoint_metric} was observed; best.pt was not created")
    run.update(status="complete",stage="training",epoch=epoch,iteration=iteration,best_val_top20_ade=stopper.best if val_loader is not None else None,best_validation_metric=stopper.best if val_loader is not None else None,best_checkpoint_epoch=best_epoch,checkpoint_metric=a.optimization.checkpoint_metric,early_stopped=stopped)

if __name__=="__main__":main()
