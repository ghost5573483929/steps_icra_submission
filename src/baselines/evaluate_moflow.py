from __future__ import annotations

import argparse
import math
import warnings
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from baselines.device import training_device
from baselines.moflow import PrimaryTrajectoryDataset,build,sample_predictions
from baselines.status import RunStatus,atomic_json
from baselines.provenance import snapshot_dataset,snapshot_split_inputs
from baselines.registry import resolve
from baselines.qualitative import contact_sheet,render_scene
from config import Config,SeedConfig
from baselines.preprocessing_options import add_preprocessing_arguments,config_preprocessing,handle_preprocessing_introspection
from forecasting_config import config_as_dict
from baselines.transfer_data import downstream_test_dataset,resolve_target_name


def main():
    p=argparse.ArgumentParser();p.add_argument("--fold",choices=["eth","hotel","univ","zara1","zara2"])
    p.add_argument("--target-dataset",choices=["eth_ucy","grand_central","madras"])
    p.add_argument("--checkpoint",required=True);p.add_argument("--run-dir",required=True,type=Path)
    p.add_argument("--device",default="cuda");p.add_argument("--batch-size",type=int,default=256);p.add_argument("--max-scenes",type=int)
    p.add_argument("--data-seed",type=int,default=42);p.add_argument("--visual-scenes",type=int,default=10);p.add_argument("--visual-seed",type=int,default=42);add_preprocessing_arguments(p);handle_preprocessing_introspection()
    a=p.parse_args()
    try:a.target_dataset=resolve_target_name(a.target_dataset,a.fold)
    except ValueError as error:p.error(str(error))
    a.visual_scenes=max(0,min(30,a.visual_scenes));torch.manual_seed(a.visual_seed);np.random.seed(a.visual_seed);device=training_device(a.device)
    cfg=Config(seeds=SeedConfig(a.data_seed,42),**config_preprocessing(a));test=downstream_test_dataset(a.target_dataset,a.fold,cfg);dataset=PrimaryTrajectoryDataset(test,a.max_scenes)
    loader=DataLoader(dataset,batch_size=a.batch_size);_,model=build(device);checkpoint=resolve(a.checkpoint)
    payload=torch.load(checkpoint,map_location=device,weights_only=False)
    checkpoint_run_config=payload.get("run_config",{})
    checkpoint_hash=checkpoint_run_config.get("forecast_preprocessing_hash")
    if checkpoint_hash and checkpoint_hash!=cfg.forecasting.forecast_preprocessing_hash:
        raise ValueError(
            "Evaluation preprocessing differs from the checkpoint. Pass the same preset/config "
            f"overrides used for training ({checkpoint_hash} != {cfg.forecasting.forecast_preprocessing_hash})."
        )
    checkpoint_data_seed=checkpoint_run_config.get("data_seed")
    if checkpoint_data_seed is not None and checkpoint_data_seed!=a.data_seed:
        warnings.warn(
            f"Checkpoint data_seed {checkpoint_data_seed} does not match --data-seed {a.data_seed}"
        )
    model.load_state_dict(payload["model"]);model.eval()
    run=RunStatus(a.run_dir,{**vars(a),"checkpoint":str(checkpoint),"checkpoint_reference":a.checkpoint,"run_dir":str(a.run_dir),"model":"moflow_teacher","k":20,**cfg.forecasting.flat_provenance(),"forecasting_config":config_as_dict(cfg.forecasting)})
    snapshot_split_inputs(a.run_dir);snapshot_dataset(a.run_dir,"test",dataset)
    run.update(status="running",stage="evaluation");ades=[];fdes=[];visuals=[];visual_metrics=[]
    with torch.no_grad():
        for batch in loader:
            batch={k:v.to(device) for k,v in batch.items()};batch["batch_size"]=torch.tensor(batch["past_traj"].shape[0],device=device)
            pred=sample_predictions(model,batch);gt=batch["fut_traj"][:,0];distance=torch.linalg.norm(pred-gt[:,None],dim=-1)
            ade=distance.mean(-1).min(-1).values;fde=distance[:,:,-1].min(-1).values;ades.extend(ade.cpu().tolist());fdes.extend(fde.cpu().tolist())
            for i in range(len(pred)):
                if len(visuals)>=a.visual_scenes:break
                item=dataset[len(ades)-len(pred)+i];past=item["past_traj_original_scale"][0,:,:2];future=item["fut_traj_original_scale"][0]
                xy=torch.cat([past,future+past[-1]],0).numpy()[:,None];number=len(visuals)+1;stem=a.run_dir/"visuals"/f"scene_{number:02d}"
                label=a.fold if a.target_dataset=="eth_ucy" else a.target_dataset
                visual_metrics.append(render_scene(xy,np.ones(xy.shape[:2],bool),pred[i].cpu().numpy()+past[-1].numpy(),stem,f"{label.upper()} held-out scene {number}"));visuals.append(stem.with_suffix(".png"))
    contact_sheet(visuals,a.run_dir/"visuals/contact_sheet.png")
    if not ades:raise RuntimeError("MoFlow evaluation population is empty")
    training_config=payload.get("run_config",{});selection=payload.get("checkpoint_selection",{});n=len(ades)
    stage=training_config.get("stage")
    if stage=="finetune":
        fine_tuning_learning_rate=training_config.get("transfer_optimization",{}).get("learning_rate")
        fine_tuning_best_epoch=selection.get("best_epoch");fine_tuning_validation_metric=selection.get("value")
        pretraining_best_epoch=None;pretraining_validation_metric=None
    else:
        fine_tuning_learning_rate=None;fine_tuning_best_epoch=None;fine_tuning_validation_metric=None
        pretraining_best_epoch=selection.get("best_epoch");pretraining_validation_metric=selection.get("value")
    result={"k":20,"n_predictions":n,"topk_ade":float(np.mean(ades)),"topk_fde":float(np.mean(fdes)),
            "topk_ade_standard_error":float(np.std(ades,ddof=1)/math.sqrt(n)) if n>1 else 0.0,
            "topk_fde_standard_error":float(np.std(fdes,ddof=1)/math.sqrt(n)) if n>1 else 0.0,
            "visual_scenes":len(visuals),"source_datasets":training_config.get("source_datasets",[]),
            "target_dataset":a.target_dataset,"target_fold":a.fold,"target_fraction":training_config.get("target_fraction"),
            "seed":training_config.get("seed"),"pretraining_checkpoint":training_config.get("pretraining_checkpoint"),
            "pretraining_seed":training_config.get("pretraining_seed"),
            "finetuning_seed":training_config.get("finetuning_seed") if training_config.get("finetuning_seed") is not None else training_config.get("seed"),
            "pretraining_best_validation_metric":training_config.get("pretraining_best_validation_metric"),
            "pretraining_best_epoch":pretraining_best_epoch,"pretraining_validation_metric":pretraining_validation_metric,
            "fine_tuning_learning_rate":fine_tuning_learning_rate,
            "fine_tuning_best_epoch":fine_tuning_best_epoch,"fine_tuning_validation_metric":fine_tuning_validation_metric,
            "checkpoint_metric":selection.get("metric"),
            "transfer_optimization_hash":training_config.get("transfer_optimization_hash"),
            "forecast_preprocessing_hash":training_config.get("forecast_preprocessing_hash"),
            "data_seed":a.data_seed,"checkpoint_data_seed":checkpoint_data_seed,
            "training_stage":stage,"checkpoint_reference":a.checkpoint}
    atomic_json(a.run_dir/"visuals/metrics.json",{"scenes":visual_metrics})
    atomic_json(a.run_dir/"evaluation.json",result);run.metric({"stage":"test",**result});run.update(status="complete",stage="evaluation",**result)

if __name__=="__main__":main()
