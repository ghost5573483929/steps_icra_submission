from __future__ import annotations

import contextlib
import logging
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

ROOT=Path(__file__).resolve().parents[2]
REFERENCE=ROOT/"references/moflow"
if str(REFERENCE) not in sys.path:sys.path.insert(0,str(REFERENCE))


@contextlib.contextmanager
def _scoped_reference_namespace(reference:Path,package_names:tuple[str,...]=("models","utils")):
    """Register moflow's synthetic ``models``/``utils`` packages for one import only.

    Other modules in this repository occupy the same bare names, so a permanent
    ``sys.modules`` overwrite would shadow whichever of them is imported later.
    """
    saved={name:sys.modules.get(name) for name in package_names}
    for name in package_names:
        package=types.ModuleType(name);package.__path__=[str(reference/name)];sys.modules[name]=package
    try:
        yield
    finally:
        for key in [k for k in sys.modules if any(k==name or k.startswith(name+".") for name in package_names)]:
            del sys.modules[key]
        for name,module in saved.items():
            if module is not None:sys.modules[name]=module


with _scoped_reference_namespace(REFERENCE):
    from models.backbone_eth_ucy import ETHMotionTransformer  # noqa:E402
    from models.flow_matching import FlowMatcher  # noqa:E402
    from utils.config import Config as MoFlowConfig  # noqa:E402

UPSTREAM_COMMIT="f1b89b0e80ce95646214923e271e3c280d305cfa"


MOFLOW_WINDOW=(8,12)


def _source_window(source)->tuple[int,int]|None:
    """Find the forecasting window of the scene dataset(s) behind ``source``.

    ``None`` means the source carries no resolvable window, i.e. a synthetic in-memory
    stand-in rather than a configured dataset.
    """
    observation=getattr(source,"history_steps",None);prediction=getattr(source,"future_steps",None)
    if observation is not None and prediction is not None:return int(observation),int(prediction)
    window=getattr(getattr(getattr(source,"cfg",None),"forecasting",None),"window",None)
    if window is not None:return int(window.observation_steps),int(window.prediction_steps)
    for attribute in ("datasets","dataset"):
        nested=getattr(source,attribute,None)
        if nested is None:continue
        found={_source_window(item) for item in (nested if isinstance(nested,(list,tuple)) else [nested])}
        found.discard(None)
        if len(found)>1:raise ValueError(f"MoFlow source combines different forecasting windows: {sorted(found)}")
        if found:return found.pop()
    return None


class PrimaryTrajectoryDataset(Dataset):
    """Expose unified primary-agent samples in MoFlow's official tensor contract.

    The pinned upstream contract is the ETH/UCY 8-observed/12-predicted window, so the
    resolved window is checked here: a different split of the same total length (e.g.
    10/10) would silently turn observed steps into prediction targets.
    """
    def __init__(self,source,limit:int|None=None,*,window:tuple[int,int]|None=None):
        resolved=tuple(window) if window is not None else _source_window(source)
        if resolved is not None and resolved!=MOFLOW_WINDOW:
            raise ValueError(
                f"MoFlow's pinned tensor contract is {MOFLOW_WINDOW[0]} observed / {MOFLOW_WINDOW[1]} "
                f"predicted steps, but the resolved forecasting window is {resolved[0]}/{resolved[1]}; "
                "use a forecast configuration with that window or adapt the MoFlow adapter"
            )
        self.window=resolved
        self.source=source;self.limit=min(len(source),limit) if limit is not None else len(source)
    def __len__(self):return self.limit
    def __getitem__(self,index):
        scene=self.source[index];xy=torch.as_tensor(scene.xy[:,0],dtype=torch.float32)
        past_abs=xy[:8];origin=past_abs[-1:];past_rel=past_abs-origin;future=xy[8:]-origin
        past_vel=torch.cat([past_rel[1:]-past_rel[:-1],torch.zeros_like(past_rel[-1:])])
        future_vel=torch.cat([future[1:]-future[:-1],torch.zeros_like(future[-1:])])
        past=torch.cat([past_abs,past_rel,past_vel],dim=-1).unsqueeze(0)
        return {"indexes":torch.tensor(index),"past_traj":past,"fut_traj":future.unsqueeze(0),
                "past_traj_original_scale":past,"fut_traj_original_scale":future.unsqueeze(0),
                "fut_traj_vel":future_vel.unsqueeze(0)}


def configuration(device:torch.device,epochs:int=150):
    cfg=MoFlowConfig(str(REFERENCE/"cfg/eth_ucy/cor_fm.yml"),"adapter",train_mode=False)
    cfg.device=device;cfg.agents=1;cfg.MODEL.CONTEXT_ENCODER.AGENTS=1
    cfg.data_norm="original";cfg.rotate=False;cfg.rotate_aug=False
    cfg.tied_noise=True;cfg.fm_in_scaling=True;cfg.fm_wrapper="direct";cfg.fm_rew_sqrt=False
    cfg.logit_norm_mean=-0.5;cfg.logit_norm_std=1.5
    cfg.drop_method="emb";cfg.drop_logi_k=20.0;cfg.drop_logi_m=0.5
    cfg.LOSS_NN_MODE="agent";cfg.LOSS_REG_REDUCTION="sum"
    cfg.LOSS_REG_SQUARED=False;cfg.LOSS_VELOCITY=False
    cfg.OPTIMIZATION.NUM_EPOCHS=epochs
    return cfg


def build(device:torch.device,epochs:int=150):
    cfg=configuration(device,epochs);logger=logging.getLogger("moflow-adapter")
    backbone=ETHMotionTransformer(cfg.MODEL,logger,cfg)
    return cfg,FlowMatcher(cfg,backbone,logger).to(device)


def sample_predictions(model:FlowMatcher,batch:dict,k:int=20):
    prediction,*_=model.sample(batch,num_trajs=k)
    return prediction.reshape(prediction.shape[0],k,1,12,2)[:,:,0]


def best_of_k(model:FlowMatcher,batch:dict,k:int=20):
    pred=sample_predictions(model,batch,k)
    gt=batch["fut_traj"][:,0]
    distance=torch.linalg.norm(pred-gt[:,None],dim=-1)
    return distance.mean(-1).min(-1).values,distance[:,:,-1].min(-1).values
