from __future__ import annotations

import functools
import json
import hashlib
import logging
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from canonical import CANONICAL_HZ, CANONICAL_SCHEMA_VERSION

logger = logging.getLogger(__name__)

# Minimum wall-clock gap between curve renders for status="running" updates. Wall-clock
# rather than call-count throttling bounds how stale the curves of a hard-killed run can
# be, independently of how fast a trainer's inner loop spins.
CURVE_REFRESH_SECONDS = 120.0


def environment_info() -> dict[str, Any]:
    """Lightweight environment snapshot for run provenance."""
    info: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "torch_version": torch.__version__,
    }
    try:
        info["cuda_available"] = torch.cuda.is_available()
        if info["cuda_available"]:
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as error:
        logger.warning("Could not resolve CUDA environment info: %r", error)
    return info


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clean(value):
    """Single JSON serialisation contract for every file in a run directory.

    Path->str, dataclass->asdict, non-finite float->None, recursive over dict/list/tuple.
    """
    if isinstance(value,Path):return str(value)
    if hasattr(value,"__dataclass_fields__"):
        return _clean(asdict(value))
    if isinstance(value,float) and not math.isfinite(value):return None
    if isinstance(value,dict):return {key:_clean(item) for key,item in value.items()}
    if isinstance(value,(list,tuple)):return [_clean(item) for item in value]
    return value


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    payload=_clean(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True,allow_nan=False)+"\n")
    os.replace(temporary,path)


@functools.lru_cache(maxsize=None)
def _checkpoint_metadata_cached(run_dir:Path,mtime_ns:int)->dict[str,Any]:
    path=run_dir/"config.json"
    raw=path.read_bytes()
    return {"run_config":json.loads(raw),"run_config_sha256":"sha256:"+hashlib.sha256(raw).hexdigest()}


def checkpoint_metadata(run_dir:Path)->dict[str,Any]:
    """Embed the immutable run/preprocessing contract in every checkpoint.

    ``config.json`` is immutable for the life of a run, so its contents and SHA-256 are
    cached per run directory, keyed on mtime so that a rewritten file is re-read.
    """
    run_dir=Path(run_dir).resolve()
    path=run_dir/"config.json"
    if not path.exists():return {"run_config":{},"run_config_sha256":None}
    return _checkpoint_metadata_cached(run_dir,path.stat().st_mtime_ns)


def git_provenance(root:Path,timeout:float=5.0)->dict[str,Any]:
    """Capture this repository's commit and uncommitted-change status for run provenance.

    Any resolution failure is warned about and reported as
    ``{"git_commit": "unknown", "git_dirty": None}``; it never aborts a run.
    """
    root=Path(root)
    try:
        commit=subprocess.run(["git","rev-parse","HEAD"],cwd=root,capture_output=True,
                               text=True,timeout=timeout,check=True).stdout.strip()
        porcelain=subprocess.run(["git","status","--porcelain"],cwd=root,capture_output=True,
                                  text=True,timeout=timeout,check=True).stdout
        return {"git_commit":commit,"git_dirty":bool(porcelain.strip())}
    except Exception as error:
        logger.warning("Could not resolve git provenance for %s: %r",root,error)
        return {"git_commit":"unknown","git_dirty":None}


class RunStatus:
    def __init__(self,run_dir:Path,config:dict[str,Any],*,resume:bool=False):
        self.run_dir=Path(run_dir)
        self._last_curve_time=None
        existing=[name for name in ("config.json","status.json") if (self.run_dir/name).exists()]
        if existing and not resume:
            raise RuntimeError(
                f"Refusing to overwrite existing run at {self.run_dir} (found {', '.join(existing)} "
                "from a prior run); pass resume=True to continue that run or use a new run directory."
            )
        self.run_dir.mkdir(parents=True,exist_ok=True)
        if resume and existing:
            if not (self.run_dir/"config.json").is_file():
                raise RuntimeError(f"Cannot resume {self.run_dir}: config.json is missing")
            self.log("run opened for resume")
            return
        root=Path(__file__).resolve().parents[2]
        config={**config,**{key:config.get(key,value) for key,value in git_provenance(root).items()}}
        config={**config,**{key:config.get(key,value) for key,value in environment_info().items()}}
        task_path=root/"configs/task.json"
        if task_path.exists():
            task=json.loads(task_path.read_text())
            forecasting=task.get("forecasting",{});analysis=task.get("analysis",{})
            config={**config,"canonical_release":config.get("canonical_release","v3_2p5hz"),
                    "canonical_schema_version":CANONICAL_SCHEMA_VERSION,"canonical_hz":CANONICAL_HZ,
                    "analysis_preprocessing":config.get("analysis_preprocessing",analysis.get("preprocessing","opentraj_kalman_smooth")),
                    "forecast_input_preprocessing":config.get("forecast_input_preprocessing",forecasting.get("input_preprocessing","raw")),
                    "forecast_target_preprocessing":config.get("forecast_target_preprocessing",forecasting.get("target_preprocessing","raw")),
                    # Deprecated compatibility field: follows the resolved coordinate frame.
                    "model_normalization":config.get(
                        "model_normalization",
                        config.get(
                            "coordinate_frame",
                            forecasting.get("normalization","target_centric_se2_observations_only"),
                        ),
                    )}
            config["forecast_input_uses_future_information"]=config["forecast_input_preprocessing"]=="opentraj_offline_smooth_compat"
        atomic_json(self.run_dir/"config.json",config)
        self.log("run created",config=config)
        self.update(status="created",stage="setup",iteration=0)

    def log(self,message:str,**fields:Any)->None:
        record={"created_utc":utc_now(),"message":message,**fields}
        with (self.run_dir/"training.log").open("a") as stream:
            stream.write(json.dumps(_clean(record),sort_keys=True,default=str,allow_nan=False)+"\n")
        summary=" ".join(f"{key}={value}" for key,value in fields.items()
                         if key in {"status","stage","epoch","iteration","n_train","n_val","n_train_items","n_val_items","train_loss","loss","best_val_loss","best_val_nll","best_val_top20_ade"})
        tqdm.write(f"[{record['created_utc']}] {message}"+(f" | {summary}" if summary else ""))

    def update(self,**fields:Any)->None:
        path=self.run_dir/"status.json"
        current=json.loads(path.read_text()) if path.exists() else {}
        current.update(fields);current["last_update_utc"]=utc_now()
        atomic_json(path,current)
        self.log("status",**fields)
        status=fields.get("status")
        if status in {"complete","failed","running"}:
            now=time.monotonic()
            # The clock advances only when figures were actually written, so "running" updates
            # with nothing to plot yet cannot consume the refresh window.
            throttled=(status=="running" and self._last_curve_time is not None
                       and now-self._last_curve_time<CURVE_REFRESH_SECONDS)
            if not throttled:
                try:
                    if plot_curves(self.run_dir):self._last_curve_time=now
                except Exception as error:self.log("curve generation failed",error=repr(error),traceback=traceback.format_exc())

    def metric(self,payload:dict[str,Any])->None:
        payload={**payload,"created_utc":utc_now()}
        with (self.run_dir/"metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(payload,sort_keys=True)+"\n")
        self.log("metric",**payload)

    def progress(self,iterable,*,desc:str,total:int|None=None,unit:str="batch",leave:bool=False):
        """Visible terminal progress with lifecycle events persisted to training.log."""
        if total is None:
            try:total=len(iterable)
            except TypeError:pass
        self.log("progress started",stage=desc,total=total,unit=unit)
        return tqdm(iterable,desc=desc,total=total,unit=unit,leave=leave,dynamic_ncols=True)


def plot_curves(run_dir:Path)->bool:
    """Render every numeric train/validation series after a run.

    Returns True when figures were written, False when there was nothing to plot; callers
    use this to avoid starting a refresh throttle on a render that never happened.
    """
    path=Path(run_dir)/"metrics.jsonl"
    if not path.exists():return False
    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    series={}
    excluded={"epoch","iteration","created_utc","k","n_predictions","n_recordings"}
    for record in records:
        x=record.get("iteration",record.get("epoch"))
        if x is None:continue
        stage=record.get("stage","metric")
        for key,value in record.items():
            if key in excluded or not isinstance(value,(int,float)) or not math.isfinite(value):continue
            series.setdefault(f"{stage}/{key}",[]).append((x,value))
    if not series:return False
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    count=len(series);fig,axes=plt.subplots(count,1,figsize=(7,max(3,2.6*count)),squeeze=False)
    for axis,(name,points) in zip(axes[:,0],sorted(series.items())):
        points=sorted(points);axis.plot([p[0] for p in points],[p[1] for p in points],marker="o",markersize=2)
        axis.set_title(name);axis.set_xlabel("iteration");axis.grid(alpha=.25)
    fig.tight_layout();output=Path(run_dir)/"curves";output.mkdir(exist_ok=True)
    fig.savefig(output/"training_curves.png",dpi=180);fig.savefig(output/"training_curves.pdf");plt.close(fig)
    return True
