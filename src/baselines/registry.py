from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from baselines.status import atomic_json,utc_now

ROOT=Path(__file__).resolve().parents[2]


def registry_dir()->Path:
    return Path(os.environ.get("TRAJECTORY_MODEL_REGISTRY",ROOT/"experiments/model_registry"))


def register(model_id:str,checkpoint:Path,**metadata)->Path:
    checkpoint=Path(checkpoint).resolve()
    if not checkpoint.is_file():raise FileNotFoundError(checkpoint)
    path=registry_dir()/f"{model_id}.json"
    if path.exists():
        current=json.loads(path.read_text())
        if Path(current["checkpoint"]).resolve()!=checkpoint:
            raise FileExistsError(f"Model ID {model_id!r} already points to {current['checkpoint']}")
    atomic_json(path,{"model_id":model_id,"checkpoint":str(checkpoint),"registered_utc":utc_now(),**metadata})
    return path


def resolve(value:str|Path)->Path:
    candidate=Path(value).expanduser()
    if candidate.is_dir():candidate=candidate/"checkpoints/best.pt"
    if candidate.is_file():return candidate.resolve()
    entry=registry_dir()/f"{value}.json"
    if not entry.exists():raise FileNotFoundError(f"No checkpoint path or registered model ID {value!r} in {registry_dir()}")
    checkpoint=Path(json.loads(entry.read_text())["checkpoint"])
    if not checkpoint.is_file():raise FileNotFoundError(f"Registered checkpoint is missing: {checkpoint}")
    return checkpoint


def main():
    parser=argparse.ArgumentParser(description="Named pretrained-checkpoint registry")
    sub=parser.add_subparsers(dest="command",required=True)
    add=sub.add_parser("add");add.add_argument("model_id");add.add_argument("checkpoint",type=Path);add.add_argument("--model");add.add_argument("--dataset");add.add_argument("--seed",type=int)
    get=sub.add_parser("resolve");get.add_argument("model_id")
    sub.add_parser("list")
    args=parser.parse_args()
    if args.command=="add":print(register(args.model_id,args.checkpoint,model=args.model,dataset=args.dataset,seed=args.seed))
    elif args.command=="resolve":print(resolve(args.model_id))
    else:
        for path in sorted(registry_dir().glob("*.json")):print(path.stem,json.loads(path.read_text())["checkpoint"])

if __name__=="__main__":main()
