"""Aggregate matched transfer evaluations across model seeds.

The standard error reported here is across independent model-seed results, not
across the highly correlated forecasting windows within one evaluation run.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from baselines.status import atomic_json


CONDITION_FIELDS = (
    "model",
    "target_dataset",
    "target_fold",
    "target_fraction",
    "source_datasets",
    "fine_tuning_learning_rate",
    "checkpoint_metric",
    "transfer_optimization_hash",
    "forecast_preprocessing_hash",
    "data_seed",
    "training_stage",
    "k",
)
METRICS = ("topk_ade", "topk_fde")


def aggregate(run_dirs: list[Path]) -> dict:
    if not run_dirs:
        raise ValueError("At least one evaluation directory is required")
    rows = []
    for directory in run_dirs:
        path = directory / "evaluation.json"
        if not path.exists():
            raise FileNotFoundError(f"Missing evaluation result: {path}")
        result = json.loads(path.read_text(encoding="utf-8"))
        config_path = directory / "config.json"
        config = json.loads(config_path.read_text(encoding="utf-8")) if config_path.exists() else {}
        rows.append({"run_dir": str(directory), **result, "model": config.get("model")})

    reference = {field: rows[0].get(field) for field in CONDITION_FIELDS}
    for row in rows[1:]:
        different = {
            field: (reference[field], row.get(field))
            for field in CONDITION_FIELDS
            if row.get(field) != reference[field]
        }
        if different:
            raise ValueError(f"Cannot aggregate different experimental conditions: {different}")
    seeds = [
        row.get("finetuning_seed") if row.get("finetuning_seed") is not None else row.get("seed")
        for row in rows
    ]
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"Fine-tuning seeds are not unique: {seeds}")

    summary = {}
    for metric in METRICS:
        values = np.asarray([float(row[metric]) for row in rows], dtype=np.float64)
        n = len(values)
        std = float(values.std(ddof=1)) if n > 1 else 0.0
        summary[metric] = {
            "mean": float(values.mean()),
            "std": std,
            "standard_error": std / math.sqrt(n) if n > 1 else 0.0,
            "n_seeds": n,
        }
    return {
        "condition": reference,
        "seeds": seeds,
        "metrics": summary,
        "runs": rows,
        "standard_error_unit": "independent model seeds",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dirs", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = aggregate(args.run_dirs)
    atomic_json(args.output, result)
    print(json.dumps(result["metrics"], indent=2))


if __name__ == "__main__":
    main()
