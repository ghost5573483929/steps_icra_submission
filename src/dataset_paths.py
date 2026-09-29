from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PREPROCESSING_ROOT = ROOT / "src" / "datasets_preprocessing"
CONFIG_ROOT = ROOT / "configs"
TASK_CONFIG_PATH = CONFIG_ROOT / "task.json"
PROTOCOL_ROOT = CONFIG_ROOT / "protocols"
PAPER_ROOT = ROOT / "paper"
PAPER_STATS_ROOT = PAPER_ROOT / "stats"
LOG_ROOT = ROOT / "logs"
EXPERIMENT_ROOT = ROOT / "experiments"
MANIFEST_ROOT = EXPERIMENT_ROOT / "manifests"
RESULTS_ROOT = EXPERIMENT_ROOT / "results"


def metadata_path(dataset_dir: str | Path, filename: str) -> Path:
    """Return local generated metadata, or its committed preprocessing template."""
    dataset_dir = Path(dataset_dir)
    local = dataset_dir / filename
    if local.exists():
        return local
    template = PREPROCESSING_ROOT / dataset_dir.name / filename
    if template.exists():
        return template
    raise FileNotFoundError(f"Missing {filename} in {dataset_dir} and {template.parent}")
