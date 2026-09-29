#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=""
SOURCES=""
SEED=""
DATA_SEED=42
DEVICE=cuda
OUTPUT=experiments/pretrained
CHECKPOINT=""
TRANSFER_CONFIG=""
FORECAST_CONFIG=""
LEARNING_RATE=""
EPOCHS=""
BATCH_SIZE=""
NUM_WORKERS=""
PRETRAIN_FRACTION=""
RUN_VERSION="transfer_v1"
RESUME=""
PYTHON_BIN="${PYTHON_BIN:-python}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --sources) SOURCES="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --data-seed) DATA_SEED="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --transfer-config) TRANSFER_CONFIG="$2"; shift 2;;
    --forecast-config) FORECAST_CONFIG="$2"; shift 2;;
    --learning-rate) LEARNING_RATE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --pretrain-fraction) PRETRAIN_FRACTION="$2"; shift 2;;
    --run-version) RUN_VERSION="$2"; shift 2;;
    --resume|--resuming)
      if [[ $# -gt 1 && "$2" != --* ]]; then RESUME="$2"; shift 2; else RESUME="auto"; shift; fi;;
    --python) PYTHON_BIN="$2"; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

[[ "$MODEL" == "moflow" || "$MODEL" == "socialvae" || "$MODEL" == "social_vae" ]] || { echo "--model must be moflow or socialvae" >&2; exit 2; }
[[ -n "$SOURCES" ]] || { echo "--sources is required, for example steps" >&2; exit 2; }
[[ -n "$SEED" ]] || { echo "--seed is required" >&2; exit 2; }
MODEL_KEY="${MODEL/socialvae/social_vae}"
IFS=',' read -r -a SOURCE_ARRAY <<< "$SOURCES"
SOURCE_TAG="${SOURCES//,/_}"
[[ "$RUN_VERSION" =~ ^[a-zA-Z0-9_.-]+$ ]] || { echo "--run-version must contain only letters, digits, dots, underscores, or hyphens" >&2; exit 2; }
MODEL_ID="${MODEL_KEY/social_vae/socialvae}_${RUN_VERSION}_${SOURCE_TAG}_${SEED}"
RUN_DIR="$OUTPUT/$MODEL_ID"

# Existing runs are immutable unless continuation was requested explicitly.
if [[ -z "$RESUME" && ( -e "$RUN_DIR/status.json" || -d "$RUN_DIR/checkpoints" ) ]]; then
  echo "Refusing to overwrite existing pretraining run: $RUN_DIR" >&2
  echo "Pass --resume/--resuming to continue it, or choose another --run-version." >&2
  exit 3
fi

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1

printf 'MODEL=%q\nSOURCES=%q\nSEED=%q\nDATA_SEED=%q\nDEVICE=%q\nCHECKPOINT=%q\nTRANSFER_CONFIG=%q\nFORECAST_CONFIG=%q\nLEARNING_RATE=%q\nEPOCHS=%q\nBATCH_SIZE=%q\nNUM_WORKERS=%q\nPRETRAIN_FRACTION=%q\nRUN_VERSION=%q\nRESUME=%q\n' \
  "$MODEL_KEY" "$SOURCES" "$SEED" "$DATA_SEED" "$DEVICE" "$CHECKPOINT" "$TRANSFER_CONFIG" "$FORECAST_CONFIG" "$LEARNING_RATE" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$PRETRAIN_FRACTION" "$RUN_VERSION" "$RESUME" > "$RUN_DIR/params.env"

arguments=(--stage pretrain --pretrain-datasets "${SOURCE_ARRAY[@]}" --seed "$SEED" --data-seed "$DATA_SEED" --device "$DEVICE" --run-dir "$RUN_DIR")
[[ -z "$CHECKPOINT" ]] || arguments+=(--checkpoint "$CHECKPOINT")
[[ -z "$TRANSFER_CONFIG" ]] || arguments+=(--transfer-config "$TRANSFER_CONFIG")
[[ -z "$FORECAST_CONFIG" ]] || arguments+=(--forecast-config "$FORECAST_CONFIG")
[[ -z "$LEARNING_RATE" ]] || arguments+=(--learning-rate "$LEARNING_RATE")
[[ -z "$EPOCHS" ]] || arguments+=(--epochs "$EPOCHS")
[[ -z "$BATCH_SIZE" ]] || arguments+=(--batch-size "$BATCH_SIZE")
[[ -z "$NUM_WORKERS" ]] || arguments+=(--num-workers "$NUM_WORKERS")
[[ -z "$PRETRAIN_FRACTION" ]] || arguments+=(--pretrain-fraction "$PRETRAIN_FRACTION")
if [[ "$RESUME" == "auto" ]]; then arguments+=(--resume); elif [[ -n "$RESUME" ]]; then arguments+=(--resume "$RESUME"); fi

echo "[$(date --iso-8601=seconds)] pretraining $MODEL_KEY sources=$SOURCES seed=$SEED"
if [[ "$MODEL_KEY" == "moflow" ]]; then
  "$PYTHON_BIN" -m baselines.train_moflow "${arguments[@]}"
else
  "$PYTHON_BIN" -m baselines.train_social_vae "${arguments[@]}"
fi

export TRAJECTORY_MODEL_REGISTRY="${TRAJECTORY_MODEL_REGISTRY:-experiments/model_registry}"
"$PYTHON_BIN" -m baselines.registry add "$MODEL_ID" "$RUN_DIR/checkpoints/best.pt" --model "$MODEL_KEY" --dataset "$SOURCES" --seed "$SEED"
echo "[$(date --iso-8601=seconds)] complete; registered $MODEL_ID"
