#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$PROJECT_DIR/src${PYTHONPATH:+:$PYTHONPATH}"

MODEL=""
STAGE=""
CHECKPOINT=""
TARGET=""
FOLD=""
FRACTION="1.0"
SEED=""
DATA_SEED=42
DEVICE=cuda
OUTPUT=experiments/downstream
TRANSFER_CONFIG=""
FORECAST_CONFIG=""
LEARNING_RATE=""
EPOCHS=""
BATCH_SIZE=""
NUM_WORKERS=""
VALIDATION_MODE=""  # ETH/UCY only; the trainer resolves the default (grouped) and rejects it for other targets
VISUAL_SCENES=10
RESUME=""
PYTHON_BIN="${PYTHON_BIN:-python}"

# Resolves --seed from a pretrained checkpoint when it is omitted for --stage
# finetune/direct: from the registry entry's "seed" field for a bare model ID,
# or from the run's config.json (two directories above a .../checkpoints/best.pt
# path) for an explicit path. Always uses the system python3, not --python, so
# this works even when --python points at a non-general-purpose stub interpreter.
resolve_seed_from_checkpoint() {
  local checkpoint="$1"
  local seed_value=""
  local read_seed='
import json, sys
data = json.load(open(sys.argv[1]))
value = data.get("seed")
print(value if value is not None else "")
'
  if [[ "$checkpoint" == */* ]]; then
    local checkpoint_path="${checkpoint%/}"
    if [[ "$(basename "$checkpoint_path")" == "best.pt" ]]; then
      local config_path
      config_path="$(dirname "$(dirname "$checkpoint_path")")/config.json"
      if [[ -f "$config_path" ]]; then
        seed_value="$(python3 -c "$read_seed" "$config_path" 2>/dev/null || true)"
      fi
    fi
  else
    local registry_dir="${TRAJECTORY_MODEL_REGISTRY:-experiments/model_registry}"
    local entry_path="$registry_dir/$checkpoint.json"
    if [[ -f "$entry_path" ]]; then
      seed_value="$(python3 -c "$read_seed" "$entry_path" 2>/dev/null || true)"
    fi
  fi
  printf '%s' "$seed_value"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model) MODEL="$2"; shift 2;;
    --stage) STAGE="$2"; shift 2;;
    --checkpoint) CHECKPOINT="$2"; shift 2;;
    --target) TARGET="$2"; shift 2;;
    --fold) FOLD="$2"; shift 2;;
    --fraction) FRACTION="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --data-seed) DATA_SEED="$2"; shift 2;;
    --device) DEVICE="$2"; shift 2;;
    --output) OUTPUT="$2"; shift 2;;
    --transfer-config) TRANSFER_CONFIG="$2"; shift 2;;
    --forecast-config) FORECAST_CONFIG="$2"; shift 2;;
    --learning-rate) LEARNING_RATE="$2"; shift 2;;
    --epochs) EPOCHS="$2"; shift 2;;
    --batch-size) BATCH_SIZE="$2"; shift 2;;
    --num-workers) NUM_WORKERS="$2"; shift 2;;
    --validation-mode) VALIDATION_MODE="$2"; shift 2;;
    --visual-scenes) VISUAL_SCENES="$2"; shift 2;;
    --resume|--resuming)
      if [[ $# -gt 1 && "$2" != --* ]]; then RESUME="$2"; shift 2; else RESUME="auto"; shift; fi;;
    --python) PYTHON_BIN="$2"; shift 2;;
    *) echo "Unknown argument: $1" >&2; exit 2;;
  esac
done

[[ "$MODEL" == "moflow" || "$MODEL" == "socialvae" || "$MODEL" == "social_vae" ]] || { echo "--model must be moflow or socialvae" >&2; exit 2; }
[[ "$STAGE" == "finetune" || "$STAGE" == "scratch" || "$STAGE" == "direct" ]] || { echo "--stage must be finetune, scratch, or direct" >&2; exit 2; }
[[ "$TARGET" == "eth_ucy" || "$TARGET" == "grand_central" || "$TARGET" == "madras" ]] || { echo "--target must be eth_ucy, grand_central, or madras" >&2; exit 2; }
if [[ "$TARGET" == "eth_ucy" ]]; then
  [[ "$FOLD" =~ ^(eth|hotel|univ|zara1|zara2)$ ]] || { echo "ETH/UCY requires --fold eth|hotel|univ|zara1|zara2" >&2; exit 2; }
elif [[ -n "$FOLD" ]]; then
  echo "--fold is only valid for --target eth_ucy" >&2; exit 2
fi
if [[ -n "$VALIDATION_MODE" && "$TARGET" != "eth_ucy" ]]; then
  echo "--validation-mode is only valid for --target eth_ucy" >&2; exit 2
fi
if [[ "$STAGE" == "finetune" || "$STAGE" == "direct" ]]; then
  [[ -n "$CHECKPOINT" ]] || { echo "--checkpoint is required for --stage $STAGE" >&2; exit 2; }
elif [[ -n "$CHECKPOINT" ]]; then
  echo "Scratch training cannot load --checkpoint" >&2; exit 2
fi
if [[ "$STAGE" == "direct" && "$FRACTION" != "1.0" ]]; then
  echo "--fraction must not be given (or must be 1.0) for --stage direct" >&2; exit 2
fi

if [[ "$STAGE" == "scratch" ]]; then
  [[ -n "$SEED" ]] || { echo "--seed is required for --stage scratch" >&2; exit 2; }
elif [[ -z "$SEED" ]]; then
  SEED="$(resolve_seed_from_checkpoint "$CHECKPOINT")"
  [[ -n "$SEED" ]] || { echo "Could not resolve --seed from --checkpoint $CHECKPOINT; pass --seed explicitly" >&2; exit 2; }
  echo "Resolved --seed=$SEED from --checkpoint $CHECKPOINT"
fi

MODEL_KEY="${MODEL/socialvae/social_vae}"
SCOPE="$TARGET${FOLD:+-$FOLD}"
if [[ "$STAGE" == "scratch" ]]; then
  INITIALIZATION_TAG="random"
elif [[ "$CHECKPOINT" == */* ]]; then
  checkpoint_path="${CHECKPOINT%/}"
  if [[ "$(basename "$checkpoint_path")" == "best.pt" ]]; then
    INITIALIZATION_TAG="$(basename "$(dirname "$(dirname "$checkpoint_path")")")"
  else
    INITIALIZATION_TAG="$(basename "$checkpoint_path")"
  fi
else
  INITIALIZATION_TAG="$CHECKPOINT"
fi
INITIALIZATION_TAG="${INITIALIZATION_TAG//[^a-zA-Z0-9_.-]/_}"
if [[ "$STAGE" == "direct" ]]; then
  RUN_DIR="$OUTPUT/direct/$MODEL_KEY/$INITIALIZATION_TAG/seed$SEED/$SCOPE"
else
  RUN_DIR="$OUTPUT/$STAGE/$MODEL_KEY/$INITIALIZATION_TAG/seed$SEED/$SCOPE/fraction-$FRACTION"
fi
TRAIN_DIR="$RUN_DIR/train"
EVAL_DIR="$RUN_DIR/test"

# Existing runs are immutable unless continuation was requested explicitly. Checked
# before any artifact (params.env, driver.log) is written so a refused invocation
# leaves a prior run untouched.
if [[ -z "$RESUME" && ( -e "$TRAIN_DIR/status.json" || -d "$TRAIN_DIR/checkpoints" || -e "$EVAL_DIR/evaluation.json" ) ]]; then
  echo "Refusing to overwrite existing run: $RUN_DIR" >&2
  echo "Pass --resume/--resuming to continue it, or choose another --output/--seed/etc." >&2
  exit 3
fi

mkdir -p "$RUN_DIR"
exec > >(tee -a "$RUN_DIR/driver.log") 2>&1
printf '\n===== run_transfer_experiment.sh invocation %s =====\n' "$(date --iso-8601=seconds)"

printf 'MODEL=%q\nSTAGE=%q\nINITIALIZATION_TAG=%q\nCHECKPOINT=%q\nTARGET=%q\nFOLD=%q\nFRACTION=%q\nSEED=%q\nDATA_SEED=%q\nDEVICE=%q\nTRANSFER_CONFIG=%q\nFORECAST_CONFIG=%q\nLEARNING_RATE=%q\nEPOCHS=%q\nBATCH_SIZE=%q\nNUM_WORKERS=%q\nVALIDATION_MODE=%q\nVISUAL_SCENES=%q\n' \
  "$MODEL_KEY" "$STAGE" "$INITIALIZATION_TAG" "$CHECKPOINT" "$TARGET" "$FOLD" "$FRACTION" "$SEED" "$DATA_SEED" "$DEVICE" "$TRANSFER_CONFIG" "$FORECAST_CONFIG" "$LEARNING_RATE" "$EPOCHS" "$BATCH_SIZE" "$NUM_WORKERS" "$VALIDATION_MODE" "$VISUAL_SCENES" > "$RUN_DIR/params.env"

shared=(--stage "$STAGE" --target-dataset "$TARGET" --target-fraction "$FRACTION" --seed "$SEED" --data-seed "$DATA_SEED" --device "$DEVICE" --run-dir "$TRAIN_DIR")
[[ -z "$VALIDATION_MODE" ]] || shared+=(--validation-mode "$VALIDATION_MODE")
[[ -z "$FOLD" ]] || shared+=(--fold "$FOLD")
[[ -z "$CHECKPOINT" ]] || shared+=(--checkpoint "$CHECKPOINT")
[[ -z "$TRANSFER_CONFIG" ]] || shared+=(--transfer-config "$TRANSFER_CONFIG")
[[ -z "$FORECAST_CONFIG" ]] || shared+=(--forecast-config "$FORECAST_CONFIG")
[[ -z "$LEARNING_RATE" ]] || shared+=(--learning-rate "$LEARNING_RATE")
[[ -z "$EPOCHS" ]] || shared+=(--epochs "$EPOCHS")
[[ -z "$BATCH_SIZE" ]] || shared+=(--batch-size "$BATCH_SIZE")
[[ -z "$NUM_WORKERS" ]] || shared+=(--num-workers "$NUM_WORKERS")
if [[ "$RESUME" == "auto" ]]; then shared+=(--resume); elif [[ -n "$RESUME" ]]; then shared+=(--resume "$RESUME"); fi

if [[ "$STAGE" == "direct" ]]; then
  eval_checkpoint="$CHECKPOINT"
else
  eval_checkpoint="$TRAIN_DIR/checkpoints/best.pt"
fi
evaluation=(--target-dataset "$TARGET" --checkpoint "$eval_checkpoint" --run-dir "$EVAL_DIR" --device "$DEVICE" --data-seed "$DATA_SEED" --visual-scenes "$VISUAL_SCENES")
[[ -z "$FOLD" ]] || evaluation+=(--fold "$FOLD")
[[ -z "$FORECAST_CONFIG" ]] || evaluation+=(--forecast-config "$FORECAST_CONFIG")
# evaluate_moflow does not accept --num-workers; evaluate_social_vae does.
if [[ "$MODEL_KEY" == "social_vae" && -n "$NUM_WORKERS" ]]; then evaluation+=(--num-workers "$NUM_WORKERS"); fi

if [[ "$STAGE" == "direct" ]]; then
  echo "[$(date --iso-8601=seconds)] direct evaluation $MODEL_KEY $SCOPE seed=$SEED checkpoint=$CHECKPOINT"
  if [[ "$MODEL_KEY" == "moflow" ]]; then
    "$PYTHON_BIN" -m baselines.evaluate_moflow "${evaluation[@]}"
  else
    "$PYTHON_BIN" -m baselines.evaluate_social_vae "${evaluation[@]}"
  fi
else
  echo "[$(date --iso-8601=seconds)] training $MODEL_KEY $STAGE $SCOPE fraction=$FRACTION seed=$SEED"
  if [[ "$MODEL_KEY" == "moflow" ]]; then
    "$PYTHON_BIN" -m baselines.train_moflow "${shared[@]}"
    "$PYTHON_BIN" -m baselines.evaluate_moflow "${evaluation[@]}"
  else
    "$PYTHON_BIN" -m baselines.train_social_vae "${shared[@]}"
    "$PYTHON_BIN" -m baselines.evaluate_social_vae "${evaluation[@]}"
  fi
fi
echo "[$(date --iso-8601=seconds)] complete; metrics: $EVAL_DIR/evaluation.json"
