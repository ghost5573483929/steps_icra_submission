# STEPS — trajectory forecasting

Preprocessing, pretraining, and transfer evaluation for the STEPS experiments.

## Setup

```bash
git clone --recurse-submodules <ANONYMIZED-REPOSITORY-URL>
cd steps-trajectory-forecasting
uv sync
```

An existing clone without submodules needs `git submodule update --init --recursive`; the
reference implementations under `references/` are required for training and evaluation.

## Reproducing the experiments

### 1. Build the datasets

Convert each raw corpus to the canonical schema, then build scene indices and splits:

```bash
uv run python -m datasets_preprocessing.steps.convert datasets/steps
uv run python -m preprocess datasets/steps --data-seed 42
```

Repeat for each downstream benchmark (`eth`, `ucy`, `grand_central`, `madras`). The split
manifests used in the paper are in `experiments/manifests/`.

### 2. Pretrain on STEPS

One run per model, seed, and training-set fraction. Omit `--pretrain-fraction` for the full set.

```bash
scripts/pretrain_transfer_model.sh \
  --model socialvae --sources steps --seed 0

scripts/pretrain_transfer_model.sh \
  --model moflow --sources steps --seed 0 --pretrain-fraction 0.1
```

`--model` is `socialvae` or `moflow`. Fractions reported in the paper are `0.05`, `0.1`,
`0.25`, `0.5`, and full. Each run writes `experiments/pretrained/<model_id>/checkpoints/best.pt`
and registers it under `experiments/model_registry/<model_id>.json`.

### 3. Evaluate on the downstream benchmarks

```bash
scripts/run_transfer_experiment.sh \
  --model socialvae --stage finetune --target eth_ucy --fold eth \
  --checkpoint experiments/pretrained/<model_id>/checkpoints/best.pt
```

`--stage` is `finetune` (transfer), `direct` (zero-shot), or `scratch` (target-only baseline).
`--target` is `eth_ucy`, `grand_central`, or `madras`; `eth_ucy` additionally requires `--fold`
(`eth`, `hotel`, `univ`, `zara1`, `zara2`). `--fraction` limits the target training data.

To evaluate a checkpoint without the driver:

```bash
uv run python -m baselines.evaluate_social_vae \
  --checkpoint <path>/best.pt --fold eth \
  --forecast-preset social_vae --run-dir runs/social_vae_eth

uv run python -m baselines.evaluate_moflow \
  --checkpoint <path>/best.pt --fold eth \
  --forecast-preset steps_default --run-dir runs/moflow_eth
```

The preset must match the one the checkpoint was trained with. Each run writes
`<run-dir>/evaluation.json` with minADE and minFDE; SocialVAE takes `--k` for the sample count
(default 20). Add `--device cpu` where no GPU is available, and `--max-scenes N` to shorten a run.

### 4. Aggregate

```bash
uv run python -m baselines.aggregate_transfer experiments/downstream
```

## Tests

```bash
uv run python -m pytest
```
