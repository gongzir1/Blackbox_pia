# Blackbox PIA

Standalone text black-box prompt inversion extracted from the parent sparse PIA
research project.

## Main attack

`invert_sparse_black.py` implements query-only sparse recovery. It uses the
target model only through `BlackBoxOracle.query`, estimates updates with
two-sided finite differences, and decodes recovered hidden states back to
tokens. The target model weights are not inspected by the optimizer and no
target gradients are requested.

Variants included here:

- `invert_sparse_black_tokenwise.py`: per-position loss variant.
- `invert_sparse_black_structured.py`: adaptive block-coordinate variant.
- `run_black_sparse_table4.py`: resumable multi-layer, multi-seed runner.
- `run_structured_black_comparison.sh`: local comparison runner.
- `run_black_tuning.py`: bounded development/validation tuner.

## Setup

```bash
conda create -n blackbox-pia python=3.10
conda activate blackbox-pia
pip install -r requirements.txt
```

The base model and private LoRA adapter remain external. The sparse dictionary
is included at `sparse_dict/auto_encoder.pt`; replace it with a compatible
dictionary when needed. The included `data/heldout_prompts.json` is the prompt
manifest used by the Table 4 runner.

## Run one attack

```bash
python invert_sparse_black.py \
  --base-model-name /path/to/base-model \
  --target-adapter /path/to/private-lora-adapter \
  --dictionary-path sparse_dict/auto_encoder.pt \
  --dataset-path glue/sst2 --dataset-type datasets \
  --dataset-len 1 --split-layer 5 \
  --output-dir results_black
```

Use `python invert_sparse_black.py --help` for all query-budget and optimizer
options. GPU execution is required by the attack.

## Run a small Table 4 grid

```bash
python run_black_sparse_table4.py \
  --base-model-name /path/to/base-model \
  --target-adapter /path/to/private-lora-adapter \
  --dictionary-path sparse_dict/auto_encoder.pt \
  --manifest data/heldout_prompts.json \
  --layers 5 --budgets 8192 --seeds 0 --prompts 2 \
  --wandb-mode disabled --output-dir results_black_sparse_table4
```

Generated results belong under `results_*` directories and are intentionally
not part of this extracted project.
