# Repository Guidelines

## Project Structure & Module Organization

This repository contains research code for sparse prompt inversion from transformer internal states. `invert_l1_auto_encoder.py` is the proposed white-box sparse attack, `invert.py` is the dense PIA baseline, and `dict_learn.py` supports dictionary construction. Shared dataset and utility code lives in `dataset.py`, `utils.py`, and `misc.py`. Store generated metrics under a dedicated `results_*` directory and model artifacts under clearly named checkpoint directories. `img/` and `EIA_in_LLM_Collerative_Inference/` contain publication assets and paper source.

## Build, Test, and Development Commands

Create the documented environment with Python 3.10:

```bash
conda create -n llama python=3.10
conda activate llama
pip install -r requirements.txt
```

Run the proposed method with `sh run_inversion.sh sparse` and the baseline with `sh run_inversion.sh baseline`. Both entry points expose CLI help.

Before committing, syntax-check changed Python and shell files:

```bash
python -m py_compile path/to/changed.py
bash -n path/to/changed.sh
```

## Coding Style & Naming Conventions

Use four-space indentation, snake_case functions and variables, PascalCase classes, and UPPER_CASE constants. Keep CLI options in kebab-case while mapping them to snake_case `argparse` destinations. Prefer small reusable functions over adding another near-duplicate experiment script. No formatter is enforced; follow PEP 8 and preserve surrounding style.

## Testing Guidelines

There is no root automated test suite or coverage requirement. Add focused deterministic smoke tests for hooks, tensor shapes, attention masks, dictionary compatibility, and decoding. Use small dataset limits during local validation. Do not run full GPU experiment grids as routine tests.

## Commit & Pull Request Guidelines

Use concise imperative subjects such as `Fix sparse dictionary dimensions`. Keep commits scoped to one experiment or correction. Pull requests should describe the hypothesis, changed commands/configuration, model and dataset assumptions, validation performed, and expected output paths. Include compact metric tables for behavioral changes; screenshots are needed only for figures or rendered documentation.

## Security & Configuration Tips

Never commit API keys, private datasets, or large generated checkpoints and dictionaries. Keep cache and output paths configurable. Use identical prompts, models, split layers, seeds, optimization budgets, and decoding settings when comparing sparse recovery with the dense baseline.
