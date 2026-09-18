#!/usr/bin/env bash
# Small local development comparison; override the environment variables below.
set -Eeuo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
PYTHON_BIN=${PYTHON_BIN:-/home/test/anaconda3/envs/llama/bin/python}
MODEL=${MODEL:-/home/test/hf_cache/models--TinyLlama--TinyLlama-1.1B-Chat-v1.0/snapshots/fe8a4ea1ffedaf415f4da2f062534de366a451e6}
ADAPTER=${ADAPTER:-/home/test/DEML/sst2_lora_adapter}
DICTIONARY=${DICTIONARY:-sparse_dict/auto_encoder.pt}
OUTPUT_ROOT=${OUTPUT_ROOT:-results_black_structured/comparison_$(date +%Y%m%d_%H%M%S)}
BUDGETS=${BUDGETS:-4096 16384}
SEEDS=${SEEDS:-0 1}
LAYERS=${LAYERS:-5}
PROMPTS=${PROMPTS:-2}
PROMPT_OFFSET=${PROMPT_OFFSET:-0}
ALPHA_LR=${ALPHA_LR:-0.01}
ALPHA_LR_FINAL=${ALPHA_LR_FINAL:-0.001}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-8}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1
export PYTHONUNBUFFERED=1
mkdir -p "$OUTPUT_ROOT"
cp invert_sparse_black.py invert_sparse_black_structured.py run_structured_black_comparison.sh summarize_structured_black.py "$OUTPUT_ROOT/"
sha256sum invert_sparse_black.py invert_sparse_black_structured.py > "$OUTPUT_ROOT/source_sha256.txt"
echo "$$" > "$OUTPUT_ROOT/runner.pid"
printf 'started\tmethod\tlayer\tseed\toptimization_budget\tstatus\tdirectory\n' > "$OUTPUT_ROOT/runs.tsv"
echo "Results and logs: $OUTPUT_ROOT"
for layer in $LAYERS; do
  for budget in $BUDGETS; do
    for seed in $SEEDS; do
      for method in original uniform adaptive; do
        output="$OUTPUT_ROOT/${method}_layer${layer}_q${budget}_seed${seed}"
        mkdir -p "$output"
        script=invert_sparse_black.py
        extra=()
        if [[ $method != original ]]; then
          script=invert_sparse_black_structured.py
          extra=(--group-selection "$method" --groups-per-step 8 --probes-per-group 4)
        fi
        command=("$PYTHON_BIN" "$script" --base-model-name "$MODEL"
          --target-adapter "$ADAPTER" --dictionary-path "$DICTIONARY"
          --representation sparse --dataset-path glue/sst2 --dataset-type datasets
          --dataset-len "$PROMPTS" --prompt-offset "$PROMPT_OFFSET" --seed "$seed"
          --split-layer "$layer" --iterations "$((budget / 64))" --directions 32
          --optimization-queries "$budget" --max-queries "$((budget + 4096))"
          --query-batch-size "$QUERY_BATCH_SIZE" --perturbation-radius 0.01 --perturbation-radius-final 0.00025
          --alpha-lr "$ALPHA_LR" --alpha-lr-final "$ALPHA_LR_FINAL" --alpha-init-std 0.001 --alpha-clip 0.2
          --alpha-l1 0.001 --alpha-optimizer sgd --hidden-mse-weight 0.1
          --no-return-best-alpha --top-k 10 --refine --refinement-passes 1
          --output-dir "$output" "${extra[@]}")
        printf '%q ' "${command[@]}" > "$output/command.sh"
        printf '\n' >> "$output/command.sh"
        printf '%s\t%s\t%s\t%s\t%s\trunning\t%s\n' "$(date -Is)" "$method" "$layer" "$seed" "$budget" "$output" >> "$OUTPUT_ROOT/runs.tsv"
        echo "Running $method layer=$layer queries=$budget seed=$seed"
        if "${command[@]}" > "$output/run.log" 2>&1; then
          status=completed
        else
          status=failed
          tail -n 15 "$output/run.log"
        fi
        printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$(date -Is)" "$method" "$layer" "$seed" "$budget" "$status" "$output" >> "$OUTPUT_ROOT/runs.tsv"
        "$PYTHON_BIN" summarize_structured_black.py "$OUTPUT_ROOT"
        if [[ $status == failed ]]; then exit 1; fi
      done
    done
  done
done
