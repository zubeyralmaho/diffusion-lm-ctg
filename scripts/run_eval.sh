#!/usr/bin/env bash
set -euo pipefail

mkdir -p results/generations results/metrics

for model in gpt2_baseline t5_baseline diffusion_lm; do
  echo "=== $model ==="
  python -m src.generate \
    --config "configs/${model}.yaml" \
    --split test \
    --out "results/generations/${model}.jsonl"
  python -m src.evaluate \
    --predictions "results/generations/${model}.jsonl" \
    --out "results/metrics/${model}.json"
done
