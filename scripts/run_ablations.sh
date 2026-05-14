#!/usr/bin/env bash
# Run all ablation experiments. The base diffusion_lm checkpoint must already
# exist (`python -m src.train --config configs/diffusion_lm.yaml`).
# Only the prefix-length and schedule ablations need retraining; the DDIM-step
# ablations just resample from the base checkpoint.
set -euo pipefail

mkdir -p results/generations/ablations results/metrics/ablations

# 1) Sampling-only ablations (reuse base checkpoint).
for cfg in configs/ablations/ddim_50.yaml configs/ablations/ddim_100.yaml; do
  name=$(basename "$cfg" .yaml)
  python -m src.generate --config "$cfg" --split test \
    --out "results/generations/ablations/${name}.jsonl"
  python -m src.evaluate \
    --predictions "results/generations/ablations/${name}.jsonl" \
    --out "results/metrics/ablations/${name}.json"
done

# 2) Retraining ablations (separate checkpoints).
for cfg in configs/ablations/schedule_cosine.yaml \
           configs/ablations/prefix_16.yaml \
           configs/ablations/prefix_48.yaml; do
  name=$(basename "$cfg" .yaml)
  python -m src.train --config "$cfg"
  python -m src.generate --config "$cfg" --split test \
    --out "results/generations/ablations/${name}.jsonl"
  python -m src.evaluate \
    --predictions "results/generations/ablations/${name}.jsonl" \
    --out "results/metrics/ablations/${name}.json"
done

python -m src.compare_results --metrics_dir results/metrics --out results/summary.md
