#!/usr/bin/env bash
# Train + generate + evaluate the additional diffusion baselines that compare
# our full Diffusion-LM against other diffusion variants (not just the AR
# baselines). Each is a separate from-scratch training run.
#
#   diffusion_lm_vanilla : faithful Li et al. 2022 (x0-pred, no self-cond/EMA/guidance)
#   diffusion_lm_eps     : same but eps-prediction (classic DDPM parameterization)
#   diffusion_crossattn  : encoder-decoder / cross-attention conditioning
#                          (DiffuSeq-style) — a genuinely different method,
#                          not a variant of our prefix-conditioned model.
#
# Outputs land at the top level of results/metrics/ so they appear as their own
# rows in the comparison table alongside diffusion_lm / gpt2 / t5.
set -euo pipefail

mkdir -p results/generations results/metrics

for cfg in configs/diffusion_lm_vanilla.yaml configs/diffusion_lm_eps.yaml configs/diffusion_crossattn.yaml; do
  name=$(basename "$cfg" .yaml)
  echo "=== ${name} ==="
  python -m src.train --config "$cfg"
  python -m src.generate --config "$cfg" --split test \
    --out "results/generations/${name}.jsonl"
  python -m src.evaluate \
    --predictions "results/generations/${name}.jsonl" \
    --out "results/metrics/${name}.json"
done

python -m src.compare_results --metrics_dir results/metrics --out results/summary.md
