# Controllable Text Generation via Diffusion Models

**CENG 467 — Natural Language Understanding and Generation, Spring 2026**
İzmir Institute of Technology — Prof. Dr. Aytuğ Onan

- Student: Zübeyr Almaho (300201023)
- Topic #4: Controllable Text Generation via Diffusion Models

## Overview

This project implements **Diffusion-LM** (Li et al., 2022) for controllable text
generation on the **E2E NLG** dataset and compares it against two
autoregressive baselines (GPT-2 and T5). We measure fluency (perplexity, BLEU),
lexical diversity (distinct-n, self-BLEU), and controllability (slot-filling
accuracy on E2E meaning representations).

## Project Structure

```
.
├── configs/            # YAML configs for each model / experiment
├── data/               # raw + processed datasets (gitignored)
├── src/
│   ├── data/           # dataset loading & preprocessing
│   ├── models/         # diffusion_lm, baselines
│   ├── train.py        # unified training entrypoint
│   ├── generate.py     # decoding / sampling
│   └── evaluate.py     # BLEU, ROUGE, perplexity, distinct-n, controllability
├── scripts/            # shell helpers
├── notebooks/          # Colab notebooks (training, error analysis)
├── results/            # generated outputs + metric tables
└── report/             # LNCS paper sources
```

## Setup (local)

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
bash scripts/prepare_data.sh
```

## Setup (Google Colab)

Open `notebooks/colab_train.ipynb` in Colab. The first cell clones this repo
and installs dependencies. If the repo already exists in the runtime, the cell
pulls the latest committed changes before installing dependencies. A T4 GPU is
sufficient for the E2E experiments.

## Reproducing Results

```bash
# 1. Prepare data
bash scripts/prepare_data.sh

# 2. Train baselines
python -m src.train --config configs/gpt2_baseline.yaml
python -m src.train --config configs/t5_baseline.yaml

# 3. Train proposed method
python -m src.train --config configs/diffusion_lm.yaml

# 4. Generate and evaluate
bash scripts/run_eval.sh
```

## Diffusion Notes

The diffusion pipeline relies on three implementation details that matter for
quality:

- sampling uses a deterministic DDIM-style reverse update rather than repeatedly overwriting `x <- x0_hat`
- padded target positions are excluded from the diffusion loss
- diffusion token embeddings are initialized from the pretrained BERT tokenizer embeddings

Because of these fixes, any diffusion checkpoints trained before these changes
should be discarded and retrained from scratch.

Metrics land in `results/metrics/` as JSON; generated samples in
`results/generations/`.

## Baselines

| Model | Type | Purpose |
|-------|------|---------|
| GPT-2 small | Autoregressive decoder | Strong AR baseline |
| T5-small | Encoder-decoder | Conditional generation baseline |
| Diffusion-LM | Non-autoregressive diffusion | **Proposed method** |

## References

- Li, X. L., Thickstun, J., Gulrajani, I., Liang, P., & Hashimoto, T. B. (2022).
  *Diffusion-LM Improves Controllable Text Generation*. NeurIPS 2022.
- Novikova, J., Dušek, O., & Rieser, V. (2017). *The E2E Dataset: New Challenges
  for End-to-End Generation*. SIGDIAL 2017.
