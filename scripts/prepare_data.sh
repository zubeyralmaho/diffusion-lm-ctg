#!/usr/bin/env bash
set -euo pipefail

# E2E NLG is fetched via HuggingFace datasets at runtime; this script just
# pre-warms the cache so first training run isn't blocked on download.
python -c "from datasets import load_dataset; load_dataset('e2e_nlg_cleaned')"
echo "E2E NLG cached."
