#!/usr/bin/env bash
set -euo pipefail

# Pre-fetch the cleaned E2E NLG CSVs from tuetschek/e2e-cleaning so the first
# training run does not block on the download. The loader caches into
# data/raw/e2e/ automatically; this just warms it.
python -c "from src.data.e2e import load_e2e; [load_e2e(s) for s in ('train','validation','test')]"
echo "E2E NLG cached under data/raw/e2e/."
