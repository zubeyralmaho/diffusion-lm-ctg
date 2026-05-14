"""Generation entrypoint. Writes JSONL of {mr, slots, prediction, reference}."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError(
        f"Load checkpoint from {cfg['train']['output_dir']}, run inference on "
        f"E2E {args.split}, append each prediction as JSON line to {args.out}."
    )


if __name__ == "__main__":
    main()
