"""Unified training entrypoint. Dispatches by config.model.type."""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def train_gpt2(cfg: dict) -> None:
    raise NotImplementedError(
        "Implement HF Trainer loop: load E2E, format as `MR <sep> reference`, "
        "fine-tune GPT-2 with causal LM objective."
    )


def train_t5(cfg: dict) -> None:
    raise NotImplementedError(
        "Implement HF Trainer Seq2Seq loop: source=MR text, target=reference."
    )


def train_diffusion_lm(cfg: dict) -> None:
    raise NotImplementedError(
        "Implement training loop calling DiffusionLM.forward; log MSE + CE losses."
    )


DISPATCH = {
    "gpt2": train_gpt2,
    "t5": train_t5,
    "diffusion_lm": train_diffusion_lm,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    Path(cfg["train"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    DISPATCH[cfg["model"]["type"]](cfg)


if __name__ == "__main__":
    main()
