"""Generation entrypoint. Writes JSONL of {mr_text, slots, prediction, reference}."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoTokenizer, GPT2LMHeadModel, T5ForConditionalGeneration

from src.data.e2e import load_e2e
from src.models.diffusion_lm import DiffusionLM


def _device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


_EPOCH_RE = re.compile(r"checkpoint_epoch(\d+)\.pt$")


def find_diffusion_checkpoint(ckpt_dir: Path) -> Path:
    best = ckpt_dir / "checkpoint_best.pt"
    if best.exists():
        return best

    ckpts = []
    for path in ckpt_dir.glob("checkpoint_epoch*.pt"):
        match = _EPOCH_RE.search(path.name)
        if match:
            ckpts.append((int(match.group(1)), path))

    if not ckpts:
        raise FileNotFoundError(f"No Diffusion-LM checkpoint in {ckpt_dir}")

    ckpts.sort(key=lambda item: item[0])
    return ckpts[-1][1]


def gen_gpt2(cfg: dict, examples) -> list[str]:
    # When --pretrained is passed we skip the fine-tuned checkpoint and load
    # the base model named in cfg.model.pretrained. Useful as an untrained
    # sanity baseline before any training has been run.
    ckpt = cfg["model"]["pretrained"] if cfg.get("_use_pretrained") else cfg["train"]["output_dir"]
    tok = AutoTokenizer.from_pretrained(ckpt)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(ckpt).to(_device()).eval()
    g = cfg["generate"]
    preds = []
    for ex in tqdm(examples, desc="gpt2"):
        prompt = f"{ex['mr_text']} <|sep|> "
        ids = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **ids,
            max_new_tokens=g["max_new_tokens"],
            do_sample=True,
            temperature=g["temperature"],
            top_p=g["top_p"],
            pad_token_id=tok.pad_token_id,
        )
        text = tok.decode(out[0, ids["input_ids"].size(1):], skip_special_tokens=True)
        preds.append(text.strip())
    return preds


def gen_t5(cfg: dict, examples) -> list[str]:
    ckpt = cfg["model"]["pretrained"] if cfg.get("_use_pretrained") else cfg["train"]["output_dir"]
    tok = AutoTokenizer.from_pretrained(ckpt)
    model = T5ForConditionalGeneration.from_pretrained(ckpt).to(_device()).eval()
    g = cfg["generate"]
    preds = []
    for ex in tqdm(examples, desc="t5"):
        ids = tok(ex["mr_text"], return_tensors="pt", truncation=True).to(model.device)
        out = model.generate(
            **ids,
            num_beams=g["num_beams"],
            max_new_tokens=g["max_new_tokens"],
        )
        preds.append(tok.decode(out[0], skip_special_tokens=True).strip())
    return preds


def gen_diffusion(cfg: dict, examples) -> list[str]:
    ckpt_dir = Path(cfg["train"]["output_dir"])
    ckpt_path = find_diffusion_checkpoint(ckpt_dir)
    payload = torch.load(ckpt_path, map_location=_device())
    tok = AutoTokenizer.from_pretrained(ckpt_dir)

    model = DiffusionLM(
        vocab_size=tok.vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        max_length=cfg["data"]["max_length"],
        num_timesteps=cfg["diffusion"]["num_timesteps"],
        noise_schedule=cfg["diffusion"].get("noise_schedule", "sqrt"),
    ).to(_device())
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"Loading diffusion checkpoint: {ckpt_path.name}")

    max_len = cfg["data"]["max_length"]
    prefix_len = cfg["data"].get("prefix_length", max_len // 2)

    preds = []
    for ex in tqdm(examples, desc="diffusion_lm"):
        prefix_ids = tok(
            ex["mr_text"],
            truncation=True,
            max_length=prefix_len,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(_device())
        ids = model.sample(
            length=max_len,
            prefix_ids=prefix_ids,
            ddim_steps=cfg["generate"]["ddim_steps"],
        )
        target_ids = ids[0, prefix_len:]
        preds.append(tok.decode(target_ids, skip_special_tokens=True).strip())
    return preds


DISPATCH = {"gpt2": gen_gpt2, "t5": gen_t5, "diffusion_lm": gen_diffusion}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=None, help="Cap examples for quick debug")
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Skip the fine-tuned checkpoint and use the base pretrained model "
             "named in cfg.model.pretrained. Only valid for gpt2 / t5.",
    )
    args = parser.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.pretrained:
        if cfg["model"]["type"] == "diffusion_lm":
            raise SystemExit("--pretrained is only meaningful for gpt2 / t5 baselines.")
        cfg["_use_pretrained"] = True
    examples = list(load_e2e(args.split))
    if args.limit:
        examples = examples[: args.limit]

    preds = DISPATCH[cfg["model"]["type"]](cfg, examples)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for ex, p in zip(examples, preds):
            f.write(json.dumps({
                "mr_text": ex["mr_text"],
                "slots": ex["slots"],
                "prediction": p,
                "reference": ex["reference"],
            }) + "\n")
    print(f"Wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
