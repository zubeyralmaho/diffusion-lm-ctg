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


def _present_slots(slots: dict | None) -> dict[str, str]:
    if not slots:
        return {}
    return {
        key: value
        for key, value in slots.items()
        if isinstance(value, str) and value.strip()
    }


def _decode_diffusion_target(tok, token_ids: torch.Tensor) -> str:
    stop_ids = {
        token_id
        for token_id in (tok.sep_token_id, tok.eos_token_id, tok.pad_token_id)
        if token_id is not None
    }
    trimmed_ids = []
    for token_id in token_ids.tolist():
        if token_id in stop_ids:
            break
        trimmed_ids.append(token_id)
    return tok.decode(trimmed_ids, skip_special_tokens=True).strip()


def _slot_value(slots: dict | None, attribute: str) -> str | None:
    if not isinstance(slots, dict):
        return None
    value = slots.get(attribute)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


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
    saved_cfg = payload.get("config", cfg)
    tok = AutoTokenizer.from_pretrained(ckpt_dir)

    model = DiffusionLM(
        vocab_size=tok.vocab_size,
        embedding_dim=saved_cfg["model"]["embedding_dim"],
        hidden_dim=saved_cfg["model"]["hidden_dim"],
        num_layers=saved_cfg["model"]["num_layers"],
        num_heads=saved_cfg["model"]["num_heads"],
        max_length=saved_cfg["data"]["max_length"],
        num_timesteps=saved_cfg["diffusion"]["num_timesteps"],
        noise_schedule=saved_cfg["diffusion"].get("noise_schedule", "sqrt"),
        self_conditioning=saved_cfg["diffusion"].get("self_conditioning", False),
        classifier_num_classes=len(
            saved_cfg.get("diffusion", {}).get("classifier_guidance", {}).get("labels", [])
        ) if saved_cfg.get("diffusion", {}).get("classifier_guidance", {}).get("enabled", False) else 0,
        classifier_hidden_dim=saved_cfg.get("diffusion", {}).get("classifier_guidance", {}).get("hidden_dim"),
        classifier_loss_weight=saved_cfg.get("diffusion", {}).get("classifier_guidance", {}).get("loss_weight", 0.0),
        mse_lambda=saved_cfg.get("diffusion", {}).get("mse_lambda", 1.0),
        pad_token_id=tok.pad_token_id,
    ).to(_device())
    model.load_state_dict(payload["model"])
    model.eval()
    print(f"Loading diffusion checkpoint: {ckpt_path.name}")

    max_len = saved_cfg["data"]["max_length"]
    prefix_len = saved_cfg["data"].get("prefix_length", max_len // 2)
    ddim_steps = cfg["generate"].get(
        "ddim_steps",
        saved_cfg.get("generate", {}).get("ddim_steps", 200),
    )
    rounding_interval = cfg["generate"].get(
        "rounding_interval",
        saved_cfg.get("generate", {}).get("rounding_interval"),
    )
    batch_size = cfg["generate"].get(
        "batch_size",
        saved_cfg.get("generate", {}).get("batch_size", 16),
    )
    guidance_scale = cfg["generate"].get(
        "classifier_guidance_scale",
        saved_cfg.get("generate", {}).get("classifier_guidance_scale", 0.0),
    )
    guidance_cfg = saved_cfg.get("diffusion", {}).get("classifier_guidance", {})
    guidance_attribute = guidance_cfg.get("attribute") if guidance_cfg.get("enabled", False) else None
    guidance_label_to_id = {
        value: index for index, value in enumerate(guidance_cfg.get("labels", []))
    }

    preds = []
    for start in tqdm(range(0, len(examples), batch_size), desc="diffusion_lm"):
        batch_examples = examples[start : start + batch_size]
        prefix_ids = tok(
            [ex["mr_text"] for ex in batch_examples],
            truncation=True,
            max_length=prefix_len,
            padding="max_length",
            add_special_tokens=False,
            return_tensors="pt",
        )["input_ids"].to(_device())
        guidance_labels = None
        if guidance_attribute and guidance_scale > 0.0 and guidance_label_to_id:
            guidance_labels = torch.tensor(
                [
                    guidance_label_to_id.get(_slot_value(ex.get("slots"), guidance_attribute), 0)
                    for ex in batch_examples
                ],
                device=prefix_ids.device,
                dtype=torch.long,
            )
        ids = model.sample(
            length=max_len,
            prefix_ids=prefix_ids,
            ddim_steps=ddim_steps,
            rounding_interval=rounding_interval,
            guidance_labels=guidance_labels,
            guidance_scale=guidance_scale,
        )
        for row in ids:
            preds.append(_decode_diffusion_target(tok, row[prefix_len:]))
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
                "slots": _present_slots(ex["slots"]),
                "prediction": p,
                "reference": ex["reference"],
            }) + "\n")
    print(f"Wrote {len(preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
