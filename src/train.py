"""Unified training entrypoint. Dispatches by config.model.type."""
from __future__ import annotations

import argparse
import copy
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    DataCollatorForLanguageModeling,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    TrainingArguments,
    get_linear_schedule_with_warmup,
)

from src.data.e2e import load_e2e
from src.models.baselines import build_gpt2, build_t5
from src.models.diffusion_lm import DiffusionLM


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def evaluate_diffusion(model: DiffusionLM, loader: DataLoader, device: str) -> float:
    model.eval()
    running = 0.0
    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            noise_mask = batch.get("noise_mask")
            if noise_mask is not None:
                noise_mask = noise_mask.to(device)
            guidance_labels = batch.get("guidance_label")
            if guidance_labels is not None:
                guidance_labels = guidance_labels.to(device)
            loss = model(
                input_ids,
                target_mask=target_mask,
                noise_mask=noise_mask,
                guidance_labels=guidance_labels,
            )
            running += loss.item()
    return running / max(len(loader), 1)


def collate_diffusion_batch(features: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    batch = {
        "input_ids": torch.tensor([feature["input_ids"] for feature in features], dtype=torch.long),
        "target_mask": torch.tensor([feature["target_mask"] for feature in features], dtype=torch.long),
    }
    # noise_mask is optional for backward compatibility with old encodings.
    if "noise_mask" in features[0]:
        batch["noise_mask"] = torch.tensor(
            [feature["noise_mask"] for feature in features], dtype=torch.long
        )
    if "guidance_label" in features[0]:
        batch["guidance_label"] = torch.tensor(
            [feature["guidance_label"] for feature in features], dtype=torch.long
        )
    return batch


def slot_value(slots: dict | None, attribute: str) -> str | None:
    if not isinstance(slots, dict):
        return None
    value = slots.get(attribute)
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def encode_diffusion_example(
    ex: dict,
    tok,
    prefix_len: int,
    target_len: int,
    guidance_attribute: str | None = None,
    guidance_label_to_id: dict[str, int] | None = None,
) -> dict[str, list[int] | int]:
    prefix_ids = tok(
        ex["mr_text"],
        truncation=True,
        max_length=prefix_len,
        padding="max_length",
        add_special_tokens=False,
    )["input_ids"]

    pad_token_id = tok.pad_token_id
    if pad_token_id is None:
        raise ValueError("Diffusion tokenizer must define a pad_token_id.")

    end_token_id = tok.sep_token_id if tok.sep_token_id is not None else tok.eos_token_id
    max_target_tokens = target_len - 1 if end_token_id is not None else target_len
    target_ids = tok(
        ex["reference"],
        truncation=True,
        max_length=max_target_tokens,
        add_special_tokens=False,
    )["input_ids"]

    if end_token_id is not None:
        target_ids = target_ids + [end_token_id]

    target_ids = target_ids[:target_len]
    actual_target_len = len(target_ids)
    padded_target_ids = target_ids + [pad_token_id] * (target_len - actual_target_len)

    # Two separate masks with distinct jobs:
    # - noise_mask  : covers the ENTIRE target window (prefix=0, target+pads=1).
    #   The forward process noises these positions; prefix stays clean.
    # - target_mask : same extent as noise_mask.  CE and MSE are applied to
    #   ALL target positions including trailing pads, so the model learns
    #   to emit PAD tokens in the trailing region.  Without this signal the
    #   model is never trained on what to produce past the actual text, and
    #   at inference it generates high-norm vocab noise in those slots.
    encoded = {
        "input_ids": prefix_ids + padded_target_ids,
        "target_mask": [0] * prefix_len + [1] * target_len,
        "noise_mask":  [0] * prefix_len + [1] * target_len,
    }
    if guidance_attribute is not None and guidance_label_to_id is not None:
        encoded["guidance_label"] = guidance_label_to_id.get(slot_value(ex.get("slots"), guidance_attribute), 0)
    return encoded


def build_guidance_label_values(dataset, attribute: str) -> list[str]:
    values = sorted({
        value
        for row in dataset
        if (value := slot_value(row.get("slots"), attribute)) is not None
    })
    return ["<missing>"] + values


def build_ema_model(model: DiffusionLM) -> DiffusionLM:
    ema_model = copy.deepcopy(model)
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)
    return ema_model


def update_ema_model(ema_model: DiffusionLM, model: DiffusionLM, decay: float) -> None:
    with torch.no_grad():
        for ema_param, model_param in zip(ema_model.parameters(), model.parameters()):
            ema_param.data.mul_(decay).add_(model_param.data, alpha=1.0 - decay)
        for ema_buffer, model_buffer in zip(ema_model.buffers(), model.buffers()):
            ema_buffer.copy_(model_buffer)


def diffusion_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps_backend = getattr(torch.backends, "mps", None)
    if mps_backend is not None and mps_backend.is_available():
        return "mps"
    return "cpu"


# ---------- GPT-2 ----------

def train_gpt2(cfg: dict) -> None:
    model, tok = build_gpt2(cfg["model"]["pretrained"])
    max_len = cfg["data"]["max_length"]

    def encode(ex):
        text = f"{ex['mr_text']} <|sep|> {ex['reference']}{tok.eos_token}"
        out = tok(text, truncation=True, max_length=max_len, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    train_ds = load_e2e("train").map(encode, remove_columns=["mr_text", "slots", "reference"])
    val_ds = load_e2e("validation").map(encode, remove_columns=["mr_text", "slots", "reference"])

    args = TrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["data"]["batch_size"],
        per_device_eval_batch_size=cfg["data"]["batch_size"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_steps=cfg["train"]["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        logging_steps=100,
        report_to=[],
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    # Save tokenizer + final model so generate.py can from_pretrained() the dir.
    tok.save_pretrained(cfg["train"]["output_dir"])
    model.save_pretrained(cfg["train"]["output_dir"])


# ---------- T5 ----------

def train_t5(cfg: dict) -> None:
    model, tok = build_t5(cfg["model"]["pretrained"])
    src_len = cfg["data"]["max_source_length"]
    tgt_len = cfg["data"]["max_target_length"]

    def encode(ex):
        src = tok(ex["mr_text"], truncation=True, max_length=src_len)
        tgt = tok(text_target=ex["reference"], truncation=True, max_length=tgt_len)
        src["labels"] = tgt["input_ids"]
        return src

    train_ds = load_e2e("train").map(encode, remove_columns=["mr_text", "slots", "reference"])
    val_ds = load_e2e("validation").map(encode, remove_columns=["mr_text", "slots", "reference"])

    args = Seq2SeqTrainingArguments(
        output_dir=cfg["train"]["output_dir"],
        num_train_epochs=cfg["train"]["epochs"],
        per_device_train_batch_size=cfg["data"]["batch_size"],
        per_device_eval_batch_size=cfg["data"]["batch_size"],
        learning_rate=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
        warmup_steps=cfg["train"]["warmup_steps"],
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        logging_steps=100,
        report_to=[],
    )
    Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    ).train()
    tok.save_pretrained(cfg["train"]["output_dir"])
    model.save_pretrained(cfg["train"]["output_dir"])


# ---------- Diffusion-LM ----------

def train_diffusion_lm(cfg: dict) -> None:
    from transformers import AutoTokenizer

    device = diffusion_device()
    tok = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer"])
    embedding_init = None
    if cfg["model"].get("init_from_tokenizer_embeddings", False):
        token_model = AutoModel.from_pretrained(cfg["model"]["tokenizer"])
        embedding_init = token_model.get_input_embeddings().weight.detach().cpu()
        cfg["model"]["embedding_dim"] = int(embedding_init.size(1))
        print(
            "Initializing diffusion token embeddings from",
            cfg["model"]["tokenizer"],
            f"({cfg['model']['embedding_dim']} dims)",
        )
        del token_model
    max_len = cfg["data"]["max_length"]
    prefix_len = cfg["data"].get("prefix_length", max_len // 2)
    target_len = max_len - prefix_len
    train_raw = load_e2e("train")
    val_raw = load_e2e("validation")

    classifier_cfg = cfg["diffusion"].get("classifier_guidance", {})
    classifier_enabled = classifier_cfg.get("enabled", False)
    guidance_attribute = classifier_cfg.get("attribute") if classifier_enabled else None
    guidance_label_to_id = None
    classifier_num_classes = 0
    classifier_hidden_dim = classifier_cfg.get("hidden_dim")
    classifier_loss_weight = classifier_cfg.get("loss_weight", 0.0)
    if classifier_enabled:
        label_values = classifier_cfg.get("labels") or build_guidance_label_values(train_raw, guidance_attribute)
        classifier_cfg["labels"] = label_values
        guidance_label_to_id = {value: index for index, value in enumerate(label_values)}
        classifier_num_classes = len(label_values)
        print(
            f"Training attribute classifier for '{guidance_attribute}' with {classifier_num_classes} labels"
        )

    ema_cfg = cfg["diffusion"].get("ema", {})
    ema_enabled = ema_cfg.get("enabled", False)
    ema_decay = ema_cfg.get("decay", 0.999)

    def encode(ex):
        return encode_diffusion_example(
            ex,
            tok,
            prefix_len,
            target_len,
            guidance_attribute=guidance_attribute,
            guidance_label_to_id=guidance_label_to_id,
        )

    train_ds = train_raw.map(encode, remove_columns=["mr_text", "slots", "reference"])
    val_ds = val_raw.map(encode, remove_columns=["mr_text", "slots", "reference"])

    model = DiffusionLM(
        vocab_size=tok.vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        max_length=max_len,
        num_timesteps=cfg["diffusion"]["num_timesteps"],
        noise_schedule=cfg["diffusion"].get("noise_schedule", "sqrt"),
        embedding_init=embedding_init,
        self_conditioning=cfg["diffusion"].get("self_conditioning", False),
        classifier_num_classes=classifier_num_classes,
        classifier_hidden_dim=classifier_hidden_dim,
        classifier_loss_weight=classifier_loss_weight,
        mse_lambda=cfg["diffusion"].get("mse_lambda", 1.0),
        pad_token_id=tok.pad_token_id,
    ).to(device)
    ema_model = build_ema_model(model) if ema_enabled else None

    loader = DataLoader(
        train_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=True,
        collate_fn=collate_diffusion_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg["data"]["batch_size"],
        shuffle=False,
        collate_fn=collate_diffusion_batch,
    )
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_steps = len(loader) * cfg["train"]["epochs"]
    sched = get_linear_schedule_with_warmup(optim, cfg["train"]["warmup_steps"], total_steps)

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir)
    best_val = math.inf

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            target_mask = batch["target_mask"].to(device)
            noise_mask = batch.get("noise_mask")
            if noise_mask is not None:
                noise_mask = noise_mask.to(device)
            guidance_labels = batch.get("guidance_label")
            if guidance_labels is not None:
                guidance_labels = guidance_labels.to(device)
            loss = model(
                input_ids,
                target_mask=target_mask,
                noise_mask=noise_mask,
                guidance_labels=guidance_labels,
            )
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            if ema_model is not None:
                update_ema_model(ema_model, model, ema_decay)
            running += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        train_loss = running / len(loader)
        eval_model = ema_model if ema_model is not None else model
        val_loss = evaluate_diffusion(eval_model, val_loader, device)
        print(f"epoch {epoch} done | train loss {train_loss:.4f} | val loss {val_loss:.4f}")
        payload = {
            "model": (ema_model if ema_model is not None else model).state_dict(),
            "config": cfg,
            "epoch": epoch,
            "val_loss": val_loss,
        }
        if ema_model is not None:
            payload["model_raw"] = model.state_dict()
            payload["ema_decay"] = ema_decay
        torch.save(payload, out_dir / f"checkpoint_epoch{epoch}.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(payload, out_dir / "checkpoint_best.pt")


DISPATCH = {"gpt2": train_gpt2, "t5": train_t5, "diffusion_lm": train_diffusion_lm}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    Path(cfg["train"]["output_dir"]).mkdir(parents=True, exist_ok=True)
    DISPATCH[cfg["model"]["type"]](cfg)


if __name__ == "__main__":
    main()
