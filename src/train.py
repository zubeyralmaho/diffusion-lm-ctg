"""Unified training entrypoint. Dispatches by config.model.type."""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import (
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
        logging_steps=100,
        report_to=[],
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    ).train()
    tok.save_pretrained(cfg["train"]["output_dir"])


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
        predict_with_generate=True,
        logging_steps=100,
        report_to=[],
    )
    Seq2SeqTrainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=DataCollatorForSeq2Seq(tok, model=model),
    ).train()
    tok.save_pretrained(cfg["train"]["output_dir"])


# ---------- Diffusion-LM ----------

def train_diffusion_lm(cfg: dict) -> None:
    from transformers import AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(cfg["model"]["tokenizer"])
    max_len = cfg["data"]["max_length"]

    def encode(ex):
        text = f"{ex['mr_text']} [SEP] {ex['reference']}"
        return tok(text, truncation=True, max_length=max_len, padding="max_length")

    train_ds = load_e2e("train").map(encode, remove_columns=["mr_text", "slots", "reference"])
    train_ds.set_format("torch", columns=["input_ids"])

    model = DiffusionLM(
        vocab_size=tok.vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_dim=cfg["model"]["hidden_dim"],
        num_layers=cfg["model"]["num_layers"],
        num_heads=cfg["model"]["num_heads"],
        max_length=max_len,
        num_timesteps=cfg["diffusion"]["num_timesteps"],
    ).to(device)

    loader = DataLoader(train_ds, batch_size=cfg["data"]["batch_size"], shuffle=True)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"], weight_decay=cfg["train"]["weight_decay"])
    total_steps = len(loader) * cfg["train"]["epochs"]
    sched = get_linear_schedule_with_warmup(optim, cfg["train"]["warmup_steps"], total_steps)

    out_dir = Path(cfg["train"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    tok.save_pretrained(out_dir)

    step = 0
    for epoch in range(cfg["train"]["epochs"]):
        model.train()
        running = 0.0
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            loss = model(input_ids)
            optim.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            sched.step()
            running += loss.item()
            step += 1
            if step % 100 == 0:
                print(f"epoch {epoch} step {step} loss {loss.item():.4f}")
        print(f"epoch {epoch} done | avg loss {running / len(loader):.4f}")
        torch.save(
            {"model": model.state_dict(), "config": cfg},
            out_dir / f"checkpoint_epoch{epoch}.pt",
        )


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
