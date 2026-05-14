"""Autoregressive baselines: GPT-2 (decoder-only) and T5 (encoder-decoder)."""
from __future__ import annotations

from transformers import (
    AutoTokenizer,
    GPT2LMHeadModel,
    T5ForConditionalGeneration,
)


def build_gpt2(pretrained: str = "gpt2"):
    tok = AutoTokenizer.from_pretrained(pretrained)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = GPT2LMHeadModel.from_pretrained(pretrained)
    model.config.pad_token_id = tok.pad_token_id
    return model, tok


def build_t5(pretrained: str = "t5-small"):
    tok = AutoTokenizer.from_pretrained(pretrained)
    model = T5ForConditionalGeneration.from_pretrained(pretrained)
    return model, tok
