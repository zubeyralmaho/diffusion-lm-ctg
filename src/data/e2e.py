"""E2E NLG dataset loader.

Loads via HuggingFace `datasets` (`e2e_nlg_cleaned`) and flattens meaning
representations (MR) into key=value strings suitable for seq2seq and decoder-only
models. The slot dict is preserved for controllability evaluation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from datasets import load_dataset


SLOT_RE = re.compile(r"(\w+)\[([^\]]+)\]")


@dataclass
class E2EExample:
    mr_text: str
    slots: dict[str, str]
    reference: str


def parse_mr(mr: str) -> dict[str, str]:
    return {k: v.strip() for k, v in SLOT_RE.findall(mr)}


def slots_to_prompt(slots: dict[str, str]) -> str:
    return " | ".join(f"{k}: {v}" for k, v in slots.items())


def load_e2e(split: str = "train"):
    ds = load_dataset("e2e_nlg_cleaned", split=split)

    def _map(row):
        slots = parse_mr(row["meaning_representation"])
        return {
            "mr_text": slots_to_prompt(slots),
            "slots": slots,
            "reference": row["target"],
        }

    return ds.map(_map, remove_columns=ds.column_names)
