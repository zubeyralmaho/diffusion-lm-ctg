"""E2E NLG (cleaned) loader.

Loads directly from the tuetschek/e2e-cleaning GitHub repo CSVs, avoiding the
HuggingFace `datasets` script-based loader which was removed in datasets>=3.0.
The CSVs are downloaded once to `data/raw/e2e/` and cached locally.

Columns produced per example:
    mr_text   — flattened "key: value | key: value" prompt
    slots     — dict of slot name -> value (for controllability eval)
    reference — gold target utterance
"""
from __future__ import annotations

import csv
import os
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset


CSV_URLS = {
    "train":      "https://raw.githubusercontent.com/tuetschek/e2e-cleaning/master/cleaned-data/train-fixed.no-ol.csv",
    "validation": "https://raw.githubusercontent.com/tuetschek/e2e-cleaning/master/cleaned-data/devel-fixed.no-ol.csv",
    "test":       "https://raw.githubusercontent.com/tuetschek/e2e-cleaning/master/cleaned-data/test-fixed.csv",
}

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


def _cache_dir() -> Path:
    # data/raw/e2e at the project root.
    here = Path(__file__).resolve()
    root = here.parents[2]
    out = root / "data" / "raw" / "e2e"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _download(split: str) -> Path:
    cache = _cache_dir() / f"{split}.csv"
    if cache.exists() and cache.stat().st_size > 0:
        return cache
    url = CSV_URLS[split]
    print(f"Downloading E2E {split} from {url}")
    urllib.request.urlretrieve(url, cache)
    return cache


def _read_csv(path: Path) -> list[dict]:
    """Read an E2E CSV. Cleaned files use `mr,ref` columns."""
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Tolerate either ('mr','ref') or ('mr','target') headers.
        mr_key = "mr" if "mr" in reader.fieldnames else reader.fieldnames[0]
        ref_key = "ref" if "ref" in reader.fieldnames else ("target" if "target" in reader.fieldnames else reader.fieldnames[1])
        for row in reader:
            mr_text = row[mr_key]
            reference = row[ref_key]
            slots = parse_mr(mr_text)
            rows.append({
                "mr_text": slots_to_prompt(slots),
                "slots": slots,
                "reference": reference,
            })
    return rows


def load_e2e(split: str = "train") -> Dataset:
    if split not in CSV_URLS:
        raise ValueError(f"unknown split: {split}; expected one of {list(CSV_URLS)}")
    path = _download(split)
    rows = _read_csv(path)
    return Dataset.from_list(rows)
