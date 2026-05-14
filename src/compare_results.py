"""Aggregate per-model metric JSON files into a single Markdown table.

Reads every `*.json` under --metrics_dir (recursively) and prints / writes a
sortable comparison. Used after `scripts/run_eval.sh` and
`scripts/run_ablations.sh`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

METRIC_ORDER = ["bleu", "rougeL", "distinct_1", "distinct_2", "slot_accuracy", "n_examples"]


def collect(metrics_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(metrics_dir.rglob("*.json")):
        with open(path) as f:
            m = json.load(f)
        rows.append({"name": path.relative_to(metrics_dir).with_suffix("").as_posix(), **m})
    return rows


def render_markdown(rows: list[dict]) -> str:
    if not rows:
        return "_No metrics found._\n"
    header = ["name"] + METRIC_ORDER
    lines = ["| " + " | ".join(header) + " |",
             "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(k, "")) for k in header) + " |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", default="results/metrics")
    parser.add_argument("--out", default="results/summary.md")
    args = parser.parse_args()

    rows = collect(Path(args.metrics_dir))
    md = render_markdown(rows)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(md)
    print(md)


if __name__ == "__main__":
    main()
