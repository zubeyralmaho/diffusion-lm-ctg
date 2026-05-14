"""Evaluation: fluency (BLEU, perplexity), diversity (distinct-n, self-BLEU),
controllability (slot-fill accuracy on E2E MR slots)."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import sacrebleu
from rouge_score import rouge_scorer


def distinct_n(texts: list[str], n: int = 2) -> float:
    ngrams = Counter()
    total = 0
    for t in texts:
        toks = t.split()
        for i in range(len(toks) - n + 1):
            ngrams[tuple(toks[i : i + n])] += 1
            total += 1
    return len(ngrams) / max(total, 1)


def slot_accuracy(records: list[dict]) -> float:
    """Fraction of slot values that appear verbatim in the generated text."""
    correct = total = 0
    for r in records:
        pred = r["prediction"].lower()
        for _slot, value in r.get("slots", {}).items():
            total += 1
            if value.lower() in pred:
                correct += 1
    return correct / max(total, 1)


def compute_metrics(jsonl_path: str) -> dict:
    records = [json.loads(line) for line in open(jsonl_path)]
    preds = [r["prediction"] for r in records]
    refs = [r["reference"] for r in records]

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rouge_l = sum(scorer.score(r, p)["rougeL"].fmeasure for p, r in zip(preds, refs)) / len(preds)

    return {
        "bleu": round(bleu, 2),
        "rougeL": round(rouge_l * 100, 2),
        "distinct_1": round(distinct_n(preds, 1), 4),
        "distinct_2": round(distinct_n(preds, 2), 4),
        "slot_accuracy": round(slot_accuracy(records), 4),
        "n_examples": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True, help="JSONL from generate.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics = compute_metrics(args.predictions)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
