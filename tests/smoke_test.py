"""End-to-end smoke test — no GPU, no HF model downloads, ~10 seconds.

Verifies that:
1. E2E MR parser works.
2. Diffusion-LM forward + sample run without shape errors.
3. Noise schedules (sqrt/cosine/linear) all produce monotonic alpha_bar.
4. evaluate.compute_metrics handles a fake predictions JSONL.

Run:
    python -m tests.smoke_test

Exits nonzero on any failure.
"""
from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import torch

# Make `src` importable.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.e2e import parse_mr, slots_to_prompt  # noqa: E402
from src.evaluate import compute_metrics  # noqa: E402
from src.generate import find_diffusion_checkpoint  # noqa: E402
from src.train import encode_diffusion_example  # noqa: E402
from src.models.diffusion_lm import DiffusionLM, NoiseSchedule  # noqa: E402


def test_mr_parser() -> None:
    mr = "name[The Eagle], eatType[restaurant], food[Italian], priceRange[high]"
    slots = parse_mr(mr)
    assert slots == {
        "name": "The Eagle",
        "eatType": "restaurant",
        "food": "Italian",
        "priceRange": "high",
    }, slots
    prompt = slots_to_prompt(slots)
    assert "name: The Eagle" in prompt
    print("  ✓ MR parser")


def test_noise_schedules() -> None:
    for kind in ("sqrt", "cosine", "linear"):
        s = NoiseSchedule(num_timesteps=100, kind=kind)
        ab = s.alpha_bar
        assert ab[0] >= ab[-1], f"{kind}: alpha_bar must be non-increasing"
        assert (ab > 0).all() and (ab <= 1.0 + 1e-5).all(), f"{kind}: out of [0,1]"
    print("  ✓ noise schedules (sqrt, cosine, linear)")


def test_diffusion_lm_forward_and_sample() -> None:
    vocab, B, L, P = 200, 4, 16, 4
    model = DiffusionLM(
        vocab_size=vocab,
        embedding_dim=32,
        hidden_dim=64,
        num_layers=2,
        num_heads=4,
        max_length=L,
        num_timesteps=100,
        self_conditioning=True,
        classifier_num_classes=3,
        classifier_hidden_dim=32,
        classifier_loss_weight=0.2,
    )

    input_ids = torch.randint(0, vocab, (B, L))
    target_mask = torch.cat(
        [torch.zeros(B, P, dtype=torch.long), torch.ones(B, L - P, dtype=torch.long)], dim=1
    )
    guidance_labels = torch.tensor([0, 1, 2, 1])
    loss = model(input_ids, target_mask=target_mask, guidance_labels=guidance_labels)
    assert torch.isfinite(loss), f"loss not finite: {loss}"
    loss.backward()  # gradients must flow
    print(f"  ✓ Diffusion-LM forward + backward (loss={loss.item():.3f})")

    prefix = torch.randint(0, vocab, (2, P))
    out = model.sample(
        length=L,
        prefix_ids=prefix,
        ddim_steps=5,
        guidance_labels=torch.tensor([1, 2]),
        guidance_scale=0.1,
    )
    assert out.shape == (2, L), out.shape
    assert (out[:, :P] == prefix).all(), "prefix not clamped in output"
    print(f"  ✓ Diffusion-LM sample with self-conditioning/guidance (shape={tuple(out.shape)})")


def test_compute_metrics() -> None:
    records = [
        {
            "mr_text": "name: The Eagle | food: Italian",
            "slots": {"name": "The Eagle", "food": "Italian", "area": None},
            "prediction": "The Eagle is an Italian restaurant.",
            "reference": "The Eagle serves Italian food.",
        },
        {
            "mr_text": "name: Loch Fyne | food: Japanese",
            "slots": {"name": "Loch Fyne", "food": "Japanese", "near": None},
            "prediction": "Loch Fyne offers Japanese cuisine.",
            "reference": "Loch Fyne is a Japanese restaurant.",
        },
    ]
    with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
        path = f.name

    metrics = compute_metrics(path)
    for key in ("bleu", "rougeL", "distinct_1", "distinct_2", "slot_accuracy"):
        assert key in metrics, f"missing metric: {key}"
    assert metrics["slot_accuracy"] == 1.0, metrics
    assert metrics["n_examples"] == 2
    print(f"  ✓ evaluate.compute_metrics (BLEU={metrics['bleu']}, slot_acc={metrics['slot_accuracy']})")


def test_diffusion_checkpoint_selection() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        ckpt_dir = Path(tmp)
        for name in ("checkpoint_epoch0.pt", "checkpoint_epoch2.pt", "checkpoint_epoch10.pt"):
            (ckpt_dir / name).touch()
        chosen = find_diffusion_checkpoint(ckpt_dir)
        assert chosen.name == "checkpoint_epoch10.pt", chosen

        best = ckpt_dir / "checkpoint_best.pt"
        best.touch()
        chosen = find_diffusion_checkpoint(ckpt_dir)
        assert chosen.name == "checkpoint_best.pt", chosen
    print("  ✓ diffusion checkpoint selection prefers best, then highest epoch")


class DummyTokenizer:
    pad_token_id = 0
    sep_token_id = 99
    eos_token_id = None

    def __call__(self, text: str, truncation: bool, max_length: int, padding: str | None = None, add_special_tokens: bool = False):
        del truncation, add_special_tokens
        ids = list(range(1, min(len(text.split()), max_length) + 1))
        if padding == "max_length":
            ids = ids + [self.pad_token_id] * (max_length - len(ids))
        return {"input_ids": ids}


def test_diffusion_encoding_target_window() -> None:
    """Target mask spans the entire target window (including trailing pads).

    This is critical for inference: at sampling time the model initializes
    all target positions from pure noise with no clean pad anchors. The CE
    loss on pad positions teaches the model to emit pad in the trailing
    region; without it, inference produces high-norm vocab noise.
    """
    tok = DummyTokenizer()
    out = encode_diffusion_example(
        {
            "mr_text": "name: Blue Spice | area: city centre",
            "reference": "Blue Spice",
        },
        tok,
        prefix_len=4,
        target_len=6,
    )
    assert out["input_ids"][:4] == [1, 2, 3, 4]
    assert out["input_ids"][4:10] == [1, 2, 99, 0, 0, 0]
    # mask=1 across all 6 target positions, including the 3 trailing pads
    assert out["target_mask"] == [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]
    print("  ✓ diffusion encoding masks the full target window (including pad)")


def main() -> int:
    print("Smoke test:")
    tests = [
        test_mr_parser,
        test_noise_schedules,
        test_diffusion_lm_forward_and_sample,
        test_compute_metrics,
        test_diffusion_checkpoint_selection,
        test_diffusion_encoding_target_window,
    ]
    failed = 0
    for t in tests:
        try:
            t()
        except Exception as e:
            print(f"  ✗ {t.__name__}: {e}")
            failed += 1
    print()
    if failed:
        print(f"FAIL: {failed}/{len(tests)} test(s) failed")
        return 1
    print(f"PASS: {len(tests)}/{len(tests)} tests")
    return 0


if __name__ == "__main__":
    sys.exit(main())
