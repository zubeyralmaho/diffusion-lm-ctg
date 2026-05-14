"""Diffusion-LM (Li et al., NeurIPS 2022) with prefix conditioning.

Continuous-space text diffusion: tokens are mapped to learned embeddings, noise
is added in embedding space, and a Transformer denoiser predicts the clean
embeddings (x0-prediction). A rounding step projects denoised vectors back to
the vocabulary at inference.

**Controllability via prefix conditioning**: the meaning representation (MR)
occupies the first `prefix_len` positions and is kept clean (no noise added,
not included in the loss). Only target positions are diffused. At sampling
time, the prefix embeddings are clamped to the MR throughout denoising — the
model is forced to produce a target consistent with the given attributes.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class NoiseSchedule:
    """Diffusion noise schedule. `kind` ∈ {"sqrt", "cosine", "linear"}.

    - sqrt   : Diffusion-LM (Li et al. 2022) — alpha_bar = 1 - sqrt(t/T + s)
    - cosine : Nichol & Dhariwal (2021) improved schedule
    - linear : DDPM original
    """

    def __init__(self, num_timesteps: int = 2000, kind: str = "sqrt", s: float = 1e-4):
        self.T = num_timesteps
        self.kind = kind
        t = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        if kind == "sqrt":
            alpha_bar = 1.0 - torch.sqrt(t + s)
            alpha_bar = alpha_bar / alpha_bar[0]
        elif kind == "cosine":
            f = torch.cos((t + s) / (1 + s) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
        elif kind == "linear":
            beta = torch.linspace(1e-4, 0.02, num_timesteps + 1, dtype=torch.float64)
            alpha = 1.0 - beta
            alpha_bar = torch.cumprod(alpha, dim=0)
            alpha_bar = alpha_bar / alpha_bar[0]
        else:
            raise ValueError(f"unknown schedule kind: {kind}")
        self.alpha_bar = alpha_bar.clamp(min=1e-6, max=1.0).float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


# Back-compat alias.
SqrtSchedule = NoiseSchedule


class SinusoidalTimeEmbed(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class DiffusionLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 64,
        num_timesteps: int = 2000,
        noise_schedule: str = "sqrt",
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embed = nn.Embedding(max_length, hidden_dim)
        self.in_proj = nn.Linear(embedding_dim, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight
        self.schedule = NoiseSchedule(num_timesteps, kind=noise_schedule)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_t.shape
        h = self.in_proj(x_t)
        positions = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, L)
        h = h + self.pos_embed(positions) + self.time_embed(t).unsqueeze(1)
        h = self.transformer(h)
        return self.out_proj(h)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss. `target_mask` is 1 for positions to diffuse / score and
        0 for prefix (MR) positions, which are kept clean and excluded from
        the loss. If None, the whole sequence is treated as target (uncond).
        """
        x0 = self.embedding(input_ids)
        if target_mask is None:
            target_mask = torch.ones_like(input_ids, dtype=torch.float)
        m = target_mask.unsqueeze(-1).float()

        t = torch.randint(0, self.schedule.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t_target = self.schedule.q_sample(x0, t, noise)
        # Clamp prefix positions to clean embeddings.
        x_t = m * x_t_target + (1.0 - m) * x0

        x0_hat = self.predict_x0(x_t, t)

        # Restrict both losses to target positions.
        mse = ((x0_hat - x0) ** 2 * m).sum() / m.sum().clamp_min(1.0) / x0.size(-1)
        logits = self.lm_head(x0_hat)
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction="none",
        )
        ce = (ce * target_mask.view(-1).float()).sum() / target_mask.sum().clamp_min(1.0)
        return mse + ce

    @torch.no_grad()
    def sample(
        self,
        length: int,
        prefix_ids: torch.Tensor | None = None,
        ddim_steps: int = 200,
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate token ids. If `prefix_ids` is given (shape [B, P]), the
        first P positions are clamped to the prefix embeddings throughout
        denoising and the remaining `length - P` positions are the target.
        """
        device = next(self.parameters()).device
        if prefix_ids is not None:
            B, P = prefix_ids.shape
            assert P < length, "prefix must be shorter than total length"
            prefix_emb = self.embedding(prefix_ids.to(device))
        else:
            B = batch_size
            P = 0
            prefix_emb = None

        x = torch.randn(B, length, self.embedding.embedding_dim, device=device)
        if prefix_emb is not None:
            x[:, :P] = prefix_emb

        timesteps = torch.linspace(
            self.schedule.T - 1, 0, ddim_steps, dtype=torch.long, device=device
        )
        for t in timesteps:
            t_batch = t.expand(B)
            x0_hat = self.predict_x0(x, t_batch)
            # Clamp prefix every step so the condition cannot drift.
            if prefix_emb is not None:
                x0_hat[:, :P] = prefix_emb
            x = x0_hat

        logits = self.lm_head(x)
        ids = logits.argmax(dim=-1)
        if prefix_emb is not None:
            ids[:, :P] = prefix_ids.to(device)
        return ids
