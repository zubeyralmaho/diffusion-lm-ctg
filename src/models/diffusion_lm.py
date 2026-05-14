"""Diffusion-LM (Li et al., NeurIPS 2022).

Continuous-space text diffusion: tokens are mapped to learned embeddings, noise
is added in embedding space, and a Transformer denoiser predicts the clean
embeddings (x0-prediction). A rounding step projects denoised vectors back to
the vocabulary at inference.

This file is a faithful but minimal skeleton — flesh out forward/sample for
final training runs.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SqrtSchedule:
    """Diffusion-LM's sqrt noise schedule (Section 4.1)."""

    def __init__(self, num_timesteps: int = 2000, s: float = 1e-4):
        self.T = num_timesteps
        t = torch.arange(num_timesteps + 1, dtype=torch.float64) / num_timesteps
        alpha_bar = 1.0 - torch.sqrt(t + s)
        alpha_bar = alpha_bar / alpha_bar[0]
        self.alpha_bar = alpha_bar.float()

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ab = self.alpha_bar.to(x0.device)[t].view(-1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1.0 - ab) * noise


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
        self.schedule = SqrtSchedule(num_timesteps)

    def predict_x0(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        B, L, _ = x_t.shape
        h = self.in_proj(x_t)
        positions = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, L)
        h = h + self.pos_embed(positions) + self.time_embed(t).unsqueeze(1)
        h = self.transformer(h)
        return self.out_proj(h)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x0 = self.embedding(input_ids)
        t = torch.randint(0, self.schedule.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)
        x0_hat = self.predict_x0(x_t, t)
        loss_mse = F.mse_loss(x0_hat, x0)
        logits = self.lm_head(x0_hat)
        loss_ce = F.cross_entropy(logits.view(-1, logits.size(-1)), input_ids.view(-1))
        return loss_mse + loss_ce

    @torch.no_grad()
    def sample(self, batch_size: int, length: int, ddim_steps: int = 200) -> torch.Tensor:
        device = next(self.parameters()).device
        x = torch.randn(batch_size, length, self.embedding.embedding_dim, device=device)
        timesteps = torch.linspace(self.schedule.T - 1, 0, ddim_steps, dtype=torch.long, device=device)
        for t in timesteps:
            t_batch = t.expand(batch_size)
            x = self.predict_x0(x, t_batch)
        logits = self.lm_head(x)
        return logits.argmax(dim=-1)
