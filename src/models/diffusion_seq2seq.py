"""Cross-attention (encoder-decoder) diffusion baseline.

A genuinely different conditioning mechanism from the prefix-clamping used by
our main Diffusion-LM. Here the meaning representation (MR) is encoded by a
separate Transformer encoder and the diffusion denoiser is a Transformer
*decoder* that cross-attends to that encoded MR at every layer. The target is
its own noised tensor — there is no clamped prefix inside the diffused
sequence.

This is the conditioning style of encoder-decoder / seq2seq diffusion models
(DiffuSeq, Gong et al. 2023; SeqDiffuSeq). It shares the continuous
embedding-space forward process, x0-prediction objective, and rounding step
with our main model, but replaces in-context prefix conditioning with
cross-attention conditioning.

The forward()/sample() signatures match DiffusionLM so the existing training
and generation code paths work with only a constructor-level branch.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.diffusion_lm import NoiseSchedule, SinusoidalTimeEmbed


class CrossAttnDiffusionLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 768,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        max_length: int = 64,
        num_timesteps: int = 2000,
        noise_schedule: str = "sqrt",
        embedding_init: torch.Tensor | None = None,
        self_conditioning: bool = False,
        encoder_layers: int = 3,
        mse_lambda: float = 1.0,
        pad_token_id: int | None = None,
        # Accepted for signature compatibility with DiffusionLM; this baseline
        # does not use classifier guidance.
        classifier_num_classes: int = 0,
        classifier_hidden_dim: int | None = None,
        classifier_loss_weight: float = 0.0,
        prediction_type: str = "x0",
    ):
        super().__init__()
        if prediction_type != "x0":
            raise ValueError("CrossAttnDiffusionLM only supports x0-prediction.")
        self.self_conditioning = self_conditioning
        self.mse_lambda = mse_lambda
        self.pad_token_id = pad_token_id
        self.attribute_classifier = None  # no guidance for this baseline

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_init is not None:
            if embedding_init.shape != self.embedding.weight.shape:
                raise ValueError(
                    f"embedding_init shape {tuple(embedding_init.shape)} != "
                    f"embedding table shape {tuple(self.embedding.weight.shape)}"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_init)

        # --- MR encoder ---
        self.enc_in_proj = nn.Linear(embedding_dim, hidden_dim)
        self.enc_pos_embed = nn.Embedding(max_length, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=encoder_layers)

        # --- Target denoiser (decoder, cross-attends to MR memory) ---
        self.in_proj = nn.Linear(embedding_dim, hidden_dim)
        self.self_cond_proj = nn.Linear(embedding_dim, hidden_dim) if self_conditioning else None
        self.pos_embed = nn.Embedding(max_length, hidden_dim)
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmbed(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        dec_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
            batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=num_layers)
        self.out_proj = nn.Linear(hidden_dim, embedding_dim)
        self.lm_head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.lm_head.weight = self.embedding.weight

        self.schedule = NoiseSchedule(num_timesteps, kind=noise_schedule)

    def encode_mr(self, mr_ids: torch.Tensor) -> tuple[torch.Tensor, torch.BoolTensor | None]:
        B, P = mr_ids.shape
        h = self.enc_in_proj(self.embedding(mr_ids))
        positions = torch.arange(P, device=mr_ids.device).unsqueeze(0).expand(B, P)
        h = h + self.enc_pos_embed(positions)
        pad_mask: torch.BoolTensor | None = None
        if self.pad_token_id is not None:
            pm = mr_ids == self.pad_token_id
            if pm.any():
                pad_mask = pm
        enc_pad = None if (pad_mask is not None and h.device.type == "mps") else pad_mask
        memory = self.encoder(h, src_key_padding_mask=enc_pad)
        return memory, pad_mask

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        memory: torch.Tensor,
        memory_pad_mask: torch.BoolTensor | None = None,
        tgt_pad_mask: torch.BoolTensor | None = None,
        self_cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Ltgt, _ = x_t.shape
        h = self.in_proj(x_t)
        if self.self_cond_proj is not None:
            if self_cond is None:
                self_cond = torch.zeros_like(x_t)
            h = h + self.self_cond_proj(self_cond)
        positions = torch.arange(Ltgt, device=x_t.device).unsqueeze(0).expand(B, Ltgt)
        h = h + self.pos_embed(positions) + self.time_embed(t).unsqueeze(1)
        if x_t.device.type == "mps":  # MPS does not support boolean key-padding masks reliably
            tgt_pad_mask = None
            memory_pad_mask = None
        h = self.decoder(
            tgt=h, memory=memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=memory_pad_mask,
        )
        return self.out_proj(h)

    @staticmethod
    def _prefix_len(noise_mask: torch.Tensor) -> int:
        # noise_mask is [0]*P + [1]*target_len; the prefix is the leading zero run.
        return int((noise_mask[0] == 0).sum().item())

    def forward(
        self,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        guidance_labels: torch.Tensor | None = None,  # ignored; signature compat
    ) -> torch.Tensor:
        if noise_mask is None or target_mask is None:
            raise ValueError("CrossAttnDiffusionLM requires target_mask and noise_mask.")
        P = self._prefix_len(noise_mask)
        mr_ids = input_ids[:, :P]
        tgt_ids = input_ids[:, P:]
        tgt_loss_mask = target_mask[:, P:].unsqueeze(-1).float()

        memory, mr_pad = self.encode_mr(mr_ids)
        tgt_pad = None
        if self.pad_token_id is not None:
            pm = tgt_ids == self.pad_token_id
            if pm.any():
                tgt_pad = pm

        x0 = self.embedding(tgt_ids)
        t = torch.randint(0, self.schedule.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t = self.schedule.q_sample(x0, t, noise)

        self_cond = None
        if self.self_conditioning and torch.rand((), device=x0.device) < 0.5:
            with torch.no_grad():
                self_cond = self.predict_x0(x_t, t, memory, mr_pad, tgt_pad)

        x0_hat = self.predict_x0(x_t, t, memory, mr_pad, tgt_pad, self_cond=self_cond)

        x0_target = x0.detach()
        mse = ((x0_hat - x0_target) ** 2 * tgt_loss_mask).sum() / tgt_loss_mask.sum().clamp_min(1.0) / x0.size(-1)
        logits = self.lm_head(x0_hat)
        ce = F.cross_entropy(logits.view(-1, logits.size(-1)), tgt_ids.reshape(-1), reduction="none")
        ce = (ce * target_mask[:, P:].reshape(-1).float()).sum() / target_mask[:, P:].sum().clamp_min(1.0)
        return self.mse_lambda * mse + ce

    @torch.no_grad()
    def sample(
        self,
        length: int,
        prefix_ids: torch.Tensor | None = None,
        ddim_steps: int = 200,
        rounding_interval: int | None = None,
        guidance_labels: torch.Tensor | None = None,  # ignored
        guidance_scale: float = 0.0,  # ignored
        batch_size: int = 1,
    ) -> torch.Tensor:
        """Generate token ids. Returns [B, length] = [MR prefix ; target], so
        the caller can slice row[P:] exactly as for the prefix model."""
        device = next(self.parameters()).device
        assert prefix_ids is not None, "cross-attention baseline requires an MR prefix"
        prefix_ids = prefix_ids.to(device)
        B, P = prefix_ids.shape
        target_len = length - P
        memory, mr_pad = self.encode_mr(prefix_ids)

        x = torch.randn(B, target_len, self.embedding.embedding_dim, device=device)
        alpha_bar = self.schedule.alpha_bar.to(device)
        timesteps = torch.linspace(self.schedule.T - 1, 0, ddim_steps, dtype=torch.long, device=device)
        self_cond = None
        for index, t in enumerate(timesteps):
            t_batch = t.expand(B)
            x0_hat = self.predict_x0(x, t_batch, memory, mr_pad, None, self_cond=self_cond)
            if rounding_interval and rounding_interval > 0 and (
                index == len(timesteps) - 1 or (index + 1) % rounding_interval == 0
            ):
                rounded = self.lm_head(x0_hat).argmax(dim=-1)
                x0_hat = self.embedding(rounded)
            if self.self_conditioning:
                self_cond = x0_hat.detach()
            if index == len(timesteps) - 1:
                x = x0_hat
                continue
            next_t = timesteps[index + 1]
            ab_t = alpha_bar[t].view(1, 1, 1)
            ab_next = alpha_bar[next_t].view(1, 1, 1)
            eps_hat = (x - torch.sqrt(ab_t) * x0_hat) / torch.sqrt((1.0 - ab_t).clamp_min(1e-6))
            x = torch.sqrt(ab_next) * x0_hat + torch.sqrt((1.0 - ab_next).clamp_min(1e-6)) * eps_hat

        target_ids = self.lm_head(x).argmax(dim=-1)
        return torch.cat([prefix_ids, target_ids], dim=1)
