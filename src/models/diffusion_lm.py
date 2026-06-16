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
        embedding_init: torch.Tensor | None = None,
        self_conditioning: bool = False,
        classifier_num_classes: int = 0,
        classifier_hidden_dim: int | None = None,
        classifier_loss_weight: float = 0.0,
        mse_lambda: float = 1.0,
        pad_token_id: int | None = None,
        prediction_type: str = "x0",
    ):
        super().__init__()
        if prediction_type not in ("x0", "eps"):
            raise ValueError(f"prediction_type must be 'x0' or 'eps', got {prediction_type!r}")
        self.self_conditioning = self_conditioning
        self.classifier_loss_weight = classifier_loss_weight
        self.mse_lambda = mse_lambda
        self.pad_token_id = pad_token_id
        # "x0": denoiser predicts the clean embedding (Diffusion-LM default).
        # "eps": denoiser predicts the noise (classic DDPM parameterization);
        #        x0 is then derived as (x_t - sqrt(1-abar)*eps) / sqrt(abar).
        self.prediction_type = prediction_type
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        if embedding_init is not None:
            if embedding_init.shape != self.embedding.weight.shape:
                raise ValueError(
                    f"embedding_init shape {tuple(embedding_init.shape)} does not match "
                    f"embedding table shape {tuple(self.embedding.weight.shape)}"
                )
            with torch.no_grad():
                self.embedding.weight.copy_(embedding_init)
        self.pos_embed = nn.Embedding(max_length, hidden_dim)
        self.in_proj = nn.Linear(embedding_dim, hidden_dim)
        self.self_cond_proj = nn.Linear(embedding_dim, hidden_dim) if self_conditioning else None
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
        self.attribute_classifier = None
        if classifier_num_classes > 0:
            classifier_hidden_dim = classifier_hidden_dim or hidden_dim
            self.attribute_classifier = nn.Sequential(
                nn.Linear(embedding_dim, classifier_hidden_dim),
                nn.SiLU(),
                nn.Linear(classifier_hidden_dim, classifier_num_classes),
            )
        self.schedule = NoiseSchedule(num_timesteps, kind=noise_schedule)

    def predict_x0(
        self,
        x_t: torch.Tensor,
        t: torch.Tensor,
        self_cond: torch.Tensor | None = None,
        pad_mask: torch.BoolTensor | None = None,
    ) -> torch.Tensor:
        B, L, _ = x_t.shape
        h = self.in_proj(x_t)
        if self.self_cond_proj is not None:
            if self_cond is None:
                self_cond = torch.zeros_like(x_t)
            h = h + self.self_cond_proj(self_cond)
        positions = torch.arange(L, device=x_t.device).unsqueeze(0).expand(B, L)
        h = h + self.pos_embed(positions) + self.time_embed(t).unsqueeze(1)
        transformer_pad_mask = pad_mask
        if transformer_pad_mask is not None and x_t.device.type == "mps":
            transformer_pad_mask = None
        h = self.transformer(h, src_key_padding_mask=transformer_pad_mask)
        return self.out_proj(h)

    def _to_x0(self, model_out: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Map the raw denoiser output to a clean-embedding (x0) estimate.

        Under x0-prediction this is the identity. Under eps-prediction the
        network output is the noise estimate and x0 is recovered by inverting
        the forward process: x0 = (x_t - sqrt(1-abar_t) * eps) / sqrt(abar_t).
        """
        if self.prediction_type == "x0":
            return model_out
        ab = self.schedule.alpha_bar.to(x_t.device)[t].view(-1, 1, 1)
        return (x_t - torch.sqrt((1.0 - ab).clamp_min(1e-6)) * model_out) / torch.sqrt(ab.clamp_min(1e-6))

    def predict_guidance_logits(
        self,
        x_t: torch.Tensor,
        target_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.attribute_classifier is None:
            raise RuntimeError("predict_guidance_logits called without an attribute classifier.")
        if target_mask is None:
            target_mask = torch.ones(x_t.size()[:2], device=x_t.device, dtype=torch.float)
        m = target_mask.unsqueeze(-1).float()
        pooled = (x_t * m).sum(dim=1) / m.sum(dim=1).clamp_min(1.0)
        return self.attribute_classifier(pooled)

    def forward(
        self,
        input_ids: torch.Tensor,
        target_mask: torch.Tensor | None = None,
        noise_mask: torch.Tensor | None = None,
        guidance_labels: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute loss.

        ``target_mask`` controls the LOSS: 1 on positions that contribute to
        MSE/CE (typically the real reference tokens), 0 elsewhere.
        ``noise_mask`` controls the FORWARD process: 1 on positions that get
        noised (the entire target window, including trailing pads), 0 on
        prefix positions which stay clean. If ``noise_mask`` is None, we
        fall back to ``target_mask`` (legacy behavior, equivalent to the
        single-mask design).
        """
        x0 = self.embedding(input_ids)
        if target_mask is None:
            target_mask = torch.ones_like(input_ids, dtype=torch.float)
        if noise_mask is None:
            noise_mask = target_mask
        loss_m = target_mask.unsqueeze(-1).float()
        noise_m = noise_mask.unsqueeze(-1).float()

        # Prefix positions that hold PAD tokens carry no MR information and
        # should be excluded from Transformer attention.
        pad_mask: torch.BoolTensor | None = None
        if self.pad_token_id is not None:
            prefix_pad = (noise_mask == 0) & (input_ids == self.pad_token_id)
            if prefix_pad.any():
                pad_mask = prefix_pad

        t = torch.randint(0, self.schedule.T, (x0.size(0),), device=x0.device)
        noise = torch.randn_like(x0)
        x_t_target = self.schedule.q_sample(x0, t, noise)
        # Clamp prefix positions to clean embeddings; noise everything inside
        # the target window (real tokens + trailing pads).
        x_t = noise_m * x_t_target + (1.0 - noise_m) * x0

        self_cond = None
        if self.self_conditioning and torch.rand((), device=x0.device) < 0.5:
            with torch.no_grad():
                sc_out = self.predict_x0(x_t, t, pad_mask=pad_mask)
                sc_x0 = self._to_x0(sc_out, x_t, t)
                # The self-conditioning signal is always the x0 estimate,
                # regardless of parameterization.
                self_cond = noise_m * sc_x0 + (1.0 - noise_m) * x0

        model_out = self.predict_x0(x_t, t, self_cond=self_cond, pad_mask=pad_mask)
        x0_hat = self._to_x0(model_out, x_t, t)

        # Detach x0 when used as the MSE regression *target* so that the
        # embedding table is not pulled toward x0_hat through the target side
        # (moving-target instability).  Gradients still flow through the INPUT
        # path (q_sample), which is the intended training signal.
        if self.prediction_type == "eps":
            # Classic DDPM objective: regress the noise directly. Noise has
            # unit variance, so this term is O(1) per dim (use mse_lambda ~1).
            mse = ((model_out - noise.detach()) ** 2 * loss_m).sum() / loss_m.sum().clamp_min(1.0) / x0.size(-1)
        else:
            x0_target = x0.detach()
            mse = ((x0_hat - x0_target) ** 2 * loss_m).sum() / loss_m.sum().clamp_min(1.0) / x0.size(-1)
        logits = self.lm_head(x0_hat)
        ce = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction="none",
        )
        ce = (ce * target_mask.view(-1).float()).sum() / target_mask.sum().clamp_min(1.0)
        # mse_lambda rescales the denoising term so it is not swamped by CE.
        # With BERT 768-d embeddings the raw MSE is ~13 000x smaller than CE;
        # mse_lambda brings them to a comparable magnitude.
        total = self.mse_lambda * mse + ce
        if self.attribute_classifier is not None and guidance_labels is not None:
            # Pool over noise_mask: the classifier sees the full noisy
            # target window, which matches the inference configuration.
            cls_logits = self.predict_guidance_logits(x_t, noise_mask)
            cls_loss = F.cross_entropy(cls_logits, guidance_labels)
            total = total + self.classifier_loss_weight * cls_loss
        return total

    @torch.no_grad()
    def sample(
        self,
        length: int,
        prefix_ids: torch.Tensor | None = None,
        ddim_steps: int = 200,
        rounding_interval: int | None = None,
        guidance_labels: torch.Tensor | None = None,
        guidance_scale: float = 0.0,
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

        alpha_bar = self.schedule.alpha_bar.to(device)
        sample_target_mask = torch.cat(
            [
                torch.zeros(B, P, device=device, dtype=torch.float),
                torch.ones(B, length - P, device=device, dtype=torch.float),
            ],
            dim=1,
        )
        timesteps = torch.linspace(
            self.schedule.T - 1, 0, ddim_steps, dtype=torch.long, device=device
        )
        # Build prefix-PAD mask once: True = position is ignored by attention.
        pad_mask: torch.BoolTensor | None = None
        if self.pad_token_id is not None and prefix_ids is not None:
            prefix_is_pad = (prefix_ids.to(device) == self.pad_token_id)
            if prefix_is_pad.any():
                target_false = torch.zeros(B, length - P, dtype=torch.bool, device=device)
                pad_mask = torch.cat([prefix_is_pad, target_false], dim=1)
        self_cond = None
        for index, t in enumerate(timesteps):
            if (
                self.attribute_classifier is not None
                and guidance_labels is not None
                and guidance_scale > 0.0
            ):
                with torch.enable_grad():
                    x_guided = x.detach().clone().requires_grad_(True)
                    guided_logits = self.predict_guidance_logits(x_guided, sample_target_mask)
                    guided_scores = F.log_softmax(guided_logits, dim=-1)
                    selected = guided_scores.gather(1, guidance_labels.view(-1, 1)).sum()
                    grad = torch.autograd.grad(selected, x_guided)[0]
                grad[:, :P] = 0.0
                grad_norm = grad.flatten(1).norm(dim=1).view(B, 1, 1).clamp_min(1e-6)
                x = x + guidance_scale * grad / grad_norm
                if prefix_emb is not None:
                    x[:, :P] = prefix_emb

            t_batch = t.expand(B)
            model_out = self.predict_x0(x, t_batch, self_cond=self_cond, pad_mask=pad_mask)
            x0_hat = self._to_x0(model_out, x, t_batch)
            # Clamp prefix every step so the condition cannot drift.
            if prefix_emb is not None:
                x0_hat[:, :P] = prefix_emb

            if rounding_interval and rounding_interval > 0 and (
                index == len(timesteps) - 1 or (index + 1) % rounding_interval == 0
            ):
                rounded_ids = self.lm_head(x0_hat).argmax(dim=-1)
                x0_hat = self.embedding(rounded_ids)
                if prefix_emb is not None:
                    x0_hat[:, :P] = prefix_emb

            if self.self_conditioning:
                self_cond = x0_hat.detach()

            if index == len(timesteps) - 1:
                x = x0_hat
                continue

            next_t = timesteps[index + 1]
            ab_t = alpha_bar[t].view(1, 1, 1)
            ab_next = alpha_bar[next_t].view(1, 1, 1)
            denom = torch.sqrt((1.0 - ab_t).clamp_min(1e-6))
            eps_hat = (x - torch.sqrt(ab_t) * x0_hat) / denom
            x = torch.sqrt(ab_next) * x0_hat + torch.sqrt((1.0 - ab_next).clamp_min(1e-6)) * eps_hat
            if prefix_emb is not None:
                x[:, :P] = prefix_emb

        logits = self.lm_head(x)
        ids = logits.argmax(dim=-1)
        if prefix_emb is not None:
            ids[:, :P] = prefix_ids.to(device)
        return ids
