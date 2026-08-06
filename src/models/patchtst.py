"""PatchTST (Nie et al., ICLR 2023) — patch-based transformer for time series.

Splits each channel's time series into overlapping patches treated as
tokens, then applies a standard Transformer encoder per channel.

Reference implementation: https://github.com/yuqinie98/PatchTST
We re-implement a minimal version here to avoid a heavy dependency.
"""
from __future__ import annotations

import copy

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel
from ._masking import resample_valid
from .registry import register


class _PatchEmbed(nn.Module):
    def __init__(self, patch_len: int, stride: int, d_model: int):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.proj = nn.Linear(patch_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, T) -> patches (B, C, N, patch_len) -> (B, C, N, d_model)
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        return self.proj(x)


class _PatchTSTNet(nn.Module):
    def __init__(self, n_channels: int, seq_len: int,
                 patch_len: int, stride: int,
                 d_model: int, n_heads: int, n_layers: int, dropout: float,
                 n_static: int = 0):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        # Adaptive pool brings any batch's variable-length time axis to the
        # fixed seq_len the patch projection / encoder were built for.
        self.adapt = nn.AdaptiveAvgPool1d(seq_len)
        self.embed = _PatchEmbed(patch_len, stride, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        n_patches = (seq_len - patch_len) // stride + 1
        # LayerNorm operates on d_model (last dim of encoded tokens), then we
        # flatten and project to scalar.
        self.norm = nn.LayerNorm(d_model)
        # Static covariates (soil, terrain, crop identity) enter here, at the
        # head, rather than being broadcast across the time axis. Broadcasting
        # would repeat each value ~220 times and drown it in the sequence
        # features; a dedicated head term keeps them on equal footing.
        # These were previously DROPPED ENTIRELY — the model built its input
        # as cat([x_past, x_future]) and never referenced x_static, so soil,
        # terrain, landkreis_id and the crop one-hots reached nothing.
        # Static covariates are OFF by default for this architecture.
        #
        # The target is an anomaly: yield minus that (district, crop)'s own
        # training mean. Every static feature — soil, terrain, landkreis_id,
        # crop one-hots — is CONSTANT for a given (district, crop) across all
        # years, so it cannot explain year-to-year deviation. Injected
        # additively at the head it contributes a per-series constant, which
        # against an anomaly target is pure variance.
        #
        # Measured, wiring statics in cost: dlinear -0.174, itransformer
        # -0.046, lstm -0.028, tft -0.089 mean R². XGBoost was unchanged to
        # 3 decimals because tree splits capture the interactions that an
        # additive head cannot.
        #
        # Statics could still help by MODULATING the time-varying pathway
        # (sandy soil making a dry year worse). That needs multiplicative
        # gating, not addition — left as future work. Set use_static_head:
        # true in the model config to re-enable the additive path.
        self.n_static = n_static
        self.proj = nn.Linear(d_model * n_patches * n_channels + n_static, 1)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None,
                x_static: torch.Tensor | None = None) -> torch.Tensor:
        # Resample each sample's VALID span to seq_len. Feeding the padded
        # tensor put 20 of 54 patches (37 %) of a summer_barley sample on pure
        # padding — every one projecting to the same constant token — and let
        # real patches attend to them.
        xc = resample_valid(x, mask, self.adapt)
        patches = self.embed(xc)                     # (B, C, N, D)
        B, C, N, D = patches.shape
        tokens = patches.reshape(B * C, N, D)
        encoded = self.encoder(tokens)               # (B*C, N, D)
        encoded = self.norm(encoded)                 # LayerNorm on last dim D
        encoded = encoded.reshape(B, C * N * D)
        if self.n_static and x_static is not None:
            encoded = torch.cat([encoded, x_static], dim=-1)
        return self.proj(encoded).squeeze(-1)


@register
class PatchTSTModel(BaseModel):
    name = "patchtst"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.net: _PatchTSTNet | None = None
        self.device = "cpu"

    def fit(self, train_loader, val_loader=None, *,
            max_epochs: int = 100, lr: float | None = None,
            patience: int = 15, device: str = "cpu",
            max_minutes: float | None = None):
        import time
        self.device = device
        first = next(iter(train_loader))
        n_static = (first["x_static"].shape[-1]
                     if self.config.get("use_static_head", False) else 0)
        T = first["x_past"].shape[1]
        C = first["x_past"].shape[-1] + first["x_future"].shape[-1]
        c = self.config
        self.net = _PatchTSTNet(
            n_channels=C, seq_len=T, n_static=n_static,
            patch_len=c["patch_len"], stride=c["stride"],
            d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_layers"], dropout=c["dropout"],
        ).to(device)
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=lr or c["learning_rate"])
        loss_fn = nn.MSELoss()
        deadline = time.time() + max_minutes * 60 if max_minutes else None
        best, bad = float("inf"), 0
        # Keep the best-validation weights. Early stopping only breaks out of
        # the loop, so without this the checkpoint saved is the FINAL epoch —
        # `patience` epochs past the optimum, i.e. a deliberately over-fit model.
        best_state = None
        for _ in range(max_epochs):
            self.net.train()
            for batch in train_loader:
                x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1).to(device)
                msk = batch.get("mask")
                msk = msk.to(device) if msk is not None else None
                xs = batch["x_static"].to(device)
                y = batch["y"].to(device)
                pred = self.net(x, msk, xs)
                loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()

            if deadline and time.time() > deadline:
                break
            if val_loader is None:
                continue
            val = self._eval(val_loader, loss_fn)
            if val < best:
                best, bad = val, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _eval(self, loader, loss_fn):
        self.net.eval()
        total, n = 0.0, 0
        for batch in loader:
            x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1).to(self.device)
            msk = batch.get("mask")
            msk = msk.to(self.device) if msk is not None else None
            xs = batch["x_static"].to(self.device)
            y = batch["y"].to(self.device)
            total += loss_fn(self.net(x, msk, xs), y).item() * len(y)
            n += len(y)
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        self.net.eval()
        out = []
        for batch in loader:
            x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1).to(self.device)
            msk = batch.get("mask")
            msk = msk.to(self.device) if msk is not None else None
            xs = batch["x_static"].to(self.device)
            out.append(self.net(x, msk, xs).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        torch.save({
            "n_static": getattr(self.net, "n_static", 0),
            "state_dict": self.net.state_dict(),
            "config": self.config,
            "seq_len": self.net.seq_len,
            "n_channels": self.net.n_channels,
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.config = ckpt["config"]
        c = self.config
        self.net = _PatchTSTNet(
            n_channels=ckpt["n_channels"], seq_len=ckpt["seq_len"],
            n_static=ckpt.get("n_static", 0),
            patch_len=c["patch_len"], stride=c["stride"],
            d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_layers"], dropout=c["dropout"],
        )
        self.net.load_state_dict(ckpt["state_dict"])
        return self
