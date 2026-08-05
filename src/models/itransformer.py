"""iTransformer (Liu et al., ICLR 2024) — variates as tokens.

Inverts the usual axes: each variable becomes a token, attention runs
across variables (not time). Each variable's whole time series is
linearly embedded into a single D-dimensional token.
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


class _ITransformerNet(nn.Module):
    def __init__(self, n_channels: int, seq_len: int,
                 d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float, n_static: int = 0):
        super().__init__()
        self.seq_len = seq_len
        self.n_channels = n_channels
        self.adapt = nn.AdaptiveAvgPool1d(seq_len)
        self.var_embed = nn.Linear(seq_len, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        # LayerNorm on d_model (last dim of encoded), then flatten + project.
        self.norm = nn.LayerNorm(d_model)
        # Static covariates (soil, terrain, crop identity) enter here, at the
        # head, rather than being broadcast across the time axis. Broadcasting
        # would repeat each value ~220 times and drown it in the sequence
        # features; a dedicated head term keeps them on equal footing.
        # These were previously DROPPED ENTIRELY — the model built its input
        # as cat([x_past, x_future]) and never referenced x_static, so soil,
        # terrain, landkreis_id and the crop one-hots reached nothing.
        self.n_static = n_static
        self.proj = nn.Linear(d_model * n_channels + n_static, 1)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None,
                x_static: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T_var, C) -> (B, C, seq_len), padding excluded
        x = resample_valid(x, mask, self.adapt)
        tokens = self.var_embed(x)                # (B, C, D)
        encoded = self.encoder(tokens)            # (B, C, D)
        encoded = self.norm(encoded)              # LayerNorm on D
        B, C, D = encoded.shape
        flat = encoded.reshape(B, C * D)
        if self.n_static and x_static is not None:
            flat = torch.cat([flat, x_static], dim=-1)
        return self.proj(flat).squeeze(-1)


@register
class ITransformerModel(BaseModel):
    name = "itransformer"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.net: _ITransformerNet | None = None
        self.device = "cpu"

    def fit(self, train_loader, val_loader=None, *,
            max_epochs: int = 100, lr: float | None = None,
            patience: int = 15, device: str = "cpu",
            max_minutes: float | None = None):
        import time
        self.device = device
        first = next(iter(train_loader))
        n_static = first["x_static"].shape[-1]
        T = first["x_past"].shape[1]
        C = first["x_past"].shape[-1] + first["x_future"].shape[-1]
        c = self.config
        self.net = _ITransformerNet(
            n_channels=C, seq_len=T, n_static=n_static,
            d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_layers"], d_ff=c["d_ff"], dropout=c["dropout"],
        ).to(device)
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=lr or c["learning_rate"],
                                weight_decay=c.get("weight_decay", 0.01))
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
        self.net = _ITransformerNet(
            n_channels=ckpt["n_channels"], seq_len=ckpt["seq_len"],
            d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_layers"], d_ff=c["d_ff"], dropout=c["dropout"],
            n_static=ckpt.get("n_static", 0),
        )
        self.net.load_state_dict(ckpt["state_dict"])
        return self
