"""iTransformer (Liu et al., ICLR 2024) — variates as tokens.

Inverts the usual axes: each variable becomes a token, attention runs
across variables (not time). Each variable's whole time series is
linearly embedded into a single D-dimensional token.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .base import BaseModel
from .registry import register


class _ITransformerNet(nn.Module):
    def __init__(self, n_channels: int, seq_len: int,
                 d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float):
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
        self.proj = nn.Linear(d_model * n_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T_var, C) -> (B, C, seq_len)
        x = self.adapt(x.transpose(1, 2))
        tokens = self.var_embed(x)                # (B, C, D)
        encoded = self.encoder(tokens)            # (B, C, D)
        encoded = self.norm(encoded)              # LayerNorm on D
        B, C, D = encoded.shape
        return self.proj(encoded.reshape(B, C * D)).squeeze(-1)


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
        T = first["x_past"].shape[1]
        C = first["x_past"].shape[-1] + first["x_future"].shape[-1]
        c = self.config
        self.net = _ITransformerNet(
            n_channels=C, seq_len=T,
            d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_layers"], d_ff=c["d_ff"], dropout=c["dropout"],
        ).to(device)
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=lr or c["learning_rate"],
                                weight_decay=c.get("weight_decay", 0.01))
        loss_fn = nn.MSELoss()
        deadline = time.time() + max_minutes * 60 if max_minutes else None
        best, bad = float("inf"), 0
        for _ in range(max_epochs):
            self.net.train()
            for batch in train_loader:
                x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1).to(device)
                y = batch["y"].to(device)
                pred = self.net(x)
                loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()

            if deadline and time.time() > deadline:
                break
            if val_loader is None:
                continue
            val = self._eval(val_loader, loss_fn)
            if val < best:
                best, bad = val, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        return self

    @torch.no_grad()
    def _eval(self, loader, loss_fn):
        self.net.eval()
        total, n = 0.0, 0
        for batch in loader:
            x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1).to(self.device)
            y = batch["y"].to(self.device)
            total += loss_fn(self.net(x), y).item() * len(y)
            n += len(y)
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, loader) -> np.ndarray:
        self.net.eval()
        out = []
        for batch in loader:
            x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1).to(self.device)
            out.append(self.net(x).cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        torch.save({
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
        )
        self.net.load_state_dict(ckpt["state_dict"])
        return self
