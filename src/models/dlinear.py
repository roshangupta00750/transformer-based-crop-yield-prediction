"""DLinear (Zeng et al., AAAI 2023) — linear sanity-check baseline.

If a transformer can't beat DLinear, the transformer isn't earning its
complexity. Decomposes the input into trend + seasonal via a moving
average and applies one linear projection per channel (or shared).
"""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseModel
from .registry import register


class _MovingAvg(nn.Module):
    """Moving average via avg-pool with reflective padding."""
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C)
        pad = (self.kernel_size - 1) // 2
        x_p = torch.nn.functional.pad(x.transpose(1, 2),
                                      (pad, pad + (1 - self.kernel_size % 2)),
                                      mode="replicate")
        return self.avg(x_p).transpose(1, 2)


class _DLinearNet(nn.Module):
    def __init__(self, seq_len: int, n_channels: int,
                 kernel_size: int = 25, individual: bool = False):
        super().__init__()
        self.individual = individual
        self.seq_len = seq_len
        self.decomp = _MovingAvg(kernel_size)
        # Adaptive pool brings variable-length batches to ``seq_len`` so the
        # downstream linear has a fixed input size.
        self.adapt = nn.AdaptiveAvgPool1d(seq_len)
        if individual:
            self.linear_t = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(n_channels)])
            self.linear_s = nn.ModuleList([nn.Linear(seq_len, 1) for _ in range(n_channels)])
        else:
            self.linear_t = nn.Linear(seq_len * n_channels, 1)
            self.linear_s = nn.Linear(seq_len * n_channels, 1)

    def _to_fixed(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C) -> (B, seq_len, C)
        return self.adapt(x.transpose(1, 2)).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._to_fixed(x)
        trend = self.decomp(x)
        seasonal = x - trend
        B, T, C = x.shape
        if self.individual:
            t = torch.stack([self.linear_t[c](trend[:, :, c]) for c in range(C)], -1).sum(-1).squeeze(-1)
            s = torch.stack([self.linear_s[c](seasonal[:, :, c]) for c in range(C)], -1).sum(-1).squeeze(-1)
        else:
            t = self.linear_t(trend.reshape(B, -1)).squeeze(-1)
            s = self.linear_s(seasonal.reshape(B, -1)).squeeze(-1)
        return t + s


@register
class DLinearModel(BaseModel):
    name = "dlinear"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.net: _DLinearNet | None = None
        self.device = "cpu"
        self.seq_len: int | None = None

    def fit(self, train_loader, val_loader=None, *,
            max_epochs: int = 100, lr: float | None = None,
            patience: int = 15, device: str = "cpu",
            max_minutes: float | None = None):
        import time
        self.device = device
        first = next(iter(train_loader))
        T = first["x_past"].shape[1]
        # DLinear processes past + future stacked along channel axis.
        n_channels = first["x_past"].shape[-1] + first["x_future"].shape[-1]
        self.seq_len = T
        self.net = _DLinearNet(
            seq_len=T, n_channels=n_channels,
            kernel_size=self.config["moving_avg_kernel"],
            individual=self.config["individual"],
        ).to(device)

        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=lr or self.config["learning_rate"])
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
    def predict(self, loader):
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
            "seq_len": self.seq_len,
            "n_channels": self.net.linear_t.in_features // self.net.seq_len
                           if hasattr(self.net, "linear_t") and isinstance(self.net.linear_t, nn.Linear)
                           else len(self.net.linear_t),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.config = ckpt["config"]
        self.seq_len = ckpt["seq_len"]
        self.net = _DLinearNet(
            seq_len=ckpt["seq_len"], n_channels=ckpt["n_channels"],
            kernel_size=self.config["moving_avg_kernel"],
            individual=self.config["individual"],
        )
        self.net.load_state_dict(ckpt["state_dict"])
        return self
