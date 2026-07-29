"""LSTM baseline — single-stream temporal RNN over concatenated features.

Concatenates past-observed and known-future channels at each timestep,
appends static covariates by broadcasting, runs an LSTM, takes the final
hidden state, and projects to a scalar yield prediction.
"""
from __future__ import annotations

import copy

from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .base import BaseModel
from .registry import register


class _LSTMNet(nn.Module):
    def __init__(self, n_past: int, n_future: int, n_static: int,
                 hidden_size: int, num_layers: int, dropout: float,
                 bidirectional: bool):
        super().__init__()
        input_size = n_past + n_future + n_static
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.Linear(out_dim, 64), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(64, 1),
        )

    def forward(self, x_past, x_future, x_static, mask=None):
        B, T, _ = x_past.shape
        static = x_static.unsqueeze(1).expand(-1, T, -1)
        x = torch.cat([x_past, x_future, static], dim=-1)
        out, _ = self.lstm(x)
        if mask is not None:
            lengths = mask.sum(dim=1).clamp(min=1) - 1
            last = out[torch.arange(B, device=out.device), lengths]
        else:
            last = out[:, -1]
        return self.head(last).squeeze(-1)


@register
class LSTMModel(BaseModel):
    name = "lstm"

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.net: _LSTMNet | None = None
        self.device = "cpu"

    def _build_net(self, n_past, n_future, n_static):
        c = self.config
        self._dims = {"n_past": n_past, "n_future": n_future, "n_static": n_static}
        self.net = _LSTMNet(
            n_past=n_past, n_future=n_future, n_static=n_static,
            hidden_size=c["hidden_size"], num_layers=c["num_layers"],
            dropout=c["dropout"], bidirectional=c["bidirectional"],
        )

    def fit(self, train_loader: DataLoader, val_loader: DataLoader | None = None,
            *, max_epochs: int = 50, lr: float | None = None,
            patience: int = 10, device: str = "cpu",
            max_minutes: float | None = None):
        import time
        self.device = device
        first = next(iter(train_loader))
        if self.net is None:
            self._build_net(
                n_past=first["x_past"].shape[-1],
                n_future=first["x_future"].shape[-1],
                n_static=first["x_static"].shape[-1],
            )
        self.net.to(device)
        opt = torch.optim.AdamW(self.net.parameters(),
                                lr=lr or self.config["learning_rate"])
        loss_fn = nn.MSELoss()

        deadline = time.time() + max_minutes * 60 if max_minutes else None
        best_val, bad = float("inf"), 0
        # Keep the best-validation weights. Early stopping only breaks out of
        # the loop, so without this the checkpoint saved is the FINAL epoch —
        # `patience` epochs past the optimum, i.e. a deliberately over-fit model.
        best_state = copy.deepcopy(self.net.state_dict())
        for epoch in range(max_epochs):
            self.net.train()
            for batch in train_loader:
                mask = batch.get("mask")
                pred = self.net(
                    batch["x_past"].to(device),
                    batch["x_future"].to(device),
                    batch["x_static"].to(device),
                    mask.to(device) if mask is not None else None,
                )
                loss = loss_fn(pred, batch["y"].to(device))
                opt.zero_grad(); loss.backward(); opt.step()

            if deadline and time.time() > deadline:
                break
            if val_loader is None:
                continue
            val = self._eval_loss(val_loader, loss_fn)
            if val < best_val:
                best_val, bad = val, 0
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    @torch.no_grad()
    def _eval_loss(self, loader, loss_fn):
        self.net.eval()
        total, n = 0.0, 0
        for batch in loader:
            mask = batch.get("mask")
            pred = self.net(
                batch["x_past"].to(self.device),
                batch["x_future"].to(self.device),
                batch["x_static"].to(self.device),
                mask.to(self.device) if mask is not None else None,
            )
            total += loss_fn(pred, batch["y"].to(self.device)).item() * len(pred)
            n += len(pred)
        return total / max(n, 1)

    @torch.no_grad()
    def predict(self, loader: DataLoader) -> np.ndarray:
        self.net.eval()
        out = []
        for batch in loader:
            mask = batch.get("mask")
            pred = self.net(
                batch["x_past"].to(self.device),
                batch["x_future"].to(self.device),
                batch["x_static"].to(self.device),
                mask.to(self.device) if mask is not None else None,
            )
            out.append(pred.cpu().numpy())
        return np.concatenate(out)

    def save(self, path):
        lstm_in = self.net.lstm.input_size
        # The forward pass concatenates [x_past, x_future, x_static]; we need
        # the individual widths to rebuild on load. Stash them now.
        dims = getattr(self, "_dims", None)
        torch.save({"state_dict": self.net.state_dict(),
                    "config": self.config,
                    "dims": dims,
                    "input_size": lstm_in}, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.config = ckpt["config"]
        dims = ckpt.get("dims")
        if dims is None:
            raise RuntimeError(
                "Checkpoint missing input dims; retrain with the patched save()."
            )
        self._build_net(**dims)
        self.net.load_state_dict(ckpt["state_dict"])
        return self
