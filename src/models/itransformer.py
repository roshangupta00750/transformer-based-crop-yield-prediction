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


# Static covariates that may join the attention as extra tokens, grouped so
# each group becomes ONE token. landkreis_id is deliberately absent: a district
# number carries no agronomy and only lets the model memorise districts.
STATIC_TOKEN_GROUPS = {
    "crop": lambda n: n.startswith("crop_is_"),
    "soil": lambda n: n in ("sand", "clay", "soc", "ph", "bulk_density",
                            "soil_imputed"),
    "terrain": lambda n: n.startswith(("elevation_", "slope_", "aspect_")),
}


def resolve_static_tokens(static_names: list[str],
                          groups: list[str]) -> list[list[int]]:
    """Column indices into x_static for each requested token group."""
    out = []
    for g in groups:
        if g not in STATIC_TOKEN_GROUPS:
            raise ValueError(f"unknown static token group {g!r}; "
                             f"choose from {sorted(STATIC_TOKEN_GROUPS)}")
        idx = [i for i, n in enumerate(static_names) if STATIC_TOKEN_GROUPS[g](n)]
        if not idx:
            raise ValueError(f"static token group {g!r} matched no columns in "
                             f"{static_names}")
        out.append(idx)
    return out


class _ITransformerNet(nn.Module):
    def __init__(self, n_channels: int, seq_len: int,
                 d_model: int, n_heads: int, n_layers: int,
                 d_ff: int, dropout: float, n_static: int = 0,
                 attention_axis: str = "variate",
                 static_token_idx: list[list[int]] | None = None):
        super().__init__()
        if attention_axis not in ("variate", "temporal"):
            raise ValueError(f"attention_axis must be variate|temporal, "
                             f"got {attention_axis!r}")
        self.seq_len = seq_len
        self.n_channels = n_channels
        # ABLATION AXIS. iTransformer's defining choice is that a TOKEN IS A
        # VARIATE: the series is transposed so attention runs across channels
        # rather than across time. Setting attention_axis="temporal" removes
        # exactly that transpose and nothing else — same depth, heads,
        # d_model, d_ff, dropout, optimiser and budget — turning the model
        # into a conventional time-attention encoder.
        #
        # This isolates the claim. Comparing iTransformer to PatchTST or TFT
        # confounds the attention axis with every other architectural
        # difference between them; comparing iTransformer to itself does not.
        self.attention_axis = attention_axis
        self.adapt = nn.AdaptiveAvgPool1d(seq_len)
        # variate: each of C channels embeds its length-seq_len series
        # temporal: each of seq_len steps embeds its C-vector
        self.var_embed = nn.Linear(
            seq_len if attention_axis == "variate" else n_channels, d_model)
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
        n_tokens = n_channels if attention_axis == "variate" else seq_len
        # CROP-AWARE VARIANT (static_tokens in the model config). Each static
        # group -- crop identity, soil, terrain -- is embedded into ONE extra
        # token that sits beside the weather/satellite variate tokens inside
        # the encoder. Attention can then condition how a dry June is read on
        # which crop and which soil it is, i.e. the multiplicative modulation
        # the additive head above cannot express. Without crop identity the
        # pooled model sees near-identical inputs for wheat, triticale and rye
        # in the same district-year and can only predict their average.
        self.static_token_idx = [list(i) for i in (static_token_idx or [])]
        if self.static_token_idx and attention_axis != "variate":
            raise ValueError("static tokens require attention_axis='variate'")
        self.static_embeds = nn.ModuleList(
            nn.Linear(len(i), d_model) for i in self.static_token_idx)
        n_tokens += len(self.static_token_idx)
        self.n_tokens = n_tokens
        self.proj = nn.Linear(d_model * n_tokens + n_static, 1)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None,
                x_static: torch.Tensor | None = None) -> torch.Tensor:
        # x: (B, T_var, C) -> (B, C, seq_len), padding excluded
        x = resample_valid(x, mask, self.adapt)
        if self.attention_axis == "temporal":
            # (B, C, seq_len) -> (B, seq_len, C): tokens become timesteps, so
            # the encoder attends across time instead of across variates.
            x = x.transpose(1, 2)
        tokens = self.var_embed(x)                # (B, n_tokens, D)
        if self.static_token_idx:
            extra = [emb(x_static[:, idx]) for emb, idx
                     in zip(self.static_embeds, self.static_token_idx)]
            tokens = torch.cat([tokens, torch.stack(extra, dim=1)], dim=1)
        encoded = self.encoder(tokens)            # (B, C, D)
        encoded = self.norm(encoded)              # LayerNorm on D
        B, N, D = encoded.shape
        flat = encoded.reshape(B, N * D)
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
            max_minutes: float | None = None,
            init_state: dict | None = None):
        """Train; with ``init_state`` (a state_dict of the same architecture)
        the network starts from those weights instead of a fresh init --
        per-crop fine-tuning, preregistration v4 §2.2."""
        import time
        self.device = device
        first = next(iter(train_loader))
        c = self.config
        self.net = self._build_net(first, train_loader).to(device)
        if init_state is not None:
            self.net.load_state_dict(init_state)
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
        # learning curve, kept in the run metadata (train loss = mean over the epoch's batches)
        self.history_, self.best_epoch_, self.stop_reason_ = [], None, "max_epochs"
        t_start = time.time()
        for epoch in range(1, max_epochs + 1):
            self.net.train()
            tr_sum, tr_n = 0.0, 0
            for batch in train_loader:
                x = torch.cat([batch["x_past"], batch["x_future"]], dim=-1).to(device)
                msk = batch.get("mask")
                msk = msk.to(device) if msk is not None else None
                xs = batch["x_static"].to(device)
                y = batch["y"].to(device)
                pred = self.net(x, msk, xs)
                if "w" in batch:                       # v15 §4: area-weighted training loss
                    w = batch["w"].to(device)
                    loss = (w * (pred - y) ** 2).sum() / w.sum().clamp_min(1e-8)
                else:
                    loss = loss_fn(pred, y)
                opt.zero_grad(); loss.backward(); opt.step()
                tr_sum += float(loss.item()) * len(y); tr_n += len(y)

            val = self._eval(val_loader, loss_fn) if val_loader is not None else None
            self.history_.append({"epoch": epoch, "train_loss": tr_sum / max(tr_n, 1), "val_loss": val,
                                  "seconds": round(time.time() - t_start, 1)})
            if deadline and time.time() > deadline:
                self.stop_reason_ = "time_cap"     # as before: the timed-out epoch is not a candidate (logging only)
                break
            if val_loader is None:
                continue
            if val < best:
                best, bad, self.best_epoch_ = val, 0, epoch
                best_state = copy.deepcopy(self.net.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    self.stop_reason_ = "early_stop"
                    break
        if best_state is not None:
            self.net.load_state_dict(best_state)
        return self

    def _build_net(self, first: dict, train_loader) -> nn.Module:
        """The network for this config and data; subclasses (preregistration v17: TFT,
        PatchTST) override only this, save and load, so the training loop is shared."""
        c = self.config
        n_static = (first["x_static"].shape[-1]
                    if c.get("use_static_head", False) else 0)
        # resample_len fixes the model's input length independently of the
        # padding length, so a longer season window compresses the input
        # instead of enlarging the model (preregistration §3).
        T = int(c.get("resample_len") or first["x_past"].shape[1])
        C = first["x_past"].shape[-1] + first["x_future"].shape[-1]
        static_idx = None
        if c.get("static_tokens"):
            spec = getattr(train_loader.dataset, "spec", None)
            if spec is None:
                raise RuntimeError("static_tokens needs the dataset's FeatureSpec")
            static_idx = resolve_static_tokens(list(spec.static),
                                               list(c["static_tokens"]))
        return _ITransformerNet(
            n_channels=C, seq_len=T, n_static=n_static,
            static_token_idx=static_idx,
            attention_axis=c.get("attention_axis", "variate"),
            d_model=c["d_model"], n_heads=c["n_heads"],
            n_layers=c["n_layers"], d_ff=c["d_ff"], dropout=c["dropout"],
        )

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
            "static_token_idx": getattr(self.net, "static_token_idx", []),
            "attention_axis": getattr(self.net, "attention_axis", "variate"),
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
            attention_axis=ckpt.get("attention_axis", "variate"),
            static_token_idx=ckpt.get("static_token_idx") or None,
        )
        self.net.load_state_dict(ckpt["state_dict"])
        return self
