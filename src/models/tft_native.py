"""Temporal Fusion Transformer (Lim et al., 2021) on the shared crop-year pipeline
(preregistration v17 §2, arm T17).

One sample = one district-crop-year sequence -> one target z (yield anomaly against the
district normal), exactly like the iTransformer, so TFT gets the same rows, target, folds,
training loop and per-crop fine-tuning. Components as in the paper:

  static covariate encoder   soil, terrain, crop identity -> context vectors c_s, c_e, c_h, c_c
  variable selection         per-variable embedding; GRN(x_t | c_s) -> softmax weights over variables
  sequence encoder           LSTM initialised with (c_h, c_c); gated skip + LayerNorm
  static enrichment          GRN(phi_t | c_e)
  temporal attention         interpretable multi-head attention (shared value projection); the
                             final step is the query (one output per sample)
  output                     position-wise GRN, gated skip over the attention block, linear head

Simplifications stated in the preregistration: linear per-variable embeddings (no per-variable
GRN) and a single output step instead of a multi-horizon decoder. landkreis_id is excluded from
the statics (a district number carries no agronomy). The legacy pytorch-forecasting wrapper
(src/models/tft.py, a different task) is untouched.
"""
from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ._masking import resample_valid
from .itransformer import ITransformerModel
from .registry import register

EXCLUDED_STATICS = ("landkreis_id",)


class _GLU(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.a, self.b = nn.Linear(d_in, d_out), nn.Linear(d_in, d_out)

    def forward(self, x):
        return torch.sigmoid(self.a(x)) * self.b(x)


class _GRN(nn.Module):
    """Gated residual network: LayerNorm(skip(a) + GLU(W1 ELU(W2 a + W3 c)))."""

    def __init__(self, d_in: int, d_hidden: int, d_out: int, dropout: float, d_ctx: int = 0):
        super().__init__()
        self.skip = nn.Linear(d_in, d_out) if d_in != d_out else nn.Identity()
        self.w2 = nn.Linear(d_in, d_hidden)
        self.w3 = nn.Linear(d_ctx, d_hidden, bias=False) if d_ctx else None
        self.w1 = nn.Linear(d_hidden, d_hidden)
        self.drop = nn.Dropout(dropout)
        self.glu = _GLU(d_hidden, d_out)
        self.norm = nn.LayerNorm(d_out)

    def forward(self, a, c=None):
        h = self.w2(a)
        if self.w3 is not None and c is not None:
            h = h + self.w3(c)
        h = self.drop(self.w1(F.elu(h)))
        return self.norm(self.skip(a) + self.glu(h))


class _TFTNet(nn.Module):
    def __init__(self, n_channels: int, seq_len: int, static_idx: list[int],
                 hidden: int, n_heads: int, dropout: float):
        super().__init__()
        d = hidden
        self.seq_len, self.n_channels, self.static_idx = seq_len, n_channels, list(static_idx)
        self.hidden, self.n_heads = d, n_heads
        self.adapt = nn.AdaptiveAvgPool1d(seq_len)
        S = len(self.static_idx)
        if S:
            self.s_emb_w = nn.Parameter(torch.randn(S, d) * 0.1)
            self.s_emb_b = nn.Parameter(torch.zeros(S, d))
            self.s_select = _GRN(S, d, S, dropout)
            self.ctx = nn.ModuleList(_GRN(d, d, d, dropout) for _ in range(4))   # c_s, c_e, c_h, c_c
        self.v_emb_w = nn.Parameter(torch.randn(n_channels, d) * 0.1)
        self.v_emb_b = nn.Parameter(torch.zeros(n_channels, d))
        self.v_select = _GRN(n_channels, d, n_channels, dropout, d_ctx=d if S else 0)
        self.lstm = nn.LSTM(d, d, batch_first=True)
        self.lstm_gate, self.lstm_norm = _GLU(d, d), nn.LayerNorm(d)
        self.enrich = _GRN(d, d, d, dropout, d_ctx=d if S else 0)
        dk = d // n_heads
        self.q, self.k = nn.Linear(d, dk * n_heads), nn.Linear(d, dk * n_heads)
        self.v = nn.Linear(d, d)                      # shared across heads (interpretable MHA)
        self.attn_out = nn.Linear(d, d)
        self.attn_drop = nn.Dropout(dropout)
        self.attn_gate, self.attn_norm = _GLU(d, d), nn.LayerNorm(d)
        self.pos_grn = _GRN(d, d, d, dropout)
        self.out_gate, self.out_norm = _GLU(d, d), nn.LayerNorm(d)
        self.head = nn.Linear(d, 1)
        self.last_weights: dict[str, torch.Tensor] = {}

    def forward(self, x, mask=None, x_static=None):
        x = resample_valid(x, mask, self.adapt).transpose(1, 2)       # (B, T, C)
        B, T, C = x.shape
        d = self.hidden
        if self.static_idx and x_static is not None:
            s = x_static[:, self.static_idx]                            # (B, S)
            ws = torch.softmax(self.s_select(s), dim=-1)                # (B, S)
            se = (s.unsqueeze(-1) * self.s_emb_w + self.s_emb_b)        # (B, S, d)
            s_enc = (ws.unsqueeze(-1) * se).sum(1)                      # (B, d)
            c_s, c_e, c_h, c_c = (g(s_enc) for g in self.ctx)
        else:
            c_s = c_e = None
            c_h = c_c = x.new_zeros(B, d)
        # temporal variable selection
        wv = torch.softmax(self.v_select(x, None if c_s is None else c_s.unsqueeze(1).expand(B, T, d)), dim=-1)
        emb = x.unsqueeze(-1) * self.v_emb_w + self.v_emb_b              # (B, T, C, d)
        xi = (wv.unsqueeze(-1) * emb).sum(2)                             # (B, T, d)
        h, _ = self.lstm(xi, (c_h.unsqueeze(0).contiguous(), c_c.unsqueeze(0).contiguous()))
        phi = self.lstm_norm(xi + self.lstm_gate(h))
        theta = self.enrich(phi, None if c_e is None else c_e.unsqueeze(1).expand(B, T, d))
        # interpretable multi-head attention, final step as query
        H, dk = self.n_heads, d // self.n_heads
        q = self.q(theta[:, -1:]).view(B, 1, H, dk).transpose(1, 2)      # (B, H, 1, dk)
        k = self.k(theta).view(B, T, H, dk).transpose(1, 2)               # (B, H, T, dk)
        a = torch.softmax(q @ k.transpose(-1, -2) / dk ** 0.5, dim=-1)    # (B, H, 1, T)
        v = self.v(theta)                                                 # (B, T, d)
        att = self.attn_out((self.attn_drop(a) @ v.unsqueeze(1)).mean(1))  # (B, 1, d)
        delta = self.attn_norm(theta[:, -1:] + self.attn_gate(att))
        psi = self.pos_grn(delta)
        out = self.out_norm(phi[:, -1:] + self.out_gate(psi))
        if not self.training:
            self.last_weights = {"variables": wv.mean(1).detach(), "attention": a.mean(1).squeeze(1).detach()}
        return self.head(out).squeeze(-1).squeeze(-1)


@register
class TFTNativeModel(ITransformerModel):
    """Same training loop, early stopping, sample weights and fine-tuning as the iTransformer."""
    name = "tftnet"

    def _build_net(self, first: dict, train_loader) -> nn.Module:
        c = self.config
        spec = getattr(train_loader.dataset, "spec", None)
        names = list(spec.static) if spec is not None else []
        static_idx = [i for i, n in enumerate(names) if n not in EXCLUDED_STATICS]
        T = int(c.get("resample_len") or first["x_past"].shape[1])
        C = first["x_past"].shape[-1] + first["x_future"].shape[-1]
        return _TFTNet(n_channels=C, seq_len=T, static_idx=static_idx, hidden=c["hidden_size"],
                       n_heads=c["n_heads"], dropout=c["dropout"])

    def save(self, path):
        torch.save({"state_dict": self.net.state_dict(), "config": self.config, "seq_len": self.net.seq_len,
                    "n_channels": self.net.n_channels, "static_idx": self.net.static_idx}, path)

    def load(self, path):
        ck = torch.load(path, map_location="cpu")
        self.config = c = ck["config"]
        self.net = _TFTNet(n_channels=ck["n_channels"], seq_len=ck["seq_len"], static_idx=ck["static_idx"],
                           hidden=c["hidden_size"], n_heads=c["n_heads"], dropout=c["dropout"])
        self.net.load_state_dict(ck["state_dict"])
        return self
