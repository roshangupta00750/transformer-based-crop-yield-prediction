"""Shared helper for keeping zero padding out of the models.

``collate_pad`` pads every sequence to a single global length (the longest
phenology window, 220 days) and returns a boolean ``mask`` marking the valid
timesteps. Three of the architectures then called
``nn.AdaptiveAvgPool1d(seq_len)`` on the padded tensor — but since the tensor
is already exactly ``seq_len`` long, that pool is an identity and the padding
flows straight into the model.

The damage is concentrated on short-season crops, because the padding
fraction is the difference between their growing season and winter wheat's:

    summer_barley  136 of 220 valid -> 38.2 % padding
    oats           146             -> 33.6 %
    silage_maize   151             -> 31.4 %
    potatoes       166             -> 24.5 %
    winter_wheat   220             ->  0.0 %

``resample_valid`` slices each sample to its own valid length and *then*
resamples to ``seq_len``. Two things follow: no padding reaches the network,
and position i means the same phenological phase for every crop rather than
the same absolute day-of-year — which is what a pooled multi-crop model
needs, since nothing else in the feature set identifies the crop.

Lengths are grouped before pooling, so the loop runs once per distinct
sequence length in the batch (at most one per crop, ~10) rather than once
per sample.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def resample_valid(x: torch.Tensor, mask: torch.Tensor | None,
                   pool: nn.AdaptiveAvgPool1d) -> torch.Tensor:
    """Resample the valid region of ``x`` to the pool's output length.

    Parameters
    ----------
    x
        ``(B, T, C)`` — time-major batch straight from ``collate_pad``.
    mask
        ``(B, T)`` boolean, True on valid timesteps. When None the whole
        tensor is treated as valid, reproducing the previous behaviour.
    pool
        The module's ``nn.AdaptiveAvgPool1d(seq_len)``.

    Returns
    -------
    ``(B, C, seq_len)`` — channel-major, ready for the patch/linear stages.
    """
    xc = x.transpose(1, 2)                       # (B, C, T)
    if mask is None:
        return pool(xc)

    lengths = mask.sum(dim=1).to(torch.long)     # (B,)
    # A degenerate all-False row would slice to width 0; keep at least one step.
    lengths = lengths.clamp(min=1)
    if bool((lengths == xc.shape[-1]).all()):
        return pool(xc)                          # nothing padded, skip the work

    out = None
    for L in torch.unique(lengths):
        sel = (lengths == L).nonzero(as_tuple=True)[0]
        pooled = pool(xc[sel, :, :int(L)])       # (b, C, seq_len)
        if out is None:
            out = xc.new_empty((xc.shape[0], xc.shape[1], pooled.shape[-1]))
        out[sel] = pooled
    return out
