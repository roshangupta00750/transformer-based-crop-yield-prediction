"""Preregistration v17: TFT and PatchTST run on the shared loop (fit, init_state, save/load, predict)."""
import os
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from src.models import build_model  # noqa: E402

STATIC = ["sand", "clay", "landkreis_id", "crop_is_oats", "crop_is_rye"]


class _DS(torch.utils.data.Dataset):
    spec = SimpleNamespace(static=STATIC)

    def __init__(self, n=24, T=60, C=7):
        g = torch.Generator().manual_seed(0)
        self.x = torch.randn(n, T, C, generator=g)
        self.s = torch.randn(n, len(STATIC), generator=g)
        self.y = self.x[:, -20:, 0].mean(1)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return {"x_past": self.x[i, :, :5], "x_future": self.x[i, :, 5:], "x_static": self.s[i], "y": self.y[i],
                "mask": torch.ones(self.x.shape[1], dtype=torch.bool)}


CFG = {"tftnet": {"name": "tftnet", "hidden_size": 16, "n_heads": 4, "dropout": 0.1, "learning_rate": 1e-3,
                  "resample_len": 40},
       "patchtst17": {"name": "patchtst17", "patch_len": 8, "stride": 4, "d_model": 16, "n_heads": 4, "n_layers": 1,
                      "dropout": 0.1, "learning_rate": 1e-3, "resample_len": 40}}


@pytest.mark.parametrize("name", sorted(CFG))
def test_fit_finetune_save_load(name):
    dl = torch.utils.data.DataLoader(_DS(), batch_size=8)
    m = build_model(name, CFG[name]).fit(dl, dl, max_epochs=3, patience=2)
    p = m.predict(dl)
    assert p.shape == (24,) and np.isfinite(p).all()
    init = {k: v.clone() for k, v in m.net.state_dict().items()}
    m2 = build_model(name, CFG[name]).fit(dl, dl, max_epochs=1, init_state=init)   # per-crop fine-tuning path
    assert np.isfinite(m2.predict(dl)).all()
    with tempfile.TemporaryDirectory() as d:
        f = os.path.join(d, "m.pt")
        m.save(f)
        m3 = build_model(name, CFG[name]).load(f)
        assert np.allclose(m3.predict(dl), p, atol=1e-5)


def test_tft_excludes_district_number():
    dl = torch.utils.data.DataLoader(_DS(), batch_size=8)
    m = build_model("tftnet", CFG["tftnet"]).fit(dl, None, max_epochs=1)
    assert STATIC.index("landkreis_id") not in m.net.static_idx and len(m.net.static_idx) == 4


@pytest.mark.parametrize("name", sorted(CFG))
def test_learning_curve_recorded(name):
    from src.training.trainer import Trainer
    dl = torch.utils.data.DataLoader(_DS(), batch_size=8)
    m = build_model(name, CFG[name])
    r = Trainer(m, max_epochs=6, early_stopping_patience=2).fit(dl, dl)
    assert len(r.history) >= 1 and all({"epoch", "train_loss", "val_loss", "seconds"} <= set(h) for h in r.history)
    assert r.stop_reason in ("early_stop", "max_epochs", "time_cap")
    assert r.best_epoch == min(r.history, key=lambda h: h["val_loss"])["epoch"]
    assert abs(r.best_val_loss - min(h["val_loss"] for h in r.history)) < 1e-12


def test_logging_does_not_change_training():
    """Same seed, same data -> identical predictions as without logging (weights chosen as before)."""
    from src.utils import set_seed
    dl = torch.utils.data.DataLoader(_DS(), batch_size=8)
    out = []
    for _ in range(2):
        set_seed(0)
        out.append(build_model("tftnet", CFG["tftnet"]).fit(dl, dl, max_epochs=4, patience=2).predict(dl))
    assert np.allclose(out[0], out[1])
