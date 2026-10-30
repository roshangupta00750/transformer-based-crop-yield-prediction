"""Preregistration v20 §2: stress inputs use the right windows and are causal (no future weather or stages)."""
import numpy as np
import pandas as pd

from src.data.dataset import STRESS_COLUMNS, add_stress_features


def _frame(tmax_shift=0.0):
    d = pd.date_range("2020-03-01", "2020-08-31")
    rng = np.random.default_rng(0)
    f = pd.DataFrame({"landkreis": "X", "crop": "winter_wheat", "year": 2020, "date": d,
                      "temperature_air_max_2m": 25 + 8 * np.sin(np.arange(len(d)) / 9),
                      "temperature_air_min_2m": rng.normal(5, 6, len(d)),
                      "precipitation_height": rng.gamma(1, 2, len(d)), "cwb": rng.normal(0, 2, len(d)),
                      "vpd": rng.gamma(2, 0.5, len(d)), "swroot7": rng.uniform(0.2, 0.4, len(d))})
    late = f.date > "2020-06-15"
    f.loc[late, "temperature_air_max_2m"] += tmax_shift            # change only future weather
    return f


def _obs(tmp_path):
    o = pd.DataFrame({"landkreis": "X", "crop": "winter_wheat", "year": 2020, "network": "full",
                      "slot": ["stem", "head", "ripe"],
                      "date": pd.to_datetime(["2020-04-10", "2020-05-25", "2020-07-20"])})
    p = tmp_path / "obs.parquet"; o.to_parquet(p)
    return p


def test_windows(tmp_path):
    f = add_stress_features(_frame(), "full", _obs(tmp_path)).set_index("date")
    assert set(STRESS_COLUMNS) <= set(f.columns)
    assert (f.loc[:"2020-04-09", ["se_prec", "se_heat", "frost_stem"]] == 0).all().all()       # before stem
    se = _frame().set_index("date").loc["2020-04-10":"2020-05-24", "precipitation_height"].sum()
    assert np.isclose(f.at[pd.Timestamp("2020-05-24"), "se_prec"], se)
    assert f.at[pd.Timestamp("2020-06-30"), "se_prec"] == f.at[pd.Timestamp("2020-05-24"), "se_prec"]  # frozen
    assert (f.loc[:"2020-05-24", ["fl_hdd", "fl_tmax", "gf_hdd"]] == 0).all().all()            # before heading
    assert f.at[pd.Timestamp("2020-07-31"), "fl_tmax"] == f.loc["2020-05-25":"2020-06-07", "temperature_air_max_2m"].max()


def test_causal(tmp_path):
    a = add_stress_features(_frame(), "full", _obs(tmp_path)).set_index("date")
    b = add_stress_features(_frame(tmax_shift=10.0), "full", _obs(tmp_path)).set_index("date")
    early = a.index <= "2020-06-15"
    assert np.allclose(a.loc[early, STRESS_COLUMNS], b.loc[early, STRESS_COLUMNS])            # future change, same past
