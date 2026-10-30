"""Hand-typed numbers in the dashboard must equal the computed results.

scripts/63_dashboard_data.py transcribes each version's hindcast accuracy (new vs main model, same
rows) and its paper-comparison wins from the pre-registrations; scripts/69_eval_v16.py transcribes
each decision's P(Δ>0). These tests recompute every typed value from the saved score files and
comparison tables, so a typo or a later re-run cannot leave the dashboard showing a stale number.
"""
import importlib.util
import json
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
MET = ROOT / "results" / "metrics"

# version -> (score file, path to the decision's hindcast block)
SCORES = {
    "v7": ("v7_scores.json", ["hindcast_2010_2015"]),
    "v8": ("v8_scores.json", ["hindcast_2010_2015"]),
    "v9": ("v9_scores.json", ["hindcast_2010_2015"]),
    "v10": ("v10_scores.json", ["decision_rows", "hindcast_2010_2015"]),
    "v12": ("v12_scores.json", ["decision_rows", "hindcast_2010_2015"]),
    "v13": ("v13_scores.json", ["hindcast_2010_2015"]),
    "v14": ("v14_scores_hindcast.json", ["hindcast_2010_2015"]),
    "v15b": ("v15W_scores.json", ["WE15_vs_E15", "hindcast_2010_2015"]),
    "v15": ("v15_scores.json", ["hindcast_2010_2015"]),
    "v17": ("v17_scores.json", ["TE17_vs_E15", "hindcast_2010_2015"]),
    "v18": ("v18_scores.json", ["XE18_vs_E15", "hindcast_2010_2015"]),
    "v19": ("v19_scores.json", ["S19_vs_XE18", "hindcast_2010_2015"]),
    "v20": ("v20_scores.json", ["XE20_vs_XE18", "hindcast_2010_2015"]),
}
# scripts/69 DECISIONS row (version, change prefix) -> version key above
DECISION_KEY = {("v7", "R7"): "v7", ("v8", "Y"): "v8", ("v9", "baseline"): "v9", ("v10", "D10"): "v10",
                ("v12", "Q12"): "v12", ("v13", "C13"): "v13", ("v14", "H14"): "v14",
                ("v15", "E15"): "v15", ("v15", "W15"): "v15b"}
# where the registered decision's P sits when it is not the all-crop p_delta_gt_0 (v8 decided rapeseed only)
P_PATH = {"v8": ["rapeseed", "p_Y_better"]}


def _load(name, file):
    s = importlib.util.spec_from_file_location(name, ROOT / "scripts" / file)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def dash():
    return _load("d63", "63_dashboard_data.py")


@pytest.fixture(scope="module")
def versions(dash):
    return pd.DataFrame(dash.VERSIONS, columns=["version", "planned", "result", "title", "change", "hindcast_new",
                                                "hindcast_main", "papers_won", "decision", "lesson"]).set_index("version")


def _block(version):
    f, path = SCORES[version]
    if not (MET / f).exists():
        pytest.skip(f"{f} not present")
    d = json.loads((MET / f).read_text())
    for k in path:
        d = d[k]
    return d


@pytest.mark.parametrize("version", sorted(SCORES))
def test_typed_hindcast_accuracy_matches_scores(versions, version):
    b = _block(version)
    new, main = 100 - b["mape_R7"], 100 - b["mape_F"]
    row = versions.loc[version]
    assert abs(row.hindcast_new - new) < 0.006, f"{version}: typed {row.hindcast_new} vs computed {new:.3f}"
    assert abs(row.hindcast_main - main) < 0.006, f"{version}: typed main {row.hindcast_main} vs computed {main:.3f}"


def test_typed_papers_won_matches_comparison_tables(dash, versions):
    wrong = []
    for v, f in dash.PAPER_FILES.items():
        typed = versions.loc[v, "papers_won"] if v in versions.index else None
        if typed is None or typed != typed or not (MET / f).exists():
            continue
        c = pd.read_csv(MET / f)
        won = int((c["ours_better"] if "ours_better" in c else c["verdict"].eq("ours better")).sum()) \
            if ("ours_better" in c or "verdict" in c) else None
        if won is None:
            pytest.skip(f"{f}: no verdict column")
        if won != int(typed):
            wrong.append(f"{v}: typed {int(typed)}, {f} says {won}")
    assert not wrong, wrong


def test_typed_decision_certainty_matches_scores():
    v16 = _load("v16", "69_eval_v16.py")
    wrong = []
    for version, change, p, _ in v16.DECISIONS:
        key = next(k for (v, pre), k in DECISION_KEY.items() if v == version and change.startswith(pre))
        b = _block(key)
        for k in P_PATH.get(key, ["p_delta_gt_0"]):
            b = b[k]
        computed = b
        if abs(computed - p) > 0.005:
            wrong.append(f"{version} {change}: typed {p}, computed {computed:.4f}")
    assert not wrong, wrong
