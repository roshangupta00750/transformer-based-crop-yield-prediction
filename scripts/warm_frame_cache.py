"""Build the cached crop-year modelling frame for one dataset (used before parallel training).
Usage: python scripts/warm_frame_cache.py data/processed/<dataset>.parquet"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from src.data.dataset import load_modeling_frame  # noqa: E402
from src.utils import load_data_config  # noqa: E402

f = load_modeling_frame(ROOT / sys.argv[1], load_data_config(), window="crop_year", common_sample=True,
                        cache_dir=ROOT / "data/processed/cache")
print("frame", len(f), f.groupby(["landkreis", "crop", "year"]).ngroups, "crop-years", flush=True)
