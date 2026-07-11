"""Per-source data acquisition adapters.

Each module exposes a `download(out_dir, **kwargs)` function that writes
raw files into `data/raw/<source>/`, and a `clean(raw_dir, out_path)`
function that produces an interim parquet ready for the unified builder.
"""
