"""Data acquisition, preprocessing, and PyTorch dataset interfaces."""
from .dataset import CropYieldDataset, build_unified_dataset
from .splits import make_temporal_split, make_spatial_split

__all__ = [
    "CropYieldDataset",
    "build_unified_dataset",
    "make_temporal_split",
    "make_spatial_split",
]
