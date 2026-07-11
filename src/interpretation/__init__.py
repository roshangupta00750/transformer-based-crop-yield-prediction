"""Interpretability comparison across models.

TFT has native variable-selection + attention. For the others we rely
on permutation importance (a model-agnostic baseline) and per-channel
gradient × input for the neural models.
"""
