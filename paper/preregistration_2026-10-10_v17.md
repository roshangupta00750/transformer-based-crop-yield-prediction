# Pre-registration v17: the other two transformers (TFT, PatchTST) on YieldFormer-DE's protocol

**Written:** 10 October 2026, before any v17 model is trained or smoke-tested. Scoring and the
model-change rule as v6 §4-§6. Main model: YieldFormer-DE = v15 E15 (v12 Q12 iTransformer +
district error memory). Paper title fixed: "Transformer-based Crop Yield Prediction using
multivariate Timeseries datasets"; TFT and PatchTST are its other two transformers.

## 1. Why

The only TFT and PatchTST results are from August 2026, before the feature-policy-v3 leakage fix
and the corrected 366-district data; they are not comparable and are not reported. v16 showed that
gradient boosting on the same inputs ties the iTransformer (wins 2016-2025, loses 2010-2015), so
the paper must show how the transformer architectures compare under one protocol.

The August TFT (`src/models/tft.py`, pytorch-forecasting) used a different task: raw yield as the
target (not the anomaly against the district normal), 180-day windows sliding over multi-year
series, and the previous year's yield inside the encoder. It is **not** reused. Both architectures
are run on the exact Q12 pipeline instead.

## 2. Arms (only the network changes)

Common to both, identical to Q12: dataset_v5_hybridW, crop-year window, phenology v3 + stage
features + district anomalies, reliable-series training rows, target z = (yield − y_hist) / σ_crop,
inputs resampled to 220 steps over each sample's valid span, rolling folds 2010-2025 (train ≤ Y−2,
validate Y−1, test Y), seeds 42/43/44, then per-crop fine-tuning (`scripts/47`, same 5 candidates,
selected on Y−1) and the ensemble mean. Same budget: ≤ 120 epochs, patience 15, ≤ 120 minutes.

**T17, Temporal Fusion Transformer** (Lim et al. 2021), implemented natively on the shared pipeline
(`src/models/tft_native.py`; one sample = one crop-year → one z):
static covariate encoder over soil, terrain and crop identity (landkreis_id excluded, as for
every network: a district number carries no agronomy) giving the four TFT context vectors;
temporal variable selection (per-variable linear embedding, GRN-softmax selection weights
conditioned on the static context); LSTM encoder initialised from the static context; gated
skip + LayerNorm; static enrichment GRN; interpretable multi-head attention (shared value
projection) with the final step as query; position-wise GRN; linear output. hidden 64, 4 heads,
dropout 0.15, AdamW lr 5e-4, weight decay 0.05, batch 128. Simplification against the paper,
stated in advance: per-variable embeddings are linear (no per-variable GRN), and a single output
step replaces the multi-horizon decoder (the task has one target per sample).

**P17, PatchTST** (Nie et al. 2023), existing implementation (`src/models/patchtst.py`) with the
August architecture unchanged (patch 8, stride 4, d_model 128, 8 heads, 4 layers, dropout 0.15,
lr 3e-4), plus the two pieces the protocol needs: `resample_len` 220 and `init_state` for
per-crop fine-tuning; weight decay 0.05 as Q12.

**TE17 / PE17** = T17 / P17 corrected by the v15 §2 error memory using each arm's own past errors
(λ chosen causally on 2011..Y−1).

No hyperparameter is tuned on any test year; no re-run after seeing results.

## 3. Decision (v6 §6, hindcast 2010-2015)

TE17 or PE17 replaces E15 as the main model if, on the hindcast, its macro MSE skill vs y_hist is
higher with year-block bootstrap P(Δ > 0) ≥ 0.95, pooled MAPE is lower, and no crop is worse in
every year. If both pass, the one with the higher macro skill. The name YieldFormer-DE stays for
the method (normal + network + error memory); only the network inside changes. Also reported, not
decisive: 2016-2025, per crop, T17/P17 vs Q12 without memory, vs gradient boosting + memory (GE16),
and the TFT variable-selection weights per crop (interpretability). A new main model would need a
fresh 2026 forecast to be frozen separately; the frozen YieldFormer-DE (Q12) 2026 file is never
changed.

## 4. Order of work

1. 1-epoch smoke test of both arms on one fold (train + fine-tune + prediction), before any full
   run. 2. Full runs on julia2 L40, QoS computervision (user's standing rule). 3. Evaluation
   `scripts/73_eval_v17.py` on the Mac.

## Change log

| Date | Change | Reason |
|---|---|---|
| 2026-10-10 | Plan written | - |
| 2026-10-10 | Smoke test passed (job 3476262, fold 2015 seed 42, 1 epoch, fine-tuning on 2 crops, prediction): both arms run end to end; ~2 min per training epoch on an L40 for each. No smoke number is used. Implementation: `ITransformerModel._build_net` factored out so T17 (`tftnet`) and P17 (`patchtst17`) share the identical training loop; `scripts/47` builds the network named in the config; tests `tests/test_v17_nets.py`. | Operational. |
| 2026-10-10 | **Scope decision by the user, before any v17 result exists:** the study keeps one model family, transformers, plus simple baselines (averages, trend + memory). The §3 "also reported" comparison against gradient boosting + memory (GE16) is dropped; nothing else changes (decision rule, arms, budget). | User decision. |
| 2026-10-10 | **Scope refined by the user (same day, still before any v17 result):** the candidates stay transformers only; XGBoost + memory (GE16) is reported again as the single non-transformer **reference** (vs TE17/PE17, 2010-2015 and 2016-2025), never a candidate and not part of the §3 decision. | User decision. |
| 2026-10-11 | **T17 result (train 3476278, fine-tune 3476279, predict 3476280; all 48+48 tasks completed; 37,383 rows, y_hist identical to Q12 to 2e-13).** §3 decision, hindcast 2010-2015: TE17 88.76% vs E15 88.48%, macro skill Δ +0.023 (90% CI −0.055..+0.105), P(Δ>0) 0.664, no crop worse in every year → **v6 §6 not met; E15 (iTransformer) stays the main model.** TE17 is better in 2011, 2012, 2015 and worse in 2010, 2013, 2014. Reported, not decisive: 2016-2025 TE17 88.91 vs E15 88.38 (Δ +0.070, CI +0.033..+0.120, P 1.000; better in 9 of 10 years, worse in 2023); 2010-2025 88.84 vs 88.43 (P 0.970). Without memory T17 vs Q12: 88.52 vs 88.32 (P 0.56) and 88.37 vs 87.90 (P 1.000). Reference GE16: TE17 ahead on 2010-2015 (P 0.984) and 2016-2025 (88.91 vs 88.60, P 0.925). Per crop 2016-2025 (pp): rye +1.75, wheat +1.72, potatoes +1.01, rapeseed +0.98, silage maize +0.33, winter barley +0.32, sugar beet +0.32, triticale −0.04, summer barley −0.56, oats −1.01. 2025 winter wheat, Germany: TE17 −2.0% vs E15 −15.0%. λ for TE17 0.20-0.40 (as E15). Highest mean TFT variable-selection weights: root-zone soil moisture anomaly, VPD anomaly, crop-mask NDVI/EVI/LAI anomalies (`v17_tft_variable_weights.csv`). | Registered outcome. |
| 2026-10-11 | **P17 result (train 3476281, fine-tune 3476282, predict 3476998 on GPU; the CPU prediction job 3476283 was cancelled after 2 years because it ran ~6 min/year; same code, prediction moved to the GPU). 37,383 rows.** Hindcast 2010-2015: PE17 87.81% vs E15 88.48%, Δ skill −0.089 (CI −0.174..−0.036), P 0.001 → **v6 §6 not met.** 2016-2025 87.95 vs 88.38 (P 0.10). Without memory P17 vs Q12: 87.83 vs 88.32 / 87.58 vs 87.90. Reference GE16: 2010-2015 P 0.57, 2016-2025 87.95 vs 88.60. PatchTST is the weakest of the three transformers on this task. **No v17 arm replaces E15.** | Registered outcome. |
