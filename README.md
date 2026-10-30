# Transformer-based Crop Yield Prediction using Multivariate Timeseries Datasets

Praktikum project at the Chair for Computer Science IV (Computer Vision), Julius-Maximilians-Universität Würzburg.
Authors: Roshan Kumar Gupta, Manisha Samal, Sreedeepthi Ramanath.
Supervisor: Prof. Dr. Radu Timofte. Advisor: Dr. Fayaz Ali.

The project estimates the official yields of ten field crops in German districts (NUTS-3) at the end of each
season, before the statistics are published. The final model is called YieldFormer-DE (version v18). It averages
three transformer networks for multivariate time series, the Temporal Fusion Transformer (TFT), iTransformer and
PatchTST, and then corrects each district's usual over- or underestimate with a small error memory.

## Results in short

All numbers come from a rolling evaluation: each season from 2010 to 2025 is predicted by models trained only on
earlier seasons. This covers 37,383 district-crop-years in 363 districts.

| Forecast | Accuracy (100 minus MAPE) |
|---|---|
| YieldFormer-DE (v18) | 88.8 % |
| 5-year district average | 86.9 % |
| Long-term district normal | 86.4 % |

- YieldFormer-DE beats the 5-year average in 13 of 16 seasons. The gain is largest in poor seasons, for example
  11.6 percentage points in the 2018 drought.
- Retrained on the data splits of three published German studies (AgroForecast-DE, CropFusionNet, Srivastava et
  al.), the network ensemble wins 5 of 8 comparisons, loses 2 and ties 1. CropFusionNet stays clearly better for
  silage maize.
- For the 2025 season the national error is close to that of the EU's JRC MARS bulletins (6.1 % against 6.4 %).
  This is not a like-for-like comparison: our estimate uses weather up to harvest.
- Limits: with data cut at 30 June the model is no better than the 5-year average, and it underestimates very
  large losses in single districts. The evidence rests on only 16 test seasons.

## How the work was done

- Data: district yields from Destatis (Regionalstatistik table 41241-01-03-4), daily weather from DWD stations
  and ERA5-Land, soil water, MODIS vegetation indices over arable land (Google Earth Engine), DWD crop
  development stages, SoilGrids soil data and census crop areas (Duden, Nacke and Offermann 2024). In total 69
  daily inputs over each crop's season and 23 fixed inputs per district.
- Every change was planned in writing before it was run, with its decision rule. The plans and results of all
  20 versions are in `paper/preregistration_*.md`. Of 27 planned tests, three passed: arable-land satellite
  averages (v1), the district error memory (v15) and the three-network ensemble (v18).
- Inputs that would leak information from after the forecast date are excluded in `configs/data.yaml`.
- Training ran on NVIDIA L40 and L40S GPUs of the GPU cluster of the University of Würzburg, about 317 GPU
  hours for the final model. The three networks have 0.43, 0.22 and 1.27 million parameters.
- Forecasts for 2026, 2027 and 2028 were sealed before any official figure existed. The files and their SHA-256
  fingerprints are in `results/forecasts/`.

## Model and data downloads

- Model checkpoints (the three networks of YieldFormer-DE and version v20):
  https://huggingface.co/Roshangupta007/YieldFormer-DE
- Training data set (`data/processed/dataset.parquet` and `data/processed/static/`):
  https://huggingface.co/datasets/Roshangupta007/YieldFormer-DE-data

Put the data files in `data/processed/` and the checkpoints in `results/checkpoints/` (file names unchanged)
to re-run the evaluation without downloading the raw sources.
The dashboard and its tests also need the district shapes, which `python scripts/00_build_geo.py` downloads
from GISCO.

## Dashboard

A one-screen Streamlit dashboard shows the estimates, the sealed forecasts and all comparisons. Filters on the
left (crop, years, region, state, district, comparison, model) apply to every card and view. It runs on the
local machine only.

```bash
./dashboard/run.sh          # http://127.0.0.1:8502
```

The dashboard reads `results/dashboard/`, which `scripts/63_dashboard_data.py` rebuilds on every start. Its
contract is written down in `dashboard/SPEC.md` and checked by `tests/test_dashboard_app.py`. The earlier
multi-page dashboard is kept as a backup (`./dashboard/run_old.sh`, port 8503).

## Running the code

The project was developed with Python 3.9.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pytest tests/
```

Main steps of the pipeline:

| Step | Script |
|---|---|
| District shapes | `scripts/00_build_geo.py` |
| Download weather, satellite, soil and yield data (needs Copernicus CDS and Google Earth Engine accounts) | `scripts/01_download_data.py` |
| Build the data set | `scripts/02_preprocess.py` |
| Train a network for one season and seed | `scripts/03_train.py` |
| Fine-tune it per crop | `scripts/47_finetune_per_crop.py` |
| Evaluate the final ensemble | `scripts/74_eval_v18.py` |
| Comparison with the published studies | `scripts/75_paper_compare_v18.py` |
| Score the sealed forecasts once official yields are out | `scripts/37_forecast_scoreboard.py` |
| Computing cost from the cluster job logs | `scripts/79_compute_cost.py` |

The cluster jobs are the `scripts/sbatch_*.sh` files (SLURM arrays, one task per season and seed). Submit them
from the project root after `mkdir -p slurm_logs`, because SLURM opens the log files before the job starts. They
activate `.venv` (or the environment named in `VENV`). The `--partition` and `--qos` lines are those of the
Würzburg cluster; change them for another cluster.

## Repository layout

- `configs/`: data and model settings, including the list of excluded inputs
- `src/`: data pipeline, networks, training and evaluation code
- `scripts/`: numbered pipeline and evaluation scripts
- `results/metrics/`: evaluation results of all versions
- `results/forecasts/`: sealed forecast files with their fingerprints
- `results/dashboard/`: tables the dashboard reads
- `dashboard/`: the Streamlit app (`main.py`, `app.py`), its specification and launchers
- `paper/`: pre-registrations of all versions and paper drafts
- `tests/`: unit tests for the data windows, the networks, the error memory and the dashboard

The project started in summer 2026 as a benchmark of six architectures (XGBoost, DLinear, LSTM, PatchTST,
iTransformer, TFT). Those early results were produced before leaking inputs were removed and are kept only as
history; they are not results of this project.

## Data licences

The code is released under the MIT licence (see `LICENSE`). The source data keep their own licences: DWD and
SoilGrids under CC BY 4.0, Destatis under the Data Licence Germany (dl-de/by-2-0), ERA5-Land under the Copernicus
licence, MODIS data from NASA LP DAAC, census crop areas under CC BY 4.0, and the GISCO district boundaries
(EuroGeographics) with their own attribution terms, which do not allow commercial use. Please cite the data
providers when you use the data.

## Contact

Roshan Kumar Gupta, roshangupta00750@gmail.com
Julius-Maximilians-Universität Würzburg
