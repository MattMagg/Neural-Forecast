# Copilot instructions for this repo

Intraday BTC forecasting on 15‑minute bars using NeuralForecast (NF). Horizons h ∈ {4, 8, 16, 32}. NF‑native only (no custom backtester/serializer), UTC end‑of‑bar timestamps, and zero leakage via shift(1).

## What lives where
- Entrypoints: `run_train.py` (train/CV), `run_predict.py` (inference) — stubs to implement per spec.
- Features: `features/registry.py` (indicator + MTF specs, hist/futr/stat), `features/builder.py` (build + shift/prune).
- Models: `nf_models/factory.py` (NHITS/NBEATSx/TiDE/PatchTST + losses/scalers).
- CV/HPO: `cv/runner.py` (wrap NF.cross_validation), `cv/hpo.py`.
- UQ: `uq/diag.py` (coverage/PIT), `uq/ensembles.py` (simple blends).
- Utils: `utils/validate.py`, `utils/io.py`, `utils/version.py`.
- Experiments: `experiments/*.yaml`, outputs in `experiments/h{h}/` and `reports/h{h}/`.
- Specs: `docs/forecasting_sf_plan.md` (authoritative), helpers in `CLAUDE.md` and `AGENTS.md`. Versioning practices in `docs/versioning_system.md`.

## Data/feature contract (enforced patterns)
- Long format `[unique_id, ds, y, ...]` on a regular 15‑min UTC EOB grid. No gaps in training windows.
- Target is log return; never forward‑fill `y`.
- Exogs labeled: `hist_` → must be `shift(1)`; `futr_` → known ahead; `stat_` → static.
- Run guards before fit/predict: `assert_regular_grid`, `assert_utc_eob`, `assert_shifted`, `assert_no_forward_fill_y` (see `utils/validate.py`).

## NF usage (project‑specific)
- Use only NF primitives: `NeuralForecast.fit/predict/predict_insample/cross_validation`.
- Imports: models `NHITS, NBEATSx, TiDE, PatchTST`; losses `DistributionLoss("StudentT"), MQLoss, IQLoss`.
- Conformal: `from neuralforecast.utils import PredictionIntervals`; typical `level=[80,90,95]`.
- Input sizes: 1024 default, PatchTST often 2048. Prefer NF scalers (e.g., `robust`, PatchTST `revin`).

## CV, metrics, artifacts
- CV: `cross_validation(df, n_windows≈6→10, step_size=h, val_size=4*h, refit=True, prediction_intervals=PredictionIntervals(n_windows=6), level=[80,90,95])`.
- Metrics: primary sCRPS; also MAE/RMSE. Calibration gates: coverage at 80/90/95 within ±2pp; check PIT.
- Artifacts: `experiments/h{h}/cv_results.parquet`, `experiments/h{h}/metrics.csv`, `experiments/h{h}/best/` (via `nf.save(..., save_dataset=True)`), predictions/plots in `reports/h{h}/`.

## Conventions and examples
- Always shift(1) `hist_` features and cap final feature count ≤256 after pruning.
- Align 30m/1h/4h features to 15‑min EOB; forward‑fill then shift for `hist_`.
- Save/load: `nf.save("experiments/h16/best/", save_dataset=True)` ↔ `NeuralForecast.load("experiments/h16/best/")`.
- Commands (when scripts filled in): train `python run_train.py --exp experiments/h16.yaml --save`; predict `python run_predict.py --exp experiments/h16.yaml --model_path experiments/h16/best/ --h 16`.

Read first: `AGENTS.md` → `CLAUDE.md` → `docs/forecasting_sf_plan.md` → `docs/versioning_system.md` (for version bumps/tags/artifact naming). Implement stubs to match these contracts and keep artifact paths stable.
