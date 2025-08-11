> **Mission:** intraday BTC forecasting on 15-min bars with calibrated PIs, **NeuralForecast-first**. No reinvention.

---

## TL;DR non-negotiables

- **NF-centric**: Use `neuralforecast` for models, losses, scalers (incl. RevIN via `scaler_type`), `fit`, `predict`, `predict_insample`, `cross_validation`, **conformal** via `PredictionIntervals`, and `save/load`.
- **Conformal import**: `from neuralforecast.utils import PredictionIntervals` (✅), **not** from `.auto` (❌). Pass levels via `level=[...]` to `fit/cross_validation/predict`, **not** to `PredictionIntervals(...)`.
- **Leakage hygiene**: All *historic* features must be computed then **shifted by one bar** (`shift(1)`) before modeling. **Never** join contemporaneous values.
- **UTC EOB**: All timestamps are UTC end-of-bar; grids must be regular at `freq="15min"`.
- **Target**: log returns `y = log(close_t / close_{t-1})`. No forward-fill of `y`.
- **Horizons**: `h ∈ {4,8,16,32}` (1h/2h/4h/8h).
- **Primary metric**: **sCRPS** (with MAE/RMSE as supporting), plus coverage @ 80/90/95.
- **Feature cap**: ≤ **256** columns after pruning; availability ≥ **98%** post-shift; Spearman |ρ| prune at ≥ **0.95**.
- **Do not** write custom backtesters, custom scalers, or ad-hoc serializers.

---

## Repo structure (authoritative)

```

/data/                  # canonical frames (post-regularization)
/features/              # indicator registry + builders (vectorbt/TA-Lib primary; pandas-ta-openbb & freqtrade/technical supplement)
/nf\_models/             # NF model factories & common args
/cv/                    # NF cross\_validation orchestrations
/experiments/           # YAML configs & results per horizon
/uq/                    # uncertainty & calibration (PIT, coverage, ensembles)
/reports/               # generated summaries & acceptance reports
/utils/                 # IO, validation, versioning, tail builders
run\_train.py            # training & CV entrypoint (NF-native)
run\_predict.py          # inference entrypoint (batch/loop)

```

Do **not** change the layout without a plan PR.

---

## Golden rules (what to write / avoid)

### Data contracts
- Canonical long frame columns: `["unique_id","ds","y", <exog...>]` with `ds` tz-aware UTC.
- Regularize to a strict 15-min grid **before** computing `y`.
- Assertions to always call somewhere in train/infer flows:
  - `assert_regular_grid(df,"15min")`
  - `assert_utc_eob(df,"15min")`
  - `assert_no_forward_fill_y(df)`
  - `assert_shifted(df, hist_exog_cols)`

### Features
- **Primary**: vectorbt + TA-Lib wrappers.  
  **Supplement**: pandas-ta-openbb (pure-Python/numba), freqtrade/technical for **MTF** resample/merge (30m/1h/4h).
- MTF: align to **EOB**, forward-fill to 15-min, then **shift(1)**.
- Prune: availability ≥98%, cap ≤256, |ρ|≥0.95 cluster prune, simple MI/variance screen.

### Modeling (NF only)
- Portfolio: **NHITS**, **NBEATSx**, **TiDE**, **PatchTST**.
- Losses: **DistributionLoss("StudentT")**, **MQLoss**, **IQLoss**.
- Scalers: `scaler_type="robust"` default; consider `"revin"` if drift; only pass ctor kwargs the model supports.
- Input windows: NHITS/NBEATSx/TiDE **1024**; PatchTST **2048**.
- Conformal: `from neuralforecast.utils import PredictionIntervals`; attach via `prediction_intervals=PredictionIntervals(...)`.

### Cross-validation (NF)
- `n_windows`: 6 (pilot) → 10 (final); `step_size=h`; `val_size=4*h`; `refit=1`.
- `predict_insample(step_size=1)` for PIT/diagnostics (not on conformalized models).

### HPO (tight, low-variance)
- Global knobs: `learning_rate ∈ {1e-3, 5e-4}`, `batch_size ∈ {256,512,1024}`, patience 200–400.
- Pilot h=16 → promote top configs → run full CV on {4,8,16,32}.
- Stop when mean sCRPS gains < ~1–2%.

### Selection & ensemble
- Select by mean **sCRPS**; tie-break by coverage@90 and sCRPS std.
- Optional **ENS2**: equal-weight top-2; average quantiles/intervals level-wise; never average distribution params.

### Uncertainty & calibration
- Prefer Student-t or MQ; switch to **IQLoss** if quantiles cross.
- Conformal if coverage miss > ±2pp at 80/90/95.
- Diagnostics: coverage tables, PIT ~ Uniform, coverage by volatility deciles.

### Save/Load
- `nf.save(path, save_dataset=True)` / `NeuralForecast.load(path)`. No `pickle`/`torch.save`.

---

## Commands (copy/paste)

Train + CV (example h=16):
```bash
python run_train.py --exp experiments/h16.yaml --save
````

Inference one-shot:

```bash
python run_predict.py --exp experiments/h16.yaml --model_path experiments/h16/best/ --h 16
```

Acceptance:

```bash
python reports/acceptance.py --hdir experiments/h16 --winner ENS2 --baseline_scrps 0.1234
```

---

## PR checklist (must pass)

* [ ] No leakage: all **hist** exogs verified shifted (`assert_shifted` green).
* [ ] Conformal import is correct: `from neuralforecast.utils import PredictionIntervals`.
* [ ] NF APIs only: `fit`, `predict`, `predict_insample`, `cross_validation`, `save/load`.
* [ ] CV args follow policy (`step_size=h`, `val_size=4*h`, `refit=1`).
* [ ] Features ≤ 256, availability ≥98%, EOB alignment unit test passes.
* [ ] Leaderboard includes sCRPS for finalists (dense quantiles run if needed).
* [ ] Acceptance report for each horizon: **ACCEPT** or justified waiver.
* [ ] Version manifest + lockfile updated on any dep change.
* [ ] No custom backtester/scaler/serializer introduced.

---

## Common pitfalls & exact fixes

* **Wrong conformal import/usage** → Fix:

  ```python
  from neuralforecast.utils import PredictionIntervals
  pi = PredictionIntervals(n_windows=cfg["n_windows"])
  cv = nf.cross_validation(..., prediction_intervals=pi, level=[80,90,95])
  ```
* **Intervals too narrow (under-coverage)** → Attach conformal; consider `scaler_type="revin"`; or switch MQ → IQLoss.
* **OOM/slow** → Lower `batch_size` → reduce PatchTST `hidden_size/n_heads` → shorten `input_size` (as last resort).
* **Predict < h rows** → Ensure `futr_df` has exactly `h` rows and matches `futr_exog_list`.

---

## Style guide (thin)

* Python: black/ruff defaults; type hints; no global state in modules.
* YAML: snake\_case keys; comments for any non-obvious knob.
* Logging: concise, actionable; log params and shapes, not tensors.
