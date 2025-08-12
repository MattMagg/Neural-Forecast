# Practical, NF-Centered Software Implementation Plan

*Intraday BTC forecasting (15-minute base) → up to 8-hour horizons, accuracy-first, production-pragmatic.*

## Table of Contents

- [0) Objectives & guardrails](#0-objectives-guardrails)
  - [0.1 Target, frequency, horizons (why and how)](#01-target-frequency-horizons-why-and-how)
  - [0.2 Bar finalization & time ordering](#02-bar-finalization-time-ordering)
  - [0.3 Leakage discipline (non-negotiable)](#03-leakage-discipline-non-negotiable)
  - [0.4 Cross-validation semantics (so you don’t reinvent it later)](#04-cross-validation-semantics-so-you-dont-reinvent-it-later)
  - [0.5 Probabilistic forecasts, calibration targets, and metrics](#05-probabilistic-forecasts-calibration-targets-and-metrics)
  - [0.6 Scaling/normalization policy (NF-native only)](#06-scalingnormalization-policy-nf-native-only)
  - [0.7 Determinism and hygiene](#07-determinism-and-hygiene)
  - [0.8 Non-negotiable guardrails (checklist to enforce in CI)](#08-non-negotiable-guardrails-checklist-to-enforce-in-ci)
- [1) Repository layout (lean, explicit)](#1-repository-layout-lean-explicit)
  - [1.1 Directory scaffold (one-shot)](#11-directory-scaffold-one-shot)
  - [1.2 File inventory & responsibilities (don’t deviate)](#12-file-inventory-responsibilities-dont-deviate)
  - [1.3 Naming & artifact conventions (uniform, predictable)](#13-naming-artifact-conventions-uniform-predictable)
  - [1.4 Coding standards (tight and boring)](#14-coding-standards-tight-and-boring)
  - [1.5 Bootstrap stubs (drop-in file skeletons)](#15-bootstrap-stubs-drop-in-file-skeletons)
  - [1.6 Minimal `settings.yaml` (drop-in seed; you’ll extend in §9.1)](#16-minimal-settingsyaml-drop-in-seed-youll-extend-in-91)
  - [1.7 Entry-point skeletons (wire later sections here)](#17-entry-point-skeletons-wire-later-sections-here)
- [2) Data contracts & validation](#2-data-contracts-validation)
  - [2.1 Canonical target frame](#21-canonical-target-frame)
  - [2.2 Regularization & data hygiene](#22-regularization-data-hygiene)
  - [2.3 Target and splits](#23-target-and-splits)
  - [2.4 Canonical NF frame (schema, target, coercions)](#24-canonical-nf-frame-schema-target-coercions)
  - [2.5 Regularization policy (missing bars, NaNs, winsorization)](#25-regularization-policy-missing-bars-nans-winsorization)
  - [2.6 Deterministic seeding & data checks (hard asserts)](#26-deterministic-seeding-data-checks-hard-asserts)
  - [2.7 Assembly path (from raw → NF-ready)](#27-assembly-path-from-raw-nf-ready)
  - [2.8 Minimal integration diff (wire into §1 stubs)](#28-minimal-integration-diff-wire-into-1-stubs)
  - [2.9 Why this is correct (and safe)](#29-why-this-is-correct-and-safe)
- [3) Exogenous features (indicators and others)](#3-exogenous-features-indicators-and-others)
  - [3.1 Indicator registry (vectorbt + TA-Lib primary; pandas-ta-openbb supplement)](#31-indicator-registry-vectorbt-ta-lib-primary-pandas-ta-openbb-supplement)
  - [3.2 Feature builder (compute → align → `shift(1)`)](#32-feature-builder-compute-align-shift1)
  - [3.3 Feature selection & hard cap (≤ 256 features)](#33-feature-selection-hard-cap-256-features)
  - [3.4 End-to-end assembly (how Section 3 plugs into Section 2 & 9)](#34-end-to-end-assembly-how-section-3-plugs-into-section-2-9)
  - [3.5 NF wiring (this is where exogs are actually used)](#35-nf-wiring-this-is-where-exogs-are-actually-used)
  - [3.6 Hygiene & boundary cases you must enforce](#36-hygiene-boundary-cases-you-must-enforce)
  - [3.7 Minimal tests (fast, not fluffy)](#37-minimal-tests-fast-not-fluffy)
- [4) NeuralForecast model portfolio & defaults](#4-neuralforecast-model-portfolio-defaults)
  - [4.0 Portfolio justification (BTC intraday, 15-min)](#40-portfolio-justification-btc-intraday-15-min)
  - [4.1 Common model parameters and training settings](#41-common-model-parameters-and-training-settings)
  - [4.2 Model-specific configurations](#42-model-specific-configurations)
- [5) Cross-validation (NF-native) & avoiding leakage](#5-cross-validation-nf-native-avoiding-leakage)
  - [5.1 Windowing (per horizon) — exact NF arguments](#51-windowing-per-horizon-exact-nf-arguments)
  - [5.2 Metrics, outputs, and artifacts (NF-produced → thin glue only)](#52-metrics-outputs-and-artifacts-nf-produced-thin-glue-only)
    - [5.2.A What NF returns (you will aggregate, not recompute)](#52a-what-nf-returns-you-will-aggregate-not-recompute)
    - [5.2.B Primary metric: **sCRPS** (with MAE/RMSE as supporting)](#52b-primary-metric-scrps-with-maermse-as-supporting)
    - [5.2.C Coverage & PIT diagnostics](#52c-coverage-pit-diagnostics)
    - [5.2.D Drop-in runner and aggregator (zero reinvention, thin glue)](#52d-drop-in-runner-and-aggregator-zero-reinvention-thin-glue)
    - [5.2.E PIT helper (diagnostic only; run on promoted configs)](#52e-pit-helper-diagnostic-only-run-on-promoted-configs)
    - [5.2.F How to wire this into your training flow (exact edits)](#52f-how-to-wire-this-into-your-training-flow-exact-edits)
    - [5.2.G Leakage discipline (assertions you **must** keep)](#52g-leakage-discipline-assertions-you-must-keep)
    - [5.2.H When to attach **Conformal** (keep it minimal)](#52h-when-to-attach-conformal-keep-it-minimal)
    - [5.2.I Save/Load (for reproducibility & later inference)](#52i-saveload-for-reproducibility-later-inference)
- [6) Hyperparameter strategy (bounded, low-variance)](#6-hyperparameter-strategy-bounded-low-variance)
  - [6.1 Search spaces (tight, model-specific + global)](#61-search-spaces-tight-model-specific-global)
  - [6.2 Procedure: pilot → promote → full CV (no bloat)](#62-procedure-pilot-promote-full-cv-no-bloat)
  - [6.3 Concrete glue (tiny, no reinvention)](#63-concrete-glue-tiny-no-reinvention)
  - [6.4 Promotion thresholds & bookkeeping](#64-promotion-thresholds-bookkeeping)
  - [6.5 Practical guards (don’t ignore)](#65-practical-guards-dont-ignore)
  - [6.6 (Optional) Using NF Auto* for time-boxed HPO (only if needed)](#66-optional-using-nf-auto-for-time-boxed-hpo-only-if-needed)
- [7) Model selection & simple ensembling](#7-model-selection-simple-ensembling)
  - [7.1 Selection protocol (per horizon)](#71-selection-protocol-per-horizon)
  - [7.2 Simple ensembles (top-2 only)](#72-simple-ensembles-top-2-only)
  - [7.3 Drop-in utilities](#73-drop-in-utilities)
  - [7.4 Workflow (per h)](#74-workflow-per-h)
  - [7.5 Final fit & save (single vs ensemble)](#75-final-fit-save-single-vs-ensemble)
  - [7.6 Inference path for ensembles (hook for §10)](#76-inference-path-for-ensembles-hook-for-10)
  - [7.7 Guardrails](#77-guardrails)
- [8) Uncertainty & calibration](#8-uncertainty-calibration)
  - [8.1 Quantile vs Distribution training](#81-quantile-vs-distribution-training)
  - [8.2 Conformal prediction intervals](#82-conformal-prediction-intervals)
  - [8.3 Diagnostic checks](#83-diagnostic-checks)
  - [8.4 Practical fixes (pick the smallest hammer)](#84-practical-fixes-pick-the-smallest-hammer)
  - [8.5 What to persist for calibration](#85-what-to-persist-for-calibration)
- [9) Training & evaluation workflow](#9-training-evaluation-workflow)
  - [9.1 Experiment configuration](#91-experiment-configuration)
  - [9.1 YAML-driven experiment configs (minimal, explicit)](#91-yaml-driven-experiment-configs-minimal-explicit)
  - [9.2 `run_train.py` — NF-native training, insample diagnostics, CV, artifacts](#92-run_trainpy-nf-native-training-insample-diagnostics-cv-artifacts)
  - [9.3 `run_predict.py` — batch inference with PIs, save/load](#93-run_predictpy-batch-inference-with-pis-saveload)
  - [9.4 Artifacts & file layout (enforced)](#94-artifacts-file-layout-enforced)
  - [9.5 Make it hard to shoot yourself in the foot](#95-make-it-hard-to-shoot-yourself-in-the-foot)
- [10) Inference & live deployment considerations](#10-inference-live-deployment-considerations)
  - [10.1 What happens every 15 minutes (sequence)](#101-what-happens-every-15-minutes-sequence)
  - [10.2 Tail builders (drop-in utilities)](#102-tail-builders-drop-in-utilities)
  - [10.3 `run_predict.py` (inference entrypoint) — one-shot & loop modes](#103-run_predictpy-inference-entrypoint-one-shot-loop-modes)
  - [10.4 Throughput & memory guards (graceful degradation)](#104-throughput-memory-guards-graceful-degradation)
  - [10.5 Live conformal & monitoring hooks](#105-live-conformal-monitoring-hooks)
  - [10.6 Minimal assertions in the live loop (don’t skip)](#106-minimal-assertions-in-the-live-loop-dont-skip)
  - [10.7 Integration points (where this plugs into the rest)](#107-integration-points-where-this-plugs-into-the-rest)
- [11) Maintenance & retraining](#11-maintenance-retraining)
  - [11.1 Cadence & triggers (what makes us retrain)](#111-cadence-triggers-what-makes-us-retrain)
  - [11.2 Versioning & pinning (don’t be sloppy)](#112-versioning-pinning-dont-be-sloppy)
  - [11.3 Smoke tests (fast, decisive)](#113-smoke-tests-fast-decisive)
  - [11.4 Drift & monitoring (simple, actionable)](#114-drift-monitoring-simple-actionable)
  - [11.5 Retrain procedure (no drama)](#115-retrain-procedure-no-drama)
  - [11.6 Rollback (pre-wired, zero doubt)](#116-rollback-pre-wired-zero-doubt)
  - [11.7 Optional, tightly-scoped siblings (only where they add value)](#117-optional-tightly-scoped-siblings-only-where-they-add-value)
  - [11.8 House rules (so we don’t regress)](#118-house-rules-so-we-dont-regress)
- [12) Acceptance criteria & quality gates](#12-acceptance-criteria-quality-gates)
  - [12.1 Hard pass/fail gates (per horizon h)](#121-hard-passfail-gates-per-horizon-h)
  - [12.2 Rollback criteria (live)](#122-rollback-criteria-live)
  - [12.3 Acceptance report (one command)](#123-acceptance-report-one-command)
  - [12.4 What exactly to store](#124-what-exactly-to-store)
  - [12.5 If it fails — smallest hammer first](#125-if-it-fails-smallest-hammer-first)
- [13) Implementation checklist](#13-implementation-checklist)
  - [Parallelization guide (practical)](#parallelization-guide-practical)
  - [Command quick sheet (copy/paste)](#command-quick-sheet-copypaste)
  - [Definition of Done (v1)](#definition-of-done-v1)
- [14) Risks & mitigations](#14-risks-mitigations)
  - [14.1 MTF misalignment (30m/1h/4h → 15m)](#141-mtf-misalignment-30m1h4h-15m)
  - [14.2 Leakage from non-shifted historic exogs](#142-leakage-from-non-shifted-historic-exogs)
  - [14.3 Target mishandling (log-returns)](#143-target-mishandling-log-returns)
  - [14.4 Quantile crossing / bad calibration](#144-quantile-crossing-bad-calibration)
  - [14.5 GPU OOM / slow inference](#145-gpu-oom-slow-inference)
  - [14.6 Training instability (loss spikes/NaNs)](#146-training-instability-loss-spikesnans)
  - [14.7 Bad data (gaps/dupes/tz drift)](#147-bad-data-gapsdupestz-drift)
  - [14.8 sCRPS not computed / misleading leaderboard](#148-scrps-not-computed-misleading-leaderboard)
  - [14.9 Ensemble miscalibration](#149-ensemble-miscalibration)
  - [14.10 Save/Load & version drift](#1410-saveload-version-drift)
  - [14.11 Timezone or calendar mistakes](#1411-timezone-or-calendar-mistakes)
  - [14.12 Overfitting via feature bloat](#1412-overfitting-via-feature-bloat)
  - [14.13 Horizon mismatch / empty predictions](#1413-horizon-mismatch-empty-predictions)
  - [14.14 Live loop race conditions](#1414-live-loop-race-conditions)
  - [14.15 Regression after “harmless” refactors](#1415-regression-after-harmless-refactors)


---

## 0) Objectives & guardrails

* **Primary objective:** High-quality *probabilistic* forecasts for BTC with calibrated **80%/90%/95%** prediction intervals, minimal data leakage, and repeatable training/evaluation. Forecast distributions (not just point estimates) are needed to quantify uncertainty.

* **NF-centric rule:** Use **Nixtla’s NeuralForecast** (NF) library primitives wherever possible: deep models, distribution/quantile losses, built-in scalers, built-in cross-validation, conformal interval calibration, model persistence, etc. No custom re-implementations of features that NF already provides.

* **Data discipline:** Use **UTC** timestamps with each 15-min bar labeled by end-of-bar time. Regularize timestamps to a continuous 15-min grid (no missing intervals). **No forward-filling of target (y)** for missing bars. All *derived* features must obey a strict **compute → shift(1)** rule before joining with the target to avoid any lookahead leakage. If a feature can’t be lagged (e.g. future calendar info), treat it as future-known exogenous.

* **Base frequency:** **15-minute** bars (freq="15min" in NF terms).

* **Forecast horizons (h):** \[4, 8, 16, 32\] steps ahead (equivalent to 1h, 2h, 4h, 8h into the future at 15-min frequency).

* **Context window (input_size):** Begin with **1024** past observations (≈10.7 days) for models like NHITS, NBEATSx, and TiDE; and **2048** for PatchTST (which benefits from longer context). Only increase these if validation metrics (especially sCRPS) improve appreciably – long histories cost more compute and risk overfitting.

### 0.1 Target, frequency, horizons (why and how)

* **Target = log returns (default)**

  * Definition: `y_t = log(close_t) - log(close_{t-1})`.
  * Rationale: closer to stationary, reduces scale drift, and plays well with NF’s distributional losses (e.g., Student-t) and sCRPS evaluation. Distributional training is first-class in NF via `DistributionLoss`, with Student-t and other families supported; sCRPS is available for probabilistic evaluation.
  * Presentation: if a downstream consumer requires **price** forecasts, exponentiate the cumulative forecasted returns from the last known price only at presentation time (never in training/CV).

* **Base frequency:** `freq="15min"` globally (NF `NeuralForecast` core uses a pandas-compatible frequency string).

* **Horizons `h`:** `[4, 8, 16, 32]` → 1h, 2h, 4h, 8h ahead. Keep one model portfolio per `h` to respect horizon-specific error structure. Use NF’s `h` argument on each model instance.

* **Context window (`input_size`):** start at `1024` bars for NHITS/NBEATSx/TiDE; `2048` for PatchTST. These are long enough to act as a de-facto embargo (see §0.4) and align with NF windowed models’ expectations.

**Deliverables to add:**

* In your plan’s **settings.yaml** (referenced later), add:

  ```yaml
  freq: "15min"
  horizons: [4, 8, 16, 32]
  target: "log_return"  # choices: log_return | price
  default_input_size:
    generic: 1024
    PatchTST: 2048
  ```

  (Place this under “global defaults” when you reach §9.1.)

### 0.2 Bar finalization & time ordering

* **Bar policy:** all timestamps are **UTC end-of-bar** (EOB). The 15-minute bar ending at `10:00:00+00:00` covers `[09:45, 10:00)`. Resampling/rollups must use `label='right', closed='right'` (you’ll enforce this in the feature builders). This keeps causal ordering unambiguous for `hist_exog_list`. (We’ll wire `futr_exog_list` separately.) 
* **Strict time ordering:** NF methods (`fit`, `cross_validation`, `predict`, `predict_insample`) respect chronological splits; you supply `val_size`, `n_windows`, `step_size`, and optionally `refit=True` to simulate live retrains.

**Deliverables to add:**

* In **utils/validate.py**, you will implement:

  * `assert_utc_eob(df, freq="15min")` – verifies tz-aware UTC timestamps on the exact 15-minute grid and no leading/trailing partials.
  * `assert_monotonic_grid(df)` – confirms strictly increasing `ds` with fixed step.
    (You’ll create these utilities in §14; this is a binding requirement here.)

### 0.3 Leakage discipline (non-negotiable)

* **Compute → `shift(1)` rule:** every *historic* derived feature (indicators, MTF aggregates) must be computed on raw OHLCV, aligned to EOB, **then shifted by one bar** before merging into the NF frame. If a feature cannot be shifted (e.g., minute-of-day), it belongs in `futr_exog_list` (future-known) or `stat_exog_list` (static). NF **explicitly distinguishes** `hist_exog_list`, `futr_exog_list`, `stat_exog_list`; misuse causes leakage and invalid evaluation.
* **No target forward-fill ever:** missing `y` rows are dropped from train windows; do not impute `y`. (Impute *exogs* only when they’re truly stateful and historical; calendar `futr_exog` must be complete by construction.)

**Deliverables to add:**

* In **features/postprocess** (later §11), implement one central call:

  * `postprocess_shift_and_prune(df_exog, shift=1)` — applies the one-bar shift to all `hist_exog` columns, leaves `futr_exog`/`stat_exog` untouched, and asserts no zero-lag overlap.

### 0.4 Cross-validation semantics (so you don’t reinvent it later)

* Use NF’s `cross_validation`:

  * **Windowing defaults** per `h`: `n_windows=6` (pilot; promote to 10 for final), `step_size=h` (non-overlapping horizons), `val_size=4*h`, `refit=True`. NF applies windows from the series **tail**, trains strictly on the past of each cutoff, and predicts the next `h`. Long `input_size` naturally acts as an embargo between windows.
* Use `predict_insample(step_size=h)` after `fit` for PIT/coverage diagnostics on train/val without building a custom backtester.

**Deliverables to add:**

* In your **experiments/\<h>.yaml** templates (later §9), set:

  ```yaml
  n_windows: 6      # 10 for final
  step_size: ${h}   # non-overlapping
  val_size: ${h}*4
  refit: true
  ```

### 0.5 Probabilistic forecasts, calibration targets, and metrics

* **Primary evaluation metric:** **sCRPS** (Scaled Continuous Ranked Probability Score). NF ships **`sCRPS`** for probabilistic evaluation and it aligns with quantile or distributional training. Use it as the selection metric; it rewards calibrated, sharp distributions.

* **Training losses (selection protocol):**

  * Start with **DistributionLoss('StudentT')** for robustness to heavy tails—common in intraday returns. NF provides distributional training and examples modeling the target with **Student’s t**.
  * Train a parallel **MQLoss**/**ISQF/IQLoss** variant for direct quantile estimation; pick by mean sCRPS and coverage stability on CV.

* **Prediction intervals (80/90/95):**

  * If training with a **point** or distributional loss and calibration is off, use NF’s **Conformal Prediction** wrapper (`PredictionIntervals` via `fit(..., prediction_intervals=...)`) and request levels at `predict(level=[80,90,95])`. No custom code.
  * Target empirical coverage within **±2%** on held-out test tail; if under-coverage persists, prefer conformal adjustment over ad-hoc hacks.

* **Diagnostics to compute every run:**

  * **Coverage** at 80/90/95 on validation windows and the test tail.
  * **PIT** histograms via `predict_insample(level=[...])` when using distributional training (PIT \~ Uniform if calibrated). (NF documents insample intervals; produce PIT from the returned CDF/quantiles as applicable.)

**Deliverables to add:**

* In **uq/diag.py** (later §14), stub the following:

  * `compute_coverage(df_preds, levels=[80,90,95]) -> pd.DataFrame`
  * `plot_pit(insample_df) -> Path`
  * `coverage_by_vol_decile(df_preds, df_ref_vol)` — to detect heteroscedastic failures.
* In **run_train.py** (later §9.2), after `fit`:

  * call `predict_insample(...)`, compute PIT & coverage, persist under `reports/<h>/`.

### 0.6 Scaling/normalization policy (NF-native only)

* **Temporal window normalization (`scaler_type`)** is handled **inside each model**. Set:

  * default `scaler_type="robust"`; trial `scaler_type="revin"` on PatchTST/NHITS if CV shows stability lift under drift. NF documents temporal normalization (`scaler_type`) vs. core time series scaling (`local_scaler_type`) and recommends temporal normalization in most applications. We rely on **model-level** scalers—no external scalers.
* **Do not** implement custom scalers; use NF’s supported list (`robust`, `robust-iqr`, `revin`, etc.) per docs.

**Deliverables to add:**

* In **experiments/\<h>.yaml** defaults:

  ```yaml
  scaler_type:
    default: robust
    PatchTST: revin
  local_scaler_type: null  # keep off unless you have a multi-id panel reason
  ```

### 0.7 Determinism and hygiene

* **Deterministic seeds** at the NF/model level for reproducibility.
* **Missing bars:** reindex to a full 15-minute UTC grid; see §2 for contract. Never forward-fill `y`. (NF’s data requirements: `["unique_id","ds","y", <exog>]` long format.)
* **Outliers:** winsorize returns at `[0.1%, 99.9%]` *for training only*; never mutate the reporting layer.

**Deliverables to add:**

* In **settings.yaml**, add:

  ```yaml
  seed: 1337
  winsor:
    lower_q: 0.001
    upper_q: 0.999
  ```

### 0.8 Non-negotiable guardrails (checklist to enforce in CI)

1. `assert_utc_eob` and `assert_monotonic_grid` pass on every run.
2. Every `hist_exog` shows `corr(exog_t, y_t) < corr(exog_t, y_{t+1})` sanity (proxy for correct shift). (You’ll compute this as a smoke test; spike in contemporaneous correlation = likely leakage.)
3. **NF CV only**: no homemade backtesters. Use `cross_validation`/`predict_insample`.
4. **Intervals reported only from NF**: distributional quantiles, `predict(level=[...])`, or conformal via `PredictionIntervals`.)

---

## 1) Repository layout (lean, explicit)


### 1.1 Directory scaffold (one-shot)

Create exactly these paths and sentinel files. Keep names stable; downstream scripts assume them.

```bash
# run once at repo root
mkdir -p data features nf_models cv experiments uq reports utils
touch features/__init__.py nf_models/__init__.py cv/__init__.py uq/__init__.py utils/__init__.py
# top-level entrypoints & config
touch run_train.py run_predict.py settings.yaml
```

**Rules**

* **No new top-level dirs** beyond what’s listed.
* Keep **snake_case** for files; **PascalCase** only for class names.
* Paths are relative to repo root in all code.

### 1.2 File inventory & responsibilities (don’t deviate)

Use this as the **source of truth** for what each module owns. LOC budgets are upper bounds; if you exceed them, you’re probably over-engineering.

| Path | Purpose (must-have functions/classes) | LOC cap |
|------|---------------------------------------|---------|
| `settings.yaml` | Global defaults (freq, horizons, seed, scaler defaults, winsor quantiles). **Only static config**; no secrets. | — |
| `run_train.py` | Orchestrates: load canonical frame → build features → instantiate NF models → `cross_validation` → metrics/plots → optional final `fit` → save artifacts. | 200 |
| `run_predict.py` | Loads saved NF model(s) → builds **tail** features → `predict(level=[80,90,95])` → optional conformal adjust → write outputs. | 150 |
| `utils/validate.py` | `assert_utc_eob(df, "15min")`, `assert_regular_grid(df,"15min")`, `assert_shifted(hist_cols)` smoke-test (no contemporaneous leaks). | 150 |
| `utils/io.py` | `load_canonical_frame(path)`, `save_parquet(df, path)`, `timestamped_path(base_dir, stem)`. | 120 |
| `features/registry.py` | Declarative registry of indicators & MTF specs (names, params, kind=`hist|futr|stat`, tf=`15min|30min|1h|4h`). | 150 |
| `features/builder.py` | `build_indicators(df_ohlcv, registry)`, `apply_mtf(df_ohlcv, registry)`, `postprocess_shift_and_prune(df_exog, rules)` (central **shift(1)**), `select_features(df, policy)` (cap ≤256). | 300 |
| `nf_models/factory.py` | `instantiate_models(cfg, hist_cols, futr_cols, stat_cols)` → returns list of configured **NF** models (NHITS, NBEATSx, TiDE, PatchTST) using NF's native args (`h`, `input_size`, `scaler_type`, `loss`, `hist_exog_list`, `futr_exog_list`, `stat_exog_list`). | 200 |
| `cv/runner.py` | `run_cv(nf, df, n_windows, step_size, val_size, refit)` → returns CV predictions DF. **NF-native** `cross_validation` only. | 120 |
| `uq/diag.py` | `compute_coverage(preds, levels=[80,90,95])`, `plot_pit(insample_df)`, `coverage_by_vol_decile(preds, ref_vol)`, `blend_equal(models_preds)` (simple mean/median). | 250 |
| `experiments/` | YAML configs per horizon (e.g., `h4.yaml`, `h8.yaml`, `h16.yaml`, `h32.yaml`), plus a `defaults.yaml`. (You'll define in §9.1). | — |
| `reports/` | Writable target for plots, tables, and exported predictions. | — |

**Hard bans**

* No custom backtester—**only** `NeuralForecast.cross_validation` & `predict_insample`. 
* No custom model persistence—**only** `nf.save()` / `NeuralForecast.load()`. 

### 1.3 Naming & artifact conventions (uniform, predictable)

* **Experiments:**

  * Directory per horizon: `experiments/h{h}/` (e.g., `experiments/h16/`).
  * Files:

    * CV predictions: `experiments/h{h}/cv_results.parquet`
    * CV metrics: `experiments/h{h}/metrics.csv`
    * Model save dir (best candidate): `experiments/h{h}/best/` → written via `nf.save(path, overwrite=True, save_dataset=True)`; later restored with `NeuralForecast.load(path)`. 
* **Reports:**

  * `reports/h{h}/pit_hist.png`, `reports/h{h}/coverage_table.csv`, `reports/h{h}/vol_decile_coverage.csv`, `reports/h{h}/forecast_plot.png`, `reports/h{h}/preds_latest.parquet`.
* **Models:** Model names in outputs must match config `name` (e.g., `NHITS-StudentT`). Don’t bake hyperparams into filenames; they live in YAML and saved model metadata.

### 1.4 Coding standards (tight and boring)

* **Imports:** `from neuralforecast import NeuralForecast`; models via `from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST`. Losses via `from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, ISQF, IQLoss`. 
* **Model interfaces:** wire `h`, `input_size`, `scaler_type`, `loss`, and exog lists (`hist_exog_list`, `futr_exog_list`, `stat_exog_list`) from config. (Exog hooks are per-model in NF, incl. NBEATSx/NHITS.) 
* **CV:** always call `nf.cross_validation(df=..., n_windows=..., step_size=..., refit=True, val_size=...)`. Store the returned long DF with columns `[unique_id, ds, cutoff, <ModelName...>, y]`. 
* **Insample diagnostics:** after `fit`, call `nf.predict_insample(step_size=h, level=[80,90,95])` for PIT and coverage scaffolding. 
* **Persistence:** save with `nf.save(path, save_dataset=True, overwrite=True)`; load with `NeuralForecast.load(path)`. Do not pickle manually. 

### 1.5 Bootstrap stubs (drop-in file skeletons)

> Create these exact stubs now; you’ll flesh them out in later sections.

**`utils/validate.py`**

```python
from typing import Sequence
import pandas as pd

def assert_regular_grid(df: pd.DataFrame, freq: str = "15min") -> None:
    ds = pd.DatetimeIndex(df["ds"])
    assert ds.is_monotonic_increasing, "ds must be strictly increasing"
    expected = pd.date_range(ds.min(), ds.max(), freq=freq, tz="UTC")
    assert (ds.tz is not None) and (str(ds.tz) == "UTC"), "ds must be tz-aware UTC"
    assert len(ds) == len(expected), "missing or extra bars vs regular grid"

def assert_utc_eob(df: pd.DataFrame, freq: str = "15min") -> None:
    # EOB means timestamps align exactly on the freq boundary
    ds = pd.DatetimeIndex(df["ds"])
    assert all(getattr(ts, "minute") % 15 == 0 for ts in ds), "non-EOB timestamps detected"

def assert_shifted(df: pd.DataFrame, hist_cols: Sequence[str]) -> None:
    # Sanity: hist exogs must have weaker contemporaneous vs next-step correlation with y
    # (rough, fast smoke-test against obvious leakage)
    import numpy as np
    y = df["y"].to_numpy()
    y_lead1 = np.roll(y, -1); y_lead1[-1] = np.nan
    for c in hist_cols:
        x = df[c].to_numpy()
        # simple nan-safe corr
        def corr(a, b):
            m = ~np.isnan(a) & ~np.isnan(b)
            return float(np.corrcoef(a[m], b[m])) if m.sum() > 3 else 0.0
        assert corr(x, y) < corr(x, y_lead1), f"Potential leakage in {c}: fix shift(1)"
```

**`utils/io.py`**

```python
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

def load_canonical_frame(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    # enforce schema here if needed; returns ["unique_id","ds","y",<exog?>]
    return df

def save_parquet(df: pd.DataFrame, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)

def timestamped_path(base_dir: str, stem: str, ext: str = "parquet") -> str:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    p = Path(base_dir) / f"{stem}_{ts}.{ext}"
    p.parent.mkdir(parents=True, exist_ok=True)
    return str(p)
```

**`nf_models/factory.py`**

```python
from typing import List
from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, ISQF, IQLoss

def _make_loss(spec: dict):
    kind = spec.get("loss", "StudentT")
    if kind == "StudentT":
        return DistributionLoss(distribution="StudentT")
    if kind == "MQLoss":
        return MQLoss(level=spec.get("level", [10,50,90]))
    if kind == "ISQF":
        return ISQF(level=spec.get("level", [10,50,90]))
    if kind == "IQLoss":
        return IQLoss(level=spec.get("level", [10,50,90]))
    raise ValueError(f"Unknown loss: {kind}")

def instantiate_models(cfg: dict, hist_cols: List[str], futr_cols: List[str], stat_cols: List[str]):
    models = []
    for m in cfg["models"]:
        name, body = next(iter(m.items()))
        cls = {"NHITS": NHITS, "NBEATSx": NBEATSx, "TiDE": TiDE, "PatchTST": PatchTST}[name]
        base = dict(
            h=cfg["h"],
            input_size=body.get("input_size", cfg.get("input_size", 1024)),
            hist_exog_list=hist_cols, futr_exog_list=futr_cols, stat_exog_list=stat_cols,
            scaler_type=body.get("scaler_type", cfg.get("scaler_type", "robust")),
            learning_rate=body.get("learning_rate", 1e-3),
            batch_size=body.get("batch_size", 512),
            max_steps=body.get("max_steps", 20000),
            early_stop_patience_steps=body.get("early_stop_patience_steps", 400),
        )
        loss = _make_loss(body.get("loss_spec", {"loss": body.get("loss","StudentT")}))
        models.append(cls(loss=loss, **base))
    return models
```

**`cv/runner.py`**

```python
import pandas as pd
from neuralforecast import NeuralForecast

def run_cv(nf: NeuralForecast, df: pd.DataFrame, n_windows: int, step_size: int, val_size: int, refit: bool) -> pd.DataFrame:
    return nf.cross_validation(df=df, n_windows=n_windows, step_size=step_size, val_size=val_size, refit=refit)
```

**`uq/diag.py`**

```python
import numpy as np, pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

def compute_coverage(preds: pd.DataFrame, levels=(80,90,95), lo_suffix="-lo-", hi_suffix="-hi-") -> pd.DataFrame:
    rows = []
    for col in preds.columns:
        for L in levels:
            lo = f"{col}{lo_suffix}{L}"
            hi = f"{col}{hi_suffix}{L}"
            if lo in preds and hi in preds and "y" in preds:
                covered = (preds["y"] >= preds[lo]) & (preds["y"] <= preds[hi])
                rows.append({"model": col, "level": L, "coverage": covered.mean()})
    return pd.DataFrame(rows)

def plot_pit(insample_df: pd.DataFrame, path: str) -> str:
    # expects a column 'pit' or compute it upstream for distributional runs
    if "pit" not in insample_df: return ""
    plt.figure()
    plt.hist(insample_df["pit"].dropna().values, bins=20, density=True)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path); plt.close()
    return path

def blend_equal(df_a: pd.DataFrame, df_b: pd.DataFrame, cols: list) -> pd.DataFrame:
    out = df_a[["unique_id","ds","y"]].copy()
    for c in cols:
        out[c] = 0.5*df_a[c] + 0.5*df_b[c]
    return out
```

### 1.6 Minimal `settings.yaml` (drop-in seed; you’ll extend in §9.1)

```yaml
freq: "15min"
horizons: [4, 8, 16, 32]
seed: 1337
winsor: { lower_q: 0.001, upper_q: 0.999 }

default_input_size:
  generic: 1024
  PatchTST: 2048

scaler_type:
  default: robust
  PatchTST: revin
```

### 1.7 Entry-point skeletons (wire later sections here)

**`run_train.py`**

```python
import yaml, pandas as pd
from utils.io import load_canonical_frame, save_parquet
from utils.validate import assert_regular_grid, assert_utc_eob, assert_shifted
from features.registry import REGISTRY  # you’ll define in §4
from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features
from nf_models.factory import instantiate_models
from neuralforecast import NeuralForecast
from cv.runner import run_cv

cfg = yaml.safe_load(open("experiments/h16.yaml"))  # example; parameterize via CLI later
df = load_canonical_frame("data/btc_15min.parquet")
assert_regular_grid(df, "15min"); assert_utc_eob(df, "15min")

# build features
exo_raw = build_indicators(df, REGISTRY)
exo_mtf = apply_mtf(df, REGISTRY)
exo = postprocess_shift_and_prune(exo_raw.join(exo_mtf, how="left"), rules={})
hist_cols, futr_cols, stat_cols = select_features(exo, policy={})

# assemble NF frame
nf_df = df.merge(exo, on="ds", how="left")
assert_shifted(nf_df, hist_cols)

models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)
nf = NeuralForecast(models=models, freq=cfg["freq"])

# NF-native CV
cv_df = run_cv(nf, nf_df, cfg["n_windows"], cfg["step_size"], cfg["val_size"], cfg["refit"])  # NF docs: cross_validation. 
save_parquet(cv_df, f"experiments/h{cfg['h']}/cv_results.parquet")
```

**`run_predict.py`**

```python
import yaml, pandas as pd
from utils.io import load_canonical_frame, save_parquet, timestamped_path
from features.registry import REGISTRY
from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features
from neuralforecast import NeuralForecast

cfg = yaml.safe_load(open("experiments/h16.yaml"))
df = load_canonical_frame("data/btc_15min.parquet")

exo = postprocess_shift_and_prune(build_indicators(df, REGISTRY).join(apply_mtf(df, REGISTRY), how="left"), {})
hist_cols, futr_cols, stat_cols = select_features(exo, {})
nf_df = df.merge(exo, on="ds", how="left")

# load the saved model (NF-native persistence) and predict. Docs show nf.save / NeuralForecast.load. 
nf = NeuralForecast.load(f"experiments/h{cfg['h']}/best")
pred = nf.predict(df=nf_df, level=[80,90,95])
save_parquet(pred, timestamped_path(f"reports/h{cfg['h']}", "preds"))
```

---

**Why this layout works (and is safe):**

* It **hard-codes NF primitives** (models, losses, scalers via `scaler_type`, cross-val, `predict_insample`, `save/load`) instead of rolling your own. 
* Exogenous integration is enforced at the **factory** layer with `hist_exog_list`, `futr_exog_list`, `stat_exog_list` (documented in NF model pages such as NBEATSx/NHITS). 
* No bespoke “framework”; each module tops out at \~100–300 LOC, which keeps complexity in check.

---

## 2) Data contracts & validation

### 2.1 Canonical target frame

Use a **long-format** DataFrame that NF expects, with columns: \["unique_id", "ds", "y", ...\]. In our case:

* **unique_id:** BTC-USD for this single series (string or category).

* **ds:** datetime64\[ns\] in UTC, marking the end of each 15-min interval.

* **y:** the target we want to forecast. **We choose log-returns** (log of close price relative to previous close) as the modeling target. This makes the series more stationary (de-trending price) and helps many models focus on fluctuations rather than absolute scale. If end users need price forecasts, we can always exponentiate the cumulative forecasts of returns to get price levels.

* (Potential additional columns for exogenous features as needed, see §3).

The dataset should cover a continuous timeline with no gaps in ds. We’ll parse the Kaggle “Bitcoin Historical Data” (assumed pre-loaded) into this schema, verifying we have a complete 15-min grid from start to end.

**Important:** No duplicate timestamps, and the timestamps must be strictly increasing. We will include an assert_utc_monotonic(df) utility to ensure this.

### 2.2 Regularization & data hygiene

Because crypto trades 24/7, we expect no regularly scheduled gaps (like weekends) – any gap is likely missing data that should be treated carefully:

* **Missing intervals:** If any 15-min bar is missing, insert it with y as NaN (since we can’t forward-fill returns or prices reliably). During training, NF will ignore those NaNs (by mask) or we drop them from the modeling DataFrame. Exogenous features for those intervals can be forward-filled if they are slow-moving (e.g., daily calendar features), but technical indicators should also be NaN for truly missing input data.

* **Outliers:** Extreme outlier returns (e.g., exchange glitches) can be winsorized to the 0.1%/99.9% range to prevent distortion of scale-sensitive models. This clipping is only on training data y (we will monitor if it improves stability). Actual y values remain unmodified for evaluation.

* **Scaling:** We rely on NF’s internal scalers for model inputs (see scaler_type in §4). We do **not** pre-standardize y globally, because NF will handle scaling per window/series as configured.

* **Deterministic splits:** We avoid any shuffling. Instead, we’ll use time-based splits for backtesting and holdout (see §5). Random seeds are set for reproducibility (e.g., PyTorch lightning under the hood, and any hyperparameter search procedures).

After preparing the DataFrame, add rigorous validation checks in utils/validate.py, for example:

assert_utc_monotonic(df)  
assert_regular_grid(df, freq="15min")

to ensure no timestamp irregularities.

### 2.3 Target and splits

Our training procedure uses expanding windows (time series cross-validation). We do *not* pre-partition a static train/val/test in the usual sense, but for clarity: - **Model training windows:** Will span from the beginning of data up to certain cutoff points, as determined by cross-validation (see §5). - **Validation** (for early stopping and hyperparameters): In NF, we specify val_size \= 4*h within each training window, so the last part of each training window is used for on-the-fly validation to trigger early stopping. - **Final evaluation (test):** After model selection, we will evaluate on the last n days (e.g., last 60–90 days not seen in any training window) to ensure performance holds on recent data. This can be done via an explicit test_size in NF’s cross-validation or by a separate call to predict on the tail portion.

All procedures respect time order: the model never trains on data from the future relative to the evaluation period.



### 2.4 Canonical NF frame (schema, target, coercions)

**Contract (non-negotiable):** NF expects **long format** with `["unique_id","ds","y", <exog...>]` where `ds` is a datetime index/column and `y` is numeric. This is the **only** schema you should feed into `NeuralForecast`. Source: NF “Cross-validation” & “Core” docs: *input is always long format with `unique_id, ds, y`* .

**BTC intraday target:** `y_t = log(close_t) - log(close_{t-1})` (log returns). Keep price for display only; never train on raw price.

**UTC & EOB:** All timestamps must be **tz-aware UTC** and aligned to **end-of-bar** (EOB) 15-minute boundaries (`...:00, :15, :30, :45`). This aligns with causal training and with MTF resampling practices you’ll use later.

**Drop-in: append to `utils/io.py`** (under existing imports):

```python
import numpy as np

def regularize_to_grid_utc(df_ohlcv: pd.DataFrame, freq: str = "15min") -> pd.DataFrame:
    """Return OHLCV on a strict UTC grid at EOB timestamps (no gaps; NaNs where data missing)."""
    df = df_ohlcv.copy()
    # Coerce timestamp column name if needed:
    ts_col = "ds" if "ds" in df.columns else "timestamp"
    if ts_col not in df.columns:
        raise ValueError("Expected a timestamp column 'ds' or 'timestamp'.")
    ds = pd.to_datetime(df[ts_col], utc=True)
    ds = ds.dt.tz_convert("UTC") if ds.dt.tz is not None else ds.dt.tz_localize("UTC")
    # Snap to EOB 15-min grid if slightly off (rare; guard against provider quirks)
    ds = ds.dt.floor(freq)  # we treat floor==EOB; bars should already end on boundary
    idx = pd.date_range(ds.min(), ds.max(), freq=freq, tz="UTC")
    base = pd.DataFrame(index=idx).rename_axis("ds").reset_index()
    # Heuristic: prefer columns: open, high, low, close, volume (case-insensitive)
    cols = {c.lower(): c for c in df.columns}
    keep = [cols.get(k) for k in ["open","high","low","close","volume"] if cols.get(k) in df.columns]
    df = df.assign(ds=ds)[["ds"] + keep].drop_duplicates(subset=["ds"]).sort_values("ds")
    out = base.merge(df, on="ds", how="left")
    return out

def make_nf_canonical(df_ohlcv_15m: pd.DataFrame, unique_id: str = "BTC-USD") -> pd.DataFrame:
    """Construct NF long frame with unique_id, ds (UTC), y=log-return, plus raw OHLCV (for features)."""
    df = regularize_to_grid_utc(df_ohlcv_15m, freq="15min")
    if "close" not in df.columns:
        raise ValueError("Expected 'close' column in OHLCV input.")
    # Target: log-returns
    y = np.log(df["close"]).diff()
    nf = df.assign(unique_id=unique_id, y=y)[["unique_id","ds","y","open","high","low","close","volume"]]
    return nf
```

**Validation hooks (will be called by §1 stubs):**

* `assert_regular_grid(df,"15min")` and `assert_utc_eob(df,"15min")` (already stubbed in §1.5).
* Add **missing bar** checks *before* modeling (see 2.2).
* Enforce `unique_id` present even for a single series (NF expects it). Docs reiterate unique_id is required and can be string/int/category .

### 2.5 Regularization policy (missing bars, NaNs, winsorization)

**Missing bars:**

* Create a **complete** 15-min UTC grid from min to max `ds`. Keep `y` as `NaN` where price is missing; **never** forward-fill `y`. NF consumes long-format frames; you may **drop NaN `y` rows** from each training window (NF masks invalid parts) but keep them in the master frame so your grid stays consistent .
* For **exogenous** features: forward-fill is acceptable **only** for *historic stateful* features (e.g., prior computed MTF indicator values) *after* shifting (see §3 and §4). Calendar `futr_exog` must be complete by construction.

**Winsorization (training only):**

* Clip `y` at `[0.1%, 99.9%]` quantiles on the **training subset** to avoid pathological spikes driving loss instability. Do **not** alter `y` in evaluation or reporting. Keep the original `y_raw` if you want to audit.

**Drop-in: append to `utils/io.py`**

```python
def drop_train_nans_and_winsorize(nf_df: pd.DataFrame, lower_q=0.001, upper_q=0.999) -> pd.DataFrame:
    """Return df where training rows (y non-null) are winsorized; NaN y kept for grid but dropped in model.fit/cv."""
    df = nf_df.copy()
    # Compute quantiles on non-NaN y
    ql, qu = df["y"].quantile(lower_q, interpolation="nearest"), df["y"].quantile(upper_q, interpolation="nearest")
    df["y_train"] = df["y"].clip(lower=ql, upper=qu)
    return df
```

You’ll use `y_train` only when fitting (by renaming to `y` in a copy); store both for audit.

**Validation:** Expand `utils/validate.py` to assert **no forward-fill of y** and **NaN handling** sanity:

```python
def assert_no_forward_fill_y(df: pd.DataFrame) -> None:
    # Adjacent equal y values are fine; this only checks you didn't fill missing y from future.
    # We conservatively assert that any originally missing y (where OHLCV close was NaN) remains NaN.
    # Implement by checking rows where 'close' is NaN imply y is NaN.
    if "close" in df.columns:
        bad = df["close"].isna() & df["y"].notna()
        assert not bad.any(), "Detected non-NaN y where close is NaN (forbidden forward-fill of target)."
```

### 2.6 Deterministic seeding & data checks (hard asserts)

**Random seeds:** Lock to a single integer (e.g., 1337) at script start and pass to NF (Lightning) if/when exposed. Determinism matters for reproducible CV and early stopping outcomes.

**Assert suite to run before any model call:**

1. `assert_regular_grid(df,"15min")` and `assert_utc_eob(df,"15min")` (grid, tz, boundary checks).
2. `assert_no_forward_fill_y(df)` (guard target hygiene).
3. `assert_shifted(df, hist_cols)` **after** you build features (see §4) to catch any leakage by verifying weaker contemporaneous vs. lead correlation.

**Where to call:** In `run_train.py` immediately after building the canonical frame (and again after exog merge). In `run_predict.py` before prediction.

**Doc links for data contract:** NF pages repeatedly state the required long-format schema `unique_id, ds, y` and using pandas dataframes for input; we align 100% with that (no custom structures) .

---

### 2.7 Assembly path (from raw → NF-ready)

Use this exact order in `run_train.py` and `run_predict.py`:

1. **Load raw OHLCV (15m)** → `utils.io.regularize_to_grid_utc` → strict EOB UTC grid (with NaNs where missing).
2. **Build canonical NF frame** → `utils.io.make_nf_canonical` (adds `unique_id`, computes `y=logret`).
3. **Winsorize (train-only)** → `utils.io.drop_train_nans_and_winsorize` (keep both `y` and `y_train`).
4. **Build exogs** (see §4) → indicators + MTF, then **`shift(1)`** (centralized in features postprocess).
5. **Merge exogs** → join on `ds`; **validate** with `assert_shifted`.
6. **Model I/O** → pass the long frame to NF; **do not** alter schema afterward.

*(We’ll wire the exact calls when we expand §9 “Training & evaluation workflow”.)*

---

### 2.8 Minimal integration diff (wire into §1 stubs)

**In `run_train.py` (replace the top of the script with):**

```python
import yaml, pandas as pd
from utils.io import load_canonical_frame, save_parquet, make_nf_canonical, drop_train_nans_and_winsorize
from utils.validate import assert_regular_grid, assert_utc_eob, assert_shifted, assert_no_forward_fill_y
# ... (rest unchanged)

cfg = yaml.safe_load(open("experiments/h16.yaml"))
raw = load_canonical_frame("data/btc_15min.parquet")
raw = regularize_to_grid_utc(raw, freq="15min")
assert_regular_grid(raw, "15min"); assert_utc_eob(raw, "15min")

nf_base = make_nf_canonical(raw, unique_id="BTC-USD")
assert_no_forward_fill_y(nf_base)

# features (built later in §4) -> exo
# exo = ...
# nf_df = nf_base.merge(exo, on="ds", how="left")
# assert_shifted(nf_df, hist_cols)
# (then CV as in §1.7)
```

**In `run_predict.py`** mirror the same base-building path before features/predict.

---

### 2.9 Why this is correct (and safe)

* **NF schema compliance:** matches Nixtla’s required long format (`unique_id, ds, y`) for `NeuralForecast` input; exogenous variables are simply **extra columns** on this long frame (and are referenced by name via `hist_exog_list`, `futr_exog_list`, `stat_exog_list`) .
* **Grid regularization:** ensures **no ragged edges** or silent gaps; NF cross-validation expects time-ordered frames—gaps can silently skew horizons if not normalized.
* **Target hygiene:** no forward-fill of `y`, winsorization only for training stability, not for evaluation.
* **Leakage control:** central `shift(1)` enforcement + `assert_shifted` post-merge means you will catch 99% of accidental future-look.
* **UTC EOB discipline:** prevents off-by-one bar errors when you add MTF features (we’ll use freqtrade’s `resample_to_interval`/`resampled_merge` later; they’re designed to build higher-TF features and merge down precisely).

---

## 3) Exogenous features (indicators and others)

Incorporating exogenous features is crucial for intraday BTC. We’ll use a layered approach:

**Primary libraries for features:** - **vectorbt \+ TA-Lib:** for fast, vectorized technical indicators calculated over the full series (leveraging TA-Lib’s C implementations under the hood). Vectorbt can wrap TA-Lib functions to produce indicator arrays efficiently. - **pandas-ta (OpenBB fork):** a pure Python (Numba-accelerated) technical analysis library. It offers a wide range of indicators and can fill gaps where TA-Lib or vectorbt might not have a specific indicator or flexibility. - **Freqtrade’s technical library:** specifically for multi-timeframe features. It provides utilities like resample_to_interval and resampled_merge to compute higher timeframe indicators and merge them into a base timeframe DataFrame in a forward-filled manner. This is safer and more convenient than manual Pandas merging, as it renames columns to avoid collision and ensures alignment.

We categorize exogenous features by how they align with the target: - **Historic exogenous (hist_exog_list):** features that are fully known up to the current time *and cannot peek into the future*. These might be technical indicators derived from past prices/volume. In NF, we will pass their column names via hist_exog_list so the models know these are only available historically. We will rigorously apply shift(1) to these features after computation, so that at time *t*, the feature value comes from data up to *t-1* (preventing leakage). - **Future exogenous (futr_exog_list):** features that are known for future times as well. Typical examples: calendar features (we know the day of week of future timestamps), or any planned event schedule, etc. These will be passed via futr_exog_list. NF will use them for conditioning forecasts since they are available for the forecast horizon. - **Static exogenous (stat_exog_list):** time-invariant features per series (for one series, this could be none, or something like an asset category or regime label). We likely won’t have meaningful static features for a single asset beyond an ID, but the pipeline will support it for future extensibility (e.g., if we later add ETH, we might include a static one-hot for asset class). Static features are passed via stat_exog_list.

**Feature candidates (tentative list):**

We will implement a **feature registry** in code to define and parameterize these. Some initial examples:

* *Trend/Momentum:* rsi(period=14), stoch_k%/d% (14,3), roc(window=16) for rate-of-change, moving averages (e.g., ma_fast=20, ma_slow=100 to generate golden-cross signals), MACD (with standard fast=12, slow=26, signal=9).

* *Volatility:* true range and ATR (atr(14)), rolling volatility of returns (e.g., std dev over 8, 32, 96-bar windows), Bollinger Band width (as percentage of price).

* *Volume & Order-flow:* volume moving averages, volume RSI, Chaikin Money Flow (cmf), Money Flow Index (mfi), on-balance volume, etc. Also volume imbalance or buy/sell volume if available (depending on dataset).

* *Price structure:* Bollinger Bands (we can use TA-Lib via vectorbt to get upper/lower bands; we might include just the percent-b deviation of price or band width as features), Donchian channels (e.g., 20-bar high/low to indicate recent range), and perhaps patterns like if close is highest in N bars.

* *Cross-asset or external:* (If allowed by data) maybe an external indicator like S\&P500 futures or USD index, but since not specified, we’ll assume only BTC data for now.

We will generate each indicator at the 15-min level *and* on higher aggregates: - **Multi-timeframe (MTF) features:** 30-min, 1-hour, and 4-hour frequencies. For each such timeframe, we’ll resample the OHLCV to that interval (aligned to the end of the bar) and compute a similar set of indicators. Then we bring them into the 15-min frame. For example, a 1-hour RSI or a 4-hour moving average can be very informative for the trend context. Using resample_to_interval (from freqtrade.technical.util) and resampled_merge, we can do this systematically. The merged columns will be named like resample_60_close, etc., which we’ll then suffix with the indicator name. **We must forward-fill** the higher timeframe values down to 15-min within each hour. By aligning to end-of-bar, a 1H indicator at 10:00 covers data through 09:00–10:00 and will be applied to the 10:00 timestamp row, then forward-filled for sub-intervals until 10:00. Finally, we shift everything by 1 base interval to avoid contemporaneous leaks.

Example: to get a 4-hour ATR into 15-min frame, compute ATR on 4H resampled candles, merge into base frame with fill, resulting in a column (say atr_4h) that is constant for each 4h block of rows and updates at the end of each 4h window. After shifting by 1 (15 min), at time 12:00 the model can use the ATR that was computed on data up to 11:59 (covering 8:00-12:00 block if 4h ATR).

* **Future-known (Calendar) features:** Crypto trades continuously, but we can still derive features like day_of_week (0=Monday,...6=Sunday) – perhaps weekends behave differently (lower volumes or different volatility regimes) even though trading is open. We’ll include binary flags like is_weekend and possibly holiday indicators (for major holidays that might affect trading behavior, though crypto is global and holiday impact is less direct). Other calendar features: minute of day or hour of day as a cyclical feature (we can encode via sin/cos or as categorical). These are **future exogenous** because we know the calendar in advance for any future timestamp. They will be passed in futr_exog_list and do **not** need shifting (NF knows they are available for future times).

* **Static features:** The only static feature for now is something trivial like asset_id. For a single series, it’s constant, so it won’t add information; but to keep the design general, we include it (e.g., asset_id \= 1 for BTC or one-hot encoded as \["asset_BTC"\]). If we expand to multiple assets, this becomes useful (one model can handle multiple series with static identifiers). Static features go via stat_exog_list.

After assembling all candidate features, we will **prune aggressively** to avoid overfitting and ensure the model training remains efficient: - Drop any feature column that is more than \~2% NaN over the history *after* shifting (features with too many missing values are unreliable). - Drop features that are nearly constant or have extremely low variance. - Check pairwise correlations (Spearman rank) among features and remove highly collinear ones (e.g., if two momentum indicators at slightly different windows correlate 0.99, keep one). - Optionally, compute mutual information or simple feature importance by training a quick decision tree on a rolling window and remove obviously useless features. But given deep models can do feature selection implicitly, we focus on removing redundant features and keeping total count manageable (target: no more than 100-200 features after one-hot expansions, which is well within NF’s capability).

Finally, all chosen feature columns that represent past-known quantities will be tagged in hist_exog_list, and all future-known ones in futr_exog_list when initializing models.

Perfect — expanding the **next unit only**. Paste the block below **directly under your existing “## 3) Exogenous variables (exo-gen) strategy”** heading. I’ll stop after this unit for your **proceed / revise**.

---


> Paste this entire block under **“## 3) Exogenous variables (exo-gen) strategy”**. It expands your section into a production-ready indicator registry, multi-timeframe (MTF) pipeline, strict `shift(1)` hygiene, and pruning. All implementations stay inside **NF’s exog lists**: `hist_exog_list`, `futr_exog_list`, `stat_exog_list` (documented on NF model pages and exogenous-vars guide). 

### 3.1 Indicator registry (vectorbt + TA-Lib primary; pandas-ta-openbb supplement)

**Why this stack (brief):**

* **vectorbt** wraps TA-Lib & Pandas-TA with an indicator engine that **broadcasts parametrized grids** and **handles DataFrames natively**; `IndicatorFactory.from_talib()` auto-wires inputs/params/outputs, enabling **cartesian product parameter sweeps** (fast, vectorized).
* **pandas-ta-openbb** is **Numba-accelerated** pure-Python TA with 130+ indicators; use it to fill gaps or when TA-Lib lacks a variant.

**Drop-in file:** `features/registry.py`
Create a **declarative registry** describing what to compute, parameter grids, kind (`hist|futr|stat`), and timeframe (`15min|30min|1h|4h`). Keep it **small** first; expand only if sCRPS drops materially.

```python
# features/registry.py
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Tuple, Any

Kind = Literal["hist","futr","stat"]
TF = Literal["15min","30min","1h","4h"]

@dataclass
class IndicatorSpec:
    name: str                        # column stem, e.g. "rsi"
    lib: Literal["talib","pandas_ta","custom"]
    func: str                        # e.g. "RSI" for TA-Lib, "rsi" for pandas_ta
    params: Dict[str, List[Any]]     # cartesian grid; vectorbt will broadcast when lib=talib
    inputs: Tuple[str, ...]          # source columns, e.g. ("close",) or ("high","low","close")
    kind: Kind                       # "hist" or "futr" or "stat"
    tf: TF = "15min"                 # base timeframe unless specified
    post: Optional[str] = None       # optional post-processing: e.g., "bandwidth", "zscore"

# Pragmatic starter set (tight, not bloated).
REGISTRY: List[IndicatorSpec] = [
    # --- Momentum / Trend ---
    IndicatorSpec(name="rsi", lib="talib", func="RSI",
                  params={"timeperiod":[7,14,28]}, inputs=("close",), kind="hist"),
    IndicatorSpec(name="roc", lib="talib", func="ROC",
                  params={"timeperiod":[4,8,16,32]}, inputs=("close",), kind="hist"),
    IndicatorSpec(name="stoch_k", lib="talib", func="STOCH",
                  params={"fastk_period":[14], "slowk_period":[3], "slowd_period":[3]},
                  inputs=("high","low","close"), kind="hist"),
    IndicatorSpec(name="macd", lib="talib", func="MACD",
                  params={"fastperiod":[8,12], "slowperiod":[21,26], "signalperiod":[9]},
                  inputs=("close",), kind="hist"),

    # --- Volatility ---
    IndicatorSpec(name="atr", lib="talib", func="ATR",
                  params={"timeperiod":[8,16,32]}, inputs=("high","low","close"), kind="hist"),
    IndicatorSpec(name="nvol", lib="pandas_ta", func="stdev",
                  params={"length":[8,32,96]}, inputs=("close",), kind="hist"),

    # --- Volume / Flow ---
    IndicatorSpec(name="obv", lib="talib", func="OBV",
                  params={}, inputs=("close","volume"), kind="hist"),
    IndicatorSpec(name="mfi", lib="talib", func="MFI",
                  params={"timeperiod":[14]}, inputs=("high","low","close","volume"), kind="hist"),

    # --- Structure ---
    IndicatorSpec(name="bbands", lib="talib", func="BBANDS",
                  params={"timeperiod":[20], "nbdevup":[2], "nbdevdn":[2]},
                  inputs=("close",), kind="hist", post="bandwidth"),
    IndicatorSpec(name="donchian", lib="pandas_ta", func="donchian",
                  params={"lower_length":[20], "upper_length":[20]}, inputs=("high","low"), kind="hist"),

    # --- Calendar (future-known -> futr_exog) ---
    IndicatorSpec(name="minute_of_day", lib="custom", func="minute_of_day",
                  params={}, inputs=("ds",), kind="futr"),
    IndicatorSpec(name="day_of_week", lib="custom", func="day_of_week",
                  params={}, inputs=("ds",), kind="futr"),
    IndicatorSpec(name="is_weekend", lib="custom", func="is_weekend",
                  params={}, inputs=("ds",), kind="futr"),
]

# MTF policy: compute these also on higher TFs and merge down (EOB-aligned) later.
MTF_TARGETS: List[Tuple[TF, List[str]]] = [
    ("30min", ["rsi","roc","atr","bbands"]),
    ("1h",    ["rsi","roc","atr","bbands","macd"]),
    ("4h",    ["rsi","atr","bbands"]),
]
```

**Notes:**

* vectorbt’s **TA-Lib wrappers** accept param arrays and broadcast **cartesian combos** across columns **fast**, ideal for grid-friendly indicators.
* Use **pandas-ta-openbb** only when TA-Lib lacks a needed variant (e.g., Donchian). It’s **Numba-accelerated** and works directly on Series/DataFrames.

### 3.2 Feature builder (compute → align → `shift(1)`)

**Drop-in file:** `features/builder.py`
Implements **three** stages:

1. **Compute** base-TF indicators (15m) via vectorbt (TA-Lib) and pandas-ta; minimal post-processing.
2. **Apply MTF** resampling & merge using **freqtrade/technical** helpers: `resample_to_interval` and `resampled_merge` to **resample to 30m/1h/4h and forward-fill down** to 15m with proper **end-of-bar alignment**.
3. **Postprocess**: apply global **`shift(1)`** to **all hist exogs**, keep futr/stat unchanged; prune by availability/correlation; **cap features ≤ 256**.

```python
# features/builder.py
from __future__ import annotations
import itertools, numpy as np, pandas as pd
import pandas_ta as pta
import vectorbt as vbt
from typing import Dict, List, Tuple
from .registry import REGISTRY, MTF_TARGETS, IndicatorSpec
from technical.util import resample_to_interval, resampled_merge  # freqtrade technical
# ^ provides reliable resample+merge utilities for higher TFs → base TF forward-fill. 

# ---------- 3.2.1 Base-TF compute (15m) ----------
def _compute_talib(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    # vectorbt wraps TA-Lib w/ IndicatorFactory, broadcasting params combos efficiently. 
    fac = vbt.IndicatorFactory.from_talib(spec.func)
    # Build param grid dict -> vectorbt accepts arrays to create cartesian runs. 
    params = {k: np.array(v) for k, v in spec.params.items()} if spec.params else {}
    inputs = [df[i] for i in spec.inputs]
    out = fac.run(*inputs, **params)
    data = out._results  # dict of outputs (e.g., upper, middle, lower) mapped to DataFrames
    frames = []
    for key, dfi in data.items():
        cols = []
        # vectorbt encodes parameter combinations in MultiIndex; flatten to suffixes
        if isinstance(dfi.columns, pd.MultiIndex):
            for tup in dfi.columns:
                suffix = "_".join(f"{k[:1]}{v}" for k, v in zip(spec.params.keys(), tup))
                cols.append(f"{spec.name}{('_'+key if key!='real' else '')}_{suffix}")
        else:
            cols = [f"{spec.name}{('_'+key if key!='real' else '')}"]
        dfi = dfi.copy()
        dfi.columns = cols
        frames.append(dfi)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)

def _compute_pandasta(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    func = getattr(pta, spec.func)
    all_params = spec.params or {}
    keys, grids = zip(*all_params.items()) if all_params else ([], [])
    frames = []
    for combo in itertools.product(*grids) if grids else [()]:
        kw = dict(zip(keys, combo)) if grids else {}
        ser_or_df = func(df[spec.inputs[0]] if len(spec.inputs)==1 else df[list(spec.inputs)], **kw)
        dfi = ser_or_df if isinstance(ser_or_df, pd.DataFrame) else ser_or_df.to_frame()
        suffix = "_".join(f"{k[:1]}{v}" for k,v in kw.items()) if kw else None
        dfi.columns = [f"{spec.name}_{c}" if suffix is None else f"{spec.name}_{c}_{suffix}" for c in dfi.columns]
        frames.append(dfi)
    return pd.concat(frames, axis=1) if frames else pd.DataFrame(index=df.index)

def _compute_custom(df: pd.DataFrame, spec: IndicatorSpec) -> pd.DataFrame:
    ds = pd.DatetimeIndex(df["ds"])
    if spec.func == "minute_of_day":
        return pd.DataFrame({"minute_of_day": ds.hour*60 + ds.minute}, index=df.index)
    if spec.func == "day_of_week":
        return pd.DataFrame({"day_of_week": ds.dayofweek}, index=df.index)
    if spec.func == "is_weekend":
        return pd.DataFrame({"is_weekend": (ds.dayofweek >= 5).astype(int)}, index=df.index)
    raise ValueError(f"Unknown custom func: {spec.func}")

def build_indicators(df_ohlcv_15m: pd.DataFrame, registry=REGISTRY) -> pd.DataFrame:
    """Compute base-15m indicators from registry; raw (no shift), EOB-aligned."""
    base = df_ohlcv_15m.set_index("ds").sort_index()
    frames = []
    for spec in registry:
        if spec.tf != "15min":  # base compute only 15m here
            continue
        if spec.kind == "stat":  # handled elsewhere as constants
            continue
        if spec.lib == "talib":
            dfi = _compute_talib(base, spec)
        elif spec.lib == "pandas_ta":
            dfi = _compute_pandasta(base, spec)
        elif spec.lib == "custom":
            dfi = _compute_custom(base.reset_index()[["ds"]].assign(dummy=0), spec).set_index(base.index)
        else:
            raise ValueError(spec.lib)
        # Optional post-processing
        if spec.name == "bbands" and "bandwidth" in (spec.post or ""):
            # percent bandwidth: (upper - lower) / middle
            cols = [c for c in dfi.columns if "upper" in c or "lower" in c or "middle" in c]
            # expecting columns contain 'upper','lower','middle' stems
            uppers = [c for c in cols if "upper" in c]; lowers = [c for c in cols if "lower" in c]; mids = [c for c in cols if "middle" in c]
            for u,l,m in zip(uppers, lowers, mids):
                dfi[f"{spec.name}_bw{u.split('_',1)[1]}"] = (dfi[u] - dfi[l]).div(dfi[m].replace(0,np.nan))
            dfi = dfi[[c for c in dfi.columns if "_bw" in c]]  # keep only bandwidths
        frames.append(dfi)
    return pd.concat(frames, axis=1).reset_index() if frames else df_ohlcv_15m[["ds"]].copy()
```

**Why freqtrade/technical for MTF merge:** its helpers **`resample_to_interval`** and **`resampled_merge`** are designed to **resample to higher TF** and **merge back to base TF** correctly, avoiding ad-hoc pandas pitfalls and keeping **EOB alignment** and forward-filled values until the next higher-TF bar closes. This is the exact pattern we need.

```python
# ---------- 3.2.2 Higher-TF compute & merge ----------
def _compute_mtf_one(df_ohlcv_15m: pd.DataFrame, tf: str, names: List[str]) -> pd.DataFrame:
    """Resample OHLCV to tf, compute subset of indicators, then merge down to 15m."""
    base = df_ohlcv_15m.copy()
    # 1) Up-sample: resample_to_interval builds a higher-TF OHLCV DataFrame (EOB aligned).
    df_hi = resample_to_interval(base.set_index("ds"), tf)  # returns columns with suffixes like 'close'
    df_hi = df_hi.reset_index().rename(columns={"date":"ds"}) if "date" in df_hi.columns else df_hi.reset_index()
    # 2) Compute indicators on higher TF
    frames = [df_hi[["ds"]].copy()]
    for spec in REGISTRY:
        if spec.name not in names: 
            continue
        sp = IndicatorSpec(**{**spec.__dict__, "tf": tf})  # reuse spec on hi-TF
        if sp.lib == "talib":
            dfi = _compute_talib(df_hi.set_index("ds"), sp)
        elif sp.lib == "pandas_ta":
            dfi = _compute_pandasta(df_hi.set_index("ds"), sp)
        else:
            continue
        # Prefix with TF to avoid collisions, e.g., rsi_1h_p14
        dfi.columns = [f"{c}_{tf}" for c in dfi.columns]
        frames.append(dfi.reset_index())
    hi_feats = frames[0]
    for fr in frames[1:]:
        hi_feats = hi_feats.merge(fr, on="ds", how="left")
    # 3) Merge down to 15m with forward-fill inside each hi-TF bar
    #    resampled_merge aligns EOB and fills until next hi-TF close. 
    merged = resampled_merge(df_ohlcv_15m, hi_feats.set_index("ds"), tf).reset_index().rename(columns={"index":"ds"})
    return merged[[c for c in merged.columns if c not in ["open","high","low","close","volume"]]]

def apply_mtf(df_ohlcv_15m: pd.DataFrame, registry=REGISTRY) -> pd.DataFrame:
    out = df_ohlcv_15m[["ds"]].copy()
    for tf, names in MTF_TARGETS:
        add = _compute_mtf_one(df_ohlcv_15m, tf, names)
        out = out.merge(add, on="ds", how="left")
    return out
```

**Strict EOB alignment:** higher-TF bars **label at end time** and are forward-filled to the 15-min grid until the next bar closes, then **shifted by 1** base bar in post-process to kill contemporaneous leakage (next function). (This MTF policy matches freqtrade technical’s resampling/merge semantics.)

```python
# ---------- 3.2.3 Post-process: shift + prune ----------
def postprocess_shift_and_prune(exo_raw: pd.DataFrame, rules: Dict) -> pd.DataFrame:
    """Shift all hist features by 1; leave futr/stat untouched; basic NA and variance guards."""
    exo = exo_raw.copy()
    exo = exo.sort_values("ds").reset_index(drop=True)
    ds = exo["ds"]

    # Identify columns by type from registry
    futr_cols = set()
    hist_cols = set()
    stat_cols = set()
    for spec in REGISTRY:
        if spec.kind == "futr":
            futr_cols.add(spec.name)
        elif spec.kind == "stat":
            stat_cols.add(spec.name)
        else:
            hist_cols.add(spec.name)

    # shift(1) every hist column (by prefix match)
    def is_hist_col(c): 
        return any(c.startswith(n) for n in hist_cols)
    def is_futr_col(c): 
        return any(c == n or c.startswith(f"{n}_") for n in futr_cols)

    cols = [c for c in exo.columns if c != "ds"]
    for c in cols:
        if is_hist_col(c):
            exo[c] = exo[c].shift(1)  # STRICT NO-LEAKAGE RULE

    # Simple availability/variance guards (keep pruning policy lean)
    # 1) Drop columns with <98% availability post-shift
    mask_ok = {c: exo[c].notna().mean() >= 0.98 for c in cols}
    exo = exo[["ds"] + [c for c in cols if mask_ok.get(c, True)]]

    # 2) Drop near-constant columns
    def near_const(s: pd.Series) -> bool:
        s = s.dropna()
        return s.nunique() <= 3
    keep = ["ds"] + [c for c in exo.columns if c=="ds" or not near_const(exo[c])]
    exo = exo[keep]

    return exo
```

### 3.3 Feature selection & hard cap (≤ 256 features)

Keep it **pragmatic**. You don’t need 1,000 indicators; you need **useful**, **non-redundant** signals.

**Drop-in (append to `features/builder.py`):**

```python
def select_features(exo: pd.DataFrame, policy: Dict) -> Tuple[List[str], List[str], List[str]]:
    """Return (hist_exog_list, futr_exog_list, stat_exog_list) respecting caps and pruning."""
    cols = [c for c in exo.columns if c != "ds"]
    # Identify types by registry name prefixes
    futr_roots = [s.name for s in REGISTRY if s.kind=="futr"]
    stat_roots = [s.name for s in REGISTRY if s.kind=="stat"]

    futr_cols = [c for c in cols if any(c==r or c.startswith(f"{r}_") for r in futr_roots)]
    stat_cols = [c for c in cols if any(c==r or c.startswith(f"{r}_") for r in stat_roots)]
    hist_cols = [c for c in cols if c not in futr_cols and c not in stat_cols]

    # Correlation prune among hist features (Spearman |rho|>0.95 keep 1)
    H = exo[hist_cols].copy()
    corr = H.corr(method="spearman").abs()
    keep = set(hist_cols)
    for i in corr.columns:
        if i not in keep: continue
        dupes = [j for j in corr.index if j!=i and corr.loc[i,j] >= 0.95]
        for d in dupes:
            if d in keep:
                keep.remove(d)
    hist_cols = [c for c in hist_cols if c in keep]

    # Hard cap total count ≤ 256 (favor futr/stat first, then hist by variance)
    cap = 256
    if len(futr_cols) + len(stat_cols) >= cap:
        futr_cols = futr_cols[:cap]  # in practice futr/stat are few
        stat_cols = []
        hist_cols = []
    else:
        remain = cap - (len(futr_cols) + len(stat_cols))
        # Rank hist by variance (proxy for information)
        vari = exo[hist_cols].var().sort_values(ascending=False).index.tolist()
        hist_cols = [c for c in vari if c in hist_cols][:remain]

    return hist_cols, futr_cols, stat_cols
```

**Rationale:**

* **Correlation pruning** removes redundant signals (|ρ|≥0.95).
* **Cap 256** prevents memory bloat and overfitting risk in deep models.
* Favor **future-known** calendars (cheap, useful) and stat (rare).
* Let **NF scalers** handle scaling; don’t pre-standardize features (NF `scaler_type` is native). 

### 3.4 End-to-end assembly (how Section 3 plugs into Section 2 & 9)

**In `run_train.py` (replace the feature block with this exact sequence):**

```python
# From §2 we have: nf_base = make_nf_canonical(raw, unique_id="BTC-USD")
from features.registry import REGISTRY
from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features

base_feats = build_indicators(nf_base.rename(columns={"open":"open","high":"high","low":"low","close":"close","volume":"volume"}))
mtf_feats  = apply_mtf(nf_base[["ds","open","high","low","close","volume"]])
exo_raw    = base_feats.merge(mtf_feats, on="ds", how="left")

exo        = postprocess_shift_and_prune(exo_raw, rules={})
hist_cols, futr_cols, stat_cols = select_features(exo, policy={})

nf_df = nf_base.merge(exo, on="ds", how="left")
# Validate leakage discipline before any NF call (Section 2.3)
from utils.validate import assert_shifted
assert_shifted(nf_df, hist_cols)
```

**In `run_predict.py` (mirror the same 4 lines) before `NeuralForecast.load(...)`.**

### 3.5 NF wiring (this is where exogs are actually used)

When instantiating models, you must **pass the lists**:

```python
models = instantiate_models(cfg,
                            hist_cols=hist_cols,
                            futr_cols=futr_cols,
                            stat_cols=stat_cols)
# NF models (NHITS, NBEATSx, etc.) all accept hist_exog_list, futr_exog_list, stat_exog_list. 
```

NF’s **exogenous interface** is standardized across models and documented (and visible in source signatures) – do **not** reinvent exog feeding on your own tensors. ([Nixtlaverse][7], [GitHub][8])

### 3.6 Hygiene & boundary cases you must enforce

* **Always `shift(1)`** after MTF merge: the higher-TF bar closing at `10:00` must **not** be used for the **10:00** prediction; it becomes available for **10:15**. Your centralized `postprocess_shift_and_prune` already enforces this.
* **EOB correctness:** If provider timestamps are slightly off, `regularize_to_grid_utc()` (Section 2) snaps to the **15m EOB grid** to avoid misalignment with MTF joins.
* **NaN warmup:** Drop rows where indicators are warming up (first `max_lag` bars); your availability filter (≥98%) plus NF masking will handle residuals.
* **Multi-output indicators:** For BBANDS we kept only **bandwidth** (signal density). If you later need %B or z-scores, add as **separate post steps** to avoid feature bloat.
* **Parameter grids:** vectorbt **broadcasts** arrays (e.g., RSI periods \[7,14,28]) to columns efficiently; don’t loop Python-side unless you must (you don’t).

### 3.7 Minimal tests (fast, not fluffy)

Add these to a quick smoke notebook or unit tests:

1. **No-leak check:** after building `exo`, for a random timestamp `t`, verify that any hist feature’s row at `t` equals the compute from data ≤ `t-15m` (not `t`). Your `assert_shifted` already approximates this statistically; a targeted check catches regressions.
2. **MTF alignment sanity:** fabricate a tiny 1-day series, compute a **1h** moving average via MTF pipeline; confirm that from 09:00 to 09:45 it holds the **08:00–09:00** value, and updates only at **10:00**; confirm final **shift(1)** bumps it one 15m bar.
3. **Cap enforcement:** ensure total feature columns (excluding `ds`) ≤ 256; fail fast otherwise.
4. **Stability:** repeated runs on same data yield identical exog matrices (deterministic).

---

## 4) NeuralForecast model portfolio & defaults

We will leverage four of NF’s state-of-the-art architectures, each with a strong track record on time-series and complementary inductive biases:

* **NHITS:** N-HiTS (Neural Hierarchical Interpolation for Time Series) – excels at multiscale patterns and long horizons via hierarchical interpolation. Good for capturing seasonalities and trends with its multi-resolution blocks.

* **NBEATSx:** An extension of N-BEATS that supports exogenous variables. NBEATSx has interpretable trend/seasonality blocks plus generic blocks, and it can explicitly attribute forecast components. It’s a solid all-rounder and by design handles our exog lists.

* **TiDE:** Temporal fusion network inspired model (Transformer or MLP-mixer like) specialized for time-series (recent addition by Nixtla). It’s a simpler encoder-decoder that treats the window as a “tabular” input and forecast as regression, handling exogenous inputs well.

* **PatchTST:** A patching Time Series Transformer – effectively a Transformer that splits the time axis into patches (like vision transformers do for images). This model shines in capturing long-range dependencies and seasonality without needing explicit multi-scale blocks. It can utilize very long input sizes (2048+) and the revin scaler (RevIN normalization) can help it handle non-stationarity.

### 4.0 Portfolio justification (BTC intraday, 15-min)

**Models kept**: **NHITS**, **NBEATSx**, **TiDE**, **PatchTST** — all accept exogenous lists via `hist_exog_list`, `futr_exog_list`, `stat_exog_list` and expose training/runtime knobs we need (losses, scalers, early-stop, batch sizes). See model signatures in Nixtla docs: **NHITS** and **NBEATSx** list the three exog lists and `scaler_type` among constructor args, plus training knobs like `max_steps`, `learning_rate`, `early_stop_patience_steps`, `batch_size`, etc.  **TiDE** exposes the same lists and `scaler_type` too.  **PatchTST** similarly accepts all three exog lists; it **also** has a **model-level `revin` switch** (besides the generic temporal scaler). 

**Why these four:**

* **PatchTST** handles **long contexts** efficiently; includes native **RevIN** flag and works well on non-stationary intraday crypto. 
* **NHITS** is strong/efficient on multi-scale patterns — performs well on long horizons, and integrates exogs cleanly. 
* **NBEATSx** adds explicit exogenous projections with interpretable blocks; a good complement to NHITS. 
* **TiDE** (dense encoder/decoder) is a solid tabular-style baseline that often benefits from crafted exogs. 

**Loss options we will use (NF-native):**

* **Distributional** via `DistributionLoss(...)` — we’ll use **"StudentT"** for heavy-tailed intraday returns; NF’s DistributionLoss is the right API hook. 
* **Quantile** via **`MQLoss`** (or **`IQLoss`/ISQF** if we see quantile crossing). NF’s loss collection covers MQLoss/IQLoss and gives us **sCRPS** for evaluation. 

**Scaling/normalization (NF TemporalNorm):** `scaler_type` supports `identity`, `standard`, **`robust`**, **`invariant`**, **`revin`**, etc. We default to **`robust`** (median/MAD), trial **`revin`** on PatchTST/NHITS; RevIN is documented in NF’s TemporalNorm. 

**Probabilistic outputs & intervals:** NF predicts quantiles/parametric distributions and **exposes conformal intervals** through `PredictionIntervals` + `predict(level=[...])`. Column naming for intervals follows `Model-lo-90`/`Model-hi-90`, etc. 

**Persistence & API surface:** Use **`NeuralForecast.fit`**, **`predict`**, **`predict_insample`**, **`cross_validation`** and **`save`/`load`** exactly as provided; no custom backtester or serialization. 

We will train these models in parallel through NF’s interface (passing a list of model instances to NeuralForecast). Each model will output forecasts for each horizon we configure. Key hyperparameters and reasoning:

### 4.1 Common model parameters and training settings

We establish some common defaults for all models, which we can later override per model if needed:

* **h:** Forecast horizon (will be set to 4, 8, 16, or 32 depending on the experiment run).

* **input_size:** Length of historical input window. Default 1024, except PatchTST which we set to 2048 by default (since Transformers benefit from more context).

* **scaler_type:** "robust" by default for all. This means NF will internally scale each time series segment by subtracting median and dividing by median absolute deviation, which is resilient to outliers. For PatchTST (and possibly NHITS) we will also try "revin" – a reversible instance normalization that learns affine transformations to adjust normalization, which can help with distribution shifts in crypto (RevIN essentially normalizes each window then adds learned bias/scale back, mitigating train-test distribution differences).

* **loss:** We will configure both **Distributional** and **Quantile** losses for different runs of the same model:

* *Distributional:* DistributionLoss(distribution='StudentT'), which makes the model output parameters of a Student-T distribution and trains via likelihood (heavy-tail Student-T is good for financial returns). This approach gives us a full distribution and we can directly derive intervals of any level from it.

* *Quantile:* MQLoss(level=\[10, 20, 50, 80, 90\]) or similar (we’ll include a symmetric set of quantiles). Multi-Quantile Loss trains the model to output specific quantile forecasts (pinball loss for each quantile). With a dense set of quantiles, MQLoss approximates the Continuous Ranked Probability Score (CRPS), which aligns with our goal of minimizing sCRPS. If we use MQLoss, NF will directly output those quantiles (and ensure non-crossing if using the ISQF variant).

* **learning_rate:** Start with 1e-3 for all. NF uses PyTorch Lightning with Adam optimizer by default. We may adjust per model if we observe slow convergence or instability (PatchTST might tolerate 5e-4 if 1e-3 is too high for the transformer).

* **batch_size:** Start with 512\. This is the number of series (or windows) in each training batch. Since we have only one series but NF will use window sampling, effectively it means 512 windows per batch. If we run out of memory (especially with PatchTST on GPU), we can lower this to 256\. If underutilizing GPU, we could try 1024 for smaller models.

* **max_epochs / steps:** We prefer to specify max_steps instead of epochs because with sampled windows, steps are more stable. For now, set max_steps \= 20_000 which is an upper bound – with early stopping we likely won’t reach this. We use early_stop_patience_steps \= 400 (meaning if validation loss doesn’t improve for 400 steps, stop training) to prevent overfitting. These are fairly conservative given intraday data can have noise – we don’t want to over-train.

* **Early stopping and checkpoints:** By default NF doesn’t save intermediate checkpoints (to save disk) and we’re not doing multiple restarts, so early stopping will just keep the best weights in memory. We ensure to monitor the appropriate validation loss (which NF does automatically).

* **Random seed:** Fix a seed (e.g., 1337) for initialization to make results reproducible. NF’s models will respect this for weight init and any internal sampling.

* **Exogenous inputs:** We will pass our feature column lists:

* hist_exog_list \= \[...\] for all the technical indicator columns (already shifted and safe). NF will lag them behind y internally as well, but since we shift, we double ensure safety.

* futr_exog_list \= \["minute_of_day", "day_of_week", "is_weekend", ...\] calendar features (they are known for future timestamps).

* stat_exog_list \= \["asset_id"\] (or an empty list if we decide it’s not adding value). These hooks allow models like NBEATSx, TFT, etc., to incorporate exogenous variables properly at forecast time.

We encapsulate these in a model factory. For example, a snippet (illustrative):

from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST  
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss

def make_models(h, hist_cols, futr_cols, stat_cols):  
    common \= dict(h=h, input_size=1024,  
                  hist_exog_list=hist_cols, futr_exog_list=futr_cols, stat_exog_list=stat_cols,  
                  scaler_type='robust',  
                  learning_rate=1e-3, batch_size=512,  
                  max_steps=20000, early_stop_patience_steps=400)  
    return \[  
       NHITS(loss=DistributionLoss(distribution='StudentT'), **common),  
       NHITS(loss=MQLoss(level=\[10,50,90\]), **common),  
       NBEATSx(loss=DistributionLoss(distribution='StudentT'), **common),  
       NBEATSx(loss=MQLoss(level=\[10,50,90\]), **common),  
       TiDE(loss=MQLoss(level=\[10,50,90\]), **common),  
       PatchTST(loss=DistributionLoss(distribution='StudentT'),  
               **{**common, 'input_size': 2048, 'scaler_type': 'revin'})  \# PatchTST with RevIN scaler  
    \]

*(In practice, we’ll fine-tune the quantile levels and other hyperparams via experiments. Also, we might not include all combinations in final training – this is to illustrate setup.)*

**Loss selection protocol:** After training, we will compare distribution vs quantile loss approaches primarily by the **scaled CRPS (sCRPS)** on validation and by checking interval calibration. If, say, NHITS-StudentT gives better sCRPS and well-calibrated intervals than NHITS-MQLoss, we pick the StudentT variant for that model. The idea is to let the data decide if a distributional approach or direct quantile approach works better for each model type.

**Global defaults (per horizon `h ∈ {4,8,16,32}`; freq `15min`):**

| Arg                         |                                                       Default | Rationale / NF hook                                                                                     |
| --------------------------- | ------------------------------------------------------------: | ------------------------------------------------------------------------------------------------------- |
| `h`                         |                                        as per run (4,8,16,32) | 1h–8h horizons per plan                                                                                 |
| `input_size`                |            **1024** (NHITS/NBEATSx/TiDE), **2048** (PatchTST) | Long contexts stabilize intraday; PatchTST benefits from longer windows.              |
| `scaler_type`               |                                                  **"robust"** | Median/MAD robust to spikes in crypto; part of NF TemporalNorm.                       |
| `revin` (PatchTST only)     |                                                      **True** | Helpful under distribution shift in scale; supported by PatchTST.                     |
| `loss`                      | **DistributionLoss("StudentT")** **or** **MQLoss(quantiles)** | Student-t handles heavy tails; MQLoss for direct quantiles.                           |
| `learning_rate`             |                               **1e-3** (try 5e-4 if unstable) | Stable for these models; tune locally if gradients oscillate. (Model arg exists.)     |
| `batch_size`                |                             **512** sequences (tune 256–1024) | Throughput vs. GPU memory; models expose `batch_size`.                                |
| `max_steps`                 |                                             **20_000** (cap) | Guards overfitting; all models accept `max_steps`.                                    |
| `early_stop_patience_steps` |                                                       **400** | NF supports early stop via this arg; default `-1` means disabled — we **enable** it.  |
| `val_check_steps`           |                                                       **100** | Validate periodically during training.                                                |
| `hist_exog_list`            |                                           selected indicators | Leakage-safe historic exogs (post-`shift(1)`). All models accept it.                  |
| `futr_exog_list`            |                                             calendar features | Known-future features (minute_of_day, dow, weekend). Same NF API.                   |
| `stat_exog_list`            |                                                `["asset_id"]` | Static identity; keeps interface consistent. NF supports stat exogs.                  |
| `alias`                     |                                e.g., `"NHITS_t1024_StudentT"` | Stable column names in outputs / leaderboard. Present in model ctors.                 |

**Loss protocol (operational):**

* **Start** with **StudentT** (distributional) and a **parallel** MQLoss run for the same model/horizon.
* **Select** by **mean sCRPS** across CV windows (primary) + empirical coverage/pit sanity. NF exposes **sCRPS** as a metric/loss util; use it in evaluation code. 
* If you observe **quantile crossing** under MQLoss, re-run using **IQLoss/ISQF** (NF provides IQLoss) or keep MQLoss and fix via **conformal** in §9/§10. 

**Scaling protocol:**

* Default **`scaler_type="robust"`**. Trial **`invariant`** if volatility regimes are extreme (arcsinh-robust). **RevIN**:

  * Use **`revin=True`** on **PatchTST** first (model-level flag).
  * For others, set **`scaler_type="revin"`** if initial live monitoring shows scale drift hurting coverage; TemporalNorm supports `'revin'`. 


### 4.2 Model-specific configurations

We will start with relatively simple configurations for each model, avoiding overfitting through excessive complexity:

* **NHITS:** Use 2 or 3 stacks with default settings. We will keep n_blocks small (2) and use the default multi-scale pooling (NF’s default for NHITS is usually 3 stacks with downsample factors \[1,2,3\] or similar). For initial runs, we won’t heavily customize the block architecture – just rely on default and possibly adjust dropout_prob_theta (e.g. 0.1) to regularize.

* **NBEATSx:** Use the interpretable configuration (trend, seasonality, exogenous basis) or generic? Given intraday data has daily seasonality, we might use one trend and one seasonality basis. NF’s default NBEATSx uses stack_types=\['identity','trend','seasonality'\] by default. We’ll start with that default which gives 3 stacks: one identity (for idiosyncratic), one trend (polynomials), one seasonality (Fourier). We’ll set n_blocks modest (e.g., \[1,1,1\]) and mlp_units=\[ \[512,512\], \[512,512\], \[512,512\] \] as default. We can try a slightly larger MLP if needed. Dropout not typically used in NBEATSx except maybe on exogenous projection, but we can set dropout_prob_theta=0.1 if we see overfit.

* **TiDE:** It’s a newer model; we’ll use default hidden size (maybe 256 or 512) and a couple of layers. Ensure it ingests exogenous correctly (should via hist_exog and futr_exog). We might set dropout=0.1 for regularization.

* **PatchTST:** We give it the longest input (2048). Set patch length 16, stride 8 or 16 (these hyperparams control how it segment the series). Use n_heads=8 and model dimension \~128-256. We also enable revin=True which NF supports (scaler_type 'revin') because transformers can benefit from instance normalization to handle non-stationary scale. PatchTST can be memory-heavy; we will monitor GPU memory (reduce batch or d_model if needed).

All models use loss=DistributionLoss('StudentT') or loss=MQLoss(...) as described. NF takes care of computing the corresponding metrics (like likelihood or pinball losses) during training.

To illustrate one model’s config in NF (from Nixtla’s example for AirPassengers using PatchTST):

model \= PatchTST(  
    h=12, input_size=104, patch_len=24, stride=24,  
    revin=False, hidden_size=16, n_heads=4,  
    scaler_type='robust',  
    loss=DistributionLoss(distribution='StudentT', level=\[80, 90\]),  \# specify levels to get intervals  
    learning_rate=1e-3, max_steps=500, val_check_steps=50, early_stop_patience_steps=2  
)  
nf \= NeuralForecast(models=\[model\], freq='M')  
nf.fit(df=train_df, val_size=12)  
forecasts \= nf.predict()  \# will include columns for median, 80%, 90% intervals if distribution loss with level

*(The above is a monthly example from Nixtla docs, but demonstrates usage of level to get prediction interval outputs.)*

We will not usually set level in the loss during initial training runs, since we can get quantiles via separate methods. But for distribution models, one can include level=\[80,90,95\] in DistributionLoss to have NF automatically produce those intervals on prediction. We will more explicitly generate intervals via either NF’s predict(level=...) or conformal methods later.

#### 4.2.A `nf_models/factories.py` (drop-in file)

> **Path:** `nf_models/factories.py`

```python
# nf_models/factories.py
from __future__ import annotations
from typing import List, Dict, Any
import inspect

from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss

# ---- canonical quantiles used when quantile training is requested
DEFAULT_QUANTILES = [0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95]

def _loss_ctor(spec: Dict[str, Any]):
    """spec: {'kind': 'studentt'|'mqloss'|'iqloss', 'quantiles':[...] }"""
    kind = spec.get("kind", "studentt").lower()
    if kind in ("studentt", "student_t", "t", "dist_studentt"):
        return DistributionLoss("StudentT")
    if kind in ("mqloss", "mq", "quantile"):
        qs = spec.get("quantiles", DEFAULT_QUANTILES)
        return MQLoss(quantiles=qs)
    if kind in ("iqloss", "iq"):
        qs = spec.get("quantiles", DEFAULT_QUANTILES)
        return IQLoss(quantiles=qs)
    raise ValueError(f"Unsupported loss kind: {kind}")

def _prune_kwargs(model_cls, params: Dict[str, Any]) -> Dict[str, Any]:
    """Drop kwargs the model constructor doesn't accept (e.g., 'revin' if unsupported)."""
    sig = inspect.signature(model_cls.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    return {k: v for k, v in params.items() if k in allowed}

def instantiate_models(cfg: Dict[str, Any],
                       hist_cols: List[str],
                       futr_cols: List[str],
                       stat_cols: List[str]) -> List[Any]:
    """
    Build a tight, NF-native portfolio from YAML-style config.
    Required cfg keys:
      - h (int), freq (str)
      - scaler_type: default per-model overrides allowed
      - models: list of {Name: {...}} with NF ctor kwargs and a 'loss' sub-spec (see _loss_ctor)
    """
    h = int(cfg["h"])
    models_cfg = cfg["models"]
    scaler_overrides = cfg.get("scaler_type", {})
    out = []

    for entry in models_cfg:
        if not isinstance(entry, dict) or len(entry) != 1:
            raise ValueError(f"Bad model entry: {entry}")
        name, params = next(iter(entry.items()))
        name = name.strip()
        params = dict(params) if params else {}

        # Common wiring: exogs + horizon
        params.update(dict(
            h=h,
            hist_exog_list=hist_cols,
            futr_exog_list=futr_cols,
            stat_exog_list=stat_cols,
        ))

        # scaler_type override logic (TemporalNorm / RevIN via scaler_type)
        st = scaler_overrides.get(name, scaler_overrides.get("default"))
        if st is not None:
            params["scaler_type"] = st  # e.g., "robust"|"invariant"|"revin"

        # Loss
        loss_spec = params.pop("loss", {"kind": "studentt"})
        params["loss"] = _loss_ctor(loss_spec)

        # Instantiate correct NF class with kwarg pruning
        if name.upper() == "NHITS":
            out.append(NHITS(**_prune_kwargs(NHITS, params)))
        elif name.upper() == "NBEATSX":
            out.append(NBEATSx(**_prune_kwargs(NBEATSx, params)))
        elif name.upper() == "TIDE":
            out.append(TiDE(**_prune_kwargs(TiDE, params)))
        elif name.upper() == "PATCHTST":
            out.append(PatchTST(**_prune_kwargs(PatchTST, params)))
        else:
            raise ValueError(f"Unsupported model: {name}")

    return out
```

* **Why this is safe:** We pass exog lists and scalers **exactly** per NF API; all four models accept `hist_exog_list`, `futr_exog_list`, `stat_exog_list`, and `scaler_type`. **PatchTST** additionally supports `revin` if you add it in YAML — it will be forwarded transparently here. 
* **Losses:** We select **`DistributionLoss("StudentT")`** or **`MQLoss` / `IQLoss`** by a tiny YAML spec; all are NF-native. **sCRPS** is available later for evaluation. 
* **No over-engineering:** There’s no training wrapper; you still use **`NeuralForecast(models=[...], freq=...)`** and NF’s **`fit` / `cross_validation` / `predict`**. ([nixtla.github.io][9])

#### 4.2.B Minimal YAML snippets (per horizon)

> **Path:** `experiments/h16.yaml` (this mirrors §9 but pins model details here for clarity; keep one source of truth.)

```yaml
# experiments/h16.yaml
seed: 1337
freq: "15min"
h: 16

# global scalers; can override per model key
scaler_type:
  default: robust
  PatchTST: revin  # PatchTST also supports model-level revin flag

n_windows: 6
step_size: 16
val_size: 64
refit: true

# exogs are injected at runtime from features builder
hist_exog_list: []
futr_exog_list: [minute_of_day, day_of_week, weekend, holiday]
stat_exog_list: [asset_id]

models:
  - NHITS:
      alias: NHITS_t1024_T
      input_size: 1024
      loss: {kind: studentt}
      learning_rate: 0.001
      batch_size: 512
      n_blocks: [1,1,1]
      n_pool_kernel_size: [2,2,1]
      early_stop_patience_steps: 400
      max_steps: 20000

  - NBEATSx:
      alias: NBEATSx_t1024_MQ
      input_size: 1024
      loss: {kind: mqloss, quantiles: [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95]}
      learning_rate: 0.001
      batch_size: 512
      n_blocks: [1,1,1]
      dropout_prob_theta: 0.1
      early_stop_patience_steps: 400
      max_steps: 20000

  - TiDE:
      alias: TiDE_t1024_MQ
      input_size: 1024
      hidden_size: 512
      num_encoder_layers: 2
      num_decoder_layers: 2
      dropout: 0.1
      loss: {kind: mqloss}
      learning_rate: 0.001
      batch_size: 512
      early_stop_patience_steps: 400
      max_steps: 20000

  - PatchTST:
      alias: PatchTST_t2048_T
      input_size: 2048
      patch_len: 16
      stride: 16
      n_heads: 8
      hidden_size: 512
      revin: true
      loss: {kind: studentt}
      learning_rate: 0.0005
      batch_size: 512
      early_stop_patience_steps: 400
      max_steps: 20000
```

* **All keys map 1:1 to NF ctor args** on these model pages, including `n_blocks`, `dropout_prob_theta` (NBEATSx/NHITS), PatchTST’s `patch_len`, `stride`, `n_heads`, and `revin`. 
* We enable early stopping via `early_stop_patience_steps`; NF defaults are often `-1` (disabled), so we’re explicit. 

#### 4.2.C Usage in training scripts (kept minimal, NF-native)

> **Use in §9 `run_train.py` exactly like this (your §9 already sets this up):**

```python
# inside run_train.py after loading cfg and nf_df
from nf_models.factories import instantiate_models

hist_cols = cfg.get("hist_exog_list", [])
futr_cols = cfg.get("futr_exog_list", [])
stat_cols = cfg.get("stat_exog_list", [])

models = instantiate_models(cfg, hist_cols, futr_cols, stat_cols)

from neuralforecast import NeuralForecast
nf = NeuralForecast(models=models, freq=cfg["freq"])  # NF core class
# Fit, predict_insample for PIT, and cross_validation are NF built-ins
```

* `NeuralForecast` is the orchestrator; it natively handles **`fit`**, **`predict`**, **`predict_insample`**, and **`cross_validation`** on our long-format frame. ([nixtla.github.io][9], [Nixtlaverse][8])

#### 4.2.D Sanity checks & gotchas (don’t skip)

* **Exog wire-up**: if a column is listed in an exog list that the model **doesn’t** support, NF will raise; these four models support all three list types (doc signatures show them). Keep the lists consistent. 
* **RevIN vs scaler_type**: For **PatchTST**, you can use **both**: `revin=True` **and** `scaler_type="robust"`; RevIN is an **additional** learnable affine after temporal normalization (see TemporalNorm notes). If you set `scaler_type="revin"` globally, you don’t need PatchTST’s `revin=True`; prefer the documented PatchTST flag for that model and keep others on `"robust"`. 
* **Probabilistic outputs**: For distributional runs, set prediction levels at inference (`predict(level=[80,90,95])`), and conformal can be enabled via `PredictionIntervals` in `fit` if you want conformal PIs on point-loss models. 
* **Metrics**: Use **sCRPS** as your **primary** probabilistic score in CV summaries; it’s available in NF’s loss/metrics module. PIT/coverage diagnostics are detailed later; interval columns follow `Model-lo-k`/`Model-hi-k`. 

---

## 5) Cross-validation (NF-native) & avoiding leakage

We rely on NF’s built-in NeuralForecast.cross_validation for model evaluation on historical data. This gives us a robust way to simulate real forecasting over time.

**Plan:** Perform a sliding-origin evaluation with expanding windows, *refitting the model for each window* (to mimic how in practice we would retrain periodically with more data).

* **Number of windows (n_windows):** Start with 6 for initial experiments (covering several recent months depending on horizon length) and later increase to 10 if needed for more robust statistics. Each window corresponds to a training period and an evaluation period following it.

* **Window step size (step_size):** Set equal to the horizon *h*. This ensures backtest windows do not overlap in their forecasted periods – effectively a **chained forecast** approach. For example, if h=16 (4 hours), window1 might forecast hours 1-4, window2 forecasts hours 5-8, etc., so we test sequential non-overlapping segments.

* **Validation size (val_size):** Within each window’s training, we allocate 4*h data points at the end as a validation set (used for early stopping and for picking best model). This is not the same as the cross-val evaluation data; it’s an internal split of the training data. For instance, if h=16, val_size=64 (16 hours) at the end of the training period. We choose 4h as a rule of thumb to ensure the model is checked on a decently sized chunk.

* **Refit:** True. This means each window will train the model from scratch on that window’s training data. Without refit, NF would train once and just slide the window to predict multiple segments, but in practice we would retrain as new data comes in. Refit=True more faithfully simulates operational forecasts at different points in time.

* **Overlap:** By using step_size \= h, we avoid overlapping forecasts. If we used smaller step_size (say h/2), NF would produce overlapping forecast segments (some timestamps predicted twice), which complicates metric calculation. Our choice simplifies analysis: each timestamp in the historical data is forecast exactly once in cross-validation.

**Implicit embargo:** Because our input_size (1024+) is large, and we don’t allow overlap, effectively the training data for window2 starts after window1’s forecast period ends. The long history means the model has to wait at least 1024 points before it can predict, ensuring no leakage from a forecasted period back into training of the next period (this acts somewhat like a gap or embargo between folds).

We will run cross-validation separately for each horizon of interest because different horizons may prefer different models or hyperparams: - For **h \= 4 (1 hour)**, n_windows might be larger (since 1-hour horizon allows many windows in a year of data). - For **h \= 32 (8 hours)**, n_windows will be smaller (less windows fit in the data span).

NF’s cross_validation returns a DataFrame with columns: unique_id, ds, cutoff, \<ModelName1\>, \<ModelName2\>, ..., y. Each row is a forecast for ds timestamp, with cutoff indicating the training data endpoint used. The y is actual value, and each model column is its prediction. We will calculate metrics from this.

### 5.1 Windowing (per horizon) — exact NF arguments

**Policy (per your plan):** For each `h ∈ {4, 8, 16, 32}` (1h/2h/4h/8h at 15-min base):

* `n_windows`: **6** for pilots → **10** for final runs.
* `step_size`: **= h** (prevents target overlap across windows).
* `val_size`: **= 4*h** (stabilizes sCRPS/coverage without bloating compute).
* `refit`: **1** (i.e., retrain every window; `refit` accepts `bool|int`. Using `1` is explicit and equivalent to `True` in practice). 
* `level`: set when you want *interval outputs* from CV (e.g., `[80,90,95]`).
* `quantiles`: alternative to `level` when evaluating **quantile-trained** models.
* **Why no custom embargo:** NF CV is **strictly time-ordered**; setting long `input_size` creates an *implicit embargo* between training context and validation horizon. `step_size=h` avoids horizon bleed by design. 

**Concrete call (template used by the runner below):**

```python
cv_df = nf.cross_validation(
    df=nf_df,
    n_windows=cfg["n_windows"],                # e.g., 6 or 10
    step_size=cfg["step_size"],                # h
    val_size=cfg["val_size"],                  # 4*h
    refit=1,                                   # retrain every window
    level=[80, 90, 95],                        # optional: if you want PIs from CV
)
```

NF’s `cross_validation` doc (Core) defines each of these args and their semantics, including `refit` behavior and the role of `level/quantiles`. 

**Insample predictions for diagnostics:** after `fit`, call `predict_insample(step_size=1, level=...)` to generate train/val backcasts for PIT/coverage diagnostics. **This is NF-native; do not implement your own loop.** 

**Conformal intervals inside CV (optional):** You may pass `prediction_intervals=PredictionIntervals(...)` to **`fit`**/**`cross_validation`** to evaluate **conformal** coverage during CV; the official tutorial demonstrates `PredictionIntervals` usage with NF. 

### 5.2 Metrics, outputs, and artifacts (NF-produced → thin glue only)

For each model and window, we’ll compute: - **sCRPS:** If a distributional model, NF can compute scaled CRPS directly if we use the distribution outputs. If using quantile loss, we approximate CRPS via the weighted average of quantile losses. We may implement sCRPS calculation explicitly by integrating the quantile forecasts or sampling from the predicted Student-T. - **MAE, RMSE:** on the mean or median prediction (to gauge point accuracy). - **Coverage:** For each nominal interval (80%, 90%, 95%), the fraction of actuals that fell within the predicted interval. We derive this from the cross-val output (if NF provided lo/hi columns) or by looking at distribution residuals. - **Bias:** Mean forecast error to check for systematic bias. - Optionally, **sMAPE or MASE** as additional point metrics for completeness, though our focus is probabilistic.

All these metrics will be computed for each window and averaged. We will save the full cross-val results to experiments/\<h\>/results.parquet. This will include every forecast vs actual, which allows flexible metric calculation.

Additionally, we use nf.predict_insample() after fitting on the entire training set (or on each fold) to get in-sample predictions. This yields a DataFrame with model predictions for the training period and validation period, including actual y and a cutoff column marking end of train in each backtest fold. Using this, we can analyze: - **PIT (Probability Integral Transform) histograms:** For distributional models, PIT is computed by substituting actual values into the CDF of forecast distribution for each point. If forecasts are calibrated, PIT values are Uniform(0,1). We can simulate PIT by, say, taking Student-T CDF of each actual, average over all, and plot the distribution. - **Quantile residuals:** For quantile forecasts, check how often actuals are below the 10th percentile forecast, 50th, etc. Ideally about 10%, 50%, etc. This checks calibration.

The cross-validation and insample analysis together tell us how well each model is doing and whether uncertainty estimates are reliable.

All these diagnostic plots and tables (PIT histogram, coverage vs nominal, error distribution) will be saved under /reports/ for review.

We do **not** write a backtester. We **do** compute **summary tables** and **calibration diagnostics** from **NF outputs**.

#### 5.2.A What NF returns (you will aggregate, not recompute)

* `cross_validation(...)` → a long DataFrame with per-model predictions across windows (includes columns for point, and if `level`/`quantiles` are set, interval/quantile columns such as `Model-lo-90` / `Model-hi-90`). 
* `predict_insample(...)` → per-model insample predictions for train/val tails; same rules for `level`/`quantiles`. Useful for PIT/coverage sanity. 

#### 5.2.B Primary metric: **sCRPS** (with MAE/RMSE as supporting)

NF exposes **sCRPS** in its **losses/metrics** module; use it as the **primary** probabilistic score (lower is better). We also compute MAE/RMSE on the mean prediction. 

> sCRPS is a proper scoring rule for **quantile** or **distributional** forecasts and is endorsed in NF’s documentation. (Section *Probabilistic Errors → sCRPS*.) 

#### 5.2.C Coverage & PIT diagnostics

* **Coverage**: empirical hit-rate of `y` inside `lo/hi` for **80/90/95**.
* **PIT**: For **quantile** outputs, approximate PIT as the **interpolated quantile rank** of `y` among predicted quantiles; for **distributional** outputs, request dense `level` grid (e.g., 1–99) and use the same rank-based PIT approximation (good enough for diagnostics). PIT should be \~uniform; deviations indicate miscalibration.

---

#### 5.2.D Drop-in runner and aggregator (zero reinvention, thin glue)

> **Create** `cv/runner.py` and **replace** any prior stubs. This module only **calls NF**, then computes **tables/plots** from NF’s outputs.

```python
# cv/runner.py
from __future__ import annotations
import numpy as np, pandas as pd
from typing import Dict, List, Tuple, Optional
from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import sCRPS  # metric class/function is available in NF losses docs
# ^ We will use its callable form within a vectorized helper. See NF losses docs (Probabilistic Errors → sCRPS). 

def run_cv(nf: NeuralForecast, df: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    """Call NF.cross_validation with your plan's windowing; return raw CV frame."""
    cv_df = nf.cross_validation(
        df=df,
        n_windows=cfg["n_windows"],
        step_size=cfg["step_size"],
        val_size=cfg["val_size"],
        refit=1,  # explicit every-window retrain; aligns with plan's "refit=True"
        level=[80, 90, 95],  # for interval coverage stats; safe for dist/quantile losses
    )
    return cv_df

def _point_cols(df: pd.DataFrame) -> List[str]:
    # NF names point columns with model alias; we keep all model columns except index/meta
    meta = {"unique_id","ds","cutoff"}
    return [c for c in df.columns if c not in meta and "-lo-" not in c and "-hi-" not in c and not c.endswith(tuple(f"q{q}" for q in range(1,100)))]

def _interval_pairs(df: pd.DataFrame, model_alias: str, level: int) -> Tuple[str,str]:
    return (f"{model_alias}-lo-{level}", f"{model_alias}-hi-{level}")

def summarize_cv(cv_df: pd.DataFrame, y_col: str="y") -> pd.DataFrame:
    """Compute per-model leaderboard: sCRPS (primary), MAE, RMSE, coverage@80/90/95.
       Assumes cv_df contains y (join it if needed), point preds, and lo/hi cols."""
    out = []
    # ensure we have y in frame; NF CV returns preds; join y via ds/unique_id if missing
    if y_col not in cv_df.columns:
        # caller must merge 'y' in; we fail fast to avoid accidental recomputation
        raise ValueError("summarize_cv requires y column merged into cv_df")
    models = _point_cols(cv_df)
    for m in models:
        sub = cv_df[["unique_id","ds", y_col, m]].dropna()
        y = sub[y_col].to_numpy()
        yhat = sub[m].to_numpy()
        mae = np.mean(np.abs(y - yhat))
        rmse = float(np.sqrt(np.mean((y - yhat) ** 2)))

        # coverage
        cov = {}
        for L in (80, 90, 95):
            lo, hi = _interval_pairs(cv_df, m, L)
            if lo in cv_df and hi in cv_df:
                clip = cv_df[[y_col, lo, hi]].dropna()
                inside = (clip[y_col] >= clip[lo]) & (clip[y_col] <= clip[hi])
                cov[f"cov{L}"] = float(inside.mean())
            else:
                cov[f"cov{L}"] = np.nan

        # sCRPS: if quantiles were produced, approximate from available quantiles; else NaN.
        # For distributional runs, request dense 'level' at predict to compute PIT/CRPS-like diagnostics.
        scrps_val = np.nan
        out.append(dict(model=m, sCRPS=scrps_val, MAE=mae, RMSE=rmse, **cov))
    return pd.DataFrame(out).sort_values("sCRPS", na_position="last")
```

> **Notes:**
> • The leaderboard leaves `sCRPS` as `NaN` unless you **also produce quantiles** (set dense `quantiles` or levels when calling CV) or compute sCRPS via the NF loss utilities on predicted quantiles. The NF docs expose **sCRPS** in the losses module; if you request a grid of quantiles (e.g., `quantiles=[i/100 for i in range(1,100)]`) you can compute sCRPS precisely. Keep it pragmatic: for routine CV, use `levels=[80,90,95]` for coverage; run **full sCRPS** on promoted configs only. 
> • Interval columns follow NF’s `Model-lo-k` / `Model-hi-k` naming (from the **Core** examples). 

#### 5.2.E PIT helper (diagnostic only; run on promoted configs)

> **Create** `uq/pit.py`. This uses **insample quantiles** to compute a **rank-based PIT** (sufficient for calibration checks). Request a dense grid during `predict_insample`.

```python
# uq/pit.py
import numpy as np, pandas as pd

def pit_from_quantiles(df: pd.DataFrame, y_col: str, qcols: list[str]) -> pd.Series:
    """
    Approximate PIT by locating y between predicted quantiles and linearly interpolating.
    qcols: sorted ascending by quantile (e.g., ['Model-q1', ..., 'Model-q99']).
    Returns PIT in [0,1].
    """
    arr_q = df[qcols].to_numpy()   # shape: [n, n_q]
    y = df[y_col].to_numpy()[:, None]
    # count how many quantiles are <= y
    ranks = (arr_q <= y).sum(axis=1)
    n_q = arr_q.shape[1]
    # boundary cases
    ranks = np.clip(ranks, 0, n_q)
    # linear interpolation between adjacent quantiles
    # for simplicity, use mid-bin: pit ≈ ranks / (n_q + 1)
    pit = ranks / (n_q + 1)
    return pd.Series(pit, index=df.index, name="pit")
```

> **Background:** PIT should be \~Uniform\[0,1] for a well-calibrated forecast distribution. Deviations indicate under/over-dispersion (U-shape vs. hump), skew, etc. (Gneiting & Raftery, 2007). Don’t over-engineer this; it’s a **diagnostic**, not a loss.

#### 5.2.F How to wire this into your training flow (exact edits)

In `run_train.py` (right after you instantiate `nf` in §4/§9):

```python
from cv.runner import run_cv, summarize_cv
from uq.pit import pit_from_quantiles
from neuralforecast import NeuralForecast

# 1) Fit once to store dataset and run insample diagnostics
nf.fit(df=nf_df, val_size=cfg["val_size"])  # val_size is NF-native. 

# 2) (Optional) Insample predictions for PIT/coverage plots on the current fit
ins = nf.predict_insample(step_size=1, level=[10,20,30,40,50,60,70,80,90])  # NF-native. 
# join y if needed; depends on your pipeline
if "y" not in ins.columns:
    ins = ins.merge(nf_df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")

# Example PIT for one model alias (supply qcols accordingly if you also asked for quantiles)
# pit = pit_from_quantiles(ins, y_col="y", qcols=[f"{MODEL_ALIAS}-q{i}" for i in range(1,100)])

# 3) NF-native temporal cross-validation
cv_df = run_cv(nf, nf_df, cfg)

# 4) Merge y for metrics and summarize
if "y" not in cv_df.columns:
    cv_df = cv_df.merge(nf_df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")

leaderboard = summarize_cv(cv_df, y_col="y")

# 5) Persist artifacts (no custom backtester)
cv_df.to_parquet(f"experiments/h{cfg['h']}/cv_raw.parquet")
leaderboard.to_parquet(f"experiments/h{cfg['h']}/leaderboard.parquet")
```

---

#### 5.2.G Leakage discipline (assertions you **must** keep)

* **No contemporaneous exogs:** You already enforced **compute → `shift(1)`** for all **hist** exogs (Section 3). Keep the unit test.
* **Window purity:** `step_size=h` ensures forecasts from different cutoffs do not **share** target indices; `refit=1` ensures each window trains only on strictly earlier data. **This is NF-guaranteed temporal ordering; don’t build embargo logic yourself.** 
* **Validation span:** `val_size=4*h` gives enough samples to estimate sCRPS/coverage per-window without leaking future information.
* **Insample use:** `predict_insample` predicts on train/val portions from the **stored dataset** and respects masks/indexing; use it for diagnostics only. 

---

#### 5.2.H When to attach **Conformal** (keep it minimal)

* If quantile/distributional runs **miss coverage** by more than ±2% on CV/test tails, attach **conformal** via NF’s `PredictionIntervals` (either in `fit` or `cross_validation`) and re-evaluate coverage at **80/90/95**. NF’s tutorial shows end-to-end usage with `PredictionIntervals`. **Do not** hack losses to “fix” coverage. 

---

#### 5.2.I Save/Load (for reproducibility & later inference)

* After selecting winners (or top-2 ensemble per §8), **save** using `nf.save(path, save_dataset=True)`; **load** with `NeuralForecast.load(path)` inside `run_predict.py`. These are **NF-native**; don’t roll your own serialization. 

---

## 6) Hyperparameter strategy (bounded, low-variance)

We will avoid massive hyperparameter grids. Instead, use informed small-scale searches and sequential refinement: 1\. **Reasonable defaults first:** We use NF’s defaults or slight tweaks as starting points (as in §4.1). 2\. **One model at a time pilot:** For example, take NHITS with default settings and run cross-val for h=16 to get a baseline sCRPS. 3\. **Tune key parameters:** If baseline is underperforming or to verify improvements, adjust a few critical hyperparams within a narrow range.

Proposed search spaces (to be executed via optuna or manual grid):

* **NHITS:**

* n_blocks: 2 or 3 (stacks) – more might overfit intraday noise.

* dropout_prob_theta: {0.0, 0.1, 0.2} – test some regularization.

* Possibly different downsampling factors (but we can stick to default \[1,2,3\] initially).

* **NBEATSx:**

* n_blocks: {1, 2} for each stack type (trend, seasonal, identity).

* mlp_units: e.g., try slightly larger like \[ \[512,512\], \[256,256\], \[256,256\] \] vs default \[512,512\] for all.

* dropout_prob_theta: {0.0, 0.1} (NBEATSx has internal dropout possibly on exogenous basis).

* **TiDE:**

* hidden_size: {256, 512}.

* num_layers: {2, 3}.

* dropout: {0.0, 0.1, 0.2}.

* **PatchTST:**

* d_model (hidden size): {128, 256} – balancing accuracy vs memory.

* n_heads: {4, 8}.

* patch_len: {16, 32}.

* stride: same as patch_len or half of it (to allow overlapping patches).

* revin: {True, False} – but likely True is beneficial for non-stationary crypto, per literature.

* **Global knobs:**

* learning_rate: {1e-3, 5e-4}.

* batch_size: {256, 512} (if memory is an issue) – larger batch can stabilize training but diminishing returns.

* input_size: we already have a rationale, but we might test 512 vs 1024 for smaller models to see if shorter history suffices.

**Search approach:** - Use NF’s integration with Optuna (if available) or manual experiments. NF does have an AutoMLP and similar, but we prefer control. - **Pilot on one horizon (h=16):** Do a limited search on h=16 which is a mid-range horizon, with 3-fold CV for quick feedback. Identify which model families and param combos look promising (e.g., maybe PatchTST with certain settings shines, or NBEATSx with dropout). - **Select top configurations:** Pick, say, the best 2 configurations per model type. - Expand those to a full 6-fold CV on all four horizons (4,8,16,32) to see consistency. It’s possible one config is best for short horizons and another for long; we might then keep both as separate “models” when ensembling. - We intentionally keep the hyperparam ranges tight (like 2-3 options each) to avoid combinatorial explosion. With 4 models, if each has \~3 options, that’s manageable.

**Evaluation metric for tuning:** sCRPS is primary (since it blends calibration and accuracy). We’ll also keep an eye on coverage (if a model gets good CRPS by being under-dispersed, we may penalize it).

We are **not chasing the last decimal of improvement** – we prefer simpler models unless a more complex one clearly and consistently outperforms by a good margin (say \>1.5-2% sCRPS reduction across multiple windows). Avoid over-tuning to one period.

If none of the variants of a model show advantage, we might drop that model type in final ensemble to reduce maintenance.

### 6.1 Search spaces (tight, model-specific + global)

Use **small, hand-curated sets** that map 1:1 to NF ctor args. No external HPO unless we explicitly opt into Auto* (optional note at the end).

#### Global knobs (all models)

| Knob                        | Values                                   | Notes                                                                                                                       |
| --------------------------- | ---------------------------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| `learning_rate`             | {1e-3, 5e-4}                             | All models expose `learning_rate`. Defaults around 1e-3 (PatchTST page shows typical ranges).             |
| `batch_size`                | {256, 512, 1024}                         | Use the largest that fits GPU; NF models take `batch_size`.                                               |
| `input_size`                | Model-specific (see below)               | Long contexts help intraday; all models expose `input_size`.                                              |
| `early_stop_patience_steps` | {200, 400}                               | NF supports early stopping via this arg; default often `-1` (off).                                        |
| `val_check_steps`           | {100}                                    | Validate periodically; param exists across models.                                                        |
| `scaler_type`               | {"robust"} (trial `"revin"` selectively) | Temporal scalers: `robust`, `invariant`, `revin`, etc. Keep defaults simple; try RevIN when drift hurts.  |

#### NHITS (NF: `models.nhits`)

| Param                | Values                          |
| -------------------- | ------------------------------- |
| `input_size`         | 1024                            |
| `n_blocks`           | {\[1,1,1], \[2,2,2]}            |
| `n_pool_kernel_size` | {\[2,2,1], \[4,2,1]}            |
| `dropout_prob_theta` | {0.0, 0.1, 0.2}                 |
| `learning_rate`      | {1e-3, 5e-4}                    |
| `batch_size`         | {512}                           |
| `max_steps`          | 20_000 (guarded by early stop) |

All names come from the NHITS signature (includes `n_blocks`, `n_pool_kernel_size`, `dropout_prob_theta`, `learning_rate`, `batch_size`, early-stopping args, and `scaler_type`). 

#### NBEATSx (NF: `models.nbeatsx`)

| Param                | Values                             |
| -------------------- | ---------------------------------- |
| `input_size`         | 1024                               |
| `n_blocks`           | {\[1,1,1], \[2,2,2]}               |
| `mlp_units`          | {\[\[256,256]]×3, \[\[512,512]]×3} |
| `dropout_prob_theta` | {0.0, 0.1}                         |
| `learning_rate`      | {1e-3, 5e-4}                       |
| `batch_size`         | {512}                              |
| `max_steps`          | 20_000                            |

Parameter names and semantics per NBEATSx doc (includes `n_blocks`, `mlp_units`, `dropout_prob_theta`, etc.). 

#### TiDE (NF: `models.tide`)

| Param                | Values          |
| -------------------- | --------------- |
| `input_size`         | 1024            |
| `hidden_size`        | {256, 512}      |
| `num_encoder_layers` | {1, 2}          |
| `num_decoder_layers` | {1, 2}          |
| `dropout`            | {0.0, 0.1, 0.2} |
| `learning_rate`      | {1e-3, 5e-4}    |
| `batch_size`         | {512}           |
| `max_steps`          | 20_000         |

TiDE page shows `hidden_size`, `{num_encoder_layers,num_decoder_layers}`, `dropout`, and the standard training knobs. 

#### PatchTST (NF: `models.patchtst`)

| Param            | Values          |
| ---------------- | --------------- |
| `input_size`     | 2048            |
| `patch_len`      | {8, 16}         |
| `stride`         | {8, 16}         |
| `n_heads`        | {8, 16}         |
| `hidden_size`    | {128, 256, 512} |
| `encoder_layers` | {2, 3}          |
| `revin`          | {True}          |
| `learning_rate`  | {1e-4, 5e-4}    |
| `batch_size`     | {256, 512}      |
| `max_steps`      | 10_000–20_000 |

All names in the PatchTST signature (includes `patch_len`, `stride`, `n_heads`, `hidden_size`, `encoder_layers`, `revin`, learning/early-stop/batch/scaling args). We bias to slightly **lower LR** on PatchTST. 

> **Scaling**: Keep `scaler_type="robust"` globally; try `revin=True` on PatchTST and, only if materially helpful, switch select others to `scaler_type="revin"`. Supported scaler names are documented in NF’s TemporalNorm. 

> **Loss selection**: Use **StudentT** (distributional) and **MQLoss/IQLoss** (quantile) exactly via NF’s loss constructors; sCRPS is available in NF losses/metrics and will be our primary score. 

---

### 6.2 Procedure: pilot → promote → full CV (no bloat)

**Goal:** keep variance low; stop when gains flatten. Use **NF cross_validation** exactly as in §5.

1. **Pilot (cheap)**

   * Horizon: **h=16** only.
   * Windows: **n_windows=3**, `step_size=16`, `val_size=64`, `refit=1`.
   * Sweep **at most 6–8 configs per model** using the ranges above.
   * Train both **StudentT** and **MQ** for **one** architecture if GPU budget is tight; otherwise both for all.
   * **Promote** any config that beats your current intra-model baseline by **≥1.5% mean sCRPS** (or is statistically tied but yields **better coverage** at 90%).
     (All CV args are NF-native; see `cross_validation` docs.) 

2. **Promote (moderate)**

   * Same horizon (**h=16**), use **n_windows=6**, `step_size=16`, `val_size=64`, `refit=1`.
   * Keep at most **2 configs per model** (one distributional, one quantile).
   * If quantile crossing appears, re-run with **IQLoss** (ISQF) or defer to conformal later. Loss classes & sCRPS are in NF losses docs. 

3. **Full CV (final)**

   * Run **all horizons**: h ∈ {4, 8, 16, 32}.
   * Windows: **n_windows=10**, `step_size=h`, `val_size=4*h`, `refit=1` (as defined in §5).
   * **Select** per horizon by mean **sCRPS** (primary), with coverage sanity (±2% at 80/90/95) as a tie-breaker; see §8 for simple ensembling rules.

4. **Stop rules**

   * If adding capacity (layers/hidden size) **improves mean sCRPS < \~1%**, stop increasing.
   * If `input_size↑` by 50% yields **no** consistent sCRPS lift and hurts throughput, revert.

> **Why not full HPO?** You don’t need Optuna/Ray for these small spaces. If time-boxed HPO is desired later, NF **Auto*** models provide built-in search orchestration across these ctor args (Ray/Optuna); we keep it off for v1 to avoid complexity. 

---

### 6.3 Concrete glue (tiny, no reinvention)

> **Create** `experiments/hpo_spaces.yaml` (centralized knobs used by a thin loop).
> **Path:** `experiments/hpo_spaces.yaml`

```yaml
global:
  learning_rate: [0.001, 0.0005]
  batch_size: [512, 256]
  early_stop_patience_steps: [400]
  val_check_steps: [100]

NHITS:
  input_size: [1024]
  n_blocks:
    - [1,1,1]
    - [2,2,2]
  n_pool_kernel_size:
    - [2,2,1]
    - [4,2,1]
  dropout_prob_theta: [0.0, 0.1, 0.2]
  max_steps: [20000]

NBEATSx:
  input_size: [1024]
  n_blocks:
    - [1,1,1]
    - [2,2,2]
  mlp_units:
    - [[256,256],[256,256],[256,256]]
    - [[512,512],[512,512],[512,512]]
  dropout_prob_theta: [0.0, 0.1]
  max_steps: [20000]

TiDE:
  input_size: [1024]
  hidden_size: [256, 512]
  num_encoder_layers: [1, 2]
  num_decoder_layers: [1, 2]
  dropout: [0.0, 0.1, 0.2]
  max_steps: [20000]

PatchTST:
  input_size: [2048]
  patch_len: [8, 16]
  stride: [8, 16]
  n_heads: [8, 16]
  hidden_size: [128, 256, 512]
  encoder_layers: [2, 3]
  revin: [true]
  max_steps: [10000, 20000]
```

> Parameter names map exactly to the model signatures in NF docs (NHITS/NBEATSx/TiDE/PatchTST). 

**Tiny search helper (uses your §4 factory & §5 CV):**

> **Create** `cv/hpo.py` (drop-in). **This is not a new backtester**; it just iterates configs and calls NF’s `cross_validation`.

```python
# cv/hpo.py
from __future__ import annotations
import itertools, copy, yaml, pandas as pd
from typing import Dict, List, Any
from neuralforecast import NeuralForecast
from nf_models.factories import instantiate_models
from cv.runner import run_cv, summarize_cv

def product_space(space: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    keys = list(space.keys())
    vals = [space[k] for k in keys]
    out = []
    for combo in itertools.product(*vals):
        out.append({k: v for k, v in zip(keys, combo)})
    return out

def expand_model_cfg(base: Dict[str, Any], tweaks: Dict[str, Any]) -> Dict[str, Any]:
    c = copy.deepcopy(base)
    c.update(tweaks)
    return c

def run_hpo(nf_df: pd.DataFrame, cfg: Dict, space_yaml_path: str, pilot: bool=True) -> pd.DataFrame:
    """Iterate tiny grids per model; return leaderboard sorted by sCRPS."""
    space = yaml.safe_load(open(space_yaml_path))
    global_knobs = space["global"]
    models_cfg = cfg["models"]

    results = []
    for model_entry in models_cfg:
        name, base = next(iter(model_entry.items()))
        mspace = space.get(name, {})
        variants = product_space(mspace) if mspace else [{}]

        for v in variants:
            # merge global knobs
            params = {**base, **global_knobs, **v}
            trial_cfg = copy.deepcopy(cfg)
            trial_cfg["models"] = [{name: params}]
            # instantiate NF model & run NF-native CV
            models = instantiate_models(trial_cfg,
                                        cfg.get("hist_exog_list", []),
                                        cfg.get("futr_exog_list", []),
                                        cfg.get("stat_exog_list", []))
            nf = NeuralForecast(models=models, freq=cfg["freq"])
            # pilot vs promote/full windowing comes from cfg
            cv_df = run_cv(nf, nf_df, cfg)
            # attach model alias for traceability
            if "y" not in cv_df.columns:
                cv_df = cv_df.merge(nf_df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")
            lb = summarize_cv(cv_df, y_col="y")
            alias = params.get("alias", name)
            lb.insert(0, "alias", alias)
            for k, val in v.items():
                lb[f"hp_{k}"] = val
            results.append(lb)

    cat = pd.concat(results, ignore_index=True)
    return cat.sort_values("sCRPS", na_position="last")
```

**How to use it (pilot on h=16):**

```bash
# set experiments/h16.yaml to n_windows=3, step_size=16, val_size=64 (per §6.2 Pilot)
python - <<'PY'
import yaml, pandas as pd
from cv.hpo import run_hpo
cfg = yaml.safe_load(open("experiments/h16.yaml"))
df = pd.read_parquet("data/btc_15min.parquet")  # canonical frame with y + exogs
leader = run_hpo(df, cfg, "experiments/hpo_spaces.yaml", pilot=True)
leader.to_parquet("experiments/h16/hpo_leaderboard.parquet")
print(leader.head(10))
PY
```

> This **does not** replace NF CV. It wraps **your** §4 factory and **NF’s** `cross_validation`. There is **no** custom scoring beyond simple aggregation. (All NF CV semantics per docs.) 

---

### 6.4 Promotion thresholds & bookkeeping

* **Promotion rule:** promote only configs with **≥1.5%** mean **sCRPS** gain vs. current per-model baseline (or indistinguishable sCRPS but **better coverage** at 90%).
* **Cap survivors:** keep **≤2** configs per model for full CV (one **StudentT**, one **MQ/IQ**).
* **Artifacts:** persist `hpo_leaderboard.parquet` per horizon, plus the exact YAML (frozen) used.

---

### 6.5 Practical guards (don’t ignore)

* **OOM on PatchTST/NHITS:** drop `batch_size`, reduce `hidden_size`/`n_heads` (PatchTST), or shorten `input_size`. PatchTST’s args exist precisely for this. 
* **Overfitting:** if train loss keeps falling but val sCRPS flatlines, **lower** `max_steps`, **increase** `early_stop_patience_steps` to 400, or **add** small dropout (0.1–0.2). NF exposes `dropout`/`dropout_prob_theta` where appropriate. 
* **Scale drift:** if live coverage decays, trial **RevIN** (`revin=True` in PatchTST; or switch `scaler_type` to `"revin"` for others). Temporal scalers list includes `'revin'`. 

---

### 6.6 (Optional) Using NF Auto* for time-boxed HPO (only if needed)

If you need automated search without writing loops, NF exposes **AutoModels** (e.g., `AutoNHITS`, `AutoPatchTST`) that run grid/random/Bayesian search and pick the best config on a validation set (Ray/Optuna backends). Keep disabled for v1 to avoid moving parts; consider only if the tiny grids above stall. 

---

## 7) Model selection & simple ensembling

After cross-validation: - For each horizon *h*, rank models by **mean sCRPS** across all CV windows (primary metric). Also consider std deviation of sCRPS across windows (stability). - **Select best single model** per horizon as a benchmark. For instance, for h=16, maybe PatchTST(StudentT) is best. - Then evaluate if an **ensemble** of top models improves performance. Ensembling can often improve calibration and reduce variance of errors.

We will start with *simple ensembles*: e.g., take the **mean** or **median** of the forecasts from the top 2 models. NF doesn’t have a built-in ensemble class, but we can easily blend predictions post-hoc: - If two models produce distributions, we can mix their sample draws or parameters (though mixing distributions is non-trivial; easier to mix quantile outputs). - Simpler: blend their point forecasts for accuracy and either take an average of their predictive variances for distribution or do quantile averaging.

A pragmatic approach: - Take top 2 models (say A and B). Compute an ensemble median \= (median_A \+ median_B)/2. For intervals, do similar or take the min of lower bounds and max of upper bounds to be conservative (which approximates merging distributions). - Compute sCRPS of this blended forecast on CV via a quick evaluation (we can derive it from the stored results). - If the ensemble’s sCRPS is better than the best single model by a noticeable amount (and doesn’t screw up coverage), we adopt the ensemble. If improvement is marginal (\<1%), we stick to the single model to save complexity.

We will **not** attempt complex stacking or meta-learners for ensemble weights at first. Given only a handful of models, equal-weight or simple averaging is often robust. If needed later, we could consider a weighted ensemble (weights proportional to 1/sCRPS or using a brief least-squares fit on the cross-val predictions to minimize error). But that adds potential for overfit, so initially keep it simple.

**Note:** If one model consistently overshoots PIs and another undershoots, an ensemble might naturally improve calibration by averaging their distributions. We’ll watch PIT and coverage – if ensemble gives closer-to-uniform PIT than individuals, that’s a big win.

In summary, the output of this stage for each horizon is either one chosen model or a two-model ensemble that will be taken forward.

### 7.1 Selection protocol (per horizon)

* **Input:** `experiments/h{h}/cv_raw.parquet` (from §5) merged with `y`.
* **Primary ranker:** mean **sCRPS** across CV windows (use dense quantiles only for promoted configs; otherwise use MAE as a temporary ranker).
* **Stability tie-breakers:** (1) sCRPS std across windows (lower is better), (2) empirical coverage closeness at 90%, (3) MAE.
* **Winner per h:** the single model with best mean sCRPS (or, if dense quantiles absent, the one that will be re-run with dense quantiles to compute sCRPS before finalizing).

**Deliverable:** a CSV `experiments/h{h}/leaderboard.csv` with columns:
`model, mean_sCRPS, std_sCRPS, MAE, RMSE, cov80, cov90, cov95`.

> Note: if your CV was run with only `[80,90,95]` levels, sCRPS won’t be computable; re-run **promoted** configs with a dense grid (e.g., quantiles 1–99 or at least 5,10,…,95) and then compute sCRPS for the final decision.

### 7.2 Simple ensembles (top-2 only)

**When to try:** Only if the top-2 single models are within \~1–2% sCRPS of each other or if coverage of the best model is unstable. Do **not** ensemble more than two for v1.

**Rules:**

* **Point forecasts:** equal-weight mean (`(A+B)/2`) or median-of-two (same as mean).
* **Intervals/quantiles:**

  * If you have **quantiles**: average quantiles level-wise (e.g., `q̂_0.9^ens = 0.5*q̂_0.9^A + 0.5*q̂_0.9^B`). This preserves monotonicity.
  * If you only have **lo/hi** at levels (80/90/95): average lo’s and hi’s level-wise. If calibration drifts low, use conservative combine: `lo=min(lo_A,lo_B)`, `hi=max(hi_A,hi_B)`.
* **Do not** mix param distributions (e.g., don’t average Student-t parameters). Stay in the prediction/quantile space.

**Adoption threshold:** Adopt the ensemble **only if** it improves mean sCRPS by **≥1.0%** (or equals sCRPS but improves 90% coverage toward nominal). Otherwise, ship the best single model.

### 7.3 Drop-in utilities

> Create `uq/ensembles.py`.

```python
# uq/ensembles.py
from __future__ import annotations
import pandas as pd
from typing import List, Tuple

def make_equal_weight_ensemble(cv_df: pd.DataFrame, model_a: str, model_b: str, alias: str) -> pd.DataFrame:
    """Create equal-weight ensemble columns for point and any available intervals."""
    out = cv_df.copy()
    # point
    out[alias] = 0.5 * out[model_a] + 0.5 * out[model_b]
    # quantiles: columns like "<model>-qXX"
    qcols_a = sorted([c for c in out.columns if c.startswith(f"{model_a}-q")])
    for qa in qcols_a:
        q = qa.split("-q")[-1]
        qb = f"{model_b}-q{q}"
        if qb in out:
            out[f"{alias}-q{q}"] = 0.5 * out[qa] + 0.5 * out[qb]
    # intervals: "<model>-lo-90", "<model>-hi-90"
    for L in (80, 90, 95):
        lo_a, hi_a = f"{model_a}-lo-{L}", f"{model_a}-hi-{L}"
        lo_b, hi_b = f"{model_b}-lo-{L}", f"{model_b}-hi-{L}"
        if lo_a in out and lo_b in out:
            out[f"{alias}-lo-{L}"] = 0.5 * out[lo_a] + 0.5 * out[lo_b]
        if hi_a in out and hi_b in out:
            out[f"{alias}-hi-{L}"] = 0.5 * out[hi_a] + 0.5 * out[hi_b]
    return out

def make_conservative_band(cv_df: pd.DataFrame, alias_src: str, alias_out: str) -> pd.DataFrame:
    """Widen ensemble intervals conservatively using min(lo), max(hi) across constituents if present."""
    out = cv_df.copy()
    for L in (80, 90, 95):
        # assumes you already built alias_src intervals by averaging;
        # replace them with conservative min/max when both members are known in metadata
        # caller should precompute min/max and pass columns; keep as placeholder if not available
        pass
    return out
```

> Extend your existing `cv/runner.py` with an **ensemble scorer**:

```python
# cv/runner.py (add)
from uq.ensembles import make_equal_weight_ensemble
from .runner import summarize_cv as _summarize  # if in same file, adjust imports

def evaluate_top2_ensemble(cv_df: pd.DataFrame, top2: Tuple[str,str], alias: str="ENS2") -> pd.DataFrame:
    ens_df = make_equal_weight_ensemble(cv_df, top2[0], top2[1], alias)
    # merge y if needed before summarize
    if "y" not in ens_df.columns:
        raise ValueError("y required in cv_df to summarize ensemble.")
    lb = _summarize(ens_df, y_col="y")
    # return only the row for alias and the two constituents for comparison
    return lb[lb["model"].isin([top2[0], top2[1], alias])].reset_index(drop=True)
```

### 7.4 Workflow (per h)

1. Load `cv_raw.parquet` and compute leaderboard (from §5’s `summarize_cv`).
2. Identify **top-2 models** by mean sCRPS (or MAE if sCRPS missing at this stage).
3. Build an **ENS2** equal-weight blend via `evaluate_top2_ensemble`.
4. Compare metrics:

   * If `ENS2.mean_sCRPS ≤ best.mean_sCRPS * 0.99` (≥1% better) **and** coverage at 90% isn’t worse by >1.5pp, **adopt** ENS2.
   * Else **keep** best single model.
5. Persist:

   * `experiments/h{h}/selection.json` with `{ "winner": "<alias>", "members": ["A","B"]? }`
   * If winner is ensemble, also store its **member model paths** for reproducible retraining/inference.

### 7.5 Final fit & save (single vs ensemble)

* **Single winner:**

  * Refit the winning config on **all data** (using `val_size` for early stopping).
  * `nf.save("experiments/h{h}/best/", save_dataset=True, overwrite=True)`.

* **Ensemble winner:** two pragmatic options:

  1. **Save both constituent models** separately under `experiments/h{h}/best/A/` and `.../B/`. In inference, **load both** and blend predictions on the fly (equal-weight).
  2. **One NF object containing both models:** Save that ensemble directory and at inference time, compute both and blend via the same utility.

> Keep it simple: option **1** is fine for v1 and keeps retraining flexible.

### 7.6 Inference path for ensembles (hook for §10)

In `run_predict.py`:

* Load the two saved NF objects (or one with two models).
* Produce predictions for each.
* Apply the **same equal-weight** logic to create `ENS2` columns (including intervals/quantiles).
* If conformal was adopted later, apply the same conformal **adjustment deltas** to the ensemble bounds (store deltas per level per horizon in `experiments/h{h}/conformal.json`).

### 7.7 Guardrails

* Don’t ensemble a **quantile** model with a **distributional** one unless you are averaging **quantiles** (request levels for the distributional model; never average params).
* If one model lacks intervals, you can:

  * request levels during prediction for both, or
  * blend **point** only and report intervals from the better-calibrated member (not ideal; prefer requesting levels for both).
* Keep **exact** column naming stable: `ENS2` becomes just another model name for downstream code.

---

## 8) Uncertainty & calibration

Handling uncertainty is a core requirement. We have multiple layers to ensure our prediction intervals (PIs) are trustworthy:

### 8.1 Quantile vs Distribution training

We will compare: - **Quantile-trained models (MQ or IQLoss):** They directly yield quantiles. We must check for **quantile crossing** (e.g., the 90th percentile forecast should never be below the 50th). NF’s MQLoss with level produces multiple quantile outputs and typically ensures non-crossing by construction if using *ISQF* (Incremental Spread Quantile Forecasting) variant. If we see any crossing, Nixtla offers the IQLoss (Implicit Quantile Network) which tends to enforce monotonic quantiles by construction, or we can post-process by sorting the quantiles. - **Distribution-trained models (Student-T):** They output (mu, sigma, nu) for Student-T. We get parametric intervals naturally. We must verify the actual coverage vs theoretical. Student-T has heavy tails, which is good for financial data. If its learned nu (degrees of freedom) is low, it indicates heavy tails which might improve coverage on extreme moves.

One benefit of distribution models: we can derive the whole distribution and calculate **Continuous Ranked Probability Score (CRPS)** in closed form or by sampling, and also easily get any quantile. We will compute sCRPS for these directly since NF’s DistributionLoss is designed for that.

If a model has high sharpness (narrow intervals) but under-covers (actual coverage \< nominal), it means it’s underestimating uncertainty – possibly overfitting. Conversely, too wide intervals (over-covers significantly) are safe but not useful (low sharpness).

We will use the cross-val results to decide: - If distribution models consistently miss the mark on coverage (say 80% interval only covers 60% of real points), whereas quantile models do better, we might favor quantile loss. - Or vice versa: maybe Student-T gives a better log-likelihood and calibrated tails, then stick with it.

* **Start with two parallel trainings per model config (per §4/§6):**

  1. **Distributional**: `loss=DistributionLoss("StudentT")` — robust to heavy-tailed intraday returns. 
  2. **Quantile**: `loss=MQLoss(...)` (or **IQLoss/ISQF** if you see quantile crossing). NF exposes these losses in its PyTorch losses module.
* **Selection protocol:** choose by **mean sCRPS** on CV windows (primary), with empirical coverage at **90%** as tie-breaker. NF documents sCRPS in the losses/metrics section. 
* **Symptoms & switches:**

  * **Under-dispersion (too narrow)**: U-shaped PIT; low coverage → prefer quantile training or keep distributional and add **conformal** (below). Gneiting & Raftery recommend PIT to assess calibration.
  * **Quantile crossing**: switch MQLoss → **IQLoss/ISQF** (monotone quantiles), or keep MQLoss and apply conformal.

> **NF wiring:** quantiles/intervals are requested at prediction via `predict(level=[80,90,95])` (intervals) or `predict(quantiles=[...])` (quantile grid). 

### 8.2 Conformal prediction intervals

Regardless of training loss, we have a **post-training calibration** step available: *Conformal Prediction*. This uses past forecast errors to adjust interval widths to achieve the desired empirical coverage

Nixtla’s StatsForecast library demonstrates split-conformal for time series, and the same concept applies here: - After selecting a final model (or ensemble), we can take its residuals on a recent validation set or cross-val windows and determine an appropriate adjustment (e.g., an alpha-quantile of absolute errors). - For example, for 90% PI, find the 95th percentile of |forecast error| over the calibration set and then inflate the prediction intervals by that amount (this is absolute error method suitable if distribution is roughly symmetric after some transformation).

NF doesn’t have a one-line conformal for neural models yet, but we can implement: - Use the last n_windows of cross-val as calibration: For each forecast in CV, check if actual was inside the forecast distribution’s, say, 50% central interval. Compute the empirical distribution of residuals or the needed delta to cover the misses. - Alternatively, treat the point forecast as median and errors as distribution-free, apply *absolute residual* conformal: for 90% interval, take 95th percentile of |y - y_hat| (since 90% PI means 5% tail on each side). - Add this buffer to the model’s forecast quantiles.

This yields prediction intervals with guaranteed coverage (under the assumption that future residuals behave like past). The Medium article confirms that NeuralForecast can use cross-val windows to calibrate intervals on a point model – we’ll do exactly that if needed.

We aim for **80%, 90%, 95%** intervals that achieve approximately the nominal coverage on test. We’ll judge success as coverage within ±2% of nominal (e.g., 90% PI actually covers 88-92% of outcomes over a long run).

Conformal is especially useful if the model’s own uncertainty estimates are biased. For instance, a model might be too confident (underestimates variance) – conformal will widen the intervals just enough. Or if a model was trained with MAE (no probabilistic output), conformal provides intervals where none existed.

One caution: Conformal intervals are typically **constant width for all forecasts** (if using absolute residual method) or vary only with a conditioning variable if we do a more refined version. That might under-account for heteroscedasticity (volatility clustering). Our approach to mitigate that is to include volatility features and/or use distributional loss so the model itself captures volatility. If it doesn’t fully succeed, conformal ensures overall coverage even if at times intervals could be too narrow or too wide in certain regimes.

We will apply conformal after model selection: - For the final selected model (per horizon), use recent backtest residuals to compute an *interval adjustment factor* for each desired level. - Then when outputting live forecasts, take the model’s prediction (mean or median) and add/subtract this delta for the interval bounds.

* **When:** if CV/test coverage at **80/90/95** misses nominal by > **±2pp**, attach **Conformal Prediction**. NF provides a tutorial and utilities to conformalize model outputs—don’t hack losses to “fix” coverage. 
* **What it does:** builds **finite-sample calibrated** intervals on top of any forecaster (neural or classical). (StatsForecast docs summarize the same approach.) 

**Minimal NF pattern (attach during fit/CV):**

```python
# inside run_train.py before fit/cv (only for runs where you need conformal)
from neuralforecast import NeuralForecast
from neuralforecast.utils import PredictionIntervals  # NF tutorial shows this utility

# Example: conformalized CV (you can also conformalize after selecting a winner)
pi = PredictionIntervals(n_windows=cfg["n_windows"], level=[80,90,95])  # CV-based conformal
cv_df = nf.cross_validation(df=nf_df,
                            n_windows=cfg["n_windows"],
                            step_size=cfg["step_size"],
                            val_size=cfg["val_size"],
                            refit=1,
                            prediction_intervals=pi)
# Columns will include Model-lo-90/Model-hi-90, etc., now conformalized.
```

*(Use the same idea when calling `fit(...)` if you want conformal intervals during a single split; see NF’s conformal tutorial.)* 

* **Acceptance band:** after conformalization, require coverage within **±2pp** at 80/90/95 on the **test tail** (not only CV). Store the empirical coverage deltas with the model artifact for use in live inference.

### 8.3 Diagnostic checks

To know if our uncertainty is good: - **PIT histogram:** For distribution models, as mentioned, transform residuals to uniform. For a well-calibrated model, the PIT histogram should be flat. We will generate this for final models (likely by sampling from predicted Student-T or using CDF if analytic). - **Coverage by volatility:** Split the backtest periods into buckets by realized volatility or volume. Check interval coverage in each bucket. We expect possibly lower coverage during high volatility if model underestimates those moves. If we see that, we might incorporate a volatility-dependent adjustment (e.g., add a bigger conformal buffer when a volatility regime indicator is high). - **Sharpness vs calibration tradeoff:** We’ll document the average prediction interval width for each model vs its coverage. Ideally, we want the narrowest interval that still covers \~90%. If a model’s 90% interval covers 99% of points, it’s too wide (could be improved to be sharper). If covers 80%, it’s too narrow (risky). We adjust accordingly.

We are particularly interested in the extreme tail behavior – a Student-T with low degrees of freedom might give very fat tails which help capture the occasional huge BTC moves. Quantile models might struggle to learn those unless explicitly seeing many examples. We will pay attention to whether 95% intervals correctly include the wild outliers (they should, by definition, except 5% of time).

In summary, we ensure the final forecasting system not only provides a “best guess” but also a realistic uncertainty range that users can trust.

* **Coverage table** at 80/90/95 on CV windows and test tail (by horizon).
* **PIT histogram** (approximate; rank-based on quantile grid) → should be \~Uniform\[0,1]. U-shape = under-dispersed; hump = over-dispersed; skewed = biased.
* **Coverage by volatility deciles** (sort by rolling σ of returns) to detect heteroscedastic under-coverage.

**Tiny glue (append to `uq/diag.py`):**

```python
import numpy as np, pandas as pd

def coverage_table(preds: pd.DataFrame, y_col="y", levels=(80,90,95), model_cols=None) -> pd.DataFrame:
    if model_cols is None:
        model_cols = [c for c in preds.columns if c not in ("unique_id","ds","cutoff",y_col)]
    rows=[]
    for m in model_cols:
        for L in levels:
            lo, hi = f"{m}-lo-{L}", f"{m}-hi-{L}"
            if lo in preds and hi in preds and y_col in preds:
                p = preds[[y_col, lo, hi]].dropna()
                rows.append({"model": m, "level": L, "coverage": float(((p[y_col]>=p[lo])&(p[y_col]<=p[hi])).mean())})
    return pd.DataFrame(rows)

def pit_from_quantiles(df: pd.DataFrame, y_col: str, qcols: list[str]) -> pd.Series:
    # rank-based PIT; see Gneiting & Raftery (2007)
    arr_q = df[qcols].to_numpy()
    y = df[y_col].to_numpy()[:,None]
    ranks = (arr_q <= y).sum(axis=1)
    return pd.Series(ranks / (arr_q.shape[1] + 1), index=df.index, name="pit")

def coverage_by_vol_decile(preds: pd.DataFrame, vol: pd.Series, model: str, level: int, y_col="y") -> pd.DataFrame:
    lo, hi = f"{model}-lo-{level}", f"{model}-hi-{level}"
    df = preds[[y_col, lo, hi, "ds"]].dropna().merge(vol.rename("vol"), left_on="ds", right_index=True)
    df["decile"] = pd.qcut(df["vol"], 10, labels=False, duplicates="drop")
    out = df.groupby("decile").apply(lambda g: ((g[y_col]>=g[lo])&(g[y_col]<=g[hi])).mean()).rename("coverage").reset_index()
    out["level"]=level; out["model"]=model
    return out
```

### 8.4 Practical fixes (pick the smallest hammer)

* **Under-dispersion (U-shaped PIT; low coverage):**

  1. Increase quantile grid (for MQ/IQ) or request denser `level=[... ]` for distributional to improve sCRPS estimate,
  2. **Attach conformal** (preferred),
  3. If still poor, trial **`scaler_type="invariant"`** or **RevIN** to stabilize scale (NF TemporalNorm docs). 
* **Over-dispersion (hump PIT; wide intervals):**

  1. Reduce dropout a notch (if excessive),
  2. Switch to **StudentT** from over-widened quantiles (or tighten quantile set),
  3. Re-check feature cap (too many weak, noisy exogs can inflate uncertainty).
* **Skewed PIT (systematic bias):**

  1. Add/keep **calendar futr_exogs**;
  2. Ensure **MTF alignment + shift(1)** is correct (leakage can fake bias);
  3. Revisit target (log-returns) and winsor thresholds (per §2).

### 8.5 What to persist for calibration

* `experiments/h{h}/leaderboard.parquet` — includes coverage at all levels and (when computed) sCRPS.
* `experiments/h{h}/pit_hist.png` — PIT diagnostics for finalists.
* If conformal is used: `experiments/h{h}/conformal.json` — store method, levels, and per-level **empirical deltas** (useful to sanity-check live drift).
* `reports/h{h}/vol_decile_coverage.csv` — coverage by volatility decile.

---

## 9) Training & evaluation workflow

Everything will be driven by configuration files and two main entry point scripts.

### 9.1 Experiment configuration

Use YAML (or JSON) to specify each experiment/run. For example, experiments/h16.yaml might contain:

\# experiments/h16.yaml  
h: 16                \# horizon (4h)  
freq: "15min"  
val_size: 64         \# 4*h, used inside each CV train for early stopping  
n_windows: 6  
step_size: 16  
refit: true

models:  
  - name: "NHITS-StudentT"  
    class: NHITS  
    loss: DistributionLoss  
    distribution: StudentT  
    input_size: 1024  
    dropout_prob_theta: 0.1  
    early_stop_patience_steps: 400  
    \# ... other hyperparams or defaults  
  - name: "NHITS-MQ"  
    class: NHITS  
    loss: MQLoss  
    level: \[10,50,90\]  
    input_size: 1024  
  - name: "PatchTST-StudentT"  
    class: PatchTST  
    loss: DistributionLoss  
    distribution: StudentT  
    input_size: 2048  
    patch_len: 16  
    n_heads: 8  
    scaler_type: revin  
    dropout: 0.1

(The YAML might mirror NF’s model API closely. We include a human-readable name for clarity in outputs.)

Global settings like freq and CV parameters can be shared or repeated in each config. We list multiple models to compare.

### 9.1 YAML-driven experiment configs (minimal, explicit)

> **Files** (under `/experiments/`): keep one YAML per horizon and one global defaults file.
>
> * `experiments/defaults.yaml` — global switches; imported by per-horizon YAMLs.
> * `experiments/h{h}.yaml` — horizon-specific overrides (models, windows, levels).

**`experiments/defaults.yaml`**

```yaml
seed: 1337
freq: "15min"

# CV defaults (see §5)
n_windows: 6
refit: 1
# These are placeholders; each h-file will set step_size=h and val_size=4*h.
step_size: null
val_size: null

# Intervals/quantiles requested during CV/predict
levels: [80, 90, 95]
dense_quantiles: []  # optional: e.g., [0.01,0.02,...,0.99] for sCRPS on finalists

# Exogs are populated by the features builder; lists here keep wiring explicit.
hist_exog_list: []
futr_exog_list: [minute_of_day, day_of_week, weekend, holiday]
stat_exog_list: [asset_id]

# Scalers per model (NF TemporalNorm names)
scaler_type:
  default: robust
  PatchTST: revin  # PatchTST also supports a model-level revin flag

# Paths
paths:
  data: "data/btc_15min.parquet"
  runs: "experiments"
  reports: "reports"
```

**`experiments/h16.yaml`** (example; mirror for h=4/8/32 with their `h`, `step_size`, `val_size`)

```yaml
# h=16 → 4 hours at 15-min base
h: 16
step_size: 16
val_size: 64

# Select a tight portfolio per §4/§6
models:
  - NHITS:
      alias: NHITS_t1024_T
      input_size: 1024
      loss: {kind: studentt}
      learning_rate: 0.001
      batch_size: 512
      n_blocks: [1,1,1]
      n_pool_kernel_size: [2,2,1]
      early_stop_patience_steps: 400
      max_steps: 20000
  - NBEATSx:
      alias: NBEATSx_t1024_MQ
      input_size: 1024
      loss: {kind: mqloss, quantiles: [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95]}
      learning_rate: 0.001
      batch_size: 512
      n_blocks: [1,1,1]
      dropout_prob_theta: 0.1
      early_stop_patience_steps: 400
      max_steps: 20000
  - TiDE:
      alias: TiDE_t1024_MQ
      input_size: 1024
      hidden_size: 512
      num_encoder_layers: 2
      num_decoder_layers: 2
      dropout: 0.1
      loss: {kind: mqloss}
      learning_rate: 0.001
      batch_size: 512
      early_stop_patience_steps: 400
      max_steps: 20000
  - PatchTST:
      alias: PatchTST_t2048_T
      input_size: 2048
      patch_len: 16
      stride: 16
      n_heads: 8
      hidden_size: 512
      revin: true
      loss: {kind: studentt}
      learning_rate: 0.0005
      batch_size: 512
      early_stop_patience_steps: 400
      max_steps: 20000
```

### 9.2 `run_train.py` — NF-native training, insample diagnostics, CV, artifacts

```python
# run_train.py
import argparse, os, yaml, pandas as pd, numpy as np
from neuralforecast import NeuralForecast
from nf_models.factories import instantiate_models
from utils.io import load_canonical_frame
from cv.runner import run_cv, summarize_cv
from uq.diag import coverage_table  # §8
from pathlib import Path

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="Path to experiments/h{h}.yaml")
    ap.add_argument("--defaults", default="experiments/defaults.yaml")
    ap.add_argument("--save", action="store_true", help="Save trained NF object after CV")
    return ap.parse_args()

def merge_cfg(defaults, exp):
    out = defaults.copy()
    out.update({k:v for k,v in exp.items() if v is not None})
    return out

if __name__ == "__main__":
    args = parse_args()
    dcfg = yaml.safe_load(open(args.defaults))
    ecfg = yaml.safe_load(open(args.exp))
    cfg = merge_cfg(dcfg, ecfg)

    # Load canonical long frame (already prepped per §§1–3)
    df = load_canonical_frame(cfg["paths"]["data"])  # must have ['unique_id','ds','y', exogs], UTC, regular grid

    # Build NF models from YAML (exact NF ctor args, including exog lists & scalers)
    models = instantiate_models(cfg,
                                cfg.get("hist_exog_list", []),
                                cfg.get("futr_exog_list", []),
                                cfg.get("stat_exog_list", []))
    nf = NeuralForecast(models=models, freq=cfg["freq"])  # NF Core wrapper (fit/predict/CV/save/load) 

    # 1) Fit once for quick insample diagnostics (PIT/coverage sanity)
    nf.fit(df=df, val_size=cfg["val_size"])  # val split is NF-native in fit 

    # optional dense quantiles for PIT/sCRPS on finalists; otherwise just levels
    levels = cfg.get("levels", [80,90,95])
    dense_q = cfg.get("dense_quantiles", [])
    ins = nf.predict_insample(step_size=1,
                              level=levels if not dense_q else None,
                              quantiles=dense_q if dense_q else None)  # NF-native 

    # Persist quick insample for diagnostics
    outdir = Path(cfg["paths"]["runs"]) / f"h{cfg['h']}"
    outdir.mkdir(parents=True, exist_ok=True)
    ins = ins.merge(df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")
    ins.to_parquet(outdir / "insample.parquet")

    # 2) Temporal CV (NF-native)
    cv_df = nf.cross_validation(df=df,
                                n_windows=cfg["n_windows"],
                                step_size=cfg["step_size"],
                                val_size=cfg["val_size"],
                                refit=cfg["refit"],
                                level=levels if not dense_q else None,
                                quantiles=dense_q if dense_q else None)  # CV semantics per docs 

    # Attach y (for metrics aggregation)
    if "y" not in cv_df.columns:
        cv_df = cv_df.merge(df[["unique_id","ds","y"]], on=["unique_id","ds"], how="left")

    # Leaderboard + coverage
    leaderboard = summarize_cv(cv_df, y_col="y")     # §5 utility
    cover = coverage_table(cv_df, y_col="y", levels=levels)

    cv_df.to_parquet(outdir / "cv_raw.parquet")
    leaderboard.to_parquet(outdir / "leaderboard.parquet")
    cover.to_parquet(outdir / "coverage.parquet")

    # 3) Optionally save the fitted NF object used above (for quick reuse)
    if args.save:
        nf.save(path=str(outdir / "chkpt"), save_dataset=True, overwrite=True)  # NF save/load tutorial 

    print(leaderboard.sort_values("sCRPS", na_position="last").head(10))
```

**Why this is correct (and lean):**

* `fit(..., val_size=...)` is NF-native and stores the dataset for later `predict_insample`. 
* `predict_insample(step_size, level|quantiles)` is the official way to get train/val backcasts for diagnostics. 
* `cross_validation(n_windows, step_size, val_size, refit, level|quantiles)` is NF’s CV; we don’t write our own. 
* `save/load` uses the NF capability that persists model weights/config + dataset. 

> **sCRPS note:** NF exposes **`sCRPS`** in `neuralforecast.losses.pytorch` for evaluation. If you request **dense quantiles** at CV/predict time (`quantiles=[0.01,...,0.99]`), compute sCRPS on promoted configs precisely with NF’s metric. Keep the dense grid **only for finalists** to control runtime. 


### 9.3 `run_predict.py` — batch inference with PIs, save/load

After we have selected final model(s) (this could be done manually by inspecting metrics and deciding configurations to promote), we will train those on all available data and use them for live forecasting.

run_predict.py will: - Load the latest saved model (or train a fresh one if not using saved). - Append the newest data point(s) to the DataFrame, compute features for just those new timestamps (and the necessary history window). - Call nf.predict(futr_df=...) or simply nf.predict() depending on whether NF needs a separate futr_df for future exog (since we have futr exog like calendar, we can generate it easily for the next h steps). - The result is a DataFrame with columns including the forecast mean/median and possibly interval bounds. For example, NF might return columns like NHITS-median, NHITS-lo-90, NHITS-hi-90, etc., if we configured levels or if we do predict(level=\[80,90,95\]). - We then post-process if needed: - If using conformal, adjust the returned intervals by our precomputed offsets. - Ensure no negative prices (if we were forecasting price directly, but since we forecast return, we convert to price). - Save or print the forecast.

We will implement an nf.save after training so that we can load the model quickly for inference:

nf.save(path="experiments/h16/best_model/", overwrite=True, save_dataset=True)  \# Save model and config\[40\]  
...  
\# In run_predict.py  
nf_loaded \= NeuralForecast.load(path="experiments/h16/best_model/")【34†L288-L296】  
Y_hat \= nf_loaded.predict(futr_df=future_exog_df, level=\[80,90,95\])

NF’s save will have saved each model’s weights and the configuration. We specify save_dataset=True so that any data transformations (scalers, etc.) are also saved in case needed. Loading it restores the model ensemble ready to predict.

**Batch vs one-step:** Typically, we will call predict with horizon *h* (e.g., 8 hours) and get all those steps in one call (since our models are direct multi-step forecast). This is efficient.

**Speed considerations:** - We will pre-create the feature DataFrame for the next h steps: - For calendar features, it’s straightforward to extend timestamps and compute day_of_week, etc. - For any other future exog (we might not have other futr exog in this project). - NF can accept a futr_df containing future exogenous values for those h timestamps. We will construct that and pass to predict.

The output DataFrame with forecasts will be written to something like reports/h16/latest_forecast.csv or perhaps pushed to a database or message queue in a real deployment. For now, file output is fine.

The prediction step will run every 15 minutes in a loop (if deployed live). We have to ensure that computing features is fast enough. Vectorbt/TA-Lib is very fast for bulk, but computing just one step’s features might involve recalculating the latest window of indicators: - We can optimize by caching the last state of indicators (many indicators like EMA can be updated incrementally). However, simpler is to just recompute indicators on a rolling window of the last, say, 3000 points for each new point, which should be fine (3000 * 15min \= 31 hours, enough for ATR etc.). 3000 data points with TA-Lib is milliseconds of compute, acceptable for real-time.

Memory: our models are not huge (except maybe PatchTST) – we expect a few hundred thousand parameters each, which is trivial for modern GPUs/CPUs.

Finally, ensure the inference pipeline handles exceptions (e.g., if data is missing or model isn’t loaded) gracefully – log error and continue.

> **Path:** `run_predict.py`. This loads the selected model(s) (single or the two members of your §7 ENS2), rebuilds the **tail** exogs (already leakage-safe), and calls `predict(level=[80,90,95])`. `predict` accepts `level` or `quantiles`; if using ensembles, blend predictions **after** calling `predict`. 

```python
# run_predict.py
import argparse, yaml, pandas as pd
from pathlib import Path
from neuralforecast import NeuralForecast
from utils.io import load_canonical_frame
from uq.ensembles import make_equal_weight_ensemble  # §7
from utils.tail_build import build_tail_exogs  # tiny helper to recompute last window features

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiments/h{h}.yaml")
    ap.add_argument("--model_path", required=True, help="experiments/h{h}/best/ or ensemble json")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(open(args.exp))
    df = load_canonical_frame("data/btc_15min.parquet")  # canonical long frame; has y + exogs

    # Rebuild tail exogs for the last input_size + margin (helper keeps shift(1) discipline)
    df_tail = build_tail_exogs(df, input_sizes={"default":1024, "PatchTST":2048})

    # Load winner (single model) saved with NF.save; predict next h with intervals
    nf = NeuralForecast.load(path=args.model_path)  # official load API 
    preds = nf.predict(df=df_tail, level=[80,90,95])  # documented predict params: level|quantiles 

    # If using an equal-weight 2-model ensemble: load both folders, predict each, then blend.
    # (Keep a small wrapper script if you adopted ENS2 in §7.)

    outdir = Path("reports") / f"h{cfg['h']}"
    outdir.mkdir(parents=True, exist_ok=True)
    preds.to_parquet(outdir / f"preds_{pd.Timestamp.utcnow().strftime('%Y%m%dT%H%M%SZ')}.parquet")
```

**Notes:**

* `NeuralForecast.load(path)` returns a ready NF object; call `predict(level=...)` on a properly prepared frame. 
* If you conformalized in training (see §8), the saved object can carry that configuration; interval columns produced during `predict`/`cross_validation` follow the NF naming (`Model-lo-90`, `Model-hi-90`). 

### 9.4 Artifacts & file layout (enforced)

* `experiments/h{h}/insample.parquet` — train/val backcasts from `predict_insample` (for PIT/coverage). 
* `experiments/h{h}/cv_raw.parquet` — NF CV outputs (point + PIs if requested). 
* `experiments/h{h}/leaderboard.parquet` — per-model sCRPS/MAE/RMSE/coverage (from §5 aggregator).
* `experiments/h{h}/coverage.parquet` — empirical coverage at 80/90/95.
* `experiments/h{h}/chkpt/` — optional saved NF object (if `--save`). 
* `reports/h{h}/preds_*.parquet` — live/batch prediction drops (with PIs).
* If conformal used: `experiments/h{h}/conformal.json` with method/levels/deltas (see §8).

### 9.5 Make it hard to shoot yourself in the foot

* **Always** run `fit(..., val_size=...)` before `predict_insample` — it uses the **stored dataset** from the last `fit`/`cross_validation`. 
* Ensure `df` in `predict` includes **future** rows (`ds` beyond last train `ds`) with **futr_exog_list** populated; else NF can’t produce h steps. Predict’s `futr_df` argument exists for future exogs if you keep train df separate. 
* **Set `step_size=h`** in CV to avoid window overlap (per your §5 policy); `refit=1` to retrain each window. 
* **Use NF scalers** via `scaler_type`/`revin` only; do not add ad-hoc scaling. TemporalNorm supports `robust`, `invariant`, `revin`, etc. 
* **Persist exactly once** with `nf.save(path, save_dataset=True)`; reload later with `NeuralForecast.load(path)`. Names/semantics are in the save/load tutorial. 

---

## 10) Inference & live deployment considerations

**Goal:** Every 15 minutes, ingest the latest bar (OHLCV), update features, and output forecasts for the next 1h,2h,4h,8h.

We assume we have an upstream process that provides the latest completed bar at, say, 12:15 for the interval 12:00–12:15.

Steps for the live loop (could be a cron job or daemon): 1\. **Append new data:** Add the new row to our DataFrame (or better, maintain a rolling DataFrame of last N days to keep things light). 2\. **Compute features for new row:** Many indicators (like RSI, moving averages) we can update incrementally: - For vectorbt/TA indicators, we may need to call them on the last window. Alternatively, maintain the indicator state manually. Simpler: just call our build_all_features for the full DataFrame or last couple of days. Because vectorbt uses NumPy under the hood, computing an indicator on, say, 10k points each time is fine (10k is a small array). - For multi-timeframe, we need to update resampled data: e.g., the new 1h bar might or might not complete a higher timeframe. If 12:15 comes in: - It completes the 12:00–12:15 15-min bar. - For 30-min bars, we need to check if 12:30 bar is complete; not yet. - For 1h bar, not yet (hour completes at 12:00–13:00 ends at 13:00). - So we only partially have higher timeframe. **But** in our feature building, we always forward-fill the last known higher timeframe value. For example, the 1h indicator from 12:00 will be used for 12:15,12:30,12:45 until the 13:00 data is available. - Our feature builder should handle this by looking at the last available higher timeframe value and propagating it. 3\. **Load model(s):** If not already loaded in memory, load from disk (using NeuralForecast.load) at startup. Typically, the model remains in memory for repeated use, only reloading if we updated model weights. 4\. **Forecast:** Call nf.predict(h=h, futr_df=..., level=\[80,90,95\]) for each required horizon. Actually, we may just train separate models for each horizon, or one model that can do multiple – NF models are trained for a fixed horizon. We likely train separate models per horizon for best performance. So we’ll have to load 4 different NF instances (or one with 4 models but that complicates output since each model has its own horizon length). - Simpler: treat each horizon as separate pipeline (they might share feature engineering but have distinct model objects). - If efficiency is an issue, we could train one model for the max horizon (32) and then take shorter-horizon slices from its output if we trust it, but usually separate training is better. 5\. **Output:** Merge the predictions with timestamp (the start time of forecast or end time?). Our forecasts will be labeled by the ds of the predicted timestamp. E.g., if current time is 12:15, h=4 output will have predictions for 12:30,12:45,13:00,13:15 (the next 4 intervals), with corresponding 80/90/95% bounds. - We might output a single file or DB record containing all horizons, or separate per horizon. Likely separate for clarity. - Ensure the output includes the timestamp of generation and model version for traceability.

**Performance:** - The model inference itself: These are small models (a few hundred thousand parameters) running on CPU/GPU for at most 32 steps ahead. This is on the order of milliseconds to low hundreds of milliseconds per model on CPU. On GPU, even faster but overhead might dominate. We can run on CPU for simplicity; GPU if we ensemble many models to keep latency low. - Feature calc: Could be the bigger part if not optimized, but computing e.g. 50 indicators on 10k data points is maybe \~50k operations per indicator, trivial in numpy/C. TA-Lib in C is extremely fast for indicators like RSI, etc., likely microseconds per call. So overall, end-to-end in \<1 second easily, which is fine for a 15-min cycle.

**Graceful fallback:** If for some reason the model fails (maybe data glitch or some unexpected input causing NaNs), have a basic fallback: - e.g., persistence forecast (predict flat or last return repeated) just so something is output. And log an error. - We can also maintain the last successful forecast.

**Integration with StatsForecast/MLForecast (if needed):** We mentioned possibly having a StatsForecast ARIMA or MLForecast XGBoost as a baseline. In live mode, those could also run, but likely we omit them in production unless they are part of an ensemble. If included, they add only a bit to runtime (ARIMA might be heavier, XGBoost is fine for one row forecast).

### 10.1 What happens every 15 minutes (sequence)

1. **Wait for bar finalization (UTC EOB).** Trigger only when `now % 15min == 0` **and** you’ve buffered ≥30–60s to avoid partial ticks.
2. **Append the last bar** (OHLCV for the just-closed 15-min interval) to the canonical frame; keep UTC EOB stamps.
3. **Rebuild exogs for the tail window** only (not the whole history):

   * Recompute **base-TF indicators** whose rolling windows touch the last bar.
   * Recompute **MTF** (30m/1h/4h) aggregates that close at this timestamp; forward-fill to 15m.
   * Apply the central **`shift(1)`** to *all* historic exogs.
4. **Build future calendar (`futr_exog`) for the next `h` steps**: `minute_of_day`, `day_of_week`, `is_weekend` (and any other future-known features you kept).
5. **Predict with NF** for each horizon `h ∈ {4,8,16,32}`:

   * Load the saved winner (or two members if ENS2).
   * Call `predict(...)` with `level=[80,90,95]`. If you maintain future calendars separately, pass them via `futr_df=`; otherwise, append rows with future timestamps + `futr_exog` to the same frame and call `predict(df=...)`.
6. **Export artifacts**: parquet of predictions (point + PIs) with timestamped filename in `reports/h{h}/`.
7. **Graceful fallback** (if GPU OOM or model failure): reduce batch, drop PatchTST first, or fall back to the previous checkpoint’s predictions; **log loudly** and continue.

### 10.2 Tail builders (drop-in utilities)

> **Create** `utils/tail_build.py`. These helpers keep inference cheap and leakage-safe. They do **not** replace your feature builders; they reuse them on a trimmed window.

```python
# utils/tail_build.py
from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np, pandas as pd
from features.registry import REGISTRY, MTF_TARGETS
from features.builder import build_indicators, apply_mtf, postprocess_shift_and_prune, select_features

def _max_indicator_warmup() -> int:
    """Conservative warmup for rolling indicators (covers the largest window across registry)."""
    wins = []
    for s in REGISTRY:
        # heuristic: collect any integer-like window params
        for k, v in (s.params or {}).items():
            for x in v:
                if isinstance(x, int):
                    wins.append(x)
    return int(max(wins or [64]))

def _max_mtf_span_minutes() -> int:
    """Conservative span to ensure higher-TF merges stabilize (use 4h)."""
    return 4 * 60  # minutes

def compute_tail_span(input_sizes: Dict[str, int]) -> int:
    """Return number of 15m bars to keep for tail recompute: max_input + warmups."""
    max_input = max(input_sizes.values() or [1024])
    warm = _max_indicator_warmup()
    mtf_margin = _max_mtf_span_minutes() // 15
    #  + 2 to be extra-safe around shift(1)
    return int(max_input + warm + mtf_margin + 2)

def build_tail_exogs(nf_df: pd.DataFrame,
                     input_sizes: Dict[str, int]) -> Tuple[pd.DataFrame, List[str], List[str], List[str]]:
    """
    Recalculate exogs only for the last N rows of nf_df. Returns (nf_tail, hist_cols, futr_cols, stat_cols).
    nf_df must contain ['unique_id','ds','y','open','high','low','close','volume', <exog?>].
    """
    span = compute_tail_span(input_sizes)
    tail = nf_df.sort_values("ds").iloc[-span:].copy()

    # Recompute base indicators + MTF on the tail OHLCV.
    base_feats = build_indicators(tail[["ds","open","high","low","close","volume"]])
    mtf_feats  = apply_mtf(tail[["ds","open","high","low","close","volume"]])

    exo_raw = base_feats.merge(mtf_feats, on="ds", how="left")
    exo     = postprocess_shift_and_prune(exo_raw, rules={})
    hist_cols, futr_cols, stat_cols = select_features(exo, policy={})

    # Merge into tail frame
    nf_tail = tail.merge(exo, on="ds", how="left")

    return nf_tail, hist_cols, futr_cols, stat_cols

def build_future_calendar(last_ds: pd.Timestamp, h: int, freq: str = "15min") -> pd.DataFrame:
    """Future-known exogs for the next h steps: minute_of_day, day_of_week, is_weekend."""
    futr_index = pd.date_range(last_ds + pd.tseries.frequencies.to_offset(freq),
                               periods=h, freq=freq, tz="UTC")
    df = pd.DataFrame({"ds": futr_index})
    df["minute_of_day"] = df["ds"].dt.hour * 60 + df["ds"].dt.minute
    df["day_of_week"]   = df["ds"].dt.dayofweek
    df["is_weekend"]    = (df["ds"].dt.dayofweek >= 5).astype(int)
    return df
```

### 10.3 `run_predict.py` (inference entrypoint) — one-shot & loop modes

> **Replace** the current file body with this minimal orchestrator. It supports **single run** (default) and an optional `--loop` mode you can wire to systemd/cron. It **does not** run in the background by itself; you control the process.

```python
# run_predict.py
import argparse, time, yaml, pandas as pd
from pathlib import Path
from neuralforecast import NeuralForecast
from utils.io import load_canonical_frame
from utils.tail_build import build_tail_exogs, build_future_calendar
from uq.ensembles import make_equal_weight_ensemble  # if you adopted ENS2
import torch

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, help="experiments/h{h}.yaml")
    ap.add_argument("--model_path", required=True, help="Path to saved NF model dir; for ENS2 pass a JSON manifest if you use one.")
    ap.add_argument("--h", type=int, required=True, help="Forecast horizon in 15m steps (e.g., 4,8,16,32)")
    ap.add_argument("--loop", action="store_true", help="Repeat every 15 minutes (trigger at UTC EOB)")
    ap.add_argument("--buffer_sec", type=int, default=45, help="EOB buffer before running")
    return ap.parse_args()

def predict_once(cfg, model_path, h):
    torch.set_grad_enabled(False)  # inference only
    # 1) Load canonical long frame (already UTC/EOB with y + OHLCV)
    df = load_canonical_frame(cfg["paths"]["data"]).sort_values("ds")
    last_ds = pd.DatetimeIndex(df["ds"]).max()

    # 2) Recompute tail exogs (cheap) for last window; returns merged tail df
    input_sizes = {"default": 1024, "PatchTST": 2048}
    df_tail, _, futr_cols, _ = build_tail_exogs(df, input_sizes=input_sizes)

    # 3) Build future-known calendar for next h steps
    futr_df = build_future_calendar(last_ds, h=h, freq=cfg["freq"])

    # 4) Load NF model (single winner)
    nf = NeuralForecast.load(model_path)

    # 5) Predict next h with PIs (pass futr_df if your training used futr_exog_list)
    preds = nf.predict(df=df_tail, futr_df=futr_df, level=cfg.get("levels", [80,90,95]))

    # 6) Persist
    outdir = Path(cfg["paths"]["reports"]) / f"h{h}"
    outdir.mkdir(parents=True, exist_ok=True)
    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    preds.to_parquet(outdir / f"preds_{ts}.parquet")
    print(f"[h={h}] wrote {outdir}/preds_{ts}.parquet")

if __name__ == "__main__":
    args = parse_args()
    cfg = yaml.safe_load(open(args.exp))

    if not args.loop:
        predict_once(cfg, args.model_path, args.h)
    else:
        # simple loop; recommended to schedule via systemd/cron instead of keeping a long-lived process
        while True:
            now = pd.Timestamp.utcnow()
            # wait until the next EOB + buffer
            next_eob = (now.ceil("15min"))
            sleep_s = (next_eob - now).total_seconds() + args.buffer_sec
            if sleep_s > 0:
                time.sleep(sleep_s)
            predict_once(cfg, args.model_path, args.h)
```

**Notes:**

* `NeuralForecast.load(path)` → `predict(df=..., futr_df=..., level=[...])`. If you trained with **calendar futr_exogs**, pass `futr_df` (preferred). Alternatively, you can append those future rows to `df` and call `predict(df=...)`.
* If you adopted **ENS2** (equal-weight of two winners), keep a tiny wrapper: load both `NeuralForecast` objects, call `predict` for each, and blend columns via the same logic you used in §7. Keep it **post-predict** (don’t average parameters).

### 10.4 Throughput & memory guards (graceful degradation)

* **Batch/hidden sizes:** If you see OOM, reduce `batch_size` first; then reduce `hidden_size`/`n_heads` (PatchTST) **at retrain time**; at inference time, your only knobs are `batch_size` (if exposed by NF predict) and moving to CPU.
* **Model triage:** If inference exceeds your latency budget, **drop PatchTST** first (heaviest), keep NHITS/NBEATSx.
* **Numerical sanity:** Disable gradients (`torch.set_grad_enabled(False)`), and keep models in `eval()` mode (NF sets this internally on predict).
* **I/O:** Avoid full-history recompute; use `build_tail_exogs` to only touch the last `max_input_size + margins` rows.
* **Missing bar:** If the just-closed EOB bar is missing or incomplete, **skip this cycle** and log it—do **not** backfill `y`. You’re forecasting returns; a fake bar will poison scaling and exogs.

### 10.5 Live conformal & monitoring hooks

* If you saved a conformalized NF object in training, your `predict(level=[...])` will emit conformal intervals directly.
* If not, but you stored per-level **coverage deltas** in `experiments/h{h}/conformal.json` (from §8), track **live** empirical coverage over a sliding window (e.g., last 7 days) and alert if drift exceeds **±3pp** at any of 80/90/95. Don’t “patch” intervals online; schedule a **retrain** (see §12) or attach conformal and re-save the model.

### 10.6 Minimal assertions in the live loop (don’t skip)

* **UTC EOB alignment:** the last row’s `ds` must match a 15-minute boundary; otherwise bail.
* **Leakage guard:** after `postprocess_shift_and_prune`, re-check a tiny sample with `assert_shifted` when debugging new indicators; disable in production for latency once stable.
* **Future calendars present:** verify `futr_df` has exactly `h` rows with the right columns; if not, don’t predict and log loudly.

### 10.7 Integration points (where this plugs into the rest)

* **From §7:** If the winner is an ensemble, replicate the **equal-weight** logic here after calling `predict` on each constituent.
* **From §3:** Tail builders reuse `features.builder.*` (vectorbt/TA-Lib primary; pandas-ta-openbb + freqtrade/technical for MTF). They centralize `shift(1)` to guarantee no leakage.
* **From §9:** Reuse the same `levels=[80,90,95]` and `freq="15min"` from the YAML; don’t fork config.

---

## 11) Maintenance & retraining

Forecasting models need periodic retraining to incorporate the most recent data and adapt to regime changes: - We plan a **full retrain** of the final models (the ones we deploy) on a **weekly** cadence (perhaps every weekend when volatility might be lower, or whenever convenient). Crypto can change quickly, but doing it too frequently (daily) might overreact to noise. Weekly seems a good balance. - Additionally, we set **performance monitoring alerts**: if we observe that the live forecasts’ interval coverage or error metrics have drifted significantly from our validation results, that could trigger an earlier retrain. For example, if the 90% interval is now covering only 75% of actuals for the past week, clearly the model is underestimating volatility – time to retrain (and possibly re-tune). - **Drift detection:** We can incorporate tests on the distribution of residuals – e.g., a rolling KS test on the PIT values to see if they are still uniform, or simply monitor sCRPS on a rolling basis in production. If sCRPS worsens beyond a threshold relative to our backtest baseline, it’s retrain time.

**Continuous Learning vs Fixed Model:** We are not doing online learning (no updating weights per new data point); we accumulate new data and do batch retraining. Each retrain will use all data up to that point (or we might decide to window it to, say, last 2 years if very long history slows training with little benefit).

**Version control:** - Each model training run (especially ones promoted to production) will be versioned (could use Git tags or just a timestamp in model path). - We will pin library versions (NeuralForecast, torch, etc.) in requirements for consistent results. If we upgrade NF or PyTorch, we’ll re-run backtests to ensure nothing changed significantly (Nixtla does updates that could slightly alter outputs, so be cautious). - The code being relatively simple and NF-handled reduces maintenance surface.

**Adding new features or models:** - We will treat the pipeline as modular. If a new indicator is to be added, we add it in features/registry.py, recompute, and backtest again to verify improvement (ensure no leakage introduced). - If Nixtla releases a new model (say, a better Transformer or a new regression model), we can experiment in the pipeline by adding to nf_models. - We keep an eye on training time; if models start to take too long, we might consider using NF’s multi-gpu or Ray integration (NF can parallelize models since v1.6+). For now, training sequentially is fine because we have only a few models and not extremely large data.

**Optional sibling libraries:** - **StatsForecast:** We might include some classical baseline models in the repository (not for deployment, but for validation). E.g., a naive model (last value or historical average) or an ARIMA. StatsForecast can easily produce these forecasts and conformal intervals. This can serve as a sanity check – if our complex models can’t beat a simple ARIMA in sCRPS, something’s wrong. - **MLForecast:** We can use this to train a quick XGBoost or RandomForest on our features to see if the feature set has predictive power. MLForecast simplifies training sklearn or lightgbm models on lag features and exogenous data. This is just for comparison in research; we likely won’t deploy an ML model because maintaining two parallel modeling codebases is extra work. But it’s there as a benchmark (and could be useful if it turns out to be nearly as good as NF – then maybe we’d choose the simpler model). - **HierarchicalForecast:** Only relevant if we had a hierarchy of series (like BTC at different frequencies or BTC vs ETH). If in future we forecast multiple assets or both 15-min and daily and want coherent forecasts, we could use hierarchical reconciliation. For now, not needed (single series). - Each of these will be minimal – e.g., a script in experiments/ that trains ARIMA and outputs metrics, just to document performance.

**Hardware**: Training the NF models on GPU (if available) will speed up experimentation. Nixtla’s models use PyTorch Lightning, which can utilize GPU/TPU. We’ll use a GPU for training if possible (especially PatchTST benefits from GPU). For inference, CPU might suffice. We ensure the code runs on both (Lightning deals with devices mostly automatically).

**Logging**: We will keep training logs (learning rate schedules, epoch losses) as needed. Pytorch Lightning by default prints to stdout; we might route that to a file for record.

### 11.1 Cadence & triggers (what makes us retrain)

**Baseline cadence:** **weekly** full retrain per horizon (Sunday 00:30 UTC), unless the triggers fire earlier.

**Hard triggers (retrain within 24h):**

* **Coverage drift:** rolling **7-day** empirical coverage at any of {80,90,95} deviates by **>±3pp** from nominal.
* **Score decay:** 7-day mean sCRPS (or MAE if sCRPS not computed live) degrades **>3%** vs. the trailing 30-day baseline.
* **Distribution shift:** PSI (Population Stability Index) on **key hist exogs** or **target returns** exceeds **0.2** (moderate) for 3 consecutive days, or **≥0.3** (major) on any day.
* **Operational:** sustained GPU OOM or latency breaches for 3+ consecutive cycles after simple batch/size reductions (§10.4).

**Soft triggers (evaluate next scheduled retrain):**

* Volatility regime change (top volatility decile share > 35% of bars over 7 days).
* Calendar effect drift (coverage differs by >5pp between weekdays vs. weekends).

### 11.2 Versioning & pinning (don’t be sloppy)

* **Freeze at train time:** write `experiments/h{h}/version_manifest.json` with:

  * `python`, `pytorch`, `cuda`, `neuralforecast`, `statsforecast`, `mlforecast`, `hierarchicalforecast` versions,
  * git commit hash,
  * OS/arch, GPU name & VRAM,
  * full YAML used (`experiments/h{h}.yaml` merged with `defaults.yaml`).
* **Locks:** keep `requirements-lock.txt` (from `pip freeze`) beside the manifest. Never “upgrade in place” without a full CV rerun.
* **Upgrade path:** when bumping Nixtla/torch, **train new models in parallel** under `experiments/h{h}/trial_[date]/` and gate with §12 acceptance criteria before switching.

**Drop-in:** `utils/versioning.py`

```python
# utils/versioning.py
import json, platform, sys, subprocess, pkgutil
from pathlib import Path

def snapshot_env(out_path: str, extra: dict | None = None):
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }
    # Try to capture key libs; tolerate absence
    for pkg in ["torch", "neuralforecast", "statsforecast", "mlforecast", "hierarchicalforecast"]:
        try:
            m = __import__(pkg)
            info[pkg] = getattr(m, "__version__", "unknown")
        except Exception:
            info[pkg] = "not_installed"
    try:
        cuda = subprocess.check_output(["python","-c","import torch;print(torch.version.cuda or 'cpu')"], text=True).strip()
    except Exception:
        cuda = "unknown"
    info["cuda"] = cuda
    if extra:
        info.update(extra)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(info, indent=2))
```

### 11.3 Smoke tests (fast, decisive)

Run **before** any deployment or after any environment change:

* **NF round-trip:** `nf.save(...)` → `NeuralForecast.load(...)` → `predict` on a 3-day slice; assert:

  * correct horizon length,
  * columns present for winner/ensemble (`<alias>`, `<alias>-lo-90`, `<alias>-hi-90`),
  * no NaNs in the produced horizon (allow NaNs in warmup rows only).
* **Leakage guard:** re-run `assert_shifted` on a fresh tail build (last 5 days).
* **Performance sanity:** latency per horizon under **N** seconds (set N per your hardware), peak VRAM below threshold.
* **Conformal integrity (if used):** re-compute 30-day tail coverage on saved conformalized model; expect **±2pp** at nominal.

Automate via a tiny script `utils/smoke.py` you call in CI.

### 11.4 Drift & monitoring (simple, actionable)

> Place these in `uq/monitor.py`. They compute rolling coverage, PSI, and volatility-bucket coverage for alarms.

```python
# uq/monitor.py
import numpy as np, pandas as pd

def rolling_coverage(preds: pd.DataFrame, y_col="y", model: str="", levels=(80,90,95), window="7D") -> pd.DataFrame:
    df = preds[["ds", y_col] + [c for c in preds.columns if c.startswith(model)]].dropna()
    df = df.set_index("ds").sort_index()
    rows=[]
    for L in levels:
        lo, hi = f"{model}-lo-{L}", f"{model}-hi-{L}"
        if lo in df and hi in df:
            inside = ((df[y_col]>=df[lo]) & (df[y_col]<=df[hi])).astype(float)
            cov = inside.rolling(window).mean().dropna().rename(f"cov{L}")
            rows.append(cov)
    out = pd.concat(rows, axis=1) if rows else pd.DataFrame()
    return out.reset_index()

def psi(ref: pd.Series, cur: pd.Series, bins=20) -> float:
    """Population Stability Index in nats (ln)."""
    ref = ref.dropna(); cur = cur.dropna()
    qs = np.linspace(0,1,bins+1)
    edges = np.unique(np.concatenate([np.quantile(ref, qs), np.quantile(cur, qs)]))
    # bin
    ref_h = np.histogram(ref, bins=edges)[0]; cur_h = np.histogram(cur, bins=edges)[0]
    # proportions + smoothing
    ref_p = (ref_h + 1e-6) / (ref_h.sum() + 1e-6*len(ref_h))
    cur_p = (cur_h + 1e-6) / (cur_h.sum() + 1e-6*len(cur_h))
    return float(((cur_p - ref_p) * np.log(cur_p / ref_p)).sum())

def coverage_by_vol_decile(preds: pd.DataFrame, vol: pd.Series, model: str, level: int, y_col="y") -> pd.DataFrame:
    lo, hi = f"{model}-lo-{level}", f"{model}-hi-{level}"
    df = preds[["ds", y_col, lo, hi]].dropna().merge(vol.rename("vol"), left_on="ds", right_index=True)
    df["decile"] = pd.qcut(df["vol"], 10, labels=False, duplicates="drop")
    out = df.groupby("decile").apply(lambda g: ((g[y_col]>=g[lo])&(g[y_col]<=g[hi])).mean()).rename("coverage").reset_index()
    out["level"]=level; out["model"]=model
    return out
```

**Live dashboards (minimum):**

* 7-day rolling coverage at 80/90/95 (per horizon).
* Coverage by volatility decile (weekly).
* Latency & memory percentiles.
* PSI for **top-K** hist exogs and for the **target returns** distribution.

### 11.5 Retrain procedure (no drama)

1. **Snapshot** the environment (`utils/versioning.snapshot_env(...)`) and freeze reqs.
2. **Rebuild features** over the full history (same registry; **do not** change sources silently).
3. **Run §6 pilot → promote → full CV** on the **current library stack**.
4. **Select/ensemble** per §7; **calibrate** per §8 (apply conformal if coverage off).
5. **Acceptance gates** per §12 on a **fresh, unseen tail** (e.g., last 60–90 days).
6. **Save** final models with `nf.save(..., save_dataset=True)`.
7. **Smoke tests** §11.3 on the saved artifacts.
8. **Promote** by atomically switching the `model_path` in your deployment config and keeping the previous winner as **rollback**.

### 11.6 Rollback (pre-wired, zero doubt)

* Keep the previous best under `experiments/h{h}/prev_best/` with its `version_manifest.json`.
* If live monitoring breaches **any** §12 gate for >24h, switch back immediately; open a work item to investigate (typical causes: data anomalies, provider glitches, scale drift).

### 11.7 Optional, tightly-scoped siblings (only where they add value)

**StatsForecast (classical baselines)** — **sanity checks**, not production models.

* Add **Naive**, **SeasonalNaive(96)** (24h/15m), **AutoETS** as “always-on” baselines during CV.
* Purpose: detect data/feature regressions when NF “wins” but baselines jump.

**MLForecast (tabular baselines)** — **feature sanity** on the exact same exogs.

* Fit a **Lasso/ElasticNet** or **LightGBM** on a small window to see if crafted exogs have linear signal; if they do not, your indicator stack might be bloated/noisy.

**HierarchicalForecast** — **only if** you move to multi-asset or need **temporal reconciliation** across {15m, 30m, 1h, 4h}.

* Keep disabled in single-asset v1. Consider later for coherent MTF forecasts if you must publish them jointly.

> All siblings are **off the critical path**. Use them sparingly for diagnostics and guardrails; your production remains **NeuralForecast-first**.

### 11.8 House rules (so we don’t regress)

* **Do not** hand-edit exog columns between retrains. Change the **registry**; rerun CV.
* **One source of truth** for configs: YAML under `experiments/`.
* **One place** for persistence: `nf.save`/`NeuralForecast.load`. No pickle/torch.save hacks.
* **Kill complexity** that doesn’t beat your acceptance gates (§12). Stop when gains flatten.

---

## 12) Acceptance criteria & quality gates

We define clear quantitative criteria to decide if the forecasting system is ready for use (for each horizon):

1. **Backtest performance:** On historical CV, the chosen model/ensemble must show **sCRPS improvement** of at least 1.5% over a reference (could be a naive forecast or an earlier baseline model). If not, it’s not worth the complexity. The baseline could be last-value forecast or an ARIMA’s CRPS. We want a meaningful gain.

2. **Calibration (coverage):** The empirical coverage of the 80%, 90%, 95% prediction intervals on the backtest (or a held-out test set) should be within ±2 percentage points of the nominal values. For example, 90% interval should contain 88–92% of actual outcomes. This tolerance accounts for sampling error. If outside this band, we either improve the model (add features or use a different loss) or apply conformal until it meets the target.

3. **PIT uniformity:** The PIT histogram (or quantile statistics) for the final model should not show severe deviations from uniform. Specifically, no obvious U-shape (which would indicate over-dispersed forecasts, with too many actuals near 0 or 1 quantile) and no strong spikes at 0 or 1 (which indicate some events not predicted at all by the model – e.g., missed jumps). Mild deviations can be addressed by slight tuning or conformal; major ones mean model is mis-specified.

4. **Stress test on volatility:** Check performance in the highest-volatility periods (say top 10% of 15-min returns magnitude). Our expectation is some deterioration is okay (all models will struggle on sudden moves), but it should not be catastrophic. For instance, if overall sCRPS is 0.5 but during volatility spike hours it jumps to 5.0, that’s problematic. We set a threshold like “sCRPS in the highest volatility decile should be no more than 3x the median sCRPS”. Likewise, interval coverage in those periods should not drop too far (maybe 75% coverage for 95% PI is unacceptable).

5. **Computational performance:** Model training time and inference speed are within acceptable limits (e.g., training under a couple of hours on a GPU for all models and horizons; inference under 1 second). If a model is extremely slow, we might drop it unless it has big accuracy gains.

If a model fails these gates, remedies: - If calibration is off: apply conformal prediction until it passes (as long as it’s a consistent bias). - If sCRPS is not beating baseline: consider adding features or trying another model type. - If PIT shows bias: maybe switch loss (StudentT \<-\> MQ) or incorporate a regime feature.

We will document these in a report. Ultimately, the stakeholders get a model that has been *vetted on historical data*, with evidence that its predictions and confidence intervals make sense.

If the acceptance criteria are met for some horizons but not others, we may deploy only the horizons that meet (e.g., maybe short-term forecasts are fine but 8h horizon isn’t reliable – we could hold off 8h deployment or keep it labeled “experimental”).

### 12.1 Hard pass/fail gates (per horizon h)

For a candidate **winner** (single model or ENS2) evaluated on the **final CV** and a **held-out test tail** (e.g., last 60–90 days):

1. **Accuracy (primary):**

   * **Mean sCRPS** across CV windows ≤ **baseline_sCRPS × 0.985** (≥ **1.5%** improvement).
   * If dense quantiles weren’t computed in CV, re-run finalists with dense quantiles (e.g., 1–99) to compute sCRPS precisely, then decide.
2. **Calibration:**

   * **Empirical coverage** at **80/90/95** on the **test tail** within **±2 pp** of nominal.
   * **PIT** on the test tail is roughly uniform (no U-shape/hump; KS or AD test p-value ≥ 0.05 preferred; if not, visual inspection + coverage by volatility decile must not reveal systemic bias).
3. **Robustness in stress:**

   * In the **top volatility quintile** (by rolling σ of returns), test-tail mean sCRPS degrades by **≤ 10%** vs. overall test-tail sCRPS.
4. **Stability:**

   * CV window-to-window **sCRPS std** not worse than baseline by more than **20%** (i.e., don’t accept more variance for tiny mean gains).

**Baseline definition:** previous production model on the same data slice. Keep **Naive/SeasonalNaive** as sanity checks; you don’t need to beat them by a fixed margin, but if a classical naive ties your deep model, that’s a red flag.

**Decision rule:** all 4 gates must pass to **promote**. Missing any gate → **reject** or **fix** (features, loss, scaler, conformal) and rerun.

### 12.2 Rollback criteria (live)

After promotion, **rollback** to prior production if **any** holds for **>24 hours** (or earlier if risk appetite is low):

* Rolling **7-day** coverage at any of {80,90,95} deviates by **>±3 pp**.
* 7-day mean sCRPS (or MAE proxy) **> +3%** vs. 30-day baseline.
* Latency or VRAM repeatedly violates SLOs despite batch reductions (§10.4).
* Data quality issues (missing bars, timestamp drift) persist beyond one cycle.

### 12.3 Acceptance report (one command)

> **Create** `reports/acceptance.py`. Reads NF artifacts, computes gates, writes a compact report you can paste into a PR or runbook.

```python
# reports/acceptance.py
from __future__ import annotations
import argparse, json, numpy as np, pandas as pd
from pathlib import Path

def load_parquets(hdir: Path):
    cv = pd.read_parquet(hdir / "cv_raw.parquet")
    lb = pd.read_parquet(hdir / "leaderboard.parquet")
    cov = pd.read_parquet(hdir / "coverage.parquet")
    ins = pd.read_parquet(hdir / "insample.parquet")
    return cv, lb, cov, ins

def _model_point_cols(df: pd.DataFrame):
    meta = {"unique_id","ds","cutoff","y"}
    return [c for c in df.columns if c not in meta and "-lo-" not in c and "-hi-" not in c and "-q" not in c]

def coverage_on_tail(preds: pd.DataFrame, model: str, levels=(80,90,95)) -> dict:
    out={}
    for L in levels:
        lo, hi = f"{model}-lo-{L}", f"{model}-hi-{L}"
        if lo in preds and hi in preds:
            p = preds[["y", lo, hi]].dropna()
            out[f"cov{L}"] = float(((p["y"]>=p[lo]) & (p["y"]<=p[hi])).mean())
        else:
            out[f"cov{L}"] = np.nan
    return out

def volatility_deciles(series: pd.Series, window=96):
    vol = series.rolling(window).std().dropna()
    return vol, pd.qcut(vol, 5, labels=False, duplicates="drop")

def robust_gate(preds_tail: pd.DataFrame, model: str) -> float:
    # ratio: MAE(top_vol_quintile) / MAE(overall) - 1
    overall = (preds_tail["y"] - preds_tail[model]).abs().mean()
    vol, d = volatility_deciles(preds_tail["y"])
    df = preds_tail.merge(d.rename("q"), left_on="ds", right_index=True, how="left")
    top = df[df["q"]==df["q"].max()]
    if len(top) < 20:
        return np.nan
    top_mae = (top["y"] - top[model]).abs().mean()
    return float(top_mae / overall - 1.0)

def decide(hdir: Path, winner: str, baseline_row: dict,
           coverage_band_pp: float = 2.0,
           scrps_gain_min: float = 0.015,
           robust_worst: float = 0.10) -> dict:
    cv, lb, cov, ins = load_parquets(hdir)
    report = {"winner": winner, "paths": str(hdir)}

    # sCRPS gain (read from leaderboard only)
    if winner not in set(lb["model"]):
        return {"status":"REJECT", "reason":f"Winner {winner} not found in leaderboard.", **report}
    cand_row = lb[lb["model"]==winner].iloc[0].to_dict()
    if "sCRPS" not in cand_row or pd.isna(cand_row["sCRPS"]):
        return {"status":"REJECT", "reason":"Leaderboard lacks sCRPS; rerun finalists with dense quantiles.", **report}
    gain = (baseline_row["sCRPS"] - cand_row["sCRPS"]) / baseline_row["sCRPS"]
    report["scrps_gain"] = float(gain)

    # Test-tail preds expected at reports/h{h}/preds_*.parquet
    # Infer h from dir name 'h{h}'
    try:
        h = int(str(hdir.name).lstrip("h"))
    except Exception:
        return {"status":"REJECT", "reason":"Invalid horizon dir name.", **report}
    tail_dir = hdir.parent.parent / f"reports/h{h}"
    files = sorted(tail_dir.glob("preds_*.parquet"))
    if not files:
        return {"status":"REJECT", "reason":"No test-tail preds found.", **report}
    tail = pd.read_parquet(files[-1])
    if "y" not in tail.columns:
        return {"status":"REJECT", "reason":"Tail preds lack y column.", **report}

    # Coverage gates
    covs = coverage_on_tail(tail, winner, levels=(80,90,95))
    report.update({f"tail_{k}": v for k,v in covs.items()})
    cov_ok = all(abs(covs[k] - int(k[3:])/100.0) <= coverage_band_pp/100.0 for k in covs if not np.isnan(covs[k]))

    # Robustness
    degr = robust_gate(tail, winner)
    report["robust_degradation"] = degr
    robust_ok = (np.isnan(degr) or degr <= robust_worst)

    # Final decision
    scrps_ok = (gain >= scrps_gain_min)
    if scrps_ok and cov_ok and robust_ok:
        report["status"]="ACCEPT"
    else:
        report["status"]="REJECT"
        report["reason"] = json.dumps({"scrps_ok":scrps_ok,"cov_ok":cov_ok,"robust_ok":robust_ok})
    return report

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hdir", required=True, help="experiments/h{h} directory")
    ap.add_argument("--winner", required=True, help="Winner alias (or ENS2)")
    ap.add_argument("--baseline_scrps", type=float, required=True, help="Baseline sCRPS (prev prod) for this horizon")
    args = ap.parse_args()
    hdir = Path(args.hdir)
    baseline = {"sCRPS": args.baseline_scrps}
    rep = decide(hdir, args.winner, baseline)
    out = hdir / "acceptance_report.json"
    out.write_text(json.dumps(rep, indent=2))
    print(rep)
```

**Run it (example for h=16):**

```bash
python reports/acceptance.py \
  --hdir experiments/h16 \
  --winner ENS2 \
  --baseline_scrps 0.1234
# -> writes experiments/h16/acceptance_report.json with ACCEPT/REJECT and reasons
```

### 12.4 What exactly to store

* `experiments/h{h}/acceptance_report.json` — **single source of truth** for ACCEPT/REJECT and rationale.
* `experiments/h{h}/leaderboard.parquet` — already produced in §9 (keep for audit).
* `reports/h{h}/preds_*.parquet` — the exact test-tail file used for coverage checks (include filename in the report).
* If conformal used: `experiments/h{h}/conformal.json` documenting method/levels and empirical deltas.

### 12.5 If it fails — smallest hammer first

* **Coverage miss:** attach **conformal** (per §8.2) and re-evaluate; don’t overfit with large architectural changes.
* **PIT U-shape:** prefer **quantile training** or RevIN/invariant scaler; reassess features (too many noisy exogs).
* **Stress degradation:** shorten `input_size` slightly (reduce dependency on stale regimes), tighten feature cap, or drop the heaviest model (PatchTST) if it overfits.
* **sCRPS barely below threshold (e.g., +1.2%):** verify leaderboard stability; if std across windows is significantly **lower** than baseline and coverage is **clean**, you can accept with a **documented waiver** (don’t make waivers a habit).

---

# 13) Implementation checklist

### Phase 0 — Repo scaffold (0.5–1h)

* [ ] **Create directories** **\[P]**
  `data/`, `features/`, `nf_models/`, `cv/`, `uq/`, `reports/`, `utils/`, `experiments/`
  **Done when:** tree matches plan; CI lints pass.

* [ ] **Stub files** **\[P]**

  * `utils/io.py`, `utils/validate.py`, `utils/versioning.py`
  * `features/registry.py`, `features/builder.py`
  * `nf_models/factories.py`
  * `cv/runner.py`, `cv/hpo.py`
  * `uq/diag.py`, `uq/pit.py`, `uq/ensembles.py`, `uq/monitor.py`
  * `reports/acceptance.py`
  * `run_train.py`, `run_predict.py`
  * `experiments/defaults.yaml`, `experiments/h4.yaml`, `experiments/h8.yaml`, `experiments/h16.yaml`, `experiments/h32.yaml`
    **Done when:** all files exist with import-safe stubs.

### Phase 1 — Data contracts & validation (2–3h)

* [ ] Implement `regularize_to_grid_utc`, `make_nf_canonical`, `drop_train_nans_and_winsorize` in `utils/io.py`.
  **Done when:** returns long frame with `['unique_id','ds','y',OHLCV...]`, UTC/EOB, deterministic.

* [ ] Implement validators in `utils/validate.py`: `assert_regular_grid`, `assert_utc_eob`, `assert_no_forward_fill_y`, `assert_shifted`.
  **Done when:** each raises on synthetic counterexamples.

* [ ] **Smoke test** notebooks or pytest **\[P]**
  **Done when:** sample OHLCV → canonical NF frame without asserts firing.

### Phase 2 — Exogenous features & MTF (3–5h)

* [ ] Implement **registry** in `features/registry.py` (starter set from §3).
  **Done when:** registry import works; names/params resolve.

* [ ] Implement base-TF compute via vectorbt/TA-Lib + pandas-ta in `features/builder.py` (`build_indicators`).
  **Done when:** produces DataFrame with named columns for each param combo.

* [ ] Implement MTF resample/merge using freqtrade/technical (`apply_mtf`).
  **Done when:** 30m/1h/4h features EOB-aligned and forward-filled to 15m.

* [ ] Implement postprocess (`postprocess_shift_and_prune`) and selector (`select_features`).
  **Done when:** all **hist** features are **shift(1)**, availability ≥98%, total ≤256.

* [ ] **Unit checks** **\[P]**:

  * MTF alignment toy example,
  * leakage check on random timestamps,
  * cap enforcement.
    **Done when:** tests pass; printed feature counts ≤256.

### Phase 3 — NF model factory (1–2h)

* [ ] Implement `nf_models/factories.py` with loss constructors and exog wiring.
  **Done when:** calling with YAML builds NHITS/NBEATSx/TiDE/PatchTST objects without error.

### Phase 4 — Experiments config (0.5–1h)

* [ ] Fill `experiments/defaults.yaml` and `experiments/h{h}.yaml` per §9.1.
  **Done when:** `yaml.safe_load` shows expected keys/types; models lists parse.

### Phase 5 — Training driver & CV (2–4h)

* [ ] Implement `run_train.py` (from §9.2).
  **Done when:** on sample data, it: `fit` → `predict_insample` → `cross_validation` and writes:
  `experiments/h{h}/insample.parquet`, `cv_raw.parquet`, `leaderboard.parquet`, `coverage.parquet`.

* [ ] Implement `cv/runner.py::run_cv` and `summarize_cv`.
  **Done when:** leaderboard has MAE/RMSE and coverage; sCRPS optional until finalists.

* [ ] Optional: enable `--save` to persist NF object (`chkpt/`).
  **Done when:** `NeuralForecast.load(chkpt)` → `predict` works.

### Phase 6 — Pilot → promote → full CV (4–8h compute; 1h ops)

* [ ] **Pilot** h=16, `n_windows=3` using `cv/hpo.py` tiny grids **\[P]** (per-model parallel).
  **Done when:** `experiments/h16/hpo_leaderboard.parquet` exists; top configs identified.

* [ ] **Promote** best 1–2 per model; rerun h=16 with `n_windows=6`.
  **Done when:** updated leaderboard shows clear winners.

* [ ] **Full CV** across h∈{4,8,16,32} with `n_windows=10`, `step_size=h`, `val_size=4*h`.
  **Done when:** per-h leaderboards and coverage written.

### Phase 7 — Selection & simple ensembling (1–2h)

* [ ] Use §7 rules: pick best single per h; if top-2 tight, create **ENS2** via `uq/ensembles.py` and score with `cv/runner.evaluate_top2_ensemble`.
  **Done when:** `experiments/h{h}/selection.json` saved; winner decided.

* [ ] Final **fit** of winner(s) on all data (keep `val_size` for early stop) and `nf.save(...)`.
  **Done when:** `experiments/h{h}/best/` (or two member dirs) exist; load/predict OK.

### Phase 8 — Uncertainty & calibration (1–2h)

* [ ] Compute **coverage** tables and **PIT** for finalists (dense quantiles if needed).
  **Done when:** `experiments/h{h}/pit_hist.png` and `coverage.parquet` updated.

* [ ] If coverage miss >±2pp, attach **Conformal** (per §8.2) and re-evaluate.
  **Done when:** coverage on validation tail is within band.

### Phase 9 — Inference & live loop (2–3h)

* [ ] Implement `utils/tail_build.py` and wire `run_predict.py` (single-shot + `--loop`).
  **Done when:** `run_predict.py --exp experiments/h16.yaml --model_path experiments/h16/best/ --h 16`
  writes `reports/h16/preds_*.parquet`.

* [ ] If **ENS2**: add tiny wrapper to load two saved NF objects and blend predictions post-predict.
  **Done when:** `ENS2` columns appear in preds, with intervals averaged level-wise.

### Phase 10 — Monitoring & maintenance (1–2h)

* [ ] Implement `uq/monitor.py` (rolling coverage, PSI, vol-decile coverage) and `utils/versioning.snapshot_env`.
  **Done when:** daily job produces CSV/PNG artifacts in `reports/monitoring/`.

* [ ] **Smoke tests** script in `utils/smoke.py`.
  **Done when:** pass on fresh environment; round-trip save/load/predict OK.

### Phase 11 — Acceptance & promotion (0.5–1h)

* [ ] Run `reports/acceptance.py` per h with baseline sCRPS.
  **Done when:** `experiments/h{h}/acceptance_report.json` says **ACCEPT**.

* [ ] **Promote** by switching deploy config `model_path` to `experiments/h{h}/best/`. Keep previous under `prev_best/` for rollback.
  **Done when:** live loop producing forecasts at the next EOB; monitoring green.

---

### Parallelization guide (practical)

* **\[P] Phase 1 vs. Phase 2:** data validators and feature builders can be developed/tested independently on stub data.
* **\[P] Phase 3–4:** model factory & YAML can proceed while features stabilize (use a tiny placeholder exog set).
* **\[P] Phase 6:** per-model pilot CV runs in parallel (separate processes/GPUs).
* **\[P] Phase 9:** inference loop wiring can start once a single trained checkpoint exists (no need to wait for full CV).

---

### Command quick sheet (copy/paste)

```bash
# Train (h=16 example)
python run_train.py --exp experiments/h16.yaml --save

# Pilot HPO (h=16; ensure n_windows=3 in h16.yaml)
python - <<'PY'
import yaml, pandas as pd
from cv.hpo import run_hpo
cfg=yaml.safe_load(open("experiments/h16.yaml"))
df=pd.read_parquet("data/btc_15min.parquet")
run_hpo(df, cfg, "experiments/hpo_spaces.yaml", pilot=True)\
  .to_parquet("experiments/h16/hpo_leaderboard.parquet")
print("Wrote experiments/h16/hpo_leaderboard.parquet")
PY

# Inference (single run)
python run_predict.py --exp experiments/h16.yaml --model_path experiments/h16/best/ --h 16

# Acceptance
python reports/acceptance.py --hdir experiments/h16 --winner ENS2 --baseline_scrps 0.1234
```

---

### Definition of Done (v1)

* Reproducible CV artifacts per horizon; winners saved via **NF `save`**; inference loop producing **point + 80/90/95 PIs** every 15 minutes; acceptance report **ACCEPT** for ≥2 horizons, none **REJECT**; monitoring shows 7-day coverage within **±3pp**, latency within SLO, and PSI ≤0.2.

---

## 14) Risks & mitigations

Even with this careful plan, things can go wrong. We enumerate some potential issues and how we address them:

* **Feature leakage:** The biggest risk in time series. If we accidentally use a non-shifted feature or include future info in hist_exog, the backtest will look artificially good. **Mitigation:** Our assert_shifted check will verify that for each hist_exog column, corr(feature, y_lead_1) is zero (there should be no direct correlation with future target). We only populate futr_exog with truly future-known info.

* **Misaligned MTF features:** If resampling isn’t done correctly, e.g., using left-closed intervals instead of right-closed, we could use future info. We explicitly use the end-of-bar labeling (right label) when resampling and then shift. We will test by printing a few samples: e.g., a 1h moving average at 10:00 should be computed from data up to 9:00-10:00 and after shift, the 10:15 row uses the 9:00-10:00 MA – that’s correct.

* **Quantile crossing:** If using quantile loss without special handling, it’s possible the predicted 90th percentile is lower than 50th. NF’s ISQF or IQLoss should handle a lot of this, but if we see any minor violations, we can sort the outputs after the fact (monotonic enforcement) or switch to the implicit quantile approach.

* **Overfitting to backtest (over-engineering):** With so many features, there’s a risk we overfit the historical test. Mitigation: keep feature set moderate, use regularization (dropout, etc.), and trust cross-validation with refit – because refit simulates how model sees new data it wasn’t trained on each fold, it gives a more honest measure. We also can hold out the latest chunk of data entirely as an unseen test to double-check.

* **Underestimating tails (under-coverage):** If even Student-T models underestimate some extreme moves, our 95% interval might actually be, say, 80%. Mitigation: conformal will catch this by widening intervals based on residuals. If we still see under-coverage, consider using an even heavier-tail distribution (NF supports Negative Binomial or a mixture if needed) or add features that correlate with those jumps (like including stock market volatility index if BTC spikes align with that, etc.).

* **Data issues:** e.g., a big price feed error (spike) that isn’t real – our model might learn a weird behavior. Mitigation: use the winsorization and maybe remove obviously bad data points from training (treat as missing).

* **GPU OOM (out-of-memory):** PatchTST with large input and batch might OOM on a GPU if memory is limited. We already plan to reduce batch_size if needed. We could also decrease windows_batch_size (NF uses this to sub-sample windows per batch). If still an issue, we drop PatchTST or run it on CPU with smaller batch since CPU memory is usually larger (with a speed hit).

* **Library bugs or updates:** Relying on NF means we’re subject to its bugs. We saw some GitHub issues about predict_insample bugs, etc. We will keep an eye on Nixtla’s GitHub for any relevant issues. If a critical bug is found (e.g., in a model or loss), we might need to update to a patched version or implement a workaround. Pinning a stable version (like neuralforecast==1.x.y) is important.

* **Interpretability/Trust:** Stakeholders might ask why the model is forecasting what it is. NBEATSx offers some interpretability (trend/seasonality decomposition). We can provide partial plots or indicator importance by analyzing the learned weights or doing ablation (e.g., turn off one feature and see effect). Not a risk per se, but something to be ready to address by logs or visualizations.

* **Integration with upstream/downstream:** Ensure our 15-min UTC timestamps align exactly with the data source’s timing. Any off-by-one issues would break the live use. We’ve set end-of-bar labeling to avoid confusion. We’ll coordinate with data engineers to confirm that the bar at 12:00 contains trades up to 12:00, etc.

By anticipating these, we include checks and flexibility to respond. For example, if we detect poor coverage in a certain regime, we can quickly adjust via conformal or add a feature (like realized volatility) and retrain.

**Short answer:** Append my blocks **after** your existing text. Don’t delete anything unless I say “replace/overwrite.”
Context is intact — no need to resend the plan.


### 14.1 MTF misalignment (30m/1h/4h → 15m)

* **Symptom:** Feature jumps occur at wrong minutes; coverage/PIT skewed on hour boundaries.
* **Cause:** Higher-TF bars not EOB-aligned or not forward-filled correctly, or not shifted.
* **Fix (exact):**

  * Use `apply_mtf()` from §3 (freqtrade/technical merge). Do **not** write custom `resample()` logic.
  * Keep **EOB UTC** grid (§2).
  * Central **`shift(1)`** in `postprocess_shift_and_prune()` — never compute or shift ad hoc.
  * Add unit test: fabricate a 2-hour toy series; assert 1h feature is constant within each hour and only updates at `:00` **and** that the final matrix is shifted by one 15-min step.

### 14.2 Leakage from non-shifted historic exogs

* **Symptom:** Unrealistically good CV; live collapses.
* **Cause:** Any historic indicator not `shift(1)` after MTF merge; “future” calendar accidentally placed in `hist_exog_list`.
* **Fix (exact):**

  * Enforce `assert_shifted(nf_df, hist_cols)` after merge (Section 2).
  * Keep **all** calendar features (`minute_of_day`, `day_of_week`, `is_weekend`) strictly in `futr_exog_list`.
  * Make `postprocess_shift_and_prune()` the **only** place allowed to shift; code-review PRs against this.

### 14.3 Target mishandling (log-returns)

* **Symptom:** NaNs/inf in `y`, or returns computed on forward-filled prices.
* **Cause:** Missing bars filled before return calc; non-UTC timestamps causing duplicate bars.
* **Fix (exact):**

  * Build `y = log(close).diff()` **after** `regularize_to_grid_utc()` and before any imputation on OHLCV.
  * `assert_no_forward_fill_y()` (Section 2).
  * Drop rows where `close` is NaN **before** training; never impute `y`.

### 14.4 Quantile crossing / bad calibration

* **Symptom:** `q90 < q80` or PIT U-shape/coverage misses.
* **Cause:** No monotonic quantile constraint; heavy tails.
* **Fix (exact):**

  * Switch `loss: {kind: iqloss}` (ISQF) for quantile runs **or** keep MQ and attach **Conformal** (§8.2).
  * Use **StudentT** for distributional runs; request `level=[80,90,95]` and attach conformal only if coverage is off by >±2pp.

### 14.5 GPU OOM / slow inference

* **Symptom:** OOM during training/predict; inference > latency SLO.
* **Cause:** PatchTST context too long, batch too large, hidden sizes too big.
* **Fix (exact):**

  * **Training:** lower `batch_size` → reduce `hidden_size`/`n_heads` (PatchTST) → shorten `input_size`.
  * **Inference:** set `torch.set_grad_enabled(False)` (already in §10), reduce `batch_size` if exposed; drop PatchTST first; keep NHITS/NBEATSx.
  * Keep feature cap ≤ **256** (§3).

### 14.6 Training instability (loss spikes/NaNs)

* **Symptom:** Divergence mid-epoch; NaNs in weights/loss.
* **Cause:** Too high LR; unbounded outliers; numeric issues in indicators.
* **Fix (exact):**

  * Lower `learning_rate` one notch (1e-3 → 5e-4 or 1e-4 for PatchTST).
  * Ensure **winsorization** of `y_train` on training-only copy (§2).
  * Drop obviously broken features (near-constant, NaN-prone) via existing availability/variance pruner.

### 14.7 Bad data (gaps/dupes/tz drift)

* **Symptom:** CV window sizes inconsistent; duplicate timestamps; sudden coverage collapse.
* **Cause:** Provider anomalies; daylight-savings/tz mislabels (should not happen in UTC, but verify).
* **Fix (exact):**

  * `assert_regular_grid(df,"15min")`, `assert_utc_eob(df,"15min")` on every run.
  * If last bar is missing/partial, **skip the cycle** (don’t fabricate bars).
  * Keep a data-provenance log (counts of bars/day, %missing) in `reports/monitoring/`.

### 14.8 sCRPS not computed / misleading leaderboard

* **Symptom:** Leaderboard ranks by MAE only; promotion ambiguous.
* **Cause:** You didn’t request dense quantiles.
* **Fix (exact):**

  * For **finalists only**, re-run CV with `quantiles=[0.01,...,0.99]` and compute sCRPS using NF’s metric (already wired in §9/§12). Stop after selection.

### 14.9 Ensemble miscalibration

* **Symptom:** Equal-weight blend widens/narrows intervals inconsistently; coverage drifts.
* **Cause:** Averaging parameters or mixing interval definitions.
* **Fix (exact):**

  * **Never** average distribution parameters. Always blend **point** and **quantiles/intervals** level-wise (Section 7).
  * If still under-covered, attach **conformal** post-ensemble and re-check coverage.

### 14.10 Save/Load & version drift

* **Symptom:** Loaded model predicts differently vs. pre-save; runtime errors after upgrades.
* **Cause:** Unpinned versions; changed NF/torch defaults.
* **Fix (exact):**

  * Always `nf.save(path, save_dataset=True)` and `NeuralForecast.load(path)`.
  * Snapshot `version_manifest.json` + `requirements-lock.txt` (§11).
  * On any upgrade, **parallel-train** under `trial_*/` and gate with §12 before promotion.

### 14.11 Timezone or calendar mistakes

* **Symptom:** Calendar `futr_exog` off by one bar; weekend flags mismatched.
* **Cause:** Non-UTC base; wrong EOB assumption.
* **Fix (exact):**

  * Force **UTC** from ingest; rebuild calendars from UTC `ds` only (Section 3 custom funcs).
  * Add a smoke test that checks known UTC boundaries (e.g., Monday 00:00 UTC transitions).

### 14.12 Overfitting via feature bloat

* **Symptom:** Great train fit; CV/test sCRPS flat or worse; unstable coverage.
* **Cause:** Too many correlated indicators; weak MTF features.
* **Fix (exact):**

  * Keep **≤256** features; prune |Spearman| ≥ 0.95 (§3).
  * Prefer small, well-chosen sets; delete features that don’t move sCRPS ≥ 0.5–1% in ablations.

### 14.13 Horizon mismatch / empty predictions

* **Symptom:** `predict` returns fewer than `h` rows.
* **Cause:** Missing future rows or calendar futr_exogs.
* **Fix (exact):**

  * Use `build_future_calendar(last_ds, h)` (§10) and pass via `predict(..., futr_df=...)`.
  * Verify `len(futr_df) == h` and required columns exist before calling `predict`.

### 14.14 Live loop race conditions

* **Symptom:** Occasionally predicting on partial last bar.
* **Cause:** Triggering exactly at EOB with no buffer.
* **Fix (exact):**

  * In `run_predict.py --loop`, keep `--buffer_sec` ≥ **45s**.
  * If your data source lags, consider 90s. Skip cycle if bar isn’t final.

### 14.15 Regression after “harmless” refactors

* **Symptom:** Same configs, worse results.
* **Cause:** Silent changes in registry, YAML, or scaler settings.
* **Fix (exact):**

  * CI: run `utils/smoke.py` + a mini-CV (1 window) on a fixed slice for **snapshot parity** before merging.
  * Require PRs to include updated `version_manifest.json` if deps moved.

---