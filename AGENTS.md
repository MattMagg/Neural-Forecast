# Repository Guidelines

Read this first: The canonical, detailed reference for this repo is docs/forecasting_sf_plan.md (core technical plan). For navigation and project-level context for another agent, see CLAUDE.md. This AGENTS.md gives a high-level map plus entry points into those sources.

## Project Overview
- Purpose: Intraday BTC forecasting (15‑minute base) with calibrated prediction intervals across horizons h ∈ {4, 8, 16, 32}.
- Principles: NeuralForecast‑native only (no custom backtester or persistence), strict UTC end‑of‑bar timestamps, zero‑leakage features (shift(1)).

## Project Structure & Module Organization
- Entrypoints: `run_train.py`, `run_predict.py`, `settings.yaml`.
- Packages: `features/` (registry, builder), `nf_models/` (factory), `cv/` (cross‑validation runner), `uq/` (diagnostics/ensembles), `utils/` (I/O, validation).
- Experiments & outputs: `experiments/` (YAML per horizon: `h4.yaml`, `h8.yaml`, `h16.yaml`, `h32.yaml`, plus `defaults.yaml`), `reports/` (plots, metrics), `data/` (canonical 15‑min frame).
- Notes: Some `.py` files are stubs—follow the specifications in docs/forecasting_sf_plan.md when implementing. Do not add new top‑level dirs; keep snake_case files and PascalCase classes.

## Core Workflows & Commands
```bash
# Train (example: h=16)
python run_train.py --exp experiments/h16.yaml --save

# Inference (batch)
python run_predict.py --exp experiments/h16.yaml --model_path experiments/h16/best/ --h 16

# Acceptance report (per horizon)
python reports/acceptance.py --hdir experiments/h16 --winner ENS2 --baseline_scrps 0.1234
```
What they do: orchestrate NF‑native training/CV; restore saved models to predict with PIs (80/90/95); produce ACCEPT/REJECT rationale and artifacts.

## Coding Standards & Guardrails
- Style: Python, PEP 8, 4‑space indents, type hints for new code.
- NF‑native only: `NeuralForecast` for `fit/predict/predict_insample/cross_validation`; save/load via `nf.save()` / `NeuralForecast.load()`.
- Imports: `from neuralforecast import NeuralForecast`; models `NHITS`, `NBEATSx`, `TiDE`, `PatchTST`; losses `DistributionLoss`, `MQLoss`, `ISQF`, `IQLoss`.
- Exogenous variables: wire by name via `hist_exog_list`, `futr_exog_list`, `stat_exog_list`.
- Data contract: long format `[unique_id, ds, y, ...]`, tz‑aware UTC 15‑min EOB; never forward‑fill `y`; winsorization (training only) allowed per plan.
- Artifacts/naming: `experiments/h{h}/cv_results.parquet`, `experiments/h{h}/metrics.csv`, `experiments/h{h}/best/`, `reports/h{h}/preds_*.parquet`.

## Testing & Quality Gates
- Smoke tests with pytest are encouraged (fast only).
- Enforce validators: `assert_regular_grid(df, "15min")`, `assert_utc_eob(df, "15min")`, `assert_shifted(df, hist_cols)`, `assert_no_forward_fill_y(df)`.
- Feature pipeline: ensure strict `shift(1)` on hist exogs and end‑of‑bar alignment for MTF features.
- Where to look: See docs/forecasting_sf_plan.md → Sections 2 (Data), 3 (Features), 5 (CV), 12 (Acceptance).

## Commit & Pull Request Guidelines
- Follow repo versioning practices in `docs/versioning_system.md` (version bumps, tags, artifact naming).
- Conventional Commits (e.g., `feat:`, `fix:`, `docs:`); small, focused changes; present tense.
- PRs include: intent, brief design notes, commands used, updated YAMLs, and links to artifacts (e.g., `experiments/h16/cv_results.parquet`).
- Must confirm NF‑native primitives only; UTC/EOB and leakage checks enforced.

## Quick Links
Plan sections (anchor links):
- 1) Repository layout → docs/forecasting_sf_plan.md#1-repository-layout-lean-explicit
- 2) Data contracts & validation → docs/forecasting_sf_plan.md#2-data-contracts-validation
- 3) Exogenous features (indicators) → docs/forecasting_sf_plan.md#3-exogenous-features-indicators-and-others
- 4) NF model portfolio & defaults → docs/forecasting_sf_plan.md#4-neuralforecast-model-portfolio-defaults
- 5) Cross-validation (NF-native) → docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage
- 6) Hyperparameter strategy → docs/forecasting_sf_plan.md#6-hyperparameter-strategy-bounded-low-variance
- 7) Model selection & ensembling → docs/forecasting_sf_plan.md#7-model-selection-simple-ensembling
- 8) Uncertainty & calibration → docs/forecasting_sf_plan.md#8-uncertainty-calibration
- 9) Training & evaluation workflow → docs/forecasting_sf_plan.md#9-training-evaluation-workflow
- 10) Inference & live deployment → docs/forecasting_sf_plan.md#10-inference-live-deployment-considerations
- 11) Maintenance & retraining → docs/forecasting_sf_plan.md#11-maintenance-retraining
- 12) Acceptance criteria & quality gates → docs/forecasting_sf_plan.md#12-acceptance-criteria-quality-gates
- 13) Implementation checklist → docs/forecasting_sf_plan.md#13-implementation-checklist
- Command quick sheet → docs/forecasting_sf_plan.md#command-quick-sheet-copypaste

Other docs:
- Core plan: `docs/forecasting_sf_plan.md` (authoritative spec)
- Project guide (companion): `CLAUDE.md`
- Versioning: `docs/versioning_system.md`
- Implementation timeline: `implementation_workflow.md`

## Security & Configuration Tips
- `settings.yaml` holds static defaults only—no secrets. Pin NF versions; upgrade deliberately.
- Keep artifact structure stable; downstream processes depend on folder/file names.
