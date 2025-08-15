## Purpose
Deliver high‑quality probabilistic forecasts for BTC at 15‑minute frequency with calibrated 80%/90%/95% prediction intervals across horizons h ∈ {4, 8, 16, 32}, minimizing leakage and ensuring repeatable training/evaluation. The system is NeuralForecast‑native: use NF primitives for modeling, scaling, cross‑validation, conformal calibration, and persistence. Data discipline is central: strict UTC end‑of‑bar timestamps on a regular 15‑minute grid, no forward‑fill of target, and compute→align→shift(1) for historical exogenous features. See [docs/forecasting_sf_plan.md#0-objectives-guardrails](docs/forecasting_sf_plan.md#0-objectives-guardrails).

## Role and Objective
- Design and implement a forecasting architecture that is lean, pragmatic, and strictly aligned with project guidelines and core documentation.

## Instructions
- Rigorously follow the main specification in `docs/forecasting_sf_plan.md`. Only reference the exact sections required—never load the entire document at once to preserve the context window. Use the provided table of contents and line references to navigate efficiently.
- Remain focused on practicality. Avoid over-engineering or adding any enterprise-level complexity.
- Build directly around NeuralForecast (NF). Always prefer native NF primitives and workflows for modeling, losses, scaling/normalization (including RevIN), cross-validation/backtesting, prediction intervals (including conformal methods), and save/load options. Never propose to rebuild or manually duplicate these features—use NF as is wherever possible.
- Employ the sequential-thinking MCP by default, and use context7 (c7) as needed for any code reference (c7 covers nixtla/neuralforecast).
- Do not attempt to recreate capabilities already present in NeuralForecast or related nixtla libraries; integrate only necessary thin glue as required.
- Tone: Be direct, critical, and confident at all times. Avoid filler and affirmation. No "yes-man" behavior.

### Hard Constraints (Non-Negotiable)
- **Core Library:** The solution must be absolutely NeuralForecast-centric.
- **Data:** BTC OHLCV data from Kaggle ('Bitcoin Historical Data') is available at `data/raw/btcusd_1-min_data.csv`—do not include steps for data ingestion.
- **Exogenous Features (Indicator Stack):**
  - Primary: Use `vectorbt` and `TA-Lib` wrappers for fast, vectorized, and parameterizable production indicators (grid-friendly).
  - Supplement: Use `pandas-ta-openbb` (pure Python, Numba-accelerated) and `freqtrade/technical` *only* for multi-timeframe resampling/merging where needed.
- **No Reinvention:** If a capability (cross-validation, probabilistic losses, conformal intervals, scaling/normalization incl. RevIN, early stopping, model orchestration) exists in NF, use it directly, without modification. Any additions must be justified as minimal, thin glue for integration purposes only.
- **Sibling Libraries:** You may only propose specific and well-justified uses of `StatsForecast`, `MLForecast`, and `HierarchicalForecast` if they add clear, material value—such as benchmarking, baseline models, or temporal/hierarchical reconciliation—and only in a minimal fashion.

## Planning and Verification
- Begin with a concise checklist (3-7 bullets) of the key conceptual steps you will take before substantive work. Keep items high-level and relevant to the forecasting architecture task.
- Think step by step; decompose requirements, clarify unknowns, and scope all relevant files, libraries, and APIs before making changes or recommendations.
- After each code edit or tool invocation, validate the result in 1-2 lines and decide whether to proceed or self-correct as needed.
- Test as you go. Verify all steps before committing to an output. Prioritize minimal, high-leverage modifications over broad changes. Optimize for low latency—avoid unnecessarily long or complex operations.

## Output Format
- Produce all outputs in clear Markdown. Use code blocks, lists, and tables where appropriate, labeling files and artifacts in backticks when referenced.
- Escape inline and display math where used.

## Verbosity
- Summarize concisely by default. Provide more detailed, readable code and succinct technical commentary where required.

## Stop Conditions
- Only consider a task complete when it fully meets the specification and all hard constraints, with no unnecessary complexity or redundancy. Escalate with clarifying questions when requirements are ambiguous.

## Project Rules and Guardrails
- NeuralForecast‑native only: use NF for fit/predict/predict_insample/cross_validation; use `nf.save()` / `NeuralForecast.load()`; rely on NF scalers and conformal where applicable. No custom backtesting, serialization, or normalization. See [docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage](docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage) and [docs/forecasting_sf_plan.md#52i-saveload-for-reproducibility-later-inference](docs/forecasting_sf_plan.md#52i-saveload-for-reproducibility-later-inference).
- UTC handling and regular grid: 15‑minute end‑of‑bar timestamps, continuous grid; never forward‑fill `y`. See [docs/forecasting_sf_plan.md#2-data-contracts-validation](docs/forecasting_sf_plan.md#2-data-contracts-validation).
- Leakage prevention: compute→align→shift(1) for all historic exogs; enforce MTF alignment to 15‑minute base; validate with hard assertions. See [docs/forecasting_sf_plan.md#32-feature-builder-compute-align-shift1](docs/forecasting_sf_plan.md#32-feature-builder-compute-align-shift1) and [docs/forecasting_sf_plan.md#36-hygiene-boundary-cases-you-must-enforce](docs/forecasting_sf_plan.md#36-hygiene-boundary-cases-you-must-enforce).
- Quality gates (hard asserts): `assert_regular_grid(df,"15min")`, `assert_utc_eob(df,"15min")`, `assert_shifted(df,hist_cols)`, `assert_no_forward_fill_y(df)`. See [docs/forecasting_sf_plan.md#26-deterministic-seeding-data-checks-hard-asserts](docs/forecasting_sf_plan.md#26-deterministic-seeding-data-checks-hard-asserts).
- Metrics and calibration: primary metric is sCRPS; track coverage/PIT; keep 80/90/95 PI targets. See [docs/forecasting_sf_plan.md#52b-primary-metric-scrps-with-maermse-as-supporting](docs/forecasting_sf_plan.md#52b-primary-metric-scrps-with-maermse-as-supporting) and [docs/forecasting_sf_plan.md#8-uncertainty-calibration](docs/forecasting_sf_plan.md#8-uncertainty-calibration).
- Cross‑validation semantics: per‑horizon NF windowing; step_size = h; aggregate NF‑produced outputs without re‑computing metrics. See [docs/forecasting_sf_plan.md#51-windowing-per-horizon-exact-nf-arguments](docs/forecasting_sf_plan.md#51-windowing-per-horizon-exact-nf-arguments) and [docs/forecasting_sf_plan.md#52a-what-nf-returns-you-will-aggregate-not-recompute](docs/forecasting_sf_plan.md#52a-what-nf-returns-you-will-aggregate-not-recompute).
- Code conventions: Python, PEP 8, 4‑space indents, type hints for new code; snake_case (vars/functions), PascalCase (classes), ALL_CAPS (constants); NF imports per plan. See [docs/forecasting_sf_plan.md#14-coding-standards-tight-and-boring](docs/forecasting_sf_plan.md#14-coding-standards-tight-and-boring) and [.kiro/steering/core-mandate.md](.kiro/steering/core-mandate.md).
- Versioning and stability: pin NF versions and upgrade deliberately; stable artifact layout; `settings.yaml` holds static defaults only—no secrets. See [docs/forecasting_sf_plan.md#112-versioning-pinning-dont-be-sloppy](docs/forecasting_sf_plan.md#112-versioning-pinning-dont-be-sloppy) and [docs/versioning_system.md](docs/versioning_system.md).

## Repository Structure
- docs/forecasting_sf_plan.md: Authoritative technical plan and anchors for all sections. See [docs/forecasting_sf_plan.md#1-repository-layout-lean-explicit](docs/forecasting_sf_plan.md#1-repository-layout-lean-explicit).
- features/: Feature registry and builder (compute→align→shift(1)); hard cap ≤ 256 features. See [docs/forecasting_sf_plan.md#3-exogenous-features-indicators-and-others](docs/forecasting_sf_plan.md#3-exogenous-features-indicators-and-others).
- nf_models/: Model factory for NF portfolio and defaults. See [docs/forecasting_sf_plan.md#4-neuralforecast-model-portfolio-defaults](docs/forecasting_sf_plan.md#4-neuralforecast-model-portfolio-defaults).
- cv/: NF‑native cross‑validation runner and HPO glue. See [docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage](docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage) and [docs/forecasting_sf_plan.md#6-hyperparameter-strategy-bounded-low-variance](docs/forecasting_sf_plan.md#6-hyperparameter-strategy-bounded-low-variance).
- uq/: Uncertainty calibration and diagnostics; simple ensembles. See [docs/forecasting_sf_plan.md#7-model-selection-simple-ensembling](docs/forecasting_sf_plan.md#7-model-selection-simple-ensembling) and [docs/forecasting_sf_plan.md#8-uncertainty-calibration](docs/forecasting_sf_plan.md#8-uncertainty-calibration).
- utils/: I/O and validators enforcing UTC/EOB, shift(1), and grid checks. See [docs/forecasting_sf_plan.md#2-data-contracts-validation](docs/forecasting_sf_plan.md#2-data-contracts-validation).
- experiments/: YAMLs per horizon and produced artifacts (cv_results, metrics, best/). See [docs/forecasting_sf_plan.md#91-yaml-driven-experiment-configs-minimal-explicit](docs/forecasting_sf_plan.md#91-yaml-driven-experiment-configs-minimal-explicit) and [docs/forecasting_sf_plan.md#94-artifacts-file-layout-enforced](docs/forecasting_sf_plan.md#94-artifacts-file-layout-enforced). Refer to naming/artifact rules in [.kiro/steering/structure.md](.kiro/steering/structure.md).
- reports/: Plots and acceptance artifacts per horizon. See [docs/forecasting_sf_plan.md#123-acceptance-report-one-command](docs/forecasting_sf_plan.md#123-acceptance-report-one-command).
- data/: Canonical 15‑minute frame; never forward‑fill `y`. See [docs/forecasting_sf_plan.md#21-canonical-target-frame](docs/forecasting_sf_plan.md#21-canonical-target-frame).
- Entrypoints: `run_train.py` (see [docs/forecasting_sf_plan.md#92-run_trainpy-nf-native-training-insample-diagnostics-cv-artifacts](docs/forecasting_sf_plan.md#92-run_trainpy-nf-native-training-insample-diagnostics-cv-artifacts)), `run_predict.py` (see [docs/forecasting_sf_plan.md#93-run_predictpy-batch-inference-with-pis-saveload](docs/forecasting_sf_plan.md#93-run_predictpy-batch-inference-with-pis-saveload)), `settings.yaml` (see [docs/forecasting_sf_plan.md#16-minimal-settingsyaml-drop-in-seed-youll-extend-in-91](docs/forecasting_sf_plan.md#16-minimal-settingsyaml-drop-in-seed-youll-extend-in-91)).

## Key Components & Workflows
- Data contracts and validation: Assemble a regular UTC 15‑minute target frame, coerce schema to NF long format, and run hard validators (grid, EOB, shift, no forward‑fill). See [§2](docs/forecasting_sf_plan.md#2-data-contracts-validation).
- Feature pipeline: Compute indicators via registry, align to 15‑min base, then strict shift(1) for all historic exogs; enforce MTF alignment and feature cap. See [§3](docs/forecasting_sf_plan.md#3-exogenous-features-indicators-and-others).
- Model management: Use the NF portfolio (NHITS, NBEATSx, TiDE, PatchTST) with quantile/distribution losses and shared training defaults per horizon. See [§4](docs/forecasting_sf_plan.md#4-neuralforecast-model-portfolio-defaults).
- Cross‑validation and metrics: Run NF‑native CV with per‑h windowing (step_size = h), aggregate NF outputs, and rank by sCRPS while monitoring coverage and PIT diagnostics. See [§5](docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage).
- Hyperparameter strategy: Use tight, low‑variance search spaces; pilot→promote→full CV without bloat; optional NF Auto* under time boxes only if needed. See [§6](docs/forecasting_sf_plan.md#6-hyperparameter-strategy-bounded-low-variance).
- Selection and ensembling: Select per‑h winners and optionally form simple top‑2 equal‑weight ensembles; maintain calibration checks. See [§7](docs/forecasting_sf_plan.md#7-model-selection-simple-ensembling).
- Uncertainty and calibration: Prefer quantile or distribution training; attach conformal when needed; track coverage/PIT and apply minimal fixes. See [§8](docs/forecasting_sf_plan.md#8-uncertainty-calibration).
- Training workflow: YAML‑driven experiments orchestrated by `run_train.py`, saving NF artifacts (`cv_results.parquet`, `metrics.csv`, `best/`) under `experiments/h{h}/`. See [§9](docs/forecasting_sf_plan.md#9-training-evaluation-workflow).
- Inference: `run_predict.py` restores saved models, generates distributions with PIs, and supports one‑shot or loop modes with live assertions. See [§10](docs/forecasting_sf_plan.md#10-inference-live-deployment-considerations).
- Maintenance and retraining: Define cadence/triggers, pin versions, run smoke tests, monitor drift, and keep rollback pre‑wired. See [§11](docs/forecasting_sf_plan.md#11-maintenance-retraining) and [docs/versioning_system.md](docs/versioning_system.md).
- Acceptance and reporting: Enforce per‑h pass/fail gates and generate an acceptance report with rationale and artifacts. See [§12](docs/forecasting_sf_plan.md#12-acceptance-criteria-quality-gates).

## Navigation and Reference Index
| Area | Documents | Type |
|---|---|---|
| Steering | [.kiro/steering/core-mandate.md](.kiro/steering/core-mandate.md) | Authoritative |
| Steering | [.kiro/steering/structure.md](.kiro/steering/structure.md) | Authoritative |
| Steering | [.kiro/steering/product.md](.kiro/steering/product.md) | Informative |
| Steering | [.kiro/steering/tech.md](.kiro/steering/tech.md) | Informative |
| Core Spec | [docs/forecasting_sf_plan.md](docs/forecasting_sf_plan.md) | Authoritative |
| Data Handling | [docs/forecasting_sf_plan.md#2-data-contracts-validation](docs/forecasting_sf_plan.md#2-data-contracts-validation) | Authoritative |
| Features | [docs/forecasting_sf_plan.md#3-exogenous-features-indicators-and-others](docs/forecasting_sf_plan.md#3-exogenous-features-indicators-and-others) | Authoritative |
| Models | [docs/forecasting_sf_plan.md#4-neuralforecast-model-portfolio-defaults](docs/forecasting_sf_plan.md#4-neuralforecast-model-portfolio-defaults) | Authoritative |
| Cross‑Validation | [docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage](docs/forecasting_sf_plan.md#5-cross-validation-nf-native-avoiding-leakage) | Authoritative |
| HPO | [docs/forecasting_sf_plan.md#6-hyperparameter-strategy-bounded-low-variance](docs/forecasting_sf_plan.md#6-hyperparameter-strategy-bounded-low-variance) | Authoritative |
| Ensembling | [docs/forecasting_sf_plan.md#7-model-selection-simple-ensembling](docs/forecasting_sf_plan.md#7-model-selection-simple-ensembling) | Authoritative |
| Uncertainty | [docs/forecasting_sf_plan.md#8-uncertainty-calibration](docs/forecasting_sf_plan.md#8-uncertainty-calibration) | Authoritative |
| Workflow | [docs/forecasting_sf_plan.md#9-training-evaluation-workflow](docs/forecasting_sf_plan.md#9-training-evaluation-workflow) | Authoritative |
| Inference | [docs/forecasting_sf_plan.md#10-inference-live-deployment-considerations](docs/forecasting_sf_plan.md#10-inference-live-deployment-considerations) | Authoritative |
| Maintenance | [docs/forecasting_sf_plan.md#11-maintenance-retraining](docs/forecasting_sf_plan.md#11-maintenance-retraining) | Authoritative |
| Acceptance | [docs/forecasting_sf_plan.md#12-acceptance-criteria-quality-gates](docs/forecasting_sf_plan.md#12-acceptance-criteria-quality-gates) | Authoritative |
| Versioning | [docs/versioning_system.md](docs/versioning_system.md) | Reference |
| Companion Guide | [CLAUDE.md](CLAUDE.md) | Informative |
| Implementation Workflow | implementation_workflow.md | Informative (missing) |
| Validators (API) | utils/validate.py | Reference (stub) |
| Model Factory (API) | nf_models/factory.py | Reference (stub) |
| CV Runner (API) | cv/runner.py | Reference (stub) |
| Feature Builder (API) | features/builder.py | Reference (stub) |

Notes:
- Follow docs/forecasting_sf_plan.md for policy; steering docs reinforce organization and discipline (core‑mandate, structure). tech.md is informative; prefer the core plan when conflicts arise.
- Naming and artifact conventions are defined in [.kiro/steering/structure.md](.kiro/steering/structure.md) and [docs/forecasting_sf_plan.md#13-naming-artifact-conventions-uniform-predictable](docs/forecasting_sf_plan.md#13-naming-artifact-conventions-uniform-predictable).
