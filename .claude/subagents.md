# Claude Code Sub-Agents Proposal

- Establish architecture, configuration, and artifact conventions.
- Implement data contracts, features, models, CV/metrics, and UQ.
- Orchestrate training, selection/ensembling, and HPO.
- Deliver inference, monitoring, and maintenance for production.
- Ensure QA, documentation, and release/version practices.

---

## 1) system-architect-and-config-steward

- Purpose: Own architecture, repo conventions, and config surfaces to ensure coherence, reproducibility, and artifact stability.
- Spec Scope: docs/forecasting_sf_plan.md §0 (132–290), §1 (291–583); docs/proposed-spec-structure.md Configuration/Repository sections
- Specialized Domain Knowledge:
  - ML system architecture; YAML config design; artifact/version governance
  - NeuralForecast usage patterns and experiment structuring
  - CI/CD-friendly settings and environment isolation
- Responsibilities (operating procedures):
  - Define/maintain global defaults (freq=15min UTC EOB, horizons h∈{4,8,16,32}, seeds, scalers) in settings.yaml and experiments/defaults.yaml.
  - Standardize artifact paths: experiments/h{h}/cv_results.parquet, metrics.csv, best/, reports/h{h}/.
  - Enforce naming conventions and horizon portfolio; require level=[80,90,95] where applicable.
  - Specify CLI contracts for run_train.py and run_predict.py; ensure compatibility with NF save/load.
  - Approve changes to spec boundaries; resolve ownership ambiguities across agents.
- Inputs: repo docs/specs; team proposals/PRs; NF constraints.
- Outputs: canonical configuration schemas, path conventions, and decision records in docs.
- Interfaces/Dependencies: None; consulted by all agents for conventions.
- Guardrails & KPIs: stable paths, deterministic seeds, zero breaking changes to CLI without version bump; config drift checks in CI.
- Failure modes & recovery: path drift → migration notes + symlinks; config conflicts → RFC process and versioned schema.

---

## 2) data-contracts-and-validation

- Purpose: Enforce canonical NF long frame on 15-min UTC EOB grid with safe target assembly and CI guardrails.
- Spec Scope: docs/forecasting_sf_plan.md §2 (584–783) with sub-§2.1–2.9; supporting §0.2–0.3
- Specialized Domain Knowledge:
  - Pandas time indexing; UTC/EOB alignment; resampling and gap checks
  - Log-returns computation; winsorization policy (train-only)
  - Assertion design for CI (fast, deterministic)
- Responsibilities (operating procedures):
  - Implement and run: assert_regular_grid, assert_utc_eob, assert_shifted, assert_no_forward_fill_y before any fit/predict.
  - Build canonical frame: columns [unique_id, ds, y, ...], with y as log return; never forward-fill y.
  - Provide IO helpers for parquet read/write; add timestamped paths per experiment.
  - Generate validation summaries (missingness, grid conformity) for CI logs.
  - Seed handling: ensure deterministic sampling; record hashes of input data snapshots if applicable.
  - Winsorization explicitly TRAIN-ONLY (never applied to inference-time inputs or evaluation windows); log parameters.
- Inputs: raw OHLCV and any exogs; config for winsorization levels (train-only).
- Outputs: validated canonical dataframe; validation logs; failure exceptions with guidance.
- Interfaces/Dependencies: system-architect-and-config-steward; upstream to feature-engineering and training-orchestrator.
- Guardrails & KPIs: 100% grid regularity; UTC/EOB confirmed; hist_ features shifted(1) confirmed; zero forward-fill on y.
- Failure modes & recovery: grid gaps → raise with suggested resample/repair; tz mismatch → normalize to UTC and re-assert; leakage risk → block and report offending columns.

---

## 3) feature-engineering

- Purpose: Build leakage-safe exogenous features with strict shift(1) and precise MTF alignment to 15-min EOB.
- Spec Scope: docs/forecasting_sf_plan.md §3 (784–1203) with sub-§3.1–3.7
- Specialized Domain Knowledge:
  - vectorbt / TA-Lib / pandas-ta; MTF aggregation (30m/1h/4h → 15m EOB)
  - Availability filtering and pruning to ≤256 features; hist/futr/stat semantics
- Responsibilities (operating procedures):
  - Maintain declarative indicator registry: fields [name, params, kind, timeframe, label, role(hist/futr/stat)].
  - Build indicators per timeframe; forward-fill to 15-min grid; then shift(1) for hist_.
  - Implement postprocess_shift_and_prune(df, shift=1): drop columns <98% availability; cap ≤256 using simple variance / redundancy heuristics (NO model-based feature importance or embedded model training for selection).
  - Produce exog lists: hist_exog_list, futr_exog_list, stat_exog_list for NF wiring.
  - Validate no hist_ column remains unshifted; align higher TF EOB to base grid.
- Inputs: canonical dataframe; registry specs; pruning thresholds.
- Outputs: enriched dataframe with prefixed columns; exog lists; build logs.
- Interfaces/Dependencies: data-contracts-and-validation; model-factory consumes exog lists; training-orchestrator orchestrates build.
- Guardrails & KPIs: zero leakage (all hist_ shifted); ≤256 features post-prune; ≥98% availability for retained features; correct MTF alignment.
- Failure modes & recovery: misalignment → reindex with EOB alignment and assert; over-cap features → prune by variance/importance; missing columns → warn and continue if optional.

---

## 4) model-factory

- Purpose: Provide NF-native model instantiation with consistent losses, scalers, and exog wiring per horizon.
- Spec Scope: docs/forecasting_sf_plan.md §4 (1204–1564), sub-§4.0–4.2; supporting §1.5 stubs
- Specialized Domain Knowledge:
  - NF models: NHITS, NBEATSx, TiDE, PatchTST; scalers: robust, revin
  - Losses: DistributionLoss("StudentT"), MQLoss, IQLoss; input sizes (1024, PatchTST 2048)
- Responsibilities (operating procedures):
  - Implement instantiate_models(exp_cfg, exog_lists, h) producing list of NF model instances.
  - Configure losses and scalers per model family; set input size defaults; set exog lists (hist/futr/stat) appropriately.
  - Respect horizon h in model configs; set seeds for reproducibility.
  - Support nf.save(..., save_dataset=True) and NeuralForecast.load(...), documenting paths.
  - DO NOT implement custom training loops—only use NeuralForecast.fit/predict/cross_validation.
- Inputs: exog lists; experiment config; horizon h.
- Outputs: ready-to-fit NF model instances; model config logs.
- Interfaces/Dependencies: feature-engineering (exogs); system-architect-and-config-steward (conventions); consumed by CV and training-orchestrator.
- Guardrails & KPIs: consistent wiring across models; correct losses/scalers; reproducible instantiation.
- Failure modes & recovery: incompatible exogs → drop/notify; OOM → reduce batch/input size per spec defaults.

---

## 5) cross-validation-and-metrics

- Purpose: Execute NF-native cross_validation with documented windowing; compute sCRPS/MAE/RMSE and aggregate metrics.
- Spec Scope: docs/forecasting_sf_plan.md §5 (1565–1807) with sub-§5.1–5.2; supporting §0.4, §1.5
- Specialized Domain Knowledge:
  - NF cross_validation arguments; PredictionIntervals(level=[80,90,95])
  - sCRPS computation and metrics aggregation across windows
- Responsibilities (operating procedures):
  - Run validation guards; construct NF object with models; execute cross_validation(df, n_windows≈6→10, step_size=h, val_size=4*h, refit=True, prediction_intervals=PredictionIntervals(n_windows=6), level=[80,90,95]).
  - Save raw CV outputs; compute per-model aggregated metrics; build leaderboard.
  - Provide PIT helper hooks and pass conformal settings through.
  - Ensure no leakage (shifted hist_, proper splits) before CV.
  - Prohibit wrapping/subclassing NF cross_validation; parameterize only.
- Inputs: validated dataframe; model instances; horizon h; levels.
- Outputs: cv_results (parquet); aggregated metrics (csv/json); leaderboard view.
- Interfaces/Dependencies: model-factory; data-contracts-and-validation; feeds uncertainty-and-calibration and selection.
- Guardrails & KPIs: correct windowing; reproducible metrics; sCRPS as primary; artifacts saved to stable paths.
- Failure modes & recovery: mis-specified windows → adjust per h; missing columns → fail-fast with guidance; long runtimes → reduce n_windows for pilot.

---

## 6) uncertainty-and-calibration

- Purpose: Diagnose and enforce calibration using coverage and PIT; feed calibration gates back into training.
- Spec Scope: docs/forecasting_sf_plan.md §8 (2291–2431) with sub-§8.1–8.5; supporting §0.5
- Specialized Domain Knowledge:
  - Conformal intervals via NF; quantile/distribution training impacts
  - Coverage/PIT diagnostics; vol-conditional analysis
- Responsibilities (operating procedures):
  - Compute coverage at 80/90/95 and check ±2pp calibration gates; report pass/fail.
  - Produce PIT histograms; coverage-by-vol-decile diagnostics; suggest remedies from §8.4.
  - Persist diagnostics in reports/h{h}/; annotate CV leaderboard with calibration notes.
  - Provide callable for training-orchestrator to gate promotions.
  - Use only NF PredictionIntervals / native quantiles; DO NOT implement custom conformal algorithms.
- Inputs: CV predictions/quantiles; levels; validation data segments.
- Outputs: coverage metrics; PIT plots; calibration status for gate checks.
- Interfaces/Dependencies: cross-validation-and-metrics; training-orchestrator consumes gates; monitoring later uses outputs.
- Guardrails & KPIs: coverage within ±2pp; stable PIT shape; reproducible diagnostics.
- Failure modes & recovery: under-coverage → widen intervals or adjust conformal settings; unstable PIT → review loss/scale.

---

## 7) training-orchestrator

- Purpose: Drive end-to-end training per YAML config, coordinating validation, features, models, CV, and persistence.
- Spec Scope: docs/forecasting_sf_plan.md §9 (2432–2757) sub-§9.1–9.5; §1.7 entry-points
- Specialized Domain Knowledge:
  - CLI/YAML orchestration; NF fit/predict/predict_insample; artifact management
- Responsibilities (operating procedures):
  - Parse --exp; load settings; call validation; build features; instantiate models; run CV.
  - Attach conformal via PredictionIntervals(levels); enforce guards pre/post steps.
  - Select best (with selection agent); save nf.save(save_dataset=True) to experiments/h{h}/best/.
  - Generate insample diagnostics and write reports/h{h}/.
  - Persist experiment artifacts strictly to filesystem; NO external tracking service introduction.
- Inputs: experiment YAML; settings; data; exog; models.
- Outputs: trained artifacts in experiments/h{h}/; logs and reports.
- Interfaces/Dependencies: system-architect-and-config-steward; data-contracts-and-validation; feature-engineering; model-factory; cross-validation-and-metrics; uncertainty-and-calibration; model-selection-and-ensembling.
- Guardrails & KPIs: successful run with artifacts at stable paths; config reproducibility; failures are actionable with logs.
- Failure modes & recovery: partial runs → resume from checkpoints; missing config → fallback to defaults; CV failures → pilot mode.

---

## 8) model-selection-and-ensembling

- Purpose: Rank models by sCRPS, apply guardrails, and produce simple ensembles for promotion.
- Spec Scope: docs/forecasting_sf_plan.md §7 (2142–2290) sub-§7.1–7.7
- Specialized Domain Knowledge:
  - Ranking/selection; blending; artifact tracking
- Responsibilities (operating procedures):
  - Aggregate CV metrics; rank by mean sCRPS; compute deltas vs baseline.
  - Apply promotion guardrail: require ≥1% sCRPS improvement; if close, evaluate ensemble.
  - Build equal-weight top-2 ensemble ONLY (no stacking/weight optimization); evaluate sCRPS; choose winner.
  - Provide selection report to orchestrator; tag artifacts accordingly.
- Inputs: aggregated metrics; CV outputs; ensemble utilities.
- Outputs: selection decision (winner/ensemble); metadata for artifact tagging.
- Interfaces/Dependencies: cross-validation-and-metrics; training-orchestrator; uncertainty-and-calibration for calibration gates.
- Guardrails & KPIs: transparent criteria; ensemble only if improves; audit trail of decisions.
- Failure modes & recovery: ties → prefer simpler model; inconsistent metrics → re-aggregate or rerun CV subset.

---

## 9) hyperparameter-optimization

- Purpose: Perform bounded HPO with staged evaluation and promotion criteria to improve sCRPS efficiently.
- Spec Scope: docs/forecasting_sf_plan.md §6 (1808–2141) sub-§6.1–6.6
- Specialized Domain Knowledge:
  - NF model hyperparameters; early stopping and budget control
- Responsibilities (operating procedures):
  - Define tight search spaces per model; run pilot CV (n_windows≈6); rank params by sCRPS.
  - Promote promising configs to extended CV (n_windows≈10) if improvement ≥0.5%.
  - Record best params; pass to selection agent for final comparison.
  - Optionally leverage NF Auto* under time-boxed constraints.
  - Early stop: if two successive promotion attempts yield <0.2–0.3% incremental sCRPS gain, terminate search.
- Inputs: model families; initial configs; CV runner.
- Outputs: candidate params with metrics; promotion logs.
- Interfaces/Dependencies: model-factory; cross-validation-and-metrics; training-orchestrator; model-selection-and-ensembling.
- Guardrails & KPIs: bounded search; promotion thresholds enforced; compute budget respected.
- Failure modes & recovery: overfitting → require holdout check; noisy gains → rerun subset.

---

## 10) inference-and-live-deployment

- Purpose: Execute one-shot and live-loop inference with strict guards, timing buffers, and robust logging.
- Spec Scope: docs/forecasting_sf_plan.md §10 (2758–2955) sub-§10.1–10.7
- Specialized Domain Knowledge:
  - NF predict flows; real-time system considerations; UTC timing rules
- Responsibilities (operating procedures):
  - Implement one-shot mode and live loop with 45+ sec post-EOB buffer; build tail features on the fly.
  - Load best model (NeuralForecast.load); run predict with conformal intervals; write outputs to reports/h{h}/.
  - Validate inputs on each iteration; on failure retry bounded (e.g., ≤3 attempts) then log and skip bar (no complex circuit breaker logic).
  - Optional simple latency logging (timestamps) only; no external monitoring system integration.
- Inputs: model_path; experiment config; recent data; exog builders.
- Outputs: prediction files with intervals; runtime logs.
- Interfaces/Dependencies: training-orchestrator; feature-engineering; model-factory; monitoring-and-maintenance.
- Guardrails & KPIs: on-time completion before next bar; zero leakage; consistent output schema.
- Failure modes & recovery: missing bars → skip with log; model load fail → fallback to previous version; runtime errors → circuit-breaker and alert.

---

## 11) monitoring-and-maintenance

- Purpose: Monitor live performance and calibration drift; manage retraining cadence, versioning, and rollback.
- Spec Scope: docs/forecasting_sf_plan.md §11 (2956–3135) sub-§11.1–11.8; docs/versioning_system.md
- Specialized Domain Knowledge:
  - Drift metrics (sCRPS regression %, coverage deviation ±pp, PIT stability)
  - Version bump/tagging; rollback and retraining SOPs
- Responsibilities (operating procedures):
  - Compute rolling sCRPS vs baseline, coverage drift, PIT stability metrics; flag thresholds (no external alert infra—write flags to report files).
  - Define and execute retrain triggers; coordinate artifact versioning and rollback plans.
  - Produce periodic drift reports under reports/h{h}/ and maintain lightweight logs/.
  - Integrate with orchestrator to schedule retrains and with documentation for release notes.
- Inputs: live predictions; realized outcomes; baseline metrics; version metadata.
- Outputs: drift flag files; retrain notes; version bump notes; drift reports.
- Interfaces/Dependencies: uncertainty-and-calibration; training-orchestrator; documentation-and-release-manager.
- Guardrails & KPIs: timely detection (next-bar evaluation) of material drift; clean rollback path; version traceability.
- Failure modes & recovery: false positives → smoothing; missed drift → add redundancy and thresholds per deciles.

---

## 12) qa-and-test-runner

- Purpose: Enforce acceptance criteria via fast unit/smoke tests and static checks; keep guardrails green.
- Spec Scope: docs/forecasting_sf_plan.md §12 (3136–3323) and §13 (3324–3492); risk references §14 (3493–3664)
- Specialized Domain Knowledge:
  - pytest-based testing; CI wiring; minimal synthetic datasets for NF
- Responsibilities (operating procedures):
  - Write tests for validate guards, feature shift discipline, model factory wiring.
  - Add CV smoke tests on tiny synthetic frames; assert artifact creation and metrics schema.
  - Test UQ coverage computations and PIT helper with controlled quantiles.
  - Provide lint/format tasks; integrate tests into CI.
  - Keep suite minimal: no large backtests; single tiny dataset per horizon suffices.
- Inputs: codebase; sample configs; synthetic data builders.
- Outputs: test reports; CI pass/fail signals; coverage snapshots for QA.
- Interfaces/Dependencies: all build agents; feeds signals to system-architect-and-config-steward.
- Guardrails & KPIs: sub-5min local test runtime; deterministic results; failing tests are actionable.
- Failure modes & recovery: flaky tests → seed control and retries; slow suites → split and parallelize.

---

## 13) documentation-and-release-manager

- Purpose: Keep docs accurate, concise, and actionable; manage release notes and versioning narrative.
- Spec Scope: docs/forecasting_sf_plan.md cross-references; docs/versioning_system.md; §9–§11 integration points
- Specialized Domain Knowledge:
  - Technical ML documentation; changelog and semantic versioning practices
- Responsibilities (operating procedures):
  - Maintain end-to-end guides for train, predict, CV, UQ with minimal examples.
  - Align docs/forecasting_sf_plan.md with implementation; add validation appendices as needed.
  - Curate experiment READMEs and release notes per version bump; document rollback procedures.
  - Avoid over-documenting internals; focus on user entrypoints and guardrail explanation.
- Inputs: implementation changes; monitoring signals; version changes.
- Outputs: updated docs; release notes; troubleshooting guides.
- Interfaces/Dependencies: system-architect-and-config-steward; training-orchestrator; monitoring-and-maintenance.
- Guardrails & KPIs: docs updated within PRs; zero stale CLI examples; clear rollback instructions.

---

## Notes on coordination and non-overlap

- Boundaries: CV/metrics handle evaluation; UQ owns calibration diagnostics; selection/ensembling consumes CV outputs; HPO tunes models pre-selection.
- Guards: Validation runs before any fit/CV/predict. Feature shift and exog wiring enforced centrally in feature-engineering and validated by QA.
- Artifacts: All agents follow experiments/h{h}/ and reports/h{h}/ structures with stable names to support downstream automation.
