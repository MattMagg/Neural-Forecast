# Implementation Plan

- [ ] 1. Implement bootstrap stubs and configuration (from §1.5-1.6 of forecasting_sf_plan.md)
  - [ ] 1.1 Create settings.yaml with minimal configuration
    - Implement exact configuration from §1.6 "Minimal settings.yaml (drop-in seed)"
    - Add freq: "15min", horizons: [4, 8, 16, 32], seed: 1337
    - Add winsor: {lower_q: 0.001, upper_q: 0.999}
    - Add default_input_size: {generic: 1024, PatchTST: 2048}
    - Add scaler_type: {default: robust, PatchTST: revin}
    - _Reference: §1.6 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 1.2 Implement utils/validate.py bootstrap stubs
    - Copy exact implementation from §1.5 "Bootstrap stubs (drop-in file skeletons)"
    - Write assert_regular_grid function with exact code from plan
    - Write assert_utc_eob function with exact code from plan  
    - Write assert_shifted function with exact code from plan
    - _Reference: §1.5 utils/validate.py in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 1.3 Implement utils/io.py bootstrap stubs
    - Copy exact implementation from §1.5 "Bootstrap stubs (drop-in file skeletons)"
    - Write load_canonical_frame function with exact code from plan
    - Write save_parquet function with exact code from plan
    - Write timestamped_path function with exact code from plan
    - _Reference: §1.5 utils/io.py in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 1.4 Implement nf_models/factory.py bootstrap stubs
    - Copy exact implementation from §1.5 "Bootstrap stubs (drop-in file skeletons)"
    - Write _make_loss function with exact code from plan
    - Write instantiate_models function with exact code from plan
    - Support NHITS, NBEATSx, TiDE, PatchTST with exact parameter wiring from plan
    - _Reference: §1.5 nf_models/factory.py in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 1.5 Implement cv/runner.py bootstrap stubs
    - Copy exact implementation from §1.5 "Bootstrap stubs (drop-in file skeletons)"
    - Write run_cv function with exact code from plan using NF-native cross_validation
    - _Reference: §1.5 cv/runner.py in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 1.6 Implement uq/diag.py bootstrap stubs
    - Copy exact implementation from §1.5 "Bootstrap stubs (drop-in file skeletons)"
    - Write compute_coverage function with exact code from plan
    - Write plot_pit function with exact code from plan
    - Write blend_equal function with exact code from plan
    - _Reference: §1.5 uq/diag.py in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 2. Extend core data processing utilities (from §2 Data contracts & validation)
  - [ ] 2.1 Extend utils/io.py with additional data transformation functions
    - Implement regularize_to_grid_utc function as specified in §2.5 "Regularization policy"
    - Implement make_nf_canonical function as specified in §2.4 "Canonical NF frame" with unique_id="BTC-USD"
    - Implement drop_train_nans_and_winsorize function as specified in §2.5 with [0.1%, 99.9%] winsorization
    - Follow exact specifications from §2.7 "Assembly path (from raw → NF-ready)"
    - _Reference: §2.4, §2.5, §2.7 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 2.2 Extend utils/validate.py with additional validation functions
    - Implement assert_no_forward_fill_y function as specified in §2.6 "Deterministic seeding & data checks"
    - Follow exact implementation from code block in §2.6 of the plan
    - Add comprehensive error handling as specified in §2.9 "Why this is correct (and safe)"
    - _Reference: §2.6 assert_no_forward_fill_y in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 3. Implement feature engineering pipeline (from §3 Exogenous features)
  - [ ] 3.1 Create features/registry.py declarative feature specification
    - Implement exact REGISTRY structure from §3.1 "Indicator registry (vectorbt + TA-Lib primary)"
    - Define indicators section with RSI [7,14,28], BB bandwidth [20], SMA/EMA [20,50] as specified
    - Add MTF features section for 30min/1h/4h timeframes as specified in §3.1
    - Add calendar features (minute_of_day, day_of_week, is_weekend) marked as kind="futr" per §3.5
    - Enforce ≤256 feature cap as specified in §3.3 "Feature selection & hard cap"
    - _Reference: §3.1, §3.3, §3.5 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 3.2 Implement features/builder.py feature computation pipeline
    - Implement build_indicators function using vectorbt/TA-Lib primary, pandas-ta supplement per §3.1
    - Implement apply_mtf function with EOB alignment using label='right', closed='right' per §3.2
    - Implement postprocess_shift_and_prune function with central shift(1) rule per §3.2 "compute → align → shift(1)"
    - Implement select_features function with ≥98% availability filter per §3.3 and ≤256 cap
    - Follow exact specifications from §3.4 "End-to-end assembly" for pipeline integration
    - Handle boundary cases per §3.6 "Hygiene & boundary cases you must enforce"
    - _Reference: §3.1, §3.2, §3.3, §3.4, §3.6 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 4. Extend NeuralForecast model factory beyond bootstrap stubs (from §4 NeuralForecast model portfolio)
  - [ ] 4.1 Extend nf_models/factory.py model instantiation system
    - Extend bootstrap stubs with full model portfolio per §4.0 "Portfolio justification (BTC intraday, 15-min)"
    - Support exact models specified: NHITS, NBEATSx, TiDE, PatchTST per §4.1 "Common model parameters"
    - Configure exact training parameters from §4.1: batch_size=512, learning_rate=1e-3, max_steps=20000, early_stop_patience_steps=400
    - Set input_size per §4.1: 1024 for NHITS/NBEATSx/TiDE and 2048 for PatchTST
    - Wire exog lists per §4.2 "Model-specific configurations" and §3.5 "NF wiring"
    - Use scaler_type="robust" default and revin=True for PatchTST per §4.1 table
    - _Reference: §4.0, §4.1, §4.2 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 5. Extend cross-validation and evaluation system beyond bootstrap stubs (from §5 Cross-validation NF-native)
  - [ ] 5.1 Extend cv/runner.py cross-validation execution
    - Extend bootstrap run_cv function per §5.1 "Windowing (per horizon) — exact NF arguments"
    - Configure exact windowing: n_windows=6 for pilot, step_size=h, val_size=4*h, refit=True per §5.1
    - Add summarize_cv function per §5.2.D "Drop-in runner and aggregator"
    - Compute sCRPS as primary metric per §5.2.B "Primary metric: sCRPS (with MAE/RMSE as supporting)"
    - Return exact CV format per §5.2.A "What NF returns (you will aggregate, not recompute)"
    - _Reference: §5.1, §5.2.A, §5.2.B, §5.2.D in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 6. Extend uncertainty quantification and diagnostics beyond bootstrap stubs (from §8 Uncertainty & calibration)
  - [ ] 6.1 Extend uq/diag.py probabilistic evaluation system
    - Extend bootstrap compute_coverage function per §8.3 "Diagnostic checks"
    - Extend bootstrap plot_pit function per §5.2.E "PIT helper (diagnostic only)"
    - Add coverage_by_vol_decile function per §8.3 for volatility regime analysis
    - Extend bootstrap blend_equal function per §7.2 "Simple ensembles (top-2 only)"
    - Implement PredictionIntervals conformal prediction per §8.2 "Conformal prediction intervals"
    - Add dense quantiles generation per §5.2.B for sCRPS calculation on finalists
    - _Reference: §8.2, §8.3, §5.2.E, §7.2 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 7. Create experiment configuration system (from §9.1 YAML-driven experiment configs)
  - [ ] 7.1 Implement YAML-driven experiment configs
    - Create experiments/defaults.yaml with global settings per §9.1 specifications
    - Create experiments/h4.yaml with h=4, step_size=4, val_size=16 for 1-hour horizon
    - Create experiments/h8.yaml with h=8, step_size=8, val_size=32 for 2-hour horizon  
    - Create experiments/h16.yaml with h=16, step_size=16, val_size=64 for 4-hour horizon (copy exact example from plan)
    - Create experiments/h32.yaml with h=32, step_size=32, val_size=128 for 8-hour horizon
    - Use exact model configurations from §9.1 example: NHITS, NBEATSx, TiDE, PatchTST with specified parameters
    - Configure exact loss specifications from plan: {kind: studentt} and {kind: mqloss, quantiles: [0.05,0.1,0.2,0.3,0.5,0.7,0.8,0.9,0.95]}
    - Set exact training parameters from plan: learning_rate=0.001, batch_size=512, max_steps=20000, early_stop_patience_steps=400
    - _Reference: §9.1 experiments/h16.yaml example in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 8. Implement training workflow orchestration (from §1.7 Entry-point skeletons and §9.2)
  - [ ] 8.1 Create run_train.py entry-point skeleton and extend to full pipeline
    - Start with exact entry-point skeleton from §1.7 "Entry-point skeletons (wire later sections here)"
    - Copy exact imports and workflow structure from plan's run_train.py skeleton
    - Extend with command-line argument parsing per §9.2 "run_train.py — NF-native training"
    - Add insample predictions for PIT analysis using predict_insample(step_size=h, level=[80,90,95]) per §5.2.F
    - Implement CV results summarization with sCRPS as primary metric per §5.2.B
    - Add model saving to experiments/h{h}/best/ using NF's native save per §5.2.I
    - _Reference: §1.7 run_train.py skeleton and §9.2 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 8.2 Create run_predict.py entry-point skeleton and extend to full inference
    - Start with exact entry-point skeleton from §1.7 "Entry-point skeletons (wire later sections here)"
    - Copy exact imports and workflow structure from plan's run_predict.py skeleton
    - Extend with command-line argument parsing per §9.3 "run_predict.py — batch inference"
    - Add prediction interval generation at 80/90/95 levels per §10.3 "one-shot & loop modes"
    - Implement live loop mode with 45+ second buffer per §10.1 "What happens every 15 minutes"
    - Add graceful error handling per §10.4 "Throughput & memory guards"
    - _Reference: §1.7 run_predict.py skeleton, §9.3, §10.1, §10.3, §10.4 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 9. Implement model selection and ensembling (from §7 Model selection & simple ensembling)
  - [ ] 9.1 Create model selection system
    - Implement selection protocol per §7.1 "Selection protocol (per horizon)"
    - Rank models by mean sCRPS across CV windows per horizon per §7.1
    - Require sCRPS improvement ≥1% over baseline for promotion per §7.4 "Workflow (per h)"
    - Implement simple equal-weight averaging per §7.2 "Simple ensembles (top-2 only)"
    - Blend point forecasts and quantiles separately per §7.3 "Drop-in utilities"
    - Save models per §7.5 "Final fit & save (single vs ensemble)"
    - _Reference: §7.1, §7.2, §7.3, §7.4, §7.5 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 10. Implement hyperparameter optimization and additional utilities (from §6 Hyperparameter strategy)
  - [ ] 10.1 Create cv/hpo.py hyperparameter search system
    - Implement search spaces per §6.1 "Search spaces (tight, model-specific + global)"
    - Implement pilot → promote → full CV per §6.2 "Procedure: pilot → promote → full CV"
    - Add promotion logic per §6.4 "Promotion thresholds & bookkeeping" requiring sCRPS improvement ≥0.5%
    - Implement resource limits per §6.5 "Practical guards (don't ignore)"
    - Support exact workflow from §6.2 with n_windows progression (6 → 10)
    - _Reference: §6.1, §6.2, §6.4, §6.5 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 10.2 Create uq/ensembles.py ensemble utilities
    - Implement ensemble utilities per §7.3 "Drop-in utilities"
    - Support top-2 model selection per §7.2 "Simple ensembles (top-2 only)"
    - Implement ensemble inference path per §7.6 "Inference path for ensembles"
    - Follow guardrails per §7.7 "Guardrails" for proper interval handling
    - _Reference: §7.2, §7.3, §7.6, §7.7 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

  - [ ] 10.3 Create utils/version.py versioning and environment management
    - Implement versioning per §11.2 "Versioning & pinning (don't be sloppy)"
    - Add smoke tests per §11.3 "Smoke tests (fast, decisive)"
    - Create rollback procedures per §11.6 "Rollback (pre-wired, zero doubt)"
    - Support environment snapshots for reproducibility per §11.2
    - _Reference: §11.2, §11.3, §11.6 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 11. Implement inference and deployment system (from §10 Inference & live deployment)
  - [ ] 11.1 Extend run_predict.py inference pipeline beyond skeleton
    - Extend skeleton per §10.3 "run_predict.py (inference entrypoint) — one-shot & loop modes"
    - Implement 15-minute sequence per §10.1 "What happens every 15 minutes (sequence)"
    - Add tail builders per §10.2 "Tail builders (drop-in utilities)"
    - Implement 45+ second buffer per §10.1 and §14.14 "Live loop race conditions"
    - Add throughput & memory guards per §10.4 "Throughput & memory guards (graceful degradation)"
    - Implement live conformal & monitoring per §10.5 "Live conformal & monitoring hooks"
    - Add minimal assertions per §10.6 "Minimal assertions in the live loop (don't skip)"
    - _Reference: §10.1, §10.2, §10.3, §10.4, §10.5, §10.6 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 12. Implement monitoring and maintenance system (from §11 Maintenance & retraining)
  - [ ] 12.1 Create monitoring and alerting system
    - Implement cadence & triggers per §11.1 "Cadence & triggers (what makes us retrain)"
    - Track coverage drift per §11.1: rolling 7-day empirical coverage >±3pp deviation
    - Track score decay per §11.1: 7-day mean sCRPS degrades >3% vs 30-day baseline
    - Monitor PSI per §11.1: thresholds 0.2 moderate (3 consecutive days), 0.3 major (any day)
    - Implement drift & monitoring per §11.4 "Drift & monitoring (simple, actionable)"
    - Add retrain procedure per §11.5 "Retrain procedure (no drama)"
    - Implement rollback per §11.6 "Rollback (pre-wired, zero doubt)"
    - _Reference: §11.1, §11.4, §11.5, §11.6 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 13. Implement risk mitigation and error handling (from §14 Risks & mitigations)
  - [ ] 13.1 Create comprehensive error handling system
    - Prevent MTF misalignment per §14.1 "MTF misalignment (30m/1h/4h → 15m)"
    - Prevent leakage per §14.2 "Leakage from non-shifted historic exogs"
    - Handle target computation per §14.3 "Target mishandling (log-returns)"
    - Manage quantile crossing per §14.4 "Quantile crossing / bad calibration"
    - Handle GPU OOM per §14.5 "GPU OOM / slow inference"
    - Manage training instability per §14.6 "Training instability (loss spikes/NaNs)"
    - Handle bad data per §14.7 "Bad data (gaps/dupes/tz drift)"
    - Ensure model persistence per §14.10 "Save/Load & version drift"
    - _Reference: §14.1-§14.7, §14.10 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 14. Implement quality gates and acceptance criteria (from §12 Acceptance criteria & quality gates)
  - [ ] 14.1 Create acceptance testing and validation system
    - Implement hard pass/fail gates per §12.1 "Hard pass/fail gates (per horizon h)"
    - Add rollback criteria per §12.2 "Rollback criteria (live)"
    - Generate acceptance report per §12.3 "Acceptance report (one command)"
    - Store required artifacts per §12.4 "What exactly to store"
    - Implement remediation guidance per §12.5 "If it fails — smallest hammer first"
    - Validate leakage checks per §0.8 "Non-negotiable guardrails (checklist to enforce in CI)"
    - _Reference: §12.1, §12.2, §12.3, §12.4, §12.5, §0.8 in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_

- [ ] 15. Final integration and testing (from §13 Implementation checklist)
  - [ ] 15.1 Perform end-to-end integration testing per Phase 11 "Acceptance & promotion"
    - Execute complete pipeline following exact phases from §13 "Implementation checklist"
    - Validate Definition of Done per §13 "Definition of Done (v1)"
    - Test command quick sheet per §13 "Command quick sheet (copy/paste)"
    - Follow parallelization guide per §13 "Parallelization guide (practical)"
    - Ensure all Phase 0-11 deliverables are complete and functional
    - Validate reproducible CV artifacts, saved winners, 15-min inference loop with 80/90/95 PIs
    - Confirm acceptance reports show ACCEPT for ≥2 horizons, monitoring within ±3pp coverage
    - _Reference: §13 Implementation checklist, Definition of Done in /Users/mac-main/Neural-Forecast/docs/forecasting_sf_plan.md_
