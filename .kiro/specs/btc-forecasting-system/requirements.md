# Requirements Document

## Introduction

This document outlines the requirements for implementing a comprehensive BTC intraday forecasting system using NeuralForecast (NF) library. The system will generate probabilistic forecasts for Bitcoin with calibrated prediction intervals at multiple horizons (1h, 2h, 4h, 8h) using 15-minute base frequency data. The implementation prioritizes data discipline, leakage prevention, and NF-native approaches for training, evaluation, and deployment.

## Requirements

### Requirement 1: Data Infrastructure and Validation

**User Story:** As a quantitative researcher, I want a robust data pipeline that ensures data quality and prevents leakage, so that I can trust the forecasting results and avoid false signals.

#### Acceptance Criteria

1. WHEN processing OHLCV data THEN the system SHALL enforce UTC end-of-bar timestamps on exact 15-minute grid boundaries
2. WHEN validating data integrity THEN the system SHALL implement assert_regular_grid, assert_utc_eob, assert_no_forward_fill_y, and assert_shifted functions
3. WHEN handling missing bars THEN the system SHALL drop missing target values and never forward-fill the target variable, validated by assert_no_forward_fill_y
4. WHEN processing derived features THEN the system SHALL apply strict compute → shift(1) rule to prevent lookahead bias
5. WHEN computing target variable THEN the system SHALL use log returns formula: y_t = log(close_t) - log(close_{t-1})
6. WHEN handling outliers THEN the system SHALL winsorize returns at [0.1%, 99.9%] for training only, never mutating reporting layer
7. WHEN creating NF canonical frame THEN the system SHALL use long format with ["unique_id","ds","y", <exog>] where unique_id="BTC-USD"
8. WHEN detecting data anomalies THEN the system SHALL halt processing and log specific validation failures
9. WHEN ensuring reproducibility THEN the system SHALL use deterministic seeds at NF/model level

### Requirement 2: Feature Engineering and Selection

**User Story:** As a machine learning engineer, I want a systematic feature engineering pipeline with leakage protection, so that I can create predictive features while maintaining temporal causality.

#### Acceptance Criteria

1. WHEN building technical indicators THEN the system SHALL use vectorbt and TA-Lib as primary libraries with pandas-ta as supplement
2. WHEN creating multi-timeframe features THEN the system SHALL aggregate from 15min to 30min/1h/4h using proper EOB alignment with label='right', closed='right'
3. WHEN applying feature selection THEN the system SHALL enforce a hard cap of ≤256 features to prevent overfitting
4. WHEN filtering features THEN the system SHALL require availability ≥98% and drop features with insufficient data coverage
5. WHEN processing historical exogenous features THEN the system SHALL shift all features by one bar before joining with target using postprocess_shift_and_prune
6. WHEN handling future-known features THEN the system SHALL properly categorize calendar features (minute_of_day, day_of_week, is_weekend) as futr_exog_list for NF models
7. WHEN validating feature quality THEN the system SHOULD include an optional diagnostic: hist_exog shows corr(exog_t, y_t) < corr(exog_t, y_{t+1}); hard gates remain the assert_* validators

### Requirement 3: Model Training and Cross-Validation

**User Story:** As a forecasting practitioner, I want NF-native model training with proper cross-validation, so that I can evaluate model performance without custom backtesting code.

#### Acceptance Criteria

1. WHEN training models THEN the system SHALL use NeuralForecast's cross_validation method exclusively with no custom backtesters
2. WHEN configuring cross-validation THEN the system SHALL use n_windows=6 for pilot, step_size=h, val_size=4*h, refit=True
3. WHEN training multiple horizons THEN the system SHALL maintain separate model portfolios for h=[4,8,16,32] steps (1h,2h,4h,8h)
4. WHEN using context windows THEN the system SHALL set input_size=1024 for NHITS/NBEATSx/TiDE and 2048 for PatchTST
5. WHEN evaluating models THEN the system SHALL use sCRPS as primary metric with MAE/RMSE as supporting metrics
6. WHEN implementing models THEN the system SHALL use exactly NHITS, NBEATSx, TiDE, and PatchTST from NeuralForecast
7. WHEN configuring scalers THEN the system SHALL use scaler_type="robust" as default and revin=True for PatchTST
8. WHEN setting training parameters THEN the system SHALL use batch_size=512, learning_rate=1e-3, max_steps=20000, early_stop_patience_steps=400
9. WHEN using loss functions THEN the system SHALL use DistributionLoss("StudentT") as default or MQLoss for direct quantile estimation
10. WHEN persisting models THEN the system SHALL use only NF's native save/load with save_dataset=True, never custom pickle

### Requirement 4: Probabilistic Forecasting and Uncertainty Quantification

**User Story:** As a risk manager, I want calibrated prediction intervals at multiple confidence levels, so that I can quantify forecast uncertainty and make informed decisions.

#### Acceptance Criteria

1. WHEN training models THEN the system SHALL use DistributionLoss('StudentT') as default with MQLoss/ISQF/IQLoss as alternatives
2. WHEN generating predictions THEN the system SHALL produce prediction intervals at 80%, 90%, and 95% confidence levels using predict(level=[80,90,95])
3. WHEN calibrating intervals THEN the system SHALL use NF's PredictionIntervals conformal prediction if empirical coverage deviates >±2%
4. WHEN computing diagnostics THEN the system SHALL generate PIT histograms via predict_insample and coverage statistics by volatility decile
5. WHEN detecting miscalibration THEN the system SHALL trigger conformal adjustment before deployment
6. WHEN using conformal prediction THEN the system SHALL implement PredictionIntervals(n_windows=cfg["n_windows"], level=[80,90,95]) for CV-based conformal
7. WHEN evaluating probabilistic performance THEN the system SHALL compute dense quantiles [0.01,...,0.99] for sCRPS calculation on finalists only
8. WHEN persisting artifacts THEN the system SHALL store CV outputs and diagnostics under stable paths per plan: experiments/h{h}/ and reports/h{h}/

### Requirement 5: Model Selection and Ensembling

**User Story:** As a model developer, I want systematic model selection and simple ensembling capabilities, so that I can combine the best performing models for each horizon.

#### Acceptance Criteria

1. WHEN selecting models THEN the system SHALL rank by mean sCRPS across CV windows per horizon
2. WHEN creating ensembles THEN the system SHALL use simple equal-weight averaging of top-2 models only
3. WHEN blending predictions THEN the system SHALL average point forecasts and quantiles separately, never distribution parameters
4. WHEN promoting models THEN the system SHALL require sCRPS improvement ≥1% over baseline for selection
5. WHEN saving ensembles THEN the system SHALL persist individual models and blending metadata separately
6. WHEN storing ensemble metadata THEN the system SHALL save a minimal JSON (model_names, weights, horizon, timestamp) under experiments/h{h}/best/ensemble.json

### Requirement 6: Hyperparameter Optimization

**User Story:** As a machine learning engineer, I want bounded hyperparameter search with efficient promotion criteria, so that I can optimize models without excessive compute overhead.

#### Acceptance Criteria

1. WHEN defining search spaces THEN the system SHALL use tight, model-specific bounds to prevent unrealistic configurations
2. WHEN running pilot studies THEN the system SHALL use 6 CV windows for initial screening
3. WHEN promoting candidates THEN the system SHALL require sCRPS improvement ≥0.5% to advance to full 10-window CV
4. WHEN managing compute THEN the system SHALL implement early stopping and resource limits per trial
5. WHEN tracking experiments THEN the system SHALL log all hyperparameters and metrics for reproducibility

### Requirement 7: Training and Evaluation Workflow

**User Story:** As a data scientist, I want YAML-driven experiment configuration with automated training workflows, so that I can run reproducible experiments with minimal manual intervention.

#### Acceptance Criteria

1. WHEN configuring experiments THEN the system SHALL use YAML files per horizon (h4.yaml, h8.yaml, h16.yaml, h32.yaml) plus defaults.yaml
2. WHEN structuring YAML configs THEN the system SHALL include required keys: h, n_windows, step_size, val_size, refit, models list with loss specifications
3. WHEN running training THEN the system SHALL execute via run_train.py with NF-native fit and cross_validation
4. WHEN generating diagnostics THEN the system SHALL produce insample predictions for PIT analysis and coverage validation
5. WHEN invoking training THEN the system SHALL call assert_regular_grid, assert_utc_eob, assert_no_forward_fill_y, and assert_shifted prior to NF.fit/cross_validation
6. WHEN saving artifacts THEN the system SHALL use NF's native save/load with save_dataset=True and overwrite=True
7. WHEN organizing outputs THEN the system SHALL maintain consistent directory structure under experiments/h{h}/ with cv_results.parquet, metrics.csv, best/
8. WHEN implementing run_train.py THEN the system SHALL orchestrate: load canonical frame → build features → instantiate NF models → cross_validation (with PredictionIntervals, level=[80,90,95]) → metrics/plots → optional final fit → save artifacts

### Requirement 8: Inference and Deployment

**User Story:** As a production engineer, I want reliable inference capabilities with live deployment support, so that I can generate forecasts every 15 minutes in production.

#### Acceptance Criteria

1. WHEN running inference THEN the system SHALL execute via run_predict.py in both one-shot and loop modes
2. WHEN building prediction features THEN the system SHALL construct tail features using the same pipeline as training
3. WHEN generating forecasts THEN the system SHALL produce predictions with calibrated intervals at specified confidence levels
4. WHEN operating in live mode THEN the system SHALL include 45+ second buffer after EOB to ensure bar finalization
5. WHEN handling failures THEN the system SHALL implement graceful degradation and skip cycles for incomplete data

### Requirement 9: Monitoring and Maintenance

**User Story:** As a system operator, I want automated monitoring and retraining capabilities, so that I can maintain model performance and detect drift in production.

#### Acceptance Criteria

1. WHEN monitoring performance THEN the system SHALL track rolling 7-day empirical coverage and sCRPS on sliding windows
2. WHEN detecting coverage drift THEN the system SHALL trigger alerts when coverage deviates >±3pp from nominal levels at any of 80/90/95
3. WHEN detecting score decay THEN the system SHALL alert when 7-day mean sCRPS degrades >3% vs trailing 30-day baseline
4. WHEN detecting distribution shift THEN the system SHALL monitor PSI (Population Stability Index) with thresholds: 0.2 moderate (3 consecutive days), 0.3 major (any day)
5. WHEN retraining models THEN the system SHALL follow versioned procedures with rollback capabilities
6. WHEN managing versions THEN the system SHALL maintain version_manifest.json and requirements-lock.txt snapshots
7. WHEN validating updates THEN the system SHALL run smoke tests and mini-CV before promoting new models

### Requirement 10: Repository Structure and Organization

**User Story:** As a developer, I want a standardized repository structure with clear module responsibilities, so that I can maintain and extend the codebase efficiently.

#### Acceptance Criteria

1. WHEN setting up the repository THEN the system SHALL create exactly these directories: data, features, nf_models, cv, experiments, uq, reports, utils
2. WHEN implementing utils/io.py THEN the system SHALL provide regularize_to_grid_utc, make_nf_canonical, drop_train_nans_and_winsorize, load_canonical_frame, save_parquet, timestamped_path functions (120 LOC cap)
3. WHEN implementing utils/validate.py THEN the system SHALL provide assert_utc_eob, assert_regular_grid, assert_no_forward_fill_y, assert_shifted functions (150 LOC cap)
4. WHEN implementing features/builder.py THEN the system SHALL provide build_indicators, apply_mtf, postprocess_shift_and_prune, select_features functions (300 LOC cap)
5. WHEN implementing features/registry.py THEN the system SHALL provide declarative registry of indicators & MTF specs with names, params, kind=hist|futr|stat, tf=15min|30min|1h|4h (150 LOC cap)
6. WHEN implementing nf_models/factory.py THEN the system SHALL provide instantiate_models function returning configured NF models with proper exog wiring (200 LOC cap)
7. WHEN implementing cv/runner.py THEN the system SHALL provide run_cv function using NF-native cross_validation only (120 LOC cap)
8. WHEN implementing uq/diag.py THEN the system SHALL provide compute_coverage, plot_pit, coverage_by_vol_decile, blend_equal functions (250 LOC cap)
9. WHEN naming artifacts THEN the system SHALL use experiments/h{h}/ directories with cv_results.parquet, metrics.csv, and best/ model saves
10. WHEN implementing modules THEN the system SHALL enforce snake_case for files and PascalCase only for class names
11. WHEN creating entry points THEN the system SHALL implement run_train.py (200 LOC) and run_predict.py (150 LOC) with settings.yaml configuration

### Requirement 11: Implementation Phases and Deliverables

**User Story:** As a project manager, I want a structured implementation approach with clear phases and deliverables, so that I can track progress and ensure systematic development.

#### Acceptance Criteria

1. WHEN implementing Phase 0 THEN the system SHALL create exact directory structure: data/, features/, nf_models/, cv/, uq/, reports/, utils/, experiments/
2. WHEN implementing Phase 1 THEN the system SHALL deliver data contracts with regularize_to_grid_utc, make_nf_canonical, validators in utils/validate.py and utils/io.py
3. WHEN implementing Phase 2 THEN the system SHALL deliver feature registry, MTF builders, postprocess_shift_and_prune with ≤256 feature cap
4. WHEN implementing Phase 3 THEN the system SHALL deliver NF model factory supporting NHITS/NBEATSx/TiDE/PatchTST with proper exog wiring
5. WHEN implementing Phase 4 THEN the system SHALL deliver YAML experiment configs for all horizons h=[4,8,16,32]
6. WHEN implementing Phase 5 THEN the system SHALL deliver run_train.py with NF-native cross_validation and insample diagnostics
7. WHEN implementing Phase 6 THEN the system SHALL deliver pilot→promote→full CV workflow with n_windows progression
8. WHEN implementing Phase 7 THEN the system SHALL deliver model selection and simple ensembling with top-2 blending
9. WHEN implementing Phase 8 THEN the system SHALL deliver uncertainty quantification with PIT analysis and conformal calibration
10. WHEN implementing Phase 9 THEN the system SHALL deliver run_predict.py with single-shot and live loop modes
11. WHEN implementing Phase 10 THEN the system SHALL deliver monitoring and maintenance with rolling coverage tracking
12. WHEN implementing Phase 11 THEN the system SHALL deliver acceptance reports and promotion procedures

### Requirement 12: Risk Mitigation and Error Handling

**User Story:** As a system operator, I want comprehensive risk mitigation and error handling, so that I can prevent common failure modes and maintain system reliability.

#### Acceptance Criteria

1. WHEN preventing MTF misalignment THEN the system SHALL use EOB-aligned resampling with central shift(1) and unit tests
2. WHEN preventing leakage THEN the system SHALL enforce assert_shifted validation and restrict calendar features to futr_exog_list
3. WHEN handling target computation THEN the system SHALL compute log returns after regularization and never forward-fill y
4. WHEN managing quantile crossing THEN the system SHALL use ISQF/IQLoss or attach conformal prediction for monotonic constraints
5. WHEN handling GPU OOM THEN the system SHALL implement batch size reduction and graceful model fallback strategies
6. WHEN detecting training instability THEN the system SHALL implement learning rate adjustment and winsorization safeguards
7. WHEN handling bad data THEN the system SHALL implement grid validation, skip cycles for missing bars, and maintain provenance logs
8. WHEN ensuring model persistence THEN the system SHALL use only NF native save/load with version manifest tracking

### Requirement 13: Coding Standards and Implementation Details

**User Story:** As a developer, I want strict coding standards and implementation guidelines, so that I can build a maintainable and consistent codebase.

#### Acceptance Criteria

1. WHEN importing NF components THEN the system SHALL use: from neuralforecast import NeuralForecast; from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST; from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, ISQF, IQLoss
2. WHEN implementing model interfaces THEN the system SHALL wire h, input_size, scaler_type, loss, and exog lists (hist_exog_list, futr_exog_list, stat_exog_list) from config
3. WHEN calling cross-validation THEN the system SHALL always use nf.cross_validation(df=..., n_windows=..., step_size=..., refit=True, val_size=...)
4. WHEN implementing insample diagnostics THEN the system SHALL call nf.predict_insample(step_size=h, level=[80,90,95]) after fit
5. WHEN persisting models THEN the system SHALL use nf.save(path, save_dataset=True, overwrite=True) and NeuralForecast.load(path)
6. WHEN creating bootstrap stubs THEN the system SHALL implement exact function signatures as specified in the plan's bootstrap section
7. WHEN enforcing hard bans THEN the system SHALL never implement custom backtesters or custom model persistence outside NF's native methods

### Requirement 14: Quality Gates and Acceptance Criteria

**User Story:** As a project stakeholder, I want clear pass/fail criteria for model acceptance, so that I can ensure quality standards before deployment.

#### Acceptance Criteria

1. WHEN evaluating model quality THEN the system SHALL enforce sCRPS thresholds per horizon for acceptance
2. WHEN checking calibration THEN the system SHALL require empirical coverage within ±2% of nominal levels
3. WHEN validating stability THEN the system SHALL ensure consistent performance across CV windows
4. WHEN generating reports THEN the system SHALL produce comprehensive acceptance reports with all key metrics
5. WHEN failing quality gates THEN the system SHALL provide clear guidance on remediation steps
6. WHEN enforcing guardrails THEN the system SHOULD include the correlation-based leakage diagnostic (optional, non-blocking); blocking gates are assert_regular_grid, assert_utc_eob, assert_no_forward_fill_y, assert_shifted
7. WHEN achieving definition of done THEN the system SHALL deliver reproducible CV artifacts, saved winners, 15-min inference loop with 80/90/95 PIs, acceptance reports, and monitoring within ±3pp coverage