# Requirements Document

## Introduction

The Cross-Validation and Metrics spec implements the NeuralForecast-native cross-validation system and metrics computation framework specified in `docs/forecasting_sf_plan.md` Section 5 (lines 1583-1825). This component provides a lean, NF-native approach to model evaluation through sliding-origin cross-validation with proper windowing (Section 5.1, lines 1605-1635), comprehensive metrics computation centered on sCRPS (Section 5.2.B, lines 1655-1660), and calibration diagnostics including coverage and PIT analysis (Section 5.2.C, lines 1661-1665). The implementation follows the "zero reinvention" principle, using NeuralForecast's built-in cross_validation method exclusively (lines 1585, 1620-1628) while adding only thin glue code for metrics aggregation and diagnostic visualization.

## Requirements

### Requirement 1: Implement NF-Native Cross-Validation with Proper Windowing

**User Story:** As a model evaluation system, I want to use NeuralForecast's native cross_validation method with horizon-specific windowing parameters, so that I can evaluate models through time-ordered backtesting without implementing custom loops or risking leakage.

#### Acceptance Criteria

1. WHEN implementing cross-validation THEN the system SHALL use NeuralForecast.cross_validation() exclusively as specified in Section 5.0 (docs/forecasting_sf_plan.md lines 1583-1585)
2. WHEN configuring n_windows THEN the system SHALL use 6 for pilot experiments and 10 for final runs as specified in Section 5.1 (lines 1589, 1609)
3. WHEN setting step_size THEN the system SHALL use step_size=h to ensure non-overlapping forecast horizons as specified (lines 1591, 1610)
4. WHEN setting val_size THEN the system SHALL use val_size=4*h for internal validation splits as specified (lines 1593, 1611)
5. WHEN setting refit parameter THEN the system SHALL use refit=1 (True) to retrain models for each window as specified (lines 1595, 1612)
6. WHEN requesting prediction intervals THEN the system SHALL pass level=[80, 90, 95] to cross_validation for interval outputs
7. WHEN handling long input_size (1024+) THEN the system SHALL rely on implicit embargo from NF's time-ordered windowing (lines 1599, 1615)
8. IF custom backtesting loops are implemented THEN the system SHALL fail validation to enforce NF-native approach

### Requirement 2: Implement sCRPS as Primary Metric with Supporting Metrics

**User Story:** As a probabilistic forecasting system, I want to compute scaled Continuous Ranked Probability Score (sCRPS) as the primary evaluation metric, so that I can properly score probabilistic forecasts and select models based on calibrated uncertainty estimates.

#### Acceptance Criteria

1. WHEN computing primary metric THEN the system SHALL use sCRPS from neuralforecast.losses.pytorch as specified in Section 5.2.B (lines 1655-1660)
2. WHEN evaluating distributional models THEN the system SHALL compute sCRPS directly from distribution outputs
3. WHEN evaluating quantile models THEN the system SHALL approximate sCRPS via weighted quantile losses
4. WHEN computing supporting metrics THEN the system SHALL calculate MAE and RMSE on mean/median predictions
5. WHEN aggregating metrics THEN the system SHALL compute per-window metrics and report mean/std across windows
6. WHEN ranking models THEN the system SHALL use mean sCRPS as the primary selection criterion (lower is better)
7. WHEN storing metrics THEN the system SHALL save detailed per-window results for analysis
8. IF custom sCRPS implementations are added THEN the system SHALL fail validation to use NF-native metric

### Requirement 3: Implement Coverage Diagnostics at Target Confidence Levels

**User Story:** As a calibration validation system, I want to compute empirical coverage at 80%, 90%, and 95% confidence levels, so that I can verify that prediction intervals achieve their nominal coverage rates within acceptable tolerances.

#### Acceptance Criteria

1. WHEN computing coverage THEN the system SHALL calculate empirical hit-rate of actuals within predicted intervals
2. WHEN evaluating 80% intervals THEN the system SHALL verify coverage is 80±2% as specified in Section 5.2.C (lines 1661-1665)
3. WHEN evaluating 90% intervals THEN the system SHALL verify coverage is 90±2% as specified
4. WHEN evaluating 95% intervals THEN the system SHALL verify coverage is 95±2% as specified
5. WHEN intervals are provided THEN the system SHALL expect columns like "Model-lo-80", "Model-hi-80" from NF
6. WHEN aggregating coverage THEN the system SHALL compute per-window coverage and report mean/std
7. WHEN coverage deviates from nominal THEN the system SHALL flag miscalibration in diagnostic reports
8. IF coverage targets are not met THEN the system SHALL recommend conformal prediction adjustment

### Requirement 4: Implement PIT Analysis for Distributional Calibration

**User Story:** As a calibration diagnostic system, I want to compute and visualize Probability Integral Transform (PIT) histograms, so that I can assess whether distributional forecasts are properly calibrated through uniformity testing.

#### Acceptance Criteria

1. WHEN computing PIT for distributional models THEN the system SHALL use CDF evaluation at actual values
2. WHEN computing PIT for quantile models THEN the system SHALL use interpolated quantile rank approximation
3. WHEN requesting dense quantiles THEN the system SHALL use level grid 1-99 for PIT approximation
4. WHEN generating PIT histograms THEN the system SHALL create 20-bin density plots as specified (lines 487, 1664)
5. WHEN PIT values are uniform THEN the system SHALL indicate proper calibration
6. WHEN using conformal prediction THEN the system SHALL skip insample PIT (not available) and rely on CV coverage (lines 1632, 1642, 1664)
7. WHEN saving PIT diagnostics THEN the system SHALL store plots in reports/h{horizon}/pit_histogram.png
8. IF PIT deviates from uniform THEN the system SHALL document calibration issues in diagnostic summary

### Requirement 5: Implement Conformal Prediction Integration

**User Story:** As an uncertainty quantification system, I want to integrate NeuralForecast's PredictionIntervals utility for conformal prediction, so that I can improve calibration through distribution-free interval construction when needed.

#### Acceptance Criteria

1. WHEN using conformal prediction THEN the system SHALL use NF's PredictionIntervals class as specified in Section 5.2.H (lines 1814-1817)
2. WHEN passing to fit/cross_validation THEN the system SHALL support prediction_intervals parameter
3. WHEN configuring conformal windows THEN the system SHALL use appropriate n_windows for calibration set
4. WHEN conformal is enabled THEN the system SHALL not expect insample prediction intervals
5. WHEN evaluating conformal coverage THEN the system SHALL rely on CV/test coverage statistics only
6. WHEN combining with base models THEN the system SHALL apply conformal as post-processing step
7. WHEN documenting conformal usage THEN the system SHALL note it's optional for calibration improvement
8. IF custom conformal methods are implemented THEN the system SHALL fail validation to use NF-native

### Requirement 6: Implement Leakage Prevention Discipline

**User Story:** As a validation system, I want to enforce strict leakage prevention in cross-validation setup, so that I ensure no future information contaminates training data and maintain realistic backtest conditions.

#### Acceptance Criteria

1. WHEN setting up CV windows THEN the system SHALL use NF's strictly time-ordered windowing
2. WHEN using step_size=h THEN the system SHALL ensure no overlap between forecast horizons
3. WHEN leveraging input_size THEN the system SHALL rely on implicit embargo from long history requirements
4. WHEN validating data splits THEN the system SHALL verify train < validation < test ordering (lines 1806-1812)
5. WHEN checking features THEN the system SHALL call assert_shifted() to verify shift(1) was applied (lines 380, 1807)
6. WHEN preventing forward fill THEN the system SHALL call assert_no_forward_fill_y() on target (lines 705, 721)
7. WHEN documenting leakage THEN the system SHALL explicitly note prevention measures in reports
8. IF leakage is detected THEN the system SHALL immediately fail validation with detailed error

### Requirement 7: Implement Metrics Aggregation and Leaderboard Generation

**User Story:** As a model selection system, I want to aggregate cross-validation metrics and generate ranked leaderboards, so that I can systematically compare models and select the best performers for each horizon.

#### Acceptance Criteria

1. WHEN aggregating metrics THEN the system SHALL compute mean, std, min, max across CV windows
2. WHEN creating leaderboard THEN the system SHALL rank models by mean sCRPS (ascending)
3. WHEN including metrics THEN the system SHALL show sCRPS, MAE, RMSE, coverage at 80/90/95
4. WHEN formatting output THEN the system SHALL create DataFrame with model names as index
5. WHEN saving leaderboard THEN the system SHALL write to experiments/h{horizon}/leaderboard.csv
6. WHEN selecting top models THEN the system SHALL identify best StudentT and best quantile model
7. WHEN reporting confidence THEN the system SHALL include standard errors for metric estimates
8. IF models tie on sCRPS THEN the system SHALL use coverage deviation as tiebreaker

### Requirement 8: Implement Model Persistence and Artifact Management

**User Story:** As a model lifecycle system, I want to save cross-validation results and trained models with proper versioning, so that I can reproduce results, analyze failures, and deploy selected models to production.

#### Acceptance Criteria

1. WHEN saving CV results THEN the system SHALL write raw predictions to experiments/h{horizon}/cv_results.parquet (lines 1799)
2. WHEN saving models THEN the system SHALL use NF's native save() method to experiments/h{horizon}/models/ (lines 1820-1822)
3. WHEN naming artifacts THEN the system SHALL include timestamp in format YYYYMMDDTHHMMSSZ (lines 411-415)
4. WHEN storing metrics THEN the system SHALL save detailed metrics to experiments/h{horizon}/metrics.json
5. WHEN saving diagnostics THEN the system SHALL write plots to reports/h{horizon}/ directory (lines 1646)
6. WHEN loading models THEN the system SHALL use NF's native load() method for inference (lines 569, 1822)
7. WHEN versioning models THEN the system SHALL follow tagging rules from docs/versioning_system.md (separate versioning doc)
8. IF save/load fails THEN the system SHALL provide detailed error with recovery instructions

### Requirement 9: Implement Integration with Training Pipeline

**User Story:** As a training orchestration system, I want to seamlessly integrate cross-validation with the model training workflow, so that I can evaluate models as part of the standard training process and make data-driven model selection decisions.

#### Acceptance Criteria

1. WHEN calling from run_train.py THEN the system SHALL accept NeuralForecast instance and configuration
2. WHEN receiving data THEN the system SHALL expect canonical frame with unique_id, ds, y, and exog columns
3. WHEN configuring CV THEN the system SHALL read parameters from experiments/h{horizon}.yaml (lines 530, 1622-1628)
4. WHEN executing CV THEN the system SHALL call run_cv() function from cv/runner.py module
5. WHEN computing metrics THEN the system SHALL call summarize_cv() for aggregation and ranking
6. WHEN handling errors THEN the system SHALL provide informative messages with debugging hints
7. WHEN logging progress THEN the system SHALL report window completion and time remaining
8. IF CV fails THEN the system SHALL save partial results before raising exception

### Requirement 10: Implement Diagnostic Visualization Suite

**User Story:** As a model diagnostics system, I want to generate comprehensive visualization plots for cross-validation results, so that I can visually assess model performance, calibration quality, and identify potential issues.

#### Acceptance Criteria

1. WHEN plotting predictions THEN the system SHALL show actuals vs forecasts with prediction intervals
2. WHEN plotting residuals THEN the system SHALL create QQ plots and residual distributions
3. WHEN plotting coverage THEN the system SHALL show nominal vs empirical coverage by confidence level
4. WHEN plotting PIT THEN the system SHALL create uniformity histograms with KS test statistics
5. WHEN plotting metrics THEN the system SHALL show per-window performance with confidence bands
6. WHEN saving plots THEN the system SHALL use high-resolution PNG format in reports directory
7. WHEN creating summaries THEN the system SHALL generate HTML reports with embedded visualizations
8. IF matplotlib is unavailable THEN the system SHALL gracefully skip plots with warning message