# Implementation Plan

## Core CV Runner Implementation

- [ ] 1. Set up CV runner module structure
  - Create cv/runner.py with NeuralForecast imports
  - Import pandas, numpy, and typing annotations
  - Add imports: from neuralforecast.losses.pytorch import sCRPS
  - Create __init__.py for module organization
  - Add logging configuration for progress tracking
  - **Deliverables**: cv/runner.py module with proper imports
  - **Dependencies**: None (foundational task)
  - **Success Criteria**: Module imports successfully, all NF classes available
  - _Requirements: 1.1, 1.8, 9.1_

- [ ] 2. Implement core run_cv function
  - Write run_cv function accepting NeuralForecast instance, DataFrame, and config (lines 1681-1691)
  - Extract CV parameters: n_windows, step_size, val_size from config (lines 1622-1628)
  - Set refit=1 for proper window retraining (lines 1595, 1612, 1625)
  - Pass level=[80, 90, 95] for prediction intervals (lines 1613, 1626, 1689)
  - Handle both pilot (n_windows=6) and final (n_windows=10) configurations (lines 1589, 1609)
  - **Deliverables**: Functional run_cv that calls NF.cross_validation
  - **Dependencies**: Task 1
  - **Success Criteria**: Successfully executes NF cross-validation with correct parameters
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 1.6_

- [ ] 3. Implement CV results validation
  - Add validation for CV output DataFrame structure
  - Verify presence of unique_id, ds, cutoff, y columns
  - Check for model prediction columns
  - Validate interval columns when level is specified
  - Add error handling for incomplete CV runs
  - **Deliverables**: Validation functions for CV results
  - **Dependencies**: Task 2
  - **Success Criteria**: Detects and reports malformed CV outputs
  - _Requirements: 1.7, 6.1, 6.4_

## Metrics Computation Implementation

- [ ] 4. Implement sCRPS computation wrapper
  - Create compute_scrps function using NF's native sCRPS (lines 1655-1660, 1678)
  - Support both distributional and quantile model outputs (lines 1638, 1657-1659)
  - Handle vectorized computation for efficiency
  - Add proper error handling for numerical issues
  - Document sCRPS interpretation (lower is better) (lines 1657, 1659)
  - **Deliverables**: sCRPS computation function in uq/metrics.py
  - **Dependencies**: Task 1
  - **Success Criteria**: Correctly computes sCRPS matching NF's implementation
  - _Requirements: 2.1, 2.2, 2.3, 2.6_

- [ ] 5. Implement supporting metrics computation
  - Add MAE calculation on point predictions
  - Add RMSE calculation for error magnitude
  - Add bias computation for systematic error detection
  - Support both mean and median as point estimates
  - Implement vectorized operations for performance
  - **Deliverables**: Supporting metrics functions
  - **Dependencies**: Task 4
  - **Success Criteria**: Metrics match sklearn/numpy implementations
  - _Requirements: 2.4, 2.5_

- [ ] 6. Implement metrics aggregation across windows
  - Create aggregate_metrics function for window statistics
  - Compute mean, std, min, max for each metric
  - Handle missing values gracefully
  - Support per-model aggregation
  - Add confidence intervals for mean estimates
  - **Deliverables**: Metrics aggregation logic
  - **Dependencies**: Tasks 4, 5
  - **Success Criteria**: Produces summary statistics table
  - _Requirements: 2.5, 7.1, 7.7_

## Coverage and Calibration Implementation

- [ ] 7. Implement coverage computation
  - Create compute_coverage function for interval evaluation (lines 472-481, 1717-1726)
  - Support 80%, 90%, 95% confidence levels (lines 1661-1665)
  - Calculate empirical hit rates from CV results (lines 1663, 1723-1724)
  - Compare against nominal rates with ±2% tolerance (lines 1663)
  - Flag miscalibration when outside tolerance
  - **Deliverables**: Coverage analysis function in uq/calibration.py
  - **Dependencies**: Task 2
  - **Success Criteria**: Coverage matches manual calculation
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.6, 3.7_

- [ ] 8. Implement PIT computation for distributional models
  - Create compute_pit function for calibration assessment (lines 1743-1766)
  - Support CDF evaluation for distributional outputs (lines 1642, 1664)
  - Implement quantile rank interpolation for quantile models (lines 1664, 1747-1763)
  - Use dense quantile grid: quantiles = [i/100 for i in range(1, 100)]
  - Handle dense quantile grids (1-99) for approximation (lines 1664, 1750)
  - Skip PIT for conformal models (document reason) (lines 1632, 1642, 1664)
  - **Deliverables**: PIT computation logic
  - **Dependencies**: Task 2
  - **Success Criteria**: PIT values uniformly distributed for calibrated models
  - _Requirements: 4.1, 4.2, 4.3, 4.6_

- [ ] 9. Implement calibration diagnostics visualization
  - Create PIT histogram plotting with 20 bins (lines 483-490, 1664)
  - Add KS test for uniformity assessment
  - Create coverage reliability diagrams
  - Plot nominal vs empirical coverage (lines 1663)
  - Save plots to reports/h{horizon}/ directory (lines 1646)
  - **Deliverables**: Diagnostic plotting functions
  - **Dependencies**: Tasks 7, 8
  - **Success Criteria**: Generates interpretable diagnostic plots
  - _Requirements: 4.4, 4.7, 10.3, 10.4_

## Leaderboard and Ranking Implementation

- [ ] 10. Implement summarize_cv main function
  - Create summarize_cv accepting CV DataFrame and model list
  - Orchestrate metrics computation for all models
  - Call aggregation functions for summary statistics
  - Generate coverage analysis for all models
  - Return dictionary with metrics, leaderboard, coverage DataFrames
  - **Deliverables**: Main summarization function
  - **Dependencies**: Tasks 4, 5, 6, 7
  - **Success Criteria**: Produces complete evaluation summary
  - _Requirements: 7.1, 7.2, 9.5_

- [ ] 11. Implement leaderboard generation
  - Create ranking logic based on mean sCRPS
  - Include all metrics in leaderboard table
  - Add model metadata (loss type, parameters)
  - Implement tiebreaker using coverage deviation
  - Format as DataFrame with model names as index
  - **Deliverables**: Leaderboard generation function
  - **Dependencies**: Task 10
  - **Success Criteria**: Produces correctly ranked model comparison
  - _Requirements: 7.2, 7.3, 7.4, 7.6, 7.8_

- [ ] 12. Implement best model selection
  - Identify best distributional model (StudentT)
  - Identify best quantile model (MQLoss/IQLoss)
  - Document selection rationale
  - Create model selection summary
  - Support configurable selection criteria
  - **Deliverables**: Model selection logic
  - **Dependencies**: Task 11
  - **Success Criteria**: Correctly identifies top performers
  - _Requirements: 7.6, 2.6_

## Conformal Prediction Integration

- [ ] 13. Implement conformal prediction support
  - Add PredictionIntervals configuration support (lines 1634, 1814-1817)
  - Integrate with NF's fit and cross_validation calls (lines 1634-1635)
  - Handle conformal window configuration
  - Document insample PI limitations (lines 1632, 1642)
  - Implement conformal coverage validation (lines 1816)
  - **Deliverables**: Conformal prediction integration
  - **Dependencies**: Task 2
  - **Success Criteria**: Successfully applies conformal methods when configured
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5, 5.6_

## Artifact Management Implementation

- [ ] 14. Implement results persistence
  - Save raw CV predictions to parquet format (lines 1799)
  - Store metrics summary as JSON
  - Save leaderboard as CSV (lines 1800)
  - Implement timestamped naming convention (lines 411-415)
  - Create directory structure if missing (lines 407-409, 414)
  - **Deliverables**: Persistence functions in utils/io.py
  - **Dependencies**: Tasks 10, 11
  - **Success Criteria**: All artifacts saved with proper versioning
  - _Requirements: 8.1, 8.3, 8.4, 8.5, 8.7_

- [ ] 15. Implement model saving integration
  - Use NF's native save() method for models (lines 1820-1822)
  - Save to experiments/h{horizon}/models/ directory
  - Include model metadata in filenames
  - Support selective saving of best models
  - Document model artifact structure
  - **Deliverables**: Model persistence logic
  - **Dependencies**: Task 11
  - **Success Criteria**: Models saved and loadable via NF.load()
  - _Requirements: 8.2, 8.6, 8.7_

## Integration and Testing

- [ ] 16. Implement integration with run_train.py
  - Create CV execution block in training script (lines 1770-1801)
  - Wire configuration loading from YAML (lines 530, 1622-1628)
  - Add progress logging during CV
  - Implement error recovery for partial failures
  - Add timing instrumentation
  - **Deliverables**: Integration code in run_train.py
  - **Dependencies**: Tasks 2, 10, 14
  - **Success Criteria**: Seamless CV execution from training pipeline
  - _Requirements: 9.1, 9.3, 9.4, 9.5, 9.7, 9.8_

- [ ] 17. Create comprehensive unit tests
  - Test sCRPS computation accuracy
  - Test coverage calculation correctness
  - Test PIT uniformity for known distributions
  - Test metrics aggregation logic
  - Test leaderboard ranking algorithm
  - **Deliverables**: Test suite in tests/test_cv.py
  - **Dependencies**: All implementation tasks
  - **Success Criteria**: >90% code coverage, all tests passing
  - _Requirements: All validation criteria_

- [ ] 18. Create integration tests
  - Test end-to-end CV execution with toy data
  - Test artifact persistence and loading
  - Test error handling and recovery
  - Test configuration validation
  - Benchmark performance on realistic data
  - **Deliverables**: Integration tests in tests/test_cv_integration.py
  - **Dependencies**: Task 17
  - **Success Criteria**: Full pipeline executes without errors
  - _Requirements: 9.6, 9.8_

## Documentation and Examples

- [ ] 19. Create usage documentation
  - Write comprehensive docstrings for all functions
  - Create README for cv/ module
  - Add configuration examples for each horizon
  - Document metrics interpretation
  - Create troubleshooting guide
  - **Deliverables**: Documentation in cv/README.md
  - **Dependencies**: All implementation tasks
  - **Success Criteria**: Clear, complete documentation
  - _Requirements: 10.7, 10.8_

- [ ] 20. Create example notebooks
  - Build CV execution example notebook
  - Create metrics analysis notebook
  - Add calibration diagnostics notebook
  - Include model selection workflow
  - Add performance tuning guide
  - **Deliverables**: Jupyter notebooks in examples/cv/
  - **Dependencies**: Task 19
  - **Success Criteria**: Runnable examples demonstrating all features
  - _Requirements: Documentation and usability_