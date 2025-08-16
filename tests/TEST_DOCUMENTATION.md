# CV and Metrics Test Documentation

## Overview

This document describes the comprehensive test suites for the cross-validation and metrics modules of the Neural-Forecast BTC prediction system.

## Test Structure

### Unit Tests (`test_cv.py`)
Comprehensive unit tests covering individual functions and classes with >90% coverage target.

#### Test Classes

1. **TestMetricsComputation**
   - `test_compute_mae_basic`: MAE computation with perfect and known error cases
   - `test_compute_mae_with_nan`: NaN handling in MAE computation
   - `test_compute_rmse_basic`: RMSE computation validation
   - `test_rmse_ge_mae`: Verify RMSE ≥ MAE mathematical property
   - `test_compute_bias_basic`: Bias computation for over/underestimation
   - `test_compute_scrps_point_predictions`: sCRPS with point forecasts
   - `test_compute_scrps_quantile_predictions`: sCRPS with quantile forecasts
   - `test_compute_scrps_distributional`: sCRPS with StudentT distribution

2. **TestCoverageAnalysis**
   - `test_compute_coverage_basic`: Basic coverage computation
   - `test_compute_coverage_multiple_models`: Multi-model coverage
   - `test_coverage_within_tolerance`: Verify ±2% tolerance requirement

3. **TestPITAnalysis**
   - `test_compute_pit_uniform`: PIT for uniform distribution
   - `test_compute_pit_distributional`: PIT for distributional models
   - `test_check_pit_uniformity`: KS test for uniformity

4. **TestMetricsAggregation**
   - `test_aggregate_metrics_mean`: Mean aggregation across windows
   - `test_aggregate_metrics_median`: Median aggregation (outlier robust)
   - `test_rank_models_by_scrps`: Model ranking by sCRPS

5. **TestCVRunner**
   - `test_validate_cv_results`: CV results validation
   - `test_compute_cv_metrics`: Metrics computation from CV
   - `test_summarize_cv`: CV summarization and leaderboard
   - `test_run_cv_with_conformal`: Conformal prediction intervals

6. **TestIOFunctions**
   - `test_save_cv_artifacts`: Artifact persistence
   - `test_load_cv_artifacts`: Artifact loading
   - `test_load_missing_artifacts`: Graceful handling of missing files

7. **TestErrorHandling**
   - `test_empty_dataframe_handling`: Empty input handling
   - `test_missing_columns_handling`: Missing column detection
   - `test_invalid_quantiles`: Invalid quantile validation
   - `test_mismatched_array_sizes`: Array size mismatch detection
   - `test_all_nan_handling`: All-NaN array handling

8. **TestPerformanceBenchmarks**
   - `test_metrics_computation_speed`: <100ms for 10K samples
   - `test_coverage_computation_speed`: <1s for 50K samples

### Integration Tests (`test_cv_integration.py`)
End-to-end tests with realistic data and complete workflows.

#### Test Classes

1. **TestEndToEndCVExecution**
   - `test_complete_cv_workflow`: Full pipeline from data to leaderboard
   - `test_cv_with_ensemble`: CV with ensemble creation
   - Fixtures: `realistic_data`, `models_config`

2. **TestArtifactPersistence**
   - `test_save_and_load_artifacts`: Round-trip artifact persistence
   - `test_versioned_artifacts`: Multiple version handling
   - `test_partial_artifact_loading`: Missing file tolerance

3. **TestErrorRecovery**
   - `test_cv_partial_failure_recovery`: Window failure recovery
   - `test_invalid_configuration_handling`: Config validation
   - `test_nan_handling_in_metrics`: NaN in metrics computation
   - `test_empty_window_handling`: Empty CV window handling

4. **TestConfigurationValidation**
   - `test_cv_config_validation`: CV configuration checking
   - `test_model_configuration_validation`: Model parameter validation
   - `test_horizon_consistency`: Multi-horizon consistency

5. **TestPerformanceBenchmarks**
   - `test_cv_execution_time`: <30s for small datasets
   - `test_metrics_computation_performance`: <5s for 100K samples
   - `test_memory_usage`: <500MB increase for large datasets

6. **TestRealDataScenarios**
   - `test_with_missing_data`: Gap detection in time series
   - `test_with_extreme_values`: Outlier handling with robust scalers
   - `test_with_different_horizons`: Multi-horizon (4,8,16,32) testing

7. **TestIntegrationWithOtherModules**
   - `test_integration_with_io_module`: I/O utilities integration
   - `test_integration_with_validation_module`: Validation checks
   - `test_integration_with_ensemble_module`: Ensemble creation

### Foundation Tests (`test_cv_foundation.py`)
Basic smoke tests for quick CI/CD validation (<5 minutes).

## Running Tests

### Quick Start
```bash
# Run all tests with coverage
python tests/run_tests.py

# Run only unit tests
python tests/run_tests.py --type unit

# Run only integration tests
python tests/run_tests.py --type integration

# Run quick smoke tests (CI/CD)
python tests/run_tests.py --type quick

# Run without coverage
python tests/run_tests.py --no-coverage
```

### Using pytest directly
```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_cv.py -v

# Run specific test class
pytest tests/test_cv.py::TestMetricsComputation -v

# Run specific test
pytest tests/test_cv.py::TestMetricsComputation::test_compute_mae_basic -v

# Run with coverage
pytest tests/ --cov=cv --cov=uq --cov-report=term-missing

# Run with markers
pytest tests/ -m "not slow"
```

## Test Coverage

### Target Coverage
- **Overall**: >90%
- **cv.runner**: >95%
- **uq.metrics**: >95%
- **uq.calibration**: >90%
- **utils.io**: >85%

### Coverage Report
After running tests with coverage, view the HTML report:
```bash
open tests/coverage_report/index.html
```

## Test Data

### Synthetic Data
- Used for unit tests
- Controlled random seeds for reproducibility
- Small datasets for speed

### Realistic Data
- BTC-like price series with trends and volatility
- 15-minute frequency with UTC timestamps
- Log returns as target variable
- Includes volume, volatility, momentum features

## Key Test Scenarios

### sCRPS Computation
- Point predictions: Simple MAE-based approximation
- Quantile predictions: Integration over quantile levels
- Distributional predictions: StudentT distribution parameters

### Coverage Analysis
- Nominal levels: 80%, 90%, 95%
- Tolerance: ±2% of nominal
- Multi-model comparison

### PIT Analysis
- Uniformity testing with KS statistic
- Support for both quantile and distributional models
- P-value threshold: 0.05

### Model Selection
- Ranking by sCRPS (primary metric)
- Tie-breaking with MAE
- Best model identification

## Performance Requirements

### Metrics Computation
- 10K samples: <100ms
- 100K samples: <1s
- 1M samples: <10s

### CV Execution
- Small dataset (200 samples): <30s
- Medium dataset (1000 samples): <2min
- Large dataset (10000 samples): <10min

### Memory Usage
- Training: <16GB
- Inference: <2GB
- Metrics computation: <500MB increase

## Error Handling

### Graceful Failures
- Missing data detection
- NaN handling in computations
- Empty window tolerance
- Partial artifact loading

### Validation Checks
- Regular grid validation
- UTC timestamp checking
- Feature shift verification
- No forward-fill on target

## CI/CD Integration

### GitHub Actions
```yaml
- name: Run Tests
  run: |
    python tests/run_tests.py --type quick
    
- name: Full Test Suite
  if: github.event_name == 'push'
  run: |
    python tests/run_tests.py --type full
```

### Pre-commit Hooks
```yaml
- repo: local
  hooks:
    - id: pytest-check
      name: pytest-check
      entry: python tests/run_tests.py --type quick
      language: system
      pass_filenames: false
      always_run: true
```

## Debugging Failed Tests

### Common Issues

1. **Import Errors**
   - Ensure parent directory is in path
   - Check module structure

2. **NaN Values**
   - Use `np.nanmean()` instead of `np.mean()`
   - Check for empty arrays after filtering

3. **Assertion Failures**
   - Verify tolerance levels (use `pytest.approx()`)
   - Check random seeds for reproducibility

4. **Performance Failures**
   - Profile slow sections
   - Consider data size reduction for tests
   - Use mocking for expensive operations

### Debug Commands
```bash
# Run with debugging output
pytest tests/test_cv.py -vv -s

# Run with pdb on failure
pytest tests/test_cv.py --pdb

# Run with traceback
pytest tests/test_cv.py --tb=long

# Profile test execution
pytest tests/test_cv.py --profile
```

## Test Maintenance

### Adding New Tests
1. Follow existing test class structure
2. Use descriptive test names
3. Include docstrings explaining test purpose
4. Add to appropriate test class
5. Update this documentation

### Test Data Management
- Use fixtures for reusable test data
- Control random seeds for reproducibility
- Keep test data small but representative
- Use tempfile for file I/O tests

### Mocking Guidelines
- Mock external dependencies (APIs, databases)
- Don't mock core functionality being tested
- Use `unittest.mock` for Python mocking
- Document mock behavior in test docstrings

## Quality Standards

### Test Quality Checklist
- [ ] Tests are independent (no shared state)
- [ ] Tests are deterministic (fixed seeds)
- [ ] Tests are fast (<1s for unit tests)
- [ ] Tests have clear assertions
- [ ] Tests cover edge cases
- [ ] Tests include error scenarios
- [ ] Tests are documented
- [ ] Tests follow naming conventions

### Code Review Checklist
- [ ] New features have tests
- [ ] Bug fixes include regression tests
- [ ] Coverage hasn't decreased
- [ ] Tests pass locally
- [ ] Tests are maintainable
- [ ] Performance benchmarks met

## Contact

For questions about tests or to report issues:
- Review test files in `tests/` directory
- Check CI/CD logs for failures
- Consult CV/metrics module documentation