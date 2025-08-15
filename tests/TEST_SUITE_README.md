# Feature Engineering Test Suite

## Overview
Minimal testing suite for the feature engineering pipeline as specified in Task 7 of `.kiro/specs/feature-engineering-pipeline/tasks.md`.

## Test Files Created

### 1. `test_features.py`
Main test file containing 4 critical tests:

#### Test 1: No-Leak Verification (`test_no_leak`)
- **Purpose**: Verify that historical features at time t only use data ≤ t-15m (not t)
- **Method**: Creates data with known patterns, applies feature pipeline, checks shift(1) is properly applied
- **Assertion**: Features at row t should equal raw computation at row t-1

#### Test 2: MTF Alignment Sanity (`test_mtf_alignment`)
- **Purpose**: Verify multi-timeframe features align correctly with shift
- **Method**: Creates 1-day series (96 bars), computes 1h features, checks alignment
- **Assertion**: 09:00-09:45 holds 08:00-09:00 value, updates at 10:00

#### Test 3: Feature Cap Enforcement (`test_feature_cap`)
- **Purpose**: Ensure total features ≤ 256
- **Method**: Builds full feature set, applies select_features()
- **Assertion**: Total of hist + futr + stat features ≤ 256

#### Test 4: Deterministic Stability (`test_deterministic`)
- **Purpose**: Verify pipeline determinism
- **Method**: Runs full pipeline twice with same input
- **Assertion**: Outputs are exactly identical

### 2. Supporting Files

- **`verify_test_structure.py`**: Verifies test file structure without running tests
- **`run_tests.sh`**: Shell script to run tests with or without pytest

## Design Principles

Following the "fast, not fluffy" philosophy:
- Uses minimal synthetic data (200-500 rows max)
- Deterministic with fixed random seeds
- Fast execution (< 30 seconds total)
- Clear failure messages
- No unnecessary complexity

## Running the Tests

### With pytest installed:
```bash
python3 -m pytest tests/test_features.py -v
```

### Without pytest:
```bash
python3 tests/test_features.py
```

### Using the test runner:
```bash
./tests/run_tests.sh
```

## Test Data Characteristics

- **No-leak test**: 200 bars with linearly increasing pattern
- **MTF test**: 96 bars (1 day) with hourly distinct values
- **Feature cap test**: 500 bars with random data
- **Deterministic test**: 300 bars with fixed seed

## Key Validations

Each test targets a specific critical requirement:
1. **Leakage Prevention**: Ensures shift(1) is applied to all historical features
2. **MTF Correctness**: Validates multi-timeframe alignment post-shift
3. **Feature Limits**: Enforces the 256 feature hard cap
4. **Reproducibility**: Guarantees deterministic behavior

## Dependencies

- pandas
- numpy
- features.builder module
- features.registry module
- pytest (optional, for better test output)

## Success Criteria

All 4 tests must pass to ensure:
- No future information leakage
- Correct multi-timeframe alignment
- Feature count within limits
- Deterministic pipeline behavior