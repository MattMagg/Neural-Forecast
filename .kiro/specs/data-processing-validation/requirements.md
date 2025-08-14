# Requirements Document

## Introduction

The Data Processing and Validation spec establishes the foundational data contracts, regularization procedures, and validation utilities for the BTC forecasting system. This component ensures data quality, prevents leakage, and creates canonical NeuralForecast-compatible data frames with proper UTC timestamp handling and grid regularization. It serves as the critical first layer that all other system components depend on for reliable, validated data inputs.

## Requirements

### Requirement 1: UTC Timestamp Handling and Grid Regularization

**User Story:** As a forecasting system, I want to ensure all timestamps are properly handled in UTC with end-of-bar (EOB) semantics and regular 15-minute grids, so that temporal alignment is consistent and predictable across all data processing operations.

#### Acceptance Criteria

1. WHEN processing raw price data THEN the system SHALL convert all timestamps to UTC timezone with tz-aware datetime64[ns] format
2. WHEN creating time grids THEN the system SHALL use end-of-bar semantics where timestamps align exactly on 15-minute boundaries (:00, :15, :30, :45)
3. WHEN validating time series data THEN the system SHALL implement assert_regular_grid(df, "15min") to ensure no gaps or duplicates
4. WHEN validating timestamps THEN the system SHALL implement assert_utc_eob(df, "15min") to verify UTC EOB alignment
5. WHEN detecting irregular grids THEN the system SHALL raise AssertionError with specific diagnostic information about missing or extra bars
6. IF timestamp conversion fails THEN the system SHALL halt processing with clear error messages indicating the problematic timestamps

### Requirement 2: Target Computation and Log Returns

**User Story:** As a forecasting model, I want properly computed log returns as the target variable with appropriate handling of price data, so that the model can learn from stationary, normalized price movements.

#### Acceptance Criteria

1. WHEN computing targets THEN the system SHALL calculate log returns as y_t = log(close_t) - log(close_t-1) using numpy.log
2. WHEN processing price data THEN the system SHALL ensure no zero or negative prices before log return computation
3. WHEN target computation encounters missing close prices THEN the system SHALL set corresponding y values to NaN (never forward-fill)
4. WHEN creating target series THEN the system SHALL maintain proper alignment with the regularized UTC timestamp grid
5. WHEN winsorizing for training THEN the system SHALL clip y at [0.1%, 99.9%] quantiles on training subset only
6. IF price data contains anomalies THEN the system SHALL preserve original y_raw values for evaluation while using y_train for model fitting

### Requirement 3: Canonical NeuralForecast Frame Creation

**User Story:** As a NeuralForecast model, I want data formatted in the canonical NF long-format schema with proper unique_id, ds, and y columns, so that I can consume the data without additional preprocessing.

#### Acceptance Criteria

1. WHEN creating NF frames THEN the system SHALL use the exact schema: ["unique_id", "ds", "y", <exog...>] in long format
2. WHEN setting unique_id THEN the system SHALL use "BTC-USD" as the string identifier for all rows
3. WHEN setting ds column THEN the system SHALL use tz-aware UTC datetime64[ns] timestamps
4. WHEN setting y column THEN the system SHALL use computed log returns as float64 dtype
5. WHEN including OHLCV data THEN the system SHALL preserve ["open", "high", "low", "close", "volume"] columns for feature engineering
6. WHEN implementing make_nf_canonical function THEN the system SHALL return DataFrame with columns ["unique_id", "ds", "y", "open", "high", "low", "close", "volume"]
7. IF schema validation fails THEN the system SHALL provide detailed error messages about column types and names

### Requirement 4: Data Quality Gates and Validation

**User Story:** As a system administrator, I want comprehensive data validation utilities that catch quality issues early, so that downstream components receive clean, validated data and training doesn't fail due to data problems.

#### Acceptance Criteria

1. WHEN validating data THEN the system SHALL implement assert_regular_grid(df, "15min") checking monotonic increasing ds and complete 15-minute grid
2. WHEN validating timestamps THEN the system SHALL implement assert_utc_eob(df, "15min") verifying UTC timezone and EOB alignment on 15-minute boundaries
3. WHEN validating features THEN the system SHALL implement assert_shifted(df, hist_cols) using correlation analysis to detect leakage
4. WHEN validating targets THEN the system SHALL implement assert_no_forward_fill_y(df) ensuring NaN close prices correspond to NaN y values
5. WHEN any validation fails THEN the system SHALL raise AssertionError with specific diagnostic information about the failure
6. WHEN validation passes THEN the system SHALL complete silently without logging (following assert pattern)
7. WHEN assert_shifted detects leakage THEN the system SHALL compare contemporaneous vs next-step correlation with y for each hist_col
8. IF correlation analysis indicates leakage THEN the system SHALL raise AssertionError with column name and correlation values

### Requirement 5: Missing Data Handling and Winsorization

**User Story:** As a data processing pipeline, I want robust handling of missing data and outliers through winsorization policies, so that the system can handle real-world data imperfections without compromising model training.

#### Acceptance Criteria

1. WHEN detecting missing 15-minute bars THEN the system SHALL insert rows with y as NaN on the complete UTC grid
2. WHEN handling missing target values THEN the system SHALL NEVER forward-fill y column to prevent leakage
3. WHEN applying winsorization THEN the system SHALL implement drop_train_nans_and_winsorize function with configurable quantiles (default 0.1%, 99.9%)
4. WHEN winsorizing data THEN the system SHALL create y_train column with clipped values while preserving original y for evaluation
5. WHEN processing exogenous features THEN the system SHALL allow forward-fill only for historic stateful features after shifting
6. WHEN handling missing OHLCV data THEN the system SHALL preserve NaN values in the regularized grid rather than imputing
7. IF missing data creates gaps in the time grid THEN the system SHALL maintain grid completeness with NaN placeholders

### Requirement 6: Leakage Prevention and Data Integrity

**User Story:** As a forecasting system, I want strict leakage prevention mechanisms that ensure no future information contaminates historical features, so that model performance estimates are realistic and unbiased.

#### Acceptance Criteria

1. WHEN processing historical features THEN the system SHALL enforce the compute → shift(1) rule for all hist_exog_list variables
2. WHEN validating shifted data THEN the system SHALL use assert_shifted function to detect leakage via correlation analysis
3. WHEN detecting potential leakage THEN the system SHALL raise AssertionError with specific column name and correlation diagnostics
4. WHEN creating train/validation splits THEN the system SHALL ensure temporal ordering is preserved (no shuffling)
5. WHEN assembling final datasets THEN the system SHALL validate that all quality gates pass before model training
6. WHEN implementing deterministic seeding THEN the system SHALL use fixed random seed (e.g., 1337) for reproducibility
7. IF leakage is detected THEN the system SHALL halt processing and provide detailed diagnostic information about the problematic feature

### Requirement 7: Data Assembly and Integration Path

**User Story:** As a system integrator, I want a clear data assembly path that combines all validation, processing, and formatting steps into a single, reliable pipeline, so that data preparation is reproducible and maintainable.

#### Acceptance Criteria

1. WHEN assembling data THEN the system SHALL execute steps in exact order: load raw OHLCV → regularize_to_grid_utc → make_nf_canonical → winsorize → build exogs → merge → validate
2. WHEN implementing regularize_to_grid_utc THEN the system SHALL create complete UTC grid with NaNs for missing data and snap to EOB boundaries
3. WHEN implementing make_nf_canonical THEN the system SHALL compute log returns and format as NF long-format schema
4. WHEN merging exogenous features THEN the system SHALL join on ds column and validate with assert_shifted
5. WHEN assembly completes THEN the system SHALL pass all validation gates before proceeding to model training
6. WHEN integration is used in run_train.py THEN the system SHALL wire the exact assembly path as specified
7. IF any assembly step fails THEN the system SHALL provide clear error messages indicating which step failed and why

### Requirement 8: Utility Functions and File Organization

**User Story:** As a developer, I want well-organized utility functions in the correct modules with proper imports and dependencies, so that the data processing components are maintainable and reusable.

#### Acceptance Criteria

1. WHEN implementing validation functions THEN the system SHALL place assert_regular_grid, assert_utc_eob, assert_shifted, assert_no_forward_fill_y in utils/validate.py
2. WHEN implementing data processing functions THEN the system SHALL place regularize_to_grid_utc, make_nf_canonical, drop_train_nans_and_winsorize in utils/io.py
3. WHEN implementing load/save functions THEN the system SHALL place load_canonical_frame, save_parquet, timestamped_path in utils/io.py
4. WHEN validation functions fail THEN the system SHALL raise AssertionError with specific diagnostic messages
5. WHEN utility functions encounter errors THEN the system SHALL raise ValueError with clear descriptions of the problem
6. WHEN functions succeed THEN the system SHALL return expected data types (DataFrame, None for assertions, str for paths)
7. IF imports are missing THEN the system SHALL include all required dependencies (pandas, numpy, pathlib, datetime, typing)