# Implementation Plan

- [ ] 1. Create validation utilities in utils/validate.py
  - Implement assert_regular_grid function checking monotonic ds and complete 15-min grid
  - Implement assert_utc_eob function verifying UTC timezone and EOB alignment
  - Implement assert_shifted function using correlation analysis to detect leakage
  - Implement assert_no_forward_fill_y function preventing target forward-fill
  - _Requirements: 1.3, 1.5, 4.1, 4.2, 4.3, 4.4_

- [ ] 2. Create data processing functions in utils/io.py
  - Implement regularize_to_grid_utc function with UTC conversion and grid creation
  - Implement make_nf_canonical function computing log returns and NF schema
  - Implement drop_train_nans_and_winsorize function with quantile clipping
  - Add load_canonical_frame, save_parquet, and timestamped_path utility functions
  - _Requirements: 1.1, 1.2, 2.1, 2.4, 3.1, 3.6, 5.3, 8.2, 8.3_

- [ ] 3. Wire assembly path into run_train.py
  - Add imports for validation and processing functions
  - Implement exact sequence: load → regularize_to_grid_utc → make_nf_canonical → validate
  - Add validation calls: assert_regular_grid, assert_utc_eob, assert_no_forward_fill_y
  - Prepare for exogenous feature integration with assert_shifted
  - _Requirements: 7.1, 7.2, 7.4, 7.6_

- [ ] 4. Test and validate implementation
  - Test validation functions with valid and invalid data
  - Test data processing functions with sample BTC data
  - Verify complete assembly path produces NF-compatible output
  - Confirm all validation gates work correctly
  - _Requirements: All requirements validation_