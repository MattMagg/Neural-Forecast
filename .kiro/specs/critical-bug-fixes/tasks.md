# Implementation Plan

## Standard Task Completion Workflow

**MANDATORY**: Every task completion must follow this workflow:

### 1. Implementation Phase
- Complete the technical implementation as specified in the task
- Ensure all code changes are tested and validated
- Verify requirements are met and functionality works as expected

### 2. GitHub Issue Management
- **If task relates to GitHub issues**: Update issue status on GitHub
  - Add resolution comment with implementation details
  - Close the issue with appropriate status (completed/resolved)
  - Reference the commit hash in the resolution comment

### 3. Documentation Updates
- **Always required**: Update TASK_TRACKER.md with completion details
  - Update project version following semantic versioning
  - Add task to completed specifications table if completing a spec
  - Add detailed changelog entry with:
    - Implementation summary
    - Files modified and functions created
    - GitHub issues resolved (if applicable)
    - Technical details and key changes

### 4. Version Control
- Commit all changes with descriptive commit messages
- Push changes to the appropriate branch
- Ensure commit messages reference issue numbers when applicable

### 5. Validation
- Verify all changes are properly committed and pushed
- Confirm GitHub issues are updated (if applicable)
- Validate TASK_TRACKER.md reflects current project state

**Note**: This workflow ensures consistent project tracking, proper issue management, and comprehensive documentation of all completed work.

---

- [x] 1. Fix Critical Blocking Bugs (Issues #3 and #4)
  - Resolve duplicate column bug and sCRPS computation to unblock pipeline execution
  - Implement safe feature merging and proper metrics computation
  - **Complete with standard workflow**: Update GitHub issues, TASK_TRACKER.md, commit/push
  - _Requirements: 1.1, 1.2, 1.3, 1.4, 1.5, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6_

- [x] 1.1 Implement Safe Feature Merge Function
  - Create `safe_feature_merge()` function in utils/io.py to handle column conflicts
  - Add conflict detection logic for 'ds' and other duplicate columns
  - Implement automatic deduplication with proper logging
  - _Requirements: 1.1, 1.2, 1.3_

- [x] 1.2 Fix Feature Integration in Training Pipeline
  - Update `run_train.ipynb` to use `safe_feature_merge()` instead of direct merge
  - Add pre-merge and post-merge validation checks
  - Test with both base and MTF features to ensure no data loss
  - _Requirements: 1.1, 1.4, 1.5_

- [x] 1.3 Implement Proper sCRPS Computation
  - Replace `scrps_val = np.nan` stub in `cv/runner.py` with actual computation
  - Use NeuralForecast's native sCRPS function from losses.pytorch module
  - Handle both distributional (StudentT) and quantile (MQLoss) model types
  - _Requirements: 2.1, 2.2, 2.3_

- [x] 1.4 Add sCRPS Aggregation and Model Ranking
  - Implement proper aggregation of sCRPS values across CV windows
  - Update model selection logic to use computed sCRPS for ranking decisions
  - Add validation that sCRPS values are reasonable and consistent
  - _Requirements: 2.4, 2.5, 2.6_

- [x] 1.5 Complete Task 1 Documentation and Issue Management
  - Update GitHub issues #3 and #4 with resolution comments and close them
  - Update TASK_TRACKER.md with completion details and version bump
  - Commit and push all changes with proper commit messages
  - _Requirements: Standard completion workflow_

- [ ] 2. Fix Import Path and Conformal Prediction (Issue #5)
  - Correct PredictionIntervals import path to enable conformal prediction functionality
  - Update all files using incorrect import pattern
  - **Complete with standard workflow**: Update GitHub issue #5, TASK_TRACKER.md, commit/push
  - _Requirements: 3.1, 3.2, 3.3, 3.4, 3.5_

- [ ] 2.1 Search and Fix Import Paths
  - Search codebase for incorrect `from neuralforecast import PredictionIntervals`
  - Replace with correct `from neuralforecast.utils import PredictionIntervals`
  - Update cv/runner.py, run_train.ipynb, and any documentation notebooks
  - _Requirements: 3.1, 3.2_

- [ ] 2.2 Test Conformal Prediction Functionality
  - Verify PredictionIntervals can be instantiated without import errors
  - Test CV execution with prediction_intervals parameter
  - Validate conformal correction applies when coverage misses ±2pp threshold
  - _Requirements: 3.3, 3.4, 3.5_

- [ ] 2.3 Complete Task 2 Documentation and Issue Management
  - Update GitHub issue #5 with resolution comment and close it
  - Update TASK_TRACKER.md with completion details and version bump
  - Commit and push all changes with proper commit messages
  - _Requirements: Standard completion workflow_

- [ ] 3. Improve Data Processing Robustness (Issue #6)
  - Address data validation gaps and processing order issues
  - Implement proper train/validation isolation and feature computation order
  - **Complete with standard workflow**: Update GitHub issue #6, TASK_TRACKER.md, commit/push
  - _Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7_

- [ ] 3.1 Implement Winsorization Isolation
  - Create `isolate_winsorization()` function that only affects training data
  - Update `drop_train_nans_and_winsorize()` to use proper train/val separation
  - Add validation that winsorization never affects validation/test sets
  - _Requirements: 4.1, 4.5_

- [ ] 3.2 Fix Feature Computation Order
  - Implement proper pipeline: compute → align → shift(1) → validate
  - Ensure features are computed on raw data before NaN handling
  - Add validation that shift(1) is applied after computation but before validation
  - _Requirements: 4.2, 4.7_

- [ ] 3.3 Add Calendar Feature Validation
  - Create `validate_calendar_features()` function to check feature placement
  - Ensure calendar features (day_of_week, hour_of_day, is_weekend) are future-only
  - Add automatic validation in feature integration pipeline
  - _Requirements: 4.3_

- [ ] 3.4 Ensure CV Uses Original Target Values
  - Validate that cross-validation uses original (non-winsorized) target values
  - Add explicit checks that CV never uses modified training targets
  - Implement proper data flow to maintain target integrity
  - _Requirements: 4.4_

- [ ] 3.5 Add Comprehensive Data Processing Validation
  - Implement validation that NaN handling never forward-fills target variable
  - Add checks for proper train/validation isolation throughout pipeline
  - Create comprehensive test suite for data processing edge cases
  - _Requirements: 4.6_

- [ ] 3.6 Complete Task 3 Documentation and Issue Management
  - Update GitHub issue #6 with resolution comment and close it
  - Update TASK_TRACKER.md with completion details and version bump
  - Commit and push all changes with proper commit messages
  - _Requirements: Standard completion workflow_

- [ ] 4. Handle Dependency Updates Safely (Issue #7)
  - Manage PyTorch upgrade from 2.6.0 to 2.8.0 with compatibility testing
  - Ensure system stability while benefiting from improvements
  - **Complete with standard workflow**: Update GitHub issue #7, TASK_TRACKER.md, commit/push
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [ ] 4.1 Test PyTorch 2.8.0 Compatibility
  - Local: Install PyTorch 2.8.0 in test environment and run basic compatibility tests
  - Remote: Push changes to repo, connect to TNR instance, pull and test GPU functionality
  - Remote: Run existing model training tests to verify compatibility on A100XL hardware
  - Remote: Check GPU memory usage patterns and performance with new PyTorch version
  - _Requirements: 5.1, 5.3_

- [ ] 4.2 Validate NeuralForecast Compatibility
  - Test NeuralForecast functionality with PyTorch 2.8.0
  - Verify model training, cross-validation, and inference work correctly
  - Check for any breaking changes or performance regressions
  - _Requirements: 5.4_

- [ ] 4.3 Run Comprehensive Test Suite
  - Execute all existing tests with new PyTorch version
  - Compare results with previous version to ensure consistency
  - Document any differences or required adjustments
  - _Requirements: 5.2, 5.5_

- [ ] 4.4 Complete Task 4 Documentation and Issue Management
  - Update GitHub issue #7 with resolution comment and close it (if accepting the dependency update)
  - Update TASK_TRACKER.md with completion details and version bump
  - Commit and push all changes with proper commit messages
  - _Requirements: Standard completion workflow_

- [ ] 5. Create Comprehensive Test Suite
  - Develop tests for all bug fixes to prevent regression
  - Implement validation for each component and integration point
  - **Complete with standard workflow**: Update TASK_TRACKER.md, commit/push
  - _Requirements: All requirements validation_

- [ ] 5.1 Create Unit Tests for Bug Fixes
  - Test safe feature merge with various conflict scenarios
  - Test sCRPS computation with known test cases and edge conditions
  - Test import path corrections and conformal prediction instantiation
  - Test data processing functions with edge cases and validation scenarios

- [ ] 5.2 Implement Integration Tests
  - Local: Create integration test framework and basic pipeline tests
  - Remote: Connect to TNR instance and run end-to-end pipeline with all fixes applied
  - Remote: Test cross-validation with proper sCRPS computation and conformal prediction on GPU
  - Remote: Test feature integration pipeline from raw data to model-ready format with real BTC data
  - Remote: Test data processing pipeline with proper train/validation isolation using GPU resources

- [ ] 5.3 Add Regression Tests
  - Ensure fixes don't break existing functionality
  - Test performance impact of changes
  - Validate result consistency with previous versions
  - Create baseline comparisons for critical metrics

- [ ] 5.4 Complete Task 5 Documentation and Issue Management
  - Update TASK_TRACKER.md with test suite completion details and version bump
  - Commit and push all changes with proper commit messages
  - Document test coverage and validation results
  - _Requirements: Standard completion workflow_

- [ ] 6. Documentation and Validation
  - Update documentation to reflect fixes and improvements
  - Create validation reports for each bug fix
  - **Complete with standard workflow**: Update TASK_TRACKER.md, commit/push
  - _Requirements: All requirements documentation_

- [ ] 6.1 Update Technical Documentation
  - Document safe feature merge patterns and best practices
  - Update sCRPS computation examples and usage patterns
  - Correct import path examples in documentation notebooks
  - Document improved data processing pipeline and validation checks

- [ ] 6.2 Create Bug Fix Validation Report
  - Document each bug, its root cause, and the implemented solution
  - Provide before/after comparisons showing fixes work correctly
  - Include test results and performance impact analysis
  - Create troubleshooting guide for similar issues in the future

- [ ] 6.3 Complete Task 6 Documentation and Issue Management
  - Update TASK_TRACKER.md with documentation completion details and version bump
  - Commit and push all changes with proper commit messages
  - Finalize all documentation and validation reports
  - _Requirements: Standard completion workflow_
