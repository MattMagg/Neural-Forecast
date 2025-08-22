# Requirements Document

## Introduction

The Training Guide Optimization spec creates a clear, step-by-step training guide for the fully-implemented BTC forecasting system (v0.5.1.2). The system includes complete data processing (Spec 1), feature engineering (Spec 2), model factory (Spec 3), and cross-validation (Spec 4) implementations. This spec focuses on creating a practical guide that provides sequential steps for autonomous training execution, emphasizing local validation to minimize GPU debugging time.

## Requirements

### Requirement 1: Sequential Training Workflow

**User Story:** As an autonomous agent deploying to Thunder Compute, I want a clear sequential workflow using existing components, so that I can execute training from setup to results without gaps.

#### Acceptance Criteria

1. WHEN starting deployment THEN the system SHALL provide the exact sequence: setup.sh → kaggle_download_btc.py → run_train.ipynb → validation
2. WHEN documenting each step THEN the system SHALL explain what each component accomplishes and expected outputs
3. WHEN providing commands THEN the system SHALL include specific commands and file paths relative to workspace root
4. WHEN estimating time THEN the system SHALL provide realistic time estimates for each step on A100XL
5. WHEN handling dependencies THEN the system SHALL clearly show step dependencies and what can be parallelized
6. IF any step fails THEN the system SHALL provide recovery procedures to continue the workflow

### Requirement 2: Local Pre-GPU Validation

**User Story:** As a developer with expensive GPU resources, I want comprehensive local validation procedures, so that I can catch all preventable issues before GPU deployment.

#### Acceptance Criteria

1. WHEN validating imports THEN the system SHALL test all module imports: utils.io, features.builder, nf_models.factory, cv.runner
2. WHEN testing configurations THEN the system SHALL validate YAML configs can be loaded and parsed correctly
3. WHEN testing data processing THEN the system SHALL run data pipeline on small sample (first 10k rows)
4. WHEN testing model instantiation THEN the system SHALL verify models can be created without GPU
5. WHEN identifying GPU-only operations THEN the system SHALL clearly distinguish local vs. GPU-required steps
6. WHEN providing validation script THEN the system SHALL create a single validation command that tests everything locally
7. IF local validation fails THEN the system SHALL provide specific fixes for each type of failure

### Requirement 3: Environment Setup and Data Acquisition

**User Story:** As a system deployer, I want foolproof environment setup and data acquisition procedures, so that the foundation is solid before training begins.

#### Acceptance Criteria

1. WHEN running setup.sh THEN the system SHALL document expected output and success indicators
2. WHEN installing dependencies THEN the system SHALL explain the critical order: Python 3.13 → CUDA → TA-Lib → requirements.txt
3. WHEN downloading data THEN the system SHALL provide validation steps for kaggle_download_btc.py output
4. WHEN verifying data THEN the system SHALL include checks for file size, date range, and basic statistics
5. WHEN activating environment THEN the system SHALL provide commands to verify virtual environment is working
6. IF setup fails THEN the system SHALL provide diagnostic commands and common failure solutions

### Requirement 4: Training Execution with run_train.ipynb

**User Story:** As a training executor, I want clear guidance for using run_train.ipynb effectively, so that I can execute the complete pipeline with proper monitoring.

#### Acceptance Criteria

1. WHEN configuring training THEN the system SHALL explain how to set horizon (h4/h8/h16/h32) in configuration cells
2. WHEN executing cells THEN the system SHALL provide cell-by-cell guidance with expected outputs
3. WHEN monitoring progress THEN the system SHALL document GPU monitoring commands and training progress indicators
4. WHEN optimizing for A100XL THEN the system SHALL provide optimal batch sizes and memory settings per model
5. WHEN handling the pipeline THEN the system SHALL explain the sequence: data processing → features → models → CV
6. WHEN tracking artifacts THEN the system SHALL document what files are created in experiments/h{horizon}/
7. IF training fails THEN the system SHALL provide debugging steps for common failure modes

### Requirement 5: Results Validation and Quality Assessment

**User Story:** As a model validator, I want clear procedures for assessing training results, so that I can determine if models meet quality standards.

#### Acceptance Criteria

1. WHEN checking artifacts THEN the system SHALL list expected files: cv_results.parquet, metrics.json, leaderboard.csv, best/ models
2. WHEN interpreting metrics THEN the system SHALL explain sCRPS scores, coverage percentages, and acceptance thresholds
3. WHEN validating models THEN the system SHALL provide commands to load and test saved models
4. WHEN assessing quality THEN the system SHALL define success criteria: sCRPS < baseline, coverage within ±2pp
5. WHEN generating reports THEN the system SHALL document how to create acceptance reports
6. IF results are poor THEN the system SHALL provide guidance on parameter tuning and retraining

### Requirement 6: Configuration Management and Horizon Selection

**User Story:** As a configuration manager, I want clear guidance on using existing YAML configurations, so that I can train different horizons effectively.

#### Acceptance Criteria

1. WHEN selecting horizons THEN the system SHALL explain h4.yaml (1h), h8.yaml (2h), h16.yaml (4h), h32.yaml (8h)
2. WHEN understanding configs THEN the system SHALL document key parameters: models, batch_size, learning_rate, n_windows
3. WHEN modifying settings THEN the system SHALL identify safe parameters to adjust vs. those that should not be changed
4. WHEN validating configs THEN the system SHALL provide YAML syntax validation and parameter range checks
5. WHEN training multiple horizons THEN the system SHALL provide guidance on sequential vs. parallel execution
6. IF configs are invalid THEN the system SHALL provide specific error messages and fixes

### Requirement 7: Troubleshooting and Error Recovery

**User Story:** As a system operator, I want systematic troubleshooting procedures, so that I can quickly resolve issues and minimize downtime.

#### Acceptance Criteria

1. WHEN encountering CUDA errors THEN the system SHALL provide GPU memory diagnostics and batch size reduction steps
2. WHEN facing import errors THEN the system SHALL provide module path diagnostics and environment fixes
3. WHEN handling data errors THEN the system SHALL provide data validation and re-download procedures
4. WHEN resolving training failures THEN the system SHALL provide model-specific debugging steps
5. WHEN recovering from interruptions THEN the system SHALL provide checkpoint recovery and resume procedures
6. WHEN escalating issues THEN the system SHALL provide diagnostic information collection commands
7. IF multiple failures occur THEN the system SHALL provide systematic diagnosis workflow

### Requirement 8: Performance Optimization and Monitoring

**User Story:** As a performance optimizer, I want guidance on maximizing A100XL utilization, so that I can achieve optimal training efficiency.

#### Acceptance Criteria

1. WHEN optimizing memory THEN the system SHALL provide A100XL-specific batch sizes and memory settings
2. WHEN monitoring GPU THEN the system SHALL provide nvidia-smi commands and utilization targets
3. WHEN tracking progress THEN the system SHALL document training time estimates and progress indicators
4. WHEN optimizing throughput THEN the system SHALL provide guidance on parallel training strategies
5. WHEN managing resources THEN the system SHALL document memory usage patterns and optimization techniques
6. WHEN preventing bottlenecks THEN the system SHALL identify common performance issues and solutions
7. IF performance is poor THEN the system SHALL provide systematic optimization procedures