# Requirements Document

## Introduction

The GPU Training Workflow spec defines the complete end-to-end training execution process for the BTC forecasting system on Thunder Compute A100XL instances. This spec captures the practical, sequential workflow from environment setup through model training to results validation, ensuring autonomous agents can execute the complete training pipeline efficiently and reliably.

## Requirements

### Requirement 1: Sequential Training Pipeline

**User Story:** As an autonomous agent, I want a clear sequential training pipeline, so that I can execute the complete workflow from setup to results without gaps.

#### Acceptance Criteria

1. WHEN starting training THEN the system SHALL execute the 3-step sequence: setup.sh → kaggle_download_btc.py → run_train.ipynb
2. WHEN executing each step THEN the system SHALL provide clear success indicators and expected outputs
3. WHEN handling dependencies THEN the system SHALL ensure each step completes before proceeding to the next
4. WHEN estimating time THEN the system SHALL provide realistic duration estimates for A100XL hardware
5. WHEN monitoring progress THEN the system SHALL provide real-time progress indicators and GPU utilization metrics
6. IF any step fails THEN the system SHALL provide specific error diagnostics and recovery procedures

### Requirement 2: Environment Setup and Validation

**User Story:** As a system deployer, I want automated environment setup with comprehensive validation, so that the training environment is guaranteed to work correctly.

#### Acceptance Criteria

1. WHEN running setup.sh THEN the system SHALL install all dependencies in the correct order: Python 3.13 → CUDA → TA-Lib → requirements.txt
2. WHEN validating environment THEN the system SHALL test all critical imports and GPU accessibility
3. WHEN creating directories THEN the system SHALL establish the complete project structure with proper permissions
4. WHEN verifying setup THEN the system SHALL generate a setup report with package versions and system status
5. WHEN detecting issues THEN the system SHALL provide specific diagnostic information and remediation steps
6. IF setup fails THEN the system SHALL provide cleanup procedures and retry mechanisms

### Requirement 3: Data Acquisition and Processing

**User Story:** As a data manager, I want reliable data acquisition with integrity validation, so that training uses high-quality, validated datasets.

#### Acceptance Criteria

1. WHEN downloading data THEN the system SHALL use kaggle_download_btc.py to fetch Bitcoin historical data
2. WHEN validating data THEN the system SHALL verify file size (>400MB), row count (>7M), and date range coverage
3. WHEN processing data THEN the system SHALL execute the complete assembly path: 1min → 15min → canonical → validated
4. WHEN running validation gates THEN the system SHALL pass all quality checks: regular grid, UTC EOB, no leakage, no forward-fill
5. WHEN detecting data issues THEN the system SHALL provide specific error messages and re-download procedures
6. IF data validation fails THEN the system SHALL prevent training from proceeding with invalid data

### Requirement 4: Model Training Execution

**User Story:** As a training executor, I want guided notebook execution with optimal GPU utilization, so that I can train models efficiently on A100XL hardware.

#### Acceptance Criteria

1. WHEN configuring training THEN the system SHALL support horizon selection (h4, h8, h16, h32) via YAML configuration
2. WHEN executing notebook cells THEN the system SHALL provide cell-by-cell guidance with expected outputs
3. WHEN optimizing for A100XL THEN the system SHALL use optimal batch sizes (512 default, 256 fallback) and memory settings
4. WHEN running cross-validation THEN the system SHALL execute 6-window CV with proper progress tracking
5. WHEN monitoring GPU THEN the system SHALL provide real-time utilization metrics and memory usage
6. WHEN generating artifacts THEN the system SHALL save results to experiments/h{horizon}/ with proper timestamping

### Requirement 5: Results Validation and Quality Assessment

**User Story:** As a model validator, I want comprehensive results assessment with clear quality metrics, so that I can determine if training was successful.

#### Acceptance Criteria

1. WHEN checking artifacts THEN the system SHALL validate presence of cv_results.parquet, metrics.json, leaderboard.csv, and best/ models
2. WHEN interpreting metrics THEN the system SHALL assess sCRPS scores (target <0.12) and coverage percentages (±2% tolerance)
3. WHEN ranking models THEN the system SHALL generate leaderboard with model performance comparison
4. WHEN validating quality THEN the system SHALL apply acceptance criteria for model promotion
5. WHEN generating reports THEN the system SHALL create comprehensive training summary with key metrics
6. IF quality is insufficient THEN the system SHALL provide guidance on parameter tuning and retraining

### Requirement 6: Error Handling and Recovery

**User Story:** As a system operator, I want robust error handling with systematic recovery procedures, so that I can quickly resolve issues and minimize downtime.

#### Acceptance Criteria

1. WHEN encountering CUDA errors THEN the system SHALL provide GPU memory diagnostics and batch size reduction procedures
2. WHEN facing import errors THEN the system SHALL provide module path diagnostics and environment repair steps
3. WHEN handling data errors THEN the system SHALL provide validation diagnostics and re-download procedures
4. WHEN experiencing training failures THEN the system SHALL provide model-specific debugging guidance
5. WHEN recovering from interruptions THEN the system SHALL support checkpoint recovery and training resumption
6. WHEN escalating issues THEN the system SHALL collect comprehensive diagnostic information for troubleshooting

### Requirement 7: Performance Optimization

**User Story:** As a performance optimizer, I want A100XL-specific optimizations and monitoring, so that I can achieve maximum training efficiency.

#### Acceptance Criteria

1. WHEN optimizing memory THEN the system SHALL use A100XL-specific batch sizes and memory configurations
2. WHEN monitoring performance THEN the system SHALL track GPU utilization (target >80%) and memory usage
3. WHEN estimating duration THEN the system SHALL provide realistic time estimates: 2-4 hours per horizon
4. WHEN managing resources THEN the system SHALL prevent memory leaks and optimize GPU allocation
5. WHEN tracking progress THEN the system SHALL provide detailed progress indicators for each training phase
6. IF performance is suboptimal THEN the system SHALL provide systematic optimization procedures

### Requirement 8: Configuration Management

**User Story:** As a configuration manager, I want flexible horizon and parameter management, so that I can train different forecasting horizons effectively.

#### Acceptance Criteria

1. WHEN selecting horizons THEN the system SHALL support h4 (1h), h8 (2h), h16 (4h), h32 (8h) configurations
2. WHEN loading configurations THEN the system SHALL validate YAML syntax and parameter ranges
3. WHEN modifying parameters THEN the system SHALL identify safe adjustments vs. critical settings
4. WHEN auto-correcting configs THEN the system SHALL fix val_size to 4*h as per specification
5. WHEN switching horizons THEN the system SHALL provide clear guidance on configuration changes
6. IF configurations are invalid THEN the system SHALL provide specific error messages and correction procedures