# Requirements Document

## Introduction

The Thunder Compute Setup spec provides a simple, practical setup for deploying the completed BTC forecasting system (v0.5.1) on a fresh Thunder Compute instance. The system is already fully implemented and tested - this spec just handles environment setup and deployment to get training running quickly on the A100XL GPU (80GB VRAM).

## Requirements

### Requirement 1: System Setup Script

**User Story:** As a developer, I want a single setup script that prepares the Thunder Compute instance, so that I can get the environment ready without manual configuration.

#### Acceptance Criteria

1. WHEN running the setup script THEN the system SHALL update Ubuntu packages and install build essentials
2. WHEN installing Python THEN the system SHALL install Python 3.13.6 with pip and venv
3. WHEN installing CUDA THEN the system SHALL install NVIDIA drivers and CUDA toolkit for A100XL
4. WHEN installing TA-Lib THEN the system SHALL install the C library before Python package
5. WHEN creating environment THEN the system SHALL create virtual environment and install dependencies in correct order
6. WHEN verifying setup THEN the system SHALL test all critical imports and GPU access
7. IF any step fails THEN the system SHALL provide clear error messages and continue where possible

### Requirement 2: Dependency Installation

**User Story:** As a dependency manager, I want all packages installed in the correct order from the existing requirements.txt, so that there are no version conflicts.

#### Acceptance Criteria

1. WHEN installing core packages THEN the system SHALL follow the exact order from dependencies.yaml installation notes
2. WHEN installing numpy THEN the system SHALL install numpy==2.3.2 first
3. WHEN installing pandas THEN the system SHALL install pandas==2.3.1 and pyarrow==20.0.0 second  
4. WHEN installing PyTorch THEN the system SHALL install torch==2.8.0 third
5. WHEN installing NeuralForecast THEN the system SHALL install neuralforecast==3.0.2 fourth
6. WHEN installing remaining packages THEN the system SHALL install all other requirements.txt packages
7. IF installation fails THEN the system SHALL retry with fallback strategies

### Requirement 3: Project Deployment

**User Story:** As a project deployer, I want the complete implemented system available immediately, so that I can start training without additional setup.

#### Acceptance Criteria

1. WHEN deploying project THEN the system SHALL clone or copy the complete repository
2. WHEN setting up directories THEN the system SHALL create data/, experiments/, and reports/ directories
3. WHEN verifying deployment THEN the system SHALL test that all implemented modules import correctly
4. WHEN testing pipeline THEN the system SHALL run a quick validation test
5. IF modules are missing THEN the system SHALL report which components need attention

### Requirement 4: Training Guide

**User Story:** As a user, I want a practical training guide with specific commands, so that I can immediately start training models on the A100XL.

#### Acceptance Criteria

1. WHEN providing training commands THEN the system SHALL include specific examples for each horizon (h4, h8, h16, h32)
2. WHEN optimizing for A100XL THEN the system SHALL provide optimal batch sizes for 80GB VRAM
3. WHEN monitoring training THEN the system SHALL include GPU monitoring commands
4. WHEN handling common issues THEN the system SHALL provide troubleshooting for typical problems
5. WHEN estimating performance THEN the system SHALL provide expected training times and memory usage

### Requirement 5: Environment Validation

**User Story:** As a validator, I want to confirm the environment works correctly, so that I can trust the setup before starting long training runs.

#### Acceptance Criteria

1. WHEN validating Python THEN the system SHALL test critical imports (pandas, torch, neuralforecast)
2. WHEN validating GPU THEN the system SHALL confirm A100XL is accessible with 80GB VRAM
3. WHEN validating project THEN the system SHALL test that implemented modules work
4. WHEN validating pipeline THEN the system SHALL run a minimal end-to-end test
5. IF validation fails THEN the system SHALL provide specific error messages and fixes