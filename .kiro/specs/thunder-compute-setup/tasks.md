# Implementation Plan

- [x] 1. Create Thunder Compute setup script
  - [x] 1.1 Create setup.sh script for complete environment setup
    - Install system packages (build-essential, git, curl, wget)
    - Install Python 3.13.6 from deadsnakes PPA
    - Install NVIDIA drivers and CUDA toolkit for A100XL
    - Install TA-Lib C library (apt or compile from source)
    - Create virtual environment and activate
    - Install dependencies in correct order from dependencies.yaml
    - Clone/deploy the complete BTC forecasting project
    - Create required directory structure (data/, experiments/, reports/)
    - Run basic validation tests using existing test suite
    - _Reference: dependencies.yaml installation_notes section_

  - [x] 1.2 Create validation script using existing tests
    - Use existing validation functions from utils/validate.py
    - Test GPU access with torch.cuda.is_available()
    - Test critical imports (neuralforecast, talib, vectorbt)
    - Test existing project modules can be imported
    - Run existing test suite to verify everything works
    - Generate setup report with package versions
    - _Reference: existing tests in tests/ directory_

- [x] 2. Create practical training guide
  - [x] 2.1 Create TRAINING_GUIDE.md with A100XL-specific commands
    - Provide specific training commands for each horizon (h4, h8, h16, h32)
    - Include optimal batch sizes for 80GB VRAM (512 default, 256 fallback)
    - Add GPU monitoring commands (nvidia-smi, gpustat)
    - Include expected training times and memory usage
    - Add troubleshooting section for common issues
    - Provide performance optimization tips for A100XL
    - _Reference: settings.yaml for current batch sizes and configurations_

- [x] 3. Create cleanup and utility scripts
  - [x] 3.1 Create cleanup.sh for failed installations
    - Remove partial virtual environment
    - Clean up downloaded packages
    - Reset CUDA/driver installations if needed
    - Provide fresh start capability
    - _Reference: common cleanup patterns_

  - [x] 3.2 Create quick validation script
    - Test that all implemented modules work
    - Verify GPU is accessible and has 80GB VRAM
    - Run smoke test using existing pipeline
    - Generate environment report
    - _Reference: existing test files for validation patterns_
