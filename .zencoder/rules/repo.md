---
description: Repository Information Overview
alwaysApply: true
---

# Bitcoin Forecasting System Information

## Summary
A production-ready probabilistic forecasting system for Bitcoin price prediction using 15-minute intervals with calibrated prediction intervals. The system delivers high-quality probabilistic forecasts for BTC at 15-minute frequency with calibrated 80%/90%/95% prediction intervals across multiple time horizons (1h, 2h, 4h, 8h). Built with a NeuralForecast-centric approach, the system emphasizes data discipline, leakage prevention, and production readiness.

## Structure
- **data/**: Raw and processed datasets
- **features/**: Feature engineering pipeline with registry and builders
- **nf_models/**: Model factory and configurations
- **cv/**: Cross-validation and metrics
- **uq/**: Uncertainty quantification and calibration diagnostics
- **experiments/**: Per-horizon configurations and results
- **utils/**: Validation and I/O utilities
- **tests/**: Comprehensive test suite
- **examples/**: Usage examples and tutorials
- **docs/**: Documentation and technical specifications
- **.kiro/**: Specifications and steering documents
- **.claude/**: Agent configuration and orchestration

## Language & Runtime
**Language**: Python
**Version**: Python 3.13.6
**Build System**: pip
**Package Manager**: pip

## Dependencies
**Main Dependencies**:
- neuralforecast==3.0.2 (core forecasting framework)
- torch==2.8.0
- pytorch-lightning==2.5.3
- pandas==2.3.1
- numpy==2.3.2
- pyarrow==20.0.0
- TA-Lib==0.6.5 (requires system-level installation)
- vectorbt==0.28.0
- technical==1.4.0 (MTF resampling utilities)
- freqtrade==2025.7
- scikit-learn==1.7.1
- statsmodels==0.14.5

**Development Dependencies**:
- pytest==8.4.1
- pytest-cov==6.0.0

## Build & Installation
```bash
# Install system dependencies (TA-Lib)
sudo apt install -y libta-lib-dev

# Create and activate virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

## Infrastructure
**Local Development**: This repository is used for development, testing, and experiment configuration
**GPU Training**: Actual model training runs on Thunder Compute (TNR) GPU instances
**Connection Method**: TNR CLI tool for SSH connection and file transfer
**Commands**:
```bash
# Connect to GPU instance
tnr connect <instance_id>
ssh tnr-0

# Run training on GPU
ssh tnr-0 "cd /workspace && python run_train.py --horizon 4"

# Transfer files
tnr scp ./experiments/h4.yaml <instance_id>:/workspace/experiments/
tnr scp <instance_id>:/workspace/experiments/h4/metrics.json ./experiments/h4/
```

## Main Entry Points
**Training Pipeline**: `run_train.py`
- Implements complete data assembly and training path
- Loads and processes raw data
- Integrates features with leakage prevention
- Instantiates NeuralForecast models
- Runs cross-validation with metrics
- Command-line interface with `--horizon`, `--save-processed`, `--cv` options

**Prediction Pipeline**: `run_predict.py`
- Loads trained models
- Generates forecasts for specified horizons
- Outputs probabilistic predictions with calibrated intervals

**Jupyter Notebooks**:
- `run_train.ipynb`: Interactive version of training pipeline
- `run_predict.ipynb`: Interactive version of prediction pipeline
- `test_cv_integration.ipynb`: Cross-validation integration testing
- `nf_models/01_model_factory_core.ipynb`: Core model factory implementation
- `nf_models/02_integration_tests.ipynb`: Model integration testing
- `nf_models/03_performance_profiling.ipynb`: Performance analysis
- `nf_models/04_usage_documentation.ipynb`: Documentation and examples

## Testing
**Framework**: pytest
**Test Location**: tests/
**Naming Convention**: test_*.py
**Configuration**: pytest.ini
**Run Command**:
```bash
# Run all tests with coverage
python tests/run_tests.py --type full

# Run quick smoke tests
python tests/run_tests.py --type quick

# Run specific test types
python tests/run_tests.py --type unit
python tests/run_tests.py --type integration
python tests/run_tests.py --type foundation
```

## Data Processing
**Input**: 1-minute OHLCV Bitcoin data from Kaggle
**Output**: Regular 15-minute UTC grid with log returns
**Validation**: Comprehensive quality gates preventing data leakage
**Quality Gates**:
- `assert_regular_grid()`: Validates 15-min grid completeness
- `assert_utc_eob()`: Verifies UTC timezone and EOB alignment
- `assert_shifted()`: Correlation-based leakage detection
- `assert_no_forward_fill_y()`: Prevents target forward-fill

## Feature Engineering
**Indicators**: RSI, MACD, Bollinger Bands, ATR, and more (13 total)
**Multi-timeframe**: 30min, 1h, 4h aligned to 15min base
**Leakage Prevention**: Strict shift(1) rule for historical features
**Feature Selection**: Correlation pruning (|rho| ≥ 0.95), 256 feature cap
**Registry**: Centralized indicator specifications in `features/registry.py`

## Model Training
**Models**: NHITS, NBEATSx, TiDE, PatchTST with probabilistic losses
**Cross-validation**: NeuralForecast-native with proper windowing
**Model Selection**: sCRPS-based ranking with ensemble options
**Persistence**: NF-native save/load with versioning
**Configurations**: Per-horizon YAML files in `experiments/`
**GPU Acceleration**: Training runs on Thunder Compute GPU instances

## Uncertainty Quantification
**Coverage Diagnostics**: Empirical coverage validation at 80/90/95%
**PIT Analysis**: Probability integral transform uniformity testing
**Conformal Prediction**: NF's PredictionIntervals integration
**Metrics**: sCRPS (primary), MAE, RMSE, bias