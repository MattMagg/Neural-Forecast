# Technology Stack & Build System

## Core Dependencies
- **Python**: 3.12+ (required)
- **NeuralForecast**: 3.0.2+ (pinned version - core framework)
- **GPU Support**: Required for model training
- **Database**: PostgreSQL (data storage), Redis (caching)

## Key Libraries
- **Modeling**: NeuralForecast (primary), PyTorch (backend)
- **Technical Indicators**: vectorbt, TA-Lib (primary), pandas-ta-openbb, freqtrade/technical (supplement)
- **Data Processing**: pandas, numpy, pyarrow (parquet support)
- **Configuration**: PyYAML
- **Validation**: Custom utilities in `utils/validate.py` (IMPLEMENTED)

## Implemented Dependencies
- **pandas**: Data manipulation and analysis (INSTALLED)
- **pyarrow**: Parquet file support for data persistence (INSTALLED)
- **numpy**: Numerical computations for log returns and validation (AVAILABLE)

## NeuralForecast-Only Policy
- Use NF's native cross-validation - no custom backtesting
- Use NF's save/load mechanisms - no custom serialization  
- Use NF's scalers - no external normalization
- Use NF's conformal prediction when needed
- Import pattern: `from neuralforecast.utils import PredictionIntervals`

## Common Commands

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Training & Prediction
```bash
# Train models for specific horizon
python run_train.py --horizon h4

# Run predictions
python run_predict.py --horizon h4

# Cross-validation
python run_train.py --cv --horizon h4
```

### Development
```bash
# Run data processing pipeline (IMPLEMENTED)
python run_train.py --save-processed

# Run validation checks
python -m utils.validate

# Check project status
python -m utils.version
```

## Configuration Management
- **Global Config**: `settings.yaml` (project-wide parameters)
- **Experiment Configs**: `experiments/h{horizon}.yaml` (per-horizon settings)
- **Defaults**: `experiments/defaults.yaml` (shared parameters)