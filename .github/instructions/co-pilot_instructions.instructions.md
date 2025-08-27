---
applyTo: '**'
---
# Neural-Forecast Co-Pilot Instructions

## Project Overview
**Neural-Forecast** is a production-ready probabilistic forecasting system for Bitcoin price prediction using 15-minute intervals with calibrated prediction intervals. The system delivers high-quality probabilistic forecasts for BTC at 15-minute frequency with calibrated 80%/90%/95% prediction intervals across horizons h ∈ {4, 8, 16, 32} (1h, 2h, 4h, 8h).

**Current Version**: 0.5.2.11  
**Branch**: instance-training-v0.5.2.X  
**Status**: Multiple specifications completed, production-ready foundation implemented

## Architecture Principles

### NeuralForecast-Native (ABSOLUTE REQUIREMENT)
- **Use NeuralForecast library directly** without abstraction layers
- **ZERO custom backtesting** - Use NF's native cross-validation only
- **ZERO custom serialization** - Use NF's save/load mechanisms only
- **ZERO external normalization** - Use NF's scalers only
- **Import pattern**: `from neuralforecast.utils import PredictionIntervals`

### Simple Scripts
- Write focused scripts that do one thing well
- Single-machine execution, no distributed systems
- Local development only

### Code Organization Rules (NON-NEGOTIABLE)
**Check project structure before creating ANY file. Files MUST go in their designated directories:**

- **Features**: `features/` (registry.py, builder.py)
- **Models**: `nf_models/` (factory.py)
- **Cross-validation**: `cv/` (runner.py, hpo.py)
- **Utilities**: `utils/` (validate.py, io.py, version.py)
- **Experiments**: `experiments/h{h}.yaml`
- **Data**: `data/` (processed datasets)

**NO EXCEPTIONS:** Creating files in wrong locations is unacceptable. Organization is not optional.

## Data Quality Gates - MANDATORY VALIDATION

**EVERY DATASET MUST PASS - NO EXCEPTIONS:**

```python
assert_regular_grid(df, "15min")
assert_utc_eob(df, "15min")
assert_shifted(df, hist_cols)
assert_no_forward_fill_y(df)
```

**FAILURE = STOP EXECUTION:** If validation fails, fix the data. Don't proceed with broken datasets.

## Critical Developer Workflows

### Data Processing Pipeline
**Command**: `python run_train.py --save-processed`
**Sequence** (from `utils/io.py`):
1. `load_raw_1min_data()` - Load 1-minute OHLCV data
2. `aggregate_1min_to_15min()` - Aggregate to 15-minute bars with UTC EOB alignment
3. `regularize_to_grid_utc()` - Create complete UTC time grid
4. `make_nf_canonical()` - Transform to NeuralForecast canonical schema with log returns
5. `drop_train_nans_and_winsorize()` - Apply winsorization while preserving evaluation data

### Feature Engineering Pipeline
**Location**: `features/`
**Sequence** (from `features/builder.py`):
1. `build_indicators()` - Compute base indicators from registry
2. `apply_mtf()` - Apply multi-timeframe features (30min, 1h, 4h)
3. `postprocess_shift_and_prune()` - Apply shift(1) and pruning
4. `select_features()` - Select ≤256 features with hard cap

### Training & Cross-Validation
**Command**: `python run_train.py --horizon h4`
**Location**: `run_train.py`
**Key Functions**:
- `load_experiment_config()` - Load YAML config
- `integrate_features()` - Add features with leakage prevention
- `train_and_evaluate()` - Run NF-native training and CV
- `run_cv_with_progress()` - Execute cross-validation with progress logging

### Inference Pipeline
**Command**: `python run_predict.py --horizon h4`
**Location**: `run_predict.py`
**Features**: Batch inference with prediction intervals, save/load support

## Project-Specific Conventions

### Naming Conventions
- **Horizons**: `h4`, `h8`, `h16`, `h32` (1h, 2h, 4h, 8h)
- **Variables**: snake_case
- **Classes**: PascalCase
- **Constants**: ALL_CAPS
- **Files**: Descriptive names with clear purpose

### Configuration Patterns
**Global Settings**: `settings.yaml`
```yaml
freq: "15min"
horizons: [4, 8, 16, 32]
seed: 1337
```

**Per-Horizon Configs**: `experiments/h{h}.yaml`
```yaml
h: 4
n_windows: 6
step_size: 4
val_size: 16
models:
  - NHITS:
      input_size: 1024
      loss: {kind: studentt}
```

### Import Patterns
```python
# NeuralForecast
from neuralforecast import NeuralForecast
from neuralforecast.models import NHITS, NBEATSx, TiDE, PatchTST
from neuralforecast.losses.pytorch import DistributionLoss, MQLoss, IQLoss

# Data processing
from utils.io import (
    load_raw_1min_data,
    aggregate_1min_to_15min,
    regularize_to_grid_utc,
    make_nf_canonical
)

# Validation
from utils.validate import (
    assert_regular_grid,
    assert_utc_eob,
    assert_shifted,
    assert_no_forward_fill_y
)
```

## Integration Points & Dependencies

### Core Libraries
- **NeuralForecast**: 3.0.2+ (pinned version - core framework)
- **Technical Indicators**:
  - Primary: `vectorbt` and `TA-Lib` wrappers
  - Supplement: `pandas-ta-openbb` (pure Python, Numba-accelerated)
  - MTF: `freqtrade/technical` for multi-timeframe features
- **Data Processing**: `pandas`, `numpy`, `pyarrow`
- **Configuration**: `PyYAML`

### External Dependencies
- **Database**: PostgreSQL (data storage), Redis (caching)
- **GPU Support**: Required for model training
- **Python**: 3.12+ (required)

### Model Portfolio
```python
# From nf_models/factory.py
models = [
    NHITS(input_size=1024, loss=DistributionLoss('StudentT')),
    NBEATSx(input_size=1024, loss=DistributionLoss('StudentT')),
    TiDE(input_size=1024, loss=DistributionLoss('StudentT')),
    PatchTST(input_size=2048, loss=DistributionLoss('StudentT'))
]
```

## Common Commands & Workflows

### Environment Setup
```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
pip install -r requirements.txt
```

### Data Processing
```bash
# Process data only (no training)
python run_train.py --save-processed

# Full pipeline for specific horizon
python run_train.py --horizon h4
```

### Training & Evaluation
```bash
# Train models for specific horizon
python run_train.py --horizon h4

# Run cross-validation only
python run_train.py --cv --horizon h4

# Use conformal prediction
python run_train.py --use-conformal --horizon h4
```

### Inference
```bash
# Run predictions
python run_predict.py --horizon h4

# Batch inference with custom data
python run_predict.py --input-data custom_data.parquet --horizon h4
```

### Validation & Testing
```bash
# Run data validation checks
python -m utils.validate

# Run feature integration tests
python test_feature_integration.py

# Run CV integration tests
python test_cv_integration.py
```

## Data Discipline Requirements

### UTC End-of-Bar Timestamps
**MANDATORY**: All timestamps must be UTC timezone-aware and align to 15-minute boundaries:
- Valid minutes: [0, 15, 30, 45]
- Seconds and microseconds must be 0
- No forward-filling of target variable `y`

### Leakage Prevention
**MANDATORY**: All historical exogenous features must follow:
1. **Compute**: Calculate the indicator/feature
2. **Align**: Align to 15-minute base frequency
3. **Shift(1)**: Shift by 1 period to prevent leakage

### Feature Engineering
- **Maximum features**: 256 after pruning
- **Multi-timeframe**: 30min, 1h, 4h aligned to 15min base
- **Availability filter**: ≥98% availability required
- **Correlation pruning**: Remove highly collinear features (|ρ|≥0.95)

## Quality Gates & Acceptance Criteria

### Hard Pass/Fail Gates
1. **Accuracy**: Mean sCRPS ≤ baseline × 0.985 (≥1.5% improvement)
2. **Calibration**: Coverage at 80/90/95 within ±2pp of nominal
3. **Robustness**: sCRPS in top volatility quintile degrades ≤10%
4. **Stability**: CV std deviation not worse than baseline by >20%

### Validation Assertions
```python
# These MUST pass before any NF operations
assert_regular_grid(df, "15min")      # Regular time grid
assert_utc_eob(df, "15min")           # UTC EOB alignment
assert_shifted(df, hist_cols)         # No data leakage
assert_no_forward_fill_y(df)          # No target forward-fill
```

## Cross-Validation Semantics

### Windowing Strategy
- **n_windows**: 6 (pilot) → 10 (final)
- **step_size**: = h (non-overlapping horizons)
- **val_size**: = 4*h
- **refit**: True (retrain every window)

### Primary Metric
- **sCRPS**: Scaled Continuous Ranked Probability Score
- **Supporting**: MAE, RMSE, coverage at 80/90/95

### NF-Native Only
```python
# Correct approach
from neuralforecast import NeuralForecast
nf = NeuralForecast(models=models, freq='15min')
cv_df = nf.cross_validation(df=df, n_windows=6, step_size=h)
```

## Model Configuration Patterns

### Common Parameters
```python
# Shared across all models
common_params = {
    'input_size': 1024,
    'loss': DistributionLoss('StudentT'),
    'learning_rate': 0.001,
    'batch_size': 512,
    'max_steps': 20000,
    'early_stop_patience_steps': 400,
    'val_check_steps': 100
}
```

### Model-Specific Overrides
```python
# PatchTST requires larger input size
PatchTST(
    input_size=2048,  # Larger than default
    revin=True,       # Reversible instance normalization
    **common_params
)
```

## Error Handling & Debugging

### Common Issues
1. **Memory Errors**: Reduce `batch_size`, `input_size`, `n_heads`, `hidden_size`
2. **Leakage Detection**: Check `assert_shifted()` failures - fix shift(1)
3. **Grid Issues**: Fix timestamp alignment with `regularize_to_grid_utc()`
4. **Feature Limits**: Prune features to stay under 256 limit

### GPU Memory Management
```python
# Graceful degradation strategy
try:
    # Try full configuration
    model = PatchTST(input_size=2048, hidden_size=512)
except RuntimeError as e:
    if "memory" in str(e).lower():
        # Fallback to smaller configuration
        model = PatchTST(input_size=1024, hidden_size=256)
```

## File Structure Expectations

### Expected Directories
```
data/               # Raw and processed datasets
├── raw/            # Original 1-minute data
└── processed/      # 15-minute canonical frames

features/           # Feature engineering
├── registry.py     # Indicator specifications
└── builder.py      # Feature computation and assembly

nf_models/          # Model factory
└── factory.py      # NF model instantiation

cv/                 # Cross-validation
├── runner.py       # CV execution
└── hpo.py          # Hyperparameter optimization

experiments/        # Per-horizon configs
├── h4.yaml         # 1-hour horizon config
├── h8.yaml         # 2-hour horizon config
└── ...

utils/              # Utilities
├── validate.py     # Data validation functions
├── io.py           # Data I/O and processing
└── ...

uq/                 # Uncertainty quantification
├── diag.py         # Diagnostics
├── calibration.py  # Calibration utilities
└── ...
```

### Key Files to Reference
- **`docs/forecasting_sf_plan.md`**: Authoritative technical specification
- **`AGENTS.md`**: AI agent instructions and project philosophy
- **`settings.yaml`**: Global configuration defaults
- **`run_train.py`**: Main training entry point
- **`run_predict.py`**: Main inference entry point

## Success Criteria

- **Reproducible CV artifacts** with sCRPS as primary metric
- **Saved model winners** with proper versioning
- **15-minute inference loop** with calibrated prediction intervals
- **Acceptance reports** showing ACCEPT for ≥2 horizons
- **Monitoring system** with coverage within ±3pp of nominal levels

## Operational Efficiency

**Batch Operations:** When performing multiple related operations, execute them concurrently in a single message rather than sequentially. This applies to file operations, bash commands, and tool invocations.

**VERIFICATION REQUIRED:** All claims must be backed by verification commands showing success. No assumptions about system state. No "it should work" - PROVE IT WORKS.

## Final Reminders

- **Be direct, critical, and confident** at all times
- **Avoid filler and affirmation** - no "yes-man" behavior
- **State problems clearly** with evidence when they exist
- **Your assessment matters** more than satisfaction
- **Iterate until solved** - never give up on problems
- **Clean up your mess** - proper file organization is mandatory
- **Follow NeuralForecast principles** - no custom implementations of NF features
- **Maintain data discipline** - UTC EOB, regular grid, no forward-fill, shift(1)
- **Verify everything** - no claims without proof

This repository represents a sophisticated, production-ready forecasting system built around NeuralForecast best practices. Every decision should reinforce the core principles of data discipline, leakage prevention, and NF-native implementation.