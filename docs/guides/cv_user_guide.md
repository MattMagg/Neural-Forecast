# Cross-Validation User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Understanding Time Series CV](#understanding-time-series-cv)
3. [Configuration Guide](#configuration-guide)
4. [Running Cross-Validation](#running-cross-validation)
5. [Understanding Metrics](#understanding-metrics)
6. [Model Selection](#model-selection)
7. [Best Practices](#best-practices)
8. [Advanced Topics](#advanced-topics)

## Introduction

This guide provides comprehensive instructions for using the Neural-Forecast cross-validation system for BTC intraday forecasting. The system is built on NeuralForecast's native CV capabilities and focuses on rigorous model evaluation using proper time series validation techniques.

## Understanding Time Series CV

### Why Time Series CV is Different

Unlike standard cross-validation, time series CV must respect temporal ordering:

- **No random splits**: Future data cannot be used to predict the past
- **Expanding windows**: Training set grows with each fold
- **Gap handling**: Optional gap between train and test to simulate deployment lag

### Our CV Strategy

We use an expanding window approach with the following characteristics:

```
Window 1: [========TRAIN========][VAL]
Window 2: [==========TRAIN==========][VAL]
Window 3: [============TRAIN============][VAL]
...
```

Key parameters:
- **n_windows**: Number of CV folds (6 for pilot, 10 for production)
- **step_size**: Bars between window starts (equals horizon h)
- **val_size**: Validation set size (64 bars = 4 hours)
- **refit**: Always true - models retrain on each window

## Configuration Guide

### Basic Configuration Structure

```yaml
# experiments/h{horizon}.yaml
seed: 1337              # Reproducibility
freq: "15min"          # Data frequency
h: 4                   # Forecast horizon

# Cross-validation parameters
n_windows: 6           # Number of CV windows
step_size: 4           # Steps between windows (= h)
val_size: 64          # Validation size in bars
refit: true           # Retrain on each window

# Model portfolio
models:
  - NHITS:
      input_size: 1024
      loss: {kind: studentt}
      # ... other parameters
```

### Horizon-Specific Settings

| Horizon | Config File | h | step_size | Description |
|---------|-------------|---|-----------|-------------|
| 1 hour | h4.yaml | 4 | 4 | Ultra-short term |
| 2 hours | h8.yaml | 8 | 8 | Short term |
| 4 hours | h16.yaml | 16 | 16 | Medium term |
| 8 hours | h32.yaml | 32 | 32 | Long term |

### Pilot vs Production Settings

**Pilot Phase** (faster iteration):
```yaml
n_windows: 6
max_steps: 20000
```

**Production Phase** (final evaluation):
```yaml
n_windows: 10
max_steps: 50000
```

## Running Cross-Validation

### Step 1: Prepare Data

```python
import pandas as pd
from utils.validate import assert_regular_grid, assert_utc_eob

# Load your prepared data
df = pd.read_parquet('data/processed/btc_15min_canonical.parquet')

# Validate data quality
assert_regular_grid(df, "15min")
assert_utc_eob(df, "15min")

print(f"Data shape: {df.shape}")
print(f"Date range: {df['ds'].min()} to {df['ds'].max()}")
```

### Step 2: Load Configuration

```python
import yaml

horizon = 4  # 1-hour horizon
config_path = f'experiments/h{horizon}.yaml'

with open(config_path, 'r') as f:
    cfg = yaml.safe_load(f)
    
print(f"Loaded config for h={cfg['h']} ({cfg['h'] * 15} minutes)")
print(f"CV windows: {cfg['n_windows']}")
print(f"Models: {len(cfg['models'])} configured")
```

### Step 3: Initialize Models

```python
from neuralforecast import NeuralForecast
from nf_models.factory import create_models

# Create model instances from config
models = create_models(cfg)
print(f"Created {len(models)} model instances")

# Initialize NeuralForecast
nf = NeuralForecast(
    models=models,
    freq='15min'
)
```

### Step 4: Run Cross-Validation

```python
from cv.runner import run_cv, summarize_cv
import time

# Fit models first (required before CV)
print("Fitting models...")
start_time = time.time()
nf.fit(df, val_size=cfg['val_size'])
fit_time = time.time() - start_time
print(f"Fitting completed in {fit_time/60:.1f} minutes")

# Run cross-validation
print(f"Running CV with {cfg['n_windows']} windows...")
cv_results = run_cv(nf, df, cfg)
cv_time = time.time() - start_time - fit_time
print(f"CV completed in {cv_time/60:.1f} minutes")

# Generate summary
model_names = [m.alias for m in models]
summary = summarize_cv(cv_results, model_names)
```

### Step 5: Analyze Results

```python
# View leaderboard
print("\n=== Model Leaderboard ===")
print(summary['leaderboard'])

# Check best models
print(f"\nBest overall: {summary['best_models']['overall']}")
print(f"Best distributional: {summary['best_models']['distributional']}")
print(f"Best quantile: {summary['best_models']['quantile']}")

# Coverage analysis
print("\n=== Coverage Analysis ===")
for model in model_names:
    cov = summary['coverage'][model]
    print(f"\n{model}:")
    print(f"  80% coverage: {cov['80']:.1%} (target: 80±2%)")
    print(f"  90% coverage: {cov['90']:.1%} (target: 90±2%)")
    print(f"  95% coverage: {cov['95']:.1%} (target: 95±2%)")
```

## Understanding Metrics

### sCRPS (Scaled Continuous Ranked Probability Score)

**What it measures**: The quality of the entire predictive distribution, not just point forecasts.

**Formula**: sCRPS = CRPS / MAE(y)

**Interpretation**:
- Range: 0 to ∞ (lower is better)
- 0.0: Perfect predictions
- 0.3-0.5: Good performance on financial data
- 0.5-0.7: Acceptable performance
- \>0.7: Poor performance

**Why we use it**:
1. Evaluates full distribution, not just point estimates
2. Scale-free, allowing comparison across different periods
3. Proper scoring rule (incentivizes honest uncertainty estimates)

### Coverage Metrics

**What they measure**: How often true values fall within prediction intervals.

**Target tolerances**:
- 80% level: Should cover 78-82% of observations
- 90% level: Should cover 88-92% of observations
- 95% level: Should cover 93-97% of observations

**Diagnosing coverage issues**:

| Coverage | Diagnosis | Likely Cause | Solution |
|----------|-----------|--------------|----------|
| Too low | Overconfident | Variance underestimated | Increase dropout, use heavier-tailed distribution |
| Too high | Underconfident | Variance overestimated | Reduce regularization, tune distribution parameters |
| Varies by level | Miscalibrated | Distribution shape wrong | Try different loss function |

### MAE and RMSE

**MAE (Mean Absolute Error)**:
- Robust to outliers
- Same units as target
- Good for understanding typical error

**RMSE (Root Mean Squared Error)**:
- Sensitive to outliers
- Penalizes large errors more
- Good for risk assessment

**Relationship**: RMSE/MAE ratio indicates error distribution:
- ≈1.0: Errors are uniform
- ≈1.25: Errors are normally distributed
- \>1.5: Heavy-tailed errors or outliers

## Model Selection

### Selection Criteria

1. **Primary**: Lowest mean sCRPS
2. **Secondary**: Coverage within tolerance
3. **Tertiary**: Consistency (low std of sCRPS)

### Model Categories

**Distributional Models** (StudentT loss):
- Provide full probability distributions
- Better uncertainty quantification
- Support PIT analysis
- Generally preferred for risk-sensitive applications

**Quantile Models** (MQLoss/IQLoss):
- Direct quantile predictions
- No distribution assumptions
- More flexible for asymmetric risks
- Good for specific percentile requirements

### Ensemble Strategies

```python
from uq.ensembles import create_ensemble

# Simple average of top 3 models
top_3 = summary['leaderboard'].head(3).index.tolist()
ensemble_avg = create_ensemble(cv_results, top_3, method='mean')

# Weighted by inverse sCRPS
ensemble_weighted = create_ensemble(
    cv_results, 
    top_3, 
    method='weighted',
    weights='inverse_scrps'
)
```

## Best Practices

### 1. Data Validation

Always validate data before CV:

```python
from utils.validate import run_all_validations

# Run comprehensive checks
issues = run_all_validations(df)
if issues:
    print("Data issues found:")
    for issue in issues:
        print(f"  - {issue}")
    raise ValueError("Fix data issues before CV")
```

### 2. Incremental Testing

Start with small experiments:

```python
# Quick test with reduced data and iterations
test_cfg = cfg.copy()
test_cfg['n_windows'] = 2
test_cfg['models'][0]['NHITS']['max_steps'] = 100

# Run quick test
quick_results = run_cv(nf, df.tail(10000), test_cfg)
```

### 3. Monitor Progress

Enable detailed logging:

```python
import logging

# Set up comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'logs/cv_h{horizon}.log'),
        logging.StreamHandler()
    ]
)
```

### 4. Save Intermediate Results

```python
from pathlib import Path
import json

# Create output directory
output_dir = Path(f'experiments/h{horizon}')
output_dir.mkdir(exist_ok=True)

# Save after each window (in case of failure)
for window in range(cfg['n_windows']):
    window_results = run_cv_window(nf, df, cfg, window)
    window_results.to_parquet(
        output_dir / f'cv_window_{window}.parquet'
    )
```

### 5. Resource Management

Monitor and manage resources:

```python
import psutil
import torch

# Check available resources
print(f"CPU cores: {psutil.cpu_count()}")
print(f"RAM available: {psutil.virtual_memory().available / 1e9:.1f} GB")
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Adjust batch size if needed
if psutil.virtual_memory().available < 8e9:  # Less than 8GB RAM
    for model_cfg in cfg['models']:
        model_name = list(model_cfg.keys())[0]
        model_cfg[model_name]['batch_size'] = 256
```

## Advanced Topics

### Conformal Prediction

Enable conformal prediction intervals for better coverage:

```python
# Run CV with conformal intervals
cv_conformal = run_cv(nf, df, cfg, use_conformal=True)

# Note: Conformal intervals are computed post-hoc
# They guarantee coverage but may be wider than parametric intervals
```

### Custom Metrics

Add custom metrics to evaluation:

```python
def pinball_loss(y_true, y_pred, quantile):
    """Compute pinball loss for a specific quantile."""
    errors = y_true - y_pred
    return np.mean(np.maximum(quantile * errors, (quantile - 1) * errors))

# Add to summary
for model in model_names:
    pred_col = f'{model}_q50'
    if pred_col in cv_results.columns:
        summary['metrics'][model]['pinball_50'] = pinball_loss(
            cv_results['y'].values,
            cv_results[pred_col].values,
            0.5
        )
```

### Parallel Cross-Validation

Run CV for multiple horizons in parallel:

```python
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

def run_cv_horizon(horizon):
    """Run CV for a specific horizon."""
    # Load config
    with open(f'experiments/h{horizon}.yaml', 'r') as f:
        cfg = yaml.safe_load(f)
    
    # Create models and run CV
    models = create_models(cfg)
    nf = NeuralForecast(models=models, freq='15min')
    nf.fit(df)
    
    return run_cv(nf, df, cfg)

# Run all horizons in parallel
horizons = [4, 8, 16, 32]
with ProcessPoolExecutor(max_workers=mp.cpu_count()) as executor:
    results = dict(zip(horizons, executor.map(run_cv_horizon, horizons)))
```

### Debugging Failed CV Runs

If CV fails, use these debugging steps:

```python
# 1. Check data quality
from utils.validate import assert_regular_grid
try:
    assert_regular_grid(df, "15min")
except AssertionError as e:
    print(f"Data issue: {e}")
    # Fix data issues

# 2. Run single window
test_window = run_cv_single_window(nf, df, cfg, window_idx=0)
print(f"Single window shape: {test_window.shape}")

# 3. Check model predictions
for col in test_window.columns:
    if col.startswith(('NHITS', 'TIDE', 'NBEATS', 'PATCH')):
        null_count = test_window[col].isna().sum()
        if null_count > 0:
            print(f"Warning: {col} has {null_count} NaN values")

# 4. Verify memory usage
import tracemalloc
tracemalloc.start()
cv_results = run_cv(nf, df, cfg)
current, peak = tracemalloc.get_traced_memory()
print(f"Peak memory usage: {peak / 1e9:.2f} GB")
tracemalloc.stop()
```

## Next Steps

1. Review example notebooks in `examples/cv/`
2. Run pilot CV on your data (n_windows=6)
3. Analyze results and tune if needed
4. Run production CV (n_windows=10)
5. Select best models for deployment

For troubleshooting, see `docs/cv_troubleshooting.md`.