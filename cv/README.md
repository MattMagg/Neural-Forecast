# Cross-Validation Module

## Overview

The `cv` module provides NeuralForecast-native cross-validation capabilities for intraday BTC forecasting. It implements time series cross-validation with proper windowing, comprehensive metrics computation (sCRPS, coverage, calibration), and model ranking/selection.

## Key Features

- **NF-Native CV**: Uses `NeuralForecast.cross_validation()` exclusively - no custom backtesting
- **Proper Windowing**: Implements expanding window CV with configurable n_windows, step_size, val_size
- **Comprehensive Metrics**: sCRPS (primary), MAE, RMSE, coverage at 80%/90%/95% levels
- **Calibration Diagnostics**: PIT analysis for distributional models, coverage reliability
- **Model Ranking**: Automatic leaderboard generation based on sCRPS performance
- **Conformal Support**: Optional conformal prediction intervals via NF's PredictionIntervals

## Module Structure

```
cv/
├── __init__.py       # Module exports
├── runner.py         # Core CV execution and orchestration
├── hpo.py           # Hyperparameter optimization utilities
└── README.md        # This file
```

## Quick Start

### Basic Cross-Validation

```python
from neuralforecast import NeuralForecast
from cv.runner import run_cv, summarize_cv
import yaml

# Load configuration
with open('experiments/h4.yaml', 'r') as f:
    cfg = yaml.safe_load(f)

# Assume you have a fitted NeuralForecast instance
nf = NeuralForecast(models=models, freq='15min')
nf.fit(df)

# Run cross-validation
cv_results = run_cv(nf, df, cfg)

# Compute metrics and generate leaderboard
summary = summarize_cv(cv_results, model_names=['NHITS_t1024_T', 'TiDE_t1024_T'])
print(summary['leaderboard'])
```

### With Conformal Prediction

```python
# Enable conformal intervals
cv_results = run_cv(nf, df, cfg, use_conformal=True)

# Conformal intervals will be included in the results
# Note: PIT analysis is skipped for conformal models
```

## Configuration

Cross-validation parameters are specified in experiment YAML files:

```yaml
# experiments/h4.yaml
n_windows: 6        # Number of CV windows (6 for pilot, 10 for final)
step_size: 4        # Steps between windows (horizon-aligned)
val_size: 64        # Validation size (4 hours = 16 * 15min)
refit: true         # Refit models for each window
```

### Parameter Guidelines

| Horizon | h | step_size | val_size | n_windows (pilot) | n_windows (final) |
|---------|---|-----------|----------|-------------------|-------------------|
| 1 hour  | 4 | 4         | 64       | 6                 | 10                |
| 2 hours | 8 | 8         | 64       | 6                 | 10                |
| 4 hours | 16| 16        | 64       | 6                 | 10                |
| 8 hours | 32| 32        | 64       | 6                 | 10                |

## Metrics

### Primary Metric: sCRPS

Scaled Continuous Ranked Probability Score (sCRPS) is our primary metric for model selection:

- **What it measures**: Quality of probabilistic forecasts
- **Scale**: 0 to ∞ (lower is better)
- **Interpretation**: sCRPS = CRPS / MAE(y), providing scale-free comparison
- **Typical values**: 0.3-0.7 for good models on BTC data

### Coverage Metrics

Coverage measures how often true values fall within prediction intervals:

- **Target levels**: 80%, 90%, 95%
- **Tolerance**: ±2% from nominal
- **Good calibration**: 80% level covers 78-82% of observations
- **Undercoverage**: Model is overconfident
- **Overcoverage**: Model is too conservative

### Supporting Metrics

- **MAE**: Mean Absolute Error for point predictions
- **RMSE**: Root Mean Squared Error for error magnitude
- **Bias**: Systematic over/under prediction

## Model Selection

The module automatically ranks models and identifies:

1. **Best Overall**: Lowest mean sCRPS across CV windows
2. **Best Distributional**: Best model using StudentT loss
3. **Best Quantile**: Best model using MQLoss/IQLoss
4. **Most Calibrated**: Closest to nominal coverage rates

### Leaderboard Format

```
               sCRPS_mean  sCRPS_std  MAE    RMSE   Coverage_80  Coverage_90  Coverage_95
NHITS_t1024_T  0.421      0.052      0.0023  0.0031  0.812        0.908        0.951
TiDE_t1024_T   0.438      0.061      0.0024  0.0032  0.798        0.892        0.943
```

## Calibration Diagnostics

### PIT (Probability Integral Transform)

For distributional models, PIT values should be uniformly distributed [0,1]:

- **Interpretation**: U-shaped = underdispersed, ∩-shaped = overdispersed
- **KS Test**: Tests uniformity (p-value > 0.05 indicates good calibration)
- **Note**: PIT is skipped for conformal models (not meaningful)

### Coverage Reliability

Plots nominal vs empirical coverage across all levels to identify systematic miscalibration.

## Artifact Management

Results are automatically saved to:

```
experiments/h{horizon}/
├── cv_predictions_{timestamp}.parquet  # Raw CV predictions
├── metrics_summary_{timestamp}.json    # Aggregated metrics
├── leaderboard_{timestamp}.csv        # Model ranking
└── models/                            # Saved NF models
    ├── best_overall/
    ├── best_distributional/
    └── best_quantile/

reports/h{horizon}/
├── pit_histogram_{model}.png          # PIT diagnostic plots
├── coverage_reliability.png           # Coverage analysis
└── metrics_comparison.png             # Model comparison charts
```

## Performance Considerations

### Memory Management

- **Large datasets**: Use `step_size = h` to reduce overlap
- **Many models**: Run CV in batches if memory constrained
- **GPU OOM**: Reduce batch_size in model configs

### Speed Optimization

- **Parallel CV**: Set `n_jobs=-1` in NeuralForecast for parallel windows
- **Early stopping**: Configure `early_stop_patience_steps` to avoid overtraining
- **Selective saving**: Only save best models to reduce I/O

## Troubleshooting

### Common Issues

1. **"CV results missing columns"**
   - Ensure models are fitted before CV
   - Check that `level=[80, 90, 95]` is passed

2. **"Coverage outside tolerance"**
   - Normal for pilot runs (n_windows=6)
   - May indicate model miscalibration
   - Try adjusting loss function parameters

3. **"PIT computation failed"**
   - Only works for distributional models
   - Requires dense quantile predictions for quantile models
   - Skipped for conformal models (expected)

4. **"Memory error during CV"**
   - Reduce n_windows or val_size
   - Decrease model batch_size
   - Run models sequentially instead of all at once

## API Reference

### Core Functions

```python
run_cv(nf, df, cfg, use_conformal=False)
    Execute NF-native cross-validation
    
summarize_cv(cv_df, model_names)
    Compute metrics and generate leaderboard
    
validate_cv_results(cv_df, model_names, h, level)
    Validate CV output structure
```

### Metrics Functions

```python
compute_scrps(y_true, y_pred, quantiles, distribution, distribution_params)
    Compute scaled CRPS
    
compute_coverage(y_true, y_lower, y_upper)
    Calculate empirical coverage
    
compute_pit(y_true, distribution_params)
    Compute PIT values for calibration
```

## Examples

See the `examples/cv/` directory for detailed notebooks:

1. `01_basic_cv_example.ipynb` - Simple CV execution
2. `02_metrics_analysis.ipynb` - Understanding metrics
3. `03_calibration_diag.ipynb` - PIT and coverage analysis
4. `04_model_selection.ipynb` - Selecting best models
5. `05_performance_tuning.ipynb` - Optimization tips

## References

- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [sCRPS Paper](https://doi.org/10.1016/j.ijforecast.2021.05.008)
- [Time Series Cross-Validation](https://otexts.com/fpp3/tscv.html)