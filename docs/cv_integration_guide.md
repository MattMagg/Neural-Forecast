# Cross-Validation Integration Guide

This document describes the integration of the CV modules into the training pipeline, following the pattern specified in `docs/forecasting_sf_plan.md` lines 1770-1801.

## Overview

The CV integration connects the cross-validation infrastructure with the main training pipeline, providing:
- Automated model training and evaluation
- Progress logging and error recovery
- Comprehensive metrics computation
- Model selection and saving

## Integration Architecture

```
run_train.py
    ├── Load Data (utils/io.py)
    ├── Feature Engineering (features/)
    ├── Model Instantiation (nf_models/factory.py)
    ├── CV Execution (cv/runner.py)
    │   ├── run_cv() - Execute NF-native CV
    │   └── summarize_cv() - Compute metrics
    ├── Model Selection (cv/runner.py)
    └── Save Artifacts (experiments/h{horizon}/)
```

## Key Integration Points

### 1. Configuration Loading

```python
# Load experiment configuration
cfg = load_experiment_config("experiments/h4.yaml")
```

The configuration contains:
- `h`: Forecast horizon (4, 8, 16, 32)
- `n_windows`: Number of CV windows (6 for pilot, 10 for final)
- `step_size`: Window step size (must equal h)
- `val_size`: Validation size (must equal 4*h)
- `models`: Model configurations with losses

### 2. Training Script Integration

The modified `run_train.py` now includes:

```python
from cv.runner import run_cv, summarize_cv
from nf_models.factory import instantiate_models
from neuralforecast import NeuralForecast

# After data processing and feature engineering...

# Step 1: Instantiate models
models = instantiate_models(cfg, verbose=True)
nf = NeuralForecast(models=models, freq=cfg['freq'])

# Step 2: Fit once to store dataset
nf.fit(df=nf_df, val_size=cfg["val_size"])

# Step 3: Optional insample predictions
ins = nf.predict_insample(step_size=1, level=[10,20,30,40,50,60,70,80,90])

# Step 4: Run cross-validation
cv_df = run_cv(nf, nf_df, cfg, use_conformal=False)

# Step 5: Summarize results
results = summarize_cv(cv_df, models, h=cfg['h'], ...)

# Step 6: Save artifacts
cv_df.to_parquet(f"experiments/h{cfg['h']}/cv_raw.parquet")
results['leaderboard'].to_parquet(f"experiments/h{cfg['h']}/leaderboard.parquet")
```

### 3. Progress Logging

The integration provides detailed progress logging at each step:

```
==================================================================
STEP 1: Model Instantiation
==================================================================
Successfully instantiated 4 models

==================================================================
STEP 2: Initial Model Fitting
==================================================================
Fitting 4 models with val_size=64...
✅ Model fitting completed in 45.2 seconds

==================================================================
STEP 4: Cross-Validation
==================================================================
Starting 6-window cross-validation...
Running pre-CV validation checks...
Cross-validation completed: 1536 predictions generated
✅ Cross-validation completed in 180.3 seconds
```

### 4. Error Recovery

The integration includes comprehensive error handling:

#### Retry Logic for Transient Failures
```python
def run_cv_with_progress(nf, df, cfg, max_retries=3):
    """Run CV with retry logic for transient failures."""
    for attempt in range(max_retries):
        try:
            return run_cv(nf, df, cfg)
        except Exception as e:
            if is_retryable(e) and attempt < max_retries - 1:
                wait_time = attempt * 5  # Exponential backoff
                time.sleep(wait_time)
            else:
                raise
```

#### Partial Failure Recovery
- Saves completed CV windows before raising exceptions
- Provides informative error messages with debugging hints
- Suggests solutions for common errors (OOM, shape mismatches)

### 5. Command-Line Interface

The enhanced CLI supports:

```bash
# Full training with CV
python run_train.py \
    --config experiments/h4.yaml \
    --output-dir data/processed \
    --save-models

# With conformal prediction
python run_train.py \
    --config experiments/h4.yaml \
    --use-conformal \
    --save-models

# Skip CV for testing data processing
python run_train.py \
    --config experiments/h4.yaml \
    --skip-cv \
    --save-processed
```

## Testing the Integration

A test script is provided to verify the integration:

```bash
# Run integration test with synthetic data
python test_cv_integration.py
```

The test script:
1. Creates a small synthetic dataset
2. Runs the complete pipeline with reduced settings
3. Verifies all outputs are generated correctly
4. Reports success/failure with detailed logs

## Output Structure

After successful execution, the following artifacts are created:

```
experiments/h{horizon}/
├── cv_raw.parquet              # Raw CV predictions
├── leaderboard.parquet          # Model rankings
├── cv_metrics_h{horizon}.csv    # Detailed metrics
├── cv_summary_h{horizon}.md     # Markdown report
├── training_config.yaml         # Configuration used
├── models/                      # Saved models
│   ├── best_model_h{horizon}_*/
│   ├── best_distributional_h{horizon}_*/
│   └── best_quantile_h{horizon}_*/
└── plots/                       # Calibration plots (if enabled)
```

## Performance Benchmarks

Expected performance on a GPU-enabled system:
- Data processing: < 30 seconds
- Model instantiation: < 5 seconds  
- Initial fitting: 30-60 seconds per model
- CV execution: 2-5 minutes per window
- Total for 6 windows: 15-30 minutes

## Troubleshooting

### Common Issues and Solutions

1. **Out of Memory (OOM)**
   - Reduce `batch_size` in model configs
   - Reduce `input_size` for models
   - Use fewer CV windows (`n_windows`)

2. **Shape Mismatches**
   - Verify all features are properly aligned
   - Check that `hist_exog_list` matches actual columns
   - Ensure regular 15-minute grid

3. **Slow Performance**
   - Reduce `max_steps` for testing
   - Use GPU acceleration if available
   - Disable plot generation for speed

4. **CV Failures**
   - Check that `step_size = h`
   - Verify `val_size = 4 * h`
   - Ensure sufficient data (>10 horizons)

## Next Steps

After successful CV integration:

1. **Run experiments for all horizons**:
   ```bash
   for h in 4 8 16 32; do
       python run_train.py --config experiments/h${h}.yaml
   done
   ```

2. **Analyze results**:
   - Review leaderboards in `experiments/h{horizon}/`
   - Check calibration plots
   - Compare model performance across horizons

3. **Deploy best models**:
   - Load saved models for inference
   - Use `run_predict.py` for production predictions
   - Monitor performance with production data

## References

- Main specification: `docs/forecasting_sf_plan.md` Section 5
- CV module implementation: `cv/runner.py`
- Model factory: `nf_models/factory.py`
- Integration test: `test_cv_integration.py`