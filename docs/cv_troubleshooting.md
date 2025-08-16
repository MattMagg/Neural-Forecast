# Cross-Validation Troubleshooting Guide

## Common Issues and Solutions

### 1. Data Issues

#### Issue: "AssertionError: Grid is not regular"

**Symptoms:**
```
AssertionError: Grid is not regular. Found gaps at: [timestamp1, timestamp2, ...]
```

**Causes:**
- Missing timestamps in the data
- Incorrect frequency specification
- Data not properly resampled to 15-minute intervals

**Solutions:**

```python
# 1. Check for gaps
from utils.validate import find_grid_gaps

gaps = find_grid_gaps(df, "15min")
if gaps:
    print(f"Found {len(gaps)} gaps in data")
    
    # Option A: Forward fill small gaps (< 1 hour)
    df = df.set_index('ds').resample('15min').ffill().reset_index()
    
    # Option B: Remove periods with gaps
    for gap_start, gap_end in gaps:
        df = df[(df['ds'] < gap_start) | (df['ds'] > gap_end)]

# 2. Ensure correct frequency
df = df.set_index('ds').asfreq('15min').reset_index()

# 3. Re-validate
from utils.validate import assert_regular_grid
assert_regular_grid(df, "15min")
```

#### Issue: "ValueError: unique_id column not found"

**Symptoms:**
```
ValueError: DataFrame must have columns: ['unique_id', 'ds', 'y']
```

**Solutions:**

```python
# Add unique_id column (required by NeuralForecast)
df['unique_id'] = 'BTC'

# Verify required columns exist
required_cols = ['unique_id', 'ds', 'y']
missing = [col for col in required_cols if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")
```

#### Issue: "Features have forward-looking bias"

**Symptoms:**
```
AssertionError: Column 'rsi' appears to have forward-looking bias
```

**Solutions:**

```python
# Shift all historical features by 1 period
hist_cols = [col for col in df.columns if col not in ['unique_id', 'ds', 'y']]
for col in hist_cols:
    df[f'{col}_shifted'] = df.groupby('unique_id')[col].shift(1)
    
# Drop original unshifted columns
df = df.drop(columns=hist_cols)

# Rename shifted columns back
df.columns = df.columns.str.replace('_shifted', '')
```

### 2. Model Issues

#### Issue: "CUDA out of memory"

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X.XX GiB
```

**Solutions:**

```python
# 1. Reduce batch size
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    model_cfg[model_name]['batch_size'] = 128  # Reduce from 512

# 2. Reduce input size
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    if 'input_size' in model_cfg[model_name]:
        model_cfg[model_name]['input_size'] = 512  # Reduce from 1024

# 3. Use CPU instead
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''  # Force CPU usage

# 4. Clear GPU cache between models
import torch
torch.cuda.empty_cache()

# 5. Run models sequentially
for model_cfg in cfg['models'][:1]:  # Run one model at a time
    models = create_models({'models': [model_cfg]})
    nf = NeuralForecast(models=models, freq='15min')
    cv_results = run_cv(nf, df, cfg)
```

#### Issue: "Model predictions are all NaN"

**Symptoms:**
- CV results contain NaN values for model predictions
- Metrics computation fails

**Solutions:**

```python
# 1. Check for NaN in input data
nan_cols = df.columns[df.isna().any()].tolist()
if nan_cols:
    print(f"NaN values found in: {nan_cols}")
    # Fill or drop NaN values
    df = df.fillna(method='ffill').fillna(0)

# 2. Check for infinite values
inf_cols = df.columns[np.isinf(df.select_dtypes(include=[np.number])).any()].tolist()
if inf_cols:
    print(f"Inf values found in: {inf_cols}")
    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

# 3. Scale features to reasonable range
from sklearn.preprocessing import RobustScaler
scaler = RobustScaler()
feature_cols = [col for col in df.columns if col not in ['unique_id', 'ds', 'y']]
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# 4. Reduce learning rate
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    model_cfg[model_name]['learning_rate'] = 0.0001  # Reduce from 0.001
```

#### Issue: "Training loss is NaN"

**Symptoms:**
```
Training loss: nan
Model training stopped early
```

**Solutions:**

```python
# 1. Use gradient clipping
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    model_cfg[model_name]['gradient_clip_val'] = 1.0

# 2. Check target variable scale
print(f"Target stats: mean={df['y'].mean():.6f}, std={df['y'].std():.6f}")
if df['y'].std() < 1e-6:
    print("Warning: Target has very low variance")
    # Scale target
    df['y'] = df['y'] * 1000

# 3. Use more stable loss function
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    # Switch from StudentT to Gaussian temporarily
    model_cfg[model_name]['loss'] = {'kind': 'normal'}
```

### 3. Cross-Validation Issues

#### Issue: "CV taking too long"

**Symptoms:**
- CV runs for hours without completing
- Memory usage keeps increasing

**Solutions:**

```python
# 1. Reduce number of windows for testing
test_cfg = cfg.copy()
test_cfg['n_windows'] = 2  # Start with just 2 windows

# 2. Reduce validation size
test_cfg['val_size'] = 32  # Reduce from 64

# 3. Use subset of data
df_subset = df.tail(10000)  # Use last 10k rows only

# 4. Disable refit (for testing only)
test_cfg['refit'] = False  # Models won't retrain each window

# 5. Add progress monitoring
import logging
logging.getLogger('neuralforecast').setLevel(logging.INFO)

# 6. Run with timeout
import signal
from contextlib import contextmanager

@contextmanager
def timeout(duration):
    def handler(signum, frame):
        raise TimeoutError("CV took too long")
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(duration)
    try:
        yield
    finally:
        signal.alarm(0)

# Use 1-hour timeout
with timeout(3600):
    cv_results = run_cv(nf, df, test_cfg)
```

#### Issue: "Coverage metrics are way off"

**Symptoms:**
- Coverage at 80% level is <70% or >90%
- All models showing similar miscalibration

**Solutions:**

```python
# 1. Check if models are actually producing intervals
cv_cols = cv_results.columns.tolist()
interval_cols = [col for col in cv_cols if '-lo-' in col or '-hi-' in col]
if not interval_cols:
    print("No interval predictions found!")
    # Ensure level parameter is passed
    cv_results = run_cv(nf, df, cfg, level=[80, 90, 95])

# 2. Verify interval width
for model in model_names:
    lo_col = f'{model}-lo-80'
    hi_col = f'{model}-hi-80'
    if lo_col in cv_results.columns:
        width = (cv_results[hi_col] - cv_results[lo_col]).mean()
        print(f"{model} 80% interval width: {width:.6f}")
        if width < 1e-6:
            print(f"Warning: {model} intervals are too narrow!")

# 3. Check distribution parameters for StudentT models
# If df (degrees of freedom) is too high, intervals will be too narrow
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    if model_cfg[model_name].get('loss', {}).get('kind') == 'studentt':
        # Consider adding explicit df parameter
        model_cfg[model_name]['loss']['df'] = 4  # Lower df = heavier tails

# 4. Enable dropout for uncertainty
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    if 'dropout' in model_cfg[model_name]:
        model_cfg[model_name]['dropout'] = 0.2  # Increase dropout
```

### 4. Memory Issues

#### Issue: "MemoryError during CV"

**Solutions:**

```python
import gc
import psutil

# 1. Monitor memory usage
def print_memory():
    process = psutil.Process()
    mem = process.memory_info().rss / 1e9
    print(f"Memory usage: {mem:.2f} GB")

# 2. Run CV in chunks
def run_cv_chunked(nf, df, cfg, chunk_size=2):
    """Run CV in chunks to manage memory."""
    all_results = []
    n_windows = cfg['n_windows']
    
    for i in range(0, n_windows, chunk_size):
        chunk_cfg = cfg.copy()
        chunk_cfg['n_windows'] = min(chunk_size, n_windows - i)
        chunk_cfg['cv_start_offset'] = i  # Custom parameter
        
        print(f"Running windows {i} to {i + chunk_cfg['n_windows']}")
        chunk_results = run_cv(nf, df, chunk_cfg)
        all_results.append(chunk_results)
        
        # Clear memory
        del chunk_results
        gc.collect()
        print_memory()
    
    return pd.concat(all_results, ignore_index=True)

# 3. Reduce number of models
# Run each model separately
for model_cfg in cfg['models']:
    single_model_cfg = cfg.copy()
    single_model_cfg['models'] = [model_cfg]
    
    models = create_models(single_model_cfg)
    nf = NeuralForecast(models=models, freq='15min')
    nf.fit(df)
    
    cv_results = run_cv(nf, df, single_model_cfg)
    
    # Save immediately
    model_name = list(model_cfg.keys())[0]
    cv_results.to_parquet(f'cv_results_{model_name}.parquet')
    
    # Clear memory
    del nf, models, cv_results
    gc.collect()
```

### 5. Metric Computation Issues

#### Issue: "sCRPS computation returns inf"

**Solutions:**

```python
# 1. Check for zero variance in predictions
for model in model_names:
    pred_col = f'{model}'
    if pred_col in cv_results.columns:
        pred_std = cv_results[pred_col].std()
        if pred_std < 1e-10:
            print(f"Warning: {model} has zero variance predictions")

# 2. Handle edge cases in sCRPS
def safe_scrps(y_true, y_pred, **kwargs):
    """Compute sCRPS with safety checks."""
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred).any(axis=-1))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) == 0:
        return np.nan
    
    # Check for zero variance
    if y_true_clean.std() < 1e-10:
        return np.nan
    
    return compute_scrps(y_true_clean, y_pred_clean, **kwargs)

# 3. Use robust scaling for normalization
mae_robust = np.median(np.abs(y_true - np.median(y_true)))
if mae_robust > 0:
    scrps = crps / mae_robust
else:
    scrps = crps  # Fall back to unnormalized
```

### 6. File I/O Issues

#### Issue: "Permission denied when saving results"

**Solutions:**

```python
import os
from pathlib import Path

# 1. Check directory permissions
output_dir = Path(f'experiments/h{horizon}')
if not output_dir.exists():
    output_dir.mkdir(parents=True, exist_ok=True)
    
# 2. Check write permissions
if not os.access(output_dir, os.W_OK):
    print(f"No write permission for {output_dir}")
    # Use alternative directory
    output_dir = Path.home() / 'tmp' / f'h{horizon}'
    output_dir.mkdir(parents=True, exist_ok=True)

# 3. Handle file locks
output_file = output_dir / 'cv_results.parquet'
if output_file.exists():
    # Create backup
    backup_file = output_dir / f'cv_results_backup_{int(time.time())}.parquet'
    shutil.copy(output_file, backup_file)
    
# 4. Use try-except for saving
try:
    cv_results.to_parquet(output_file)
except Exception as e:
    print(f"Failed to save to parquet: {e}")
    # Try CSV as fallback
    cv_results.to_csv(output_file.with_suffix('.csv'))
```

## Performance Optimization Tips

### 1. Speed Optimizations

```python
# Use these settings for faster CV during development

# Reduce training iterations
for model_cfg in cfg['models']:
    model_name = list(model_cfg.keys())[0]
    model_cfg[model_name]['max_steps'] = 1000  # Reduce from 20000
    model_cfg[model_name]['val_check_steps'] = 50  # Check more frequently

# Use smaller validation set
cfg['val_size'] = 32  # Reduce from 64

# Use fewer CV windows
cfg['n_windows'] = 3  # Reduce from 6

# Disable some models
cfg['models'] = cfg['models'][:2]  # Use only first 2 models
```

### 2. Memory Optimizations

```python
# Settings for memory-constrained environments

# Use float32 instead of float64
df = df.astype({col: 'float32' for col in df.select_dtypes('float64').columns})

# Reduce feature count
from features.selection import select_top_features
top_features = select_top_features(df, n=50)  # Keep only top 50
df = df[['unique_id', 'ds', 'y'] + top_features]

# Use data generator for large datasets
def create_data_generator(df, batch_size=10000):
    """Yield data in batches."""
    for i in range(0, len(df), batch_size):
        yield df.iloc[i:i+batch_size]
```

### 3. GPU Optimizations

```python
import torch

# Check GPU availability
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # Set GPU options
    torch.cuda.set_per_process_memory_fraction(0.8)  # Use max 80% of GPU memory
    
    # Enable mixed precision training
    for model_cfg in cfg['models']:
        model_name = list(model_cfg.keys())[0]
        model_cfg[model_name]['use_amp'] = True  # Automatic mixed precision
else:
    print("No GPU available, using CPU")
    # Reduce batch sizes for CPU
    for model_cfg in cfg['models']:
        model_name = list(model_cfg.keys())[0]
        model_cfg[model_name]['batch_size'] = 32
```

## Getting Help

If you encounter issues not covered here:

1. Check the [NeuralForecast GitHub Issues](https://github.com/Nixtla/neuralforecast/issues)
2. Review the [official documentation](https://nixtla.github.io/neuralforecast/)
3. Enable debug logging:
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
4. Create a minimal reproducible example:
   ```python
   # Minimal example that reproduces the issue
   import pandas as pd
   from neuralforecast import NeuralForecast
   from neuralforecast.models import NHITS
   
   # Small synthetic data
   df = pd.DataFrame({
       'unique_id': ['BTC'] * 1000,
       'ds': pd.date_range('2024-01-01', periods=1000, freq='15min'),
       'y': np.random.randn(1000) * 0.01
   })
   
   # Simple model
   model = NHITS(h=4, input_size=96, loss={'kind': 'normal'})
   nf = NeuralForecast(models=[model], freq='15min')
   
   # Try to reproduce issue
   nf.fit(df)
   # ... your issue here
   ```