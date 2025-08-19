# Neural-Forecast Debugging Guide

## Overview

This guide provides debugging strategies and context7 queries for common error scenarios in the Neural-Forecast system. It follows the Agent-Oriented Error Recovery Protocols from the design specification.

## Quick Reference

| Error Type | Detection | Primary Recovery | Context7 Query |
|------------|-----------|------------------|----------------|
| GPU OOM | RuntimeError, kernel crash | Reduce batch_size by 50% | `neuralforecast memory optimization` |
| sCRPS Failure | NaN values, import errors | Check NaN/Inf, use fallback | `neuralforecast sCRPS import path` |
| Coverage Issues | Outside ±2% tolerance | Add conformal, check intervals | `neuralforecast conformal prediction` |
| Quantile Crossing | q90 < q50 | Switch to IQLoss | `neuralforecast IQLoss quantile crossing` |
| PIT Non-uniform | KS test p < 0.05 | Check quantile grid | `probability integral transform quantile` |

## Error Scenarios & Recovery

### 1. Memory Exhaustion During CV

**Detection Signs:**
- `RuntimeError: CUDA out of memory`
- Kernel crashes or restarts
- CV hanging indefinitely
- `torch.cuda.OutOfMemoryError`

**Recovery Protocol:**

```python
# Step 1: Check current batch_size in experiments/*.yaml
import yaml
with open('experiments/h4.yaml') as f:
    config = yaml.safe_load(f)
print(f"Current batch_size: {config.get('batch_size', 512)}")

# Step 2: Reduce batch_size by 50%
config['batch_size'] = config.get('batch_size', 512) // 2

# Step 3: Monitor GPU memory during execution
import torch
if torch.cuda.is_available():
    print(f"GPU memory: {torch.cuda.memory_allocated()/1e9:.2f}GB used")
    print(f"GPU memory: {torch.cuda.memory_reserved()/1e9:.2f}GB reserved")
```

**Progressive Reduction Strategy:**
1. **Attempt 1**: batch_size → batch_size // 2
2. **Attempt 2**: batch_size // 4 + gradient accumulation
3. **Attempt 3**: Reduce n_windows from 10 to 6
4. **Attempt 4**: Enable mixed precision (AMP)
5. **Final**: Switch to CPU with minimal config

**Context7 Queries:**
- `neuralforecast memory optimization batch_size`
- `neuralforecast inference_windows_batch_size GPU`
- `neuralforecast gradient accumulation training`
- `neuralforecast mixed precision AMP`

### 2. sCRPS Computation Failures

**Detection Signs:**
- `ValueError: NaN values in sCRPS`
- `ImportError: cannot import name 'sCRPS'`
- Metric computation crashes
- Non-finite metric values

**Recovery Protocol:**

```python
# Step 1: Verify correct import path
try:
    from neuralforecast.losses.pytorch import sCRPS
    print("sCRPS import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    # Fallback to MAE-based approximation

# Step 2: Check for NaN/Inf in data
import numpy as np
has_nan = np.isnan(y_true).any() or np.isnan(predictions).any()
has_inf = np.isinf(y_true).any() or np.isinf(predictions).any()
print(f"NaN: {has_nan}, Inf: {has_inf}")

# Step 3: Fix numerical issues
if has_nan or has_inf:
    # Winsorize extreme values
    from scipy.stats import mstats
    y_true = mstats.winsorize(y_true, limits=(0.001, 0.001))
    
# Step 4: Use fallback metric if needed
if scrps_failed:
    # Use MAE as proxy
    mae = np.mean(np.abs(y_true - y_pred))
    scale = np.mean(np.abs(y_true))
    fallback_scrps = mae / scale if scale > 0 else mae
```

**Context7 Queries:**
- `neuralforecast sCRPS import losses.pytorch`
- `neuralforecast sCRPS numerical stability`
- `neuralforecast metric computation NaN handling`
- `neuralforecast StudentT distribution heavy tails`

### 3. Coverage Calibration Issues

**Detection Signs:**
- Coverage outside target ±2% (e.g., 80% ±2%)
- Interval crossing (lo-90 > lo-80)
- Missing interval columns
- U-shaped or inverse-U PIT histograms

**Recovery Protocol:**

```python
# Step 1: Check interval columns exist
expected_cols = ['Model-lo-80', 'Model-hi-80', 'Model-lo-90', 'Model-hi-90']
missing = [col for col in expected_cols if col not in predictions.columns]
print(f"Missing columns: {missing}")

# Step 2: Verify level parameter in CV
if 'level' not in cv_config:
    cv_config['level'] = [80, 90, 95]

# Step 3: Add conformal prediction if needed
from neuralforecast.utils import PredictionIntervals
pi = PredictionIntervals(
    n_windows=6,
    method='conformal_error'
)

# Step 4: Check for quantile crossing
for i, row in predictions.iterrows():
    if row['Model-lo-90'] > row['Model-lo-80']:
        print(f"Quantile crossing at row {i}")
        # Fix: Switch to IQLoss
```

**Context7 Queries:**
- `neuralforecast conformal prediction intervals`
- `neuralforecast PredictionIntervals configuration`
- `neuralforecast IQLoss monotonic quantiles`
- `neuralforecast level parameter cross_validation`

### 4. PIT Computation Errors

**Detection Signs:**
- Non-uniform PIT histogram
- Interpolation failures
- `ValueError: invalid quantile grid`
- KS test p-value < 0.05

**Recovery Protocol:**

```python
# Step 1: Verify quantile grid
quantiles = [i/100 for i in range(1, 100)]
print(f"Quantile grid size: {len(quantiles)}")

# Step 2: Perform KS test for uniformity
from scipy import stats
ks_stat, ks_pval = stats.kstest(pit_values, 'uniform')
print(f"KS test p-value: {ks_pval:.4f}")

# Step 3: Analyze PIT shape
import matplotlib.pyplot as plt
plt.hist(pit_values, bins=10, density=True)
plt.axhline(y=1.0, color='r', linestyle='--')
plt.title(f"PIT Histogram (KS p={ks_pval:.3f})")

# Step 4: Skip PIT for conformal models
if using_conformal:
    print("PIT not available for conformal prediction")
    # Use coverage metrics instead
```

**Context7 Queries:**
- `probability integral transform quantile interpolation`
- `neuralforecast distributional calibration diagnostics`
- `scipy.stats.kstest uniformity test`
- `neuralforecast PIT histogram interpretation`

### 5. CV Window Configuration Errors

**Detection Signs:**
- `ValueError: insufficient data points`
- Overlapping CV windows
- step_size != h
- val_size != 4*h

**Recovery Protocol:**

```python
# Step 1: Calculate minimum data requirements
min_data = (n_windows - 1) * step_size + val_size + input_size
print(f"Minimum data points needed: {min_data}")
print(f"Available data points: {len(df)}")

# Step 2: Verify relationships
assert step_size == h, f"step_size ({step_size}) should equal h ({h})"
assert val_size == 4 * h, f"val_size ({val_size}) should be 4*h ({4*h})"

# Step 3: Adjust n_windows if needed
if len(df) < min_data:
    max_windows = (len(df) - val_size - input_size) // step_size + 1
    n_windows = min(n_windows, max_windows)
    print(f"Adjusted n_windows to {n_windows}")

# Step 4: Reduce input_size as last resort
if still_insufficient:
    input_size = min(256, input_size // 2)
```

**Context7 Queries:**
- `neuralforecast cross validation window parameters`
- `neuralforecast n_windows step_size configuration`
- `neuralforecast val_size validation size`
- `neuralforecast minimum data requirements CV`

## Risk Mitigation Checks

### MTF Misalignment Prevention

```python
from utils.risk_mitigation import MTFAlignmentValidator

validator = MTFAlignmentValidator()

# Always use these parameters for resampling
resampled = df.resample('1h', label='right', closed='right').last()

# Validate MTF features
results = validator.validate_mtf_features(df, '15min', ['30min', '1h', '4h'])
```

### Data Leakage Detection

```python
from utils.risk_mitigation import DataLeakageDetector

detector = DataLeakageDetector()

# Check correlation patterns
suspicious = detector.check_correlation_leakage(features, target)

# Verify shift(1) applied
shift_results = detector.verify_historical_shift(df, hist_cols)
```

### Quantile Crossing Fix

```python
from utils.risk_mitigation import QuantileCrossingFixer

fixer = QuantileCrossingFixer()

# Detect crossing
has_crossing, violations = fixer.detect_crossing(predictions, quantile_cols)

# Fix crossing
if has_crossing:
    fixed = fixer.fix_crossing_simple(predictions, quantile_cols)
    
    # Consider switching to IQLoss
    if fixer.recommend_loss_switch(len(violations), len(predictions)):
        print("Switch from MQLoss to IQLoss recommended")
```

### GPU Memory Monitoring

```python
from utils.risk_mitigation import GPUMemoryManager

gpu_manager = GPUMemoryManager()

# Get memory status
mem_info = gpu_manager.get_gpu_memory_info()
print(f"GPU free: {mem_info['free_gb']:.2f}GB")

# Suggest batch size
suggested = gpu_manager.suggest_batch_size(
    model_size_mb=500,
    current_batch=512
)

# Progressive reduction if OOM
config = gpu_manager.progressive_reduction(config, stage=1)
```

## Debugging Workflow

### Step 1: Initial Diagnosis

1. Check error type and message
2. Review recent changes
3. Verify data quality
4. Check GPU/memory status

### Step 2: Apply Recovery Protocol

1. Use appropriate recovery strategy from above
2. Start with least invasive fix
3. Progress to more aggressive solutions
4. Document what worked

### Step 3: Verify Fix

1. Re-run failed operation
2. Check metrics and outputs
3. Validate results quality
4. Monitor for recurrence

### Step 4: Prevention

1. Add validation checks
2. Update configuration defaults
3. Document in comments
4. Add to CI/CD tests

## Common Context7 Queries

### General NeuralForecast
- `neuralforecast getting started`
- `neuralforecast model selection guide`
- `neuralforecast best practices`

### Models & Training
- `neuralforecast NHITS configuration`
- `neuralforecast NBEATSx parameters`
- `neuralforecast TiDE implementation`
- `neuralforecast PatchTST memory requirements`

### Loss Functions
- `neuralforecast DistributionLoss StudentT`
- `neuralforecast MQLoss quantile regression`
- `neuralforecast IQLoss monotonic quantiles`

### Cross-Validation
- `neuralforecast cross_validation method`
- `neuralforecast refit parameter`
- `neuralforecast validation strategies`

### Uncertainty Quantification
- `neuralforecast prediction intervals`
- `neuralforecast conformal prediction`
- `neuralforecast probabilistic forecasting`

## Error Reporting Template

When reporting errors, include:

```markdown
### Error Summary
- **Error Type**: [e.g., GPU OOM, sCRPS failure]
- **Error Message**: [exact error text]
- **When**: [during CV, training, prediction]
- **Model**: [NHITS, NBEATSx, etc.]
- **Config**: [horizon, batch_size, etc.]

### Environment
- GPU: [model, memory]
- PyTorch: [version]
- NeuralForecast: [version]
- Data shape: [rows, features]

### What I Tried
1. [First recovery attempt]
2. [Second recovery attempt]
3. [What worked/didn't work]

### Logs
```
[relevant log output]
```
```

## Quick Fixes Checklist

- [ ] Clear GPU cache: `torch.cuda.empty_cache()`
- [ ] Reduce batch_size by 50%
- [ ] Check for NaN/Inf in data
- [ ] Verify shift(1) on historical features
- [ ] Use label='right', closed='right' for MTF
- [ ] Check quantile monotonicity
- [ ] Verify CV window parameters
- [ ] Try fallback metrics if sCRPS fails
- [ ] Consider conformal prediction for coverage
- [ ] Switch to IQLoss if quantile crossing persists

## References

- [NeuralForecast Documentation](https://nixtla.github.io/neuralforecast/)
- [Design Specification](../docs/forecasting_sf_plan.md)
- [Error Recovery Module](../utils/error_recovery.py)
- [Risk Mitigation Module](../utils/risk_mitigation.py)