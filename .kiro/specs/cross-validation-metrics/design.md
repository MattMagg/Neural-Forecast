# Design Document

## Overview

The Cross-Validation and Metrics system is a lean, NF-native evaluation framework that leverages NeuralForecast's built-in cross_validation method (docs/forecasting_sf_plan.md lines 1583-1585, 1620-1628) for time-series backtesting and implements comprehensive metrics computation centered on sCRPS (lines 1655-1660). The design follows the "zero reinvention" principle, using NF's native capabilities for windowing, training, and prediction (Section 5.1, lines 1605-1635) while adding minimal glue code for metrics aggregation, calibration diagnostics, and visualization (Section 5.2, lines 1636-1825). The system integrates seamlessly with the model factory and feature engineering pipeline to provide end-to-end model evaluation capabilities.

## Architecture

### High-Level Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Trained NF    │    │   Experiment     │    │  Canonical      │
│   Models        │───▶│  Configuration   │───▶│  DataFrame      │
│                 │    │   (YAML)         │    │  (NF format)    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ NF.cross_       │───▶│  CV Results      │───▶│  Metrics        │
│ validation()    │    │  DataFrame       │    │  Computation    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│  Leaderboard    │◀───│  Calibration     │◀───│  Aggregation    │
│  Generation     │    │  Diagnostics     │    │  & Ranking      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
```

### Component Architecture

The system consists of four main architectural layers:

1. **Cross-Validation Layer**: NF-native windowing and evaluation execution
2. **Metrics Layer**: sCRPS computation, MAE/RMSE calculation, metric aggregation
3. **Diagnostics Layer**: Coverage analysis, PIT computation, calibration assessment
4. **Persistence Layer**: Results storage, model artifacts, diagnostic visualizations

### Design Principles

- **NF-Native**: Use NeuralForecast.cross_validation() exclusively, no custom backtesting (lines 1583-1585)
- **Metrics-Centric**: sCRPS as primary metric with comprehensive supporting metrics
- **Calibration-Focused**: Rigorous coverage and PIT diagnostics for uncertainty validation
- **Leakage-Free**: Strict time ordering with proper windowing parameters
- **Reproducible**: Deterministic evaluation with versioned artifacts

## Components and Interfaces

### Core CV Runner Module (`cv/runner.py`)

#### Primary Cross-Validation Interface

```python
def run_cv(nf: NeuralForecast, 
           df: pd.DataFrame, 
           cfg: Dict[str, Any]) -> pd.DataFrame:
    """
    Execute NF-native cross-validation with configured windowing.
    
    Args:
        nf: Fitted NeuralForecast instance with models
        df: Canonical frame with unique_id, ds, y, exog columns
        cfg: Configuration with n_windows, step_size, val_size
        
    Returns:
        DataFrame with CV predictions from all models and windows
    """
```

#### Metrics Aggregation Interface

```python
def summarize_cv(cv_df: pd.DataFrame,
                 models: List[str],
                 h: int) -> Dict[str, pd.DataFrame]:
    """
    Compute and aggregate metrics from CV results.
    
    Args:
        cv_df: Raw CV output from NeuralForecast
        models: List of model names to evaluate
        h: Forecast horizon for this experiment
        
    Returns:
        Dictionary with 'metrics', 'leaderboard', 'coverage' DataFrames
    """
```

### Metrics Computation Module (`uq/metrics.py`)

#### sCRPS Computation

```python
def compute_scrps(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  quantiles: Optional[np.ndarray] = None,
                  distribution: Optional[str] = None) -> float:
    """
    Compute scaled CRPS using NF's native implementation.
    Import: from neuralforecast.losses.pytorch import sCRPS
    
    Args:
        y_true: Actual values
        y_pred: Predictions (point or distributional)
        quantiles: For quantile models, the quantile levels
        distribution: For distributional models, the distribution type
        
    Returns:
        Scaled CRPS score (lower is better)
    """
```

#### Coverage Analysis

```python
def compute_coverage(cv_df: pd.DataFrame,
                    levels: List[int] = [80, 90, 95]) -> pd.DataFrame:
    """
    Compute empirical coverage at specified confidence levels.
    
    Args:
        cv_df: CV results with interval predictions
        levels: Confidence levels to evaluate
        
    Returns:
        DataFrame with model, level, coverage, and deviation
    """
```

### Calibration Diagnostics Module (`uq/calibration.py`)

#### PIT Analysis

```python
def compute_pit(y_true: np.ndarray,
                predictions: Dict[str, np.ndarray],
                model_type: str) -> np.ndarray:
    """
    Compute Probability Integral Transform values.
    For quantile models, use dense grid: quantiles = [i/100 for i in range(1, 100)]
    
    Args:
        y_true: Actual values
        predictions: Model predictions (quantiles or distribution params)
        model_type: 'distributional', 'quantile', or 'conformal'
        
    Returns:
        PIT values for uniformity testing
    """
```

#### Calibration Visualization

```python
def plot_calibration_diagnostics(pit_values: np.ndarray,
                                 coverage_df: pd.DataFrame,
                                 output_dir: Path) -> None:
    """
    Generate calibration diagnostic plots.
    
    Args:
        pit_values: PIT values for histogram
        coverage_df: Coverage analysis results
        output_dir: Directory for saving plots
    """
```

## Data Flow

### Cross-Validation Execution Flow

1. **Input Preparation**
   - Load canonical DataFrame with features
   - Validate data quality (regular grid, UTC timestamps, shifted features)
   - Configure CV parameters from YAML

2. **NF Cross-Validation**
   - Call NeuralForecast.cross_validation() with configured parameters (lines 1620-1628)
   - Use n_windows=6 (pilot) or 10 (final) (lines 1589, 1609)
   - Set step_size=h for non-overlapping windows (lines 1591, 1610)
   - Set val_size=4*h for internal validation (lines 1593, 1611)
   - Enable refit=1 for realistic evaluation (lines 1595, 1612)

3. **Results Processing**
   - Extract predictions for each model and window
   - Compute residuals and prediction errors
   - Generate interval predictions if configured

### Metrics Computation Flow

1. **Primary Metrics**
   - Compute sCRPS using NF's native implementation (lines 1655-1660, 1678)
   - Calculate per-window scores
   - Aggregate mean and standard deviation

2. **Supporting Metrics**
   - Calculate MAE on point predictions
   - Calculate RMSE for error magnitude
   - Compute bias for systematic error detection

3. **Aggregation**
   - Create per-model summary statistics
   - Rank models by mean sCRPS
   - Generate leaderboard DataFrame

### Calibration Assessment Flow

1. **Coverage Analysis**
   - Check 80%, 90%, 95% interval coverage (lines 1661-1665)
   - Compare empirical vs nominal rates
   - Flag deviations beyond ±2% tolerance (lines 1663)

2. **PIT Computation**
   - For distributional: evaluate CDF at actuals (lines 1642, 1664)
   - For quantile: interpolate quantile ranks (lines 1664, 1743-1766)
   - For conformal: skip (not available insample) (lines 1632, 1642, 1664)

3. **Diagnostic Visualization**
   - Generate PIT uniformity histograms
   - Plot coverage reliability diagrams
   - Create residual QQ plots

## Integration Points

### Model Factory Integration

- Receives fitted NeuralForecast instance with configured models
- Uses model names for metrics attribution
- Accesses model configurations for interpretation

### Feature Engineering Integration

- Expects features already shifted and validated
- Uses feature column names for diagnostics
- Validates no leakage through assert_shifted()

### Training Pipeline Integration

- Called from run_train.py after model fitting
- Reads configuration from experiment YAML
- Saves results to experiment directory

### Inference Pipeline Integration

- Provides model selection via leaderboard
- Saves best models for deployment
- Documents calibration quality for production

## Error Handling

### Data Validation Errors

- **Missing columns**: Clear error with expected schema
- **Irregular grid**: Fail fast with timestamp diagnostics
- **Unshifted features**: Block execution with leakage warning
- **Forward-filled target**: Reject with data quality error

### Cross-Validation Errors

- **Insufficient data**: Provide minimum data requirements
- **Window configuration**: Validate parameters before execution
- **Memory issues**: Suggest batch processing or reduction
- **Convergence failures**: Save partial results, report problematic windows

### Metrics Computation Errors

- **Missing predictions**: Handle gracefully with NaN
- **Numerical instability**: Use robust statistics
- **Distribution errors**: Fallback to empirical methods
- **Visualization failures**: Continue with text reports

## Agent-Oriented Error Recovery Protocols

### Memory Exhaustion During CV
**Detection**: OOM errors, kernel crashes, or CV hanging
**Recovery Protocol for Agents**:
1. Use Grep tool to search for "batch_size" in experiments/*.yaml files
2. Use Edit tool to reduce batch_size by 50% (e.g., 512 → 256)
3. Use context7 to query "neuralforecast memory optimization" for additional strategies
4. If persists, use Edit to reduce n_windows from 10 to 6 for pilot testing
5. Use Bash to monitor memory with: `nvidia-smi -l 1` during execution

### sCRPS Computation Failures
**Detection**: NaN values, import errors, or metric computation crashes
**Recovery Protocol for Agents**:
1. Use context7 to verify correct import: "neuralforecast sCRPS import path"
2. Use Grep to check if `from neuralforecast.losses.pytorch import sCRPS` exists
3. Use Read to verify data has no NaN/Inf values in y_true or predictions
4. Use context7 to research "neuralforecast sCRPS numerical stability"
5. If quantile model, verify quantiles are monotonic using numpy.diff checks

### Coverage Calibration Issues
**Detection**: Coverage outside ±2% tolerance or interval crossing
**Recovery Protocol for Agents**:
1. Use Grep to find "level=" in CV configuration to verify [80, 90, 95] specified
2. Use context7 to query "neuralforecast conformal prediction" for calibration methods
3. Use Read to check if predictions have proper interval columns (e.g., "Model-lo-80")
4. Use Edit to add PredictionIntervals configuration if missing
5. Use context7 to research "neuralforecast IQLoss" for quantile crossing prevention

### PIT Computation Errors
**Detection**: Non-uniform PIT histograms or interpolation failures
**Recovery Protocol for Agents**:
1. Use Edit to verify quantile grid: `quantiles = [i/100 for i in range(1, 100)]`
2. Use context7 to query "probability integral transform quantile interpolation"
3. Use Bash to run scipy.stats.kstest for uniformity validation
4. If conformal model, use Read to verify PIT is skipped (not available)
5. Use context7 to research "neuralforecast distributional calibration diagnostics"

### CV Window Configuration Errors
**Detection**: Overlapping windows, insufficient data, or step_size issues
**Recovery Protocol for Agents**:
1. Use Grep to verify step_size=h in configuration files
2. Use Read to check val_size=4*h relationship is maintained
3. Use context7 to query "neuralforecast cross validation window parameters"
4. Use Bash to calculate minimum data requirements: (n_windows-1)*step_size + val_size + input_size
5. Use Edit to adjust n_windows if insufficient data points available

## Performance Considerations

### Memory Management

- Process CV windows incrementally when possible
- Store only necessary columns in results
- Use chunked processing for large datasets
- Clear intermediate results after aggregation

### Computational Optimization

- Leverage vectorized metrics computation
- Cache computed metrics for reuse
- Parallelize window evaluation if supported
- Use efficient DataFrame operations

### Storage Optimization

- Compress parquet files with snappy
- Store only essential diagnostic data
- Implement rolling cleanup of old artifacts
- Use symbolic links for model references

## Quality Assurance

### Validation Checks

1. **Pre-CV Validation**
   - assert_regular_grid(df, "15min") (lines 368, 720)
   - assert_utc_eob(df, "15min") (lines 375, 720)
   - assert_shifted(df, hist_cols) (lines 380, 722, 1807)
   - assert_no_forward_fill_y(df) (lines 705, 721)

2. **Post-CV Validation**
   - Verify all windows completed
   - Check prediction completeness
   - Validate metric calculations
   - Confirm coverage tolerances

### Testing Strategy

- Unit tests for metrics computation
- Integration tests for CV execution
- Regression tests for known datasets
- Performance benchmarks for scalability

### Documentation Requirements

- Clear docstrings with type hints
- Usage examples in module headers
- Configuration templates in YAML
- Diagnostic interpretation guides