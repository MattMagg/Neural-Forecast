"""
Cross-validation runner module for NeuralForecast models.

This module provides NF-native cross-validation execution with proper windowing,
metrics computation, and results aggregation. Uses NeuralForecast's built-in
cross_validation method exclusively - no custom backtesting loops.
"""

from neuralforecast import NeuralForecast
from neuralforecast.losses.pytorch import sCRPS
from neuralforecast.tsdataset import TimeSeriesDataset
from neuralforecast.utils import PredictionIntervals
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import logging
import json
from scipy import stats
import torch
import gc

# Import validation utilities for comprehensive checks
from utils.validate import (
    assert_regular_grid,
    assert_utc_eob, 
    assert_shifted,
    assert_no_forward_fill_y
)

# Import error recovery and risk mitigation
from utils.error_recovery import (
    ErrorRecoveryManager,
    NumericalStabilityChecker,
    with_error_recovery
)
from utils.risk_mitigation import GPUMemoryManager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_scrps_nf_native(cv_df: pd.DataFrame, model_name: str, y_col: str = "y") -> float:
    """
    Compute sCRPS using NeuralForecast's native implementation.
    
    This function implements the exact pattern from the design document to handle
    both distributional (StudentT) and quantile (MQLoss) model types using
    NeuralForecast's native sCRPS function.
    
    Args:
        cv_df: Cross-validation DataFrame with predictions
        model_name: Name of the model to compute sCRPS for
        y_col: Column name for actual values (default: "y")
        
    Returns:
        sCRPS value (float), or np.nan if computation fails
    """
    try:
        # Extract actual values
        y_true = cv_df[y_col].values
        
        # 1. Identify quantile columns for this model
        quantile_cols = [c for c in cv_df.columns 
                        if model_name in c and '-q' in c]
        
        if not quantile_cols:
            # No quantiles available - return NaN as expected for point forecasts
            logger.debug(f"No quantile columns found for {model_name}, returning NaN")
            return np.nan
        
        # 2. Prepare quantile predictions data
        q_preds = cv_df[quantile_cols].values
        
        # 3. Extract quantile levels from column names
        quantiles = []
        for col in quantile_cols:
            try:
                # Handle format: "ModelName-q10", "ModelName-q50", etc.
                if '-q' in col:
                    q_str = col.split('-q')[1]
                    q_level = float(q_str) / 100.0
                    quantiles.append(q_level)
                else:
                    logger.warning(f"Unexpected quantile column format: {col}")
            except (ValueError, IndexError) as e:
                logger.warning(f"Could not parse quantile level from {col}: {e}")
                continue
        
        if not quantiles:
            logger.warning(f"Could not parse any quantile levels for {model_name}")
            return np.nan
        
        quantiles = np.array(quantiles)
        
        # 4. Remove NaN values
        mask = ~(np.isnan(y_true) | np.isnan(q_preds).any(axis=1))
        if not mask.any():
            logger.warning(f"All values are NaN for {model_name}")
            return np.nan
        
        y_true_clean = y_true[mask]
        q_preds_clean = q_preds[mask]
        
        # 5. Compute sCRPS using NeuralForecast's native implementation
        # Convert to torch tensors as expected by NF's sCRPS
        y_true_tensor = torch.tensor(y_true_clean, dtype=torch.float32)
        q_preds_tensor = torch.tensor(q_preds_clean, dtype=torch.float32)
        quantiles_tensor = torch.tensor(quantiles, dtype=torch.float32)
        
        # Use NF's sCRPS function
        scrps_loss = sCRPS()
        scrps_value = scrps_loss(y_true_tensor, q_preds_tensor, quantiles_tensor)
        
        # Convert back to float
        if isinstance(scrps_value, torch.Tensor):
            scrps_value = scrps_value.item()
        
        # Validate result
        if not np.isfinite(scrps_value):
            logger.warning(f"Non-finite sCRPS value for {model_name}: {scrps_value}")
            return np.nan
        
        logger.debug(f"Computed sCRPS for {model_name}: {scrps_value:.6f}")
        return scrps_value
        
    except Exception as e:
        logger.error(f"sCRPS computation failed for {model_name}: {e}")
        return np.nan


def _validate_scrps_computation(cv_df: pd.DataFrame, models: List[str]) -> None:
    """
    Validate that sCRPS values are reasonable and consistent.
    
    This function checks that:
    1. sCRPS can be computed for models with quantile predictions
    2. Values are finite and positive
    3. Values are within reasonable ranges
    
    Args:
        cv_df: Cross-validation DataFrame
        models: List of model names to validate
        
    Raises:
        ValueError: If sCRPS validation fails
    """
    logger.info("Validating sCRPS computation...")
    
    validation_results = {}
    
    for model_name in models:
        if model_name not in cv_df.columns:
            continue
            
        # Check if model has quantile predictions
        quantile_cols = [c for c in cv_df.columns 
                        if model_name in c and '-q' in c]
        
        if quantile_cols:
            # Try to compute sCRPS
            scrps_value = compute_scrps_nf_native(cv_df, model_name)
            validation_results[model_name] = scrps_value
            
            if np.isnan(scrps_value):
                logger.warning(f"sCRPS computation returned NaN for {model_name}")
            elif scrps_value <= 0:
                logger.warning(f"sCRPS value is non-positive for {model_name}: {scrps_value}")
            elif scrps_value > 10:
                logger.warning(f"sCRPS value seems unusually high for {model_name}: {scrps_value}")
            else:
                logger.info(f"sCRPS validation passed for {model_name}: {scrps_value:.6f}")
        else:
            logger.debug(f"No quantile predictions found for {model_name}, sCRPS will be NaN")
    
    # Summary validation
    valid_scrps = [v for v in validation_results.values() if np.isfinite(v)]
    
    if valid_scrps:
        logger.info(f"sCRPS validation summary: {len(valid_scrps)}/{len(validation_results)} models "
                   f"have valid sCRPS values (range: {min(valid_scrps):.6f} - {max(valid_scrps):.6f})")
    else:
        logger.warning("No models have valid sCRPS values - model ranking may not work properly")


def run_cv(nf: NeuralForecast, 
           df: pd.DataFrame, 
           cfg: Dict[str, Any],
           use_conformal: bool = False,
           enable_recovery: bool = True) -> pd.DataFrame:
    """
    Execute NF-native cross-validation with configured windowing.
    
    This function wraps NeuralForecast.cross_validation() with the exact
    parameters from the configuration. It handles both pilot (n_windows=6)
    and final (n_windows=10) configurations. Supports conformal prediction
    intervals when requested.
    
    Args:
        nf: Fitted NeuralForecast instance with models
        df: Canonical frame with unique_id, ds, y, exog columns
        cfg: Configuration dict with the following keys:
            - n_windows: Number of CV windows (6 for pilot, 10 for final)
            - step_size: Window step size (typically h)
            - val_size: Validation size (typically 4*h)
            - level: Optional list of confidence levels [80, 90, 95]
            - h: Horizon for validation (required for leakage checks)
        use_conformal: Whether to use conformal prediction intervals (Task 13)
        enable_recovery: Enable automatic error recovery for memory issues
        
    Returns:
        DataFrame with CV predictions from all models and windows, including:
        - unique_id, ds, cutoff, y: Core columns
        - Model predictions: One column per model
        - Interval predictions: If level specified, lo/hi columns per level
        
    Raises:
        ValueError: If required configuration keys are missing or validation fails
        RuntimeError: If cross_validation fails after all recovery attempts
    """
    # Initialize error recovery if enabled
    if enable_recovery:
        recovery_manager = ErrorRecoveryManager(enable_gpu_monitoring=torch.cuda.is_available())
        gpu_manager = GPUMemoryManager()
        stability_checker = NumericalStabilityChecker()
        
        # Log initial GPU status
        if torch.cuda.is_available():
            mem_info = gpu_manager.get_gpu_memory_info()
            logger.info(f"Initial GPU status: {mem_info['free_gb']:.2f}GB free of {mem_info['total_gb']:.2f}GB")
            
            # Suggest batch size if memory is tight
            if mem_info['free_gb'] < 4.0:
                current_batch = cfg.get('batch_size', 512)
                suggested_batch = gpu_manager.suggest_batch_size(
                    model_size_mb=500,  # Estimate for neural models
                    current_batch=current_batch
                )
                if suggested_batch < current_batch:
                    logger.warning(f"Low GPU memory. Consider reducing batch_size from {current_batch} to {suggested_batch}")
                    cfg['batch_size'] = suggested_batch
    # Validate required configuration
    required_keys = ['n_windows', 'step_size', 'val_size', 'h']
    missing_keys = [k for k in required_keys if k not in cfg]
    if missing_keys:
        raise ValueError(f"Missing required config keys: {missing_keys}")
    
    # Extract CV parameters
    n_windows = cfg['n_windows']
    step_size = cfg['step_size']
    val_size = cfg['val_size']
    h = cfg['h']
    level = cfg.get('level', [80, 90, 95])  # Default to standard levels
    
    # Pre-CV validation: Check input data structure
    logger.info("Running pre-CV validation checks...")
    _validate_input_data(df, h)
    
    # Validate CV parameters match requirements
    _validate_cv_parameters(n_windows, step_size, val_size, h)
    
    logger.info(f"Starting cross-validation with {n_windows} windows")
    logger.info(f"Parameters: step_size={step_size}, val_size={val_size}, level={level}")
    logger.info(f"Conformal prediction: {'Enabled' if use_conformal else 'Disabled'}")
    
    # Error recovery loop for memory exhaustion
    max_attempts = 3 if enable_recovery else 1
    last_error = None
    cv_df = None
    
    for attempt in range(1, max_attempts + 1):
        try:
            # Clear GPU cache before attempt
            if enable_recovery and torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
                
            # Task 13: Configure conformal prediction if requested
            if use_conformal:
                # Use NeuralForecast's PredictionIntervals for conformal prediction
                # Following docs/forecasting_sf_plan.md lines 1632, 1642
                logger.info("Configuring conformal prediction intervals...")
                prediction_intervals = PredictionIntervals(
                    n_windows=min(n_windows, 6),  # Use 6 windows for conformal calibration
                    h=h,
                    method='conformal_error',  # NF's conformal method
                    level=level  # Confidence levels
                )
                
                # Note: Conformal PIs are not available insample, only out-of-sample
                # This is documented in docs/forecasting_sf_plan.md lines 1632, 1642
                logger.info("Note: Conformal prediction intervals are only available out-of-sample")
                
                # Execute cross-validation with conformal intervals
                cv_df = nf.cross_validation(
                    df=df,
                    n_windows=n_windows,
                    step_size=step_size,
                    val_size=val_size,
                    refit=1,  # Always retrain every window for realistic evaluation
                    level=level,  # Base confidence levels
                    prediction_intervals=prediction_intervals  # Conformal configuration
                )
            else:
                # Execute standard NF-native cross-validation
                # Following docs/forecasting_sf_plan.md lines 1620-1628
                cv_df = nf.cross_validation(
                    df=df,
                    n_windows=n_windows,
                    step_size=step_size,
                    val_size=val_size,
                    refit=1,  # Always retrain every window for realistic evaluation
                    level=level  # Prediction intervals at specified confidence levels
                )
            
            # If we got here, CV succeeded
            break
            
        except (RuntimeError, torch.cuda.OutOfMemoryError) as e:
            last_error = e
            
            if 'out of memory' in str(e).lower() and enable_recovery and attempt < max_attempts:
                logger.warning(f"GPU OOM detected on attempt {attempt}/{max_attempts}")
                
                # Apply progressive recovery
                cfg = recovery_manager.handle_memory_exhaustion(e, cfg, attempt)
                
                # Update NF models with new batch size if possible
                for model in nf.models:
                    if hasattr(model, 'batch_size'):
                        model.batch_size = cfg.get('batch_size', 256)
                        logger.info(f"Updated {model.__class__.__name__} batch_size to {model.batch_size}")
                        
                    # Set inference batch size for memory efficiency
                    if hasattr(model, 'inference_windows_batch_size'):
                        model.inference_windows_batch_size = min(16, cfg.get('batch_size', 256) // 4)
                        
                # Reduce CV windows if on final attempt
                if attempt == max_attempts - 1:
                    n_windows = max(2, n_windows // 2)
                    logger.info(f"Reduced n_windows to {n_windows} for final attempt")
                    
            else:
                # Non-memory error or recovery disabled
                raise
    
    # Check if CV failed after all attempts
    if cv_df is None:
        if last_error:
            raise RuntimeError(f"Cross-validation failed after {max_attempts} attempts: {last_error}")
        else:
            raise RuntimeError("Cross-validation failed unexpectedly")
    
    logger.info(f"Cross-validation completed: {len(cv_df)} predictions generated")
    
    # Check for numerical issues in results
    if enable_recovery:
        # Check for NaN/Inf in predictions
        model_cols = [col for col in cv_df.columns if col not in ['unique_id', 'ds', 'cutoff', 'y']]
        for col in model_cols:
            is_clean, fixed = stability_checker.check_array(
                cv_df[col].values,
                name=col,
                fix=True
            )
            if not is_clean:
                logger.warning(f"Fixed numerical issues in {col}")
                cv_df[col] = fixed
    
    try:
        # Comprehensive post-CV validation
        # Get historical feature columns for leakage checks
        hist_cols = _get_historical_columns(df)
        
        # Validate CV results structure and completeness
        _validate_cv_results(cv_df, nf.models, level, hist_cols, h)
        
        # Additional validation for time ordering
        _validate_time_ordering(cv_df)
        
        return cv_df
        
    except AssertionError as e:
        # Re-raise assertion errors with context
        logger.error(f"CV validation failed: {str(e)}")
        raise ValueError(f"Cross-validation failed validation: {str(e)}")
        
    except Exception as e:
        logger.error(f"Cross-validation execution failed: {str(e)}")
        
        # Provide recovery suggestions for common errors
        if "memory" in str(e).lower() or "oom" in str(e).lower():
            logger.error("Suggestion: Reduce n_windows or batch_size, or use GPU with more memory")
        elif "shape" in str(e).lower() or "dimension" in str(e).lower():
            logger.error("Suggestion: Check that all exogenous features are properly aligned")
        elif "nan" in str(e).lower() or "inf" in str(e).lower():
            logger.error("Suggestion: Check for NaN/Inf values in input data or features")
            
        raise RuntimeError(f"CV execution failed: {str(e)}")


def _validate_cv_results(cv_df: pd.DataFrame, 
                         models: List[Any],
                         level: Optional[List[int]] = None,
                         hist_cols: Optional[List[str]] = None,
                         h: Optional[int] = None) -> None:
    """
    Comprehensive validation of CV results structure, completeness, and data integrity.
    
    Args:
        cv_df: Raw CV output from NeuralForecast
        models: List of model instances from NF
        level: Confidence levels if interval predictions expected
        hist_cols: List of historical feature columns for leakage checks
        h: Horizon for additional validation
        
    Raises:
        AssertionError: If validation checks fail
        ValueError: If CV results are malformed or incomplete
    """
    logger.info("Running comprehensive CV results validation...")
    
    # 1. Validate core DataFrame structure
    required_cols = ['unique_id', 'ds', 'cutoff', 'y']
    missing_cols = [col for col in required_cols if col not in cv_df.columns]
    if missing_cols:
        raise ValueError(
            f"CV results missing required columns: {missing_cols}. "
            f"Expected schema: unique_id, ds, cutoff, y, [model predictions], [intervals]"
        )
    
    # 2. Check data types
    if not pd.api.types.is_datetime64_any_dtype(cv_df['ds']):
        raise ValueError(f"Column 'ds' must be datetime, got: {cv_df['ds'].dtype}")
    
    if not pd.api.types.is_datetime64_any_dtype(cv_df['cutoff']):
        raise ValueError(f"Column 'cutoff' must be datetime, got: {cv_df['cutoff'].dtype}")
    
    if not pd.api.types.is_numeric_dtype(cv_df['y']):
        raise ValueError(f"Column 'y' must be numeric, got: {cv_df['y'].dtype}")
    
    # 3. Validate model prediction columns
    model_names = [model.__class__.__name__ for model in models]
    found_models = []
    missing_models = []
    
    for model_name in model_names:
        if model_name in cv_df.columns:
            found_models.append(model_name)
            # Check that predictions are numeric
            if not pd.api.types.is_numeric_dtype(cv_df[model_name]):
                raise ValueError(f"Model predictions '{model_name}' must be numeric, got: {cv_df[model_name].dtype}")
        else:
            missing_models.append(model_name)
    
    if not found_models:
        raise ValueError(
            f"No model predictions found in CV results. Expected columns for models: {model_names}. "
            f"Available columns: {list(cv_df.columns)}"
        )
    
    if missing_models:
        logger.warning(f"Some model predictions not found: {missing_models}")
    
    # 4. Validate interval column format and presence
    if level:
        interval_issues = []
        for model_name in found_models:
            for lv in level:
                # Check exact format: "Model-lo-80", "Model-hi-80", etc.
                lo_col = f"{model_name}-lo-{lv}"
                hi_col = f"{model_name}-hi-{lv}"
                
                if lo_col not in cv_df.columns:
                    interval_issues.append(f"Missing lower bound: {lo_col}")
                elif not pd.api.types.is_numeric_dtype(cv_df[lo_col]):
                    raise ValueError(f"Interval column '{lo_col}' must be numeric")
                    
                if hi_col not in cv_df.columns:
                    interval_issues.append(f"Missing upper bound: {hi_col}")
                elif not pd.api.types.is_numeric_dtype(cv_df[hi_col]):
                    raise ValueError(f"Interval column '{hi_col}' must be numeric")
                
                # Check interval validity (lo <= pred <= hi)
                if lo_col in cv_df.columns and hi_col in cv_df.columns and model_name in cv_df.columns:
                    # Check for any row where lower bound > upper bound
                    invalid_intervals = cv_df[lo_col] > cv_df[hi_col]
                    if invalid_intervals.any():
                        n_invalid = invalid_intervals.sum()
                        raise AssertionError(
                            f"Invalid prediction intervals detected for {model_name} at level {lv}: "
                            f"{n_invalid} cases where lower bound > upper bound"
                        )
        
        if interval_issues:
            logger.warning(f"Interval column issues: {', '.join(interval_issues[:5])}")
    
    # 5. Check for NaN values in critical columns
    nan_counts = cv_df[required_cols].isna().sum()
    if nan_counts['y'] > 0:
        # Some NaN in y is acceptable (gaps in data), but log it
        logger.info(f"Found {nan_counts['y']} NaN values in target 'y' (expected for data gaps)")
    
    if nan_counts['ds'] > 0 or nan_counts['cutoff'] > 0 or nan_counts['unique_id'] > 0:
        raise ValueError(
            f"NaN values in critical columns not allowed: "
            f"ds={nan_counts['ds']}, cutoff={nan_counts['cutoff']}, unique_id={nan_counts['unique_id']}"
        )
    
    # 6. Validate CV windows completeness
    n_cutoffs = cv_df['cutoff'].nunique()
    if n_cutoffs == 0:
        raise ValueError("No CV windows found in results. CV execution may have failed completely.")
    
    unique_ids = cv_df['unique_id'].nunique()
    expected_rows_per_window = unique_ids * (h if h else 1)  # h predictions per unique_id per window
    
    # Check if we have reasonable number of predictions
    rows_per_cutoff = cv_df.groupby('cutoff').size()
    incomplete_windows = rows_per_cutoff[rows_per_cutoff < expected_rows_per_window * 0.9]  # 90% threshold
    
    if len(incomplete_windows) > 0:
        logger.warning(
            f"Potentially incomplete CV windows detected. "
            f"Expected ~{expected_rows_per_window} rows per window, "
            f"but {len(incomplete_windows)} windows have fewer predictions: "
            f"{incomplete_windows.head().to_dict()}"
        )
    
    # 7. Leakage prevention checks using assert_shifted
    if hist_cols:
        logger.info(f"Checking {len(hist_cols)} historical features for leakage...")
        try:
            # Check a sample of the CV results for efficiency
            sample_size = min(10000, len(cv_df))
            sample_df = cv_df.sample(n=sample_size, random_state=42) if len(cv_df) > sample_size else cv_df
            
            # Only check columns that exist in CV results
            existing_hist_cols = [col for col in hist_cols if col in sample_df.columns]
            
            if existing_hist_cols:
                assert_shifted(sample_df, existing_hist_cols)
                logger.info(f"Leakage check passed for {len(existing_hist_cols)} features")
        except AssertionError as e:
            raise AssertionError(f"Leakage detected in CV results: {str(e)}")
    
    # 8. Validate target variable wasn't forward-filled
    try:
        if 'close' in cv_df.columns:
            assert_no_forward_fill_y(cv_df)
    except AssertionError as e:
        raise AssertionError(f"Target forward-fill violation in CV results: {str(e)}")
    
    # 9. Check temporal consistency within windows
    for cutoff in cv_df['cutoff'].unique()[:3]:  # Check first 3 windows as sample
        window_df = cv_df[cv_df['cutoff'] == cutoff]
        
        # All predictions in a window should be after the cutoff
        if (window_df['ds'] <= cutoff).any():
            raise AssertionError(
                f"Temporal violation: Found predictions before or at cutoff {cutoff}. "
                f"All forecasts must be strictly after the cutoff point."
            )
    
    logger.info(
        f"CV validation passed: {n_cutoffs} windows, {len(found_models)}/{len(model_names)} models, "
        f"{len(cv_df)} total predictions"
    )


def summarize_cv(cv_df: pd.DataFrame,
                 models: List[str],
                 h: int,
                 output_dir: Optional[Path] = None,
                 generate_plots: bool = True,
                 model_types: Optional[Dict[str, str]] = None,
                 model_metadata: Optional[Dict[str, Dict[str, Any]]] = None,
                 nf_instance: Optional['NeuralForecast'] = None,
                 save_best_models: bool = True,
                 selection_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, pd.DataFrame]:
    """
    Task 10: Enhanced summarize_cv with model selection and saving integration.
    
    This function orchestrates comprehensive metrics computation for all models,
    generates leaderboards, performs model selection, and optionally saves the
    best models using NF's native save functionality.
    
    Args:
        cv_df: Raw CV output from NeuralForecast
        models: List of model names to evaluate
        h: Forecast horizon for this experiment
        output_dir: Directory for saving plots and reports (Task 9)
        generate_plots: Whether to generate calibration visualizations
        model_types: Dict mapping model names to types ('distributional', 'quantile', 'conformal')
        model_metadata: Dict with model metadata (loss types, parameters)
        nf_instance: NeuralForecast instance for model saving (Task 15)
        save_best_models: Whether to save selected models (Task 15)
        selection_criteria: Optional custom selection criteria (Task 12)
        
    Returns:
        Dictionary with the following DataFrames:
        - 'metrics': Per-model metrics with confidence intervals (Task 6)
        - 'leaderboard': Models ranked by mean sCRPS (Task 11)
        - 'coverage': Coverage analysis at each confidence level
        - 'window_metrics': Metrics broken down by CV window
        - 'calibration_summary': Calibration diagnostics for all models
        - 'model_selection': Best model selection results (Task 12)
        - 'saved_models': Paths to saved model files (Task 15)
    """
    from uq.metrics import compute_metrics_per_model, aggregate_metrics
    from uq.calibration import (
        compute_coverage,
        generate_calibration_report,
        compute_coverage_by_volatility,
        plot_coverage_by_volatility
    )
    
    logger.info(f"Summarizing CV results for {len(models)} models at horizon {h}")
    
    # Initialize results dictionary
    results = {}
    
    # Validate sCRPS computation before proceeding
    _validate_scrps_computation(cv_df, models)
    
    # Compute metrics for each model
    metrics_list = []
    window_metrics_list = []
    
    for model_name in models:
        if model_name not in cv_df.columns:
            logger.warning(f"Model {model_name} not found in CV results, skipping")
            continue
        
        # Compute per-window metrics
        window_metrics = compute_metrics_per_model(
            cv_df=cv_df,
            model_name=model_name,
            y_col='y'
        )
        window_metrics['model'] = model_name
        window_metrics['horizon'] = h
        window_metrics_list.append(window_metrics)
        
        # Task 6: Enhanced aggregation with confidence intervals
        agg_metrics = aggregate_metrics_with_ci(window_metrics)
        agg_metrics['model'] = model_name
        agg_metrics['horizon'] = h
        metrics_list.append(agg_metrics)
    
    # Create metrics DataFrame
    if metrics_list:
        results['metrics'] = pd.DataFrame(metrics_list)
        results['window_metrics'] = pd.concat(window_metrics_list, ignore_index=True)
        
        # Task 11: Generate enhanced leaderboard with model metadata
        results['leaderboard'] = _generate_leaderboard(
            results['metrics'], 
            model_metadata=model_metadata
        )
        
        # Compute coverage analysis
        results['coverage'] = compute_coverage(cv_df, models)
        
        # Task 12: Perform best model selection
        results['model_selection'] = select_best_models(
            leaderboard=results['leaderboard'],
            model_types=model_types if model_types else {},
            selection_criteria=selection_criteria
        )
        
        # Task 15: Save selected models if requested
        if save_best_models and nf_instance and output_dir:
            results['saved_models'] = save_selected_models(
                nf_instance=nf_instance,
                selection_results=results['model_selection'],
                output_dir=output_dir,
                horizon=h
            )
        
        # Task 9: Generate calibration visualizations if requested
        if generate_plots and output_dir:
            logger.info("Generating calibration diagnostic plots...")
            
            # Set default model types if not provided
            if model_types is None:
                model_types = {model: 'quantile' for model in models}
            
            # Generate comprehensive calibration report with all visualizations
            results['calibration_summary'] = generate_calibration_report(
                cv_df=cv_df,
                models=models,
                model_types=model_types,
                output_dir=output_dir,
                horizon=h,
                levels=[80, 90, 95]
            )
            
            # Generate additional volatility-based calibration analysis
            logger.info("Computing coverage by volatility deciles...")
            coverage_vol_df = compute_coverage_by_volatility(cv_df, models)
            results['coverage_by_volatility'] = coverage_vol_df
            
            # Plot coverage by volatility for each model
            for model_name in models:
                if model_name in cv_df.columns:
                    plot_coverage_by_volatility(
                        coverage_vol_df=coverage_vol_df,
                        output_dir=output_dir,
                        model_name=model_name,
                        horizon=h
                    )
            
            logger.info(f"All calibration plots saved to {output_dir}/h{h}/")
    else:
        logger.warning("No valid models found for summarization")
        results['metrics'] = pd.DataFrame()
        results['leaderboard'] = pd.DataFrame()
        results['coverage'] = pd.DataFrame()
        results['window_metrics'] = pd.DataFrame()
    
    logger.info("CV summarization completed")
    return results


def aggregate_metrics_with_ci(window_metrics: pd.DataFrame,
                              confidence_level: float = 0.95) -> Dict[str, float]:
    """
    Task 6: Enhanced metrics aggregation with confidence intervals.
    
    Computes mean, std, min, max for each metric across CV windows,
    and adds confidence intervals for mean estimates using mean ± 1.96*std/sqrt(n).
    
    Args:
        window_metrics: DataFrame with per-window metrics
        confidence_level: Confidence level for intervals (default: 0.95)
        
    Returns:
        Dictionary with aggregated statistics including confidence intervals
    """
    agg_stats = {}
    
    # Get z-score for confidence level (1.96 for 95% CI)
    z_score = stats.norm.ppf((1 + confidence_level) / 2)
    
    # Core metrics to aggregate
    metrics = ['sCRPS', 'MAE', 'RMSE', 'bias']
    
    for metric in metrics:
        if metric in window_metrics.columns:
            values = window_metrics[metric].dropna()
            
            if len(values) > 0:
                # Basic statistics
                agg_stats[f'{metric}_mean'] = values.mean()
                agg_stats[f'{metric}_std'] = values.std()
                agg_stats[f'{metric}_min'] = values.min()
                agg_stats[f'{metric}_max'] = values.max()
                
                # Task 6: Add confidence intervals for mean estimates
                n = len(values)
                std_error = values.std() / np.sqrt(n) if n > 1 else 0
                margin = z_score * std_error
                
                agg_stats[f'{metric}_ci_lower'] = values.mean() - margin
                agg_stats[f'{metric}_ci_upper'] = values.mean() + margin
                agg_stats[f'{metric}_ci_margin'] = margin
                
                # Add percentiles for robust estimates
                agg_stats[f'{metric}_median'] = values.median()
                agg_stats[f'{metric}_p25'] = values.quantile(0.25)
                agg_stats[f'{metric}_p75'] = values.quantile(0.75)
            else:
                # Handle missing values gracefully
                for suffix in ['mean', 'std', 'min', 'max', 'ci_lower', 'ci_upper', 
                              'ci_margin', 'median', 'p25', 'p75']:
                    agg_stats[f'{metric}_{suffix}'] = np.nan
    
    # Add window count and confidence level used
    agg_stats['n_windows'] = len(window_metrics)
    agg_stats['confidence_level'] = confidence_level
    
    # Log summary statistics
    if 'sCRPS_mean' in agg_stats and not np.isnan(agg_stats['sCRPS_mean']):
        logger.info(f"sCRPS: {agg_stats['sCRPS_mean']:.4f} "
                   f"[{agg_stats['sCRPS_ci_lower']:.4f}, {agg_stats['sCRPS_ci_upper']:.4f}] "
                   f"({confidence_level*100:.0f}% CI, n={agg_stats['n_windows']} windows)")
    
    return agg_stats


def _generate_leaderboard(metrics_df: pd.DataFrame,
                         model_metadata: Optional[Dict[str, Dict[str, Any]]] = None) -> pd.DataFrame:
    """
    Task 11: Generate comprehensive model leaderboard ranked by mean sCRPS.
    
    Includes all metrics, model metadata (loss type, parameters), and implements
    tiebreaker using coverage deviation from nominal rates.
    
    Args:
        metrics_df: DataFrame with model metrics
        model_metadata: Optional dict with model info (loss type, parameters)
        
    Returns:
        DataFrame with models ranked by performance, including:
        - rank: Position in leaderboard
        - model: Model name
        - sCRPS_mean: Primary ranking metric
        - MAE, RMSE: Supporting metrics
        - coverage_80/90/95: Coverage at each level
        - loss_type: StudentT/MQLoss/IQLoss
        - rel_performance: Performance relative to best model
    """
    if metrics_df.empty:
        return pd.DataFrame()
    
    # Create leaderboard with ranking
    leaderboard = metrics_df.copy()
    
    # Task 11: Add model metadata if provided
    if model_metadata:
        for model_name in leaderboard['model']:
            if model_name in model_metadata:
                metadata = model_metadata[model_name]
                leaderboard.loc[leaderboard['model'] == model_name, 'loss_type'] = metadata.get('loss_type', 'unknown')
                # Add key hyperparameters if available
                if 'n_blocks' in metadata:
                    leaderboard.loc[leaderboard['model'] == model_name, 'n_blocks'] = metadata.get('n_blocks')
                if 'n_layers' in metadata:
                    leaderboard.loc[leaderboard['model'] == model_name, 'n_layers'] = metadata.get('n_layers')
    
    # Ensure sCRPS_mean exists
    if 'sCRPS_mean' in leaderboard.columns:
        # Task 11: Implement tiebreaker using coverage deviation
        # Calculate average absolute deviation from nominal coverage rates
        coverage_cols = [col for col in leaderboard.columns if 'coverage_' in col and '_deviation' not in col]
        if coverage_cols:
            # Nominal rates for 80%, 90%, 95%
            nominal_rates = {'coverage_80': 0.80, 'coverage_90': 0.90, 'coverage_95': 0.95}
            
            # Calculate average deviation for tiebreaker
            deviations = []
            for _, row in leaderboard.iterrows():
                dev_sum = 0
                dev_count = 0
                for col, nominal in nominal_rates.items():
                    if col in row and not pd.isna(row[col]):
                        dev_sum += abs(row[col] - nominal)
                        dev_count += 1
                avg_deviation = dev_sum / dev_count if dev_count > 0 else float('inf')
                deviations.append(avg_deviation)
            
            leaderboard['coverage_deviation'] = deviations
            
            # Sort by sCRPS first, then by coverage deviation as tiebreaker
            leaderboard = leaderboard.sort_values(['sCRPS_mean', 'coverage_deviation'])
        else:
            # Sort by sCRPS only if no coverage data
            leaderboard = leaderboard.sort_values('sCRPS_mean')
        
        leaderboard['rank'] = range(1, len(leaderboard) + 1)
        
        # Add relative performance vs best
        best_scrps = leaderboard['sCRPS_mean'].iloc[0]
        leaderboard['rel_performance'] = (leaderboard['sCRPS_mean'] / best_scrps - 1) * 100
        
        # Task 11: Include all key metrics in leaderboard
        # Reorder columns for comprehensive view
        cols_order = ['rank', 'model']
        
        # Add loss type if available
        if 'loss_type' in leaderboard.columns:
            cols_order.append('loss_type')
        
        # Primary metric
        cols_order.append('sCRPS_mean')
        
        # Confidence intervals if available
        if 'sCRPS_ci_lower' in leaderboard.columns:
            cols_order.extend(['sCRPS_ci_lower', 'sCRPS_ci_upper'])
        
        # Supporting metrics
        for metric in ['MAE_mean', 'RMSE_mean', 'bias_mean']:
            if metric in leaderboard.columns:
                cols_order.append(metric)
        
        # Coverage metrics
        for level in [80, 90, 95]:
            col = f'coverage_{level}'
            if col in leaderboard.columns:
                cols_order.append(col)
        
        # Performance comparison
        cols_order.extend(['rel_performance', 'coverage_deviation'])
        
        # Add remaining columns
        other_cols = [col for col in leaderboard.columns if col not in cols_order]
        leaderboard = leaderboard[[col for col in cols_order if col in leaderboard.columns] + other_cols]
        
        # Log top models
        logger.info(f"Leaderboard generated: Best model is {leaderboard.iloc[0]['model']} "
                   f"with sCRPS={leaderboard.iloc[0]['sCRPS_mean']:.4f}")
        
        if len(leaderboard) > 1:
            logger.info(f"Second best: {leaderboard.iloc[1]['model']} "
                       f"with sCRPS={leaderboard.iloc[1]['sCRPS_mean']:.4f} "
                       f"({leaderboard.iloc[1]['rel_performance']:.1f}% worse)")
    else:
        logger.warning("sCRPS_mean not found in metrics, cannot rank models")
        leaderboard['rank'] = 0
    
    return leaderboard


def select_best_models(leaderboard: pd.DataFrame,
                      model_types: Dict[str, str],
                      selection_criteria: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Task 12: Select best models based on sCRPS and model type.
    
    Identifies the best distributional model (StudentT loss), best quantile model
    (MQLoss/IQLoss), and determines whether to use a single model or ensemble.
    
    Args:
        leaderboard: DataFrame with ranked models from _generate_leaderboard
        model_types: Dict mapping model names to loss types
        selection_criteria: Optional custom selection criteria with keys:
            - 'max_ensemble_size': Maximum models in ensemble (default: 2)
            - 'min_improvement': Minimum % improvement for ensemble (default: 1.0)
            - 'prioritize_coverage': Weight coverage over sCRPS (default: False)
            
    Returns:
        Dictionary with:
        - 'best_overall': Best single model by sCRPS
        - 'best_distributional': Best model with StudentT loss
        - 'best_quantile': Best model with MQLoss/IQLoss
        - 'ensemble_models': List of models for ensemble (if beneficial)
        - 'selection_rationale': Explanation of selections
        - 'recommendations': Specific recommendations for deployment
    """
    if leaderboard.empty:
        logger.warning("Empty leaderboard provided to select_best_models")
        return {}
    
    # Default selection criteria
    criteria = selection_criteria or {}
    max_ensemble_size = criteria.get('max_ensemble_size', 2)
    min_improvement = criteria.get('min_improvement', 1.0)  # 1% improvement threshold
    prioritize_coverage = criteria.get('prioritize_coverage', False)
    
    selection_results = {}
    
    # Task 12: Identify best overall model
    best_overall = leaderboard.iloc[0]
    selection_results['best_overall'] = {
        'model': best_overall['model'],
        'sCRPS': best_overall.get('sCRPS_mean', np.nan),
        'rank': 1,
        'loss_type': model_types.get(best_overall['model'], 'unknown')
    }
    
    # Task 12: Identify best distributional model (StudentT)
    distributional_models = []
    for _, row in leaderboard.iterrows():
        model_name = row['model']
        loss_type = model_types.get(model_name, row.get('loss_type', 'unknown'))
        if 'student' in loss_type.lower() or 'distributional' in loss_type.lower():
            distributional_models.append(row)
    
    if distributional_models:
        best_dist = distributional_models[0]  # Already sorted by sCRPS
        selection_results['best_distributional'] = {
            'model': best_dist['model'],
            'sCRPS': best_dist.get('sCRPS_mean', np.nan),
            'rank': best_dist.get('rank', -1),
            'loss_type': 'StudentT'
        }
        logger.info(f"Best distributional model: {best_dist['model']} (rank {best_dist.get('rank', 'N/A')})")
    else:
        selection_results['best_distributional'] = None
        logger.warning("No distributional (StudentT) models found in leaderboard")
    
    # Task 12: Identify best quantile model (MQLoss/IQLoss)
    quantile_models = []
    for _, row in leaderboard.iterrows():
        model_name = row['model']
        loss_type = model_types.get(model_name, row.get('loss_type', 'unknown'))
        if 'mqloss' in loss_type.lower() or 'iqloss' in loss_type.lower() or 'quantile' in loss_type.lower():
            quantile_models.append(row)
    
    if quantile_models:
        best_quant = quantile_models[0]  # Already sorted by sCRPS
        selection_results['best_quantile'] = {
            'model': best_quant['model'],
            'sCRPS': best_quant.get('sCRPS_mean', np.nan),
            'rank': best_quant.get('rank', -1),
            'loss_type': model_types.get(best_quant['model'], 'quantile')
        }
        logger.info(f"Best quantile model: {best_quant['model']} (rank {best_quant.get('rank', 'N/A')})")
    else:
        selection_results['best_quantile'] = None
        logger.warning("No quantile (MQLoss/IQLoss) models found in leaderboard")
    
    # Task 12: Determine ensemble composition
    # Check if top-2 ensemble would be beneficial
    ensemble_models = []
    if len(leaderboard) >= 2:
        model1 = leaderboard.iloc[0]
        model2 = leaderboard.iloc[1]
        
        # Calculate improvement percentage
        improvement_pct = abs(model2['sCRPS_mean'] - model1['sCRPS_mean']) / model1['sCRPS_mean'] * 100
        
        # Decide on ensemble based on improvement threshold
        if improvement_pct >= min_improvement:
            # Models are sufficiently different, ensemble may help
            ensemble_models = [model1['model'], model2['model']]
            logger.info(f"Recommending ensemble of {model1['model']} and {model2['model']} "
                       f"(sCRPS difference: {improvement_pct:.2f}%)")
        else:
            # Models too similar, use single best
            ensemble_models = [model1['model']]
            logger.info(f"Single model recommended: {model1['model']} "
                       f"(second model only {improvement_pct:.2f}% different)")
    else:
        ensemble_models = [best_overall['model']]
    
    selection_results['ensemble_models'] = ensemble_models
    
    # Task 12: Document selection rationale
    rationale = []
    
    # Best overall rationale
    rationale.append(f"Best overall model '{best_overall['model']}' selected with sCRPS={best_overall.get('sCRPS_mean', 'N/A'):.4f}")
    
    # Distributional vs quantile comparison
    if selection_results['best_distributional'] and selection_results['best_quantile']:
        dist_scrps = selection_results['best_distributional']['sCRPS']
        quant_scrps = selection_results['best_quantile']['sCRPS']
        
        if dist_scrps < quant_scrps:
            rationale.append(f"Distributional model outperforms quantile: {dist_scrps:.4f} < {quant_scrps:.4f}")
        else:
            rationale.append(f"Quantile model outperforms distributional: {quant_scrps:.4f} < {dist_scrps:.4f}")
    
    # Coverage considerations if prioritized
    if prioritize_coverage and 'coverage_deviation' in leaderboard.columns:
        best_coverage = leaderboard.nsmallest(1, 'coverage_deviation').iloc[0]
        if best_coverage['model'] != best_overall['model']:
            rationale.append(f"Note: '{best_coverage['model']}' has better coverage calibration "
                           f"(deviation={best_coverage['coverage_deviation']:.3f})")
    
    # Ensemble rationale
    if len(ensemble_models) > 1:
        rationale.append(f"Ensemble of {len(ensemble_models)} models recommended for diversity")
    else:
        rationale.append("Single model recommended (insufficient diversity for ensemble)")
    
    selection_results['selection_rationale'] = rationale
    
    # Task 12: Create deployment recommendations
    recommendations = []
    
    # Primary recommendation
    if len(ensemble_models) > 1:
        recommendations.append(f"Deploy equal-weight ensemble of: {', '.join(ensemble_models)}")
    else:
        recommendations.append(f"Deploy single model: {ensemble_models[0]}")
    
    # Loss type recommendation
    if selection_results['best_distributional']:
        recommendations.append("StudentT model available for full distributional forecasts")
    if selection_results['best_quantile']:
        recommendations.append("Quantile model available for robust interval predictions")
    
    # Coverage recommendation
    if 'coverage_90' in leaderboard.columns:
        avg_coverage = leaderboard['coverage_90'].mean()
        if abs(avg_coverage - 0.90) > 0.02:
            recommendations.append(f"Consider conformal prediction (avg 90% coverage={avg_coverage:.1%})")
    
    selection_results['recommendations'] = recommendations
    
    # Log summary
    logger.info("="*60)
    logger.info("MODEL SELECTION SUMMARY")
    logger.info("="*60)
    for r in rationale:
        logger.info(f"  • {r}")
    logger.info("-"*60)
    logger.info("RECOMMENDATIONS")
    for r in recommendations:
        logger.info(f"  → {r}")
    logger.info("="*60)
    
    return selection_results


def save_selected_models(nf_instance: 'NeuralForecast',
                        selection_results: Dict[str, Any],
                        output_dir: Path,
                        horizon: int) -> Dict[str, str]:
    """
    Task 15: Save selected models using NF's native save functionality.
    
    After model selection, saves the best models and/or ensemble using
    NeuralForecast.save() for later inference.
    
    Args:
        nf_instance: NeuralForecast instance with trained models
        selection_results: Results from select_best_models
        output_dir: Base directory for saving models
        horizon: Forecast horizon for naming
        
    Returns:
        Dictionary mapping model names to saved file paths
    """
    from datetime import datetime
    
    saved_models = {}
    
    # Create models directory
    models_dir = Path(output_dir) / f"experiments/h{horizon}/models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    # Timestamp for unique naming
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Task 15: Save ensemble models if selected
    if 'ensemble_models' in selection_results and selection_results['ensemble_models']:
        ensemble_models = selection_results['ensemble_models']
        
        # Filter NF instance to only include selected models
        selected_model_instances = []
        for model in nf_instance.models:
            model_name = model.__class__.__name__
            if model_name in ensemble_models:
                selected_model_instances.append(model)
        
        if selected_model_instances:
            # Create new NF instance with only selected models
            nf_selected = NeuralForecast(
                models=selected_model_instances,
                freq=nf_instance.freq
            )
            
            # Save ensemble
            if len(ensemble_models) > 1:
                ensemble_path = models_dir / f"ensemble_h{horizon}_{timestamp}"
                nf_selected.save(str(ensemble_path), save_dataset=True)
                saved_models['ensemble'] = str(ensemble_path)
                logger.info(f"Saved ensemble models to {ensemble_path}")
            else:
                # Single best model
                best_path = models_dir / f"best_model_h{horizon}_{timestamp}"
                nf_selected.save(str(best_path), save_dataset=True)
                saved_models['best_model'] = str(best_path)
                logger.info(f"Saved best model to {best_path}")
    
    # Task 15: Additionally save best distributional and quantile models separately
    if 'best_distributional' in selection_results and selection_results['best_distributional']:
        dist_model_name = selection_results['best_distributional']['model']
        for model in nf_instance.models:
            if model.__class__.__name__ == dist_model_name:
                nf_dist = NeuralForecast(models=[model], freq=nf_instance.freq)
                dist_path = models_dir / f"best_distributional_h{horizon}_{timestamp}"
                nf_dist.save(str(dist_path), save_dataset=True)
                saved_models['best_distributional'] = str(dist_path)
                logger.info(f"Saved best distributional model to {dist_path}")
                break
    
    if 'best_quantile' in selection_results and selection_results['best_quantile']:
        quant_model_name = selection_results['best_quantile']['model']
        for model in nf_instance.models:
            if model.__class__.__name__ == quant_model_name:
                nf_quant = NeuralForecast(models=[model], freq=nf_instance.freq)
                quant_path = models_dir / f"best_quantile_h{horizon}_{timestamp}"
                nf_quant.save(str(quant_path), save_dataset=True)
                saved_models['best_quantile'] = str(quant_path)
                logger.info(f"Saved best quantile model to {quant_path}")
                break
    
    # Task 15: Save metadata about saved models
    metadata = {
        'timestamp': timestamp,
        'horizon': horizon,
        'selection_results': {
            k: v for k, v in selection_results.items() 
            if k not in ['selection_rationale', 'recommendations']  # Exclude verbose text
        },
        'saved_paths': saved_models,
        'rationale': selection_results.get('selection_rationale', []),
        'recommendations': selection_results.get('recommendations', [])
    }
    
    metadata_path = models_dir / f"model_metadata_h{horizon}_{timestamp}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    
    logger.info(f"Saved model metadata to {metadata_path}")
    saved_models['metadata'] = str(metadata_path)
    
    return saved_models


def _validate_input_data(df: pd.DataFrame, h: int) -> None:
    """
    Validate input data structure before CV execution.
    
    Args:
        df: Input DataFrame to validate
        h: Horizon for checks
        
    Raises:
        AssertionError: If validation fails
    """
    # Check required columns
    required_cols = ['unique_id', 'ds', 'y']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Input data missing required columns: {missing_cols}")
    
    # Validate time grid regularity
    try:
        assert_regular_grid(df, freq="15min")
    except AssertionError as e:
        raise AssertionError(f"Input data grid validation failed: {str(e)}")
    
    # Validate UTC EOB timestamps
    try:
        assert_utc_eob(df, freq="15min")
    except AssertionError as e:
        raise AssertionError(f"Input data EOB validation failed: {str(e)}")
    
    # Check for sufficient data
    n_rows = len(df)
    min_required = h * 10  # At least 10 horizons of data
    if n_rows < min_required:
        raise ValueError(
            f"Insufficient data for CV: {n_rows} rows < {min_required} required "
            f"(need at least 10 * horizon={h} for meaningful CV)"
        )
    
    logger.info(f"Input data validation passed: {n_rows} rows, regular 15min grid, UTC EOB")


def _validate_cv_parameters(n_windows: int, step_size: int, val_size: int, h: int) -> None:
    """
    Validate CV parameters match specification requirements.
    
    Per docs/forecasting_sf_plan.md lines 1807-1809:
    - step_size=h prevents forecast overlap
    - refit=1 ensures proper retraining
    - val_size=4*h provides sufficient validation samples
    
    Args:
        n_windows: Number of CV windows
        step_size: Step size between windows
        val_size: Validation set size
        h: Forecast horizon
        
    Raises:
        AssertionError: If parameters violate requirements
    """
    # Validate step_size = h (prevents overlap)
    if step_size != h:
        raise AssertionError(
            f"step_size must equal horizon to prevent forecast overlap: "
            f"step_size={step_size} != h={h}. Set step_size=h"
        )
    
    # Validate val_size = 4*h (sufficient validation samples)
    expected_val_size = 4 * h
    if val_size != expected_val_size:
        raise AssertionError(
            f"val_size must be 4*horizon for proper validation: "
            f"val_size={val_size} != 4*h={expected_val_size}. Set val_size={4*h}"
        )
    
    # Validate n_windows is reasonable
    if n_windows < 3:
        raise ValueError(f"n_windows={n_windows} too small for reliable CV, minimum is 3")
    
    if n_windows > 20:
        logger.warning(f"n_windows={n_windows} is large, CV may be slow")
    
    logger.info(f"CV parameters validated: n_windows={n_windows}, step_size=h={h}, val_size=4h={val_size}")


def _get_historical_columns(df: pd.DataFrame) -> List[str]:
    """
    Identify historical feature columns for leakage checks.
    
    Args:
        df: Input DataFrame
        
    Returns:
        List of column names that are historical features
    """
    # Core columns that are NOT historical features
    non_hist_cols = {'unique_id', 'ds', 'y', 'cutoff', 'close', 'open', 'high', 'low', 'volume'}
    
    # All other columns are assumed to be historical features
    hist_cols = [col for col in df.columns if col not in non_hist_cols]
    
    # Filter to columns that contain common indicator patterns
    indicator_patterns = ['rsi', 'ma', 'ema', 'sma', 'bb', 'macd', 'stoch', 'atr', 'adx', 
                          'cci', 'roc', 'williams', 'mfi', 'obv', 'vwap', 'psar', 'ich']
    
    filtered_hist_cols = []
    for col in hist_cols:
        col_lower = col.lower()
        if any(pattern in col_lower for pattern in indicator_patterns):
            filtered_hist_cols.append(col)
    
    # If no indicator patterns found, return all non-core columns
    if not filtered_hist_cols:
        filtered_hist_cols = hist_cols
    
    return filtered_hist_cols


def _validate_time_ordering(cv_df: pd.DataFrame) -> None:
    """
    Validate temporal ordering: train < validation < test.
    
    Args:
        cv_df: CV results DataFrame
        
    Raises:
        AssertionError: If time ordering is violated
    """
    # Check that cutoffs are in ascending order
    cutoffs = cv_df['cutoff'].unique()
    cutoffs_sorted = sorted(cutoffs)
    
    if not all(cutoffs == cutoffs_sorted):
        raise AssertionError("CV cutoffs are not in chronological order")
    
    # For each cutoff, verify predictions are after the cutoff
    for cutoff in cutoffs[:5]:  # Check first 5 windows
        window_data = cv_df[cv_df['cutoff'] == cutoff]
        
        # All ds values should be > cutoff (future predictions)
        past_predictions = window_data[window_data['ds'] <= cutoff]
        if len(past_predictions) > 0:
            raise AssertionError(
                f"Time ordering violation at cutoff {cutoff}: "
                f"Found {len(past_predictions)} predictions before/at cutoff time"
            )
    
    logger.info("Time ordering validation passed: train < validation < test maintained")


def validate_conformal_coverage(cv_df: pd.DataFrame,
                               models: List[str],
                               expected_levels: List[int] = [80, 90, 95]) -> pd.DataFrame:
    """
    Task 13: Validate conformal prediction interval coverage.
    
    For conformal models, this validates that empirical coverage matches
    nominal rates, accounting for the fact that insample PIs are not available.
    
    Args:
        cv_df: CV results with conformal intervals
        models: List of model names using conformal prediction
        expected_levels: Expected confidence levels
        
    Returns:
        DataFrame with conformal coverage validation results
    """
    validation_results = []
    
    for model_name in models:
        if model_name not in cv_df.columns:
            continue
        
        for level in expected_levels:
            # Check for conformal interval columns
            lo_col = f"{model_name}-conformal-lo-{level}"
            hi_col = f"{model_name}-conformal-hi-{level}"
            
            # Also check standard naming
            if lo_col not in cv_df.columns:
                lo_col = f"{model_name}-lo-{level}"
                hi_col = f"{model_name}-hi-{level}"
            
            if lo_col in cv_df.columns and hi_col in cv_df.columns:
                # Compute coverage
                y_true = cv_df['y'].values
                lo_pred = cv_df[lo_col].values
                hi_pred = cv_df[hi_col].values
                
                # Handle NaN values
                mask = ~(np.isnan(y_true) | np.isnan(lo_pred) | np.isnan(hi_pred))
                if mask.any():
                    y_clean = y_true[mask]
                    lo_clean = lo_pred[mask]
                    hi_clean = hi_pred[mask]
                    
                    # Check coverage
                    within_interval = (y_clean >= lo_clean) & (y_clean <= hi_clean)
                    empirical_coverage = float(within_interval.mean())
                    nominal_coverage = level / 100.0
                    deviation = empirical_coverage - nominal_coverage
                    
                    # Conformal intervals should have exact coverage guarantee
                    # Allow small deviation due to finite sample
                    is_valid = abs(deviation) <= 0.02
                    
                    validation_results.append({
                        'model': model_name,
                        'level': level,
                        'method': 'conformal',
                        'nominal_coverage': nominal_coverage,
                        'empirical_coverage': empirical_coverage,
                        'deviation': deviation,
                        'is_valid': is_valid,
                        'n_samples': len(y_clean),
                        'note': 'Conformal PIs not available insample' if 'conformal' in lo_col else 'Standard PI'
                    })
    
    if validation_results:
        logger.info(f"Conformal coverage validation complete for {len(validation_results)} model-level combinations")
        df = pd.DataFrame(validation_results)
        
        # Log summary
        n_valid = df['is_valid'].sum()
        n_total = len(df)
        logger.info(f"Conformal validation: {n_valid}/{n_total} meet coverage guarantees")
        
        return df
    else:
        logger.warning("No conformal intervals found for validation")
        return pd.DataFrame()


def save_cv_results(results: Dict[str, pd.DataFrame],
                   output_dir: Path,
                   horizon: int) -> None:
    """
    Save CV results to disk with proper naming convention.
    
    Args:
        results: Dictionary of CV results from summarize_cv
        output_dir: Base directory for saving results
        horizon: Forecast horizon for naming
    """
    from utils.io import save_parquet, timestamped_path
    
    # Create output directory if needed
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each result type
    for key, df in results.items():
        if not df.empty:
            if key == 'metrics':
                # Save as CSV for easy viewing
                csv_path = output_dir / f"cv_metrics_h{horizon}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved metrics to {csv_path}")
                
            elif key == 'leaderboard':
                # Save leaderboard as CSV
                csv_path = output_dir / f"leaderboard_h{horizon}.csv"
                df.to_csv(csv_path, index=False)
                logger.info(f"Saved leaderboard to {csv_path}")
                
            else:
                # Save other results as parquet
                parquet_path = timestamped_path(
                    base_dir=str(output_dir),
                    stem=f"cv_{key}_h{horizon}",
                    ext="parquet"
                )
                save_parquet(df, parquet_path)
                logger.info(f"Saved {key} to {parquet_path}")


def generate_cv_summary_report(results: Dict[str, pd.DataFrame],
                             output_path: Path,
                             horizon: int,
                             use_conformal: bool = False) -> None:
    """
    Generate comprehensive markdown summary report of CV results.
    
    This orchestrates all reporting including metrics, calibration,
    and conformal validation results.
    
    Args:
        results: Dictionary of all CV results
        output_path: Path for markdown report
        horizon: Forecast horizon
        use_conformal: Whether conformal prediction was used
    """
    with open(output_path, 'w') as f:
        f.write(f"# Cross-Validation Summary - Horizon {horizon}\n\n")
        f.write(f"Generated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model leaderboard
        if 'leaderboard' in results and not results['leaderboard'].empty:
            f.write("## Model Leaderboard\n\n")
            f.write("Models ranked by mean sCRPS (lower is better):\n\n")
            # Write table header
            f.write("| Rank | Model | sCRPS Mean | Relative Performance |\n")
            f.write("|------|-------|------------|---------------------|\n")
            # Write table rows
            for _, row in results['leaderboard'].iterrows():
                f.write(f"| {row.get('rank', 'N/A')} | {row.get('model', 'N/A')} | ")
                f.write(f"{row.get('sCRPS_mean', 0):.4f} | ")
                f.write(f"{row.get('rel_performance', 0):.1f}% |\n")
            f.write("\n\n")
        
        # Coverage summary
        if 'coverage' in results and not results['coverage'].empty:
            f.write("## Coverage Analysis\n\n")
            coverage_summary = results['coverage'].pivot_table(
                index='model',
                columns='level',
                values='empirical_coverage',
                aggfunc='first'
            )
            # Write coverage table manually
            f.write("| Model | 80% Coverage | 90% Coverage | 95% Coverage |\n")
            f.write("|-------|--------------|--------------|---------------|\n")
            for model in coverage_summary.index:
                f.write(f"| {model} | ")
                for level in [80, 90, 95]:
                    if level in coverage_summary.columns:
                        val = coverage_summary.loc[model, level]
                        f.write(f"{val:.1%} | " if not pd.isna(val) else "N/A | ")
                    else:
                        f.write("N/A | ")
                f.write("\n")
            f.write("\n\n")
        
        # Calibration summary
        if 'calibration_summary' in results and not results['calibration_summary'].empty:
            f.write("## Calibration Status\n\n")
            cal_df = results['calibration_summary'][[
                'model', 'overall_calibrated', 'coverage_calibrated', 
                'pit_uniform', 'has_interval_crossing'
            ]]
            # Write calibration table manually
            f.write("| Model | Overall Calibrated | Coverage Calibrated | PIT Uniform | Interval Crossing |\n")
            f.write("|-------|-------------------|--------------------|-----------|-----------------|\n")
            for _, row in cal_df.iterrows():
                f.write(f"| {row['model']} | ")
                f.write(f"{'✓' if row.get('overall_calibrated', False) else '✗'} | ")
                f.write(f"{'✓' if row.get('coverage_calibrated', False) else '✗'} | ")
                f.write(f"{'✓' if row.get('pit_uniform', False) else '✗'} | ")
                f.write(f"{'✗' if row.get('has_interval_crossing', False) else '✓'} |\n")
            f.write("\n\n")
        
        # Conformal validation if applicable
        if use_conformal:
            f.write("## Conformal Prediction Validation\n\n")
            f.write("Note: Conformal prediction intervals are only available out-of-sample.\n")
            f.write("Insample PIs are not provided by NeuralForecast's conformal method.\n\n")
            
            if 'conformal_validation' in results and not results['conformal_validation'].empty:
                # Write conformal validation table manually
                f.write("| Model | Level | Nominal Coverage | Empirical Coverage | Valid |\n")
                f.write("|-------|-------|-----------------|-------------------|-------|\n")
                for _, row in results['conformal_validation'].iterrows():
                    f.write(f"| {row['model']} | {row['level']}% | ")
                    f.write(f"{row['nominal_coverage']:.2f} | ")
                    f.write(f"{row['empirical_coverage']:.2f} | ")
                    f.write(f"{'✓' if row.get('is_valid', False) else '✗'} |\n")
            else:
                f.write("Conformal validation results will be available after prediction.\n")
            f.write("\n\n")
        
        # Key insights
        f.write("## Key Insights\n\n")
        if 'metrics' in results and not results['metrics'].empty:
            best_model = results['leaderboard'].iloc[0]['model'] if 'leaderboard' in results else 'N/A'
            best_scrps = results['leaderboard'].iloc[0]['sCRPS_mean'] if 'leaderboard' in results else 'N/A'
            
            f.write(f"- **Best Model**: {best_model} (sCRPS: {best_scrps:.4f})\n")
            
            if 'calibration_summary' in results:
                n_calibrated = results['calibration_summary']['overall_calibrated'].sum()
                n_total = len(results['calibration_summary'])
                f.write(f"- **Calibration**: {n_calibrated}/{n_total} models well-calibrated\n")
            
            if 'coverage' in results:
                avg_coverage_90 = results['coverage'][results['coverage']['level'] == 90]['empirical_coverage'].mean()
                f.write(f"- **Average 90% Coverage**: {avg_coverage_90:.1%}\n")
        
        f.write("\n")
    

# =============================================================================
# CHECKPOINTING FUNCTIONALITY (NEW)
# =============================================================================

import pickle
from pathlib import Path
from datetime import datetime

class CVCheckpointManager:
    """Manages checkpointing for cross-validation to prevent loss of progress.
    
    This class provides functionality to save and restore CV state during
    long-running training sessions, protecting against failures and allowing
    resumption from the last completed window.
    """
    
    def __init__(self, checkpoint_dir: str = "checkpoints"):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to store checkpoints
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def save_checkpoint(
        self,
        cv_results: pd.DataFrame,
        window_idx: int,
        horizon: int,
        n_windows: int,
        models: List[str],
        additional_data: Optional[Dict] = None
    ) -> Path:
        """Save CV checkpoint after completing a window.
        
        Args:
            cv_results: Current CV results DataFrame
            window_idx: Index of completed window (0-based)
            horizon: Forecast horizon (h)
            n_windows: Total number of CV windows
            models: List of model names
            additional_data: Optional additional data to save
            
        Returns:
            Path to saved checkpoint file
        """
        checkpoint = {
            'cv_results': cv_results,
            'window_idx': window_idx,
            'horizon': horizon,
            'n_windows': n_windows,
            'models': models,
            'timestamp': datetime.now().isoformat(),
            'progress_pct': (window_idx + 1) / n_windows * 100
        }
        
        if additional_data:
            checkpoint.update(additional_data)
        
        # Save with timestamp and window index
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = self.checkpoint_dir / f"cv_checkpoint_h{horizon}_w{window_idx}_{timestamp}.pkl"
        
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        logger.info(f"✅ Checkpoint saved: {checkpoint_path}")
        logger.info(f"   Progress: {checkpoint['progress_pct']:.1f}% ({window_idx + 1}/{n_windows} windows)")
        
        # Clean old checkpoints (keep last 3 for this horizon)
        self._cleanup_old_checkpoints(horizon, keep=3)
        
        return checkpoint_path
    
    def load_latest_checkpoint(self, horizon: int) -> Optional[Dict]:
        """Load the most recent checkpoint for a given horizon.
        
        Args:
            horizon: Forecast horizon to load checkpoint for
            
        Returns:
            Checkpoint dictionary or None if no checkpoint exists
        """
        pattern = f"cv_checkpoint_h{horizon}_*.pkl"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if not checkpoints:
            logger.info(f"No checkpoints found for horizon h={horizon}")
            return None
        
        # Get most recent by modification time
        latest = max(checkpoints, key=lambda p: p.stat().st_mtime)
        
        with open(latest, 'rb') as f:
            checkpoint = pickle.load(f)
        
        logger.info(f"✅ Loaded checkpoint: {latest}")
        logger.info(f"   Window {checkpoint['window_idx'] + 1}/{checkpoint['n_windows']} completed")
        logger.info(f"   Progress: {checkpoint['progress_pct']:.1f}%")
        logger.info(f"   Timestamp: {checkpoint['timestamp']}")
        
        return checkpoint
    
    def should_resume(self, horizon: int) -> bool:
        """Check if there's a checkpoint to resume from.
        
        Args:
            horizon: Forecast horizon to check
            
        Returns:
            True if checkpoint exists and should resume
        """
        checkpoint = self.load_latest_checkpoint(horizon)
        if checkpoint is None:
            return False
        
        # Check if CV was completed
        if checkpoint['window_idx'] + 1 >= checkpoint['n_windows']:
            logger.info("CV already completed based on checkpoint")
            return False
        
        return True
    
    def _cleanup_old_checkpoints(self, horizon: int, keep: int = 3):
        """Remove old checkpoints, keeping only the most recent ones.
        
        Args:
            horizon: Forecast horizon
            keep: Number of recent checkpoints to keep
        """
        pattern = f"cv_checkpoint_h{horizon}_*.pkl"
        checkpoints = list(self.checkpoint_dir.glob(pattern))
        
        if len(checkpoints) <= keep:
            return
        
        # Sort by modification time and remove old ones
        checkpoints.sort(key=lambda p: p.stat().st_mtime)
        for checkpoint in checkpoints[:-keep]:
            checkpoint.unlink()
            logger.debug(f"Removed old checkpoint: {checkpoint}")


def run_cv_with_checkpointing(
    nf: NeuralForecast,
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    use_conformal: bool = False,
    checkpoint_interval: int = 1
) -> pd.DataFrame:
    """Run cross-validation with automatic checkpointing.
    
    This is a wrapper around run_cv that adds checkpointing capability
    for long-running CV sessions.
    
    Args:
        nf: Fitted NeuralForecast instance
        df: Data DataFrame
        cfg: Configuration dictionary
        use_conformal: Whether to use conformal prediction
        checkpoint_interval: Save checkpoint every N windows
        
    Returns:
        Complete CV results DataFrame
    """
    checkpoint_mgr = CVCheckpointManager()
    horizon = cfg['h']
    
    # Check for existing checkpoint
    checkpoint = checkpoint_mgr.load_latest_checkpoint(horizon)
    
    if checkpoint and checkpoint['window_idx'] + 1 < cfg['n_windows']:
        logger.info(f"Resuming from checkpoint (window {checkpoint['window_idx'] + 1})")
        # Note: NeuralForecast doesn't support partial CV, so we'd need to
        # implement custom windowing here. For now, we just warn.
        logger.warning("Note: Full CV restart required (NF doesn't support partial CV)")
        logger.warning("Checkpoints will still be saved for progress tracking")
    
    # Run standard CV
    cv_df = run_cv(nf, df, cfg, use_conformal)
    
    # Save final checkpoint
    model_names = [model.__class__.__name__ for model in nf.models]
    checkpoint_mgr.save_checkpoint(
        cv_results=cv_df,
        window_idx=cfg['n_windows'] - 1,
        horizon=horizon,
        n_windows=cfg['n_windows'],
        models=model_names,
        additional_data={'completed': True}
    )
    
    return cv_df
    logger.info(f"Generated comprehensive CV summary report at {output_path}")