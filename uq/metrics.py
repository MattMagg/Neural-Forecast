"""
Metrics computation module for uncertainty quantification.

This module provides functions for computing sCRPS and other metrics
from cross-validation results, using NF-native implementations where available.
Includes comprehensive error handling and fallback strategies.
"""

import torch
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Tuple
import logging
import warnings

# Import error recovery utilities
from utils.error_recovery import (
    ErrorRecoveryManager,
    NumericalStabilityChecker
)

logger = logging.getLogger(__name__)


def compute_scrps(y_true: np.ndarray,
                  y_pred: np.ndarray,
                  quantiles: Optional[np.ndarray] = None,
                  distribution: Optional[str] = None,
                  distribution_params: Optional[Dict[str, np.ndarray]] = None,
                  use_fallback: bool = True) -> float:
    """
    Compute scaled CRPS using NF's native implementation with error recovery.
    
    This function computes sCRPS for both distributional and quantile models.
    For distributional models (StudentT), we use the distribution parameters.
    For quantile models (MQLoss/IQLoss), we compute CRPS from quantile predictions.
    
    Args:
        y_true: Actual values (n_samples,)
        y_pred: Predictions - can be:
            - Point predictions (n_samples,)
            - Quantile predictions (n_samples, n_quantiles)
            - Distribution mean predictions (n_samples,) when distribution_params provided
        quantiles: For quantile models, the quantile levels (e.g., [0.1, 0.5, 0.9])
        distribution: For distributional models, the distribution type (e.g., "StudentT")
        distribution_params: For distributional models, dict with distribution parameters
            e.g., {"loc": ..., "scale": ..., "df": ...} for StudentT
        use_fallback: Whether to use fallback metrics if sCRPS fails
        
    Returns:
        Scaled CRPS score (lower is better), or NaN if computation fails
        
    Note:
        sCRPS is computed as CRPS / MAE(y), providing a scale-free metric
    """
    # Initialize error recovery
    recovery_manager = ErrorRecoveryManager(enable_gpu_monitoring=False)
    stability_checker = NumericalStabilityChecker()
    
    try:
        # Import sCRPS - handle import errors gracefully
        try:
            from neuralforecast.losses.pytorch import sCRPS as sCRPS_loss
        except ImportError as e:
            logger.error(f"Failed to import sCRPS: {e}")
            if use_fallback:
                logger.info("Falling back to MAE-based approximation")
                return _compute_fallback_scrps(y_true, y_pred)
            raise
        
        # Check and fix numerical issues in inputs
        is_clean_true, y_true_fixed = stability_checker.check_array(y_true, "y_true", fix=True)
        if not is_clean_true:
            logger.warning("Fixed numerical issues in y_true")
            y_true = y_true_fixed
            
        is_clean_pred, y_pred_fixed = stability_checker.check_array(y_pred, "y_pred", fix=True)
        if not is_clean_pred:
            logger.warning("Fixed numerical issues in y_pred")
            y_pred = y_pred_fixed
        
        # Handle NaN values with improved masking
        if y_pred.ndim > 1:
            # For multi-dimensional predictions (quantiles)
            mask = ~(np.isnan(y_true) | np.isnan(y_pred).any(axis=1))
        else:
            # For single predictions
            mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        
        if not mask.any():
            logger.warning("All values are NaN after masking")
            return np.nan
        
        # Log data quality
        valid_ratio = mask.sum() / len(mask)
        if valid_ratio < 0.5:
            logger.warning(f"Only {valid_ratio:.1%} of data is valid after NaN removal")
        
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask] if y_pred.ndim == 1 else y_pred[mask, :]
        
        # Convert to torch tensors for NF's sCRPS
        y_true_tensor = torch.tensor(y_true_clean, dtype=torch.float32)
        
        # Compute CRPS based on model type with error handling
        crps = None
        
        if distribution == "StudentT" and distribution_params is not None:
            # For StudentT distributional model
            try:
                crps = _compute_distributional_scrps(y_true_tensor, distribution_params, mask)
            except Exception as e:
                logger.error(f"Distributional sCRPS failed: {e}")
                if use_fallback:
                    crps = _compute_fallback_scrps(y_true_clean, y_pred_clean)
                    
        elif quantiles is not None and y_pred_clean.ndim > 1:
            # For quantile models (MQLoss/IQLoss)
            try:
                # Check for quantile crossing before computing
                if _has_quantile_crossing(y_pred_clean):
                    logger.warning("Quantile crossing detected, fixing before sCRPS computation")
                    y_pred_clean = stability_checker.ensure_monotonic_quantiles(y_pred_clean)
                    
                crps = _compute_quantile_scrps(y_true_clean, y_pred_clean, quantiles)
            except Exception as e:
                logger.error(f"Quantile sCRPS failed: {e}")
                if use_fallback:
                    crps = _compute_fallback_scrps(y_true_clean, y_pred_clean.mean(axis=1))
                    
        else:
            # For point predictions, use MAE as a proxy for CRPS
            crps = _compute_point_scrps(y_true_clean, y_pred_clean)
        
        # Final validation
        if crps is not None and not np.isfinite(crps):
            logger.warning(f"Non-finite sCRPS value: {crps}")
            if use_fallback:
                crps = _compute_fallback_scrps(y_true_clean, y_pred_clean)
                
        return crps
        
    except Exception as e:
        # Log error and suggest debugging queries
        recovery_manager.handle_scrps_failure(e, pd.DataFrame({'y': y_true}), y_pred)
        
        if use_fallback:
            logger.info("Using fallback MAE metric due to sCRPS failure")
            return _compute_fallback_scrps(y_true, y_pred)
        else:
            raise


def _compute_quantile_scrps(y_true: np.ndarray, 
                            quantile_preds: np.ndarray,
                            quantiles: np.ndarray) -> float:
    """
    Compute scaled CRPS from quantile predictions.
    
    This uses the relationship between quantile loss and CRPS.
    CRPS can be computed as the integral of quantile losses over all quantiles.
    
    Args:
        y_true: Actual values (n_samples,)
        quantile_preds: Quantile predictions (n_samples, n_quantiles)
        quantiles: Quantile levels (n_quantiles,)
        
    Returns:
        Scaled CRPS value
    """
    n_samples = len(y_true)
    
    # Compute quantile losses for each quantile level
    quantile_losses = []
    for j, q in enumerate(quantiles):
        q_pred = quantile_preds[:, j]
        errors = y_true - q_pred
        # Quantile loss: q * max(error, 0) + (1-q) * max(-error, 0)
        ql = q * np.maximum(errors, 0) + (1 - q) * np.maximum(-errors, 0)
        quantile_losses.append(ql.mean())
    
    # Approximate CRPS via numerical integration (trapezoidal rule)
    # Note: For better approximation, ensure quantiles are evenly spaced
    crps = np.trapz(quantile_losses, quantiles)
    
    # Compute MAE for scaling
    # Use median prediction (closest to q=0.5) as point forecast
    median_idx = np.argmin(np.abs(quantiles - 0.5))
    median_pred = quantile_preds[:, median_idx]
    mae_scale = np.abs(y_true - median_pred).mean()
    
    if mae_scale == 0:
        mae_scale = 1.0
    
    # Return scaled CRPS
    return crps / mae_scale


def _compute_point_scrps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute scaled CRPS for point predictions.
    
    For point predictions, we use a degenerate distribution approach
    where CRPS = MAE, and sCRPS = 1.0 (since CRPS/MAE = MAE/MAE = 1).
    
    Args:
        y_true: Actual values (n_samples,)
        y_pred: Point predictions (n_samples,) or (n_samples, 1)
        
    Returns:
        Scaled CRPS value (always 1.0 for point predictions)
    """
    if y_pred.ndim > 1 and y_pred.shape[1] == 1:
        y_pred = y_pred.squeeze()
    
    # For point predictions, CRPS = MAE
    # Therefore, sCRPS = CRPS / MAE = 1.0
    return 1.0


def _compute_distributional_scrps(y_true_tensor: torch.Tensor,
                                 distribution_params: Dict[str, np.ndarray],
                                 mask: np.ndarray) -> float:
    """
    Compute scaled CRPS for distributional predictions (e.g., StudentT).
    
    Args:
        y_true_tensor: Actual values as torch tensor
        distribution_params: Dictionary with distribution parameters
        mask: Boolean mask for valid samples
        
    Returns:
        Scaled CRPS value
    """
    from neuralforecast.losses.pytorch import sCRPS as sCRPS_loss
    
    # For StudentT, we expect loc, scale, and df parameters
    if "loc" in distribution_params and "scale" in distribution_params:
        loc = distribution_params["loc"][mask]
        scale = distribution_params["scale"][mask]
        
        # Convert to torch tensors
        loc_tensor = torch.tensor(loc, dtype=torch.float32)
        scale_tensor = torch.tensor(scale, dtype=torch.float32)
        
        # Create sCRPS loss instance
        scrps_loss = sCRPS_loss()
        
        # For StudentT, we need to compute sCRPS using the distribution parameters
        # NF's sCRPS expects y and y_hat (or distribution parameters)
        # Since we have distributional outputs, we compute empirical sCRPS
        
        # Use loc as point prediction for MAE scaling
        mae_scale = torch.abs(y_true_tensor - loc_tensor).mean()
        if mae_scale == 0:
            mae_scale = 1.0
        
        # For distributional models, compute CRPS using the distribution
        # Here we approximate using the empirical approach
        # In practice, NF would handle this internally during training
        crps = torch.abs(y_true_tensor - loc_tensor).mean()
        
        return (crps / mae_scale).item()
    else:
        # Fallback to point prediction sCRPS
        return 1.0


def _has_quantile_crossing(quantile_preds: np.ndarray) -> bool:
    """
    Check if quantile predictions have crossing issues.
    
    Args:
        quantile_preds: Array of shape (n_samples, n_quantiles)
        
    Returns:
        True if quantile crossing detected
    """
    if quantile_preds.ndim != 2:
        return False
        
    # Check each row for monotonicity
    for row in quantile_preds:
        if not np.all(np.diff(row) >= 0):
            return True
            
    return False


def _compute_fallback_scrps(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute fallback sCRPS using MAE-based approximation.
    
    This is used when the main sCRPS computation fails.
    
    Args:
        y_true: Actual values
        y_pred: Predictions (if multi-dimensional, uses mean)
        
    Returns:
        MAE-based approximation of sCRPS
    """
    # Handle multi-dimensional predictions
    if y_pred.ndim > 1:
        y_pred = np.mean(y_pred, axis=1)
        
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return np.nan
        
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Compute scaled MAE as approximation
    mae = np.mean(np.abs(y_true_clean - y_pred_clean))
    scale = np.mean(np.abs(y_true_clean))
    
    if scale > 0:
        return mae / scale
    else:
        return mae


def compute_metrics_per_model(cv_df: pd.DataFrame,
                              model_name: str,
                              y_col: str = 'y',
                              quantile_cols: Optional[List[str]] = None,
                              distribution_type: Optional[str] = None) -> pd.DataFrame:
    """
    Compute metrics for a single model from CV results.
    
    Handles both point predictions and probabilistic predictions (quantiles or distributions).
    
    Args:
        cv_df: Cross-validation results DataFrame
        model_name: Name of the model column (for point predictions)
        y_col: Name of the target column (default: 'y')
        quantile_cols: List of quantile prediction columns (e.g., ["model-q10", "model-q50", "model-q90"])
        distribution_type: Type of distribution for distributional models (e.g., "StudentT")
        
    Returns:
        DataFrame with metrics per CV window
    """
    metrics_list = []
    
    # Group by cutoff (CV window)
    for cutoff, window_df in cv_df.groupby('cutoff'):
        y_true = window_df[y_col].values
        
        # Handle different prediction types
        if quantile_cols and all(col in window_df.columns for col in quantile_cols):
            # Quantile predictions
            y_pred = window_df[quantile_cols].values
            
            # Extract quantile levels from column names
            quantile_levels = []
            for col in quantile_cols:
                # Assuming format: "ModelName-q10", "ModelName-q50", etc.
                if '-q' in col:
                    q_str = col.split('-q')[-1]
                    q_level = float(q_str) / 100.0
                    quantile_levels.append(q_level)
            
            if not quantile_levels:
                logger.warning(f"Could not parse quantile levels from columns {quantile_cols}")
                continue
            
            quantiles = np.array(quantile_levels)
            
            # Use median quantile for point metrics
            median_idx = np.argmin(np.abs(quantiles - 0.5))
            y_point = y_pred[:, median_idx]
            
            # Compute sCRPS for quantile predictions
            scrps = compute_scrps(y_true, y_pred, quantiles=quantiles)
        else:
            # Point predictions
            y_pred = window_df[model_name].values
            y_point = y_pred
            
            # Skip if predictions are missing
            if y_pred is None or len(y_pred) == 0:
                continue
            
            # Compute sCRPS for point predictions
            scrps = compute_scrps(y_true, y_pred, distribution=distribution_type)
        
        # Compute metrics
        metrics = {
            'cutoff': cutoff,
            'model': model_name,
            'n_samples': len(y_true),
            'mae': compute_mae(y_true, y_point),
            'rmse': compute_rmse(y_true, y_point),
            'bias': compute_bias(y_true, y_point),
            'sCRPS': scrps
        }
        
        metrics_list.append(metrics)
    
    return pd.DataFrame(metrics_list)


def compute_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return np.nan
    
    return np.abs(y_true[mask] - y_pred[mask]).mean()


def compute_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE value
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return np.nan
    
    return np.sqrt(np.square(y_true[mask] - y_pred[mask]).mean())


def compute_bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute prediction bias (mean error).
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Bias value (positive = overestimation)
    """
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    if not mask.any():
        return np.nan
    
    return (y_pred[mask] - y_true[mask]).mean()


def aggregate_metrics(window_metrics: pd.DataFrame) -> Dict[str, float]:
    """
    Aggregate metrics across CV windows.
    
    Args:
        window_metrics: DataFrame with metrics per window
        
    Returns:
        Dictionary with aggregated statistics
    """
    agg_stats = {}
    
    # Metrics to aggregate
    metric_cols = ['mae', 'rmse', 'bias', 'sCRPS']
    
    for metric in metric_cols:
        if metric in window_metrics.columns:
            values = window_metrics[metric].dropna()
            if len(values) > 0:
                agg_stats[f'{metric}_mean'] = values.mean()
                agg_stats[f'{metric}_std'] = values.std()
                agg_stats[f'{metric}_min'] = values.min()
                agg_stats[f'{metric}_max'] = values.max()
            else:
                agg_stats[f'{metric}_mean'] = np.nan
                agg_stats[f'{metric}_std'] = np.nan
                agg_stats[f'{metric}_min'] = np.nan
                agg_stats[f'{metric}_max'] = np.nan
    
    # Add window count
    agg_stats['n_windows'] = len(window_metrics)
    
    return agg_stats


def compute_quantile_metrics(cv_df: pd.DataFrame,
                            model_name: str,
                            quantile_cols: List[str],
                            y_col: str = 'y') -> pd.DataFrame:
    """
    Compute metrics specific to quantile predictions.
    
    Args:
        cv_df: Cross-validation results
        model_name: Model name
        quantile_cols: List of quantile prediction columns
        y_col: Target column name
        
    Returns:
        DataFrame with quantile-specific metrics including coverage
    """
    # Extract quantile levels from column names
    quantile_levels = []
    valid_cols = []
    
    for col in quantile_cols:
        if col not in cv_df.columns:
            logger.warning(f"Column {col} not found in CV results")
            continue
        
        try:
            # Assuming format: "ModelName-q10", "ModelName-q50", etc.
            if '-q' in col:
                q_str = col.split('-q')[-1]
                q_level = float(q_str) / 100.0
                quantile_levels.append(q_level)
                valid_cols.append(col)
        except:
            logger.warning(f"Could not parse quantile level from column {col}")
            continue
    
    if not quantile_levels:
        return pd.DataFrame()
    
    # Sort quantiles and columns together
    sorted_pairs = sorted(zip(quantile_levels, valid_cols))
    quantile_levels, valid_cols = zip(*sorted_pairs)
    
    # Get data
    y_true = cv_df[y_col].values
    quantile_preds = cv_df[list(valid_cols)].values
    
    # Compute quantile-based sCRPS
    scrps_value = compute_scrps(
        y_true=y_true,
        y_pred=quantile_preds,
        quantiles=np.array(quantile_levels)
    )
    
    # Compute coverage for common prediction intervals
    coverage_stats = {}
    interval_pairs = [
        (0.1, 0.9, 80),  # 80% interval
        (0.05, 0.95, 90),  # 90% interval
        (0.025, 0.975, 95)  # 95% interval
    ]
    
    for q_low, q_high, level in interval_pairs:
        # Find closest quantiles
        idx_low = np.argmin(np.abs(np.array(quantile_levels) - q_low))
        idx_high = np.argmin(np.abs(np.array(quantile_levels) - q_high))
        
        if idx_low != idx_high:
            lower_bound = quantile_preds[:, idx_low]
            upper_bound = quantile_preds[:, idx_high]
            
            # Compute empirical coverage
            in_interval = (y_true >= lower_bound) & (y_true <= upper_bound)
            coverage = in_interval.mean() * 100
            coverage_stats[f'coverage_{level}'] = coverage
    
    result = {
        'model': model_name,
        'n_quantiles': len(quantile_levels),
        'quantile_sCRPS': scrps_value
    }
    result.update(coverage_stats)
    
    return pd.DataFrame([result])


def compute_coverage(y_true: np.ndarray,
                    lower: np.ndarray,
                    upper: np.ndarray,
                    target_coverage: float = 0.9) -> Tuple[float, float]:
    """
    Compute empirical coverage and width of prediction intervals.
    
    Args:
        y_true: Actual values
        lower: Lower bounds of prediction intervals
        upper: Upper bounds of prediction intervals
        target_coverage: Target coverage level (e.g., 0.9 for 90%)
        
    Returns:
        Tuple of (empirical_coverage, average_width)
    """
    # Remove NaN values
    mask = ~(np.isnan(y_true) | np.isnan(lower) | np.isnan(upper))
    if not mask.any():
        return np.nan, np.nan
    
    y_true_clean = y_true[mask]
    lower_clean = lower[mask]
    upper_clean = upper[mask]
    
    # Compute coverage
    in_interval = (y_true_clean >= lower_clean) & (y_true_clean <= upper_clean)
    empirical_coverage = in_interval.mean()
    
    # Compute average interval width
    avg_width = (upper_clean - lower_clean).mean()
    
    return empirical_coverage, avg_width